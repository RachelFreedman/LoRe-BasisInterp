"""Circularity controls for the planted-direction check.

The pre-committed concern (Discord, 14/08): the planted direction and the
concept->feature scoring map are built from the SAME 50 contrast pairs per
concept, so a hit may be self-matching rather than concept reading. Two
separable questions live inside that, and this script runs both on CPU.

CONTROL 1 -- disjoint split (item-level circularity).
  Each concept's 50 pairs are split in half. Users are planted from half A only;
  the SAE concept profiles are built from half B only. Same concepts, disjoint
  text. If accuracy holds, a hit is not item memorisation. If it collapses, the
  headline number was self-matching.

CONTROL 2 -- answer absent from the library (false-positive rate).
  A truly novel concept cannot be generated here (that needs Bedrock credentials
  and a Skywork embedding pass), so we make the planted concept unavailable to
  the readout instead: score group g with its own concept column removed. From
  the readout's point of view the planted axis is then outside its library.
  Compared against masking a RANDOM other concept, so both conditions score over
  the same number of columns and the margin is not inflated by the denominator.
  A well-calibrated readout should be visibly less confident when the true answer
  is absent. Equal confidence means a confident name on real data is worthless.

CPU only, by design: utils.py resolves its device at import, so CUDA draws a
different init stream and the numbers stop being comparable to teammates'.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "PRISM"))

from PRISM.synthetic_concept_users import (DEFAULT_CONCEPTS, build_users,  # noqa: E402
                                           group_dir_match, set_seed,
                                           signed_ground_truth, unit)
from PRISM.planted_directions import group_mean_directions, cos_to_truth  # noqa: E402
from sae.scripts.planted_concept_readout import (build_profiles, encode,  # noqa: E402
                                                 identify, load_sae)


def half_split(emb, concepts, seed):
    """Per concept, split the 50 contrast pairs into two disjoint halves."""
    g = torch.Generator().manual_seed(seed)
    a, b = {}, {}
    for c in concepts:
        n = emb[c]["high"].shape[0]
        perm = torch.randperm(n, generator=g)
        ia, ib = perm[: n // 2], perm[n // 2:]
        a[c] = {"high": emb[c]["high"][ia], "low": emb[c]["low"][ia]}
        b[c] = {"high": emb[c]["high"][ib], "low": emb[c]["low"][ib]}
    return a, b


def concept_vectors_from(emb, concepts):
    """v_c = unit(mean high - mean low). Matches how concept_vectors.pt is defined."""
    return {c: unit(emb[c]["high"].float().mean(0) - emb[c]["low"].float().mean(0))
            for c in concepts}


def profiles_from(model, emb, concepts, bs, device):
    """SAE concept profiles + activity, in ACTIVATION space (never encode a difference:
    encode_pre_acts subtracts b_pre, so a centred vector reads as -b_pre)."""
    dz = {}
    for c in concepts:
        z_hi = encode(model, emb[c]["high"].float(), bs, device)
        z_lo = encode(model, emb[c]["low"].float(), bs, device)
        dz[c] = z_hi - z_lo
    profiles = build_profiles(dz)
    activity = torch.cat(list(dz.values())).abs().mean(0)
    return profiles, activity


def plant(emb, concepts, cvs, seed, args):
    """Fit LoReV2 on users planted from `emb`; return unit group-mean directions."""
    from utils import LoReV2
    V_true = torch.stack([cvs[c].float() for c in concepts], dim=1)
    V_signed = signed_ground_truth(V_true)
    n_groups = 2 * len(concepts)
    set_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    train_feats, _, group_id = build_users(emb, concepts, args.users_per_group,
                                           args.test_frac, gen, args.high_frac,
                                           args.subset_frac)
    model = LoReV2(len(train_feats), V_true.shape[0], 2 * len(concepts),
                   lam_pop=args.lam_pop, lam_d=args.lam_d,
                   num_iterations=args.iters, learning_rate=args.lr, verbose=False)
    model.train(train_feats)
    r_dirs = model.reward_dirs().detach().cpu().T
    gd = group_mean_directions(r_dirs, group_id, n_groups)
    return gd, V_true, V_signed, group_id, r_dirs


def score(dirs, concepts, profiles, activity, dec, top_k, drop=None):
    """Forced choice over `concepts`, optionally with one column removed per group.

    drop: list of concept indices to remove, one per direction (None = keep all).
    Returns (accuracy, concept_accuracy, mean margin, rows).
    """
    rows, hit, chit, margins = [], 0, 0, []
    for i, v in enumerate(dirs):
        keep = [j for j in range(len(concepts)) if drop is None or j != drop[i]]
        sub = [concepts[j] for j in keep]
        c, s, sig, _ = identify(F.normalize(v, dim=0), dec, profiles[:, keep], activity, top_k)
        true_c, true_s = i // 2, (1 if i % 2 == 0 else -1)
        ok_c = sub[c] == concepts[true_c]
        rows.append({"predicted": sub[c], "sign": "high" if s > 0 else "low",
                     "true": concepts[true_c], "true_sign": "high" if true_s > 0 else "low",
                     "margin": float(sig.abs().max() / (sig.abs().sum() + 1e-12)),
                     "answer_available": drop is None or drop[i] != true_c})
        hit += int(ok_c and s == true_s)
        chit += int(ok_c)
        margins.append(rows[-1]["margin"])
    n = len(dirs)
    return hit / n, chit / n, float(np.mean(margins)), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default=os.path.join(REPO, "data", "prism",
                                                  "contrastive_pair_embeddings.pt"))
    ap.add_argument("--concept_vectors", default=os.path.join(REPO, "data", "prism",
                                                              "concept_vectors.pt"))
    ap.add_argument("--checkpoint", default=os.path.join(REPO, "sae", "checkpoints", "d3", "model.pt"))
    ap.add_argument("--concepts", nargs="+", default=DEFAULT_CONCEPTS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--users_per_group", type=int, default=20)
    ap.add_argument("--test_frac", type=float, default=0.3)
    ap.add_argument("--subset_frac", type=float, default=0.7)
    ap.add_argument("--high_frac", type=float, default=0.5)
    ap.add_argument("--lam_pop", type=float, default=0.01)
    ap.add_argument("--lam_d", type=float, default=0.01)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--top_k", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--split_seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(REPO, "results", "planted",
                                                  "circularity_controls.json"))
    args = ap.parse_args()

    if torch.cuda.is_available():
        print("[warn] CUDA visible -- re-run with CUDA_VISIBLE_DEVICES='' to compare "
              "against teammates' numbers.\n")
    device = torch.device("cpu")

    emb = torch.load(args.emb, weights_only=False)
    cvs_file = torch.load(args.concept_vectors, weights_only=False)
    concepts = [c for c in args.concepts if c in emb and c in cvs_file]
    model, cfg = load_sae(args.checkpoint, device)
    dec = model.decoder.weight.detach().cpu().float()

    # sanity: confirm the stored concept vectors are the mean-difference definition
    rebuilt = concept_vectors_from(emb, concepts)
    agree = float(np.mean([float(F.cosine_similarity(rebuilt[c],
                                                     cvs_file[c].float(), dim=0)) for c in concepts]))
    print(f"stored concept_vectors vs recomputed mean-diff: mean cos {agree:.4f}\n")

    emb_a, emb_b = half_split(emb, concepts, args.split_seed)
    cvs_a = concept_vectors_from(emb_a, concepts)
    prof_a, act_a = profiles_from(model, emb_a, concepts, args.batch_size, device)
    prof_b, act_b = profiles_from(model, emb_b, concepts, args.batch_size, device)
    cross = float(np.mean([float(F.cosine_similarity(rebuilt[c], cvs_a[c], dim=0))
                           for c in concepts]))
    print(f"half-A concept vectors vs full-set: mean cos {cross:.4f}")
    print(f"half A/B pairs per concept: {emb_a[concepts[0]]['high'].shape[0]}/"
          f"{emb_b[concepts[0]]['high'].shape[0]}\n")

    out = {"concepts": concepts, "n_groups": 2 * len(concepts), "top_k": args.top_k,
           "chance": 1.0 / (2 * len(concepts)),
           "chance_one_dropped": 1.0 / (2 * len(concepts) - 2),
           "stored_vs_recomputed_cos": agree, "halfA_vs_full_cos": cross,
           "seeds": args.seeds, "split_seed": args.split_seed}

    same, disj, gmatch = [], [], []
    m_present, m_absent, acc_present = [], [], []
    rng = np.random.default_rng(0)

    for seed in args.seeds:
        gd, V_true_a, V_signed_a, group_id, r_dirs = plant(emb_a, concepts, cvs_a, seed, args)
        gmatch.append(group_dir_match(r_dirs, group_id, V_signed_a))

        # --- Control 1: same half (A/A, circular) vs disjoint (A/B) ---
        acc_same, _, _, _ = score(gd, concepts, prof_a, act_a, dec, args.top_k)
        acc_disj, _, _, _ = score(gd, concepts, prof_b, act_b, dec, args.top_k)
        same.append(acc_same)
        disj.append(acc_disj)

        # --- Control 2: true answer absent vs a random other absent (disjoint profiles) ---
        n_c = len(concepts)
        drop_true = [i // 2 for i in range(2 * n_c)]
        drop_rand = [int(rng.choice([j for j in range(n_c) if j != i // 2]))
                     for i in range(2 * n_c)]
        _, _, mar_abs, _ = score(gd, concepts, prof_b, act_b, dec, args.top_k, drop=drop_true)
        acc_pr, _, mar_pr, _ = score(gd, concepts, prof_b, act_b, dec, args.top_k, drop=drop_rand)
        m_absent.append(mar_abs)
        m_present.append(mar_pr)
        acc_present.append(acc_pr)

        print(f"seed {seed}: group_dir_match {gmatch[-1]:.3f} | "
              f"A/A {acc_same:.3f}  A/B {acc_disj:.3f} | "
              f"margin answer-present {mar_pr:.4f}  answer-absent {mar_abs:.4f}")

    out |= {
        "control1_same_half_accuracy_mean": float(np.mean(same)),
        "control1_same_half_per_seed": same,
        "control1_disjoint_accuracy_mean": float(np.mean(disj)),
        "control1_disjoint_per_seed": disj,
        "control1_group_dir_match_mean": float(np.mean(gmatch)),
        "control2_accuracy_answer_present": float(np.mean(acc_present)),
        "control2_margin_answer_present": float(np.mean(m_present)),
        "control2_margin_answer_absent": float(np.mean(m_absent)),
        "control2_margin_ratio_absent_over_present": float(np.mean(m_absent) / np.mean(m_present)),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n=== Control 1: item-level circularity ({len(args.seeds)} seeds) ===")
    print(f"  plant A, score A (circular) : {np.mean(same):.3f}")
    print(f"  plant A, score B (disjoint) : {np.mean(disj):.3f}")
    print(f"  chance                      : {out['chance']:.3f}")
    print(f"=== Control 2: confidence when the answer is unavailable ===")
    print(f"  margin, a random other concept dropped : {np.mean(m_present):.4f}")
    print(f"  margin, the TRUE concept dropped       : {np.mean(m_absent):.4f}")
    print(f"  ratio absent/present                   : {out['control2_margin_ratio_absent_over_present']:.3f}")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
