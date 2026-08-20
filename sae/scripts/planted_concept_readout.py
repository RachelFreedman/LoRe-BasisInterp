#!/usr/bin/env python3
"""
Stage 2 / 2b of the planted-concept check: does the SAE decomposition NAME the
concept a direction was planted from?

Stage 1 (PRISM/planted_directions.py) produced 12 reward directions whose true
(concept, sign) we know. This script runs each through the SAE and asks the SAE to
identify it. Because we planted the answer, a wrong answer is unambiguous -- which
is the whole point, and what the real-data concept lists can never provide.

The readout
-----------
A pair's reward under direction v is v @ (e_high - e_low). Passing the responses
through the SAE decomposes that difference into features:

    e_high - e_low  ~  sum_i (z_high_i - z_low_i) * d_i          (d_i = decoder col)
    v @ (e_high - e_low)  ~  sum_i dz_i * (d_i @ v)

so feature i's contribution to what v rewards is dz_i * (d_i @ v). That is an
additive decomposition of the actual score, not a heuristic ranking.

Two rules fixed before running (per the pre-registration):

  * differences are taken in ACTIVATION space -- encode each response, then
    z_high - z_low. NOT by encoding (e_high - e_low): `encode_pre_acts` is
    `encoder(x - b_pre)` with b_pre the train-embedding mean, so a difference
    vector (mean ~ 0) would be read as roughly -b_pre and the top-k selection
    would be driven by the bias rather than the signal.
  * features are ranked by mean-absolute-contribution over the pair-difference
    set: |d_i @ v| * mean_pairs|dz_i|.

Scoring: forced identification, not "did the concept appear"
------------------------------------------------------------
Each feature gets a profile p_i[c] = mean dz_i over concept c's pairs -- what the
feature responds to. A direction's signature over the concepts is then

    sig(v)[c] = sum_{i in top-K} (d_i @ v) * p_i[c]

and the SAE's answer is argmax_c |sig(v)[c]| with the sign of sig at that concept.

Identification rather than presence-in-top-20, because the concept library is
entangled with itself: formatting and diversity sit at +0.601 in the ground-truth
vectors, creativity and diversity at -0.682. "Did diversity features show up for a
formatting direction" has no correct answer. "Which of the 12 (concept, sign)
groups is this?" does, with chance at 1/12, and a plausible neighbour fails it.

Three references, or the number means nothing
---------------------------------------------
  planted   -- the Stage 1 recovered directions (the actual test)
  ceiling   -- the same readout on the ground-truth concept vectors (Stage 2b).
               Bounds what the SAE can do at all on these concepts, so a weak
               planted result can be attributed to the SAE or to the 0.736
               cos_to_truth gap left by training, instead of staying ambiguous.
  null      -- random unit directions. Should land at chance.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "PRISM"))

from PRISM.random_baseline import random_unit_directions  # noqa: E402
from sae.src.io import ensure_dir, write_json  # noqa: E402
from sae.src.topk_sae import TopKSAE  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="sae/checkpoints/d3/model.pt")
    p.add_argument("--directions", default="results/planted/planted_directions.pt")
    p.add_argument("--embeddings", default="data/prism/contrastive_pair_embeddings.pt")
    p.add_argument("--results-dir", default="results/planted")
    p.add_argument("--top-k", type=int, default=256,
                   help="features kept per direction for the signature. Default matches "
                        "the SAE's own k, so the readout is as sparse as the model is")
    p.add_argument("--top-n-table", type=int, default=20)
    p.add_argument("--n-null", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_sae(path: Path, device: torch.device) -> tuple[TopKSAE, dict]:
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["config"]
    train_cfg = cfg.get("training", {})
    model = TopKSAE(
        input_dim=int(cfg["input_dim"]),
        dict_size=int(cfg["dict_size"]),
        k=int(cfg["k"]),
        normalize_decoder=bool(train_cfg.get("normalize_decoder", True)),
        aux_k=int(train_cfg.get("aux_k", cfg["k"])),
        sparsity_mode=str(train_cfg.get("sparsity_mode", "topk")),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), cfg


@torch.no_grad()
def encode(model: TopKSAE, x: torch.Tensor, bs: int, device: torch.device) -> torch.Tensor:
    return torch.cat([model.encode(x[i:i + bs].to(device)).cpu() for i in range(0, len(x), bs)])


@torch.no_grad()
def recon_error(model: TopKSAE, x: torch.Tensor, bs: int, device: torch.device) -> float:
    """Mean L2 norm of the reconstruction residual, comparable to the 11.98 the SAE
    reported on PRISM test. This set is a different distribution (LLM-written concept
    responses), so a much larger value means the features are extrapolating and every
    list below is worth less."""
    errs = []
    for i in range(0, len(x), bs):
        b = x[i:i + bs].to(device)
        errs.append((model(b)[0] - b).norm(dim=1).cpu())
    return float(torch.cat(errs).mean())


def build_profiles(dz_by_concept: dict[str, torch.Tensor]) -> torch.Tensor:
    """[dict_size, n_concepts] -- what each feature responds to, as mean dz per concept."""
    return torch.stack([dz.mean(0) for dz in dz_by_concept.values()], dim=1)


def identify(v, dec, profiles, activity, top_k):
    """SAE's answer for direction v: (concept index, sign, signature, top feature ids).

    v: [F] unit. dec: [F, dict]. profiles: [dict, n_concepts]. activity: [dict].
    """
    align = dec.T @ v                                   # [dict]
    contrib = align.abs() * activity                    # pre-registered ranking
    top = torch.topk(contrib, top_k).indices
    sig = align[top] @ profiles[top]                    # [n_concepts]
    c = int(sig.abs().argmax())
    return c, (1 if sig[c] > 0 else -1), sig, top


def run_set(directions, labels_true, dec, profiles, activity, concepts, top_k):
    """directions: [n, F]. labels_true: list of (concept_idx, sign) or None for a null."""
    rows, correct, concept_only = [], 0, 0
    for i, v in enumerate(directions):
        c, s, sig, _ = identify(F.normalize(v, dim=0), dec, profiles, activity, top_k)
        row = {"predicted_concept": concepts[c], "predicted_sign": "high" if s > 0 else "low",
               "margin": float(sig.abs().max() / (sig.abs().sum() + 1e-12))}
        if labels_true is not None:
            tc, ts = labels_true[i]
            row |= {"true_concept": concepts[tc], "true_sign": "high" if ts > 0 else "low",
                    "correct": int(c == tc and s == ts), "concept_correct": int(c == tc)}
            correct += row["correct"]
            concept_only += row["concept_correct"]
        rows.append(row)
    n = len(directions)
    return rows, (correct / n if labels_true else None), (concept_only / n if labels_true else None)


def main() -> int:
    args = parse_args()
    device = choose_device(args.device)
    results_dir = ensure_dir(args.results_dir)

    model, cfg = load_sae(Path(args.checkpoint), device)
    payload = torch.load(args.directions, weights_only=False)
    emb = torch.load(args.embeddings, weights_only=False)
    concepts = payload["concepts"]
    n_concepts = len(concepts)

    # activation-space pair differences, per concept
    dz_by_concept, all_x = {}, []
    for c in concepts:
        high, low = emb[c]["high"].float(), emb[c]["low"].float()
        z_high = encode(model, high, args.batch_size, device)
        z_low = encode(model, low, args.batch_size, device)
        dz_by_concept[c] = z_high - z_low
        all_x += [high, low]
    dz_all = torch.cat(list(dz_by_concept.values()))
    profiles = build_profiles(dz_by_concept)
    activity = dz_all.abs().mean(0)
    dec = model.decoder.weight.detach().cpu().float()

    err = recon_error(model, torch.cat(all_x), args.batch_size, device)
    print(f"SAE: dict={cfg['dict_size']}, k={cfg['k']}, top_k readout={args.top_k}")
    print(f"reconstruction error on the contrast set: {err:.2f}  (PRISM test baseline 11.98)")
    if err > 2 * 11.98:
        print("  [warn] much worse than the training distribution -- features are "
              "extrapolating and the lists below are correspondingly weaker evidence")
    live = int((activity > 0).sum())
    print(f"features active anywhere on this set: {live} / {cfg['dict_size']}\n")

    # ---- ceiling (Stage 2b): the ground-truth concept vectors themselves ----
    V_signed = payload["V_signed"]                       # [F, 2*n_concepts]
    truth_dirs = V_signed.T
    truth_labels = [(g // 2, 1 if g % 2 == 0 else -1) for g in range(2 * n_concepts)]
    ceil_rows, ceil_acc, ceil_concept = run_set(truth_dirs, truth_labels, dec, profiles,
                                                activity, concepts, args.top_k)

    # ---- null: random unit directions ----
    # Scored the same way the planted set is: a random direction is asked to be each
    # of the 12 groups in turn, so the null sits on the identical metric rather than
    # on a concept-only variant with a different chance level.
    null_dirs = torch.stack(random_unit_directions(dec.shape[0], args.n_null, seed=0))
    null_rows, _, _ = run_set(null_dirs, None, dec, profiles, activity, concepts, args.top_k)
    null_pred = [(concepts.index(r["predicted_concept"]), r["predicted_sign"]) for r in null_rows]
    null_hits = float(np.mean([[int(pc == tc and ps == ("high" if ts > 0 else "low"))
                                for tc, ts in truth_labels] for pc, ps in null_pred]))
    null_concept_hits = float(np.mean([[int(pc == tc) for tc, _ in truth_labels]
                                       for pc, _ in null_pred]))

    # ---- planted (the actual test), per seed ----
    all_rows, accs, concept_accs = [], [], []
    for rec in payload["records"]:
        labels_true = [(concepts.index(c), 1 if s == "high" else -1) for c, s in rec["labels"]]
        rows, acc, cacc = run_set(rec["group_dirs"], labels_true, dec, profiles,
                                  activity, concepts, args.top_k)
        for r in rows:
            r["seed"] = rec["seed"]
        all_rows += rows
        accs.append(acc)
        concept_accs.append(cacc)

    with (results_dir / "planted_readout.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    chance = 1 / (2 * n_concepts)
    print(f"=== identification: which of the {2 * n_concepts} (concept, sign) groups is this? ===")
    print(f"  chance                     {chance:.3f}")
    print(f"  null (random directions)   {null_hits:.3f}   concept-only {null_concept_hits:.3f}"
          f"   ({args.n_null} draws)")
    print(f"  ceiling (ground-truth v)   {ceil_acc:.3f}   concept-only {ceil_concept:.3f}")
    print(f"  planted (recovered dirs)   {np.mean(accs):.3f} +/- {np.std(accs):.3f}   "
          f"concept-only {np.mean(concept_accs):.3f}")
    print("\n  per seed: " + ", ".join(f"{a:.3f}" for a in accs))

    # Where the errors fall. Every misidentification so far has been onto a concept the
    # library itself cannot separate, so report the ground-truth cosine for each error
    # rather than counting them as flat failures.
    truth_cos = F.normalize(payload["V_true"], dim=0).T @ F.normalize(payload["V_true"], dim=0)
    print("\n=== errors, against the library's own geometry ===")
    for tag, rows in (("ceiling", ceil_rows), ("planted", all_rows)):
        errs = [r for r in rows if not r["correct"]]
        if not errs:
            print(f"  {tag}: none")
            continue
        print(f"  {tag}: {len(errs)}/{len(rows)}")
        seen = set()
        for r in errs:
            pair = (r["true_concept"], r["predicted_concept"])
            if pair in seen:
                continue
            seen.add(pair)
            cos = float(truth_cos[concepts.index(pair[0]), concepts.index(pair[1])])
            print(f"    {pair[0]:<11} -> {r['predicted_concept']:<11} "
                  f"(ground-truth cos {cos:+.3f})")
    entangled = sorted({(concepts[i], concepts[j], float(truth_cos[i, j]))
                        for i in range(n_concepts) for j in range(i + 1, n_concepts)
                        if abs(truth_cos[i, j]) >= 0.5}, key=lambda t: -abs(t[2]))
    print("  library pairs at |cos| >= 0.50 (i.e. not separable in principle):")
    for a, b, c in entangled:
        print(f"    {a:<11} {b:<11} {c:+.3f}")

    # ---- top features for one direction, for the write-up ----
    rec0 = payload["records"][0]
    table = []
    for g, (c, s) in enumerate(rec0["labels"]):
        v = F.normalize(rec0["group_dirs"][g], dim=0)
        align = dec.T @ v
        top = torch.topk(align.abs() * activity, args.top_n_table).indices
        for rank, i in enumerate(top.tolist()):
            prof = profiles[i]
            table.append({
                "group": f"{c}_{s}", "rank": rank, "feature_id": i,
                "alignment": float(align[i]),
                "mean_abs_dz": float(activity[i]),
                "contribution": float(align[i] * prof[concepts.index(c)]),
                "feature_top_concept": concepts[int(prof.abs().argmax())],
                "feature_top_concept_sign": "high" if prof[int(prof.abs().argmax())] > 0 else "low",
            })
    with (results_dir / "planted_top_features.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)

    write_json(results_dir / "planted_readout_summary.json", {
        "chance": chance,
        "null_accuracy": float(null_hits),
        "null_concept_accuracy": float(null_concept_hits),
        "ceiling_accuracy": ceil_acc,
        "ceiling_concept_accuracy": ceil_concept,
        "planted_accuracy_mean": float(np.mean(accs)),
        "planted_accuracy_std": float(np.std(accs)),
        "planted_concept_accuracy_mean": float(np.mean(concept_accs)),
        "per_seed_accuracy": accs,
        "reconstruction_error_contrast_set": err,
        "reconstruction_error_prism_test_reference": 11.98,
        "top_k_readout": args.top_k,
        "dict_size": int(cfg["dict_size"]),
        "sae_k": int(cfg["k"]),
        "note": ("Identification is forced-choice over 12 (concept, sign) groups. "
                 "Presence-in-top-20 is not used as the primary metric because the "
                 "concept library is internally correlated up to |cos| 0.68."),
    })
    print(f"\nWrote {results_dir}/planted_readout.csv, planted_top_features.csv, "
          f"planted_readout_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
