"""
Stage 1 of the planted-concept check: extract the reward directions LoReV2 learned
for user groups whose true concept we planted ourselves.

synthetic_concept_users.py builds users who disagree along known concept axes and
measures whether TRAINING recovers them. It stops there. This script keeps the
trained model's output: one direction per (concept, sign) group, each carrying a
label we know is correct. Those labelled directions are the answer key the SAE
readout (Stage 2) is scored against.

Why group means rather than per-user directions: individual users see a random 70%
subset of their group's pairs, so a single user's direction carries subset noise on
top of the concept. The group is the unit the ground truth is defined on.

IMPORTANT -- what this direction is and is not
----------------------------------------------
The model recovers WHICH USERS ARE ALIKE far better than it recovers WHAT THE AXES
ARE: group_dir_match ~0.99, but subspace_align 0.755 and best_axis_match 0.674
(random floors 0.050 / 0.030). A group-mean reward direction is therefore only a
partial reconstruction of the concept vector it was planted from. The per-group
`cos_to_truth` reported here quantifies exactly how partial, per group, and it is
the number Stage 2b's ceiling control needs in order to attribute a weak SAE
result to the SAE rather than to training.

Device: pinned to CPU by default. utils.py resolves `device` at import and creates
parameters with device=device, so manual_seed draws a different init stream on CUDA
and the metrics shift (subspace 0.755 -> 0.795 on one Blackwell run). Results are
only comparable to teammates' numbers on CPU.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(SCRIPT_DIR)

from random_baseline import percentile_of, random_unit_directions  # noqa: E402
from synthetic_concept_users import (DEFAULT_CONCEPTS, build_users,  # noqa: E402
                                     ground_truth_matrix, set_seed,
                                     signed_ground_truth, unit)


def group_mean_directions(reward_dirs, group_id, n_groups):
    """[n_users, F] user directions -> [n_groups, F] unit group means."""
    out = torch.zeros(n_groups, reward_dirs.shape[1])
    for g in range(n_groups):
        members = reward_dirs[group_id == g]
        out[g] = unit(members.mean(0))
    return out


def cos_to_truth(group_dirs, V_signed):
    """Cosine of each group's direction with its OWN signed ground-truth column.

    Sign-sensitive on purpose: a group that prefers low-C must come back negative
    on +C, so an unsigned |cos| would score a direction that got the concept right
    and the polarity backwards as a success.
    """
    Gn = F.normalize(group_dirs, dim=1)
    Tn = F.normalize(V_signed, dim=0)
    return torch.stack([Gn[g] @ Tn[:, g] for g in range(group_dirs.shape[0])])


def null_cosines(V_signed, n_null, seed):
    """Cosines of random unit directions against the ground-truth columns.

    Without this the per-group cosines are unreadable: Skywork embeddings sit in a
    narrow cone, so a random direction is not centred on 0 against these probes.
    """
    dirs = random_unit_directions(V_signed.shape[0], n_null, seed=seed)
    Tn = F.normalize(V_signed, dim=0)
    return np.array([float(d @ Tn[:, g]) for d in dirs for g in range(Tn.shape[1])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default=os.path.join(SCRIPT_DIR, "..", "data", "prism",
                                                  "contrastive_pair_embeddings.pt"))
    ap.add_argument("--concept_vectors", default=os.path.join(SCRIPT_DIR, "..", "data", "prism",
                                                              "concept_vectors.pt"))
    ap.add_argument("--concepts", nargs="+", default=DEFAULT_CONCEPTS)
    ap.add_argument("--users_per_group", type=int, default=20)
    ap.add_argument("--test_frac", type=float, default=0.3)
    ap.add_argument("--subset_frac", type=float, default=0.7)
    ap.add_argument("--high_frac", type=float, default=0.5)
    ap.add_argument("--k_fit", type=int, default=None, help="default 2 x #concepts")
    ap.add_argument("--lam_pop", type=float, default=0.01)
    ap.add_argument("--lam_d", type=float, default=0.01,
                    help="the committed synthetic config. NOT 10 -- that is the "
                         "Community Alignment setting and it suppresses the deltas "
                         "this control depends on")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_null", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "..", "results", "planted",
                                                  "planted_directions.pt"))
    args = ap.parse_args()

    if torch.cuda.is_available():
        print("[warn] CUDA is visible. Re-run with CUDA_VISIBLE_DEVICES='' if these "
              "numbers will be compared against anyone else's.\n")

    from utils import LoReV2  # after the CUDA warning: utils picks its device at import

    emb = torch.load(args.emb, weights_only=False)
    cvs = torch.load(args.concept_vectors, weights_only=False)
    concepts = [c for c in args.concepts if c in emb and c in cvs]
    if missing := [c for c in args.concepts if c not in concepts]:
        print(f"[warn] skipping concepts absent from the embeddings/vectors: {missing}")
    k_fit = args.k_fit or 2 * len(concepts)
    n_groups = 2 * len(concepts)

    V_true = ground_truth_matrix(cvs, concepts)
    V_signed = signed_ground_truth(V_true)
    labels = [(c, "high" if s == 0 else "low") for c in concepts for s in (0, 1)]

    null = null_cosines(V_signed, args.n_null, seed=0)
    print(f"Concepts ({len(concepts)}): {concepts}")
    print(f"{n_groups} groups, {args.users_per_group * n_groups} users, k_fit={k_fit}, "
          f"lam_pop={args.lam_pop}, lam_d={args.lam_d}, iters={args.iters}")
    print(f"random-direction null for cos_to_truth: mean {null.mean():+.4f}, "
          f"95th pct {np.percentile(null, 95):+.4f}\n")

    records = []
    for seed in args.seeds:
        set_seed(seed)
        gen = torch.Generator().manual_seed(seed)
        train_feats, _, group_id = build_users(emb, concepts, args.users_per_group,
                                               args.test_frac, gen, args.high_frac,
                                               args.subset_frac)

        model = LoReV2(len(train_feats), V_true.shape[0], k_fit, lam_pop=args.lam_pop,
                       lam_d=args.lam_d, num_iterations=args.iters,
                       learning_rate=args.lr, verbose=False)
        model.train(train_feats)

        r_dirs = model.reward_dirs().detach().cpu().T          # [n_users, F]
        group_dirs = group_mean_directions(r_dirs, group_id, n_groups)
        cos = cos_to_truth(group_dirs, V_signed)

        records.append({
            "seed": seed,
            "group_dirs": group_dirs,                          # [n_groups, F], unit
            "labels": labels,
            "group_id": group_id,
            "cos_to_truth": cos,
        })

        print(f"seed {seed}:")
        for g, (concept, sign) in enumerate(labels):
            pct = percentile_of(float(cos[g]), null)
            print(f"  {concept:<12} {sign:<4}  cos_to_truth {cos[g]:+.4f}  "
                  f"(beats {pct:.1f}% of random)")
        print(f"  mean {cos.mean():+.4f}\n")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({
        "records": records,
        "V_true": V_true,
        "V_signed": V_signed,
        "concepts": concepts,
        "labels": labels,
        "null_cosines": null,
        "config": vars(args),
    }, args.out)

    all_cos = torch.stack([r["cos_to_truth"] for r in records])
    print(f"=== across {len(records)} seeds ===")
    print(f"  cos_to_truth  mean {all_cos.mean():+.4f}  min {all_cos.min():+.4f}  "
          f"max {all_cos.max():+.4f}")
    print(f"  every group positive: {bool((all_cos > 0).all())}  "
          f"(a negative group means the polarity came back backwards)")
    print(f"\nSaved {args.out}")
    print("These directions are Stage 2's input. cos_to_truth is how much of the "
          "planted concept survived training -- Stage 2b runs the same SAE readout "
          "on V_true itself, so a weak feature ranking can be attributed to the SAE "
          "or to this gap, not left ambiguous.")


if __name__ == "__main__":
    main()
