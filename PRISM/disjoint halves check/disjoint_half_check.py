#!/usr/bin/env python3
"""
stability_check.py's --check split varies which of each user's OWN pairs land in train vs.
test, holding the user population fixed. That measures the model's sensitivity to which pairs
you happened to draw. This script measures: how much would
v_pop differ just from being fit on two independent, non-overlapping halves of the same 941
users? 
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(SCRIPT_DIR)
from community_alignment_lore import build_user_diffs  # noqa: E402
from stability_check import fit_one_run, unit, summarize  # noqa: E402
from random_baseline import random_unit_directions       # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description="How much does v_pop differ between two disjoint halves of the same user "
                     "pool, purely from finite-sample noise? Baseline for split-stability.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pairs", required=True)
    p.add_argument("--emb", required=True)
    p.add_argument("--min_pairs", type=int, default=100)
    p.add_argument("--test_frac", type=float, default=0.3)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--max_pairs", type=int, default=None)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--lam_pop", type=float, default=0.01)
    p.add_argument("--lam_d", type=float, default=10.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--iters", type=int, default=10000)
    p.add_argument("--n-splits", type=int, default=5,
                    help="number of independent random 2-way partitions to try")
    p.add_argument("--split-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="seeds controlling both the user-pool build and each partition's "
                         "shuffle; also used as the init seed for each half's model")
    p.add_argument("--n-baseline", type=int, default=200,
                    help="isotropic random directions for the usual random-direction null")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.pairs) as f:
        pairs = json.load(f)
    blob = torch.load(args.emb, weights_only=False)
    emb_lookup = {k: blob["emb"][i] for i, k in enumerate(blob["keys"])}
    print(f"{len(pairs)} pairs, {len(emb_lookup)} embeddings")
    num_features = next(iter(emb_lookup.values())).shape[-1]

    half_cosines = []
    for run_idx, s in enumerate(args.split_seeds):
        random.seed(s); np.random.seed(s)
        gen = torch.Generator().manual_seed(s)
        users = build_user_diffs(pairs, emb_lookup, args.min_pairs, args.test_frac, gen,
                                 split_by_turn=True, max_pairs=args.max_pairs,
                                 val_frac=args.val_frac)
        uids = sorted(users)
        if len(uids) < 2:
            print("Not enough users; nothing to run."); return

        # shuffle the (fixed, sorted) user list with this run's seed, split into two disjoint
        # halves -- same 941-user pool, non-overlapping members, not just non-overlapping pairs
        rng = random.Random(s)
        shuffled = uids[:]
        rng.shuffle(shuffled)
        mid = len(shuffled) // 2
        half_a, half_b = shuffled[:mid], shuffled[mid:]
        print(f"\n[split {run_idx}] seed={s}  half A: {len(half_a)} users, "
              f"half B: {len(half_b)} users")

        halves_vpop = []
        for label, half_uids in (("A", half_a), ("B", half_b)):
            train_feats = [users[u][0] for u in half_uids]
            val_feats = [users[u][1] for u in half_uids] if args.val_frac > 0 else None
            test_feats = [users[u][2] for u in half_uids]
            torch.manual_seed(s)  # same init seed for both halves, isolates the user-set effect
            v_pop, _, train_acc, test_acc = fit_one_run(
                train_feats, num_features, args.rank, args.lam_pop, args.lam_d, args.iters,
                args.lr, val_feats, test_feats)
            print(f"[split {run_idx}]   half {label}: train_acc={train_acc:.4f} "
                  f"test_acc={test_acc:.4f}")
            halves_vpop.append(v_pop)

        cos_ab = float((halves_vpop[0] @ halves_vpop[1]).item())
        half_cosines.append(cos_ab)
        print(f"[split {run_idx}]   cos(v_pop_A, v_pop_B) = {cos_ab:+.4f}")

    print("\n" + "#" * 90)
    print("# DISJOINT-HALVES SAMPLING-NOISE BASELINE")
    print("#" * 90)
    mean, std = summarize("disjoint-half v_pop", half_cosines)

    print(f"\nIsotropic random-direction baseline (n={args.n_baseline}, for scale reference):")
    base_dirs = random_unit_directions(num_features, args.n_baseline, seed=0)
    from stability_check import pairwise_cosines
    base_mean, base_std = summarize("random vs random", pairwise_cosines(base_dirs))

    print("\n" + "-" * 90)
    print(f"disjoint-half v_pop cosine : {mean:+.4f} +/- {std:.4f}  (n={len(half_cosines)})")


if __name__ == "__main__":
    main()
