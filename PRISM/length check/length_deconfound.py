#!/usr/bin/env python3
"""
Length de-confound check.
"""
import argparse
import json
from collections import defaultdict

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", required=True, help="e.g. data/community_alignment/pairs.json")
    p.add_argument("--min_pairs", type=int, default=100)
    p.add_argument("--test_frac", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--metric", choices=["chars", "words"], default="chars",
                    help="which length measure predicts 'preferred'")
    return p.parse_args()


def acc_with_ties(correct, total_ties, total):
    return (correct + 0.5 * total_ties) / max(total, 1)


def main():
    args = parse_args()
    with open(args.pairs) as f:
        pairs = json.load(f)
    print(f"{len(pairs)} pairs loaded")

    # group by user -> turn -> [pair, ...], exactly mirroring build_user_diffs' leak-free split
    by_user = defaultdict(lambda: defaultdict(list))
    for p in pairs:
        turn_key = (p.get("conversation_id"), p.get("turn"))
        by_user[p["user_id"]][turn_key].append(p)

    gen = torch.Generator().manual_seed(args.seed)
    test_pairs = []
    n_users_used = 0
    for u, groups in by_user.items():
        keys = list(groups)
        total = sum(len(groups[k]) for k in keys)
        if total < args.min_pairs:
            continue
        n_users_used += 1
        perm = torch.randperm(len(keys), generator=gen).tolist()
        n_test_turns = max(1, int(round(args.test_frac * len(keys))))
        test_keys = {keys[i] for i in perm[:n_test_turns]}
        for k in test_keys:
            test_pairs.extend(groups[k])

    print(f"{n_users_used} users >= {args.min_pairs} pairs; {len(test_pairs)} held-out test "
          f"pairs across {n_users_used} users\n")
    if not test_pairs:
        print("No held-out pairs; nothing to score."); return

    def length(text):
        return len(text) if args.metric == "chars" else len(text.split())

    correct = ties = 0
    per_user_acc = defaultdict(lambda: [0, 0, 0])  # [correct, ties, total]
    for p in test_pairs:
        lc, lr = length(p["chosen"]), length(p["rejected"])
        u = p["user_id"]
        per_user_acc[u][2] += 1
        if lc > lr:
            correct += 1; per_user_acc[u][0] += 1
        elif lc == lr:
            ties += 1; per_user_acc[u][1] += 1

    overall = acc_with_ties(correct, ties, len(test_pairs))
    per_user_vals = [acc_with_ties(c, t, n) for c, t, n in per_user_acc.values()]

    print(f"Predict preferred = longer response ({args.metric})")
    print(f"  overall accuracy (pooled over all held-out pairs) : {overall:.4f}")
    print(f"  per-user accuracy: mean={np.mean(per_user_vals):.4f}  "
          f"std={np.std(per_user_vals):.4f}  n_users={len(per_user_vals)}")
    print(f"  ties (chosen/rejected exactly same length)        : {ties}/{len(test_pairs)}")
    
if __name__ == "__main__":
    main()
