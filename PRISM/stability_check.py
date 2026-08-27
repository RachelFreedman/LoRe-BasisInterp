#!/usr/bin/env python3
"""
Stability check: do LoReV2's learned directions stay the same across random
seeds and across data splits, or do they vary run to run?

  --check seed  : SAME train/test split (one fixed split-seed), VARY only
                  the model's random init across --seeds. Answers: given the
                  same data, does optimization reliably land in the same
                  place?
  --check split : SAME model init seed, VARY only which pairs land in train
                  vs test across --seeds (each seed redraws the turn-level
                  split). Answers: does the learned direction depend on
                  which held-out pairs happened to get excluded?

IMPORTANT -- rotational ambiguity: LoReV2
drops the simplex, so V -> VR, w -> R^-1 w leaves the loss unchanged for any
invertible R. Comparing raw V columns or raw wbar across runs would treat
this arbitrary rotation as "instability" when the actual reward function is
identical. This script never compares raw V/wbar/delta directly -- only the
ROTATION-INVARIANT induced directions:
    v_pop   = unit(V @ wbar)                    (population direction)
    v_u     = unit(V @ (wbar + delta_u))         (each user's full direction)
Both are invariant under V->VR, w->R^-1w (V@wbar = VR @ R^-1wbar), so this
sidesteps the ambiguity rather than needing to solve it.


Baseline: pairwise cosine among N isotropic random unit directions in the
same 4096-dim space, for a numeric sense of "near 0" in this space. 

USAGE
-----
    # seed stability: fixed split, 5 different inits
    python stability_check.py --pairs data/community_alignment/pairs.json \
        --emb data/community_alignment/embeddings.pt --check seed \
        --seeds 0 1 2 3 4 --split-seed 0 --rank 8 \
        --lam_pop 0.01 --lam_d 10 --lr 1e-4

    # split stability: fixed init, 5 different train/test splits
    python stability_check.py --pairs data/community_alignment/pairs.json \
        --emb data/community_alignment/embeddings.pt --check split \
        --seeds 0 1 2 3 4 --init-seed 42 --rank 8 \
        --lam_pop 0.01 --lam_d 10 --lr 1e-4

Config above (rank=8, lam_pop=0.01, lam_d=10, lr=1e-4)
"""
import argparse
import json
import os
import random
import sys
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(SCRIPT_DIR)
from utils import LoReV2                              
from community_alignment_lore import build_user_diffs  
from random_baseline import random_unit_directions      
from synthetic_recovery import eval_acc             
from rm_head_utils import load_reward_head               


def unit(v):
    return v / (v.norm() + 1e-8)


def fit_one_run(train_feats, num_features, rank, lam_pop, lam_d, iters, lr, val_feats=None,
                test_feats=None):
    model = LoReV2(len(train_feats), num_features, rank, lam_pop=lam_pop, lam_d=lam_d,
                   num_iterations=iters, learning_rate=lr, verbose=False)
    W_eff, V = model.train(train_feats, val=val_feats)

    train_acc = eval_acc(W_eff, V, train_feats)
    test_acc = eval_acc(W_eff, V, test_feats) if test_feats is not None else float("nan")

    with torch.no_grad():
        v_pop = unit((model.V @ model.wbar).detach().cpu())
        # reward_dirs() is [F, C] -- one column per user, already V @ (wbar + delta_u)
        user_dirs_raw = model.reward_dirs().detach().cpu()  # [F, C]
        per_user = {i: unit(user_dirs_raw[:, i]) for i in range(user_dirs_raw.shape[1])}
    return v_pop, per_user, train_acc, test_acc


def pairwise_cosines(vectors):
    """vectors: list of [F] unit tensors. Returns list of cosine sims for every pair."""
    sims = []
    for a, b in combinations(range(len(vectors)), 2):
        sims.append(float((vectors[a] @ vectors[b]).item()))
    return sims


def print_cosine_matrix(labels, vectors, title):
    """Full NxN pairwise cosine matrix, not just the aggregate mean/std. """
    n = len(vectors)
    print(f"\n{title} -- full {n}x{n} pairwise cosine matrix:")
    header = "        " + "".join(f"{str(l):>9}" for l in labels)
    print(header)
    for i in range(n):
        row = f"{str(labels[i]):>8}"
        for j in range(n):
            if i == j:
                row += f"{'--':>9}"
            else:
                cos_ij = float((vectors[i] @ vectors[j]).item())
                row += f"{cos_ij:>9.4f}"
        print(row)


def summarize(name, sims):
    sims = np.array(sims)
    print(f"  {name:<22}: mean={sims.mean():+.4f}  std={sims.std():.4f}  "
          f"min={sims.min():+.4f}  max={sims.max():+.4f}  (n_pairs={len(sims)})")
    return float(sims.mean()), float(sims.std())


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", required=True)
    p.add_argument("--emb", required=True)
    p.add_argument("--check", choices=["seed", "split"], required=True,
                   help="'seed' = fixed split, vary init. 'split' = fixed init, vary split.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   help="the varying axis: init seeds if --check seed, split seeds if --check split")
    p.add_argument("--split-seed", type=int, default=0,
                   help="fixed split seed to use for every run when --check seed")
    p.add_argument("--init-seed", type=int, default=42,
                   help="fixed init seed to use for every run when --check split")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--lam_pop", type=float, default=0.01)
    p.add_argument("--lam_d", type=float, default=10.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--iters", type=int, default=1500)
    p.add_argument("--min_pairs", type=int, default=50)
    p.add_argument("--test_frac", type=float, default=0.3)
    p.add_argument("--val_frac", type=float, default=0.0)
    p.add_argument("--max_pairs", type=int, default=None)
    p.add_argument("--n_baseline", type=int, default=200,
                   help="number of random directions for the isotropic null")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.pairs) as f:
        pairs = json.load(f)
    blob = torch.load(args.emb, weights_only=False)
    emb_lookup = {k: blob["emb"][i] for i, k in enumerate(blob["keys"])}
    print(f"{len(pairs)} pairs, {len(emb_lookup)} embeddings")
    num_features = next(iter(emb_lookup.values())).shape[-1]

    pop_dirs = []
    per_user_dirs_by_run = []   # list of dict[user_id -> unit vector], one per run
    uid_sets = []
    run_accs = []                # list of (train_acc, test_acc), one per run, same order as args.seeds

    for run_idx, s in enumerate(args.seeds):
        if args.check == "seed":
            split_seed, init_seed = args.split_seed, s
        else:
            split_seed, init_seed = s, args.init_seed

        random.seed(split_seed); np.random.seed(split_seed)
        gen = torch.Generator().manual_seed(split_seed)
        users = build_user_diffs(pairs, emb_lookup, args.min_pairs, args.test_frac, gen,
                                 split_by_turn=True, max_pairs=args.max_pairs,
                                 val_frac=args.val_frac)
        uids = sorted(users)
        if not uids:
            print("No users met the threshold; nothing to run."); return
        train_feats = [users[u][0] for u in uids]
        val_feats = [users[u][1] for u in uids] if args.val_frac > 0 else None
        test_feats = [users[u][2] for u in uids]

        # init seed controls LoReV2's parameter init (torch.randn calls in __init__)
        torch.manual_seed(init_seed)
        print(f"\n[run {run_idx}] {args.check}={s}  (split_seed={split_seed}, "
              f"init_seed={init_seed})  {len(uids)} users")
        v_pop, per_user, train_acc, test_acc = fit_one_run(
            train_feats, num_features, args.rank, args.lam_pop, args.lam_d, args.iters,
            args.lr, val_feats, test_feats)
        print(f"[run {run_idx}]   train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")
        pop_dirs.append(v_pop)
        run_accs.append((train_acc, test_acc))
        # re-key per-user directions by actual user_id (not positional index), so runs
        # with a possibly-different uids ordering still match correctly
        per_user_dirs_by_run.append({uids[i]: per_user[i] for i in per_user})
        uid_sets.append(set(uids))

    common_uids = set.intersection(*uid_sets)
    if any(s != uid_sets[0] for s in uid_sets):
        dropped = set.union(*uid_sets) - common_uids

    print("\n" + "#" * 90)
    print(f"# STABILITY CHECK: {args.check}  (varying {args.seeds})")
    print("#" * 90)

    print("\nPopulation direction (v_pop = unit(V @ wbar)), pairwise cosine across runs:")
    print_cosine_matrix(args.seeds, pop_dirs, "v_pop")
    pop_mean, pop_std = summarize("v_pop", pairwise_cosines(pop_dirs))

    print(f"\nPer-run accuracy (cross-reference against the matrix above -- does the outlier "
          f"run, if any, also have unusual accuracy, or just a different direction?):")
    print(f"  {'run':>4} | {'label':>6} | {'train_acc':>10} | {'test_acc':>9}")
    for i, (s, (tr, te)) in enumerate(zip(args.seeds, run_accs)):
        print(f"  {i:>4} | {s:>6} | {tr:>10.4f} | {te:>9.4f}")
    test_accs = np.array([te for _, te in run_accs])
    print(f"  {'mean':>4} | {'':>6} | {np.mean([tr for tr, _ in run_accs]):>10.4f} | "
          f"{test_accs.mean():>9.4f}")
    print(f"  {'std':>4} | {'':>6} | {np.std([tr for tr, _ in run_accs]):>10.4f} | "
          f"{test_accs.std():>9.4f}")

    print("\nPer-user directions (v_u = unit(V @ (wbar + delta_u))), pairwise cosine across "
          "runs, averaged over users:")
    per_user_means = []
    for uid in sorted(common_uids):
        vecs = [per_user_dirs_by_run[r][uid] for r in range(len(args.seeds))]
        sims = pairwise_cosines(vecs)
        per_user_means.append(np.mean(sims))
    per_user_means = np.array(per_user_means)
    print(f"  {'per-user (avg)':<22}: mean={per_user_means.mean():+.4f}  "
          f"std={per_user_means.std():.4f}  min={per_user_means.min():+.4f}  "
          f"max={per_user_means.max():+.4f}  (n_users={len(per_user_means)})")

    base_dirs = random_unit_directions(num_features, args.n_baseline, seed=0)
    base_mean, base_std = summarize("random vs random", pairwise_cosines(base_dirs))

    # Stricter baseline than isotropic random: how close is v_pop to the actual base RM
    # direction, vs. how close random directions are to it? Two random 4096-dim directions sit
    # at 0 +/- 0.016 by construction, so that check "can't really fail" (per review comment) --
    # this one can, since the base RM is a real, non-random direction in the space.
    head = unit(load_reward_head().reshape(-1))
    pop_vs_head = [float((v @ head).item()) for v in pop_dirs]
    rand_vs_head = [float((v @ head).item()) for v in base_dirs]
    print(f"\ncos(v_pop, base_rm) -- stricter than the isotropic baseline above:")
    for i, (s, c) in enumerate(zip(args.seeds, pop_vs_head)):
        print(f"  run {i} ({args.check}={s}): {c:+.4f}")
    pop_head_mean, pop_head_std = summarize("v_pop vs base_rm", pop_vs_head)
    rand_head_mean, rand_head_std = summarize("random vs base_rm", rand_vs_head)

    print("\n" + "-" * 90)
    print(f"v_pop stability   : {pop_mean:+.4f} (vs random baseline {base_mean:+.4f})")
    print(f"per-user stability: {per_user_means.mean():+.4f} (vs random baseline {base_mean:+.4f})")
    print(f"v_pop vs base_rm  : {pop_head_mean:+.4f} +/- {pop_head_std:.4f} "
          f"(vs random-vs-base_rm {rand_head_mean:+.4f} +/- {rand_head_std:.4f})")
    print("Read-out: stability meaningfully above the random baseline (and its std) means the "
          "learned direction(s) are reproducible, not an artifact of this particular run. "
          "Stability indistinguishable from the random baseline means whatever gets "
          "interpreted downstream may just be describing this one run.")


if __name__ == "__main__":
    main()
