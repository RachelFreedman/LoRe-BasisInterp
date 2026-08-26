#!/usr/bin/env python3
"""
LoRe-PRISM replication + collapse-vs-accuracy sweep.

  1. FULL (seed x K) GRID -- runs every (seed, K) combination, so you can check whether the
     flat-accuracy-across-K curve and the near-total basis collapse
     (cos to V_sft ~ 0.9999) actually hold across MULTIPLE seeds
  2. --init {random,vsft} -- 'vsft' initializes every basis column at V_sft + small
     independent noise instead of the repo's default random init.

"""
import os
import sys
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

device = "cuda:0" if torch.cuda.is_available() else "cpu"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils import LoRe_regularized, set_seed  
from modeling import PAIR_FORMAT, load_reference_direction  


def group_embeddings_by_user(dataset, seen_value, split_name):
    """Per user, stack the (chosen - rejected) embedding diffs into [m_i, 4096]."""
    grouped = defaultdict(list)
    for ex in dataset:
        info = ex.get("extra_info", {})
        if info.get("seen") == seen_value and info.get("split") == split_name:
            uid = info.get("user_id")
            if uid:
                if info.get("pair_format") != PAIR_FORMAT:
                    raise ValueError(
                        "Cached embeddings predate the corrected PRISM "
                        "string-vs-string format. Rerun prepare.py and "
                        "generate-prism-embeddings.py."
                    )
                chosen = torch.tensor(info["chosen_conv_embedding"], dtype=torch.float32, device=device)
                rejected = torch.tensor(info["rejected_conv_embedding"], dtype=torch.float32, device=device)
                grouped[uid].append(chosen - rejected)
    return [torch.stack(grouped[u]) for u in sorted(grouped.keys())]


def accuracy(W, V, features):
    """Device-safe pairwise accuracy: fraction of pairs where the personalized
    reward of (chosen - rejected) is positive, i.e. chosen scored higher."""
    accs = []
    for i, X in enumerate(features):
        X = X.to(V.device, dtype=V.dtype)
        scores = X @ (V @ W[i])
        accs.append((scores > 0).float().mean().item())
    return float(np.mean(accs)), float(np.std(accs))


def basis_collapse_metrics(V):
    """Diagnostics for whether V's columns share one direction (same
    definitions as basis_reproducibility_check.py, kept identical so numbers
    are directly comparable)."""
    singular_values = torch.linalg.svdvals(V.float())
    if singular_values.numel() > 1 and singular_values[0] > 0:
        s2_s1 = float((singular_values[1] / singular_values[0]).item())
    else:
        s2_s1 = float("nan")

    rank_threshold = singular_values[0] * 1e-3
    numerical_rank = int((singular_values > rank_threshold).sum().item())

    K = V.shape[1]
    if K > 1:
        V_unit = F.normalize(V.float(), dim=0)
        abs_cos = torch.abs(V_unit.T @ V_unit)
        diagonal = torch.eye(K, dtype=torch.bool, device=abs_cos.device)
        mean_abs_basis_cos = float(abs_cos[~diagonal].mean().item())
    else:
        mean_abs_basis_cos = float("nan")

    return {
        "numerical_rank": numerical_rank,
        "s2_s1": s2_s1,
        "mean_abs_basis_cos": mean_abs_basis_cos,
    }


def cos_to_vsft(V, V_sft):
    """Mean and min |cosine| of each basis column to the reference direction."""
    V_unit = F.normalize(V.float(), dim=0)
    vsft = V_sft.float().reshape(-1)
    vsft_unit = vsft / (torch.linalg.norm(vsft) + 1e-12)
    cos = (V_unit.T @ vsft_unit).abs()
    return float(cos.mean().item()), float(cos.min().item())


def weight_concentration_metrics(W_full):
    """Per-user diagnostics on softmax(W): how concentrated is each user's
    mixture on a single basis? top1 = max weight (1.0 = fully collapsed onto
    one basis); entropy in nats (0 = fully concentrated, log(K) = uniform
    over all K bases)."""
    W = W_full.float()
    top1 = W.max(dim=1).values
    eps = 1e-12
    entropy = -(W * (W + eps).log()).sum(dim=1)
    return {
        "mean_top1_weight": float(top1.mean().item()),
        "mean_entropy": float(entropy.mean().item()),
        "max_possible_entropy": float(np.log(W.shape[1])) if W.shape[1] > 1 else 0.0,
    }



class LoRe_regularized_VSFTInit(LoRe_regularized):
    def __init__(self, V_sft, alpha, num_classes, num_features, num_basis_vectors,
                 num_iterations, learning_rate, noise_std=1e-2):
        nn.Module.__init__(self)
        self.V_sft = V_sft.to(device)
        self.V_sft_norm = F.normalize(self.V_sft, dim=0)
        self.alpha = alpha
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_basis_vectors = num_basis_vectors
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate

        self.W = nn.Parameter(torch.rand(num_classes, num_basis_vectors, device=device))
        vsft_col = V_sft.to(device).reshape(-1, 1)             # [num_features, 1]
        base = vsft_col.repeat(1, num_basis_vectors)            # [num_features, K]
        noise = torch.randn_like(base) * noise_std
        self.V = nn.Parameter(base + noise)


def build_solver(init, V_sft, alpha, num_classes, num_features, K, iters, lr, noise_std):
    if init == "random":
        return LoRe_regularized(V_sft, alpha, num_classes, num_features, K, iters, lr)
    elif init == "vsft":
        return LoRe_regularized_VSFTInit(V_sft, alpha, num_classes, num_features, K,
                                          iters, lr, noise_std=noise_std)
    else:
        raise ValueError(f"Unknown --init {init!r}")


# ---------------------------------------------------------------------------
# One (K, seed) run
# ---------------------------------------------------------------------------

def run_one(K, seed, init, noise_std, V_final, train_seen, test_seen, base_test_acc, args):
    print("\n" + "=" * 64)
    print(f"K={K} seed={seed} init={init} alpha={args.alpha} "
          f"iters={args.iters} lr={args.lr}")
    print("=" * 64)
    set_seed(seed)  # lock RNG before the solve, matching train_basis.py's convention
    am = build_solver(init, V_final, args.alpha, len(train_seen), 4096, K,
                       args.iters, args.lr, noise_std)
    W_kept, V_kept = am.train(train_seen)         # pruned: what accuracy eval uses
    V_full = am.V.detach()                        # [4096, K] every basis column
    W_full = torch.softmax(am.W, dim=1).detach()  # [num_users, K] full weights

    train_acc, _ = accuracy(W_kept, V_kept, train_seen)
    test_acc, test_std = accuracy(W_kept, V_kept, test_seen)
    collapse = basis_collapse_metrics(V_full)
    mean_cos_vsft, min_cos_vsft = cos_to_vsft(V_full, V_final)
    weight_stats = weight_concentration_metrics(W_full)

    row = {
        "K": K, "seed": seed, "init": init,
        "alpha": args.alpha, "iters": args.iters, "lr": args.lr,
        "bases_kept": V_kept.shape[1],
        "train_acc": train_acc, "test_acc": test_acc, "test_std": test_std,
        "base_test_acc": base_test_acc, "delta_vs_base": test_acc - base_test_acc,
        "mean_cos_vsft": mean_cos_vsft, "min_cos_vsft": min_cos_vsft,
        **collapse, **weight_stats,
    }
    return row


def parse_args():
    p = argparse.ArgumentParser(
        description="Full (K x seed) replication grid, weight/basis collapse "
                     "diagnostics, and an optional V_sft-init ablation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--k-values", type=lambda s: [int(x) for x in s.split(",")],
                   default=[1, 5, 10, 25, 50], help="K values to sweep")
    p.add_argument("--seed-values", type=lambda s: [int(x) for x in s.split(",")],
                   default=[0, 1, 2, 42],
                   help="seeds to sweep (42 = team-agreed canonical value)")
    p.add_argument("--init", choices=["random", "vsft"], default="random",
                   help="'random' matches the repo's current default; 'vsft' "
                        "is the diagnostic ablation (init every basis column "
                        "near V_sft instead of from scratch)")
    p.add_argument("--vsft-noise-std", type=float, default=1e-2,
                   help="stddev of per-column noise added to V_sft when --init vsft")
    p.add_argument("--alpha", type=float, default=1e4, help="regularization strength")
    p.add_argument("--iters", type=int, default=20000, help="optimization steps")
    p.add_argument("--lr", type=float, default=0.5, help="learning rate")
    p.add_argument("--out", default="replication_sweep_results.csv",
                   help="CSV output path (gitignored; copy off-box to share)")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Device: {device}")
    print(f"Hyperparameters: alpha={args.alpha} iters={args.iters} lr={args.lr} "
          f"init={args.init} (data split fixed by prepare.py seed=123)")

    print("Loading cached embeddings...")
    train_emb = torch.load("data/prism/train_embeddings.pkl")
    test_emb = torch.load("data/prism/test_embeddings.pkl")
    train_seen = group_embeddings_by_user(train_emb, seen_value=True, split_name="train")
    test_seen = group_embeddings_by_user(test_emb, seen_value=True, split_name="test")
    print(f"Seen users: {len(train_seen)}")

    print("Loading backbone once for reference direction (V_sft)...")
    V_final = load_reference_direction(device)
    base_W = [torch.ones(1, device=device) for _ in train_seen]
    base_test_acc, base_test_std = accuracy(base_W, V_final, test_seen)

    rows = []
    for seed in args.seed_values:
        for K in args.k_values:
            row = run_one(K, seed, args.init, args.vsft_noise_std, V_final,
                           train_seen, test_seen, base_test_acc, args)
            rows.append(row)


    header = (f"{'seed':>4} | {'K':>3} | {'test_acc':>8} | {'vs_base':>8} | "
              f"{'s2/s1':>7} | {'mean|cos_basis|':>15} | {'mean_cos_vsft':>13} | "
              f"{'top1_wt':>7} | {'entropy':>7}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['seed']:>4} | {r['K']:>3} | {r['test_acc']:>8.4f} | "
              f"{r['delta_vs_base']:>+8.4f} | {r['s2_s1']:>7.4f} | "
              f"{r['mean_abs_basis_cos']:>15.4f} | {r['mean_cos_vsft']:>13.4f} | "
              f"{r['mean_top1_weight']:>7.4f} | {r['mean_entropy']:>7.4f}")

    fields = list(rows[0].keys())
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


if __name__ == "__main__":
    main()
