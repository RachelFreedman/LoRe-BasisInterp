#!/usr/bin/env python3
"""
Log collapse/accuracy metrics DURING training, not just at the end.

"""
import os
import sys
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = "cuda:0" if torch.cuda.is_available() else "cpu"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils import LoRe_regularized, set_seed  
from modeling import load_reference_direction 


from replication_sweep import ( 
    group_embeddings_by_user,
    accuracy,
    basis_collapse_metrics,
    cos_to_vsft,
    weight_concentration_metrics,
)


class LoRe_regularized_Logged(LoRe_regularized):

    def train_with_logging(self, X, test_features, log_every=500):
        self.to(device)
        X_cat, y = self._prepare_batch(X)
        X_cat = X_cat.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer_W = optim.Adam([self.W], lr=self.learning_rate)
        optimizer_V = optim.Adam([self.V], lr=self.learning_rate)

        log_rows = []

        for step in range(self.num_iterations):
            alpha_curr = self._alpha_at_step(step)

            # ---- Update W: freeze V (identical to LoRe_regularized.train) ----
            optimizer_W.zero_grad()
            nll_W, _, _ = self._forward_from_packed(X_cat, y, alpha_curr=0.0)
            nll_W.backward()
            optimizer_W.step()

            # ---- Update V: freeze W (identical to LoRe_regularized.train) ----
            optimizer_V.zero_grad()
            nll_V, reg, _ = self._forward_from_packed(X_cat, y, alpha_curr=alpha_curr)
            total_loss_V = nll_V + alpha_curr * reg
            total_loss_V.backward()
            optimizer_V.step()

            is_last = (step == self.num_iterations - 1)
            if step % log_every == 0 or is_last:
                with torch.no_grad():
                    W_probs = F.softmax(self.W, dim=1).detach()   # [C, K], unpruned
                    V_full = self.V.detach()                      # [F, K], unpruned
                    W_list = [W_probs[i] for i in range(W_probs.shape[0])]

                    collapse = basis_collapse_metrics(V_full)
                    mean_cos_vsft_val, min_cos_vsft_val = cos_to_vsft(V_full, self.V_sft)
                    weight_stats = weight_concentration_metrics(W_probs)
                    train_acc, _ = accuracy(W_list, V_full, X)
                    test_acc, test_std = accuracy(W_list, V_full, test_features)

                row = {
                    "alpha": self.alpha, "step": step, "alpha_curr": alpha_curr,
                    "nll_W": float(nll_W.item()), "nll_V": float(nll_V.item()),
                    "reg": float(reg.detach()) if isinstance(reg, torch.Tensor) else float(reg),
                    "train_acc": train_acc, "test_acc": test_acc, "test_std": test_std,
                    "mean_cos_vsft": mean_cos_vsft_val, "min_cos_vsft": min_cos_vsft_val,
                    **collapse, **weight_stats,
                }
                log_rows.append(row)
                print(
                    f"[alpha={self.alpha:g} step={step:>6}] "
                    f"nll_W={row['nll_W']:.4f} nll_V={row['nll_V']:.4f} "
                    f"reg={row['reg']:.4f} alpha_curr={alpha_curr:.2f} | "
                    f"train_acc={train_acc:.4f} test_acc={test_acc:.4f} | "
                    f"cos_basis={collapse['mean_abs_basis_cos']:.4f} "
                    f"cos_vsft={mean_cos_vsft_val:.4f} | "
                    f"top1_wt={weight_stats['mean_top1_weight']:.4f} "
                    f"entropy={weight_stats['mean_entropy']:.4f}"
                )

        return log_rows


def parse_args():
    p = argparse.ArgumentParser(
        description="Log collapse/accuracy metrics every N steps during "
                     "training, across one or more alpha values.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--alphas", type=lambda s: [float(x) for x in s.split(",")],
                   default=[0.0, 100.0, 1000.0, 10000.0],
                   help="comma-separated alpha values to log curves for")
    p.add_argument("--K", type=int, default=10, help="number of basis vectors")
    p.add_argument("--seed", type=int, default=42,
                   help="seed, locked identically before each alpha's run so "
                        "differences are attributable to alpha, not init")
    p.add_argument("--iters", type=int, default=20000, help="optimization steps")
    p.add_argument("--lr", type=float, default=0.5, help="learning rate")
    p.add_argument("--log-every", type=int, default=500,
                   help="snapshot metrics every this many steps")
    p.add_argument("--out", default="training_curves.csv",
                   help="CSV output path (one row per alpha per logged step)")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Device: {device}")
    print(f"K={args.K} seed={args.seed} iters={args.iters} lr={args.lr} "
          f"log_every={args.log_every} alphas={args.alphas}")

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
    print(f"Base RM test accuracy: {base_test_acc:.4f} +/- {base_test_std:.4f}")

    all_rows = []
    for alpha in args.alphas:
        print("\n" + "=" * 80)
        print(f"TRAINING CURVE: alpha={alpha:g}  (K={args.K}, seed={args.seed})")
        print("=" * 80)
        set_seed(args.seed)  # identical init across alphas -- isolates alpha's effect
        am = LoRe_regularized_Logged(V_final, alpha, len(train_seen), 4096, args.K,
                                      args.iters, args.lr)
        rows = am.train_with_logging(train_seen, test_seen, log_every=args.log_every)
        for r in rows:
            r["base_test_acc"] = base_test_acc
            r["delta_vs_base"] = r["test_acc"] - base_test_acc
        all_rows.extend(rows)

    fields = list(all_rows[0].keys())
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)
    print(f"\nWrote {len(all_rows)} rows ({len(args.alphas)} alphas x "
          f"~{args.iters // args.log_every} snapshots) to {args.out}")


if __name__ == "__main__":
    main()
