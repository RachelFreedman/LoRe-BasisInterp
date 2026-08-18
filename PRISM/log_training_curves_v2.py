#!/usr/bin/env python3
"""
This script sweeps RANK by default -- the same axis
community_alignment_lore.py sweeps for the final accuracy check -- and logs accuracy every
--log-every steps for each rank, at fixed (lam_pop, lam_d, lr) so the curves are comparable.

Pass --lams as an alternative sweep (fixed rank, several lam_pop/lam_d pairs) if you want to see
how regularization strength shapes the training curve instead.
"""
import argparse
import csv
import os
import random
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(SCRIPT_DIR)
from utils import LoReV2                                    
from community_alignment_lore import build_user_diffs, unit  
from synthetic_recovery import eval_acc, collapse_metrics    
from rm_head_utils import load_reward_head                  

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class LoReV2_Logged(LoReV2):

    def train_with_logging(self, X, val, test_feats, log_every=500):
        self.to(device)
        from utils import _pack_users  
        X_cat, uid = _pack_users(X)
        X_cat = X_cat.to(device, non_blocking=True).float()
        uid = uid.to(device, non_blocking=True)
        self._rescale_init(X_cat, uid)

        Xv_cat = uidv = None
        if val is not None:
            Xv_cat, uidv = _pack_users(val)
            Xv_cat = Xv_cat.to(device, non_blocking=True).float()
            uidv = uidv.to(device, non_blocking=True)

        opt = torch.optim.Adam([self.V, self.wbar, self.delta], lr=self.learning_rate)

        log_rows = []
        best, since_best = float("inf"), 0
        for step in range(self.num_iterations):
            opt.zero_grad()
            loss = self._nll(X_cat, uid) + self._reg()
            loss.backward()
            opt.step()

            if val is not None:
                with torch.no_grad():
                    monitored = self._nll(Xv_cat, uidv).item()
            else:
                monitored = loss.item()

            is_last = (step == self.num_iterations - 1)
            if step % log_every == 0 or is_last:
                with torch.no_grad():
                    W_eff = self.user_weights().detach()
                    V_now = self.V.detach()
                    train_acc = eval_acc(W_eff, V_now, X)
                    test_acc = eval_acc(W_eff, V_now, test_feats)
                    collapse = collapse_metrics(V_now.cpu())
                row = {
                    "rank": self.num_basis_vectors, "lam_pop": self.lam_pop, "lam_d": self.lam_d,
                    "step": step, "loss": float(loss.item()), "monitored": float(monitored),
                    "train_acc": train_acc, "test_acc": test_acc,
                    "min_abs_basis_cos": collapse,
                }
                log_rows.append(row)
                print(f"[K={self.num_basis_vectors} step={step:>6}] "
                      f"loss={row['loss']:.4f} monitored={row['monitored']:.4f} | "
                      f"train_acc={train_acc:.4f} test_acc={test_acc:.4f} | "
                      f"min|cos_basis|={collapse:.4f}")

            if self.patience is not None:
                if monitored < best - self.tol:
                    best, since_best = monitored, 0
                else:
                    since_best += 1
                    if since_best >= self.patience:
                        self.stopped_at = step + 1
                        break
        else:
            self.stopped_at = self.num_iterations

        return log_rows


def parse_args():
    p = argparse.ArgumentParser(
        description="Log accuracy/collapse metrics every N steps while training LoReV2, "
                     "across a swept rank (default) or lam_pop/lam_d grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pairs", required=True, help="e.g. data/community_alignment/pairs.json")
    p.add_argument("--emb", required=True, help="e.g. data/community_alignment/embeddings.pt")
    p.add_argument("--min_pairs", type=int, default=50)
    p.add_argument("--test_frac", type=float, default=0.3)
    p.add_argument("--val_frac", type=float, default=0.2, help="required for early stopping")
    p.add_argument("--ranks", type=int, nargs="+", default=[1, 5, 10, 20],
                    help="ranks to log training curves for (the swept axis, like --alphas in v1)")
    p.add_argument("--lam_pop", type=float, default=0.01)
    p.add_argument("--lam_d", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42,
                    help="locked identically before each rank's run, like v1's --seed")
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--no-early-stop", action="store_true",
                    help="run the full --iters regardless of plateau (default: stop after "
                         "patience=100 steps without improvement, matching LoReV2.train)")
    p.add_argument("--out", default="training_curves_v2.csv")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Device: {device}")
    print(f"ranks={args.ranks} lam_pop={args.lam_pop} lam_d={args.lam_d} lr={args.lr} "
          f"seed={args.seed} iters={args.iters} log_every={args.log_every}")

    import json
    with open(args.pairs) as f:
        pairs = json.load(f)
    blob = torch.load(args.emb, weights_only=False)
    emb_lookup = {k: blob["emb"][i] for i, k in enumerate(blob["keys"])}
    print(f"{len(pairs)} pairs, {len(emb_lookup)} embeddings")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    users = build_user_diffs(pairs, emb_lookup, args.min_pairs, args.test_frac, gen,
                             split_by_turn=True, val_frac=args.val_frac)
    uids = sorted(users)
    if not uids:
        print("No users met the threshold; nothing to run.")
        return
    train_feats = [users[u][0] for u in uids]
    val_feats = [users[u][1] for u in uids]
    test_feats = [users[u][2] for u in uids]
    print(f"{len(uids)} users\n")

    head = unit(load_reward_head().reshape(-1))
    base_test_acc = float(np.mean([(te @ head > 0).float().mean().item() for te in test_feats]))
    print(f"Base RM test accuracy: {base_test_acc:.4f}")

    all_rows = []
    for K in args.ranks:
        print("\n" + "=" * 80)
        print(f"TRAINING CURVE: rank={K}  (lam_pop={args.lam_pop}, lam_d={args.lam_d}, "
              f"seed={args.seed})")
        print("=" * 80)
        torch.manual_seed(args.seed)  # identical init across ranks -- isolates rank's effect
        model = LoReV2_Logged(len(train_feats), 4096, K, lam_pop=args.lam_pop, lam_d=args.lam_d,
                              num_iterations=args.iters, learning_rate=args.lr, verbose=False)
        if args.no_early_stop:
            model.patience = None
        rows = model.train_with_logging(train_feats, val_feats, test_feats,
                                        log_every=args.log_every)
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
    print(f"\nWrote {len(all_rows)} rows ({len(args.ranks)} ranks x "
          f"~{args.iters // args.log_every} snapshots) to {args.out}")


if __name__ == "__main__":
    main()
