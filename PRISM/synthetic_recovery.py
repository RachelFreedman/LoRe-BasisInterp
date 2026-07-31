"""
Positive control: can LoRe recover KNOWN preference axes when the signal definitely exists?

The question being tested: "Construct a synthetic dataset with really clearly defined axes of
variation and tons of data per user, and see if LoRe can pick them up."

This is the experiment that has to come before anything else: PRISM gave a null, but a null is only
interpretable if we know the method CAN succeed when signal is present. Here we plant the ground
truth ourselves, so any failure is the method/optimizer, not the data.

Setup
-----
  * K_true orthonormal ground-truth axes V* in R^d.
  * Each user u gets a weight vector w*_u over those axes (by default: dominated by one axis, which
    is the regime LoRe's softmax-simplex W is designed for).
  * Their true reward direction is  r_u = V* w*_u.
  * For each pair we draw a feature difference vector `diff` (= chosen - rejected) and LABEL it by
    the user's own direction: we orient diff so that <diff, r_u> > 0, flipping a `noise` fraction.
    So the ONLY thing distinguishing users is which axis they care about.

We then train LoRe and measure whether it recovers the planted structure:
  test_acc          : held-out pair accuracy (per user)
  subspace_align    : mean cos of principal angles between span(V_learned) and span(V*)
                      (1.0 = the learned bases span exactly the true axes)
  best_axis_match   : for each true axis, the best |cos| to any learned column (are axes individually
                      recovered, not just the subspace?)
  min_abs_basis_cos : collapse metric (1.0 = all learned columns identical)
  w_recovery        : agreement between learned per-user argmax basis and the true dominant axis,
                      up to a permutation matched greedily (chance = 1/K_true)

Sweeps pairs-per-user, because "tons of data per user" is exactly the condition PRISM lacked
(~15 pairs/user). If LoRe recovers the axes at high pairs/user and degrades as pairs shrink, that
localizes the PRISM failure to data volume rather than the method.

CPU-only, no embeddings needed.
"""
import argparse
import os
import sys
import random
import csv

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(SCRIPT_DIR)
from utils import LoRe_regularized  # noqa: E402


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def make_ground_truth(d, k_true, gen):
    """k_true orthonormal axes in R^d -> [d, k_true]."""
    A = torch.randn(d, k_true, generator=gen)
    Q, _ = torch.linalg.qr(A)
    return Q[:, :k_true]


def make_users(n_users, k_true, gen, concentration):
    """Per-user weights over the true axes. concentration=1.0 -> one-hot (pure single-axis user);
    lower values mix in the other axes."""
    dom = torch.randint(0, k_true, (n_users,), generator=gen)
    W = torch.rand(n_users, k_true, generator=gen) * (1.0 - concentration)
    W[torch.arange(n_users), dom] += concentration
    W = W / W.sum(1, keepdim=True)
    return W, dom


def make_pairs(n, r_u, d, gen, noise, signal_scale):
    """Feature diffs labeled by the user's own direction r_u.

    diff = signal_scale * s * r_u_hat + isotropic noise, with s>0 meaning 'chosen preferred'.
    We return diffs already oriented as (chosen - rejected), so a correct model scores diff@r > 0.
    A `noise` fraction of labels is flipped (diff negated) to keep the problem non-separable.
    """
    r_hat = r_u / (r_u.norm() + 1e-8)
    # component along the user's true direction (always positive before flipping)
    mag = torch.rand(n, 1, generator=gen) * signal_scale + 0.05
    # isotropic distractor content orthogonal to nothing in particular (realistic clutter)
    clutter = torch.randn(n, d, generator=gen)
    clutter = clutter - (clutter @ r_hat).unsqueeze(1) * r_hat.unsqueeze(0)  # remove r component
    diffs = mag * r_hat.unsqueeze(0) + clutter
    flip = torch.rand(n, generator=gen) < noise
    diffs[flip] = -diffs[flip]
    return diffs


def collapse_metrics(V):
    Vn = F.normalize(V, dim=0)
    c = Vn.t() @ Vn
    k = V.shape[1]
    if k == 1:
        return 1.0
    off = c[~torch.eye(k, dtype=torch.bool)]
    return off.abs().min().item()


def subspace_alignment(V_learned, V_true):
    """Mean cosine of principal angles between the two column spaces (1.0 = identical span)."""
    Ql, _ = torch.linalg.qr(V_learned)
    Qt, _ = torch.linalg.qr(V_true)
    k = min(Ql.shape[1], Qt.shape[1])
    s = torch.linalg.svdvals(Ql[:, :Ql.shape[1]].t() @ Qt[:, :Qt.shape[1]])
    return s[:k].mean().item()


def best_axis_match(V_learned, V_true):
    """For each true axis, the best |cos| against any learned column; averaged."""
    Ln = F.normalize(V_learned, dim=0)
    Tn = F.normalize(V_true, dim=0)
    sim = (Tn.t() @ Ln).abs()          # [k_true, k_learned]
    return sim.max(dim=1).values.mean().item()


def eval_acc(W, V, feats):
    V = V.detach().cpu(); W = W.detach().cpu()
    return float(np.mean([(X.cpu() @ (V @ W[i]) > 0).float().mean().item()
                          for i, X in enumerate(feats)]))


def w_recovery(W_learned, dom_true, k_true):
    """Do users who share a true dominant axis get assigned the same learned basis?
    Greedy-match learned argmax basis -> true axis, then report agreement (chance = 1/k_true)."""
    pred = W_learned.detach().cpu().argmax(1).numpy()
    true = dom_true.numpy()
    # build contingency and greedily match
    ks = sorted(set(pred.tolist())); ts = sorted(set(true.tolist()))
    table = np.zeros((len(ts), len(ks)))
    for t_i, t in enumerate(ts):
        for k_i, k in enumerate(ks):
            table[t_i, k_i] = np.sum((true == t) & (pred == k))
    matched, used = 0, set()
    for _ in range(min(len(ts), len(ks))):
        i, j = np.unravel_index(np.argmax(table), table.shape)
        if table[i, j] <= 0:
            break
        matched += table[i, j]
        table[i, :] = -1; table[:, j] = -1
    return matched / len(true)


def run_one(pairs_per_user, args, seed):
    gen = torch.Generator().manual_seed(seed)
    set_seed(seed)
    d, k_true, n_users = args.dim, args.k_true, args.n_users

    V_true = make_ground_truth(d, k_true, gen)
    Wu, dom = make_users(n_users, k_true, gen, args.concentration)

    train, test = [], []
    for u in range(n_users):
        r_u = V_true @ Wu[u]
        train.append(make_pairs(pairs_per_user, r_u, d, gen, args.noise, args.signal))
        test.append(make_pairs(args.test_pairs, r_u, d, gen, args.noise, args.signal))

    # anchor for the regularizer: the population-mean direction (analogue of v_sft)
    anchor = torch.cat(train, 0).mean(0).reshape(-1, 1)

    model = LoRe_regularized(anchor, args.alpha, n_users, d, args.k_fit,
                             args.iters, args.lr)
    Wk, Vk = model.train(train)
    V_full = model.V.detach().cpu()

    return {
        "pairs_per_user": pairs_per_user,
        "seed": seed,
        "test_acc": eval_acc(Wk, Vk, test),
        "train_acc": eval_acc(Wk, Vk, train),
        "subspace_align": subspace_alignment(V_full, V_true),
        "best_axis_match": best_axis_match(V_full, V_true),
        "min_abs_basis_cos": collapse_metrics(V_full),
        "w_recovery": w_recovery(model.W, dom, k_true),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=256, help="feature dimension")
    ap.add_argument("--k_true", type=int, default=5, help="number of planted axes")
    ap.add_argument("--k_fit", type=int, default=5, help="rank LoRe is given")
    ap.add_argument("--n_users", type=int, default=200)
    ap.add_argument("--pairs", type=int, nargs="+", default=[10, 25, 50, 100, 400, 1000],
                    help="pairs per user to sweep ('tons of data per user' = the large end)")
    ap.add_argument("--test_pairs", type=int, default=200)
    ap.add_argument("--noise", type=float, default=0.1, help="fraction of flipped labels")
    ap.add_argument("--signal", type=float, default=1.0, help="scale of the true-direction component")
    ap.add_argument("--concentration", type=float, default=1.0,
                    help="1.0 = each user cares about exactly one axis")
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "..", "results", "synthetic",
                                                  "synthetic_recovery.csv"))
    args = ap.parse_args()

    print(f"Synthetic positive control: d={args.dim}, {args.k_true} planted axes, "
          f"{args.n_users} users, k_fit={args.k_fit}, alpha={args.alpha}, "
          f"label noise={args.noise}, concentration={args.concentration}")
    print(f"Chance accuracy = 0.5 ; ceiling given label noise = {1 - args.noise:.2f}")
    print(f"w_recovery chance = {1/args.k_true:.2f}\n")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    f = open(args.out, "w", newline="")
    w = csv.DictWriter(f, fieldnames=["pairs_per_user", "seed", "train_acc", "test_acc",
                                      "subspace_align", "best_axis_match",
                                      "min_abs_basis_cos", "w_recovery"])
    w.writeheader()

    print(f"{'pairs/user':>10} | {'test_acc':>8} | {'subspace':>8} | {'axis_match':>10} | "
          f"{'min|cos|':>8} | {'w_recov':>7}")
    print("-" * 68)
    for p in args.pairs:
        rows = [run_one(p, args, s) for s in args.seeds]
        w.writerows(rows); f.flush()
        agg = {k: np.mean([r[k] for r in rows]) for k in
               ("test_acc", "subspace_align", "best_axis_match", "min_abs_basis_cos", "w_recovery")}
        print(f"{p:>10} | {agg['test_acc']:>8.4f} | {agg['subspace_align']:>8.4f} | "
              f"{agg['best_axis_match']:>10.4f} | {agg['min_abs_basis_cos']:>8.4f} | "
              f"{agg['w_recovery']:>7.4f}")
    f.close()

    print(f"\nSaved {args.out}")
    print("Read-out: if subspace_align and axis_match -> ~1.0 with enough pairs/user, LoRe CAN "
          "recover planted axes, and the PRISM null is about the data (signal/volume), not the "
          "method. If they stay low even with 1000 pairs/user and clean axes, the method/optimizer "
          "itself cannot recover known structure.")


if __name__ == "__main__":
    main()
