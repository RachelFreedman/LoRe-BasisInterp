"""
Perturbation recovery test on PRISM response pairs.

Construction
------------
Real PRISM labels are discarded. For each real (chosen, rejected) embedding pair, we take
the real diff d = chosen - rejected, strip out the real component, then inject a controlled signal on top:

  d_clutter = d - proj_onto_span(planted directions)(d)
  d_final   = signal_scale * median_norm * (+/- r_hat) + d_clutter

signal_scale=0 reports each direction's NATURAL separation in real PRISM pairs before any injection. Sweeping signal_scale finds
how large a real effect needs to be, given PRISM's actual noise floor, before recovery kicks in.
Sweeping pairs-per-user (default includes PRISM's real ~15/user regime) checks whether recovery is
gated by data volume the same way the original PRISM null was suspected to be.

Two ground-truth structures (--mode):
  bipolar     : one direction r; group A prefers +r, group B prefers -r (rank-1 recovery problem)
  independent : two distinct directions r1, r2; group A prefers +r1, group B prefers +r2

Two ground-truth sources (--source):
  random  : orthonormal random unit direction(s) -- no semantic content
  concept : named vector(s) from the existing 11-concept library (default: confidence for bipolar,
            confidence + formatting for independent -- the two strongest wbar-aligned concepts in
            the population direction concept-alignment analysis)

Respects the real PRISM train/test prompt-split boundary: synthetic train pairs are drawn only from
train_embeddings.pkl, synthetic test pairs only from test_embeddings.pkl, so no synthetic user can see
the "same" real pair in both halves. 

Metrics, controls, and CSV/print conventions are reused directly from synthetic_recovery.py and
synthetic_concept_users.py -- no new evaluation code, only new data construction.
"""
import argparse
import csv
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(SCRIPT_DIR)
from utils import LoRe_regularized, LoReV2, canonical_variation_axes       # noqa: E402
from synthetic_recovery import (subspace_alignment, best_axis_match,      # noqa: E402
                                collapse_metrics, eval_acc, w_recovery)
from synthetic_concept_users import (group_dir_match, null_recovery,      # noqa: E402
                                     _acc, signed_ground_truth)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def unit(v):
    return v / (v.norm() + 1e-8)


def _to_tensor(x):
    if torch.is_tensor(x):
        return x.clone().detach().to(dtype=torch.float32)
    return torch.tensor(x, dtype=torch.float32)


def load_real_diff_pool(path):
    """All real (chosen - rejected) diffs in a train_embeddings.pkl / test_embeddings.pkl file,
    pooled across every real user -- real labels and user identity are discarded entirely; only
    the response-embedding geometry is kept."""
    data = torch.load(path, weights_only=False)
    diffs = []
    for ex in data:
        info = ex.get("extra_info", {})
        c, r = info.get("chosen_conv_embedding"), info.get("rejected_conv_embedding")
        if c is not None and r is not None:
            diffs.append(_to_tensor(c) - _to_tensor(r))
    return torch.stack(diffs) if diffs else torch.empty(0)


def make_ground_truth(mode, source, dim, concepts, concept_vectors_path, gen):
    """Returns (V_true [dim, n_true_axes], V_signed [dim, 2]) -- V_signed's two columns are each
    group's actual target direction, in group-index order, ready for group_dir_match/w_recovery."""
    if source == "random":
        n_axes = 1 if mode == "bipolar" else 2
        A = torch.randn(dim, n_axes, generator=gen)
        Q, _ = torch.linalg.qr(A)
        V_true = Q[:, :n_axes]
    else:
        cvs = torch.load(concept_vectors_path, weights_only=False)
        missing = [c for c in concepts if c not in cvs]
        if missing:
            raise ValueError(f"concepts not in concept_vectors.pt: {missing}")
        V_true = torch.stack([unit(cvs[c].float()) for c in concepts], dim=1)

    if mode == "bipolar":
        V_signed = signed_ground_truth(V_true)   # [dim, 2] = [r, -r]
    else:
        V_signed = V_true                         # [dim, 2] already, one column per group
    return V_true, V_signed


def make_pairs(n, diff_pool, span_basis, sign, signal_scale, median_norm, gen, noise):
    """n synthetic pairs for one user: real PRISM diffs, own component along the planted
    direction(s) removed, controlled signal re-injected along this user's group direction."""
    idx = torch.randint(0, diff_pool.shape[0], (n,), generator=gen)
    d = diff_pool[idx].clone()
    # remove the full span of planted direction(s) -- both groups start from the same clean clutter
    proj = (d @ span_basis) @ span_basis.T
    clutter = d - proj
    injected = signal_scale * median_norm * sign.unsqueeze(0)
    diffs = injected + clutter
    flip = torch.rand(n, generator=gen) < noise
    diffs[flip] = -diffs[flip]
    return diffs


def build_synthetic_users(train_pool, test_pool, V_signed, users_per_group, pairs_per_user,
                          test_pairs, subset_frac, noise, signal_scale, gen):
    """2 groups (A prefers V_signed[:,0], B prefers V_signed[:,1]); users_per_group each."""
    span_basis, _ = torch.linalg.qr(V_signed)  # orthonormal basis for the planted span
    median_norm = train_pool.norm(dim=1).median().item()

    train_feats, test_feats, group_id = [], [], []
    for g in range(2):
        sign = unit(V_signed[:, g])
        for _ in range(users_per_group):
            n_sub = max(2, int(round(subset_frac * pairs_per_user)))
            tr = make_pairs(n_sub, train_pool, span_basis, sign, signal_scale, median_norm, gen, noise)
            te = make_pairs(test_pairs, test_pool, span_basis, sign, signal_scale, median_norm, gen, noise)
            train_feats.append(tr); test_feats.append(te); group_id.append(g)
    return train_feats, test_feats, torch.tensor(group_id), median_norm


def personal_vs_global(train_feats, test_feats, group_id):
    """Same control as synthetic_concept_users.py: a user's own direction must beat one pooled
    global direction, since the two groups disagree by construction."""
    pooled_mean = torch.cat(train_feats, 0).mean(0)
    global_dir = unit(pooled_mean)
    personal, glob, other = [], [], []
    n = len(train_feats)
    for i in range(n):
        p = unit(train_feats[i].mean(0))
        personal.append(_acc(test_feats[i] @ p))
        glob.append(_acc(test_feats[i] @ global_dir))
        others = [j for j in range(n) if group_id[j] != group_id[i]]
        j = random.choice(others) if others else i
        other.append(_acc(test_feats[i] @ unit(train_feats[j].mean(0))))
    return (float(np.mean(personal)), float(np.mean(glob)), float(np.mean(other)),
            pooled_mean.norm().item())


def run_one(pairs_per_user, args, seed, train_pool, test_pool, V_true, V_signed):
    gen = torch.Generator().manual_seed(seed)
    set_seed(seed)
    dim = train_pool.shape[1]
    n_groups = 2

    train_feats, test_feats, group_id, median_norm = build_synthetic_users(
        train_pool, test_pool, V_signed, args.users_per_group, pairs_per_user, args.test_pairs,
        args.subset_frac, args.noise, args.signal_scale, gen)

    k_fit = args.k_fit
    if args.model == "v2":
        model = LoReV2(len(train_feats), dim, k_fit, lam_pop=args.lam_pop, lam_d=args.lam_d,
                       num_iterations=args.iters, learning_rate=args.lr, verbose=False)
        Wk, Vk = model.train(train_feats)
        V_raw = model.V.detach().cpu()
        axes, _, _ = canonical_variation_axes(model.V, model.delta, k=k_fit)
        V_axes = axes
        W_assign = (model.delta.detach().cpu() @ V_raw.T) @ axes
        signed = True
        r_dirs = model.reward_dirs().detach().cpu().T
    else:
        anchor = torch.cat(train_feats, 0).mean(0).reshape(-1, 1)
        model = LoRe_regularized(anchor, args.alpha, len(train_feats), dim, k_fit, args.iters, args.lr)
        Wk, Vk = model.train(train_feats)
        V_raw = V_axes = model.V.detach().cpu()
        W_assign, signed = model.W, False
        r_dirs = (Vk.detach().cpu() @ Wk.detach().cpu().T).T

    pers, glob, oth, gnorm = personal_vs_global(train_feats, test_feats, group_id)
    return {
        "pairs_per_user": pairs_per_user, "signal_scale": args.signal_scale, "seed": seed,
        "train_acc": eval_acc(Wk, Vk, train_feats),
        "test_acc": eval_acc(Wk, Vk, test_feats),
        "subspace_align": subspace_alignment(V_axes, V_true),
        "best_axis_match": best_axis_match(V_axes, V_true),
        "min_abs_basis_cos": collapse_metrics(V_raw),
        "w_recovery": w_recovery(W_assign, group_id, n_groups, signed=signed),
        "group_dir_match": group_dir_match(r_dirs, group_id, V_signed),
        "personal": pers, "global": glob, "other_group": oth, "global_dir_norm": gnorm,
        "median_real_diff_norm": median_norm,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    default_dir = os.path.join(SCRIPT_DIR, "..", "data", "prism")
    ap.add_argument("--train", default=os.path.join(default_dir, "train_embeddings.pkl"),
                    help="path to CORRECTED train_embeddings.pkl -- source pool for synthetic train pairs")
    ap.add_argument("--test", default=os.path.join(default_dir, "test_embeddings.pkl"),
                    help="path to CORRECTED test_embeddings.pkl -- source pool for synthetic test pairs")
    ap.add_argument("--concept_vectors", default=os.path.join(default_dir, "concept_vectors.pt"))
    ap.add_argument("--mode", choices=["bipolar", "independent"], required=True)
    ap.add_argument("--source", choices=["random", "concept"], required=True)
    ap.add_argument("--concepts", nargs="+", default=None,
                    help="required if --source concept: 1 concept for bipolar, 2 for independent")
    ap.add_argument("--gt_seed", type=int, default=0, help="seed for the random ground-truth direction(s)")
    ap.add_argument("--users_per_group", type=int, default=20)
    ap.add_argument("--pairs", type=int, nargs="+", default=[15, 50, 100, 400, 1000],
                    help="pairs/user to sweep -- 15 matches PRISM's real regime")
    ap.add_argument("--test_pairs", type=int, default=200)
    ap.add_argument("--subset_frac", type=float, default=1.0,
                    help="fraction of pairs_per_user each user actually gets (independent draws)")
    ap.add_argument("--noise", type=float, default=0.1, help="fraction of flipped labels")
    ap.add_argument("--signal", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 4.0],
                    help="injected-signal scale(s) to sweep, as a multiple of the median real diff norm; "
                         "0.0 reports natural separation with no injection")
    ap.add_argument("--k_fit", type=int, default=4)
    ap.add_argument("--model", choices=["vanilla", "v2"], default="v2")
    ap.add_argument("--alpha", type=float, default=0.0, help="vanilla only")
    ap.add_argument("--lam_pop", type=float, default=0.01, help="v2 only")
    ap.add_argument("--lam_d", type=float, default=0.01, help="v2 only")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=None, help="default 0.5 vanilla, 1e-2 v2")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.lr is None:
        args.lr = 1e-2 if args.model == "v2" else 0.5
    if args.source == "concept":
        n_needed = 1 if args.mode == "bipolar" else 2
        if not args.concepts:
            args.concepts = (["confidence"] if args.mode == "bipolar"
                             else ["confidence", "formatting"])
        if len(args.concepts) != n_needed:
            raise ValueError(f"--mode {args.mode} needs exactly {n_needed} concept(s), "
                             f"got {args.concepts}")
    if args.out is None:
        args.out = os.path.join(SCRIPT_DIR, "..", "results", "synthetic",
                                f"prism_perturbation_{args.mode}_{args.source}_{args.model}.csv")

    print(f"Loading real PRISM diff pools:\n  train: {args.train}\n  test:  {args.test}")
    train_pool = load_real_diff_pool(args.train)
    test_pool = load_real_diff_pool(args.test)
    print(f"{train_pool.shape[0]} real train diffs, {test_pool.shape[0]} real test diffs "
          f"(dim={train_pool.shape[1]})")

    gt_gen = torch.Generator().manual_seed(args.gt_seed)
    V_true, V_signed = make_ground_truth(args.mode, args.source, train_pool.shape[1],
                                         args.concepts, args.concept_vectors, gt_gen)
    label = args.concepts if args.source == "concept" else f"random(gt_seed={args.gt_seed})"
    print(f"mode={args.mode} source={args.source} directions={label} "
          f"users_per_group={args.users_per_group} k_fit={args.k_fit} model={args.model}")
    print(f"w_recovery chance = 0.500 (2 groups)\n")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    f = open(args.out, "w", newline="")
    fields = ["pairs_per_user", "signal_scale", "seed", "train_acc", "test_acc", "subspace_align",
              "best_axis_match", "min_abs_basis_cos", "w_recovery", "group_dir_match", "personal",
              "global", "other_group", "global_dir_norm", "median_real_diff_norm"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()

    print(f"{'signal':>7} | {'pairs/usr':>9} | {'test_acc':>8} | {'subspace':>8} | "
          f"{'axis':>6} | {'w_recov':>7} | {'grp_dir':>7} | {'personal':>8} | {'global':>7}")
    print("-" * 92)
    for s in args.signal:
        args.signal_scale = s
        for p in args.pairs:
            rows = [run_one(p, args, seed, train_pool, test_pool, V_true, V_signed)
                   for seed in args.seeds]
            w.writerows(rows); f.flush()
            agg = {k: np.mean([r[k] for r in rows]) for k in
                   ("test_acc", "subspace_align", "best_axis_match", "w_recovery",
                    "group_dir_match", "personal", "global")}
            print(f"{s:>7.2f} | {p:>9} | {agg['test_acc']:>8.4f} | {agg['subspace_align']:>8.4f} | "
                  f"{agg['best_axis_match']:>6.4f} | {agg['w_recovery']:>7.4f} | "
                  f"{agg['group_dir_match']:>7.4f} | {agg['personal']:>8.4f} | {agg['global']:>7.4f}")
    f.close()

    null_sub, null_axis = null_recovery(V_true, args.k_fit)
    print(f"\nrandom-basis floor: subspace={null_sub:.4f}  axis_match={null_axis:.4f}")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
