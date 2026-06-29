#!/usr/bin/env python3
"""
LoRe-PRISM basis reproducibility sanity check (two-part).

PURPOSE
-------
Confirm that everyone on the team learns the SAME basis reward functions from
the PRISM preference data. Each teammate runs this script and compares the
`basis_fp` column for the agreed configuration.

The script runs TWO parts in a single invocation:

  PART 1 - RANK SWEEP  (vary K, hold the seed fixed)
      Sweeps the number of basis functions K (the rank of V). Shows how
      generalization accuracy changes with rank. Each row uses a different K,
      so the learned bases are EXPECTED to differ between rows -- this part is
      about the rank/accuracy trade-off, not about matching teammates.

  PART 2 - SEED VARIANCE / TEAM REPRODUCIBILITY  (hold K fixed, vary the seed)
      The original LoRe code does NOT lock the random seed, so the basis
      initialization -- and therefore the learned bases -- randomizes on every
      run. We lock it here (see set_seed). Seeds 0,1,2 measure how much the
      seed actually moves the result; seed 42 is the TEAM-AGREED value. Compare
      your seed=42 row's basis_fp against your teammates' seed=42 row to confirm
      you all converge to the same bases.

HYPERPARAMETERS (documented so the script is self-explanatory)
--------------------------------------------------------------
  K            number of basis functions = rank of the basis matrix V. The
               central LoRe knob. (PART 1 sweeps it; PART 2 fixes it.)
  seed         RNG seed controlling basis initialization during training. MUST
               match across teammates to reproduce the same bases. Team value = 42.
  alpha        strength of regularization pulling the bases toward the backbone's
               single SFT reward direction (V_sft). Default 1e4 (repo default).
  iters        optimization steps for the alternating W / V solve. Default 20000.
  lr           learning rate for that solve. Default 0.5.
  data split   fixed by prepare.py at seed=123. Do NOT change it -- it keeps the
               train/test user split identical for everyone on the team.

  bases_kept   how many of the K basis functions survive pruning: a basis is
               dropped if EVERY user gives it <1% weight (softmax(W) max < 1e-2,
               see utils.py). This is informational -- the model often collapses
               onto fewer "effective" bases than K. It can differ between
               seeds/machines, which is exactly why it must NOT be the thing you
               compare.

  basis_fp     = ||V||_F over the FULL [4096, K] basis matrix (ALL K columns,
               BEFORE pruning) -- so it includes sub-threshold bases and is
               comparable across teammates even when bases_kept differs. It is
               invariant to column permutation and sign, so it survives the
               harmless reordering of bases between runs while still flagging a
               genuinely different fit. Same GPU + same seed -> matches tightly;
               different GPUs -> agree to ~3-4 decimals (CUDA float
               nondeterminism), NOT bitwise. So compare approximately. The full
               matrices themselves are also saved (see OUTPUT) for direct
               inspection.

PREREQUISITE (one-time, slow): the cached embeddings must already exist at
    data/prism/train_embeddings.pkl
    data/prism/test_embeddings.pkl
produced by:  python prepare.py  &&  python generate-prism-embeddings.py

OUTPUT
------
Prints a summary table for each part AND writes:
  - basis_reproducibility_results.csv : the summary rows (gitignored)
  - basis_matrices.pt                 : the FULL [4096, K] basis matrix for every
                                        run, keyed by run (e.g. PART2_K10_seed42),
                                        so you can load and compare the actual
                                        basis vectors -- including sub-threshold
                                        ones -- not just the scalar fingerprint.
Both are gitignored; copy them to your local machine to share / compare.

USAGE (run from inside the PRISM/ directory)
--------------------------------------------
    python basis_reproducibility_check.py
    python basis_reproducibility_check.py --rank-values 1,5,10,20,50
    python basis_reproducibility_check.py --seed-values 0,1,2,42 --fixed-K 10
    python basis_reproducibility_check.py --out my_results.csv
"""
import os
import sys
import csv
import argparse
import random
import numpy as np
import torch
from collections import defaultdict

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Make utils.py importable (same trick train_basis.py uses)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils import LoRe_regularized  # noqa: E402


def set_seed(s):
    """Lock every RNG so basis initialization (and the learned bases) is
    deterministic. This is the fix for LoRe's default non-seeded training that
    the team agreed on -- without it the bases randomize on every run."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def group_embeddings_by_user(dataset, seen_value, split_name):
    """Per user, stack the (chosen - rejected) embedding diffs into [m_i, 4096]."""
    grouped = defaultdict(list)
    for ex in dataset:
        info = ex.get("extra_info", {})
        if info.get("seen") == seen_value and info.get("split") == split_name:
            uid = info.get("user_id")
            if uid:
                chosen = torch.tensor(info["chosen_conv_embedding"], dtype=torch.float32, device=device)
                rejected = torch.tensor(info["rejected_conv_embedding"], dtype=torch.float32, device=device)
                grouped[uid].append(chosen - rejected)
    return [torch.stack(grouped[u]) for u in sorted(grouped.keys())]


def get_reference_direction():
    """The single global reward direction = backbone's final linear-layer weight.
    Used as the regularization anchor (V_sft). Loads the 8B backbone once."""
    from transformers import AutoModel
    model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
    rm = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
        num_labels=1,
    )
    last = None
    for _, m in rm.named_modules():
        if isinstance(m, torch.nn.Linear):
            last = m
    return last.weight[:, 0].to(device).to(torch.float32).reshape(-1, 1)


def accuracy(W, V, features):
    """Device-safe pairwise accuracy. A pair is correct when the personalized
    reward of (chosen - rejected) is positive, i.e. chosen scored higher."""
    accs = []
    for i, X in enumerate(features):
        X = X.to(V.device, dtype=V.dtype)
        scores = X @ (V @ W[i])  # [m_i]
        accs.append((scores > 0).float().mean().item())
    return float(np.mean(accs)), float(np.std(accs))


def run_one(part, label, K, seed, V_final, train_seen, test_seen, args):
    """Train one basis set under a given (K, seed). Returns (row, V_full) where
    V_full is the FULL [4096, K] basis matrix BEFORE pruning -- that full matrix
    is the thing teammates should compare, since pruning can keep a different
    number of columns on different machines/seeds."""
    print("\n" + "=" * 64)
    print(f"[{part}] {label}  (K={K}, seed={seed}, alpha={args.alpha}, "
          f"iters={args.iters}, lr={args.lr})")
    print("=" * 64)
    set_seed(seed)  # lock RNG BEFORE the solve so the basis init is reproducible
    # Build the solver directly (instead of solve_regularized_simplex) so we can
    # grab the FULL basis matrix am.V (all K columns). solve_regularized_simplex
    # would only hand back the pruned columns, hiding the sub-threshold bases.
    am = LoRe_regularized(V_final, args.alpha, len(train_seen), 4096, K,
                          args.iters, args.lr)
    W_kept, V_kept = am.train(train_seen)   # pruned: what downstream eval uses
    V_full = am.V.detach()                  # [4096, K] -- every basis column

    # Accuracy uses the pruned (canonical) bases; the dropped near-zero-weight
    # columns add ~nothing, so this matches the repo's reported metric.
    train_acc, _ = accuracy(W_kept, V_kept, train_seen)
    test_acc, test_std = accuracy(W_kept, V_kept, test_seen)
    row = {
        "part": part, "run": label, "K": K, "seed": seed,
        "alpha": args.alpha, "iters": args.iters, "lr": args.lr,
        "bases_kept": V_kept.shape[1],                        # survived pruning (info)
        "basis_fp": float(torch.linalg.norm(V_full).item()),  # ||V||_F over ALL K cols
        "train_acc": train_acc, "test_acc": test_acc, "test_std": test_std,
    }
    return row, V_full.cpu()


def print_table(title, rows):
    print("\n\n" + "#" * 80)
    print(f"# {title}")
    print("#" * 80)
    header = (f"{'run':>9} | {'K':>3} | {'seed':>4} | {'bases_kept':>10} | "
              f"{'basis_fp':>10} | {'train_acc':>9} | {'test_acc':>9} | {'test_std':>8}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['run']:>9} | {r['K']:>3} | {r['seed']:>4} | {r['bases_kept']:>10} | "
              f"{r['basis_fp']:>10.4f} | {r['train_acc']:>9.4f} | "
              f"{r['test_acc']:>9.4f} | {r['test_std']:>8.4f}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Two-part LoRe-PRISM basis reproducibility check.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # PART 1 -- rank sweep
    p.add_argument("--rank-values", type=lambda s: [int(x) for x in s.split(",")],
                   default=[5, 10, 20],
                   help="PART 1: comma-separated K values to sweep (seed held fixed)")
    p.add_argument("--rank-seed", type=int, default=42,
                   help="PART 1: seed held fixed while sweeping K")
    # PART 2 -- seed variance / team reproducibility
    p.add_argument("--seed-values", type=lambda s: [int(x) for x in s.split(",")],
                   default=[0, 1, 2, 42],
                   help="PART 2: comma-separated seeds to sweep (42 = team-agreed value)")
    p.add_argument("--fixed-K", type=int, default=10,
                   help="PART 2: K held fixed while sweeping the seed")
    # shared hyperparameters (documented in the module docstring)
    p.add_argument("--alpha", type=float, default=1e4, help="regularization strength")
    p.add_argument("--iters", type=int, default=20000, help="optimization steps")
    p.add_argument("--lr", type=float, default=0.5, help="learning rate")
    p.add_argument("--out", default="basis_reproducibility_results.csv",
                   help="CSV path for results (gitignored; copy to your machine when done)")
    p.add_argument("--matrices-out", default="basis_matrices.pt",
                   help="path to save the FULL [4096,K] basis matrix per run "
                        "(gitignored; copy down to inspect/compare ALL bases, "
                        "including sub-threshold ones)")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Device: {device}")
    print(f"Hyperparameters: alpha={args.alpha} iters={args.iters} lr={args.lr} "
          f"(data split fixed by prepare.py seed=123)")
    print("Loading cached embeddings...")
    train_emb = torch.load("data/prism/train_embeddings.pkl")
    test_emb = torch.load("data/prism/test_embeddings.pkl")
    train_seen = group_embeddings_by_user(train_emb, seen_value=True, split_name="train")
    test_seen = group_embeddings_by_user(test_emb, seen_value=True, split_name="test")
    print(f"Seen users: {len(train_seen)}")

    print("Loading backbone once for reference direction (V_sft)...")
    V_final = get_reference_direction()

    part1, part2 = [], []
    matrices = {}  # run-key -> full [4096, K] basis matrix, saved for inspection

    # ---- PART 1: rank sweep (vary K, fixed seed) -- bases differ by design ----
    for K in args.rank_values:
        row, V_full = run_one("PART1-rank", f"K={K}", K, args.rank_seed,
                              V_final, train_seen, test_seen, args)
        part1.append(row)
        matrices[f"PART1_K{K}_seed{args.rank_seed}"] = V_full

    # ---- PART 2: seed variance / team reproducibility (fixed K, vary seed) ----
    for s in args.seed_values:
        row, V_full = run_one("PART2-seed", f"seed={s}", args.fixed_K, s,
                              V_final, train_seen, test_seen, args)
        part2.append(row)
        matrices[f"PART2_K{args.fixed_K}_seed{s}"] = V_full

    print_table(
        f"PART 1  RANK SWEEP  (seed fixed at {args.rank_seed}; bases differ between rows by design)",
        part1)
    print_table(
        f"PART 2  SEED VARIANCE  (K fixed at {args.fixed_K}; compare the seed=42 row to teammates)",
        part2)

    # ---- save the FULL basis matrices (all K columns, pre-pruning) ----
    torch.save(matrices, args.matrices_out)
    print(f"\nSaved {len(matrices)} full basis matrices (each [4096, K], "
          f"pre-pruning) to {args.matrices_out}")
    print("  load with: torch.load('basis_matrices.pt') -> dict[run_key] = V[4096,K]")

    # ---- write combined CSV (gitignored; copy to your machine to share) ----
    fields = ["part", "run", "K", "seed", "alpha", "iters", "lr",
              "bases_kept", "basis_fp", "train_acc", "test_acc", "test_std"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in part1 + part2:
            row = dict(r)
            for k in ("basis_fp", "train_acc", "test_acc", "test_std"):
                row[k] = f"{row[k]:.4f}"
            writer.writerow(row)
    print(f"\nWrote {len(part1) + len(part2)} rows to {args.out}")
    print("basis_fp = ||V||_F (permutation/sign-invariant). Compare the PART 2 "
          "seed=42 row against teammates to confirm matching bases.")


if __name__ == "__main__":
    main()
