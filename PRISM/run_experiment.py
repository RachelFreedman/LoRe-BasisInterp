#!/usr/bin/env python3
"""
Quick LoRe-PRISM experiment runner.

Runs the PRISM basis-learning experiment 3 times, varying ONE hyperparameter
(the number of basis functions K by default), then prints a summary table.

Prerequisite (one-time, slow): the cached embeddings must already exist:
    data/prism/train_embeddings.pkl
    data/prism/test_embeddings.pkl
produced by:  python prepare.py  &&  python generate-prism-embeddings.py

Run from inside the PRISM/ directory:
    python run_experiment.py
"""
import os
import sys
import random
import numpy as np
import torch
from collections import defaultdict

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Make utils.py importable (same trick train_basis.py uses)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils import solve_regularized_simplex  # noqa: E402

# ---------------------------------------------------------------------------
# EXPERIMENT CONFIG  --  edit these rows to change the experiment.
# Each run changes ONE knob. Default: vary K (number of basis functions),
# seed held fixed. To study the seed instead, set all K equal and vary "seed".
# ---------------------------------------------------------------------------
ALPHA = 1e4        # regularization strength toward the base reward (fixed)
NUM_ITERS = 20000  # optimization steps (paper uses 20000; lower => faster)
LR = 0.5           # learning rate

RUNS = [
    {"label": "K=5",  "K": 5,  "seed": 0},
    {"label": "K=10", "K": 10, "seed": 0},
    {"label": "K=20", "K": 20, "seed": 0},
]
# ---------------------------------------------------------------------------


def set_seed(s):
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


def main():
    print(f"Device: {device}")
    print("Loading cached embeddings...")
    train_emb = torch.load("data/prism/train_embeddings.pkl")
    test_emb = torch.load("data/prism/test_embeddings.pkl")

    train_seen = group_embeddings_by_user(train_emb, seen_value=True, split_name="train")
    test_seen = group_embeddings_by_user(test_emb, seen_value=True, split_name="test")
    N = len(train_seen)
    print(f"Seen users: {N}")

    print("Loading backbone once for reference direction (V_sft)...")
    V_final = get_reference_direction()

    results = []
    for cfg in RUNS:
        print("\n" + "=" * 64)
        print(f"RUN {cfg['label']}  (K={cfg['K']}, seed={cfg['seed']}, alpha={ALPHA}, iters={NUM_ITERS})")
        print("=" * 64)
        set_seed(cfg["seed"])
        W, V = solve_regularized_simplex(
            V_final, ALPHA, train_seen, cfg["K"],
            num_iterations=NUM_ITERS, learning_rate=LR,
        )
        train_acc, _ = accuracy(W, V, train_seen)
        test_acc, test_std = accuracy(W, V, test_seen)
        kept = V.shape[1]  # bases surviving the pruning step
        results.append((cfg["label"], cfg["K"], cfg["seed"], kept, train_acc, test_acc, test_std))

    print("\n\n" + "#" * 64)
    print("# SUMMARY  (test = seen users, UNSEEN prompts -- the generalization metric)")
    print("#" * 64)
    header = f"{'run':>7} | {'K':>3} | {'seed':>4} | {'bases_kept':>10} | {'train_acc':>9} | {'test_acc':>9} | {'test_std':>8}"
    print(header)
    print("-" * len(header))
    for label, K, seed, kept, tr, te, std in results:
        print(f"{label:>7} | {K:>3} | {seed:>4} | {kept:>10} | {tr:>9.4f} | {te:>9.4f} | {std:>8.4f}")


if __name__ == "__main__":
    main()
