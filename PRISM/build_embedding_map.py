"""
build_embedding_map.py

1. Loads train/test embeddings
2. Builds embedding_map.pkl: lookup dict keyed by (user_id, dialog_id, turn_nb)
   with conversation text + chosen/rejected/diff embeddings
3. Trains LoRe bases for all (K, alpha) combinations using solve_regularized_simplex
4. Scores all N conversations against all K basis vectors for every (K, alpha) run
5. Saves basis_scores.pkl: nested dict keyed by "K{K}_alpha{alpha}"

Usage (run from PRISM/ directory on GPU cluster):
    python build_embedding_map.py

Requires:
    data/prism/train_embeddings.pkl
    data/prism/test_embeddings.pkl
    Skywork model (for V_sft extraction)
    ../utils.py

Outputs:
    embedding_map.pkl    -- flat lookup dict, one entry per conversation turn
    basis_scores.pkl     -- dict keyed by "K{K}_alpha{alpha}", each with:
                              "keys"   : list of (user_id, dialog_id, turn_nb)
                              "scores" : [N, K] tensor
                              "V"      : [4096, K] basis matrix
                              "K"      : int
                              "alpha"  : float
"""

import os
import sys
import torch
import torch.nn.functional as F
import pickle
from collections import defaultdict
from transformers import AutoModel

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_PKL   = "data/prism/train_embeddings.pkl"
TEST_PKL    = "data/prism/test_embeddings.pkl"
MODEL_NAME  = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
DEVICE      = os.environ.get("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
SEED        = 42

K_LIST      = [1, 5, 10, 15, 20, 25, 50]
ALPHA_LIST  = [1e3, 1e4, 1e5]
NUM_ITERS   = 20000
LR          = 0.5

OUT_MAP     = "embedding_map.pkl"
OUT_SCORES  = "basis_scores.pkl"
# ──────────────────────────────────────────────────────────────────────────────

# import utils from parent directory (same as train_basis.py does)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils import solve_regularized_simplex, set_seed


# Load embeddings

def load_pkl(path):
    print(f"Loading {path} ...")
    data = torch.load(path, map_location="cpu")
    print(f"  → {len(data)} entries")
    return data


# Build flat embedding map 

def build_embedding_map(train_data, test_data):
    """
    Returns a dict keyed by (user_id, dialog_id, turn_nb).
    Each value has: chosen/rejected/diff embeddings + conversation text + metadata.
    """
    embedding_map = {}

    for split_name, dataset in [("train", train_data), ("test", test_data)]:
        for entry in dataset:
            info = entry["extra_info"]
            key = (info["user_id"], info["dialog_id"], info["turn_nb"])

            chosen_emb   = info.get("chosen_conv_embedding")
            rejected_emb = info.get("rejected_conv_embedding")
            if chosen_emb is None or rejected_emb is None:
                continue

            if not isinstance(chosen_emb, torch.Tensor):
                chosen_emb   = torch.tensor(chosen_emb,   dtype=torch.float32)
                rejected_emb = torch.tensor(rejected_emb, dtype=torch.float32)
            else:
                chosen_emb   = chosen_emb.to(torch.float32)
                rejected_emb = rejected_emb.to(torch.float32)

            embedding_map[key] = {
                "chosen_embedding":   chosen_emb,
                "rejected_embedding": rejected_emb,
                "diff_embedding":     chosen_emb - rejected_emb,
                "prompt":             entry.get("prompt", []),
                "chosen_utterance":   info.get("chosen_utterance", ""),
                "rejected_utterance": info.get("rejected_utterance", []),
                "user_id":   info["user_id"],
                "dialog_id": info["dialog_id"],
                "turn_nb":   info["turn_nb"],
                "split":     split_name,
                "seen":      info.get("seen", None),
            }

    print(f"embedding_map: {len(embedding_map)} entries")
    return embedding_map


# Group embeddings by user 

def group_by_user(train_data, test_data, device):
    """Returns train_seen, train_unseen, test_seen, test_unseen as lists of tensors."""
    def process(dataset, seen_value, split_name):
        grouped = defaultdict(list)
        for entry in dataset:
            info = entry.get("extra_info", {})
            if info.get("seen") == seen_value and info.get("split") == split_name:
                user_id = info.get("user_id")
                if user_id:
                    chosen   = torch.tensor(info["chosen_conv_embedding"],   dtype=torch.float32, device=device)
                    rejected = torch.tensor(info["rejected_conv_embedding"],  dtype=torch.float32, device=device)
                    grouped[user_id].append(chosen - rejected)
        result = []
        for uid in sorted(grouped.keys()):
            result.append(torch.stack(grouped[uid]))
        return result

    train_seen   = process(train_data, seen_value=True,  split_name="train")
    train_unseen = process(train_data, seen_value=False, split_name="train")
    test_seen    = process(test_data,  seen_value=True,  split_name="test")
    test_unseen  = process(test_data,  seen_value=False, split_name="test")

    print(f"  train_seen users:   {len(train_seen)}")
    print(f"  train_unseen users: {len(train_unseen)}")
    print(f"  test_seen users:    {len(test_seen)}")
    print(f"  test_unseen users:  {len(test_unseen)}")
    return train_seen, train_unseen, test_seen, test_unseen


# Extract V_sft from Skywork backbone 

def get_v_sft(model_name, device):
    print(f"\nLoading backbone {model_name} to extract V_sft ...")
    rm = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
        num_labels=1,
    )
    last_linear = None
    for _, module in rm.named_modules():
        if isinstance(module, torch.nn.Linear):
            last_linear = module
    V_sft = last_linear.weight[:, 0].to(device).to(torch.float32).reshape(-1, 1)
    print(f"  V_sft shape: {V_sft.shape}")
    del rm  # free GPU memory
    torch.cuda.empty_cache()
    return V_sft


# Score all conversations against a basis V 

def score_all(embedding_map, V):
    """
    Returns:
        keys   : list of (user_id, dialog_id, turn_nb)
        scores : [N, K] float tensor  (on CPU)
    """
    keys  = list(embedding_map.keys())
    diffs = torch.stack([embedding_map[k]["diff_embedding"] for k in keys])  # [N, 4096]
    V_cpu = V.cpu().to(torch.float32)
    scores = diffs @ V_cpu  # [N, K]
    return keys, scores


def main():
    # load embeddings
    train_data = load_pkl(TRAIN_PKL)
    test_data  = load_pkl(TEST_PKL)

    # build flat lookup dict
    print("\nBuilding embedding map ...")
    embedding_map = build_embedding_map(train_data, test_data)
    with open(OUT_MAP, "wb") as f:
        pickle.dump(embedding_map, f)
    print(f"Saved embedding_map → {OUT_MAP}")

    # group by user for LoRe training
    print("\nGrouping embeddings by user ...")
    train_seen, train_unseen, test_seen, test_unseen = group_by_user(
        train_data, test_data, DEVICE
    )

    # extract V_sft (backbone reward direction) — loaded once, shared across all runs
    V_sft = get_v_sft(MODEL_NAME, DEVICE)

    # train LoRe bases for all (K, alpha) and score
    all_scores = {}
    total_runs = len(K_LIST) * len(ALPHA_LIST)
    run_idx = 0

    for alpha in ALPHA_LIST:
        for K in K_LIST:
            run_idx += 1
            run_key = f"K{K}_alpha{alpha:.0e}"
            print(f"\n{'='*60}")
            print(f"[{run_idx}/{total_runs}] Training: K={K}, alpha={alpha:.0e}")
            print(f"{'='*60}")

            set_seed(SEED)

            if K == 0:
                V_joint = V_sft
            else:
                _, V_joint = solve_regularized_simplex(
                    V_sft, alpha, train_seen, K,
                    num_iterations=NUM_ITERS,
                    learning_rate=LR
                )

            V_joint = V_joint.detach().cpu().to(torch.float32)
            print(f"  V_joint shape: {V_joint.shape}")

            # score all conversations
            keys, scores = score_all(embedding_map, V_joint)
            print(f"  Score matrix: {scores.shape}")
            for i in range(scores.shape[1]):
                col = scores[:, i]
                print(f"    basis_{i:2d} | min={col.min():.4f} max={col.max():.4f} "
                      f"mean={col.mean():.4f} std={col.std():.4f}")

            all_scores[run_key] = {
                "keys":   keys,
                "scores": scores,
                "V":      V_joint,
                "K":      K,
                "alpha":  alpha,
            }

    with open(OUT_SCORES, "wb") as f:
        pickle.dump(all_scores, f)
    print(f"\nSaved basis_scores → {OUT_SCORES}")

    # 7. summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for run_key, data in all_scores.items():
        print(f"  {run_key:25s}: scores {data['scores'].shape}, V {data['V'].shape}")


if __name__ == "__main__":
    main()
