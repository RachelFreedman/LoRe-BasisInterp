"""
build_embedding_map.py

Builds a lookup dictionary from the train/test embedding pkl files that maps
each (user_id, dialog_id, turn_nb) to its embeddings + full conversation text.

Also computes basis scores for every conversation against every basis vector
and saves the full score matrix.

Usage:
    python build_embedding_map.py

Outputs:
    embedding_map.pkl       -- full lookup dict, one entry per conversation turn
    basis_scores.pkl        -- dict with:
                                 "keys"   : list of (user_id, dialog_id, turn_nb)
                                 "scores" : [N, K] tensor of basis scores
                                 "V"      : [4096, K] basis matrix

"""

import torch
import pickle

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_PKL     = "data/prism/train_embeddings.pkl"
TEST_PKL      = "data/prism/test_embeddings.pkl"
BASIS_PT      = "basis_matrices.pt"
BASIS_RUN_KEY = "PART2_K10_seed42"  # ['PART1_K5_seed42', 'PART1_K10_seed42', 'PART1_K20_seed42','PART2_K10_seed0', 'PART2_K10_seed1', 'PART2_K10_seed2', 'PART2_K10_seed42']
OUT_MAP       = "embedding_map.pkl"
OUT_SCORES    = "basis_scores.pkl"
# ──────────────────────────────────────────────────────────────────────────────


def load_pkl(path):
    print(f"Loading {path} ...")
    data = torch.load(path, map_location="cpu")
    print(f"  → {len(data)} entries")
    return data


def build_embedding_map(train_data, test_data):
    """
    Returns a dict keyed by (user_id, dialog_id, turn_nb).
    Each value contains:
        chosen_embedding    [4096] float tensor
        rejected_embedding  [4096] float tensor
        diff_embedding      chosen - rejected  [4096]
        prompt              list of {"role", "content"} dicts
        chosen_utterance    str
        rejected_utterance  list[str]
        user_id, dialog_id, turn_nb, split, seen
    """
    embedding_map = {}

    for split_name, dataset in [("train", train_data), ("test", test_data)]:
        for entry in dataset:
            info = entry["extra_info"]

            key = (
                info["user_id"],
                info["dialog_id"],
                info["turn_nb"],
            )

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

    print(f"\nembedding_map: {len(embedding_map)} entries")
    return embedding_map


def score_all_conversations(embedding_map, V):
    """
    Score every conversation against every basis vector.
        score[n, i] = diff_embedding[n] @ V[:, i]

    Returns:
        keys   : list of (user_id, dialog_id, turn_nb) in row order
        scores : [N, K] float tensor
    """
    K    = V.shape[1]
    keys = list(embedding_map.keys())
    N    = len(keys)

    # stack all diff embeddings: [N, 4096]
    diffs = torch.stack([embedding_map[k]["diff_embedding"] for k in keys])

    # full score matrix: [N, K]
    scores = diffs @ V

    print(f"  Score matrix shape: {scores.shape}  ({N} conversations × {K} bases)")
    for i in range(K):
        col = scores[:, i]
        print(f"  basis_{i:2d} | min: {col.min():.4f}  max: {col.max():.4f}"
              f"  mean: {col.mean():.4f}  std: {col.std():.4f}")

    return keys, scores


def main():
    #load embeddings
    train_data = load_pkl(TRAIN_PKL)
    test_data  = load_pkl(TEST_PKL)

    #build lookup dict
    embedding_map = build_embedding_map(train_data, test_data)

    #save embedding map
    with open(OUT_MAP, "wb") as f:
        pickle.dump(embedding_map, f)
    print(f"\nSaved embedding_map → {OUT_MAP}")

    #load basis vectors
    print(f"\nLoading basis matrices from {BASIS_PT} (run: {BASIS_RUN_KEY}) ...")
    matrices = torch.load(BASIS_PT, map_location="cpu")
    V = matrices[BASIS_RUN_KEY]["V"].to(torch.float32)  # [4096, K]
    print(f"  V shape: {V.shape}")

    #score all conversations against all bases
    print(f"\nScoring {len(embedding_map)} conversations against {V.shape[1]} bases ...")
    keys, scores = score_all_conversations(embedding_map, V)

    #save full score matrix
    with open(OUT_SCORES, "wb") as f:
        pickle.dump({
            "keys":   keys,    # list of (user_id, dialog_id, turn_nb)
            "scores": scores,  # [N, K] tensor
            "V":      V,       # [4096, K] basis matrix
        }, f)
    print(f"\nSaved basis_scores → {OUT_SCORES}")
    print(f"  keys:   list of {len(keys)} (user_id, dialog_id, turn_nb) tuples")
    print(f"  scores: {scores.shape} tensor — every conversation × every basis")
    print(f"  V:      {V.shape} basis matrix")

if __name__ == "__main__":
    main()
