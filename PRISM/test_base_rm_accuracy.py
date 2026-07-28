"""
Experiment 1: Does the Base RM (V_sft) Already Get High Accuracy?

Tests whether the pretrained Skywork reward head alone can classify
PRISM chosen vs rejected responses, without any LoRe training.
"""
import torch
import numpy as np
from collections import defaultdict

device = "cpu"

def group_embeddings_by_user(dataset, seen_value, split_name):
    """Replicates the grouping logic from train_basis.py"""
    grouped = defaultdict(lambda: {"embeddings": []})
    for example in dataset:
        extra_info = example.get("extra_info", {})
        if extra_info.get("seen") == seen_value and extra_info.get("split") == split_name:
            user_id = extra_info.get("user_id")
            if user_id:
                chosen = torch.tensor(extra_info["chosen_conv_embedding"], dtype=torch.float32)
                rejected = torch.tensor(extra_info["rejected_conv_embedding"], dtype=torch.float32)
                grouped[user_id]["embeddings"].append(chosen - rejected)
    
    sorted_grouped = []
    for user_id in sorted(grouped.keys()):
        sorted_grouped.append(torch.stack(grouped[user_id]["embeddings"]))
    return sorted_grouped

def evaluate_with_vector(test_features, V):
    """Evaluate accuracy using a single reward direction V."""
    accuracies = []
    for user_diffs in test_features:
        # user_diffs: [num_pairs, 4096] (chosen - rejected embeddings)
        scores = user_diffs @ V  # [num_pairs, 1]
        correct = (scores > 0).float().mean().item()
        accuracies.append(correct)
    return accuracies

print("Loading embeddings...")
train_embeddings = torch.load("data/prism/train_embeddings.pkl", map_location="cpu", weights_only=False)
test_embeddings = torch.load("data/prism/test_embeddings.pkl", map_location="cpu", weights_only=False)

print("Grouping by user...")
test_seen = group_embeddings_by_user(test_embeddings, seen_value=True, split_name="test")
test_unseen = group_embeddings_by_user(test_embeddings, seen_value=False, split_name="test")

print(f"Seen users (test): {len(test_seen)}")
print(f"Unseen users (test): {len(test_unseen)}")

# --- Test 1: V_sft (the "grabbed" anchor from train_basis.py) ---
print("\n" + "=" * 60)
print("Loading Skywork model to extract V_sft...")
from transformers import AutoModel
model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
rm = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation="eager", num_labels=1)

last_linear_layer = None
for name, module in rm.named_modules():
    if isinstance(module, torch.nn.Linear):
        last_linear_layer = module

V_sft = last_linear_layer.weight[:, 0].to(torch.float32).reshape(-1, 1)
print(f"V_sft shape: {V_sft.shape}")

# Also grab the ACTUAL reward head for comparison
print("\nExtracting actual reward head...")
for name, module in rm.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(f"  Linear layer: {name}, shape: {module.weight.shape}")

del rm  # free memory

# --- Evaluate V_sft ---
print("\n" + "=" * 60)
print("EXPERIMENT 1: Base RM (V_sft) Accuracy")
print("=" * 60)

acc_seen = evaluate_with_vector(test_seen, V_sft)
acc_unseen = evaluate_with_vector(test_unseen, V_sft)

print(f"\nV_sft on SEEN users (test):   {np.mean(acc_seen):.4f} +/- {np.std(acc_seen):.4f}")
print(f"V_sft on UNSEEN users (test): {np.mean(acc_unseen):.4f} +/- {np.std(acc_unseen):.4f}")

# --- Test 2: Compare with LoRe K=20 collapsed direction ---
print("\n" + "=" * 60)
print("COMPARISON: LoRe K=20 (Collapsed) Direction")
print("=" * 60)

V_lore = torch.load("checkpoints/checkpoints/PRISM_V_lore_K_20_alpha_10000.0.pt", map_location="cpu", weights_only=True)
# Since all bases are identical, just use the first one
V_lore_single = V_lore[:, 0:1]  # [4096, 1]
print(f"V_lore shape: {V_lore.shape}, using first column: {V_lore_single.shape}")

# Check cosine sim between V_sft and collapsed direction
import torch.nn.functional as F
cos_sim = F.cosine_similarity(V_sft.squeeze(), V_lore_single.squeeze(), dim=0)
print(f"Cosine similarity between V_sft and collapsed LoRe direction: {cos_sim.item():.6f}")

acc_lore_seen = evaluate_with_vector(test_seen, V_lore_single)
acc_lore_unseen = evaluate_with_vector(test_unseen, V_lore_single)

print(f"\nLoRe K=20 on SEEN users (test):   {np.mean(acc_lore_seen):.4f} +/- {np.std(acc_lore_seen):.4f}")
print(f"LoRe K=20 on UNSEEN users (test): {np.mean(acc_lore_unseen):.4f} +/- {np.std(acc_lore_unseen):.4f}")

# --- Summary ---
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Base RM (V_sft):        {np.mean(acc_seen):.2%} seen / {np.mean(acc_unseen):.2%} unseen")
print(f"  LoRe K=20 (collapsed):  {np.mean(acc_lore_seen):.2%} seen / {np.mean(acc_lore_unseen):.2%} unseen")
print(f"  Accuracy gain from LoRe: {np.mean(acc_lore_seen) - np.mean(acc_seen):.2%} seen / {np.mean(acc_lore_unseen) - np.mean(acc_unseen):.2%} unseen")
print(f"  V_sft vs LoRe cosine:   {cos_sim.item():.4f}")
