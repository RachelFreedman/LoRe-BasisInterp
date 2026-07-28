"""
Experiments to prove the "Lazy Optimization" and "Basis Collapse" theories to Rachel.
"""
import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import defaultdict
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

def evaluate_accuracy(features, V_list, W_list=None):
    """Evaluates training accuracy."""
    accuracies = []
    # If W_list is None, we assume V is a single vector (K=1) used for all users
    for i, diffs in enumerate(features):
        if W_list is not None:
            # Reconstruct the user's specific reward vector: V @ softmax(W)
            w = F.softmax(W_list[i], dim=0) # [K, 1]
            V_user = V_list @ w # [4096, 1]
        else:
            V_user = V_list
            
        scores = diffs @ V_user # [num_pairs, 1]
        correct = (scores > 0).float().mean().item()
        accuracies.append(correct)
    return np.mean(accuracies)

print("Loading embeddings...")
train_emb = torch.load('data/prism/train_embeddings.pkl', map_location='cpu', weights_only=False)

# Dummy V_sft for initialization (just random noise to simulate the anchor)
# We use 4096 dimensions
V_sft_dummy = torch.randn(4096, 1, dtype=torch.float32)

print("\n" + "="*80)
print("EXPERIMENT A: The 'Random Labels' Memorization Test")
print("="*80)
# Hypothesis: 4096 dimensions is so large that the model can easily memorize 
# completely random labels (pure noise) with a single direction (K=1).

random.seed(42)
torch.manual_seed(42)

# Create a single 'global' user with 2000 random pairs to simulate a dataset
# We use 2000 pairs to make training fast but statistically significant
sample_embs = random.sample(train_emb, 2000)

noise_diffs = []
for ex in sample_embs:
    c = torch.tensor(ex['extra_info']['chosen_conv_embedding'], dtype=torch.float32)
    r = torch.tensor(ex['extra_info']['rejected_conv_embedding'], dtype=torch.float32)
    diff = c - r
    
    # 50% chance to flip the label (make the rejected one 'chosen')
    if random.random() < 0.5:
        diff = -diff
        
    noise_diffs.append(diff)

train_features_noise = [torch.stack(noise_diffs)] # 1 user, 2000 pairs

print(f"Training LoRe (K=1) on {len(noise_diffs)} pure NOISE pairs...")
W_noise, V_noise = utils.solve_regularized(
    V_sft=V_sft_dummy, 
    alpha=0.0, 
    train_features=train_features_noise, 
    num_basis_vectors=1, 
    num_iterations=200, 
    learning_rate=0.1
)

acc_noise = evaluate_accuracy(train_features_noise, V_noise[:, 0:1])
print(f"--> Training Accuracy on Pure Noise: {acc_noise:.2%}")
print("If this is near 100%, it proves 4096-dim space easily allows finding a lazy separating direction!")


print("\n" + "="*80)
print("EXPERIMENT B: Artificial Conflict Injection")
print("="*80)
# Hypothesis: The lack of overlapping, contradictory prompts causes basis collapse.
# If we force two users to have perfectly opposite preferences on the exact same prompts,
# a single direction cannot solve it, and the model will be FORCED to use distinct bases (Rank > 1).

sample_embs_conflict = random.sample(train_emb, 500)

user_a_diffs = []
user_b_diffs = []

for ex in sample_embs_conflict:
    c = torch.tensor(ex['extra_info']['chosen_conv_embedding'], dtype=torch.float32)
    r = torch.tensor(ex['extra_info']['rejected_conv_embedding'], dtype=torch.float32)
    diff = c - r
    
    # User A prefers chosen
    user_a_diffs.append(diff)
    # User B prefers rejected (perfect contradiction)
    user_b_diffs.append(-diff)

train_features_conflict = [torch.stack(user_a_diffs), torch.stack(user_b_diffs)]

print(f"Training LoRe (K=2) on 2 users with {len(sample_embs_conflict)} PERFECTLY CONTRADICTORY pairs...")
W_conf, V_conf = utils.solve_regularized(
    V_sft=V_sft_dummy, 
    alpha=0.0, 
    train_features=train_features_conflict, 
    num_basis_vectors=2, 
    num_iterations=300, 
    learning_rate=0.1
)

# Check collapse
V_conf_norm = F.normalize(V_conf, p=2, dim=0)
cos_sim_conflict = torch.sum(V_conf_norm[:, 0] * V_conf_norm[:, 1]).item()
print(f"--> Cosine Similarity between the 2 learned bases: {cos_sim_conflict:.4f}")

# Check accuracy
acc_conf = evaluate_accuracy(train_features_conflict, V_conf, W_conf)
print(f"--> Training Accuracy on Contradictory Dataset: {acc_conf:.2%}")
print("If similarity is low (e.g. negative), the basis collapse is BROKEN! The model was forced to learn distinct bases.")
