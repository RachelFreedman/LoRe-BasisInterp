import torch
import numpy as np
import random
import os
import sys

# Change to PRISM directory to run the tests locally without breaking paths
os.chdir("/Users/mohamedeldagla/Desktop/F/PRISM /LoRe Interpretation/LoRe-BasisInterp/PRISM")
sys.path.append("..")

import utils

# Generate mock data
def generate_mock_data():
    utils.set_seed(42)
    device = utils.device
    N = 2 # 2 seen users
    N_unseen = 1 # 1 unseen user
    num_prompts = 5
    num_features = 4096
    
    # train_seen
    train_seen = [torch.randn(num_prompts, num_features, device=device) for _ in range(N)]
    train_unseen = [torch.randn(num_prompts, num_features, device=device) for _ in range(N_unseen)]
    test_seen = [torch.randn(num_prompts, num_features, device=device) for _ in range(N)]
    test_unseen = [torch.randn(num_prompts, num_features, device=device) for _ in range(N_unseen)]
    
    V_final = torch.randn(num_features, 1, device=device)
    
    return train_seen, test_seen, train_unseen, test_unseen, V_final, N, N_unseen

def run_experiment(seed):
    train_seen, test_seen, train_unseen, test_unseen, V_final, N, N_unseen = generate_mock_data()
    utils.set_seed(seed)
    
    # We'll just run one configuration to test reproducibility
    K_list = [5]
    alpha_list = [10000.0]
    
    # We must patch num_iterations to be very small for quick testing
    original_num_iter = utils.solve_regularized_simplex.__defaults__
    
    # Redefine solve_regularized_simplex to run fast for this test
    original_solve = utils.solve_regularized_simplex
    def solve_regularized_simplex_fast(V_sft, alpha, train_features, num_basis_vectors, num_iterations=5, learning_rate=0.01):
        return original_solve(V_sft, alpha, train_features, num_basis_vectors, num_iterations=10, learning_rate=0.5)
    
    utils.solve_regularized_simplex = solve_regularized_simplex_fast
    
    results = utils.run_regularized(
        K_list, alpha_list, V_final, train_seen, test_seen, train_unseen, test_unseen, N, N_unseen, utils.device
    )
    
    utils.solve_regularized_simplex = original_solve
    return results

print("Running experiment with seed 42 (run 1)...")
res1 = run_experiment(42)

print("Running experiment with seed 42 (run 2)...")
res2 = run_experiment(42)

print("Running experiment with seed 99 (run 3)...")
res3 = run_experiment(99)

def compare_results(r1, r2):
    for i in range(len(r1)):
        if not np.allclose(r1[i], r2[i]):
            return False
    return True

print(f"Run 1 and Run 2 identical: {compare_results(res1, res2)}")
print(f"Run 1 and Run 3 identical: {compare_results(res1, res3)}")

if compare_results(res1, res2) and not compare_results(res1, res3):
    print("SUCCESS: Reproducibility confirmed.")
else:
    print("FAILED: Reproducibility broken.")
