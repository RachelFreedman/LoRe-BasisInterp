# D3 Sparse Autoencoder for LoRe Basis Interpretation

## What this is

This trains a TopK sparse autoencoder on the 4096-dimensional PRISM/Skywork
embeddings used by LoRe. The goal is to preserve LoRe basis scores while
producing sparse features for later inspection. **D3** is the current selected
configuration (`16384` dictionary size, `k=256`, centered inputs, unit-norm
decoder, auxiliary dead-feature loss). Semantic labeling is not part of this
branch.

## How to run

Run all commands from the **repository root**.

Required local files (not committed):

```text
PRISM/data/prism/train_embeddings.pkl
PRISM/data/prism/test_embeddings.pkl
PRISM/basis_matrices.pt
```

```bash
# 0. Install dependencies
pip install -r requirements.txt

# 1. Build SAE train/validation/test tensors
bash sae/run_d3.sh build

# 2. Train D3 — GPU recommended
bash sae/run_d3.sh train

# 3. Evaluate LoRe preservation
bash sae/run_d3.sh evaluate

# 4. Check feature usage and concentration
bash sae/run_d3.sh diagnose

# 5. Optional: numeric LoRe-basis feature attribution
bash sae/run_d3.sh analyze
```

One-command pipeline (build → train → evaluate → diagnose → analyze):

```bash
bash sae/run_d3.sh all
```

- **`train` is the GPU-intensive step.**
- `build` / `evaluate` / `diagnose` / `analyze` auto-select CUDA, MPS, or CPU
  when available (via each script’s `--device auto` default).
- The runner fails with a clear message if an input or checkpoint is missing.

## Defaults

| Setting | Value |
| --- | ---: |
| Input dimension | 4096 |
| Dictionary size | 16384 |
| Active features k | 256 |
| Sparsity mode | TopK |
| Training steps | 20000 |
| Batch size | 256 |
| Learning rate | 3e-4 |
| Center inputs | Yes |
| Unit-normalized decoder | Yes |
| Auxiliary dead-feature coefficient | 0.03125 |
| LoRe run key | PART2_K10_seed42 |
| Split seed | 123 |

Full config: `sae/configs/d3.yaml`.

## Expected D3 result

| Metric | Expected |
| --- | ---: |
| Mean LoRe basis Pearson | 0.9542 |
| Minimum LoRe basis Pearson | 0.9538 |
| Mean pair-score Pearson | 0.9152 |
| Explained variance | 0.9924 |
| Reconstruction MSE | 0.0384 |
| Original LoRe accuracy | 0.9301 |
| Reconstructed LoRe accuracy | 0.9387 |
| Accuracy drop | -0.0086 |
| Test live features | 10668 / 16384 |
| Test dead-feature rate | 0.3489 |
| Test live-feature Gini | 0.9096 |
| Top 5% activation mass | 0.8557 |
| Effective feature count | approximately 666 |

Small floating-point differences across devices are acceptable. The committed
reference row is in `sae/d3_results.csv`.

**D3 passes the predefined LoRe-preservation gate.**

## Artifacts to look at

| File | What it contains | What to compare |
| --- | --- | --- |
| `sae/d3_results.csv` | Committed reference result | Your main metrics vs this row |
| `sae/checkpoints/d3/model.pt` | Trained D3 checkpoint | Required for evaluate / diagnose / analyze |
| `sae/results/d3/sae_eval_summary.json` | Reconstruction + LoRe metrics | Mean/min basis Pearson, pair Pearson, accuracy drop |
| `sae/results/d3/basis_score_correlations.csv` | One row per LoRe basis | Per-basis Pearson and pair Pearson |
| `sae/results/d3/sae_diagnostics_summary.json` | Train/val/test feature health | Live/dead rates, Gini |
| `sae/results/d3/top_active_features.csv` | Most frequently used features | Feature concentration |
| `sae/results/d3/top_features_per_basis_operational.csv` | Attribution for operational bases | Top contribution-ranked features for bases **1, 3, 9** |
| `sae/results/d3/attribution_meta.json` | Attribution settings and scope | Run key and operational basis IDs |

Generated paths under `sae/data/`, `sae/checkpoints/`, and `sae/results/` are
gitignored. Optional supplementary attribution for all 10 bases:
`sae/results/d3/top_features_per_basis.csv`.

## What the main metrics mean

- **Mean basis Pearson:** how well reconstructed embeddings preserve LoRe basis scores.
- **Pair-score Pearson:** how well chosen-minus-rejected LoRe scores are preserved.
- **Accuracy drop:** original LoRe accuracy minus reconstructed accuracy (lower or negative is better).
- **Dead-feature rate:** fraction of SAE features that never activate on the split.
- **Live-feature Gini:** inequality of usage among features that activate (higher = more concentrated).
- **Top 5% activation mass:** share of total activation carried by the most-used 5% of features.

## Current limitation

Feature usage remains concentrated. D3 passes the LoRe-preservation gate, but
the test live-feature Gini is about 0.91 and the top 5% of features carry about
86% of activation mass. I am still exploring whether this concentration can be
reduced without weakening LoRe preservation.

## Scope

- No semantic feature labels are assigned.
- Numeric basis-feature attribution is observational, not causal.
- Embeddings, checkpoints, and generated results are not committed.
- LLM judging, persona vectors, and behavioral validation are outside this branch.
