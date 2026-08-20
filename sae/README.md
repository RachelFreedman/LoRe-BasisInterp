# D3 Sparse Autoencoder for LoRe Basis Interpretation

## What this is

This trains a TopK sparse autoencoder on the 4096-dimensional PRISM/Skywork embeddings used by LoRe. The goal is to preserve LoRe basis scores while producing sparse features. **D3** is the current selected configuration (`16384` dict, `k=256`, center inputs, unit-norm decoder, aux dead-feature loss). Semantic labeling is not part of this branch.

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
- `build` / `evaluate` / `diagnose` / `analyze` auto-select CUDA, MPS, or CPU (`--device auto`).
- The runner errors clearly if an input or checkpoint is missing.

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

Retrained on the corrected PRISM embeddings (see **Data version** below).

| Metric | Expected |
| --- | ---: |
| Mean LoRe basis Pearson | 0.9994 |
| Minimum LoRe basis Pearson | 0.9994 |
| Mean pair-score Pearson | 0.9994 |
| Explained variance | 0.9941 |
| Reconstruction MSE | 0.0299 |
| Original LoRe accuracy | 0.5966 |
| Reconstructed LoRe accuracy | 0.6014 |
| Accuracy drop | -0.0048 |
| Test live features | 11148 / 16384 |
| Test dead-feature rate | 0.3196 |
| Test live-feature Gini | 0.9190 |
| Top 5% activation mass | 0.8646 |
| Effective feature count | approximately 611 |

Small floating-point differences across devices are acceptable. The committed reference
row is in `sae/d3_results.csv`.

**D3 passes the LoRe-preservation gate, but the gate no longer discriminates.** Basis
Pearson of 0.9994 reads like an improvement on the previous 0.9542 and is not one. The
Phase-1 bases refit on corrected embeddings predict held-out preference at 0.5904 and
carry a twelfth of their former norm, so `V^T e` is dominated by the shared component of
the embedding cloud and the correlation is close to free. Do not cite these three numbers
as evidence of reconstruction quality; use explained variance and the pair-score
correlation instead, and read the gate as a floor rather than a discriminator.

## Data version

These numbers come from the corrected PRISM embeddings. The earlier regeneration fixed a
bug where rejected responses were rendered as a Python list rather than prose, so every
rejected embedding changed while chosen ones stayed identical. Corrected files carry
`pair_format: string-vs-string-v1` in `extra_info` and hold 19,638 train / 19,727 test
comparisons; the pre-fix files hold 12,999 and lack the marker.

The SAE trains on chosen and rejected embeddings pooled, not on their differences, so
half of the previous training set was affected. Both the SAE and the Phase-1 basis were
refit. The previous reference row is preserved in git history at `57dbef7` and does not
reproduce on corrected data.

The retrained checkpoint is `D3_16384_k256_string-vs-string-v1.pt` on the team Drive.
Feature indices do not correspond between it and the previous dictionary, so results from
the two cannot be compared feature-by-feature.

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

Generated paths under `sae/data/`, `sae/checkpoints/`, and `sae/results/` are gitignored. Optional all-bases attribution: `sae/results/d3/top_features_per_basis.csv`.

## What the main metrics mean

- **Mean basis Pearson:** how well reconstructed embeddings preserve LoRe basis scores.
- **Pair-score Pearson:** how well chosen-minus-rejected LoRe scores are preserved.
- **Accuracy drop:** original LoRe accuracy minus reconstructed accuracy.
- **Dead-feature rate:** fraction of SAE features that never activate on the split.
- **Live-feature Gini:** inequality of usage among features that activate.
- **Top 5% activation mass:** share of total activation carried by the most-used 5% of features.

## Current limitation

Feature usage remains concentrated. D3 passes the LoRe-preservation gate, but the test live-feature Gini is about 0.91 and the top 5% of features carry about 86% of activation mass. I am still exploring whether this concentration can be reduced without weakening LoRe preservation.

## Scope

- No semantic feature labels are assigned.
- Numeric basis-feature attribution is observational, not causal.
- Embeddings, checkpoints, and generated results are not committed.
- LLM judging, persona vectors, and behavioral validation are outside this branch.
