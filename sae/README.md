# Sparse Autoencoder for LoRe Basis Interpretation

## Goal

This package trains a sparse autoencoder (SAE) on the same 4096-dimensional
Skywork reward-model embeddings used by LoRe. Each response embedding is
reconstructed from a small set of active TopK features.

The SAE is useful for LoRe interpretation only if reconstruction preserves the
reward-relevant geometry of the LoRe bases. Evaluation therefore emphasizes
LoRe basis-score correlation and pairwise preference accuracy, not only
reconstruction MSE.

D3 was selected after internal capacity experiments; exploratory configurations
and outputs are not included in this branch.

## Required inputs

Provide these generated files locally (they are **not** committed):

```text
PRISM/data/prism/train_embeddings.pkl
PRISM/data/prism/test_embeddings.pkl
PRISM/basis_matrices.pt
```

Canonical LoRe run key: `PART2_K10_seed42`.

## Installation

From the repository root:

```bash
pip install -r requirements.txt
```

## Quick start

```bash
bash sae/run_d3.sh build
bash sae/run_d3.sh train
bash sae/run_d3.sh evaluate
bash sae/run_d3.sh diagnose
```

Or end-to-end (includes optional numeric basis-feature attribution):

```bash
bash sae/run_d3.sh all
```

| Command | Purpose |
| --- | --- |
| `build` | Build SAE train/val/test tensors from PRISM embeddings |
| `train` | Train the D3 TopK SAE |
| `evaluate` | LoRe preservation + reconstruction metrics |
| `diagnose` | Live/dead features and usage concentration |
| `analyze` | Numeric basis–feature attribution (observational) |
| `all` | build → train → evaluate → diagnose → analyze |

### Outputs (gitignored)

```text
sae/data/                      # tensors + metadata
sae/checkpoints/d3/model.pt    # trained checkpoint (~hundreds of MB)
sae/results/d3/                # eval, diagnostics, attribution CSVs/JSON
```

Committed machine-readable summary: `sae/d3_results.csv`.

## D3 configuration

| Setting | Value |
| --- | ---: |
| Input dimension | 4096 |
| Dictionary size | 16384 |
| Active features `k` | 256 |
| Sparsity | TopK |
| Steps | 20000 |
| Batch size | 256 |
| Learning rate | 3e-4 |
| Center inputs | Yes |
| Unit-normalized decoder | Yes |
| Auxiliary dead-feature loss | 0.03125 |

Config file: `sae/configs/d3.yaml`.

Training data: chosen + rejected response embeddings (not pairwise differences).
Pairwise differences are used only at evaluation time.

## D3 results (test)

| Metric | D3 result |
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

**D3 passes the predefined LoRe-preservation gate**
(mean basis Pearson ≥ 0.95, min ≥ 0.90, LoRe accuracy drop ≤ 0.02).

Feature usage remains concentrated. Live-feature Gini and activation
concentration are still under investigation: the open question is whether they
can be improved without reducing LoRe preservation.

## LoRe bases used in evaluation

`PART2_K10_seed42` stores **10** columns in `V`. Personalized LoRe accuracy uses
only bases with non-negligible user weight (`max_user_weight ≥ 1e-2`). For this
run, **three bases are operational** (ids **1, 3, 9**); the other seven remain
in the matrix with near-zero user mass. Attribution writes a primary table for
operational bases and a supplementary table for all ten.

## Scope and limitations

- No semantic feature labels are assigned in this package.
- Basis–feature projections (`decoder · V`, contribution scores) are
  **observational**, not causal.
- The trained checkpoint is not committed (size).
- Generated data, checkpoints, logs, and large result tables are gitignored.
- Labeling, LLM judging, persona vectors, and behavioral validation are out of
  scope for this branch.

## Layout

```text
sae/
├── README.md
├── .gitignore
├── d3_results.csv
├── run_d3.sh
├── configs/d3.yaml
├── scripts/
│   ├── build_sae_dataset.py
│   ├── train_sae.py
│   ├── evaluate_sae.py
│   ├── diagnose_sae.py
│   └── analyze_basis_features.py
└── src/
    ├── data.py
    ├── io.py
    ├── metrics.py
    ├── topk_sae.py
    └── attribution.py
```
