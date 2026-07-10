#!/usr/bin/env bash
# Minimal runner for the D3 SAE pipeline.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="sae/configs/d3.yaml"
DATA_DIR="sae/data"
CKPT_DIR="sae/checkpoints/d3"
CKPT_PATH="${CKPT_DIR}/model.pt"
RESULTS_DIR="sae/results/d3"
TRAIN_EMB="PRISM/data/prism/train_embeddings.pkl"
TEST_EMB="PRISM/data/prism/test_embeddings.pkl"
BASIS="PRISM/basis_matrices.pt"
RUN_KEY="PART2_K10_seed42"

usage() {
  cat <<'EOF'
Usage: bash sae/run_d3.sh <command>

Commands:
  build      Build SAE train/val/test tensors from PRISM embeddings
  train      Train the D3 TopK SAE
  evaluate   Evaluate reconstruction and LoRe preservation
  diagnose   Compute live/dead and concentration diagnostics
  analyze    Numeric basis-feature attribution (contribution-ranked)
  all        build → train → evaluate → diagnose → analyze

Required local inputs (not committed):
  PRISM/data/prism/train_embeddings.pkl
  PRISM/data/prism/test_embeddings.pkl
  PRISM/basis_matrices.pt

Outputs (gitignored):
  sae/data/
  sae/checkpoints/d3/model.pt
  sae/results/d3/
EOF
}

require_file() {
  local path="$1"
  local hint="$2"
  if [[ ! -f "$path" ]]; then
    echo "error: missing required file: $path" >&2
    echo "hint: $hint" >&2
    exit 1
  fi
}

cmd_build() {
  require_file "$TRAIN_EMB" "Provide Phase 1 train embeddings locally."
  require_file "$TEST_EMB" "Provide Phase 1 test embeddings locally."
  python sae/scripts/build_sae_dataset.py \
    --train-embeddings "$TRAIN_EMB" \
    --test-embeddings "$TEST_EMB" \
    --output-dir "$DATA_DIR" \
    --split-seed 123
}

cmd_train() {
  require_file "$CONFIG" "D3 config should live at sae/configs/d3.yaml."
  require_file "$DATA_DIR/sae_train.pt" "Run: bash sae/run_d3.sh build"
  require_file "$DATA_DIR/sae_val.pt" "Run: bash sae/run_d3.sh build"
  python sae/scripts/train_sae.py \
    --config "$CONFIG" \
    --data-dir "$DATA_DIR" \
    --checkpoint-dir "$CKPT_DIR" \
    --results-dir "$RESULTS_DIR" \
    --checkpoint-name model.pt
}

cmd_evaluate() {
  require_file "$CKPT_PATH" "Run: bash sae/run_d3.sh train  (or place model.pt under sae/checkpoints/d3/)"
  require_file "$DATA_DIR/sae_test.pt" "Run: bash sae/run_d3.sh build"
  require_file "$BASIS" "Provide PRISM/basis_matrices.pt locally."
  python sae/scripts/evaluate_sae.py \
    --checkpoint "$CKPT_PATH" \
    --data-dir "$DATA_DIR" \
    --basis-matrices "$BASIS" \
    --run-key "$RUN_KEY" \
    --split test \
    --results-dir "$RESULTS_DIR"
}

cmd_diagnose() {
  require_file "$CKPT_PATH" "Run: bash sae/run_d3.sh train  (or place model.pt under sae/checkpoints/d3/)"
  require_file "$DATA_DIR/sae_train.pt" "Run: bash sae/run_d3.sh build"
  require_file "$DATA_DIR/sae_val.pt" "Run: bash sae/run_d3.sh build"
  require_file "$DATA_DIR/sae_test.pt" "Run: bash sae/run_d3.sh build"
  python sae/scripts/diagnose_sae.py \
    --checkpoint "$CKPT_PATH" \
    --data-dir "$DATA_DIR" \
    --results-dir "$RESULTS_DIR"
}

cmd_analyze() {
  require_file "$CKPT_PATH" "Run: bash sae/run_d3.sh train  (or place model.pt under sae/checkpoints/d3/)"
  require_file "$DATA_DIR/sae_test.pt" "Run: bash sae/run_d3.sh build"
  require_file "$BASIS" "Provide PRISM/basis_matrices.pt locally."
  python sae/scripts/analyze_basis_features.py \
    --checkpoint "$CKPT_PATH" \
    --data-dir "$DATA_DIR" \
    --basis-matrices "$BASIS" \
    --run-key "$RUN_KEY" \
    --split test \
    --results-dir "$RESULTS_DIR"
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

case "$1" in
  build) cmd_build ;;
  train) cmd_train ;;
  evaluate) cmd_evaluate ;;
  diagnose) cmd_diagnose ;;
  analyze) cmd_analyze ;;
  all)
    cmd_build
    cmd_train
    cmd_evaluate
    cmd_diagnose
    cmd_analyze
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "error: unknown command: $1" >&2
    usage
    exit 1
    ;;
esac
