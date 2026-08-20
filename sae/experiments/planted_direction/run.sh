#!/usr/bin/env bash
# Planted-direction check -- full replication.
# Owner: Hassan   Red-teamer: Harsh
#
# CPU IS NOT OPTIONAL. utils.py resolves its device at import and builds parameters
# with device=device, so manual_seed draws a different init stream under CUDA and
# every fitted number shifts (subspace alignment 0.755 -> 0.795 on one GPU run).
# Numbers below are only comparable across machines with CUDA disabled.
#
# Usage:  bash sae/experiments/planted_direction/run.sh [stage]
#         stages: geometry | plant | readout | naming | circularity | width | seeds | all
set -euo pipefail
cd "$(dirname "$0")/../../.."          # repo root
export CUDA_VISIBLE_DEVICES=""
S=sae/scripts
stage="${1:-all}"

run_geometry()    { python3 $S/concept_library_geometry.py; }
run_plant()       { python3 PRISM/planted_directions.py; }
run_readout()     { python3 $S/planted_concept_readout.py; }
run_profiles()    { python3 $S/feature_text_profiles.py; }
run_naming()      { python3 $S/planted_concept_naming.py; }
run_circularity() { python3 $S/planted_circularity_controls.py; }
run_width()       { python3 $S/readout_width_check.py; }
run_seeds() {
  python3 $S/resampled_seeds_check.py \
    --out results/planted/resampled_seeds.json \
    --rows results/planted/resampled_seeds_rows.csv
  python3 $S/resampled_seeds_check.py --fixed-split \
    --out results/planted/fixed_split_seeds.json \
    --rows results/planted/fixed_split_seeds_rows.csv
}

case "$stage" in
  geometry)    run_geometry ;;
  plant)       run_plant ;;
  readout)     run_readout ;;
  profiles)    run_profiles ;;
  naming)      run_naming ;;
  circularity) run_circularity ;;
  width)       run_width ;;
  seeds)       run_seeds ;;
  all)
    run_geometry; run_plant; run_readout; run_profiles
    run_naming; run_circularity; run_width; run_seeds ;;
  *) echo "unknown stage: $stage" >&2; exit 1 ;;
esac
