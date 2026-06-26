#!/usr/bin/env bash
set -euo pipefail

# Full pipeline: extract global GMM Top-K features -> train all Phase-1 models -> build Phase-2 clips from EViT -> train all Phase-2 models.
TOP_K=${TOP_K:-30}
export TOP_K
bash scripts/run_extract_features.sh
bash scripts/run_phase1_all_models.sh
MODEL=evit bash scripts/run_build_phase2.sh
SOURCE_MODEL=evit bash scripts/run_phase2_all_models.sh
