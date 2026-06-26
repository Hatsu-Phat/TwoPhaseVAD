#!/usr/bin/env bash
set -euo pipefail

# Train only the NASNetMobile-EViT paper-style Phase-1 model on global GMM Top-K features.
python -u phase1/train_phase1.py --config configs/phase1_evit.yaml 2>&1 | tee train_phase1_top30_evit.log
