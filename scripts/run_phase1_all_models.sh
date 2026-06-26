#!/usr/bin/env bash
set -euo pipefail

for MODEL in evit lstm tcn transformer stgnn; do
  echo "========== Train phase1 paper-topk $MODEL =========="
  python -u phase1/train_phase1.py --config "configs/phase1_${MODEL}.yaml" 2>&1 | tee "train_phase1_top30_${MODEL}.log"
done
