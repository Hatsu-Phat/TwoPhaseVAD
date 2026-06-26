#!/usr/bin/env bash
set -euo pipefail

UCF=${UCF:-/home/grouphahieu/imagenet/UCF-Crime}
VIDEO_ROOT=${VIDEO_ROOT:-$UCF}
MODEL=${MODEL:-evit}
TOP_K=${TOP_K:-30}
THRESHOLD_MODE=${THRESHOLD_MODE:-max_normal}

python -u phase1/infer_phase1_scores.py \
  --features-root "$UCF/features_nasnet_gmm_top${TOP_K}" \
  --checkpoint "$UCF/outputs_phase1_top${TOP_K}/$MODEL/best_auc.pth" \
  --out-dir "$UCF/phase1_scores_top${TOP_K}/$MODEL" \
  --split all \
  --device cuda \
  2>&1 | tee "infer_phase1_top${TOP_K}_${MODEL}.log"

python -u phase2/build_phase2_clips.py \
  --video-root "$VIDEO_ROOT" \
  --scores-root "$UCF/phase1_scores_top${TOP_K}/$MODEL/train" \
  --out-root "$UCF/phase2_clips_top${TOP_K}_from_${MODEL}" \
  --threshold-mode "$THRESHOLD_MODE" \
  --clip-len 32 \
  --image-size 224 \
  --max-clips-per-video 5 \
  --fallback-topk 1 \
  --use-center selected_index \
  --exclude-normal \
  --val-ratio 0.2 \
  2>&1 | tee "build_phase2_clips_top${TOP_K}_${MODEL}_${THRESHOLD_MODE}.log"
