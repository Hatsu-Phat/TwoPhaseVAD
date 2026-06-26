#!/usr/bin/env bash
set -euo pipefail

UCF=${UCF:-/home/grouphahieu/imagenet/UCF-Crime}
VIDEO_ROOT=${VIDEO_ROOT:-$UCF}
SPLIT_DIR=${SPLIT_DIR:-$UCF/UCF_Crimes-Train-Test-Split/Anomaly_Detection_splits}
TOP_K=${TOP_K:-30}
FRAME_STRIDE=${FRAME_STRIDE:-1}

python -u phase1/extract_nasnet_gmm_topk_features.py \
  --video-root "$VIDEO_ROOT" \
  --out-root "$UCF/features_nasnet_gmm_top${TOP_K}" \
  --annotation-file "$UCF/Temporal_Anomaly_Annotation_for_Testing_Videos.txt" \
  --train-split-files "$SPLIT_DIR/Anomaly_Train.txt" \
  --test-split-files "$SPLIT_DIR/Anomaly_Test.txt" \
  --top-k "$TOP_K" \
  --image-size 224 \
  --backbone nasnetamobile \
  --batch-size-frames 64 \
  --frame-stride "$FRAME_STRIDE" \
  --warmup-frames 5 \
  --device cuda \
  --save-float16 \
  --skip-existing \
  2>&1 | tee "extract_nasnet_gmm_top${TOP_K}.log"
