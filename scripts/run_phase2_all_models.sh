#!/usr/bin/env bash
set -euo pipefail

UCF=${UCF:-/home/grouphahieu/imagenet/UCF-Crime}
SOURCE_MODEL=${SOURCE_MODEL:-evit}
TOP_K=${TOP_K:-30}
MANIFEST=${MANIFEST:-$UCF/phase2_clips_top${TOP_K}_from_${SOURCE_MODEL}/manifest_phase2.jsonl}

for MODEL in simple_cnn resnet18 resnet50 efficientnet_b0 efficientnet_b3 vit_b_16 swin_t convnext_tiny; do
  echo "========== Train phase2 top${TOP_K} $MODEL =========="
  BS=8
  CLIP=16
  LR=0.0001
  if [[ "$MODEL" == "vit_b_16" || "$MODEL" == "swin_t" || "$MODEL" == "convnext_tiny" ]]; then
    BS=4
    CLIP=8
    LR=0.00005
  fi
  python -u phase2/train_phase2_classifier.py \
    --manifest "$MANIFEST" \
    --out-dir "$UCF/outputs_phase2_top${TOP_K}/${SOURCE_MODEL}_${MODEL}" \
    --model "$MODEL" \
    --num-classes 13 \
    --clip-len "$CLIP" \
    --image-size 224 \
    --batch-size "$BS" \
    --epochs 50 \
    --lr "$LR" \
    --weight-decay 0.0001 \
    --pretrained \
    --device cuda \
    --num-workers 0 \
    2>&1 | tee "train_phase2_top${TOP_K}_${SOURCE_MODEL}_${MODEL}.log"
done
