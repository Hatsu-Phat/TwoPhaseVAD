#!/usr/bin/env bash
set -e

cd /home/grouphahieu/imagenet/phatlam/phatlam_pipeline_paper_topk

export UCF=/home/grouphahieu/imagenet/UCF-Crime
export FEATURES_ROOT=$UCF/features_nasnet_gmm_top30_fp32_norm05
export OUT_ROOT=$UCF/outputs_phase1_top30_fp32_norm05

mkdir -p "$OUT_ROOT"

python - <<'PY'
from pathlib import Path
import os
import yaml

UCF = os.environ["UCF"]
FEATURES_ROOT = os.environ["FEATURES_ROOT"]
OUT_ROOT = os.environ["OUT_ROOT"]

models = ["lstm", "tcn", "transformer", "stgnn"]

for model in models:
    src = Path(f"configs/phase1_{model}.yaml")
    dst = Path(f"configs/phase1_{model}_top30_fp32_norm05.yaml")

    if not src.exists():
        raise FileNotFoundError(f"Missing config: {src}")

    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))

    cfg["features_root"] = FEATURES_ROOT
    cfg["out_dir"] = f"{OUT_ROOT}/{model}"
    cfg["model"] = model
    cfg["input_dim"] = 1056
    cfg["seq_len"] = 30

    cfg["batch_size"] = 32
    cfg["epochs"] = 100
    cfg["lr"] = cfg.get("lr", 0.0001)
    cfg["weight_decay"] = cfg.get("weight_decay", 0.0001)
    cfg["device"] = "cuda"
    cfg["num_workers"] = 0
    cfg["seed"] = 42

    dst.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print("saved", dst)
    print(yaml.safe_dump(cfg, sort_keys=False))
PY

for MODEL in lstm tcn transformer stgnn; do
  echo "=============================="
  echo "TRAIN PHASE 1 MODEL: $MODEL"
  echo "=============================="

  python -u phase1/train_phase1.py \
    --config "configs/phase1_${MODEL}_top30_fp32_norm05.yaml" \
    2>&1 | tee "train_phase1_${MODEL}_top30_fp32_norm05.log"
done
