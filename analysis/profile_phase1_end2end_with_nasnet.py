#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn


class NASNetMobileFeatureWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        import pretrainedmodels

        self.backbone = pretrainedmodels.__dict__["nasnetamobile"](
            num_classes=1000,
            pretrained="imagenet",
        )

    def forward(self, x):
        feat = self.backbone.features(x)
        feat = torch.nn.functional.adaptive_avg_pool2d(feat, 1)
        feat = torch.flatten(feat, 1)
        return feat


def profile_macs(model, dummy):
    from thop import profile
    macs, params = profile(model, inputs=(dummy,), verbose=False)
    return float(macs), float(params)


@torch.no_grad()
def latency_ms(model, dummy, device, warmup=20, iters=100):
    model = model.to(device).eval()
    dummy = dummy.to(device)

    for _ in range(warmup):
        _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(iters):
        _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()

    return (end - start) * 1000.0 / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--efficiency-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-tex", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-selected-frames", type=int, default=30)
    ap.add_argument("--image-size", type=int, default=224)
    args = ap.parse_args()

    device = torch.device(
        args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )

    df = pd.read_csv(args.efficiency_csv)

    nasnet = NASNetMobileFeatureWrapper()
    dummy = torch.randn(1, 3, args.image_size, args.image_size)

    print("Profiling NASNetMobile per frame...")
    nasnet_macs, nasnet_params = profile_macs(nasnet.cpu(), dummy.cpu())
    nasnet_time = latency_ms(nasnet, dummy, device=device, warmup=20, iters=100)

    nasnet_row = {
        "phase": "feature_extractor",
        "model": "nasnetamobile_per_frame",
        "input_shape": f"[1,3,{args.image_size},{args.image_size}]",
        "params": nasnet_params,
        "trainable_params": nasnet_params,
        "params_million": nasnet_params / 1e6,
        "macs_g": nasnet_macs / 1e9,
        "flops_g_approx_2x_macs": 2 * nasnet_macs / 1e9,
        "inference_time_ms": nasnet_time,
        "config": "NASNetMobile feature extractor per selected frame",
    }

    nasnet30_row = {
        "phase": "feature_extractor",
        "model": f"nasnetamobile_x{args.num_selected_frames}_frames",
        "input_shape": f"[{args.num_selected_frames},3,{args.image_size},{args.image_size}]",
        "params": nasnet_params,
        "trainable_params": nasnet_params,
        "params_million": nasnet_params / 1e6,
        "macs_g": nasnet_macs * args.num_selected_frames / 1e9,
        "flops_g_approx_2x_macs": 2 * nasnet_macs * args.num_selected_frames / 1e9,
        "inference_time_ms": nasnet_time * args.num_selected_frames,
        "config": f"NASNetMobile × {args.num_selected_frames} selected frames",
    }

    rows = [nasnet_row, nasnet30_row]

    phase1 = df[df["phase"] == "phase1_temporal"].copy()

    for _, r in phase1.iterrows():
        model_name = r["model"]

        temporal_macs_g = float(r["macs_g"]) if pd.notna(r.get("macs_g")) else None
        temporal_flops_g = float(r["flops_g_approx_2x_macs"]) if pd.notna(r.get("flops_g_approx_2x_macs")) else None
        temporal_time_ms = float(r["inference_time_ms"]) if pd.notna(r.get("inference_time_ms")) else None

        e2e_macs_g = None if temporal_macs_g is None else nasnet30_row["macs_g"] + temporal_macs_g
        e2e_flops_g = None if temporal_flops_g is None else nasnet30_row["flops_g_approx_2x_macs"] + temporal_flops_g
        e2e_time_ms = None if temporal_time_ms is None else nasnet30_row["inference_time_ms"] + temporal_time_ms

        rows.append({
            "phase": "phase1_end_to_end",
            "model": f"NASNetMobile+{model_name}",
            "input_shape": f"video -> top{args.num_selected_frames} frames -> [1,30,1056]",
            "params": nasnet_params + float(r["params"]) if pd.notna(r.get("params")) else None,
            "trainable_params": nasnet_params + float(r["trainable_params"]) if pd.notna(r.get("trainable_params")) else None,
            "params_million": (nasnet_params + float(r["params"])) / 1e6 if pd.notna(r.get("params")) else None,
            "macs_g": e2e_macs_g,
            "flops_g_approx_2x_macs": e2e_flops_g,
            "inference_time_ms": e2e_time_ms,
            "config": f"NASNetMobile × {args.num_selected_frames} + Phase1 {model_name}; excludes video decoding and GMM selection",
        })

    out_df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_tex = Path(args.out_tex)

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    out_df.to_csv(out_csv, index=False)
    out_df.to_markdown(out_md, index=False)
    out_df.to_latex(out_tex, index=False, float_format="%.4f")

    print("Saved:")
    print(out_csv)
    print(out_md)
    print(out_tex)

    print("\n===== NASNetMobile per frame =====")
    print(pd.DataFrame([nasnet_row]).to_string(index=False))

    print("\n===== Phase 1 end-to-end rows =====")
    print(out_df[out_df["phase"] == "phase1_end_to_end"].to_string(index=False))


if __name__ == "__main__":
    main()
