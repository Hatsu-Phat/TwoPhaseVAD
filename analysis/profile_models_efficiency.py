#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def profile_macs(model, dummy):
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        return float(macs)
    except Exception as e:
        print(f"[WARN] FLOPs/MACs failed: {repr(e)}")
        return None


@torch.no_grad()
def measure_latency_ms(model, dummy, device, warmup=20, iters=100):
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


def find_phase1_config(model_name: str):
    candidates = [
        ROOT / f"configs/phase1_{model_name}_top30_rollback.yaml",
        ROOT / f"configs/phase1_cv_{model_name}_fold2.yaml",
        ROOT / f"configs/phase1_{model_name}.yaml",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(f"Missing config for {model_name}: {candidates}")


def load_cfg(model_name: str):
    cfg_path = find_phase1_config(model_name)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["model"] = model_name
    cfg["input_dim"] = int(cfg.get("input_dim", 1056))
    cfg["seq_len"] = int(cfg.get("seq_len", 30))
    cfg["hidden_dim"] = int(cfg.get("hidden_dim", 256))
    cfg["num_layers"] = int(cfg.get("num_layers", 2))
    cfg["num_heads"] = int(cfg.get("num_heads", 4))
    cfg["dropout"] = float(cfg.get("dropout", 0.3))
    cfg["topk"] = int(cfg.get("topk", 5))
    return cfg, cfg_path


def try_forward(model, shapes):
    model.eval()

    for shape in shapes:
        dummy = torch.randn(*shape)
        try:
            with torch.no_grad():
                _ = model(dummy)
            return dummy
        except Exception:
            pass

    raise RuntimeError("Model cannot forward with candidate dummy shapes")


def build_phase1_from_module(model_name: str, cfg: dict):
    import importlib

    module = importlib.import_module(f"phase1.models.{model_name}")

    preferred = {
        "evit": [
            "EViT",
            "EVIT",
            "EViTModel",
            "EVITModel",
            "EViTAnomalyDetector",
            "EVITAnomalyDetector",
            "AnomalyEViT",
        ],
        "lstm": [
            "LSTMModel",
            "LSTMAnomalyDetector",
            "TemporalLSTM",
            "MIL_LSTM",
            "LSTM",
        ],
        "tcn": [
            "TCNModel",
            "TCNAnomalyDetector",
            "TemporalConvNet",
            "TemporalTCN",
            "TCN",
        ],
        "transformer": [
            "TransformerModel",
            "TransformerAnomalyDetector",
            "TemporalTransformer",
            "MILTransformer",
            "Transformer",
        ],
        "stgnn": [
            "STGNNModel",
            "STGNN",
            "STGNNAnomalyDetector",
            "TemporalSTGNN",
        ],
    }

    candidate_classes = []

    for cname in preferred.get(model_name, []):
        if hasattr(module, cname):
            obj = getattr(module, cname)
            if inspect.isclass(obj) and issubclass(obj, nn.Module):
                candidate_classes.append(obj)

    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, nn.Module) and obj is not nn.Module and obj not in candidate_classes:
            if obj.__module__ == module.__name__:
                candidate_classes.append(obj)

    if not candidate_classes:
        available = [name for name, obj in inspect.getmembers(module, inspect.isclass)]
        raise RuntimeError(f"No nn.Module class found in {module.__name__}. Available classes={available}")

    errors = []

    for cls in candidate_classes:
        attempts = []

        attempts.append(lambda cls=cls: cls(cfg))

        def build_by_signature(cls=cls):
            sig = inspect.signature(cls.__init__)
            params = sig.parameters

            values = {
                "input_dim": cfg.get("input_dim", 1056),
                "in_dim": cfg.get("input_dim", 1056),
                "feature_dim": cfg.get("input_dim", 1056),
                "feat_dim": cfg.get("input_dim", 1056),
                "num_features": cfg.get("input_dim", 1056),

                "seq_len": cfg.get("seq_len", 30),
                "num_segments": cfg.get("seq_len", 30),
                "n_segments": cfg.get("seq_len", 30),

                "hidden_dim": cfg.get("hidden_dim", 256),
                "hidden_size": cfg.get("hidden_dim", 256),
                "d_model": cfg.get("hidden_dim", 256),

                "num_layers": cfg.get("num_layers", 2),
                "n_layers": cfg.get("num_layers", 2),
                "layers": cfg.get("num_layers", 2),

                "num_heads": cfg.get("num_heads", 4),
                "nhead": cfg.get("num_heads", 4),
                "n_heads": cfg.get("num_heads", 4),

                "dropout": cfg.get("dropout", 0.3),
                "topk": cfg.get("topk", 5),
                "top_k": cfg.get("topk", 5),

                "graph": cfg.get("graph", "temporal_knn"),
                "num_classes": 1,
                "out_dim": 1,
                "output_dim": 1,
            }

            kwargs = {}
            for k in params:
                if k == "self":
                    continue
                if k in values:
                    kwargs[k] = values[k]

            return cls(**kwargs)

        attempts.append(build_by_signature)

        attempts.append(lambda cls=cls: cls(cfg.get("input_dim", 1056), cfg.get("hidden_dim", 256)))
        attempts.append(lambda cls=cls: cls(cfg.get("input_dim", 1056), cfg.get("hidden_dim", 256), cfg.get("num_layers", 2)))
        attempts.append(lambda cls=cls: cls())

        for attempt in attempts:
            try:
                model = attempt()

                dummy = try_forward(
                    model,
                    shapes=[
                        (1, cfg.get("seq_len", 30), cfg.get("input_dim", 1056)),
                        (1, cfg.get("input_dim", 1056), cfg.get("seq_len", 30)),
                    ],
                )

                print(f"[OK] built {model_name} using class {cls.__name__}, input_shape={tuple(dummy.shape)}")
                return model, dummy

            except Exception as e:
                errors.append(f"{cls.__name__}: {repr(e)}")

    raise RuntimeError("Cannot instantiate phase1 model. Errors:\n" + "\n".join(errors[:30]))


def build_phase2_model(model_name: str, num_classes: int):
    from phase2.models import build_phase2_model
    return build_phase2_model(model_name, num_classes=num_classes, pretrained=False)


def profile_phase1(models, device):
    rows = []

    for model_name in models:
        print("=" * 80)
        print("Profile Phase 1:", model_name)

        try:
            cfg, cfg_path = load_cfg(model_name)
            model, dummy = build_phase1_from_module(model_name, cfg)

            params, trainable = count_params(model)
            macs = profile_macs(model.cpu(), dummy.cpu())

            latency = measure_latency_ms(
                model=model,
                dummy=dummy,
                device=device,
                warmup=20,
                iters=100,
            )

            rows.append({
                "phase": "phase1_temporal",
                "model": model_name,
                "input_shape": str(list(dummy.shape)),
                "params": params,
                "trainable_params": trainable,
                "params_million": params / 1e6,
                "macs_g": None if macs is None else macs / 1e9,
                "flops_g_approx_2x_macs": None if macs is None else 2 * macs / 1e9,
                "inference_time_ms": latency,
                "config": str(cfg_path),
            })

        except Exception as e:
            print("[ERROR]", model_name, repr(e))
            rows.append({
                "phase": "phase1_temporal",
                "model": model_name,
                "input_shape": "[1,30,1056]",
                "error": repr(e),
            })

    return rows


def profile_phase2(models, device, num_classes=13, clip_len=16, image_size=224):
    rows = []

    for model_name in models:
        print("=" * 80)
        print("Profile Phase 2:", model_name)

        try:
            model = build_phase2_model(model_name, num_classes=num_classes)
            dummy = torch.randn(1, clip_len, 3, image_size, image_size)

            params, trainable = count_params(model)
            macs = profile_macs(model.cpu(), dummy.cpu())

            latency = measure_latency_ms(
                model=model,
                dummy=dummy,
                device=device,
                warmup=10,
                iters=30,
            )

            rows.append({
                "phase": "phase2_classifier",
                "model": model_name,
                "input_shape": str([1, clip_len, 3, image_size, image_size]),
                "params": params,
                "trainable_params": trainable,
                "params_million": params / 1e6,
                "macs_g": None if macs is None else macs / 1e9,
                "flops_g_approx_2x_macs": None if macs is None else 2 * macs / 1e9,
                "inference_time_ms": latency,
                "config": "",
            })

        except Exception as e:
            print("[ERROR]", model_name, repr(e))
            rows.append({
                "phase": "phase2_classifier",
                "model": model_name,
                "input_shape": str([1, clip_len, 3, image_size, image_size]),
                "error": repr(e),
            })

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/home/grouphahieu/imagenet/UCF-Crime/analysis_outputs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--phase1-models", nargs="+", default=["evit", "lstm", "tcn", "transformer", "stgnn"])
    ap.add_argument("--phase2-models", nargs="+", default=[
        "simple_cnn",
        "resnet18",
        "resnet50",
        "efficientnet_b0",
        "efficientnet_b3",
        "vit_b_16",
        "swin_t",
        "convnext_tiny",
    ])
    ap.add_argument("--clip-len", type=int, default=16)
    ap.add_argument("--image-size", type=int, default=224)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    print("device:", device)

    rows = []
    rows += profile_phase1(args.phase1_models, device)
    rows += profile_phase2(args.phase2_models, device, clip_len=args.clip_len, image_size=args.image_size)

    df = pd.DataFrame(rows)

    out_csv = out_dir / "efficiency_flops_params_inference_time.csv"
    out_md = out_dir / "efficiency_flops_params_inference_time.md"
    out_tex = out_dir / "efficiency_flops_params_inference_time.tex"

    df.to_csv(out_csv, index=False)
    df.to_markdown(out_md, index=False)
    df.to_latex(out_tex, index=False, float_format="%.4f")

    print("Saved:")
    print(out_csv)
    print(out_md)
    print(out_tex)
    print()
    print(df)


if __name__ == "__main__":
    main()
