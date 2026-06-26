#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GREEN = "#2ca02c"

MODEL_DISPLAY = {
    "evit": "EViT",
    "lstm": "LSTM",
    "tcn": "TCN",
    "transformer": "Transformer",
    "stgnn": "ST-GNN",
    "swin_t": "Swin-T",
    "convnext_tiny": "ConvNeXt-Tiny",
}


def get_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def save_bar(df, x_col, y_col, err_col, title, ylabel, out_path, rotate=True):
    if y_col not in df.columns:
        print("[SKIP missing column]", y_col)
        return

    temp = df.dropna(subset=[y_col]).copy()
    if temp.empty:
        print("[SKIP empty]", title)
        return

    x = np.arange(len(temp))
    y = temp[y_col].astype(float).values

    err = None
    if err_col and err_col in temp.columns:
        err = temp[err_col].astype(float).values

    plt.figure(figsize=(8, 5))
    plt.bar(x, y, yerr=err, capsize=4, color=GREEN)
    plt.xticks(
        x,
        temp[x_col].astype(str).values,
        rotation=25 if rotate else 0,
        ha="right" if rotate else "center",
    )
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    print("saved:", out_path)


def phase1_sensitivity(xd: Path, out_dir: Path):
    p = xd / "phase1_cv_temporal_metrics_model_comparison.csv"
    if not p.exists():
        print("[SKIP] Missing Phase 1 CSV:", p)
        return

    df = pd.read_csv(p)

    if "model" not in df.columns:
        print("[SKIP] Phase 1 CSV has no model column")
        return

    df["model_display"] = df["model"].map(MODEL_DISPLAY).fillna(df["model"])

    save_bar(
        df=df,
        x_col="model_display",
        y_col="frame_ap_mean",
        err_col="frame_ap_std",
        title="XD-Violence Phase 1 sensitivity: frame-level AP",
        ylabel="Frame-level AP",
        out_path=out_dir / "xd_phase1_sensitivity_frame_ap_green.png",
    )

    save_bar(
        df=df,
        x_col="model_display",
        y_col="frame_auc_mean",
        err_col="frame_auc_std",
        title="XD-Violence Phase 1 sensitivity: frame-level AUC",
        ylabel="Frame-level AUC",
        out_path=out_dir / "xd_phase1_sensitivity_frame_auc_green.png",
    )

    save_bar(
        df=df,
        x_col="model_display",
        y_col="false_alarms_per_hour_mean",
        err_col="false_alarms_per_hour_std",
        title="XD-Violence Phase 1 sensitivity: false alarms per hour",
        ylabel="False alarms/hour",
        out_path=out_dir / "xd_phase1_sensitivity_false_alarms_per_hour_green.png",
    )

    save_bar(
        df=df,
        x_col="model_display",
        y_col="mean_best_temporal_iou_mean",
        err_col="mean_best_temporal_iou_std",
        title="XD-Violence Phase 1 sensitivity: temporal IoU",
        ylabel="Mean best temporal IoU",
        out_path=out_dir / "xd_phase1_sensitivity_temporal_iou_green.png",
    )

    delay_col = get_col(df, ["detection_delay_sec_mean_mean", "detection_delay_sec_mean"])
    delay_std_col = get_col(df, ["detection_delay_sec_mean_std", "detection_delay_sec_std"])

    if delay_col:
        save_bar(
            df=df,
            x_col="model_display",
            y_col=delay_col,
            err_col=delay_std_col,
            title="XD-Violence Phase 1 sensitivity: detection delay",
            ylabel="Detection delay (seconds)",
            out_path=out_dir / "xd_phase1_sensitivity_detection_delay_green.png",
        )

    if "frame_ap_mean" in df.columns and "false_alarms_per_hour_mean" in df.columns:
        temp = df.dropna(subset=["frame_ap_mean", "false_alarms_per_hour_mean"]).copy()
        if not temp.empty:
            plt.figure(figsize=(7, 5))
            plt.scatter(
                temp["false_alarms_per_hour_mean"].astype(float),
                temp["frame_ap_mean"].astype(float),
                color=GREEN,
            )

            for _, r in temp.iterrows():
                plt.annotate(
                    str(r["model_display"]),
                    (
                        float(r["false_alarms_per_hour_mean"]),
                        float(r["frame_ap_mean"]),
                    ),
                    fontsize=9,
                    xytext=(5, 5),
                    textcoords="offset points",
                )

            plt.xlabel("False alarms/hour")
            plt.ylabel("Frame-level AP")
            plt.title("XD-Violence Phase 1 trade-off: AP vs false alarms/hour")
            plt.tight_layout()
            out_path = out_dir / "xd_phase1_tradeoff_ap_vs_false_alarms_green.png"
            plt.savefig(out_path, dpi=220)
            plt.close()
            print("saved:", out_path)


def phase2_sensitivity(xd: Path, out_dir: Path, run_prefix: str):
    p = xd / f"phase2_cv_full_metrics_model_comparison_{run_prefix}.csv"

    if not p.exists():
        print("[WARN] Missing:", p)
        fallback = xd / "phase2_cv_full_metrics_model_comparison.csv"
        if fallback.exists():
            print("[INFO] Using fallback:", fallback)
            p = fallback
        else:
            print("[SKIP] Missing Phase 2 CSV")
            return

    df = pd.read_csv(p)

    if "model" not in df.columns:
        print("[SKIP] Phase 2 CSV has no model column")
        return

    df = df[df["model"].isin(["swin_t", "convnext_tiny"])].copy()
    df["model_display"] = df["model"].map(MODEL_DISPLAY).fillna(df["model"])

    if "macro_f1_mean" in df.columns and "f1_macro_mean" not in df.columns:
        df["f1_macro_mean"] = df["macro_f1_mean"]
    if "macro_f1_std" in df.columns and "f1_macro_std" not in df.columns:
        df["f1_macro_std"] = df["macro_f1_std"]

    if "acc_mean" in df.columns and "accuracy_mean" not in df.columns:
        df["accuracy_mean"] = df["acc_mean"]
    if "acc_std" in df.columns and "accuracy_std" not in df.columns:
        df["accuracy_std"] = df["acc_std"]

    out_csv = out_dir / f"xd_phase2_sensitivity_{run_prefix}_only.csv"
    df.to_csv(out_csv, index=False)
    print("saved:", out_csv)

    save_bar(
        df=df.sort_values("f1_macro_mean", ascending=False) if "f1_macro_mean" in df.columns else df,
        x_col="model_display",
        y_col="f1_macro_mean",
        err_col="f1_macro_std",
        title="XD-Violence Phase 2 sensitivity: macro-F1",
        ylabel="Macro-F1",
        out_path=out_dir / f"xd_phase2_sensitivity_macro_f1_{run_prefix}_green.png",
        rotate=False,
    )

    save_bar(
        df=df.sort_values("accuracy_mean", ascending=False) if "accuracy_mean" in df.columns else df,
        x_col="model_display",
        y_col="accuracy_mean",
        err_col="accuracy_std",
        title="XD-Violence Phase 2 sensitivity: accuracy",
        ylabel="Accuracy",
        out_path=out_dir / f"xd_phase2_sensitivity_accuracy_{run_prefix}_green.png",
        rotate=False,
    )

    save_bar(
        df=df,
        x_col="model_display",
        y_col="precision_macro_mean",
        err_col="precision_macro_std",
        title="XD-Violence Phase 2 sensitivity: macro precision",
        ylabel="Macro precision",
        out_path=out_dir / f"xd_phase2_sensitivity_precision_{run_prefix}_green.png",
        rotate=False,
    )

    save_bar(
        df=df,
        x_col="model_display",
        y_col="recall_macro_mean",
        err_col="recall_macro_std",
        title="XD-Violence Phase 2 sensitivity: macro recall",
        ylabel="Macro recall",
        out_path=out_dir / f"xd_phase2_sensitivity_recall_{run_prefix}_green.png",
        rotate=False,
    )


def efficiency_tradeoff(xd: Path, out_dir: Path, run_prefix: str):
    eff_path = out_dir / "xd_efficiency_params_flops_inference.csv"
    if not eff_path.exists():
        print("[SKIP] Missing efficiency CSV:", eff_path)
        return

    phase2_path = xd / f"phase2_cv_full_metrics_model_comparison_{run_prefix}.csv"
    if not phase2_path.exists():
        phase2_path = xd / "phase2_cv_full_metrics_model_comparison.csv"

    if not phase2_path.exists():
        print("[SKIP] Missing phase2 metrics for trade-off")
        return

    eff = pd.read_csv(eff_path)
    phase2 = pd.read_csv(phase2_path)

    if "macro_f1_mean" in phase2.columns and "f1_macro_mean" not in phase2.columns:
        phase2["f1_macro_mean"] = phase2["macro_f1_mean"]

    phase2 = phase2[phase2["model"].isin(["swin_t", "convnext_tiny"])].copy()

    eff2 = eff[eff["phase"] == "phase2_classifier"].copy()
    merged = phase2.merge(eff2, on="model", how="left")
    merged["model_display"] = merged["model"].map(MODEL_DISPLAY).fillna(merged["model"])

    out_csv = out_dir / f"xd_phase2_efficiency_vs_macro_f1_{run_prefix}.csv"
    merged.to_csv(out_csv, index=False)
    print("saved:", out_csv)

    temp = merged.dropna(subset=["f1_macro_mean", "inference_time_ms"])
    if not temp.empty:
        plt.figure(figsize=(7, 5))
        plt.scatter(
            temp["inference_time_ms"].astype(float),
            temp["f1_macro_mean"].astype(float),
            color=GREEN,
        )

        for _, r in temp.iterrows():
            plt.annotate(
                str(r["model_display"]),
                (float(r["inference_time_ms"]), float(r["f1_macro_mean"])),
                fontsize=9,
                xytext=(5, 5),
                textcoords="offset points",
            )

        plt.xlabel("Inference time (ms/sample)")
        plt.ylabel("Macro-F1")
        plt.title("XD-Violence Phase 2 trade-off: inference time vs macro-F1")
        plt.tight_layout()

        out_path = out_dir / f"xd_phase2_tradeoff_inference_time_vs_macro_f1_{run_prefix}_green.png"
        plt.savefig(out_path, dpi=220)
        plt.close()
        print("saved:", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xd-root", default="/home/grouphahieu/imagenet/XD_Violence")
    ap.add_argument("--out-dir", default="/home/grouphahieu/imagenet/XD_Violence/analysis_outputs")
    ap.add_argument("--run-prefix", default="topk8_transformer_fold0")
    args = ap.parse_args()

    xd = Path(args.xd_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phase1_sensitivity(xd, out_dir)
    phase2_sensitivity(xd, out_dir, args.run_prefix)
    efficiency_tradeoff(xd, out_dir, args.run_prefix)

    print("DONE. outputs in:", out_dir)


if __name__ == "__main__":
    main()
