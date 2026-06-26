#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_DISPLAY = {
    "swin_t": "Swin-T",
    "convnext_tiny": "ConvNeXt-Tiny",
    "vit_b_16": "ViT-B/16",
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "efficientnet_b0": "EfficientNet-B0",
    "efficientnet_b3": "EfficientNet-B3",
    "simple_cnn": "CNN",
}


def get_col(df, candidates, default=None):
    for c in candidates:
        if c in df.columns:
            return c
    return default


def save_bar_with_error(df, x_col, y_col, err_col, title, ylabel, out_path, rotate=True):
    df = df.copy()
    df = df.dropna(subset=[y_col])

    if df.empty:
        print("[SKIP EMPTY]", title)
        return

    x = np.arange(len(df))
    y = df[y_col].astype(float).values

    err = None
    if err_col and err_col in df.columns:
        err = df[err_col].astype(float).values

    plt.figure(figsize=(8, 5))
    plt.bar(x, y, yerr=err, capsize=4)
    plt.xticks(
        x,
        df[x_col].astype(str).values,
        rotation=25 if rotate else 0,
        ha="right" if rotate else "center",
    )
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

    print("saved:", out_path)


def phase1_plots(ucf: Path, out_dir: Path):
    p = ucf / "phase1_cv_temporal_metrics_model_comparison.csv"

    if not p.exists():
        print("[SKIP] Missing Phase 1 CV:", p)
        return

    df = pd.read_csv(p)

    if "model" not in df.columns:
        print("[SKIP] Phase 1 CSV has no model column:", p)
        return

    df["model_display"] = df["model"].map(MODEL_DISPLAY).fillna(df["model"])

    save_bar_with_error(
        df=df,
        x_col="model_display",
        y_col="frame_auc_mean",
        err_col="frame_auc_std",
        title="Phase 1 sensitivity: frame-level AUC across temporal models",
        ylabel="Frame-level AUC",
        out_path=out_dir / "phase1_sensitivity_frame_auc.png",
    )

    if "false_alarms_per_hour_mean" in df.columns:
        save_bar_with_error(
            df=df,
            x_col="model_display",
            y_col="false_alarms_per_hour_mean",
            err_col="false_alarms_per_hour_std",
            title="Phase 1 sensitivity: false alarms per hour",
            ylabel="False alarms/hour",
            out_path=out_dir / "phase1_sensitivity_false_alarms_per_hour.png",
        )

    if "mean_best_temporal_iou_mean" in df.columns:
        save_bar_with_error(
            df=df,
            x_col="model_display",
            y_col="mean_best_temporal_iou_mean",
            err_col="mean_best_temporal_iou_std",
            title="Phase 1 sensitivity: mean best temporal IoU",
            ylabel="Mean best temporal IoU",
            out_path=out_dir / "phase1_sensitivity_temporal_iou.png",
        )

    delay_col = get_col(df, ["detection_delay_sec_mean_mean", "detection_delay_sec_mean"])
    delay_std_col = get_col(df, ["detection_delay_sec_mean_std", "detection_delay_sec_std"])

    if delay_col:
        save_bar_with_error(
            df=df,
            x_col="model_display",
            y_col=delay_col,
            err_col=delay_std_col,
            title="Phase 1 sensitivity: detection delay",
            ylabel="Detection delay (seconds)",
            out_path=out_dir / "phase1_sensitivity_detection_delay.png",
        )

    if "frame_auc_mean" in df.columns and "false_alarms_per_hour_mean" in df.columns:
        temp = df.dropna(subset=["frame_auc_mean", "false_alarms_per_hour_mean"])
        if not temp.empty:
            plt.figure(figsize=(7, 5))
            plt.scatter(
                temp["false_alarms_per_hour_mean"].astype(float),
                temp["frame_auc_mean"].astype(float),
            )

            for _, r in temp.iterrows():
                plt.annotate(
                    str(r["model_display"]),
                    (
                        float(r["false_alarms_per_hour_mean"]),
                        float(r["frame_auc_mean"]),
                    ),
                    fontsize=9,
                    xytext=(5, 5),
                    textcoords="offset points",
                )

            plt.xlabel("False alarms/hour")
            plt.ylabel("Frame-level AUC")
            plt.title("Phase 1 trade-off: AUC vs false alarms/hour")
            plt.tight_layout()
            out_path = out_dir / "phase1_tradeoff_auc_vs_false_alarms.png"
            plt.savefig(out_path, dpi=220)
            plt.close()
            print("saved:", out_path)


def load_phase2_transformer_fold2_only(ucf: Path):
    path = ucf / "phase2_cv_full_metrics_model_comparison_transformer_fold2_noleak.csv"

    if not path.exists():
        print("[SKIP] Missing Phase 2 CV transformer fold2 no-leak:", path)
        return pd.DataFrame()

    df = pd.read_csv(path)

    if "model" not in df.columns:
        print("[SKIP] Phase 2 CSV has no model column:", path)
        return pd.DataFrame()

    df = df[df["model"].isin(["swin_t", "convnext_tiny"])].copy()

    if "macro_f1_mean" in df.columns and "f1_macro_mean" not in df.columns:
        df["f1_macro_mean"] = df["macro_f1_mean"]
    if "macro_f1_std" in df.columns and "f1_macro_std" not in df.columns:
        df["f1_macro_std"] = df["macro_f1_std"]

    if "acc_mean" in df.columns and "accuracy_mean" not in df.columns:
        df["accuracy_mean"] = df["acc_mean"]
    if "acc_std" in df.columns and "accuracy_std" not in df.columns:
        df["accuracy_std"] = df["acc_std"]

    df["model_display"] = df["model"].map(MODEL_DISPLAY).fillna(df["model"])

    return df


def phase2_plots(ucf: Path, out_dir: Path):
    df = load_phase2_transformer_fold2_only(ucf)

    if df.empty:
        print("[SKIP] No Phase 2 transformer fold2 no-leak results found")
        return

    combined_csv = out_dir / "phase2_sensitivity_transformer_fold2_noleak_only.csv"
    df.to_csv(combined_csv, index=False)
    print("saved:", combined_csv)

    if "f1_macro_mean" in df.columns:
        temp = df.dropna(subset=["f1_macro_mean"]).copy()
        temp = temp.sort_values("f1_macro_mean", ascending=False)

        save_bar_with_error(
            df=temp,
            x_col="model_display",
            y_col="f1_macro_mean",
            err_col="f1_macro_std" if "f1_macro_std" in temp.columns else None,
            title="Phase 2 sensitivity: macro-F1 of classifiers",
            ylabel="Macro-F1",
            out_path=out_dir / "phase2_sensitivity_macro_f1_transformer_fold2_noleak.png",
            rotate=False,
        )

    if "accuracy_mean" in df.columns:
        temp = df.dropna(subset=["accuracy_mean"]).copy()
        temp = temp.sort_values("accuracy_mean", ascending=False)

        save_bar_with_error(
            df=temp,
            x_col="model_display",
            y_col="accuracy_mean",
            err_col="accuracy_std" if "accuracy_std" in temp.columns else None,
            title="Phase 2 sensitivity: accuracy of classifiers",
            ylabel="Accuracy",
            out_path=out_dir / "phase2_sensitivity_accuracy_transformer_fold2_noleak.png",
            rotate=False,
        )

    metric_specs = [
        ("precision_macro_mean", "precision_macro_std", "Macro precision", "phase2_sensitivity_precision_transformer_fold2_noleak.png"),
        ("recall_macro_mean", "recall_macro_std", "Macro recall", "phase2_sensitivity_recall_transformer_fold2_noleak.png"),
        ("event_recall_macro_mean", "event_recall_macro_std", "Event-level recall", "phase2_sensitivity_event_recall_transformer_fold2_noleak.png"),
    ]

    for mean_col, std_col, ylabel, filename in metric_specs:
        if mean_col in df.columns:
            temp = df.dropna(subset=[mean_col]).copy()
            temp = temp.sort_values(mean_col, ascending=False)

            save_bar_with_error(
                df=temp,
                x_col="model_display",
                y_col=mean_col,
                err_col=std_col if std_col in temp.columns else None,
                title=f"Phase 2 sensitivity: {ylabel}",
                ylabel=ylabel,
                out_path=out_dir / filename,
                rotate=False,
            )


def efficiency_tradeoff(ucf: Path, out_dir: Path):
    eff_path = out_dir / "efficiency_flops_params_inference_time_end2end.csv"
    if not eff_path.exists():
        eff_path = out_dir / "efficiency_flops_params_inference_time.csv"

    if not eff_path.exists():
        print("[SKIP] Missing efficiency table:", eff_path)
        return

    eff = pd.read_csv(eff_path)
    phase2 = load_phase2_transformer_fold2_only(ucf)

    if phase2.empty or "f1_macro_mean" not in phase2.columns:
        print("[SKIP] Missing Phase 2 macro-F1 for efficiency trade-off")
        return

    phase2_eff = eff[eff["phase"] == "phase2_classifier"].copy()

    merged = phase2.merge(phase2_eff, on="model", how="left")
    merged["model_display"] = merged["model"].map(MODEL_DISPLAY).fillna(merged["model"])

    merged_csv = out_dir / "phase2_efficiency_vs_macro_f1_transformer_fold2_noleak.csv"
    merged.to_csv(merged_csv, index=False)
    print("saved:", merged_csv)

    temp = merged.dropna(subset=["f1_macro_mean", "inference_time_ms"])

    if not temp.empty:
        plt.figure(figsize=(7, 5))
        plt.scatter(temp["inference_time_ms"], temp["f1_macro_mean"])

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
        plt.title("Phase 2 trade-off: inference time vs macro-F1")
        plt.tight_layout()

        out_path = out_dir / "phase2_tradeoff_inference_time_vs_macro_f1_transformer_fold2_noleak.png"
        plt.savefig(out_path, dpi=220)
        plt.close()
        print("saved:", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ucf-root", default="/home/grouphahieu/imagenet/UCF-Crime")
    ap.add_argument("--out-dir", default="/home/grouphahieu/imagenet/UCF-Crime/analysis_outputs")
    args = ap.parse_args()

    ucf = Path(args.ucf_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phase1_plots(ucf, out_dir)
    phase2_plots(ucf, out_dir)
    efficiency_tradeoff(ucf, out_dir)

    print("DONE. outputs in:", out_dir)


if __name__ == "__main__":
    main()
