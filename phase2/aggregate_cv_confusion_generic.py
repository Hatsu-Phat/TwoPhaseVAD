#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import numpy as np


def read_confusion_csv(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    rows = reader[1:]
    class_names = [r[0] for r in rows]
    matrix = np.array([[float(x) for x in r[1:]] for r in rows], dtype=np.float64)
    return class_names, matrix


def save_confusion_csv(path: Path, matrix: np.ndarray, class_names):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + class_names)
        for name, row in zip(class_names, matrix):
            writer.writerow([name] + row.tolist())


def save_confusion_png(path: Path, matrix: np.ndarray, class_names, title: str, normalized: bool):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(matrix)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            text = f"{matrix[i, j]:.2f}" if normalized else str(int(matrix[i, j]))
            ax.text(j, i, text, ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def metrics_from_confusion(cm: np.ndarray, class_names):
    eps = 1e-12
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    pred_count = cm.sum(axis=0)

    precision = tp / np.maximum(pred_count, eps)
    recall = tp / np.maximum(support, eps)
    f1 = 2 * precision * recall / np.maximum(precision + recall, eps)

    accuracy = tp.sum() / np.maximum(cm.sum(), eps)

    summary = {
        "accuracy": float(accuracy),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_precision": float((precision * support).sum() / np.maximum(support.sum(), eps)),
        "weighted_recall": float((recall * support).sum() / np.maximum(support.sum(), eps)),
        "weighted_f1": float((f1 * support).sum() / np.maximum(support.sum(), eps)),
        "total_samples": int(cm.sum()),
    }

    per_class = []
    for i, name in enumerate(class_names):
        per_class.append({
            "class_name": name,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        })

    return summary, per_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ucf-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--run-prefix", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    ucf = Path(args.ucf_root)
    out_root = Path(args.output_root)

    all_model_summary = []

    for model in args.models:
        total_cm = None
        class_names_ref = None
        used_folds = []

        for fold in args.folds:
            cm_path = (
                out_root
                / f"{args.run_prefix}_{model}_fold{fold}"
                / "eval_full_metrics"
                / f"confusion_matrix_fold{fold}.csv"
            )

            if not cm_path.exists():
                print("[SKIP] Missing:", cm_path)
                continue

            class_names, cm = read_confusion_csv(cm_path)

            if total_cm is None:
                class_names_ref = class_names
                total_cm = np.zeros_like(cm, dtype=np.float64)
            else:
                if class_names != class_names_ref:
                    raise RuntimeError(f"Class order mismatch: {model} fold {fold}")

            total_cm += cm
            used_folds.append(fold)

        if total_cm is None:
            print("[SKIP] No confusion matrices found for", model)
            continue

        out_dir = out_root / f"{args.run_prefix}_{model}_cv_aggregate"
        out_dir.mkdir(parents=True, exist_ok=True)

        cm_norm = total_cm / np.maximum(total_cm.sum(axis=1, keepdims=True), 1)

        save_confusion_csv(out_dir / "confusion_matrix_cv_sum.csv", total_cm.astype(int), class_names_ref)
        save_confusion_csv(out_dir / "confusion_matrix_cv_sum_normalized.csv", cm_norm, class_names_ref)

        save_confusion_png(
            out_dir / "confusion_matrix_cv_sum.png",
            total_cm,
            class_names_ref,
            f"{model} CV aggregated confusion matrix",
            normalized=False,
        )

        save_confusion_png(
            out_dir / "confusion_matrix_cv_sum_normalized.png",
            cm_norm,
            class_names_ref,
            f"{model} CV aggregated normalized confusion matrix",
            normalized=True,
        )

        summary, per_class = metrics_from_confusion(total_cm, class_names_ref)
        summary["model"] = model
        summary["used_folds"] = used_folds
        summary["num_folds"] = len(used_folds)

        with open(out_dir / "cv_aggregate_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        with open(out_dir / "cv_aggregate_per_class_metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["class_name", "precision", "recall", "f1", "support"],
            )
            writer.writeheader()
            writer.writerows(per_class)

        all_model_summary.append(summary)

        print("=" * 100)
        print("MODEL:", model)
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    if all_model_summary:
        all_model_summary = sorted(all_model_summary, key=lambda r: r["macro_f1"], reverse=True)
        out_csv = ucf / f"phase2_cv_aggregate_confusion_model_comparison_{args.tag}.csv"

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "model",
                "num_folds",
                "total_samples",
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "weighted_precision",
                "weighted_recall",
                "weighted_f1",
                "used_folds",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_model_summary:
                row = dict(r)
                row["used_folds"] = ",".join(map(str, row["used_folds"]))
                writer.writerow(row)

        print("saved:", out_csv)
        for r in all_model_summary:
            print(r)


if __name__ == "__main__":
    main()
