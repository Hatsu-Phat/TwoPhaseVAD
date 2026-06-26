#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


UCF = Path("/home/grouphahieu/imagenet/UCF-Crime")
OUT_ROOT = UCF / "outputs_phase2_cv"

MODELS = [
    "swin_t",
    "convnext_tiny",
    "vit_b_16",
]

FOLDS = [0, 1, 2, 3, 4]


def read_confusion_csv(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))

    header = reader[0][1:]
    rows = reader[1:]

    class_names = [r[0] for r in rows]
    matrix = np.array([[float(x) for x in r[1:]] for r in rows], dtype=np.float64)

    if header and header != class_names:
        print(f"[WARN] header class order differs from row class order in {path}")

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
            if normalized:
                text = f"{matrix[i, j]:.2f}"
            else:
                text = str(int(matrix[i, j]))
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

    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()

    weighted_precision = (precision * support).sum() / np.maximum(support.sum(), eps)
    weighted_recall = (recall * support).sum() / np.maximum(support.sum(), eps)
    weighted_f1 = (f1 * support).sum() / np.maximum(support.sum(), eps)

    per_class = []
    for i, name in enumerate(class_names):
        per_class.append({
            "class_name": name,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        })

    summary = {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "total_samples": int(cm.sum()),
    }

    return summary, per_class


def main():
    all_model_summary = []

    for model in MODELS:
        print("=" * 100)
        print("MODEL:", model)

        total_cm = None
        class_names_ref = None
        used_folds = []

        for fold in FOLDS:
            cm_path = (
                OUT_ROOT
                / f"topk8_lstm_{model}_fold{fold}"
                / "eval_full_metrics"
                / f"confusion_matrix_fold{fold}.csv"
            )

            if not cm_path.exists():
                print(f"[SKIP] Missing: {cm_path}")
                continue

            class_names, cm = read_confusion_csv(cm_path)

            if class_names_ref is None:
                class_names_ref = class_names
                total_cm = np.zeros_like(cm, dtype=np.float64)
            else:
                if class_names != class_names_ref:
                    raise RuntimeError(
                        f"Class order mismatch for model={model}, fold={fold}"
                    )

            total_cm += cm
            used_folds.append(fold)

        if total_cm is None:
            print(f"[SKIP] No confusion matrices found for {model}")
            continue

        out_dir = OUT_ROOT / f"topk8_lstm_{model}_cv_aggregate"
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

        print("saved:", out_dir)
        print(json.dumps(summary, indent=2, ensure_ascii=False))

        print("\nPer-class metrics:")
        for r in per_class:
            print(
                f"{r['class_name']:15s} "
                f"P={r['precision']:.4f} "
                f"R={r['recall']:.4f} "
                f"F1={r['f1']:.4f} "
                f"support={r['support']}"
            )

    if all_model_summary:
        out_csv = UCF / "phase2_cv_aggregate_confusion_model_comparison.csv"

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

        all_model_summary = sorted(
            all_model_summary,
            key=lambda r: r["macro_f1"],
            reverse=True,
        )

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_model_summary:
                row = dict(r)
                row["used_folds"] = ",".join(map(str, row["used_folds"]))
                writer.writerow(row)

        print("\n" + "=" * 100)
        print("Saved model comparison:", out_csv)
        print("Rank by aggregated macro-F1:")
        for i, r in enumerate(all_model_summary, 1):
            print(
                f"{i}. {r['model']:15s} "
                f"macro_f1={r['macro_f1']:.6f} "
                f"acc={r['accuracy']:.6f} "
                f"weighted_f1={r['weighted_f1']:.6f}"
            )


if __name__ == "__main__":
    main()
