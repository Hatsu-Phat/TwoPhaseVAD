#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
)

sys.path.append(str(Path(__file__).resolve().parents[1]))

from phase2.datasets import Phase2ClipDataset
from phase2.models import build_phase2_model


def collate(batch):
    return {
        "clip": torch.stack([b["clip"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "clip_path": [b["clip_path"] for b in batch],
        "class_name": [b["class_name"] for b in batch],
    }


def load_class_names(manifest_path: Path, num_classes: int):
    class_map_path = manifest_path.parent / "class_map.json"

    if class_map_path.exists():
        with open(class_map_path, "r", encoding="utf-8") as f:
            class_map = json.load(f)

        names = [None] * num_classes
        for name, idx in class_map.items():
            idx = int(idx)
            if 0 <= idx < num_classes:
                names[idx] = name

        for i in range(num_classes):
            if names[i] is None:
                names[i] = f"class_{i}"

        return names

    id_to_name = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            id_to_name[int(r["class_id"])] = r["class_name"]

    return [id_to_name.get(i, f"class_{i}") for i in range(num_classes)]


def save_matrix_csv(path: Path, matrix: np.ndarray, class_names):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + class_names)
        for name, row in zip(class_names, matrix):
            writer.writerow([name] + row.tolist())


def save_matrix_png(path: Path, matrix: np.ndarray, class_names, title: str, normalized: bool):
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
            val = matrix[i, j]
            text = f"{val:.2f}" if normalized else str(int(val))
            ax.text(j, i, text, ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-classes", type=int, default=13)
    ap.add_argument("--clip-len", type=int, default=16)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )

    ds = Phase2ClipDataset(
        manifest=manifest_path,
        split=args.split,
        clip_len=args.clip_len,
        image_size=args.image_size,
        train=False,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )

    model = build_phase2_model(
        args.model,
        num_classes=args.num_classes,
        pretrained=False,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=True)
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)

    model.eval()

    y_true = []
    y_pred = []
    pred_rows = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval {args.model} {args.split}"):
            x = batch["clip"].to(device)
            y = batch["label"].to(device)

            logits = model(x)
            prob = torch.softmax(logits, dim=1)
            pred = prob.argmax(dim=1)

            y_true.extend(y.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())

            for clip_path, true_id, pred_id, probs in zip(
                batch["clip_path"],
                y.cpu().tolist(),
                pred.cpu().tolist(),
                prob.cpu().tolist(),
            ):
                pred_rows.append({
                    "clip_path": clip_path,
                    "true_id": int(true_id),
                    "pred_id": int(pred_id),
                    "pred_score": float(max(probs)),
                    "probs": [float(v) for v in probs],
                })

    class_names = load_class_names(manifest_path, args.num_classes)
    labels = list(range(args.num_classes))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    summary = {
        "model": args.model,
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "num_samples": len(y_true),
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "class_names": class_names,
    }

    save_matrix_csv(out_dir / f"confusion_matrix_{args.split}.csv", cm, class_names)
    save_matrix_csv(out_dir / f"confusion_matrix_{args.split}_normalized.csv", cm_norm, class_names)

    save_matrix_png(
        out_dir / f"confusion_matrix_{args.split}.png",
        cm,
        class_names,
        f"{args.model} confusion matrix ({args.split})",
        normalized=False,
    )

    save_matrix_png(
        out_dir / f"confusion_matrix_{args.split}_normalized.png",
        cm_norm,
        class_names,
        f"{args.model} normalized confusion matrix ({args.split})",
        normalized=True,
    )

    with open(out_dir / f"classification_report_{args.split}.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    with open(out_dir / f"classification_report_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)

    with open(out_dir / f"summary_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(out_dir / f"predictions_{args.split}.jsonl", "w", encoding="utf-8") as f:
        for r in pred_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print()
    print(report_text)
    print("Saved to:", out_dir)


if __name__ == "__main__":
    main()
