#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
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
        class_map = json.load(open(class_map_path, encoding="utf-8"))
        names = [None] * num_classes
        for name, idx in class_map.items():
            idx = int(idx)
            if 0 <= idx < num_classes:
                names[idx] = name
        return [n if n is not None else f"class_{i}" for i, n in enumerate(names)]

    id_to_name = {}
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            id_to_name[int(r["class_id"])] = r["class_name"]
    return [id_to_name.get(i, f"class_{i}") for i in range(num_classes)]


def load_manifest_rows(manifest_path: Path):
    rows = []
    by_clip = {}
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append(r)
            by_clip[str(r["clip_path"])] = r
    return rows, by_clip


def parse_temporal_annotations(path: Path | None):
    if path is None or not path.exists():
        return {}

    ann = {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        for raw in f:
            parts = raw.strip().split()
            if len(parts) < 6:
                continue

            video_name = parts[0]
            event = parts[1]

            try:
                nums = [int(x) for x in parts[2:6]]
            except Exception:
                continue

            ranges = []
            for a, b in [(nums[0], nums[1]), (nums[2], nums[3])]:
                if a > 0 and b > 0:
                    ranges.append((min(a, b), max(a, b)))

            ann[video_name] = {
                "event": event,
                "ranges": ranges,
            }

    return ann


def load_checkpoint(model, ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=True)
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)

    return model


def save_confusion_matrix(out_dir: Path, cm: np.ndarray, class_names, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{prefix}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + class_names)
        for name, row in zip(class_names, cm):
            w.writerow([name] + row.tolist())

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(13, 11))
        im = ax.imshow(cm)
        ax.set_title(prefix)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)

        is_float = np.issubdtype(cm.dtype, np.floating)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                txt = f"{cm[i, j]:.2f}" if is_float else str(int(cm[i, j]))
                ax.text(j, i, txt, ha="center", va="center", fontsize=7)

        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}.png", dpi=220)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] Could not save PNG confusion matrix: {e}")


def safe_mean_std(vals):
    vals = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals, ddof=0))


def iou_1d(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / max(1, union)


def temporal_optional_metrics(pred_rows, annotation, class_names, iou_thresholds=(0.1, 0.3, 0.5)):
    by_video = defaultdict(list)
    for r in pred_rows:
        src = r.get("rel_path") or r.get("source_video") or ""
        video_name = Path(src).name
        by_video[video_name].append(r)

    gt_events = []
    detected_by_thr = {thr: 0 for thr in iou_thresholds}
    best_ious = []
    delays = []

    false_alarms = 0
    normal_frames = 0

    for video_name, ann in annotation.items():
        ranges = ann.get("ranges", [])
        if not ranges:
            continue

        rows = by_video.get(video_name, [])
        if not rows:
            for rg in ranges:
                gt_events.append((video_name, rg))
            continue

        gt_intervals = [(a - 1, b) for a, b in ranges]

        total_frames = 0
        for r in rows:
            total_frames = max(total_frames, int(r.get("end_frame", 0)))
        if total_frames <= 0:
            total_frames = max(b for _, b in gt_intervals)

        gt_mask = np.zeros(total_frames, dtype=np.uint8)
        for s, e in gt_intervals:
            gt_mask[max(0, s): min(total_frames, e)] = 1
        normal_frames += int((gt_mask == 0).sum())

        pred_intervals = []
        for r in rows:
            s = int(r.get("start_frame", 0))
            e = int(r.get("end_frame", s + 1))
            pred_id = int(r["pred_id"])
            pred_name = class_names[pred_id] if 0 <= pred_id < len(class_names) else str(pred_id)
            score = float(r.get("pred_score", 0.0))
            pred_intervals.append((s, e, pred_name, score))

            if all(iou_1d((s, e), gt) <= 0 for gt in gt_intervals):
                false_alarms += 1

        for gt in gt_intervals:
            gt_events.append((video_name, gt))

            ious = []
            candidates = []
            for pred in pred_intervals:
                ps, pe, pred_name, score = pred
                tiou = iou_1d((ps, pe), gt)
                ious.append(tiou)
                if tiou > 0:
                    candidates.append((ps, pe, tiou, score))

            best_iou = max(ious) if ious else 0.0
            best_ious.append(best_iou)

            for thr in iou_thresholds:
                if best_iou >= thr:
                    detected_by_thr[thr] += 1

            if candidates:
                candidates = sorted(candidates, key=lambda x: x[0])
                first_s = candidates[0][0]
                delay_frames = max(0, first_s - gt[0])
                delays.append(delay_frames / 30.0)

    num_gt = len(gt_events)
    out = {
        "temporal_num_gt_events": num_gt,
        "temporal_mean_best_tiou": float(np.mean(best_ious)) if best_ious else None,
        "temporal_detection_delay_sec_mean": float(np.mean(delays)) if delays else None,
        "temporal_detection_delay_sec_median": float(np.median(delays)) if delays else None,
    }

    for thr in iou_thresholds:
        out[f"event_recall_at_tiou_{thr}"] = (
            detected_by_thr[thr] / num_gt if num_gt > 0 else None
        )

    normal_hours = normal_frames / 30.0 / 3600.0 if normal_frames > 0 else 0.0
    out["false_alarms"] = false_alarms if num_gt > 0 else None
    out["false_alarms_per_hour_proxy"] = (
        false_alarms / normal_hours if normal_hours > 0 and num_gt > 0 else None
    )

    return out


@torch.no_grad()
def evaluate_one_fold(
    model_name,
    fold,
    manifest_path,
    checkpoint_path,
    out_dir,
    num_classes,
    clip_len,
    image_size,
    batch_size,
    device,
    num_workers,
    annotation,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(manifest_path, num_classes)
    _, manifest_by_clip = load_manifest_rows(manifest_path)

    ds = Phase2ClipDataset(
        manifest=manifest_path,
        split="val",
        clip_len=clip_len,
        image_size=image_size,
        train=False,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
    )

    model = build_phase2_model(model_name, num_classes=num_classes, pretrained=False).to(device)
    model = load_checkpoint(model, checkpoint_path)
    model.eval()

    y_true = []
    y_pred = []
    pred_rows = []

    for batch in tqdm(loader, desc=f"Eval {model_name} fold {fold}"):
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
            row = dict(manifest_by_clip.get(str(clip_path), {}))
            row.update({
                "clip_path": str(clip_path),
                "true_id": int(true_id),
                "pred_id": int(pred_id),
                "pred_score": float(max(probs)),
                "probs": [float(v) for v in probs],
            })
            pred_rows.append(row)

    labels = list(range(num_classes))

    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    save_confusion_matrix(out_dir, cm, class_names, f"confusion_matrix_fold{fold}")
    save_confusion_matrix(out_dir, cm_norm, class_names, f"confusion_matrix_fold{fold}_normalized")

    with open(out_dir / f"classification_report_fold{fold}.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    with open(out_dir / f"classification_report_fold{fold}.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)

    with open(out_dir / f"predictions_fold{fold}.jsonl", "w", encoding="utf-8") as f:
        for r in pred_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_src = defaultdict(list)
    for r in pred_rows:
        src = r.get("rel_path") or r.get("source_video") or r["clip_path"]
        by_src[src].append(r)

    event_y_true = []
    event_y_pred = []

    for src, rows in by_src.items():
        true_id = int(rows[0]["true_id"])

        prob_mean = np.mean(np.array([r["probs"] for r in rows], dtype=np.float64), axis=0)
        pred_id = int(np.argmax(prob_mean))

        event_y_true.append(true_id)
        event_y_pred.append(pred_id)

    event_acc = accuracy_score(event_y_true, event_y_pred) if event_y_true else None
    event_p_macro, event_r_macro, event_f_macro, _ = precision_recall_fscore_support(
        event_y_true,
        event_y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    ) if event_y_true else (None, None, None, None)

    temporal = temporal_optional_metrics(pred_rows, annotation, class_names)

    summary = {
        "model": model_name,
        "fold": fold,
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "num_val_clips": len(y_true),

        "accuracy": float(acc),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f_macro),
        "precision_weighted": float(p_weighted),
        "recall_weighted": float(r_weighted),
        "f1_weighted": float(f_weighted),

        "event_num_source_videos": len(event_y_true),
        "event_accuracy": float(event_acc) if event_acc is not None else None,
        "event_precision_macro": float(event_p_macro) if event_p_macro is not None else None,
        "event_recall_macro": float(event_r_macro) if event_r_macro is not None else None,
        "event_f1_macro": float(event_f_macro) if event_f_macro is not None else None,
    }
    summary.update(temporal)

    with open(out_dir / f"summary_fold{fold}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ucf-root", default="/home/grouphahieu/imagenet/UCF-Crime")
    ap.add_argument("--phase2-root", default=None)
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--run-prefix", default="topk8_lstm")
    ap.add_argument("--models", nargs="+", default=["swin_t", "convnext_tiny", "vit_b_16"])
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--num-classes", type=int, default=13)
    ap.add_argument("--clip-len", type=int, default=16)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--vit-batch-size", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--annotation-file", default=None)
    args = ap.parse_args()

    ucf = Path(args.ucf_root)
    phase2_root = Path(args.phase2_root) if args.phase2_root else ucf / "phase2_clips_topk8_framelevel_top30_from_lstm"
    output_root = Path(args.output_root) if args.output_root else ucf / "outputs_phase2_cv"

    annotation_file = Path(args.annotation_file) if args.annotation_file else ucf / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
    annotation = parse_temporal_annotations(annotation_file)

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    all_rows = []

    for model_name in args.models:
        for fold in args.folds:
            manifest_path = phase2_root / f"manifest_phase2_video_cv_fold{fold}.jsonl"
            checkpoint_path = output_root / f"{args.run_prefix}_{model_name}_fold{fold}" / "best.pth"
            eval_dir = output_root / f"{args.run_prefix}_{model_name}_fold{fold}" / "eval_full_metrics"

            if not manifest_path.exists():
                print(f"[SKIP] Missing manifest: {manifest_path}")
                continue

            if not checkpoint_path.exists():
                print(f"[SKIP] Missing checkpoint: {checkpoint_path}")
                continue

            bs = args.vit_batch_size if model_name == "vit_b_16" else args.batch_size

            summary = evaluate_one_fold(
                model_name=model_name,
                fold=fold,
                manifest_path=manifest_path,
                checkpoint_path=checkpoint_path,
                out_dir=eval_dir,
                num_classes=args.num_classes,
                clip_len=args.clip_len,
                image_size=args.image_size,
                batch_size=bs,
                device=device,
                num_workers=args.num_workers,
                annotation=annotation,
            )
            all_rows.append(summary)

            print(json.dumps(summary, indent=2, ensure_ascii=False))

    per_fold_csv = ucf / "phase2_cv_full_metrics_per_fold.csv"
    if all_rows:
        fieldnames = sorted(set().union(*[r.keys() for r in all_rows]))
        with open(per_fold_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)

    model_rows = []
    by_model = defaultdict(list)
    for r in all_rows:
        by_model[r["model"]].append(r)

    metrics_to_avg = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "event_accuracy",
        "event_precision_macro",
        "event_recall_macro",
        "event_f1_macro",
        "temporal_mean_best_tiou",
        "temporal_detection_delay_sec_mean",
        "temporal_detection_delay_sec_median",
        "event_recall_at_tiou_0.1",
        "event_recall_at_tiou_0.3",
        "event_recall_at_tiou_0.5",
        "false_alarms_per_hour_proxy",
    ]

    for model_name, rows in by_model.items():
        out = {
            "model": model_name,
            "num_folds": len(rows),
        }

        for m in metrics_to_avg:
            vals = [r.get(m) for r in rows if r.get(m) is not None]
            mean, std = safe_mean_std(vals)
            out[f"{m}_mean"] = mean
            out[f"{m}_std"] = std

        model_rows.append(out)

    model_rows = sorted(
        model_rows,
        key=lambda r: -1 if r.get("f1_macro_mean") is None else float(r["f1_macro_mean"]),
        reverse=True,
    )

    model_csv = ucf / "phase2_cv_full_metrics_model_comparison.csv"
    if model_rows:
        fieldnames = sorted(set().union(*[r.keys() for r in model_rows]))
        with open(model_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(model_rows)

    print("\nSaved per-fold:", per_fold_csv)
    print("Saved model comparison:", model_csv)

    print("\n===== CV MODEL COMPARISON, ranked by f1_macro_mean =====")
    for r in model_rows:
        print(
            f"{r['model']:15s} "
            f"f1_macro={r.get('f1_macro_mean')} ± {r.get('f1_macro_std')} "
            f"accuracy={r.get('accuracy_mean')} ± {r.get('accuracy_std')} "
            f"event_recall={r.get('event_recall_macro_mean')} ± {r.get('event_recall_macro_std')}"
        )


if __name__ == "__main__":
    main()
