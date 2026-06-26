#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)


VIDEO_EXTS = [".mp4", ".avi", ".mkv", ".mov", ".webm"]

def normalize_video_id(x):
    x = str(x).strip()
    x = x.replace("\\", "/").split("/")[-1]

    lower = x.lower()
    for ext in VIDEO_EXTS:
        if lower.endswith(ext):
            x = x[: -len(ext)]
            break

    x = x.replace("#", "")

    return x

def parse_xd_annotations(path: Path):
    ann = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            parts = raw.strip().split()
            if len(parts) < 3:
                continue

            vid = normalize_video_id(parts[0])

            nums = []
            for x in parts[1:]:
                try:
                    nums.append(int(float(x)))
                except Exception:
                    pass

            ranges = []
            for i in range(0, len(nums) - 1, 2):
                a, b = nums[i], nums[i + 1]
                if a >= 0 and b > a:
                    ranges.append([int(a), int(b)])

            ann[vid] = ranges
    return ann


def unique_idx_score(indices, scores):
    indices = np.asarray(indices, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(indices)
    indices = indices[order]
    scores = scores[order]

    ui, us = [], []
    for idx in np.unique(indices):
        mask = indices == idx
        ui.append(int(idx))
        us.append(float(scores[mask].max()))
    return np.array(ui), np.array(us)


def frame_scores_nearest(total_frames, indices, scores):
    indices, scores = unique_idx_score(indices, scores)

    if total_frames <= 0:
        total_frames = int(indices.max() + 1) if len(indices) else 1

    if len(indices) == 0:
        return np.zeros(total_frames)

    if len(indices) == 1:
        return np.ones(total_frames) * scores[0]

    mids = [(indices[i] + indices[i + 1]) // 2 + 1 for i in range(len(indices) - 1)]
    bounds = [0] + mids + [total_frames]

    fs = np.zeros(total_frames, dtype=np.float64)
    for i, sc in enumerate(scores):
        s = max(0, min(total_frames, int(bounds[i])))
        e = max(0, min(total_frames, int(bounds[i + 1])))
        if e > s:
            fs[s:e] = float(sc)
    return fs


def make_gt(n, ranges):
    y = np.zeros(n, dtype=np.uint8)
    for a, b in ranges:
        s = max(0, int(a))
        e = min(n, int(b))
        if e > s:
            y[s:e] = 1
    return y


def intervals_from_mask(mask):
    intervals = []
    i, n = 0, len(mask)
    while i < n:
        while i < n and not mask[i]:
            i += 1
        if i >= n:
            break
        s = i
        while i < n and mask[i]:
            i += 1
        intervals.append([s, i])
    return intervals


def merge_intervals(intervals, gap):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s - merged[-1][1] <= gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return merged


def iou_1d(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / max(1, union)


def safe_auc(y, s):
    y = np.asarray(y)
    s = np.asarray(s)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def safe_ap(y, s):
    y = np.asarray(y)
    s = np.asarray(s)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    return float(average_precision_score(y, s))


def eval_scores_root(scores_root: Path, annotations, threshold_mode, merge_gap, fps_default):
    files = sorted((scores_root / "test").rglob("*.pt"))

    all_y, all_s = [], []
    video_y, video_s = [], []
    normal_scores = []
    per_video = []

    for p in files:
        x = torch.load(p, map_location="cpu")

        rel = str(x.get("rel_path", ""))
        vid = normalize_video_id(rel)
        cls = str(x.get("class_name", "Normal"))
        video_label = int(x.get("video_label", 0))
        fps = float(x.get("fps", fps_default) or fps_default)
        total_frames = int(x.get("total_frames", 0))

        selected = x["selected_indices"].cpu().numpy()
        scores = x["segment_scores"].float().cpu().numpy()
        fs = frame_scores_nearest(total_frames, selected, scores)

        ranges = annotations.get(vid, [])
        if video_label == 0 or cls == "Normal":
            ranges = []

        y = make_gt(len(fs), ranges)

        all_y.extend(y.tolist())
        all_s.extend(fs.tolist())

        video_y.append(video_label)
        video_s.append(float(x.get("video_score", np.max(scores))))

        if video_label == 0 or cls == "Normal":
            normal_scores.extend(fs.tolist())

        per_video.append({
            "frame_scores": fs,
            "gt": y,
            "ranges": ranges,
            "fps": fps,
        })

    all_y = np.array(all_y)
    all_s = np.array(all_s)

    normal_scores = np.array(normal_scores) if normal_scores else all_s

    if threshold_mode == "p99_normal":
        threshold = float(np.percentile(normal_scores, 99))
    elif threshold_mode == "p95_normal":
        threshold = float(np.percentile(normal_scores, 95))
    elif threshold_mode == "max_normal":
        threshold = float(np.max(normal_scores))
    else:
        threshold = float(np.percentile(normal_scores, 99))

    pred = (all_s >= threshold).astype(np.uint8)

    cm = confusion_matrix(all_y, pred, labels=[0, 1])
    acc = accuracy_score(all_y, pred)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        all_y, pred, labels=[0, 1], average="macro", zero_division=0
    )

    gt_events = 0
    detected = {0.1: 0, 0.3: 0, 0.5: 0}
    best_ious = []
    delays = []
    false_alarms = 0
    normal_frames = 0

    for v in per_video:
        fs = v["frame_scores"]
        gt = v["gt"]
        fps = v["fps"]

        pred_intervals = merge_intervals(intervals_from_mask(fs >= threshold), gap=merge_gap)
        gt_intervals = [[max(0, int(a)), min(len(fs), int(b))] for a, b in v["ranges"]]

        normal_frames += int((gt == 0).sum())

        for pi in pred_intervals:
            if not gt_intervals or all(iou_1d(pi, gi) <= 0 for gi in gt_intervals):
                false_alarms += 1

        for gi in gt_intervals:
            gt_events += 1
            ious = [iou_1d(pi, gi) for pi in pred_intervals]
            best_iou = max(ious) if ious else 0.0
            best_ious.append(best_iou)

            for thr in detected:
                if best_iou >= thr:
                    detected[thr] += 1

            overlaps = [pi for pi in pred_intervals if iou_1d(pi, gi) > 0]
            if overlaps:
                first = sorted(overlaps, key=lambda x: x[0])[0]
                delays.append(max(0, first[0] - gi[0]) / fps)

    normal_hours = normal_frames / fps_default / 3600.0 if normal_frames > 0 else 0

    return {
        "num_videos": len(files),
        "num_frames": int(len(all_y)),
        "num_positive_frames": int(all_y.sum()),
        "threshold_mode": threshold_mode,
        "threshold": threshold,

        "frame_auc": safe_auc(all_y, all_s),
        "frame_ap": safe_ap(all_y, all_s),
        "video_auc": safe_auc(video_y, video_s),
        "video_ap": safe_ap(video_y, video_s),

        "binary_accuracy": float(acc),
        "binary_precision_macro": float(p_macro),
        "binary_recall_macro": float(r_macro),
        "binary_f1_macro": float(f_macro),

        "false_alarms": int(false_alarms),
        "normal_hours": float(normal_hours),
        "false_alarms_per_hour": float(false_alarms / normal_hours) if normal_hours > 0 else None,

        "gt_events": int(gt_events),
        "mean_best_temporal_iou": float(np.mean(best_ious)) if best_ious else None,
        "event_recall_tiou_0.1": float(detected[0.1] / gt_events) if gt_events else None,
        "event_recall_tiou_0.3": float(detected[0.3] / gt_events) if gt_events else None,
        "event_recall_tiou_0.5": float(detected[0.5] / gt_events) if gt_events else None,
        "detection_delay_sec_mean": float(np.mean(delays)) if delays else None,
        "detection_delay_sec_median": float(np.median(delays)) if delays else None,

        "cm_tn": int(cm[0, 0]),
        "cm_fp": int(cm[0, 1]),
        "cm_fn": int(cm[1, 0]),
        "cm_tp": int(cm[1, 1]),
    }


def mean_std(vals):
    vals = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals, ddof=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xd-root", required=True)
    ap.add_argument("--models", nargs="+", default=["evit", "lstm", "tcn", "transformer", "stgnn"])
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--threshold-mode", default="p99_normal")
    ap.add_argument("--merge-gap", type=int, default=16)
    ap.add_argument("--fps", type=float, default=24.0)
    args = ap.parse_args()

    xd = Path(args.xd_root)
    annotations = parse_xd_annotations(xd / "annotations.txt")

    rows = []
    model_rows = []

    metrics = [
        "frame_auc", "frame_ap", "video_auc", "video_ap",
        "binary_accuracy", "binary_precision_macro", "binary_recall_macro", "binary_f1_macro",
        "false_alarms_per_hour",
        "mean_best_temporal_iou",
        "event_recall_tiou_0.1", "event_recall_tiou_0.3", "event_recall_tiou_0.5",
        "detection_delay_sec_mean", "detection_delay_sec_median",
    ]

    for model in args.models:
        sub = []
        for fold in args.folds:
            root = xd / "phase1_scores_cv" / f"top30_{model}_fold{fold}"
            if not root.exists():
                print("[SKIP missing]", root)
                continue

            d = eval_scores_root(root, annotations, args.threshold_mode, args.merge_gap, args.fps)
            d["model"] = model
            d["fold"] = fold
            rows.append(d)
            sub.append(d)
            print(json.dumps(d, indent=2, ensure_ascii=False))

        out = {"model": model, "num_folds": len(sub)}
        for m in metrics:
            avg, std = mean_std([r.get(m) for r in sub])
            out[m + "_mean"] = avg
            out[m + "_std"] = std
        model_rows.append(out)

    per_fold_csv = xd / "phase1_cv_temporal_metrics_per_fold.csv"
    comp_csv = xd / "phase1_cv_temporal_metrics_model_comparison.csv"

    if rows:
        fields = sorted(set().union(*[r.keys() for r in rows]))
        with open(per_fold_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    if model_rows:
        model_rows = sorted(
            model_rows,
            key=lambda r: -1 if r.get("frame_ap_mean") is None else float(r["frame_ap_mean"]),
            reverse=True,
        )
        fields = sorted(set().union(*[r.keys() for r in model_rows]))
        with open(comp_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(model_rows)

    print("saved:", per_fold_csv)
    print("saved:", comp_csv)

    print("\n===== RANK BY FRAME AP =====")
    for r in model_rows:
        print(r["model"], "AP=", r.get("frame_ap_mean"), "AUC=", r.get("frame_auc_mean"))


if __name__ == "__main__":
    main()
