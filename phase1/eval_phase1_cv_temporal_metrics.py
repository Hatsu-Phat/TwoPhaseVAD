#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)


def parse_annotations(path: Path):
    ann = {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        for raw in f:
            p = raw.strip().split()
            if len(p) < 6:
                continue
            try:
                nums = [int(x) for x in p[2:6]]
            except Exception:
                continue

            ranges = []
            for a, b in [(nums[0], nums[1]), (nums[2], nums[3])]:
                if a > 0 and b > 0:
                    ranges.append((min(a, b), max(a, b)))

            ann[p[0]] = {"event": p[1], "ranges": ranges}
    return ann


def make_gt(total_frames: int, ranges):
    y = np.zeros(total_frames, dtype=np.uint8)
    for a, b in ranges:
        s = max(0, a - 1)
        e = min(total_frames, b)
        if e > s:
            y[s:e] = 1
    return y


def unique_idx_score(indices, scores):
    indices = np.asarray(indices, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(indices)
    indices, scores = indices[order], scores[order]

    ui, us = [], []
    for idx in np.unique(indices):
        us.append(scores[indices == idx].max())
        ui.append(idx)
    return np.array(ui, dtype=np.int64), np.array(us, dtype=np.float64)


def frame_scores_nearest(total_frames: int, indices, scores):
    indices, scores = unique_idx_score(indices, scores)

    if total_frames <= 0:
        total_frames = int(indices.max() + 1) if len(indices) else 1

    if len(indices) == 0:
        return np.zeros(total_frames, dtype=np.float64)

    if len(indices) == 1:
        return np.ones(total_frames, dtype=np.float64) * float(scores[0])

    mids = [(indices[i] + indices[i + 1]) // 2 + 1 for i in range(len(indices) - 1)]
    bounds = [0] + mids + [total_frames]

    fs = np.zeros(total_frames, dtype=np.float64)
    for i, sc in enumerate(scores):
        s = max(0, min(total_frames, int(bounds[i])))
        e = max(0, min(total_frames, int(bounds[i + 1])))
        if e > s:
            fs[s:e] = float(sc)
    return fs


def intervals_from_mask(mask):
    out = []
    n = len(mask)
    i = 0
    while i < n:
        while i < n and not mask[i]:
            i += 1
        if i >= n:
            break
        s = i
        while i < n and mask[i]:
            i += 1
        e = i
        out.append((s, e))
    return out


def merge_intervals(intervals, gap):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s - merged[-1][1] <= gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [tuple(x) for x in merged]


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


def eval_one(scores_root: Path, annotation, threshold_mode: str, manual_threshold: float | None, fps_default: float, merge_gap: int):
    files = sorted((scores_root / "test").rglob("*.pt"))

    all_y = []
    all_s = []
    video_y = []
    video_s = []
    normal_scores = []

    per_video = []

    for p in files:
        x = torch.load(p, map_location="cpu")
        rel_path = str(x.get("rel_path", ""))
        name = Path(rel_path).name
        cls = str(x.get("class_name", ""))
        video_label = int(x.get("video_label", 0))
        total_frames = int(x.get("total_frames", 0))
        fps = float(x.get("fps", fps_default) or fps_default)

        ranges = annotation.get(name, {"ranges": []}).get("ranges", [])
        if video_label == 0 or "Normal" in cls:
            ranges = []

        idx = x["selected_indices"].cpu().numpy()
        ss = x["segment_scores"].float().cpu().numpy()
        fs = frame_scores_nearest(total_frames, idx, ss)
        y = make_gt(len(fs), ranges)

        all_y.extend(y.tolist())
        all_s.extend(fs.tolist())

        video_y.append(video_label)
        video_s.append(float(x.get("video_score", np.max(ss))))

        if video_label == 0 or "Normal" in cls:
            normal_scores.extend(fs.tolist())

        per_video.append({
            "name": name,
            "rel_path": rel_path,
            "video_label": video_label,
            "fps": fps,
            "frame_scores": fs,
            "gt": y,
            "ranges": ranges,
        })

    all_y = np.array(all_y, dtype=np.uint8)
    all_s = np.array(all_s, dtype=np.float64)

    if threshold_mode == "manual":
        if manual_threshold is None:
            raise ValueError("--manual-threshold required")
        threshold = float(manual_threshold)
    elif threshold_mode == "p99_normal":
        threshold = float(np.percentile(np.array(normal_scores), 99))
    elif threshold_mode == "p95_normal":
        threshold = float(np.percentile(np.array(normal_scores), 95))
    elif threshold_mode == "max_normal":
        threshold = float(np.max(np.array(normal_scores)))
    else:
        raise ValueError(threshold_mode)

    y_pred = (all_s >= threshold).astype(np.uint8)

    cm = confusion_matrix(all_y, y_pred, labels=[0, 1])
    acc = accuracy_score(all_y, y_pred)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        all_y, y_pred, labels=[0, 1], average="macro", zero_division=0
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
        pred_mask = fs >= threshold
        pred_intervals = merge_intervals(intervals_from_mask(pred_mask), gap=merge_gap)

        gt_intervals = []
        for a, b in v["ranges"]:
            gt_intervals.append((max(0, a - 1), min(len(fs), b)))

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
                delay = max(0, first[0] - gi[0]) / fps
                delays.append(delay)

    normal_hours = normal_frames / fps_default / 3600.0 if normal_frames > 0 else 0.0

    summary = {
        "num_videos": len(files),
        "num_frames": int(len(all_y)),
        "num_positive_frames": int(all_y.sum()),
        "threshold_mode": threshold_mode,
        "threshold": threshold,

        "frame_auc": safe_auc(all_y, all_s),
        "frame_ap": safe_ap(all_y, all_s),
        "video_auc": safe_auc(video_y, video_s),

        "binary_accuracy": float(acc),
        "binary_precision_macro": float(p_macro),
        "binary_recall_macro": float(r_macro),
        "binary_f1_macro": float(f_macro),

        "false_alarms": int(false_alarms),
        "normal_hours": float(normal_hours),
        "false_alarms_per_hour": float(false_alarms / normal_hours) if normal_hours > 0 else None,

        "gt_events": int(gt_events),
        "mean_best_temporal_iou": float(np.mean(best_ious)) if best_ious else None,
        "event_recall_tiou_0.1": float(detected[0.1] / gt_events) if gt_events > 0 else None,
        "event_recall_tiou_0.3": float(detected[0.3] / gt_events) if gt_events > 0 else None,
        "event_recall_tiou_0.5": float(detected[0.5] / gt_events) if gt_events > 0 else None,
        "detection_delay_sec_mean": float(np.mean(delays)) if delays else None,
        "detection_delay_sec_median": float(np.median(delays)) if delays else None,
    }

    return summary, cm


def save_cm_csv(path: Path, cm):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred", "Normal", "Anomaly"])
        w.writerow(["Normal", int(cm[0, 0]), int(cm[0, 1])])
        w.writerow(["Anomaly", int(cm[1, 0]), int(cm[1, 1])])


def mean_std(vals):
    vals = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals, ddof=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ucf-root", default="/home/grouphahieu/imagenet/UCF-Crime")
    ap.add_argument("--models", nargs="+", default=["evit", "lstm", "tcn", "transformer", "stgnn"])
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--threshold-mode", default="p99_normal", choices=["p99_normal", "p95_normal", "max_normal", "manual"])
    ap.add_argument("--manual-threshold", type=float, default=None)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--merge-gap", type=int, default=16)
    args = ap.parse_args()

    ucf = Path(args.ucf_root)
    ann = parse_annotations(ucf / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt")

    rows = []
    model_summary = []

    for model in args.models:
        cms = []
        model_rows = []

        for fold in args.folds:
            score_root = ucf / "phase1_scores_cv" / f"top30_{model}_fold{fold}"
            out_dir = ucf / "outputs_phase1_cv" / f"top30_{model}_fold{fold}" / "eval_temporal_metrics"
            out_dir.mkdir(parents=True, exist_ok=True)

            if not score_root.exists():
                print("[SKIP] missing:", score_root)
                continue

            summary, cm = eval_one(
                score_root,
                ann,
                args.threshold_mode,
                args.manual_threshold,
                args.fps,
                args.merge_gap,
            )

            summary["model"] = model
            summary["fold"] = fold

            with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            save_cm_csv(out_dir / "binary_confusion_matrix.csv", cm)

            rows.append(summary)
            model_rows.append(summary)
            cms.append(cm)

            print(json.dumps(summary, indent=2, ensure_ascii=False))

        metrics = [
            "frame_auc",
            "frame_ap",
            "video_auc",
            "binary_accuracy",
            "binary_precision_macro",
            "binary_recall_macro",
            "binary_f1_macro",
            "false_alarms_per_hour",
            "mean_best_temporal_iou",
            "event_recall_tiou_0.1",
            "event_recall_tiou_0.3",
            "event_recall_tiou_0.5",
            "detection_delay_sec_mean",
            "detection_delay_sec_median",
        ]

        out = {"model": model, "num_folds": len(model_rows)}
        for m in metrics:
            avg, std = mean_std([r.get(m) for r in model_rows])
            out[m + "_mean"] = avg
            out[m + "_std"] = std

        if cms:
            cm_sum = np.sum(np.stack(cms), axis=0)
            out["cv_cm_tn"] = int(cm_sum[0, 0])
            out["cv_cm_fp"] = int(cm_sum[0, 1])
            out["cv_cm_fn"] = int(cm_sum[1, 0])
            out["cv_cm_tp"] = int(cm_sum[1, 1])

            agg_dir = ucf / "outputs_phase1_cv" / f"top30_{model}_cv_aggregate"
            agg_dir.mkdir(parents=True, exist_ok=True)
            save_cm_csv(agg_dir / "binary_confusion_matrix_cv_sum.csv", cm_sum)

        model_summary.append(out)

    per_fold_csv = ucf / "phase1_cv_temporal_metrics_per_fold.csv"
    if rows:
        fields = sorted(set().union(*[r.keys() for r in rows]))
        with open(per_fold_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    comp_csv = ucf / "phase1_cv_temporal_metrics_model_comparison.csv"
    if model_summary:
        model_summary = sorted(
            model_summary,
            key=lambda r: -1 if r.get("frame_auc_mean") is None else float(r["frame_auc_mean"]),
            reverse=True,
        )
        fields = sorted(set().union(*[r.keys() for r in model_summary]))
        with open(comp_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(model_summary)

    print("saved:", per_fold_csv)
    print("saved:", comp_csv)

    print("\n===== MODEL RANK BY FRAME AUC =====")
    for r in model_summary:
        print(
            f"{r['model']:12s} "
            f"frame_auc={r.get('frame_auc_mean')} ± {r.get('frame_auc_std')} "
            f"FA/h={r.get('false_alarms_per_hour_mean')} "
            f"delay={r.get('detection_delay_sec_mean_mean')} "
            f"mTIoU={r.get('mean_best_temporal_iou_mean')}"
        )


if __name__ == "__main__":
    main()
