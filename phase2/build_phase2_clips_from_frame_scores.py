#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


ANOMALY_CLASSES = [
    "Abuse",
    "Arrest",
    "Arson",
    "Assault",
    "Burglary",
    "Explosion",
    "Fighting",
    "RoadAccidents",
    "Robbery",
    "Shooting",
    "Shoplifting",
    "Stealing",
    "Vandalism",
]


def stable_split(key: str, val_ratio: float) -> str:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    v = int(h[:8], 16) / 0xFFFFFFFF
    return "val" if v < val_ratio else "train"


def resolve_video_path(video_root: Path, rel_path: str, class_name: str) -> Path | None:
    roots = []
    if (video_root / "videos").exists():
        roots.append(video_root / "videos")
    roots.append(video_root)

    rel = Path(rel_path)
    name = rel.name

    candidates = []
    for root in roots:
        candidates.append(root / rel)
        candidates.append(root / class_name / name)

    for c in candidates:
        if c.exists():
            return c

    for root in roots:
        found = list(root.rglob(name))
        if found:
            return found[0]

    return None


def unique_idx_score(indices, scores):
    indices = np.asarray(indices, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    order = np.argsort(indices)
    indices = indices[order]
    scores = scores[order]

    uniq_idx, uniq_score = [], []
    for idx in np.unique(indices):
        mask = indices == idx
        uniq_idx.append(int(idx))
        uniq_score.append(float(scores[mask].max()))

    return np.array(uniq_idx, dtype=np.int64), np.array(uniq_score, dtype=np.float64)


def frame_scores_nearest(total_frames: int, indices, scores):
    indices, scores = unique_idx_score(indices, scores)

    total_frames = int(total_frames)
    if total_frames <= 0:
        total_frames = int(indices.max() + 1) if len(indices) else 1

    if len(indices) == 0:
        return np.zeros(total_frames, dtype=np.float64)

    if len(indices) == 1:
        return np.ones(total_frames, dtype=np.float64) * float(scores[0])

    mids = []
    for i in range(len(indices) - 1):
        mids.append((int(indices[i]) + int(indices[i + 1])) // 2 + 1)

    bounds = [0] + mids + [total_frames]
    frame_scores = np.zeros(total_frames, dtype=np.float64)

    for i, score in enumerate(scores):
        s = max(0, min(total_frames, int(bounds[i])))
        e = max(0, min(total_frames, int(bounds[i + 1])))
        if e > s:
            frame_scores[s:e] = float(score)

    return frame_scores


def find_intervals(mask, merge_gap: int, min_len: int):
    intervals = []
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
        intervals.append([s, e])

    if not intervals:
        return []

    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s - merged[-1][1] <= merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    merged = [(s, e) for s, e in merged if e - s >= min_len]
    return merged


def cut_clip(video_path: Path, out_path: Path, start_frame: int, end_frame: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, "cannot_open"

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))

    written = 0
    cur = int(start_frame)
    while cur < int(end_frame):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
        cur += 1

    writer.release()
    cap.release()

    if written <= 0:
        try:
            out_path.unlink(missing_ok=True)
        except Exception:
            pass
        return False, "zero_frames"

    return True, written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--scores-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--threshold-mode", choices=["max_normal", "p99_normal", "p95_normal", "manual"], default="max_normal")
    ap.add_argument("--manual-threshold", type=float, default=None)
    ap.add_argument("--clip-len", type=int, default=64)
    ap.add_argument("--merge-gap", type=int, default=16)
    ap.add_argument("--min-interval-len", type=int, default=8)
    ap.add_argument("--max-clips-per-video", type=int, default=5)
    ap.add_argument("--fallback-topk", type=int, default=1)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    args = ap.parse_args()

    video_root = Path(args.video_root)
    scores_root = Path(args.scores_root)
    out_root = Path(args.out_root)

    class_map = {c: i for i, c in enumerate(ANOMALY_CLASSES)}
    out_root.mkdir(parents=True, exist_ok=True)

    with open(out_root / "class_map.json", "w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=2, ensure_ascii=False)

    score_files = sorted(scores_root.rglob("*.pt"))

    normal_scores = []
    for p in score_files:
        x = torch.load(p, map_location="cpu")
        class_name = str(x.get("class_name", ""))
        video_label = int(x.get("video_label", 1))

        if video_label == 0 or "Normal" in class_name:
            ss = x["segment_scores"].float().cpu().numpy()
            normal_scores.extend(ss.tolist())

    if args.threshold_mode == "manual":
        if args.manual_threshold is None:
            raise ValueError("--manual-threshold is required when threshold-mode=manual")
        threshold = float(args.manual_threshold)
    else:
        if not normal_scores:
            raise RuntimeError("No normal scores found for threshold computation.")
        normal_scores = np.array(normal_scores, dtype=np.float64)

        if args.threshold_mode == "max_normal":
            threshold = float(normal_scores.max())
        elif args.threshold_mode == "p99_normal":
            threshold = float(np.percentile(normal_scores, 99))
        elif args.threshold_mode == "p95_normal":
            threshold = float(np.percentile(normal_scores, 95))
        else:
            raise ValueError(args.threshold_mode)

    print("scores_root:", scores_root)
    print("out_root:", out_root)
    print("threshold_mode:", args.threshold_mode)
    print("threshold:", threshold)
    print("normal_score_count:", len(normal_scores))

    manifest = []
    failures = []

    for p in tqdm(score_files, desc="Build phase2 clips"):
        x = torch.load(p, map_location="cpu")

        class_name = str(x.get("class_name", ""))
        video_label = int(x.get("video_label", 1))

        if video_label == 0 or "Normal" in class_name:
            continue

        if class_name not in class_map:
            failures.append({"score_file": str(p), "error": f"unknown_class {class_name}"})
            continue

        rel_path = str(x.get("rel_path", ""))
        video_path = resolve_video_path(video_root, rel_path, class_name)

        if video_path is None:
            failures.append({"score_file": str(p), "rel_path": rel_path, "error": "video_not_found"})
            continue

        selected_indices = x["selected_indices"].cpu().numpy()
        segment_scores = x["segment_scores"].float().cpu().numpy()
        total_frames = int(x.get("total_frames", 0))

        frame_scores = frame_scores_nearest(total_frames, selected_indices, segment_scores)
        mask = frame_scores > threshold

        intervals = find_intervals(
            mask,
            merge_gap=int(args.merge_gap),
            min_len=int(args.min_interval_len),
        )

        candidates = []

        for s, e in intervals:
            local = frame_scores[s:e]
            peak = s + int(np.argmax(local))
            peak_score = float(frame_scores[peak])
            candidates.append((peak_score, s, e, peak, "threshold_interval"))

        if not candidates and args.fallback_topk > 0:
            idx_sorted = np.argsort(segment_scores)[::-1][: int(args.fallback_topk)]
            for idx in idx_sorted:
                peak = int(selected_indices[idx])
                score = float(segment_scores[idx])
                s = max(0, peak - args.clip_len // 2)
                e = min(len(frame_scores), s + args.clip_len)
                s = max(0, e - args.clip_len)
                candidates.append((score, s, e, peak, "fallback_topk"))

        candidates = sorted(candidates, key=lambda z: z[0], reverse=True)
        candidates = candidates[: int(args.max_clips_per_video)]

        video_key = rel_path or str(video_path.name)
        split = stable_split(video_key, args.val_ratio)

        for clip_idx, (peak_score, s, e, peak, reason) in enumerate(candidates):
            center = int(peak)
            start = max(0, center - args.clip_len // 2)
            end = min(max(total_frames, len(frame_scores)), start + args.clip_len)
            start = max(0, end - args.clip_len)

            stem = Path(rel_path).stem if rel_path else video_path.stem
            out_path = (
                out_root
                / split
                / class_name
                / f"{stem}_clip{clip_idx:03d}_f{start:06d}_{end:06d}.mp4"
            )

            ok, info = cut_clip(video_path, out_path, start, end)
            if not ok:
                failures.append({
                    "score_file": str(p),
                    "video_path": str(video_path),
                    "error": info,
                })
                continue

            manifest.append({
                "clip_path": str(out_path),
                "split": split,
                "class_name": class_name,
                "class_id": class_map[class_name],
                "source_video": str(video_path),
                "rel_path": rel_path,
                "start_frame": int(start),
                "end_frame": int(end),
                "peak_frame": int(peak),
                "peak_score": float(peak_score),
                "threshold": float(threshold),
                "reason": reason,
            })

    with open(out_root / "manifest_phase2.jsonl", "w", encoding="utf-8") as f:
        for r in manifest:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(out_root / "failures.jsonl", "w", encoding="utf-8") as f:
        for r in failures:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("DONE")
    print("clips:", len(manifest))
    print("failures:", len(failures))
    print("manifest:", out_root / "manifest_phase2.jsonl")
    print("class_map:", out_root / "class_map.json")


if __name__ == "__main__":
    main()
