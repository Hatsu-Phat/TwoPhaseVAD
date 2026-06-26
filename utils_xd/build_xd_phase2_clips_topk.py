#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

ANOMALY_CLASSES = [
    "Fighting",
    "Shooting",
    "Riot",
    "Abuse",
    "Car accident",
    "Explosion",
]

CLASS_MAP = {c: i for i, c in enumerate(ANOMALY_CLASSES)}


def safe_class_dir(name: str):
    return name.replace("/", "_").replace(" ", "_")


def select_topk_centers(selected_indices, segment_scores, k, min_center_gap):
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    segment_scores = np.asarray(segment_scores, dtype=np.float64)

    order = np.argsort(segment_scores)[::-1]
    chosen = []

    for idx in order:
        center = int(selected_indices[idx])
        score = float(segment_scores[idx])

        if all(abs(center - c[0]) >= min_center_gap for c in chosen):
            chosen.append((center, score, int(idx)))

        if len(chosen) >= k:
            break

    return chosen


def resolve_video_path(x, xd_root=None):
    p = x.get("video_path")
    if p and Path(p).exists():
        return Path(p)

    feature_file = x.get("feature_file")
    if feature_file and Path(feature_file).exists():
        try:
            fx = torch.load(feature_file, map_location="cpu")
            p = fx.get("video_path")
            if p and Path(p).exists():
                return Path(p)

            rel = fx.get("rel_path")
            if xd_root is not None and rel:
                cand = Path(xd_root) / rel
                if cand.exists():
                    return cand
        except Exception:
            pass

    rel = x.get("rel_path")
    if xd_root is not None and rel:
        cand = Path(xd_root) / rel
        if cand.exists():
            return cand

        cand = Path(xd_root) / "videos" / Path(rel).name
        if cand.exists():
            return cand

    if xd_root is not None and rel:
        name = Path(rel).name
        found = list(Path(xd_root).rglob(name))
        if found:
            return found[0]

    raise RuntimeError(
        f"Cannot resolve video path. rel_path={x.get('rel_path')} "
        f"feature_file={x.get('feature_file')} video_path={x.get('video_path')}"
    )


def cut_clip(video_path: Path, out_path: Path, start_frame: int, end_frame: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, "cannot_open"

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 24.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

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
    ap.add_argument("--scores-root", required=True)
    ap.add_argument("--xd-root", default=None)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--k-per-video", type=int, default=8)
    ap.add_argument("--clip-len", type=int, default=64)
    ap.add_argument("--min-center-gap", type=int, default=24)
    ap.add_argument("--max-per-class", type=int, default=400)
    args = ap.parse_args()

    scores_root = Path(args.scores_root)
    out_root = Path(args.out_root)
    xd_root = Path(args.xd_root or os.environ.get("XD", "/home/grouphahieu/imagenet/XD_Violence"))
    out_root.mkdir(parents=True, exist_ok=True)

    with open(out_root / "class_map.json", "w", encoding="utf-8") as f:
        json.dump(CLASS_MAP, f, indent=2, ensure_ascii=False)

    candidates = []
    failures = []

    for p in tqdm(sorted(scores_root.rglob("*.pt")), desc="Collect XD phase2 candidates"):
        x = torch.load(p, map_location="cpu")
        video_label = int(x.get("video_label", 0))
        class_name = str(x.get("class_name", "Normal"))

        if video_label == 0 or class_name == "Normal":
            continue

        if class_name not in CLASS_MAP:
            failures.append({"score_file": str(p), "error": f"unknown_class {class_name}"})
            continue

        selected = x["selected_indices"].cpu().numpy()
        scores = x["segment_scores"].float().cpu().numpy()
        total_frames = int(x.get("total_frames", 0))

        centers = select_topk_centers(selected, scores, args.k_per_video, args.min_center_gap)

        for rank, (center, score, token_idx) in enumerate(centers):
            start = max(0, center - args.clip_len // 2)
            end = start + args.clip_len

            if total_frames > 0:
                end = min(total_frames, end)
                start = max(0, end - args.clip_len)

            candidates.append({
                "score_file": str(p),
                "source_video": str(resolve_video_path(x, xd_root=xd_root)),
                "rel_path": str(x.get("rel_path", "")),
                "class_name": class_name,
                "class_id": CLASS_MAP[class_name],
                "start_frame": int(start),
                "end_frame": int(end),
                "peak_frame": int(center),
                "peak_score": float(score),
                "local_rank": int(rank),
            })

    if args.max_per_class > 0:
        capped = []
        for cls in ANOMALY_CLASSES:
            items = [c for c in candidates if c["class_name"] == cls]
            items = sorted(items, key=lambda r: r["peak_score"], reverse=True)
            capped.extend(items[: args.max_per_class])
        candidates = capped

    manifest = []

    for c in tqdm(candidates, desc="Cut XD phase2 clips"):
        class_dir = safe_class_dir(c["class_name"])
        stem = Path(c["rel_path"]).stem or Path(c["source_video"]).stem

        out_path = out_root / "clips" / class_dir / f"{stem}_rank{c['local_rank']:02d}_f{c['start_frame']:06d}_{c['end_frame']:06d}.mp4"

        ok, info = cut_clip(
            Path(c["source_video"]),
            out_path,
            c["start_frame"],
            c["end_frame"],
        )

        if not ok:
            failures.append({"source_video": c["source_video"], "error": info})
            continue

        r = dict(c)
        r["clip_path"] = str(out_path)
        r["split"] = "train"
        r["reason"] = "topk_per_video"
        manifest.append(r)

    with open(out_root / "manifest_phase2.jsonl", "w", encoding="utf-8") as f:
        for r in manifest:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(out_root / "failures.jsonl", "w", encoding="utf-8") as f:
        for r in failures:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("clips:", len(manifest))
    print("failures:", len(failures))
    print("manifest:", out_root / "manifest_phase2.jsonl")


if __name__ == "__main__":
    main()
