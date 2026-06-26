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


def stable_hash_float(s: str) -> float:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def resolve_video_path(video_root: Path, rel_path: str, class_name: str) -> Path | None:
    roots = []
    if (video_root / "videos").exists():
        roots.append(video_root / "videos")
    roots.append(video_root)

    name = Path(rel_path).name
    rel = Path(rel_path)

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


def select_topk_centers(selected_indices, segment_scores, k: int, min_center_gap: int):
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

    ap.add_argument("--k-per-video", type=int, default=5)
    ap.add_argument("--clip-len", type=int, default=64)
    ap.add_argument("--min-center-gap", type=int, default=32)

    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--max-per-class", type=int, default=0)
    ap.add_argument("--seed-key", default="phase2_topk_per_video_v1")

    args = ap.parse_args()

    video_root = Path(args.video_root)
    scores_root = Path(args.scores_root)
    out_root = Path(args.out_root)

    class_map = {c: i for i, c in enumerate(ANOMALY_CLASSES)}
    out_root.mkdir(parents=True, exist_ok=True)

    with open(out_root / "class_map.json", "w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=2, ensure_ascii=False)

    score_files = sorted(scores_root.rglob("*.pt"))

    candidates = []
    failures = []

    for p in tqdm(score_files, desc="Collect candidates"):
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

        centers = select_topk_centers(
            selected_indices=selected_indices,
            segment_scores=segment_scores,
            k=int(args.k_per_video),
            min_center_gap=int(args.min_center_gap),
        )

        for local_rank, (center, score, token_idx) in enumerate(centers):
            start = max(0, int(center) - args.clip_len // 2)
            end = start + int(args.clip_len)

            if total_frames > 0:
                end = min(total_frames, end)
                start = max(0, end - int(args.clip_len))

            stem = Path(rel_path).stem if rel_path else video_path.stem

            candidates.append({
                "class_name": class_name,
                "class_id": class_map[class_name],
                "source_video": str(video_path),
                "rel_path": rel_path,
                "stem": stem,
                "start_frame": int(start),
                "end_frame": int(end),
                "peak_frame": int(center),
                "peak_score": float(score),
                "token_idx": int(token_idx),
                "local_rank": int(local_rank),
            })

    if args.max_per_class and args.max_per_class > 0:
        capped = []
        for cls in ANOMALY_CLASSES:
            cls_items = [c for c in candidates if c["class_name"] == cls]
            cls_items = sorted(cls_items, key=lambda r: r["peak_score"], reverse=True)
            capped.extend(cls_items[: int(args.max_per_class)])
        candidates = capped

    for c in candidates:
        key = f'{args.seed_key}|{c["class_name"]}|{c["rel_path"]}|{c["peak_frame"]}|{c["local_rank"]}'
        c["split"] = "val" if stable_hash_float(key) < float(args.val_ratio) else "train"

    manifest = []

    for c in tqdm(candidates, desc="Cut clips"):
        class_name = c["class_name"]
        split = c["split"]
        video_path = Path(c["source_video"])

        out_path = (
            out_root
            / split
            / class_name
            / f'{c["stem"]}_rank{c["local_rank"]:02d}_f{c["start_frame"]:06d}_{c["end_frame"]:06d}.mp4'
        )

        ok, info = cut_clip(
            video_path=video_path,
            out_path=out_path,
            start_frame=c["start_frame"],
            end_frame=c["end_frame"],
        )

        if not ok:
            failures.append({
                "source_video": str(video_path),
                "rel_path": c["rel_path"],
                "error": info,
            })
            continue

        row = dict(c)
        row["clip_path"] = str(out_path)
        row["reason"] = "topk_per_video"
        manifest.append(row)

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
