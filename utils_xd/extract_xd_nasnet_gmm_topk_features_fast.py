#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


class NASNetMobileFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        import pretrainedmodels

        self.model = pretrainedmodels.__dict__["nasnetamobile"](
            num_classes=1000,
            pretrained="imagenet",
        )
        self.model.eval()

    def forward(self, x):
        feat = self.model.features(x)
        feat = torch.nn.functional.adaptive_avg_pool2d(feat, 1)
        feat = torch.flatten(feat, 1)
        return feat


def read_manifest(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def safe_class_dir(name: str):
    return name.replace("/", "_").replace(" ", "_")


def get_video_meta(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 24.0

    cap.release()
    return fps, total


def select_topk_frames_onepass(video_path: Path, top_k: int, motion_stride: int):
    """
    One-pass motion-guided top-k frame selection.

    Instead of random seeking many times, this scans the video sequentially.
    It computes motion every motion_stride frames and keeps only top-k RGB frames.
    """

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 1

    heap = []
    prev_gray = None
    frame_idx = 0
    sampled_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % motion_stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray, (160, 90))

            if prev_gray is None:
                score = 0.0
            else:
                diff = cv2.absdiff(gray_small, prev_gray)
                score = float(diff.mean())

            prev_gray = gray_small
            sampled_count += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            item = (score, frame_idx, rgb)

            if len(heap) < top_k:
                heapq.heappush(heap, item)
            else:
                if score > heap[0][0]:
                    heapq.heapreplace(heap, item)

        frame_idx += 1

    cap.release()

    if not heap:
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        selected = [(0.0, 0, dummy) for _ in range(top_k)]
    else:
        selected = sorted(heap, key=lambda x: x[1])

        while len(selected) < top_k:
            selected.append(selected[-1])

    selected = selected[:top_k]

    indices = np.array([x[1] for x in selected], dtype=np.int64)
    frames = [x[2] for x in selected]

    return indices, frames, total_frames


def frames_to_tensor(frames, image_size: int):
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    out = []
    for frame in frames:
        img = Image.fromarray(frame)
        out.append(tfm(img))

    return torch.stack(out, dim=0)


def make_segment_bounds(selected_indices, total_frames):
    bounds = []
    for idx in selected_indices:
        s = max(0, int(idx))
        e = min(int(total_frames), int(idx) + 1)
        bounds.append([s, e])

    return torch.tensor(bounds, dtype=torch.long)


def make_gt_selected_labels(selected_indices, annotation_ranges):
    labels = []

    for idx in selected_indices:
        y = 0
        for a, b in annotation_ranges:
            if int(a) <= int(idx) <= int(b):
                y = 1
                break
        labels.append(y)

    return torch.tensor(labels, dtype=torch.long)


def atomic_torch_save(payload, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    if tmp_path.exists():
        tmp_path.unlink()

    torch.save(payload, tmp_path)
    tmp_path.replace(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--motion-stride", type=int, default=4)
    ap.add_argument("--batch-size-frames", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-float16", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cv2.setNumThreads(2)

    manifest_dir = Path(args.manifest_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for split in ["train", "test"]:
        rows.extend(read_manifest(manifest_dir / f"{split}.jsonl"))

    if args.limit > 0:
        rows = rows[:args.limit]

    device = torch.device(
        args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )

    print("total rows:", len(rows))
    print("device:", device)
    print("motion_stride:", args.motion_stride)

    model = NASNetMobileFeatureExtractor().to(device).eval()

    failures = []
    saved = 0
    skipped = 0

    for r in tqdm(rows, desc="Extract XD NASNet-GMMTopK FAST"):
        video_path = Path(r["video_path"])
        split = r["split"]
        class_name = r["class_name"]
        rel_path = r["rel_path"]

        out_path = out_root / split / safe_class_dir(class_name) / (video_path.stem + ".pt")

        if args.skip_existing and out_path.exists() and out_path.stat().st_size > 0:
            skipped += 1
            continue

        try:
            fps, total_meta = get_video_meta(video_path)

            selected_indices, frames, total_frames = select_topk_frames_onepass(
                video_path=video_path,
                top_k=args.top_k,
                motion_stride=args.motion_stride,
            )

            if total_meta > 0:
                total_frames = total_meta

            frame_tensor = frames_to_tensor(frames, args.image_size)

            feats = []
            with torch.no_grad():
                for i in range(0, len(frame_tensor), args.batch_size_frames):
                    batch = frame_tensor[i:i + args.batch_size_frames].to(device)
                    out = model(batch).cpu()
                    feats.append(out)

            features = torch.cat(feats, dim=0)

            if args.save_float16:
                features = features.half()

            annotation_ranges = r.get("annotation_ranges", [])

            payload = {
                "features": features,
                "selected_indices": torch.tensor(selected_indices, dtype=torch.long),
                "segment_bounds": make_segment_bounds(selected_indices, total_frames),
                "gt_segment_labels": make_gt_selected_labels(selected_indices, annotation_ranges),
                "video_label": int(r["video_label"]),
                "class_name": class_name,
                "phase2_class_id": int(r.get("phase2_class_id", -1)),
                "multi_classes": r.get("multi_classes", []),
                "codes": r.get("codes", []),
                "rel_path": rel_path,
                "video_path": str(video_path),
                "fps": float(fps),
                "total_frames": int(total_frames),
                "annotation_ranges": annotation_ranges,
                "extractor": "fast_onepass_motion_topk",
                "motion_stride": int(args.motion_stride),
            }

            atomic_torch_save(payload, out_path)
            saved += 1

        except Exception as e:
            failures.append({
                "video_path": str(video_path),
                "rel_path": rel_path,
                "error": repr(e),
            })
            print("[FAILED]", rel_path, repr(e))

    fail_path = out_root / "failures_fast.jsonl"
    with open(fail_path, "w", encoding="utf-8") as f:
        for item in failures:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("DONE")
    print("saved:", saved)
    print("skipped:", skipped)
    print("failures:", len(failures))
    print("failure log:", fail_path)


if __name__ == "__main__":
    main()
