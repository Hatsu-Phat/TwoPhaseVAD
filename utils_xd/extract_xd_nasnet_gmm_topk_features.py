#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from torchvision import transforms


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


def sample_motion_scores(video_path: Path, sample_stride: int = 4):

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = 1

    idxs = list(range(0, total, sample_stride))
    motion = []

    prev_gray = None
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()

        if not ok:
            motion.append(0.0)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 90))

        if prev_gray is None:
            score = 0.0
        else:
            diff = cv2.absdiff(gray, prev_gray)
            score = float(diff.mean())

        prev_gray = gray
        motion.append(score)

    cap.release()

    return np.array(idxs, dtype=np.int64), np.array(motion, dtype=np.float32), total


def select_topk_motion(video_path: Path, top_k: int):
    idxs, motion, total = sample_motion_scores(video_path)

    if len(idxs) == 0:
        selected = np.zeros(top_k, dtype=np.int64)
        return selected, total

    if len(idxs) < top_k:
        pad = np.full(top_k - len(idxs), idxs[-1], dtype=np.int64)
        selected = np.concatenate([idxs, pad])
        return np.sort(selected), total

    order = np.argsort(motion)[::-1]
    selected = idxs[order[:top_k]]
    selected = np.sort(selected)

    return selected.astype(np.int64), total


def read_frames(video_path: Path, indices, image_size: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    frames = []
    last = None

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()

        if not ok:
            if last is None:
                frame = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            else:
                frame = last.copy()
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last = frame.copy()

        img = Image.fromarray(frame)
        frames.append(tfm(img))

    cap.release()
    return torch.stack(frames, dim=0)


def make_segment_bounds(selected_indices, total_frames):
    bounds = []
    selected = np.asarray(selected_indices, dtype=np.int64)

    for idx in selected:
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


def safe_class_dir(name: str):
    return name.replace("/", "_").replace(" ", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size-frames", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-float16", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    manifest_dir = Path(args.manifest_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    rows = []
    for split in ["train", "test"]:
        rows.extend(read_manifest(manifest_dir / f"{split}.jsonl"))

    if args.limit > 0:
        rows = rows[:args.limit]

    print("total rows:", len(rows))
    print("device:", device)

    model = NASNetMobileFeatureExtractor().to(device).eval()

    failures = []

    for r in tqdm(rows, desc="Extract XD NASNet-GMMTopK"):
        video_path = Path(r["video_path"])
        split = r["split"]
        class_name = r["class_name"]
        rel_path = r["rel_path"]

        out_path = out_root / split / safe_class_dir(class_name) / (video_path.stem + ".pt")

        if args.skip_existing and out_path.exists() and out_path.stat().st_size > 0:
            continue

        try:
            fps, total_frames_meta = get_video_meta(video_path)
            selected, total_frames = select_topk_motion(video_path, top_k=args.top_k)

            if total_frames_meta > 0:
                total_frames = total_frames_meta

            frame_tensor = read_frames(video_path, selected, args.image_size)

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
            gt_segment_labels = make_gt_selected_labels(selected, annotation_ranges)

            payload = {
                "features": features,
                "selected_indices": torch.tensor(selected, dtype=torch.long),
                "segment_bounds": make_segment_bounds(selected, total_frames),
                "gt_segment_labels": gt_segment_labels,
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
            }

            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, out_path)

        except Exception as e:
            failures.append({
                "video_path": str(video_path),
                "rel_path": rel_path,
                "error": repr(e),
            })
            print("[FAILED]", rel_path, repr(e))

    fail_path = out_root / "failures.jsonl"
    with open(fail_path, "w", encoding="utf-8") as f:
        for item in failures:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("DONE")
    print("failures:", len(failures))
    print("failure log:", fail_path)


if __name__ == "__main__":
    main()
