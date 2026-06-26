#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torchvision import transforms
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from phase2.models import build_phase2_model


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


def parse_annotations(path: Path):
    ann = {}
    if not path.exists():
        return ann

    with open(path, encoding="utf-8", errors="ignore") as f:
        for raw in f:
            parts = raw.strip().split()
            if len(parts) < 6:
                continue

            try:
                nums = [int(x) for x in parts[2:6]]
            except Exception:
                continue

            ranges = []
            for a, b in [(nums[0], nums[1]), (nums[2], nums[3])]:
                if a > 0 and b > 0:
                    ranges.append((min(a, b), max(a, b)))

            ann[parts[0]] = {
                "event": parts[1],
                "ranges": ranges,
            }

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
    indices = indices[order]
    scores = scores[order]

    ui, us = [], []
    for idx in np.unique(indices):
        mask = indices == idx
        ui.append(int(idx))
        us.append(float(scores[mask].max()))

    return np.array(ui, dtype=np.int64), np.array(us, dtype=np.float64)


def frame_scores_nearest(total_frames, indices, scores):
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
    for i, score in enumerate(scores):
        s = max(0, min(total_frames, int(bounds[i])))
        e = max(0, min(total_frames, int(bounds[i + 1])))
        if e > s:
            fs[s:e] = float(score)

    return fs


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


def resolve_video_path(video_root: Path, rel_path: str, class_name: str):
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


def read_clip_tensor(video_path: Path, center_frame: int, build_clip_len: int, train_clip_len: int, image_size: int, total_frames: int):
    start = max(0, int(center_frame) - build_clip_len // 2)
    end = start + build_clip_len

    if total_frames > 0:
        end = min(total_frames, end)
        start = max(0, end - build_clip_len)

    frame_indices = np.linspace(start, max(start, end - 1), train_clip_len).astype(np.int64)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    frames = []

    last_frame = None
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()

        if not ok:
            if last_frame is None:
                frame = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            else:
                frame = last_frame.copy()
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last_frame = frame.copy()

        img = Image.fromarray(frame)
        frames.append(tfm(img))

    cap.release()

    return torch.stack(frames, dim=0)


def load_checkpoint(model, ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=True)
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)

    return model


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


def save_confusion_csv(path: Path, cm, class_names):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + class_names)
        for name, row in zip(class_names, cm):
            w.writerow([name] + row.tolist())


def save_confusion_png(path: Path, cm, class_names, title):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cm)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-list", required=True)
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--annotation-file", required=True)
    ap.add_argument("--phase2-checkpoint", required=True)
    ap.add_argument("--phase2-model", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--build-clip-len", type=int, default=64)
    ap.add_argument("--train-clip-len", type=int, default=16)
    ap.add_argument("--min-center-gap", type=int, default=24)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--num-classes", type=int, default=13)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    clean_list = Path(args.clean_list)
    video_root = Path(args.video_root)
    annotation_file = Path(args.annotation_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    ann = parse_annotations(annotation_file)

    model = build_phase2_model(
        args.phase2_model,
        num_classes=args.num_classes,
        pretrained=False,
    ).to(device)

    model = load_checkpoint(model, Path(args.phase2_checkpoint))
    model.eval()

    rows = []
    with open(clean_list, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    all_frame_y = []
    all_frame_s = []
    video_y = []
    video_s = []

    event_true = []
    event_pred = []
    per_video_rows = []
    failures = []

    for r in tqdm(rows, desc="Eval clean test videos"):
        score_file = Path(r["phase1_score_file"])
        x = torch.load(score_file, map_location="cpu")

        rel_path = str(x.get("rel_path", r.get("rel_path", "")))
        class_name = str(x.get("class_name", r.get("class_name", "")))
        video_label = int(x.get("video_label", r.get("video_label", 0)))
        total_frames = int(x.get("total_frames", 0))

        selected_indices = x["selected_indices"].cpu().numpy()
        segment_scores = x["segment_scores"].float().cpu().numpy()

        video_score = float(x.get("video_score", np.max(segment_scores)))

        name = Path(rel_path).name
        ranges = ann.get(name, {"ranges": []}).get("ranges", [])
        if video_label == 0 or "Normal" in class_name:
            ranges = []

        fs = frame_scores_nearest(total_frames, selected_indices, segment_scores)
        gt = make_gt(len(fs), ranges)

        all_frame_y.extend(gt.tolist())
        all_frame_s.extend(fs.tolist())

        video_y.append(video_label)
        video_s.append(video_score)

        out_row = {
            "rel_path": rel_path,
            "class_name": class_name,
            "video_label": video_label,
            "phase1_video_score": video_score,
            "phase2_pred_class": None,
            "phase2_pred_id": None,
            "phase2_confidence": None,
            "num_phase2_clips": 0,
        }

        if video_label == 1 and class_name in ANOMALY_CLASSES:
            video_path = resolve_video_path(video_root, rel_path, class_name)

            if video_path is None:
                failures.append({"rel_path": rel_path, "error": "video_not_found"})
                per_video_rows.append(out_row)
                continue

            centers = select_topk_centers(
                selected_indices=selected_indices,
                segment_scores=segment_scores,
                k=args.top_k,
                min_center_gap=args.min_center_gap,
            )

            clip_tensors = []
            for center, score, token_idx in centers:
                try:
                    clip = read_clip_tensor(
                        video_path=video_path,
                        center_frame=center,
                        build_clip_len=args.build_clip_len,
                        train_clip_len=args.train_clip_len,
                        image_size=args.image_size,
                        total_frames=total_frames,
                    )
                    clip_tensors.append(clip)
                except Exception as e:
                    failures.append({"rel_path": rel_path, "center": center, "error": repr(e)})

            if clip_tensors:
                probs_all = []

                with torch.no_grad():
                    for i in range(0, len(clip_tensors), args.batch_size):
                        batch = torch.stack(clip_tensors[i:i + args.batch_size], dim=0).to(device)
                        logits = model(batch)
                        probs = torch.softmax(logits, dim=1)
                        probs_all.append(probs.cpu())

                probs_all = torch.cat(probs_all, dim=0).numpy()
                probs_mean = probs_all.mean(axis=0)

                pred_id = int(np.argmax(probs_mean))
                pred_class = ANOMALY_CLASSES[pred_id]
                conf = float(probs_mean[pred_id])

                true_id = ANOMALY_CLASSES.index(class_name)

                event_true.append(true_id)
                event_pred.append(pred_id)

                out_row.update({
                    "phase2_pred_class": pred_class,
                    "phase2_pred_id": pred_id,
                    "phase2_confidence": conf,
                    "num_phase2_clips": len(clip_tensors),
                })

        per_video_rows.append(out_row)

    summary = {
        "num_clean_videos": len(rows),
        "num_frame_samples": len(all_frame_y),
        "num_positive_frames": int(np.sum(all_frame_y)),
        "phase1_frame_auc": safe_auc(all_frame_y, all_frame_s),
        "phase1_frame_ap": safe_ap(all_frame_y, all_frame_s),
        "phase1_video_auc": safe_auc(video_y, video_s),
        "num_event_videos": len(event_true),
        "phase2_event_accuracy": float(accuracy_score(event_true, event_pred)) if event_true else None,
        "phase2_event_macro_f1": float(f1_score(event_true, event_pred, average="macro", zero_division=0)) if event_true else None,
        "failures": len(failures),
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(out_dir / "per_video_predictions.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = list(per_video_rows[0].keys()) if per_video_rows else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(per_video_rows)

    with open(out_dir / "failures.jsonl", "w", encoding="utf-8") as f:
        for item in failures:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    if event_true:
        labels = list(range(args.num_classes))
        cm = confusion_matrix(event_true, event_pred, labels=labels)
        save_confusion_csv(out_dir / "phase2_event_confusion_matrix.csv", cm, ANOMALY_CLASSES)
        save_confusion_png(out_dir / "phase2_event_confusion_matrix.png", cm, ANOMALY_CLASSES, "Phase2 event confusion matrix")

        report = classification_report(
            event_true,
            event_pred,
            labels=labels,
            target_names=ANOMALY_CLASSES,
            zero_division=0,
        )

        with open(out_dir / "phase2_event_classification_report.txt", "w", encoding="utf-8") as f:
            f.write(report)

        print(report)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Saved to:", out_dir)


if __name__ == "__main__":
    main()
