#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

CODE_TO_CLASS = {
    "A": "Normal",
    "B1": "Fighting",
    "B2": "Shooting",
    "B4": "Riot",
    "B5": "Abuse",
    "B6": "Car accident",
    "G": "Explosion",
}

ANOMALY_CLASSES = [
    "Fighting",
    "Shooting",
    "Riot",
    "Abuse",
    "Car accident",
    "Explosion",
]

PHASE2_CLASS_MAP = {c: i for i, c in enumerate(ANOMALY_CLASSES)}


def parse_codes_from_stem(stem: str):

    tail = stem.split("_")[-1]
    raw_codes = tail.split("-")

    codes = []
    for c in raw_codes:
        c = c.strip()
        if c in CODE_TO_CLASS:
            codes.append(c)

    if codes:
        return codes

    found = re.findall(r"(B1|B2|B4|B5|B6|G|A)", stem)
    return [c for c in found if c in CODE_TO_CLASS]


def labels_from_name(path: Path):
    codes = parse_codes_from_stem(path.stem)

    if not codes:
        return {
            "codes": [],
            "video_label": 0,
            "class_name": "Normal",
            "phase2_class_id": -1,
            "multi_classes": [],
        }

    classes = [CODE_TO_CLASS[c] for c in codes if CODE_TO_CLASS[c] != "Normal"]

    video_label = 1 if len(classes) > 0 else 0
    class_name = classes[0] if classes else "Normal"
    phase2_class_id = PHASE2_CLASS_MAP.get(class_name, -1)

    return {
        "codes": codes,
        "video_label": video_label,
        "class_name": class_name,
        "phase2_class_id": phase2_class_id,
        "multi_classes": classes,
    }


def parse_annotations(path: Path):

    mapper = {}

    if not path.exists():
        return mapper

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue

            parts = raw.split()
            if len(parts) < 3:
                continue

            vid = Path(parts[0]).stem

            nums = []
            for x in parts[1:]:
                try:
                    nums.append(int(float(x)))
                except Exception:
                    pass

            ranges = []
            for i in range(0, len(nums) - 1, 2):
                a, b = nums[i], nums[i + 1]
                if a >= 0 and b >= 0 and b > a:
                    ranges.append([int(a), int(b)])

            mapper[vid] = ranges

    return mapper


def scan_videos(root: Path, subdirs):
    rows = []

    for sd in subdirs:
        d = root / sd
        if not d.exists():
            print("[WARN] missing dir:", d)
            continue

        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                rows.append(p)

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xd-root", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    xd_root = Path(args.xd_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dirs = ["1-1004", "1005-2004", "2005-2804", "2805-3319", "3320-3954"]
    test_dirs = ["videos"]

    ann = parse_annotations(xd_root / "annotations.txt")

    train_files = scan_videos(xd_root, train_dirs)
    test_files = scan_videos(xd_root, test_dirs)

    print("train files:", len(train_files))
    print("test files :", len(test_files))
    print("annotation videos:", len(ann))

    all_rows = []

    for split, files in [("train", train_files), ("test", test_files)]:
        for p in files:
            info = labels_from_name(p)
            rel_path = str(p.relative_to(xd_root))
            stem = p.stem

            ranges = ann.get(stem, [])

            if split == "test" and ranges and info["video_label"] == 0:
                info["video_label"] = 1

            row = {
                "video_path": str(p),
                "rel_path": rel_path,
                "video_id": stem,
                "split": split,
                "video_label": int(info["video_label"]),
                "class_name": info["class_name"],
                "phase2_class_id": int(info["phase2_class_id"]),
                "codes": info["codes"],
                "multi_classes": info["multi_classes"],
                "annotation_ranges": ranges,
            }

            all_rows.append(row)

    for split in ["train", "test"]:
        out_path = out_dir / f"{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in all_rows:
                if r["split"] == split:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("saved:", out_path)

    with open(out_dir / "all.jsonl", "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(out_dir / "class_map_phase2.json", "w", encoding="utf-8") as f:
        json.dump(PHASE2_CLASS_MAP, f, indent=2, ensure_ascii=False)

    with open(out_dir / "class_map_binary.json", "w", encoding="utf-8") as f:
        json.dump({"Normal": 0, "Violence": 1}, f, indent=2, ensure_ascii=False)

    print("\n===== Split / binary count =====")
    cnt = Counter((r["split"], r["video_label"]) for r in all_rows)
    for k, v in sorted(cnt.items()):
        print(k, v)

    print("\n===== Split / class count =====")
    cnt = Counter((r["split"], r["class_name"]) for r in all_rows)
    for k, v in sorted(cnt.items()):
        print(k, v)

    print("\n===== Test videos with annotation ranges =====")
    print(sum(1 for r in all_rows if r["split"] == "test" and r["annotation_ranges"]))


if __name__ == "__main__":
    main()
