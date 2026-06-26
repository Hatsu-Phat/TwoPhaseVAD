#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from collections import defaultdict
from pathlib import Path

import torch


def hval(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def safe_symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def collect_files(root: Path, split: str):
    files = sorted((root / split).rglob("*.pt"))
    rows = []

    for p in files:
        x = torch.load(p, map_location="cpu")
        cls = str(x.get("class_name", p.parent.name))
        rel = p.relative_to(root / split)
        rows.append({
            "path": p,
            "rel": rel,
            "class_name": cls,
            "video_label": int(x.get("video_label", 1)),
            "rel_path": str(x.get("rel_path", rel.as_posix())),
        })

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    features_root = Path(args.features_root)
    out_root = Path(args.out_root)

    train_rows = collect_files(features_root, "train")
    test_rows = collect_files(features_root, "test")

    by_class = defaultdict(list)
    for r in test_rows:
        by_class[r["class_name"]].append(r)

    test_fold = {}
    for cls, rows in by_class.items():
        rows = sorted(rows, key=lambda r: hval(cls + "|" + r["rel_path"]))
        for i, r in enumerate(rows):
            test_fold[r["rel_path"]] = i % args.folds

    if args.clean and out_root.exists():
        shutil.rmtree(out_root)

    out_root.mkdir(parents=True, exist_ok=True)

    for fold in range(args.folds):
        fold_root = out_root / f"fold{fold}"

        val_rows = [r for r in test_rows if test_fold[r["rel_path"]] == fold]

        cv_train_rows = train_rows + [r for r in test_rows if test_fold[r["rel_path"]] != fold]

        for r in cv_train_rows:
            dst = fold_root / "train" / r["rel"]
            safe_symlink(r["path"], dst)

        for r in val_rows:
            dst = fold_root / "test" / r["rel"]
            safe_symlink(r["path"], dst)

        print(f"fold={fold} train={len(cv_train_rows)} val/test={len(val_rows)} root={fold_root}")

    print("DONE:", out_root)


if __name__ == "__main__":
    main()
