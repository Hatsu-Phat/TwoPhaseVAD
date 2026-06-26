#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from collections import defaultdict, Counter


def hval(s: str):
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase2-root", required=True)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    root = Path(args.phase2_root)
    src = root / "manifest_phase2.jsonl"

    rows = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    class_to_sources = defaultdict(set)
    for r in rows:
        source = r.get("rel_path") or r.get("source_video")
        class_to_sources[r["class_name"]].add(source)

    source_fold = {}
    for cls, sources in sorted(class_to_sources.items()):
        sources = sorted(list(sources), key=lambda s: hval(cls + "|" + s))
        for i, source in enumerate(sources):
            source_fold[source] = i % args.folds

    for fold in range(args.folds):
        out = root / f"manifest_phase2_video_cv_fold{fold}.jsonl"
        new_rows = []

        for r in rows:
            r = dict(r)
            source = r.get("rel_path") or r.get("source_video")
            r["split"] = "val" if source_fold[source] == fold else "train"
            r["cv_fold"] = fold
            new_rows.append(r)

        with open(out, "w", encoding="utf-8") as f:
            for r in new_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        source_splits = defaultdict(set)
        cnt = Counter()

        for r in new_rows:
            source = r.get("rel_path") or r.get("source_video")
            source_splits[source].add(r["split"])
            cnt[(r["split"], r["class_name"])] += 1

        leaks = [s for s, sp in source_splits.items() if len(sp) > 1]

        print("=" * 80)
        print("fold:", fold)
        print("saved:", out)
        print("leaked source videos:", len(leaks))
        print("VAL per class:")
        for cls in sorted(class_to_sources):
            print(f"{cls:15s} {cnt[('val', cls)]}")


if __name__ == "__main__":
    main()
