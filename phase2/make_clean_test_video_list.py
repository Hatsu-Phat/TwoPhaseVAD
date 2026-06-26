#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch


def load_phase2_train_sources(manifest_path: Path):
    train_sources = set()
    val_sources = set()

    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            src = r.get("rel_path") or r.get("source_video")
            if r.get("split") == "train":
                train_sources.add(src)
            elif r.get("split") == "val":
                val_sources.add(src)

    return train_sources, val_sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1-test-scores", required=True)
    ap.add_argument("--phase2-manifest", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-txt", required=True)
    args = ap.parse_args()

    phase1_test_scores = Path(args.phase1_test_scores)
    phase2_manifest = Path(args.phase2_manifest)

    train_sources, val_sources = load_phase2_train_sources(phase2_manifest)

    rows = []
    excluded_phase2_train = 0

    for p in sorted(phase1_test_scores.rglob("*.pt")):
        x = torch.load(p, map_location="cpu")

        rel_path = str(x.get("rel_path", ""))
        class_name = str(x.get("class_name", p.parent.name))
        video_label = int(x.get("video_label", 0))

        if rel_path in train_sources:
            excluded_phase2_train += 1
            continue

        rows.append({
            "rel_path": rel_path,
            "class_name": class_name,
            "video_label": video_label,
            "phase1_score_file": str(p),
            "in_phase2_val": rel_path in val_sources,
            "phase2_manifest": str(phase2_manifest),
        })

    out_jsonl = Path(args.out_jsonl)
    out_txt = Path(args.out_txt)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(out_txt, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(r["rel_path"] + "\n")

    print("phase1_test_scores:", phase1_test_scores)
    print("phase2_manifest:", phase2_manifest)
    print("phase2 train sources:", len(train_sources))
    print("phase2 val sources:", len(val_sources))
    print("excluded because in phase2 train:", excluded_phase2_train)
    print("clean test videos:", len(rows))
    print("saved jsonl:", out_jsonl)
    print("saved txt:", out_txt)

    print("\nFirst 30 clean videos:")
    for r in rows[:30]:
        print(r["class_name"], r["video_label"], r["rel_path"], "in_phase2_val=", r["in_phase2_val"])


if __name__ == "__main__":
    main()
