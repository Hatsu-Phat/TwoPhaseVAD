#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.annotations import parse_temporal_annotations, frame_labels_from_ranges, is_normal_relpath
from utils.metrics import safe_roc_auc, safe_average_precision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores-root', required=True, help='Usually .../phase1_scores/evit/test')
    ap.add_argument('--annotation-file', required=True)
    ap.add_argument('--out-json', required=True)
    args = ap.parse_args()

    anno = parse_temporal_annotations(args.annotation_file)
    y_true_all, y_score_all = [], []
    video_rows = []
    for p in sorted(Path(args.scores_root).rglob('*.pt')):
        d = torch.load(p, map_location='cpu')
        rel = d.get('rel_path', '')
        total = int(d.get('total_frames', 0))
        if total <= 0:
            continue
        seg_scores = torch.as_tensor(d['segment_scores']).float().numpy()
        bounds = torch.as_tensor(d['segment_bounds']).long().numpy()
        frame_scores = np.zeros(total, dtype=np.float32)
        for i, (s, e) in enumerate(bounds):
            s = max(0, min(int(s), total - 1)); e = max(s, min(int(e), total - 1))
            frame_scores[s:e+1] = seg_scores[min(i, len(seg_scores)-1)]
        ranges = anno.get(Path(rel).name, {}).get('ranges', [])
        frame_labels = frame_labels_from_ranges(total, ranges)
        y_true_all.extend(frame_labels.tolist())
        y_score_all.extend(frame_scores.tolist())
        video_rows.append({'rel_path': rel, 'video_score': float(d.get('video_score', np.max(seg_scores))), 'video_label': 0 if is_normal_relpath(rel) else 1})

    frame_auc = safe_roc_auc(y_true_all, y_score_all)
    frame_ap = safe_average_precision(y_true_all, y_score_all)
    video_auc = safe_roc_auc([r['video_label'] for r in video_rows], [r['video_score'] for r in video_rows]) if video_rows else float('nan')
    out = {'frame_auc': frame_auc, 'frame_ap': frame_ap, 'video_auc': video_auc, 'num_frames': len(y_true_all), 'num_videos': len(video_rows)}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
