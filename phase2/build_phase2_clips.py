#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.video_io import find_video_path, read_clip_rgb, write_clip_mp4
from utils.annotations import is_normal_relpath, class_name_from_relpath
from utils.logging_utils import setup_logger

ANOMALY_CLASSES = ['Abuse','Arrest','Arson','Assault','Burglary','Explosion','Fighting','RoadAccidents','Robbery','Shooting','Shoplifting','Stealing','Vandalism']
CLASS_TO_ID = {c: i for i, c in enumerate(ANOMALY_CLASSES)}


def load_scores(scores_root: Path):
    return sorted(scores_root.rglob('*.pt'))


def compute_threshold(files, mode='max_normal'):
    normal_scores = []
    for p in files:
        d = torch.load(p, map_location='cpu')
        if int(d.get('video_label', 0)) == 0 or is_normal_relpath(d.get('rel_path', '')):
            normal_scores.extend(torch.as_tensor(d['segment_scores']).float().tolist())
    if not normal_scores:
        raise RuntimeError('No normal segment scores found. Cannot compute threshold.')
    arr = np.asarray(normal_scores, dtype=np.float32)
    if mode == 'max_normal':
        return float(arr.max())
    if mode == 'p99_normal':
        return float(np.percentile(arr, 99))
    if mode == 'p95_normal':
        return float(np.percentile(arr, 95))
    if mode == 'mean3std_normal':
        return float(arr.mean() + 3 * arr.std())
    raise ValueError(f'Unknown threshold mode: {mode}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video-root', required=True)
    ap.add_argument('--scores-root', required=True, help='Usually .../phase1_scores/evit/train')
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--threshold-mode', default='max_normal', choices=['max_normal','p99_normal','p95_normal','mean3std_normal'])
    ap.add_argument('--clip-len', type=int, default=32)
    ap.add_argument('--image-size', type=int, default=224)
    ap.add_argument('--max-clips-per-video', type=int, default=5)
    ap.add_argument('--fallback-topk', type=int, default=1)
    ap.add_argument('--use-center', default='selected_index', choices=['selected_index','segment_center'])
    ap.add_argument('--exclude-normal', action='store_true')
    ap.add_argument('--val-ratio', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger('build_phase2_clips', out_root / 'build_phase2_clips.log')
    files = load_scores(Path(args.scores_root))
    threshold = compute_threshold(files, args.threshold_mode)
    logger.info(f'Loaded score files={len(files)} threshold_mode={args.threshold_mode} threshold={threshold:.6f}')

    rows = []
    for p in tqdm(files, desc='Build phase2 clips'):
        d = torch.load(p, map_location='cpu')
        rel = d.get('rel_path', '')
        cls = d.get('class_name') or class_name_from_relpath(rel)
        video_label = int(d.get('video_label', 0))
        if args.exclude_normal and video_label == 0:
            continue
        if cls not in CLASS_TO_ID:
            continue
        scores = torch.as_tensor(d['segment_scores']).float().numpy()
        selected = torch.as_tensor(d.get('selected_indices', [])).long().numpy()
        bounds = torch.as_tensor(d.get('segment_bounds', [])).long().numpy()
        candidates = np.where(scores > threshold)[0].tolist()
        if not candidates and args.fallback_topk > 0:
            candidates = np.argsort(-scores)[:args.fallback_topk].tolist()
        candidates = sorted(candidates, key=lambda i: float(scores[i]), reverse=True)[:args.max_clips_per_video]
        if not candidates:
            continue
        vp = find_video_path(args.video_root, rel)
        if vp is None:
            logger.warning(f'Video not found for clip build: {rel}')
            continue
        split = 'val' if random.random() < args.val_ratio else 'train'
        for seg_id in candidates:
            if args.use_center == 'selected_index' and len(selected) > seg_id and selected[seg_id] >= 0:
                center = int(selected[seg_id])
            elif len(bounds) > seg_id:
                center = int((int(bounds[seg_id][0]) + int(bounds[seg_id][1])) // 2)
            else:
                center = 0
            start = center - args.clip_len // 2
            end = start + args.clip_len - 1
            score = float(scores[seg_id])
            clip_name = f'{Path(rel).stem}_seg{seg_id:02d}_score{score:.3f}.mp4'
            out_path = out_root / split / cls / clip_name
            try:
                frames = read_clip_rgb(vp, start, end, clip_len=args.clip_len, image_size=args.image_size)
                write_clip_mp4(frames, out_path, fps=15.0)
                rows.append({
                    'clip_path': str(out_path), 'split': split, 'video_rel_path': rel,
                    'class_name': cls, 'class_id': CLASS_TO_ID[cls], 'segment_id': int(seg_id),
                    'score': score, 'threshold': threshold, 'center_frame': center,
                    'start_frame': int(start), 'end_frame': int(end), 'source_score_file': str(p),
                    'threshold_mode': args.threshold_mode,
                })
            except Exception as e:
                logger.error(f'CLIP FAILED | {rel} seg={seg_id} | {repr(e)}')

    manifest = out_root / 'manifest_phase2.jsonl'
    with open(manifest, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    with open(out_root / 'class_map.json', 'w', encoding='utf-8') as f:
        json.dump(CLASS_TO_ID, f, indent=2)
    logger.info(f'DONE. clips={len(rows)} manifest={manifest}')


if __name__ == '__main__':
    main()
