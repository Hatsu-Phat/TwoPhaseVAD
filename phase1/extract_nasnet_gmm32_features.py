#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.annotations import read_split_file, parse_temporal_annotations, is_normal_relpath, class_name_from_relpath, segment_labels_from_ranges
from utils.video_io import find_video_path, get_video_info, temporal_segments
from utils.logging_utils import setup_logger


class NASNetMobileFeatureExtractor(nn.Module):
    def __init__(self, model_name='nasnetamobile', pretrained=True):
        super().__init__()
        self.backend = None
        self.model_name = model_name
        self.out_dim = None

        if model_name.lower() in {'nasnetamobile', 'nasnetmobile', 'nasnet_mobile'}:
            try:
                import pretrainedmodels
                m = pretrainedmodels.__dict__['nasnetamobile'](num_classes=1000, pretrained='imagenet' if pretrained else None)
                self.model = m
                self.backend = 'pretrainedmodels_nasnetamobile'
                self.out_dim = 1056
                return
            except Exception as e:
                self._pretrainedmodels_error = repr(e)

        try:
            import timm
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')
            self.backend = 'timm'
            self.out_dim = getattr(self.model, 'num_features', None)
            return
        except Exception as e:
            msg = (
                f'Cannot create backbone {model_name}. For NASNetMobile install pretrainedmodels: '\
                f'pip install pretrainedmodels. pretrainedmodels_error={getattr(self, "_pretrainedmodels_error", None)} timm_error={repr(e)}'
            )
            raise RuntimeError(msg)

    def forward(self, x):
        if self.backend == 'pretrainedmodels_nasnetamobile':
            feat = self.model.features(x)
            feat = torch.nn.functional.adaptive_avg_pool2d(feat, 1).flatten(1)
            return feat
        return self.model(x)


def gmm_select_one_frame_per_segment(video_path: Path, num_segments: int, image_size: int, frame_stride: int = 1):
    fps, total_frames, width, height = get_video_info(video_path)
    segs = temporal_segments(total_frames, num_segments)
    best_scores = np.full((num_segments,), -1.0, dtype=np.float32)
    best_indices = np.full((num_segments,), -1, dtype=np.int64)
    best_frames = [None for _ in range(num_segments)]
    last_good = None

    backsub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f'Cannot open video: {video_path}')

    idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break
        if idx % frame_stride != 0:
            idx += 1
            continue
        mask = backsub.apply(frame_bgr)
        score = float(mask.sum()) / float(mask.size * 255.0)
        seg_id = min(num_segments - 1, int(idx * num_segments / max(total_frames, 1)))
        if score > best_scores[seg_id]:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            best_scores[seg_id] = score
            best_indices[seg_id] = idx
            best_frames[seg_id] = frame_rgb
            last_good = frame_rgb
        idx += 1

    cap.release()

    available = [i for i, f in enumerate(best_frames) if f is not None]
    if not available:
        cap = cv2.VideoCapture(str(video_path))
        ok, frame_bgr = cap.read()
        cap.release()
        if not ok or frame_bgr is None:
            raise RuntimeError(f'Cannot decode any frame: {video_path}')
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        for i, (s, e) in enumerate(segs):
            best_frames[i] = frame_rgb
            best_indices[i] = s
            best_scores[i] = 0.0
    else:
        last = None
        last_idx = None
        last_score = 0.0
        for i in range(num_segments):
            if best_frames[i] is None:
                if last is None:
                    j = available[0]
                    last = best_frames[j]
                    last_idx = best_indices[j]
                    last_score = best_scores[j]
                best_frames[i] = last.copy()
                best_indices[i] = int(last_idx)
                best_scores[i] = float(last_score)
            else:
                last = best_frames[i]
                last_idx = best_indices[i]
                last_score = best_scores[i]

    resized = [cv2.resize(f, (image_size, image_size), interpolation=cv2.INTER_LINEAR) for f in best_frames]
    return resized, best_indices, best_scores, np.array(segs, dtype=np.int64), fps, total_frames, width, height


def save_jsonl(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video-root', required=True)
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--annotation-file', default=None)
    ap.add_argument('--train-split-files', nargs='*', default=[])
    ap.add_argument('--test-split-files', nargs='*', default=[])
    ap.add_argument('--num-segments', type=int, default=32)
    ap.add_argument('--image-size', type=int, default=224)
    ap.add_argument('--backbone', default='nasnetamobile')
    ap.add_argument('--batch-size-frames', type=int, default=64)
    ap.add_argument('--frame-stride', type=int, default=1)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--save-float16', action='store_true')
    ap.add_argument('--skip-existing', action='store_true')
    ap.add_argument('--limit', type=int, default=-1)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    logger = setup_logger('extract_nasnet_gmm32', out_root / 'logs' / 'extract_nasnet_gmm32.log')
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    logger.info(f'Using device={device}')

    anno_map = parse_temporal_annotations(args.annotation_file)
    logger.info(f'Loaded temporal annotations: {len(anno_map)}')

    split_files = [('train', p) for p in args.train_split_files] + [('test', p) for p in args.test_split_files]
    items = []
    for split, sf in split_files:
        rels = read_split_file(sf)
        logger.info(f'Loaded {len(rels)} videos from {sf} as split={split}')
        for rel in rels:
            vp = find_video_path(args.video_root, rel)
            if vp is None:
                logger.warning(f'Video not found: {rel}')
                continue
            cls = class_name_from_relpath(rel)
            label = 0 if is_normal_relpath(rel) else 1
            items.append({'split': split, 'rel_path': rel, 'abs_path': str(vp), 'class_name': cls, 'video_label': label})
    if args.limit > 0:
        items = items[:args.limit]
    logger.info(f'Total valid items: {len(items)}')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model = NASNetMobileFeatureExtractor(args.backbone, pretrained=True).to(device).eval()
    logger.info(f'Backbone={args.backbone}, backend={getattr(model, "backend", None)}, out_dim={getattr(model, "out_dim", None)}')

    manifest = []
    failures = []
    with torch.no_grad():
        for item in tqdm(items, desc='Extract NASNet-GMM32'):
            rel = item['rel_path']
            out_path = out_root / item['split'] / item['class_name'] / f'{Path(rel).stem}.pt'
            if args.skip_existing and out_path.exists() and out_path.stat().st_size > 0:
                continue
            try:
                frames, sel_idx, motion, segs, fps, total, width, height = gmm_select_one_frame_per_segment(
                    Path(item['abs_path']), args.num_segments, args.image_size, frame_stride=args.frame_stride
                )
                batch = torch.stack([transform(f) for f in frames], dim=0)
                feats = []
                for start in range(0, batch.shape[0], args.batch_size_frames):
                    x = batch[start:start + args.batch_size_frames].to(device, non_blocking=True)
                    y = model(x).detach().cpu()
                    feats.append(y)
                features = torch.cat(feats, dim=0)
                if args.save_float16:
                    features = features.half()

                anno = anno_map.get(Path(rel).name, None)
                ranges = anno['ranges'] if anno else []
                gt_seg = segment_labels_from_ranges(total, ranges, args.num_segments) if item['split'] == 'test' else np.full((args.num_segments,), -1, dtype=np.int64)

                out_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = out_path.with_suffix('.tmp')
                torch.save({
                    'features': features,
                    'selected_indices': torch.as_tensor(sel_idx, dtype=torch.long),
                    'segment_bounds': torch.as_tensor(segs, dtype=torch.long),
                    'motion_scores': torch.as_tensor(motion, dtype=torch.float32),
                    'video_label': int(item['video_label']),
                    'class_name': item['class_name'],
                    'rel_path': rel,
                    'fps': float(fps),
                    'total_frames': int(total),
                    'width': int(width),
                    'height': int(height),
                    'gt_segment_labels': torch.as_tensor(gt_seg, dtype=torch.long),
                    'annotation_ranges': ranges,
                    'backbone': args.backbone,
                    'feature_shape': list(features.shape),
                }, tmp)
                tmp.replace(out_path)
                manifest.append({'path': str(out_path), 'split': item['split'], 'rel_path': rel, 'class_name': item['class_name'], 'video_label': item['video_label'], 'feature_shape': list(features.shape)})
            except Exception as e:
                failures.append({'rel_path': rel, 'error': repr(e)})
                logger.error(f'FAILED | {rel} | {repr(e)}')

    save_jsonl(out_root / 'annotations' / 'manifest.jsonl', manifest)
    save_jsonl(out_root / 'annotations' / 'failures.jsonl', failures)
    logger.info(f'DONE. saved={len(manifest)} failures={len(failures)}')


if __name__ == '__main__':
    main()
