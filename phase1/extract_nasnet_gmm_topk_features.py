#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
import os
import ssl
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.annotations import (
    read_split_file,
    parse_temporal_annotations,
    is_normal_relpath,
    class_name_from_relpath,
)
from utils.video_io import find_video_path, get_video_info
from utils.logging_utils import setup_logger


class NASNetMobileFeatureExtractor(nn.Module):
    def __init__(self, model_name: str = 'nasnetamobile', pretrained: bool = True):
        super().__init__()
        self.backend = None
        self.model_name = model_name
        self.out_dim = None

        if model_name.lower() in {'nasnetamobile', 'nasnetmobile', 'nasnet_mobile'}:
            try:
                import pretrainedmodels
                try:
                    import pretrainedmodels.models.nasnet_mobile as nasnet_mobile_settings
                    fallback_url = os.environ.get(
                        'NASNETMOBILE_CKPT_URL',
                        'https://github.com/veronikayurchuk/pretrained-models.pytorch/releases/download/v1.0/nasnetmobile-7e03cead.pth.tar'
                    )
                    nasnet_mobile_settings.pretrained_settings['nasnetamobile']['imagenet']['url'] = fallback_url
                except Exception:
                    pass

                m = pretrainedmodels.__dict__['nasnetamobile'](
                    num_classes=1000,
                    pretrained='imagenet' if pretrained else None,
                )
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
                f'Cannot create backbone {model_name}. For NASNetMobile install pretrainedmodels: '
                f'pip install pretrainedmodels. pretrainedmodels_error={getattr(self, "_pretrainedmodels_error", None)} '
                f'timm_error={repr(e)}'
            )
            raise RuntimeError(msg)

    def forward(self, x):
        if self.backend == 'pretrainedmodels_nasnetamobile':
            feat = self.model.features(x)
            feat = torch.nn.functional.adaptive_avg_pool2d(feat, 1).flatten(1)
            return feat
        return self.model(x)


def token_bounds_from_selected_indices(selected_indices: np.ndarray, total_frames: int) -> np.ndarray:

    total_frames = max(int(total_frames), 1)
    centers = np.asarray(selected_indices, dtype=np.int64)
    if centers.size == 0:
        return np.zeros((0, 2), dtype=np.int64)

    centers = np.clip(centers, 0, total_frames - 1)
    bounds = []
    n = len(centers)
    for i in range(n):
        if i == 0:
            s = 0
        else:
            s = int((int(centers[i - 1]) + int(centers[i])) // 2) + 1

        if i == n - 1:
            e = total_frames - 1
        else:
            e = int((int(centers[i]) + int(centers[i + 1])) // 2)

        s = max(0, min(s, total_frames - 1))
        e = max(s, min(e, total_frames - 1))
        bounds.append((s, e))
    return np.asarray(bounds, dtype=np.int64)


def token_labels_from_ranges(bounds_0based: np.ndarray, ranges_1based: List[Tuple[int, int]]) -> np.ndarray:
    labels = np.zeros((len(bounds_0based),), dtype=np.int64)
    if not ranges_1based:
        return labels
    for i, (s0, e0) in enumerate(bounds_0based):
        s1, e1 = int(s0) + 1, int(e0) + 1
        for a, b in ranges_1based:
            if a > 0 and b > 0 and max(s1, int(a)) <= min(e1, int(b)):
                labels[i] = 1
                break
    return labels


def gmm_select_global_topk_frames(
    video_path: Path,
    top_k: int,
    image_size: int,
    frame_stride: int = 1,
    warmup_frames: int = 5,
    alpha_min: float | None = None,
    alpha_max: float | None = None,
):

    fps, total_frames, width, height = get_video_info(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f'Cannot open video: {video_path}')

    backsub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    heap: List[Tuple[float, int, np.ndarray]] = []
    first_good = None
    last_good = None
    last_idx = 0

    idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break

        if first_good is None:
            first_good = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        last_good = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        last_idx = idx

        mask = backsub.apply(frame_bgr)

        if idx >= warmup_frames and idx % max(1, frame_stride) == 0:
            alpha = float(mask.sum()) / float(mask.size * 255.0)
            if alpha_min is not None and alpha < alpha_min:
                idx += 1
                continue
            if alpha_max is not None and alpha > alpha_max:
                idx += 1
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            item = (alpha, idx, frame_rgb)
            if len(heap) < top_k:
                heapq.heappush(heap, item)
            else:
                if alpha > heap[0][0]:
                    heapq.heapreplace(heap, item)
        idx += 1

    cap.release()

    if first_good is None:
        raise RuntimeError(f'Cannot decode any frame: {video_path}')

    selected = sorted(heap, key=lambda x: x[1])
    padding_mask = []

    if not selected:
        selected = [(0.0, 0, first_good)]

    while len(selected) < top_k:
        score, frame_idx, frame_rgb = selected[-1]
        selected.append((float(score), int(frame_idx), frame_rgb.copy()))
        padding_mask.append(True)

    if len(padding_mask) < top_k:
        padding_mask = [False] * (top_k - len(padding_mask)) + padding_mask
    padding_mask = padding_mask[:top_k]

    selected = selected[:top_k]
    selected_indices = np.asarray([x[1] for x in selected], dtype=np.int64)
    motion_scores = np.asarray([x[0] for x in selected], dtype=np.float32)
    frames = [cv2.resize(x[2], (image_size, image_size), interpolation=cv2.INTER_LINEAR) for x in selected]
    token_bounds = token_bounds_from_selected_indices(selected_indices, total_frames)

    return frames, selected_indices, motion_scores, token_bounds, np.asarray(padding_mask, dtype=bool), fps, total_frames, width, height


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
    ap.add_argument('--top-k', '--num-frames', dest='top_k', type=int, default=30,
                    help='Paper-style global Top-K motion frames. Default: 30.')
    ap.add_argument('--image-size', type=int, default=224)
    ap.add_argument('--backbone', default='nasnetamobile')
    ap.add_argument('--batch-size-frames', type=int, default=64)
    ap.add_argument('--frame-stride', type=int, default=1)
    ap.add_argument('--warmup-frames', type=int, default=5,
                    help='Frames used to warm up GMM; they are not eligible for Top-K selection.')
    ap.add_argument('--alpha-min', type=float, default=None)
    ap.add_argument('--alpha-max', type=float, default=None)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--save-float16', action='store_true')
    ap.add_argument('--skip-existing', action='store_true')
    ap.add_argument('--limit', type=int, default=-1)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    logger = setup_logger('extract_nasnet_gmm_topk', out_root / 'logs' / 'extract_nasnet_gmm_topk.log')
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith('cuda') else 'cpu')
    logger.info(f'Using device={device}')
    logger.info(f'Paper-style selector: global GMM Top-K, top_k={args.top_k}, no uniform temporal segmentation')

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
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    model = NASNetMobileFeatureExtractor(args.backbone, pretrained=True).to(device).eval()
    logger.info(f'Backbone={args.backbone}, backend={getattr(model, "backend", None)}, out_dim={getattr(model, "out_dim", None)}')

    manifest = []
    failures = []
    with torch.no_grad():
        for item in tqdm(items, desc=f'Extract NASNet-GMMTop{args.top_k}'):
            rel = item['rel_path']
            out_path = out_root / item['split'] / item['class_name'] / f'{Path(rel).stem}.pt'
            if args.skip_existing and out_path.exists() and out_path.stat().st_size > 0:
                continue
            try:
                frames, sel_idx, motion, bounds, pad_mask, fps, total, width, height = gmm_select_global_topk_frames(
                    Path(item['abs_path']),
                    top_k=args.top_k,
                    image_size=args.image_size,
                    frame_stride=args.frame_stride,
                    warmup_frames=args.warmup_frames,
                    alpha_min=args.alpha_min,
                    alpha_max=args.alpha_max,
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
                if item['split'] == 'test':
                    gt_tokens = token_labels_from_ranges(bounds, ranges)
                else:
                    gt_tokens = np.full((args.top_k,), -1, dtype=np.int64)

                out_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = out_path.with_suffix('.tmp')
                torch.save({
                    'features': features,
                    'selected_indices': torch.as_tensor(sel_idx, dtype=torch.long),
                    'segment_bounds': torch.as_tensor(bounds, dtype=torch.long),
                    'token_bounds': torch.as_tensor(bounds, dtype=torch.long),
                    'motion_scores': torch.as_tensor(motion, dtype=torch.float32),
                    'selected_padding_mask': torch.as_tensor(pad_mask, dtype=torch.bool),
                    'video_label': int(item['video_label']),
                    'class_name': item['class_name'],
                    'rel_path': rel,
                    'fps': float(fps),
                    'total_frames': int(total),
                    'width': int(width),
                    'height': int(height),
                    'gt_segment_labels': torch.as_tensor(gt_tokens, dtype=torch.long),
                    'annotation_ranges': ranges,
                    'backbone': args.backbone,
                    'feature_shape': list(features.shape),
                    'selection_mode': 'gmm_global_topk_paper',
                    'top_k': int(args.top_k),
                    'frame_stride': int(args.frame_stride),
                    'warmup_frames': int(args.warmup_frames),
                    'alpha_min': args.alpha_min,
                    'alpha_max': args.alpha_max,
                }, tmp)
                tmp.replace(out_path)
                manifest.append({
                    'path': str(out_path),
                    'split': item['split'],
                    'rel_path': rel,
                    'class_name': item['class_name'],
                    'video_label': item['video_label'],
                    'feature_shape': list(features.shape),
                    'selection_mode': 'gmm_global_topk_paper',
                    'top_k': int(args.top_k),
                })
            except Exception as e:
                failures.append({'rel_path': rel, 'error': repr(e)})
                logger.error(f'FAILED | {rel} | {repr(e)}')

    save_jsonl(out_root / 'annotations' / 'manifest.jsonl', manifest)
    save_jsonl(out_root / 'annotations' / 'failures.jsonl', failures)
    logger.info(f'DONE. saved={len(manifest)} failures={len(failures)}')


if __name__ == '__main__':
    main()
