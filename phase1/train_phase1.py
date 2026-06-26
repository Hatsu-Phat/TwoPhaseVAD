#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))
from phase1.datasets import Phase1FeatureDataset
from phase1.losses import phase1_loss, topk_video_logits
from phase1.models import build_phase1_model
from utils.logging_utils import setup_logger
from utils.metrics import safe_roc_auc, safe_average_precision


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None)
    ap.add_argument('--features-root', default=None)
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--model', default='evit')
    ap.add_argument('--input-dim', type=int, default=1056)
    ap.add_argument('--seq-len', type=int, default=32)
    ap.add_argument('--hidden-dim', type=int, default=256)
    ap.add_argument('--num-layers', type=int, default=4)
    ap.add_argument('--num-heads', type=int, default=4)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--smooth-weight', type=float, default=0.1)
    ap.add_argument('--sparsity-weight', type=float, default=0.001)
    ap.add_argument('--ranking-weight', type=float, default=0.5)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--num-workers', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            setattr(args, k.replace('-', '_'), v)
    return args


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def collate(batch):
    return {
        'features': torch.stack([b['features'] for b in batch]),
        'video_label': torch.stack([b['video_label'] for b in batch]),
        'gt_segment_labels': torch.stack([b['gt_segment_labels'] for b in batch]),
        'path': [b['path'] for b in batch],
        'rel_path': [b['rel_path'] for b in batch],
        'class_name': [b['class_name'] for b in batch],
    }


def evaluate(model, loader, device, topk):
    model.eval()
    ys, scores = [], []
    seg_y, seg_s = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch['features'].to(device)
            y = batch['video_label'].numpy()
            out = model(x)
            v_logits = 0.5 * (out['video_logits'] + topk_video_logits(out['segment_logits'], topk=topk))
            v_score = torch.sigmoid(v_logits).cpu().numpy()
            ys.extend(y.tolist()); scores.extend(v_score.tolist())
            gt = batch['gt_segment_labels'].numpy()
            ss = torch.sigmoid(out['segment_logits']).cpu().numpy()
            mask = gt >= 0
            if mask.any():
                seg_y.extend(gt[mask].tolist())
                seg_s.extend(ss[mask].tolist())
    return {
        'video_auc': safe_roc_auc(ys, scores),
        'video_ap': safe_average_precision(ys, scores),
        'segment_auc': safe_roc_auc(seg_y, seg_s) if len(seg_y) > 0 else float('nan'),
        'segment_ap': safe_average_precision(seg_y, seg_s) if len(seg_y) > 0 else float('nan'),
    }


def main():
    args = parse_args()
    seed_all(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger('train_phase1', out_dir / 'train.log')
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith('cuda') else 'cpu')
    logger.info(vars(args))

    train_ds = Phase1FeatureDataset(args.features_root, split='train')
    test_ds = Phase1FeatureDataset(args.features_root, split='test')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate, pin_memory=True)

    model_kwargs = dict(input_dim=args.input_dim, seq_len=args.seq_len, hidden_dim=args.hidden_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, topk=args.topk)
    model = build_phase1_model(args.model, **model_kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = -1.0
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}'):
            x = batch['features'].to(device)
            y = batch['video_label'].to(device)
            out = model(x)
            loss, parts = phase1_loss(out, y, topk=args.topk, smooth_weight=args.smooth_weight, sparsity_weight=args.sparsity_weight, ranking_weight=args.ranking_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(parts['loss'])
        metrics = evaluate(model, test_loader, device, topk=args.topk)
        score = metrics['segment_auc'] if not np.isnan(metrics['segment_auc']) else metrics['video_auc']
        row = {'epoch': epoch + 1, 'train_loss': float(np.mean(losses)), **metrics}
        history.append(row)
        logger.info(json.dumps(row, ensure_ascii=False))
        with open(out_dir / 'history.json', 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        if score > best:
            best = score
            torch.save({'model_state': model.state_dict(), 'model': args.model, 'model_kwargs': model_kwargs, 'args': vars(args), 'best_score': best, 'epoch': epoch + 1}, out_dir / 'best_auc.pth')
            logger.info(f'Saved best checkpoint: score={best:.6f}')
    torch.save({'model_state': model.state_dict(), 'model': args.model, 'model_kwargs': model_kwargs, 'args': vars(args)}, out_dir / 'last.pth')


if __name__ == '__main__':
    main()
