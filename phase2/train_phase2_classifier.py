#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from phase2.datasets import Phase2ClipDataset
from phase2.models import build_phase2_model
from utils.logging_utils import setup_logger
from utils.metrics import classification_report_dict


def collate(batch):
    return {
        'clip': torch.stack([b['clip'] for b in batch]),
        'label': torch.stack([b['label'] for b in batch]),
        'clip_path': [b['clip_path'] for b in batch],
        'class_name': [b['class_name'] for b in batch],
    }


def evaluate(model, loader, device):
    model.eval(); ys=[]; preds=[]; losses=[]
    ce = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            x = batch['clip'].to(device); y = batch['label'].to(device)
            logits = model(x)
            loss = ce(logits, y)
            losses.append(float(loss.cpu()))
            ys.extend(y.cpu().tolist())
            preds.extend(logits.argmax(dim=1).cpu().tolist())
    m = classification_report_dict(ys, preds)
    m['loss'] = float(np.mean(losses)) if losses else float('nan')
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--model', default='resnet50')
    ap.add_argument('--num-classes', type=int, default=13)
    ap.add_argument('--clip-len', type=int, default=16)
    ap.add_argument('--image-size', type=int, default=224)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--pretrained', action='store_true')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--num-workers', type=int, default=4)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger('train_phase2', out_dir / 'train.log')
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    logger.info(vars(args))

    train_ds = Phase2ClipDataset(args.manifest, split='train', clip_len=args.clip_len, image_size=args.image_size, train=True)
    val_ds = Phase2ClipDataset(args.manifest, split='val', clip_len=args.clip_len, image_size=args.image_size, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate, pin_memory=True)

    model = build_phase2_model(args.model, num_classes=args.num_classes, pretrained=args.pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce = torch.nn.CrossEntropyLoss()
    best = -1.0
    history = []
    for epoch in range(args.epochs):
        model.train(); losses=[]
        for batch in tqdm(train_loader, desc=f'Phase2 {args.model} epoch {epoch+1}/{args.epochs}'):
            x = batch['clip'].to(device); y = batch['label'].to(device)
            logits = model(x)
            loss = ce(logits, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(float(loss.detach().cpu()))
        metrics = evaluate(model, val_loader, device)
        row = {'epoch': epoch+1, 'train_loss': float(np.mean(losses)), **metrics}
        history.append(row)
        logger.info(json.dumps(row, ensure_ascii=False))
        with open(out_dir / 'history.json', 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
        if metrics['macro_f1'] > best:
            best = metrics['macro_f1']
            torch.save({'model_state': model.state_dict(), 'model': args.model, 'args': vars(args), 'best_macro_f1': best}, out_dir / 'best.pth')
            logger.info(f'Saved best macro_f1={best:.6f}')
    torch.save({'model_state': model.state_dict(), 'model': args.model, 'args': vars(args)}, out_dir / 'last.pth')


if __name__ == '__main__':
    main()
