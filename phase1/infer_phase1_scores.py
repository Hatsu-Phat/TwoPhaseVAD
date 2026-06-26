#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from phase1.datasets import Phase1FeatureDataset
from phase1.models import build_phase1_model
from phase1.losses import topk_video_logits
from utils.logging_utils import setup_logger


def collate(batch):
    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features-root', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--split', choices=['train', 'test', 'all'], default='all')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger('infer_phase1', out_dir / 'infer.log')
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model = build_phase1_model(ckpt['model'], **ckpt['model_kwargs']).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    topk = ckpt.get('model_kwargs', {}).get('topk', 5)

    splits = ['train', 'test'] if args.split == 'all' else [args.split]
    with torch.no_grad():
        for split in splits:
            ds = Phase1FeatureDataset(args.features_root, split=split)
            loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)
            for items in tqdm(loader, desc=f'Infer {split}'):
                item = items[0]
                x = item['features'].unsqueeze(0).to(device)
                out = model(x)
                seg_logits = out['segment_logits'][0]
                seg_scores = torch.sigmoid(seg_logits).detach().cpu()
                v_logits = 0.5 * (out['video_logits'] + topk_video_logits(out['segment_logits'], topk=topk))
                v_score = float(torch.sigmoid(v_logits)[0].detach().cpu())
                class_name = item['class_name'] or 'Unknown'
                rel_stem = Path(item['rel_path']).stem if item['rel_path'] else Path(item['path']).stem
                save_path = out_dir / split / class_name / f'{rel_stem}.pt'
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'rel_path': item['rel_path'],
                    'class_name': class_name,
                    'video_label': int(item['video_label'].item()),
                    'segment_scores': seg_scores,
                    'video_score': v_score,
                    'selected_indices': item['selected_indices'],
                    'segment_bounds': item['segment_bounds'],
                    'fps': item['fps'],
                    'total_frames': item['total_frames'],
                    'feature_file': item['path'],
                    'checkpoint': args.checkpoint,
                }, save_path)
    logger.info(f'DONE. scores saved to {out_dir}')


if __name__ == '__main__':
    main()
