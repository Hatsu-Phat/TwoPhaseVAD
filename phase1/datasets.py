from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset


class Phase1FeatureDataset(Dataset):
    def __init__(self, features_root: str | Path, split: str = 'train'):
        self.root = Path(features_root) / split
        self.split = split
        self.files = sorted(self.root.rglob('*.pt'))
        if not self.files:
            raise FileNotFoundError(f'No .pt feature files under {self.root}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        data = torch.load(p, map_location='cpu')
        features = data['features'].float()
        video_label = int(data.get('video_label', 0))
        gt_seg = data.get('gt_segment_labels', data.get('segment_labels', None))
        if gt_seg is None:
            gt_seg = torch.full((features.shape[0],), -1, dtype=torch.long)
        else:
            gt_seg = torch.as_tensor(gt_seg, dtype=torch.long)
        return {
            'features': features,
            'video_label': torch.tensor(video_label, dtype=torch.float32),
            'gt_segment_labels': gt_seg,
            'path': str(p),
            'rel_path': data.get('rel_path', ''),
            'class_name': data.get('class_name', ''),
            'selected_indices': torch.as_tensor(data.get('selected_indices', []), dtype=torch.long),
            'segment_bounds': torch.as_tensor(data.get('segment_bounds', []), dtype=torch.long),
            'fps': float(data.get('fps', 0.0)),
            'total_frames': int(data.get('total_frames', 0)),
        }
