from __future__ import annotations

import json
from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class Phase2ClipDataset(Dataset):
    def __init__(self, manifest: str | Path, split: str = 'train', clip_len: int = 16, image_size: int = 224, train: bool = True):
        self.rows = []
        with open(manifest, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get('split') == split:
                    self.rows.append(r)
        if not self.rows:
            raise FileNotFoundError(f'No rows for split={split} in {manifest}')
        self.clip_len = clip_len
        self.image_size = image_size
        aug = []
        if train:
            aug += [transforms.RandomHorizontalFlip(p=0.5)]
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            *aug,
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def __len__(self):
        return len(self.rows)

    def _read_mp4(self, path):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release(); raise RuntimeError(f'Cannot open clip: {path}')
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            frames = [np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)]
        if len(frames) >= self.clip_len:
            idxs = np.linspace(0, len(frames)-1, self.clip_len).round().astype(int)
            frames = [frames[i] for i in idxs]
        else:
            while len(frames) < self.clip_len:
                frames.append(frames[-1].copy())
        return frames[:self.clip_len]

    def __getitem__(self, idx):
        r = self.rows[idx]
        frames = self._read_mp4(r['clip_path'])
        x = torch.stack([self.transform(f) for f in frames], dim=0)  # [T,C,H,W]
        return {'clip': x, 'label': torch.tensor(int(r['class_id']), dtype=torch.long), 'clip_path': r['clip_path'], 'class_name': r['class_name']}
