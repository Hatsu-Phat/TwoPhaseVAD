from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=pad, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.drop(F.relu(self.norm1(self.conv1(x))))
        h = self.drop(F.relu(self.norm2(self.conv2(h))))
        return x + h


class TCNAnomalyModel(nn.Module):
    def __init__(self, input_dim=1056, seq_len=32, hidden_dim=256, num_layers=4, dropout=0.3, topk=5, **kwargs):
        super().__init__()
        self.topk = topk
        self.proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList([TCNBlock(hidden_dim, dilation=2 ** i, dropout=dropout) for i in range(num_layers)])
        self.segment_head = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.video_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        z = x.transpose(1, 2)
        z = self.proj(z)
        for blk in self.blocks:
            z = blk(z)
        segment_logits = self.segment_head(z).squeeze(1)
        pooled = z.mean(dim=2)
        video_logits = self.video_head(pooled).squeeze(-1)
        return {'video_logits': video_logits, 'segment_logits': segment_logits, 'features': z.transpose(1, 2)}
