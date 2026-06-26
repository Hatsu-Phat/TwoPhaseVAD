from __future__ import annotations

import torch
from torch import nn


class LSTMAnomalyModel(nn.Module):
    def __init__(self, input_dim=1056, seq_len=32, hidden_dim=256, num_layers=2, dropout=0.3, topk=5, **kwargs):
        super().__init__()
        self.topk = topk
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0, bidirectional=True)
        out_dim = hidden_dim * 2
        self.norm = nn.LayerNorm(out_dim)
        self.segment_head = nn.Sequential(nn.Linear(out_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.video_head = nn.Sequential(nn.Linear(out_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        z, _ = self.lstm(x)
        z = self.norm(z)
        segment_logits = self.segment_head(z).squeeze(-1)
        pooled = z.mean(dim=1)
        video_logits = self.video_head(pooled).squeeze(-1)
        return {'video_logits': video_logits, 'segment_logits': segment_logits, 'features': z}
