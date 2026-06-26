from __future__ import annotations

import math
import torch
from torch import nn


def posenc(seq_len: int, dim: int, device):
    pe = torch.zeros(seq_len, dim, device=device)
    pos = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
    return pe.unsqueeze(0)


class TransformerAnomalyModel(nn.Module):
    def __init__(self, input_dim=1056, seq_len=32, hidden_dim=256, num_layers=4, num_heads=4, dropout=0.3, topk=5, **kwargs):
        super().__init__()
        self.topk = topk
        self.proj = nn.Linear(input_dim, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4, dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.segment_head = nn.Linear(hidden_dim, 1)
        self.video_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        z = self.proj(x)
        z = z + posenc(z.shape[1], z.shape[2], z.device)
        z = self.norm(self.encoder(z))
        segment_logits = self.segment_head(z).squeeze(-1)
        video_logits = self.video_head(z.mean(dim=1)).squeeze(-1)
        return {'video_logits': video_logits, 'segment_logits': segment_logits, 'features': z}
