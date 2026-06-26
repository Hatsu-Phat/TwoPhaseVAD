from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class EViTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = RMSNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


def sinusoidal_position_encoding(seq_len: int, dim: int, device):
    pe = torch.zeros(seq_len, dim, device=device)
    pos = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
    return pe.unsqueeze(0)


class EViTAnomalyModel(nn.Module):
    def __init__(self, input_dim=1056, seq_len=32, hidden_dim=256, num_layers=4, num_heads=4, dropout=0.3, topk=5, **kwargs):
        super().__init__()
        self.seq_len = seq_len
        self.topk = topk
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.in_norm = RMSNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([EViTBlock(hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.final_norm = RMSNorm(hidden_dim)
        self.query = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.segment_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))
        self.video_head = nn.Sequential(nn.Linear(hidden_dim, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(dropout), nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(dropout), nn.Linear(256, 1))

    def forward(self, x):
        # x: [B, T, D]
        z = self.in_norm(self.proj(x))
        z = z + sinusoidal_position_encoding(z.shape[1], z.shape[2], z.device)
        z = self.drop(z)
        for blk in self.blocks:
            z = blk(z)
        z = self.final_norm(z)
        segment_logits = self.segment_head(z).squeeze(-1)
        attn_logits = torch.matmul(z, self.query)
        attn = F.softmax(attn_logits, dim=1)
        pooled = torch.sum(z * attn.unsqueeze(-1), dim=1)
        video_logits = self.video_head(pooled).squeeze(-1)
        return {'video_logits': video_logits, 'segment_logits': segment_logits, 'features': z}
