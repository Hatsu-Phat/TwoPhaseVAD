from __future__ import annotations

import math
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / rms


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, dim: int):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        z = self.norm1(x)
        attn_out, _ = self.attn(z, z, z, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class QueryAdaptivePooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) * 0.02)

    def forward(self, x):
        logits = torch.matmul(x, self.query)
        weights = torch.softmax(logits, dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class PaperEViT(nn.Module):
    def __init__(
        self,
        input_dim: int = 1056,
        seq_len: int = 30,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()
        self.seq_len = seq_len

        self.embed = nn.Linear(input_dim, hidden_dim)
        self.embed_norm = RMSNorm(hidden_dim)
        self.pos = SinusoidalPositionalEncoding(seq_len, hidden_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(hidden_dim)
        self.qap = QueryAdaptivePooling(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.embed_norm(self.embed(x))
        x = self.pos(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.final_norm(x)
        pooled, qap_weights = self.qap(x)

        video_logits = self.classifier(pooled)
        video_prob = torch.softmax(video_logits, dim=1)
        video_scores = video_prob[:, 1]

        token_scores = qap_weights * video_scores.unsqueeze(1)

        return {
            "video_logits": video_logits,
            "video_scores": video_scores,
            "segment_scores": token_scores,
            "qap_weights": qap_weights,
        }
