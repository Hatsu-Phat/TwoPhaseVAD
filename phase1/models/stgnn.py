from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        h = torch.einsum('ij,bjd->bid', adj, x)
        return self.lin(h)


def build_temporal_adj(seq_len: int, k: int = 2, device=None):
    adj = torch.eye(seq_len, device=device)
    for i in range(seq_len):
        for d in range(1, k + 1):
            if i - d >= 0:
                adj[i, i - d] = 1
            if i + d < seq_len:
                adj[i, i + d] = 1
    deg = adj.sum(dim=1, keepdim=True).clamp_min(1)
    return adj / deg


class STGNNAnomalyModel(nn.Module):
    def __init__(self, input_dim=1056, seq_len=32, hidden_dim=256, num_layers=3, dropout=0.3, topk=5, **kwargs):
        super().__init__()
        self.seq_len = seq_len
        self.topk = topk
        layers = []
        dims = [input_dim] + [hidden_dim] * num_layers
        for a, b in zip(dims[:-1], dims[1:]):
            layers.append(GraphConv(a, b))
        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.drop = nn.Dropout(dropout)
        self.segment_head = nn.Linear(hidden_dim, 1)
        self.video_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        adj = build_temporal_adj(x.shape[1], k=2, device=x.device)
        z = x
        for layer, norm in zip(self.layers, self.norms):
            z = self.drop(F.relu(norm(layer(z, adj))))
        segment_logits = self.segment_head(z).squeeze(-1)
        video_logits = self.video_head(z.mean(dim=1)).squeeze(-1)
        return {'video_logits': video_logits, 'segment_logits': segment_logits, 'features': z}
