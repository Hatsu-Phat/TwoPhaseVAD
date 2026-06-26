from __future__ import annotations

import torch
import torch.nn.functional as F


def topk_video_logits(segment_logits: torch.Tensor, topk: int = 5):
    k = min(topk, segment_logits.shape[1])
    return torch.topk(segment_logits, k=k, dim=1).values.mean(dim=1)


def phase1_loss(outputs, labels, topk=5, smooth_weight=0.1, sparsity_weight=0.001, ranking_weight=0.5):
    seg_logits = outputs['segment_logits']
    model_video_logits = outputs['video_logits']
    mil_video_logits = topk_video_logits(seg_logits, topk=topk)
    video_logits = 0.5 * (model_video_logits + mil_video_logits)

    bce = F.binary_cross_entropy_with_logits(video_logits, labels)
    scores = torch.sigmoid(seg_logits)
    smooth = (scores[:, 1:] - scores[:, :-1]).pow(2).mean()
    sparsity = scores.mean()

    ranking = torch.tensor(0.0, device=seg_logits.device)
    anom = labels > 0.5
    norm = labels <= 0.5
    if anom.any() and norm.any():
        a = torch.sigmoid(mil_video_logits[anom])
        n = torch.sigmoid(mil_video_logits[norm])
        ranking = F.relu(1.0 - a[:, None] + n[None, :]).mean()

    total = bce + smooth_weight * smooth + sparsity_weight * sparsity + ranking_weight * ranking
    return total, {'loss': float(total.detach().cpu()), 'bce': float(bce.detach().cpu()), 'smooth': float(smooth.detach().cpu()), 'sparsity': float(sparsity.detach().cpu()), 'ranking': float(ranking.detach().cpu())}
