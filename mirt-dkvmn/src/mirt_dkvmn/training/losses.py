"""Loss composition for ordinal MIRT models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalCrossEntropy(nn.Module):
    """Ordinal loss operating on logits."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


class CombinedOrdinalLoss(nn.Module):
    """Cross-entropy + differentiable QWK + ordinal MAE."""

    def __init__(self, qwk_weight: float = 0.5, ordinal_weight: float = 0.2) -> None:
        super().__init__()
        self.qwk_weight = qwk_weight
        self.ordinal_weight = ordinal_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        probs = F.softmax(logits, dim=-1)
        qwk_loss = 1.0 - soft_qwk(probs, targets)
        ordinal = ordinal_mae(probs, targets)
        return ce + self.qwk_weight * qwk_loss + self.ordinal_weight * ordinal


def ordinal_mae(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    n_cats = probs.size(-1)
    indices = torch.arange(n_cats, device=probs.device).float()
    expected = torch.sum(probs * indices, dim=-1)
    return torch.mean(torch.abs(expected - targets.float()))


def soft_qwk(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    n_cats = probs.size(-1)
    targets_onehot = F.one_hot(targets, num_classes=n_cats).float()

    conf_mat = torch.einsum("btk,btj->kj", targets_onehot, probs)
    total = conf_mat.sum()
    if total == 0:
        return torch.tensor(0.0, device=probs.device)

    hist_true = conf_mat.sum(dim=1)
    hist_pred = conf_mat.sum(dim=0)
    expected = torch.outer(hist_true, hist_pred) / total

    weights = torch.zeros((n_cats, n_cats), device=probs.device)
    for i in range(n_cats):
        for j in range(n_cats):
            weights[i, j] = ((i - j) ** 2) / ((n_cats - 1) ** 2)

    observed = (weights * conf_mat).sum() / total
    expected_score = (weights * expected).sum() / total
    if expected_score == 0:
        return torch.tensor(0.0, device=probs.device)
    return 1.0 - observed / expected_score
