"""Graded Response Model (GRM) head."""

import torch
import torch.nn as nn


class GRMHead(nn.Module):
    """Compute GRM probabilities from MIRT parameters."""

    def forward(self, theta: torch.Tensor, alpha: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = theta.shape
        n_cats = thresholds.shape[-1] + 1

        dot = torch.sum(theta * alpha, dim=-1)
        cum_probs = []
        for k in range(n_cats - 1):
            cum = torch.sigmoid(dot - thresholds[:, :, k])
            cum_probs.append(cum)

        probs = torch.zeros(batch, seq, n_cats, device=theta.device)
        probs[:, :, 0] = 1.0 - cum_probs[0]
        for k in range(1, n_cats - 1):
            probs[:, :, k] = cum_probs[k - 1] - cum_probs[k]
        probs[:, :, n_cats - 1] = cum_probs[-1]
        return probs
