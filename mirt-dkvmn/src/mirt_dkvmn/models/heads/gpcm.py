"""GPCM head placeholder."""

import torch
import torch.nn as nn


class GPCMHead(nn.Module):
    """Wraps logits to probabilities for ordinal categories."""

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1)
