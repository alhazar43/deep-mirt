"""Loss composition for ordinal MIRT models."""

import torch
import torch.nn as nn


class OrdinalCrossEntropy(nn.Module):
    """Placeholder ordinal loss operating on logits."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
