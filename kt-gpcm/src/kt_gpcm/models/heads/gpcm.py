"""GPCM probability head.

``GPCMHead`` is intentionally trivial: it applies softmax along the
category dimension.  Separating it from ``GPCMLogits`` lets the model
return *both* logits (for loss computation) and probabilities (for
metrics and interpretation) without any flag-based dispatch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GPCMHead(nn.Module):
    """Convert GPCM logits to category probabilities.

    Args:
        None — this module is stateless.

    Shape:
        - Input  ``logits``: ``(B, S, K)``
        - Output ``probs``:  ``(B, S, K)``  (sums to 1 over K)
    """

    def forward(self, logits: Tensor) -> Tensor:
        """Apply softmax over the category dimension.

        Args:
            logits: Unnormalised GPCM logits ``(B, S, K)``.

        Returns:
            Category probabilities ``(B, S, K)``.
        """
        return F.softmax(logits, dim=-1)
