"""Prediction losses for deep_irt training.

Copied verbatim from ``ma-irt/utils/losses.py`` (the frozen Chapter 0
reference) so deep_irt stays self-contained and does not import across the
ma-irt boundary.  ma-irt is the canonical source; if the two ever diverge,
ma-irt wins.

Design principle (the reason this module exists separately from the
decoders): IRT is a READOUT FLAVOR, not the training objective.  The decoder
emits per-category ``logits``; this module scores ``(logits, targets)`` with a
generic prediction loss.  The training loop never optimizes an IRT-model
likelihood directly.

Loss map used by ``DeepIRTModel`` (set by response format, not by IRT readout):
    ordinal  (GPCM)  -> WeightedOrdinalLoss  (CE on logits * ordinal-distance
                        penalty + sqrt-balanced class weights), exactly ma-irt's
                        recipe: weighted_ordinal_weight=1.0, ordinal_penalty=0.5,
                        every other ma-irt term 0.
    nominal  (NRM)   -> plain CrossEntropy   (CombinedLoss with
                        weighted_ordinal_weight=0 -> ce_fallback).  No ordinal
                        penalty: the categories carry no order.
    binary           -> BCE on the single binary logit (handled in the model,
                        not here).

All loss classes accept pre-flattened, pre-masked logits + targets
(shape ``(N, K)`` and ``(N,)``).  The caller flattens ``(B, S, K) -> (N, K)``
and masks out padding before calling these.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Utility: class weight computation
# ---------------------------------------------------------------------------


def compute_class_weights(
    targets: Tensor,
    n_classes: int,
    strategy: str = "sqrt_balanced",
    device: Optional[torch.device] = None,
) -> Tensor:
    """Compute class weights from a flat target tensor.

    Args:
        targets: 1-D integer tensor of class labels.
        n_classes: Total number of classes.
        strategy: ``"balanced"`` -- inverse frequency;
                  ``"sqrt_balanced"`` -- square-root of inverse frequency
                  (gentler, recommended for ordinal data);
                  anything else -> uniform weights.
        device: Target device.  Defaults to ``targets.device``.

    Returns:
        Float tensor of shape ``(n_classes,)``.
    """
    if device is None:
        device = targets.device

    targets_flat = targets.view(-1)
    weights_list = []
    total = targets_flat.numel()

    for i in range(n_classes):
        count = max((targets_flat == i).sum().item(), 1)
        if strategy == "balanced":
            w = total / (n_classes * count)
        elif strategy == "sqrt_balanced":
            w = math.sqrt(total / (n_classes * count))
        else:
            w = 1.0
        weights_list.append(w)

    return torch.tensor(weights_list, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# WeightedOrdinalLoss
# ---------------------------------------------------------------------------


class WeightedOrdinalLoss(nn.Module):
    """Weighted cross-entropy with ordinal distance penalty.

    For each sample the CE loss is multiplied by
        1 + ordinal_penalty * |argmax(logits) - target|

    so that predictions far from the true category are penalised more
    heavily.  Class weights handle label-frequency imbalance.

    Args:
        n_categories: K.
        class_weights: Optional ``(K,)`` tensor of per-class weights.
        ordinal_penalty: Multiplier on the ordinal distance weighting.
        reduction: ``"mean"`` or ``"sum"``.

    Inputs:
        logits:  ``(N, K)``.
        targets: ``(N,)``.
    """

    def __init__(
        self,
        n_categories: int,
        class_weights: Optional[Tensor] = None,
        ordinal_penalty: float = 0.5,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.n_categories = n_categories
        self.ordinal_penalty = ordinal_penalty
        self.reduction = reduction

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.register_buffer("class_weights", torch.ones(n_categories))

        # Pre-compute ordinal distance matrix |i - j|
        idx = torch.arange(n_categories, dtype=torch.float32)
        self.register_buffer(
            "ordinal_dist", torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
        )

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        cw = self.class_weights.to(logits.device)
        od = self.ordinal_dist.to(logits.device)

        ce = F.cross_entropy(logits, targets, weight=cw, reduction="none")  # (N,)

        if self.ordinal_penalty > 0.0:
            with torch.no_grad():
                pred_cats = logits.argmax(dim=-1)
                dist = od[targets, pred_cats]            # (N,)
                scale = 1.0 + self.ordinal_penalty * dist
            ce = ce * scale

        if self.reduction == "mean":
            return ce.mean()
        return ce.sum()


# ---------------------------------------------------------------------------
# CombinedLoss
# ---------------------------------------------------------------------------


class CombinedLoss(nn.Module):
    """Weighted ordinal loss wrapper.

    Default recipe:
        L = weighted_ordinal_weight * WeightedOrdinalLoss(ordinal_penalty)

    With ``weighted_ordinal_weight == 0`` it collapses to plain
    ``CrossEntropyLoss`` (the ce_fallback).  This is exactly the switch
    deep_irt uses to pick the NOMINAL loss (NRM): set the weight to 0 and the
    ordinal penalty disappears, leaving order-free cross-entropy.

    Args:
        n_categories: K.
        class_weights: Optional per-class weights for ``WeightedOrdinalLoss``.
        weighted_ordinal_weight: Weight of ``WeightedOrdinalLoss`` component.
        ordinal_penalty: Internal distance penalty inside ``WeightedOrdinalLoss``.

    Inputs (already flattened + masked):
        logits:  ``(N, K)``
        targets: ``(N,)``
    """

    def __init__(
        self,
        n_categories: int,
        class_weights: Optional[Tensor] = None,
        weighted_ordinal_weight: float = 1.0,
        ordinal_penalty: float = 0.5,
    ) -> None:
        super().__init__()
        self.weighted_ordinal_weight = weighted_ordinal_weight

        if weighted_ordinal_weight > 0.0:
            self.weighted_ordinal_loss = WeightedOrdinalLoss(
                n_categories,
                class_weights=class_weights,
                ordinal_penalty=ordinal_penalty,
            )

        # Ensure at least a CE fallback (the NOMINAL path).
        if weighted_ordinal_weight == 0.0:
            self.ce_fallback = nn.CrossEntropyLoss()

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        if self.weighted_ordinal_weight > 0.0:
            return self.weighted_ordinal_weight * self.weighted_ordinal_loss(
                logits, targets
            )

        return self.ce_fallback(logits, targets)
