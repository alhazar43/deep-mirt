"""Loss functions for Deep-GPCM training.

Kept from original deep-gpcm (pruned):
    FocalLoss            — class-imbalance handling via (1-p_t)^γ modulation
    QWKLoss              — ordinal agreement (1 - QWK), pure-torch CM
    WeightedOrdinalLoss  — weighted CE + ordinal distance penalty
    CombinedLoss         — linear combination of the above three

Dropped:
    OrdinalCrossEntropyLoss, EducationalOrdinalLoss, CORALLoss,
    create_loss_function factory, compute_educational_class_weights,
    analyze_class_distribution.

All loss classes accept pre-flattened, pre-masked logits + targets
(shape (N, K) and (N,) respectively).  The trainer is responsible for
flattening (B, S, K) → (N, K) and masking out padding tokens before
calling these.

Default loss recipe (Architecture Decision A6):
    L = 0.5 * FocalLoss(logits, targets)
      + 0.5 * WeightedOrdinalLoss(logits, targets)

``WeightedOrdinalLoss`` internally uses ``ordinal_penalty=0.5``.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
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
        strategy: ``"balanced"`` — inverse frequency;
                  ``"sqrt_balanced"`` — square-root of inverse frequency
                  (gentler, recommended for ordinal data);
                  anything else → uniform weights.
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
# FocalLoss
# ---------------------------------------------------------------------------


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017) for ordinal class imbalance.

    L_focal = α (1 − p_t)^γ · CE(logits, targets)

    Args:
        alpha: Overall scale factor (default 1.0).
        gamma: Focusing parameter (default 2.0).
        reduction: ``"mean"`` or ``"sum"``.

    Inputs:
        logits:  ``(N, K)`` unnormalised logits.
        targets: ``(N,)``  integer class labels in [0, K-1].
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1.0 - p_t) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ---------------------------------------------------------------------------
# QWKLoss
# ---------------------------------------------------------------------------


class QWKLoss(nn.Module):
    """Quadratic Weighted Kappa loss: minimise 1 - QWK.

    QWK is computed from a vectorised confusion matrix built entirely in
    PyTorch, so no numpy / sklearn dependency.

    Args:
        n_categories: Number of ordinal categories K.

    Inputs:
        logits:  ``(N, K)`` — predictions via argmax.
        targets: ``(N,)``  — integer ground-truth in [0, K-1].
    """

    def __init__(self, n_categories: int) -> None:
        super().__init__()
        self.n_categories = n_categories

        # Pre-compute quadratic weight matrix W[i,j] = 1 - (i-j)^2 / (K-1)^2
        i_grid, j_grid = torch.meshgrid(
            torch.arange(n_categories, dtype=torch.float32),
            torch.arange(n_categories, dtype=torch.float32),
            indexing="ij",
        )
        qwk_weights = 1.0 - (i_grid - j_grid) ** 2 / (n_categories - 1) ** 2
        self.register_buffer("qwk_weights", qwk_weights)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        preds = logits.argmax(dim=-1)
        qwk = self._compute_qwk(targets, preds)
        return 1.0 - qwk

    def _compute_qwk(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        n = y_true.numel()
        if n == 0:
            return torch.tensor(0.0, device=y_true.device)

        # Vectorised confusion matrix via bincount
        indices = y_true * self.n_categories + y_pred
        cm = torch.bincount(indices, minlength=self.n_categories ** 2)
        cm = cm.view(self.n_categories, self.n_categories).float() / n

        # Marginals and expected matrix
        marginal_true = cm.sum(dim=1)
        marginal_pred = cm.sum(dim=0)
        expected = torch.outer(marginal_true, marginal_pred)

        eps = 1e-7
        qwk_w = self.qwk_weights.to(cm.device)
        Po = (qwk_w * cm).sum()
        Pe = torch.clamp((qwk_w * expected).sum(), 0.0, 1.0 - eps)

        return torch.clamp((Po - Pe) / (1.0 - Pe + eps), -1.0, 1.0)


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
    """Linear combination of up to three loss components.

    Mirrors the validated Deep-GPCM recipe:
        L = focal_weight * FocalLoss
          + weighted_ordinal_weight * WeightedOrdinalLoss(ordinal_penalty)

    The QWK component is off by default (``qwk_weight=0.0``) because QWK
    requires hard argmax predictions, introducing a gradient discontinuity.
    Include it only if using a straight-through estimator.

    Args:
        n_categories: K.
        class_weights: Optional per-class weights for ``WeightedOrdinalLoss``.
        focal_weight: Weight of ``FocalLoss`` component.
        weighted_ordinal_weight: Weight of ``WeightedOrdinalLoss`` component.
        ordinal_penalty: Internal distance penalty inside ``WeightedOrdinalLoss``.
        qwk_weight: Weight of ``QWKLoss`` component (default 0.0).

    Inputs — expected by the trainer (already flattened + masked):
        logits:  ``(N, K)``
        targets: ``(N,)``
    """

    def __init__(
        self,
        n_categories: int,
        class_weights: Optional[Tensor] = None,
        focal_weight: float = 0.5,
        weighted_ordinal_weight: float = 0.5,
        ordinal_penalty: float = 0.5,
        qwk_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.focal_weight = focal_weight
        self.weighted_ordinal_weight = weighted_ordinal_weight
        self.qwk_weight = qwk_weight

        if focal_weight > 0.0:
            self.focal_loss = FocalLoss()

        if weighted_ordinal_weight > 0.0:
            self.weighted_ordinal_loss = WeightedOrdinalLoss(
                n_categories,
                class_weights=class_weights,
                ordinal_penalty=ordinal_penalty,
            )

        if qwk_weight > 0.0:
            self.qwk_loss = QWKLoss(n_categories)

        # Ensure at least CE fallback
        _total = focal_weight + weighted_ordinal_weight + qwk_weight
        if _total == 0.0:
            self.ce_fallback = nn.CrossEntropyLoss()

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        total: Tensor = torch.zeros(1, device=logits.device, dtype=logits.dtype).squeeze()

        if self.focal_weight > 0.0:
            total = total + self.focal_weight * self.focal_loss(logits, targets)

        if self.weighted_ordinal_weight > 0.0:
            total = total + self.weighted_ordinal_weight * self.weighted_ordinal_loss(
                logits, targets
            )

        if self.qwk_weight > 0.0:
            total = total + self.qwk_weight * self.qwk_loss(logits, targets)

        if not (self.focal_weight + self.weighted_ordinal_weight + self.qwk_weight):
            total = self.ce_fallback(logits, targets)

        return total
