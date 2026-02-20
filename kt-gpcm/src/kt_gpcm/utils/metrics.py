"""Pure-PyTorch evaluation metrics for ordinal prediction.

No sklearn, no scipy.  All functions operate on GPU tensors and handle
edge cases (empty mask, all-same-class) gracefully.

Primary entry point: :func:`compute_metrics`.

Metrics returned
----------------
categorical_accuracy  exact-match accuracy
ordinal_accuracy      fraction of predictions within ±1 category
qwk                   quadratic weighted kappa ∈ [-1, 1]
mae                   mean absolute error (treating categories as ordinal)
spearman              Spearman rank correlation via argsort-of-argsort
confusion_matrix      (K, K) integer tensor, rows=true, cols=pred
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rankdata(x: Tensor) -> Tensor:
    """Rank a 1-D tensor (average-rank ties via argsort-of-argsort)."""
    order = x.argsort()
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(len(x), dtype=torch.float32, device=x.device)
    return ranks + 1.0   # 1-based


def _spearman(pred: Tensor, target: Tensor) -> float:
    """Spearman ρ between two 1-D float tensors."""
    n = pred.numel()
    if n < 2:
        return 0.0
    rp = _rankdata(pred.float())
    rt = _rankdata(target.float())
    # Pearson on ranks
    rp_c = rp - rp.mean()
    rt_c = rt - rt.mean()
    denom = (rp_c.norm() * rt_c.norm()).clamp(min=1e-8)
    rho = (rp_c * rt_c).sum() / denom
    return float(rho.item())


def _qwk(pred: Tensor, target: Tensor, n_categories: int) -> float:
    """Quadratic weighted kappa from hard predictions."""
    n = pred.numel()
    if n == 0:
        return 0.0

    device = pred.device

    # Confusion matrix
    idx = target * n_categories + pred
    cm = torch.bincount(idx, minlength=n_categories ** 2)
    cm = cm.view(n_categories, n_categories).float() / n

    # QWK weight matrix W[i,j] = 1 - (i-j)^2 / (K-1)^2
    i_g, j_g = torch.meshgrid(
        torch.arange(n_categories, dtype=torch.float32, device=device),
        torch.arange(n_categories, dtype=torch.float32, device=device),
        indexing="ij",
    )
    W = 1.0 - (i_g - j_g) ** 2 / (n_categories - 1) ** 2

    marginal_true = cm.sum(dim=1)
    marginal_pred = cm.sum(dim=0)
    expected = torch.outer(marginal_true, marginal_pred)

    eps = 1e-7
    Po = (W * cm).sum()
    Pe = torch.clamp((W * expected).sum(), 0.0, 1.0 - eps)
    qwk = (Po - Pe) / (1.0 - Pe + eps)
    return float(qwk.clamp(-1.0, 1.0).item())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_metrics(
    probs: Tensor,
    targets: Tensor,
    mask: Optional[Tensor] = None,
) -> dict:
    """Compute evaluation metrics for ordinal predictions.

    Args:
        probs: ``(B, S, K)`` predicted category probabilities.
        targets: ``(B, S)`` integer ground-truth in [0, K-1].
        mask: ``(B, S)`` boolean mask where ``True`` = valid position.
              Padding positions (``False``) are excluded from all metrics.
              If ``None``, all positions are treated as valid.

    Returns:
        Dict with keys:
            ``categorical_accuracy`` float
            ``ordinal_accuracy``     float (within ±1)
            ``qwk``                  float
            ``mae``                  float
            ``spearman``             float
            ``confusion_matrix``     Tensor(K, K)
    """
    B, S, K = probs.shape
    device = probs.device

    # ---- Flatten -----------------------------------------------------------
    probs_flat = probs.view(-1, K)      # (N, K)
    targets_flat = targets.view(-1)     # (N,)

    # ---- Apply mask --------------------------------------------------------
    if mask is not None:
        valid = mask.view(-1).bool()
    else:
        valid = torch.ones(B * S, dtype=torch.bool, device=device)

    if valid.sum() == 0:
        # Edge case: entirely padded batch
        return {
            "categorical_accuracy": 0.0,
            "ordinal_accuracy": 0.0,
            "qwk": 0.0,
            "mae": 0.0,
            "spearman": 0.0,
            "confusion_matrix": torch.zeros(K, K, dtype=torch.long, device=device),
        }

    probs_v = probs_flat[valid]     # (V, K)
    targets_v = targets_flat[valid] # (V,)
    preds_v = probs_v.argmax(dim=-1)  # (V,)

    # ---- Metrics -----------------------------------------------------------
    # Categorical accuracy
    cat_acc = float((preds_v == targets_v).float().mean().item())

    # Ordinal accuracy (±1)
    ord_acc = float((torch.abs(preds_v - targets_v) <= 1).float().mean().item())

    # MAE
    mae = float(torch.abs(preds_v.float() - targets_v.float()).mean().item())

    # QWK
    qwk = _qwk(preds_v, targets_v, K)

    # Spearman
    spearman = _spearman(preds_v.float(), targets_v.float())

    # Confusion matrix
    idx = targets_v * K + preds_v
    cm = torch.bincount(idx, minlength=K ** 2).view(K, K)

    return {
        "categorical_accuracy": cat_acc,
        "ordinal_accuracy": ord_acc,
        "qwk": qwk,
        "mae": mae,
        "spearman": spearman,
        "confusion_matrix": cm,
    }
