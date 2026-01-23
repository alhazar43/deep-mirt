"""Evaluation metrics for ordinal models."""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import torch


def _apply_mask(
    preds: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    if mask is None:
        return preds.reshape(-1), targets.reshape(-1)
    flat_mask = mask.reshape(-1)
    return preds.reshape(-1)[flat_mask], targets.reshape(-1)[flat_mask]


def categorical_accuracy(
    preds: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> float:
    preds, targets = _apply_mask(preds, targets, mask)
    if preds.numel() == 0:
        return float("nan")
    return (preds == targets).float().mean().item()


def per_class_accuracy(
    preds: torch.Tensor,
    targets: torch.Tensor,
    n_cats: int,
    mask: Optional[torch.Tensor] = None,
) -> Dict[int, float]:
    preds, targets = _apply_mask(preds, targets, mask)
    results: Dict[int, float] = {}
    for cat in range(n_cats):
        idx = targets == cat
        if idx.sum() == 0:
            results[cat] = float("nan")
        else:
            results[cat] = (preds[idx] == targets[idx]).float().mean().item()
    return results


def mean_absolute_error(
    preds: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> float:
    preds, targets = _apply_mask(preds, targets, mask)
    if preds.numel() == 0:
        return float("nan")
    return torch.abs(preds.float() - targets.float()).mean().item()


def within_one_accuracy(
    preds: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> float:
    preds, targets = _apply_mask(preds, targets, mask)
    if preds.numel() == 0:
        return float("nan")
    return (torch.abs(preds.float() - targets.float()) <= 1).float().mean().item()


def balanced_accuracy(
    preds: torch.Tensor,
    targets: torch.Tensor,
    n_cats: int,
    mask: Optional[torch.Tensor] = None,
) -> float:
    per_class = per_class_accuracy(preds, targets, n_cats, mask)
    values = [v for v in per_class.values() if not np.isnan(v)]
    if not values:
        return float("nan")
    return float(np.mean(values))


def spearman_correlation(
    preds: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> float:
    preds, targets = _apply_mask(preds, targets, mask)
    if preds.numel() == 0:
        return float("nan")
    preds_np = preds.cpu().numpy().astype(float)
    targets_np = targets.cpu().numpy().astype(float)
    pred_ranks = np.argsort(np.argsort(preds_np))
    target_ranks = np.argsort(np.argsort(targets_np))
    if np.std(pred_ranks) == 0 or np.std(target_ranks) == 0:
        return float("nan")
    return float(np.corrcoef(pred_ranks, target_ranks)[0, 1])


def expected_calibration_error(
    probs: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    n_bins: int = 15,
) -> float:
    preds = probs.argmax(dim=-1)
    confs = probs.max(dim=-1).values

    if mask is not None:
        flat_mask = mask.reshape(-1)
        preds = preds.reshape(-1)[flat_mask]
        targets = targets.reshape(-1)[flat_mask]
        confs = confs.reshape(-1)[flat_mask]
    else:
        preds = preds.reshape(-1)
        targets = targets.reshape(-1)
        confs = confs.reshape(-1)

    if preds.numel() == 0:
        return 0.0

    bin_bounds = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = torch.zeros(1, device=probs.device)

    for i in range(n_bins):
        in_bin = (confs > bin_bounds[i]) & (confs <= bin_bounds[i + 1])
        if in_bin.any():
            acc = (preds[in_bin] == targets[in_bin]).float().mean()
            conf = confs[in_bin].mean()
            ece += torch.abs(acc - conf) * in_bin.float().mean()

    return ece.item()


def negative_log_likelihood(
    probs: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> float:
    if mask is not None:
        flat_mask = mask.reshape(-1)
        flat_probs = probs.view(-1, probs.size(-1))[flat_mask]
        flat_targets = targets.reshape(-1)[flat_mask].unsqueeze(1)
    else:
        flat_probs = probs.view(-1, probs.size(-1))
        flat_targets = targets.view(-1, 1)

    if flat_probs.numel() == 0:
        return 0.0
    gathered = torch.gather(flat_probs, 1, flat_targets).clamp_min(eps)
    return (-torch.log(gathered)).mean().item()


def confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    n_cats: int,
    mask: Optional[torch.Tensor] = None,
) -> np.ndarray:
    preds, targets = _apply_mask(preds, targets, mask)
    if preds.numel() == 0:
        return np.zeros((n_cats, n_cats), dtype=np.int64)
    preds_np = preds.cpu().numpy().astype(int)
    targets_np = targets.cpu().numpy().astype(int)
    conf = np.zeros((n_cats, n_cats), dtype=np.int64)
    for p, t in zip(preds_np, targets_np):
        conf[t, p] += 1
    return conf


def quadratic_weighted_kappa(
    preds: torch.Tensor,
    targets: torch.Tensor,
    n_cats: int,
    mask: Optional[torch.Tensor] = None,
) -> float:
    preds, targets = _apply_mask(preds, targets, mask)
    if preds.numel() == 0:
        return float("nan")
    preds_np = preds.cpu().numpy().astype(int)
    targets_np = targets.cpu().numpy().astype(int)
    return _qwk_numpy(preds_np, targets_np, n_cats)


def _qwk_numpy(preds: np.ndarray, targets: np.ndarray, n_cats: int) -> float:
    conf_mat = np.zeros((n_cats, n_cats), dtype=np.float64)
    for p, t in zip(preds, targets):
        conf_mat[t, p] += 1

    row_marginals = conf_mat.sum(axis=1)
    col_marginals = conf_mat.sum(axis=0)
    n = conf_mat.sum()

    if n == 0:
        return float("nan")

    expected = np.outer(row_marginals, col_marginals) / n
    weights = np.zeros((n_cats, n_cats), dtype=np.float64)
    for i in range(n_cats):
        for j in range(n_cats):
            weights[i, j] = ((i - j) ** 2) / ((n_cats - 1) ** 2)

    observed = (weights * conf_mat).sum() / n
    expected_score = (weights * expected).sum() / n

    if expected_score == 0:
        return float("nan")
    return 1.0 - observed / expected_score


def compute_metrics(
    probs: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    n_cats = probs.size(-1)
    preds = probs.argmax(dim=-1)
    metrics = {
        "cat_acc": categorical_accuracy(preds, targets, mask),
        "mae": mean_absolute_error(preds, targets, mask),
        "balanced_acc": balanced_accuracy(preds, targets, n_cats, mask),
        "within_one_acc": within_one_accuracy(preds, targets, mask),
        "spearman": spearman_correlation(preds, targets, mask),
        "qwk": quadratic_weighted_kappa(preds, targets, n_cats, mask),
        "ece": expected_calibration_error(probs, targets, mask),
        "nll": negative_log_likelihood(probs, targets, mask),
    }
    per_class = per_class_accuracy(preds, targets, n_cats, mask)
    for cat, value in per_class.items():
        metrics[f"cat_acc_{cat}"] = value
    return metrics
