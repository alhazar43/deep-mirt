"""Tests for loss functions in kt_gpcm.training.losses.

Verifies:
- Each loss returns a finite scalar with factory defaults
- CombinedLoss with default weights is finite
- compute_class_weights returns correct shape
- WeightedOrdinalLoss with class weights is finite
"""

from __future__ import annotations

import torch
import pytest

from kt_gpcm.training.losses import (
    FocalLoss,
    QWKLoss,
    WeightedOrdinalLoss,
    CombinedLoss,
    compute_class_weights,
)

N, K = 64, 5


def make_batch(K=K, N=N):
    logits = torch.randn(N, K)
    targets = torch.randint(0, K, (N,))
    return logits, targets


class TestFocalLoss:
    def test_finite_default(self):
        logits, targets = make_batch()
        loss = FocalLoss()(logits, targets)
        assert torch.isfinite(loss)

    def test_scalar(self):
        logits, targets = make_batch()
        loss = FocalLoss()(logits, targets)
        assert loss.ndim == 0

    def test_gamma_zero_matches_ce(self):
        """With gamma=0, FocalLoss should reduce to CrossEntropy."""
        import torch.nn.functional as F
        torch.manual_seed(0)
        logits, targets = make_batch()
        focal = FocalLoss(alpha=1.0, gamma=0.0)(logits, targets)
        ce = F.cross_entropy(logits, targets)
        assert torch.allclose(focal, ce, atol=1e-5)


class TestQWKLoss:
    def test_finite_default(self):
        logits, targets = make_batch()
        loss = QWKLoss(K)(logits, targets)
        assert torch.isfinite(loss)

    def test_range(self):
        """1 - QWK should be in [0, 2]."""
        logits, targets = make_batch()
        loss = QWKLoss(K)(logits, targets)
        assert 0.0 <= loss.item() <= 2.0 + 1e-5

    def test_perfect_prediction(self):
        """Perfect predictions → QWK = 1 → loss ≈ 0."""
        targets = torch.arange(K).repeat(10)
        # Logits with huge values at correct positions
        logits = torch.zeros(len(targets), K)
        for i, t in enumerate(targets):
            logits[i, t] = 100.0
        loss = QWKLoss(K)(logits, targets)
        assert loss.item() < 0.05


class TestWeightedOrdinalLoss:
    def test_finite_default(self):
        logits, targets = make_batch()
        loss = WeightedOrdinalLoss(K)(logits, targets)
        assert torch.isfinite(loss)

    def test_with_class_weights(self):
        logits, targets = make_batch()
        cw = torch.rand(K) + 0.1
        loss = WeightedOrdinalLoss(K, class_weights=cw)(logits, targets)
        assert torch.isfinite(loss)

    def test_zero_penalty_matches_weighted_ce(self):
        """ordinal_penalty=0 should equal standard weighted cross-entropy."""
        import torch.nn.functional as F
        logits, targets = make_batch()
        cw = torch.ones(K)
        loss_wol = WeightedOrdinalLoss(K, class_weights=cw, ordinal_penalty=0.0)(logits, targets)
        loss_ce = F.cross_entropy(logits, targets, weight=cw)
        assert torch.allclose(loss_wol, loss_ce, atol=1e-5)


class TestCombinedLoss:
    def test_default_recipe_finite(self):
        """Default recipe: focal=0.5, weighted_ordinal=0.5."""
        logits, targets = make_batch()
        loss = CombinedLoss(K)(logits, targets)
        assert torch.isfinite(loss)

    def test_scalar(self):
        logits, targets = make_batch()
        loss = CombinedLoss(K)(logits, targets)
        assert loss.ndim == 0

    def test_with_class_weights(self):
        logits, targets = make_batch()
        cw = compute_class_weights(targets, K)
        loss = CombinedLoss(K, class_weights=cw)(logits, targets)
        assert torch.isfinite(loss)

    def test_all_zero_weights_fallback(self):
        """Zero weights → CE fallback, still finite."""
        logits, targets = make_batch()
        loss = CombinedLoss(K, focal_weight=0.0, weighted_ordinal_weight=0.0)(logits, targets)
        assert torch.isfinite(loss)

    def test_positive_loss(self):
        """Loss should be non-negative."""
        logits, targets = make_batch()
        loss = CombinedLoss(K)(logits, targets)
        assert loss.item() >= 0.0


class TestComputeClassWeights:
    def test_shape(self):
        targets = torch.randint(0, K, (200,))
        weights = compute_class_weights(targets, K)
        assert weights.shape == (K,)

    def test_all_positive(self):
        targets = torch.randint(0, K, (200,))
        weights = compute_class_weights(targets, K)
        assert (weights > 0).all()

    def test_uniform_targets(self):
        """With uniform class distribution, sqrt_balanced weights should be ~equal."""
        # Create exactly equal counts
        targets = torch.arange(K).repeat(50)
        weights = compute_class_weights(targets, K, strategy="sqrt_balanced")
        assert torch.allclose(weights, weights[0].expand(K), atol=0.01)

    def test_strategy_balanced(self):
        targets = torch.randint(0, K, (500,))
        weights = compute_class_weights(targets, K, strategy="balanced")
        assert weights.shape == (K,)
        assert (weights > 0).all()
