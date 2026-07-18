"""Direct unit tests for kt_mirt.core.losses.

None of the ported kt-irt tests exercise WeightedOrdinalLoss / CombinedLoss /
compute_class_weights directly -- the source suite only reaches them
indirectly through DeepIRTModel.fit's training loop. This file closes that
gap with small synthetic tensors.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from kt_mirt.core.losses import (
    CombinedLoss,
    WeightedOrdinalLoss,
    compute_class_weights,
)


# ---------------------------------------------------------------------------
# compute_class_weights
# ---------------------------------------------------------------------------

def test_compute_class_weights_uniform_for_unknown_strategy():
    targets = torch.tensor([0, 0, 1, 2, 2, 2])
    w = compute_class_weights(targets, n_classes=3, strategy="uniform")
    assert torch.allclose(w, torch.ones(3))


def test_compute_class_weights_balanced_matches_inverse_frequency():
    # counts: class0=2, class1=1, class2=3; total=6
    targets = torch.tensor([0, 0, 1, 2, 2, 2])
    w = compute_class_weights(targets, n_classes=3, strategy="balanced")
    expected = torch.tensor([1.0, 2.0, 2.0 / 3.0])
    assert torch.allclose(w, expected, atol=1e-6)


def test_compute_class_weights_sqrt_balanced_is_sqrt_of_balanced():
    targets = torch.tensor([0, 0, 1, 2, 2, 2])
    w_bal = compute_class_weights(targets, n_classes=3, strategy="balanced")
    w_sqrt = compute_class_weights(targets, n_classes=3, strategy="sqrt_balanced")
    assert torch.allclose(w_sqrt, w_bal.sqrt(), atol=1e-6)


def test_compute_class_weights_empty_class_floors_count_at_one():
    # class 1 never appears; its count is floored at 1, not a division by zero.
    targets = torch.tensor([0, 0, 2, 2])
    w = compute_class_weights(targets, n_classes=3, strategy="balanced")
    assert torch.isfinite(w).all()


# ---------------------------------------------------------------------------
# WeightedOrdinalLoss
# ---------------------------------------------------------------------------

def test_weighted_ordinal_loss_zero_penalty_matches_weighted_ce():
    """With ordinal_penalty=0, WeightedOrdinalLoss is per-sample weighted CE
    averaged by PLAIN count N -- NOT torch's own reduction="mean" convention,
    which instead normalizes by the sum of the per-sample class weights. The
    two conventions diverge whenever class weights are non-uniform, so the
    reference here must reduce "none" and average by hand.
    """
    torch.manual_seed(0)
    logits = torch.randn(8, 4)
    targets = torch.randint(0, 4, (8,))
    weights = torch.tensor([1.0, 2.0, 0.5, 1.5])
    loss_fn = WeightedOrdinalLoss(4, class_weights=weights, ordinal_penalty=0.0)
    ref = F.cross_entropy(logits, targets, weight=weights, reduction="none").mean()
    assert torch.allclose(loss_fn(logits, targets), ref, atol=1e-6)


def test_weighted_ordinal_loss_penalizes_far_misses_more():
    """A confident miss 3 categories away must cost more than a confident miss
    1 category away: the raw CE term is identical (same logit multiset
    relative to the target), so the gap isolates the ordinal-distance scale.
    """
    n_cats = 4
    loss_fn = WeightedOrdinalLoss(n_cats, ordinal_penalty=0.5)
    target = torch.tensor([0])
    near = torch.tensor([[0.0, 5.0, -5.0, -5.0]])   # argmax=1, distance 1
    far = torch.tensor([[0.0, -5.0, -5.0, 5.0]])     # argmax=3, distance 3
    assert loss_fn(far, target).item() > loss_fn(near, target).item()


def test_weighted_ordinal_loss_reduction_sum_vs_mean():
    torch.manual_seed(1)
    logits = torch.randn(5, 3)
    targets = torch.randint(0, 3, (5,))
    mean_loss = WeightedOrdinalLoss(3, reduction="mean")(logits, targets)
    sum_loss = WeightedOrdinalLoss(3, reduction="sum")(logits, targets)
    assert math.isclose(sum_loss.item(), mean_loss.item() * 5, rel_tol=1e-5)


def test_weighted_ordinal_loss_gradient_flow():
    logits = torch.randn(4, 3, requires_grad=True)
    targets = torch.randint(0, 3, (4,))
    loss = WeightedOrdinalLoss(3, ordinal_penalty=0.5)(logits, targets)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


# ---------------------------------------------------------------------------
# CombinedLoss
# ---------------------------------------------------------------------------

def test_combined_loss_default_matches_weighted_ordinal_loss():
    torch.manual_seed(2)
    logits = torch.randn(6, 4)
    targets = torch.randint(0, 4, (6,))
    combined = CombinedLoss(4, weighted_ordinal_weight=1.0, ordinal_penalty=0.5)
    direct = WeightedOrdinalLoss(4, ordinal_penalty=0.5)
    assert torch.allclose(combined(logits, targets), direct(logits, targets), atol=1e-6)


def test_combined_loss_zero_weight_is_plain_cross_entropy():
    torch.manual_seed(3)
    logits = torch.randn(6, 4)
    targets = torch.randint(0, 4, (6,))
    combined = CombinedLoss(4, weighted_ordinal_weight=0.0)
    ref = F.cross_entropy(logits, targets)
    assert torch.allclose(combined(logits, targets), ref, atol=1e-6)
    # the nominal (NRM) path never builds the ordinal submodule at all.
    assert not hasattr(combined, "weighted_ordinal_loss")
    assert hasattr(combined, "ce_fallback")
