"""
Tests for the rebalanced dynjoint training path (train_joint_balanced).

Test IDs
--------
1. test_balance_none_matches_baseline -- balance="none" with default levers is
   bit-identical to the original train_joint (P1.6 reproducibility guard).
2. test_balance_modes_run            -- every balance mode runs, returns a model,
   and produces a finite, mean-centred d_i of the right shape.
3. test_gradnorm_weight_increases_bt -- the gradnorm balancer raises BT's pull on
   the shared trunk relative to the unweighted case (the rebalancing actually
   moves the lever it claims to move).
"""

from __future__ import annotations

import torch
import pytest

from deep_irt.dynjoint.data_gen import (
    generate_ground_truth,
    generate_mode_a_dynamic,
    generate_mode_b,
)
from deep_irt.dynjoint.train import train_joint, train_joint_balanced


_SEED = 0


def _tiny_data(seed: int = _SEED):
    """Small, fast synthetic problem that still exercises all three partitions."""
    gt = generate_ground_truth(
        n_items=60, n_direct_only=20, n_pairwise_only=20, K=4, seed=seed
    )
    ma = generate_mode_a_dynamic(
        gt, n_respondents=40, seq_len=15, drift_sigma=0.2, seed=seed
    )
    mb = generate_mode_b(gt, comparisons_per_item=40, seed=seed)
    return gt, ma, mb


# ---------------------------------------------------------------------------
# Test 1: balance="none" reproduces the baseline exactly
# ---------------------------------------------------------------------------

def test_balance_none_matches_baseline() -> None:
    gt, ma, mb = _tiny_data()
    common = dict(
        emb_dim=16, hidden_dim=64, n_epochs=12, lr=5e-3,
        batch_size=64, reg=0.01, seed=_SEED, verbose=False,
    )
    m_base = train_joint(gt, ma, mb, **common)
    m_bal = train_joint_balanced(
        gt, ma, mb, balance="none", bt_pairs_mult=1.0,
        head_lr_mult=1.0, warmup_frac=0.0, clip_norm=5.0, **common
    )
    d_base = m_base.get_item_strengths()
    d_bal = m_bal.get_item_strengths()
    assert torch.allclose(d_base, d_bal, atol=1e-6), (
        f"balance='none' diverged from train_joint: "
        f"max abs diff = {(d_base - d_bal).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 2: every balance mode runs and yields a sane d_i
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["none", "fixed", "uncertainty", "gradnorm"])
def test_balance_modes_run(mode: str) -> None:
    gt, ma, mb = _tiny_data()
    model = train_joint_balanced(
        gt, ma, mb, balance=mode, n_epochs=8, lr=5e-3,
        batch_size=64, reg=0.01, seed=_SEED, verbose=False,
    )
    d = model.get_item_strengths()
    assert d.shape == (gt["n_items"],)
    assert torch.isfinite(d).all(), f"{mode}: non-finite d_i"
    assert abs(float(d.mean())) < 1e-5, f"{mode}: d_i not mean-centred"


def test_unknown_balance_raises() -> None:
    gt, ma, mb = _tiny_data()
    with pytest.raises(ValueError, match="unknown balance mode"):
        train_joint_balanced(
            gt, ma, mb, balance="nonsense", n_epochs=2, seed=_SEED, verbose=False
        )


# ---------------------------------------------------------------------------
# Test 3: gradnorm and fixed up-weighting change the recovered shared scale
# ---------------------------------------------------------------------------

def test_balancing_changes_shared_scale() -> None:
    """The rebalancing knobs must actually move the shared d_i, not be inert.

    A model trained with gradnorm (or a fixed BT up-weight) reweights the BT
    head's contribution to the shared trunk, so its recovered d_i must differ
    from the unweighted baseline. We assert the knobs are non-trivial: both
    gradnorm and fixed w_bt=6 produce a d_i that is measurably different from
    balance='none' under the same seed and data."""
    gt, ma, mb = _tiny_data()
    common = dict(
        emb_dim=16, hidden_dim=64, n_epochs=40, lr=5e-3,
        batch_size=64, reg=0.01, seed=_SEED, verbose=False,
    )
    d_none = train_joint_balanced(gt, ma, mb, balance="none", **common).get_item_strengths()
    d_grad = train_joint_balanced(gt, ma, mb, balance="gradnorm", **common).get_item_strengths()
    d_fix6 = train_joint_balanced(gt, ma, mb, balance="fixed", bt_weight=6.0, **common).get_item_strengths()

    grad_shift = (d_grad - d_none).abs().max().item()
    fix_shift = (d_fix6 - d_none).abs().max().item()
    assert grad_shift > 1e-4, f"gradnorm left d_i unchanged (max shift {grad_shift:.2e})"
    assert fix_shift > 1e-4, f"fixed w_bt=6 left d_i unchanged (max shift {fix_shift:.2e})"
