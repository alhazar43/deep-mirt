"""Welford running mean/variance freeze invariants."""

from __future__ import annotations

import math

import torch

from ordrec.reward.running_norm import RunningMeanStd


def test_welford_matches_numpy_after_n_samples() -> None:
    rms = RunningMeanStd(freeze_after=0)  # never freeze
    xs = torch.linspace(-3.0, 5.0, 64)
    for x in xs:
        rms.update(x.item())
    expected_mean = xs.mean().item()
    # Sample variance with Bessel's correction matches Welford's M2.
    expected_var = xs.var(unbiased=True).item()
    assert math.isclose(rms.mean, expected_mean, abs_tol=1e-6)
    assert math.isclose(rms.var, expected_var, abs_tol=1e-6)


def test_freeze_at_threshold_is_no_op_after() -> None:
    rms = RunningMeanStd(freeze_after=5)
    for x in [1.0, 2.0, 3.0, 4.0, 5.0]:
        rms.update(x)
    assert rms.frozen is True
    mean_before = rms.mean
    var_before = rms.var
    # 1001st update is a no-op.
    rms.update(100.0)
    rms.update(-100.0)
    assert rms.mean == mean_before
    assert rms.var == var_before


def test_normalise_uses_current_stats() -> None:
    rms = RunningMeanStd(freeze_after=0)
    for x in range(100):
        rms.update(float(x))
    x = torch.tensor([0.0, 50.0, 99.0])
    out = rms.normalise(x)
    # Manual centring/scaling check.
    expected = (x - rms.mean) / rms.std
    assert torch.allclose(out, expected, atol=1e-6)


def test_batched_update_counts_each_element() -> None:
    rms = RunningMeanStd(freeze_after=0)
    batch = torch.tensor([1.0, 2.0, 3.0, 4.0])
    rms.update(batch)
    assert rms.count == 4
    assert math.isclose(rms.mean, 2.5, abs_tol=1e-6)


def test_state_dict_round_trip() -> None:
    rms = RunningMeanStd(freeze_after=10)
    for x in [1.0, 2.5, -0.5, 3.0]:
        rms.update(x)
    state = rms.state_dict()
    rms2 = RunningMeanStd(freeze_after=10)
    rms2.load_state_dict(state)
    assert rms2.count == rms.count
    assert math.isclose(rms2.mean, rms.mean, abs_tol=0.0)
    assert math.isclose(rms2.var, rms.var, abs_tol=0.0)
