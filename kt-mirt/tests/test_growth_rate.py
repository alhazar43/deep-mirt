"""Tests for `kt_mirt.growth.rate`: MIX-L's stage-2 bounded-exponential
rate fits (E-M1 pooled, E-M2 per-slice) and the family-misfit flag E-M3
(`_planning/design/a4_design.md` v1.1, section 2.4)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kt_mirt.growth import bank, rate
from kt_mirt.growth.slices import Slice


def _bank_from_b(b_hat: np.ndarray) -> bank.FrozenBank:
    hier = bank.flat_hierarchy(len(b_hat))
    fit = bank.BankFitResult(
        hierarchy=hier, growth_mode="none", b_hat=np.asarray(b_hat, dtype=float),
        eligible_leaf=np.ones(len(b_hat), dtype=bool), problem_seen=np.ones(len(b_hat), dtype=bool),
        calib_exposure_count=np.full(len(b_hat), 100), converged=True, n_epochs_run=1, final_data_nll=0.0,
    )
    return bank.freeze_bank(fit)


def _slice(learner, kc, item_id, response, opportunity=None, block_id=None):
    item_id = np.asarray(item_id)
    response = np.asarray(response, dtype=np.int8)
    if opportunity is None:
        opportunity = np.arange(1, len(item_id) + 1)
    if block_id is None:
        block_id = bank.opportunity_block(opportunity)
    return Slice(learner=learner, kc=kc, item_id=item_id, response=response,
                 opportunity=np.asarray(opportunity), block_id=np.asarray(block_id))


def _bounded_exp_slices(rng, n_slices, T, m, r, theta0, kc=0, b=0.0):
    out = []
    n = np.arange(1, T + 1)
    for i in range(n_slices):
        theta_n = m - (m - theta0) * np.exp(-r * (n - 1))
        p = 1.0 / (1.0 + np.exp(-(theta_n - b)))
        resp = (rng.random(T) < p).astype(np.int8)
        out.append(_slice(i, kc, item_id=np.zeros(T, dtype=int), response=resp))
    return out


# ---------------------------------------------------------------------------
# E-M1: pooled per-KC rate
# ---------------------------------------------------------------------------


def test_fit_kc_rate_recovers_growth_direction_and_stays_finite():
    rng = np.random.default_rng(0)
    frozen = _bank_from_b(np.array([0.0]))
    kc_slices = _bounded_exp_slices(rng, n_slices=60, T=20, m=3.0, r=0.2, theta0=-2.0)
    fit = rate.fit_kc_rate(kc_slices, frozen)
    assert fit is not None
    assert np.isfinite(fit.m_c)
    assert np.isfinite(fit.r_c)
    assert np.isfinite(fit.theta0).all()
    assert fit.r_c > 0  # true r=0.2 > 0, growth is real and should be recovered as positive


def test_fit_kc_rate_wald_z_large_for_strong_growth():
    rng = np.random.default_rng(1)
    frozen = _bank_from_b(np.array([0.0]))
    kc_slices = _bounded_exp_slices(rng, n_slices=100, T=20, m=3.0, r=0.25, theta0=-2.5)
    fit = rate.fit_kc_rate(kc_slices, frozen)
    assert np.isfinite(fit.r_c_se)
    assert np.isfinite(fit.r_c_z)
    assert fit.r_c_z > 2.0  # should clearly reject r=0 given this much signal


def test_fit_kc_rate_near_zero_r_on_flat_no_growth_kc():
    """Genuinely flat (i.i.d. Bernoulli, no opportunity dependence) data,
    NOT an exact m==theta0 construction: the latter puts every slice at a
    literal non-identifiability point for r (zero data curvature in the r
    direction), which makes the Wald SE estimate itself unstable -- a
    known Wald-test limitation under exact degeneracy, not something to
    assert against. A small amount of natural per-slice level variation
    is the honest way to probe "no true growth"."""
    rng = np.random.default_rng(2)
    frozen = _bank_from_b(np.array([0.0]))
    n_slices, T = 80, 20
    kc_slices = []
    for i in range(n_slices):
        p = 1.0 / (1.0 + np.exp(-(0.3 + rng.normal(0.0, 0.2))))
        resp = (rng.random(T) < p).astype(np.int8)
        kc_slices.append(_slice(i, 0, item_id=np.zeros(T, dtype=int), response=resp))
    fit = rate.fit_kc_rate(kc_slices, frozen)
    assert abs(fit.r_c) < 0.1  # no true opportunity dependence: recovered rate should be small


def test_fit_kc_rate_empty_input_returns_none():
    assert rate.fit_kc_rate([], None) is None


def test_fit_kc_rate_all_masked_out_returns_none():
    hier = bank.flat_hierarchy(1)
    fit_bank = bank.BankFitResult(
        hierarchy=hier, growth_mode="none", b_hat=np.array([0.0]),
        eligible_leaf=np.array([True]), problem_seen=np.array([False]),
        calib_exposure_count=np.array([0]), converged=True, n_epochs_run=1, final_data_nll=0.0,
    )
    frozen = bank.freeze_bank(fit_bank)
    sl = [_slice(0, 0, item_id=np.zeros(10, dtype=int), response=[1] * 10)]
    assert rate.fit_kc_rate(sl, frozen) is None


# ---------------------------------------------------------------------------
# E-M2: per-slice rate, m_c fixed as external data
# ---------------------------------------------------------------------------


def test_fit_slice_rate_batched_shapes_and_finite():
    rng = np.random.default_rng(3)
    frozen = _bank_from_b(np.array([0.0]))
    kc_slices = _bounded_exp_slices(rng, n_slices=30, T=20, m=3.0, r=0.2, theta0=-2.0)
    kc_fit = rate.fit_kc_rate(kc_slices, frozen)
    fitted = rate.fit_slice_rate(kc_slices, {0: kc_fit.m_c}, frozen)
    assert fitted.theta0.shape == (30,)
    assert fitted.r_ic.shape == (30,)
    assert np.isfinite(fitted.theta0).all()
    assert np.isfinite(fitted.r_ic).all()


def test_fit_slice_rate_skips_slices_whose_kc_has_no_m_c():
    rng = np.random.default_rng(4)
    frozen = _bank_from_b(np.array([0.0]))
    kc0_slices = _bounded_exp_slices(rng, n_slices=5, T=15, m=3.0, r=0.2, theta0=-2.0, kc=0)
    kc1_slices = _bounded_exp_slices(rng, n_slices=5, T=15, m=3.0, r=0.2, theta0=-2.0, kc=1)
    fitted = rate.fit_slice_rate(kc0_slices + kc1_slices, {0: 3.0}, frozen)
    assert len(fitted.keys) == 5
    assert all(kc == 0 for _, kc in fitted.keys)


def test_fit_slice_rate_empty_input():
    fitted = rate.fit_slice_rate([], {}, None)
    assert fitted.keys == []
    assert fitted.theta0.shape == (0,)


# ---------------------------------------------------------------------------
# E-M3: family-misfit flag
# ---------------------------------------------------------------------------


def test_misfit_flag_silent_when_bounded_exponential_is_the_true_family():
    rng = np.random.default_rng(5)
    frozen = _bank_from_b(np.array([0.0]))
    kc_slices = _bounded_exp_slices(rng, n_slices=100, T=20, m=3.0, r=0.2, theta0=-2.5)
    kc_fit = rate.fit_kc_rate(kc_slices, frozen)
    flag = rate.misfit_flag(kc_slices, frozen, kc_fit, n_bootstrap=200, seed=0)
    assert np.isfinite(flag.p_value)
    assert flag.fired is False  # blockwise should not systematically beat the TRUE family


def test_misfit_flag_fires_on_genuinely_non_standard_shape():
    """A step-function shape (insight-like jump) is badly misfit by a
    smooth bounded-exponential but well captured by the blockwise model
    (design's SYN-NS scenario, section 3.2)."""
    rng = np.random.default_rng(6)
    frozen = _bank_from_b(np.array([0.0]))
    n_slices, T = 120, 20
    kc_slices = []
    n = np.arange(1, T + 1)
    for i in range(n_slices):
        cp = 8  # jump at opportunity 8
        theta_n = np.where(n < cp, -3.0, 2.0)
        p = 1.0 / (1.0 + np.exp(-theta_n))
        resp = (rng.random(T) < p).astype(np.int8)
        kc_slices.append(_slice(i, 0, item_id=np.zeros(T, dtype=int), response=resp))
    kc_fit = rate.fit_kc_rate(kc_slices, frozen)
    flag = rate.misfit_flag(kc_slices, frozen, kc_fit, n_bootstrap=200, seed=1)
    assert flag.fired is True
    assert flag.mean_diff > 0  # blockwise's held-out NLL beats bounded-exponential's


def test_misfit_flag_empty_input_is_defensive():
    flag = rate.misfit_flag([], None, rate.KCRateFit(theta0=np.zeros(0), m_c=0.0, r_c=0.0, r_c_se=float("nan"), r_c_z=float("nan")))
    assert flag.fired is False
    assert np.isnan(flag.p_value)


# ---------------------------------------------------------------------------
# Exponent clamping (module docstring note 3): extreme r*n does not raise
# or produce non-finite results.
# ---------------------------------------------------------------------------


def test_bounded_exp_kc_nll_stays_finite_under_extreme_rate_params():
    y = torch.tensor([1.0, 0.0, 1.0])
    mask = torch.ones(3, dtype=torch.bool)
    b = torch.zeros(3)
    opp = torch.tensor([1.0, 30.0, 60.0])
    onehot = torch.ones(3, 1)
    extreme_params = torch.tensor([0.0, 5.0, 50.0])  # r_c = 50, way past pre-registered range
    out = rate._bounded_exp_kc_nll(extreme_params, y, mask, b, opp, onehot)
    assert torch.isfinite(out)
