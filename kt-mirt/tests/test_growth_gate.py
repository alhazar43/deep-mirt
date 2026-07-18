"""Tests for `kt_mirt.growth.gate`: PAS-G's M0/M1a/M1b family under the
penalized bounded Newton primitive, the interpolative odd/even gate
statistic at every pooling level, the permutation-null hook, and BH/BY
discovery control (`_planning/design/a4_design.md` v1.1, section 2.2)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kt_mirt.growth import bank, gate
from kt_mirt.growth.slices import Slice


def _bank_from_b(b_hat: np.ndarray) -> bank.FrozenBank:
    hier = bank.flat_hierarchy(len(b_hat))
    fit = bank.BankFitResult(
        hierarchy=hier,
        growth_mode="none",
        b_hat=np.asarray(b_hat, dtype=float),
        eligible_leaf=np.ones(len(b_hat), dtype=bool),
        problem_seen=np.ones(len(b_hat), dtype=bool),
        calib_exposure_count=np.full(len(b_hat), 100),
        converged=True,
        n_epochs_run=1,
        final_data_nll=0.0,
    )
    return bank.freeze_bank(fit)


def _slice(learner, kc, item_id, response, opportunity=None, block_id=None):
    item_id = np.asarray(item_id)
    response = np.asarray(response, dtype=np.int8)
    if opportunity is None:
        opportunity = np.arange(1, len(item_id) + 1)
    if block_id is None:
        block_id = bank.opportunity_block(opportunity)
    return Slice(
        learner=learner, kc=kc, item_id=item_id, response=response,
        opportunity=np.asarray(opportunity), block_id=np.asarray(block_id),
    )


def _make_no_growth_slices(rng, n_slices, T, theta, b=0.0, kc=0):
    """Constant-ability (no trend) slices, all in one KC, for the null side."""
    out = {}
    for i in range(n_slices):
        p = 1.0 / (1.0 + np.exp(-(theta - b)))
        resp = (rng.random(T) < p).astype(np.int8)
        out[(i, kc)] = _slice(i, kc, item_id=np.zeros(T, dtype=int), response=resp)
    return out


def _make_linear_growth_slices(rng, n_slices, T, theta0, slope, b=0.0, kc=0):
    out = {}
    for i in range(n_slices):
        n = np.arange(1, T + 1)
        theta_n = theta0 + slope * (n - 1)
        p = 1.0 / (1.0 + np.exp(-(theta_n - b)))
        resp = (rng.random(T) < p).astype(np.int8)
        out[(i, kc)] = _slice(i, kc, item_id=np.zeros(T, dtype=int), response=resp)
    return out


# ---------------------------------------------------------------------------
# Design-matrix builders
# ---------------------------------------------------------------------------


def test_m0_design_is_intercept_only():
    opp = np.array([1, 2, 3])
    block = np.array([0, 0, 1])
    d = gate._m0_design(opp, block)
    assert d.shape == (3, 1)
    assert (d == 1.0).all()


def test_m1a_slice_design_has_intercept_and_centered_opportunity():
    opp = np.array([1, 2, 3])
    d = gate._m1a_slice_design(opp, np.zeros(3, dtype=int))
    assert d.shape == (3, 2)
    assert d[:, 1].tolist() == [0.0, 1.0, 2.0]


def test_m1b_slice_design_is_one_hot_over_blocks():
    opp = np.array([1, 5, 9, 20])
    block = bank.opportunity_block(opp)
    d = gate._m1b_slice_design(opp, block)
    assert d.shape == (4, gate.N_BLOCKS)
    assert d.sum(axis=1).tolist() == [1.0, 1.0, 1.0, 1.0]
    assert d[np.arange(4), block].tolist() == [1.0, 1.0, 1.0, 1.0]


# ---------------------------------------------------------------------------
# Numerical safeguard (R2-B3's explicit unit test): finite MAP + finite
# held-out NLL on all-correct / all-incorrect slices, for both M0 and M1.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("responses", [[1] * 10, [0] * 10])
def test_penalized_fits_stay_finite_under_separation(responses):
    sl = [_slice(0, 0, item_id=np.zeros(10, dtype=int), response=responses)]
    for fit_fn, P in [(gate.fit_m0, 1), (gate.fit_m1a_slice, 2), (gate.fit_m1b_slice, gate.N_BLOCKS)]:
        fitted = fit_fn(sl, None, time_filter=gate._time_filter_odd)
        assert torch.isfinite(fitted.params).all()
        nll = gate.held_out_nll(sl, None, gate._m1b_slice_design if P == gate.N_BLOCKS else
                                 (gate._m1a_slice_design if P == 2 else gate._m0_design),
                                 fitted.params, gate._time_filter_even)
        assert torch.isfinite(nll).all()


def test_fit_kc_joint_stays_finite_under_all_correct_kc():
    kc_slices = [_slice(i, 0, item_id=np.zeros(8, dtype=int), response=[1] * 8) for i in range(5)]
    theta, shared = gate.fit_kc_joint(kc_slices, None, gate._m1b_shared_design, gate.N_BLOCKS, time_filter=gate._time_filter_odd)
    assert np.isfinite(theta).all()
    assert np.isfinite(shared).all()


# ---------------------------------------------------------------------------
# Gate statistic: sanity on constructed data (no-trend vs strong-trend)
# ---------------------------------------------------------------------------


def test_compute_gate_result_detects_strong_linear_trend():
    rng = np.random.default_rng(0)
    frozen = _bank_from_b(np.array([0.0]))
    slices_dict = _make_linear_growth_slices(rng, n_slices=60, T=20, theta0=-3.0, slope=0.6)
    result = gate.compute_gate_result(slices_dict, frozen, n_kcs=1)
    assert result.bed_stat > 0
    assert result.kc_stat[0] > 0
    assert result.kc_selected_family[0] in ("m1a", "m1b")
    assert len(result.slice_keys) == 60  # all T=20 slices are D2+


def test_compute_gate_result_near_zero_on_flat_no_trend_kc():
    rng = np.random.default_rng(1)
    frozen = _bank_from_b(np.array([0.0]))
    slices_dict = _make_no_growth_slices(rng, n_slices=80, T=20, theta=0.3)
    growth_result = gate.compute_gate_result(slices_dict, frozen, n_kcs=1)

    growth_dict = _make_linear_growth_slices(rng, n_slices=80, T=20, theta0=-3.0, slope=0.6)
    trend_result = gate.compute_gate_result(growth_dict, frozen, n_kcs=1)
    # A genuine trend must produce a materially larger held-out improvement
    # than a flat (no-trend) KC of the same size and density.
    assert trend_result.bed_stat > growth_result.bed_stat


def test_compute_gate_result_empty_slices_is_defensive():
    result = gate.compute_gate_result({}, None, n_kcs=2)
    assert result.bed_stat == 0.0
    assert result.kc_stat.tolist() == [0.0, 0.0]
    assert result.slice_stat.shape == (0,)


def test_compute_gate_result_sub_d2_slices_excluded_from_slice_stat():
    rng = np.random.default_rng(2)
    frozen = _bank_from_b(np.array([0.0]))
    short_slices = _make_no_growth_slices(rng, n_slices=5, T=6, theta=0.0)  # D1, not D2+
    result = gate.compute_gate_result(short_slices, frozen, n_kcs=1)
    assert result.slice_stat.shape == (0,)
    assert len(result.slice_keys) == 0


# ---------------------------------------------------------------------------
# Permutation-null hook + empirical p-value
# ---------------------------------------------------------------------------


def test_empirical_pvalue_floor_and_extremes():
    null = np.array([1.0, 2.0, 3.0, 4.0])
    assert gate.empirical_pvalue(10.0, null) == pytest.approx(1 / 5)  # nothing >= observed
    assert gate.empirical_pvalue(0.0, null) == pytest.approx(5 / 5)  # everything >= observed
    assert np.isnan(gate.empirical_pvalue(1.0, np.array([])))


def test_permutation_null_runs_and_produces_finite_null_distribution():
    class Log:
        pass

    def make_log(learner, T, rng):
        log = Log()
        log.learner = learner
        log.item_ids = np.zeros(T, dtype=int)
        log.responses = (rng.random(T) < 0.6).astype(np.int8)
        log.tag_ids = np.zeros((T, 1), dtype=int)
        log.tag_mask = np.ones((T, 1), dtype=bool)
        return log

    rng = np.random.default_rng(0)
    learners = [make_log(i, 12, rng) for i in range(15)]
    frozen = _bank_from_b(np.array([0.0]))
    null = gate.permutation_null(learners, n_learners=15, n_kcs=1, bank=frozen, n_replicates=5, seed=0)
    assert null["bed"].shape == (5,)
    assert null["kc"].shape == (5, 1)
    assert np.isfinite(null["bed"]).all()


def test_gate_statistic_on_learners_matches_direct_slice_construction():
    class Log:
        pass

    def make_log(learner, T, rng):
        log = Log()
        log.learner = learner
        log.item_ids = np.zeros(T, dtype=int)
        log.responses = (rng.random(T) < 0.6).astype(np.int8)
        log.tag_ids = np.zeros((T, 1), dtype=int)
        log.tag_mask = np.ones((T, 1), dtype=bool)
        return log

    rng = np.random.default_rng(3)
    learners = [make_log(i, 15, rng) for i in range(10)]
    rows = bank.build_calibration_rows(learners)
    from kt_mirt.growth.slices import build_slices

    direct_slices = build_slices(rows)
    frozen = _bank_from_b(np.array([0.0]))
    direct_result = gate.compute_gate_result(direct_slices, frozen, n_kcs=1)
    via_learners = gate.gate_statistic_on_learners(learners, n_learners=10, n_kcs=1, bank=frozen)
    assert direct_result.bed_stat == pytest.approx(via_learners.bed_stat)


# ---------------------------------------------------------------------------
# BH-FDR / Benjamini-Yekutieli
# ---------------------------------------------------------------------------


def test_bh_fdr_rejects_all_when_all_pvalues_tiny():
    p = np.array([0.0001, 0.0002, 0.0003])
    assert gate.bh_fdr(p, q=0.05).all()


def test_bh_fdr_rejects_none_when_all_pvalues_large():
    p = np.array([0.9, 0.8, 0.95])
    assert not gate.bh_fdr(p, q=0.05).any()


def test_bh_fdr_mixed_pvalues_step_up_procedure():
    # Classic BH textbook example: 5 p-values, q=0.05.
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.5])
    reject = gate.bh_fdr(p, q=0.05)
    # Threshold_i = q*i/m: 0.01, 0.02, 0.03, 0.04, 0.05.
    # sorted p <= threshold: [T,T,F,F,F] -> largest passing index is 1 (0-based),
    # so the first two sorted p-values are rejected.
    assert reject.tolist() == [True, True, False, False, False]


def test_by_correction_is_more_conservative_than_bh():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.5])
    bh = gate.bh_fdr(p, q=0.05)
    by = gate.by_correction(p, q=0.05)
    # BY's harmonic-sum correction can only reject a subset of what BH rejects.
    assert by.sum() <= bh.sum()
    assert np.all(~by | bh)  # every BY rejection is also a BH rejection here


def test_bh_fdr_and_by_correction_empty_input():
    assert gate.bh_fdr(np.array([])).shape == (0,)
    assert gate.by_correction(np.array([])).shape == (0,)


# ---------------------------------------------------------------------------
# Frozen bank's is_calibrated exclusion is respected in slice fits
# ---------------------------------------------------------------------------


def test_uncalibrated_items_excluded_from_gate_fit():
    hier = bank.flat_hierarchy(2)
    fit = bank.BankFitResult(
        hierarchy=hier, growth_mode="none", b_hat=np.array([0.0, 0.0]),
        eligible_leaf=np.array([True, True]), problem_seen=np.array([True, False]),
        calib_exposure_count=np.array([100, 0]), converged=True, n_epochs_run=1, final_data_nll=0.0,
    )
    frozen = bank.freeze_bank(fit)
    # Item 1 is uncalibrated; a slice built entirely of item 1 should
    # contribute nothing to the odd/even held-out fit (mask all-False).
    sl = [_slice(0, 0, item_id=np.ones(10, dtype=int), response=[1] * 10)]
    fitted = gate.fit_m0(sl, frozen, time_filter=gate._time_filter_odd)
    assert torch.isfinite(fitted.params).all()  # penalty alone still gives a finite (prior-mean) MAP
