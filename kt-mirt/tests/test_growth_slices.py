"""Tests for `kt_mirt.growth.slices`: density strata, slice construction
from `CalibrationRows`, the D2+ selection rule, the raw saturation flag,
and the permutation-null hook (`_planning/design/a4_design.md` v1.1,
sections 1, 2.1, 2.2, battery arm 1)."""

from __future__ import annotations

import numpy as np
import pytest

from kt_mirt.growth import bank, slices


class _Log:
    """Minimal `LearnerRecord`-shaped stub (mirrors `test_growth_bank.py`)."""


def _log(item_ids, responses, tag_ids, tag_mask=None, learner=0):
    log = _Log()
    log.learner = learner
    log.item_ids = np.asarray(item_ids)
    log.responses = np.asarray(responses, dtype=np.int8)
    log.tag_ids = np.asarray(tag_ids)
    log.tag_mask = np.asarray(tag_mask) if tag_mask is not None else np.ones_like(log.tag_ids, dtype=bool)
    return log


# ---------------------------------------------------------------------------
# Density strata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "T,expected",
    [(0, "sub"), (3, "sub"), (4, "D0"), (5, "D0"), (6, "D1"), (9, "D1"), (10, "D2"), (19, "D2"), (20, "D3"), (100, "D3")],
)
def test_density_stratum_edges(T, expected):
    assert slices.density_stratum(T) == expected


def test_is_d2_plus_vectorized():
    out = slices.is_d2_plus(np.array([3, 9, 10, 19, 20]))
    assert out.tolist() == [False, False, True, True, True]


# ---------------------------------------------------------------------------
# Slice construction (single-tag)
# ---------------------------------------------------------------------------


def test_build_slices_single_tag_groups_by_learner_kc_in_arrival_order():
    log0 = _log([5, 6, 5], [1, 0, 1], [[0], [1], [0]])
    log1 = _log([7], [1], [[0]])
    rows = bank.build_calibration_rows([log0, log1])
    sl = slices.build_slices(rows)
    assert set(sl.keys()) == {(0, 0), (0, 1), (1, 0)}
    s00 = sl[(0, 0)]
    assert s00.item_id.tolist() == [5, 5]
    assert s00.response.tolist() == [1, 1]
    assert s00.opportunity.tolist() == [1, 2]
    assert s00.T == 2
    s01 = sl[(0, 1)]
    assert s01.T == 1
    assert s01.opportunity.tolist() == [1]
    s10 = sl[(1, 0)]
    assert s10.item_id.tolist() == [7]


def test_build_slices_block_id_matches_bank_opportunity_block():
    log0 = _log(list(range(20)), [1] * 20, [[0]] * 20)
    rows = bank.build_calibration_rows([log0])
    sl = slices.build_slices(rows)
    s = sl[(0, 0)]
    assert s.opportunity.tolist() == list(range(1, 21))
    expected_blocks = bank.opportunity_block(np.arange(1, 21))
    assert s.block_id.tolist() == expected_blocks.tolist()


def test_build_slices_empty_input_returns_empty_dict():
    rows = bank.build_calibration_rows([])
    assert slices.build_slices(rows) == {}


# ---------------------------------------------------------------------------
# Slice construction (multi-tag): opportunity index follows arrival order
# regardless of which tag slot a KC occupies (bank.py's own docstring point)
# ---------------------------------------------------------------------------


def test_build_slices_multi_tag_opportunity_follows_row_order_not_slot():
    # Item A tags KC0 in slot 0; item B tags KC0 in slot 1. KC0's opportunity
    # count must still follow arrival (row) order across both items.
    log0 = _log(
        item_ids=[100, 200, 100],
        responses=[1, 0, 1],
        tag_ids=[[0, -1], [1, 0], [0, -1]],
        tag_mask=[[True, False], [True, True], [True, False]],
    )
    rows = bank.build_calibration_rows([log0])
    sl = slices.build_slices(rows)
    s_kc0 = sl[(0, 0)]
    assert s_kc0.T == 3
    assert s_kc0.opportunity.tolist() == [1, 2, 3]
    assert s_kc0.item_id.tolist() == [100, 200, 100]
    s_kc1 = sl[(0, 1)]
    assert s_kc1.T == 1
    assert s_kc1.item_id.tolist() == [200]


# ---------------------------------------------------------------------------
# slices_by_kc / select_d2_plus
# ---------------------------------------------------------------------------


def test_slices_by_kc_groups_correctly():
    log0 = _log([1, 2], [1, 1], [[0], [1]])
    rows = bank.build_calibration_rows([log0])
    sl = slices.build_slices(rows)
    by_kc = slices.slices_by_kc(sl, n_kcs=3)
    assert len(by_kc) == 3
    assert len(by_kc[0]) == 1 and len(by_kc[1]) == 1 and len(by_kc[2]) == 0


def test_select_d2_plus_filters_short_slices():
    log0 = _log(list(range(3)), [1, 1, 1], [[0]] * 3)  # T=3, sub
    log1 = _log(list(range(12)), [1] * 12, [[0]] * 12)  # T=12, D2+
    rows = bank.build_calibration_rows([log0, log1])
    sl = slices.build_slices(rows)
    d2 = slices.select_d2_plus(sl)
    assert set(d2.keys()) == {(1, 0)}


# ---------------------------------------------------------------------------
# Saturation flag (section 2.1: raw, model-free)
# ---------------------------------------------------------------------------


def test_saturation_stats_correct_rate_and_threshold():
    # KC0: 3 correct / 4 -> 0.75 (unsaturated at 0.85). KC1: all correct -> 1.0 (saturated).
    log0 = _log([0, 0, 0, 0], [1, 1, 1, 0], [[0]] * 4)
    log1 = _log([1, 1], [1, 1], [[1]] * 2)
    rows = bank.build_calibration_rows([log0, log1])
    rate, is_unsat = slices.saturation_stats(rows, n_kcs=3, threshold=0.85)
    assert rate[0] == pytest.approx(0.75)
    assert rate[1] == pytest.approx(1.0)
    assert np.isnan(rate[2])
    assert is_unsat.tolist() == [True, False, False]


def test_saturation_stats_multi_tag_counts_every_tagged_occurrence():
    # One item tagged with both KC0 and KC1 -> both KCs get the response.
    log0 = _log([0], [1], [[0, 1]])
    rows = bank.build_calibration_rows([log0])
    rate, is_unsat = slices.saturation_stats(rows, n_kcs=2)
    assert rate[0] == pytest.approx(1.0)
    assert rate[1] == pytest.approx(1.0)


def test_saturation_stats_empty_rows_all_nan():
    rows = bank.build_calibration_rows([])
    rate, is_unsat = slices.saturation_stats(rows, n_kcs=2)
    assert np.isnan(rate).all()
    assert is_unsat.tolist() == [False, False]


# ---------------------------------------------------------------------------
# Permutation hook (battery arm 1)
# ---------------------------------------------------------------------------


def test_permute_learner_order_preserves_multiset_shuffles_order():
    log0 = _log([1, 2, 3, 4, 5], [1, 0, 1, 0, 1], [[0]] * 5)
    rng = np.random.default_rng(0)
    permuted = slices.permute_learner_order([log0], rng)
    p = permuted[0]
    assert sorted(p.item_ids.tolist()) == sorted(log0.item_ids.tolist())
    assert sorted(p.responses.tolist()) == sorted(log0.responses.tolist())
    # Order-item and order-response pairing must move together (same permutation).
    orig_pairs = set(zip(log0.item_ids.tolist(), log0.responses.tolist()))
    perm_pairs = set(zip(p.item_ids.tolist(), p.responses.tolist()))
    assert orig_pairs == perm_pairs


def test_permute_learner_order_is_seed_reproducible():
    log0 = _log(list(range(20)), [1] * 20, [[0]] * 20)
    r1 = np.random.default_rng(3)
    r2 = np.random.default_rng(3)
    p1 = slices.permute_learner_order([log0], r1)
    p2 = slices.permute_learner_order([log0], r2)
    assert np.array_equal(p1[0].item_ids, p2[0].item_ids)


def test_permute_learner_order_destroys_within_slice_trend_on_average():
    """The permutation null's whole point: a strong monotone response
    trend (e.g. all-wrong-then-all-right) should, after permutation, no
    longer be recoverable as a trend by simple position correlation
    (mechanical smoke check, not a statistical claim)."""
    n = 40
    responses = np.array([0] * (n // 2) + [1] * (n // 2))
    log0 = _log(list(range(n)), responses, [[0]] * n)
    rng = np.random.default_rng(1)
    orig_corr = np.corrcoef(np.arange(n), responses)[0, 1]
    permuted = slices.permute_learner_order([log0], rng)[0]
    perm_corr = np.corrcoef(np.arange(n), permuted.responses)[0, 1]
    assert abs(orig_corr) > 0.8
    assert abs(perm_corr) < abs(orig_corr)
