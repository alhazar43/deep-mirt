"""Tests for `kt_mirt.growth.bank`: the frozen item-difficulty bank
(design section 2.1, `_planning/design/a4_design.md` v1.1) -- hierarchical
MAP calibration via Adam, blockwise growth absorption, the cohort split,
the `FrozenBank` read object, and battery arm 10's two check functions
(synthetic b-recovery, tri-spec RB0 stability).

Scale note: calibration-fit tests use small-to-moderate synthetic data
(tens to a couple hundred items, hundreds to ~1500 learners) sized for
CPU speed (design's own compute plan reserves GPU-hours for the true
KDD/EdNet-scale fits, section 7); they check that the mechanism recovers
signal and that the tri-spec pairwise stability bar (>=0.95, RB0) is met
at this scale, NOT that the full pre-registered CG3/CG5 (>=0.9 vs.
generator truth) bank-recovery bar is met -- that gate is evaluated later
by the battery module against the actual full-scale twins (`SYN_DEV`
itself carries no certification number, per `growth/synth.py`'s own
docstring, and this dev-scale data is smaller still).
"""

from __future__ import annotations

import numpy as np
import pytest

from kt_mirt.growth import bank


# ---------------------------------------------------------------------------
# Opportunity blocks
# ---------------------------------------------------------------------------


def test_opportunity_block_edges():
    n = np.array([1, 2, 3, 4, 7, 8, 15, 16, 17, 100])
    blocks = bank.opportunity_block(n)
    assert blocks.tolist() == [0, 0, 0, 1, 1, 2, 2, 3, 3, 3]


def test_opportunity_block_matches_matrix_shape():
    n = np.array([[1, 4], [8, 16]])
    blocks = bank.opportunity_block(n)
    assert blocks.shape == n.shape
    assert blocks.tolist() == [[0, 1], [2, 3]]


# ---------------------------------------------------------------------------
# Item hierarchy
# ---------------------------------------------------------------------------


def test_flat_hierarchy_is_single_level_identity():
    hier = bank.flat_hierarchy(5)
    assert hier.n_items == 5
    assert len(hier.level_paths) == 1
    assert hier.level_sizes == (5,)
    assert hier.level_paths[0].tolist() == [0, 1, 2, 3, 4]


def test_build_item_hierarchy_factorizes_non_contiguous_ids():
    raw = np.array([10, 20, 10, 30])
    hier = bank.build_item_hierarchy([raw])
    assert hier.level_sizes == (3,)
    # Items 0 and 2 share raw id 10, so they must share a factorized id.
    assert hier.level_paths[0][0] == hier.level_paths[0][2]
    assert hier.level_paths[0][1] != hier.level_paths[0][0]


def test_build_item_hierarchy_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        bank.build_item_hierarchy([np.arange(3), np.arange(4)])


def test_build_kdd_hierarchy_qualifies_repeated_problem_and_step_names():
    """Two different hierarchies both containing a "P1"/"S1" step must be
    kept distinct (raw problem/step names are not globally unique)."""
    hierarchy_id = np.array(["H1", "H1", "H2", "H2"])
    problem_name = np.array(["P1", "P1", "P1", "P1"])
    step_name = np.array(["S1", "S2", "S1", "S2"])
    hier = bank.build_kdd_hierarchy(hierarchy_id, problem_name, step_name)
    assert hier.level_sizes == (2, 2, 4)  # 2 hierarchies, 2 qualified problems, 4 steps
    # The two "P1" problems (under H1 and H2) must be distinct level-1 ids.
    assert hier.level_paths[1][0] != hier.level_paths[1][2]
    # Same-hierarchy same-problem rows share the level-1 id.
    assert hier.level_paths[1][0] == hier.level_paths[1][1]
    # Every step is unique (4 distinct step ids).
    assert len(set(hier.level_paths[2].tolist())) == 4


# ---------------------------------------------------------------------------
# Learner-cohort split
# ---------------------------------------------------------------------------


def test_split_learners_is_disjoint_and_covers_all():
    calib, analysis = bank.split_learners(101, seed=0, calib_frac=0.5)
    assert set(calib.tolist()).isdisjoint(set(analysis.tolist()))
    assert set(calib.tolist()) | set(analysis.tolist()) == set(range(101))
    assert len(calib) == 50 or len(calib) == 51  # round(101 * 0.5)


def test_split_learners_is_seed_reproducible():
    c1, a1 = bank.split_learners(50, seed=7)
    c2, a2 = bank.split_learners(50, seed=7)
    assert np.array_equal(c1, c2)
    assert np.array_equal(a1, a2)


# ---------------------------------------------------------------------------
# Calibration rows: opportunity-index construction
# ---------------------------------------------------------------------------


class _Log:
    """Minimal `LearnerRecord`-shaped stub."""


def test_build_calibration_rows_single_tag_sequential_opportunities():
    log = _Log()
    log.item_ids = np.array([5, 6, 5, 7])
    log.responses = np.array([1, 0, 1, 1], dtype=np.int8)
    log.tag_ids = np.array([[0], [1], [0], [0]])
    log.tag_mask = np.ones((4, 1), dtype=bool)

    rows = bank.build_calibration_rows([log])
    assert rows.opportunity[:, 0].tolist() == [1, 1, 2, 3]
    assert rows.block_id[:, 0].tolist() == [0, 0, 0, 0]


def test_build_calibration_rows_multi_tag_cross_slot_ordering():
    """Regression test: a KC appearing in different tag-slot COLUMNS
    across rows must still be counted in row (time) order, not per-slot-
    column order (the bug caught during build: a slot-by-slot vectorized
    pass silently loses cross-slot ordering)."""
    log = _Log()
    log.item_ids = np.array([0, 1, 2, 3])
    log.responses = np.array([1, 0, 1, 0], dtype=np.int8)
    # t0: item0 tags=[kc0, kc1]; t1: item1 tags=[kc0]; t2: item2 tags=[kc1, kc2]
    # (kc1 in slot 1 here, slot 0 at t0); t3: item3 tags=[kc0, kc2].
    log.tag_ids = np.array([[0, 1], [0, -1], [1, 2], [0, 2]])
    log.tag_mask = np.array([[True, True], [True, False], [True, True], [True, True]])

    rows = bank.build_calibration_rows([log])
    expected = np.array(
        [
            [1, 1],  # t0: kc0 opp1, kc1 opp1
            [2, 0],  # t1: kc0 opp2
            [2, 1],  # t2: kc1 opp2 (its 2nd occurrence, despite slot change), kc2 opp1
            [3, 2],  # t3: kc0 opp3, kc2 opp2
        ]
    )
    assert np.array_equal(rows.opportunity, expected)


def test_build_calibration_rows_across_multiple_learners_independent_counters():
    l0 = _Log()
    l0.item_ids = np.array([0, 0])
    l0.responses = np.array([1, 1], dtype=np.int8)
    l0.tag_ids = np.array([[0], [0]])
    l0.tag_mask = np.ones((2, 1), dtype=bool)

    l1 = _Log()
    l1.item_ids = np.array([0])
    l1.responses = np.array([0], dtype=np.int8)
    l1.tag_ids = np.array([[0]])
    l1.tag_mask = np.ones((1, 1), dtype=bool)

    rows = bank.build_calibration_rows([l0, l1])
    assert rows.opportunity[:, 0].tolist() == [1, 2, 1]  # learner 1 restarts at 1
    assert rows.learner_idx.tolist() == [0, 0, 1]


def test_build_calibration_rows_empty_learner_list():
    rows = bank.build_calibration_rows([])
    assert len(rows) == 0


# ---------------------------------------------------------------------------
# Leaf eligibility / parent-seen
# ---------------------------------------------------------------------------


def test_compute_leaf_eligibility_flat_hierarchy_always_true():
    elig = bank.compute_leaf_eligibility(np.array([0, 0, 1]), n_items=3, floor=None)
    assert elig.tolist() == [True, True, True]


def test_compute_leaf_eligibility_floor_applies():
    item_ids = np.array([0] * 25 + [1] * 5)  # item 0: 25 responses, item 1: 5
    elig = bank.compute_leaf_eligibility(item_ids, n_items=2, floor=20)
    assert elig.tolist() == [True, False]


def test_compute_parent_seen_flat_hierarchy_is_item_level():
    hier = bank.flat_hierarchy(3)
    seen = bank.compute_parent_seen(hier, np.array([0, 0, 1]))
    assert seen.tolist() == [True, True, False]


def test_compute_parent_seen_hierarchical_uses_problem_level():
    hierarchy_id = np.array(["H1", "H1", "H1"])
    problem_name = np.array(["P1", "P1", "P1"])
    step_name = np.array(["S1", "S2", "S3"])
    hier = bank.build_kdd_hierarchy(hierarchy_id, problem_name, step_name)
    # Only step S1 (item 0) seen in calibration, but S2/S3 share its problem.
    seen = bank.compute_parent_seen(hier, np.array([0, 0, 0]))
    assert seen.tolist() == [True, True, True]


# ---------------------------------------------------------------------------
# End-to-end calibration: flat hierarchy, moderate scale, signal recovery
# ---------------------------------------------------------------------------


def _make_flat_dataset(rng, n_items, n_kcs, n_learners, t_lo, t_hi):
    items_per_kc = n_items // n_kcs
    primary_kc = np.repeat(np.arange(n_kcs), items_per_kc)
    b_true = rng.normal(0.0, 1.0, size=n_items)
    learners = []
    for _ in range(n_learners):
        T = rng.integers(t_lo, t_hi)
        theta_i = rng.normal(0.0, 1.0)
        item_ids = rng.integers(0, n_items, size=T)
        tag_ids = primary_kc[item_ids].reshape(-1, 1)
        tag_mask = np.ones((T, 1), dtype=bool)
        p = 1.0 / (1.0 + np.exp(-(theta_i - b_true[item_ids])))
        responses = (rng.random(T) < p).astype(np.int8)
        log = _Log()
        log.item_ids, log.responses, log.tag_ids, log.tag_mask = (
            item_ids,
            responses,
            tag_ids,
            tag_mask,
        )
        learners.append(log)
    return learners, b_true, primary_kc


@pytest.fixture(scope="module")
def flat_dataset():
    rng = np.random.default_rng(0)
    learners, b_true, primary_kc = _make_flat_dataset(
        rng, n_items=80, n_kcs=8, n_learners=1000, t_lo=20, t_hi=35
    )
    return learners, b_true, primary_kc


def test_calibrate_bank_blockwise_recovers_no_growth_difficulty(flat_dataset):
    learners, b_true, _ = flat_dataset
    n_learners = len(learners)
    rows = bank.build_calibration_rows(learners)
    calib, _ = bank.split_learners(n_learners, seed=0)
    hierarchy = bank.flat_hierarchy(len(b_true))
    cfg = bank.BankModelConfig(n_epochs_max=200, lr=0.05, batch_size=8192, patience_epochs=3)

    fit = bank.calibrate_bank(
        rows, n_learners, n_kcs=8, calib_learners=calib, hierarchy=hierarchy,
        hierarchy_spec=bank.FLAT_HIERARCHY_SPEC, growth_mode="blockwise", config=cfg,
    )
    assert np.isfinite(fit.b_hat).all()
    corr = bank.spearman_rank_correlation(fit.b_hat, b_true)
    assert corr > 0.85  # near-ceiling recovery at this scale; official 0.9 bar is a later-stage battery result


def test_calibrate_bank_no_growth_and_linear_modes_run_and_are_finite(flat_dataset):
    learners, b_true, _ = flat_dataset
    n_learners = len(learners)
    rows = bank.build_calibration_rows(learners)
    calib, _ = bank.split_learners(n_learners, seed=0)
    hierarchy = bank.flat_hierarchy(len(b_true))
    cfg = bank.BankModelConfig(n_epochs_max=80, lr=0.05, batch_size=8192, patience_epochs=3)

    for mode in ("none", "linear"):
        fit = bank.calibrate_bank(
            rows, n_learners, n_kcs=8, calib_learners=calib, hierarchy=hierarchy,
            hierarchy_spec=bank.FLAT_HIERARCHY_SPEC, growth_mode=mode, config=cfg,
        )
        assert np.isfinite(fit.b_hat).all()
        assert fit.n_epochs_run > 0


def test_calibrate_bank_rejects_mismatched_hierarchy_spec(flat_dataset):
    learners, b_true, _ = flat_dataset
    n_learners = len(learners)
    rows = bank.build_calibration_rows(learners)
    calib, _ = bank.split_learners(n_learners, seed=0)
    hierarchy = bank.flat_hierarchy(len(b_true))
    with pytest.raises(ValueError):
        bank.calibrate_bank(
            rows, n_learners, n_kcs=8, calib_learners=calib, hierarchy=hierarchy,
            hierarchy_spec=bank.KDD_HIERARCHY_SPEC,  # 3 sigmas, 1-level hierarchy
        )


def test_calibrate_bank_empty_calibration_cohort_is_defensive():
    hierarchy = bank.flat_hierarchy(4)
    rows = bank.build_calibration_rows([])
    fit = bank.calibrate_bank(
        rows, n_learners=0, n_kcs=1, calib_learners=np.array([], dtype=np.int64),
        hierarchy=hierarchy, hierarchy_spec=bank.FLAT_HIERARCHY_SPEC,
    )
    assert fit.converged is True
    assert fit.n_epochs_run == 0
    assert fit.b_hat.shape == (4,)


# ---------------------------------------------------------------------------
# The zero-mean-across-blocks identifiability fix (module docstring note 8)
# ---------------------------------------------------------------------------


def test_blockwise_growth_does_not_degrade_recovery_vs_no_growth(flat_dataset):
    """The blockwise growth-absorption term must not systematically steal
    a KC-level share of item difficulty when there is no true opportunity-
    difficulty confound (build-time regression: an unconstrained per-(KC,
    block) offset is exactly collinear with a KC's mean item difficulty
    and, left unconstrained, corrupted recovery from ~0.94 to ~0.33 at a
    comparable scale)."""
    learners, b_true, _ = flat_dataset
    n_learners = len(learners)
    rows = bank.build_calibration_rows(learners)
    calib, _ = bank.split_learners(n_learners, seed=0)
    hierarchy = bank.flat_hierarchy(len(b_true))
    cfg = bank.BankModelConfig(n_epochs_max=200, lr=0.05, batch_size=8192, patience_epochs=3)

    fit_bw = bank.calibrate_bank(
        rows, n_learners, n_kcs=8, calib_learners=calib, hierarchy=hierarchy,
        hierarchy_spec=bank.FLAT_HIERARCHY_SPEC, growth_mode="blockwise", config=cfg,
    )
    fit_none = bank.calibrate_bank(
        rows, n_learners, n_kcs=8, calib_learners=calib, hierarchy=hierarchy,
        hierarchy_spec=bank.FLAT_HIERARCHY_SPEC, growth_mode="none", config=cfg,
    )
    corr_bw = bank.spearman_rank_correlation(fit_bw.b_hat, b_true)
    corr_none = bank.spearman_rank_correlation(fit_none.b_hat, b_true)
    assert corr_bw > corr_none - 0.1  # comparable recovery, not a large degradation


# ---------------------------------------------------------------------------
# Hierarchical (KDD-shaped) calibration and the exposure floor
# ---------------------------------------------------------------------------


def test_hierarchical_calibration_zeros_out_offsets_below_floor():
    """Two sparse sibling steps (same problem, both below the exposure
    floor) must collapse to an IDENTICAL b_hat (both read as b_H + e_P,
    zero step offset), even though their true difficulties differ --
    this is a structural (not approximate) property of the floor mask."""
    rng = np.random.default_rng(5)
    hierarchy_id = np.array(["H1", "H1", "H1"])
    problem_name = np.array(["P1", "P1", "P1"])
    step_name = np.array(["A", "B", "C"])
    hier = bank.build_kdd_hierarchy(hierarchy_id, problem_name, step_name)
    b_true = np.array([-1.0, 2.0, -2.0])

    n_learners = 200
    learners = []
    for i in range(n_learners):
        theta_i = rng.normal(0.0, 1.0)
        if i < 60:
            item_ids = np.array([0])  # item A: well exposed
        elif i < 65:
            item_ids = np.array([1])  # item B: only 5 total responses (< floor 20)
        elif i < 68:
            item_ids = np.array([2])  # item C: only 3 total responses (< floor 20)
        else:
            item_ids = np.array([0])
        tag_ids = item_ids.reshape(-1, 1)
        tag_mask = np.ones_like(tag_ids, dtype=bool)
        p = 1.0 / (1.0 + np.exp(-(theta_i - b_true[item_ids])))
        responses = (rng.random(len(item_ids)) < p).astype(np.int8)
        log = _Log()
        log.item_ids, log.responses, log.tag_ids, log.tag_mask = (
            item_ids,
            responses,
            tag_ids,
            tag_mask,
        )
        learners.append(log)

    rows = bank.build_calibration_rows(learners)
    calib = np.arange(n_learners)  # everyone in calibration (deterministic exposure counts)
    cfg = bank.BankModelConfig(n_epochs_max=150, lr=0.05, batch_size=4096, patience_epochs=3)
    fit = bank.calibrate_bank(
        rows, n_learners, n_kcs=3, calib_learners=calib, hierarchy=hier,
        hierarchy_spec=bank.KDD_HIERARCHY_SPEC, growth_mode="none", config=cfg,
    )
    assert fit.eligible_leaf.tolist() == [True, False, False]
    assert fit.b_hat[1] == pytest.approx(fit.b_hat[2], abs=1e-5)


# ---------------------------------------------------------------------------
# FrozenBank
# ---------------------------------------------------------------------------


def test_freeze_bank_difficulty_and_is_calibrated(flat_dataset):
    learners, b_true, _ = flat_dataset
    n_learners = len(learners)
    rows = bank.build_calibration_rows(learners)
    calib, _ = bank.split_learners(n_learners, seed=0)
    hierarchy = bank.flat_hierarchy(len(b_true))
    cfg = bank.BankModelConfig(n_epochs_max=60, lr=0.05, batch_size=8192, patience_epochs=3)
    fit = bank.calibrate_bank(
        rows, n_learners, n_kcs=8, calib_learners=calib, hierarchy=hierarchy,
        hierarchy_spec=bank.FLAT_HIERARCHY_SPEC, growth_mode="blockwise", config=cfg,
    )
    frozen = bank.freeze_bank(fit)
    assert np.array_equal(frozen.difficulty(np.arange(len(b_true))), fit.b_hat)
    assert frozen.is_calibrated(0) == fit.problem_seen[0]
    assert frozen.growth_mode == "blockwise"


def test_frozen_bank_marks_never_seen_item_uncalibrated():
    hier = bank.flat_hierarchy(3)
    fit = bank.BankFitResult(
        hierarchy=hier,
        growth_mode="none",
        b_hat=np.zeros(3),
        eligible_leaf=np.array([True, True, False]),
        problem_seen=np.array([True, True, False]),
        calib_exposure_count=np.array([10, 5, 0]),
        converged=True,
        n_epochs_run=1,
        final_data_nll=0.0,
    )
    frozen = bank.freeze_bank(fit)
    assert frozen.is_calibrated(2) == False  # noqa: E712 (numpy bool clarity)
    assert frozen.is_calibrated(0) == True  # noqa: E712


# ---------------------------------------------------------------------------
# Battery arm 10: synthetic recovery + tri-spec stability
# ---------------------------------------------------------------------------


def test_synthetic_bank_recovery_check_shape_and_floor(flat_dataset):
    learners, b_true, _ = flat_dataset
    n_learners = len(learners)
    rows = bank.build_calibration_rows(learners)
    calib, _ = bank.split_learners(n_learners, seed=0)
    hierarchy = bank.flat_hierarchy(len(b_true))
    cfg = bank.BankModelConfig(n_epochs_max=150, lr=0.05, batch_size=8192, patience_epochs=3)
    fit = bank.calibrate_bank(
        rows, n_learners, n_kcs=8, calib_learners=calib, hierarchy=hierarchy,
        hierarchy_spec=bank.FLAT_HIERARCHY_SPEC, growth_mode="blockwise", config=cfg,
    )
    check = bank.synthetic_bank_recovery_check(fit, b_true)
    assert set(check.keys()) == {"rank_corr", "n_items", "passed"}
    assert check["n_items"] <= len(b_true)
    assert check["rank_corr"] > 0.5

    # A prohibitively high exposure floor should shrink the eligible set.
    strict = bank.synthetic_bank_recovery_check(fit, b_true, min_calib_exposure=10_000)
    assert strict["n_items"] == 0
    assert strict["passed"] is False


def test_tri_spec_refit_and_pairwise_stability(flat_dataset):
    learners, b_true, _ = flat_dataset
    n_learners = len(learners)
    rows = bank.build_calibration_rows(learners)
    calib, _ = bank.split_learners(n_learners, seed=0)
    hierarchy = bank.flat_hierarchy(len(b_true))
    cfg = bank.BankModelConfig(n_epochs_max=200, lr=0.05, batch_size=8192, patience_epochs=3)

    tri = bank.tri_spec_refit(
        rows, n_learners, n_kcs=8, calib_learners=calib, hierarchy=hierarchy,
        hierarchy_spec=bank.FLAT_HIERARCHY_SPEC, config=cfg,
    )
    assert tri.no_growth.growth_mode == "none"
    assert tri.linear.growth_mode == "linear"
    assert tri.blockwise.growth_mode == "blockwise"

    stability = bank.bank_pairwise_stability(tri, min_calib_responses=20)
    assert set(stability.keys()) == {
        "n_items",
        "no_growth_vs_linear",
        "no_growth_vs_blockwise",
        "linear_vs_blockwise",
        "passed",
    }
    # RB0's pairwise bar (design section 5.2/battery arm 10): >= 0.95.
    assert stability["no_growth_vs_blockwise"] >= 0.95
    assert stability["passed"] is True


# ---------------------------------------------------------------------------
# Rank correlation helper
# ---------------------------------------------------------------------------


def test_spearman_rank_correlation_perfect_and_reversed():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert bank.spearman_rank_correlation(x, x) == pytest.approx(1.0)
    assert bank.spearman_rank_correlation(x, x[::-1]) == pytest.approx(-1.0)


def test_spearman_rank_correlation_ties_and_constant_input():
    x = np.array([1.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 2.0, 3.0])
    corr = bank.spearman_rank_correlation(x, y)
    assert -1.0 <= corr <= 1.0
    constant = np.array([5.0, 5.0, 5.0])
    assert np.isnan(bank.spearman_rank_correlation(constant, x[:3]))


# ---------------------------------------------------------------------------
# Interop with stage 1's synthetic generator (duck-typed `LearnerRecord`)
# ---------------------------------------------------------------------------


def test_interop_with_synth_learner_log():
    """`build_calibration_rows` must accept `synth.LearnerLog` objects
    directly (structural typing, no import dependency required)."""
    from kt_mirt.growth import synth

    twin = synth.generate_twin("syn_ng", synth.SYN_DEV, seed=0)
    rows = bank.build_calibration_rows(twin.learners)
    assert len(rows) == sum(len(log.item_ids) for log in twin.learners)
    assert rows.kc_ids.shape[1] == twin.item_bank.max_arity
