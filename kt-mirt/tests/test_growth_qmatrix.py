"""Tests for `kt_mirt.growth.qmatrix`: the item-to-KC expansion policy,
pure-anchor identification, and the circularity guard
(`_planning/design/a4_design.md` v1.1, sections 2.5, 6)."""

from __future__ import annotations

import numpy as np
import pytest

from kt_mirt.growth import qmatrix


# ---------------------------------------------------------------------------
# build_qmatrix: shape, padding, arity
# ---------------------------------------------------------------------------


def test_build_qmatrix_pads_ragged_kc_lists():
    item_keys = ["s0", "s1", "s2"]
    kc_lists = [[0], [0, 1], []]
    kc_names = ["kc_a", "kc_b"]
    qm = qmatrix.build_qmatrix(item_keys, kc_lists, kc_names)
    assert qm.n_items == 3
    assert qm.n_kcs == 2
    assert qm.arity_max == 2
    assert qm.kc_ids[0].tolist() == [0, -1]
    assert qm.kc_ids[1].tolist() == [0, 1]
    assert qm.kc_ids[2].tolist() == [-1, -1]
    assert qm.kc_mask[0].tolist() == [True, False]
    assert qm.kc_mask[1].tolist() == [True, True]
    assert qm.kc_mask[2].tolist() == [False, False]


def test_build_qmatrix_arity_property():
    item_keys = ["s0", "s1"]
    kc_lists = [[0], [0, 1]]
    qm = qmatrix.build_qmatrix(item_keys, kc_lists, ["a", "b"])
    assert qm.arity(0) == 1
    assert qm.arity(1) == 2
    assert qm.arity(np.array([0, 1])).tolist() == [1, 2]


def test_build_qmatrix_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        qmatrix.build_qmatrix(["s0", "s1"], [[0]], ["a"])


def test_build_qmatrix_rejects_arity_max_too_small():
    with pytest.raises(ValueError):
        qmatrix.build_qmatrix(["s0"], [[0, 1]], ["a", "b"], arity_max=1)


def test_build_qmatrix_empty_input():
    qm = qmatrix.build_qmatrix([], [], [])
    assert qm.n_items == 0
    assert qm.n_kcs == 0
    assert qm.arity_max == 1  # minimum width, no items to size from


# ---------------------------------------------------------------------------
# Circularity guard (the ASSISTments lesson)
# ---------------------------------------------------------------------------


def test_check_no_circularity_passes_on_disjoint_vocabularies():
    # No exception.
    qmatrix.check_no_circularity(["H1||P1||S1", "H1||P1||S2"], ["Using simple numbers", "Find Y"])


def test_check_no_circularity_fires_on_total_identity():
    """The ASSISTments failure mode: the item-key vocabulary IS the
    KC-name vocabulary (a skill id doubling as the item id)."""
    item_keys = ["skill_a", "skill_b", "skill_c"]
    kc_names = ["skill_a", "skill_b", "skill_c", "skill_d"]
    with pytest.raises(qmatrix.CircularQMatrixError):
        qmatrix.check_no_circularity(item_keys, kc_names)


def test_check_no_circularity_tolerates_incidental_overlap():
    """A handful of coincidental string collisions (well under the 50%
    overlap threshold) must NOT raise -- the guard targets total-identity
    namespace collapse, not any overlap at all."""
    item_keys = ["H1||P1||S1", "H1||P1||S2", "H1||P1||S3", "coincidence"]
    kc_names = ["coincidence", "Find Y"]
    qmatrix.check_no_circularity(item_keys, kc_names)  # no raise


def test_build_qmatrix_runs_circularity_guard_by_default():
    item_keys = ["skill_a", "skill_b"]
    kc_lists = [[0], [1]]
    kc_names = ["skill_a", "skill_b"]
    with pytest.raises(qmatrix.CircularQMatrixError):
        qmatrix.build_qmatrix(item_keys, kc_lists, kc_names)


def test_build_qmatrix_can_skip_circularity_guard():
    item_keys = ["skill_a", "skill_b"]
    kc_lists = [[0], [1]]
    kc_names = ["skill_a", "skill_b"]
    qm = qmatrix.build_qmatrix(item_keys, kc_lists, kc_names, skip_circularity_check=True)
    assert qm.n_items == 2


def test_check_no_circularity_noop_on_empty_vocab():
    qmatrix.check_no_circularity([], ["a"])
    qmatrix.check_no_circularity(["a"], [])


# ---------------------------------------------------------------------------
# Pure-anchor identification
# ---------------------------------------------------------------------------


def test_pure_anchor_item_mask():
    qm = qmatrix.build_qmatrix(
        ["s0", "s1", "s2", "s3"], [[0], [0, 1], [1], []], ["a", "b"],
    )
    mask = qmatrix.pure_anchor_item_mask(qm)
    assert mask.tolist() == [True, False, True, False]


def test_pure_anchor_item_mask_empty():
    qm = qmatrix.build_qmatrix([], [], [])
    assert qmatrix.pure_anchor_item_mask(qm).tolist() == []


def test_pure_anchor_kc_ids():
    qm = qmatrix.build_qmatrix(
        ["s0", "s1", "s2"], [[0], [0, 1], [1]], ["a", "b"],
    )
    pure = qmatrix.pure_anchor_kc_ids(qm)
    assert sorted(pure.tolist()) == [0, 1]  # s0 -> kc0 (pure), s2 -> kc1 (pure); s1 impure


def test_pure_anchor_stats_matches_triage_field_shape():
    """Reproduces `triage_common.pure_anchor_and_arity_stats`'s exact
    field set on a small hand-built example, checked against hand
    computation."""
    # KC0 gets 3 pure items (s0,s1,s2); KC1 gets 1 pure item (s4); s3 is
    # impure (tags both KC0 and KC1).
    item_keys = ["s0", "s1", "s2", "s3", "s4"]
    kc_lists = [[0], [0], [0], [0, 1], [1]]
    kc_names = ["kc0", "kc1"]
    qm = qmatrix.build_qmatrix(item_keys, kc_lists, kc_names)
    stats = qmatrix.pure_anchor_stats(qm, min_pure=3)
    assert stats["n_items"] == 5
    assert stats["n_kc_in_item_bank"] == 2
    assert stats["n_pure_items_total"] == 4  # s0,s1,s2,s4
    assert stats["min_pure_items_gate"] == 3
    # Only kc0 clears >=3 pure items (kc1 has 1 pure item).
    assert stats["frac_kc_with_ge_min_pure_items"] == pytest.approx(0.5)
    assert stats["arity_mean"] == pytest.approx((1 + 1 + 1 + 2 + 1) / 5)
    assert stats["arity_max"] == 2


def test_pure_anchor_stats_empty_qmatrix():
    qm = qmatrix.build_qmatrix([], [], [])
    stats = qmatrix.pure_anchor_stats(qm)
    assert stats["n_items"] == 0
    assert stats["n_kc_in_item_bank"] == 0
    assert stats["arity_mean"] is None
    assert stats["frac_kc_with_ge_min_pure_items"] is None
