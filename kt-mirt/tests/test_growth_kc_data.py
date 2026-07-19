"""Tests for `kt_mirt.growth.kc_data`: the KDD Algebra 2008-2009 chunked
real-bed loader (`_planning/design/a4_design.md` v1.1 section 6).

Unit tests build a small, synthetic KDD-FORMAT fixture in-test (never
commit real data, per the harness instructions) that exercises: multi-KC
``~~``-split tags, an untagged row, a KC/Opportunity length mismatch, an
invalid Correct-First-Attempt value, out-of-file-order chronology (a
later ``Row`` can have an earlier timestamp), and a same-timestamp tie
broken by ``Row``. A separate integration test is gated to skip when the
real raw file is absent.
"""

from __future__ import annotations

import csv as csv_mod
from pathlib import Path

import numpy as np
import pytest

from kt_mirt.growth import kc_data, qmatrix


# ---------------------------------------------------------------------------
# Synthetic KDD-format fixture
# ---------------------------------------------------------------------------

_HEADER = (
    "Row\tAnon Student Id\tProblem Hierarchy\tProblem Name\tStep Name\t"
    "Step Start Time\tCorrect First Attempt\tKC(KTracedSkills)\tOpportunity(KTracedSkills)\n"
)

# Row, student, hierarchy, problem, step, time, CFA, KC, Opportunity
_ROWS = [
    ("10", "stuA", "H1", "P1", "S1", "2020-01-01 10:00:00.0", "1", "skillX", "1"),
    ("1", "stuA", "H1", "P1", "S2", "2020-01-01 09:00:00.0", "0", "skillX~~skillY", "1~~1"),
    ("2", "stuA", "H1", "P1", "S1", "2020-01-01 10:00:00.0", "1", "skillX", "2"),  # ties R1 on time, Row breaks tie
    ("3", "stuB", "H1", "P1", "S3", "2020-01-01 09:30:00.0", "1", "", ""),  # untagged
    ("4", "stuB", "H2", "P2", "S4", "2020-01-01 09:40:00.0", "1", "skillY", "1~~2"),  # length mismatch
    ("6", "stuA", "H1", "P1", "S1", "2020-01-01 10:30:00.0", "X", "skillX", "1"),  # invalid CFA
    ("7", "stuC", "H1", "P3", "S5", "2020-01-01 08:00:00.0", "1", "skillZ", "1"),
    # invalid CFA AND a KC/Opportunity length mismatch -- must be excluded
    # from n_kc_opp_mismatch too (triage_kdd.py's mismatch count is scoped
    # to valid_cfa & has_kc, not to has_kc alone; see kc_data.py note 8 /
    # realbed_bridge_notes.md section 3).
    ("8", "stuC", "H1", "P3", "S6", "2020-01-01 08:10:00.0", "Y", "skillZ~~skillW", "1"),
]


def _write_fixture(path: Path) -> None:
    lines = [_HEADER]
    for row in _ROWS:
        lines.append("\t".join(row) + "\n")
    path.write_text("".join(lines), encoding="utf-8")


def _write_rows(path: Path, rows) -> None:
    lines = [_HEADER]
    for row in rows:
        lines.append("\t".join(row) + "\n")
    path.write_text("".join(lines), encoding="utf-8")


@pytest.fixture
def fixture_path(tmp_path) -> Path:
    p = tmp_path / "kdd_fixture.txt"
    _write_fixture(p)
    return p


# ---------------------------------------------------------------------------
# LoadStats: row-accounting fidelity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunksize", [1, 2, 3, 100])
def test_load_stats_row_accounting(fixture_path, chunksize):
    """Row counts must be identical regardless of chunk boundaries."""
    result = kc_data.load_kdd_kc_traced(fixture_path, chunksize=chunksize)
    stats = result.stats
    assert stats.n_rows_read == 8
    assert stats.n_rows_invalid_cfa == 2  # R6, R8
    # Matches triage_kdd's own scoping exactly (valid_cfa & has_kc &
    # length_ok): R1,R2,R3,R7 usable; R5 is valid-CFA but length-mismatched
    # (excluded here, unlike the pre-fix count); R4 carries no KC at all;
    # R6/R8 are invalid-CFA (excluded regardless of their KC tagging).
    assert stats.n_rows_has_kc == 4
    # Scoped to valid_cfa & has_kc & ~length_ok (triage_kdd's own scope):
    # only R5 (valid CFA, mismatched). R8 is ALSO length-mismatched but has
    # an invalid CFA, so it must NOT be counted here -- this is the exact
    # distinction the pre-fix code got wrong.
    assert stats.n_kc_opp_mismatch == 1
    assert stats.n_rows_used == 6  # every valid-CFA row with a resolvable student id (note 1), i.e. all but R6/R8
    assert stats.n_rows_missing_student_id == 0
    assert stats.n_rows_bad_row_num == 0
    assert stats.n_rows_missing_timestamp == 0


def test_missing_student_id_row_excluded_and_counted(tmp_path):
    """A row with no resolvable Anon Student Id cannot be attributed to any
    LearnerLog; it must be counted and dropped, not silently folded into a
    bogus "NaN" learner (module docstring note 8)."""
    rows = [
        ("1", "stuA", "H1", "P1", "S1", "2020-01-01 09:00:00.0", "1", "skillX", "1"),
        ("2", "", "H1", "P1", "S2", "2020-01-01 09:05:00.0", "1", "skillX", "1"),  # missing student id
    ]
    p = tmp_path / "missing_student.txt"
    _write_rows(p, rows)
    result = kc_data.load_kdd_kc_traced(p, chunksize=10)
    assert result.stats.n_rows_read == 2
    assert result.stats.n_rows_invalid_cfa == 0
    assert result.stats.n_rows_missing_student_id == 1
    assert result.stats.n_rows_used == 1
    assert result.n_learners == 1
    assert result.learners[0].item_ids.shape[0] == 1


def test_unparseable_row_number_is_counted_not_silently_cast(tmp_path):
    """An unparseable ``Row`` value is still a real response (kept as an
    interaction) but must not silently corrupt the same-timestamp tie-break
    via an implementation-defined NaN->int64 cast (module docstring note
    8): it is counted and sorts deterministically LAST in its tie group."""
    rows = [
        ("1", "stuA", "H1", "P1", "S1", "2020-01-01 10:00:00.0", "1", "skillX", "1"),
        ("garbage", "stuA", "H1", "P1", "S2", "2020-01-01 10:00:00.0", "0", "skillX", "1"),
    ]
    p = tmp_path / "bad_row.txt"
    _write_rows(p, rows)
    result = kc_data.load_kdd_kc_traced(p, chunksize=10)
    assert result.stats.n_rows_bad_row_num == 1
    assert result.stats.n_rows_used == 2
    stu_a = result.learners[0]
    assert stu_a.item_ids.shape[0] == 2
    # Row=1 (valid) sorts before the unparseable Row within the same
    # student/timestamp tie group.
    assert stu_a.responses.tolist() == [1, 0]


def test_missing_timestamp_is_counted_and_sorts_first(tmp_path):
    """A missing ``Step Start Time`` already fell back to "" (sorts before
    any real timestamp); it now also gets a diagnostic counter (module
    docstring note 8)."""
    rows = [
        ("1", "stuA", "H1", "P1", "S1", "", "1", "skillX", "1"),  # missing timestamp
        ("2", "stuA", "H1", "P1", "S2", "2020-01-01 09:00:00.0", "0", "skillX", "1"),
    ]
    p = tmp_path / "missing_ts.txt"
    _write_rows(p, rows)
    result = kc_data.load_kdd_kc_traced(p, chunksize=10)
    assert result.stats.n_rows_missing_timestamp == 1
    stu_a = result.learners[0]
    assert stu_a.responses.tolist() == [1, 0]  # empty-string timestamp sorts first


def test_n_learners_and_n_kcs(fixture_path):
    result = kc_data.load_kdd_kc_traced(fixture_path, chunksize=2)
    assert result.n_learners == 3  # stuA, stuB, stuC
    assert result.n_kcs == 3  # skillX, skillY, skillZ
    assert set(result.kc_names) == {"skillX", "skillY", "skillZ"}


# ---------------------------------------------------------------------------
# Chronological ordering (timestamp primary, Row tie-break)
# ---------------------------------------------------------------------------


def test_learner_a_is_chronologically_ordered_with_row_tiebreak(fixture_path):
    result = kc_data.load_kdd_kc_traced(fixture_path, chunksize=2)
    # stuA is whichever learner has 3 events (R2 09:00, R3 10:00/Row2, R1 10:00/Row10).
    stu_a = next(l for l in result.learners if l.item_ids.shape[0] == 3)
    # Chronological + Row-tiebreak order: R2 (09:00), then R3 (Row2), then R1 (Row10).
    assert stu_a.responses.tolist() == [0, 1, 1]  # R2=0, R3=1, R1=1
    # R2 (H1||P1||S2) is a distinct item from R1/R3 (H1||P1||S1); R3 and R1 are the SAME item.
    assert stu_a.item_ids[1] == stu_a.item_ids[2]
    assert stu_a.item_ids[0] != stu_a.item_ids[1]


def test_learner_a_tag_ids_reflect_multi_kc_and_single_kc_rows(fixture_path):
    result = kc_data.load_kdd_kc_traced(fixture_path, chunksize=2)
    stu_a = next(l for l in result.learners if l.item_ids.shape[0] == 3)
    kc_name_to_id = {name: i for i, name in enumerate(result.kc_names)}
    # First event (R2): tagged with both skillX and skillY.
    tags0 = set(stu_a.tag_ids[0][stu_a.tag_mask[0]].tolist())
    assert tags0 == {kc_name_to_id["skillX"], kc_name_to_id["skillY"]}
    # Second/third events (R3, R1): tagged with skillX only.
    for row in (1, 2):
        tags = set(stu_a.tag_ids[row][stu_a.tag_mask[row]].tolist())
        assert tags == {kc_name_to_id["skillX"]}


def test_untagged_and_mismatched_rows_carry_all_false_mask(fixture_path):
    result = kc_data.load_kdd_kc_traced(fixture_path, chunksize=2)
    stu_b = next(l for l in result.learners if l.item_ids.shape[0] == 2)
    # Both of stuB's events (R4 untagged, R5 length-mismatched) must be
    # present as interactions (note 1/2) with an all-False tag mask.
    assert stu_b.responses.tolist() == [1, 1]
    assert not stu_b.tag_mask.any()


def test_invalid_cfa_row_excluded_entirely(fixture_path):
    result = kc_data.load_kdd_kc_traced(fixture_path, chunksize=2)
    stu_a = next(l for l in result.learners if l.item_ids.shape[0] == 3)
    # R6 (invalid CFA) must never appear: stuA has exactly 3 events, not 4.
    assert stu_a.item_ids.shape[0] == 3


def test_learner_ids_are_positional_matching_synth_convention(fixture_path):
    result = kc_data.load_kdd_kc_traced(fixture_path, chunksize=2)
    for i, log in enumerate(result.learners):
        assert log.learner == i


def test_learner_timestamps_aligned_with_item_ids(fixture_path):
    result = kc_data.load_kdd_kc_traced(fixture_path, chunksize=2)
    for log, ts in zip(result.learners, result.learner_timestamps):
        assert ts.shape[0] == log.item_ids.shape[0]
    stu_a = next(
        (l, t) for l, t in zip(result.learners, result.learner_timestamps) if l.item_ids.shape[0] == 3
    )
    log, ts = stu_a
    assert ts.tolist() == [
        "2020-01-01 09:00:00.0", "2020-01-01 10:00:00.0", "2020-01-01 10:00:00.0",
    ]


# ---------------------------------------------------------------------------
# Item hierarchy + Q-matrix
# ---------------------------------------------------------------------------


def test_item_hierarchy_has_expected_levels(fixture_path):
    result = kc_data.load_kdd_kc_traced(fixture_path, chunksize=2)
    hier = result.item_hierarchy
    assert hier.n_items == 5  # S1, S2, S3, S4, S5 (distinct steps)
    assert len(hier.level_sizes) == 3  # hierarchy > problem > step
    # H1/P1 covers S1, S2, S3; H2/P2 covers S4; H1/P3 covers S5.
    assert hier.level_sizes[0] == 2  # H1, H2
    assert hier.level_sizes[1] == 3  # H1||P1, H2||P2, H1||P3
    assert hier.level_sizes[2] == 5  # 5 distinct steps


def test_qmatrix_pure_anchor_and_untagged_items(fixture_path):
    result = kc_data.load_kdd_kc_traced(fixture_path, chunksize=2)
    qm = result.qmatrix
    assert qm.n_items == 5
    pure_mask = qmatrix.pure_anchor_item_mask(qm)
    # S1 (skillX only) and S5 (skillZ only) are pure anchors; S2 is
    # multi-tagged; S3/S4 never got a usable tag at all (arity 0).
    assert int(pure_mask.sum()) == 2


def test_qmatrix_circularity_guard_not_tripped_on_real_shaped_keys(fixture_path):
    """Item keys ("H1||P1||S1"-style) and KC names ("skillX"-style) are
    disjoint vocabularies -- the ASSISTments guard must stay silent."""
    result = kc_data.load_kdd_kc_traced(fixture_path, chunksize=2)
    qmatrix.check_no_circularity(result.qmatrix.item_keys, result.qmatrix.kc_names)


# ---------------------------------------------------------------------------
# Growth-package downstream compatibility: bank.build_calibration_rows
# ---------------------------------------------------------------------------


def test_learners_feed_build_calibration_rows(fixture_path):
    """The exact consumer contract `kc_data.py` exists to satisfy
    (module docstring): `bank.build_calibration_rows` must accept the
    returned `LearnerLog`s with no adaptation."""
    from kt_mirt.growth.bank import build_calibration_rows

    result = kc_data.load_kdd_kc_traced(fixture_path, chunksize=2)
    rows = build_calibration_rows(result.learners)
    assert len(rows) == sum(l.item_ids.shape[0] for l in result.learners) == 6


# ---------------------------------------------------------------------------
# Empty file
# ---------------------------------------------------------------------------


def test_empty_file_returns_empty_result(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text(_HEADER, encoding="utf-8")
    result = kc_data.load_kdd_kc_traced(p, chunksize=10)
    assert result.n_learners == 0
    assert result.n_kcs == 0
    assert result.learners == []
    assert result.stats.n_rows_read == 0


# ---------------------------------------------------------------------------
# Integration test: the real file, gated to skip when absent
# ---------------------------------------------------------------------------

_REAL_KDD_PATH = Path(__file__).resolve().parents[2] / "data" / "kdd" / "algebra_2008_2009_train.txt"


@pytest.mark.skipif(not _REAL_KDD_PATH.exists(), reason="real KDD raw file not present locally")
def test_real_kdd_file_loads_a_small_prefix():
    """Light smoke test on a small row prefix of the real file (the full
    acceptance run lives in `_planning/design/realbed_bridge_notes.md`,
    not in the test suite)."""
    result = kc_data.load_kdd_kc_traced(_REAL_KDD_PATH, chunksize=10_000, nrows=20_000)
    assert result.stats.n_rows_read == 20_000
    assert result.n_learners > 0
    assert result.n_kcs > 0
    assert result.item_hierarchy.n_items > 0
    assert len(result.learners) == result.n_learners
    for log in result.learners:
        assert log.item_ids.shape[0] == log.responses.shape[0] == log.tag_ids.shape[0] == log.tag_mask.shape[0]
