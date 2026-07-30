"""Tests for `kt_mirt.growth.junyi_data`: the Junyi Academy 2015 chunked
real-bed loader (`_planning/design/a4_design.md` v1.1).

Unit tests build a small, synthetic Junyi15-FORMAT fixture pair in-test
(never commit real data, per the harness instructions): an exercise table
with a dropped exact-duplicate row, a missing-topic item, a self-loop
prerequisite, and a topic attempted by nobody in the log; a problem log
exercising a missing user id, an exercise absent from the table, an
invalid ``correct`` value, a missing timestamp (sorts first), and a
same-timestamp/-user tie broken by file row order. A separate integration
test is gated to skip when the real raw files are absent.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from kt_mirt.growth import junyi_data, qmatrix


# ---------------------------------------------------------------------------
# Synthetic Junyi15-format fixtures
# ---------------------------------------------------------------------------

# Item catalog (post-dedup order): exA(t1), exB(t1), exC(t2), exD(no topic),
# exE(t3, never attempted in the log below), exG(t1, self-loop prereq). The
# second "exA" row is an exact-duplicate NAME, dropped by
# drop_duplicates(subset="name", keep="first").
_EX_ROWS = [
    ("exA", "", "t1"),
    ("exB", "exA", "t1"),
    ("exC", "exB", "t2"),
    ("exD", "", ""),
    ("exE", "exA,exC", "t3"),
    ("exA", "", "t1"),  # duplicate name -> dropped, keep first
    ("exG", "exG", "t1"),  # self-loop
]


def _write_exercise_table(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "prerequisites", "topic"])
        w.writerows(_EX_ROWS)


_LOG_HEADER = "user_id,exercise,correct,time_done\n"
# (user_id, exercise, correct, time_done) in file order -- row_pos tie-break.
_LOG_ROWS = [
    ("u1", "exB", "false", "50"),  # row0
    ("u1", "exA", "true", "100"),  # row1 (ties row2 on user+time; file order wins)
    ("u1", "exD", "true", "100"),  # row2 (untagged: exD has no topic)
    ("u2", "exC", "true", "10"),  # row3
    ("", "exG", "true", "5"),  # row4: missing user id
    ("u1", "nonexistent_ex", "true", "99"),  # row5: exercise not in table
    ("u2", "exA", "maybe", "20"),  # row6: invalid correct
    ("u2", "exB", "true", ""),  # row7: missing timestamp -> sentinel 0, sorts first
]


def _write_log(path: Path, rows) -> None:
    lines = [_LOG_HEADER]
    for row in rows:
        lines.append(",".join(row) + "\n")
    path.write_text("".join(lines), encoding="utf-8")


@pytest.fixture
def ex_path(tmp_path) -> Path:
    p = tmp_path / "junyi_Exercise_table.csv"
    _write_exercise_table(p)
    return p


@pytest.fixture
def log_path(tmp_path) -> Path:
    p = tmp_path / "junyi_ProblemLog_original.csv"
    _write_log(p, _LOG_ROWS)
    return p


# ---------------------------------------------------------------------------
# LoadStats: row-accounting fidelity, chunk-invariant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunksize", [1, 2, 3, 8, 100])
def test_load_stats_row_accounting(ex_path, log_path, chunksize):
    result = junyi_data.load_junyi_kc_traced(log_path, ex_path, chunksize=chunksize)
    stats = result.stats
    assert stats.n_rows_read == 8
    assert stats.n_rows_missing_user_id == 1  # row4
    assert stats.n_rows_exercise_not_in_table == 1  # row5
    assert stats.n_rows_invalid_correct == 1  # row6
    assert stats.n_rows_used == 5  # rows 0,1,2,3,7
    assert stats.n_rows_missing_timestamp == 1  # row7
    assert stats.n_rows_topic_unmapped == 1  # row2 (exD, no topic)


def test_missing_user_id_row_excluded(ex_path, log_path):
    result = junyi_data.load_junyi_kc_traced(log_path, ex_path, chunksize=100)
    total_events = sum(l.item_ids.shape[0] for l in result.learners)
    assert total_events == 5


# ---------------------------------------------------------------------------
# Item/KC catalog: full-exercise-table vocabulary (module docstring note 2)
# ---------------------------------------------------------------------------


def test_item_catalog_is_full_deduped_table(ex_path, log_path):
    result = junyi_data.load_junyi_kc_traced(log_path, ex_path, chunksize=100)
    assert result.item_names == ["exA", "exB", "exC", "exD", "exE", "exG"]
    assert result.item_hierarchy.n_items == 6
    assert result.item_hierarchy.level_sizes == (6,)  # flat (module docstring note 6)


def test_kc_catalog_vs_attempted_distinction(ex_path, log_path):
    result = junyi_data.load_junyi_kc_traced(log_path, ex_path, chunksize=100)
    assert set(result.kc_names) == {"t1", "t2", "t3"}
    assert result.n_kcs == 3  # full catalog (t3/exE is never attempted below)
    assert result.n_kcs_attempted == 2  # only t1 (exA/exB), t2 (exC) appear in kept rows


def test_n_learners(ex_path, log_path):
    result = junyi_data.load_junyi_kc_traced(log_path, ex_path, chunksize=100)
    assert result.n_learners == 2  # u1, u2


# ---------------------------------------------------------------------------
# `max_learners`: deterministic learner subsample (module docstring note
# 10), the Junyi real-bed pilot's OOM fix. `log_path`/`ex_path` above only
# carry 2 distinct users (u1, u2), which cannot distinguish "capped" from
# "uncapped" -- both u3 and a 3rd distinct user are added here so
# max_learners=2 actually has someone to exclude.
# ---------------------------------------------------------------------------


def test_max_learners_caps_learners_but_not_kc_catalog(ex_path, tmp_path):
    rows = list(_LOG_ROWS) + [("u3", "exC", "true", "200")]
    log_path_3users = tmp_path / "junyi_log_3users.csv"
    _write_log(log_path_3users, rows)

    uncapped = junyi_data.load_junyi_kc_traced(log_path_3users, ex_path, chunksize=100)
    capped = junyi_data.load_junyi_kc_traced(log_path_3users, ex_path, chunksize=100, max_learners=2)

    assert uncapped.n_learners == 3  # u1, u2, u3
    assert capped.n_learners == 2  # u3 sorts last, dropped by the cap
    assert len(capped.learners) == 2
    assert capped.stats.n_learners_kept == 2
    assert capped.stats.n_rows_excluded_by_subsample == 1  # u3's one row

    # The full exercise-table catalog is unaffected by dropping a learner.
    assert capped.n_kcs == uncapped.n_kcs == 3
    assert capped.item_hierarchy.n_items == uncapped.item_hierarchy.n_items == 6


# ---------------------------------------------------------------------------
# `subsample_seed`: SEEDED RANDOM cohort draw (module docstring note 11),
# additive to `max_learners`'s pre-existing first-N-by-sort rule (note 10).
# A 4th distinct user is added on top of the 3-user fixture above so
# max_learners=2 has room to draw a genuinely different 2-subset per seed
# (with only 2 distinct users, every subsample mode would trivially agree).
# ---------------------------------------------------------------------------


def test_subsample_seed_draws_seeded_random_cohort(ex_path, tmp_path):
    rows = list(_LOG_ROWS) + [
        ("u3", "exC", "true", "200"),
        ("u4", "exA", "true", "300"),
    ]
    log_path_4users = tmp_path / "junyi_log_4users.csv"
    _write_log(log_path_4users, rows)
    full_ids = {"u1", "u2", "u3", "u4"}

    # (c) subsample_seed=None: unchanged first-2-by-sort behavior.
    kept_none, mode_none, depth_none = junyi_data._collect_kept_user_ids(
        log_path_4users, max_learners=2, chunksize=100,
    )
    assert kept_none == set(sorted(full_ids)[:2])
    assert mode_none == "first" and depth_none is None

    # (a) seeded draws: exactly 2 ids each, both valid subsets of the full id set.
    kept_seed0, mode_seed0, depth_seed0 = junyi_data._collect_kept_user_ids(
        log_path_4users, max_learners=2, chunksize=100, subsample_seed=0,
    )
    kept_seed1, _, _ = junyi_data._collect_kept_user_ids(
        log_path_4users, max_learners=2, chunksize=100, subsample_seed=1,
    )
    assert mode_seed0 == "random" and depth_seed0 is None
    assert len(kept_seed0) == 2 and kept_seed0.issubset(full_ids)
    assert len(kept_seed1) == 2 and kept_seed1.issubset(full_ids)

    # (b) same seed twice -> identical set (reproducibility); different
    # seeds generally draw different subsets (verified true for this
    # fixture/seed pair, not asserted as a universal RNG property).
    kept_seed0_again, _, _ = junyi_data._collect_kept_user_ids(
        log_path_4users, max_learners=2, chunksize=100, subsample_seed=0,
    )
    assert kept_seed0 == kept_seed0_again
    assert kept_seed0 != kept_seed1

    # End-to-end via load_junyi_kc_traced: subsample_seed threads through,
    # n_learners is capped at 2 either way, and the full exercise-table KC
    # catalog (n_kcs) is untouched by which subsample mode is in force.
    result_seed0 = junyi_data.load_junyi_kc_traced(
        log_path_4users, ex_path, chunksize=100, max_learners=2, subsample_seed=0,
    )
    result_none = junyi_data.load_junyi_kc_traced(
        log_path_4users, ex_path, chunksize=100, max_learners=2,
    )
    assert result_seed0.n_learners == 2
    assert result_none.n_learners == 2
    assert result_seed0.n_kcs == result_none.n_kcs == 3


# ---------------------------------------------------------------------------
# `selection="deepest"`: top-N-by-raw-row-count cohort (module docstring
# note 12), the density-boundary pilot's kept-id rule. Raw counts below:
# u1=4 (rows 0,1,2,5 -- the pre-pass counts RAW rows incl. ones the main
# pass later drops), u2=3 (rows 3,6,7), u3=5, u4=3 (tie with u2).
# ---------------------------------------------------------------------------


def test_selection_deepest_keeps_top_n_by_row_count(ex_path, tmp_path):
    rows = list(_LOG_ROWS) + [
        ("u3", "exA", "true", str(t)) for t in (300, 310, 320, 330, 340)
    ] + [
        ("u4", "exB", "true", str(t)) for t in (400, 410, 420)
    ]
    log_path_deep = tmp_path / "junyi_log_deep.csv"
    _write_log(log_path_deep, rows)

    # (a) top-2 by raw count: u3 (5) and u1 (4); cutoff = kept minimum = 4.
    kept2, mode2, depth2 = junyi_data._collect_kept_user_ids(
        log_path_deep, max_learners=2, chunksize=100, selection="deepest",
    )
    assert kept2 == {"u3", "u1"}
    assert mode2 == "deepest" and depth2 == 4

    # (b) tie-break at count 3 (u2 vs u4): ascending id keeps u2.
    kept3, _, depth3 = junyi_data._collect_kept_user_ids(
        log_path_deep, max_learners=3, chunksize=100, selection="deepest",
    )
    assert kept3 == {"u3", "u1", "u2"}
    assert depth3 == 3

    # (c) subsample_seed PRECEDENCE (note 12): a non-None seed forces the
    # effective mode to "random" regardless of selection.
    _, mode_prec, depth_prec = junyi_data._collect_kept_user_ids(
        log_path_deep, max_learners=2, chunksize=100, subsample_seed=0, selection="deepest",
    )
    assert mode_prec == "random" and depth_prec is None

    # (d) unknown selection rejected.
    with pytest.raises(ValueError, match="selection"):
        junyi_data._collect_kept_user_ids(
            log_path_deep, max_learners=2, chunksize=100, selection="shallowest",
        )

    # (e) end-to-end: selection threads through load_junyi_kc_traced, the
    # stats report the effective mode + depth cutoff, and the kept cohort
    # is {u1, u3} (valid-event total 3 + 5 = 8; u2/u4 rows excluded).
    result_deep = junyi_data.load_junyi_kc_traced(
        log_path_deep, ex_path, chunksize=100, max_learners=2, selection="deepest",
    )
    assert result_deep.n_learners == 2
    assert result_deep.stats.selection_used == "deepest"
    assert result_deep.stats.depth_cutoff == 4
    assert sum(l.item_ids.shape[0] for l in result_deep.learners) == 8

    # (f) default selection (no flag) on an uncapped load: stats stay at
    # their neutral defaults (no pre-pass runs at all).
    result_uncapped = junyi_data.load_junyi_kc_traced(log_path_deep, ex_path, chunksize=100)
    assert result_uncapped.stats.selection_used == "first"
    assert result_uncapped.stats.depth_cutoff is None


# ---------------------------------------------------------------------------
# Chronological ordering: timestamp primary, file row-order tie-break
# ---------------------------------------------------------------------------


def test_u1_chronological_order_with_row_pos_tiebreak(ex_path, log_path):
    result = junyi_data.load_junyi_kc_traced(log_path, ex_path, chunksize=100)
    stu1 = next(l for l in result.learners if l.item_ids.shape[0] == 3)
    item_id = {name: i for i, name in enumerate(result.item_names)}
    # row0 (exB, ts50) first; then ts100 tie between row1 (exA) and row2
    # (exD), broken by file order -> exA before exD.
    assert stu1.item_ids.tolist() == [item_id["exB"], item_id["exA"], item_id["exD"]]
    assert stu1.responses.tolist() == [0, 1, 1]


def test_u2_missing_timestamp_sorts_first(ex_path, log_path):
    result = junyi_data.load_junyi_kc_traced(log_path, ex_path, chunksize=100)
    stu2 = next(l for l in result.learners if l.item_ids.shape[0] == 2)
    item_id = {name: i for i, name in enumerate(result.item_names)}
    # row7 (exB, missing timestamp -> sentinel 0) sorts before row3 (exC, ts10).
    assert stu2.item_ids.tolist() == [item_id["exB"], item_id["exC"]]


def test_learner_ids_are_positional(ex_path, log_path):
    result = junyi_data.load_junyi_kc_traced(log_path, ex_path, chunksize=100)
    for i, log in enumerate(result.learners):
        assert log.learner == i


def test_learner_timestamps_aligned(ex_path, log_path):
    result = junyi_data.load_junyi_kc_traced(log_path, ex_path, chunksize=100)
    for log, ts in zip(result.learners, result.learner_timestamps):
        assert ts.shape[0] == log.item_ids.shape[0]


# ---------------------------------------------------------------------------
# Tag ids / mask: single-topic arity, untagged items (module docstring note 4)
# ---------------------------------------------------------------------------


def test_tag_ids_single_topic_and_untagged_item(ex_path, log_path):
    result = junyi_data.load_junyi_kc_traced(log_path, ex_path, chunksize=100)
    stu1 = next(l for l in result.learners if l.item_ids.shape[0] == 3)
    assert stu1.tag_ids.shape[1] == 1  # arity always 1
    # exB, exA both tag t1; exD (untagged item) carries an all-False mask.
    assert stu1.tag_mask[0, 0] and stu1.tag_mask[1, 0]
    assert not stu1.tag_mask[2, 0]


# ---------------------------------------------------------------------------
# Item hierarchy + Q-matrix
# ---------------------------------------------------------------------------


def test_qmatrix_arity_always_one(ex_path, log_path):
    result = junyi_data.load_junyi_kc_traced(log_path, ex_path, chunksize=100)
    qm = result.qmatrix
    assert qm.n_items == 6
    assert qm.arity_max == 1
    pure_mask = qmatrix.pure_anchor_item_mask(qm)
    # exA, exB, exC, exE, exG carry a topic (5 pure anchors); exD does not.
    assert int(pure_mask.sum()) == 5


def test_qmatrix_circularity_guard_not_tripped(ex_path, log_path):
    result = junyi_data.load_junyi_kc_traced(log_path, ex_path, chunksize=100)
    qmatrix.check_no_circularity(result.qmatrix.item_keys, result.qmatrix.kc_names)


# ---------------------------------------------------------------------------
# Growth-package downstream compatibility: bank.build_calibration_rows
# ---------------------------------------------------------------------------


def test_learners_feed_build_calibration_rows(ex_path, log_path):
    from kt_mirt.growth.bank import build_calibration_rows

    result = junyi_data.load_junyi_kc_traced(log_path, ex_path, chunksize=100)
    rows = build_calibration_rows(result.learners)
    assert len(rows) == sum(l.item_ids.shape[0] for l in result.learners) == 5


# ---------------------------------------------------------------------------
# Empty file
# ---------------------------------------------------------------------------


def test_empty_log_returns_empty_learners_but_full_catalog(ex_path, tmp_path):
    p = tmp_path / "empty_log.csv"
    p.write_text(_LOG_HEADER, encoding="utf-8")
    result = junyi_data.load_junyi_kc_traced(p, ex_path, chunksize=10)
    assert result.n_learners == 0
    assert result.learners == []
    assert result.stats.n_rows_read == 0
    # The item/KC catalog is still built from the exercise table (note 2).
    assert result.qmatrix.n_items == 6
    assert result.n_kcs == 3


# ---------------------------------------------------------------------------
# Exercise-table dedup + prerequisite graph
# ---------------------------------------------------------------------------


def test_exact_duplicate_exercise_row_dropped(ex_path, log_path):
    result = junyi_data.load_junyi_kc_traced(log_path, ex_path, chunksize=100)
    assert result.item_names.count("exA") == 1  # the second "exA" row is a dropped duplicate
    assert len(result.item_names) == 6


def test_prerequisite_graph_edges_and_self_loop(ex_path):
    graph = junyi_data.load_prerequisite_graph(ex_path)
    assert graph.n_exact_duplicate_rows_dropped == 1
    assert set(graph.edges) == {
        ("exA", "exB"), ("exB", "exC"), ("exA", "exE"), ("exC", "exE"), ("exG", "exG"),
    }
    assert graph.self_loop_exercises == ["exG"]


def test_prerequisite_graph_id_edges_match_loader_vocabulary(ex_path, log_path):
    result = junyi_data.load_junyi_kc_traced(log_path, ex_path, chunksize=100)
    item_name_to_id = {name: i for i, name in enumerate(result.item_names)}
    graph = junyi_data.load_prerequisite_graph(ex_path, item_name_to_id=item_name_to_id)
    assert graph.n_edges_endpoint_not_in_vocab == 0  # every edge endpoint survived dedup
    id_edge_set = set(graph.id_edges)
    assert (item_name_to_id["exA"], item_name_to_id["exB"]) in id_edge_set
    assert (item_name_to_id["exG"], item_name_to_id["exG"]) in id_edge_set  # self-loop preserved


# ---------------------------------------------------------------------------
# Integration test: the real files, gated to skip when absent
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "junyi15"
_REAL_LOG_PATH = _DATA_DIR / "junyi_ProblemLog_original.csv"
_REAL_EX_PATH = _DATA_DIR / "junyi_Exercise_table.csv"


@pytest.mark.skipif(
    not (_REAL_LOG_PATH.exists() and _REAL_EX_PATH.exists()),
    reason="real Junyi15 raw files not present locally",
)
def test_real_junyi_file_loads_a_small_prefix():
    """Light smoke test on a small row prefix of the real file (the full
    acceptance run lives in the agreement-table script, not the test
    suite)."""
    result = junyi_data.load_junyi_kc_traced(
        _REAL_LOG_PATH, _REAL_EX_PATH, chunksize=10_000, nrows=20_000,
    )
    assert result.stats.n_rows_read == 20_000
    assert result.n_learners > 0
    assert result.n_kcs > 0
    assert result.qmatrix.n_items > 0
    assert len(result.learners) == result.n_learners
    for log in result.learners:
        assert log.item_ids.shape[0] == log.responses.shape[0] == log.tag_ids.shape[0] == log.tag_mask.shape[0]
