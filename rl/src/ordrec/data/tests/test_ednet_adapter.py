"""EdNetAdapter tests against the ednet_mini.csv fixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ordrec.data import AdapterConfig, EdNetAdapter
from ordrec.data.ednet import (
    N_CATEGORIES_EDNET,
    ORD_CORRECT_FAST,
    ORD_CORRECT_SLOW,
    ORD_INCORRECT_FAST,
    ORD_INCORRECT_SLOW,
)
from ordrec.data.schema import (
    COERCION_FILENAME,
    METADATA_FILENAME,
    SEQUENCES_FILENAME,
    load_coercion_artefacts,
    load_metadata,
    load_sequences,
)


FIXTURE = Path(__file__).parent / "fixtures" / "ednet_mini.csv"


def _make_cfg(tmp_path: Path, seed: int = 7) -> AdapterConfig:
    return AdapterConfig(
        name="ednet_mini_k4",
        raw_dir=FIXTURE,
        out_dir=tmp_path,
        split_seed=seed,
        test_frac=0.2,
        valid_frac=0.2,
        min_seq_len=3,
        max_seq_len=0,
        chunk_long_sequences=False,
    )


def test_materialise_writes_all_artefacts(tmp_path: Path) -> None:
    a = EdNetAdapter(_make_cfg(tmp_path))
    a.materialise()
    out = tmp_path / "ednet_mini_k4"
    assert (out / SEQUENCES_FILENAME).exists()
    assert (out / METADATA_FILENAME).exists()
    assert (out / COERCION_FILENAME).exists()


def test_metadata_block(tmp_path: Path) -> None:
    a = EdNetAdapter(_make_cfg(tmp_path))
    a.materialise()
    meta = load_metadata(tmp_path / "ednet_mini_k4" / METADATA_FILENAME)
    assert meta["adapter_class"] == "EdNetAdapter"
    assert meta["n_categories"] == N_CATEGORIES_EDNET == 4
    assert meta["n_questions"] == 5
    assert meta["n_students"] == 10
    assert meta["ordinal_coercion_method"] == "correctness_x_time_quadrant"
    assert meta["splits"]["split_seed"] == 7


def test_quadrant_mapping_table(tmp_path: Path) -> None:
    """Recoded responses must align with the (correct, fast/slow) quadrant
    given the per-question train-fold median.
    """
    a = EdNetAdapter(_make_cfg(tmp_path))
    a.materialise()
    a.load()

    coercion = load_coercion_artefacts(
        tmp_path / "ednet_mini_k4" / COERCION_FILENAME
    )
    rt_median_per_q = {int(k): float(v) for k, v in coercion["rt_median_per_q"].items()}

    df = pd.read_csv(FIXTURE)
    # Map question_id -> 1-based item id matches the adapter's first-
    # appearance-order convention.
    question_ids_unique = list(pd.unique(df["question_id"]))
    qid_to_item = {qid: i + 1 for i, qid in enumerate(question_ids_unique)}
    df["_item"] = df["question_id"].map(qid_to_item)

    train_idx = a.get_split("train")
    n_train = len(train_idx)
    assert n_train > 0

    # For each (student, item) row in the merged artefact, the recoded
    # category must match the quadrant table given correctness and the
    # rt_median lookup. We verify across all kept records.
    records = load_sequences(tmp_path / "ednet_mini_k4" / SEQUENCES_FILENAME)

    # Build a lookup (student_id, item) -> (correct, rt) from the raw csv.
    student_ids_unique = list(pd.unique(df["student_id"]))
    sid_to_idx = {sid: i for i, sid in enumerate(student_ids_unique)}
    df["_stu"] = df["student_id"].map(sid_to_idx)
    raw_lookup: dict[tuple[int, int], tuple[int, float]] = {}
    for stu, item, correct, rt in df[["_stu", "_item", "correct", "response_time"]].to_numpy():
        raw_lookup[(int(stu), int(item))] = (int(correct), float(rt))

    # Match records to students by row order (records are emitted in
    # student-index order).
    student_iter = iter(range(len(student_ids_unique)))
    for rec in records:
        # find the next student whose questions match the record's
        stu_idx = next(student_iter)
        while [it for it in df[df["_stu"] == stu_idx]["_item"].tolist()] != rec["questions"]:
            stu_idx = next(student_iter)
        for item, code in zip(rec["questions"], rec["responses"]):
            correct, rt = raw_lookup[(stu_idx, item)]
            median_q = rt_median_per_q.get(item, coercion["global_train_median_rt"])
            is_fast = rt <= median_q
            if correct == 1 and is_fast:
                expected = ORD_CORRECT_FAST
            elif correct == 1 and not is_fast:
                expected = ORD_CORRECT_SLOW
            elif correct == 0 and is_fast:
                expected = ORD_INCORRECT_FAST
            else:
                expected = ORD_INCORRECT_SLOW
            assert int(code) == int(expected), (item, correct, rt, median_q)


def test_median_computed_train_only(tmp_path: Path) -> None:
    """``rt_median_per_q`` must equal the train-fold median (not the global)."""
    a = EdNetAdapter(_make_cfg(tmp_path))
    a.materialise()
    a.load()
    coercion = load_coercion_artefacts(
        tmp_path / "ednet_mini_k4" / COERCION_FILENAME
    )

    df = pd.read_csv(FIXTURE)
    student_ids_unique = list(pd.unique(df["student_id"]))
    sid_to_idx = {sid: i for i, sid in enumerate(student_ids_unique)}
    df["_stu"] = df["student_id"].map(sid_to_idx)
    question_ids_unique = list(pd.unique(df["question_id"]))
    qid_to_item = {qid: i + 1 for i, qid in enumerate(question_ids_unique)}
    df["_item"] = df["question_id"].map(qid_to_item)

    train_idx = set(int(i) for i in a.get_split("train"))
    # The adapter applies the user-level split BEFORE the
    # min_seq_len filter; ``get_split("train")`` returns row indices
    # into the kept records, so we need to recover the original student
    # indices. For this fixture every student survives min_seq_len so
    # the kept-record order matches the materialised-student order.
    train_stu = set()
    for r_idx in train_idx:
        train_stu.add(r_idx)

    train_df = df[df["_stu"].isin(train_stu)]
    for item_str, median_str in coercion["rt_median_per_q"].items():
        item = int(item_str)
        median_expected = float(train_df[train_df["_item"] == item]["response_time"].median())
        assert pytest.approx(float(median_str), rel=1e-9) == median_expected


def test_kt3_no_hint_field(tmp_path: Path) -> None:
    """The fixture lacks a hint_used column, so the adapter must record
    ``ednet_level == 'kt3'`` and ``has_hint_field == False``."""
    a = EdNetAdapter(_make_cfg(tmp_path))
    a.materialise()
    coercion = load_coercion_artefacts(
        tmp_path / "ednet_mini_k4" / COERCION_FILENAME
    )
    assert coercion["ednet_level"] == "kt3"
    assert coercion["has_hint_field"] is False
    assert "kt3" in coercion["hint_handling"].lower()


def test_responses_in_valid_range_after_load(tmp_path: Path) -> None:
    a = EdNetAdapter(_make_cfg(tmp_path))
    a.materialise()
    a.load()
    for r_seq in a._responses:
        arr = np.asarray(r_seq, dtype=int)
        assert ((arr >= 0) & (arr <= 3)).all()
    for q_seq in a._questions:
        arr = np.asarray(q_seq, dtype=int)
        assert ((arr >= 1) & (arr <= a.get_n_questions())).all()


def test_kt4_detected_when_hint_field_present(tmp_path: Path) -> None:
    """A fixture with a hint_used column flips ``ednet_level`` to kt4."""
    # Create a tiny csv with the hint column on the fly.
    csv_path = tmp_path / "ednet_kt4_mini.csv"
    df = pd.read_csv(FIXTURE)
    df["hint_used"] = 0
    df.to_csv(csv_path, index=False)

    cfg = AdapterConfig(
        name="ednet_kt4_mini",
        raw_dir=csv_path,
        out_dir=tmp_path,
        split_seed=7,
        test_frac=0.2,
        valid_frac=0.2,
        min_seq_len=3,
        max_seq_len=0,
        chunk_long_sequences=False,
    )
    a = EdNetAdapter(cfg)
    a.materialise()
    coercion = load_coercion_artefacts(
        tmp_path / "ednet_kt4_mini" / COERCION_FILENAME
    )
    assert coercion["ednet_level"] == "kt4"
    assert coercion["has_hint_field"] is True
    assert "kt4" in coercion["hint_handling"].lower()


def test_missing_response_time_labeled_as_slow(tmp_path: Path) -> None:
    """A row with NaN response_time must be coerced to the slow bucket."""
    csv_path = tmp_path / "ednet_nan_rt.csv"
    df = pd.read_csv(FIXTURE)
    # Inject a NaN response_time at a known position.
    target_idx = df.index[(df["student_id"] == 2001) & (df["question_id"] == 15)][0]
    correct_at_target = int(df.loc[target_idx, "correct"])
    df.loc[target_idx, "response_time"] = float("nan")
    df.to_csv(csv_path, index=False)

    cfg = AdapterConfig(
        name="ednet_nan_rt",
        raw_dir=csv_path,
        out_dir=tmp_path,
        split_seed=7,
        test_frac=0.2,
        valid_frac=0.2,
        min_seq_len=3,
        max_seq_len=0,
        chunk_long_sequences=False,
    )
    a = EdNetAdapter(cfg)
    a.materialise()
    a.load()
    coercion = load_coercion_artefacts(
        tmp_path / "ednet_nan_rt" / COERCION_FILENAME
    )
    assert coercion["n_missing_rt_labeled_as_slow"] >= 1
    # Student 2001's recorded response for item 15 should be the slow
    # quadrant given the (correct == correct_at_target, rt == NaN
    # -> slow) coercion.
    # student_id 2001 maps to first row in first-appearance order -> 0.
    rec0 = a._questions[0], a._responses[0]
    questions, responses = rec0
    # Item id for question_id == 15 is index of 15 in first-appearance
    # order. The csv has questions 11, 12, 13, 14, 15 in order so item
    # for 15 is 5.
    pos = questions.index(5)
    expected = ORD_CORRECT_SLOW if correct_at_target == 1 else ORD_INCORRECT_SLOW
    assert responses[pos] == expected
