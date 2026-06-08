"""AssistAdapter tests against the assist_mini.csv fixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ordrec.data import AdapterConfig, AssistAdapter
from ordrec.data.assist import N_CATEGORIES_ASSIST
from ordrec.data.schema import (
    COERCION_FILENAME,
    METADATA_FILENAME,
    SEQUENCES_FILENAME,
    load_coercion_artefacts,
    load_metadata,
    load_sequences,
)


FIXTURE = Path(__file__).parent / "fixtures" / "assist_mini.csv"


def _make_cfg(tmp_path: Path, seed: int = 5) -> AdapterConfig:
    return AdapterConfig(
        name="assist_mini_k2",
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
    a = AssistAdapter(_make_cfg(tmp_path))
    a.materialise()
    out = tmp_path / "assist_mini_k2"
    assert (out / SEQUENCES_FILENAME).exists()
    assert (out / METADATA_FILENAME).exists()
    assert (out / COERCION_FILENAME).exists()


def test_metadata_block_k2(tmp_path: Path) -> None:
    a = AssistAdapter(_make_cfg(tmp_path))
    a.materialise()
    meta = load_metadata(tmp_path / "assist_mini_k2" / METADATA_FILENAME)
    assert meta["adapter_class"] == "AssistAdapter"
    assert meta["n_categories"] == N_CATEGORIES_ASSIST == 2
    assert meta["ordinal_coercion_method"] == "binary"
    assert meta["n_questions"] == 5
    assert meta["n_students"] == 10
    assert meta["splits"]["split_seed"] == 5


def test_identity_passthrough(tmp_path: Path) -> None:
    """Every recoded response must equal the original correct flag."""
    a = AssistAdapter(_make_cfg(tmp_path))
    a.materialise()
    a.load()
    records = load_sequences(tmp_path / "assist_mini_k2" / SEQUENCES_FILENAME)

    df = pd.read_csv(FIXTURE)
    student_ids_unique = list(pd.unique(df["student_id"]))
    sid_to_idx = {sid: i for i, sid in enumerate(student_ids_unique)}
    df["_stu"] = df["student_id"].map(sid_to_idx)
    question_ids_unique = list(pd.unique(df["question_id"]))
    qid_to_item = {qid: i + 1 for i, qid in enumerate(question_ids_unique)}
    df["_item"] = df["question_id"].map(qid_to_item)

    # Build (stu, item) -> correct lookup.
    raw_correct: dict[tuple[int, int], int] = {}
    for stu, item, correct in df[["_stu", "_item", "correct"]].to_numpy():
        raw_correct[(int(stu), int(item))] = int(correct)

    # The fixture has every student passing min_seq_len, so kept order
    # equals the materialised-student order.
    for stu_idx, rec in enumerate(records):
        for item, code in zip(rec["questions"], rec["responses"]):
            assert int(code) == raw_correct[(stu_idx, int(item))]


def test_responses_in_valid_range_after_load(tmp_path: Path) -> None:
    a = AssistAdapter(_make_cfg(tmp_path))
    a.materialise()
    a.load()
    for r_seq in a._responses:
        arr = np.asarray(r_seq, dtype=int)
        assert ((arr == 0) | (arr == 1)).all()
    for q_seq in a._questions:
        arr = np.asarray(q_seq, dtype=int)
        assert ((arr >= 1) & (arr <= a.get_n_questions())).all()


def test_no_test_in_train(tmp_path: Path) -> None:
    """User-level split is exhaustive and exclusive."""
    a = AssistAdapter(_make_cfg(tmp_path))
    a.materialise()
    a.load()
    tr = set(a.get_split("train").tolist())
    va = set(a.get_split("valid").tolist())
    te = set(a.get_split("test").tolist())
    assert tr | va | te == set(range(len(a)))
    assert not (tr & va)
    assert not (tr & te)
    assert not (va & te)
