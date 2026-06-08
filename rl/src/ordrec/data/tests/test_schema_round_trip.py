"""Schema round-trip and validator tests."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

from ordrec.data.schema import (
    COMMON_RECORD_SCHEMA,
    SchemaError,
    dump_coercion_artefacts,
    dump_metadata,
    dump_sequences,
    load_coercion_artefacts,
    load_metadata,
    load_sequences,
    validate_metadata,
    validate_q_matrix,
    validate_sequences,
)


def _well_formed_meta(n_items: int = 5, n_cat: int = 4) -> dict:
    return {
        "dataset_name": "demo",
        "adapter_class": "EediAdapter",
        "n_students": 10,
        "n_items": n_items,
        "n_categories": n_cat,
        "n_kcs": 0,
        "seq_len_range": [3, 8],
        "ordinal_coercion_method": "distractor_difficulty_2pl",
        "splits": {
            "split_seed": 0, "train_frac": 0.8, "valid_frac": 0.1,
            "test_frac": 0.1, "n_train": 8, "n_valid": 1, "n_test": 1,
        },
        "question_id_map": {"q11": 1, "q12": 2, "q13": 3, "q14": 4, "q15": 5},
    }


def test_metadata_validator_accepts_well_formed() -> None:
    validate_metadata(_well_formed_meta())


def test_metadata_validator_rejects_missing_key() -> None:
    meta = _well_formed_meta()
    del meta["n_categories"]
    with pytest.raises(SchemaError):
        validate_metadata(meta)


def test_metadata_validator_rejects_bad_K() -> None:
    meta = _well_formed_meta()
    meta["n_categories"] = 1
    with pytest.raises(SchemaError):
        validate_metadata(meta)


def test_metadata_validator_rejects_bad_method() -> None:
    meta = _well_formed_meta()
    meta["ordinal_coercion_method"] = "not_a_method"
    with pytest.raises(SchemaError):
        validate_metadata(meta)


def test_metadata_validator_rejects_inconsistent_fractions() -> None:
    meta = _well_formed_meta()
    meta["splits"]["test_frac"] = 0.5
    with pytest.raises(SchemaError):
        validate_metadata(meta)


def test_sequences_validator_rejects_out_of_range_qid() -> None:
    with pytest.raises(SchemaError):
        validate_sequences(
            [{"questions": [99], "responses": [0], "split": "train"}],
            n_items=5, n_categories=4,
        )


def test_sequences_validator_rejects_out_of_range_response() -> None:
    with pytest.raises(SchemaError):
        validate_sequences(
            [{"questions": [1], "responses": [9], "split": "train"}],
            n_items=5, n_categories=4,
        )


def test_sequences_validator_rejects_length_mismatch() -> None:
    with pytest.raises(SchemaError):
        validate_sequences(
            [{"questions": [1, 2, 3], "responses": [0, 1], "split": "train"}],
            n_items=5, n_categories=4,
        )


def test_sequences_validator_rejects_bad_split() -> None:
    with pytest.raises(SchemaError):
        validate_sequences(
            [{"questions": [1], "responses": [0], "split": "blah"}],
            n_items=5, n_categories=4,
        )


def test_q_matrix_validator() -> None:
    q = np.zeros((5, 3), dtype=np.uint8)
    q[0, 1] = 1
    validate_q_matrix(q, n_items=5, n_kcs=3)
    with pytest.raises(SchemaError):
        validate_q_matrix(q.astype(np.int32), n_items=5, n_kcs=3)
    with pytest.raises(SchemaError):
        bad = np.zeros((5, 3), dtype=np.uint8)
        bad[0, 0] = 2
        validate_q_matrix(bad, n_items=5, n_kcs=3)
    with pytest.raises(SchemaError):
        validate_q_matrix(q, n_items=4, n_kcs=3)


def test_metadata_round_trip(tmp_path: Path) -> None:
    meta = _well_formed_meta()
    p = tmp_path / "metadata.json"
    dump_metadata(meta, p)
    loaded = load_metadata(p)
    assert loaded == meta


def test_sequences_round_trip_is_byte_identical(tmp_path: Path) -> None:
    records = [
        {"questions": [1, 2, 3], "responses": [0, 1, 2], "split": "train"},
        {"questions": [4, 5], "responses": [3, 2], "split": "valid"},
        {"questions": [1, 1, 1], "responses": [0, 0, 0], "split": "test"},
    ]
    p1 = tmp_path / "sequences_1.json"
    p2 = tmp_path / "sequences_2.json"
    dump_sequences(records, p1)
    dump_sequences(records, p2)
    assert p1.read_bytes() == p2.read_bytes()
    loaded = load_sequences(p1)
    assert loaded == records


def test_coercion_round_trip(tmp_path: Path) -> None:
    payload = {
        "method": "distractor_difficulty_2pl",
        "distractor_order_per_q": {"1": [2, 3, 4], "2": [3, 4, 2]},
    }
    p = tmp_path / "coercion_artefacts.json"
    dump_coercion_artefacts(payload, p)
    assert load_coercion_artefacts(p) == payload


def test_common_record_schema_keys_unchanged() -> None:
    # Lock the public schema so we notice silent renames.
    expected = {
        "dataset_name", "adapter_class", "n_students", "n_items",
        "n_categories", "n_kcs", "seq_len_range", "ordinal_coercion_method",
        "splits", "question_id_map",
    }
    assert set(COMMON_RECORD_SCHEMA.keys()) == expected
