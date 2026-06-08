"""Determinism tests for the user-level split machinery."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ordrec.data import AdapterConfig, SyntheticAdapter
from ordrec.data.split import (
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALID,
    _chunk_sequences,
    make_split,
    stratified_split,
)


def test_make_split_reproducible_with_same_seed() -> None:
    a = make_split(n_students=100, test_frac=0.1, valid_frac=0.1, split_seed=0)
    b = make_split(n_students=100, test_frac=0.1, valid_frac=0.1, split_seed=0)
    assert np.array_equal(a, b)


def test_make_split_different_with_different_seed() -> None:
    a = make_split(n_students=100, test_frac=0.1, valid_frac=0.1, split_seed=0)
    b = make_split(n_students=100, test_frac=0.1, valid_frac=0.1, split_seed=1)
    assert not np.array_equal(a, b)


def test_make_split_fractions_respected() -> None:
    codes = make_split(n_students=1000, test_frac=0.1, valid_frac=0.1, split_seed=0)
    counts = np.bincount(codes.astype(int) + 1, minlength=4)[1:]
    # ~10% test, ~10% valid, ~80% train
    assert 90 <= int(counts[SPLIT_TEST]) <= 110
    assert 90 <= int(counts[SPLIT_VALID]) <= 110
    assert 780 <= int(counts[SPLIT_TRAIN]) <= 820
    assert int(counts.sum()) == 1000


def test_make_split_rejects_bad_args() -> None:
    with pytest.raises(ValueError):
        make_split(n_students=0, test_frac=0.1, valid_frac=0.1, split_seed=0)
    with pytest.raises(ValueError):
        make_split(n_students=10, test_frac=0.7, valid_frac=0.6, split_seed=0)


def test_stratified_split_proportional_within_strata() -> None:
    # Two strata: 80 in stratum 0, 20 in stratum 1.
    strata = np.concatenate([np.zeros(80, dtype=int), np.ones(20, dtype=int)])
    codes = stratified_split(strata, test_frac=0.2, valid_frac=0.2, split_seed=0)
    # The stratum-0 test fraction should be ~16 (80*0.2), stratum-1 test ~4.
    n_test_s0 = int(((codes == SPLIT_TEST) & (strata == 0)).sum())
    n_test_s1 = int(((codes == SPLIT_TEST) & (strata == 1)).sum())
    assert 13 <= n_test_s0 <= 19
    assert 2 <= n_test_s1 <= 6


def test_chunk_sequences_preserves_parent_when_short() -> None:
    q = [[1, 2, 3]]
    r = [[0, 1, 2]]
    cq, cr, par = _chunk_sequences(q, r, max_len=10)
    assert cq == [[1, 2, 3]]
    assert cr == [[0, 1, 2]]
    assert par == [0]


def test_chunk_sequences_splits_long_inputs() -> None:
    q = [list(range(1, 11))]   # length 10
    r = [list(range(10))]
    cq, cr, par = _chunk_sequences(q, r, max_len=4)
    assert [len(c) for c in cq] == [4, 4, 2]
    assert cr == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert par == [0, 0, 0]


def test_chunk_zero_means_no_chunking() -> None:
    q = [[1, 2, 3, 4, 5]]
    r = [[0, 0, 0, 0, 0]]
    cq, cr, par = _chunk_sequences(q, r, max_len=0)
    assert cq == q and cr == r and par == [0]


def test_synthetic_adapter_byte_identical_rebuild(tmp_path: Path) -> None:
    """The headline test for the spec, ``split_determinism``."""
    raw = tmp_path / "raw"
    raw.mkdir()
    seq = [
        {"questions": [(i + j) % 7 + 1 for j in range(12)],
         "responses": [(i * j) % 4 for j in range(12)]}
        for i in range(30)
    ]
    (raw / "sequences.json").write_text(json.dumps(seq), encoding="utf-8")
    (raw / "metadata.json").write_text(
        json.dumps({"n_questions": 7, "n_categories": 4, "n_students": 30}),
        encoding="utf-8",
    )
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    cfg_a = AdapterConfig(
        name="d_a", raw_dir=raw, out_dir=out_a, split_seed=123,
        test_frac=0.2, valid_frac=0.2, min_seq_len=5,
        max_seq_len=0, chunk_long_sequences=False,
    )
    cfg_b = AdapterConfig(
        name="d_b", raw_dir=raw, out_dir=out_b, split_seed=123,
        test_frac=0.2, valid_frac=0.2, min_seq_len=5,
        max_seq_len=0, chunk_long_sequences=False,
    )
    SyntheticAdapter(cfg_a).materialise()
    SyntheticAdapter(cfg_b).materialise()
    seq_a = (out_a / "d_a" / "sequences.json").read_bytes()
    seq_b = (out_b / "d_b" / "sequences.json").read_bytes()
    assert seq_a == seq_b
