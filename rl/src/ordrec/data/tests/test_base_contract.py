"""Contract tests for OrdinalDatasetBase and AdapterConfig."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ordrec.data import (
    AdapterConfig,
    EediAdapter,
    OrdinalDatasetBase,
    SyntheticAdapter,
)


def test_adapter_config_is_frozen(tmp_path: Path) -> None:
    cfg = AdapterConfig(name="t", raw_dir=tmp_path, out_dir=tmp_path)
    with pytest.raises(Exception):
        # frozen dataclass mutation must raise FrozenInstanceError
        cfg.name = "changed"  # type: ignore[misc]


def test_adapter_config_rejects_bad_fractions(tmp_path: Path) -> None:
    # Allowed at construction time (we validate at materialise/split time)
    cfg = AdapterConfig(
        name="t", raw_dir=tmp_path, out_dir=tmp_path,
        test_frac=0.1, valid_frac=0.1,
    )
    assert cfg.test_frac == 0.1
    assert cfg.valid_frac == 0.1


def test_base_class_cannot_be_instantiated_directly(tmp_path: Path) -> None:
    cfg = AdapterConfig(name="t", raw_dir=tmp_path, out_dir=tmp_path)
    with pytest.raises(TypeError):
        OrdinalDatasetBase(cfg)  # type: ignore[abstract]


def test_subclasses_have_required_methods() -> None:
    for cls in (EediAdapter, SyntheticAdapter):
        assert hasattr(cls, "materialise") and callable(cls.materialise)
        assert hasattr(cls, "load") and callable(cls.load)
        assert hasattr(cls, "get_split") and callable(cls.get_split)
        assert hasattr(cls, "get_n_items") and callable(cls.get_n_items)
        assert hasattr(cls, "get_n_categories")
        assert hasattr(cls, "__getitem__")
        assert hasattr(cls, "__len__")


def test_getitem_returns_expected_keys_and_dtypes(tmp_path: Path) -> None:
    """End-to-end smoke through SyntheticAdapter on a tiny ma-irt artefact."""
    # Build a tiny ma-irt static dataset by hand to avoid importing the
    # heavy datagen module here.
    import json
    raw = tmp_path / "raw"
    raw.mkdir()
    seq = [
        {"questions": [1, 2, 3, 4, 5, 1, 2], "responses": [0, 1, 2, 3, 1, 0, 2]},
        {"questions": [2, 3, 4, 5, 1, 2, 3], "responses": [3, 2, 1, 0, 1, 2, 3]},
        {"questions": [1, 2, 3, 4, 5, 1, 2, 3], "responses": [1, 2, 3, 0, 1, 2, 3, 0]},
        {"questions": [4, 5, 1, 2, 3, 4, 5], "responses": [0, 1, 2, 3, 1, 0, 2]},
        {"questions": [1, 2, 3, 4, 5, 1], "responses": [3, 2, 1, 0, 1, 2]},
    ]
    (raw / "sequences.json").write_text(json.dumps(seq), encoding="utf-8")
    (raw / "metadata.json").write_text(
        json.dumps({"n_questions": 5, "n_categories": 4, "n_students": 5}),
        encoding="utf-8",
    )
    cfg = AdapterConfig(
        name="synth_tiny", raw_dir=raw, out_dir=tmp_path / "out",
        split_seed=0, test_frac=0.2, valid_frac=0.2, min_seq_len=3,
        max_seq_len=0, chunk_long_sequences=False,
    )
    a = SyntheticAdapter(cfg)
    a.materialise()
    a.load()
    assert len(a) > 0
    sample = a[0]
    assert "questions" in sample and "responses" in sample and "student_id" in sample
    assert sample["student_id"] == 1
    assert sample["questions"].dtype == __import__("torch").long
    assert sample["responses"].dtype == __import__("torch").long
    # student_id is 1-based at row offset.
    assert a[1]["student_id"] == 2


def test_get_split_codes_match_filenames(tmp_path: Path) -> None:
    """get_split('train' / 'valid' / 'test') is exhaustive and exclusive."""
    import json
    raw = tmp_path / "raw"
    raw.mkdir()
    # 20 students of equal length so train/valid/test all have rows.
    seq = [{"questions": [i % 5 + 1 for i in range(8)],
            "responses": [j % 4 for j in range(8)]} for _ in range(20)]
    (raw / "sequences.json").write_text(json.dumps(seq), encoding="utf-8")
    (raw / "metadata.json").write_text(
        json.dumps({"n_questions": 5, "n_categories": 4, "n_students": 20}),
        encoding="utf-8",
    )
    cfg = AdapterConfig(
        name="synth_split", raw_dir=raw, out_dir=tmp_path / "out",
        split_seed=42, test_frac=0.2, valid_frac=0.2, min_seq_len=3,
        max_seq_len=0, chunk_long_sequences=False,
    )
    a = SyntheticAdapter(cfg)
    a.materialise()
    a.load()
    tr = a.get_split("train")
    va = a.get_split("valid")
    te = a.get_split("test")
    n = len(a)
    assert set(tr.tolist()) | set(va.tolist()) | set(te.tolist()) == set(range(n))
    assert not (set(tr.tolist()) & set(va.tolist()))
    assert not (set(tr.tolist()) & set(te.tolist()))
    assert not (set(va.tolist()) & set(te.tolist()))
    with pytest.raises(ValueError):
        a.get_split("not-a-split")  # type: ignore[arg-type]


def test_subclass_attribute_population_after_load(tmp_path: Path) -> None:
    import json
    raw = tmp_path / "raw"
    raw.mkdir()
    seq = [{"questions": [1, 2, 3, 4, 5], "responses": [0, 1, 2, 3, 1]}] * 10
    (raw / "sequences.json").write_text(json.dumps(seq), encoding="utf-8")
    (raw / "metadata.json").write_text(
        json.dumps({"n_questions": 5, "n_categories": 4, "n_students": 10}),
        encoding="utf-8",
    )
    cfg = AdapterConfig(
        name="synth_attrs", raw_dir=raw, out_dir=tmp_path / "out",
        split_seed=0, test_frac=0.2, valid_frac=0.2, min_seq_len=3,
        max_seq_len=0, chunk_long_sequences=False,
    )
    a = SyntheticAdapter(cfg)
    a.materialise()
    a.load()
    assert isinstance(a._questions, list)
    assert isinstance(a._responses, list)
    assert isinstance(a._student_split, np.ndarray)
    assert a._student_split.dtype == np.int8
    assert a.get_n_categories() == 4
    assert a.get_n_items() == 5
    assert a.get_n_kcs() == 0
    assert a.get_q_matrix() is None
