"""Tests for ``ma_irt_bridge`` end-to-end through a SyntheticAdapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ordrec.data import (
    AdapterConfig,
    SyntheticAdapter,
    adapter_to_dataloader,
    adapter_to_sequence_dataset,
)


@pytest.fixture
def loaded_synth(tmp_path: Path) -> SyntheticAdapter:
    raw = tmp_path / "raw"
    raw.mkdir()
    seq = [
        {"questions": [(i + j) % 5 + 1 for j in range(8)],
         "responses": [(i * j) % 4 for j in range(8)]}
        for i in range(20)
    ]
    (raw / "sequences.json").write_text(json.dumps(seq), encoding="utf-8")
    (raw / "metadata.json").write_text(
        json.dumps({"n_questions": 5, "n_categories": 4, "n_students": 20}),
        encoding="utf-8",
    )
    cfg = AdapterConfig(
        name="synth_bridge", raw_dir=raw, out_dir=tmp_path,
        split_seed=0, test_frac=0.2, valid_frac=0.2, min_seq_len=3,
        max_seq_len=0, chunk_long_sequences=False,
    )
    a = SyntheticAdapter(cfg)
    a.materialise()
    a.load()
    return a


def test_sequence_dataset_first_item_shape(loaded_synth) -> None:
    pytest.importorskip("utils.dataloader")  # confirms ma-irt is on PYTHONPATH
    ds = adapter_to_sequence_dataset(loaded_synth, "train")
    sample = ds[0]
    assert "questions" in sample and "responses" in sample and "student_id" in sample
    assert sample["questions"].dtype == torch.long
    assert sample["responses"].dtype == torch.long
    # Shape: 1-D tensor of length >= min_seq_len (3 in this fixture).
    assert sample["questions"].ndim == 1
    assert sample["questions"].shape[0] >= 3
    # student_id is 1-based and >= id_offset (default 1).
    assert int(sample["student_id"]) >= 1


def test_dataloader_emits_collate_compatible_batch(loaded_synth) -> None:
    pytest.importorskip("utils.dataloader")
    loader = adapter_to_dataloader(
        loaded_synth, "train", batch_size=4, shuffle=False,
    )
    batch = next(iter(loader))
    # Tuple of (q_pad, r_pad, mask, student_ids) per ma-irt collation.
    assert isinstance(batch, (tuple, list))
    assert len(batch) == 4
    q_pad, r_pad, mask, sid = batch
    assert q_pad.dtype == torch.long
    assert r_pad.dtype == torch.long
    assert mask.dtype == torch.bool
    assert sid.dtype == torch.long
    # Shape consistency (B, S_max).
    B, S = q_pad.shape
    assert r_pad.shape == (B, S)
    assert mask.shape == (B, S)
    assert sid.shape == (B, S)
    assert B == 4


def test_n_questions_matches_metadata(loaded_synth) -> None:
    pytest.importorskip("utils.dataloader")
    loader = adapter_to_dataloader(loaded_synth, "train", batch_size=2)
    q_pad, _, mask, _ = next(iter(loader))
    valid_ids = q_pad[mask]
    assert int(valid_ids.max().item()) <= loaded_synth.get_n_questions()
    assert int(valid_ids.min().item()) >= 1


def test_dataloader_full_pass_no_errors(loaded_synth) -> None:
    pytest.importorskip("utils.dataloader")
    loader = adapter_to_dataloader(loaded_synth, "train", batch_size=3)
    n_batches = 0
    for batch in loader:
        assert len(batch) == 4
        n_batches += 1
    assert n_batches > 0


def test_splits_disjoint(loaded_synth) -> None:
    pytest.importorskip("utils.dataloader")
    ds_train = adapter_to_sequence_dataset(loaded_synth, "train")
    ds_valid = adapter_to_sequence_dataset(loaded_synth, "valid")
    ds_test = adapter_to_sequence_dataset(loaded_synth, "test")
    n = len(loaded_synth)
    assert len(ds_train) + len(ds_valid) + len(ds_test) == n
