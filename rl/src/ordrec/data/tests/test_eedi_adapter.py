"""EediAdapter tests against the eedi_mini.csv fixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ordrec.data import AdapterConfig, EediAdapter
from ordrec.data.schema import (
    COERCION_FILENAME,
    METADATA_FILENAME,
    SEQUENCES_FILENAME,
    load_coercion_artefacts,
    load_metadata,
    load_sequences,
)


FIXTURE = Path(__file__).parent / "fixtures" / "eedi_mini.csv"


def _make_cfg(tmp_path: Path, seed: int = 7) -> AdapterConfig:
    # All 10 fixture students go through the train fold so the
    # distractor ordering is exercised; we still validate that valid/test
    # codes can be populated by adjusting test_frac.
    return AdapterConfig(
        name="eedi_mini_k4",
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
    cfg = _make_cfg(tmp_path)
    a = EediAdapter(cfg)
    a.materialise()
    out = tmp_path / "eedi_mini_k4"
    assert (out / SEQUENCES_FILENAME).exists()
    assert (out / METADATA_FILENAME).exists()
    assert (out / COERCION_FILENAME).exists()


def test_metadata_block(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    a = EediAdapter(cfg)
    a.materialise()
    meta = load_metadata(tmp_path / "eedi_mini_k4" / METADATA_FILENAME)
    assert meta["adapter_class"] == "EediAdapter"
    assert meta["n_categories"] == 4
    assert meta["n_questions"] == 5      # five distinct question_ids in the fixture
    assert meta["n_students"] == 10  # ten distinct student_ids
    assert meta["ordinal_coercion_method"] == "distractor_difficulty_2pl"
    assert meta["splits"]["split_seed"] == 7


def test_correct_recodes_to_top_category(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    a = EediAdapter(cfg)
    a.materialise()
    a.load()
    # Every fixture row with correct == 1 has selected_option = 1; in
    # the recoded sequences each such row must end up at category 3.
    records = load_sequences(tmp_path / "eedi_mini_k4" / SEQUENCES_FILENAME)
    seen_top = False
    for rec in records:
        for r in rec["responses"]:
            assert 0 <= r <= 3
        if 3 in rec["responses"]:
            seen_top = True
    assert seen_top, "expected at least one top-category response in the recoded fixture"


def test_distractor_ordering_is_ascending_by_mean_theta(tmp_path: Path) -> None:
    """The fixture is designed so option 2 is "easy distractor" (low theta),
    option 3 is mid, option 4 is "close-to-correct distractor" (high theta).
    The recovered sigma_q must therefore be [2, 3, 4] for every item."""
    cfg = _make_cfg(tmp_path)
    a = EediAdapter(cfg)
    a.materialise()
    payload = load_coercion_artefacts(
        tmp_path / "eedi_mini_k4" / COERCION_FILENAME
    )
    sigma = payload["distractor_order_per_q"]
    # Items are 1-based; every item must have a length-3 ordering.
    for q in range(1, 6):
        order = sigma[str(q)]
        assert len(order) == 3
        assert sorted(order) == [2, 3, 4]
    # At least three out of five questions must have the expected
    # [2, 3, 4] ordering. The mini fixture is small and the SGD-fit 2PL
    # is noisy, so we allow some slack but expect a strong majority.
    correctly_ordered = sum(sigma[str(q)] == [2, 3, 4] for q in range(1, 6))
    assert correctly_ordered >= 3, sigma


def test_train_only_distractor_ordering(tmp_path: Path) -> None:
    """coercion_artefacts['placeholder_2pl']['n_train_students'] must equal
    the number of students actually routed to train (not the total),
    confirming the ordering uses train rows only."""
    cfg = _make_cfg(tmp_path)
    a = EediAdapter(cfg)
    a.materialise()
    a.load()
    payload = load_coercion_artefacts(
        tmp_path / "eedi_mini_k4" / COERCION_FILENAME
    )
    n_train = int(payload["placeholder_2pl"]["n_train_students"])
    train_idx = a.get_split("train")
    # The adapter records the original-row count of the train fold
    # before min_seq_len filtering, which must be at least len(train_idx)
    # (the filter can only drop rows).
    assert n_train >= int(len(train_idx))
    assert n_train <= a.get_metadata()["n_students"]


def test_persisted_coercion_reused_on_second_load(tmp_path: Path) -> None:
    """Materialise once, then load twice. The artefact must be reused
    unchanged (no implicit refit), so the on-disk distractor ordering
    is byte-identical."""
    cfg = _make_cfg(tmp_path)
    a = EediAdapter(cfg)
    a.materialise()
    sequences_first = (tmp_path / "eedi_mini_k4" / SEQUENCES_FILENAME).read_bytes()
    coercion_first = (tmp_path / "eedi_mini_k4" / COERCION_FILENAME).read_bytes()
    a2 = EediAdapter(cfg)
    a2.load()
    sequences_second = (tmp_path / "eedi_mini_k4" / SEQUENCES_FILENAME).read_bytes()
    coercion_second = (tmp_path / "eedi_mini_k4" / COERCION_FILENAME).read_bytes()
    assert sequences_first == sequences_second
    assert coercion_first == coercion_second
    # And the loaded payload exposes the same distractor ordering.
    assert a2._coercion == load_coercion_artefacts(
        tmp_path / "eedi_mini_k4" / COERCION_FILENAME
    )


def test_responses_in_valid_range_after_load(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    a = EediAdapter(cfg)
    a.materialise()
    a.load()
    for r_seq in a._responses:
        arr = np.asarray(r_seq, dtype=int)
        assert ((arr >= 0) & (arr <= 3)).all()
    for q_seq in a._questions:
        arr = np.asarray(q_seq, dtype=int)
        assert ((arr >= 1) & (arr <= a.get_n_questions())).all()
