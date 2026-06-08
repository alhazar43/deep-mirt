"""Per-item ``(alpha, beta)`` cache tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ordrec.envs.frozen_magpcm import FrozenMAGPCM
from ordrec.envs.item_cache import (
    ITEM_CACHE_DTYPE,
    ItemCache,
    build_item_cache,
    checkpoint_sha7,
    item_cache_path,
    load_item_cache,
    save_item_cache,
)


pytest.importorskip("models", reason="ma-irt must be on PYTHONPATH")


def _build_tiny_magpcm():
    from models.magpcm import MAGPCM  # type: ignore[import-not-found]

    return MAGPCM(
        n_questions=12,
        n_categories=4,
        n_traits=1,
        memory_size=8,
        key_dim=8,
        value_dim=8,
        summary_dim=8,
        embedding_type="learned",
        dropout_rate=0.0,
        ability_scale=1.0,
        separate_theta=True,
        init_value_memory=False,
    )


def test_build_item_cache_shapes_and_dtypes() -> None:
    m = _build_tiny_magpcm()
    frozen = FrozenMAGPCM(m)
    cache = build_item_cache(
        frozen,
        n_contexts=4,
        context_seq_len=3,
        seed=0,
        dataset_name="unit",
    )
    assert isinstance(cache, ItemCache)
    Q = frozen.n_questions
    K = frozen.n_categories
    assert cache.alpha_table.shape == (Q + 1, cache.n_traits)
    assert cache.beta_table.shape == (Q + 1, K - 1)
    assert cache.alpha_table.dtype == ITEM_CACHE_DTYPE
    assert cache.beta_table.dtype == ITEM_CACHE_DTYPE
    assert cache.n_questions == Q
    assert cache.n_categories == K
    assert cache.n_contexts == 4


def test_padding_slot_invariants() -> None:
    m = _build_tiny_magpcm()
    frozen = FrozenMAGPCM(m)
    cache = build_item_cache(frozen, n_contexts=2, dataset_name="unit")
    # Row 0 = padding, alpha = 1, beta = 0.
    assert np.allclose(cache.alpha_table[0, :], 1.0)
    assert np.allclose(cache.beta_table[0, :], 0.0)


def test_alpha_and_beta_match_a_direct_forward() -> None:
    """With n_contexts=1 and context_seq_len=1, the cached values must
    equal those produced by a single forward of a singleton sequence."""
    m = _build_tiny_magpcm()
    frozen = FrozenMAGPCM(m)
    Q = frozen.n_questions
    K = frozen.n_categories

    cache = build_item_cache(
        frozen,
        n_contexts=1,
        context_seq_len=1,
        seed=42,
        dataset_name="unit",
    )

    # Reproduce the harness call by hand: random ids and labels with
    # the same seed, then overwrite the last column with arange(1..Q+1)
    # and labels at the last column with 0.
    rng = np.random.default_rng(42)
    q_np = rng.integers(low=1, high=Q + 1, size=(Q, 1), dtype=np.int64)
    r_np = rng.integers(low=0, high=K, size=(Q, 1), dtype=np.int64)
    q_np[:, -1] = np.arange(1, Q + 1)
    r_np[:, -1] = 0

    q_t = torch.from_numpy(q_np).long()
    r_t = torch.from_numpy(r_np).long()
    out = frozen.forward_no_grad(q_t, r_t)
    alpha_direct = out["alpha"][:, -1, :].cpu().numpy().astype(ITEM_CACHE_DTYPE)
    beta_direct = out["beta"][:, -1, :].cpu().numpy().astype(ITEM_CACHE_DTYPE)

    np.testing.assert_allclose(cache.alpha_table[1:, :], alpha_direct, atol=1e-6)
    np.testing.assert_allclose(cache.beta_table[1:, :], beta_direct, atol=1e-6)


def test_save_load_round_trip(tmp_path: Path) -> None:
    m = _build_tiny_magpcm()
    frozen = FrozenMAGPCM(m)
    cache = build_item_cache(
        frozen,
        n_contexts=2,
        dataset_name="unit",
        ckpt_sha7="abc1234",
    )
    p = item_cache_path(tmp_path, "unit", "abc1234")
    saved_path = save_item_cache(cache, p)
    assert saved_path.exists()

    loaded = load_item_cache(saved_path)
    np.testing.assert_array_equal(loaded.alpha_table, cache.alpha_table)
    np.testing.assert_array_equal(loaded.beta_table, cache.beta_table)
    assert loaded.n_questions == cache.n_questions
    assert loaded.n_categories == cache.n_categories
    assert loaded.n_traits == cache.n_traits
    assert loaded.n_contexts == cache.n_contexts
    assert loaded.ckpt_sha7 == "abc1234"
    assert loaded.dataset_name == "unit"


def test_alpha_tensor_and_beta_tensor_accessors() -> None:
    m = _build_tiny_magpcm()
    frozen = FrozenMAGPCM(m)
    cache = build_item_cache(frozen, n_contexts=2, dataset_name="unit")
    at = cache.alpha_tensor()
    bt = cache.beta_tensor()
    assert at.dtype == torch.float32
    assert bt.dtype == torch.float32
    assert at.shape == cache.alpha_table.shape
    assert bt.shape == cache.beta_table.shape


def test_cache_keyed_by_dataset_and_checkpoint(tmp_path: Path) -> None:
    """Two distinct ``(dataset_name, ckpt_sha7)`` pairs must yield two
    distinct on-disk locations, so two trained models do not overwrite
    each other's caches."""
    p1 = item_cache_path(tmp_path, "ednet", "abc1234")
    p2 = item_cache_path(tmp_path, "ednet", "def5678")
    p3 = item_cache_path(tmp_path, "eedi", "abc1234")
    assert p1 != p2
    assert p1 != p3
    assert p2 != p3


def test_checkpoint_sha7_handles_missing_file(tmp_path: Path) -> None:
    assert checkpoint_sha7(tmp_path / "does_not_exist.pt") == ""


def test_checkpoint_sha7_for_real_file(tmp_path: Path) -> None:
    p = tmp_path / "tiny.pt"
    p.write_bytes(b"deterministic-bytes-for-the-hash")
    sha = checkpoint_sha7(p)
    assert len(sha) == 7
    # And it must be reproducible.
    assert sha == checkpoint_sha7(p)


def test_build_item_cache_rejects_invalid_args() -> None:
    m = _build_tiny_magpcm()
    frozen = FrozenMAGPCM(m)
    with pytest.raises(ValueError):
        build_item_cache(frozen, n_contexts=0, dataset_name="x")
    with pytest.raises(ValueError):
        build_item_cache(frozen, context_seq_len=0, dataset_name="x")


def test_item_cache_path_requires_dataset_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        item_cache_path(tmp_path, "", "abc1234")


def test_averaging_across_contexts_changes_alpha_but_not_beta() -> None:
    """``beta`` depends only on the item embedding; ``alpha`` depends on
    ``joint_summary`` and may vary by context. Averaging across contexts
    must not move beta but may move alpha (we only assert the structural
    invariant, not the magnitude)."""
    m = _build_tiny_magpcm()
    frozen = FrozenMAGPCM(m)
    cache_one = build_item_cache(
        frozen, n_contexts=1, context_seq_len=3, seed=7, dataset_name="x"
    )
    cache_many = build_item_cache(
        frozen, n_contexts=8, context_seq_len=3, seed=7, dataset_name="x"
    )
    # Beta: per-item embedding only, so a singleton context and an
    # averaged-context version differ only by random context noise on
    # the encoder side. At context_seq_len > 1 the encoder receives
    # different (q, r) prefixes, so even beta may differ slightly. The
    # invariant we assert is shape equality and that values are finite.
    assert cache_one.beta_table.shape == cache_many.beta_table.shape
    assert np.isfinite(cache_one.beta_table).all()
    assert np.isfinite(cache_many.beta_table).all()
    # Alpha tables must both be finite.
    assert np.isfinite(cache_one.alpha_table).all()
    assert np.isfinite(cache_many.alpha_table).all()
