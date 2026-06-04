r"""Tests for the M2 retrieval pillar.

Covers JobPoolSpec.from_dataframe / load_onet_pool, RetrievalIndex
deterministic top-k with and without masks, ItemTower forward shape
and L2 normalisation, and a pool-swap test that proves the system
treats the pool as a value-type input rather than a baked-in constant.

The tests use the deterministic fallback text encoder by default
(``ItemTower(force_fallback=True)``) so they do not require a live
sentence-transformers download. The real BGE path is exercised by
``rl/scripts/register_pool.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PARQUET_PATH = REPO_ROOT / "rl" / "artifacts" / "onet_v1.parquet"

# Make the rl src tree importable when running pytest from any cwd.
import sys

_RL_SRC = REPO_ROOT / "rl" / "src"
if str(_RL_SRC) not in sys.path:
    sys.path.insert(0, str(_RL_SRC))

from irtrec.retrieval import (  # noqa: E402
    ItemTower,
    JobPoolSpec,
    RetrievalIndex,
    load_onet_pool,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


def _make_synthetic_df(n: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    riasec_alphabet = ["RIA", "ECS", "IAS", "SEC", "RIC"]
    rows = []
    for i in range(n):
        rows.append(
            {
                "occupation_code": f"99-99{i:02d}.00",
                "title": f"Synth Job {i + 1}",
                "description": f"Synthetic description for job {i + 1}.",
                "tasks_concat": f"task_a_{i} | task_b_{i} | task_c_{i}",
                "work_activities_summary": "act1; act2",
                "riasec_code": riasec_alphabet[i % len(riasec_alphabet)],
                "work_zone": int(rng.integers(1, 6)),
                "education_zscore": float(rng.normal()),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def tiny_pool() -> JobPoolSpec:
    df = _make_synthetic_df(n=5, seed=0)
    return JobPoolSpec.from_dataframe(df, source="synthetic:tiny")


@pytest.fixture(scope="module")
def big_synth_pool() -> JobPoolSpec:
    df = _make_synthetic_df(n=50, seed=123)
    return JobPoolSpec.from_dataframe(df, source="synthetic:50")


# ---------------------------------------------------------------------
# JobPoolSpec
# ---------------------------------------------------------------------


def test_load_onet_pool_schema_and_rows() -> None:
    if not PARQUET_PATH.exists():
        pytest.skip(f"{PARQUET_PATH} not built")
    pool = load_onet_pool(PARQUET_PATH)
    # 923 rows in the locked v1 parquet, but be lenient if rebuilt.
    assert pool.n_jobs >= 800
    assert len(pool.occupation_codes) == pool.n_jobs
    assert len(pool.titles) == pool.n_jobs
    assert len(pool.descriptions) == pool.n_jobs
    assert len(pool.tasks_concat) == pool.n_jobs
    assert len(pool.riasec_codes) == pool.n_jobs
    assert pool.structured_features.shape == (pool.n_jobs, 8)
    assert pool.delta_j.shape == (pool.n_jobs,)
    assert pool.work_zones_raw.shape == (pool.n_jobs,)
    # work_zone_scaled column should sit in [-1, 1].
    wz_scaled = pool.structured_features[:, 0]
    assert wz_scaled.min() >= -1.0 - 1e-6
    assert wz_scaled.max() <= 1.0 + 1e-6
    # delta_j is z-scored so mean near zero, std near one.
    assert abs(float(pool.delta_j.mean())) < 1e-3
    assert abs(float(pool.delta_j.std()) - 1.0) < 1e-3


def test_attach_text_format(tiny_pool: JobPoolSpec) -> None:
    texts = tiny_pool.attach_text()
    assert len(texts) == tiny_pool.n_jobs
    for i, text in enumerate(texts):
        assert tiny_pool.titles[i] in text
        assert tiny_pool.descriptions[i] in text
        assert "Tasks:" in text


def test_education_mask_set_for_missing_values() -> None:
    df = _make_synthetic_df(n=4, seed=1)
    df.loc[1, "education_zscore"] = float("nan")
    pool = JobPoolSpec.from_dataframe(df, source="synthetic:nan")
    # education_mask sits in column 2.
    assert pool.structured_features[1, 2] == 0.0
    assert pool.structured_features[0, 2] == 1.0
    # The filled value at the NaN row must be zero.
    assert pool.structured_features[1, 1] == 0.0


# ---------------------------------------------------------------------
# ItemTower forward
# ---------------------------------------------------------------------


def test_item_tower_forward_l2_normalised(tiny_pool: JobPoolSpec) -> None:
    torch.manual_seed(0)
    tower = ItemTower(d=64, device="cpu", force_fallback=True)
    tower.eval()
    with torch.no_grad():
        v_j = tower(tiny_pool)
    assert v_j.shape == (tiny_pool.n_jobs, 64)
    norms = v_j.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_item_tower_text_encoder_is_frozen() -> None:
    torch.manual_seed(0)
    tower = ItemTower(d=64, device="cpu", force_fallback=True)
    fallback = tower.text_fallback
    assert fallback is not None
    for param in fallback.parameters():
        assert not param.requires_grad


# ---------------------------------------------------------------------
# RetrievalIndex
# ---------------------------------------------------------------------


def test_retrieval_topk_deterministic_and_unmasked() -> None:
    rng = np.random.default_rng(42)
    v_j = rng.standard_normal((10, 8)).astype(np.float32)
    v_j /= np.linalg.norm(v_j, axis=1, keepdims=True)
    ids = [f"id-{i}" for i in range(10)]
    idx = RetrievalIndex().fit(v_j, ids)

    q = v_j[3]  # Query equals job 3, so job 3 should rank first.
    top_ids, scores = idx.topk(q, k=3)
    assert top_ids[0] == "id-3"
    assert scores.shape == (3,)
    assert np.all(np.diff(scores) <= 1e-6)

    # Calling topk again with the same input must give identical output.
    again_ids, again_scores = idx.topk(q, k=3)
    assert again_ids == top_ids
    np.testing.assert_array_equal(again_scores, scores)


def test_retrieval_topk_respects_mask() -> None:
    rng = np.random.default_rng(7)
    v_j = rng.standard_normal((6, 4)).astype(np.float32)
    v_j /= np.linalg.norm(v_j, axis=1, keepdims=True)
    ids = [f"id-{i}" for i in range(6)]
    idx = RetrievalIndex().fit(v_j, ids)
    q = v_j[0]

    mask = np.array([False, True, True, True, True, True])
    top_ids, _ = idx.topk(q, k=3, mask=mask)
    assert "id-0" not in top_ids
    assert all(job_id in {f"id-{i}" for i in range(1, 6)} for job_id in top_ids)


def test_retrieval_topk_validates_inputs() -> None:
    v_j = np.eye(4, dtype=np.float32)
    ids = [f"id-{i}" for i in range(4)]
    idx = RetrievalIndex().fit(v_j, ids)
    with pytest.raises(ValueError):
        idx.topk(np.zeros(3, dtype=np.float32), k=2)
    with pytest.raises(ValueError):
        idx.topk(np.zeros(4, dtype=np.float32), k=0)
    with pytest.raises(ValueError):
        idx.topk(np.zeros(4, dtype=np.float32), k=5)


# ---------------------------------------------------------------------
# Pool swap test
# ---------------------------------------------------------------------


def test_pool_swap_round_trip(big_synth_pool: JobPoolSpec) -> None:
    torch.manual_seed(0)
    tower = ItemTower(d=64, device="cpu", force_fallback=True)
    tower.eval()
    with torch.no_grad():
        v_j = tower(big_synth_pool).cpu().numpy()
    assert v_j.shape == (50, 64)
    idx = RetrievalIndex().fit(v_j, big_synth_pool.occupation_codes)
    assert idx.n_jobs == 50
    assert idx.dim == 64

    # Query for the first job's own embedding. The deterministic top-1
    # must be that same job.
    q = v_j[0]
    top_ids, _ = idx.topk(q, k=5)
    assert top_ids[0] == big_synth_pool.occupation_codes[0]
    assert len(top_ids) == 5
    assert len(set(top_ids)) == 5

    # Swap pools and re-fit. The index state must come from the new pool.
    other_df = _make_synthetic_df(n=20, seed=999)
    other_pool = JobPoolSpec.from_dataframe(other_df, source="synthetic:swap")
    with torch.no_grad():
        v_j_other = tower(other_pool).cpu().numpy()
    idx.fit(v_j_other, other_pool.occupation_codes)
    assert idx.n_jobs == 20
    top_ids_after, _ = idx.topk(v_j_other[3], k=3)
    assert top_ids_after[0] == other_pool.occupation_codes[3]
    # And every returned id must come from the new pool.
    new_set = set(other_pool.occupation_codes)
    for job_id in top_ids_after:
        assert job_id in new_set
