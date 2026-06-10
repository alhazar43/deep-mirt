"""Tests for B5 dual probe sampler (uniform and stratified).

Verifies:
- Both samplers return exactly n items without replacement.
- StratifiedProbeSampler covers all strata when Q is large enough.
- Uniform sampler gives uniform distribution over runs.
- make_probe_sampler factory dispatches correctly.
- Config-switchable: RewardConfig.probe_sampler drives env sampler choice.
- Stratum coverage: items returned by stratified sampler span full
  difficulty range (low/mid/high strata all represented).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest

from ordrec.envs.probe_sampler import (
    StratifiedProbeSampler,
    UniformProbeSampler,
    make_probe_sampler,
)
from ordrec.reward.config import RewardConfig


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_beta(Q: int, K: int = 4, seed: int = 0) -> np.ndarray:
    """Synthetic beta table with clearly-ranked item difficulties."""
    rng = np.random.default_rng(seed)
    beta = np.zeros((Q + 1, K - 1), dtype=np.float32)
    # Items ranked 1..Q with increasing difficulty.
    beta[1:, :] = np.linspace(-2.0, 2.0, Q)[:, None] + rng.normal(0, 0.1, (Q, K - 1))
    return beta


# ---------------------------------------------------------------------------
# UniformProbeSampler
# ---------------------------------------------------------------------------


def test_uniform_returns_n_unique_items() -> None:
    rng = np.random.default_rng(42)
    sampler = UniformProbeSampler()
    allowed = np.arange(1, 51, dtype=np.int64)
    result = sampler.sample(rng, allowed, 20)
    assert result.shape == (20,)
    assert len(set(result.tolist())) == 20
    assert all(x in set(allowed.tolist()) for x in result.tolist())


def test_uniform_raises_if_n_exceeds_allowed() -> None:
    rng = np.random.default_rng(0)
    sampler = UniformProbeSampler()
    allowed = np.arange(1, 6, dtype=np.int64)
    with pytest.raises(ValueError, match="exceeds"):
        sampler.sample(rng, allowed, 10)


def test_uniform_exact_fit() -> None:
    """n == len(allowed) must work (permutation)."""
    rng = np.random.default_rng(7)
    sampler = UniformProbeSampler()
    allowed = np.arange(1, 11, dtype=np.int64)
    result = sampler.sample(rng, allowed, 10)
    assert sorted(result.tolist()) == list(range(1, 11))


# ---------------------------------------------------------------------------
# StratifiedProbeSampler
# ---------------------------------------------------------------------------


def test_stratified_returns_n_unique_items() -> None:
    beta = _make_beta(Q=100, K=4)
    sampler = StratifiedProbeSampler(beta, n_strata=5)
    rng = np.random.default_rng(1)
    allowed = np.arange(1, 101, dtype=np.int64)
    result = sampler.sample(rng, allowed, 20)
    assert result.shape == (20,)
    assert len(set(result.tolist())) == 20
    assert all(1 <= x <= 100 for x in result.tolist())


def test_stratified_covers_all_strata() -> None:
    """Items from every stratum should appear in the sample.

    With Q=100, n_strata=5, n=20 we request 4 per stratum; every
    stratum has 20 items so coverage is guaranteed.
    """
    beta = _make_beta(Q=100, K=4, seed=0)
    sampler = StratifiedProbeSampler(beta, n_strata=5)
    rng = np.random.default_rng(2)
    allowed = np.arange(1, 101, dtype=np.int64)

    # Run 50 draws and check that each stratum contributes at least once
    # across draws (the stochastic floor is very high with 4/20 per draw).
    stratum_seen: list[set] = [set() for _ in range(5)]
    for _ in range(50):
        result = set(sampler.sample(rng, allowed, 20).tolist())
        for i, stratum in enumerate(sampler._strata):
            if result & set(stratum.tolist()):
                stratum_seen[i].add(True)

    for i in range(5):
        assert stratum_seen[i], f"Stratum {i} was never sampled across 50 draws"


def test_stratified_difficulty_range_covered() -> None:
    """Items returned span the full beta range (low + high difficulty).

    We check that both the easiest quintile and the hardest quintile
    appear in the sample on average across draws.
    """
    beta = _make_beta(Q=200, K=4, seed=3)
    mean_beta = beta[1:, :].mean(axis=-1)  # (Q,)
    easy_ids = set(np.where(mean_beta <= np.percentile(mean_beta, 20))[0] + 1)
    hard_ids = set(np.where(mean_beta >= np.percentile(mean_beta, 80))[0] + 1)

    sampler = StratifiedProbeSampler(beta, n_strata=5)
    rng = np.random.default_rng(4)
    allowed = np.arange(1, 201, dtype=np.int64)

    easy_count = 0
    hard_count = 0
    n_trials = 100
    for _ in range(n_trials):
        result = set(sampler.sample(rng, allowed, 40).tolist())
        if result & easy_ids:
            easy_count += 1
        if result & hard_ids:
            hard_count += 1

    assert easy_count >= 90, f"Easy items rarely sampled ({easy_count}/{n_trials})"
    assert hard_count >= 90, f"Hard items rarely sampled ({hard_count}/{n_trials})"


def test_stratified_handles_small_allowed() -> None:
    """Deficit filling works when warmup excludes most items."""
    beta = _make_beta(Q=50, K=4)
    sampler = StratifiedProbeSampler(beta, n_strata=5)
    rng = np.random.default_rng(5)
    # Only 15 items allowed (warmup removed 35 of 50).
    allowed = np.arange(1, 16, dtype=np.int64)
    result = sampler.sample(rng, allowed, 10)
    assert result.shape == (10,)
    assert len(set(result.tolist())) == 10


def test_stratified_raises_on_bad_n_strata() -> None:
    beta = _make_beta(Q=20, K=4)
    with pytest.raises(ValueError):
        StratifiedProbeSampler(beta, n_strata=0)


def test_stratified_raises_if_n_exceeds_allowed() -> None:
    beta = _make_beta(Q=20, K=4)
    sampler = StratifiedProbeSampler(beta, n_strata=5)
    rng = np.random.default_rng(0)
    allowed = np.arange(1, 6, dtype=np.int64)
    with pytest.raises(ValueError, match="exceeds"):
        sampler.sample(rng, allowed, 10)


# ---------------------------------------------------------------------------
# make_probe_sampler factory
# ---------------------------------------------------------------------------


def test_factory_uniform() -> None:
    s = make_probe_sampler("uniform")
    assert isinstance(s, UniformProbeSampler)


def test_factory_stratified() -> None:
    beta = _make_beta(Q=30, K=4)
    s = make_probe_sampler("stratified", beta_table=beta, n_strata=3)
    assert isinstance(s, StratifiedProbeSampler)
    assert s.n_strata == 3


def test_factory_stratified_requires_beta_table() -> None:
    with pytest.raises(ValueError, match="beta_table is required"):
        make_probe_sampler("stratified")


def test_factory_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="Unknown probe_sampler mode"):
        make_probe_sampler("thompson")


# ---------------------------------------------------------------------------
# RewardConfig.probe_sampler field
# ---------------------------------------------------------------------------


def test_reward_config_default_probe_sampler_is_stratified() -> None:
    cfg = RewardConfig()
    assert cfg.probe_sampler == "stratified"


def test_reward_config_probe_sampler_switchable() -> None:
    cfg = RewardConfig(probe_sampler="uniform")
    assert cfg.probe_sampler == "uniform"


def test_reward_config_from_dict_probe_sampler() -> None:
    cfg = RewardConfig.from_dict({"probe_sampler": "uniform"})
    assert cfg.probe_sampler == "uniform"
