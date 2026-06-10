"""Probe-set samplers for OrdRecEnv.

Two strategies are provided, switchable via the ``probe_sampler``
config field in :class:`~ordrec.reward.config.RewardConfig`:

``"uniform"``
    Independent uniform random draws over the allowed item bank for
    each batch row. No item-difficulty information is used. Fast and
    suitable for quick experiments where the item cache has no
    beta information.

``"stratified"`` (default)
    The item bank is partitioned into ``n_difficulty_strata`` equal-count
    quantile bins ranked by mean step-threshold (mean across K-1 beta
    dimensions for each item). The sampler draws items evenly from each
    stratum so the probe covers the full difficulty range. The number
    per stratum is ``ceil(n_items / n_strata)``; any remainder is filled
    from a random stratum. Stratified sampling is per-episode: strata
    are computed once from the item cache on construction.

Interface. Both samplers implement the same call signature::

    sample(rng, allowed, n) -> np.ndarray of shape (n,)

where ``allowed`` is a 1D array of 1-based item ids and ``n`` is the
total number of items to draw (``probe_M + probe_H``). Return order
within each call is arbitrary; callers slice the first ``probe_M`` for
C and the remainder for H.

See ``docs/ordrec_impl_guide.md`` Section 3.6 for the design notes.
"""

from __future__ import annotations

import abc
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------


class ProbeSampler(abc.ABC):
    """Abstract base for probe-set samplers.

    Subclasses implement :meth:`sample` which draws ``n`` item ids
    without replacement from the ``allowed`` array.
    """

    @abc.abstractmethod
    def sample(
        self, rng: np.random.Generator, allowed: np.ndarray, n: int
    ) -> np.ndarray:
        """Draw ``n`` item ids without replacement from ``allowed``.

        Args:
            rng: NumPy default RNG for reproducible sampling.
            allowed: 1D int64 array of 1-based item ids eligible for
                this row's probe draw. May be a subset of the full item
                bank when warmup items are excluded.
            n: Total items to draw (``probe_M + probe_H``).

        Returns:
            1D int64 array of length ``n``.

        Raises:
            ValueError: When ``n > len(allowed)`` with the default
                ``replace=False`` setting. Callers should ensure
                ``probe_M + probe_H <= len(allowed)``.
        """


# ---------------------------------------------------------------------------
# Uniform sampler
# ---------------------------------------------------------------------------


class UniformProbeSampler(ProbeSampler):
    """Uniform random probe sampler.

    Samples ``n`` items uniformly at random without replacement from
    the ``allowed`` array. No item-difficulty information is used.
    Suitable for quick experiments and cases where the item cache has
    no beta information.
    """

    def sample(
        self, rng: np.random.Generator, allowed: np.ndarray, n: int
    ) -> np.ndarray:
        """Draw ``n`` items uniformly without replacement."""
        if n > len(allowed):
            raise ValueError(
                f"UniformProbeSampler: requested n={n} exceeds "
                f"len(allowed)={len(allowed)}."
            )
        return rng.choice(allowed, size=n, replace=False)


# ---------------------------------------------------------------------------
# Difficulty-stratified sampler
# ---------------------------------------------------------------------------


class StratifiedProbeSampler(ProbeSampler):
    """Difficulty-stratified probe sampler.

    Divides the full item bank (rows 1..Q) into ``n_strata`` equal-count
    quantile bins ordered by mean step-threshold (``beta_table[q].mean()``
    over the K-1 threshold dimensions). The sampler draws items evenly
    from each stratum so every probe covers the full difficulty range.

    Construction is O(Q log Q) (sort). Each :meth:`sample` call intersects
    the per-stratum item sets with ``allowed`` and draws proportionally.

    When a stratum has fewer eligible items than its quota, the deficit
    is filled from the remaining strata by sampling uniformly from the
    combined residual pool. This guarantees exactly ``n`` items are
    returned even with tiny item banks or large warmup sets.

    Args:
        beta_table: ``np.ndarray (Q + 1, K - 1)``. Row 0 is the padding
            row and is ignored. Rows 1..Q are used to rank items by
            difficulty.
        n_strata: Number of equal-count quantile bins. Default ``5``.
    """

    def __init__(
        self,
        beta_table: np.ndarray,
        n_strata: int = 5,
    ) -> None:
        if n_strata < 1:
            raise ValueError(f"n_strata must be >= 1, got {n_strata}")
        if beta_table.ndim != 2:
            raise ValueError(
                f"beta_table must be 2D (Q+1, K-1), got shape {beta_table.shape}"
            )
        self.n_strata = int(n_strata)

        Q = beta_table.shape[0] - 1  # rows 1..Q are real items
        if Q <= 0:
            raise ValueError(f"beta_table has no real items (Q={Q}).")

        # Rank items 1..Q by mean difficulty.
        mean_beta = beta_table[1:, :].mean(axis=-1)  # (Q,)
        ranked_ids = np.argsort(mean_beta, kind="stable") + 1  # 1-based, (Q,)

        # Partition ranked ids into n_strata equal-count bins.
        self._strata: list[np.ndarray] = []
        split_points = np.array_split(ranked_ids, n_strata)
        for sp in split_points:
            self._strata.append(sp.astype(np.int64))

        # Build a set version for fast intersection.
        self._strata_sets: list[set] = [set(s.tolist()) for s in self._strata]

    def sample(
        self, rng: np.random.Generator, allowed: np.ndarray, n: int
    ) -> np.ndarray:
        """Draw ``n`` items spread evenly across difficulty strata.

        Items in ``allowed`` that are not present in any stratum (e.g.
        items from a small synthetic bank) fall back to a uniform draw.

        Args:
            rng: NumPy default RNG.
            allowed: 1D int64 array of eligible item ids.
            n: Total number of items to draw.

        Returns:
            1D int64 array of length ``n``.
        """
        if n <= 0:
            return np.empty(0, dtype=np.int64)

        allowed_set = set(allowed.tolist())
        if len(allowed_set) < n:
            raise ValueError(
                f"StratifiedProbeSampler: requested n={n} exceeds "
                f"len(allowed)={len(allowed_set)}."
            )

        # Intersect each stratum with allowed.
        stratum_pools = [
            np.array(sorted(s & allowed_set), dtype=np.int64)
            for s in self._strata_sets
        ]

        # How many to draw per stratum (floor allocation).
        base = n // self.n_strata
        remainder = n % self.n_strata
        quotas = [base] * self.n_strata
        for i in range(remainder):
            quotas[i] += 1

        drawn: list[np.ndarray] = []
        residual: list[int] = []

        for i, (pool, quota) in enumerate(zip(stratum_pools, quotas)):
            if quota == 0:
                continue
            if len(pool) >= quota:
                drawn.append(rng.choice(pool, size=quota, replace=False))
            else:
                # Draw everything available; note deficit.
                drawn.append(pool)
                residual.append(quota - len(pool))

        total_deficit = sum(residual)
        if total_deficit > 0:
            # Collect ids already drawn to avoid repeats.
            drawn_set: set = set()
            for arr in drawn:
                drawn_set.update(arr.tolist())
            leftover = np.array(
                [x for x in allowed_set if x not in drawn_set], dtype=np.int64
            )
            if len(leftover) < total_deficit:
                # Should not happen given the len(allowed_set) >= n guard.
                leftover = np.array(
                    sorted(allowed_set - drawn_set), dtype=np.int64
                )
            extra = rng.choice(leftover, size=total_deficit, replace=False)
            drawn.append(extra)

        result = np.concatenate(drawn)
        # Shuffle so the caller's slice (first M for C, next H for H_probe)
        # does not systematically pick easy items for C and hard for H.
        rng.shuffle(result)
        return result[:n].astype(np.int64)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_probe_sampler(
    mode: str,
    beta_table: Optional[np.ndarray] = None,
    n_strata: int = 5,
) -> ProbeSampler:
    """Construct a :class:`ProbeSampler` by name.

    Args:
        mode: ``"uniform"`` or ``"stratified"``.
        beta_table: Required when ``mode == "stratified"``.
        n_strata: Number of strata for the stratified sampler.

    Returns:
        A :class:`ProbeSampler` instance.

    Raises:
        ValueError: On unknown mode or missing ``beta_table`` for the
            stratified sampler.
    """
    if mode == "uniform":
        return UniformProbeSampler()
    if mode == "stratified":
        if beta_table is None:
            raise ValueError(
                "make_probe_sampler: beta_table is required for mode='stratified'."
            )
        return StratifiedProbeSampler(beta_table, n_strata=n_strata)
    raise ValueError(
        f"Unknown probe_sampler mode '{mode}'. "
        "Expected one of: 'uniform', 'stratified'."
    )


__all__ = [
    "ProbeSampler",
    "StratifiedProbeSampler",
    "UniformProbeSampler",
    "make_probe_sampler",
]
