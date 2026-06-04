"""Attach the O*NET parquet to the synthetic data generator.

This module loads ``rl/artifacts/onet_v1.parquet`` (built by
``rl/scripts/build_onet_pool.py``), z-scores ``work_zone`` to produce the
scalar ``delta_j`` used by the Option A preference model (Section 5.3 of
the v1 plan), and exposes the structured features used by downstream
code.

The returned :class:`OnetPool` object holds parallel arrays indexed by
position in the pool. The integer ``job_id`` used by ``likes.json`` is
the 0-based position (consistent with how the long-form likes table is
emitted by :mod:`synth_likes`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class OnetPool:
    """A loaded and feature-attached O*NET pool.

    Attributes
    ----------
    job_ids
        0-based integer IDs for each pool entry. Length ``n_jobs``.
    occupation_codes
        O*NET-SOC code strings, parallel to ``job_ids``.
    titles
        Occupation titles, parallel to ``job_ids``.
    descriptions
        Occupation descriptions, parallel to ``job_ids``.
    tasks_concat
        Concatenated task statements per occupation.
    work_activities_summary
        Top-importance work activities, short text.
    riasec_code
        3-letter Holland code (may be empty string when missing).
    work_zone
        Integer 1..5 work-zone level per occupation.
    education_zscore
        Pre-computed education z-score from the parquet.
    delta_j
        Z-scored ``work_zone`` across the pool, the v1 delta_j source.
    """

    job_ids: np.ndarray
    occupation_codes: List[str]
    titles: List[str]
    descriptions: List[str]
    tasks_concat: List[str]
    work_activities_summary: List[str]
    riasec_code: List[str]
    work_zone: np.ndarray
    education_zscore: np.ndarray
    delta_j: np.ndarray

    @property
    def n_jobs(self) -> int:
        return int(self.job_ids.shape[0])

    def to_jobs_records(self) -> List[dict]:
        """Long-form ``jobs.json`` records, one per occupation.

        Each record carries the structured features needed by Section 5
        downstream consumers (ItemTower in M2 reads the same parquet
        directly, but the generated dataset duplicates them so a fitted
        artifact is self-contained).
        """
        records: List[dict] = []
        for i in range(self.n_jobs):
            records.append(
                {
                    "job_id": int(self.job_ids[i]),
                    "occupation_code": self.occupation_codes[i],
                    "title": self.titles[i],
                    "description": self.descriptions[i],
                    "tasks_concat": self.tasks_concat[i],
                    "work_activities_summary": self.work_activities_summary[i],
                    "riasec_code": self.riasec_code[i],
                    "work_zone": int(self.work_zone[i]),
                    "education_zscore": (
                        float(self.education_zscore[i])
                        if np.isfinite(self.education_zscore[i])
                        else None
                    ),
                    "delta_j": float(self.delta_j[i]),
                }
            )
        return records


def _zscore(values: np.ndarray) -> np.ndarray:
    """Population z-score, NaN-safe.

    Uses ddof=0 so the pool mean and std are population estimates.
    Constant or single-valued inputs return zeros (defined behaviour).
    """
    finite = values[np.isfinite(values)]
    if finite.size <= 1:
        return np.zeros_like(values, dtype=np.float64)
    mean = float(finite.mean())
    std = float(finite.std(ddof=0))
    if std <= 0.0:
        return np.zeros_like(values, dtype=np.float64)
    return (values.astype(np.float64) - mean) / std


def load_onet_pool(parquet_path: Optional[Path] = None) -> OnetPool:
    """Load the O*NET parquet and compute the scalar ``delta_j``.

    Parameters
    ----------
    parquet_path
        Path to the parquet built by ``build_onet_pool.py``. When
        ``None``, defaults to ``<repo>/rl/artifacts/onet_v1.parquet``
        resolved from this file's location.

    Returns
    -------
    OnetPool
        A populated pool object whose ``delta_j`` is the z-scored
        ``work_zone`` column.

    Raises
    ------
    FileNotFoundError
        If the parquet does not exist at the resolved path.
    ValueError
        If a required column is missing.
    """
    if parquet_path is None:
        repo_root = Path(__file__).resolve().parents[4]
        parquet_path = repo_root / "rl" / "artifacts" / "onet_v1.parquet"
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"missing O*NET parquet: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    required = {
        "occupation_code",
        "title",
        "description",
        "tasks_concat",
        "work_activities_summary",
        "riasec_code",
        "work_zone",
        "education_zscore",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"parquet missing columns: {sorted(missing)}")

    # Preserve parquet row order. Job IDs are 0-based positions.
    df = df.reset_index(drop=True)
    work_zone = df["work_zone"].astype(np.int64).to_numpy()
    delta_j = _zscore(work_zone.astype(np.float64))

    edu_raw = df["education_zscore"].to_numpy(dtype=np.float64)

    return OnetPool(
        job_ids=np.arange(len(df), dtype=np.int64),
        occupation_codes=df["occupation_code"].astype(str).tolist(),
        titles=df["title"].astype(str).tolist(),
        descriptions=df["description"].astype(str).tolist(),
        tasks_concat=df["tasks_concat"].astype(str).tolist(),
        work_activities_summary=df["work_activities_summary"].astype(str).tolist(),
        riasec_code=df["riasec_code"].astype(str).tolist(),
        work_zone=work_zone,
        education_zscore=edu_raw,
        delta_j=delta_j.astype(np.float64),
    )
