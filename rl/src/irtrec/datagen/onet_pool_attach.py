"""Attach the O*NET parquet to the synthetic data generator.

This module loads ``rl/artifacts/onet_v1.parquet`` (built by
``rl/scripts/build_onet_pool.py``), computes the v2 continuous
``delta_j`` composite used by the GPCM preference model, and exposes the
structured features used by downstream code.

v2 delta_j composite
--------------------

``delta_j`` is a continuous job-difficulty signal built as a weighted
sum of three z-scored features plus a fixed-seed Gaussian noise term,
then re-standardised to unit variance,

    delta_j_raw = 0.45 * z(work_zone)
                + 0.35 * z(education_zscore)
                + 0.20 * z(complexity_composite)
                + N(0, 0.30)
    delta_j     = (delta_j_raw - mean) / std

where ``complexity_composite`` is the mean z-score across the four
O*NET work_activities major categories,

    Information Input, Mental Processes, Work Output,
    Interacting With Others.

The locked ``onet_v1.parquet`` does not carry the per-category
importance scalars; it only carries a ``work_activities_summary``
string with the top-5 highest-importance activities per occupation.
We approximate the per-category importance by counting how many of an
occupation's top-5 activities fall in each major category. The four
counts are z-scored independently and averaged to produce
``complexity_composite``. This fall-back is documented here as the v2
data-availability compromise.

The final ``delta_j`` is re-standardised so that ``std(delta_j) = 1``
exactly, which makes the GPCM step thresholds ``beta = (-1.5, -0.5,
0.5, 1.5)`` interpretable on the canonical theta scale. The unweighted
formula yields ``std`` in the 0.85 - 0.88 range under the actual
parquet correlations, so the final z-scoring is a small (~5-15
percent) scale adjustment and does not change the rank order.

The returned :class:`OnetPool` object holds parallel arrays indexed by
position in the pool. The integer ``job_id`` used by ``likes.json`` is
the 0-based position (consistent with how the long-form likes table is
emitted by :mod:`synth_likes`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# v2 default RNG seed for the delta_j noise term. Fixed so the
# parquet -> delta_j mapping is byte-stable across runs.
DEFAULT_DELTA_J_NOISE_SEED: int = 20250604

# v2 composite weights (work_zone, education_zscore, complexity_composite).
DELTA_J_WEIGHTS: Dict[str, float] = {
    "work_zone": 0.45,
    "education_zscore": 0.35,
    "complexity_composite": 0.20,
}
DELTA_J_NOISE_SCALE: float = 0.30


# Mapping of the 39 O*NET work_activities Element Names that appear in
# the parquet's work_activities_summary to the four major work activity
# categories. These map cleanly onto the O*NET work activity ontology
# (4.A.1, 4.A.2, 4.A.3, 4.A.4 in the content model).
_INFORMATION_INPUT: set[str] = {
    "Getting Information",
    "Monitoring Processes, Materials, or Surroundings",
    "Identifying Objects, Actions, and Events",
    "Inspecting Equipment, Structures, or Materials",
    "Estimating the Quantifiable Characteristics of Products, Events, or Information",
}
_MENTAL_PROCESSES: set[str] = {
    "Judging the Qualities of Objects, Services, or People",
    "Processing Information",
    "Evaluating Information to Determine Compliance with Standards",
    "Analyzing Data or Information",
    "Making Decisions and Solving Problems",
    "Thinking Creatively",
    "Updating and Using Relevant Knowledge",
    "Developing Objectives and Strategies",
    "Scheduling Work and Activities",
    "Organizing, Planning, and Prioritizing Work",
}
_WORK_OUTPUT: set[str] = {
    "Performing General Physical Activities",
    "Handling and Moving Objects",
    "Controlling Machines and Processes",
    "Operating Vehicles, Mechanized Devices, or Equipment",
    "Working with Computers",
    "Drafting, Laying Out, and Specifying Technical Devices, Parts, and Equipment",
    "Repairing and Maintaining Mechanical Equipment",
    "Repairing and Maintaining Electronic Equipment",
    "Documenting/Recording Information",
}
_INTERACTING_WITH_OTHERS: set[str] = {
    "Interpreting the Meaning of Information for Others",
    "Communicating with Supervisors, Peers, or Subordinates",
    "Communicating with People Outside the Organization",
    "Establishing and Maintaining Interpersonal Relationships",
    "Assisting and Caring for Others",
    "Selling or Influencing Others",
    "Resolving Conflicts and Negotiating with Others",
    "Performing for or Working Directly with the Public",
    "Coordinating the Work and Activities of Others",
    "Developing and Building Teams",
    "Training and Teaching Others",
    "Guiding, Directing, and Motivating Subordinates",
    "Coaching and Developing Others",
    "Providing Consultation and Advice to Others",
    "Performing Administrative Activities",
}

_CATEGORY_TO_SET: Dict[str, set[str]] = {
    "information_input": _INFORMATION_INPUT,
    "mental_processes": _MENTAL_PROCESSES,
    "work_output": _WORK_OUTPUT,
    "interacting_with_others": _INTERACTING_WITH_OTHERS,
}


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
        Continuous composite difficulty per job, z-scored to unit
        variance. The v2 spec uses
        ``0.45 z(work_zone) + 0.35 z(education_zscore) + 0.20 z(complexity_composite) + N(0, 0.30)``
        followed by a final z-score.
    delta_j_components
        Per-component arrays used to construct ``delta_j``. Keys
        ``work_zone_z``, ``education_zscore_z``, ``complexity_composite``,
        ``noise``, ``delta_j_raw`` (pre-final-zscore). Useful for
        diagnostics and unit tests.
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
    delta_j_components: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_jobs(self) -> int:
        return int(self.job_ids.shape[0])

    def to_jobs_records(self) -> List[dict]:
        """Long-form ``jobs.json`` records, one per occupation.

        Each record carries the structured features needed by Section 5
        downstream consumers (ItemTower / JobTower in M2 reads the same
        parquet directly, but the generated dataset duplicates them so
        a fitted artifact is self-contained).
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
    NaN values are treated as zero after standardisation, but the
    population statistics are computed over finite values only.
    """
    arr = np.asarray(values, dtype=np.float64)
    finite_mask = np.isfinite(arr)
    if finite_mask.sum() <= 1:
        return np.zeros_like(arr, dtype=np.float64)
    finite = arr[finite_mask]
    mean = float(finite.mean())
    std = float(finite.std(ddof=0))
    if std <= 0.0:
        return np.zeros_like(arr, dtype=np.float64)
    out = (arr - mean) / std
    # NaNs become zero (the population mean on the standardised scale).
    out = np.where(finite_mask, out, 0.0)
    return out


def _count_category(summary: str, cat_set: set[str]) -> int:
    """Count how many of a row's top-5 work activities fall in ``cat_set``."""
    if not isinstance(summary, str) or not summary:
        return 0
    parts = [p.strip() for p in summary.split(";")]
    return sum(1 for p in parts if p in cat_set)


def _compute_complexity_composite(
    work_activities_summary: List[str],
) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Mean z-score across the four major work-activity categories.

    Each of the four category importances is approximated by the count
    of top-5 activities in that category (0..5 per occupation). Each
    count is independently z-scored, then the four are averaged. The
    averaged signal is returned; downstream callers re-z-score it before
    weighting in :func:`_combine_delta_j`.

    Returns
    -------
    complexity
        Length n_jobs array. The mean of the four z-scored category
        counts. Not unit variance in general.
    per_category
        Dict mapping category name -> raw count array (length n_jobs).
        Useful for diagnostics.
    """
    n = len(work_activities_summary)
    per_category: Dict[str, np.ndarray] = {}
    z_stack = np.zeros((n, len(_CATEGORY_TO_SET)), dtype=np.float64)
    for col, (name, cat_set) in enumerate(_CATEGORY_TO_SET.items()):
        counts = np.array(
            [_count_category(s, cat_set) for s in work_activities_summary],
            dtype=np.float64,
        )
        per_category[name] = counts.copy()
        z_stack[:, col] = _zscore(counts)
    complexity = z_stack.mean(axis=1)
    return complexity, per_category


def _combine_delta_j(
    work_zone: np.ndarray,
    education_zscore: np.ndarray,
    complexity_raw: np.ndarray,
    noise_seed: int,
) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Combine the three components plus noise into the v2 delta_j.

    The formula

        delta_j_raw = 0.45 z(wz) + 0.35 z(edu) + 0.20 z(complexity) + N(0, 0.30)

    is applied component-wise. ``complexity_raw`` is re-z-scored here so
    its weight applies to a unit-variance signal. ``delta_j_raw`` is
    then re-standardised so the final std equals 1 exactly. This
    final-zscore step is a small uniform rescale (~5-15 percent at the
    locked parquet) that does not change the rank order.
    """
    n = work_zone.shape[0]
    rng = np.random.default_rng(noise_seed)
    noise = rng.normal(loc=0.0, scale=DELTA_J_NOISE_SCALE, size=n)

    wz_z = _zscore(work_zone)
    ed_z = _zscore(education_zscore)
    cx_z = _zscore(complexity_raw)

    delta_j_raw = (
        DELTA_J_WEIGHTS["work_zone"] * wz_z
        + DELTA_J_WEIGHTS["education_zscore"] * ed_z
        + DELTA_J_WEIGHTS["complexity_composite"] * cx_z
        + noise
    )
    delta_j = _zscore(delta_j_raw)
    components = {
        "work_zone_z": wz_z,
        "education_zscore_z": ed_z,
        "complexity_composite": cx_z,
        "noise": noise,
        "delta_j_raw": delta_j_raw,
    }
    return delta_j, components


def load_onet_pool(
    parquet_path: Optional[Path] = None,
    noise_seed: int = DEFAULT_DELTA_J_NOISE_SEED,
) -> OnetPool:
    """Load the O*NET parquet and compute the v2 continuous ``delta_j``.

    Parameters
    ----------
    parquet_path
        Path to the parquet built by ``build_onet_pool.py``. When
        ``None``, defaults to ``<repo>/rl/artifacts/onet_v1.parquet``
        resolved from this file's location.
    noise_seed
        Seed for the Gaussian noise term inside the delta_j composite.
        Defaults to :data:`DEFAULT_DELTA_J_NOISE_SEED` so the mapping is
        byte-stable across runs.

    Returns
    -------
    OnetPool
        Pool object populated with continuous v2 ``delta_j`` and
        per-component arrays in ``delta_j_components``.

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
    edu_raw = df["education_zscore"].to_numpy(dtype=np.float64)
    work_activities = df["work_activities_summary"].astype(str).tolist()

    complexity_raw, per_category = _compute_complexity_composite(work_activities)
    delta_j, components = _combine_delta_j(
        work_zone=work_zone.astype(np.float64),
        education_zscore=edu_raw,
        complexity_raw=complexity_raw,
        noise_seed=noise_seed,
    )
    # Expose per-category counts alongside the standardised components.
    for name, arr in per_category.items():
        components[f"category_count__{name}"] = arr

    return OnetPool(
        job_ids=np.arange(len(df), dtype=np.int64),
        occupation_codes=df["occupation_code"].astype(str).tolist(),
        titles=df["title"].astype(str).tolist(),
        descriptions=df["description"].astype(str).tolist(),
        tasks_concat=df["tasks_concat"].astype(str).tolist(),
        work_activities_summary=work_activities,
        riasec_code=df["riasec_code"].astype(str).tolist(),
        work_zone=work_zone,
        education_zscore=edu_raw,
        delta_j=delta_j.astype(np.float64),
        delta_j_components=components,
    )
