"""Job pool loader and JobPoolSpec dataclass.

Per the v1 plan, Section 6.4 and 6.5, a JobPoolSpec is the in-memory
representation of an entire pool of jobs. The JobTower consumes it and
emits one vector per job. The RetrievalIndex consumes those vectors. The
spec also exposes the GPCM-style difficulty parameter ``delta_j`` so
downstream belief and policy code can read it without knowing about
parquet schemas.

The v1 source is the O*NET 2024 release built by
``rl/scripts/build_onet_pool.py``. ``delta_j`` is the z-scored work zone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


__all__ = ["JobPoolSpec", "load_onet_pool"]


@dataclass
class JobPoolSpec:
    """Frozen view over a pool of jobs.

    Attributes
    ----------
    occupation_codes
        Length ``n_jobs`` list of occupation IDs (e.g. O*NET-SOC codes).
        This is the canonical join key. The position of each code in
        this list is the integer job index used by every downstream
        component.
    titles
        Length ``n_jobs`` list of occupation titles.
    descriptions
        Length ``n_jobs`` list of long-form descriptions.
    tasks_concat
        Length ``n_jobs`` list of task statements joined by " | ".
    structured_features
        Float array of shape ``(n_jobs, 8)`` with the structured input
        for the JobTower. The columns are
        ``[work_zone_scaled, education_zscore, education_mask,
        riasec_R, riasec_I, riasec_A, riasec_S, riasec_E]``.
        We carry only 5 of the 6 RIASEC dimensions in this slot because
        the eighth column is reserved for ``education_mask``. The
        JobTower expands the riasec_code to a 6-dim weighted vector
        internally from ``riasec_codes`` below.
    riasec_codes
        Length ``n_jobs`` list of 3-letter RIASEC strings (e.g. "RIA")
        or empty strings when no interest data is present.
    delta_j
        Float array of shape ``(n_jobs,)``. The v1 plan Section 6.3 and
        Decision D8 fix delta_j to the z-scored work zone.
    work_zones_raw
        Integer array of shape ``(n_jobs,)`` with the raw 1..5 work
        zone, useful for diagnostics.
    education_zscores_raw
        Float array of shape ``(n_jobs,)`` with raw z-scored education
        (NaN preserved). Useful for diagnostics and downstream analysis.
    source
        Free-form string tag describing where the pool came from.
    """

    occupation_codes: list[str]
    titles: list[str]
    descriptions: list[str]
    tasks_concat: list[str]
    structured_features: np.ndarray  # shape (n_jobs, 8)
    riasec_codes: list[str]
    delta_j: np.ndarray  # shape (n_jobs,)
    work_zones_raw: np.ndarray  # shape (n_jobs,)
    education_zscores_raw: np.ndarray  # shape (n_jobs,)
    source: str = "unknown"

    @property
    def n_jobs(self) -> int:
        return len(self.occupation_codes)

    def attach_text(self, indices: Sequence[int] | None = None) -> list[str]:
        """Build the text input string for each occupation.

        The format is ``"<title>. <description>. Tasks: <tasks_concat>"``.

        Parameters
        ----------
        indices
            Optional subset of job indices. When ``None`` all jobs are
            returned in pool order.
        """

        if indices is None:
            iterator = range(self.n_jobs)
        else:
            iterator = list(indices)
        out: list[str] = []
        for i in iterator:
            title = self.titles[i].strip()
            desc = self.descriptions[i].strip()
            tasks = self.tasks_concat[i].strip()
            out.append(f"{title}. {desc}. Tasks: {tasks}")
        return out

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        source: str = "dataframe",
    ) -> "JobPoolSpec":
        """Build a JobPoolSpec from a dataframe with the O*NET schema.

        The dataframe must carry the columns produced by
        ``rl/scripts/build_onet_pool.py``. NaN education_zscore is
        replaced by zero in the structured features and flagged via
        the dedicated mask column.
        """

        required = {
            "occupation_code",
            "title",
            "description",
            "tasks_concat",
            "riasec_code",
            "work_zone",
            "education_zscore",
        }
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"pool dataframe missing columns {sorted(missing)}")

        df = df.reset_index(drop=True)
        n = len(df)

        work_zone_raw = df["work_zone"].to_numpy(dtype=np.int64)
        # Map 1..5 to [-1, 1] uniformly. The plan calls this work_zone
        # scaled to [-1, 1].
        work_zone_scaled = (work_zone_raw.astype(np.float32) - 3.0) / 2.0

        edu_raw = df["education_zscore"].to_numpy(dtype=np.float64)
        edu_mask = (~np.isnan(edu_raw)).astype(np.float32)
        edu_filled = np.where(np.isnan(edu_raw), 0.0, edu_raw).astype(np.float32)

        riasec_codes = df["riasec_code"].fillna("").astype(str).tolist()

        # Build a 5-dim slot from the first five riasec letters. The
        # sixth dimension is computed by the JobTower because the full
        # vector lives there. We still need a structured_features matrix
        # that downstream code can introspect, so we pack the first five
        # weighted riasec slots here and leave the C dimension to the
        # JobTower.
        riasec_5 = np.zeros((n, 5), dtype=np.float32)
        # R, I, A, S, E mapping (Conventional is the 6th and is appended
        # by JobTower at forward time).
        letter_to_col = {"R": 0, "I": 1, "A": 2, "S": 3, "E": 4}
        weights = (0.6, 0.3, 0.1)
        for row_idx, code in enumerate(riasec_codes):
            for slot, letter in enumerate(code[:3]):
                col = letter_to_col.get(letter)
                if col is None:
                    continue
                riasec_5[row_idx, col] += weights[slot]

        structured = np.concatenate(
            [
                work_zone_scaled.reshape(-1, 1),
                edu_filled.reshape(-1, 1),
                edu_mask.reshape(-1, 1),
                riasec_5,
            ],
            axis=1,
        ).astype(np.float32)
        assert structured.shape == (n, 8)

        # delta_j = z-scored work zone. The pool may not span the full
        # 1..5 range so we standardise inside the spec.
        wz = work_zone_raw.astype(np.float64)
        mu = wz.mean()
        sigma = wz.std(ddof=0)
        if sigma <= 0:
            delta_j = np.zeros_like(wz, dtype=np.float32)
        else:
            delta_j = ((wz - mu) / sigma).astype(np.float32)

        return cls(
            occupation_codes=df["occupation_code"].astype(str).tolist(),
            titles=df["title"].astype(str).tolist(),
            descriptions=df["description"].astype(str).tolist(),
            tasks_concat=df["tasks_concat"].fillna("").astype(str).tolist(),
            structured_features=structured,
            riasec_codes=riasec_codes,
            delta_j=delta_j,
            work_zones_raw=work_zone_raw,
            education_zscores_raw=edu_raw.astype(np.float64),
            source=source,
        )


def load_onet_pool(parquet_path: str | Path) -> JobPoolSpec:
    """Load the O*NET pool parquet into a JobPoolSpec.

    Parameters
    ----------
    parquet_path
        Path to the parquet emitted by ``rl/scripts/build_onet_pool.py``.

    Returns
    -------
    JobPoolSpec
    """

    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)
    df = pd.read_parquet(parquet_path)
    return JobPoolSpec.from_dataframe(df, source=f"onet:{parquet_path.name}")
