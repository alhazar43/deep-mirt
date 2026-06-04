"""RetrievalIndex.

Per the v1 plan, Section 6.5, the retrieval index turns a query vector
into the top-k jobs by inner product. The v1 backend is a plain numpy
matmul because the O*NET pool is 923 occupations. Faiss is wired in at
v2 when the pool grows.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


__all__ = ["RetrievalIndex"]


class RetrievalIndex:
    """Numpy-backed inner-product top-k index.

    The index assumes ``v_j`` is L2-normalised. When the query vector is
    also unit-norm, the inner product equals the cosine similarity.
    """

    def __init__(self) -> None:
        self._v_j: np.ndarray | None = None
        self._job_ids: list[str] | None = None
        self._job_id_to_idx: dict[str, int] | None = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def fit(self, v_j: np.ndarray, job_ids: Sequence[str]) -> "RetrievalIndex":
        """Install the embedding matrix and the parallel job ID list.

        Parameters
        ----------
        v_j
            Float array of shape ``(n_jobs, d)``.
        job_ids
            Iterable of length ``n_jobs`` whose order matches ``v_j``.
        """

        v = np.ascontiguousarray(np.asarray(v_j, dtype=np.float32))
        if v.ndim != 2:
            raise ValueError(f"v_j must be 2D, got shape {v.shape}")
        ids = list(job_ids)
        if len(ids) != v.shape[0]:
            raise ValueError(
                f"job_ids length {len(ids)} does not match v_j rows {v.shape[0]}"
            )
        if len(set(ids)) != len(ids):
            raise ValueError("job_ids must be unique")
        self._v_j = v
        self._job_ids = ids
        self._job_id_to_idx = {job_id: idx for idx, job_id in enumerate(ids)}
        return self

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def topk(
        self,
        q_t: np.ndarray,
        k: int,
        mask: np.ndarray | None = None,
    ) -> tuple[list[str], np.ndarray]:
        """Return the top-k job ids ranked by inner product with ``q_t``.

        Parameters
        ----------
        q_t
            Query vector of shape ``(d,)``.
        k
            Number of jobs to return. Must satisfy ``1 <= k <= n_jobs``
            (or ``k <= n_remaining`` when a mask is supplied).
        mask
            Optional boolean array of shape ``(n_jobs,)``. ``True``
            means the job is eligible. When ``None`` every job is
            eligible.

        Returns
        -------
        (ids, scores)
            ``ids`` is a length-``k`` list of job IDs sorted by score
            descending. ``scores`` is the matching numpy array of
            shape ``(k,)`` with float32 dtype.
        """

        if self._v_j is None or self._job_ids is None:
            raise RuntimeError("RetrievalIndex.fit must be called before topk")
        q = np.asarray(q_t, dtype=np.float32).reshape(-1)
        if q.shape[0] != self._v_j.shape[1]:
            raise ValueError(
                f"query dim {q.shape[0]} does not match index dim {self._v_j.shape[1]}"
            )
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        scores = self._v_j @ q  # (n_jobs,)
        n_jobs = scores.shape[0]

        if mask is None:
            effective_scores = scores
            eligible_idx = np.arange(n_jobs)
        else:
            mask_arr = np.asarray(mask, dtype=bool).reshape(-1)
            if mask_arr.shape[0] != n_jobs:
                raise ValueError(
                    f"mask length {mask_arr.shape[0]} != n_jobs {n_jobs}"
                )
            eligible_idx = np.flatnonzero(mask_arr)
            effective_scores = scores[eligible_idx]
        if k > eligible_idx.shape[0]:
            raise ValueError(
                f"k={k} exceeds eligible jobs ({eligible_idx.shape[0]})"
            )

        # Deterministic argsort: stable on equal scores, descending.
        order = np.argsort(-effective_scores, kind="stable")[:k]
        top_idx = eligible_idx[order]
        top_scores = scores[top_idx].astype(np.float32, copy=False)
        top_ids = [self._job_ids[int(i)] for i in top_idx]
        return top_ids, top_scores

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def n_jobs(self) -> int:
        if self._v_j is None:
            return 0
        return self._v_j.shape[0]

    @property
    def dim(self) -> int:
        if self._v_j is None:
            return 0
        return self._v_j.shape[1]

    def index_of(self, job_id: str) -> int:
        if self._job_id_to_idx is None:
            raise RuntimeError("RetrievalIndex.fit must be called before index_of")
        return self._job_id_to_idx[job_id]
