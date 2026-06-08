"""Deterministic user-level splits and sequence chunking.

The adapter framework owns the split policy. Concrete adapters call
``make_split`` to assign each student to train/valid/test, optionally
``stratified_split`` when a stratification key is available, and
``_chunk_sequences`` to enforce ``max_seq_len`` via non-overlapping
chunks.

The split is seeded purely by ``split_seed`` and the number of students,
so re-running the same adapter on the same raw input yields a
byte-identical artefact. This is what test_split_determinism asserts.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


# Numeric codes match ``base._SPLIT_CODES``: train=0, valid=1, test=2.
SPLIT_TRAIN = 0
SPLIT_VALID = 1
SPLIT_TEST = 2


def make_split(
    n_students: int,
    test_frac: float,
    valid_frac: float,
    split_seed: int,
) -> np.ndarray:
    """Assign each of ``n_students`` an integer split code.

    Args:
        n_students: Number of students to assign.
        test_frac: Fraction routed to test. Must be in ``[0, 1]``.
        valid_frac: Fraction routed to valid. Must be in ``[0, 1]``.
        split_seed: Seed for the permutation.

    Returns:
        ``np.ndarray[int8]`` of shape ``(n_students,)`` with values in
        ``{SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST}``.

    Raises:
        ValueError: When fractions are out of range.
    """
    if n_students <= 0:
        raise ValueError(f"n_students must be positive, got {n_students}")
    if not (0.0 <= test_frac <= 1.0):
        raise ValueError(f"test_frac must be in [0, 1], got {test_frac}")
    if not (0.0 <= valid_frac <= 1.0):
        raise ValueError(f"valid_frac must be in [0, 1], got {valid_frac}")
    if test_frac + valid_frac >= 1.0:
        raise ValueError(
            f"test_frac + valid_frac must be < 1, got {test_frac + valid_frac}"
        )

    rng = np.random.default_rng(int(split_seed))
    perm = rng.permutation(n_students)

    n_test = int(round(n_students * test_frac))
    n_valid = int(round(n_students * valid_frac))
    # Guarantee at least one student in train when fractions round up.
    n_train = max(1, n_students - n_test - n_valid)
    # Adjust test downward if rounding produced an overcount.
    n_test = n_students - n_train - n_valid

    codes = np.empty(n_students, dtype=np.int8)
    codes[perm[:n_train]] = SPLIT_TRAIN
    codes[perm[n_train:n_train + n_valid]] = SPLIT_VALID
    codes[perm[n_train + n_valid:]] = SPLIT_TEST
    return codes


def stratified_split(
    strata: Sequence[int],
    test_frac: float,
    valid_frac: float,
    split_seed: int,
) -> np.ndarray:
    """Stratified user-level split.

    Inside each stratum we run the same proportional ``make_split`` so
    the fold composition matches the population. Used when an external
    stratification key (e.g. cohort tag) is available; OrdRec adapters
    fall back to ``make_split`` when none is present.

    Args:
        strata: Per-student integer stratum label.
        test_frac: Fraction routed to test.
        valid_frac: Fraction routed to valid.
        split_seed: Base seed; combined per-stratum so all-train edge cases
            inside a small stratum stay deterministic.

    Returns:
        ``np.ndarray[int8]`` of shape ``(len(strata),)``.
    """
    strata_arr = np.asarray(strata, dtype=np.int64)
    n = strata_arr.shape[0]
    codes = np.zeros(n, dtype=np.int8)
    unique = np.unique(strata_arr)
    for k_idx, k in enumerate(unique):
        mask = strata_arr == k
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        # Use a stratum-specific seed derived from the base.
        sub_seed = (int(split_seed) * 1_000_003 + int(k)) & 0x7FFFFFFF
        sub_codes = make_split(n_k, test_frac, valid_frac, sub_seed)
        codes[mask] = sub_codes
    return codes


def _chunk_sequences(
    questions: List[List[int]],
    responses: List[List[int]],
    max_len: int,
) -> Tuple[List[List[int]], List[List[int]], List[int]]:
    """Split long sequences into non-overlapping chunks.

    Returns:
        ``(chunked_q, chunked_r, parent_idx)`` where ``parent_idx[i]`` is
        the index of the source student for chunk ``i``. This preserves
        the per-student split assignment when callers later expand the
        ``_student_split`` array.
    """
    if max_len <= 0:
        return list(questions), list(responses), list(range(len(questions)))
    if len(questions) != len(responses):
        raise ValueError(
            "questions and responses must align: "
            f"got {len(questions)} vs {len(responses)}"
        )

    out_q: List[List[int]] = []
    out_r: List[List[int]] = []
    parent: List[int] = []
    for u, (q, r) in enumerate(zip(questions, responses)):
        if len(q) != len(r):
            raise ValueError(
                f"student {u}: questions and responses length mismatch "
                f"({len(q)} vs {len(r)})"
            )
        if len(q) <= max_len:
            out_q.append(list(q))
            out_r.append(list(r))
            parent.append(u)
            continue
        for i in range(0, len(q), max_len):
            out_q.append(list(q[i:i + max_len]))
            out_r.append(list(r[i:i + max_len]))
            parent.append(u)
    return out_q, out_r, parent
