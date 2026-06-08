"""``fit_placeholder_2pl``, an SGD-fit 2PL wrapper used by the Eedi adapter.

The Eedi distractor-ordering algorithm needs a per-student ability
estimate on the binary correctness signal so the three distractors can
be ranked by the mean ability of students who picked them. We do not
need a state-of-the-art psychometric fit, only a stable monotone
ordering of theta_hat. ``StaticGPCM(n_categories=2)`` (the existing
ma-irt baseline) is the cheapest path that integrates with our own
loss and optimiser stack.

Roughly 5 epochs at lr=1e-2 is the budget proposed in the impl guide.
The fit is intentionally minimal, no logging, no plots, no checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class Placeholder2PLFit:
    """Output of ``fit_placeholder_2pl``.

    Attributes:
        theta_hat: ``np.ndarray (n_students,)`` ability estimate per
            student. Students not in the train fold get ``np.nan`` so
            callers can detect cold rows.
        alpha: ``np.ndarray (n_questions + 1,)`` discrimination per item.
            Index 0 is reserved for padding, identical to MA-GPCM's
            1-based item-id convention.
        b: ``np.ndarray (n_questions + 1,)`` difficulty per item.
        train_indices: Indices of students used to fit the 2PL.
        n_iters: Number of SGD epochs actually run.
    """

    theta_hat: np.ndarray
    alpha: np.ndarray
    b: np.ndarray
    train_indices: np.ndarray
    n_iters: int


def fit_placeholder_2pl(
    questions: Sequence[Sequence[int]],
    binary_responses: Sequence[Sequence[int]],
    train_indices: Sequence[int],
    n_questions: int,
    max_iters: int = 5,
    lr: float = 1e-2,
    batch_size: int = 64,
    device: str = "cpu",
    seed: int = 0,
) -> Placeholder2PLFit:
    """Fit a 2PL on the train fold and return per-student theta_hat.

    The model is a tiny re-implementation of StaticGPCM(K=2) so we can
    keep the dependency surface narrow. Mathematically,

        logit P(y_uq = 1 | theta_u, alpha_q, b_q)
            = alpha_q * (theta_u - b_q)
        alpha_q = exp(0.3 * alpha_raw_q)

    matches ``ma-irt/models/static_gpcm.py`` for D = 1, K = 2 with the
    fixed positive-mapping convention.

    Args:
        questions: Per-student lists of 1-based item ids.
        binary_responses: Per-student lists of {0, 1} responses; for the
            Eedi adapter this is ``int(chosen == correct)``.
        train_indices: Indices into ``questions``/``binary_responses``
            used to fit theta_hat and item parameters. Students outside
            this list get ``np.nan`` in the returned ``theta_hat``.
        n_questions: Total item bank size Q (1-based; 0 reserved for padding).
        max_iters: Number of full-data SGD epochs.
        lr: Adam learning rate.
        batch_size: Mini-batch size, applied across (student, item) pairs.
        device: ``"cpu"`` or ``"cuda"``.
        seed: Manual seed for reproducibility.

    Returns:
        ``Placeholder2PLFit``.
    """
    if max_iters <= 0:
        raise ValueError(f"max_iters must be positive, got {max_iters}")
    if lr <= 0:
        raise ValueError(f"lr must be positive, got {lr}")

    torch.manual_seed(int(seed))
    dev = torch.device(device)

    train_indices_arr = np.asarray(train_indices, dtype=np.int64)
    train_set = set(int(i) for i in train_indices_arr)
    n_students_total = len(questions)

    # Flatten train-fold (student_idx, item_idx, response) triples for
    # mini-batch SGD. ``student_idx`` here is the position in
    # ``train_indices_arr`` (compact), not the original index.
    flat_student: list[int] = []
    flat_item: list[int] = []
    flat_resp: list[int] = []
    compact_id_for_orig: Dict[int, int] = {}
    for k, orig in enumerate(train_indices_arr.tolist()):
        compact_id_for_orig[int(orig)] = k
        q_list = questions[int(orig)]
        r_list = binary_responses[int(orig)]
        if len(q_list) != len(r_list):
            raise ValueError(
                f"student {orig}: questions/responses length mismatch "
                f"({len(q_list)} vs {len(r_list)})"
            )
        for q, r in zip(q_list, r_list):
            qi = int(q)
            ri = int(r)
            if not (1 <= qi <= n_questions):
                raise ValueError(
                    f"student {orig}: question id {qi} out of range "
                    f"[1, {n_questions}]"
                )
            if ri not in (0, 1):
                raise ValueError(
                    f"student {orig}: binary response {ri} not in {{0, 1}}"
                )
            flat_student.append(k)
            flat_item.append(qi)
            flat_resp.append(ri)

    if not flat_student:
        raise ValueError("No training pairs to fit placeholder 2PL on.")

    n_train_students = len(train_indices_arr)
    student_idx_t = torch.tensor(flat_student, dtype=torch.long, device=dev)
    item_idx_t = torch.tensor(flat_item, dtype=torch.long, device=dev)
    resp_t = torch.tensor(flat_resp, dtype=torch.float32, device=dev)

    # ---- Parameters ------------------------------------------------------
    theta = torch.nn.Parameter(
        torch.randn(n_train_students, device=dev) * 0.1
    )
    alpha_raw = torch.nn.Parameter(torch.zeros(n_questions + 1, device=dev))
    b = torch.nn.Parameter(torch.zeros(n_questions + 1, device=dev))

    opt = torch.optim.Adam([theta, alpha_raw, b], lr=lr)

    n_pairs = student_idx_t.shape[0]
    perm_seed = int(seed)

    for epoch in range(max_iters):
        # Re-shuffle each epoch with a deterministic seed derived from
        # the base ``seed`` so re-runs are reproducible.
        g = torch.Generator(device="cpu").manual_seed(perm_seed + epoch)
        perm = torch.randperm(n_pairs, generator=g)
        for start in range(0, n_pairs, batch_size):
            sl = perm[start:start + batch_size]
            s = student_idx_t[sl]
            i = item_idx_t[sl]
            y = resp_t[sl]
            alpha_i = torch.exp(0.3 * alpha_raw[i])
            logit = alpha_i * (theta[s] - b[i])
            loss = F.binary_cross_entropy_with_logits(logit, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        alpha_final = torch.exp(0.3 * alpha_raw).cpu().numpy()
        b_final = b.cpu().numpy()
        theta_train = theta.cpu().numpy()

    theta_full = np.full(n_students_total, np.nan, dtype=np.float64)
    for orig, k in compact_id_for_orig.items():
        theta_full[orig] = float(theta_train[k])

    return Placeholder2PLFit(
        theta_hat=theta_full,
        alpha=alpha_final.astype(np.float64),
        b=b_final.astype(np.float64),
        train_indices=train_indices_arr,
        n_iters=int(max_iters),
    )
