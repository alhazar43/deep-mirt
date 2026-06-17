"""nrm_datagen.py -- synthetic SEQUENCE data from a TRUE Bock (1972) NRM.

The engine head-to-head NRM bench needs ground truth the GPCM bench does not
carry: per-option slopes a_k, per-option intercepts c_k, and a true (static or
dynamic) ability theta.  This generator samples those item parameters and emits
per-learner option-choice sequences, matched in shape (N learners x T steps, Q
items, K options) to the GPCM bench (``deep_irt/bench/datagen.py``) so the two
are directly comparable.

It deliberately mirrors ``deep_irt/nrmfmt/data_gen.py`` for the item-parameter
sampling (correct-option location planted as s_i, Bock centering applied so the
planted location survives), then wraps it into the sequence layout the engines
consume.

Datasets
--------
- ``static``  : NRM, ability fixed per learner.
- ``dynamic`` : NRM, ability a per-step random walk (the regime a richer KT
                encoder should most help).

The SAME ground truth + the SAME train/val learner split feed both engines, so
any difference is the model, never the data.  Both engines predict response[t]
from history 0..t-1 plus the known item id at t; the last ``n_holdout``
positions are scored for prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Config + containers
# ---------------------------------------------------------------------------

@dataclass
class NRMDataConfig:
    name: str = "nrm_static"
    kind: str = "static"          # "static" | "dynamic"
    n_learners: int = 800         # N
    n_items: int = 60             # Q
    seq_len: int = 60             # T (items per learner, sampled with replacement)
    n_options: int = 4            # K
    correct_option: int = 0       # designated-correct option index
    drift_sigma: float = 0.15     # per-step std for the dynamic random walk
    train_frac: float = 0.8
    n_holdout: int = 10           # last n_holdout positions scored for prediction
    seed: int = 0


@dataclass
class NRMGroundTruth:
    theta0: np.ndarray            # (N,) initial / mean ability
    a: np.ndarray                 # (Q, K) per-option slopes (Bock-centered)
    c: np.ndarray                 # (Q, K) per-option intercepts (Bock-centered)
    correct_option: int
    theta_traj: Optional[np.ndarray] = None   # (N, T) per-step true theta or None


@dataclass
class NRMDataset:
    cfg: NRMDataConfig
    gt: NRMGroundTruth
    items0: np.ndarray            # (N, T) int64 in [0, Q-1]
    responses: np.ndarray         # (N, T) int64 option indices in [0, K-1]
    train_idx: np.ndarray
    val_idx: np.ndarray
    theta_at_step: np.ndarray = field(default=None)   # (N, T) true theta used


# ---------------------------------------------------------------------------
# Ground-truth item parameters (mirrors deep_irt/nrmfmt/data_gen.py)
# ---------------------------------------------------------------------------

def _sample_item_params(Q: int, K: int, correct: int, rng) -> tuple:
    """Per-option slopes / intercepts with a planted correct-option location.

    Correct option: slope alpha_corr > 0, intercept gamma_corr = -alpha_corr*s,
    so its raw location -gamma/alpha == s.  Distractors get free N(0, 0.6) slope
    and intercept.  Then Bock-center across options (sum_k a_k = sum_k c_k = 0);
    recompute the location from the centered correct params so the planted scale
    survives centering exactly.
    """
    s = rng.standard_normal(Q)                                   # latent difficulty
    alpha_corr = np.exp(0.3 * rng.standard_normal(Q))            # > 0
    gamma_corr = -alpha_corr * s

    a = 0.6 * rng.standard_normal((Q, K))
    c = 0.6 * rng.standard_normal((Q, K))
    a[:, correct] = alpha_corr
    c[:, correct] = gamma_corr

    a = a - a.mean(axis=1, keepdims=True)                        # sum_k a_k = 0
    c = c - c.mean(axis=1, keepdims=True)                        # sum_k c_k = 0
    return a, c


def _nrm_probs(theta: float, a_row: np.ndarray, c_row: np.ndarray) -> np.ndarray:
    """softmax_k(a_k * theta + c_k)."""
    logits = a_row * theta + c_row
    logits -= logits.max()
    e = np.exp(logits)
    return e / e.sum()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(cfg: NRMDataConfig) -> NRMDataset:
    rng = np.random.default_rng(cfg.seed)
    N, Q, T, K = cfg.n_learners, cfg.n_items, cfg.seq_len, cfg.n_options

    a, c = _sample_item_params(Q, K, cfg.correct_option, rng)
    theta0 = rng.standard_normal(N)

    theta_traj = None
    if cfg.kind == "dynamic":
        steps = rng.normal(0.0, cfg.drift_sigma, size=(N, T))
        cum = np.cumsum(steps, axis=1)
        theta_traj = np.zeros((N, T))
        theta_traj[:, 0] = theta0
        theta_traj[:, 1:] = theta0[:, None] + cum[:, :-1]

    items0 = np.zeros((N, T), dtype=np.int64)
    responses = np.zeros((N, T), dtype=np.int64)
    theta_at_step = np.zeros((N, T), dtype=np.float64)

    for i in range(N):
        item_seq = rng.integers(0, Q, size=T)            # sample items w/ replacement
        items0[i] = item_seq
        for t, j in enumerate(item_seq):
            th = theta_traj[i, t] if theta_traj is not None else theta0[i]
            p = _nrm_probs(th, a[j], c[j])
            responses[i, t] = int(rng.choice(K, p=p))
            theta_at_step[i, t] = th

    perm = rng.permutation(N)
    n_train = int(round(N * cfg.train_frac))
    train_idx = np.sort(perm[:n_train])
    val_idx = np.sort(perm[n_train:])

    gt = NRMGroundTruth(
        theta0=theta0, a=a, c=c,
        correct_option=cfg.correct_option, theta_traj=theta_traj,
    )
    return NRMDataset(
        cfg=cfg, gt=gt, items0=items0, responses=responses,
        train_idx=train_idx, val_idx=val_idx, theta_at_step=theta_at_step,
    )
