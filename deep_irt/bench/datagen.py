"""
datagen.py -- one shared synthetic generator for the engine head-to-head bench.

Generates response sequences from a KNOWN ground-truth GPCM (or 2PL) so that
recovery of theta, discrimination (alpha/a) and step-thresholds (beta/b) is
measurable.  The SAME ground truth and the SAME train/val learner split feed
both engines, so any difference is the model, never the data.

Three datasets
--------------
- ``static``   : K-category GPCM, ability fixed per learner.
- ``dynamic``  : K-category GPCM, ability is a per-step random walk
                 (the regime where a richer KT encoder should most help).
- ``binary``   : 2PL (GPCM at K=2), ability fixed per learner (for AUC).

Engine data paths
-----------------
The bench feeds both engines the SAME in-memory tensors (the fairest path: no
disk round-trip, identical objects).  deep_irt reads items 0-indexed; the
ma-irt engine adds +1 at the tensor boundary so items are 1-indexed (0 =
padding) as ma-irt expects.  ``export_ma_irt`` is also provided to write a
ma-irt-loader dataset folder (sequences/metadata/true_irt_parameters JSON) when
the JSON-pipeline path is wanted for cross-checking.

Prediction alignment
--------------------
Both engines predict response[t] from the response history 0..t-1 plus the
known item id at position t.  We hold out the last ``n_holdout`` positions of
every learner's sequence for the prediction metrics (the model still consumes
the full sequence as history; the metric only scores the held-out tail).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Config + ground truth containers
# ---------------------------------------------------------------------------

@dataclass
class BenchDataConfig:
    name: str = "static"
    kind: str = "static"          # "static" | "dynamic" | "binary"
    n_learners: int = 800
    n_items: int = 60             # Q
    seq_len: int = 60             # T (each learner answers T items, sampled w/ replacement)
    n_cats: int = 4               # K
    drift_sigma: float = 0.15     # per-step std for the dynamic random walk
    train_frac: float = 0.8
    n_holdout: int = 10           # last n_holdout positions scored for prediction
    seed: int = 0
    alpha_theta_slope: float = 0.0  # >0 plants theta-DEPENDENT discrimination:
                                    # per-item gamma_j ~ N(0, slope),
                                    # alpha_eff(i,t) = a_j * exp(gamma_j * theta_it).
                                    # 0.0 (default) = static alpha, bit-identical.


@dataclass
class BenchGroundTruth:
    theta0: np.ndarray            # (N,) initial / mean ability
    a: np.ndarray                 # (Q,) discrimination
    b: np.ndarray                 # (Q, K-1) step thresholds (ascending)
    theta_traj: Optional[np.ndarray] = None   # (N, T) per-step true theta or None
    gamma: Optional[np.ndarray] = None        # (Q,) per-item log-alpha slope in theta
                                              # (planted theta-dependence; None = static)


@dataclass
class BenchDataset:
    cfg: BenchDataConfig
    gt: BenchGroundTruth
    # deep_irt tensors (0-indexed items)
    items0: np.ndarray            # (N, T) int64 in [0, Q-1]
    responses: np.ndarray         # (N, T) int64 in [0, K-1]
    train_idx: np.ndarray         # learner indices (into rows)
    val_idx: np.ndarray
    # per-(learner,step) true theta used at each answered position (N, T)
    theta_at_step: np.ndarray = field(default=None)


# ---------------------------------------------------------------------------
# GPCM sampling
# ---------------------------------------------------------------------------

def _gpcm_probs(theta: float, a: float, b: np.ndarray) -> np.ndarray:
    K = len(b) + 1
    psi = np.zeros(K)
    for k in range(1, K):
        psi[k] = np.sum(a * (theta - b[:k]))
    psi -= psi.max()
    e = np.exp(psi)
    return e / e.sum()


def _sample(theta: float, a: float, b: np.ndarray, rng) -> int:
    return int(rng.choice(len(b) + 1, p=_gpcm_probs(theta, a, b)))


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(cfg: BenchDataConfig) -> BenchDataset:
    rng = np.random.default_rng(cfg.seed)
    N, Q, T, K = cfg.n_learners, cfg.n_items, cfg.seq_len, cfg.n_cats

    # ---- ground-truth item params (shared by all kinds) ----
    a = rng.lognormal(mean=0.0, sigma=0.3, size=Q)
    b = np.sort(rng.standard_normal((Q, K - 1)), axis=1)

    theta0 = rng.standard_normal(N)

    # Optional planted theta-dependent discrimination, alpha_eff = a*exp(gamma*theta).
    # Drawn from a SEPARATE rng so the static null (slope=0) and the planted
    # (slope>0) at the same seed share identical a, b, theta, item sequences AND
    # the per-step choice draws; the responses differ ONLY through alpha_eff.
    # That makes a matched null for bias control.
    gamma = None
    if cfg.alpha_theta_slope > 0.0:
        gamma = np.random.default_rng(cfg.seed + 10007).normal(
            0.0, cfg.alpha_theta_slope, size=Q)

    items0 = np.zeros((N, T), dtype=np.int64)
    responses = np.zeros((N, T), dtype=np.int64)
    theta_at_step = np.zeros((N, T), dtype=np.float64)

    theta_traj = None
    if cfg.kind == "dynamic":
        # random walk: theta_t = theta0 + cumulative drift up to (but not
        # including) the current step, so theta at step 0 is theta0.
        steps = rng.normal(0.0, cfg.drift_sigma, size=(N, T))
        cum = np.cumsum(steps, axis=1)
        theta_traj = np.zeros((N, T))
        theta_traj[:, 0] = theta0
        theta_traj[:, 1:] = theta0[:, None] + cum[:, :-1]

    for i in range(N):
        item_seq = rng.integers(0, Q, size=T)   # sample items with replacement
        items0[i] = item_seq
        for t, j in enumerate(item_seq):
            th = theta_traj[i, t] if theta_traj is not None else theta0[i]
            a_eff = a[j] * np.exp(gamma[j] * th) if gamma is not None else a[j]
            responses[i, t] = _sample(th, a_eff, b[j], rng)
            theta_at_step[i, t] = th

    # ---- train / val learner split (shared by both engines) ----
    perm = rng.permutation(N)
    n_train = int(round(N * cfg.train_frac))
    train_idx = np.sort(perm[:n_train])
    val_idx = np.sort(perm[n_train:])

    gt = BenchGroundTruth(theta0=theta0, a=a, b=b, theta_traj=theta_traj, gamma=gamma)
    return BenchDataset(
        cfg=cfg, gt=gt, items0=items0, responses=responses,
        train_idx=train_idx, val_idx=val_idx, theta_at_step=theta_at_step,
    )


# ---------------------------------------------------------------------------
# ma-irt JSON export (1-indexed items)
# ---------------------------------------------------------------------------

def export_ma_irt(ds: BenchDataset, out_root: str, subset: str = "train") -> Path:
    """Write a ma-irt dataset folder for the given learner subset.

    Items are written 1-indexed (id 0 reserved for padding). The folder layout
    matches utils.dataloader.SequenceDataset:
        <out_root>/<name>_<subset>/sequences.json
        <out_root>/<name>_<subset>/metadata.json
        <out_root>/<name>_<subset>/true_irt_parameters.json
    Returns the dataset directory path.
    """
    idx = ds.train_idx if subset == "train" else ds.val_idx
    out_dir = Path(out_root) / f"{ds.cfg.name}_{subset}"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i in idx:
        q1 = (ds.items0[i] + 1).tolist()        # 1-indexed
        r = ds.responses[i].tolist()
        records.append({"questions": q1, "responses": r})
    with (out_dir / "sequences.json").open("w", encoding="utf-8") as fh:
        json.dump(records, fh)

    meta = {"n_questions": int(ds.cfg.n_items), "n_categories": int(ds.cfg.n_cats)}
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh)

    # ground truth aligned to the subset rows (theta is per-learner init/mean)
    true = {
        "alpha": ds.gt.a.tolist(),
        "beta": ds.gt.b.tolist(),
        "theta": ds.gt.theta0[idx].tolist(),
    }
    with (out_dir / "true_irt_parameters.json").open("w", encoding="utf-8") as fh:
        json.dump(true, fh)

    return out_dir
