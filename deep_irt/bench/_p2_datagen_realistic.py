"""_p2_datagen_realistic.py -- step-6a realistic synthetic bed for Phase 2.

CORRECTED SPEC (ma-irt-aligned; this is the PRIMARY bed for the benchmark, the
toggles, and the proof -- not a second leg).  The clean benchmark bed
(``datagen.generate``) is rectangular: every learner answers the same number of
items, sampled WITH replacement from the whole bank.  This bed instead mirrors
the way real assessment logs look, and its ONLY source of realism/noise is the
administration:

  * ADMINISTRATION IS THE ONLY NOISE.  Each learner is administered a random
    subset of DISTINCT items, drawn WITHOUT replacement; every administered item
    is answered exactly once.  The NUMBER of items per learner is an integer
    ~ Uniform[admin_min, admin_max] (default 40..80, exactly ma-irt's range).
    Selection is theta-INDEPENDENT and non-adaptive (item popularity, when used,
    is independent of difficulty/discrimination and of ability).
  * CLEAN IRT DRAWS.  Responses are drawn from the true GPCM/2PL/NRM model with
    NO guessing, NO lapse, NO response noise.  All realism comes from the random,
    incomplete administration above.
  * VARIABLE LENGTH falls out of the administration: the valid length T_n equals
    the administered count, padded tail-only, with a boolean ``mask`` so the loss
    and the prediction score skip the padding.
  * VARIABLE EXPOSURE also falls out of the administration: with a Q=200 bank and
    ~40..80 distinct items per learner each item gets ~500-600 takers; recovery
    is only meaningful on the items a fold actually saw (scored on the SEEN set,
    exposure-stratified via ``aux['coverage']``).
  * DENSE CONTROL: ``data.dense=True`` administers ALL Q items to every learner
    (rectangular, everyone sees everything) -- the small control bed.

Reuse, not reinvention (blueprint section C + E): the response draws are the
SAME wheels the clean bed uses -- ``datagen._sample`` (calls ``_gpcm_probs``) for
the GPCM/2PL family and ``nrm_datagen._sample_item_params``/``_nrm_probs`` for
the Bock NRM.  ``datagen.py`` and ``nrm_datagen.py`` are NOT edited; only their
generation kernels are imported.

The returned container is a plain :class:`datagen.BenchDataset` so the fill-in
runner (`_p2_run_cell`) and engine (`_p2_engine`) consume it with no special
case.  Everything the realistic bed adds beyond the clean bed rides in
``ds.aux`` (a declared field, so it survives ``dataclasses.replace`` in the CV
fold loop):

    aux["mask"]            (N, T) bool   valid position (True) vs tail pad (False)
    aux["administered"]    list of (n_i,) int arrays -- each learner's seen set
    aux["seen_union"]      (Q,) bool     items seen by >=1 learner (valid only)
    aux["coverage"]        (Q,) int      valid observations per item (= takers)
    aux["lengths"]         (N,) int      administered count / valid length per learner
    aux["exposure_counts"] (N,) int      distinct items each learner saw (== lengths)
    aux["holdout"]         (N,) int      per-learner prediction holdout (see below)
    aux["popularity"]      (Q,) float    exposure weights (param-independent)
    aux["nrm_a"], aux["nrm_c"]  (Q, K)   NRM ground truth (NRM decoder only)

Seen-union contract (the blueprint's seen-mask fix).  ``_p2_run_cell`` derives
the per-fold seen mask as ``np.unique(items0[train_rows])`` -- over ALL columns,
padding included -- and threads it to ``item_recovery(seen=...)``.  To keep that
derivation exact, the tail padding repeats the learner's FIRST administered item
(``seq[0]``), never a fresh id, so padded columns introduce no item that was not
genuinely administered.  Thus ``unique(items0[train_rows])`` equals the true
administered union over the training folds, and the shared-static baseline is
scored only on items it actually saw (unseen items keep at-init embeddings and
are correctly excluded).

Per-learner holdout.  ``aux["holdout"]`` = ``min(n_holdout, floor(holdout_frac *
T_n))`` so short learners keep some history.  NOTE: the current
``_p2_engine.predict_heldout`` scores the last ``cfg.n_holdout`` VALID positions
(a single scalar), so it does not yet consume this per-learner array; the array
is emitted for the engine build that will.  With admin_min >= 40 (the default)
every learner keeps ample history, so this is comfortably above ``n_holdout``.

Scratch file: ``_``-prefixed, gitignored.  Imports generation kernels from the
read-only clean generators; does not edit them.
"""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np

from deep_irt.bench.datagen import (
    BenchDataConfig,
    BenchDataset,
    BenchGroundTruth,
    _sample,  # GPCM/2PL response kernel (calls datagen._gpcm_probs internally)
)
from deep_irt.bench.nrm_datagen import _nrm_probs, _sample_item_params


# ---------------------------------------------------------------------------
# Tunables (module constants so a config field is not needed for shape-only knobs)
# ---------------------------------------------------------------------------

# Zipf exponent for the popularity draw (only when popularity == "zipf").
_ZIPF_S = 1.0

# NRM designated-correct option (matches nrm_datagen default).
_NRM_CORRECT = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _popularity_weights(Q: int, kind: str, rng) -> np.ndarray:
    """Per-item exposure weights, INDEPENDENT of (difficulty, discrimination).

    "uniform" -> equal weight.  "zipf" -> weight proportional to 1 / rank^s over
    a RANDOM rank permutation (so popularity carries no information about the item
    parameters, as the corrected spec requires).  Returns a (Q,) vector summing
    to 1.
    """
    if kind == "uniform":
        w = np.ones(Q, dtype=float)
    elif kind == "zipf":
        ranks = rng.permutation(Q).astype(float) + 1.0  # random rank, param-free
        w = 1.0 / np.power(ranks, _ZIPF_S)
    else:
        raise ValueError(f"unknown popularity {kind!r} (want 'uniform' | 'zipf')")
    return w / w.sum()


def _draw_admin_counts(N: int, admin_min: int, admin_max: int, Q: int,
                       dense: bool, rng) -> np.ndarray:
    """Per-learner number of DISTINCT administered items.

    ``dense=True`` -> every learner is administered ALL Q items (the rectangular
    dense control).  Else an integer ~ Uniform[admin_min, admin_max] (inclusive),
    clamped into [1, Q] since a learner cannot see more distinct items than the
    bank holds.
    """
    if dense:
        return np.full(N, Q, dtype=np.int64)
    lo = int(max(1, min(admin_min, Q)))
    hi = int(max(lo, min(admin_max, Q)))
    return rng.integers(lo, hi + 1, size=N).astype(np.int64)  # inclusive range


def _seen_union(items0: np.ndarray, mask: np.ndarray, Q: int) -> np.ndarray:
    """(Q,) bool: items appearing in >=1 VALID position across all learners."""
    seen = np.zeros(Q, dtype=bool)
    if mask.any():
        seen[np.unique(items0[mask])] = True
    return seen


def _coverage_counts(items0: np.ndarray, mask: np.ndarray, Q: int) -> np.ndarray:
    """(Q,) int: number of VALID observations per item (padding excluded)."""
    if not mask.any():
        return np.zeros(Q, dtype=np.int64)
    return np.bincount(items0[mask], minlength=Q).astype(np.int64)


# ---------------------------------------------------------------------------
# Ground-truth item parameters + ability
# ---------------------------------------------------------------------------

def _draw_gpcm_items(Q: int, K: int, alpha_sigma: float, beta_sigma: float,
                     rng) -> Tuple[np.ndarray, np.ndarray]:
    """GPCM/2PL ground truth: discrimination a ~ LogNormal(0, alpha_sigma) (Q,);
    difficulty as ordered step thresholds b ~ sorted N(0, beta_sigma) (Q, K-1).

    Mirrors ma-irt's block generator (``alpha ~ LogNormal(0, 0.5)``) when the
    config sets ``alpha_sigma=0.5``; honours the configurable spreads."""
    a = rng.lognormal(mean=0.0, sigma=alpha_sigma, size=Q)
    b = np.sort(rng.standard_normal((Q, K - 1)) * beta_sigma, axis=1)
    return a, b


def _ability(kind: str, N: int, T: int, drift_sigma: float, rng):
    """Return (theta0 (N,), theta_traj (N, T) or None).

    Static -> theta_traj None (theta0 ~ N(0,1) used at every step).  Dynamic ->
    per-step random-walk drift with theta at step 0 equal to theta0 (the drift
    robustness arm; matches ``datagen``)."""
    theta0 = rng.standard_normal(N)
    if kind != "dynamic":
        return theta0, None
    steps = rng.normal(0.0, drift_sigma, size=(N, T))
    cum = np.cumsum(steps, axis=1)
    theta_traj = np.zeros((N, T))
    theta_traj[:, 0] = theta0
    theta_traj[:, 1:] = theta0[:, None] + cum[:, :-1]
    return theta0, theta_traj


# ---------------------------------------------------------------------------
# Public entry point (called by _p2_run_cell._generate_dataset)
# ---------------------------------------------------------------------------

def generate_realistic(cfg, data_seed: int) -> BenchDataset:
    """Generate one realistic-bed :class:`BenchDataset` for ``data_seed``.

    ``cfg`` is a :class:`_p2_config.P2Config`.  The response family follows the
    MODEL decoder (``cfg.model.decoder``): "nrm" -> Bock NRM; anything else
    ("gpcm" / "binary") -> GPCM (2PL is GPCM at K=2).  The administration axes are
    read from ``cfg.data`` (admin_min / admin_max / dense / popularity /
    holdout_frac / kind).

    Determinism: all randomness flows from a single ``default_rng(data_seed)``,
    so the bank, the administration, the responses, and the learner split are
    reproducible for a given seed (matching the clean bed's convention).
    """
    dc = cfg.data
    decoder = getattr(cfg.model, "decoder", "gpcm")
    family = "nrm" if decoder == "nrm" else "gpcm"

    N, Q, T, K = dc.n_learners, dc.n_items, dc.seq_len, dc.n_cats
    dense = bool(getattr(dc, "dense", False))
    dynamic = dc.kind == "dynamic"
    rng = np.random.default_rng(data_seed)

    # seq_len must hold the largest administration; padding fills the rest.
    max_admin = Q if dense else int(min(getattr(dc, "admin_max", 80), Q))
    if T < max_admin:
        raise ValueError(
            f"seq_len={T} < max administered items={max_admin} "
            f"({'dense: all Q' if dense else 'admin_max clamped to Q'}); "
            f"raise data.seq_len to at least {max_admin}."
        )

    # ---- ground-truth item parameters ----
    if family == "nrm":
        a_k, c_k = _sample_item_params(Q, K, _NRM_CORRECT, rng)      # (Q,K),(Q,K)
        a_gpcm = b_gpcm = None
    else:
        a_gpcm, b_gpcm = _draw_gpcm_items(Q, K, dc.alpha_sigma, dc.beta_sigma, rng)
        a_k = c_k = None

    # ---- ability ----
    theta0, theta_traj = _ability(dc.kind, N, T, dc.drift_sigma, rng)

    # ---- administration machinery ----
    pop = _popularity_weights(Q, dc.popularity, rng)
    counts = _draw_admin_counts(N, getattr(dc, "admin_min", 40),
                                getattr(dc, "admin_max", 80), Q, dense, rng)

    items0 = np.zeros((N, T), dtype=np.int64)
    responses = np.zeros((N, T), dtype=np.int64)
    theta_at_step = np.zeros((N, T), dtype=np.float64)
    mask = np.zeros((N, T), dtype=bool)
    administered: list = []

    for i in range(N):
        n_i = int(counts[i])
        # Theta-independent administration: n_i DISTINCT items drawn WITHOUT
        # replacement (weighted by the param-free popularity; uniform => plain
        # random).  Each administered item is answered exactly once.  choice
        # without replacement already returns them in a random order.
        seq = rng.choice(Q, size=n_i, replace=False, p=pop)

        items0[i, :n_i] = seq
        mask[i, :n_i] = True
        for t in range(n_i):
            j = int(seq[t])
            th = float(theta_traj[i, t]) if dynamic else float(theta0[i])
            theta_at_step[i, t] = th
            if family == "nrm":
                p = _nrm_probs(th, a_k[j], c_k[j])
                responses[i, t] = int(rng.choice(K, p=p))
            else:
                responses[i, t] = _sample(th, a_gpcm[j], b_gpcm[j], rng)

        # Tail padding repeats seq[0] (an administered item) so the seen-union
        # derived downstream from unique(items0) stays exact.  Responses stay 0
        # (masked out of the loss + the prediction score).
        if n_i < T:
            items0[i, n_i:] = seq[0]
            theta_at_step[i, n_i:] = theta_at_step[i, n_i - 1]
            if dynamic:
                theta_traj[i, n_i:] = theta_traj[i, n_i - 1]  # GT final ability held
        administered.append(np.unique(seq))

    # ---- learner train/val split (deterministic; folds re-partition later) ----
    perm = rng.permutation(N)
    n_train = int(round(N * dc.train_frac))
    train_idx = np.sort(perm[:n_train])
    val_idx = np.sort(perm[n_train:])

    # ---- ground-truth container ----
    if family == "nrm":
        # gt.a / gt.b are placeholders: the NRM path scores a_k/c_k from
        # aux["nrm_a"]/aux["nrm_c"] (nrm_metrics.item_recovery) and never reads
        # gt.a/gt.b.  Kept as well-formed zero arrays so the dataclass is valid.
        gt = BenchGroundTruth(
            theta0=theta0,
            a=np.zeros(Q, dtype=float),
            b=np.zeros((Q, max(K - 1, 1)), dtype=float),
            theta_traj=theta_traj,
            gamma=None,
        )
    else:
        gt = BenchGroundTruth(
            theta0=theta0, a=a_gpcm, b=b_gpcm, theta_traj=theta_traj, gamma=None,
        )

    seen_union = _seen_union(items0, mask, Q)
    coverage = _coverage_counts(items0, mask, Q)
    holdout = np.minimum(dc.n_holdout,
                         np.floor(dc.holdout_frac * counts).astype(np.int64))

    aux = {
        "mask": mask,
        "administered": administered,
        "seen_union": seen_union,
        "coverage": coverage,
        "lengths": counts,
        "exposure_counts": np.array([s.size for s in administered], dtype=np.int64),
        "holdout": holdout,
        "popularity": pop,
        "regime": "realistic",
        "dense": dense,
    }
    if family == "nrm":
        aux["nrm_a"] = a_k
        aux["nrm_c"] = c_k

    _warn_identifiability(family, K, N, Q, counts, coverage, seen_union,
                          getattr(dc, "admin_min", 40), dc.n_holdout, dense)

    bench_cfg = BenchDataConfig(
        name=getattr(cfg.report, "cell_name", "realistic"),
        kind=dc.kind,
        n_learners=N, n_items=Q, seq_len=T, n_cats=K,
        drift_sigma=dc.drift_sigma, train_frac=dc.train_frac,
        n_holdout=dc.n_holdout, seed=data_seed,
    )
    return BenchDataset(
        cfg=bench_cfg, gt=gt, items0=items0, responses=responses,
        train_idx=train_idx, val_idx=val_idx, theta_at_step=theta_at_step, aux=aux,
    )


# ---------------------------------------------------------------------------
# Identifiability guard (warn, never raise: the caller may want a stress bed)
# ---------------------------------------------------------------------------

def _warn_identifiability(family, K, N, Q, counts, coverage, seen_union,
                          admin_min, n_holdout, dense) -> None:
    """Emit warnings when the bed is below the recovery floors from the spec.

    * per-item coverage floor: >=100-200 obs/item (200 for GPCM K>=4).  With
      N=2000, Q=200, admin Uniform(40,80) the bed sits at ~500-600 takers/item,
      well above the floor; the warning fires only on down-scaled smoke/test beds.
    * short-learner history: admin_min must exceed n_holdout or the shortest
      learners have no prediction history under the scalar-holdout engine.
    """
    if seen_union.any():
        med_cov = float(np.median(coverage[seen_union]))
    else:
        med_cov = 0.0
    floor = 200 if (family == "gpcm" and K >= 4) else 100
    if med_cov < floor:
        warnings.warn(
            f"realistic bed: median {med_cov:.0f} obs/item < identifiability "
            f"floor {floor} (N={N}, Q={Q}, mean_admin={counts.mean():.0f}); "
            f"recovery may be noisy -- raise N or lower Q.",
            stacklevel=3,
        )
    if not dense and admin_min <= n_holdout:
        warnings.warn(
            f"realistic bed: admin_min={admin_min} <= n_holdout={n_holdout}; "
            f"the shortest learners have no prediction history. Raise admin_min.",
            stacklevel=3,
        )
