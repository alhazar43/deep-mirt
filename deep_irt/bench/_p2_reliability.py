"""
_p2_reliability.py -- Split-half reliability metrics for Phase 2 (step 6b).

Real data carries no ground-truth item parameters, so recovery correlation is
unmeasurable.  The measurable proxy is RELIABILITY: split the learners into two
random halves, fit an independent model on each, correlate the per-item recovered
parameters on the items both halves saw, and Spearman-Brown correct the half
correlation up to full length.  A model whose recovery is meaningful reproduces
its item ordering across disjoint learner samples; a model that reads noise does
not.

Decoder-agnostic.  The correlation core does not care which decoder produced the
numbers: the slope channel (GPCM/2PL ``alpha`` or NRM ``a_k``) is scored by
sign-aligned Spearman (rank, gauge-free), the location channel (GPCM ``beta`` or
NRM ``c_k``) by sign-aligned Pearson.  ``split_half_reliability`` pulls those
arrays through the engine's ``model.recover_item_params`` and restricts to the
items seen in BOTH halves (items unseen in a half keep at-init embeddings and
must never enter the correlation).

Reuse (blueprint section E).  The bootstrap CI reuses
``_p2_aggregate.bootstrap_ci`` (the same percentile machinery the CV aggregation
uses).  Sign alignment and the correlation kernels mirror
``metrics_bench``/``nrm_metrics`` so a reliability number and a recovery number
are computed the same way.

Out of scope here.  Ability reliability by ITEM split (fit on disjoint item
halves, correlate per-learner theta) is a different partition than the learner
split used for item reliability; ``theta_reliability`` is returned as NaN with a
note rather than conflated into the learner-split path.

Scratch file: ``_``-prefixed, gitignored.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy import stats as sp_stats

from deep_irt.bench._p2_aggregate import bootstrap_ci


# ---------------------------------------------------------------------------
# Correlation kernels (pure, unit-testable)
# ---------------------------------------------------------------------------

def _corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    """Pearson (method='pearson') or Spearman (method='spearman') correlation.

    Returns NaN when either side is (near-)constant or fewer than 3 finite pairs
    remain -- the honest value, never a silent 0.  Ties in the Spearman path are
    handled by scipy's average-rank convention (same as the recovery metrics).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3:
        return float("nan")
    x, y = x[m], y[m]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    if method == "spearman":
        return float(sp_stats.spearmanr(x, y).correlation)
    return float(sp_stats.pearsonr(x, y)[0])


def sign_aligned_corr(x: np.ndarray, y: np.ndarray, method: str = "spearman") -> float:
    """|correlation| between two halves' recovered params.

    The latent scale orientation (sign) is not identified, so a global flip
    between the two independently-fit halves carries no information; the reliable
    quantity is the magnitude of the correlation.  Returns |r| (or |rho|).
    """
    r = _corr(x, y, method)
    return abs(r) if r == r else float("nan")  # NaN-safe abs


# ---------------------------------------------------------------------------
# Spearman-Brown correction
# ---------------------------------------------------------------------------

def spearman_brown(r_half: float) -> float:
    """Full-length reliability from split-half correlation.

    r_full = 2 * r_half / (1 + r_half)

    Args:
        r_half: correlation between odd-half and even-half recovered params.

    Returns:
        Spearman-Brown corrected reliability. NaN if r_half is NaN or <= -1.
    """
    if r_half != r_half or r_half <= -1.0:  # NaN or degenerate
        return float("nan")
    return 2.0 * r_half / (1.0 + r_half)


# ---------------------------------------------------------------------------
# Recovered-parameter adapter (decoder-agnostic slope / location extraction)
# ---------------------------------------------------------------------------

def _slope_location(rec: dict):
    """Pull (slope, location) arrays from a ``recover_item_params`` dict.

    GPCM/2PL -> slope = alpha (Q,), location = per-item mean threshold (Q,).
    NRM      -> slope = a_k (Q, K),  location = c_k / intercept (Q, K).
    The location is reduced to one value per item for GPCM (mean threshold) so
    both channels are per-item comparable; NRM keeps its (Q, K) option grid and
    the correlation flattens over (item, option) after the seen-item restriction.
    """
    slope = np.asarray(rec.get("alpha"), dtype=float)
    if "beta" in rec and rec["beta"] is not None:
        loc = np.asarray(rec["beta"], dtype=float)
        loc = loc.mean(axis=1) if loc.ndim == 2 else loc
    elif "intercept" in rec and rec["intercept"] is not None:
        loc = np.asarray(rec["intercept"], dtype=float)
    else:
        loc = None
    return slope, loc


def _seen_from_items(items: np.ndarray, Q: int,
                     mask: Optional[np.ndarray] = None) -> np.ndarray:
    """(Q,) bool of items appearing in a half (valid positions only if masked)."""
    seen = np.zeros(Q, dtype=bool)
    ids = items[mask] if mask is not None else items
    ids = np.asarray(ids).ravel()
    ids = ids[ids >= 0]
    if ids.size:
        seen[np.unique(ids)] = True
    return seen


# ---------------------------------------------------------------------------
# Core split-half routine
# ---------------------------------------------------------------------------

def split_half_reliability(
    responses: np.ndarray,
    items: np.ndarray,
    engine_cls: type,
    engine_kwargs: dict,
    train_kwargs: dict,
    n_splits: int = 8,
    seed: int = 0,
    mask: Optional[np.ndarray] = None,
    n_cats: int = 4,
    n_holdout: int = 10,
) -> Dict[str, float]:
    """Odd/even split-half reliability for recovered item parameters.

    Randomly splits learners into two halves, fits two independent engines, and
    correlates their per-item recovered parameters, restricted to items seen in
    BOTH halves.  The slope channel is scored by sign-aligned Spearman, the
    location channel by sign-aligned Pearson, each Spearman-Brown corrected.
    Repeated ``n_splits`` times; the mean and std across splits are returned.

    Args:
        responses: (N, T) response/category matrix (0-indexed; pad value ignored
            via ``mask``).
        items: (N, T) 0-indexed item-id matrix.  Negative ids are treated as pad.
        engine_cls: an engine class taking ``ds=<BenchDataset>`` plus
            ``engine_kwargs`` (e.g. ``_P2Engine`` -- mask-aware -- or
            ``DeepIRTEngine``).  Must expose ``.fit(**train_kwargs)`` and
            ``.model.recover_item_params()``.
        engine_kwargs: constructor kwargs common to both halves (decoder, dims,
            device, state flags, item_key_dim, seed, ...).
        train_kwargs: kwargs forwarded to ``engine.fit`` (n_epochs, lr, ...).
        n_splits: number of random split-half resamples (blueprint: 8).
        seed: base RNG seed for split generation.
        mask: optional (N, T) bool validity mask; if None it is derived as
            ``items >= 0``.  Threaded to the half datasets so a mask-aware engine
            skips padding.
        n_cats: K, for the half datasets' config.
        n_holdout: holdout width for the half datasets' config (unused by
            reliability, required by the engine's ds.cfg).

    Returns:
        Dict with a_reliability, b_reliability (mean Spearman-Brown across splits),
        their _std, theta_reliability (NaN -- see module docstring), and the
        raw per-split lists a_split / b_split for downstream bootstrap CIs.
    """
    responses = np.asarray(responses)
    items = np.asarray(items)
    N, T = items.shape
    Q = int(items[items >= 0].max()) + 1 if (items >= 0).any() else 0
    if mask is None:
        mask = items >= 0
    mask = np.asarray(mask, dtype=bool)

    a_splits: List[float] = []
    b_splits: List[float] = []

    for s in range(n_splits):
        rng = np.random.default_rng(seed + s * 100)
        perm = rng.permutation(N)
        half = N // 2
        idx_a = np.sort(perm[:half])
        idx_b = np.sort(perm[half: 2 * half])

        rec_a, seen_a = _fit_recover_half(
            responses, items, mask, idx_a, engine_cls, engine_kwargs,
            train_kwargs, Q, n_cats, n_holdout,
            init_seed=seed + s * 1000 + 1,
        )
        rec_b, seen_b = _fit_recover_half(
            responses, items, mask, idx_b, engine_cls, engine_kwargs,
            train_kwargs, Q, n_cats, n_holdout,
            init_seed=seed + s * 1000 + 2,
        )

        common = seen_a & seen_b
        if int(common.sum()) < 3:
            a_splits.append(float("nan"))
            b_splits.append(float("nan"))
            continue

        sa, la = _slope_location(rec_a)
        sb, lb = _slope_location(rec_b)

        r_a = sign_aligned_corr(sa[common], sb[common], method="spearman")
        a_splits.append(spearman_brown(r_a))

        if la is not None and lb is not None:
            r_b = sign_aligned_corr(la[common], lb[common], method="pearson")
            b_splits.append(spearman_brown(r_b))
        else:
            b_splits.append(float("nan"))

    return {
        "a_reliability": float(np.nanmean(a_splits)) if a_splits else float("nan"),
        "b_reliability": float(np.nanmean(b_splits)) if b_splits else float("nan"),
        "a_reliability_std": float(np.nanstd(a_splits)) if a_splits else float("nan"),
        "b_reliability_std": float(np.nanstd(b_splits)) if b_splits else float("nan"),
        "theta_reliability": float("nan"),  # item-split; out of this routine's scope
        "a_split": a_splits,
        "b_split": b_splits,
    }


def _fit_recover_half(responses, items, mask, idx, engine_cls, engine_kwargs,
                      train_kwargs, Q, n_cats, n_holdout, init_seed):
    """Fit one half and return (recover dict, (Q,) seen-mask for the half)."""
    # Lazy import: keep the pure helpers importable without the datagen container.
    from deep_irt.bench.datagen import (
        BenchDataConfig, BenchDataset, BenchGroundTruth,
    )

    it_half = items[idx].copy()
    rp_half = responses[idx].copy()
    m_half = mask[idx].copy()
    # Clamp pad ids to a valid index so the embedding lookup never sees a negative;
    # the mask keeps those positions out of the loss.
    it_half[~m_half] = 0
    it_half[it_half < 0] = 0

    n_half, T = it_half.shape
    cfg = BenchDataConfig(
        name="split_half", kind="static", n_learners=n_half, n_items=Q,
        seq_len=T, n_cats=n_cats, n_holdout=n_holdout, seed=init_seed,
    )
    gt = BenchGroundTruth(
        theta0=np.zeros(n_half), a=np.zeros(Q), b=np.zeros((Q, max(n_cats - 1, 1))),
    )
    ds = BenchDataset(
        cfg=cfg, gt=gt, items0=it_half, responses=rp_half,
        train_idx=np.arange(n_half), val_idx=np.arange(0),
        aux={"mask": m_half},
    )

    ekw = dict(engine_kwargs)
    ekw.setdefault("seed", init_seed)
    engine = engine_cls(ds=ds, **ekw)
    engine.fit(**train_kwargs)

    model = engine.model
    if getattr(model, "_uses_state", False):
        import torch
        it_t = torch.as_tensor(it_half, dtype=torch.long)
        rp_t = torch.as_tensor(rp_half, dtype=torch.long)
        rec = model.recover_item_params(it_t, rp_t)
        seen = np.asarray(rec["seen"], dtype=bool) if "seen" in rec else \
            _seen_from_items(it_half, Q, m_half)
    else:
        rec = model.recover_item_params()
        seen = _seen_from_items(it_half, Q, m_half)
    return rec, seen


# ---------------------------------------------------------------------------
# Item bootstrap CI
# ---------------------------------------------------------------------------

def item_bootstrap_ci(
    param_hat: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Bootstrap CI for a per-item statistic by resampling ITEMS with replacement.

    The item population is the resampling unit (the natural unit for a
    reliability/parameter summary): items are drawn with replacement, the mean is
    recomputed on each resample, and the percentile CI of that mean is returned.
    For a 2-D ``param_hat`` (Q, m) the CI is computed per column (e.g. per step
    threshold), returning length-m arrays.

    Args:
        param_hat: (Q,) or (Q, m) per-item estimates (e.g. per-item split-half
            reliability, or a recovered parameter column).
        n_boot: bootstrap resamples.
        ci: coverage level (0.95 -> 2.5th / 97.5th percentiles).
        seed: RNG seed.

    Returns:
        {"mean": np.ndarray, "ci_lo": np.ndarray, "ci_hi": np.ndarray} -- scalars
        (0-d arrays) for 1-D input, length-m arrays for 2-D input.  NaN entries in
        a column propagate to NaN for that column (explicit, not silently dropped).

    NOTE: this returns the CI of the item-population MEAN, not a per-item band
    (a single per-item estimate has no within-item replicates to bootstrap).
    """
    param_hat = np.asarray(param_hat, dtype=float)
    if param_hat.ndim == 1:
        cols = [param_hat]
    elif param_hat.ndim == 2:
        cols = [param_hat[:, k] for k in range(param_hat.shape[1])]
    else:
        raise ValueError(f"param_hat must be 1-D or 2-D, got shape {param_hat.shape}")

    means, los, his = [], [], []
    for k, col in enumerate(cols):
        # Distinct seed per column so columns are not resampled in lockstep.
        res = bootstrap_ci(col, statistic=np.mean, n_boot=n_boot, ci=ci,
                           seed=seed + k)
        means.append(res["mean"])
        los.append(res["ci_lo"])
        his.append(res["ci_hi"])

    if param_hat.ndim == 1:
        return {"mean": np.array(means[0]), "ci_lo": np.array(los[0]),
                "ci_hi": np.array(his[0])}
    return {"mean": np.asarray(means), "ci_lo": np.asarray(los),
            "ci_hi": np.asarray(his)}


# ---------------------------------------------------------------------------
# Coverage moderation helper (real-data readout)
# ---------------------------------------------------------------------------

def reliability_by_coverage(
    a_reliabilities: List[float],
    coverage_counts: List[int],
    n_bins: int = 5,
) -> Dict[str, list]:
    """Bin per-item reliability by item coverage (responses seen per item).

    Reports the low-coverage degradation (the EdNet reversal): items answered by
    few learners have noisier recovered parameters and lower split-half
    reliability even under a well-specified model.  Items are binned by coverage
    quantile; the mean reliability and item count per bin are returned.

    Args:
        a_reliabilities: (Q,) per-item reliability (or any per-item score).
        coverage_counts: (Q,) per-item number of responses in the dataset.
        n_bins: number of coverage quantile bins.

    Returns:
        {"bin_edges": [...], "mean_reliability": [...], "n_items": [...]}
        Bins are coverage quantiles; empty bins (from ties/degenerate quantiles)
        report NaN mean and 0 count rather than being dropped, so the bin index is
        stable.
    """
    rel = np.asarray(a_reliabilities, dtype=float)
    cov = np.asarray(coverage_counts, dtype=float)
    if rel.shape != cov.shape:
        raise ValueError(
            f"a_reliabilities {rel.shape} and coverage_counts {cov.shape} "
            f"must have the same shape"
        )
    n_bins = int(max(1, n_bins))
    edges = np.quantile(cov, np.linspace(0.0, 1.0, n_bins + 1))
    # Deduplicate collapsed quantile edges (heavy ties) so digitize is well-posed.
    edges = np.unique(edges)
    # np.digitize: bin index in [1, len(edges)-1] for interior; clamp the top edge.
    idx = np.clip(np.digitize(cov, edges[1:-1], right=False), 0, len(edges) - 2)

    n_eff = len(edges) - 1
    mean_rel, n_items = [], []
    for b in range(n_eff):
        sel = idx == b
        vals = rel[sel]
        finite = vals[np.isfinite(vals)]
        mean_rel.append(float(finite.mean()) if finite.size else float("nan"))
        n_items.append(int(sel.sum()))

    return {
        "bin_edges": edges.tolist(),
        "mean_reliability": mean_rel,
        "n_items": n_items,
    }
