"""
_p2_aggregate.py -- Cross-cell aggregation and figure-data builders for Phase 2.

SCAFFOLD: fill-in pending. Signatures and return-type contracts specified;
bodies raise NotImplementedError with explicit TODO(fill-in) notes.

Role:
  - aggregate_rows(): per-seed rows -> mean + bootstrap 95% CI for one cell.
  - bootstrap_ci(): non-parametric bootstrap for any scalar statistic.
  - wilcoxon_comparison(): paired Wilcoxon signed-rank + rank-biserial r.
  - build_swap_matrix(): E1 encoder x decoder panel from a manifest.
  - build_kendalls_w(): ordering concordance for E1.
  - build_budget_curve(): E-budget curve (one axis vs metric + CI).
  - load_manifest() / load_cell_results(): manifest I/O helpers.

  All functions are deterministic given their inputs. The RNG seed for bootstrap
  is an explicit argument so the CI is reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable = np.mean,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> Dict[str, float]:
    """Non-parametric bootstrap CI for a scalar statistic over 1-D values.

    Args:
        values: 1-D array of observed values (e.g. per-seed metric).
        statistic: callable applied to each bootstrap sample; defaults to mean.
        n_boot: number of bootstrap resamples.
        ci: coverage level (0.95 -> 2.5th and 97.5th percentiles).
        seed: RNG seed for reproducibility.

    Returns:
        {"mean": float, "ci_lo": float, "ci_hi": float, "n": int}
        NaN propagates: if values contains NaN, all outputs are NaN.

    Adjudication note: this bootstrap CI is the primary uncertainty report under
    the LOCKED 5-fold-CV protocol (decision 1).  Paired fold-mean difference +
    this CI + rank-biserial (see :func:`wilcoxon_comparison`) adjudicate toggle
    comparisons; a 5-sample significance test is deliberately NOT used (5 paired
    folds cannot reach p<0.05).
    """
    values = np.asarray(values, dtype=float).ravel()
    n = int(values.size)
    if n == 0 or not np.all(np.isfinite(values)):
        # NaN propagates: any missing/NaN value voids the CI (explicit, not silent).
        return {"mean": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "n": n}
    if n == 1:
        v = float(values[0])
        return {"mean": v, "ci_lo": v, "ci_hi": v, "n": 1}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = np.array([statistic(values[row]) for row in idx], dtype=float)
    lo_pct = (1.0 - ci) / 2.0 * 100.0
    hi_pct = (1.0 + ci) / 2.0 * 100.0
    lo, hi = np.percentile(samples, [lo_pct, hi_pct])
    return {
        "mean": float(statistic(values)),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n": n,
    }


# ---------------------------------------------------------------------------
# Within-cell aggregation
# ---------------------------------------------------------------------------

# Metrics to aggregate by default (all scored parameters + prediction).
SCORED_METRICS = [
    "a_spearman", "a_pearson",
    "b_spearman", "b_pearson",
    "c_spearman", "c_pearson",   # NRM per-option intercept recovery
    "theta_spearman", "theta_pearson",
    "theta_netdrift_spearman", "theta_netdrift_pearson",
    "acc", "qwk", "auc",
    "nll", "macro_auc",   # NRM prediction metrics
]


def aggregate_rows(
    rows: List[Dict],
    metrics: Optional[List[str]] = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Aggregate per-seed result rows for a single cell.

    Args:
        rows: list of dicts, each a run_one() output.
        metrics: which metric keys to aggregate (default: SCORED_METRICS).
        n_boot: bootstrap resamples for CI.
        seed: RNG seed for bootstrap.

    Returns:
        Dict mapping each metric name to {"mean", "ci_lo", "ci_hi", "n"}.
        Metrics absent from all rows or all-NaN are included with NaN values.

    A metric present in some rows but NaN/missing in others yields an all-NaN
    agg for that metric (NaN propagates via bootstrap_ci), which is the honest
    report -- a partially-unstable cell should not silently average its stable
    folds.  A metric absent from every row is included with NaN and n=0.
    """
    if metrics is None:
        metrics = SCORED_METRICS
    out: Dict[str, Dict[str, float]] = {}
    for m in metrics:
        vals = [row[m] for row in rows if m in row]
        if not vals:
            out[m] = {"mean": float("nan"), "ci_lo": float("nan"),
                      "ci_hi": float("nan"), "n": 0}
            continue
        out[m] = bootstrap_ci(np.asarray(vals, dtype=float),
                              n_boot=n_boot, seed=seed)
    return out


# ---------------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------------

def wilcoxon_comparison(
    a_values: np.ndarray,
    b_values: np.ndarray,
) -> Dict[str, float]:
    """Paired Wilcoxon signed-rank test + rank-biserial effect size.

    Suitable for paired (same seed) metric comparisons between two conditions
    (e.g. coupled vs decoupled, dynamic vs static head).

    Args:
        a_values: per-seed values for condition A (shape: (n_seeds,)).
        b_values: per-seed values for condition B (same seed order as a_values).

    Returns:
        {"statistic", "p_value", "rank_biserial", "n_pairs",
         "mean_diff", "diff_ci_lo", "diff_ci_hi"}

    Adjudication note (LOCKED decision 1).  Under 5-fold CV the paired sample has
    only 5 folds, so the Wilcoxon p_value CANNOT reach 0.05 (min two-sided p =
    2/2^5 = 0.0625) and is reported for completeness only.  The ACTUAL toggle
    adjudication is ``mean_diff`` + its bootstrap 95% CI (``diff_ci_*``) +
    ``rank_biserial`` effect size.  rank_biserial is the matched-pairs form
    (W+ - W-)/(W+ + W-) over the signed ranks of the nonzero differences;
    positive -> A > B.  (This is sign-correct and tie-aware, unlike the raw
    1 - 4W/(n(n+1)) sketch in the original stub.)
    """
    a = np.asarray(a_values, dtype=float).ravel()
    b = np.asarray(b_values, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(
            f"a_values {a.shape} and b_values {b.shape} must be paired (same shape)"
        )
    n = int(a.size)
    d = a - b
    finite = n > 0 and bool(np.all(np.isfinite(d)))

    mean_diff = float(np.mean(d)) if finite else float("nan")
    ci = (bootstrap_ci(d, n_boot=2000, seed=0)
          if finite else {"ci_lo": float("nan"), "ci_hi": float("nan")})

    # Matched-pairs rank-biserial from signed ranks of the nonzero differences.
    if finite:
        nz = d[d != 0.0]
        if nz.size == 0:
            rank_biserial = 0.0
        else:
            ranks = sp_stats.rankdata(np.abs(nz))
            w_plus = float(ranks[nz > 0].sum())
            w_minus = float(ranks[nz < 0].sum())
            tot = w_plus + w_minus
            rank_biserial = float((w_plus - w_minus) / tot) if tot > 0 else 0.0
    else:
        rank_biserial = float("nan")

    # scipy Wilcoxon signed-rank (inert at n=5; reported for completeness).
    statistic = float("nan")
    p_value = float("nan")
    if finite and np.any(d != 0.0):
        try:
            res = sp_stats.wilcoxon(a, b, alternative="two-sided",
                                    zero_method="wilcox")
            statistic = float(res.statistic)
            p_value = float(res.pvalue)
        except Exception:
            pass

    return {
        "statistic": statistic,
        "p_value": p_value,
        "rank_biserial": rank_biserial,
        "n_pairs": n,
        "mean_diff": mean_diff,
        "diff_ci_lo": float(ci["ci_lo"]),
        "diff_ci_hi": float(ci["ci_hi"]),
    }


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: str | Path) -> List[Dict]:
    """Load a manifest.json; return the list of cell entry dicts."""
    return json.loads(Path(manifest_path).read_text()).get("cells", [])


def load_cell_results(results_path: str | Path) -> Dict:
    """Load a single cell's results.json."""
    return json.loads(Path(results_path).read_text())


# ---------------------------------------------------------------------------
# Cross-cell aggregation builders
# ---------------------------------------------------------------------------

def build_swap_matrix(
    manifest_path: str | Path,
    metrics: Optional[List[str]] = None,
) -> Dict:
    """Build E1 encoder x decoder swap matrix from a manifest.

    Returns a nested dict keyed by (encoder, decoder) with per-metric agg dicts.
    Cells not present in the manifest are marked as missing.

    Returns:
        {"matrix":      {(encoder, decoder): {metric: agg}},
         "missing":     [cell_name, ...],           # results.json absent
         "encoders":    [...], "decoders": [...],   # sorted axis labels
         "kendalls_w":  {metric: W}}                # E1 ordering concordance
    """
    if metrics is None:
        metrics = SCORED_METRICS
    cells = load_manifest(manifest_path)

    matrix: Dict = {}
    missing: List[str] = []
    encoders: List[str] = []
    decoders: List[str] = []
    for entry in cells:
        rp = entry.get("results_path")
        if not rp or not Path(rp).exists():
            missing.append(entry.get("cell_name", "<unknown>"))
            continue
        res = load_cell_results(rp)
        mc = res.get("config", {}).get("model", {})
        enc, dec = mc.get("encoder"), mc.get("decoder")
        agg = aggregate_rows(res.get("per_seed_rows", []), metrics=metrics)
        matrix[(enc, dec)] = agg
        if enc not in encoders:
            encoders.append(enc)
        if dec not in decoders:
            decoders.append(dec)

    encoders.sort()
    decoders.sort()

    # Kendall's W per metric: do the encoders agree on the decoder ordering?
    kendalls_w: Dict[str, float] = {}
    for m in metrics:
        per_encoder = {}
        for enc in encoders:
            row = [matrix.get((enc, dec), {}).get(m, {}).get("mean", float("nan"))
                   for dec in decoders]
            per_encoder[enc] = row
        kendalls_w[m] = build_kendalls_w(per_encoder)

    return {
        "matrix": matrix,
        "missing": missing,
        "encoders": encoders,
        "decoders": decoders,
        "kendalls_w": kendalls_w,
    }


def build_kendalls_w(
    per_encoder_rankings: Dict[str, List[float]],
) -> float:
    """Compute Kendall's W (ordering concordance) over per-encoder rankings.

    per_encoder_rankings: {encoder: [metric_val_per_decoder_in_fixed_order]}

    Kendall's W = 1 means all encoders agree on the decoder ordering;
    W = 0 means no agreement. E1 claim: the per-parameter recovery ordering
    is consistent across encoders (high W for theta and b; lower W for a).

    Uncorrected W (no tie correction; average ranks handle ties adequately for
    the small E1 grids).  Returns NaN if fewer than 2 raters/objects or any
    ranking value is non-finite.
    """
    encoders = list(per_encoder_rankings)
    m = len(encoders)
    if m < 2:
        return float("nan")
    mat = np.asarray([per_encoder_rankings[e] for e in encoders], dtype=float)
    if mat.ndim != 2 or mat.shape[1] < 2 or not np.all(np.isfinite(mat)):
        return float("nan")
    n = mat.shape[1]
    # Rank the n objects (decoders) within each rater (encoder); higher metric
    # value -> higher rank.  Average ranks for ties (rankdata default).
    ranks = np.vstack([sp_stats.rankdata(row) for row in mat])   # (m, n)
    rank_sums = ranks.sum(axis=0)                                # (n,)
    S = float(np.sum((rank_sums - rank_sums.mean()) ** 2))
    return float(12.0 * S / (m ** 2 * (n ** 3 - n)))


def build_budget_curve(
    manifest_path: str | Path,
    axis: str,
    metric: str,
) -> Dict:
    """Extract E-budget curve data for one axis and one metric.

    axis: "T" | "N" | "sigma_b" | "K" (must match cell_name suffix convention
          set by _p2_datagen_budget.make_*_grid).
    metric: e.g. "a_spearman", "b_spearman", "theta_spearman".

    Returns:
        {"axis_values": [...], "means": [...], "ci_lo": [...], "ci_hi": [...]}
        sorted by axis value ascending.

    TODO(fill-in): for each cell in the manifest, parse the axis value from
      cell_name (e.g. "_T50" -> 50 for axis="T") and load the agg[metric] values.
    TODO(fill-in): sort by axis value, return curve arrays.
    TODO(fill-in): call fit_exponential_decay() from _p2_datagen_budget for axis="T".
    """
    raise NotImplementedError("build_budget_curve not yet implemented -- fill-in pending")


def build_levers_comparison(
    manifest_path: str | Path,
    metric: str = "a_spearman",
) -> Dict:
    """Build E-levers comparison: {static, dynamic} x {gpcm, nrm} agg values.

    TODO(fill-in): filter manifest cells by experiment tag, aggregate, run
      wilcoxon_comparison for each decoder (gpcm and nrm separately).
    """
    raise NotImplementedError("build_levers_comparison not yet implemented -- fill-in pending")
