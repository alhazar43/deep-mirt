"""run_alpha_beta_asymmetry_stats.py -- paired statistics for the RQ1 result.

The asymmetry runner (run_alpha_beta_asymmetry.py) trains, per (K, seed), three
matched arms (baseline / alpha-dynamic / beta-dynamic) and writes one row per
(K, seed, arm) to outputs/alpha_beta_asymmetry.json.  Its built-in summary takes
deltas of K-LEVEL MEANS (mean a_dyn - mean a_base), which is not a paired test.

This analysis-only script reads that JSON and computes the PAIRED-PER-SEED
statistics needed for paper rigor, on the SIGN-ALIGNED SPEARMAN recovery
(a_spearman / b_spearman, the rank metric M.item_recovery already sign-aligns):

  delta_alpha[K, seed] = a_spearman(alpha-dynamic) - a_spearman(baseline)
  delta_beta [K, seed] = b_spearman(beta-dynamic)  - b_spearman(baseline)

Per K it reports, across seeds:
  - mean delta_alpha and delta_beta
  - seed-level 95% CI (t-distribution, n-1 df) and a percentile bootstrap 95% CI
  - paired Wilcoxon signed-rank p and paired-t p (alpha-dynamic vs baseline,
    beta-dynamic vs baseline)

It then reports the K-trend of the per-K-mean delta_alpha: Spearman and Pearson
of (delta_alpha vs K).  Pure analysis, no training; reads and writes only the
gitignored outputs/ directory.  Does not modify any runner.

Usage
-----
  source ~/anaconda3/etc/profile.d/conda.sh && conda activate research
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=".;rl/src;ma-irt" \
      python deep_irt/bench/run_alpha_beta_asymmetry_stats.py \
          [--in deep_irt/bench/outputs/alpha_beta_asymmetry.json] \
          [--metric spearman|pearson] [--n-boot 10000] [--seed 0]

Outputs:
  deep_irt/bench/outputs/alpha_beta_asymmetry_stats.json
  deep_irt/bench/outputs/alpha_beta_asymmetry_stats.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "outputs"


def _ci_t(x):
    """Seed-level 95% CI for the mean (t-distribution, n-1 df)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return (float("nan"), float("nan"))
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    half = stats.t.ppf(0.975, n - 1) * se
    return (float(m - half), float(m + half))


def _ci_boot(x, n_boot, rng):
    """Percentile bootstrap 95% CI for the mean of a paired-delta vector."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(n_boot, n))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


def _paired_tests(dyn, base):
    """Paired Wilcoxon signed-rank and paired t-test, dyn vs base."""
    dyn = np.asarray(dyn, dtype=float)
    base = np.asarray(base, dtype=float)
    mask = np.isfinite(dyn) & np.isfinite(base)
    dyn, base = dyn[mask], base[mask]
    out = {"wilcoxon_p": float("nan"), "ttest_p": float("nan"), "n": int(dyn.size)}
    if dyn.size < 2:
        return out
    diff = dyn - base
    # Wilcoxon needs at least one non-zero difference.
    if np.any(diff != 0):
        try:
            out["wilcoxon_p"] = float(stats.wilcoxon(dyn, base).pvalue)
        except ValueError:
            out["wilcoxon_p"] = float("nan")
    try:
        out["ttest_p"] = float(stats.ttest_rel(dyn, base).pvalue)
    except ValueError:
        out["ttest_p"] = float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",
                    default=str(OUT / "alpha_beta_asymmetry.json"))
    ap.add_argument("--metric", choices=["spearman", "pearson"],
                    default="spearman",
                    help="recovery metric for the deltas (default: spearman, "
                         "the sign-aligned RANK recovery the RQ1 claim is about)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    blob = json.loads(Path(args.inp).read_text(encoding="utf-8"))
    meta = blob["meta"]
    rows = blob["rows"]
    a_key = f"a_{args.metric}"
    b_key = f"b_{args.metric}"

    Ks = list(meta["K"])
    seeds = list(meta["seeds"])
    rng = np.random.default_rng(args.seed)

    # index rows by (K, seed, arm)
    idx = {(r["K"], r["seed"], r["arm"]): r for r in rows}

    per_k = []
    da_means_by_k = []   # for the K-trend (per-K mean delta_alpha)
    db_means_by_k = []
    for K in Ks:
        da, db = [], []          # paired deltas per seed
        a_base, a_dyn = [], []    # for the paired test on alpha
        b_base, b_dyn = [], []    # for the paired test on beta
        for s in seeds:
            rb = idx.get((K, s, "baseline"))
            ra = idx.get((K, s, "alpha-dynamic"))
            rbe = idx.get((K, s, "beta-dynamic"))
            if rb is None or ra is None or rbe is None:
                continue
            da.append(ra[a_key] - rb[a_key])
            db.append(rbe[b_key] - rb[b_key])
            a_base.append(rb[a_key]); a_dyn.append(ra[a_key])
            b_base.append(rb[b_key]); b_dyn.append(rbe[b_key])

        da = np.array(da, dtype=float)
        db = np.array(db, dtype=float)
        mean_da = float(np.nanmean(da))
        mean_db = float(np.nanmean(db))
        da_means_by_k.append(mean_da)
        db_means_by_k.append(mean_db)

        entry = {
            "K": K,
            "n_seeds": int(np.isfinite(da).sum()),
            "mean_delta_alpha": mean_da,
            "sd_delta_alpha": float(np.nanstd(da, ddof=1)) if np.isfinite(da).sum() > 1 else float("nan"),
            "ci_t_alpha": _ci_t(da),
            "ci_boot_alpha": _ci_boot(da, args.n_boot, rng),
            "alpha_test": _paired_tests(a_dyn, a_base),
            "mean_delta_beta": mean_db,
            "sd_delta_beta": float(np.nanstd(db, ddof=1)) if np.isfinite(db).sum() > 1 else float("nan"),
            "ci_t_beta": _ci_t(db),
            "ci_boot_beta": _ci_boot(db, args.n_boot, rng),
            "beta_test": _paired_tests(b_dyn, b_base),
        }
        per_k.append(entry)

    # K-trend of per-K-mean delta_alpha (and delta_beta as a negative control).
    Karr = np.array(Ks, dtype=float)
    da_arr = np.array(da_means_by_k, dtype=float)
    db_arr = np.array(db_means_by_k, dtype=float)

    def _trend(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 2:
            return {"pearson_r": float("nan"), "pearson_p": float("nan"),
                    "spearman_r": float("nan"), "spearman_p": float("nan")}
        pr = stats.pearsonr(x[m], y[m])
        sr = stats.spearmanr(x[m], y[m])
        return {"pearson_r": float(pr.statistic), "pearson_p": float(pr.pvalue),
                "spearman_r": float(sr.correlation), "spearman_p": float(sr.pvalue)}

    trend_alpha = _trend(Karr, da_arr)
    trend_beta = _trend(Karr, db_arr)

    # Pooled across all (K, seed) pairs: trend on the seed-level deltas vs K.
    pooled_K, pooled_da, pooled_db = [], [], []
    for K in Ks:
        for s in seeds:
            rb = idx.get((K, s, "baseline"))
            ra = idx.get((K, s, "alpha-dynamic"))
            rbe = idx.get((K, s, "beta-dynamic"))
            if rb is None or ra is None or rbe is None:
                continue
            pooled_K.append(K)
            pooled_da.append(ra[a_key] - rb[a_key])
            pooled_db.append(rbe[b_key] - rb[b_key])
    trend_alpha_pooled = _trend(np.array(pooled_K, float),
                                np.array(pooled_da, float))
    trend_beta_pooled = _trend(np.array(pooled_K, float),
                               np.array(pooled_db, float))

    result = {
        "meta": {**meta, "metric": args.metric, "n_boot": args.n_boot,
                 "source": str(Path(args.inp).name)},
        "per_K": per_k,
        "trend_delta_alpha_vs_K_perKmean": trend_alpha,
        "trend_delta_beta_vs_K_perKmean": trend_beta,
        "trend_delta_alpha_vs_K_pooled": trend_alpha_pooled,
        "trend_delta_beta_vs_K_pooled": trend_beta_pooled,
    }

    (OUT / "alpha_beta_asymmetry_stats.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")

    # ---- markdown render ----
    L = []
    L.append("# RQ1 paired statistics -- state-conditioned (dynamic) vs static alpha\n")
    L.append(f"source={Path(args.inp).name}  metric={args.metric} (sign-aligned)  "
             f"seeds={len(seeds)} {seeds}  K={Ks}  "
             f"epochs={meta.get('epochs')}  N={meta.get('n_learners')}  "
             f"Q={meta.get('n_items')}  T={meta.get('seq_len')}  "
             f"boot={args.n_boot}\n")
    L.append("delta_alpha = alpha-dynamic - baseline (the effect); "
             "delta_beta = beta-dynamic - baseline (the negative control). "
             "Paired PER SEED. CI_t = seed-level t 95% CI; CI_boot = percentile "
             "bootstrap 95% CI. p = paired Wilcoxon signed-rank (and paired-t).\n")
    L.append("| K | n | mean d_alpha | 95% CI_t | 95% CI_boot | "
             "Wilcoxon p | t p | mean d_beta | 95% CI_t (beta) | "
             "Wilcoxon p (beta) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for e in per_k:
        ca = e["ci_t_alpha"]; cab = e["ci_boot_alpha"]; cb = e["ci_t_beta"]
        L.append(
            f"| {e['K']} | {e['n_seeds']} | {e['mean_delta_alpha']:+.4f} | "
            f"[{ca[0]:+.4f}, {ca[1]:+.4f}] | [{cab[0]:+.4f}, {cab[1]:+.4f}] | "
            f"{e['alpha_test']['wilcoxon_p']:.4f} | {e['alpha_test']['ttest_p']:.4f} | "
            f"{e['mean_delta_beta']:+.4f} | [{cb[0]:+.4f}, {cb[1]:+.4f}] | "
            f"{e['beta_test']['wilcoxon_p']:.4f} |")

    L.append("\n## K-trend of delta_alpha\n")
    L.append("Per-K-mean delta_alpha vs K (5 points):")
    L.append(f"  Pearson  r={trend_alpha['pearson_r']:+.4f}  p={trend_alpha['pearson_p']:.4f}")
    L.append(f"  Spearman r={trend_alpha['spearman_r']:+.4f}  p={trend_alpha['spearman_p']:.4f}")
    L.append("Pooled seed-level delta_alpha vs K (all pairs):")
    L.append(f"  Pearson  r={trend_alpha_pooled['pearson_r']:+.4f}  p={trend_alpha_pooled['pearson_p']:.4f}")
    L.append(f"  Spearman r={trend_alpha_pooled['spearman_r']:+.4f}  p={trend_alpha_pooled['spearman_p']:.4f}")
    L.append("\nNegative control, delta_beta vs K (per-K-mean):")
    L.append(f"  Pearson  r={trend_beta['pearson_r']:+.4f}  p={trend_beta['pearson_p']:.4f}")
    L.append(f"  Spearman r={trend_beta['spearman_r']:+.4f}  p={trend_beta['spearman_p']:.4f}")

    md = "\n".join(L)
    (OUT / "alpha_beta_asymmetry_stats.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\nwrote {OUT/'alpha_beta_asymmetry_stats.json'} and "
          f"{OUT/'alpha_beta_asymmetry_stats.md'}")


if __name__ == "__main__":
    main()
