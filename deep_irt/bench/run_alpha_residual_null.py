"""run_alpha_residual_null.py -- E8 null control for contextual alpha residuals.

The E8 plan defines the state-conditioned discrimination decomposition as

    log alpha_hat[j,t] = a_j + delta[j,t]
    E_t[delta[j,t] | j] = 0

On static-alpha synthetic data, the true contextual residual is exactly zero.
Any fitted residual structure is therefore a null artifact that must be recorded
before interpreting planted or real-data residuals.

This runner trains the standard state-conditioned alpha model on static GPCM
data, extracts per-occurrence alpha, item-demeans log alpha, and correlates the
residual with the variables named in the plan: theta, theta-beta gap, local
information/uncertainty, history position, previous response, and exposure.

Outputs
-------
  deep_irt/bench/outputs/alpha_residual_null_results.json
  deep_irt/bench/outputs/alpha_residual_null_table.md
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deep_irt.bench.datagen import BenchDataConfig, generate
from deep_irt.bench.engines import DeepIRTEngine

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "outputs"
OUT.mkdir(exist_ok=True)

_DEFAULT_K = [4]
_DEFAULT_NS = [200, 800, 3200]
_DEFAULT_SEEDS = [0, 1, 2]
_QUICK_NS = [80]
_QUICK_SEEDS = [0]


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average-rank transform with NaNs left as NaN."""
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan, dtype=float)
    m = np.isfinite(x)
    vals = x[m]
    if vals.size == 0:
        return out
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(vals.size, dtype=float)
    i = 0
    while i < vals.size:
        j = i + 1
        while j < vals.size and vals[order[j]] == vals[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    out[m] = ranks
    return out


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    x = x[m] - np.mean(x[m])
    y = y[m] - np.mean(y[m])
    denom = math.sqrt(float(np.sum(x * x) * np.sum(y * y)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(x * y) / denom)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


def _ols_r2(y: np.ndarray, cols: list[np.ndarray]) -> float:
    y = np.asarray(y, dtype=float).ravel()
    X = [np.asarray(c, dtype=float).ravel() for c in cols]
    m = np.isfinite(y)
    for c in X:
        m &= np.isfinite(c)
    if m.sum() < len(X) + 3:
        return float("nan")
    yy = y[m]
    if np.var(yy) <= 1e-12:
        return float("nan")
    XX = np.column_stack([np.ones(m.sum())] + [c[m] for c in X])
    coef, *_ = np.linalg.lstsq(XX, yy, rcond=None)
    pred = XX @ coef
    return float(1.0 - np.var(yy - pred) / np.var(yy))


def item_demean(values: np.ndarray, items: np.ndarray, n_items: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return item means, residuals, and item counts for flat occurrence arrays."""
    values = np.asarray(values, dtype=float).ravel()
    items = np.asarray(items, dtype=np.int64).ravel()
    sums = np.zeros(n_items, dtype=float)
    counts = np.zeros(n_items, dtype=float)
    np.add.at(sums, items, values)
    np.add.at(counts, items, 1.0)
    means = sums / np.maximum(counts, 1.0)
    residual = values - means[items]
    return means, residual, counts


def exposure_index(items: np.ndarray, n_items: int) -> np.ndarray:
    """Zero-based prior exposure count for each flat occurrence."""
    flat = np.asarray(items, dtype=np.int64).ravel()
    seen = np.zeros(n_items, dtype=np.int64)
    out = np.zeros(flat.size, dtype=float)
    for k, item in enumerate(flat):
        out[k] = seen[item]
        seen[item] += 1
    return out


def gpcm_probs(theta: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Vectorized GPCM probabilities for occurrence-level parameters."""
    theta = np.asarray(theta, dtype=float).reshape(-1, 1)
    alpha = np.asarray(alpha, dtype=float).reshape(-1, 1)
    beta = np.asarray(beta, dtype=float)
    if beta.ndim == 1:
        beta = beta.reshape(-1, 1)
    psi_pos = np.cumsum(alpha * (theta - beta), axis=1)
    logits = np.concatenate([np.zeros((theta.shape[0], 1)), psi_pos], axis=1)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(logits)
    return e / np.sum(e, axis=1, keepdims=True)


def category_variance(probs: np.ndarray) -> np.ndarray:
    """Var(Y) under categorical probabilities over scores 0..K-1."""
    probs = np.asarray(probs, dtype=float)
    scores = np.arange(probs.shape[1], dtype=float)
    mean = probs @ scores
    mean2 = probs @ (scores * scores)
    return mean2 - mean * mean


@torch.no_grad()
def extract_occurrence_params(model, items, responses, device: str) -> dict[str, np.ndarray]:
    """Read occurrence-level state-conditioned alpha, beta, and model theta."""
    enc, dec = model.encoder, model.decoder
    it = torch.as_tensor(items, dtype=torch.long, device=device)
    rp = torch.as_tensor(responses, dtype=torch.long, device=device)
    theta, state = enc.aligned_theta_and_state(it, rp)
    n, t = it.shape
    h = state.shape[-1]
    val = enc.item_val_emb(it)
    key = enc.item_key_emb(it)
    params = dec.item_params(
        val.reshape(n * t, -1),
        state=state.reshape(n * t, h),
        item_key=key.reshape(n * t, -1),
    )
    return {
        "alpha": params["alpha"].reshape(n, t).cpu().numpy(),
        "beta": params["beta"].reshape(n, t, -1).cpu().numpy(),
        "theta_model": theta.reshape(n, t).cpu().numpy(),
    }


def residual_diagnostics(ds, occ: dict[str, np.ndarray]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute null residual metrics for one fitted model."""
    items = np.asarray(ds.items0, dtype=np.int64)
    responses = np.asarray(ds.responses, dtype=float)
    n, t = items.shape
    q = ds.cfg.n_items
    k = ds.cfg.n_cats
    flat_items = items.ravel()

    alpha = np.asarray(occ["alpha"], dtype=float)
    beta = np.asarray(occ["beta"], dtype=float)
    theta_model = np.asarray(occ["theta_model"], dtype=float)
    log_alpha = np.log(np.clip(alpha.ravel(), 1e-8, None))
    log_alpha_item, delta, counts = item_demean(log_alpha, flat_items, q)

    true_log_alpha = np.log(np.clip(ds.gt.a, 1e-8, None))
    static_spearman = _spearman(log_alpha_item, true_log_alpha)
    static_pearson = _pearson(log_alpha_item, true_log_alpha)

    true_theta = np.asarray(ds.theta_at_step, dtype=float).ravel()
    model_theta = theta_model.ravel()
    true_beta_mean = ds.gt.b[flat_items].mean(axis=1)
    model_beta_mean = beta.reshape(n * t, k - 1).mean(axis=1)

    true_prob = gpcm_probs(true_theta, ds.gt.a[flat_items], ds.gt.b[flat_items])
    model_prob = gpcm_probs(model_theta, alpha.ravel(), beta.reshape(n * t, k - 1))
    true_info = category_variance(true_prob)
    model_info = category_variance(model_prob)

    hist_pos = np.tile(np.arange(t, dtype=float), n)
    prev_response = np.full((n, t), np.nan, dtype=float)
    prev_response[:, 1:] = responses[:, :-1]
    prev_response = prev_response.ravel()
    exposure = exposure_index(flat_items, q)

    variables = {
        "theta_true": true_theta,
        "theta_model": model_theta,
        "gap_true": true_theta - true_beta_mean,
        "gap_model": model_theta - model_beta_mean,
        "abs_gap_true": np.abs(true_theta - true_beta_mean),
        "abs_gap_model": np.abs(model_theta - model_beta_mean),
        "gap2_true": (true_theta - true_beta_mean) ** 2,
        "gap2_model": (model_theta - model_beta_mean) ** 2,
        "info_true": true_info,
        "info_model": model_info,
        "history_pos": hist_pos,
        "prev_response": prev_response,
        "exposure_count": exposure,
    }

    var_rows = []
    for name, values in variables.items():
        pear = _pearson(delta, values)
        spear = _spearman(delta, values)
        row = {
            "variable": name,
            "pearson": pear,
            "spearman": spear,
            "abs_pearson": abs(pear) if pear == pear else float("nan"),
            "abs_spearman": abs(spear) if spear == spear else float("nan"),
        }
        if name in ("theta_true", "theta_model", "gap_true", "gap_model"):
            row["cubic_r2"] = _ols_r2(delta, [values, values ** 2, values ** 3])
        else:
            row["cubic_r2"] = float("nan")
        var_rows.append(row)

    finite_abs = [
        row["abs_pearson"] for row in var_rows
        if math.isfinite(row["abs_pearson"])
    ]
    finite_r2 = [
        row["cubic_r2"] for row in var_rows
        if math.isfinite(row["cubic_r2"])
    ]
    strongest = max(
        var_rows,
        key=lambda r: r["abs_pearson"] if math.isfinite(r["abs_pearson"]) else -1.0,
    )

    per_item_delta_std = np.zeros(q, dtype=float)
    for item in range(q):
        m = flat_items == item
        per_item_delta_std[item] = float(np.std(delta[m])) if m.any() else float("nan")

    summary = {
        "static_log_alpha_spearman": static_spearman,
        "static_log_alpha_pearson": static_pearson,
        "delta_std": float(np.std(delta)),
        "delta_abs_mean": float(np.mean(np.abs(delta))),
        "item_delta_std_mean": float(np.nanmean(per_item_delta_std)),
        "item_delta_std_p95": float(np.nanpercentile(per_item_delta_std, 95)),
        "strongest_abs_corr": float(max(finite_abs)) if finite_abs else float("nan"),
        "strongest_corr_variable": strongest["variable"],
        "max_cubic_r2": float(max(finite_r2)) if finite_r2 else float("nan"),
        "min_item_count": float(np.min(counts)),
        "median_item_count": float(np.median(counts)),
    }
    return summary, var_rows


def _fit_cell(k: int, n_learners: int, seed: int, args) -> dict[str, Any]:
    cfg = BenchDataConfig(
        name=f"alpha_residual_null_k{k}",
        kind="static",
        n_cats=k,
        n_learners=n_learners,
        n_items=args.n_items,
        seq_len=args.seq_len,
        n_holdout=min(10, args.seq_len),
        seed=seed,
        alpha_theta_slope=0.0,
    )
    ds = generate(cfg)
    eng = DeepIRTEngine(
        ds,
        decoder="gpcm",
        emb_dim=8,
        hidden_dim=32,
        state_alpha=True,
        item_key_dim=64,
        alpha_log_scale=1.0,
        encoder="lstm",
        device=args.device,
        seed=seed,
    )
    eng.fit(n_epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
    occ = extract_occurrence_params(eng.model, ds.items0, ds.responses, args.device)
    summary, variables = residual_diagnostics(ds, occ)
    return {
        "K": k,
        "N": n_learners,
        "seed": seed,
        "dataset": cfg.name,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        **summary,
        "variables": variables,
    }


def _finite_mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr))


def aggregate(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(int(row["K"]), int(row["N"]))].append(row)

    agg = []
    var_agg = []
    metric_keys = [
        "static_log_alpha_spearman",
        "static_log_alpha_pearson",
        "delta_std",
        "delta_abs_mean",
        "item_delta_std_mean",
        "item_delta_std_p95",
        "strongest_abs_corr",
        "max_cubic_r2",
        "min_item_count",
        "median_item_count",
    ]
    for (k, n), rows in sorted(groups.items()):
        entry: dict[str, Any] = {"K": k, "N": n, "n_seeds": len(rows)}
        for key in metric_keys:
            mean, std = _finite_mean_std([r[key] for r in rows])
            entry[f"{key}_mean"] = mean
            entry[f"{key}_std"] = std
        # Most frequent strongest artifact variable across seeds.
        counts = defaultdict(int)
        for r in rows:
            counts[r["strongest_corr_variable"]] += 1
        entry["strongest_corr_variable_mode"] = max(
            counts.items(), key=lambda kv: (kv[1], kv[0])
        )[0]
        agg.append(entry)

        var_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            for v in r["variables"]:
                var_groups[v["variable"]].append(v)
        for variable, vs in sorted(var_groups.items()):
            pear, pear_sd = _finite_mean_std([v["pearson"] for v in vs])
            spear, spear_sd = _finite_mean_std([v["spearman"] for v in vs])
            r2, r2_sd = _finite_mean_std([v["cubic_r2"] for v in vs])
            var_agg.append({
                "K": k,
                "N": n,
                "variable": variable,
                "pearson_mean": pear,
                "pearson_std": pear_sd,
                "spearman_mean": spear,
                "spearman_std": spear_sd,
                "cubic_r2_mean": r2,
                "cubic_r2_std": r2_sd,
            })
    return agg, var_agg


def _fmt(value: Any, p: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    return "-" if not math.isfinite(value) else f"{value:.{p}f}"


def _fmt_ms(row: dict[str, Any], key: str, p: int = 3) -> str:
    mean = row.get(f"{key}_mean", float("nan"))
    std = row.get(f"{key}_std", float("nan"))
    return "-" if not math.isfinite(float(mean)) else f"{mean:.{p}f}+-{std:.{p}f}"


def render_markdown(meta: dict[str, Any], agg: list[dict[str, Any]], var_agg: list[dict[str, Any]]) -> str:
    lines = ["# Alpha Residual Null Control\n"]
    lines.append(
        f"mode={meta['mode']} device={meta['device']} K={meta['K']} "
        f"N={meta['N']} seeds={meta['seeds']} epochs={meta['epochs']} "
        f"Q={meta['Q']} T={meta['T']} lr={meta['lr']}\n"
    )
    lines.append(
        "Static-alpha synthetic data has true contextual residual delta[j,t]=0. "
        "The fitted residual is item-demeaned log alpha, so any nonzero structure "
        "reported here is a null artifact to preserve before real-data analysis.\n"
    )
    lines.append("## Summary\n")
    lines.append(
        "| K | N | a_static_sp | delta std | item delta std | strongest | "
        "max |corr| | max cubic R2 | item count median |"
    )
    lines.append("|---:|---:|---|---|---|---|---:|---:|---:|")
    for row in agg:
        lines.append(
            f"| {row['K']} | {row['N']} | "
            f"{_fmt_ms(row, 'static_log_alpha_spearman')} | "
            f"{_fmt_ms(row, 'delta_std')} | "
            f"{_fmt_ms(row, 'item_delta_std_mean')} | "
            f"{row['strongest_corr_variable_mode']} | "
            f"{_fmt(row['strongest_abs_corr_mean'])} | "
            f"{_fmt(row['max_cubic_r2_mean'])} | "
            f"{_fmt(row['median_item_count_mean'], 1)} |"
        )

    focus = {
        "theta_true",
        "theta_model",
        "gap_true",
        "abs_gap_true",
        "info_true",
        "info_model",
        "history_pos",
        "prev_response",
        "exposure_count",
    }
    lines.append("\n## Variable Correlations\n")
    lines.append("| K | N | variable | Pearson | Spearman | cubic R2 |")
    lines.append("|---:|---:|---|---:|---:|---:|")
    for row in var_agg:
        if row["variable"] not in focus:
            continue
        lines.append(
            f"| {row['K']} | {row['N']} | {row['variable']} | "
            f"{_fmt(row['pearson_mean'])} | {_fmt(row['spearman_mean'])} | "
            f"{_fmt(row['cubic_r2_mean'])} |"
        )

    lines.append("\n## Interpretation Rule\n")
    lines.append(
        "Do not interpret contextual alpha residuals as substantive unless their "
        "structure is larger than this null artifact, survives matched-null "
        "subtraction, or detects a planted signal directionally.  Magnitude "
        "calibration requires separate evidence."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--K", dest="k_vals", type=int, nargs="+", default=None)
    ap.add_argument("--N", dest="n_vals", type=int, nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--n-items", type=int, default=60)
    ap.add_argument("--seq-len", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    if args.quick:
        mode = "QUICK"
        k_vals = args.k_vals or [4]
        n_vals = args.n_vals or _QUICK_NS
        seeds = args.seeds or _QUICK_SEEDS
        args.epochs = args.epochs or 8
    else:
        mode = "FULL"
        k_vals = args.k_vals or _DEFAULT_K
        n_vals = args.n_vals or _DEFAULT_NS
        seeds = args.seeds or _DEFAULT_SEEDS
        args.epochs = args.epochs or 150

    total = len(k_vals) * len(n_vals) * len(seeds)
    print(
        f"=== alpha residual null [{mode}]: device={args.device} K={k_vals} "
        f"N={n_vals} seeds={seeds} epochs={args.epochs} fits={total} ==="
    )
    t0 = time.time()
    records = []
    for k in k_vals:
        for n_learners in n_vals:
            for seed in seeds:
                row = _fit_cell(k, n_learners, seed, args)
                records.append(row)
                print(
                    f"  K={k} N={n_learners} seed={seed}: "
                    f"a_sp={row['static_log_alpha_spearman']:.3f} "
                    f"delta_std={row['delta_std']:.3f} "
                    f"max|corr|={row['strongest_abs_corr']:.3f} "
                    f"strongest={row['strongest_corr_variable']}"
                )

    agg, var_agg = aggregate(records)
    meta = {
        "mode": mode,
        "device": args.device,
        "K": k_vals,
        "N": n_vals,
        "seeds": seeds,
        "epochs": args.epochs,
        "Q": args.n_items,
        "T": args.seq_len,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "elapsed_s": time.time() - t0,
        "torch_version": torch.__version__,
    }
    prefix = args.out_prefix or (
        "alpha_residual_null_quick" if args.quick else "alpha_residual_null"
    )
    out_json = OUT / f"{prefix}_results.json"
    out_md = OUT / f"{prefix}_table.md"
    out_json.write_text(
        json.dumps({"meta": meta, "records": records, "agg": agg, "var_agg": var_agg}, indent=2),
        encoding="utf-8",
    )
    out_md.write_text(render_markdown(meta, agg, var_agg), encoding="utf-8")
    print(f"\nwrote {out_json}")
    print(f"wrote {out_md}")
    print(f"elapsed {meta['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
