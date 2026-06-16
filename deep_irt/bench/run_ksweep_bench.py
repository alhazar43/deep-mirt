"""run_ksweep_bench.py -- K-sweep: does the decoupling advantage grow with K?

Hypothesis (from docs/learning_dynamics_toy.md Sec 11.5): the decoupling
advantage on alpha-Spearman grows monotonically with the number of response
categories K, tracking the GPCM Fisher stiffness ratio I(theta)/I(alpha),
which climbs 1.03 (K=2) -> 2.29 (K=4) under the data-gen priors.

For each K in {2,3,4,5,6} we generate a static GPCM dataset and train two
LSTM/GPCM models:

  SHARED    -- alpha reads the shared narrow item embedding (alpha_emb_dim=None)
  DECOUPLED -- alpha reads its own wide item table (alpha_emb_dim=64)

Both use state_alpha=True, alpha_log_scale=1.0, emb_dim=8, hidden_dim=32
(the cheap theta-encoder held fixed).  The headline metric is

  delta_K = a_spearman(DECOUPLED) - a_spearman(SHARED),

averaged over seeds.  The Fisher stiffness ratio I(theta)/I(alpha) is
computed analytically (Monte Carlo over the data-gen priors) and reported
next to delta_K so the correlation is visible.

Usage
-----
  source ~/anaconda3/etc/profile.d/conda.sh && conda activate research
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=".;rl/src;ma-irt" \\
      python deep_irt/bench/run_ksweep_bench.py [--quick] [--device cuda]

Quick mode: K in {2,4,6,9,11}, N=200, Q=30, T=30, 2 seeds, 40 epochs.
Full mode:  K in {2,3,4,5,6,7,8,9,10,11}, N=800, Q=60, T=60, 10 seeds, 150 epochs.

Outputs
-------
  deep_irt/bench/outputs/ksweep_results.json
  deep_irt/bench/outputs/ksweep_table.md
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from deep_irt.bench.datagen import BenchDataConfig, generate
from deep_irt.bench.engines import DeepIRTEngine
from deep_irt.bench.run_alpha_fix_bench import run_cell, aggregate, _ms

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "outputs"
OUT.mkdir(exist_ok=True)

# Fixed theta-encoder capacity (identical for SHARED and DECOUPLED)
_CHEAP = dict(emb_dim=8, hidden_dim=32)
_ALPHA_EMB_DIM = 64      # decoupled wide alpha key
_ALPHA_LOG_SCALE = 1.0   # exp(raw), ma-irt's link


# ---------------------------------------------------------------------------
# GPCM Fisher stiffness ratio I(theta)/I(alpha) by Monte Carlo
# ---------------------------------------------------------------------------

def _gpcm_category_probs(theta: float, alpha: float,
                          b: np.ndarray) -> np.ndarray:
    """GPCM category probabilities for a single (person, item) pair.

    b : (K-1,) sorted step thresholds.
    """
    K = len(b) + 1
    psi = np.zeros(K)
    for k in range(1, K):
        psi[k] = alpha * (k * theta - np.sum(b[:k]))
    psi -= psi.max()
    e = np.exp(psi)
    return e / e.sum()


def _gpcm_fisher_per_response(theta: float, alpha: float,
                               b: np.ndarray) -> tuple[float, float]:
    """Fisher information for theta and alpha for ONE (person, item) response.

    GPCM Fisher formulas (from docs/learning_dynamics_toy.md Sec 11.5):
      I(theta) = alpha^2 * Var_p(k)              k = 0..K-1 category index
      I(alpha) = Var_p(k * theta - B_k)          natural-stat variance
                 where B_k = sum_{c<=k} b_c

    p : category distribution, k : index variable.
    """
    K = len(b) + 1
    p = _gpcm_category_probs(theta, alpha, b)
    k_vec = np.arange(K, dtype=float)

    # Cumulative threshold sums B_k = sum_{c=1..k} b_c  (B_0 = 0)
    B = np.zeros(K)
    for k in range(1, K):
        B[k] = np.sum(b[:k])

    nat_stat = k_vec * theta - B          # k*theta - B_k, shape (K,)

    var_k = float(np.sum(p * k_vec**2) - np.sum(p * k_vec)**2)
    var_nat = float(np.sum(p * nat_stat**2) - np.sum(p * nat_stat)**2)

    i_theta = alpha**2 * var_k
    i_alpha = var_nat
    return i_theta, i_alpha


def compute_stiffness(K: int, n_mc: int = 50_000, seed: int = 42) -> float:
    """Population-mean I(theta)/I(alpha) for GPCM under the data-gen priors.

    Priors (matching datagen.py generate()):
      alpha ~ LogNormal(0, 0.3)
      theta ~ N(0, 1)
      b     ~ sorted N(0, 1) of length K-1 per item
    """
    rng = np.random.default_rng(seed)
    alphas = rng.lognormal(mean=0.0, sigma=0.3, size=n_mc)
    thetas = rng.standard_normal(n_mc)
    # Each b is (K-1,) sorted; draw and sort per MC sample
    b_raw = rng.standard_normal((n_mc, K - 1))
    bs = np.sort(b_raw, axis=1)

    i_th_sum = 0.0
    i_al_sum = 0.0
    for idx in range(n_mc):
        it, ia = _gpcm_fisher_per_response(
            thetas[idx], alphas[idx], bs[idx])
        i_th_sum += it
        i_al_sum += ia

    i_th_mean = i_th_sum / n_mc
    i_al_mean = i_al_sum / n_mc
    return i_th_mean / i_al_mean if i_al_mean > 0 else float("nan")


# ---------------------------------------------------------------------------
# Engine builder: SHARED and DECOUPLED at a given K
# ---------------------------------------------------------------------------

def build_engines_for_K(ds, device: str, seed: int):
    """Return (variant_key, engine) pairs for SHARED and DECOUPLED."""
    common = dict(decoder="gpcm", state_alpha=True,
                  alpha_log_scale=_ALPHA_LOG_SCALE,
                  device=device, seed=seed, **_CHEAP)
    return [
        ("shared",
         DeepIRTEngine(ds, alpha_emb_dim=None, **common)),
        ("decoupled",
         DeepIRTEngine(ds, alpha_emb_dim=_ALPHA_EMB_DIM, **common)),
    ]


# ---------------------------------------------------------------------------
# Table and verdict rendering
# ---------------------------------------------------------------------------

def render_table(agg, stiffness, K_vals, device, epochs, N, Q, T, seeds) -> str:
    L = []
    L.append("# K-sweep bench -- does the decoupling advantage grow with K?\n")
    L.append(
        f"device={device}  epochs={epochs}  N={N}  Q={Q}  T={T}  "
        f"seeds={seeds}  (torch {torch.__version__})\n"
    )
    L.append(
        "Both models: LSTM/GPCM, state_alpha=True, alpha_log_scale=1.0, "
        "emb_dim=8, hidden_dim=32.\n"
        "SHARED: alpha reads the shared narrow item_emb (alpha_emb_dim=None).\n"
        "DECOUPLED: alpha reads its own wide table (alpha_emb_dim=64).\n"
        "Metric: alpha Spearman (a_spearman); delta_K = decoupled - shared.\n"
        "Stiffness = population-mean I(theta)/I(alpha) from Monte Carlo over "
        "LogNormal(0,0.3) alpha, N(0,1) theta, sorted N(0,1) beta "
        "(50k samples, seed 42).\n"
    )

    # Main K-sweep table
    L.append("## K-sweep (static GPCM only -- a_spearman is the load-bearing metric)\n")
    L.append(
        "| K | shared a_sp | decoupled a_sp | delta_K (std) | "
        "shared theta_st | decoup theta_st | "
        "shared params | decoup params | stiffness I(th)/I(al) |"
    )
    L.append("|---|---|---|---|---|---|---|---|---|")

    delta_vals = []
    delta_stds = []
    for K in K_vals:
        sh = agg.get(("shared", f"static_k{K}"))
        de = agg.get(("decoupled", f"static_k{K}"))
        if sh is None or de is None:
            continue
        sh_a = sh["a_spearman_mean"]
        de_a = de["a_spearman_mean"]
        sh_a_std = sh["a_spearman_std"]
        de_a_std = de["a_spearman_std"]
        delta = de_a - sh_a
        # Propagate std conservatively (sum in quadrature)
        delta_std = float(np.sqrt(sh_a_std**2 + de_a_std**2))
        delta_vals.append(delta)
        delta_stds.append(delta_std)

        sh_th = sh["theta_spearman_mean"]
        de_th = de["theta_spearman_mean"]
        sh_p = int(sh["n_params_mean"])
        de_p = int(de["n_params_mean"])
        stiff = stiffness.get(K, float("nan"))

        L.append(
            f"| {K} | {sh_a:.3f}+-{sh_a_std:.3f} | "
            f"{de_a:.3f}+-{de_a_std:.3f} | "
            f"{delta:+.3f} ({delta_std:.3f}) | "
            f"{sh_th:.3f} | {de_th:.3f} | "
            f"{sh_p} | {de_p} | {stiff:.4f} |"
        )

    L.append("")
    L.append(_verdict(agg, stiffness, K_vals, delta_vals, delta_stds))
    return "\n".join(L)


def _verdict(agg, stiffness, K_vals, delta_vals, delta_stds) -> str:
    L = ["## Verdict\n"]

    # Per-K sign consistency: fraction of seeds where decoupled > shared
    L.append("### Per-K sign consistency (fraction of seeds with delta > 0)\n")
    L.append("| K | n_seeds | seeds+ | frac+ |")
    L.append("|---|---|---|---|")

    sign_fracs = []
    for K in K_vals:
        # Re-derive per-seed deltas from the raw rows stored in agg
        sh = agg.get(("shared", f"static_k{K}"))
        de = agg.get(("decoupled", f"static_k{K}"))
        if sh is None or de is None:
            sign_fracs.append(float("nan"))
            continue
        n_s = sh.get("n_seeds", 0)
        # We cannot access per-seed rows from the agg dict directly; we stored
        # them under a "per_seed_delta" key added in main().
        per_seed = agg.get(("delta_seeds", f"static_k{K}"), [])
        if per_seed:
            n_pos = sum(1 for d in per_seed if d > 0)
            frac = n_pos / len(per_seed)
            sign_fracs.append(frac)
            L.append(f"| {K} | {len(per_seed)} | {n_pos} | {frac:.2f} |")
        else:
            sign_fracs.append(float("nan"))
            L.append(f"| {K} | {n_s} | - | - |")

    L.append("")

    # Stiffness column next to delta_K
    L.append("### delta_K and stiffness side by side\n")
    L.append("| K | stiffness I(th)/I(al) | delta_K (mean) |")
    L.append("|---|---|---|")
    for K, delta in zip(K_vals, delta_vals):
        stiff = stiffness.get(K, float("nan"))
        L.append(f"| {K} | {stiff:.4f} | {delta:+.4f} |")
    L.append("")

    # Monotonicity check
    if len(delta_vals) >= 2:
        is_mono = all(
            delta_vals[i + 1] >= delta_vals[i] - 0.005  # tolerance for noise
            for i in range(len(delta_vals) - 1)
        )
        strictly_mono = all(
            delta_vals[i + 1] > delta_vals[i]
            for i in range(len(delta_vals) - 1)
        )
        # Check stiffness monotonicity (should be trivially true from the toy)
        stiff_vals = [stiffness.get(K, float("nan")) for K in K_vals]
        stiff_mono = all(
            stiff_vals[i + 1] > stiff_vals[i]
            for i in range(len(stiff_vals) - 1)
            if not (stiff_vals[i] != stiff_vals[i] or
                    stiff_vals[i + 1] != stiff_vals[i + 1])
        )

        L.append("### Monotonicity assessment\n")
        if strictly_mono:
            L.append(
                "MONOTONICALLY INCREASING (strict): delta_K rises at every K step. "
                "The decoupling advantage grows monotonically with the number of "
                "response categories."
            )
        elif is_mono:
            L.append(
                "MONOTONICALLY NON-DECREASING (with tolerance 0.005): delta_K does "
                "not materially fall at any K step, though it may plateau. "
                "Consistent with the stiffness hypothesis."
            )
        else:
            non_mono_pairs = [
                f"K={K_vals[i]}->{K_vals[i+1]} ({delta_vals[i]:+.3f}->{delta_vals[i+1]:+.3f})"
                for i in range(len(delta_vals) - 1)
                if delta_vals[i + 1] < delta_vals[i] - 0.005
            ]
            L.append(
                "NOT MONOTONE: delta_K decreases at " +
                ", ".join(non_mono_pairs) + ". "
                "The stiffness hypothesis is not supported, or seed noise dominates."
            )

        if stiff_mono:
            L.append("Stiffness I(theta)/I(alpha) is monotone in K (confirmed).")

        # Correlation between stiffness and delta_K
        valid = [(s, d) for s, d in zip(stiff_vals, delta_vals)
                 if s == s and d == d]
        if len(valid) >= 3:
            s_arr = np.array([v[0] for v in valid])
            d_arr = np.array([v[1] for v in valid])
            from scipy import stats as sp_stats
            corr_sp = float(sp_stats.spearmanr(s_arr, d_arr).correlation)
            corr_pe = float(sp_stats.pearsonr(s_arr, d_arr)[0])
            L.append(
                f"\nCorrelation between stiffness and delta_K: "
                f"Spearman={corr_sp:.3f}, Pearson={corr_pe:.3f}."
            )
            if corr_sp >= 0.8:
                L.append(
                    "TRACKS STIFFNESS: the decoupling advantage correlates "
                    "strongly with the Fisher stiffness ratio I(theta)/I(alpha)."
                )
            elif corr_sp >= 0.4:
                L.append(
                    "WEAK TRACKING: moderate correlation with stiffness; "
                    "seed noise or small K range may be limiting."
                )
            else:
                L.append(
                    "DOES NOT TRACK STIFFNESS: the decoupling advantage does not "
                    "follow the stiffness ratio. The hypothesis is not supported."
                )
    else:
        L.append("(Too few K points to assess monotonicity.)")

    return "\n".join(L)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Fast smoke-test: K in {2,4,6}, N=200, Q=30, T=30, "
                         "2 seeds, 40 epochs.")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if args.quick:
        K_vals = [2, 4, 6, 9, 11]
        N, Q, T = 200, 30, 30
        epochs = args.epochs or 40
        seeds = args.seeds or [0, 1]
    else:
        K_vals = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        N, Q, T = 800, 60, 60
        epochs = args.epochs or 150
        seeds = args.seeds or list(range(10))

    mode = "QUICK" if args.quick else "FULL"
    print(
        f"=== ksweep bench [{mode}]: device={device}  K={K_vals}  "
        f"epochs={epochs}  N={N}  Q={Q}  T={T}  seeds={seeds} ==="
    )

    # ------------------------------------------------------------------
    # Pre-compute analytical stiffness ratios (fast, no model needed)
    # ------------------------------------------------------------------
    print("\n--- Computing GPCM Fisher stiffness ratios (Monte Carlo, 50k) ---")
    stiffness = {}
    for K in K_vals:
        stiffness[K] = compute_stiffness(K, n_mc=50_000, seed=42)
        print(f"  K={K}: I(theta)/I(alpha) = {stiffness[K]:.4f}")

    # ------------------------------------------------------------------
    # Sweep K x seeds x variants
    # ------------------------------------------------------------------
    t_start = time.time()
    rows = []

    for K in K_vals:
        dsname = f"static_k{K}"
        print(f"\n=== K={K} (dataset: {dsname}) ===")
        for seed in seeds:
            cfg = BenchDataConfig(
                name=dsname, kind="static", n_cats=K,
                n_learners=N, n_items=Q, seq_len=T, seed=seed
            )
            ds = generate(cfg)
            print(f"  --- seed {seed} ---")
            for variant_key, eng in build_engines_for_K(ds, device, seed):
                eng.fit(n_epochs=epochs)
                cell = run_cell(eng, ds)
                cell["variant_key"] = variant_key
                cell["seed"] = seed
                cell["K"] = K
                rows.append(cell)
                print(
                    f"    {variant_key:10s} K={K}  "
                    f"theta_static={cell['theta_spearman']:.3f}  "
                    f"a={cell['a_spearman']:.3f}  "
                    f"b={cell['b_spearman']:.3f}  "
                    f"p={cell['n_params']}  t={cell['train_time']:.1f}s"
                )

    # ------------------------------------------------------------------
    # Aggregate and augment with per-seed delta for sign-consistency
    # ------------------------------------------------------------------
    agg = aggregate(rows)

    # Compute per-seed deltas and inject under special "delta_seeds" key
    for K in K_vals:
        dsname = f"static_k{K}"
        # Gather per-seed a_spearman for shared and decoupled
        sh_rows = {r["seed"]: r["a_spearman"] for r in rows
                   if r["K"] == K and r["variant_key"] == "shared"}
        de_rows = {r["seed"]: r["a_spearman"] for r in rows
                   if r["K"] == K and r["variant_key"] == "decoupled"}
        common_seeds = sorted(set(sh_rows) & set(de_rows))
        per_seed_delta = [de_rows[s] - sh_rows[s] for s in common_seeds]
        agg[("delta_seeds", dsname)] = per_seed_delta

    # ------------------------------------------------------------------
    # Render table + verdict
    # ------------------------------------------------------------------
    # Collect delta_vals in K order for the verdict
    delta_vals = []
    delta_stds = []
    for K in K_vals:
        sh = agg.get(("shared", f"static_k{K}"))
        de = agg.get(("decoupled", f"static_k{K}"))
        if sh is None or de is None:
            delta_vals.append(float("nan"))
            delta_stds.append(float("nan"))
            continue
        delta = de["a_spearman_mean"] - sh["a_spearman_mean"]
        delta_std = float(np.sqrt(sh["a_spearman_std"]**2 +
                                  de["a_spearman_std"]**2))
        delta_vals.append(delta)
        delta_stds.append(delta_std)

    table = render_table(
        agg, stiffness, K_vals, device, epochs, N, Q, T, seeds
    )

    # ------------------------------------------------------------------
    # Serialise (drop the synthetic "delta_seeds" agg entries from JSON)
    # ------------------------------------------------------------------
    agg_serialisable = {
        str(k): v for k, v in agg.items()
        if not (isinstance(k, tuple) and k[0] == "delta_seeds")
    }
    delta_seeds_map = {
        str(K): agg.get(("delta_seeds", f"static_k{K}"), [])
        for K in K_vals
    }
    blob = {
        "meta": {
            "mode": mode,
            "device": device,
            "epochs": epochs,
            "N": N, "Q": Q, "T": T,
            "seeds": seeds,
            "K_vals": K_vals,
            "cheap": _CHEAP,
            "alpha_emb_dim": _ALPHA_EMB_DIM,
            "alpha_log_scale": _ALPHA_LOG_SCALE,
        },
        "stiffness": stiffness,
        "delta_vals": dict(zip(K_vals, delta_vals)),
        "delta_stds": dict(zip(K_vals, delta_stds)),
        "delta_seeds_per_K": delta_seeds_map,
        "rows": rows,
        "agg": agg_serialisable,
    }
    with (OUT / "ksweep_results.json").open("w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)

    (OUT / "ksweep_table.md").write_text(table, encoding="utf-8")
    print("\n" + table)
    print(f"\ntotal wall {time.time() - t_start:.1f}s")
    print(f"wrote {OUT / 'ksweep_results.json'} and {OUT / 'ksweep_table.md'}")


if __name__ == "__main__":
    main()
