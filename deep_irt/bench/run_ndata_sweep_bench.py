"""run_ndata_sweep_bench.py -- N x K sweep: does the decoupling advantage narrow with data?

Context (docs/LEARNING_DYNAMICS_STUDY.md, Phase 3b / K-sweep EXTENDED):
The decoupling advantage on discrimination Spearman is a FINITE-DATA sample-
efficiency effect that vanishes at the population limit (Phase 3b toy, reps->inf).
The empirical K-sweep has only N=800.  This sweep adds the DATA axis:

  For each (N, K), SHARED (alpha_emb_dim=None) vs DECOUPLED (alpha_emb_dim=64).
  delta(N,K) = decoupled a_spearman - shared a_spearman.

Predictions:
  - delta narrows as N grows (finite-data effect).
  - Higher K gaps persist longer (Fisher stiffness is worse, so more data needed
    to wash it out).
  - At N=1600 (2x the canonical run) we are still far from the population limit,
    so delta should remain positive everywhere.

Usage
-----
  source ~/anaconda3/etc/profile.d/conda.sh && conda activate research
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=".;rl/src;ma-irt" \\
      python deep_irt/bench/run_ndata_sweep_bench.py [--quick] [--device cuda]

Quick mode: N in {100,400,1600}, K in {2,6,11}, 2 seeds, 40 epochs.
Full mode:  N in {100,200,400,800,1600}, K in {2..11}, 5 seeds, 150 epochs.

Outputs
-------
  deep_irt/bench/outputs/ndata_sweep_results.json
  deep_irt/bench/outputs/ndata_sweep_table.md
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

# Fixed theta-encoder capacity (identical for SHARED and DECOUPLED at all cells)
_CHEAP = dict(emb_dim=8, hidden_dim=32)
_ALPHA_EMB_DIM = 64
_ALPHA_LOG_SCALE = 1.0
Q = 60   # items
T = 60   # sequence length (fixed, only N varies)


# ---------------------------------------------------------------------------
# Engine builder -- explicit args, no model defaults
# ---------------------------------------------------------------------------

def build_engines(ds, device: str, seed: int):
    """Return (variant_key, engine) pairs for SHARED and DECOUPLED."""
    common = dict(
        decoder="gpcm",
        state_alpha=True,
        alpha_log_scale=_ALPHA_LOG_SCALE,
        device=device,
        seed=seed,
        **_CHEAP,
    )
    return [
        ("shared",
         DeepIRTEngine(ds, alpha_emb_dim=None, **common)),
        ("decoupled",
         DeepIRTEngine(ds, alpha_emb_dim=_ALPHA_EMB_DIM, **common)),
    ]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_table(
    results: dict,
    N_vals: list,
    K_vals: list,
    device: str,
    epochs: int,
    seeds: list,
    mode: str,
) -> str:
    L = []
    L.append("# N x K sweep -- does the decoupling advantage narrow with more data?\n")
    L.append(
        f"mode={mode}  device={device}  epochs={epochs}  "
        f"Q={Q}  T={T}  seeds={seeds}  (torch {torch.__version__})\n"
    )
    L.append(
        "Both models: LSTM/GPCM, state_alpha=True, alpha_log_scale=1.0, "
        "emb_dim=8, hidden_dim=32.  "
        "SHARED: alpha_emb_dim=None.  DECOUPLED: alpha_emb_dim=64.\n"
        "Metric: alpha Spearman.  "
        "delta(N,K) = decoupled - shared (mean over seeds).\n"
        "Prediction: delta narrows as N grows; high-K gaps persist longer.\n"
    )

    # Per-K tables showing N trend
    for K in K_vals:
        L.append(f"\n## K={K}\n")
        L.append(
            "| N | shared a_sp | decoupled a_sp | delta (std) | % reduction vs N=min |"
        )
        L.append("|---|---|---|---|---|")
        delta_at_min = None
        for N in N_vals:
            entry = results.get((K, N))
            if entry is None:
                continue
            sh = entry["shared_mean"]
            de = entry["decoupled_mean"]
            delta = entry["delta_mean"]
            delta_std = entry["delta_std"]
            if delta_at_min is None:
                delta_at_min = delta
                pct = "baseline"
            else:
                if abs(delta_at_min) > 1e-6:
                    pct = f"{100.0 * (delta_at_min - delta) / abs(delta_at_min):+.1f}%"
                else:
                    pct = "n/a"
            sh_std = entry["shared_std"]
            de_std = entry["decoupled_std"]
            L.append(
                f"| {N} | {sh:.3f}+-{sh_std:.3f} | {de:.3f}+-{de_std:.3f} | "
                f"{delta:+.3f} ({delta_std:.3f}) | {pct} |"
            )

    # Summary: delta at N_min vs N_max for each K
    N_min = N_vals[0]
    N_max = N_vals[-1]
    L.append(f"\n## Summary -- delta at N={N_min} vs N={N_max}\n")
    L.append(
        f"| K | delta(N={N_min}) | delta(N={N_max}) | % reduction | narrows? |"
    )
    L.append("|---|---|---|---|---|")
    for K in K_vals:
        e_min = results.get((K, N_min))
        e_max = results.get((K, N_max))
        if e_min is None or e_max is None:
            continue
        d_min = e_min["delta_mean"]
        d_max = e_max["delta_mean"]
        if abs(d_min) > 1e-6:
            pct = 100.0 * (d_min - d_max) / abs(d_min)
            narrows = "YES" if pct > 5.0 else ("FLAT" if abs(pct) <= 5.0 else "GROWS")
        else:
            pct = float("nan")
            narrows = "n/a"
        L.append(
            f"| {K} | {d_min:+.3f} | {d_max:+.3f} | "
            f"{pct:+.1f}% | {narrows} |"
        )

    L.append("")
    L.append(_verdict(results, N_vals, K_vals))
    return "\n".join(L)


def _verdict(results: dict, N_vals: list, K_vals: list) -> str:
    N_min = N_vals[0]
    N_max = N_vals[-1]
    L = ["## Verdict\n"]

    # 1. Does delta narrow with N (overall)?
    narrows_count = 0
    flat_count = 0
    grows_count = 0
    high_K_persist = []
    low_K_wash = []

    for K in K_vals:
        e_min = results.get((K, N_min))
        e_max = results.get((K, N_max))
        if e_min is None or e_max is None:
            continue
        d_min = e_min["delta_mean"]
        d_max = e_max["delta_mean"]
        if abs(d_min) > 1e-6:
            pct = 100.0 * (d_min - d_max) / abs(d_min)
            if pct > 5.0:
                narrows_count += 1
                if K <= 4:
                    low_K_wash.append((K, pct))
            elif pct < -5.0:
                grows_count += 1
            else:
                flat_count += 1

            # check whether high-K persists more than low-K
            if K >= 7:
                high_K_persist.append((K, pct))

    total = narrows_count + flat_count + grows_count
    L.append(f"N={N_min}->{N_max} gap reduction across {total} K cells:")
    L.append(f"  NARROWS (>5%): {narrows_count}/{total}")
    L.append(f"  FLAT (within 5%): {flat_count}/{total}")
    L.append(f"  GROWS (<-5%): {grows_count}/{total}")
    L.append("")

    # 2. Does delta -> 0 as N grows?
    # Check sign at N_max: if all deltas are still positive, we haven't washed out.
    still_positive = [
        K for K in K_vals
        if results.get((K, N_max)) is not None
        and results[(K, N_max)]["delta_mean"] > 0.02
    ]
    if still_positive:
        L.append(
            f"DELTA STILL POSITIVE at N={N_max} for K={still_positive}: "
            "finite-data effect persists well beyond N=800; consistent with "
            "the toy prediction (population limit requires reps -> inf, not just 2x)."
        )
    else:
        L.append(
            f"DELTA approaches zero at N={N_max}: the finite-data effect washes "
            "out within the empirically accessible N range."
        )
    L.append("")

    # 3. High-K vs low-K persistence comparison
    low_K_reduction = np.mean([p for _, p in low_K_wash]) if low_K_wash else float("nan")
    high_K_reduction = np.mean([p for _, p in high_K_persist]) if high_K_persist else float("nan")

    if low_K_wash and high_K_persist:
        if low_K_reduction > high_K_reduction + 5.0:
            L.append(
                f"HIGH-K GAPS PERSIST LONGER: low-K (K<={low_K_wash[-1][0]}) "
                f"reduction {low_K_reduction:.1f}% vs high-K (K>={high_K_persist[0][0]}) "
                f"{high_K_reduction:.1f}%. Consistent with worse Fisher stiffness "
                "requiring more data to overcome."
            )
        elif high_K_reduction > low_K_reduction + 5.0:
            L.append(
                f"LOW-K GAPS PERSIST LONGER (unexpected): low-K reduction "
                f"{low_K_reduction:.1f}% vs high-K {high_K_reduction:.1f}%. "
                "Contradicts Fisher stiffness prediction."
            )
        else:
            L.append(
                f"SIMILAR PERSISTENCE ACROSS K: low-K reduction {low_K_reduction:.1f}% "
                f"vs high-K {high_K_reduction:.1f}%. K-scaling of persistence not clear "
                "at these N values."
            )
    L.append("")

    # 4. Any K where gap does NOT narrow?
    no_narrow = [
        K for K in K_vals
        if results.get((K, N_min)) is not None and results.get((K, N_max)) is not None
        and abs(results[(K, N_min)]["delta_mean"]) > 1e-6
        and (100.0 * (results[(K, N_min)]["delta_mean"] -
                      results[(K, N_max)]["delta_mean"]) /
             abs(results[(K, N_min)]["delta_mean"])) < -5.0
    ]
    if no_narrow:
        L.append(
            f"SURPRISE -- gap GROWS with N at K={no_narrow}: the advantage is "
            "larger at N_max than N_min. Could be a decoupled arm still improving "
            "while shared has more stable recovery, or seed noise."
        )
    else:
        L.append(
            "No K where the gap grows substantially with N (all within noise or narrows)."
        )

    return "\n".join(L)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--quick", action="store_true",
        help="Smoke-test: N in {100,400,1600}, K in {2,6,11}, 2 seeds, 40ep.",
    )
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if args.quick:
        N_vals = [100, 400, 1600]
        K_vals = [2, 6, 11]
        epochs = args.epochs or 40
        seeds = args.seeds or [0, 1]
        mode = "QUICK"
    else:
        N_vals = [100, 200, 400, 800, 1600]
        K_vals = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        epochs = args.epochs or 150
        seeds = args.seeds or list(range(5))
        mode = "FULL"

    print(
        f"=== ndata_sweep bench [{mode}]: device={device}  "
        f"N={N_vals}  K={K_vals}  epochs={epochs}  "
        f"Q={Q}  T={T}  seeds={seeds} ==="
    )
    print(
        f"Total fits: {len(N_vals)} x {len(K_vals)} x 2 configs x {len(seeds)} seeds "
        f"= {len(N_vals)*len(K_vals)*2*len(seeds)}"
    )

    t_start = time.time()
    all_rows = []

    for N in N_vals:
        for K in K_vals:
            dsname = f"static_k{K}_n{N}"
            print(f"\n=== N={N}  K={K}  (dataset: {dsname}) ===")
            for seed in seeds:
                cfg = BenchDataConfig(
                    name=dsname,
                    kind="static",
                    n_cats=K,
                    n_learners=N,
                    n_items=Q,
                    seq_len=T,
                    seed=seed,
                )
                ds = generate(cfg)
                for variant_key, eng in build_engines(ds, device, seed):
                    eng.fit(n_epochs=epochs)
                    cell = run_cell(eng, ds)
                    cell["variant_key"] = variant_key
                    cell["seed"] = seed
                    cell["K"] = K
                    cell["N"] = N
                    all_rows.append(cell)
                    print(
                        f"  {variant_key:10s}  N={N}  K={K}  seed={seed}  "
                        f"a_sp={cell['a_spearman']:.3f}  "
                        f"theta={cell['theta_spearman']:.3f}  "
                        f"b={cell['b_spearman']:.3f}  "
                        f"t={cell['train_time']:.1f}s"
                    )

    # ------------------------------------------------------------------
    # Aggregate: for each (K, N) compute delta mean/std over seeds
    # ------------------------------------------------------------------
    # Group rows by (K, N, variant_key)
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for r in all_rows:
        groups[(r["K"], r["N"], r["variant_key"])].append(r["a_spearman"])

    results = {}  # (K, N) -> {shared_mean, shared_std, decoupled_mean, decoupled_std, delta_mean, delta_std}
    for K in K_vals:
        for N in N_vals:
            sh_vals = groups.get((K, N, "shared"), [])
            de_vals = groups.get((K, N, "decoupled"), [])
            if not sh_vals or not de_vals:
                continue
            # Per-seed delta (over common seeds)
            sh_arr = np.array(sh_vals)
            de_arr = np.array(de_vals)
            # Seeds were generated in the same order for both variants
            n_common = min(len(sh_arr), len(de_arr))
            per_seed_delta = de_arr[:n_common] - sh_arr[:n_common]
            results[(K, N)] = {
                "shared_mean": float(np.mean(sh_arr)),
                "shared_std": float(np.std(sh_arr)),
                "decoupled_mean": float(np.mean(de_arr)),
                "decoupled_std": float(np.std(de_arr)),
                "delta_mean": float(np.mean(per_seed_delta)),
                "delta_std": float(np.std(per_seed_delta)),
                "n_seeds": n_common,
                "per_seed_delta": per_seed_delta.tolist(),
                "per_seed_shared": sh_arr.tolist(),
                "per_seed_decoupled": de_arr.tolist(),
            }

    # ------------------------------------------------------------------
    # Build the load-bearing nested JSON structure:
    #   delta[K][N]     = mean delta
    #   shared[K][N]    = {mean, std}
    #   decoupled[K][N] = {mean, std}
    # ------------------------------------------------------------------
    delta_nested: dict = {}
    shared_nested: dict = {}
    decoupled_nested: dict = {}
    for K in K_vals:
        K_str = str(K)
        delta_nested[K_str] = {}
        shared_nested[K_str] = {}
        decoupled_nested[K_str] = {}
        for N in N_vals:
            N_str = str(N)
            entry = results.get((K, N))
            if entry is None:
                continue
            delta_nested[K_str][N_str] = entry["delta_mean"]
            shared_nested[K_str][N_str] = {
                "mean": entry["shared_mean"],
                "std": entry["shared_std"],
            }
            decoupled_nested[K_str][N_str] = {
                "mean": entry["decoupled_mean"],
                "std": entry["decoupled_std"],
            }

    blob = {
        "meta": {
            "mode": mode,
            "device": device,
            "epochs": epochs,
            "Q": Q,
            "T": T,
            "seeds": seeds,
            "N_vals": N_vals,
            "K_vals": K_vals,
            "cheap": _CHEAP,
            "alpha_emb_dim": _ALPHA_EMB_DIM,
            "alpha_log_scale": _ALPHA_LOG_SCALE,
        },
        "N_vals": N_vals,
        "K_vals": K_vals,
        "delta": delta_nested,
        "shared": shared_nested,
        "decoupled": decoupled_nested,
        # per-cell detail including per-seed values
        "cells": {
            f"K{K}_N{N}": results[(K, N)]
            for K in K_vals for N in N_vals
            if (K, N) in results
        },
        "rows": all_rows,
    }

    out_json = OUT / "ndata_sweep_results.json"
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)

    table = render_table(results, N_vals, K_vals, device, epochs, seeds, mode)
    out_md = OUT / "ndata_sweep_table.md"
    out_md.write_text(table, encoding="utf-8")

    print("\n" + table)
    total_wall = time.time() - t_start
    print(f"\ntotal wall {total_wall:.1f}s ({total_wall/60:.1f} min)")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
