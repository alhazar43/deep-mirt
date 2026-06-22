"""run_arch_ksweep.py -- architecture-independence of the K-stiffness tracking.

Replicates ``run_ksweep_bench.py`` (the decoupling-advantage-grows-with-K claim)
but with a swappable ENCODER, so the claim can be tested on the transformer and
DKVMN backbones, not just the LSTM it was established on.

For each K we generate a static GPCM dataset and train two models that differ
ONLY in alpha's item capacity:

  SHARED     alpha reads the shared narrow item value (item_key_dim=None)
  DECOUPLED  alpha reads its own wide item key table (item_key_dim=64)

Both use state_alpha=True, alpha_log_scale=1.0, emb_dim=8, hidden_dim=32 (the
theta-encoder held fixed).  The headline is

  delta_K = a_spearman(DECOUPLED) - a_spearman(SHARED),

averaged over seeds, reported next to the GPCM Fisher stiffness ratio
I(theta)/I(alpha) (computed analytically, reused from run_ksweep_bench).

This runner does NOT edit any Codex-owned bench file.  It imports the engine
(DeepIRTEngine), the cell/aggregation helpers (run_cell, aggregate), and the
analytic stiffness (compute_stiffness) and drives them with encoder= swapped.

LSTM reference: Spearman(delta, kappa)=0.891 over K=2..11; delta rises +0.116
(K=2) to ~+0.30 (K=9).

Usage
-----
  source ~/anaconda3/etc/profile.d/conda.sh && conda activate research
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=".;rl/src;ma-irt" \
      python deep_irt/bench/run_arch_ksweep.py --encoder transformer \
          [--K 2 4 6 8 11] [--seeds 0 1 2 3] [--epochs 150] [--device cuda] [--quick]

Outputs
-------
  deep_irt/bench/outputs/arch_ksweep_{encoder}.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sp_stats

from deep_irt.bench.datagen import BenchDataConfig, generate
from deep_irt.bench.engines import DeepIRTEngine
from deep_irt.bench.run_alpha_fix_bench import run_cell, aggregate
from deep_irt.bench.run_ksweep_bench import compute_stiffness

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "outputs"
OUT.mkdir(exist_ok=True)

_CHEAP = dict(emb_dim=8, hidden_dim=32)   # fixed theta-encoder capacity
_ITEM_KEY_DIM = 64                        # decoupled wide alpha key
_ALPHA_LOG_SCALE = 1.0                    # exp(raw), ma-irt's link


def build_engines_for_K(ds, encoder, device, seed):
    """SHARED and DECOUPLED engines at a given K, on the requested encoder."""
    common = dict(decoder="gpcm", state_alpha=True,
                  alpha_log_scale=_ALPHA_LOG_SCALE, encoder=encoder,
                  device=device, seed=seed, **_CHEAP)
    return [
        ("shared", DeepIRTEngine(ds, item_key_dim=None, **common)),
        ("decoupled", DeepIRTEngine(ds, item_key_dim=_ITEM_KEY_DIM, **common)),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True,
                    choices=["lstm", "transformer", "dkvmn"])
    ap.add_argument("--K", type=int, nargs="+", default=[2, 4, 6, 8, 11])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-learners", type=int, default=800)
    ap.add_argument("--n-items", type=int, default=60)
    ap.add_argument("--seq-len", type=int, default=60)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.K = [2, 4, 6]
        args.seeds = [0, 1]
        args.epochs = 40
        args.n_learners, args.n_items, args.seq_len = 200, 30, 30

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print(f"=== arch ksweep: encoder={args.encoder} K={args.K} seeds={args.seeds} "
          f"epochs={args.epochs} device={device} {'QUICK' if args.quick else ''} ===")

    print("\n--- GPCM Fisher stiffness (Monte Carlo, 50k) ---")
    stiffness = {}
    for K in args.K:
        stiffness[K] = compute_stiffness(K, n_mc=50_000, seed=42)
        print(f"  K={K}: I(theta)/I(alpha) = {stiffness[K]:.4f}")

    t_start = time.time()
    rows = []
    for K in args.K:
        dsname = f"static_k{K}"
        print(f"\n=== K={K} ===")
        for seed in args.seeds:
            cfg = BenchDataConfig(
                name=dsname, kind="static", n_cats=K,
                n_learners=args.n_learners, n_items=args.n_items,
                seq_len=args.seq_len, seed=seed)
            ds = generate(cfg)
            for vkey, eng in build_engines_for_K(ds, args.encoder, device, seed):
                eng.fit(n_epochs=args.epochs)
                cell = run_cell(eng, ds)
                cell["variant_key"] = vkey
                cell["seed"] = seed
                cell["K"] = K
                rows.append(cell)
            sh = [r for r in rows if r["K"] == K and r["seed"] == seed
                  and r["variant_key"] == "shared"][0]
            de = [r for r in rows if r["K"] == K and r["seed"] == seed
                  and r["variant_key"] == "decoupled"][0]
            print(f"  seed{seed}  shared a={sh['a_spearman']:.3f}  "
                  f"decoupled a={de['a_spearman']:.3f}  "
                  f"delta={de['a_spearman']-sh['a_spearman']:+.3f}")

    agg = aggregate(rows)

    # delta_K (mean over seeds) and per-seed deltas.
    delta_mean = {}
    delta_std = {}
    delta_seeds = {}
    for K in args.K:
        sh = agg.get(("shared", f"static_k{K}"))
        de = agg.get(("decoupled", f"static_k{K}"))
        delta_mean[K] = float(de["a_spearman_mean"] - sh["a_spearman_mean"])
        delta_std[K] = float(np.sqrt(sh["a_spearman_std"]**2 +
                                     de["a_spearman_std"]**2))
        sh_seed = {r["seed"]: r["a_spearman"] for r in rows
                   if r["K"] == K and r["variant_key"] == "shared"}
        de_seed = {r["seed"]: r["a_spearman"] for r in rows
                   if r["K"] == K and r["variant_key"] == "decoupled"}
        common = sorted(set(sh_seed) & set(de_seed))
        delta_seeds[K] = [de_seed[s] - sh_seed[s] for s in common]

    # Tracking: Spearman(delta_K, stiffness) and monotonicity.
    K_sorted = sorted(args.K)
    d_arr = np.array([delta_mean[K] for K in K_sorted])
    s_arr = np.array([stiffness[K] for K in K_sorted])
    corr_sp = float(sp_stats.spearmanr(s_arr, d_arr).correlation)
    corr_pe = float(sp_stats.pearsonr(s_arr, d_arr)[0])

    print(f"\n--- {args.encoder.upper()} delta_K vs stiffness ---")
    print("  K | stiffness | shared_a | decoup_a | delta_K (std)")
    for K in K_sorted:
        sh = agg[("shared", f"static_k{K}")]
        de = agg[("decoupled", f"static_k{K}")]
        print(f"  {K:2d} | {stiffness[K]:.3f}    | "
              f"{sh['a_spearman_mean']:.3f}    | {de['a_spearman_mean']:.3f}    | "
              f"{delta_mean[K]:+.3f} ({delta_std[K]:.3f})")
    print(f"\n  Spearman(delta, stiffness)={corr_sp:.3f}  "
          f"Pearson={corr_pe:.3f}")
    rises = all(d_arr[i+1] >= d_arr[i] - 0.02 for i in range(len(d_arr)-1))
    print(f"  delta rises with K (tol 0.02): {rises}  "
          f"[delta {d_arr[0]:+.3f} (K={K_sorted[0]}) -> "
          f"{d_arr[-1]:+.3f} (K={K_sorted[-1]})]")

    agg_ser = {str(k): v for k, v in agg.items()}
    blob = {
        "meta": {"encoder": args.encoder, "K_vals": K_sorted,
                 "seeds": args.seeds, "epochs": args.epochs, "device": device,
                 "n_learners": args.n_learners, "n_items": args.n_items,
                 "seq_len": args.seq_len, "quick": bool(args.quick),
                 "cheap": _CHEAP, "item_key_dim": _ITEM_KEY_DIM,
                 "alpha_log_scale": _ALPHA_LOG_SCALE},
        "stiffness": stiffness,
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "delta_seeds_per_K": {str(K): v for K, v in delta_seeds.items()},
        "tracking": {"spearman_delta_stiffness": corr_sp,
                     "pearson_delta_stiffness": corr_pe,
                     "delta_rises_with_K": bool(rises)},
        "rows": rows,
        "agg": agg_ser,
    }
    path = OUT / f"arch_ksweep_{args.encoder}.json"
    if path.exists():
        raise SystemExit(f"REFUSING to overwrite existing file: {path}.")
    with path.open("w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)

    print(f"\ntotal wall {time.time()-t_start:.1f}s")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
