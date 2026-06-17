"""run_gate_bench.py -- Phase 0 GATE: shared vs decoupled item-embedding width sweep.

THE QUESTION
Is there a theta-vs-alpha trade-off at all, or just size?  If a wide-enough
SHARED item table recovers BOTH theta and alpha simultaneously, the effect is
trivial capacity -- a bigger embedding fixes it and decoupling is unnecessary.
If the shared family traces a frontier (alpha up <=> theta down, never both) and
the decoupled point sits ABOVE that frontier, the trade-off is real and
structural.  This decides whether the learning-dynamics study is worth doing.

DESIGN
- Backbone: LSTM only (one backbone, cheapest; the gate is about allocation).
- Decoder: GPCM, state_alpha=True, alpha_log_scale=1.0 everywhere.
- hidden_dim FIXED = 32 for every cell (so only item-embedding allocation changes).
- Width grid W in {8, 16, 24, 32, 48, 64, 96, 128}.
- Per W, two variants:
    SHARED(W)    emb_dim=W, item_key_dim=None.  One table width W feeds the
                 LSTM input AND the alpha head.  Total item-embedding params = Q*W.
    DECOUPLED(W) emb_dim=8, item_key_dim=W-8.  Narrow encoder (emb=8) + a
                 separate alpha-only table of width W-8.  Total = Q*8 + Q*(W-8)
                 = Q*W.  MATCHED capacity.  Skipped at W=8 (W-8=0 is invalid).
- Datasets: static_k4 (kind=static) and dynamic_k4 (kind=dynamic, drift=0.15).
  N=800, Q=60, T=60.  Seeds: [0,1,2,3,4].  Epochs: 150.
  --quick: N=200, Q=30, T=30, W in {8,16,32,64}, 2 seeds, 40 epochs.

OUTPUTS
  deep_irt/bench/outputs/gate_results.json
  deep_irt/bench/outputs/gate_table.md

Usage
-----
  source ~/anaconda3/etc/profile.d/conda.sh && conda activate research
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=".;rl/src;ma-irt" \\
      python deep_irt/bench/run_gate_bench.py [--quick] [--device cuda]
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

_HIDDEN_DIM = 32           # fixed for every cell
_ENCODER_EMB_DIM = 8       # narrow encoder embedding for DECOUPLED cells
_ALPHA_LOG_SCALE = 1.0     # exp(raw), consistent with ma-irt and prior benches

_FULL_W_GRID = [8, 16, 24, 32, 48, 64, 96, 128]
_QUICK_W_GRID = [8, 16, 32, 64]


# ---------------------------------------------------------------------------
# variant keys and pretty labels
# ---------------------------------------------------------------------------

def _shared_key(W: int) -> str:
    return f"shared_W{W}"

def _decoupled_key(W: int) -> str:
    return f"decoupled_W{W}"

def _pretty(key: str) -> str:
    if key.startswith("shared_W"):
        W = key[len("shared_W"):]
        return f"SHARED(W={W})"
    if key.startswith("decoupled_W"):
        W = key[len("decoupled_W"):]
        return f"DECOUPLED(W={W},e8+ae{int(W)-8})"
    return key


# ---------------------------------------------------------------------------
# engine factory for one (W, variant, dataset, seed)
# ---------------------------------------------------------------------------

def make_shared(ds, W: int, device: str, seed: int) -> DeepIRTEngine:
    return DeepIRTEngine(
        ds,
        encoder="lstm",
        decoder="gpcm",
        state_alpha=True,
        item_key_dim=None,           # shared: no separate alpha table
        alpha_log_scale=_ALPHA_LOG_SCALE,
        emb_dim=W,
        hidden_dim=_HIDDEN_DIM,
        device=device,
        seed=seed,
    )


def make_decoupled(ds, W: int, device: str, seed: int) -> DeepIRTEngine:
    """emb_dim=8 (narrow LSTM input) + item_key_dim=W-8 (separate alpha table).
    Total item-embedding params = Q*8 + Q*(W-8) = Q*W, matched to SHARED(W).
    """
    assert W > 8, f"DECOUPLED requires W > 8, got W={W}"
    return DeepIRTEngine(
        ds,
        encoder="lstm",
        decoder="gpcm",
        state_alpha=True,
        item_key_dim=W - _ENCODER_EMB_DIM,
        alpha_log_scale=_ALPHA_LOG_SCALE,
        emb_dim=_ENCODER_EMB_DIM,
        hidden_dim=_HIDDEN_DIM,
        device=device,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# ordered variant list for a given width grid
# ---------------------------------------------------------------------------

def variant_order(W_grid: list[int]) -> list[str]:
    """Returns all variant keys in display order: SHARED then DECOUPLED per W."""
    keys = []
    for W in W_grid:
        keys.append(_shared_key(W))
    for W in W_grid:
        if W > 8:
            keys.append(_decoupled_key(W))
    return keys


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _fmt_row(key: str, entry: dict) -> str:
    return (
        f"| {_pretty(key)} "
        f"| {_ms(entry, 'theta_spearman')} "
        f"| {_ms(entry, 'theta_netdrift_spearman')} "
        f"| {_ms(entry, 'a_spearman')} "
        f"| {_ms(entry, 'b_spearman')} "
        f"| {int(entry['n_params_mean'])} |"
    )


def _trend_label(vals: list[float]) -> str:
    """Return a brief monotonic trend label over a value list."""
    if len(vals) < 2:
        return "n/a"
    diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    if all(d > 0.005 for d in diffs):
        return "rising"
    if all(d < -0.005 for d in diffs):
        return "falling"
    # Check if any direction is dominant.
    pos = sum(d > 0.005 for d in diffs)
    neg = sum(d < -0.005 for d in diffs)
    if pos >= 2 * max(neg, 1):
        return "mostly rising"
    if neg >= 2 * max(pos, 1):
        return "mostly falling"
    return "mixed"


def _compute_verdict(agg: dict, W_grid: list[int]) -> str:
    """Compute the Phase-0 verdict automatically from the aggregated metrics."""

    def m(key: str, dsname: str, metric: str) -> float:
        e = agg.get((key, dsname))
        return e[f"{metric}_mean"] if e else float("nan")

    # Collect shared family metrics across widths (on static_k4).
    shared_keys = [_shared_key(W) for W in W_grid]
    shared_theta = [m(k, "static_k4", "theta_spearman") for k in shared_keys]
    shared_alpha = [m(k, "static_k4", "a_spearman") for k in shared_keys]

    # Filter NaN.
    valid = [(W, th, a) for W, th, a in zip(W_grid, shared_theta, shared_alpha)
             if th == th and a == a]

    L = ["## FRONTIER\n"]
    L.append("(theta_static, alpha) per cell on static_k4.  Mean over seeds.\n")
    L.append("SHARED family:")
    for W, th, a in valid:
        L.append(f"  SHARED(W={W}):   theta_static={th:.3f}  alpha={a:.3f}")

    L.append("DECOUPLED family:")
    decoupled_points = []
    for W in W_grid:
        if W <= 8:
            continue
        dk = _decoupled_key(W)
        th = m(dk, "static_k4", "theta_spearman")
        a = m(dk, "static_k4", "a_spearman")
        if th == th and a == a:
            decoupled_points.append((W, th, a))
            L.append(f"  DECOUPLED(W={W}): theta_static={th:.3f}  alpha={a:.3f}")

    L.append("")

    # Verdict logic.
    THETA_HIGH = 0.95
    ALPHA_HIGH = 0.90

    shared_reaches_both = any(th >= THETA_HIGH and a >= ALPHA_HIGH
                              for _, th, a in valid)

    L.append("## VERDICT\n")
    L.append(f"Thresholds: theta_static >= {THETA_HIGH}, alpha >= {ALPHA_HIGH}.\n")

    if shared_reaches_both:
        # Find the W where both are high.
        good_Ws = [W for W, th, a in valid if th >= THETA_HIGH and a >= ALPHA_HIGH]
        L.append("TRIVIAL CAPACITY: a wide-enough shared table gets both; "
                 "decoupling is unnecessary.")
        L.append(f"Shared W={good_Ws} reaches theta_static >= {THETA_HIGH} AND "
                 f"alpha >= {ALPHA_HIGH} simultaneously.")
    else:
        # Check if shared shows a trade-off (alpha rising while theta falling as W grows).
        if len(valid) >= 2:
            theta_trend = _trend_label(shared_theta)
            alpha_trend = _trend_label(shared_alpha)
            L.append(f"Shared family trend as W increases: "
                     f"theta_static {theta_trend}, alpha {alpha_trend}.")
            trade_off = ("fall" in theta_trend and "ris" in alpha_trend)
        else:
            trade_off = False
            L.append("Too few valid shared points to assess trend.")

        # Find best shared alpha and compare decoupled at same W.
        if valid:
            best_shared_W, best_shared_theta, best_shared_alpha = max(
                valid, key=lambda x: x[2])  # highest shared alpha
            L.append(
                f"\nBest shared alpha: SHARED(W={best_shared_W}) "
                f"theta_static={best_shared_theta:.3f}  alpha={best_shared_alpha:.3f}.")

            # Find matching DECOUPLED(W).
            matching_dec = [(W, th, a) for W, th, a in decoupled_points
                            if W == best_shared_W]
            if matching_dec:
                dec_W, dec_theta, dec_alpha = matching_dec[0]
                dec_dominates = (dec_theta >= best_shared_theta and
                                 dec_alpha >= best_shared_alpha)
                dec_both_high = (dec_theta >= THETA_HIGH and dec_alpha >= ALPHA_HIGH)
                L.append(
                    f"DECOUPLED(W={dec_W}) at same capacity: "
                    f"theta_static={dec_theta:.3f}  alpha={dec_alpha:.3f}  "
                    f"({'DOMINATES' if dec_dominates else 'does not dominate'} "
                    f"shared; {'BOTH HIGH' if dec_both_high else 'not both high'}).")
            else:
                dec_dominates = False
                dec_both_high = False
                L.append(f"No DECOUPLED point at W={best_shared_W} to compare.")

            # Also check if any decoupled point reaches both high.
            any_dec_both = any(th >= THETA_HIGH and a >= ALPHA_HIGH
                               for _, th, a in decoupled_points)

            L.append("")
            if trade_off and any_dec_both:
                L.append(
                    "TRADE-OFF CONFIRMED: shared trades theta for alpha across "
                    "widths and never reaches both; decoupled dominates the "
                    "frontier (both high). PROCEED to the dynamics study.")
                # Print supporting numbers.
                shared_nums = "  ".join(
                    f"W={W}:(theta={th:.3f},alpha={a:.3f})" for W, th, a in valid)
                dec_nums = "  ".join(
                    f"W={W}:(theta={th:.3f},alpha={a:.3f})"
                    for W, th, a in decoupled_points)
                L.append(f"\nShared family: {shared_nums}")
                L.append(f"Decoupled family: {dec_nums}")
            elif not trade_off and not shared_reaches_both:
                L.append(
                    "INCONCLUSIVE: no clear theta-vs-alpha trade-off in the "
                    "shared family, but no shared W reaches both high.  Inspect "
                    "the per-W numbers and the alpha variance across seeds.")
            else:
                # Trade-off exists but decoupled doesn't clearly beat it.
                L.append(
                    "PARTIAL TRADE-OFF: the shared family shows a trade-off "
                    "direction, but the decoupled family does not clearly reach "
                    "both high.  Inspect the decoupled alpha and seed variance.")

    return "\n".join(L)


def render_table(agg: dict, device: str, epochs: int, N: int, Q: int, T: int,
                 seeds: list[int], W_grid: list[int]) -> str:
    L = []
    L.append("# Gate bench -- shared vs decoupled item-embedding width sweep\n")
    L.append(f"device={device}  epochs={epochs}  N={N}  Q={Q}  T={T}  "
             f"seeds={seeds}  hidden_dim={_HIDDEN_DIM}  (torch {torch.__version__})\n")
    L.append("Mean +- std over seeds.  LSTM backbone, GPCM decoder, "
             "state_alpha=True, alpha_log_scale=1.0 throughout.  hidden_dim=32 "
             "FIXED.  Only item-embedding ALLOCATION varies.\n")
    L.append("SHARED(W): single item table width W feeds both LSTM input and "
             "alpha head.  Total item params = Q*W.\n")
    L.append("DECOUPLED(W): emb_dim=8 narrow encoder + separate alpha table "
             "item_key_dim=W-8.  Total item params = Q*W (MATCHED to SHARED).\n")

    header = ("| variant | theta_static | theta_drift | a (sp) | b (sp) | params |")
    sep = "|---|---|---|---|---|---|"

    for dsname in ("static_k4", "dynamic_k4"):
        L.append(f"\n## {dsname}\n")
        L.append(header)
        L.append(sep)
        # SHARED rows first.
        for W in W_grid:
            key = _shared_key(W)
            e = agg.get((key, dsname))
            if e is None:
                continue
            L.append(_fmt_row(key, e))
        # DECOUPLED rows.
        for W in W_grid:
            if W <= 8:
                continue
            key = _decoupled_key(W)
            e = agg.get((key, dsname))
            if e is None:
                continue
            L.append(_fmt_row(key, e))

    L.append("")
    L.append(_compute_verdict(agg, W_grid))
    return "\n".join(L)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase-0 gate: sweep shared vs decoupled item-embedding width.")
    ap.add_argument("--quick", action="store_true",
                    help="Small N/Q/T, fewer widths and seeds for a fast smoke test.")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override epoch count (default 150 full / 40 quick).")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Override seed list (default [0-4] full / [0,1] quick).")
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if args.quick:
        N, Q, T = 200, 30, 30
        epochs = args.epochs or 40
        seeds = args.seeds or [0, 1]
        W_grid = _QUICK_W_GRID
    else:
        N, Q, T = 800, 60, 60
        epochs = args.epochs or 150
        seeds = args.seeds or [0, 1, 2, 3, 4]
        W_grid = _FULL_W_GRID

    mode = "quick" if args.quick else "full"
    print(f"=== gate bench [{mode}]: device={device} epochs={epochs} "
          f"N={N} Q={Q} T={T} seeds={seeds} W_grid={W_grid} ===")

    t_start = time.time()
    rows = []

    for seed in seeds:
        data_cfgs = [
            BenchDataConfig(name="static_k4", kind="static", n_cats=4,
                            n_learners=N, n_items=Q, seq_len=T, seed=seed),
            BenchDataConfig(name="dynamic_k4", kind="dynamic", n_cats=4,
                            n_learners=N, n_items=Q, seq_len=T,
                            drift_sigma=0.15, seed=seed),
        ]
        datasets = {c.name: generate(c) for c in data_cfgs}

        for dsname in ("static_k4", "dynamic_k4"):
            ds = datasets[dsname]
            print(f"\n--- seed {seed}  dataset {dsname} ---")

            for W in W_grid:
                # SHARED(W)
                key = _shared_key(W)
                eng = make_shared(ds, W, device, seed)
                eng.fit(n_epochs=epochs)
                cell = run_cell(eng, ds)
                cell["variant_key"] = key
                cell["seed"] = seed
                cell["W"] = W
                cell["variant_type"] = "shared"
                rows.append(cell)
                print(f"  SHARED(W={W:3d})     "
                      f"theta_static={cell['theta_spearman']:.3f}  "
                      f"theta_drift={cell['theta_netdrift_spearman']:.3f}  "
                      f"a={cell['a_spearman']:.3f}  "
                      f"b={cell['b_spearman']:.3f}  "
                      f"p={cell['n_params']}")

                # DECOUPLED(W) -- skip W=8 (item_key_dim=0 is invalid)
                if W <= 8:
                    continue
                key = _decoupled_key(W)
                eng = make_decoupled(ds, W, device, seed)
                eng.fit(n_epochs=epochs)
                cell = run_cell(eng, ds)
                cell["variant_key"] = key
                cell["seed"] = seed
                cell["W"] = W
                cell["variant_type"] = "decoupled"
                rows.append(cell)
                print(f"  DECOUPLED(W={W:3d})  "
                      f"theta_static={cell['theta_spearman']:.3f}  "
                      f"theta_drift={cell['theta_netdrift_spearman']:.3f}  "
                      f"a={cell['a_spearman']:.3f}  "
                      f"b={cell['b_spearman']:.3f}  "
                      f"p={cell['n_params']}")

    agg = aggregate(rows)

    blob = {
        "meta": {
            "mode": mode,
            "device": device,
            "epochs": epochs,
            "N": N, "Q": Q, "T": T,
            "seeds": seeds,
            "W_grid": W_grid,
            "hidden_dim": _HIDDEN_DIM,
            "encoder_emb_dim_for_decoupled": _ENCODER_EMB_DIM,
            "alpha_log_scale": _ALPHA_LOG_SCALE,
        },
        "rows": rows,
        "agg": [{"key": list(k), **v} for k, v in agg.items()],
    }
    out_json = OUT / "gate_results.json"
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)

    table = render_table(agg, device, epochs, N, Q, T, seeds, W_grid)
    out_md = OUT / "gate_table.md"
    out_md.write_text(table, encoding="utf-8")

    print("\n" + table)
    print(f"\ntotal wall {time.time() - t_start:.1f}s")
    print(f"wrote {out_json} and {out_md}")


if __name__ == "__main__":
    main()
