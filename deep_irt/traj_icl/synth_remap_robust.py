"""synth_remap_robust.py -- E1b robustness study: fine k-grid + mapping-seed sweep.

Runs synth_remap with:
  - Fine shot-count grid: k = {0,2,4,6,8,10,12,14,16,24,32}
  - 3 mapping seeds (1, 2, 3)
  - Models: Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct
  - 150 query items per run

Response files are written to:
    deep_irt/traj_icl/outputs_e1b_robust/seed{S}/responses_{model}.json

After all runs complete, computes per-(model, seed):
  - g(k) = acc_true(k) - acc_shuffled(k)    (priming-corrected accuracy gap)
  - k*   = smallest k where g(k) >= 0.5 * max(g)
  - max_gap

Saves summary JSON and plots g(k) curves for all seeds x models.

Run from the repo root:
    python -m deep_irt.traj_icl.synth_remap_robust [--smoke]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from typing import Any, Dict, List, Optional

import torch

HERE = os.path.dirname(__file__)
OUT_ROBUST = os.path.join(HERE, "outputs_e1b_robust")

FINE_K_GRID = [0, 2, 4, 6, 8, 10, 12, 14, 16, 24, 32]
MAPPING_SEEDS = [1, 2, 3]
MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]
N_ITEMS = 150


# ---------------------------------------------------------------------------
# Collection: run all seeds x models
# ---------------------------------------------------------------------------

def run_all(max_items: int = N_ITEMS, smoke: bool = False) -> None:
    """Run the full seed x model matrix, writing per-seed response files."""
    from deep_irt.traj_icl.synth_remap import (
        _make_bijection,
        build_bank_and_demos,
        load_bank_and_demos,
        run_model,
    )

    models = ["Qwen/Qwen2.5-0.5B-Instruct"] if smoke else MODELS
    seeds = MAPPING_SEEDS
    k_grid = [0, 2, 4] if smoke else FINE_K_GRID
    n = 20 if smoke else max_items

    t_total = time.time()
    peak_vrams: List[float] = []

    for seed in seeds:
        bijection = _make_bijection(seed)
        out_dir = os.path.join(OUT_ROBUST, f"seed{seed}")
        os.makedirs(out_dir, exist_ok=True)

        bank_path = os.path.join(out_dir, "item_bank.json")
        if os.path.exists(bank_path):
            print(f"\n[seed {seed}] Loading existing bank from {out_dir}", flush=True)
            bank, demos = load_bank_and_demos(out_dir)
        else:
            print(f"\n[seed {seed}] Building bank with bijection {bijection}", flush=True)
            bank, demos = build_bank_and_demos(out_dir, bijection=bijection)

        for model_id in models:
            safetag = model_id.split("/")[-1]
            resp_path = os.path.join(out_dir, f"responses_{safetag}.json")
            if os.path.exists(resp_path):
                print(f"[seed {seed}] {safetag}: response file exists, skipping.", flush=True)
                continue

            print(f"\n{'='*60}", flush=True)
            print(f"[seed {seed}] {model_id}", flush=True)
            print(f"{'='*60}", flush=True)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            try:
                result = run_model(
                    model_id=model_id,
                    bank=bank,
                    demos=demos,
                    out_dir=out_dir,
                    device="cuda",
                    max_items=n,
                    k_values=k_grid,
                )
                peak_vrams.append(result["meta"]["peak_vram_gb"])
                print(
                    f"\nDone {model_id} | wall={result['meta']['wall_clock_s']:.0f}s "
                    f"| peak VRAM={result['meta']['peak_vram_gb']:.2f}GB",
                    flush=True,
                )
            except torch.cuda.OutOfMemoryError:
                print(f"OOM: {model_id}, skipping.", flush=True)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    wall = time.time() - t_total
    peak = max(peak_vrams) if peak_vrams else 0.0
    print(f"\nTotal wall clock: {wall:.0f}s | Peak VRAM across runs: {peak:.2f}GB", flush=True)
    return wall, peak


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _compute_gap(resp: Dict[str, Any]) -> Dict[str, Any]:
    """Compute g(k) and derived stats from a response dict."""
    k_values = [int(k) for k in resp["k_values"]]
    acc_true = [resp["accuracy"]["true"][str(k)] for k in k_values]
    acc_shuf = [resp["accuracy"]["shuffled"][str(k)] for k in k_values]
    gap = [t - s for t, s in zip(acc_true, acc_shuf)]
    max_gap = max(gap)

    # k* = smallest k where g(k) >= 0.5 * max_gap
    threshold_val = 0.5 * max_gap
    k_star = None
    for k, g in zip(k_values, gap):
        if g >= threshold_val:
            k_star = k
            break
    if k_star is None:
        k_star = k_values[-1]  # never crossed; use last

    return {
        "k_values": k_values,
        "acc_true": acc_true,
        "acc_shuffled": acc_shuf,
        "gap": gap,
        "max_gap": max_gap,
        "k_star": k_star,
    }


def analyze(smoke: bool = False) -> Dict[str, Any]:
    """Load all response files and compute per-(model, seed) stats."""
    seeds = MAPPING_SEEDS
    models = ["Qwen/Qwen2.5-0.5B-Instruct"] if smoke else MODELS

    records: List[Dict[str, Any]] = []

    for seed in seeds:
        out_dir = os.path.join(OUT_ROBUST, f"seed{seed}")
        for model_id in models:
            safetag = model_id.split("/")[-1]
            resp_path = os.path.join(out_dir, f"responses_{safetag}.json")
            if not os.path.exists(resp_path):
                print(f"WARNING: missing {resp_path}", flush=True)
                continue
            with open(resp_path, encoding="utf-8") as f:
                resp = json.load(f)
            stats = _compute_gap(resp)
            records.append({
                "model": model_id,
                "seed": seed,
                **stats,
            })

    # Per-model summary across seeds.
    model_summary: Dict[str, Any] = {}
    for model_id in models:
        rows = [r for r in records if r["model"] == model_id]
        if not rows:
            continue
        k_stars = [r["k_star"] for r in rows]
        max_gaps = [r["max_gap"] for r in rows]
        positive_gaps_all = all(r["max_gap"] > 0.05 for r in rows)
        model_summary[model_id] = {
            "k_star_per_seed": {r["seed"]: r["k_star"] for r in rows},
            "max_gap_per_seed": {r["seed"]: round(r["max_gap"], 4) for r in rows},
            "k_star_mean": round(sum(k_stars) / len(k_stars), 1),
            "k_star_range": [min(k_stars), max(k_stars)],
            "max_gap_mean": round(sum(max_gaps) / len(max_gaps), 4),
            "max_gap_range": [round(min(max_gaps), 4), round(max(max_gaps), 4)],
            "positive_gap_all_seeds": positive_gaps_all,
        }

    # Size ordering check (3B > 1.5B in max_gap for all seeds).
    size_ordering_holds = _check_size_ordering(records, models)

    result = {
        "per_model_seed": [
            {
                "model": r["model"],
                "seed": r["seed"],
                "k_star": r["k_star"],
                "max_gap": round(r["max_gap"], 4),
            }
            for r in records
        ],
        "model_summary": model_summary,
        "size_ordering_holds_all_seeds": size_ordering_holds,
    }

    # Print table.
    print("\n" + "="*65)
    print("E1b ROBUSTNESS: per-(model, seed) k* and max-gap")
    print("="*65)
    header = f"{'Model':<30} {'Seed':>4} {'k*':>5} {'max-gap':>9}"
    print(header)
    print("-"*65)
    for r in records:
        tag = r["model"].split("/")[-1].replace("Qwen2.5-", "")
        print(f"  {tag:<28} {r['seed']:>4} {r['k_star']:>5} {r['max_gap']:>9.4f}")
    print("-"*65)
    for model_id, s in model_summary.items():
        tag = model_id.split("/")[-1].replace("Qwen2.5-", "")
        print(
            f"  {tag:<28} {'mean':>4} {s['k_star_mean']:>5.1f} "
            f"{s['max_gap_mean']:>9.4f}  "
            f"(k* range {s['k_star_range'][0]}-{s['k_star_range'][1]})"
        )
    print("="*65)
    print(f"\nPositive gap in all seeds:")
    for model_id, s in model_summary.items():
        tag = model_id.split("/")[-1]
        print(f"  {tag}: {'YES' if s['positive_gap_all_seeds'] else 'NO'}")
    print(f"\n3B > 1.5B gap ordering holds in all seeds: "
          f"{'YES' if size_ordering_holds else 'NO'}")

    # Save summary.
    summary_path = os.path.join(OUT_ROBUST, "robust_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved summary to {summary_path}", flush=True)

    return result, records


def _check_size_ordering(
    records: List[Dict[str, Any]],
    models: List[str],
) -> bool:
    """Return True iff the larger model has strictly greater max_gap for all seeds."""
    if len(models) < 2:
        return True
    # Assume models ordered small -> large.
    small, large = models[0], models[1]
    for seed in MAPPING_SEEDS:
        r_small = next((r for r in records if r["model"] == small and r["seed"] == seed), None)
        r_large = next((r for r in records if r["model"] == large and r["seed"] == seed), None)
        if r_small is None or r_large is None:
            continue
        if r_large["max_gap"] <= r_small["max_gap"]:
            return False
    return True


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(records: List[Dict[str, Any]], smoke: bool = False) -> None:
    """Plot g(k) curves for all (model, seed) combos."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot.", flush=True)
        return

    models = ["Qwen/Qwen2.5-0.5B-Instruct"] if smoke else MODELS
    seeds = MAPPING_SEEDS
    colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}
    linestyles = {
        "Qwen/Qwen2.5-1.5B-Instruct": "-",
        "Qwen/Qwen2.5-3B-Instruct": "--",
    }

    fig, ax = plt.subplots(figsize=(7, 4))

    for seed in seeds:
        for model_id in models:
            row = next(
                (r for r in records if r["model"] == model_id and r["seed"] == seed),
                None,
            )
            if row is None:
                continue
            label = f"{model_id.split('/')[-1].replace('Qwen2.5-','').replace('-Instruct','')} seed{seed}"
            ax.plot(
                row["k_values"],
                row["gap"],
                color=colors.get(seed, "black"),
                linestyle=linestyles.get(model_id, "-"),
                marker="o",
                markersize=4,
                label=label,
            )
            if row.get("k_star") is not None:
                ax.axvline(
                    x=row["k_star"],
                    color=colors.get(seed, "gray"),
                    linestyle=":",
                    alpha=0.3,
                )

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xlabel("Shot count k")
    ax.set_ylabel("g(k) = acc_true − acc_shuffled")
    ax.set_title("E1b Robustness: priming-corrected gap by seed and model size")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.set_xticks(FINE_K_GRID if not smoke else [0, 2, 4])

    out_path = os.path.join(OUT_ROBUST, "e1b_robust_gap_curves.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="E1b robustness study: fine k-grid + mapping-seed sweep"
    )
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: tiny grid, 0.5B only, 20 items")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip collection, only analyze existing response files")
    args = parser.parse_args()

    os.makedirs(OUT_ROBUST, exist_ok=True)

    if not args.analyze_only:
        wall, peak = run_all(max_items=N_ITEMS, smoke=args.smoke)
    else:
        wall, peak = 0.0, 0.0

    result, records = analyze(smoke=args.smoke)
    plot(records, smoke=args.smoke)

    # One-line verdict.
    pos_all = all(
        s["positive_gap_all_seeds"]
        for s in result["model_summary"].values()
    )
    size_ok = result["size_ordering_holds_all_seeds"]

    # Collect k* range across all.
    all_kstars = [r["k_star"] for r in records]
    kstar_min = min(all_kstars) if all_kstars else "?"
    kstar_max = max(all_kstars) if all_kstars else "?"

    print("\n" + "="*65)
    print("VERDICT")
    print("="*65)
    print(
        f"E1b headline (positive gap, 3B > 1.5B) holds in ALL seeds: "
        f"{'YES' if (pos_all and size_ok) else 'PARTIAL/NO'}"
    )
    print(f"Threshold k* consistent across seeds: k* in [{kstar_min}, {kstar_max}]")
    if wall > 0:
        print(f"Total wall clock: {wall:.0f}s | Peak VRAM: {peak:.2f}GB")


if __name__ == "__main__":
    main()
