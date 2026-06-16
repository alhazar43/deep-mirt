"""
run_v01.py  --  orchestrator for the anchoring-drift v0.1 experiment.

Runs two venues end-to-end:
  Venue A: same-distribution E (ext_shift=0.0)
  Venue B: distribution-shifted E (ext_shift=+1.0, harder items)

The key addition over v0: a genuine tracking test (Group 2) that feeds
interleaved B+E sequences through the frozen Phase-1 encoder (which now
has E-item embeddings appended) and measures whether theta stays coherent.

Usage:
  cd <repo_root>
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=deep_irt python deep_irt/run_v01.py

Outputs:
  deep_irt/outputs/v01_venue_a_results.json
  deep_irt/outputs/v01_venue_b_results.json
  deep_irt/outputs/v01_summary.txt
"""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from data import DataConfig, generate
from train import train_base
from anchor import anchor_extend
from baseline import train_baseline
from metrics import compute_all, print_table

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_seeds(seed: int = 0) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _clean(obj):
    """Recursively make dicts/lists JSON-serialisable (drop torch/model objects)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()
                if not hasattr(v, "parameters")}   # skip nn.Module
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def run_venue(label: str, ext_shift: float, device: torch.device) -> dict:
    """Run the full pipeline for one venue and return results dict."""
    print(f"\n{'='*70}")
    print(f"  VENUE {label}  (ext_shift={ext_shift:+.1f})")
    print(f"{'='*70}")

    set_seeds(0)

    cfg = DataConfig(
        n_learners=2000, n_base=100, n_ext=30, n_cats=4,
        seed=0, ext_shift=ext_shift
    )
    dataset = generate(cfg)
    n_anchor = int(dataset.anchor_mask.sum())
    print(f"\n  Data: N={cfg.n_learners}  B={cfg.n_base}  E={cfg.n_ext}"
          f"  K={cfg.n_cats}  anchor_learners={n_anchor}"
          f"  ext_shift={cfg.ext_shift:+.1f}")

    # Phase 1
    print("\n--- Phase 1: Base calibration ---")
    phase1 = train_base(dataset, emb_dim=8, hidden_dim=32,
                        n_epochs=400, lr=1e-2, device=device, verbose=True)
    print(f"  done in {phase1['train_time']:.1f}s")

    # Phase 2: anchored extension
    print("\n--- Phase 2: Anchored extension ---")
    anchor_result = anchor_extend(dataset, phase1, n_epochs=400, lr=1e-2,
                                  device=device, verbose=True)
    print(f"  done in {anchor_result['train_time']:.1f}s"
          f"  ({anchor_result['n_params']} params trained)")

    # Baseline: full recalibration
    print("\n--- Baseline: Full recalibration ---")
    baseline_result = train_baseline(dataset, emb_dim=8, hidden_dim=32,
                                     n_epochs=400, lr=1e-2,
                                     device=device, verbose=True)
    print(f"  done in {baseline_result['train_time']:.1f}s"
          f"  ({baseline_result['n_params']} params trained)")

    # Metrics
    print("\n--- Computing metrics ---")
    results = compute_all(dataset, phase1, anchor_result, baseline_result,
                          base_model=phase1["model"])

    print()
    print_table(results)

    return results


def write_summary(venue_results: dict, summary_path: str) -> None:
    lines = ["ANCHORING DRIFT v0.1 -- SUMMARY", "=" * 70, ""]

    for label, results in venue_results.items():
        ext_shift = results.get("_meta", {}).get("ext_shift", "?")
        lines.append(f"VENUE {label}  (ext_shift={ext_shift})")
        lines.append("-" * 60)

        g1 = results["group1_new_item_recovery"]
        lines.append("  Group 1: New-item recovery (E items)")
        for method in ("anchoring", "baseline"):
            m = g1[method]
            lines.append(
                f"    {method:12s}  a_corr={m['a_corr']:.4f}  a_rmse={m['a_rmse']:.4f}"
                f"  b_corr={m['b_corr']:.4f}  b_rmse={m['b_rmse']:.4f}"
            )

        g2 = results["group2_theta_consistency"]
        lines.append("  Group 2: Tracking consistency (frozen encoder on B+E sequences)")
        if "error" in g2:
            lines.append(f"    ERROR: {g2['error']}")
        else:
            lines.append(
                f"    theta_BplusE vs B-only  :  corr={g2['BplusE_vs_B_only']['corr']:.4f}"
                f"  rmse={g2['BplusE_vs_B_only']['rmse']:.4f}"
            )
            lines.append(
                f"    theta_BplusE vs truth   :  corr={g2['BplusE_vs_truth']['corr']:.4f}"
                f"  rmse={g2['BplusE_vs_truth']['rmse']:.4f}"
            )
            lines.append(
                f"    theta_B_only vs truth   :  corr={g2['B_only_vs_truth']['corr']:.4f}"
                f"  rmse={g2['B_only_vs_truth']['rmse']:.4f}"
            )

        g4 = results["group4_cost"]
        lines.append("  Group 4: Cost")
        anc_t = g4["anchoring"]["time_seconds"]
        bas_t = g4["baseline"]["time_seconds"]
        ratio = bas_t / anc_t if anc_t > 0 else float("nan")
        lines.append(
            f"    anchoring: {g4['anchoring']['n_params']:5d} params  {anc_t:.1f}s"
        )
        lines.append(
            f"    baseline : {g4['baseline']['n_params']:5d} params  {bas_t:.1f}s"
            f"  ({ratio:.1f}x more expensive)"
        )
        lines.append("")

    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSummary written to {summary_path}")


def main() -> None:
    device = torch.device("cpu")

    venue_results = {}

    # Venue A: same-distribution E
    results_a = run_venue("A (same-dist)", ext_shift=0.0, device=device)
    results_a["_meta"] = {"ext_shift": 0.0, "description": "same-distribution E"}
    venue_results["A"] = results_a

    json_a = os.path.join(OUTPUT_DIR, "v01_venue_a_results.json")
    with open(json_a, "w") as f:
        json.dump(_clean(results_a), f, indent=2)
    print(f"Venue A results written to {json_a}")

    # Venue B: distribution-shifted E (harder items, +1.0 on step thresholds)
    results_b = run_venue("B (shifted)", ext_shift=1.0, device=device)
    results_b["_meta"] = {"ext_shift": 1.0, "description": "distribution-shifted E (+1.0 step thresholds)"}
    venue_results["B"] = results_b

    json_b = os.path.join(OUTPUT_DIR, "v01_venue_b_results.json")
    with open(json_b, "w") as f:
        json.dump(_clean(results_b), f, indent=2)
    print(f"Venue B results written to {json_b}")

    # Combined summary
    summary_path = os.path.join(OUTPUT_DIR, "v01_summary.txt")
    write_summary(venue_results, summary_path)

    print("\n=== v0.1 COMPLETE ===")


if __name__ == "__main__":
    main()
