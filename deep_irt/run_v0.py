"""
run_v0.py  --  orchestrator for the anchoring-drift v0 experiment.

Usage:
  cd <repo_root>
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=deep_irt python deep_irt/run_v0.py

Outputs:
  deep_irt/outputs/v0_results.json    -- all numbers
  deep_irt/outputs/v0_summary.md      -- short readable summary
"""

import json
import os
import sys
import time
import numpy as np
import torch

# allow running from repo root with PYTHONPATH=deep_irt
sys.path.insert(0, os.path.dirname(__file__))

from data import DataConfig, generate
from train import train_base
from anchor import anchor_extend
from baseline import train_baseline
from metrics import compute_all, print_table


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_seeds(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    set_seeds(0)
    device = torch.device("cpu")

    # -----------------------------------------------------------------------
    print("\n=== ANCHORING DRIFT v0 ===\n")

    # --- Data generation ---
    print("Generating synthetic data ...")
    cfg = DataConfig(n_learners=2000, n_base=100, n_ext=30, n_cats=4, seed=0)
    dataset = generate(cfg)
    n_anchor = int(dataset.anchor_mask.sum())
    print(f"  N={cfg.n_learners}  B={cfg.n_base}  E={cfg.n_ext}  K={cfg.n_cats}"
          f"  anchor_learners={n_anchor}")

    # -----------------------------------------------------------------------
    print("\n--- Phase 1: Base calibration ---")
    phase1 = train_base(dataset, emb_dim=8, hidden_dim=32,
                         n_epochs=400, lr=1e-2, device=device, verbose=True)
    print(f"  Phase 1 done in {phase1['train_time']:.1f}s")

    # -----------------------------------------------------------------------
    print("\n--- Phase 2: Anchored extension ---")
    anchor_result = anchor_extend(dataset, phase1, n_epochs=400, lr=1e-2,
                                   device=device, verbose=True)
    print(f"  Anchored extension done in {anchor_result['train_time']:.1f}s"
          f"  ({anchor_result['n_params']} params trained)")

    # -----------------------------------------------------------------------
    print("\n--- Baseline: Full recalibration ---")
    baseline_result = train_baseline(dataset, emb_dim=8, hidden_dim=32,
                                      n_epochs=400, lr=1e-2,
                                      device=device, verbose=True)
    print(f"  Baseline done in {baseline_result['train_time']:.1f}s"
          f"  ({baseline_result['n_params']} params trained)")

    # -----------------------------------------------------------------------
    print("\n--- Computing metrics ---")
    results = compute_all(dataset, phase1, anchor_result, baseline_result,
                           base_model=phase1["model"])

    print("\n")
    print_table(results)

    # -----------------------------------------------------------------------
    # Serialize results (drop non-serialisable model objects)
    def _to_serialisable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return obj

    def _clean(d):
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out[k] = _clean(v)
            elif isinstance(v, np.ndarray):
                out[k] = v.tolist()
            elif isinstance(v, (np.floating, float)):
                out[k] = float(v)
            elif isinstance(v, (np.integer, int)):
                out[k] = int(v)
            else:
                out[k] = v
        return out

    results_clean = _clean(results)

    json_path = os.path.join(OUTPUT_DIR, "v0_results.json")
    with open(json_path, "w") as f:
        json.dump(results_clean, f, indent=2)
    print(f"\nResults written to {json_path}")

    # --- Summary markdown ---
    g1 = results["group1_new_item_recovery"]
    g2 = results["group2_theta_consistency"]
    g3 = results["group3_scale_stability"]
    g4 = results["group4_cost"]

    summary_lines = [
        "# Anchoring Drift v0 — Summary",
        "",
        f"Seed 0, N={cfg.n_learners}, B={cfg.n_base}, E={cfg.n_ext}, K={cfg.n_cats}, "
        f"anchor learners={n_anchor}",
        "",
        "## Group 1: New-item recovery (E items)",
        "",
        "| Method | a_corr | a_rmse | b_corr | b_rmse |",
        "|--------|--------|--------|--------|--------|",
    ]
    for method in ("anchoring", "baseline"):
        m = g1[method]
        summary_lines.append(
            f"| {method} | {m['a_corr']:.4f} | {m['a_rmse']:.4f} "
            f"| {m['b_corr']:.4f} | {m['b_rmse']:.4f} |"
        )

    summary_lines += [
        "",
        "## Group 2: Theta consistency (anchor learners)",
        "",
        f"- Phase-1 B-only theta vs truth: corr={g2['B_only_vs_truth']['corr']:.4f}"
        f", rmse={g2['B_only_vs_truth']['rmse']:.4f}",
        f"- Baseline B+E theta vs truth: corr={g2['BplusE_vs_truth']['corr']:.4f}"
        f", rmse={g2['BplusE_vs_truth']['rmse']:.4f}",
        f"- Phase-1 B-only vs Baseline base-prefix: "
        f"corr={g2['B_only_phase1_vs_baseline_base_prefix']['corr']:.4f}"
        f", rmse={g2['B_only_phase1_vs_baseline_base_prefix']['rmse']:.4f}",
        "",
        "## Group 3: Scale stability",
        "",
        f"- Anchoring: a_rmse={g3['anchoring']['a_rmse']:.4f}, "
        f"b_rmse={g3['anchoring']['b_rmse']:.4f}, "
        f"theta_rmse={g3['anchoring']['theta_rmse']:.4f} (frozen by construction)",
        f"- Baseline: a_corr={g3['baseline']['a_corr']:.4f}, "
        f"a_rmse={g3['baseline']['a_rmse']:.4f}, "
        f"b_corr={g3['baseline']['b_corr']:.4f}, "
        f"b_rmse={g3['baseline']['b_rmse']:.4f}, "
        f"theta_corr={g3['baseline']['theta_corr']:.4f}, "
        f"theta_rmse={g3['baseline']['theta_rmse']:.4f}",
        "",
        "## Group 4: Cost",
        "",
        "| Method | Params trained | Time (s) |",
        "|--------|---------------|----------|",
    ]
    for method in ("anchoring", "baseline"):
        m = g4[method]
        summary_lines.append(f"| {method} | {m['n_params']} | {m['time_seconds']:.1f} |")

    summary_path = os.path.join(OUTPUT_DIR, "v0_summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
