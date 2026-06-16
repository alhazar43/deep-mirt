"""
run_v1_dynamic.py  --  dynamic theta v1 experiment (random walk ability).

Runs the full pipeline end-to-end for a random_walk dataset:
  1. Generate data with drifting theta (random_walk, drift_sigma=0.12).
  2. Phase 1: train encoder + decoder + B-item embeddings on B sequences
     with moving theta.  The LSTM already outputs theta_t per step.
  3. Phase-1 trajectory recovery: how well does the encoder track moving ability?
  4. Structural anchoring (Phase 2): freeze encoder; fit E-item embeddings
     using each anchor learner's end-of-B theta as the fixed anchor.
     (Approximation: E-calibration theta = end-of-B theta; see note in data.py.)
  5. Tracking test: for anchor learners, build interleaved B+E sequences,
     run through frozen extended encoder, recover full theta_t trajectory,
     compare to (a) true per-interaction theta and (b) B-only trajectory.

Strategic focus: TRAJECTORY TRACKING is the headline result.
Does the deep encoder recover a drifting ability trajectory?
Does tracking survive interleaved new (unseen) items?

Usage:
  cd <repo_root>
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=deep_irt python deep_irt/run_v1_dynamic.py

Outputs:
  deep_irt/outputs/v1_dynamic_results.json
  deep_irt/outputs/v1_dynamic_summary.txt
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
from metrics import (
    trajectory_recovery,
    trajectory_tracking_test,
    compute_all,
    _pearson,
    _rmse,
)

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
                if not hasattr(v, "parameters")}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, bool):
        return bool(obj)
    return obj


def run_v1(drift_sigma: float = 0.12,
           device: torch.device = torch.device("cpu"),
           n_epochs: int = 400,
           lr: float = 1e-2,
           emb_dim: int = 8,
           hidden_dim: int = 32) -> dict:
    """Run dynamic theta v1 end-to-end and return full results dict."""

    set_seeds(0)

    print(f"\n{'='*70}")
    print(f"  DYNAMIC THETA v1  (random_walk, drift_sigma={drift_sigma})")
    print(f"{'='*70}")

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------
    cfg = DataConfig(
        n_learners=2000,
        n_base=100,
        n_ext=30,
        n_cats=4,
        seed=0,
        ext_shift=0.0,
        theta_dynamics="random_walk",
        drift_sigma=drift_sigma,
    )
    dataset = generate(cfg)
    n_anchor = int(dataset.anchor_mask.sum())

    print(f"\n  Data: N={cfg.n_learners}  B={cfg.n_base}  E={cfg.n_ext}"
          f"  K={cfg.n_cats}  anchor_learners={n_anchor}"
          f"  drift_sigma={drift_sigma}")

    # Quick sanity: report true drift magnitude
    theta_traj = dataset.gt.theta_traj   # (N, B)
    net_drift = theta_traj[:, -1] - theta_traj[:, 0]
    print(f"  True net drift (end - start):  mean={net_drift.mean():.3f}"
          f"  std={net_drift.std():.3f}"
          f"  |mean|={np.abs(net_drift).mean():.3f}")

    # ------------------------------------------------------------------
    # Phase 1: train on B sequences with moving theta
    # ------------------------------------------------------------------
    print("\n--- Phase 1: Base calibration (dynamic theta) ---")
    phase1 = train_base(dataset, emb_dim=emb_dim, hidden_dim=hidden_dim,
                        n_epochs=n_epochs, lr=lr, device=device, verbose=True)
    print(f"  done in {phase1['train_time']:.1f}s")

    # ------------------------------------------------------------------
    # Phase-1 trajectory recovery (HEADLINE metric)
    # ------------------------------------------------------------------
    print("\n--- Phase-1 trajectory recovery ---")
    base_model = phase1["model"]
    traj_p1 = trajectory_recovery(dataset, base_model, device, label="Phase1_B")

    print(f"  Pooled corr (recovered vs true theta_t):  {traj_p1['pooled_corr']:.4f}")
    print(f"  Pooled RMSE (recovered vs true theta_t):  {traj_p1['pooled_rmse']:.4f}")
    print(f"  Net-drift corr (end-start recovered vs true): {traj_p1['net_drift_corr']:.4f}")
    print(f"  Net-drift RMSE:                           {traj_p1['net_drift_rmse']:.4f}")

    # ------------------------------------------------------------------
    # Phase 2: Anchored extension
    # ------------------------------------------------------------------
    print("\n--- Phase 2: Anchored extension (E items, end-of-B anchor theta) ---")
    anchor_result = anchor_extend(dataset, phase1, n_epochs=n_epochs, lr=lr,
                                  device=device, verbose=True)
    print(f"  done in {anchor_result['train_time']:.1f}s"
          f"  ({anchor_result['n_params']} params trained)")

    # New-item recovery (a, b for E items)
    gt = dataset.gt
    a_corr = _pearson(anchor_result["a_ext"], gt.a_ext)
    a_rmse = _rmse(anchor_result["a_ext"], gt.a_ext)
    b_corr = _pearson(anchor_result["b_ext"].ravel(), gt.b_ext.ravel())
    b_rmse = _rmse(anchor_result["b_ext"].ravel(), gt.b_ext.ravel())
    print(f"\n  E-item recovery (anchoring):")
    print(f"    a:  corr={a_corr:.4f}  rmse={a_rmse:.4f}")
    print(f"    b:  corr={b_corr:.4f}  rmse={b_rmse:.4f}")

    # ------------------------------------------------------------------
    # The tracking test (HEADLINE: does tracking survive new items?)
    # ------------------------------------------------------------------
    print("\n--- Tracking test: interleaved B+E trajectories ---")
    track = trajectory_tracking_test(dataset, base_model, anchor_result, device)

    if "error" in track:
        print(f"  ERROR: {track['error']}")
    else:
        print(f"  B-only trajectory recovery (reference):")
        print(f"    pooled corr={track['Bonly_pooled_corr']:.4f}"
              f"  rmse={track['Bonly_pooled_rmse']:.4f}")
        print(f"  B-item positions in interleaved B+E sequence:")
        print(f"    pooled corr={track['BplusE_B_pos_corr']:.4f}"
              f"  rmse={track['BplusE_B_pos_rmse']:.4f}")
        print(f"  Final-step theta: B+E vs B-only (internal consistency):")
        print(f"    corr={track['BplusE_vs_Bonly_final_corr']:.4f}"
              f"  rmse={track['BplusE_vs_Bonly_final_rmse']:.4f}")

    # ------------------------------------------------------------------
    # Cost
    # ------------------------------------------------------------------
    n_params_phase1 = sum(p.numel() for p in base_model.parameters())
    n_params_anchor = anchor_result["n_params"]
    print(f"\n  Cost:")
    print(f"    Phase 1 params: {n_params_phase1}")
    print(f"    Phase 2 params trained: {n_params_anchor}")
    print(f"    Phase 1 time: {phase1['train_time']:.1f}s")
    print(f"    Phase 2 time: {anchor_result['train_time']:.1f}s")

    # ------------------------------------------------------------------
    # Assemble results
    # ------------------------------------------------------------------
    results = {
        "_meta": {
            "experiment": "dynamic_theta_v1",
            "drift_sigma": drift_sigma,
            "n_learners": cfg.n_learners,
            "n_base": cfg.n_base,
            "n_ext": cfg.n_ext,
            "n_cats": cfg.n_cats,
            "n_anchor": n_anchor,
            "n_epochs": n_epochs,
            "lr": lr,
            "emb_dim": emb_dim,
            "hidden_dim": hidden_dim,
        },
        "true_drift": {
            "mean_net_drift": float(net_drift.mean()),
            "std_net_drift": float(net_drift.std()),
            "abs_mean_net_drift": float(np.abs(net_drift).mean()),
        },
        "phase1_trajectory_recovery": traj_p1,
        "new_item_recovery_anchoring": {
            "a_corr": float(a_corr), "a_rmse": float(a_rmse),
            "b_corr": float(b_corr), "b_rmse": float(b_rmse),
        },
        "tracking_test": track,
        "cost": {
            "phase1_params": n_params_phase1,
            "phase2_params_trained": n_params_anchor,
            "phase1_time_s": phase1["train_time"],
            "phase2_time_s": anchor_result["train_time"],
        },
    }

    return results


def write_summary(results: dict, path: str) -> None:
    lines = [
        "DYNAMIC THETA v1 -- SUMMARY",
        "=" * 70,
        "",
    ]

    meta = results["_meta"]
    lines += [
        f"Config: N={meta['n_learners']}  B={meta['n_base']}  E={meta['n_ext']}"
        f"  K={meta['n_cats']}  drift_sigma={meta['drift_sigma']}",
        f"Anchor learners: {meta['n_anchor']}",
        "",
    ]

    td = results["true_drift"]
    lines += [
        "True drift magnitude:",
        f"  mean net drift = {td['mean_net_drift']:.3f}"
        f"  std = {td['std_net_drift']:.3f}"
        f"  |mean| = {td['abs_mean_net_drift']:.3f}",
        "",
    ]

    p1 = results["phase1_trajectory_recovery"]
    lines += [
        "HEADLINE: Phase-1 trajectory recovery (encoder vs true theta_t over B steps)",
        f"  Pooled Pearson corr : {p1['pooled_corr']:.4f}",
        f"  Pooled RMSE         : {p1['pooled_rmse']:.4f}",
        f"  Net-drift corr      : {p1['net_drift_corr']:.4f}",
        f"  Net-drift RMSE      : {p1['net_drift_rmse']:.4f}",
        "",
    ]

    tr = results["tracking_test"]
    lines.append("HEADLINE: Tracking test (interleaved B+E, anchor learners)")
    if "error" in tr:
        lines.append(f"  ERROR: {tr['error']}")
    else:
        lines += [
            f"  B-only reference (pooled corr/rmse)      : "
            f"{tr['Bonly_pooled_corr']:.4f} / {tr['Bonly_pooled_rmse']:.4f}",
            f"  B-positions in B+E sequence (corr/rmse)  : "
            f"{tr['BplusE_B_pos_corr']:.4f} / {tr['BplusE_B_pos_rmse']:.4f}",
            f"  Final-step B+E vs B-only (corr/rmse)     : "
            f"{tr['BplusE_vs_Bonly_final_corr']:.4f} / {tr['BplusE_vs_Bonly_final_rmse']:.4f}",
        ]
    lines.append("")

    ni = results["new_item_recovery_anchoring"]
    lines += [
        "New-item recovery (E items, anchoring):",
        f"  a: corr={ni['a_corr']:.4f}  rmse={ni['a_rmse']:.4f}",
        f"  b: corr={ni['b_corr']:.4f}  rmse={ni['b_rmse']:.4f}",
        "",
    ]

    c = results["cost"]
    lines += [
        "Cost:",
        f"  Phase 1: {c['phase1_params']} params  {c['phase1_time_s']:.1f}s",
        f"  Phase 2: {c['phase2_params_trained']} params trained  {c['phase2_time_s']:.1f}s",
        "",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSummary written to {path}")


def main() -> None:
    device = torch.device("cpu")

    print("\n=== DYNAMIC THETA v1 ===")

    # Primary run: drift_sigma=0.12
    results = run_v1(drift_sigma=0.12, device=device)

    json_path = os.path.join(OUTPUT_DIR, "v1_dynamic_results.json")
    with open(json_path, "w") as f:
        json.dump(_clean(results), f, indent=2)
    print(f"\nResults written to {json_path}")

    summary_path = os.path.join(OUTPUT_DIR, "v1_dynamic_summary.txt")
    write_summary(results, summary_path)

    print("\n=== v1 DYNAMIC COMPLETE ===")


if __name__ == "__main__":
    main()
