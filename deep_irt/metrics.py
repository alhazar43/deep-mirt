"""
metrics.py  --  compute all metric groups for the anchoring-drift experiment.

Group 1: New-item recovery (E items) -- anchoring vs baseline.
Group 2: Tracking consistency -- frozen encoder on interleaved B+E sequences.
          This is the real test: does theta stay coherent when E items (never
          seen during Phase 1 training) enter the frozen encoder's input?
Group 3: Scale stability -- B-item param drift between Phase 1 and extension.
Group 4: Cost -- trainable parameter count and wall-clock time.

trajectory_recovery (dynamic v1 only):
  Compare encoder's per-step theta_t to the true per-interaction theta from
  the data generator.  Reports pooled Pearson corr, RMSE across all
  (learner, timestep) pairs, and per-learner net-drift recovery.
"""

import numpy as np
import torch
from scipy.stats import pearsonr


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return float("nan")
    r, _ = pearsonr(x.ravel(), y.ravel())
    return float(r)


def _rmse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((x.ravel() - y.ravel()) ** 2)))


def tracking_consistency(dataset, phase1: dict, anchor_result: dict) -> dict:
    """
    The real tracking test for v0.1.

    After anchoring has fitted E-item embeddings, build an interleaved B+E
    sequence per shared (anchor) learner and run it through the FROZEN
    Phase-1 encoder (now extended with E-item embedding rows appended).

    Compares:
      - theta_BplusE  vs  theta_B_only  (same learner, frozen encoder)
      - theta_BplusE  vs  ground-truth theta

    Returns a dict with corr and rmse for each comparison.
    """
    extended_model = anchor_result.get("extended_model")
    if extended_model is None:
        return {"error": "extended_model not found in anchor_result"}

    device = next(extended_model.parameters()).device
    cfg = dataset.cfg
    B = cfg.n_base

    anchor_indices = np.where(dataset.anchor_mask)[0]
    # ext_seqs is ordered the same as anchor_indices
    ext_seq_map = {s["learner_idx"]: s for s in dataset.ext_seqs}

    # Build interleaved B+E sequences for each anchor learner.
    # "Interleaved" = concatenate all B interactions and all E interactions,
    # then shuffle them together using the same RNG seed per learner.
    rng = np.random.default_rng(cfg.seed + 999)  # separate from data-gen RNG

    seq_len = B + cfg.n_ext
    items_np = np.zeros((len(anchor_indices), seq_len), dtype=np.int64)
    resps_np = np.zeros((len(anchor_indices), seq_len), dtype=np.int64)

    for row, learner_i in enumerate(anchor_indices):
        base_seq = dataset.base_seqs[learner_i]
        ext_seq = ext_seq_map[learner_i]

        # base item IDs are already 0..B-1; shift ext IDs by B
        b_items = np.array(base_seq["items"], dtype=np.int64)
        b_resps = np.array(base_seq["responses"], dtype=np.int64)
        e_items = np.array(ext_seq["items"], dtype=np.int64) + B
        e_resps = np.array(ext_seq["responses"], dtype=np.int64)

        all_items = np.concatenate([b_items, e_items])
        all_resps = np.concatenate([b_resps, e_resps])

        # shuffle the combined sequence
        perm = rng.permutation(seq_len)
        items_np[row] = all_items[perm]
        resps_np[row] = all_resps[perm]

    items_t = torch.tensor(items_np, device=device)
    resps_t = torch.tensor(resps_np, device=device)

    extended_model.eval()
    with torch.no_grad():
        theta_BplusE = extended_model.get_final_theta(items_t, resps_t).cpu().numpy()

    # Phase-1 B-only theta for the same anchor learners
    theta_B_only = phase1["theta_base"][anchor_mask_to_rows(dataset.anchor_mask)]

    # Ground-truth theta for anchor learners
    theta_gt = dataset.gt.theta[anchor_mask_to_rows(dataset.anchor_mask)]

    return {
        "BplusE_vs_B_only": {
            "corr": _pearson(theta_BplusE, theta_B_only),
            "rmse": _rmse(theta_BplusE, theta_B_only),
        },
        "BplusE_vs_truth": {
            "corr": _pearson(theta_BplusE, theta_gt),
            "rmse": _rmse(theta_BplusE, theta_gt),
        },
        "B_only_vs_truth": {
            "corr": _pearson(theta_B_only, theta_gt),
            "rmse": _rmse(theta_B_only, theta_gt),
        },
    }


def anchor_mask_to_rows(anchor_mask: np.ndarray) -> np.ndarray:
    """Return integer indices where anchor_mask is True."""
    return np.where(anchor_mask)[0]


# ---------------------------------------------------------------------------
# Trajectory recovery metrics (dynamic v1 only)
# ---------------------------------------------------------------------------

def trajectory_recovery(dataset, model, device, label: str = "") -> dict:
    """
    Compare the encoder's per-step theta_t output to the true per-interaction
    theta stored in dataset.base_seqs[i]["theta_true"].

    This function is self-contained: it builds tensors, runs the frozen encoder,
    and computes all trajectory metrics.

    Parameters
    ----------
    dataset : Dataset with dynamic (random_walk) data.
              base_seqs[i]["theta_true"] must be present.
    model   : AbilityModel (or extended model) -- will be set to eval mode.
    device  : torch.device
    label   : optional string for labelling results

    Returns
    -------
    dict with:
      pooled_corr    -- Pearson r across all (learner, timestep) pairs
      pooled_rmse    -- RMSE across all (learner, timestep) pairs
      net_drift_corr -- corr of (recovered_end - recovered_start) vs
                        (true_end - true_start) per learner
      net_drift_rmse -- RMSE of the same per-learner scalar
      label          -- echoed label
    """
    cfg = dataset.cfg
    N, B = cfg.n_learners, cfg.n_base

    if not dataset.base_seqs[0].get("theta_true"):
        return {"error": "theta_true not present in base_seqs (static dataset)"}

    # Build tensors (all sequences same length B for base)
    items_np = np.array([s["items"] for s in dataset.base_seqs], dtype=np.int64)
    resps_np = np.array([s["responses"] for s in dataset.base_seqs], dtype=np.int64)
    # True per-interaction theta: (N, B)
    theta_true_np = np.array([s["theta_true"] for s in dataset.base_seqs],
                              dtype=np.float32)

    items_t = torch.tensor(items_np, device=device)
    resps_t = torch.tensor(resps_np, device=device)

    model.eval()
    with torch.no_grad():
        # encode returns (N, B) -- theta_t at each timestep
        theta_rec = model.encode(items_t, resps_t).cpu().numpy()  # (N, B)

    # Pooled metrics across all (N, B) pairs
    pooled_corr = _pearson(theta_rec.ravel(), theta_true_np.ravel())
    pooled_rmse = _rmse(theta_rec.ravel(), theta_true_np.ravel())

    # Net-drift recovery: per-learner (final - initial)
    true_net_drift = theta_true_np[:, -1] - theta_true_np[:, 0]   # (N,)
    rec_net_drift = theta_rec[:, -1] - theta_rec[:, 0]             # (N,)
    net_drift_corr = _pearson(rec_net_drift, true_net_drift)
    net_drift_rmse = _rmse(rec_net_drift, true_net_drift)

    return {
        "pooled_corr": pooled_corr,
        "pooled_rmse": pooled_rmse,
        "net_drift_corr": net_drift_corr,
        "net_drift_rmse": net_drift_rmse,
        "label": label,
    }


def trajectory_tracking_test(dataset, phase1_model, anchor_result,
                              device) -> dict:
    """
    The tracking test for the dynamic v1 experiment.

    For anchor learners: build an interleaved B+E sequence (as in v0.1),
    run it through the frozen extended model, and recover the FULL theta_t
    trajectory over the B+E steps.

    Compare:
      (a) BplusE trajectory vs true B-only trajectory (extended at B+E positions,
          true theta for B positions only; at E positions we use the anchor theta).
      (b) BplusE trajectory (B-only positions extracted) vs B-only trajectory
          from the phase-1 model  --  does tracking survive new items?

    Returns dict with:
      BplusE_B_positions_vs_truth   : corr/rmse (only the B-item timesteps in mixed seq)
      BplusE_full_vs_Bonly_final    : corr/rmse of final-step theta (B+E vs B-only)
      Bonly_pooled_corr/rmse        : reference trajectory recovery (B-only, from phase1)
      BplusE_B_positions_pooled_corr/rmse : trajectory recovery for B positions in interleaved seq
    """
    extended_model = anchor_result.get("extended_model")
    if extended_model is None:
        return {"error": "extended_model not found in anchor_result"}
    if not dataset.base_seqs[0].get("theta_true"):
        return {"error": "theta_true not present in base_seqs (static dataset)"}

    cfg = dataset.cfg
    B, E = cfg.n_base, cfg.n_ext

    anchor_indices = np.where(dataset.anchor_mask)[0]
    ext_seq_map = {s["learner_idx"]: s for s in dataset.ext_seqs}

    rng = np.random.default_rng(cfg.seed + 999)

    seq_len = B + E
    items_np = np.zeros((len(anchor_indices), seq_len), dtype=np.int64)
    resps_np = np.zeros((len(anchor_indices), seq_len), dtype=np.int64)
    # track which positions in the mixed sequence are B-items
    is_base_pos = np.zeros((len(anchor_indices), seq_len), dtype=bool)
    # true per-B-item theta for the anchor learners (B positions)
    theta_true_B = np.array([dataset.base_seqs[i]["theta_true"]
                              for i in anchor_indices], dtype=np.float32)  # (n_anc, B)

    for row, learner_i in enumerate(anchor_indices):
        base_seq = dataset.base_seqs[learner_i]
        ext_seq = ext_seq_map[learner_i]

        b_items = np.array(base_seq["items"], dtype=np.int64)
        b_resps = np.array(base_seq["responses"], dtype=np.int64)
        e_items = np.array(ext_seq["items"], dtype=np.int64) + B
        e_resps = np.array(ext_seq["responses"], dtype=np.int64)

        all_items = np.concatenate([b_items, e_items])
        all_resps = np.concatenate([b_resps, e_resps])
        # first B entries are base; last E are ext
        is_base = np.concatenate([np.ones(B, dtype=bool), np.zeros(E, dtype=bool)])

        perm = rng.permutation(seq_len)
        items_np[row] = all_items[perm]
        resps_np[row] = all_resps[perm]
        is_base_pos[row] = is_base[perm]

    items_t = torch.tensor(items_np, device=device)
    resps_t = torch.tensor(resps_np, device=device)

    extended_model.eval()
    with torch.no_grad():
        # Full trajectory through mixed B+E sequence: (n_anc, B+E)
        theta_traj_BplusE = extended_model.encode(items_t, resps_t).cpu().numpy()

    # Reference: B-only trajectory from phase-1 model (passed in as argument)
    # Get B-only trajectory for anchor learners
    base_items_t = torch.tensor(
        np.array([dataset.base_seqs[i]["items"] for i in anchor_indices], dtype=np.int64),
        device=device)
    base_resps_t = torch.tensor(
        np.array([dataset.base_seqs[i]["responses"] for i in anchor_indices], dtype=np.int64),
        device=device)

    with torch.no_grad():
        theta_traj_Bonly = phase1_model.encode(
            base_items_t, base_resps_t).cpu().numpy()  # (n_anc, B)

    # For each anchor learner, extract the B-positions from the mixed sequence
    # These are scattered through the B+E steps; flatten with their true thetas.
    # We need to align: the t-th B-item in the mixed seq has a reconstructed theta
    # at position perm_pos in the mixed seq and a true theta at its original B-position.
    # For simplicity: for each anchor learner, extract recovered theta at B-positions
    # in the mixed sequence and compare to the true theta at those same B interactions.
    #
    # true theta for position k in mixed seq = theta_true_B[row, original_B_position]
    # But "original_B_position" is the t-th B-item encountered in the shuffled base_seq,
    # which is already stored in theta_true_B in B-item encounter order.
    # In the mixed sequence, B items land at the positions is_base_pos[row]==True.
    # There are exactly B such positions per row.  We collect recovered theta there
    # and compare to theta_true_B in the order the B items appear in the mixed seq.
    # Because theta_true_B[row] is in the original item_order from base_seqs,
    # and the mixed permutation preserves that order among B items, we can just
    # take them in the order they appear.

    rec_B_in_mixed = []  # (n_anc, B) recovered theta at B-item positions in mixed seq
    for row in range(len(anchor_indices)):
        pos_b = np.where(is_base_pos[row])[0]      # positions of B items in mixed seq
        pos_b_sorted = np.sort(pos_b)               # in time order
        rec_B_in_mixed.append(theta_traj_BplusE[row, pos_b_sorted])
    rec_B_in_mixed = np.array(rec_B_in_mixed, dtype=np.float32)  # (n_anc, B)

    # Pooled trajectory recovery: B-only (reference)
    Bonly_pooled_corr = _pearson(theta_traj_Bonly.ravel(), theta_true_B.ravel())
    Bonly_pooled_rmse = _rmse(theta_traj_Bonly.ravel(), theta_true_B.ravel())

    # Pooled trajectory recovery: B-positions in mixed B+E sequence
    BplusE_B_pos_corr = _pearson(rec_B_in_mixed.ravel(), theta_true_B.ravel())
    BplusE_B_pos_rmse = _rmse(rec_B_in_mixed.ravel(), theta_true_B.ravel())

    # Final-step theta comparison: B+E vs B-only (does the final estimate agree?)
    theta_final_BplusE = theta_traj_BplusE[:, -1]
    theta_final_Bonly = theta_traj_Bonly[:, -1]
    final_corr = _pearson(theta_final_BplusE, theta_final_Bonly)
    final_rmse = _rmse(theta_final_BplusE, theta_final_Bonly)

    return {
        "Bonly_pooled_corr": Bonly_pooled_corr,
        "Bonly_pooled_rmse": Bonly_pooled_rmse,
        "BplusE_B_pos_corr": BplusE_B_pos_corr,
        "BplusE_B_pos_rmse": BplusE_B_pos_rmse,
        "BplusE_vs_Bonly_final_corr": final_corr,
        "BplusE_vs_Bonly_final_rmse": final_rmse,
    }


def compute_all(dataset,
                phase1: dict,
                anchor_result: dict,
                baseline_result: dict,
                base_model=None) -> dict:
    """
    Compute all metrics and return a nested dict.

    dataset:         Dataset (has .gt, .anchor_mask)
    phase1:          output of train_base()
    anchor_result:   output of anchor_extend()
    baseline_result: output of train_baseline()
    base_model:      the frozen AbilityModel from Phase 1 (for theta consistency)
    """
    gt = dataset.gt
    anchor_mask = dataset.anchor_mask

    # -----------------------------------------------------------------------
    # Group 1: New-item recovery (E items)
    # -----------------------------------------------------------------------
    # discrimination a
    anc_a_corr = _pearson(anchor_result["a_ext"], gt.a_ext)
    anc_a_rmse = _rmse(anchor_result["a_ext"], gt.a_ext)
    bas_a_corr = _pearson(baseline_result["a_ext"], gt.a_ext)
    bas_a_rmse = _rmse(baseline_result["a_ext"], gt.a_ext)

    # step thresholds b (flatten K-1 per item)
    anc_b_corr = _pearson(anchor_result["b_ext"].ravel(), gt.b_ext.ravel())
    anc_b_rmse = _rmse(anchor_result["b_ext"].ravel(), gt.b_ext.ravel())
    bas_b_corr = _pearson(baseline_result["b_ext"].ravel(), gt.b_ext.ravel())
    bas_b_rmse = _rmse(baseline_result["b_ext"].ravel(), gt.b_ext.ravel())

    group1 = {
        "anchoring": {
            "a_corr": anc_a_corr, "a_rmse": anc_a_rmse,
            "b_corr": anc_b_corr, "b_rmse": anc_b_rmse,
        },
        "baseline": {
            "a_corr": bas_a_corr, "a_rmse": bas_a_rmse,
            "b_corr": bas_b_corr, "b_rmse": bas_b_rmse,
        },
    }

    # -----------------------------------------------------------------------
    # Group 2: Tracking consistency (genuine test -- v0.1)
    # -----------------------------------------------------------------------
    # Run frozen Phase-1 encoder (with E-item embeddings appended) on
    # interleaved B+E sequences for each anchor learner.  This is the real
    # test: the encoder never saw E-item embeddings during training.
    group2 = tracking_consistency(dataset, phase1, anchor_result)

    # -----------------------------------------------------------------------
    # Group 3: Scale stability -- drift of B-item params and theta
    # -----------------------------------------------------------------------
    # Under anchoring: Phase 1 params vs Phase 2 params for B items.
    # Phase 2 does NOT retrain B items, so drift should be ~0 by construction.
    # Under baseline: compare baseline B-item params to Phase 1 params.
    #
    # a drift
    a_drift_anc_corr = _pearson(phase1["a_base"], phase1["a_base"])  # trivially 1.0
    a_drift_anc_rmse = 0.0   # frozen by construction

    a_drift_bas_corr = _pearson(phase1["a_base"], baseline_result["a_base"])
    a_drift_bas_rmse = _rmse(phase1["a_base"], baseline_result["a_base"])

    # b drift (flatten)
    b_drift_anc_rmse = 0.0   # frozen by construction
    b_drift_bas_rmse = _rmse(phase1["b_base"].ravel(),
                              baseline_result["b_base"].ravel())
    b_drift_bas_corr = _pearson(phase1["b_base"].ravel(),
                                 baseline_result["b_base"].ravel())

    # theta drift (all learners, B-only theta)
    theta_drift_anc_rmse = 0.0  # frozen encoder -> same output by construction
    theta_drift_bas_rmse = _rmse(phase1["theta_base"],
                                  baseline_result["theta_base_only"])
    theta_drift_bas_corr = _pearson(phase1["theta_base"],
                                     baseline_result["theta_base_only"])

    group3 = {
        "anchoring": {
            "a_rmse":     a_drift_anc_rmse,
            "b_rmse":     b_drift_anc_rmse,
            "theta_rmse": theta_drift_anc_rmse,
            "note": "all zero by construction (frozen)",
        },
        "baseline": {
            "a_corr":     a_drift_bas_corr,
            "a_rmse":     a_drift_bas_rmse,
            "b_corr":     b_drift_bas_corr,
            "b_rmse":     b_drift_bas_rmse,
            "theta_corr": theta_drift_bas_corr,
            "theta_rmse": theta_drift_bas_rmse,
        },
    }

    # -----------------------------------------------------------------------
    # Group 4: Cost
    # -----------------------------------------------------------------------
    group4 = {
        "anchoring": {
            "n_params":      anchor_result["n_params"],
            "time_seconds":  anchor_result["train_time"],
        },
        "baseline": {
            "n_params":      baseline_result["n_params"],
            "time_seconds":  baseline_result["train_time"],
        },
    }

    return {
        "group1_new_item_recovery": group1,
        "group2_theta_consistency": group2,
        "group3_scale_stability":   group3,
        "group4_cost":              group4,
    }


def print_table(results: dict) -> None:
    """Print a readable results table to stdout."""
    sep = "-" * 70

    print(sep)
    print("GROUP 1: NEW-ITEM RECOVERY (E extension items)")
    print(sep)
    g1 = results["group1_new_item_recovery"]
    for method in ("anchoring", "baseline"):
        m = g1[method]
        print(f"  {method:12s}  a_corr={m['a_corr']:.4f}  a_rmse={m['a_rmse']:.4f}"
              f"  b_corr={m['b_corr']:.4f}  b_rmse={m['b_rmse']:.4f}")

    print()
    print(sep)
    print("GROUP 2: TRACKING CONSISTENCY (frozen encoder on interleaved B+E sequences)")
    print(sep)
    g2 = results["group2_theta_consistency"]
    if "error" in g2:
        print(f"  ERROR: {g2['error']}")
    else:
        print(f"  theta_BplusE vs B-only (frozen encoder)  :"
              f"  corr={g2['BplusE_vs_B_only']['corr']:.4f}"
              f"  rmse={g2['BplusE_vs_B_only']['rmse']:.4f}")
        print(f"  theta_BplusE vs ground truth             :"
              f"  corr={g2['BplusE_vs_truth']['corr']:.4f}"
              f"  rmse={g2['BplusE_vs_truth']['rmse']:.4f}")
        print(f"  theta_B_only vs ground truth (reference) :"
              f"  corr={g2['B_only_vs_truth']['corr']:.4f}"
              f"  rmse={g2['B_only_vs_truth']['rmse']:.4f}")

    print()
    print(sep)
    print("GROUP 3: SCALE STABILITY (B-item params and theta drift after extension)")
    print(sep)
    g3 = results["group3_scale_stability"]
    print(f"  Anchoring : a_rmse={g3['anchoring']['a_rmse']:.4f}"
          f"  b_rmse={g3['anchoring']['b_rmse']:.4f}"
          f"  theta_rmse={g3['anchoring']['theta_rmse']:.4f}  ({g3['anchoring']['note']})")
    bas = g3["baseline"]
    print(f"  Baseline  : a_corr={bas['a_corr']:.4f}  a_rmse={bas['a_rmse']:.4f}"
          f"  b_corr={bas['b_corr']:.4f}  b_rmse={bas['b_rmse']:.4f}"
          f"  theta_corr={bas['theta_corr']:.4f}  theta_rmse={bas['theta_rmse']:.4f}")

    print()
    print(sep)
    print("GROUP 4: COST")
    print(sep)
    g4 = results["group4_cost"]
    for method in ("anchoring", "baseline"):
        m = g4[method]
        print(f"  {method:12s}  params={m['n_params']:6d}  time={m['time_seconds']:.1f}s")
    print(sep)
