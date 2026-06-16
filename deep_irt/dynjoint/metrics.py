"""
metrics.py -- Spearman rank recovery metrics for the dynamic-encoder joint experiment.

Three metric groups:
1. Cross-format transfer (the key question from jointfmt, now with dynamic theta).
2. Dynamic tracking (the NEW question: does the encoder recover the moving ability?).
3. Sanity checks (each model on its own items).
"""

from __future__ import annotations

import torch
import numpy as np
from scipy import stats as sp_stats


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    """Spearman rank correlation between two 1-D tensors."""
    xn = x.detach().float().numpy()
    yn = y.detach().float().numpy()
    r, _ = sp_stats.spearmanr(xn, yn)
    return float(r)


def align_sign(recovered: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Flip sign of recovered if it negatively correlates with reference."""
    xn = recovered.detach().float().numpy()
    yn = reference.detach().float().numpy()
    r, _ = sp_stats.pearsonr(xn, yn)
    return -recovered if r < 0 else recovered


def compute_metrics(
    ground_truth: dict,
    mode_a: dict,
    joint_model,
    joint_strength: torch.Tensor,
    joint_gpcm_difficulty: torch.Tensor,
    indep_gpcm_difficulty: torch.Tensor,
    indep_bt_strength: torch.Tensor,
) -> dict:
    """
    Compute all recovery and cross-format metrics.

    Parameters
    ----------
    ground_truth         : from data_gen.generate_ground_truth()
    mode_a               : from data_gen.generate_mode_a_dynamic()
    joint_model          : trained DynJointModel
    joint_strength       : (n_items,)
    joint_gpcm_difficulty: (n_items,)
    indep_gpcm_difficulty: (n_items,)
    indep_bt_strength    : (n_items,)

    Returns
    -------
    dict of named float metrics
    """
    s_true = ground_truth["s"]
    do_idx = ground_truth["direct_only_idx"]
    po_idx = ground_truth["pairwise_only_idx"]
    both_idx = ground_truth["both_idx"]

    # Sign-align globally before slicing
    js = align_sign(joint_strength, s_true)
    jg = align_sign(joint_gpcm_difficulty, s_true)
    ig = align_sign(indep_gpcm_difficulty, s_true)
    ib = align_sign(indep_bt_strength, s_true)

    # -------------------------------------------------------------------
    # 1. Cross-format transfer (mirrors jointfmt metrics)
    # -------------------------------------------------------------------
    joint_overall = spearman(js, s_true)

    joint_gpcm_on_pairwise_only = spearman(jg[po_idx], s_true[po_idx])
    indep_gpcm_pairwise_only = spearman(ig[po_idx], s_true[po_idx])
    pairwise_only_transfer_gap = joint_gpcm_on_pairwise_only - indep_gpcm_pairwise_only

    joint_bt_on_direct_only = spearman(js[do_idx], s_true[do_idx])
    indep_bt_direct_only = spearman(ib[do_idx], s_true[do_idx])
    direct_only_transfer_gap = joint_bt_on_direct_only - indep_bt_direct_only

    indep_gpcm_direct_only = spearman(ig[do_idx], s_true[do_idx])
    indep_bt_pairwise_only = spearman(ib[po_idx], s_true[po_idx])
    joint_both_bt = spearman(js[both_idx], s_true[both_idx])
    joint_both_gpcm = spearman(jg[both_idx], s_true[both_idx])
    indep_gpcm_overall = spearman(ig, s_true)
    indep_bt_overall = spearman(ib, s_true)

    # -------------------------------------------------------------------
    # 2. Dynamic tracking: net-drift Spearman
    #
    # The HONEST dynamic metric: within-learner theta_T - theta_1,
    # recovered vs true.  Pooled Pearson of trajectories would be
    # confounded by inter-person differences; net drift isolates whether
    # the encoder tracks CHANGE within a person.
    # -------------------------------------------------------------------
    seq_item_ids = mode_a["seq_item_ids"]
    seq_responses = mode_a["seq_responses"]

    joint_model.eval()
    with torch.no_grad():
        # Full theta trajectory: (N, T)
        theta_seq, _ = joint_model.encoder.encode(seq_item_ids, seq_responses)
        theta_seq = theta_seq.squeeze(-1)  # (N, T)

    theta_final_rec = theta_seq[:, -1]   # (N,)
    theta_init_rec = theta_seq[:, 0]     # (N,)
    net_drift_rec = theta_final_rec - theta_init_rec

    net_drift_true = mode_a["net_drift"]

    # Align sign of net_drift_rec
    drift_corr = float(sp_stats.pearsonr(
        net_drift_rec.numpy(), net_drift_true.numpy()
    )[0])
    if drift_corr < 0:
        net_drift_rec = -net_drift_rec

    dyn_net_drift_spearman = spearman(net_drift_rec, net_drift_true)

    # Also report pooled correlation of final theta vs true final theta
    # (less discriminative but familiar)
    theta_final_true = mode_a["theta_true_final"]
    final_corr = float(sp_stats.pearsonr(
        theta_final_rec.numpy(), theta_final_true.numpy()
    )[0])
    if final_corr < 0:
        theta_final_rec = -theta_final_rec
    dyn_final_spearman = spearman(theta_final_rec, theta_final_true)

    return {
        # Overall item recovery
        "joint_overall": joint_overall,
        "indep_gpcm_overall": indep_gpcm_overall,
        "indep_bt_overall": indep_bt_overall,
        # Cross-format transfer (the principal claim)
        "joint_gpcm_on_pairwise_only": joint_gpcm_on_pairwise_only,
        "indep_gpcm_pairwise_only": indep_gpcm_pairwise_only,
        "pairwise_only_transfer_gap": pairwise_only_transfer_gap,
        "joint_bt_on_direct_only": joint_bt_on_direct_only,
        "indep_bt_direct_only": indep_bt_direct_only,
        "direct_only_transfer_gap": direct_only_transfer_gap,
        # Sanity
        "indep_gpcm_direct_only": indep_gpcm_direct_only,
        "indep_bt_pairwise_only": indep_bt_pairwise_only,
        "joint_both_bt": joint_both_bt,
        "joint_both_gpcm": joint_both_gpcm,
        # Dynamic tracking
        "dyn_net_drift_spearman": dyn_net_drift_spearman,
        "dyn_final_spearman": dyn_final_spearman,
    }
