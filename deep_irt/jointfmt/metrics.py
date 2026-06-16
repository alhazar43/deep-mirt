"""
metrics.py -- Spearman rank recovery metrics for the joint-format experiment.
"""

import torch
from scipy import stats as sp_stats


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    """Spearman rank correlation between two 1-D tensors."""
    x_np = x.detach().float().numpy()
    y_np = y.detach().float().numpy()
    r, _ = sp_stats.spearmanr(x_np, y_np)
    return float(r)


def align_sign(recovered: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Flip sign of recovered if it negatively correlates with reference."""
    x = recovered.detach().float().numpy()
    y = reference.detach().float().numpy()
    r, _ = sp_stats.pearsonr(x, y)
    if r < 0:
        return -recovered
    return recovered


def compute_metrics(
    ground_truth: dict,
    joint_strength: torch.Tensor,
    joint_gpcm_difficulty: torch.Tensor,
    indep_gpcm_difficulty: torch.Tensor,
    indep_bt_strength: torch.Tensor,
) -> dict:
    """
    Compute all recovery and cross-format metrics.

    Cross-format transfer logic
    ---------------------------
    For PAIRWISE-ONLY items (never in GPCM data):
      - BT training shaped their embedding via the joint BT head.
      - The GPCM head can now READ those BT-trained embeddings to infer
        their position on the ordinal scale.
      - Transfer signal: joint_gpcm_difficulty[pairwise_only_idx] vs s_true.
      - Baseline: indep_gpcm has never seen these items -> noise.

    For DIRECT-ONLY items (never in BT data):
      - GPCM training shaped their embedding via the joint GPCM head.
      - The BT head can now READ those GPCM-trained embeddings to infer
        their pairwise strength.
      - Transfer signal: joint_strength[direct_only_idx] vs s_true.
      - Baseline: indep_bt has never seen these items -> noise.

    Parameters
    ----------
    ground_truth          : from data_gen.generate_ground_truth()
    joint_strength        : (n_items,)  -- BT head strength from joint model
    joint_gpcm_difficulty : (n_items,)  -- GPCM head difficulty from joint model
    indep_gpcm_difficulty : (n_items,)  -- difficulty from independent GPCM
    indep_bt_strength     : (n_items,)  -- strength from independent BT

    Returns
    -------
    dict of named Spearman correlations.
    """
    s_true = ground_truth["s"]
    do_idx = ground_truth["direct_only_idx"]    # DIRECT-ONLY items
    po_idx = ground_truth["pairwise_only_idx"]  # PAIRWISE-ONLY items
    both_idx = ground_truth["both_idx"]         # BOTH items

    # Align signs globally before slicing
    js = align_sign(joint_strength, s_true)
    jg = align_sign(joint_gpcm_difficulty, s_true)
    ig = align_sign(indep_gpcm_difficulty, s_true)
    ib = align_sign(indep_bt_strength, s_true)

    # -------------------------------------------------------------------
    # Joint model: overall Spearman (using BT strength as common currency)
    # -------------------------------------------------------------------
    joint_overall = spearman(js, s_true)

    # -------------------------------------------------------------------
    # CROSS-FORMAT TRANSFER: the principal claim
    #
    # 1. Pairwise-only transfer:
    #    BT trained embedding -> GPCM head reads it -> ordinal scale placement.
    #    joint_gpcm_difficulty for pairwise-only items vs s_true.
    #    Baseline: indep_gpcm[pairwise_only] is noise (never saw those items).
    #
    # 2. Direct-only transfer (symmetric):
    #    GPCM trained embedding -> BT head reads it -> pairwise strength.
    #    joint_strength for direct-only items vs s_true.
    #    Baseline: indep_bt[direct_only] is noise (never saw those items).
    # -------------------------------------------------------------------
    # Pairwise-only: GPCM head reads BT-trained embeddings
    joint_gpcm_on_pairwise_only = spearman(jg[po_idx], s_true[po_idx])
    indep_gpcm_pairwise_only = spearman(ig[po_idx], s_true[po_idx])   # noise
    pairwise_only_gap = joint_gpcm_on_pairwise_only - indep_gpcm_pairwise_only

    # Direct-only: BT head reads GPCM-trained embeddings
    joint_bt_on_direct_only = spearman(js[do_idx], s_true[do_idx])
    indep_bt_direct_only = spearman(ib[do_idx], s_true[do_idx])       # noise
    direct_only_gap = joint_bt_on_direct_only - indep_bt_direct_only

    # -------------------------------------------------------------------
    # Individual model performance on their OWN items (sanity check)
    # -------------------------------------------------------------------
    indep_gpcm_direct_only = spearman(ig[do_idx], s_true[do_idx])
    indep_bt_pairwise_only = spearman(ib[po_idx], s_true[po_idx])

    # Joint model on BOTH items (both heads trained on these)
    joint_both_bt = spearman(js[both_idx], s_true[both_idx])
    joint_both_gpcm = spearman(jg[both_idx], s_true[both_idx])

    # Indep overall
    indep_gpcm_overall = spearman(ig, s_true)
    indep_bt_overall = spearman(ib, s_true)

    return {
        # Overall
        "joint_overall": joint_overall,
        "indep_gpcm_overall": indep_gpcm_overall,
        "indep_bt_overall": indep_bt_overall,
        # Cross-format transfer (the key numbers)
        "joint_gpcm_on_pairwise_only": joint_gpcm_on_pairwise_only,
        "indep_gpcm_pairwise_only": indep_gpcm_pairwise_only,
        "pairwise_only_transfer_gap": pairwise_only_gap,
        "joint_bt_on_direct_only": joint_bt_on_direct_only,
        "indep_bt_direct_only": indep_bt_direct_only,
        "direct_only_transfer_gap": direct_only_gap,
        # Sanity: each model on its own items
        "indep_gpcm_direct_only": indep_gpcm_direct_only,
        "indep_bt_pairwise_only": indep_bt_pairwise_only,
        # Joint on BOTH
        "joint_both_bt": joint_both_bt,
        "joint_both_gpcm": joint_both_gpcm,
    }
