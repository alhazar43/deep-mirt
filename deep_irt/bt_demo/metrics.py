"""
Metrics for the Bradley-Terry / GPCM bridge demo.

All correlations align sign so that positive = agreement with ground truth.
"""

import torch
from scipy import stats as sp_stats


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    """Spearman rank correlation between two 1-D tensors."""
    x_np = x.detach().float().numpy()
    y_np = y.detach().float().numpy()
    r, _ = sp_stats.spearmanr(x_np, y_np)
    return float(r)


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson correlation between two 1-D tensors."""
    x_np = x.detach().float().numpy()
    y_np = y.detach().float().numpy()
    r, _ = sp_stats.pearsonr(x_np, y_np)
    return float(r)


def align_sign(recovered: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """
    Flip sign of `recovered` if it is negatively correlated with `reference`.
    Returns the (possibly sign-flipped) tensor.
    """
    r = pearson(recovered, reference)
    if r < 0:
        return -recovered
    return recovered


def compute_all_metrics(
    ground_truth: dict,
    gpcm_results: dict,
    bt_results: dict,
) -> dict:
    """
    Compute all recovery and cross-format metrics.

    Returns
    -------
    dict with keys:
        mode_a_vs_truth_spearman
        mode_a_vs_truth_pearson
        mode_b_vs_truth_spearman
        mode_b_vs_truth_pearson
        mode_ab_spearman   (key result: cross-format agreement)
        mode_ab_pearson
    """
    s_true = ground_truth["s"]                        # (n_items,)
    d_a = gpcm_results["difficulty"]                  # (n_items,) GPCM difficulty
    s_b = bt_results["strength"]                      # (n_items,) BT strength

    # Align signs to ground truth before computing cross-format metric
    d_a = align_sign(d_a, s_true)
    s_b = align_sign(s_b, s_true)

    # Mode A vs truth
    a_sp = spearman(d_a, s_true)
    a_pe = pearson(d_a, s_true)

    # Mode B vs truth
    b_sp = spearman(s_b, s_true)
    b_pe = pearson(s_b, s_true)

    # Cross-format: Mode A difficulty vs Mode B strength
    # Both are already aligned to s_true, so sign convention matches.
    ab_sp = spearman(d_a, s_b)
    ab_pe = pearson(d_a, s_b)

    return {
        "mode_a_vs_truth_spearman": a_sp,
        "mode_a_vs_truth_pearson": a_pe,
        "mode_b_vs_truth_spearman": b_sp,
        "mode_b_vs_truth_pearson": b_pe,
        "mode_ab_spearman": ab_sp,
        "mode_ab_pearson": ab_pe,
    }


def format_report(
    ground_truth: dict,
    mode_a_data: dict,
    mode_b_data: dict,
    metrics: dict,
    gpcm_results: dict,
    bt_results: dict,
) -> str:
    """Build a human-readable results report."""
    lines = [
        "=" * 60,
        "Bradley-Terry / GPCM Bridge Demo",
        "Format-agnostic one-scale recovery",
        "=" * 60,
        "",
        "DATA SUMMARY",
        f"  Items (N):              {ground_truth['n_items']}",
        f"  GPCM categories (K):   {ground_truth['K']}",
        f"  Mode-A respondents:    {mode_a_data['n_respondents']}",
        f"  Mode-A observations:   {len(mode_a_data['responses'])}",
        f"  Mode-B pairs:          {mode_b_data['n_pairs']}",
        "",
        "SIGN CONVENTION",
        "  Higher s_i = harder / stronger item.",
        "  Mode A: higher difficulty d_i -> respondents score lower.",
        "  Mode B: item i beats item j with prob sigmoid(s_i - s_j).",
        "  Both recovery outputs are sign-aligned to s_true before",
        "  computing correlations.",
        "",
        "RECOVERY vs GROUND TRUTH",
        f"  Mode A (GPCM) difficulty vs true s:",
        f"    Spearman r = {metrics['mode_a_vs_truth_spearman']:.4f}",
        f"    Pearson  r = {metrics['mode_a_vs_truth_pearson']:.4f}",
        f"  Mode B (BT)   strength   vs true s:",
        f"    Spearman r = {metrics['mode_b_vs_truth_spearman']:.4f}",
        f"    Pearson  r = {metrics['mode_b_vs_truth_pearson']:.4f}",
        "",
        "CROSS-FORMAT AGREEMENT (Key Result)",
        "  Mode-A GPCM difficulty  vs  Mode-B BT strength",
        f"    Spearman r = {metrics['mode_ab_spearman']:.4f}",
        f"    Pearson  r = {metrics['mode_ab_pearson']:.4f}",
        "",
        "FINAL NLL",
        f"  GPCM fit NLL:  {gpcm_results['final_nll']:.4f}",
        f"  BT fit NLL:    {bt_results['final_nll']:.4f}",
        "",
        "VERDICT",
    ]

    ab_sp = metrics["mode_ab_spearman"]
    ab_pe = metrics["mode_ab_pearson"]

    if ab_sp >= 0.90 and ab_pe >= 0.90:
        verdict = (
            "PASS. Both response formats recover the same latent scale "
            f"(Spearman={ab_sp:.3f}, Pearson={ab_pe:.3f}). "
            "Direct ordinal responses and pairwise comparisons are "
            "measurements of a single coherent item dimension."
        )
    elif ab_sp >= 0.75 and ab_pe >= 0.75:
        verdict = (
            f"PARTIAL. Moderate cross-format agreement "
            f"(Spearman={ab_sp:.3f}, Pearson={ab_pe:.3f}). "
            "The shared scale is visible but recovery noise is non-trivial."
        )
    else:
        verdict = (
            f"FAIL. Weak cross-format agreement "
            f"(Spearman={ab_sp:.3f}, Pearson={ab_pe:.3f}). "
            "Investigate identifiability, data volume, or fitting bugs."
        )

    lines.append(f"  {verdict}")
    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)
