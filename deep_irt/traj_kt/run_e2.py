"""run_e2.py -- E2: learning-curve rate recovery on real EdNet, full pipeline.

Usage (from repo root, research env):
    python -m deep_irt.traj_kt.run_e2

Writes results to deep_irt/traj_kt/ (RESULTS_E2.md, results_e2.json, plots).
Does NOT commit anything and does NOT write a merged CSV.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch

# Paths -----------------------------------------------------------------------
REPO_ROOT    = Path(__file__).parent.parent.parent
KT1_DIR      = REPO_ROOT / "EdNet-KT1" / "KT1"
CONTENTS_DIR = REPO_ROOT / "EdNet-Contents" / "contents"
OUT_DIR      = Path(__file__).parent


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_wall_start = time.time()
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[E2] device = {device}")

    # -----------------------------------------------------------------------
    # Step 1: Cohort
    # -----------------------------------------------------------------------
    from deep_irt.traj_kt.cohort import build_cohort

    cohort = build_cohort(
        kt1_dir=KT1_DIR,
        contents_dir=CONTENTS_DIR,
        top_n_files=7_000,
        t_min=200,
        verbose=True,
    )

    N = len(cohort.seqs_item)
    if N < 100:
        raise RuntimeError(
            f"Cohort too small ({N} learners). Increase top_n_files."
        )

    print(f"[E2] cohort ready: {N} learners, {cohort.n_items} items\n")

    # -----------------------------------------------------------------------
    # Step 2: Train and recover
    # -----------------------------------------------------------------------
    from deep_irt.traj_kt.train import train_and_recover

    # Cap seq len to 500 to stay within GPU memory for very long sequences
    model, theta_hat, r_hat = train_and_recover(
        seqs_item=cohort.seqs_item,
        seqs_resp=cohort.seqs_resp,
        n_items=cohort.n_items,
        device=device,
        n_epochs=300,
        batch_size=256,
        max_seq_len=500,
        lr=1e-2,
        verbose=True,
    )

    # Sanity: print a few example r_hat and curves
    print("\n[E2] === SANITY CHECK (first 5 learners with finite r_hat) ===")
    shown = 0
    for i in range(N):
        if shown >= 5:
            break
        if not np.isfinite(r_hat[i]):
            continue
        L = (theta_hat[i] != 0).sum()
        print(f"  learner {cohort.seqs_uid[i]:10s}  T={L:4d}  "
              f"r_hat={r_hat[i]:.4f}  "
              f"theta[0]={theta_hat[i,0]:.3f}  "
              f"theta[end]={theta_hat[i,L-1]:.3f}")
        shown += 1
    print()

    # Peak VRAM
    vram_gb = 0.0
    if device.type == "cuda":
        vram_gb = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"[E2] peak VRAM = {vram_gb:.2f} GB")

    # -----------------------------------------------------------------------
    # Step 3: Validation
    # -----------------------------------------------------------------------
    from deep_irt.traj_kt.validate import (
        predictive_validity,
        afm_concurrent,
        split_half_reliability,
        convergent_aligned_vs_responsive,
    )

    print("[E2] (a) predictive validity ...")
    pred_val = predictive_validity(
        seqs_item=cohort.seqs_item,
        seqs_resp=cohort.seqs_resp,
        seqs_corr=cohort.seqs_corr,
        model=model,
    )
    print(f"      rho={pred_val['rho']:.3f}  "
          f"95%CI=[{pred_val['ci_lo']:.3f}, {pred_val['ci_hi']:.3f}]  "
          f"n={pred_val['n']}")
    print(f"      neg-control rho={pred_val['neg_control_rho']:.3f}  "
          f"[{pred_val['neg_control_ci_lo']:.3f}, {pred_val['neg_control_ci_hi']:.3f}]")

    print("[E2] (b) AFM concurrent validity ...")
    # Recompute AFM slopes for scatter plot
    from deep_irt.traj_kt.validate import _afm_slope_per_learner
    afm_slopes = np.full(N, np.nan)
    for i in range(N):
        if np.isfinite(r_hat[i]):
            afm_slopes[i] = _afm_slope_per_learner(
                cohort.seqs_item[i], cohort.seqs_corr[i], cohort.seqs_part[i]
            )
    afm_val = afm_concurrent(
        seqs_item=cohort.seqs_item,
        seqs_corr=cohort.seqs_corr,
        seqs_part=cohort.seqs_part,
        r_hat=r_hat,
    )
    print(f"      rho={afm_val['rho']:.3f}  "
          f"95%CI=[{afm_val['ci_lo']:.3f}, {afm_val['ci_hi']:.3f}]  "
          f"n={afm_val['n']}")

    print("[E2] (c) split-half reliability ...")
    sh_val = split_half_reliability(
        seqs_item=cohort.seqs_item,
        seqs_resp=cohort.seqs_resp,
        model=model,
    )
    print(f"      rho={sh_val['rho']:.3f}  "
          f"95%CI=[{sh_val['ci_lo']:.3f}, {sh_val['ci_hi']:.3f}]  "
          f"n={sh_val['n']}")

    print("[E2] (d) convergent (aligned vs responsive theta) ...")
    conv_val = convergent_aligned_vs_responsive(
        seqs_item=cohort.seqs_item,
        seqs_resp=cohort.seqs_resp,
        model=model,
        r_hat_aligned=r_hat,
    )
    print(f"      rho={conv_val['rho']:.3f}  "
          f"95%CI=[{conv_val['ci_lo']:.3f}, {conv_val['ci_hi']:.3f}]  "
          f"n={conv_val['n']}")

    # -----------------------------------------------------------------------
    # Step 4: Plots and results
    # -----------------------------------------------------------------------
    from deep_irt.traj_kt.plots import (
        save_rate_distribution,
        save_predictive_scatter,
        save_afm_scatter,
    )

    print("\n[E2] saving plots ...")

    save_rate_distribution(r_hat, OUT_DIR)

    # Predictive scatter needs r_hat_first and delta_acc from the validation
    # Recompute r_hat_first and delta_acc arrays for the scatter
    r_hat_first, delta_acc = _recompute_pred_arrays(
        cohort.seqs_item, cohort.seqs_resp, cohort.seqs_corr, model
    )
    save_predictive_scatter(r_hat_first, delta_acc, pred_val["rho"], OUT_DIR)
    save_afm_scatter(afm_slopes, r_hat, afm_val["rho"], OUT_DIR)

    # -----------------------------------------------------------------------
    # Final report
    # -----------------------------------------------------------------------
    wall_total = time.time() - t_wall_start
    r_ok = r_hat[np.isfinite(r_hat)]
    frac_fit = len(r_ok) / N

    results = {
        "cohort": cohort.stats,
        "training": {
            "n_epochs": 300,
            "batch_size": 256,
            "max_seq_len": 500,
            "n_learners": N,
            "r_hat_frac_finite": float(frac_fit),
            "r_hat_median": float(np.median(r_ok)) if len(r_ok) else None,
            "r_hat_mean":   float(np.mean(r_ok))   if len(r_ok) else None,
        },
        "validation": {
            "a_predictive":   pred_val,
            "b_afm_concurrent": afm_val,
            "c_split_half":   sh_val,
            "d_convergent":   conv_val,
        },
        "wall_time_s": float(wall_total),
        "peak_vram_gb": float(vram_gb),
    }

    # Verdict
    primary_rho = pred_val["rho"]
    if np.isnan(primary_rho):
        verdict = "INCONCLUSIVE: predictive validity rho is NaN (fit failures)"
    elif primary_rho > 0.15 and pred_val["neg_control_rho"] < 0.05:
        verdict = (
            f"POSITIVE: recovered rate carries real learning signal "
            f"(rho={primary_rho:.3f}, neg-ctrl={pred_val['neg_control_rho']:.3f})"
        )
    elif primary_rho < 0.05:
        verdict = (
            f"NEGATIVE: recovered rate does not predict future accuracy gains "
            f"(rho={primary_rho:.3f}); noise floor likely dominates at T=200 cutoff"
        )
    else:
        verdict = (
            f"WEAK: modest correlation (rho={primary_rho:.3f}) -- "
            "marginal individual-difference signal, not robust"
        )

    results["verdict"] = verdict

    print(f"\n[E2] === FINAL RESULTS ===")
    print(f"  Wall time        : {wall_total/60:.1f} min")
    print(f"  Peak VRAM        : {vram_gb:.2f} GB")
    print(f"  Cohort           : {N} learners, {cohort.n_items} items")
    print(f"  r_hat fit rate   : {frac_fit*100:.1f}%")
    print(f"  (a) Predictive   : rho={pred_val['rho']:.3f}  "
          f"[{pred_val['ci_lo']:.3f},{pred_val['ci_hi']:.3f}]  "
          f"neg-ctrl={pred_val['neg_control_rho']:.3f}")
    print(f"  (b) AFM          : rho={afm_val['rho']:.3f}  "
          f"[{afm_val['ci_lo']:.3f},{afm_val['ci_hi']:.3f}]")
    print(f"  (c) Split-half   : rho={sh_val['rho']:.3f}  "
          f"[{sh_val['ci_lo']:.3f},{sh_val['ci_hi']:.3f}]")
    print(f"  (d) Convergent   : rho={conv_val['rho']:.3f}  "
          f"[{conv_val['ci_lo']:.3f},{conv_val['ci_hi']:.3f}]")
    print(f"\n  VERDICT: {verdict}\n")

    # Save JSON
    results_path = OUT_DIR / "results_e2.json"
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2, default=_json_default)
    print(f"[E2] results saved to {results_path}")

    # Write RESULTS_E2.md
    _write_results_md(OUT_DIR, results, cohort.stats, pred_val, afm_val,
                      sh_val, conv_val, wall_total, vram_gb, verdict, N,
                      frac_fit, r_ok)
    print(f"[E2] RESULTS_E2.md written")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _recompute_pred_arrays(
    seqs_item, seqs_resp, seqs_corr, model
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute r_hat_first and delta_acc arrays for plotting."""
    import torch
    from deep_irt.traj_synth.metrics import fit_rate

    N = len(seqs_item)
    r_hat_first = np.full(N, np.nan)
    delta_acc   = np.full(N, np.nan)
    device = model.device
    model.encoder.eval()

    for i in range(N):
        items_i = seqs_item[i]
        resp_i  = seqs_resp[i]
        corr_i  = seqs_corr[i]
        L = len(items_i)
        if L < 40:
            continue
        mid = L // 2

        items_h1 = torch.tensor(items_i[:mid], dtype=torch.long).unsqueeze(0).to(device)
        resp_h1  = torch.tensor(resp_i[:mid].astype(np.int64),
                                dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            th_h1, _ = model.encoder.aligned_theta_and_state(items_h1, resp_h1)
        th_np = th_h1.squeeze(0).cpu().numpy().astype(float)
        r, _, _ = fit_rate(th_np, robust=True, smooth=1,
                           t=np.arange(mid, dtype=float))
        r_hat_first[i] = r
        delta_acc[i]   = corr_i[mid:].mean() - corr_i[:mid].mean()

    return r_hat_first, delta_acc


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def _write_results_md(
    out_dir, results, cohort_stats, pred_val, afm_val,
    sh_val, conv_val, wall, vram, verdict, N, frac_fit, r_ok
) -> None:
    lines = [
        "# E2 Results: Learning-Curve Rate Recovery on EdNet",
        "",
        "## Cohort",
        f"- Learners (before T≥200 filter): {cohort_stats['n_learners_before_filter']}",
        f"- Learners (after  T≥200 filter): {cohort_stats['n_learners_after_filter']}",
        f"- Median sequence length: {cohort_stats['median_seq_len']:.0f}",
        f"- 90th pct sequence length: {cohort_stats['p90_seq_len']:.0f}",
        f"- Max sequence length: {cohort_stats['max_seq_len']}",
        f"- Item vocabulary size: {cohort_stats['item_vocab_size']}",
        f"- Bundle-collapsed rows: {cohort_stats['bundle_collapse_frac']*100:.2f}%",
        "",
        "## Training",
        f"- N learners: {N}",
        f"- r_hat fit success: {frac_fit*100:.1f}%",
        f"- r_hat median: {np.median(r_ok):.4f}" if len(r_ok) else "- r_hat median: N/A",
        f"- r_hat mean:   {np.mean(r_ok):.4f}"   if len(r_ok) else "- r_hat mean: N/A",
        "",
        "## Validation",
        "",
        "### (a) Predictive validity (primary)",
        f"- Spearman rho = {pred_val['rho']:.3f}  "
        f"95% CI [{pred_val['ci_lo']:.3f}, {pred_val['ci_hi']:.3f}]  "
        f"n = {pred_val['n']}",
        f"- Negative control rho = {pred_val['neg_control_rho']:.3f}  "
        f"[{pred_val['neg_control_ci_lo']:.3f}, {pred_val['neg_control_ci_hi']:.3f}]",
        "",
        "### (b) AFM concurrent validity (primary)",
        f"- Spearman rho = {afm_val['rho']:.3f}  "
        f"95% CI [{afm_val['ci_lo']:.3f}, {afm_val['ci_hi']:.3f}]  "
        f"n = {afm_val['n']}",
        "",
        "#### Part-stratified AFM breakdown",
    ]
    for part_key, pv in afm_val.get("part_stratified", {}).items():
        lines.append(
            f"- {part_key}: rho={pv['rho']:.3f}  "
            f"[{pv['ci_lo']:.3f}, {pv['ci_hi']:.3f}]  n={pv['n']}"
        )
    lines += [
        "",
        "### (c) Split-half reliability",
        f"- Spearman rho = {sh_val['rho']:.3f}  "
        f"95% CI [{sh_val['ci_lo']:.3f}, {sh_val['ci_hi']:.3f}]  "
        f"n = {sh_val['n']}",
        "",
        "### (d) Convergent (aligned vs responsive theta)",
        f"- Spearman rho = {conv_val['rho']:.3f}  "
        f"95% CI [{conv_val['ci_lo']:.3f}, {conv_val['ci_hi']:.3f}]  "
        f"n = {conv_val['n']}",
        "",
        "## Runtime",
        f"- Wall time: {wall/60:.1f} min",
        f"- Peak VRAM: {vram:.2f} GB",
        "",
        "## Verdict",
        "",
        verdict,
    ]
    (out_dir / "RESULTS_E2.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
