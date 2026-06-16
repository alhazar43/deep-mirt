"""
run_experiment.py -- Dynamic-encoder x shared-multi-head decoder experiment.

Tests whether a DYNAMIC theta (LSTM encoder) + ONE shared item embedding + TWO
heads (GPCM + BT) can simultaneously:
  1. Preserve cross-format transfer (pairwise-only items on ordinal scale, and
     vice versa) -- the result established in jointfmt with static theta.
  2. Track MOVING learner ability -- the new capability that static theta cannot
     claim.

Run from repo root:
    export KMP_DUPLICATE_LIB_OK=TRUE
    export PYTHONPATH=.
    python deep_irt/dynjoint/run_experiment.py

or via bash shortcut (Windows Anaconda):
    source ~/anaconda3/etc/profile.d/conda.sh && conda activate research
    export KMP_DUPLICATE_LIB_OK=TRUE && export PYTHONPATH=.
    python deep_irt/dynjoint/run_experiment.py
"""

import os
import sys
import json
import time
import argparse

import torch
import numpy as np

# Path bootstrap: allow running from repo root or from deep_irt/dynjoint/
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
for p in [REPO_ROOT, os.path.join(REPO_ROOT, "deep_irt")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from deep_irt.dynjoint.data_gen import (
    generate_ground_truth,
    generate_mode_a_dynamic,
    generate_mode_b,
)
from deep_irt.dynjoint.train import (
    train_joint,
    train_joint_balanced,
    train_indep_gpcm,
    train_indep_bt,
)
from deep_irt.dynjoint.metrics import compute_metrics

SEED = 0
OUTPUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# Recommended REBALANCED config (P1.7).
#
# Empirical finding (3-seed sweep, see outputs/comparison.txt): the P1.6
# direct-only->BT degradation is GPCM OVER-TRAINING of the shared d_i scale,
# NOT primarily a per-head loss imbalance. The measured ~6x GPCM-vs-BT
# gradient ratio on the shared trunk is real, but per-head loss weighting
# alone (fixed / uncertainty / gradnorm at 400 epochs) does NOT lift
# direct-only->BT -- it slightly hurts it while helping the BOTH bridge.
#
# The dominant lever is training duration: stopping at 150 epochs lifts
# direct-only->BT from 0.873 to 0.954 (3-seed mean), keeps pairwise-only->GPCM
# flat at 0.977, and IMPROVES net-drift tracking (0.433 -> 0.465). A mild fixed
# BT up-weight (w_bt=3) layered on top of early stopping is the best single
# config: it adds a consistent net-drift gain (0.465 -> 0.502) and a stronger
# BOTH bridge at no cost to direct-only->BT. So loss-balancing is a useful
# SECONDARY lever -- once the over-training is removed by early stopping.
# -------------------------------------------------------------------------
REBAL_EPOCHS = 150
REBAL_BALANCE = "fixed"
REBAL_BT_WEIGHT = 3.0
REBAL_BT_PAIRS_MULT = 1.0
REBAL_HEAD_LR_MULT = 1.0
REBAL_WARMUP_FRAC = 0.0

# jointfmt static-theta reference values (for honest comparison)
JOINTFMT_PAIRWISE_ONLY = 0.99  # Spearman: joint GPCM on pairwise-only items
JOINTFMT_DIRECT_ONLY = 0.98    # Spearman: joint BT on direct-only items


def format_report(ground_truth, mode_a, mode_b, metrics, params, wall_time) -> str:
    n = ground_truth["n_items"]
    n_do = ground_truth["n_direct_only"]
    n_po = ground_truth["n_pairwise_only"]
    n_both = ground_truth["n_both"]
    m = metrics

    po_joint = m["joint_gpcm_on_pairwise_only"]
    do_joint = m["joint_bt_on_direct_only"]
    po_indep = m["indep_gpcm_pairwise_only"]
    do_indep = m["indep_bt_direct_only"]
    po_gap = m["pairwise_only_transfer_gap"]
    do_gap = m["direct_only_transfer_gap"]
    drift_sp = m["dyn_net_drift_spearman"]
    final_sp = m["dyn_final_spearman"]

    lines = [
        "=" * 70,
        "DYNAMIC-ENCODER x SHARED-MULTI-HEAD EXPERIMENT",
        "Dynamic ability tracking + format-agnostic item placement",
        "=" * 70,
        "",
        "DESIGN",
        f"  Total items:          {n}",
        f"  DIRECT-ONLY items:    {n_do}  (GPCM only, never in BT data)",
        f"  PAIRWISE-ONLY items:  {n_po}  (BT only, never in GPCM data)",
        f"  BOTH items:           {n_both}  (seen by both formats; bridge)",
        f"  Respondents:          {mode_a['n_respondents']}",
        f"  Sequence length:      {mode_a['seq_len']}  steps per learner",
        f"  Drift sigma:          {params['drift_sigma']}  (random-walk step std)",
        f"  GPCM obs:             {mode_a['n_respondents'] * mode_a['seq_len']}",
        f"  BT pairs:             {mode_b['n_pairs']}",
        f"  Emb dim:              {params['emb_dim']}",
        f"  LSTM hidden dim:      {params['hidden_dim']}",
        f"  Epochs:               {params['n_epochs']}",
        f"  Wall time:            {wall_time:.1f}s",
        "",
        "CROSS-FORMAT TRANSFER  (the key question: does encoder-theta preserve it?)",
        "",
        f"  {'Metric':<45} {'Dyn (this)':>10}  {'Static ref':>10}",
        f"  {'-'*45} {'-'*10}  {'-'*10}",
        f"  {'Pairwise-only -> GPCM head (joint)':<45} {po_joint:>10.4f}  {JOINTFMT_PAIRWISE_ONLY:>10.4f}",
        f"  {'Direct-only -> BT head (joint)':<45} {do_joint:>10.4f}  {JOINTFMT_DIRECT_ONLY:>10.4f}",
        f"  {'Pairwise-only indep GPCM (noise baseline)':<45} {po_indep:>10.4f}  {'---':>10}",
        f"  {'Direct-only indep BT (noise baseline)':<45} {do_indep:>10.4f}  {'---':>10}",
        f"  {'Transfer gap (pairwise-only)':<45} {po_gap:>10.4f}",
        f"  {'Transfer gap (direct-only)':<45} {do_gap:>10.4f}",
        "",
        "DYNAMIC TRACKING  (new capability vs static theta)",
        "",
        f"  Net-drift Spearman (theta_T - theta_1, rec vs true): {drift_sp:.4f}",
        f"  Final-theta Spearman (theta_T, rec vs true):          {final_sp:.4f}",
        "",
        "  Net-drift is the HONEST dynamic metric: it measures within-learner",
        "  tracking of change, not confounded by inter-person ability spread.",
        "  A pooled Pearson of trajectories would inflate to ~0.9 even for a",
        "  static model because most variance is between persons, not over time.",
        "",
        "SANITY CHECKS",
        f"  Indep GPCM on DIRECT-ONLY  (trained on these):  {m['indep_gpcm_direct_only']:.4f}",
        f"  Indep BT   on PAIRWISE-ONLY (trained on these): {m['indep_bt_pairwise_only']:.4f}",
        f"  Joint GPCM on BOTH items:                       {m['joint_both_gpcm']:.4f}",
        f"  Joint BT   on BOTH items:                       {m['joint_both_bt']:.4f}",
        f"  Joint overall (BT head, all items):             {m['joint_overall']:.4f}",
        "",
    ]

    # Verdict
    transfer_pass = po_joint >= 0.70 and do_joint >= 0.70 and po_gap >= 0.30 and do_gap >= 0.30
    transfer_partial = (po_joint >= 0.50 or do_joint >= 0.50) and (po_gap >= 0.20 or do_gap >= 0.20)
    drift_pass = drift_sp >= 0.30

    lines.append("VERDICT")

    if transfer_pass and drift_pass:
        verdict = (
            f"PASS -- BOTH hold together. "
            f"Cross-format transfer with dynamic encoder: "
            f"pairwise-only Spearman={po_joint:.3f} (static ref={JOINTFMT_PAIRWISE_ONLY:.3f}), "
            f"direct-only Spearman={do_joint:.3f} (static ref={JOINTFMT_DIRECT_ONLY:.3f}). "
            f"Dynamic tracking (net-drift Spearman)={drift_sp:.3f}. "
            f"The LSTM encoder preserves cross-format unification while adding "
            f"genuine within-learner ability tracking."
        )
    elif transfer_pass and not drift_pass:
        verdict = (
            f"PARTIAL -- format-agnostic holds, dynamic tracking is weak. "
            f"Transfer: pairwise-only={po_joint:.3f}, direct-only={do_joint:.3f}. "
            f"Net-drift Spearman={drift_sp:.3f} (below threshold 0.30). "
            f"The shared embedding unifies formats but the encoder is not reliably "
            f"tracking within-learner change at seq_len={mode_a['seq_len']}."
        )
    elif not transfer_pass and drift_pass:
        verdict = (
            f"PARTIAL -- dynamic tracking holds, cross-format transfer degraded. "
            f"Transfer: pairwise-only={po_joint:.3f} (static ref={JOINTFMT_PAIRWISE_ONLY:.3f}), "
            f"direct-only={do_joint:.3f} (static ref={JOINTFMT_DIRECT_ONLY:.3f}). "
            f"Net-drift Spearman={drift_sp:.3f}. "
            f"The encoder destabilises the shared item scale -- the two optimisation "
            f"objectives conflict at this configuration."
        )
    elif transfer_partial:
        verdict = (
            f"WEAK -- measurable transfer but below threshold. "
            f"Pairwise-only={po_joint:.3f}, direct-only={do_joint:.3f}. "
            f"Net-drift Spearman={drift_sp:.3f}. "
            f"Investigate: reduce drift_sigma, increase seq_len, or tune LR/epochs."
        )
    else:
        verdict = (
            f"FAIL -- neither claim holds at these settings. "
            f"Pairwise-only={po_joint:.3f}, direct-only={do_joint:.3f}, "
            f"net-drift Spearman={drift_sp:.3f}."
        )

    if po_joint < JOINTFMT_PAIRWISE_ONLY - 0.05 or do_joint < JOINTFMT_DIRECT_ONLY - 0.05:
        delta_po = po_joint - JOINTFMT_PAIRWISE_ONLY
        delta_do = do_joint - JOINTFMT_DIRECT_ONLY
        verdict += (
            f" DEGRADATION vs static-theta reference: "
            f"pairwise-only {delta_po:+.3f}, direct-only {delta_do:+.3f}."
        )

    lines.append(f"  {verdict}")
    lines.append("")
    lines.append("EXPOSABLE KT STATE")
    lines.append(
        "  DynJointModel.get_kt_state(item_ids, responses) -> (h_n, c_n)"
    )
    lines.append(
        "  Signature: Tuple[Tensor(batch, hidden_dim), Tensor(batch, hidden_dim)]"
    )
    lines.append(
        "  Returns raw LSTM (hidden, cell) at final timestep for each learner."
    )
    lines.append(
        "  A downstream skill consumer applies W @ h_n to get per-skill mastery."
    )
    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def _base_params(seed: int) -> dict:
    """The fixed experiment hyperparameters (data-generating + model size).

    The data-generating process is identical to P1.6 so all numbers stay
    comparable.  n_epochs is the one training-side knob that the rebalanced
    path overrides.
    """
    return {
        "n_items": 300,
        "n_direct_only": 100,
        "n_pairwise_only": 100,
        "K": 4,
        "emb_dim": 16,
        "hidden_dim": 64,
        "n_epochs": 400,
        "lr": 5e-3,
        "batch_size": 64,
        "reg": 0.01,
        "drift_sigma": 0.2,
        "n_respondents": 500,
        "seq_len": 40,
        "comparisons_per_item": 120,
        "seed": seed,
    }


def _make_data(params: dict):
    """Generate (ground_truth, mode_a, mode_b) for a given seed."""
    seed = params["seed"]
    gt = generate_ground_truth(
        n_items=params["n_items"],
        n_direct_only=params["n_direct_only"],
        n_pairwise_only=params["n_pairwise_only"],
        K=params["K"],
        seed=seed,
    )
    mode_a = generate_mode_a_dynamic(
        gt,
        n_respondents=params["n_respondents"],
        seq_len=params["seq_len"],
        drift_sigma=params["drift_sigma"],
        seed=seed,
    )
    mode_b = generate_mode_b(
        gt, comparisons_per_item=params["comparisons_per_item"], seed=seed
    )
    return gt, mode_a, mode_b


def _train_baselines(gt, mode_a, mode_b, params):
    """Independent GPCM + BT baselines (static theta, separate embeddings)."""
    indep_gpcm = train_indep_gpcm(
        ground_truth=gt, mode_a=mode_a, emb_dim=params["emb_dim"],
        n_epochs=params["n_epochs"], lr=0.05, batch_size=8192,
        reg=params["reg"], seed=params["seed"], verbose=False,
    )
    indep_bt = train_indep_bt(
        ground_truth=gt, mode_b=mode_b, emb_dim=params["emb_dim"],
        n_epochs=params["n_epochs"], lr=0.05, batch_size=8192,
        reg=params["reg"], seed=params["seed"], verbose=False,
    )
    return indep_gpcm.get_difficulty(), indep_bt.get_strength()


def run_pipeline(params: dict, joint_kwargs: dict, verbose: bool = True):
    """Run one full pipeline: data -> joint train -> baselines -> metrics.

    joint_kwargs selects the training path.  An empty dict means the original
    train_joint (P1.6 baseline).  Any keys present route through
    train_joint_balanced, which reduces to train_joint at its defaults.

    Returns (metrics, gt, mode_a, mode_b, wall_time, joint_arrays).
    """
    t0 = time.time()
    gt, mode_a, mode_b = _make_data(params)

    if joint_kwargs:
        joint_model = train_joint_balanced(
            ground_truth=gt, mode_a=mode_a, mode_b=mode_b,
            emb_dim=params["emb_dim"], hidden_dim=params["hidden_dim"],
            n_epochs=params["n_epochs"], lr=params["lr"],
            batch_size=params["batch_size"], reg=params["reg"],
            seed=params["seed"], verbose=verbose, **joint_kwargs,
        )
    else:
        joint_model = train_joint(
            ground_truth=gt, mode_a=mode_a, mode_b=mode_b,
            emb_dim=params["emb_dim"], hidden_dim=params["hidden_dim"],
            n_epochs=params["n_epochs"], lr=params["lr"],
            batch_size=params["batch_size"], reg=params["reg"],
            seed=params["seed"], verbose=verbose,
        )

    joint_strength = joint_model.get_item_strengths()
    joint_gpcm_diff = joint_model.get_gpcm_difficulty()
    indep_gpcm_diff, indep_bt_strength = _train_baselines(gt, mode_a, mode_b, params)

    m = compute_metrics(
        ground_truth=gt, mode_a=mode_a, joint_model=joint_model,
        joint_strength=joint_strength, joint_gpcm_difficulty=joint_gpcm_diff,
        indep_gpcm_difficulty=indep_gpcm_diff, indep_bt_strength=indep_bt_strength,
    )
    arrays = {
        "joint_d": joint_strength.numpy(),
        "indep_gpcm_d": indep_gpcm_diff.numpy(),
        "indep_bt_d": indep_bt_strength.numpy(),
    }
    return m, gt, mode_a, mode_b, time.time() - t0, arrays


def _save_run(tag, gt, mode_a, mode_b, m, params, wall_time, arrays):
    """Write report.txt / results.json / params.json / item_arrays.npz for a run."""
    suffix = "" if tag == "baseline" else f"_{tag}"
    report = format_report(gt, mode_a, mode_b, m, params, wall_time)

    with open(os.path.join(OUTPUT_DIR, f"report{suffix}.txt"), "w") as f:
        f.write(report)
    with open(os.path.join(OUTPUT_DIR, f"results{suffix}.json"), "w") as f:
        json.dump({k: round(float(v), 6) for k, v in m.items()}, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, f"params{suffix}.json"), "w") as f:
        json.dump(params, f, indent=2)

    group = np.zeros(gt["n_items"], dtype=np.int64)
    group[gt["direct_only_idx"].numpy()] = 1
    group[gt["pairwise_only_idx"].numpy()] = 2
    np.savez(
        os.path.join(OUTPUT_DIR, f"item_arrays{suffix}.npz"),
        s_true=gt["s"].numpy(), group=group, **arrays,
    )
    return report


def _comparison_block(base_m, rebal_m, rebal_params) -> str:
    """Side-by-side P1.6 baseline vs rebalanced on the three target metrics."""
    rows = [
        ("Pairwise-only -> GPCM (keep >= ~0.96)",
         base_m["joint_gpcm_on_pairwise_only"], rebal_m["joint_gpcm_on_pairwise_only"]),
        ("Direct-only -> BT  (raise toward ~0.98)",
         base_m["joint_bt_on_direct_only"], rebal_m["joint_bt_on_direct_only"]),
        ("Net-drift tracking (keep >= ~0.44)",
         base_m["dyn_net_drift_spearman"], rebal_m["dyn_net_drift_spearman"]),
        ("Both-item bridge (BT head)",
         base_m["joint_both_bt"], rebal_m["joint_both_bt"]),
    ]
    lines = [
        "",
        "=" * 70,
        "REBALANCED vs P1.6 BASELINE",
        f"  baseline:   n_epochs=400 balance=none",
        f"  rebalanced: n_epochs={rebal_params['n_epochs']} balance={REBAL_BALANCE} "
        f"bt_weight={REBAL_BT_WEIGHT}",
        "=" * 70,
        "",
        f"  {'Metric':<40} {'P1.6':>8} {'Rebal':>8} {'Delta':>8}",
        f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8}",
    ]
    for name, b, r in rows:
        lines.append(f"  {name:<40} {b:>8.4f} {r:>8.4f} {r-b:>+8.4f}")
    lines += ["", "=" * 70]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Dynamic-encoder joint experiment: baseline + rebalanced."
    )
    parser.add_argument(
        "--mode", choices=["baseline", "rebalanced", "both"], default="both",
        help="baseline = original P1.6 (400ep); rebalanced = recommended config; "
             "both (default) = run each and emit a comparison.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("DYNAMIC-ENCODER x SHARED-MULTI-HEAD EXPERIMENT")
    print("=" * 70)

    base_params = _base_params(args.seed)
    print(f"\nData: {base_params['n_items']} items "
          f"(direct-only={base_params['n_direct_only']}, "
          f"pairwise-only={base_params['n_pairwise_only']}), "
          f"{base_params['n_respondents']}x{base_params['seq_len']} GPCM seqs, "
          f"drift_sigma={base_params['drift_sigma']}")

    base_m = rebal_m = None

    if args.mode in ("baseline", "both"):
        print("\n[baseline] Training P1.6 path (train_joint, 400 epochs)...")
        base_m, gt, ma, mb, wt, arr = run_pipeline(base_params, joint_kwargs={})
        rep = _save_run("baseline", gt, ma, mb, base_m, base_params, wt, arr)
        print("\n" + rep)

    if args.mode in ("rebalanced", "both"):
        rebal_params = dict(base_params)
        rebal_params["n_epochs"] = REBAL_EPOCHS
        rebal_kwargs = dict(
            balance=REBAL_BALANCE, bt_weight=REBAL_BT_WEIGHT,
            bt_pairs_mult=REBAL_BT_PAIRS_MULT, head_lr_mult=REBAL_HEAD_LR_MULT,
            warmup_frac=REBAL_WARMUP_FRAC,
        )
        print(f"\n[rebalanced] Training recommended path "
              f"({REBAL_EPOCHS} epochs, balance={REBAL_BALANCE})...")
        rebal_m, gt, ma, mb, wt, arr = run_pipeline(rebal_params, joint_kwargs=rebal_kwargs)
        rep = _save_run("rebalanced", gt, ma, mb, rebal_m, rebal_params, wt, arr)
        print("\n" + rep)

    if args.mode == "both":
        comparison = _comparison_block(base_m, rebal_m, rebal_params)
        print(comparison)
        with open(os.path.join(OUTPUT_DIR, "comparison.txt"), "w") as f:
            f.write(comparison.lstrip("\n"))

    print(f"\nOutputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
