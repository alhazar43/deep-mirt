"""
run_experiment.py -- Joint-format transfer experiment.

Tests whether a SINGLE shared model genuinely unifies response formats:
  - GPCM (ordinal) for direct items
  - Bradley-Terry (pairwise) for comparison items

The KEY question: do pairwise-only items get correctly placed on the
ordinal scale (and vice versa) through the shared embedding, even though
they were never seen by the format that defines that scale?

Run as:
    cd deep_irt && KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=deep_irt python jointfmt/run_experiment.py

or from repo root:
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=deep_irt python deep_irt/jointfmt/run_experiment.py
"""

import os
import sys
import json
import torch
import numpy as np

# Ensure we can import from deep_irt/jointfmt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data_gen import generate_ground_truth, generate_mode_a, generate_mode_b
from train import train_joint, train_indep_gpcm, train_indep_bt
from metrics import compute_metrics

SEED = 0
OUTPUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def format_report(ground_truth, mode_a, mode_b, metrics, params) -> str:
    n = ground_truth["n_items"]
    n_do = ground_truth["n_direct_only"]
    n_po = ground_truth["n_pairwise_only"]
    n_both = ground_truth["n_both"]

    m = metrics

    lines = [
        "=" * 65,
        "JOINT-FORMAT TRANSFER EXPERIMENT",
        "Does a shared embedding genuinely unify response formats?",
        "=" * 65,
        "",
        "DESIGN",
        f"  Total items:          {n}",
        f"  DIRECT-ONLY items:    {n_do}  (GPCM only, never in BT data)",
        f"  PAIRWISE-ONLY items:  {n_po}  (BT only, never in GPCM data)",
        f"  BOTH items:           {n_both}  (seen by both formats; bridge)",
        f"  Respondents (GPCM):   {mode_a['n_respondents']}",
        f"  GPCM observations:    {len(mode_a['responses'])}",
        f"  BT pairs:             {mode_b['n_pairs']}  (~{params.get('comparisons_per_item','?')} per item)",
        f"  Embedding dim:        {params['emb_dim']}",
        f"  Epochs:               {params['n_epochs']}",
        "",
        "HOW TRANSFER WORKS",
        "  BOTH items are seen by both formats -> their embeddings are",
        "  pulled into agreement by BOTH heads simultaneously.",
        "  PAIRWISE-ONLY items share the embedding space with BOTH items,",
        "  so the BT loss places them relative to BOTH items, which are",
        "  already anchored to the GPCM scale. Transfer flows via BOTH.",
        "  Symmetrically for DIRECT-ONLY items.",
        "",
        "OVERALL RECOVERY vs GROUND TRUTH  (Spearman, all items)",
        f"  Joint model (BT head, shared embedding):   {m['joint_overall']:.4f}",
        f"  Indep GPCM  (separate embedding):          {m['indep_gpcm_overall']:.4f}",
        f"  Indep BT    (separate embedding):          {m['indep_bt_overall']:.4f}",
        "",
        "CROSS-FORMAT TRANSFER  (the critical test)",
        "",
        "  [A] PAIRWISE-ONLY items (BT trained their embedding):",
        "      Transfer = GPCM head reads BT-trained embeddings.",
        f"    Joint GPCM difficulty:  {m['joint_gpcm_on_pairwise_only']:.4f}  <-- cross-format",
        f"    Indep GPCM (noise):     {m['indep_gpcm_pairwise_only']:.4f}  (never saw these)",
        f"    Transfer gain:          +{m['pairwise_only_transfer_gap']:.4f}",
        "",
        "  [B] DIRECT-ONLY items (GPCM trained their embedding):",
        "      Transfer = BT head reads GPCM-trained embeddings.",
        f"    Joint BT strength:      {m['joint_bt_on_direct_only']:.4f}  <-- cross-format",
        f"    Indep BT (noise):       {m['indep_bt_direct_only']:.4f}  (never saw these)",
        f"    Transfer gain:          +{m['direct_only_transfer_gap']:.4f}",
        "",
        "SANITY CHECK (each model on its OWN items)",
        f"  Indep GPCM on DIRECT-ONLY  (trained on these):  {m['indep_gpcm_direct_only']:.4f}",
        f"  Indep BT   on PAIRWISE-ONLY (trained on these): {m['indep_bt_pairwise_only']:.4f}",
        f"  Joint GPCM on BOTH items:                       {m['joint_both_gpcm']:.4f}",
        f"  Joint BT   on BOTH items:                       {m['joint_both_bt']:.4f}",
        "",
    ]

    # Verdict
    po_gap = m["pairwise_only_transfer_gap"]
    do_gap = m["direct_only_transfer_gap"]
    po_joint = m["joint_gpcm_on_pairwise_only"]
    do_joint = m["joint_bt_on_direct_only"]

    lines.append("VERDICT")
    if po_joint >= 0.70 and do_joint >= 0.70 and po_gap >= 0.30 and do_gap >= 0.30:
        verdict = (
            "PASS. The joint shared-embedding model genuinely unifies formats. "
            f"Pairwise-only items land correctly on the ordinal scale via the GPCM head "
            f"(Spearman={po_joint:.3f} vs indep noise={m['indep_gpcm_pairwise_only']:.3f}), "
            f"and direct-only items land correctly on the pairwise scale via the BT head "
            f"(Spearman={do_joint:.3f} vs indep noise={m['indep_bt_direct_only']:.3f}). "
            "The shared BOTH-item bridge carries information across formats through the "
            "shared embedding, which is the mechanism of unification."
        )
    elif (po_joint >= 0.50 or do_joint >= 0.50) and (po_gap >= 0.20 or do_gap >= 0.20):
        verdict = (
            "PARTIAL. The shared embedding shows measurable cross-format transfer "
            f"(pairwise-only GPCM: {po_joint:.3f}, direct-only BT: {do_joint:.3f}), "
            "but not fully symmetric. Transfer flows more strongly in one direction."
        )
    else:
        verdict = (
            "WEAK/FAIL. The cross-format transfer is insufficient to claim genuine "
            f"unification (pairwise-only GPCM: {po_joint:.3f}, "
            f"direct-only BT: {do_joint:.3f}). "
            "Investigate data volume, embedding capacity, or BOTH-group overlap."
        )

    lines.append(f"  {verdict}")
    lines.append("")
    lines.append("=" * 65)

    return "\n".join(lines)


def main():
    torch.manual_seed(SEED)

    # -----------------------------------------------------------------------
    # Hyperparameters
    # -----------------------------------------------------------------------
    N_ITEMS = 300
    N_DIRECT_ONLY = 100
    N_PAIRWISE_ONLY = 100
    # N_BOTH = 100  (implicit: 300 - 100 - 100)
    EMB_DIM = 16
    N_EPOCHS = 600
    LR = 0.05
    BATCH_SIZE = 8192
    REG = 0.01
    # Increased comparisons to balance GPCM/BT signal strength.
    # 80k GPCM obs need ~800 comparisons/item to roughly balance gradients.
    COMPARISONS_PER_ITEM = 400

    params = {
        "n_items": N_ITEMS,
        "n_direct_only": N_DIRECT_ONLY,
        "n_pairwise_only": N_PAIRWISE_ONLY,
        "emb_dim": EMB_DIM,
        "n_epochs": N_EPOCHS,
        "lr": LR,
        "seed": SEED,
        "comparisons_per_item": COMPARISONS_PER_ITEM,
    }

    print("\n" + "=" * 65)
    print("JOINT-FORMAT TRANSFER EXPERIMENT")
    print("=" * 65)

    # -----------------------------------------------------------------------
    # 1. Generate data
    # -----------------------------------------------------------------------
    print("\n[1/4] Generating synthetic data...")
    gt = generate_ground_truth(
        n_items=N_ITEMS,
        n_direct_only=N_DIRECT_ONLY,
        n_pairwise_only=N_PAIRWISE_ONLY,
        K=4,
        seed=SEED,
    )
    mode_a = generate_mode_a(gt, n_respondents=2000, items_per_respondent=40, seed=SEED)
    mode_b = generate_mode_b(gt, comparisons_per_item=COMPARISONS_PER_ITEM, seed=SEED)

    print(f"  Items: {gt['n_items']}  "
          f"(direct-only={gt['n_direct_only']}, "
          f"pairwise-only={gt['n_pairwise_only']}, "
          f"both={gt['n_both']})")
    print(f"  GPCM obs: {len(mode_a['responses'])}, "
          f"BT pairs: {mode_b['n_pairs']}")

    # -----------------------------------------------------------------------
    # 2. Train joint model
    # -----------------------------------------------------------------------
    print("\n[2/4] Training joint model (shared embedding, GPCM + BT heads)...")
    joint_model = train_joint(
        ground_truth=gt,
        mode_a=mode_a,
        mode_b=mode_b,
        emb_dim=EMB_DIM,
        n_epochs=N_EPOCHS,
        lr=LR,
        batch_size=BATCH_SIZE,
        reg=REG,
        seed=SEED,
        verbose=True,
    )
    joint_strength = joint_model.get_item_strengths()
    joint_gpcm_diff = joint_model.get_gpcm_difficulty()

    # -----------------------------------------------------------------------
    # 3. Train independent baselines
    # -----------------------------------------------------------------------
    print("\n[3/4] Training independent baselines (separate embeddings)...")
    indep_gpcm = train_indep_gpcm(
        ground_truth=gt,
        mode_a=mode_a,
        emb_dim=EMB_DIM,
        n_epochs=N_EPOCHS,
        lr=LR,
        batch_size=BATCH_SIZE,
        reg=REG,
        seed=SEED,
        verbose=True,
    )
    indep_bt = train_indep_bt(
        ground_truth=gt,
        mode_b=mode_b,
        emb_dim=EMB_DIM,
        n_epochs=N_EPOCHS,
        lr=LR,
        batch_size=BATCH_SIZE,
        reg=REG,
        seed=SEED,
        verbose=True,
    )
    indep_gpcm_difficulty = indep_gpcm.get_difficulty()
    indep_bt_strength = indep_bt.get_strength()

    # -----------------------------------------------------------------------
    # 4. Compute metrics and report
    # -----------------------------------------------------------------------
    print("\n[4/4] Computing metrics...")
    m = compute_metrics(
        ground_truth=gt,
        joint_strength=joint_strength,
        joint_gpcm_difficulty=joint_gpcm_diff,
        indep_gpcm_difficulty=indep_gpcm_difficulty,
        indep_bt_strength=indep_bt_strength,
    )

    report = format_report(gt, mode_a, mode_b, m, params)
    print("\n" + report)

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    report_path = os.path.join(OUTPUT_DIR, "report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    results_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump({k: round(float(v), 6) for k, v in m.items()}, f, indent=2)

    params_path = os.path.join(OUTPUT_DIR, "params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)

    # -----------------------------------------------------------------------
    # Per-item arrays for figure generation (shared scale vs independent).
    # group: 0 = BOTH, 1 = DIRECT-ONLY, 2 = PAIRWISE-ONLY.
    # joint_d is the single shared difficulty scalar (BT strength == GPCM
    # difficulty centroid, by construction). indep_* are the held-out-format
    # readouts from the separate-table baselines (noise on unseen items).
    # -----------------------------------------------------------------------
    group = np.zeros(gt["n_items"], dtype=np.int64)
    group[gt["direct_only_idx"].numpy()] = 1
    group[gt["pairwise_only_idx"].numpy()] = 2
    arrays_path = os.path.join(OUTPUT_DIR, "item_arrays.npz")
    np.savez(
        arrays_path,
        s_true=gt["s"].numpy(),
        group=group,
        joint_d=joint_strength.numpy(),
        indep_gpcm_d=indep_gpcm_difficulty.numpy(),
        indep_bt_d=indep_bt_strength.numpy(),
    )

    print(f"\nOutputs saved to {OUTPUT_DIR}/")
    print(f"  {report_path}")
    print(f"  {results_path}")
    print(f"  {arrays_path}")


if __name__ == "__main__":
    main()
