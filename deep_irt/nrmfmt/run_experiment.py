"""
run_experiment.py -- Binary-vs-NRM cross-format transfer experiment.

The data-rich analogue of deep_irt/jointfmt (GPCM-vs-BT).  Tests whether ONE
shared item embedding, read by a 2PL (right/wrong) head AND a Bock NRM
(chosen-option) head, places items on ONE latent difficulty scale, so a
binary-only item lands correctly on the NRM-derived scale and a nrm-only item
lands on the binary scale.

Run from repo root:
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=deep_irt \
        python deep_irt/nrmfmt/run_experiment.py
"""

import os
import sys
import json
import torch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data_gen import generate_ground_truth, generate_mode_a, generate_mode_b
from train import train_joint, train_indep_binary, train_indep_nrm
from metrics import compute_metrics

SEED = 0
OUTPUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def format_report(gt, mode_a, mode_b, m, params):
    lines = [
        "=" * 66,
        "BINARY-vs-NRM CROSS-FORMAT TRANSFER EXPERIMENT",
        "One shared item scale read by a 2PL head and a Bock-NRM head?",
        "=" * 66,
        "",
        "DESIGN",
        f"  Total items:        {gt['n_items']}",
        f"  BINARY-ONLY items:  {gt['n_binary_only']}  (2PL only, never in NRM data)",
        f"  NRM-ONLY items:     {gt['n_nrm_only']}  (NRM only, never in 2PL data)",
        f"  BOTH items:         {gt['n_both']}  (bridge, seen by both formats)",
        f"  Options K:          {gt['K']}   correct option: {gt['correct_option']}",
        f"  Binary respondents: {mode_a['n_respondents']}  ({len(mode_a['responses'])} obs)",
        f"  NRM respondents:    {mode_b['n_respondents']}  ({len(mode_b['responses'])} obs)",
        f"  Embedding dim:      {params['emb_dim']}   epochs: {params['n_epochs']}",
        "",
        "OVERALL RECOVERY vs GROUND TRUTH  (Spearman, all items)",
        f"  Joint (shared d_i):            {m['joint_overall']:.4f}",
        f"  Indep 2PL (separate emb):      {m['indep_binary_overall']:.4f}",
        f"  Indep NRM (separate emb):      {m['indep_nrm_overall']:.4f}",
        "",
        "CROSS-FORMAT TRANSFER  (the critical held-out-format test)",
        "",
        "  [A] NRM-ONLY items (NRM trained their embedding):",
        "      Transfer = read shared d_i as a 2PL difficulty.",
        f"    Joint d_i:            {m['joint_on_nrm_only']:.4f}  <-- cross-format",
        f"    Indep 2PL (noise):    {m['indep_binary_nrm_only']:.4f}  (never saw these)",
        f"    Transfer gap:         {m['nrm_only_transfer_gap']:+.4f}",
        "",
        "  [B] BINARY-ONLY items (2PL trained their embedding):",
        "      Transfer = read shared d_i as the NRM correct-option location.",
        f"    Joint d_i:            {m['joint_on_binary_only']:.4f}  <-- cross-format",
        f"    Indep NRM (noise):    {m['indep_nrm_binary_only']:.4f}  (never saw these)",
        f"    Transfer gap:         {m['binary_only_transfer_gap']:+.4f}",
        "",
        "SANITY (each indep model on its OWN items)",
        f"  Indep 2PL on BINARY-ONLY:   {m['indep_binary_on_binary_only']:.4f}",
        f"  Indep NRM on NRM-ONLY:      {m['indep_nrm_on_nrm_only']:.4f}",
        f"  Joint on BOTH items:        {m['joint_both']:.4f}",
        "",
    ]

    a = m["joint_on_nrm_only"]; b = m["joint_on_binary_only"]
    ga = m["nrm_only_transfer_gap"]; gb = m["binary_only_transfer_gap"]
    lines.append("VERDICT")
    if a >= 0.70 and b >= 0.70 and ga >= 0.30 and gb >= 0.30:
        v = (f"PASS. Shared embedding genuinely unifies the binary and nominal "
             f"formats. NRM-only items land on the 2PL scale "
             f"(Spearman={a:.3f} vs indep noise={m['indep_binary_nrm_only']:.3f}), "
             f"binary-only items land on the NRM location scale "
             f"(Spearman={b:.3f} vs indep noise={m['indep_nrm_binary_only']:.3f}). "
             f"The BOTH-item bridge carries the scale across formats. The "
             f"binary-vs-nominal pair works -- ready for real MC data.")
    elif (a >= 0.50 or b >= 0.50) and (ga >= 0.20 or gb >= 0.20):
        v = (f"PARTIAL. Measurable but asymmetric transfer "
             f"(nrm-only {a:.3f}, binary-only {b:.3f}).")
    else:
        v = (f"WEAK/FAIL. Insufficient transfer "
             f"(nrm-only {a:.3f}, binary-only {b:.3f}).")
    lines.append(f"  {v}")
    lines.append("")
    lines.append("=" * 66)
    return "\n".join(lines)


def main():
    torch.manual_seed(SEED)

    N_ITEMS = 300
    N_BINARY_ONLY = 100
    N_NRM_ONLY = 100
    K = 4
    CORRECT = 0
    EMB_DIM = 16
    N_EPOCHS = 600
    LR = 0.05
    BATCH = 8192
    REG = 0.01

    params = {"n_items": N_ITEMS, "n_binary_only": N_BINARY_ONLY,
              "n_nrm_only": N_NRM_ONLY, "K": K, "emb_dim": EMB_DIM,
              "n_epochs": N_EPOCHS, "lr": LR, "seed": SEED}

    print("\n[1/4] Generating synthetic binary + NRM data...")
    gt = generate_ground_truth(N_ITEMS, N_BINARY_ONLY, N_NRM_ONLY, K, CORRECT, SEED)
    mode_a = generate_mode_a(gt, n_respondents=2000, items_per_respondent=40, seed=SEED)
    mode_b = generate_mode_b(gt, n_respondents=2000, items_per_respondent=40, seed=SEED)
    print(f"  items={gt['n_items']} (bin-only={gt['n_binary_only']}, "
          f"nrm-only={gt['n_nrm_only']}, both={gt['n_both']})")
    print(f"  binary obs={len(mode_a['responses'])}, nrm obs={len(mode_b['responses'])}")

    print("\n[2/4] Training joint model (shared embedding, 2PL + NRM heads)...")
    joint = train_joint(gt, mode_a, mode_b, EMB_DIM, N_EPOCHS, LR, BATCH, REG, SEED, True)
    joint_d = joint.get_item_difficulty()

    print("\n[3/4] Training independent baselines (separate embeddings)...")
    ib = train_indep_binary(gt, mode_a, EMB_DIM, N_EPOCHS, LR, BATCH, REG, SEED, True)
    inn = train_indep_nrm(gt, mode_b, EMB_DIM, N_EPOCHS, LR, BATCH, REG, SEED, True)
    indep_binary_d = ib.get_difficulty()
    indep_nrm_d = inn.get_difficulty()

    print("\n[4/4] Computing metrics...")
    m = compute_metrics(gt, joint_d, indep_binary_d, indep_nrm_d)

    report = format_report(gt, mode_a, mode_b, m, params)
    print("\n" + report)

    with open(os.path.join(OUTPUT_DIR, "report.txt"), "w") as f:
        f.write(report)
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump({k: round(float(v), 6) for k, v in m.items()}, f, indent=2)

    group = np.zeros(gt["n_items"], dtype=np.int64)
    group[gt["binary_only_idx"].numpy()] = 1
    group[gt["nrm_only_idx"].numpy()] = 2
    np.savez(
        os.path.join(OUTPUT_DIR, "item_arrays.npz"),
        s_true=gt["s"].numpy(), group=group,
        joint_d=joint_d.numpy(),
        indep_binary_d=indep_binary_d.numpy(),
        indep_nrm_d=indep_nrm_d.numpy(),
    )
    print(f"\nOutputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
