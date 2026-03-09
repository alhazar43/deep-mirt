#!/usr/bin/env python3
"""Merge prediction metrics with recovery correlations."""

import pandas as pd
import numpy as np

# Load prediction metrics
all_exp = pd.read_csv("outputs/summary_all_experiments.csv")
baselines = pd.read_csv("outputs/summary_baselines.csv")
pred_metrics = pd.concat([all_exp, baselines], ignore_index=True)

# Load recovery correlations
recovery = pd.read_csv("outputs/recovery_correlations.csv")

# Merge
merged = pred_metrics.merge(recovery, on="experiment", how="left")

# Save full merged data
merged.to_csv("outputs/merged_metrics_recovery.csv", index=False)
print(f"Saved {len(merged)} experiments to outputs/merged_metrics_recovery.csv")

# Extract specific tables for paper

# RQ1: K=3,4,5,6 at Q=200
rq1_data = []
for k in [3, 4, 5, 6]:
    # Get baseline results
    static = merged[merged["experiment"] == f"large_q200_k{k}_static_gpcm"].iloc[0]
    dynamic = merged[merged["experiment"] == f"large_q200_k{k}_dynamic_gpcm"].iloc[0]
    softmax = merged[merged["experiment"] == f"large_q200_k{k}_dkvmn_softmax"].iloc[0]
    ordinal = merged[merged["experiment"] == f"large_q200_k{k}_dkvmn_ordinal"].iloc[0]

    # Get DEEP-GPCM result (static_item encoding)
    deepgpcm = merged[merged["experiment"] == f"q200_k{k}_static_item"].iloc[0]

    rq1_data.append({
        "K": k,
        "Static_ACC": static["acc"],
        "Static_QWK": static["qwk"],
        "Static_tau": static["tau"],
        "Static_r_alpha": static["r_alpha"],
        "Static_r_beta": static["r_beta_mean"],
        "Dynamic_ACC": dynamic["acc"],
        "Dynamic_QWK": dynamic["qwk"],
        "Dynamic_tau": dynamic["tau"],
        "Dynamic_r_alpha": dynamic["r_alpha"],
        "Dynamic_r_beta": dynamic["r_beta_mean"],
        "Softmax_ACC": softmax["acc"],
        "Softmax_QWK": softmax["qwk"],
        "Softmax_tau": softmax["tau"],
        "Ordinal_ACC": ordinal["acc"],
        "Ordinal_QWK": ordinal["qwk"],
        "Ordinal_tau": ordinal["tau"],
        "Ordinal_r_alpha": ordinal["r_alpha"],
        "Ordinal_r_beta": ordinal["r_beta_mean"],
        "DEEPGPCM_ACC": deepgpcm["acc"],
        "DEEPGPCM_QWK": deepgpcm["qwk"],
        "DEEPGPCM_tau": deepgpcm["tau"],
        "DEEPGPCM_r_alpha": deepgpcm["r_alpha"],
        "DEEPGPCM_r_beta": deepgpcm["r_beta_mean"],
    })

rq1_df = pd.DataFrame(rq1_data)
rq1_df.to_csv("outputs/rq1_table.csv", index=False)
print("\nRQ1 table saved to outputs/rq1_table.csv")

# RQ3: Parameter recovery at Q=200, K=4
rq3_data = []
for exp_name in ["q200_k4_static_item", "large_q200_k4_static_gpcm", "large_q200_k4_dynamic_gpcm"]:
    row = merged[merged["experiment"] == exp_name].iloc[0]
    model_name = "DEEP-GPCM" if "q200" in exp_name and "large" not in exp_name else \
                 "Static GPCM" if "static_gpcm" in exp_name else "Dynamic GPCM"
    rq3_data.append({
        "Model": model_name,
        "QWK": row["qwk"],
        "ACC": row["acc"],
        "r_alpha": row["r_alpha"],
        "r_beta": row["r_beta_mean"]
    })

rq3_df = pd.DataFrame(rq3_data)
rq3_df.to_csv("outputs/rq3_recovery_table.csv", index=False)
print("RQ3 recovery table saved to outputs/rq3_recovery_table.csv")

# RQ4: Scalability (Q × Encoding)
rq4_data = []
for q in [200, 500, 1000, 2000]:
    for enc in ["linear_decay", "separable", "static_item"]:
        exp_name = f"q{q}_k4_{enc}"
        if exp_name in merged["experiment"].values:
            row = merged[merged["experiment"] == exp_name].iloc[0]
            rq4_data.append({
                "Q": q,
                "Encoding": enc,
                "QWK": row["qwk"],
                "ACC": row["acc"],
                "r_alpha": row["r_alpha"],
                "r_beta": row["r_beta_mean"]
            })

rq4_df = pd.DataFrame(rq4_data)
rq4_df.to_csv("outputs/rq4_scalability_table.csv", index=False)
print("RQ4 scalability table saved to outputs/rq4_scalability_table.csv")

# Ablation: Loss components at Q=200, K=4
ablation_loss_data = []
for exp_name, label in [
    ("q200_k4_static_item", "Full (focal+WOL)"),
    ("q200_k4_focal_only", "Focal only"),
    ("q200_k4_wol_only", "WOL only")
]:
    if exp_name in merged["experiment"].values:
        row = merged[merged["experiment"] == exp_name].iloc[0]
        ablation_loss_data.append({
            "Loss": label,
            "QWK": row["qwk"],
            "tau": row["tau"],
            "r_alpha": row["r_alpha"],
            "r_beta": row["r_beta_mean"]
        })

ablation_loss_df = pd.DataFrame(ablation_loss_data)
ablation_loss_df.to_csv("outputs/ablation_loss_table.csv", index=False)
print("Loss ablation table saved to outputs/ablation_loss_table.csv")

# Ablation: Monotonic constraint at Q=200, K=4
ablation_mono_data = []
for exp_name, label in [
    ("q200_k4_static_item", "Monotonic + ordinal"),
    ("q200_k4_unconstrained", "Unconstrained + ordinal")
]:
    if exp_name in merged["experiment"].values:
        row = merged[merged["experiment"] == exp_name].iloc[0]
        ablation_mono_data.append({
            "Model": label,
            "ACC": row["acc"],
            "QWK": row["qwk"],
            "tau": row["tau"],
            "r_alpha": row["r_alpha"],
            "r_beta": row["r_beta_mean"]
        })

ablation_mono_df = pd.DataFrame(ablation_mono_data)
ablation_mono_df.to_csv("outputs/ablation_monotonic_table.csv", index=False)
print("Monotonic ablation table saved to outputs/ablation_monotonic_table.csv")

print("\nAll tables generated successfully!")
