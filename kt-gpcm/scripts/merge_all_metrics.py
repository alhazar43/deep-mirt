#!/usr/bin/env python3
"""Merge prediction metrics with parameter recovery correlations.

Creates comprehensive tables for the paper with both prediction and interpretability metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def parse_experiment_name(exp_name):
    """Extract Q, K, encoding from experiment name."""
    parts = exp_name.split("_")

    # Handle different naming patterns
    if exp_name.startswith("large_"):
        # Baseline experiments: large_q200_k4_dkvmn_ordinal
        parts = parts[1:]  # Remove 'large_'
        q = int(parts[0][1:])  # q200 -> 200
        k = int(parts[1][1:])  # k4 -> 4
        model = "_".join(parts[2:])  # dkvmn_ordinal, static_gpcm, etc.
        return {"Q": q, "K": k, "model": model, "encoding": None}
    else:
        # Main experiments: q200_k4_linear_decay
        q = int(parts[0][1:])  # q200 -> 200
        k = int(parts[1][1:])  # k4 -> 4
        encoding = "_".join(parts[2:])  # linear_decay, static_item, etc.
        return {"Q": q, "K": k, "model": "deepgpcm", "encoding": encoding}


def load_prediction_metrics():
    """Load prediction metrics from summary CSVs."""
    # Load both files
    all_exp = pd.read_csv("outputs/summary_all_experiments.csv")
    baselines = pd.read_csv("outputs/summary_baselines.csv")

    # Parse experiment names
    all_exp["parsed"] = all_exp["experiment"].apply(parse_experiment_name)
    baselines["parsed"] = baselines["experiment"].apply(parse_experiment_name)

    # Expand parsed dict into columns
    all_exp = pd.concat([all_exp, pd.json_normalize(all_exp["parsed"])], axis=1)
    baselines = pd.concat([baselines, pd.json_normalize(baselines["parsed"])], axis=1)

    # Combine
    combined = pd.concat([all_exp, baselines], ignore_index=True)

    return combined


def create_rq1_table(df):
    """Create RQ1 main comparison table (K=3,4,5,6 only, Q=200)."""
    # Filter: Q=200, K in {3,4,5,6}
    rq1_data = df[(df["Q"] == 200) & (df["K"].isin([3, 4, 5, 6]))].copy()

    # Map model names
    model_map = {
        "static_gpcm": "Static GPCM",
        "dynamic_gpcm": "Dynamic GPCM",
        "dkvmn_softmax": "DKVMN+Softmax",
        "dkvmn_ordinal": "DKVMN+Ordinal",
        "deepgpcm": "DEEP-GPCM"
    }

    # For DEEP-GPCM, use static_item encoding as default
    rq1_data = rq1_data[
        (rq1_data["model"].isin(model_map.keys())) &
        ((rq1_data["encoding"] == "static_item") | (rq1_data["encoding"].isna()))
    ]

    rq1_data["model_name"] = rq1_data["model"].map(model_map)

    # Pivot to wide format
    pivot = rq1_data.pivot_table(
        index="model_name",
        columns="K",
        values=["qwk", "acc", "r_alpha", "r_beta_mean"],
        aggfunc="first"
    )

    return pivot


def create_rq4_table(df):
    """Create RQ4 scalability table (Q × encoding × recovery)."""
    # Filter: K=4, all Q values, all encodings
    rq4_data = df[(df["K"] == 4) & (df["model"] == "deepgpcm")].copy()

    # Encoding name mapping
    encoding_map = {
        "linear_decay": "LinearDecay",
        "separable": "Separable",
        "static_item": "SIE"
    }
    rq4_data["encoding_name"] = rq4_data["encoding"].map(encoding_map)

    # Select columns
    rq4_table = rq4_data[["Q", "encoding_name", "qwk", "acc", "r_alpha", "r_beta_mean"]].copy()
    rq4_table = rq4_table.sort_values(["Q", "encoding_name"])

    return rq4_table


def create_ablation_tables(df):
    """Create ablation study tables with recovery metrics."""
    # Loss components: q200_k4_focal_only, q200_k4_wol_only, q200_k4_static_item (full)
    loss_data = df[
        (df["Q"] == 200) & (df["K"] == 4) &
        (df["encoding"].isin(["focal_only", "wol_only", "static_item"]))
    ].copy()

    loss_map = {
        "static_item": "Full (focal + WOL)",
        "focal_only": "Focal only",
        "wol_only": "WOL only"
    }
    loss_data["variant"] = loss_data["encoding"].map(loss_map)
    loss_table = loss_data[["variant", "qwk", "tau", "r_alpha", "r_beta_mean"]]

    # Monotonic constraint: q200_k4_static_item (constrained), q200_k4_unconstrained
    mono_data = df[
        (df["Q"] == 200) & (df["K"] == 4) &
        (df["encoding"].isin(["static_item", "unconstrained"]))
    ].copy()

    mono_map = {
        "static_item": "Monotonic + ordinal",
        "unconstrained": "Unconstrained + ordinal"
    }
    mono_data["variant"] = mono_data["encoding"].map(mono_map)
    mono_table = mono_data[["variant", "acc", "qwk", "tau", "r_alpha", "r_beta_mean"]]

    return loss_table, mono_table


def main():
    # Load prediction metrics
    print("Loading prediction metrics...")
    df = load_prediction_metrics()

    # Check if recovery correlations exist
    recovery_path = Path("outputs/recovery_correlations.csv")
    if recovery_path.exists():
        print("Loading recovery correlations...")
        recovery = pd.read_csv(recovery_path)

        # Merge with main dataframe
        df = df.merge(recovery, on="experiment", how="left")
    else:
        print("WARNING: recovery_correlations.csv not found. Run compute_all_recovery.py first.")
        df["r_alpha"] = np.nan
        df["r_beta_mean"] = np.nan

    # Create tables
    print("\n=== RQ1: Main Comparison (K=3,4,5,6, Q=200) ===")
    rq1 = create_rq1_table(df)
    print(rq1.to_string())
    rq1.to_csv("outputs/table_rq1_main_comparison.csv")

    print("\n=== RQ4: Scalability (Q × encoding, K=4) ===")
    rq4 = create_rq4_table(df)
    print(rq4.to_string(index=False))
    rq4.to_csv("outputs/table_rq4_scalability.csv", index=False)

    print("\n=== Ablations ===")
    loss_table, mono_table = create_ablation_tables(df)
    print("\nLoss Components:")
    print(loss_table.to_string(index=False))
    loss_table.to_csv("outputs/table_ablation_loss.csv", index=False)

    print("\nMonotonic Constraint:")
    print(mono_table.to_string(index=False))
    mono_table.to_csv("outputs/table_ablation_monotonic.csv", index=False)

    print("\n=== Summary Statistics ===")
    print(f"Total experiments: {len(df)}")
    print(f"With recovery data: {df['r_alpha'].notna().sum()}")
    print(f"Missing recovery data: {df['r_alpha'].isna().sum()}")

    # Save full merged dataset
    df.to_csv("outputs/merged_all_metrics.csv", index=False)
    print("\nSaved merged dataset to outputs/merged_all_metrics.csv")


if __name__ == "__main__":
    main()
