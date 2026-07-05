#!/usr/bin/env python
"""
Reconcile Q-MIRT p1 cell outputs: aggregate per-seed metrics,
produce summary JSON with per-seed VALUES, and print a table
with sign-inconsistency flagging.
"""
import json
import glob
from pathlib import Path
from collections import defaultdict
import numpy as np


def sign_positive(x):
    """Return 1 if x > 0, else 0."""
    return 1 if x > 0 else 0


def reconcile():
    # Paths
    data_dir = Path(__file__).parent / "outputs" / "qm2" / "p1"
    cell_pattern = str(data_dir / "cell_*.json")
    output_file = data_dir / "p1_summary_reconciled.json"

    # Load all cells
    cells = []
    for cell_path in sorted(glob.glob(cell_pattern)):
        with open(cell_path) as f:
            cells.append(json.load(f))

    # Group by (kind, calib)
    grouped = defaultdict(list)
    for cell in cells:
        key = (cell["kind"], cell["calib"])
        grouped[key].append(cell)

    # Build summary: for each group, compute stats for each metric
    summary = {}
    table_data = []

    for (kind, calib), group_cells in sorted(grouped.items()):
        group_key = f"{kind}_{calib}"
        summary[group_key] = {}

        # All metrics we might see
        all_keys = set()
        for cell in group_cells:
            all_keys.update(cell.keys())
        all_keys.discard("kind")
        all_keys.discard("calib")
        all_keys.discard("data_seed")
        all_keys.discard("secs")  # Exclude timing

        # Compute stats for each metric
        for metric in sorted(all_keys):
            values = []
            for cell in group_cells:
                if metric in cell:
                    values.append(cell[metric])

            if values:
                values_arr = np.array(values)
                summary[group_key][metric] = {
                    "values": list(values_arr),
                    "mean": float(np.mean(values_arr)),
                    "sd": float(np.std(values_arr, ddof=1) if len(values) > 1 else 0.0),
                    "min": float(np.min(values_arr)),
                    "max": float(np.max(values_arr)),
                    "n": len(values),
                    "n_sign_positive": sum(sign_positive(v) for v in values),
                }

        # Prepare table row
        row = {
            "kind": kind,
            "calib": calib,
        }

        # alpha_spearman: values + mean
        if "alpha_spearman" in summary[group_key]:
            alpha_vals = summary[group_key]["alpha_spearman"]["values"]
            alpha_mean = summary[group_key]["alpha_spearman"]["mean"]
            row["alpha_spearman"] = f"[{', '.join(f'{v:.4f}' for v in alpha_vals)}] mean={alpha_mean:.4f}"

            # Flag if values disagree in sign
            has_pos = any(v > 0 for v in alpha_vals)
            has_neg = any(v < 0 for v in alpha_vals)
            if has_pos and has_neg:
                row["flag"] = "!"
        else:
            row["alpha_spearman"] = "N/A"
            row["flag"] = ""

        # alpha_spearman_refit if present
        if "alpha_spearman_refit" in summary[group_key]:
            refit_vals = summary[group_key]["alpha_spearman_refit"]["values"]
            refit_mean = summary[group_key]["alpha_spearman_refit"]["mean"]
            row["alpha_spearman_refit"] = f"[{', '.join(f'{v:.4f}' for v in refit_vals)}] mean={refit_mean:.4f}"

        # delta if present
        if "delta" in summary[group_key]:
            delta_vals = summary[group_key]["delta"]["values"]
            delta_mean = summary[group_key]["delta"]["mean"]
            row["delta"] = f"[{', '.join(f'{v:.4f}' for v in delta_vals)}] mean={delta_mean:.4f}"

        # d_loc_spearman mean
        if "d_loc_spearman" in summary[group_key]:
            row["d_loc_spearman_mean"] = f"{summary[group_key]['d_loc_spearman']['mean']:.4f}"

        # pos_bias_r mean
        if "pos_bias_r" in summary[group_key]:
            row["pos_bias_r_mean"] = f"{summary[group_key]['pos_bias_r']['mean']:.4f}"

        table_data.append(row)

    # Write summary JSON
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary to {output_file}")

    # Print table
    print("\n" + "=" * 160)
    print("RECONCILIATION TABLE")
    print("=" * 160)

    # Print header
    header_parts = ["kind_calib", "alpha_spearman", "alpha_spearman_refit", "delta", "d_loc_mean", "pos_bias_mean", "flag"]
    print(f"{'kind':15} {'calib':15} alpha_spearman (values + mean)")
    print("-" * 160)

    sign_inconsistent = []
    for row in table_data:
        kind = row["kind"]
        calib = row["calib"]
        alpha_sp = row.get("alpha_spearman", "N/A")
        alpha_refit = row.get("alpha_spearman_refit", "")
        delta = row.get("delta", "")
        d_loc_mean = row.get("d_loc_spearman_mean", "N/A")
        pos_bias_mean = row.get("pos_bias_r_mean", "N/A")
        flag = row.get("flag", "")

        # Print the main line
        print(f"{kind:15} {calib:15} {alpha_sp}")

        # Print optional fields on separate indented lines if present
        if alpha_refit:
            print(f"{'':15} {'':15}   alpha_spearman_refit: {alpha_refit}")
        if delta:
            print(f"{'':15} {'':15}   delta: {delta}")
        print(f"{'':15} {'':15}   d_loc_spearman_mean={d_loc_mean}, pos_bias_r_mean={pos_bias_mean}")

        if flag == "!":
            sign_inconsistent.append(f"{kind}_{calib}")
            print(f"{'':15} {'':15}   ! SIGN INCONSISTENCY in alpha_spearman")

        print()

    print("=" * 160)

    # Report sign inconsistencies
    if sign_inconsistent:
        print(f"\nSign-inconsistent groups: {', '.join(sign_inconsistent)}")
    else:
        print("\nNo sign inconsistencies detected.")


if __name__ == "__main__":
    reconcile()
