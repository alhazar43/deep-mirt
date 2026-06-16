"""
run_bt_demo.py -- Orchestrator for the Bradley-Terry / GPCM bridge demo.

Usage:
    cd C:/Users/steph/documents/deep-mirt
    PYTHONPATH=deep_irt python deep_irt/bt_demo/run_bt_demo.py

Outputs:
    deep_irt/bt_demo/outputs/report.txt
    deep_irt/bt_demo/outputs/results.pt   (torch save of all arrays)
"""

import os
import sys
import time
import torch

# Seed everything globally
SEED = 0
torch.manual_seed(SEED)

# Make sure we can import sibling modules even if PYTHONPATH is set to deep_irt
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from data_gen import generate_ground_truth, generate_mode_a, generate_mode_b
from fit_gpcm import fit_gpcm
from fit_bt import fit_bt
from metrics import compute_all_metrics, format_report

OUTPUT_DIR = os.path.join(_here, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    t0 = time.time()
    print("=" * 60)
    print("BT/GPCM Bridge Demo")
    print(f"Seed: {SEED}  |  Device: CPU")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Step 1: Ground truth
    # ------------------------------------------------------------------ #
    print("\n[1/5] Generating ground-truth parameters ...")
    gt = generate_ground_truth(n_items=200, seed=SEED)
    print(f"  {gt['n_items']} items  |  K={gt['K']}  |  "
          f"s in [{gt['s'].min():.2f}, {gt['s'].max():.2f}]  "
          f"a in [{gt['a'].min():.2f}, {gt['a'].max():.2f}]")

    # ------------------------------------------------------------------ #
    # Step 2: Mode A data (direct ordinal responses)
    # ------------------------------------------------------------------ #
    print("\n[2/5] Generating Mode-A data (GPCM ordinal responses) ...")
    mode_a = generate_mode_a(
        gt,
        n_respondents=2000,
        items_per_respondent=40,
        seed=SEED,
    )
    print(f"  {mode_a['n_respondents']} respondents  |  "
          f"{len(mode_a['responses'])} observations  |  "
          f"responses in {{{mode_a['responses'].unique().tolist()}}}")

    # ------------------------------------------------------------------ #
    # Step 3: Mode B data (pairwise comparisons)
    # ------------------------------------------------------------------ #
    print("\n[3/5] Generating Mode-B data (Bradley-Terry pairwise) ...")
    mode_b = generate_mode_b(gt, comparisons_per_item=50, seed=SEED)
    print(f"  {mode_b['n_pairs']} pairs  |  "
          f"{mode_b['outcome'].sum().item()} item-i wins  "
          f"({100*mode_b['outcome'].float().mean().item():.1f}%)")

    # ------------------------------------------------------------------ #
    # Step 4a: Fit GPCM (Mode A)
    # ------------------------------------------------------------------ #
    print("\n[4/5] Fitting GPCM to Mode-A data ...")
    gpcm_results = fit_gpcm(
        mode_a,
        gt,
        n_epochs=300,
        lr=0.05,
        batch_size=81920,  # full-batch (total obs ~80k)
        seed=SEED,
        verbose=True,
    )
    print(f"  GPCM fit done. Final NLL = {gpcm_results['final_nll']:.4f}")

    # ------------------------------------------------------------------ #
    # Step 4b: Fit BT (Mode B)
    # ------------------------------------------------------------------ #
    print("\n[4b/5] Fitting Bradley-Terry to Mode-B data ...")
    bt_results = fit_bt(
        mode_b,
        gt,
        n_epochs=500,
        lr=0.05,
        batch_size=81920,  # full-batch
        seed=SEED,
        verbose=True,
    )
    print(f"  BT fit done. Final NLL = {bt_results['final_nll']:.4f}")

    # ------------------------------------------------------------------ #
    # Step 5: Metrics + report
    # ------------------------------------------------------------------ #
    print("\n[5/5] Computing metrics ...")
    metrics = compute_all_metrics(gt, gpcm_results, bt_results)

    report = format_report(gt, mode_a, mode_b, metrics, gpcm_results, bt_results)
    print("\n" + report)

    # Save report
    report_path = os.path.join(OUTPUT_DIR, "report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport written to {report_path}")

    # Save arrays
    results_path = os.path.join(OUTPUT_DIR, "results.pt")
    torch.save(
        {
            "ground_truth": gt,
            "mode_a_data": mode_a,
            "mode_b_data": mode_b,
            "gpcm_results": gpcm_results,
            "bt_results": bt_results,
            "metrics": metrics,
        },
        results_path,
    )
    print(f"Results saved to  {results_path}")

    elapsed = time.time() - t0
    print(f"\nTotal wall time: {elapsed:.1f}s")

    return metrics


if __name__ == "__main__":
    main()
