#!/usr/bin/env python3
"""
Comprehensive diagnosis of Q=2000 alpha recovery degradation.
"""

import json
import numpy as np
from pathlib import Path

print("=" * 80)
print("COMPREHENSIVE DIAGNOSIS: Q=2000 ALPHA RECOVERY DEGRADATION")
print("=" * 80)
print()

# ============================================================================
# PART 1: ITEM COVERAGE ANALYSIS
# ============================================================================
print("PART 1: ITEM COVERAGE ANALYSIS")
print("-" * 80)

datasets = {
    "Q=200": {"n_students": 5000, "n_questions": 200, "seq_len_range": [20, 80]},
    "Q=500": {"n_students": 5000, "n_questions": 500, "seq_len_range": [20, 80]},
    "Q=1000": {"n_students": 5000, "n_questions": 1000, "seq_len_range": [20, 80]},
    "Q=2000": {"n_students": 3000, "n_questions": 2000, "seq_len_range": [50, 120]},
    "Q=5000": {"n_students": 1000, "n_questions": 5000, "seq_len_range": [400, 600]},
}

coverage_data = []
for name, config in datasets.items():
    n_students = config["n_students"]
    n_questions = config["n_questions"]
    seq_min, seq_max = config["seq_len_range"]
    avg_seq_len = (seq_min + seq_max) / 2

    total_responses = n_students * avg_seq_len
    avg_responses_per_item = total_responses / n_questions

    coverage_data.append({
        "name": name,
        "n_students": n_students,
        "n_questions": n_questions,
        "avg_seq_len": avg_seq_len,
        "total_responses": total_responses,
        "responses_per_item": avg_responses_per_item,
    })

    print(f"{name}:")
    print(f"  Students: {n_students:,}")
    print(f"  Questions: {n_questions:,}")
    print(f"  Avg seq length: {avg_seq_len:.1f}")
    print(f"  Total responses: {total_responses:,.0f}")
    print(f"  Responses per item: {avg_responses_per_item:.1f}")
    print()

# ============================================================================
# PART 2: RECOVERY CORRELATIONS
# ============================================================================
print()
print("PART 2: ALPHA RECOVERY CORRELATIONS (from merged_metrics_recovery.csv)")
print("-" * 80)

recovery_results = {
    "Q=200 Separable": 0.748,
    "Q=200 SIE": 0.746,
    "Q=500 Separable": 0.806,
    "Q=500 SIE": 0.686,
    "Q=1000 Separable": 0.644,
    "Q=1000 SIE": 0.548,
    "Q=2000 Separable": 0.275,
    "Q=2000 SIE": 0.351,
    "Q=5000 SIE (archived)": 0.654,
}

for name, r_alpha in recovery_results.items():
    print(f"{name}: r_α = {r_alpha:.3f}")

print()

# ============================================================================
# PART 3: KEY FINDINGS
# ============================================================================
print()
print("PART 3: KEY FINDINGS")
print("=" * 80)
print()

print("FINDING 1: Item Coverage Paradox")
print("-" * 40)
print("Q=2000 has 127.5 responses/item")
print("Q=5000 has 100.0 responses/item (21% LESS)")
print("Yet Q=5000 achieves r_α=0.654 vs Q=2000's r_α=0.275-0.351")
print()
print("This suggests raw item coverage is NOT the primary issue.")
print()

print("FINDING 2: Sequence Length Matters More Than Item Coverage")
print("-" * 40)
print("Q=5000: avg_seq_len = 500 (each student sees 10% of items)")
print("Q=2000: avg_seq_len = 85 (each student sees 4.25% of items)")
print("Q=1000: avg_seq_len = 50 (each student sees 5% of items)")
print()
print("Longer sequences provide:")
print("  - More observations per student for θ estimation")
print("  - Better gradient signal for item parameters")
print("  - More diverse item exposure per student")
print()

print("FINDING 3: Training Convergence")
print("-" * 40)
print("Q=1000 epoch 15: train_loss=1.218, val_qwk=0.754")
print("Q=2000 epoch 15: train_loss=1.215, val_qwk=0.759")
print()
print("Both models achieve similar QWK and loss, suggesting:")
print("  - Models converge to similar predictive performance")
print("  - But Q=2000 fails to recover interpretable IRT parameters")
print("  - This is a PARAMETER IDENTIFIABILITY issue, not optimization failure")
print()

print("FINDING 4: The Critical Difference")
print("-" * 40)
print("Q=200-1000 all use:")
print("  - 5000 students")
print("  - seq_len [20, 80]")
print("  - Same training config (15 epochs, batch_size=64)")
print()
print("Q=2000 uses:")
print("  - 3000 students (40% fewer)")
print("  - seq_len [50, 120] (longer, but not enough)")
print("  - Same training config")
print()
print("Q=5000 uses:")
print("  - 1000 students (80% fewer than Q=200-1000)")
print("  - seq_len [400, 600] (10x longer!)")
print("  - Same training config")
print()

# ============================================================================
# PART 4: ROOT CAUSE ANALYSIS
# ============================================================================
print()
print("PART 4: ROOT CAUSE ANALYSIS")
print("=" * 80)
print()

print("ROOT CAUSE: Insufficient Student-Item Interaction Density")
print("-" * 40)
print()
print("The model needs sufficient observations to disentangle:")
print("  1. Student ability (θ) - requires long sequences per student")
print("  2. Item discrimination (α) - requires diverse students per item")
print("  3. Item difficulty (β) - requires diverse students per item")
print()
print("Q=2000 fails because:")
print("  [X] Only 3000 students (vs 5000 for Q<=1000)")
print("  [X] Short sequences (avg 85 vs 500 for Q=5000)")
print("  [X] Each student sees only 4.25% of items")
print("  [X] Insufficient diversity in student-item interactions")
print()
print("Q=5000 succeeds despite fewer students because:")
print("  [OK] Very long sequences (avg 500) provide rich theta estimates")
print("  [OK] Each student sees 10% of items (2.4x more than Q=2000)")
print("  [OK] Better gradient signal for item parameters")
print()

# ============================================================================
# PART 5: RECOMMENDATIONS
# ============================================================================
print()
print("PART 5: RECOMMENDATIONS")
print("=" * 80)
print()

print("OPTION 1: Regenerate Q=2000 dataset (RECOMMENDED)")
print("-" * 40)
print("New configuration:")
print("  - n_students: 5000 (match Q≤1000)")
print("  - seq_len_range: [100, 200] (2-4x longer)")
print("  - This gives ~750 responses/item (6x current)")
print("  - Each student sees 7.5% of items (1.8x current)")
print()

print("OPTION 2: Train Q=2000 for more epochs")
print("-" * 40)
print("Current: 15 epochs")
print("Suggested: 30-50 epochs")
print("Rationale:")
print("  - More epochs may help with parameter identifiability")
print("  - But unlikely to fully solve the data sparsity issue")
print("  - Training curves show continued improvement at epoch 15")
print()

print("OPTION 3: Increase batch size for Q=2000")
print("-" * 40)
print("Current: batch_size=64")
print("Suggested: batch_size=128 or 256")
print("Rationale:")
print("  - Larger batches provide more diverse gradient signal")
print("  - May help with parameter identifiability")
print("  - But won't solve fundamental data sparsity")
print()

print("OPTION 4: Add regularization for Q=2000")
print("-" * 40)
print("Current: alpha_prior_weight=0.0")
print("Suggested: alpha_prior_weight=0.01-0.1")
print("Rationale:")
print("  - Prior on alpha can help with identifiability")
print("  - But may bias recovery metrics")
print("  - Not a fundamental solution")
print()

print()
print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print()
print("The Q=2000 dataset has insufficient student-item interaction density")
print("for the model to recover interpretable IRT parameters, despite achieving")
print("good predictive performance (QWK=0.76).")
print()
print("RECOMMENDED ACTION: Regenerate Q=2000 with 5000 students and longer")
print("sequences [100, 200] to match the interaction density of Q≤1000.")
print()
print("=" * 80)
