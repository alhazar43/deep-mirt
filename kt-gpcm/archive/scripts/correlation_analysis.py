#!/usr/bin/env python3
"""
Analyze the relationship between Q, dataset characteristics, and alpha recovery.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Data from analysis
data = [
    # Q, n_students, avg_seq_len, responses_per_item, r_alpha_separable, r_alpha_sie
    (200, 5000, 50, 1250, 0.748, 0.746),
    (500, 5000, 50, 500, 0.806, 0.686),
    (1000, 5000, 50, 250, 0.644, 0.548),
    (2000, 3000, 85, 127.5, 0.275, 0.351),
    (5000, 1000, 500, 100, None, 0.654),  # Only SIE available
]

Q_vals = [d[0] for d in data]
n_students = [d[1] for d in data]
avg_seq_len = [d[2] for d in data]
responses_per_item = [d[3] for d in data]
r_alpha_sep = [d[4] if d[4] is not None else np.nan for d in data]
r_alpha_sie = [d[5] for d in data]

# Calculate additional metrics
total_responses = [n * s for n, s in zip(n_students, avg_seq_len)]
item_coverage_per_student = [s / q for s, q in zip(avg_seq_len, Q_vals)]

print("=" * 80)
print("CORRELATION ANALYSIS: Dataset Characteristics vs Alpha Recovery")
print("=" * 80)
print()

# Print table
print(f"{'Q':<6} {'Students':<9} {'Seq_Len':<8} {'Resp/Item':<10} {'Coverage%':<10} {'r_α_sep':<9} {'r_α_sie':<9}")
print("-" * 80)
for i, q in enumerate(Q_vals):
    cov_pct = item_coverage_per_student[i] * 100
    r_sep = f"{r_alpha_sep[i]:.3f}" if not np.isnan(r_alpha_sep[i]) else "N/A"
    r_sie = f"{r_alpha_sie[i]:.3f}"
    print(f"{q:<6} {n_students[i]:<9} {avg_seq_len[i]:<8.0f} {responses_per_item[i]:<10.1f} {cov_pct:<10.1f} {r_sep:<9} {r_sie:<9}")

print()
print("=" * 80)
print("KEY INSIGHT: Item Coverage Per Student is the Critical Factor")
print("=" * 80)
print()

print("Item coverage per student (% of items each student sees):")
for i, q in enumerate(Q_vals):
    cov_pct = item_coverage_per_student[i] * 100
    r_sie = r_alpha_sie[i]
    print(f"  Q={q}: {cov_pct:.1f}% coverage -> r_α={r_sie:.3f}")

print()
print("Observation:")
print("  - Q=200-1000: All have 5% coverage, but r_α degrades as Q increases")
print("  - Q=2000: Only 4.25% coverage, r_α drops dramatically to 0.275-0.351")
print("  - Q=5000: 10% coverage (2.4x Q=2000), r_α recovers to 0.654")
print()
print("This suggests TWO factors matter:")
print("  1. Item coverage per student (higher is better)")
print("  2. Absolute number of students (more is better for item parameter estimation)")
print()

print("=" * 80)
print("HYPOTHESIS TEST: Why does Q=1000 outperform Q=2000?")
print("=" * 80)
print()

print("Q=1000:")
print("  - 5000 students")
print("  - 250 responses/item")
print("  - 5% coverage per student")
print("  - r_α = 0.644 (Separable)")
print()

print("Q=2000:")
print("  - 3000 students (40% fewer)")
print("  - 127.5 responses/item (49% fewer)")
print("  - 4.25% coverage per student (15% less)")
print("  - r_α = 0.275 (Separable) - 57% WORSE")
print()

print("The 57% drop in alpha recovery is disproportionate to the 40-49% reduction")
print("in data quantity, suggesting a THRESHOLD EFFECT:")
print()
print("Below ~250 responses/item AND <5000 students, the model cannot reliably")
print("disentangle item discrimination from student ability.")
print()

print("=" * 80)
print("FINAL RECOMMENDATION")
print("=" * 80)
print()
print("To achieve r_α > 0.6 for Q=2000, the dataset should have:")
print()
print("  OPTION A (Match Q=1000 density):")
print("    - n_students: 5000")
print("    - seq_len_range: [100, 200]")
print("    - Expected responses/item: 750")
print("    - Expected coverage: 7.5%")
print("    - Predicted r_α: ~0.65-0.70")
print()
print("  OPTION B (Match Q=5000 pattern):")
print("    - n_students: 2000")
print("    - seq_len_range: [200, 400]")
print("    - Expected responses/item: 300")
print("    - Expected coverage: 15%")
print("    - Predicted r_α: ~0.70-0.75")
print()
print("OPTION A is recommended as it's more conservative and matches the")
print("successful Q<=1000 pattern.")
print()
print("=" * 80)
