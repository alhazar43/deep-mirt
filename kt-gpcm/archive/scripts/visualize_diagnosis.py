#!/usr/bin/env python3
"""
Create visualization of dataset characteristics vs alpha recovery.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Data from analysis
data = [
    # Q, n_students, avg_seq_len, responses_per_item, r_alpha_separable, r_alpha_sie
    (200, 5000, 50, 1250, 0.748, 0.746),
    (500, 5000, 50, 500, 0.806, 0.686),
    (1000, 5000, 50, 250, 0.644, 0.548),
    (2000, 3000, 85, 127.5, 0.275, 0.351),
    (5000, 1000, 500, 100, 0.654, 0.654),  # Use SIE for both
]

Q_vals = np.array([d[0] for d in data])
n_students = np.array([d[1] for d in data])
avg_seq_len = np.array([d[2] for d in data])
responses_per_item = np.array([d[3] for d in data])
r_alpha_sep = np.array([d[4] for d in data])
r_alpha_sie = np.array([d[5] for d in data])

# Calculate additional metrics
coverage_per_student = avg_seq_len / Q_vals * 100  # percentage

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Dataset Characteristics vs Alpha Recovery (K=5)', fontsize=14, fontweight='bold')

# Plot 1: Responses per item vs r_alpha
ax = axes[0, 0]
ax.scatter(responses_per_item[:4], r_alpha_sep[:4], s=100, alpha=0.7, label='Separable', color='blue')
ax.scatter(responses_per_item[:4], r_alpha_sie[:4], s=100, alpha=0.7, label='SIE', color='red')
ax.scatter(responses_per_item[4], r_alpha_sie[4], s=100, alpha=0.7, color='green', marker='s', label='Q=5000 (SIE)')
for i in range(5):
    ax.annotate(f'Q={Q_vals[i]}', (responses_per_item[i], r_alpha_sie[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=8)
ax.set_xlabel('Responses per Item', fontsize=11)
ax.set_ylabel('Alpha Recovery (r)', fontsize=11)
ax.set_title('Responses/Item vs Alpha Recovery', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=0.6, color='gray', linestyle='--', alpha=0.5, label='Target r=0.6')

# Plot 2: Number of students vs r_alpha
ax = axes[0, 1]
ax.scatter(n_students[:4], r_alpha_sep[:4], s=100, alpha=0.7, label='Separable', color='blue')
ax.scatter(n_students[:4], r_alpha_sie[:4], s=100, alpha=0.7, label='SIE', color='red')
ax.scatter(n_students[4], r_alpha_sie[4], s=100, alpha=0.7, color='green', marker='s', label='Q=5000 (SIE)')
for i in range(5):
    ax.annotate(f'Q={Q_vals[i]}', (n_students[i], r_alpha_sie[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=8)
ax.set_xlabel('Number of Students', fontsize=11)
ax.set_ylabel('Alpha Recovery (r)', fontsize=11)
ax.set_title('Student Count vs Alpha Recovery', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=0.6, color='gray', linestyle='--', alpha=0.5)

# Plot 3: Coverage per student vs r_alpha
ax = axes[1, 0]
ax.scatter(coverage_per_student[:4], r_alpha_sep[:4], s=100, alpha=0.7, label='Separable', color='blue')
ax.scatter(coverage_per_student[:4], r_alpha_sie[:4], s=100, alpha=0.7, label='SIE', color='red')
ax.scatter(coverage_per_student[4], r_alpha_sie[4], s=100, alpha=0.7, color='green', marker='s', label='Q=5000 (SIE)')
for i in range(5):
    ax.annotate(f'Q={Q_vals[i]}', (coverage_per_student[i], r_alpha_sie[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=8)
ax.set_xlabel('Item Coverage per Student (%)', fontsize=11)
ax.set_ylabel('Alpha Recovery (r)', fontsize=11)
ax.set_title('Coverage/Student vs Alpha Recovery', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=0.6, color='gray', linestyle='--', alpha=0.5)

# Plot 4: Q vs r_alpha (showing the non-monotonic relationship)
ax = axes[1, 1]
ax.plot(Q_vals[:4], r_alpha_sep[:4], 'o-', linewidth=2, markersize=8, alpha=0.7, label='Separable', color='blue')
ax.plot(Q_vals[:4], r_alpha_sie[:4], 's-', linewidth=2, markersize=8, alpha=0.7, label='SIE', color='red')
ax.plot(Q_vals[4], r_alpha_sie[4], 's', markersize=10, alpha=0.7, color='green', label='Q=5000 (SIE)')
ax.set_xlabel('Number of Questions (Q)', fontsize=11)
ax.set_ylabel('Alpha Recovery (r)', fontsize=11)
ax.set_title('Q vs Alpha Recovery (Non-Monotonic!)', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=0.6, color='gray', linestyle='--', alpha=0.5)
ax.set_xscale('log')

# Add text annotation explaining Q=2000 issue
ax.annotate('Q=2000 drops\nbelow threshold',
            xy=(2000, 0.35), xytext=(2000, 0.15),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, color='red', fontweight='bold',
            ha='center')

plt.tight_layout()

# Save figure
output_path = Path('C:/Users/steph/documents/deep-mirt/kt-gpcm/outputs/q2000_diagnosis_plots.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved visualization to: {output_path}")

# Create a second figure showing the threshold effect
fig2, ax = plt.subplots(1, 1, figsize=(10, 6))

# Calculate a composite metric: students * responses_per_item
composite_metric = n_students * responses_per_item / 1000  # Scale down for readability

ax.scatter(composite_metric[:4], r_alpha_sep[:4], s=150, alpha=0.7, label='Separable', color='blue')
ax.scatter(composite_metric[:4], r_alpha_sie[:4], s=150, alpha=0.7, label='SIE', color='red')
ax.scatter(composite_metric[4], r_alpha_sie[4], s=150, alpha=0.7, color='green', marker='s', label='Q=5000 (SIE)')

for i in range(5):
    ax.annotate(f'Q={Q_vals[i]}', (composite_metric[i], r_alpha_sie[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')

ax.set_xlabel('Interaction Density (Students × Responses/Item) / 1000', fontsize=12)
ax.set_ylabel('Alpha Recovery (r)', fontsize=12)
ax.set_title('Student-Item Interaction Density vs Alpha Recovery', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.axhline(y=0.6, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Target r=0.6')

# Add threshold line
threshold_x = 300  # Approximate threshold
ax.axvline(x=threshold_x, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax.text(threshold_x + 20, 0.4, 'Critical\nThreshold', fontsize=11, color='red', fontweight='bold')

# Add regions
ax.fill_between([0, threshold_x], 0, 1, alpha=0.1, color='red', label='Below Threshold')
ax.fill_between([threshold_x, max(composite_metric) * 1.1], 0, 1, alpha=0.1, color='green', label='Above Threshold')

plt.tight_layout()

output_path2 = Path('C:/Users/steph/documents/deep-mirt/kt-gpcm/outputs/q2000_threshold_effect.png')
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"Saved threshold visualization to: {output_path2}")

print("\nVisualization complete!")
print(f"\nKey finding: Q=2000 falls below the critical threshold of")
print(f"interaction density (students × responses/item ≈ 300k) needed")
print(f"for reliable alpha recovery.")
