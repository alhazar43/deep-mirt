#!/usr/bin/env python3
"""
Analyze true alpha distributions from datasets.
"""

import json
import numpy as np
from pathlib import Path

base_dir = Path("C:/Users/steph/documents/deep-mirt/kt-gpcm")

datasets = [
    ("Q=200", "data/large_q200_k5/true_irt_parameters.json"),
    ("Q=500", "data/large_q500_k5/true_irt_parameters.json"),
    ("Q=1000", "data/large_q1000_k5/true_irt_parameters.json"),
    ("Q=2000", "data/large_q2000_k5/true_irt_parameters.json"),
    ("Q=5000 (archived)", "data_archive_20260304_205702/large_q5000/true_irt_parameters.json"),
]

print("=" * 80)
print("TRUE ALPHA DISTRIBUTION ANALYSIS")
print("=" * 80)
print()

for name, rel_path in datasets:
    json_path = base_dir / rel_path
    if not json_path.exists():
        print(f"{name}: FILE NOT FOUND at {rel_path}")
        print()
        continue

    try:
        with open(json_path, "r") as f:
            params = json.load(f)

        # Extract alpha parameters
        if "alpha" in params:
            alpha = np.array(params["alpha"])
        elif "item_discrimination" in params:
            alpha = np.array(params["item_discrimination"])
        else:
            print(f"{name}: No alpha/discrimination found")
            print(f"  Available keys: {list(params.keys())}")
            print()
            continue

        print(f"{name}:")
        print(f"  Shape: {alpha.shape}")
        print(f"  Mean: {alpha.mean():.4f}")
        print(f"  Std: {alpha.std():.4f}")
        print(f"  Min: {alpha.min():.4f}")
        print(f"  Max: {alpha.max():.4f}")
        print(f"  Median: {np.median(alpha):.4f}")
        print(f"  Q1: {np.percentile(alpha, 25):.4f}")
        print(f"  Q3: {np.percentile(alpha, 75):.4f}")
        print()

    except Exception as e:
        print(f"{name}: Error loading: {e}")
        print()

print("=" * 80)
