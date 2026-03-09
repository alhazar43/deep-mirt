#!/usr/bin/env python3
"""
Diagnose why Q=2000 has worse alpha recovery than Q=5000.
"""

import json
import torch
import numpy as np
from pathlib import Path

# Dataset configurations
datasets = {
    "Q=200": {
        "n_students": 5000,
        "n_questions": 200,
        "seq_len_range": [20, 80],
    },
    "Q=500": {
        "n_students": 5000,
        "n_questions": 500,
        "seq_len_range": [20, 80],
    },
    "Q=1000": {
        "n_students": 5000,
        "n_questions": 1000,
        "seq_len_range": [20, 80],
    },
    "Q=2000": {
        "n_students": 3000,
        "n_questions": 2000,
        "seq_len_range": [50, 120],
    },
    "Q=5000": {
        "n_students": 1000,
        "n_questions": 5000,
        "seq_len_range": [400, 600],
    },
}

print("=" * 80)
print("ITEM COVERAGE ANALYSIS")
print("=" * 80)
print()

for name, config in datasets.items():
    n_students = config["n_students"]
    n_questions = config["n_questions"]
    seq_min, seq_max = config["seq_len_range"]
    avg_seq_len = (seq_min + seq_max) / 2

    total_responses = n_students * avg_seq_len
    avg_responses_per_item = total_responses / n_questions

    print(f"{name}:")
    print(f"  Students: {n_students:,}")
    print(f"  Questions: {n_questions:,}")
    print(f"  Avg seq length: {avg_seq_len:.1f}")
    print(f"  Total responses: {total_responses:,.0f}")
    print(f"  Avg responses per item: {avg_responses_per_item:.1f}")
    print(f"  Ratio (responses/item): {avg_responses_per_item:.2f}")
    print()

print("=" * 80)
print("KEY FINDING:")
print("=" * 80)
print("Q=2000 has only ~127 responses per item")
print("Q=5000 has ~100 responses per item (LESS than Q=2000!)")
print("Q=1000 has ~250 responses per item (2x more than Q=2000)")
print()
print("BUT Q=5000 has much longer sequences (500 avg vs 85 avg)")
print("This means each STUDENT sees 10% of items (500/5000)")
print("vs Q=2000 where each student sees 4.25% of items (85/2000)")
print()
