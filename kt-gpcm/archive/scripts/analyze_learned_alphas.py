#!/usr/bin/env python3
"""
Analyze learned alpha parameters from checkpoints.
"""

import torch
import numpy as np
from pathlib import Path
import json

base_dir = Path("C:/Users/steph/documents/deep-mirt/kt-gpcm")

# Check which checkpoints exist
experiments = [
    ("Q=1000 Separable", "outputs/q1000_k5_separable/best.pt"),
    ("Q=1000 SIE", "outputs/q1000_k5_static_item/best.pt"),
    ("Q=2000 Separable", "outputs/q2000_k5_separable/best.pt"),
    ("Q=2000 SIE", "outputs/q2000_k5_static_item/best.pt"),
]

# Also check archived Q=5000
archived_experiments = [
    ("Q=5000 SIE (archived)", "data_archive_20260304_205702/large_q5000/best.pt"),
    ("Q=5000 Linear (archived)", "outputs/large_q5000_linear/best.pt"),
    ("Q=5000 Separable (archived)", "outputs/large_q5000_separable/best.pt"),
    ("Q=5000 SIE (archived v2)", "outputs/large_q5000_static/best.pt"),
]

print("=" * 80)
print("LEARNED ALPHA PARAMETER ANALYSIS")
print("=" * 80)
print()

for name, rel_path in experiments + archived_experiments:
    ckpt_path = base_dir / rel_path
    if not ckpt_path.exists():
        print(f"{name}: CHECKPOINT NOT FOUND at {rel_path}")
        print()
        continue

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # Find alpha parameters in state dict
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        # Look for alpha-related keys
        alpha_keys = [k for k in state_dict.keys() if "alpha" in k.lower()]

        if not alpha_keys:
            print(f"{name}: No alpha parameters found")
            print(f"  Available keys (first 15): {list(state_dict.keys())[:15]}")
            print()
            continue

        print(f"{name}:")
        for key in alpha_keys:
            param = state_dict[key]
            if isinstance(param, torch.Tensor):
                param_np = param.detach().cpu().numpy()
                print(f"  {key}:")
                print(f"    Shape: {param_np.shape}")
                print(f"    Mean: {param_np.mean():.4f}")
                print(f"    Std: {param_np.std():.4f}")
                print(f"    Min: {param_np.min():.4f}")
                print(f"    Max: {param_np.max():.4f}")

                # Check for collapse (low std)
                if param_np.std() < 0.1:
                    print(f"    ⚠️  WARNING: Low std indicates parameter collapse!")
        print()

    except Exception as e:
        print(f"{name}: Error loading checkpoint: {e}")
        print()

print("=" * 80)
