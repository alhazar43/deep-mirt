#!/usr/bin/env python3
"""Compute recovery by extracting parameters directly from model state.

For Static/Dynamic GPCM, alpha and beta are model parameters, not computed per interaction.
This script extracts them directly from the checkpoint.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch

from kt_gpcm.config import load_config


def link_alpha(vals: np.ndarray, target_std: float = 0.3) -> np.ndarray:
    """IRT linking for discrimination."""
    log_v = np.log(np.maximum(vals, 1e-6))
    std = log_v.std()
    if std < 1e-6:
        return np.ones_like(vals)
    return np.exp((log_v - log_v.mean()) / std * target_std)


def link_normal(vals: np.ndarray) -> np.ndarray:
    """IRT linking for beta."""
    std = vals.std()
    if std < 1e-6:
        return vals - vals.mean()
    return (vals - vals.mean()) / std


def extract_static_dynamic_params(checkpoint_path: str, n_categories: int):
    """Extract alpha/beta directly from Static/Dynamic GPCM checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model"]

    # Extract alpha: exp(0.3 * alpha_raw)
    alpha_raw = state["alpha_raw"]  # (Q+1, D)
    alpha = torch.exp(0.3 * alpha_raw)[1:].numpy()  # Skip padding, (Q, D)

    # Extract beta: monotonic gap construction
    beta_base = state["beta_base"][1:].numpy()  # (Q, 1)

    if n_categories == 2:
        beta = beta_base  # (Q, 1)
    else:
        beta_gaps = torch.nn.functional.softplus(state["beta_gaps"][1:])  # (Q, K-2)
        betas = [beta_base]
        for i in range(beta_gaps.shape[1]):
            betas.append(betas[-1] + beta_gaps[:, i:i+1].numpy())
        beta = np.concatenate(betas, axis=1)  # (Q, K-1)

    return alpha, beta


def compute_recovery_direct(config_path: str, checkpoint_path: str):
    """Compute recovery by extracting parameters directly."""
    cfg = load_config(config_path)

    # Load true IRT params
    irt_path = Path(cfg.data.data_dir) / cfg.data.dataset_name / "true_irt_parameters.json"
    if not irt_path.exists():
        return None

    with irt_path.open("r") as f:
        true_irt = json.load(f)

    true_alpha = np.array(true_irt["alpha"])  # (Q, D) or (Q,)
    true_beta = np.array(true_irt["beta"])    # (Q, K-1)

    # Handle 1D alpha case
    if true_alpha.ndim == 1:
        true_alpha = true_alpha[:, None]  # (Q, 1)

    model_type = getattr(cfg.model, "model_type", "deepgpcm")

    if model_type in ("static_gpcm", "dynamic_gpcm"):
        # Extract directly from checkpoint
        est_alpha, est_beta = extract_static_dynamic_params(
            checkpoint_path, cfg.model.n_categories
        )
    else:
        return None  # Use inference-based method for other models

    # Compute correlations with linking
    true_a_linked = link_alpha(true_alpha[:, 0])  # First dimension
    est_a_linked = link_alpha(est_alpha[:, 0])
    r_alpha = float(np.corrcoef(true_a_linked, est_a_linked)[0, 1])

    # Beta correlations per threshold
    K = cfg.model.n_categories
    r_beta_list = []
    for k in range(K - 1):
        true_b = link_normal(true_beta[:, k])
        est_b = link_normal(est_beta[:, k])
        r = float(np.corrcoef(true_b, est_b)[0, 1])
        r_beta_list.append(r)

    r_beta_mean = np.mean(r_beta_list)

    # Also return raw stats for debugging
    return {
        "r_alpha": r_alpha,
        "r_beta_mean": r_beta_mean,
        "r_beta_per_threshold": r_beta_list,
        "alpha_mean": float(est_alpha.mean()),
        "alpha_std": float(est_alpha.std()),
        "beta_mean": float(est_beta.mean()),
        "beta_std": float(est_beta.std()),
    }


def main():
    results = []

    # Test on Q=200 experiments
    for K in [2, 3, 4, 5, 6]:
        for model_type in ["static_gpcm", "dynamic_gpcm"]:
            exp_name = f"large_q200_k{K}_{model_type}"
            config = f"configs/baselines/{exp_name}.yaml"
            checkpoint = f"outputs/{exp_name}/best.pt"

            if not Path(config).exists() or not Path(checkpoint).exists():
                continue

            print(f"Processing {exp_name}...")
            try:
                recovery = compute_recovery_direct(config, checkpoint)
                if recovery is None:
                    continue

                results.append({
                    "experiment": exp_name,
                    "model": model_type,
                    "K": K,
                    "r_alpha": recovery["r_alpha"],
                    "r_beta_mean": recovery["r_beta_mean"],
                    "alpha_mean": recovery["alpha_mean"],
                    "alpha_std": recovery["alpha_std"],
                })
                print(f"  r_α={recovery['r_alpha']:.3f}, r_β={recovery['r_beta_mean']:.3f}")
                print(f"  α stats: mean={recovery['alpha_mean']:.3f}, std={recovery['alpha_std']:.3f}")
            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()

    # Save results
    df = pd.DataFrame(results)
    df.to_csv("outputs/recovery_direct_extraction.csv", index=False)
    print(f"\nSaved {len(results)} results to outputs/recovery_direct_extraction.csv")


if __name__ == "__main__":
    main()
