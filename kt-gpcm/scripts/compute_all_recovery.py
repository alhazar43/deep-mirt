#!/usr/bin/env python3
"""Compute parameter recovery correlations for all experiments.

This script:
1. Scans all output directories for checkpoints
2. Loads each model and computes alpha/beta recovery correlations
3. Saves results to a CSV file for table generation
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from kt_gpcm.config import load_config
from kt_gpcm.data.loaders import DataModule
from kt_gpcm.models.kt_gpcm import DeepGPCM
from kt_gpcm.models.dkvmn_softmax import DKVMNSoftmax
from kt_gpcm.models.static_gpcm import StaticGPCM
from kt_gpcm.models.dynamic_gpcm import DynamicGPCM


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


def link_theta_irt(true_theta: np.ndarray, est_theta: np.ndarray,
                    true_beta: np.ndarray, est_beta: np.ndarray) -> tuple:
    """IRT linking for theta using mean/sigma method with item parameters.

    Standard IRT linking: θ_new = A·θ_old + B
    where A and B are estimated from item parameters to preserve IRT scale.

    Args:
        true_theta: True student abilities
        est_theta: Estimated student abilities
        true_beta: True item difficulties (Q, K-1)
        est_beta: Estimated item difficulties (Q, K-1)

    Returns:
        (linked_true, linked_est, A, B): Linked parameters and constants
    """
    # Flatten beta to get all difficulty parameters
    true_beta_flat = true_beta.flatten()
    est_beta_flat = est_beta.flatten()

    # Remove NaN/inf
    valid = np.isfinite(true_beta_flat) & np.isfinite(est_beta_flat)
    true_beta_flat = true_beta_flat[valid]
    est_beta_flat = est_beta_flat[valid]

    if len(true_beta_flat) < 2:
        # Fallback to z-score if not enough item parameters
        return link_normal(true_theta), link_normal(est_theta), 1.0, 0.0

    # Mean/sigma method: A = σ(β_true) / σ(β_est)
    A = true_beta_flat.std() / max(est_beta_flat.std(), 1e-6)

    # B = μ(β_true) - A·μ(β_est)
    B = true_beta_flat.mean() - A * est_beta_flat.mean()

    # Transform: θ_true_scale = A·θ_est + B
    est_theta_linked = A * est_theta + B

    return true_theta, est_theta_linked, A, B


def compute_recovery(config_path: str, checkpoint_path: str, device: torch.device):
    """Compute recovery correlations for a single experiment."""
    cfg = load_config(config_path)

    # Load data
    data_mgr = DataModule(cfg)
    train_loader, test_loader = data_mgr.build()

    # Load model
    model_kwargs = {k: v for k, v in vars(cfg.model).items() if k != "model_type"}
    model_type = getattr(cfg.model, "model_type", "deepgpcm")

    if model_type == "dkvmn_softmax":
        model = DKVMNSoftmax(**model_kwargs).to(device)
    elif model_type == "static_gpcm":
        model = StaticGPCM(n_students=data_mgr.n_students, **model_kwargs).to(device)
        model._model_type = "static_gpcm"
    elif model_type == "dynamic_gpcm":
        model = DynamicGPCM(n_students=data_mgr.n_students, **model_kwargs).to(device)
        model._model_type = "dynamic_gpcm"
    else:
        model = DeepGPCM(**model_kwargs).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    # Load true IRT params
    irt_path = Path(cfg.data.data_dir) / cfg.data.dataset_name / "true_irt_parameters.json"
    if not irt_path.exists():
        return None

    with irt_path.open("r") as f:
        true_irt = json.load(f)

    true_alpha = np.array(true_irt["alpha"])
    true_beta = np.array(true_irt["beta"])

    # Accumulate estimates
    Q = cfg.model.n_questions
    D = cfg.model.n_traits
    K = cfg.model.n_categories

    alpha_sum = np.zeros((Q, D))
    alpha_count = np.zeros((Q,))
    beta_sum = np.zeros((Q, K - 1))
    beta_count = np.zeros((Q,))
    theta_dict = {}  # Map student_id -> last theta

    def process_loader(loader):
        for batch in loader:
            questions = batch[0].to(device)
            responses = batch[1].to(device)
            mask = batch[2].to(device)
            student_ids = batch[3].to(device) if len(batch) > 3 else None

            with torch.no_grad():
                if model_type in ("static_gpcm", "dynamic_gpcm") and student_ids is not None:
                    out = model(student_ids, questions, responses)
                else:
                    out = model(questions, responses)

            alpha = out["alpha"].cpu().numpy()
            beta = out["beta"].cpu().numpy()
            theta = out["theta"].cpu().numpy()
            q_np = questions.cpu().numpy()
            m_np = mask.cpu().numpy()
            sid_np = student_ids.cpu().numpy() if student_ids is not None else None

            B, S = q_np.shape
            for b in range(B):
                theta_vals = []
                student_id = None
                for t in range(S):
                    if m_np[b, t]:
                        qid = q_np[b, t] - 1
                        if 0 <= qid < Q:
                            alpha_sum[qid] += alpha[b, t]
                            alpha_count[qid] += 1
                            beta_sum[qid] += beta[b, t]
                            beta_count[qid] += 1
                        theta_vals.append(float(theta[b, t, 0]))
                        if student_id is None and sid_np is not None:
                            student_id = int(sid_np[b, t])
                # Store all theta values for this student
                if len(theta_vals) > 0 and student_id is not None:
                    if student_id not in theta_dict:
                        theta_dict[student_id] = []
                    theta_dict[student_id].extend(theta_vals)

    process_loader(train_loader)
    process_loader(test_loader)

    # Compute means
    seen = alpha_count > 0
    alpha_est = np.where(seen[:, None], alpha_sum / np.maximum(alpha_count[:, None], 1), 0.0)
    beta_est = np.where(seen[:, None], beta_sum / np.maximum(beta_count[:, None], 1), 0.0)

    # Compute correlations with linking
    true_a_linked = link_alpha(true_alpha[seen])
    est_a_linked = link_alpha(alpha_est[seen, 0])  # Use first dimension
    r_alpha = float(np.corrcoef(true_a_linked, est_a_linked)[0, 1]) if len(true_a_linked) > 1 else np.nan

    # Beta correlations per threshold
    r_beta_list = []
    for k in range(K - 1):
        true_b = link_normal(true_beta[seen, k])
        est_b = link_normal(beta_est[seen, k])
        r = float(np.corrcoef(true_b, est_b)[0, 1]) if len(true_b) > 1 else np.nan
        r_beta_list.append(r)

    r_beta_mean = np.mean(r_beta_list)

    # Theta correlation with IRT linking
    true_theta = np.array(true_irt["theta"])

    # For Static/Dynamic GPCM: extract theta directly from embedding table
    # For DEEP-GPCM: use last timestep from forward pass
    if model_type in ("static_gpcm", "dynamic_gpcm"):
        # Extract theta embeddings directly from checkpoint (more reliable)
        theta_embed_weight = state["model"]["theta_embed.weight"].cpu().numpy()  # Shape: (N+1, D)
        theta_est_aligned = []
        true_theta_aligned = []
        # Student ID 1 -> true_theta[0], Student ID 2 -> true_theta[1], etc.
        for student_id in range(1, min(len(true_theta) + 1, theta_embed_weight.shape[0])):
            theta_est_aligned.append(float(theta_embed_weight[student_id, 0]))
            true_theta_aligned.append(true_theta[student_id - 1])
    else:
        # For DEEP-GPCM: use mean theta per student (aligned by student ID)
        # Student IDs are 1-indexed, true_theta is 0-indexed
        theta_est_aligned = []
        true_theta_aligned = []
        for student_id in sorted(theta_dict.keys()):
            if student_id >= 1 and student_id <= len(true_theta):
                # Use mean theta across sequence (more stable than last timestep)
                theta_est_aligned.append(np.mean(theta_dict[student_id]))
                true_theta_aligned.append(true_theta[student_id - 1])

    if len(theta_est_aligned) > 1:
        # Use IRT linking with item parameters for proper scale alignment
        true_theta_arr = np.array(true_theta_aligned)
        est_theta_arr = np.array(theta_est_aligned)
        true_theta_linked, est_theta_linked, A, B = link_theta_irt(
            true_theta_arr, est_theta_arr, true_beta[seen], beta_est[seen]
        )
        r_theta = float(np.corrcoef(true_theta_linked, est_theta_linked)[0, 1])
    else:
        r_theta = np.nan

    return {
        "r_alpha": r_alpha,
        "r_beta_mean": r_beta_mean,
        "r_beta_per_threshold": r_beta_list,
        "r_theta": r_theta
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--configs_dir", default="configs")
    parser.add_argument("--output_csv", default="outputs/recovery_correlations.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputs_dir = Path(args.outputs_dir)
    configs_dir = Path(args.configs_dir)

    results = []

    # Scan all output directories
    for exp_dir in sorted(outputs_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        exp_name = exp_dir.name
        checkpoint = exp_dir / "best.pt"

        # Try multiple config locations
        config = configs_dir / f"{exp_name}.yaml"
        if not config.exists():
            config = configs_dir / "generated" / f"{exp_name}.yaml"
        if not config.exists():
            config = configs_dir / "baselines" / f"{exp_name}.yaml"

        if not checkpoint.exists() or not config.exists():
            print(f"Skipping {exp_name}: missing checkpoint or config")
            continue

        print(f"Processing {exp_name}...")
        try:
            recovery = compute_recovery(str(config), str(checkpoint), device)
            if recovery is None:
                print(f"  No true IRT parameters found")
                continue

            results.append({
                "experiment": exp_name,
                "r_alpha": recovery["r_alpha"],
                "r_beta_mean": recovery["r_beta_mean"],
                "r_theta": recovery["r_theta"],
                "r_beta_thresholds": ",".join(f"{r:.4f}" for r in recovery["r_beta_per_threshold"])
            })
            print(f"  r_α={recovery['r_alpha']:.3f}, r_β={recovery['r_beta_mean']:.3f}, r_θ={recovery['r_theta']:.3f}")
        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved recovery correlations to {args.output_csv}")


if __name__ == "__main__":
    main()
