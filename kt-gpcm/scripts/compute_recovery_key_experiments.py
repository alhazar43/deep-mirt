#!/usr/bin/env python3
"""Compute recovery for key experiments only (faster than full scan)."""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch

from kt_gpcm.config import load_config
from kt_gpcm.data.loaders import DataModule
from kt_gpcm.models.kt_gpcm import DeepGPCM
from kt_gpcm.models.dkvmn_softmax import DKVMNSoftmax
from kt_gpcm.models.static_gpcm import StaticGPCM
from kt_gpcm.models.dynamic_gpcm import DynamicGPCM
import json


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
            q_np = questions.cpu().numpy()
            m_np = mask.cpu().numpy()

            B, S = q_np.shape
            for b in range(B):
                for t in range(S):
                    if m_np[b, t]:
                        qid = q_np[b, t] - 1
                        if 0 <= qid < Q:
                            alpha_sum[qid] += alpha[b, t]
                            alpha_count[qid] += 1
                            beta_sum[qid] += beta[b, t]
                            beta_count[qid] += 1

    process_loader(train_loader)
    process_loader(test_loader)

    # Compute means
    seen = alpha_count > 0
    alpha_est = np.where(seen[:, None], alpha_sum / np.maximum(alpha_count[:, None], 1), 0.0)
    beta_est = np.where(seen[:, None], beta_sum / np.maximum(beta_count[:, None], 1), 0.0)

    # Compute correlations with linking (CORRECT METHOD - use first dimension)
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

    return {
        "r_alpha": r_alpha,
        "r_beta_mean": r_beta_mean,
        "r_beta_per_threshold": r_beta_list
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Key experiments to process
    experiments = []

    # Add all Q and K combinations for baselines
    for Q in [200, 500, 1000, 2000]:
        for K in [2, 3, 4, 5, 6]:
            for model_type in ["deepgpcm", "static_gpcm", "dynamic_gpcm", "dkvmn_softmax", "dkvmn_ordinal"]:
                exp_name = f"large_q{Q}_k{K}_{model_type}"
                config = Path(f"configs/baselines/{exp_name}.yaml")
                checkpoint = Path(f"outputs/{exp_name}/best.pt")

                if config.exists() and checkpoint.exists():
                    experiments.append((exp_name, str(config), str(checkpoint)))

    results = []
    total = len(experiments)

    for idx, (exp_name, config, checkpoint) in enumerate(experiments, 1):
        print(f"[{idx}/{total}] Processing {exp_name}...")
        try:
            recovery = compute_recovery(config, checkpoint, device)
            if recovery is None:
                print(f"  No true IRT parameters found")
                continue

            results.append({
                "experiment": exp_name,
                "r_alpha": recovery["r_alpha"],
                "r_beta_mean": recovery["r_beta_mean"],
                "r_beta_thresholds": ",".join(f"{r:.4f}" for r in recovery["r_beta_per_threshold"])
            })
            print(f"  r_α={recovery['r_alpha']:.3f}, r_β={recovery['r_beta_mean']:.3f}")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
    df = pd.DataFrame(results)
    output_path = "outputs/recovery_baselines.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(results)} recovery results to {output_path}")


if __name__ == "__main__":
    main()
