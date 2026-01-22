"""Diagnostics for parameter recovery and attention behavior."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from mirt_dkvmn.config.loader import load_config
from mirt_dkvmn.data.loaders import DataLoaderManager
from mirt_dkvmn.models.implementations.dkvmn_mirt import DKVMNMIRT
from mirt_dkvmn.utils.metrics import quadratic_weighted_kappa


def normalize_theta(thetas: np.ndarray) -> np.ndarray:
    mean = thetas.mean(axis=0, keepdims=True)
    std = thetas.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (thetas - mean) / std


def normalize_alpha(alphas: np.ndarray) -> np.ndarray:
    """Match Deep-GPCM mean-sigma normalization for lognormal(0, 0.3)."""
    eps = 1e-8
    log_alphas = np.log(np.clip(alphas, eps, None))
    log_norm = (log_alphas - np.mean(log_alphas)) / np.std(log_alphas)
    return np.exp(log_norm * 0.3)


def normalize_beta(betas: np.ndarray) -> np.ndarray:
    """Match Deep-GPCM mean-sigma normalization for thresholds."""
    beta_mean = np.mean(betas)
    beta_std = np.std(betas)
    return (betas - beta_mean) / max(beta_std, 0.1)


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0:
        return float("nan")
    a = a.reshape(-1)
    b = b.reshape(-1)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def procrustes_align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Align a to b via orthogonal Procrustes (no scaling)."""
    a_center = a - a.mean(axis=0, keepdims=True)
    b_center = b - b.mean(axis=0, keepdims=True)
    u, _, vt = np.linalg.svd(a_center.T @ b_center, full_matrices=False)
    r = u @ vt
    return a_center @ r


def collect_outputs(
    model, dataloader, device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    theta_per_student: Dict[int, List[np.ndarray]] = {}
    alpha_by_item: Dict[int, List[np.ndarray]] = {}
    beta_by_item: Dict[int, List[np.ndarray]] = {}
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            questions = batch["questions"].to(device)
            responses = batch["responses"].to(device)
            mask = batch["mask"].to(device)
            student_ids = batch["student_ids"].cpu().numpy()

            theta, beta, alpha, probs = model(questions, responses)
            preds = probs.argmax(dim=-1)

            for i in range(questions.size(0)):
                sid = int(student_ids[i])
                valid = mask[i].cpu().numpy()
                theta_seq = theta[i].cpu().numpy()[valid]
                if theta_seq.size > 0:
                    theta_per_student.setdefault(sid, []).append(theta_seq[-1])

                q_seq = questions[i].cpu().numpy()[valid]
                alpha_seq = alpha[i].cpu().numpy()[valid]
                beta_seq = beta[i].cpu().numpy()[valid]
                for q, a_vec, b_vec in zip(q_seq, alpha_seq, beta_seq):
                    alpha_by_item.setdefault(int(q), []).append(a_vec)
                    beta_by_item.setdefault(int(q), []).append(b_vec)

            all_preds.append(preds[mask].cpu())
            all_targets.append(responses[mask].cpu())

    theta_est = np.array([np.mean(v, axis=0) for _, v in sorted(theta_per_student.items())])
    alpha_est = np.zeros((max(alpha_by_item) + 1, alpha_by_item[next(iter(alpha_by_item))][0].shape[0]))
    beta_est = np.zeros((max(beta_by_item) + 1, beta_by_item[next(iter(beta_by_item))][0].shape[0]))
    for item, values in alpha_by_item.items():
        alpha_est[item] = np.mean(values, axis=0)
    for item, values in beta_by_item.items():
        beta_est[item] = np.mean(values, axis=0)

    preds_all = torch.cat(all_preds, dim=0)
    targets_all = torch.cat(all_targets, dim=0)
    qwk = quadratic_weighted_kappa(preds_all, targets_all, probs.size(-1))

    return theta_est, alpha_est, beta_est, preds_all.numpy(), targets_all.numpy(), qwk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(config.base.device if torch.cuda.is_available() else "cpu")

    loader = DataLoaderManager(config.data.dataset_name, data_root=config.data.data_root)
    dataloaders = loader.build_dataloaders(batch_size=config.training.batch_size, split_ratio=0.8, val_ratio=0.1)

    model = DKVMNMIRT(
        n_questions=config.model.n_questions,
        n_cats=config.model.n_cats,
        n_traits=config.model.n_traits,
        memory_size=config.model.memory_size,
        key_dim=config.model.key_dim,
        value_dim=config.model.value_dim,
        summary_dim=config.model.summary_dim,
        concept_aligned_memory=config.model.concept_aligned_memory,
        theta_projection=config.model.theta_projection,
        memory_add_activation=config.model.memory_add_activation,
        theta_source=config.model.theta_source,
        gpcm_mode=config.model.gpcm_mode,
    ).to(device)

    payload = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(payload["model_state"])

    theta_est, alpha_est, beta_est, preds, targets, qwk = collect_outputs(model, dataloaders["test"], device)
    print(f"QWK (test): {qwk:.4f}")

    params_path = Path(config.data.data_root) / config.data.dataset_name / "true_irt_parameters.json"
    if params_path.exists():
        with params_path.open("r", encoding="utf-8") as handle:
            true_params = json.load(handle)
        theta_true = np.array(true_params["theta"])
        alpha_true = np.array(true_params["alpha"])
        beta_true = np.array(true_params["beta"])

        theta_true_trim = theta_true[: theta_est.shape[0]]
        theta_true_norm = normalize_theta(theta_true_trim)
        theta_est_norm = normalize_theta(theta_est)
        theta_aligned = procrustes_align(theta_est_norm, theta_true_norm)
        theta_corr = pearson_corr(theta_aligned, theta_true_norm)

        alpha_true_trim = alpha_true[: alpha_est.shape[0]]
        alpha_corr = pearson_corr(
            normalize_alpha(alpha_est).reshape(-1),
            normalize_alpha(alpha_true_trim).reshape(-1),
        )

        beta_true_trim = beta_true[: beta_est.shape[0]]
        beta_est_use = beta_est
        if beta_est.shape[1] == beta_true_trim.shape[1] + 1:
            beta_est_use = beta_est[:, 1:]
        beta_corr = pearson_corr(
            normalize_beta(beta_est_use).reshape(-1),
            normalize_beta(beta_true_trim).reshape(-1),
        )
        print(f"Theta corr (procrustes): {theta_corr:.4f}")
        print(f"Alpha corr (mean_sigma): {alpha_corr:.4f}")
        print(f"Beta corr (mean_sigma): {beta_corr:.4f}")
    else:
        print("No true_irt_parameters.json found; skipping parameter recovery.")


if __name__ == "__main__":
    main()
