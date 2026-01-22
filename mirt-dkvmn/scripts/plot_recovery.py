"""Plot parameter recovery and theta distributions."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from mirt_dkvmn.config.loader import load_config
from mirt_dkvmn.data.loaders import DataLoaderManager
from mirt_dkvmn.models.implementations.dkvmn_mirt import DKVMNMIRT


def normalize_theta(thetas: np.ndarray) -> np.ndarray:
    mean = thetas.mean(axis=0, keepdims=True)
    std = thetas.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (thetas - mean) / std


def normalize_alpha(alphas: np.ndarray) -> np.ndarray:
    eps = 1e-8
    log_alphas = np.log(np.clip(alphas, eps, None))
    log_norm = (log_alphas - np.mean(log_alphas)) / np.std(log_alphas)
    return np.exp(log_norm * 0.3)


def normalize_beta(betas: np.ndarray) -> np.ndarray:
    beta_mean = np.mean(betas)
    beta_std = np.std(betas)
    return (betas - beta_mean) / max(beta_std, 0.1)


def procrustes_align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_center = a - a.mean(axis=0, keepdims=True)
    b_center = b - b.mean(axis=0, keepdims=True)
    u, _, vt = np.linalg.svd(a_center.T @ b_center, full_matrices=False)
    r = u @ vt
    return a_center @ r


def collect_item_params(model, dataloader, device):
    model.eval()
    theta_per_student = {}
    alpha_by_item = {}
    beta_by_item = {}

    with torch.no_grad():
        for batch in dataloader:
            questions = batch["questions"].to(device)
            responses = batch["responses"].to(device)
            mask = batch["mask"].to(device)
            student_ids = batch["student_ids"].cpu().numpy()

            theta, beta, alpha, _ = model(questions, responses)

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

    theta_est = np.array([np.mean(v, axis=0) for _, v in sorted(theta_per_student.items())])
    alpha_est = np.zeros((max(alpha_by_item) + 1, alpha_by_item[next(iter(alpha_by_item))][0].shape[0]))
    beta_est = np.zeros((max(beta_by_item) + 1, beta_by_item[next(iter(beta_by_item))][0].shape[0]))
    for item, values in alpha_by_item.items():
        alpha_est[item] = np.mean(values, axis=0)
    for item, values in beta_by_item.items():
        beta_est[item] = np.mean(values, axis=0)

    return theta_est, alpha_est, beta_est


def plot_theta_distributions(theta_est, theta_true, out_dir):
    theta_true_norm = normalize_theta(theta_true[: theta_est.shape[0]])
    aligned = procrustes_align(normalize_theta(theta_est), theta_true_norm)
    n_traits = aligned.shape[1]
    fig, axes = plt.subplots(1, n_traits, figsize=(4 * n_traits, 3), squeeze=False)
    for idx in range(n_traits):
        ax = axes[0, idx]
        ax.hist(aligned[:, idx], bins=30, alpha=0.6, label="learned")
        ax.hist(theta_true_norm[:, idx], bins=30, alpha=0.6, label="true")
        ax.set_title(f"Theta dim {idx}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "theta_distributions.png", dpi=150)
    plt.close(fig)


def plot_item_scatter(true_vals, learned_vals, out_path, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(true_vals, learned_vals, alpha=0.5, s=10)
    min_v = min(true_vals.min(), learned_vals.min())
    max_v = max(true_vals.max(), learned_vals.max())
    ax.plot([min_v, max_v], [min_v, max_v], "k--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_alpha_by_dim(alpha_true, alpha_est, out_dir):
    alpha_true_norm = normalize_alpha(alpha_true[: alpha_est.shape[0]])
    alpha_est_norm = normalize_alpha(alpha_est)
    n_traits = alpha_est_norm.shape[1]
    for idx in range(n_traits):
        true_vals = alpha_true_norm[:, idx]
        learned_vals = alpha_est_norm[:, idx]
        corr = np.corrcoef(true_vals, learned_vals)[0, 1] if np.std(true_vals) > 0 else np.nan
        plot_item_scatter(
            true_vals,
            learned_vals,
            out_dir / f"alpha_dim_{idx}_recovery.png",
            f"Alpha recovery dim {idx} (r={corr:.3f})",
            "True alpha (norm)",
            "Learned alpha (norm)",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/large.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="mirt-dkvmn/artifacts/large/recovery_plots")
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
    ).to(device)

    payload = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(payload["model_state"])

    theta_est, alpha_est, beta_est = collect_item_params(model, dataloaders["test"], device)

    params_path = Path(config.data.data_root) / config.data.dataset_name / "true_irt_parameters.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Missing true params: {params_path}")

    with params_path.open("r", encoding="utf-8") as handle:
        true_params = json.load(handle)

    theta_true = np.array(true_params["theta"])
    alpha_true = np.array(true_params["alpha"])
    beta_true = np.array(true_params["beta"])

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_theta_distributions(theta_est, theta_true, out_dir)
    plot_alpha_by_dim(alpha_true, alpha_est, out_dir)

    max_beta = min(beta_true.shape[1], beta_est.shape[1])
    for idx in range(max_beta):
        plot_item_scatter(
            normalize_beta(beta_true[: beta_est.shape[0], idx]),
            normalize_beta(beta_est[:, idx]),
            out_dir / f"beta_{idx}_recovery.png",
            f"Beta recovery (threshold {idx})",
            "True beta (mean_sigma)",
            "Learned beta (mean_sigma)",
        )


if __name__ == "__main__":
    main()
