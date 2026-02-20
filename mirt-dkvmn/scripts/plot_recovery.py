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
from mirt_dkvmn.utils.metrics import confusion_matrix

try:
    from scipy.stats import gaussian_kde
except Exception:  # pragma: no cover - optional dependency
    gaussian_kde = None


def normalize_theta(thetas: np.ndarray) -> np.ndarray:
    mean = thetas.mean(axis=0, keepdims=True)
    std = thetas.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (thetas - mean) / std


def normalize_alpha_with_reference(alphas: np.ndarray, ref: np.ndarray) -> np.ndarray:
    eps = 1e-8
    log_alphas = np.log(np.clip(alphas, eps, None))
    log_ref = np.log(np.clip(ref, eps, None))
    mean = np.mean(log_ref)
    std = np.std(log_ref)
    std = 1.0 if std == 0 else std
    return (log_alphas - mean) / std


def normalize_beta_with_reference(betas: np.ndarray, ref: np.ndarray) -> np.ndarray:
    beta_mean = np.mean(ref)
    beta_std = np.std(ref)
    beta_std = max(beta_std, 0.1)
    return (betas - beta_mean) / beta_std


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
                    theta_per_student.setdefault(sid, []).append(theta_seq)

                q_seq = questions[i].cpu().numpy()[valid]
                alpha_seq = alpha[i].cpu().numpy()[valid]
                beta_seq = beta[i].cpu().numpy()[valid]
                for q, a_vec, b_vec in zip(q_seq, alpha_seq, beta_seq):
                    alpha_by_item.setdefault(int(q), []).append(a_vec)
                    beta_by_item.setdefault(int(q), []).append(b_vec)

    theta_est = np.array([np.mean([seq[-1] for seq in v], axis=0) for _, v in sorted(theta_per_student.items())])
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
    cols = min(3, n_traits)
    rows = int(np.ceil(n_traits / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    x_range = np.linspace(-3, 3, 200)
    for idx in range(n_traits):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        learned = aligned[:, idx]
        true = theta_true_norm[:, idx]
        if gaussian_kde is not None:
            learned_kde = gaussian_kde(learned)
            true_kde = gaussian_kde(true)
            ax.plot(x_range, learned_kde(x_range), color="tab:blue", linewidth=2, label="Learned")
            ax.plot(x_range, true_kde(x_range), "k--", linewidth=2, label="True")
        else:
            ax.hist(learned, bins=30, density=True, alpha=0.6, label="Learned")
            ax.hist(true, bins=30, density=True, alpha=0.6, label="True")
        ax.set_title(f"Theta dim {idx}", fontweight="bold")
        ax.grid(True, alpha=0.3)
    for idx in range(n_traits, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "theta_distributions.png", dpi=150)
    plt.close(fig)


def plot_item_scatter(ax, true_vals, learned_vals, title, xlabel, ylabel):
    ax.scatter(true_vals, learned_vals, alpha=0.5, s=10)
    min_v = min(true_vals.min(), learned_vals.min())
    max_v = max(true_vals.max(), learned_vals.max())
    ax.plot([min_v, max_v], [min_v, max_v], "k--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_alpha_recovery(alpha_true, alpha_est, out_dir):
    alpha_true_trim = alpha_true[: alpha_est.shape[0]]
    alpha_true_norm = normalize_alpha_with_reference(alpha_true_trim, alpha_true_trim)
    alpha_est_norm = normalize_alpha_with_reference(alpha_est, alpha_true_trim)
    alpha_aligned = procrustes_align(alpha_est_norm, alpha_true_norm)
    n_traits = alpha_est_norm.shape[1]
    cols = min(3, n_traits)
    rows = int(np.ceil(n_traits / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for idx in range(n_traits):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        true_vals = alpha_true_norm[:, idx]
        learned_vals = alpha_aligned[:, idx]
        corr = np.corrcoef(true_vals, learned_vals)[0, 1] if np.std(true_vals) > 0 else np.nan
        plot_item_scatter(
            ax,
            true_vals,
            learned_vals,
            f"Alpha dim {idx} (r={corr:.3f})",
            "True alpha (z)",
            "Learned alpha (z)",
        )
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)
    for idx in range(n_traits, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "alpha_recovery.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    x_range = np.linspace(-3, 3, 200)
    for idx in range(n_traits):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        true_vals = alpha_true_norm[:, idx]
        learned_vals = alpha_aligned[:, idx]
        if gaussian_kde is not None:
            true_kde = gaussian_kde(true_vals)
            learned_kde = gaussian_kde(learned_vals)
            ax.plot(x_range, learned_kde(x_range), color="tab:blue", linewidth=2, label="Learned")
            ax.plot(x_range, true_kde(x_range), "k--", linewidth=2, label="True")
        else:
            ax.hist(learned_vals, bins=30, density=True, alpha=0.6, label="Learned")
            ax.hist(true_vals, bins=30, density=True, alpha=0.6, label="True")
        ax.set_title(f"Alpha dim {idx} dist", fontweight="bold")
        ax.grid(True, alpha=0.3)
    for idx in range(n_traits, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "alpha_distributions.png", dpi=150)
    plt.close(fig)

def plot_beta_by_threshold(beta_true, beta_est, out_dir):
    max_beta = min(beta_true.shape[1], beta_est.shape[1])
    cols = min(3, max_beta)
    rows = int(np.ceil(max_beta / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for idx in range(max_beta):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        true_ref = beta_true[: beta_est.shape[0], idx]
        true_norm = normalize_beta_with_reference(true_ref, true_ref)
        learned_norm = normalize_beta_with_reference(beta_est[:, idx], true_ref)
        corr = np.corrcoef(true_norm, learned_norm)[0, 1] if np.std(true_norm) > 0 else np.nan
        plot_item_scatter(
            ax,
            true_norm,
            learned_norm,
            f"Threshold {idx} (r={corr:.3f})",
            f"True beta_{idx} (z)",
            f"Learned beta_{idx} (z)",
        )
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
    for idx in range(max_beta, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "beta_recovery.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for idx in range(max_beta):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        true_ref = beta_true[: beta_est.shape[0], idx]
        true_norm = normalize_beta_with_reference(true_ref, true_ref)
        learned_norm = normalize_beta_with_reference(beta_est[:, idx], true_ref)
        x_range = np.linspace(-3, 3, 200)
        if gaussian_kde is not None:
            true_kde = gaussian_kde(true_norm)
            learned_kde = gaussian_kde(learned_norm)
            ax.plot(x_range, learned_kde(x_range), color="tab:blue", linewidth=2, label="Learned")
            ax.plot(x_range, true_kde(x_range), "k--", linewidth=2, label="True")
        else:
            ax.hist(learned_norm, bins=30, density=True, alpha=0.6, label="Learned")
            ax.hist(true_norm, bins=30, density=True, alpha=0.6, label="True")
        ax.set_title(f"Beta {idx} distribution", fontweight="bold")
        ax.grid(True, alpha=0.3)
    for idx in range(max_beta, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "beta_distributions.png", dpi=150)
    plt.close(fig)


def plot_recovery(config_path: str, checkpoint_path: str, output_dir: str) -> None:
    config = load_config(config_path)
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
    ).to(device)

    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state"])

    theta_est, alpha_est, beta_est = collect_item_params(model, dataloaders["test"], device)
    conf = np.zeros((config.model.n_cats, config.model.n_cats), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for batch in dataloaders["test"]:
            questions = batch["questions"].to(device)
            responses = batch["responses"].to(device)
            mask = batch["mask"].to(device)
            probs = model(questions, responses)[-1]
            conf += confusion_matrix(probs.argmax(dim=-1), responses, probs.size(-1), mask)

    params_path = Path(config.data.data_root) / config.data.dataset_name / "true_irt_parameters.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Missing true params: {params_path}")

    with params_path.open("r", encoding="utf-8") as handle:
        true_params = json.load(handle)

    theta_true = np.array(true_params["theta"])
    alpha_true = np.array(true_params["alpha"])
    beta_true = np.array(true_params["beta"])

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_theta_distributions(theta_est, theta_true, out_dir)
    plot_alpha_recovery(alpha_true, alpha_est, out_dir)
    plot_beta_by_threshold(beta_true, beta_est, out_dir)
    np.savetxt(out_dir / "confusion_matrix_test.csv", conf, fmt="%d", delimiter=",")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(conf, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test)")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix_test.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/large.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="mirt-dkvmn/artifacts/large/recovery_plots")
    args = parser.parse_args()
    plot_recovery(args.config, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
