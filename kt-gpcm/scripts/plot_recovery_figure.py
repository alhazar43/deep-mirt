"""Combined IRT parameter recovery figure for the DEEP-GPCM paper.

Produces a single 3-panel figure (theta KDE | alpha scatter | beta scatter),
all parameters z-score normalised so the y=x diagonal is meaningful.

Usage (from repo root):
    PYTHONPATH=kt-gpcm/src python kt-gpcm/scripts/plot_recovery_figure.py \\
        --config      kt-gpcm/configs/large_q5000_static.yaml \\
        --checkpoint  kt-gpcm/outputs/large_q5000_static/best.pt \\
        --output      kt-gpcm/outputs/large_q5000_static/recovery_figure.png
"""

from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from kt_gpcm.config import load_config
from kt_gpcm.data.loaders import DataModule
from kt_gpcm.models.kt_gpcm import DeepGPCM


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def z_score(x: np.ndarray) -> np.ndarray:
    """Z-score to N(0,1)."""
    s = x.std()
    return (x - x.mean()) / max(s, 1e-8)


def link_alpha(x: np.ndarray) -> np.ndarray:
    """IRT linking for discrimination: z-score in log-space."""
    log_x = np.log(np.maximum(x, 1e-6))
    return z_score(log_x)


# ---------------------------------------------------------------------------
# Build model and run inference
# ---------------------------------------------------------------------------

def run_inference(cfg, checkpoint: str, device: torch.device):
    """Return per-item alpha/beta estimates and per-student mean-theta estimates."""
    model = DeepGPCM(**vars(cfg.model)).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"Loaded checkpoint (epoch {state.get('epoch', '?')}): {checkpoint}")

    # Resolve data_dir relative to the config file's parent (not CWD)
    data_root = Path(cfg.data.data_dir)
    if not data_root.is_absolute():
        data_root = Path(checkpoint).parent.parent.parent / cfg.data.data_dir
        if not (data_root / cfg.data.dataset_name / "sequences.json").exists():
            # fallback: relative to CWD
            data_root = Path(cfg.data.data_dir)

    data_mgr = DataModule(cfg, base_dir=str(data_root))
    train_loader, test_loader = data_mgr.build()

    Q = cfg.model.n_questions
    D = cfg.model.n_traits
    K = cfg.model.n_categories

    # Item-level accumulators
    alpha_sum   = np.zeros((Q, D))
    alpha_count = np.zeros(Q)
    beta_sum    = np.zeros((Q, K - 1))
    beta_count  = np.zeros(Q)

    # Student-level accumulators (mean theta over time)
    student_theta_sum   = []
    student_theta_count = []

    def process(loader):
        for questions, responses, mask in loader:
            questions = questions.to(device)
            responses = responses.to(device)
            mask      = mask.to(device)

            with torch.no_grad():
                out = model(questions, responses)

            alpha_b = out["alpha"].cpu().numpy()   # (B, S, D)
            beta_b  = out["beta"].cpu().numpy()    # (B, S, K-1)
            theta_b = out["theta"].cpu().numpy()   # (B, S, D)
            q_np    = questions.cpu().numpy()       # (B, S)  1-based
            m_np    = mask.cpu().numpy()            # (B, S)

            B, S = q_np.shape
            for b in range(B):
                theta_vals = []
                for t in range(S):
                    if m_np[b, t]:
                        qid = int(q_np[b, t]) - 1
                        if 0 <= qid < Q:
                            alpha_sum[qid] += alpha_b[b, t]
                            alpha_count[qid] += 1
                            beta_sum[qid] += beta_b[b, t]
                            beta_count[qid] += 1
                        # theta — use dim-0 (D=1)
                        theta_vals.append(float(theta_b[b, t, 0]))
                if theta_vals:
                    student_theta_sum.append(np.mean(theta_vals))
                    student_theta_count.append(len(theta_vals))

    process(train_loader)
    process(test_loader)

    # Per-item means
    seen        = alpha_count > 0
    alpha_est   = alpha_sum / np.maximum(alpha_count[:, None], 1)   # (Q, D)
    beta_est    = beta_sum  / np.maximum(beta_count[:, None], 1)    # (Q, K-1)

    # Per-student mean theta
    theta_est = np.array(student_theta_sum)   # (N_students,)

    return seen, alpha_est, beta_est, theta_est


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _scatter_panel(
    ax: plt.Axes,
    true_vals: np.ndarray,
    est_vals: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    color: str = "steelblue",
) -> float:
    """Scatter plot with y=x line; returns Pearson r."""
    r = float(np.corrcoef(true_vals, est_vals)[0, 1])
    ax.scatter(true_vals, est_vals, alpha=0.35, s=12, color=color, edgecolors="none")
    lo = min(true_vals.min(), est_vals.min())
    hi = max(true_vals.max(), est_vals.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.1, label="$y=x$")
    ax.set_title(f"{title}\n$r = {r:.3f}$", fontsize=9.5)
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.tick_params(labelsize=8)
    return r


def make_figure(
    true_theta: np.ndarray,   # (N,)
    theta_est:  np.ndarray,   # (N,)  mean theta over sequence
    true_alpha: np.ndarray,   # (Q,)
    alpha_est:  np.ndarray,   # (Q, D)
    true_beta:  np.ndarray,   # (Q, K-1)
    beta_est:   np.ndarray,   # (Q, K-1)
    seen:       np.ndarray,   # (Q,) boolean
    output_path: Path,
) -> None:
    from scipy.stats import gaussian_kde, norm

    K1 = true_beta.shape[1]   # K-1 thresholds

    # Layout: 2 rows × 4 columns.
    # Row 0: theta KDE (cols 0-1) | alpha scatter (cols 2-3)  [2 wide panels]
    # Row 1: β_1 | β_2 | β_3 | β_4                           [4 individual panels]
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 4, hspace=0.48, wspace=0.38)

    ax_theta = fig.add_subplot(gs[0, 0:2])
    ax_alpha = fig.add_subplot(gs[0, 2:4])
    ax_beta  = [fig.add_subplot(gs[1, k]) for k in range(K1)]

    # ------------------------------------------------------------------
    # Row 0, left — Theta KDE
    # ------------------------------------------------------------------
    true_z = z_score(true_theta)
    est_z  = z_score(theta_est)

    xs = np.linspace(-4, 4, 300)
    for vals, label, color, ls in [
        (true_z, r"True $\theta^*$",          "steelblue",  "-"),
        (est_z,  r"Estimated $\hat{\theta}$",  "darkorange", "--"),
    ]:
        kde = gaussian_kde(vals, bw_method="scott")
        ax_theta.plot(xs, kde(xs), color=color, ls=ls, lw=2.0, label=label)

    ax_theta.plot(xs, norm.pdf(xs), color="gray", lw=1.0, ls=":", label=r"$\mathcal{N}(0,1)$")
    r_theta = float(np.corrcoef(true_z, est_z)[0, 1])
    ax_theta.set_title(f"Student ability ($\\theta$)\n$r = {r_theta:.3f}$", fontsize=9.5)
    ax_theta.set_xlabel("Normalised $\\theta$", fontsize=8.5)
    ax_theta.set_ylabel("Density", fontsize=8.5)
    ax_theta.legend(fontsize=8)
    ax_theta.grid(True, alpha=0.25, lw=0.6)
    ax_theta.tick_params(labelsize=8)

    # ------------------------------------------------------------------
    # Row 0, right — Alpha scatter
    # ------------------------------------------------------------------
    true_a = link_alpha(true_alpha[seen])
    est_a  = link_alpha(alpha_est[seen, 0])

    r_a = _scatter_panel(
        ax_alpha, true_a, est_a,
        "Discrimination ($\\alpha$)",
        "True $\\alpha$ (log-linked)",
        "Estimated $\\hat{\\alpha}$ (log-linked)",
        color="steelblue",
    )

    # ------------------------------------------------------------------
    # Row 1 — Individual beta scatter plots
    # ------------------------------------------------------------------
    # Use a shared colour ramp so each threshold has a distinct hue
    beta_colors = plt.get_cmap("plasma")(np.linspace(0.15, 0.85, K1))
    rs_beta = []

    for k in range(K1):
        true_b = z_score(true_beta[seen, k])
        est_b  = z_score(beta_est[seen, k])
        r_k = _scatter_panel(
            ax_beta[k], true_b, est_b,
            f"Step difficulty $\\beta_{k+1}$",
            f"True $\\beta_{k+1}$ (z-scored)",
            f"Est.\\ $\\hat{{\\beta}}_{k+1}$ (z-scored)",
            color=beta_colors[k],
        )
        rs_beta.append(r_k)

    mean_r = float(np.mean(rs_beta))

    # ------------------------------------------------------------------
    fig.suptitle(
        f"DEEP-GPCM IRT parameter recovery — large\\_q5000  "
        f"(SIE, $Q=5{{,}}000$, $K=5$, $N=1{{,}}000$)\n"
        f"$r_\\alpha={r_a:.3f}$  |  "
        f"$r_{{\\beta_1}}={rs_beta[0]:.3f}$  "
        f"$r_{{\\beta_2}}={rs_beta[1]:.3f}$  "
        f"$r_{{\\beta_3}}={rs_beta[2]:.3f}$  "
        f"$r_{{\\beta_4}}={rs_beta[3]:.3f}$  "
        f"(mean $r_\\beta={mean_r:.3f}$)",
        fontsize=9.5,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {output_path}")
    print(f"  r_theta = {r_theta:.3f}")
    print(f"  r_alpha = {r_a:.3f}")
    for k, r_k in enumerate(rs_beta):
        print(f"  r_beta[{k}] = {r_k:.3f}")
    print(f"  mean r_beta = {mean_r:.3f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output",     required=True)
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device("cpu")   # inference only, CPU is fine

    seen, alpha_est, beta_est, theta_est = run_inference(cfg, args.checkpoint, device)

    # Load ground-truth params
    data_root = Path(cfg.data.data_dir)
    if not data_root.is_absolute():
        data_root = Path(args.config).parent.parent / data_root
    irt_path = data_root / cfg.data.dataset_name / "true_irt_parameters.json"
    with irt_path.open() as f:
        true_irt = json.load(f)

    true_theta = np.array(true_irt["theta"])   # (N,)
    true_alpha = np.array(true_irt["alpha"])   # (Q,)
    true_beta  = np.array(true_irt["beta"])    # (Q, K-1)

    print(f"True theta: N={len(true_theta)}, Est theta: N={len(theta_est)}")
    print(f"Items seen by model: {seen.sum()} / {len(seen)}")

    make_figure(
        true_theta, theta_est,
        true_alpha, alpha_est,
        true_beta,  beta_est,
        seen,
        Path(args.output),
    )


if __name__ == "__main__":
    main()
