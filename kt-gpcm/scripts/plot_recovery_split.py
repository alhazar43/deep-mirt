#!/usr/bin/env python3
"""Split recovery figure: student params (theta) vs item params (alpha, beta).

Generates two separate figures:
1. recovery_student.pgf — 3×1 theta KDE plots
2. recovery_item.pgf — 3×(K) alpha + beta scatter plots

Usage:
    PYTHONPATH=src python scripts/plot_recovery_split.py \
      --deepgpcm-config configs/deepgpcm_k5_s42.yaml \
      --deepgpcm-checkpoint outputs/deepgpcm_k5_s42/best.pt \
      --static-config configs/static_gpcm_k5_s42.yaml \
      --static-checkpoint outputs/static_gpcm_k5_s42/best.pt \
      --dynamic-config configs/dynamic_gpcm_k5_s42.yaml \
      --dynamic-checkpoint outputs/dynamic_gpcm_k5_s42/best.pt \
      --output outputs/deepgpcm_k5_s42/recovery
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "pgf.preamble": "\\usepackage{lmodern}\\usepackage{amsmath}\\usepackage{times}",
})
import matplotlib.pyplot as plt
import numpy as np
import torch

from kt_gpcm.config import load_config
from kt_gpcm.data.loaders import DataModule
from kt_gpcm.models.kt_gpcm import DeepGPCM
from kt_gpcm.models.static_gpcm import StaticGPCM
from kt_gpcm.models.dynamic_gpcm import DynamicGPCM

COLORS = {
    "Static GPCM": "#1f77b4",
    "Dynamic GPCM": "#17becf",
    "DEEP-GPCM": "#ff7f0e",
}


def z_score(x):
    s = x.std()
    return (x - x.mean()) / max(s, 1e-8)


def link_alpha(x, target_std=0.3):
    """Log-space z-score then rescale to target_std (lognormal linking)."""
    log_v = np.log(np.maximum(x, 1e-6))
    std = log_v.std()
    if std < 1e-6:
        return np.ones_like(x)
    return np.exp((log_v - log_v.mean()) / std * target_std)


def link_normal(x):
    """Z-score to N(0,1) — IRT linking for theta/beta."""
    std = x.std()
    if std < 1e-6:
        return x - x.mean()
    return (x - x.mean()) / std


def run_inference_deepgpcm(cfg, checkpoint, device):
    model_kwargs = {k: v for k, v in vars(cfg.model).items() if k != "model_type"}
    model = DeepGPCM(**model_kwargs).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    data_mgr = DataModule(cfg)
    train_loader, test_loader = data_mgr.build()
    Q, D, K = cfg.model.n_questions, cfg.model.n_traits, cfg.model.n_categories
    alpha_sum = np.zeros((Q, D)); alpha_cnt = np.zeros(Q)
    beta_sum  = np.zeros((Q, K-1)); beta_cnt = np.zeros(Q)
    theta_list = []
    def process(loader):
        for batch in loader:
            q, r, mask = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            with torch.no_grad():
                out = model(q, r)
            a = out["alpha"].cpu().numpy(); b = out["beta"].cpu().numpy()
            t = out["theta"].cpu().numpy(); qn = q.cpu().numpy(); mn = mask.cpu().numpy()
            for bi in range(qn.shape[0]):
                last_theta = None
                for s in range(qn.shape[1]):
                    if mn[bi, s]:
                        qid = int(qn[bi, s]) - 1
                        if 0 <= qid < Q:
                            alpha_sum[qid] += a[bi, s]; alpha_cnt[qid] += 1
                            beta_sum[qid]  += b[bi, s]; beta_cnt[qid]  += 1
                        last_theta = float(t[bi, s, 0])
                if last_theta is not None: theta_list.append(last_theta)
    process(train_loader); process(test_loader)
    seen = alpha_cnt > 0
    return (seen,
            alpha_sum / np.maximum(alpha_cnt[:, None], 1),
            beta_sum  / np.maximum(beta_cnt[:, None], 1),
            np.array(theta_list))


def run_inference_irt(cfg, checkpoint, device, model_class):
    model_kwargs = {k: v for k, v in vars(cfg.model).items() if k != "model_type"}
    state = torch.load(checkpoint, map_location=device)
    n_students = state["model"]["theta_embed.weight"].shape[0] - 1
    data_mgr = DataModule(cfg)
    model = model_class(n_students=n_students, **model_kwargs).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    train_loader, test_loader = data_mgr.build()
    Q, D, K = cfg.model.n_questions, cfg.model.n_traits, cfg.model.n_categories
    alpha_sum = np.zeros((Q, D)); alpha_cnt = np.zeros(Q)
    beta_sum  = np.zeros((Q, K-1)); beta_cnt = np.zeros(Q)
    theta_list = []
    def process(loader):
        for batch in loader:
            q, r, mask, sid = (batch[0].to(device), batch[1].to(device),
                               batch[2].to(device), batch[3].to(device))
            with torch.no_grad():
                out = model(sid, q, r)
            a = out["alpha"].cpu().numpy(); b = out["beta"].cpu().numpy()
            t = out["theta"].cpu().numpy(); qn = q.cpu().numpy(); mn = mask.cpu().numpy()
            for bi in range(qn.shape[0]):
                last_theta = None
                for s in range(qn.shape[1]):
                    if mn[bi, s]:
                        qid = int(qn[bi, s]) - 1
                        if 0 <= qid < Q:
                            alpha_sum[qid] += a[bi, s]; alpha_cnt[qid] += 1
                            beta_sum[qid]  += b[bi, s]; beta_cnt[qid]  += 1
                        last_theta = float(t[bi, s, 0])
                if last_theta is not None: theta_list.append(last_theta)
    process(train_loader); process(test_loader)
    seen = alpha_cnt > 0
    return (seen,
            alpha_sum / np.maximum(alpha_cnt[:, None], 1),
            beta_sum  / np.maximum(beta_cnt[:, None], 1),
            np.array(theta_list))


def _yx_line(ax):
    """Draw y=x reference line with equal axes."""
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    lo = min(xl[0], yl[0])
    hi = max(xl[1], yl[1])
    ax.plot([lo, hi], [lo, hi], color="black", ls="--", lw=0.9, zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)


def make_student_figure(models_data, true_theta, K, output_path):
    """3×1 theta KDE plots."""
    from scipy.stats import gaussian_kde, norm
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    model_names = list(models_data.keys())
    n_rows = len(model_names)

    fig, axes = plt.subplots(n_rows, 1, figsize=(3.5, 1.25 * n_rows + 0.3), squeeze=False)
    fig.subplots_adjust(hspace=0.35, top=0.86, bottom=0.08, left=0.13, right=0.96)

    xs = np.linspace(-4, 4, 300)
    true_z = z_score(true_theta)

    # Main title with larger font
    fig.suptitle(r"Student Ability $\theta$", fontsize=11, y=0.98, fontweight="bold")

    # Merged legend: model colors + line styles in single row
    # Offset x-coordinate to account for left margin (y-axis labels)
    model_handles = [Patch(facecolor=COLORS[name], label=name) for name in model_names]
    line_handles = [
        Line2D([0], [0], color="gray", lw=0.8, ls=":", label=r"$\mathcal{N}(0,1)$"),
        Line2D([0], [0], color="black", lw=1.4, ls="--", label=r"True $\theta^*$"),
    ]
    all_handles = model_handles + line_handles
    fig.legend(handles=all_handles, loc="upper center", ncol=5, fontsize=6,
               bbox_to_anchor=(0.54, 0.90), framealpha=0.92, edgecolor="lightgray",
               handlelength=1.1, columnspacing=0.5, handletextpad=0.3)

    for row_idx, name in enumerate(model_names):
        seen, alpha_est, beta_est, theta_est = models_data[name]
        color = COLORS[name]
        ax = axes[row_idx, 0]

        ax.plot(xs, norm.pdf(xs), color="gray", lw=0.8, ls=":")
        ax.plot(xs, gaussian_kde(true_z, bw_method="scott")(xs),
                color="black", ls="--", lw=1.4)
        ax.plot(xs, gaussian_kde(z_score(theta_est), bw_method="scott")(xs),
                color=color, lw=1.6)
        # No row title - parameter name is in suptitle
        ax.set_xlabel(r"Norm.\ $\theta$", fontsize=6.5, labelpad=1)
        ax.set_ylabel("Density", fontsize=6.5, labelpad=1)
        ax.grid(True, alpha=0.2, lw=0.4)
        ax.tick_params(labelsize=6, pad=1)

    pgf_path = Path(output_path + "_student.pgf")
    pgf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pgf_path)
    fig.savefig(pgf_path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"Student figure saved: {pgf_path}")


def make_item_figure(models_data, true_alpha, true_beta, K, output_path):
    """3×(1+K-1) alpha + beta scatter plots, slightly larger."""
    from matplotlib.patches import Patch

    K1 = true_beta.shape[1]
    n_cols = 1 + K1  # alpha + K-1 betas
    model_names = list(models_data.keys())
    n_rows = len(model_names)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(6.5, 1.1 * n_rows + 0.4),
                              squeeze=False)
    fig.subplots_adjust(hspace=0.40, wspace=0.32,
                        top=0.85, bottom=0.08, left=0.06, right=0.99)

    # Main title with larger font
    fig.suptitle(r"Item Parameters", fontsize=11, y=0.97, fontweight="bold")

    # Color legend after title
    legend_handles = [Patch(facecolor=COLORS[name], label=name) for name in model_names]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, fontsize=6.5,
               bbox_to_anchor=(0.5, 0.91), framealpha=0.92, edgecolor="lightgray",
               handlelength=1.0, columnspacing=0.8)

    col_titles = [r"Discrimination $\alpha$"] + [f"Threshold $\\beta_{{{k+1}}}$" for k in range(K1)]

    r_alpha_all = {}
    r_beta_all  = {name: [] for name in model_names}

    for row_idx, name in enumerate(model_names):
        seen, alpha_est, beta_est, theta_est = models_data[name]
        color = COLORS[name]
        axes_row = axes[row_idx]

        # Col 0: Alpha scatter
        ax = axes_row[0]
        ta = link_alpha(true_alpha[seen])
        ea = link_alpha(alpha_est[seen, 0])
        r  = float(np.corrcoef(ta, ea)[0, 1])
        r_alpha_all[name] = r
        ax.scatter(ta, ea, alpha=0.4, s=6, color=color, edgecolors="none")
        _yx_line(ax)
        ax.text(0.05, 0.93, f"$r={r:.3f}$", transform=ax.transAxes,
                fontsize=6, va="top", color=color)
        if row_idx == 0:
            ax.set_title(col_titles[0], fontsize=7.5, pad=2)
        ax.set_xlabel(r"True $\alpha$", fontsize=6.5, labelpad=1)
        ax.set_ylabel(r"Est.\ $\hat{\alpha}$", fontsize=6.5, labelpad=1)
        ax.grid(True, alpha=0.2, lw=0.4)
        ax.tick_params(labelsize=6, pad=1)

        # Cols 1+: Beta scatters
        for k in range(K1):
            ax = axes_row[1 + k]
            tb = link_normal(true_beta[seen, k])
            eb = link_normal(beta_est[seen, k])
            r  = float(np.corrcoef(tb, eb)[0, 1])
            r_beta_all[name].append(r)
            ax.scatter(tb, eb, alpha=0.4, s=6, color=color, edgecolors="none")
            _yx_line(ax)
            ax.text(0.05, 0.93, f"$r={r:.3f}$", transform=ax.transAxes,
                    fontsize=6, va="top", color=color)
            if row_idx == 0:
                ax.set_title(col_titles[1 + k], fontsize=7.5, pad=2)
            ax.set_xlabel(f"True $\\beta_{{{k+1}}}$", fontsize=6.5, labelpad=1)
            ax.set_ylabel(f"Est.\ $\\hat{{\\beta}}_{{{k+1}}}$", fontsize=6.5, labelpad=1)
            ax.grid(True, alpha=0.2, lw=0.4)
            ax.tick_params(labelsize=6, pad=1)

    pgf_path = Path(output_path + "_item.pgf")
    pgf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pgf_path)
    fig.savefig(pgf_path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"Item figure saved: {pgf_path}")
    for name in models_data:
        print(f"  {name}: r_alpha={r_alpha_all[name]:.3f}  "
              f"r_beta_mean={np.mean(r_beta_all[name]):.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepgpcm-config",      required=True)
    parser.add_argument("--deepgpcm-checkpoint",  required=True)
    parser.add_argument("--static-config",        required=True)
    parser.add_argument("--static-checkpoint",    required=True)
    parser.add_argument("--dynamic-config",       required=True)
    parser.add_argument("--dynamic-checkpoint",   required=True)
    parser.add_argument("--output",               required=True)
    args = parser.parse_args()

    device = torch.device("cpu")
    cfg_deep = load_config(args.deepgpcm_config)
    cfg_stat = load_config(args.static_config)
    cfg_dyn  = load_config(args.dynamic_config)

    print("Running DEEP-GPCM inference...")
    deep_data = run_inference_deepgpcm(cfg_deep, args.deepgpcm_checkpoint, device)
    print("Running Static GPCM inference...")
    stat_data = run_inference_irt(cfg_stat, args.static_checkpoint, device, StaticGPCM)
    print("Running Dynamic GPCM inference...")
    dyn_data  = run_inference_irt(cfg_dyn,  args.dynamic_checkpoint,  device, DynamicGPCM)

    # Resolve data_dir relative to project root (kt-gpcm/)
    config_path = Path(args.deepgpcm_config).resolve()
    project_root = config_path.parent
    while project_root.name != "kt-gpcm" and project_root.parent != project_root:
        project_root = project_root.parent

    data_root = Path(cfg_deep.data.data_dir)
    if not data_root.is_absolute():
        data_root = project_root / data_root

    true_params_path = data_root / cfg_deep.data.dataset_name / "true_irt_parameters.json"
    if not true_params_path.exists():
        raise FileNotFoundError(f"True IRT parameters not found: {true_params_path}")

    with true_params_path.open() as f:
        true_irt = json.load(f)

    models_data = {
        "Static GPCM": stat_data,
        "Dynamic GPCM": dyn_data,
        "DEEP-GPCM": deep_data,
    }

    K = cfg_deep.model.n_categories
    make_student_figure(models_data, np.array(true_irt["theta"]), K, args.output)
    make_item_figure(models_data, np.array(true_irt["alpha"]), np.array(true_irt["beta"]), K, args.output)


if __name__ == "__main__":
    main()
