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

FS = {
    "suptitle": 13,
    "subtitle": 8,
    "axis":     9,
    "tick":     6.0,
    "annot":    6.5,
    "legend":   7,
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
    """n_models×1 theta KDE plots — fixed cell size, figure height adapts to n_models."""
    from scipy.stats import gaussian_kde
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    model_names = list(models_data.keys())
    n_rows = len(model_names)

    # Fixed cell dimensions (inches)
    CELL_W, CELL_H = 2.8, 1.1
    left_m, right_m = 0.42, 0.14   # S1: no y-label text, just tick space
    top_m, bot_m   = 0.62, 0.40    # S2: compact top for suptitle + legend
    v_gap = 0.18                    # S1: no subplot titles, rows can be tight

    fig_w = left_m + CELL_W + right_m
    fig_h = top_m + n_rows * CELL_H + (n_rows - 1) * v_gap + bot_m

    # S2: center over plot area, not figure (asymmetric margins)
    plot_cx = (left_m + fig_w - right_m) / 2 / fig_w

    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_w, fig_h), squeeze=False)
    fig.subplots_adjust(
        left   = left_m / fig_w,
        right  = 1 - right_m / fig_w,
        bottom = bot_m / fig_h,
        top    = 1 - top_m / fig_h,
        hspace = v_gap / CELL_H,
    )

    xs = np.linspace(-4, 4, 300)
    true_z = z_score(true_theta)

    suptitle_y = 1 - 0.07 / fig_h
    legend_y   = 1 - 0.28 / fig_h   # S2: tighter gap from title
    fig.suptitle(r"\textbf{Student Ability} $\theta$", fontsize=FS["suptitle"],
                 y=suptitle_y, x=plot_cx)

    # S3: remove N(0,1) reference — true θ is drawn from N(0,1) so it's redundant
    model_handles = [Patch(facecolor=COLORS[name], label=name) for name in model_names]
    true_handle = Line2D([0], [0], color="black", lw=1.4, ls="--", label=r"True $\theta^*$")
    fig.legend(handles=model_handles + [true_handle], loc="upper center",
               ncol=min(4, n_rows + 1),
               fontsize=FS["legend"],
               bbox_to_anchor=(plot_cx, legend_y), framealpha=0.92, edgecolor="lightgray",
               handlelength=1.1, columnspacing=0.5, handletextpad=0.3)

    # S1: shared axis labels via fig.text
    plot_left = left_m / fig_w
    plot_bot  = bot_m / fig_h
    plot_top  = 1 - top_m / fig_h
    fig.text(plot_cx, plot_bot * 0.38,
             r"\textbf{Normalized} $\theta$", ha="center", va="center", fontsize=FS["axis"])
    fig.text(plot_left * 0.20, (plot_bot + plot_top) / 2,
             r"\textbf{Density}", ha="center", va="center", fontsize=FS["axis"], rotation=90)

    for row_idx, name in enumerate(model_names):
        seen, alpha_est, beta_est, theta_est = models_data[name]
        color = COLORS[name]
        ax = axes[row_idx, 0]

        # S3: no N(0,1) line; keep only true θ KDE and estimated θ KDE
        ax.plot(xs, gaussian_kde(true_z, bw_method="scott")(xs),
                color="black", ls="--", lw=1.4)
        ax.plot(xs, gaussian_kde(z_score(theta_est), bw_method="scott")(xs),
                color=color, lw=1.6)
        ax.grid(True, alpha=0.2, lw=0.4)
        ax.tick_params(labelsize=FS["tick"], pad=1)

    pgf_path = Path(output_path + "_student.pgf")
    pgf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pgf_path)
    try:
        fig.savefig(pgf_path.with_suffix(".pdf"))
    except Exception as e:
        print(f"  PDF save skipped: {e}")
    try:
        fig.savefig(pgf_path.with_suffix(".png"), dpi=150)
    except Exception as e:
        print(f"  PNG save skipped: {e}")
    plt.close(fig)
    print(f"Student figure saved: {pgf_path}")


def make_item_figure(models_data, true_alpha, true_beta, K, output_path):
    """n_models×(1+K-1) alpha + beta scatter plots.
    Fixed square cell size; figure width grows with K, height grows with n_models.
    """
    from matplotlib.patches import Patch

    K1 = true_beta.shape[1]
    n_cols = 1 + K1  # alpha + K-1 betas
    model_names = list(models_data.keys())
    n_rows = len(model_names)

    # Fixed cell size (inches) — wider than tall
    CELL_W = 1.35
    CELL_H = 1.10
    left_m, right_m = 0.44, 0.12   # no row labels; left just for y-ticks
    top_m, bot_m   = 0.66, 0.48    # compact top; bot for shared x-label
    h_gap = 0.30                    # tight: no per-cell y-axis labels
    v_gap = 0.18                    # tight: matches student figure density

    fig_w = left_m + n_cols * CELL_W + (n_cols - 1) * h_gap + right_m
    fig_h = top_m + n_rows * CELL_H + (n_rows - 1) * v_gap + bot_m

    # Center over plot area (asymmetric left/right margins)
    plot_cx = (left_m + fig_w - right_m) / 2 / fig_w

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    fig.subplots_adjust(
        left   = left_m / fig_w,
        right  = 1 - right_m / fig_w,
        bottom = bot_m / fig_h,
        top    = 1 - top_m / fig_h,
        wspace = h_gap / CELL_W,
        hspace = v_gap / CELL_H,
    )

    suptitle_y = 1 - 0.07 / fig_h
    legend_y   = 1 - 0.28 / fig_h
    fig.suptitle(r"\textbf{Item Parameters}", fontsize=FS["suptitle"],
                 y=suptitle_y, x=plot_cx)

    legend_handles = [Patch(facecolor=COLORS[name], label=name) for name in model_names]
    fig.legend(handles=legend_handles, loc="upper center", ncol=n_rows,
               fontsize=FS["legend"],
               bbox_to_anchor=(plot_cx, legend_y), framealpha=0.92, edgecolor="lightgray",
               handlelength=1.0, columnspacing=0.8)

    # Single shared axis labels via fig.text
    plot_left  = left_m / fig_w
    plot_right = 1 - right_m / fig_w
    plot_bot   = bot_m / fig_h
    plot_top   = 1 - top_m / fig_h
    fig.text(plot_cx, plot_bot * 0.38,
             r"\textbf{True}", ha="center", va="center", fontsize=FS["axis"])
    fig.text(plot_left * 0.20, (plot_bot + plot_top) / 2,
             r"\textbf{Estimated}", ha="center", va="center",
             fontsize=FS["axis"], rotation=90)

    col_titles = [r"\textbf{Discrimination} $\alpha$"] + \
                 [rf"\textbf{{Step threshold}} $\beta_{{{k+1}}}$" for k in range(K1)]

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
        ax.scatter(ta, ea, alpha=0.4, s=5, color=color, edgecolors="none")
        _yx_line(ax)
        ax.text(0.05, 0.93, f"$r={r:.3f}$", transform=ax.transAxes,
                fontsize=FS["annot"], va="top", color=color)
        if row_idx == 0:
            ax.set_title(col_titles[0], fontsize=FS["subtitle"], pad=2)
        ax.grid(True, alpha=0.2, lw=0.4)
        ax.tick_params(labelsize=FS["tick"], pad=1)

        # Cols 1+: Beta scatters
        for k in range(K1):
            ax = axes_row[1 + k]
            tb = link_normal(true_beta[seen, k])
            eb = link_normal(beta_est[seen, k])
            r  = float(np.corrcoef(tb, eb)[0, 1])
            r_beta_all[name].append(r)
            ax.scatter(tb, eb, alpha=0.4, s=5, color=color, edgecolors="none")
            _yx_line(ax)
            ax.text(0.05, 0.93, f"$r={r:.3f}$", transform=ax.transAxes,
                    fontsize=FS["annot"], va="top", color=color)
            if row_idx == 0:
                ax.set_title(col_titles[1 + k], fontsize=FS["subtitle"], pad=2)
            ax.grid(True, alpha=0.2, lw=0.4)
            ax.tick_params(labelsize=FS["tick"], pad=1)

    pgf_path = Path(output_path + "_item.pgf")
    pgf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pgf_path)
    try:
        fig.savefig(pgf_path.with_suffix(".pdf"))
    except Exception as e:
        print(f"  PDF save skipped: {e}")
    try:
        fig.savefig(pgf_path.with_suffix(".png"), dpi=150)
    except Exception as e:
        print(f"  PNG save skipped: {e}")
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
