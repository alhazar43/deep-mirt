"""Combined IRT parameter recovery figure — overlays Static, Dynamic, and DEEP-GPCM."""

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
    "pgf.preamble": r"\usepackage{lmodern}\usepackage{amsmath}\usepackage{amssymb}\usepackage{times}",
    "axes.unicode_minus": False,
})
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

from kt_gpcm.config import load_config
from kt_gpcm.data.loaders import DataModule
from kt_gpcm.models.kt_gpcm import DeepGPCM
from kt_gpcm.models.static_gpcm import StaticGPCM
from kt_gpcm.models.dynamic_gpcm import DynamicGPCM

COLORS = {
    "DEEP-GPCM":    "#E87722",
    "Static GPCM":  "#1F77B4",
    "Dynamic GPCM": "#17BECF",
}
ALPHA_BASE = 0.22


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
                tv = []
                for s in range(qn.shape[1]):
                    if mn[bi, s]:
                        qid = int(qn[bi, s]) - 1
                        if 0 <= qid < Q:
                            alpha_sum[qid] += a[bi, s]; alpha_cnt[qid] += 1
                            beta_sum[qid]  += b[bi, s]; beta_cnt[qid]  += 1
                        tv.append(float(t[bi, s, 0]))
                if tv: theta_list.append(np.mean(tv))
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
                tv = []
                for s in range(qn.shape[1]):
                    if mn[bi, s]:
                        qid = int(qn[bi, s]) - 1
                        if 0 <= qid < Q:
                            alpha_sum[qid] += a[bi, s]; alpha_cnt[qid] += 1
                            beta_sum[qid]  += b[bi, s]; beta_cnt[qid]  += 1
                        tv.append(float(t[bi, s, 0]))
                if tv: theta_list.append(np.mean(tv))
    process(train_loader); process(test_loader)
    seen = alpha_cnt > 0
    return (seen,
            alpha_sum / np.maximum(alpha_cnt[:, None], 1),
            beta_sum  / np.maximum(beta_cnt[:, None], 1),
            np.array(theta_list))


def _annotate_r(ax, r_dict):
    """Stacked per-model r values, top-left, colour-coded."""
    y = 0.97
    for name, r in r_dict.items():
        ax.text(0.04, y, f"$r={r:.3f}$", transform=ax.transAxes,
                fontsize=6.5, va="top", color=COLORS[name],
                fontweight="bold" if name == "DEEP-GPCM" else "normal")
        y -= 0.15


def _yx_line(ax):
    """Draw y=x reference line with equal axes."""
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    lo = min(xl[0], yl[0])
    hi = max(xl[1], yl[1])
    ax.plot([lo, hi], [lo, hi], color="black", ls="--", lw=0.9, zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)


def make_figure(models_data, true_theta, true_alpha, true_beta, output_path):
    from scipy.stats import gaussian_kde, norm

    K1 = true_beta.shape[1]          # K-1 beta thresholds
    n_cols = 2 + K1                   # theta + alpha + K1 betas
    model_names = list(models_data.keys())
    n_rows = len(model_names)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(6.5, 1.15 * n_rows + 0.55),
                              squeeze=False)
    fig.subplots_adjust(hspace=0.58, wspace=0.40,
                        top=0.82, bottom=0.10, left=0.06, right=0.99)

    xs = np.linspace(-4, 4, 300)
    true_z = z_score(true_theta)

    fig.suptitle(r"IRT Parameter Recovery ($K=5$, $Q=200$)", fontsize=8.5, y=0.98)

    col_titles = ([r"Ability $\theta$", r"Discrimination $\alpha$"] +
                  [f"$\\beta_{k+1}$" for k in range(K1)])

    r_alpha_all = {}
    r_beta_all  = {name: [] for name in model_names}

    for row_idx, name in enumerate(model_names):
        seen, alpha_est, beta_est, theta_est = models_data[name]
        color = COLORS[name]
        axes_row = axes[row_idx]

        # ── Col 0: Theta KDE ─────────────────────────────────────────────
        ax = axes_row[0]
        ax.plot(xs, norm.pdf(xs), color="gray", lw=0.8, ls=":",
                label=r"$\mathcal{N}(0,1)$")
        ax.plot(xs, gaussian_kde(true_z, bw_method="scott")(xs),
                color="black", ls="--", lw=1.4, label=r"True $\theta^*$")
        ax.plot(xs, gaussian_kde(z_score(theta_est), bw_method="scott")(xs),
                color=color, lw=1.6, label=r"Est.\ $\hat{\theta}$")
        ax.set_title(name, fontsize=7, pad=2, color=color,
                     fontweight="bold" if name == "DEEP-GPCM" else "normal")
        ax.set_xlabel(r"Norm.\ $\theta$", fontsize=6.5, labelpad=1)
        ax.set_ylabel("Density", fontsize=6.5, labelpad=1)
        ax.grid(True, alpha=0.2, lw=0.4)
        ax.tick_params(labelsize=6, pad=1)

        # ── Col 1: Alpha scatter ─────────────────────────────────────────
        ax = axes_row[1]
        ta = link_alpha(true_alpha[seen])
        ea = link_alpha(alpha_est[seen, 0])
        r  = float(np.corrcoef(ta, ea)[0, 1])
        r_alpha_all[name] = r
        ax.scatter(ta, ea, alpha=0.4, s=5, color=color, edgecolors="none")
        _yx_line(ax)
        ax.text(0.05, 0.93, f"$r={r:.3f}$", transform=ax.transAxes,
                fontsize=6, va="top", color=color)
        if row_idx == 0:
            ax.set_title(col_titles[1], fontsize=7.5, pad=2)
        ax.set_xlabel(r"True $\alpha$", fontsize=6.5, labelpad=1)
        ax.set_ylabel(r"Est.\ $\hat{\alpha}$", fontsize=6.5, labelpad=1)
        ax.grid(True, alpha=0.2, lw=0.4)
        ax.tick_params(labelsize=6, pad=1)

        # ── Cols 2+: Beta scatters ───────────────────────────────────────
        for k in range(K1):
            ax = axes_row[2 + k]
            tb = link_normal(true_beta[seen, k])
            eb = link_normal(beta_est[seen, k])
            r  = float(np.corrcoef(tb, eb)[0, 1])
            r_beta_all[name].append(r)
            ax.scatter(tb, eb, alpha=0.4, s=5, color=color, edgecolors="none")
            _yx_line(ax)
            ax.text(0.05, 0.93, f"$r={r:.3f}$", transform=ax.transAxes,
                    fontsize=6, va="top", color=color)
            if row_idx == 0:
                ax.set_title(col_titles[2 + k], fontsize=7.5, pad=2)
            ax.set_xlabel(f"True $\\beta_{k+1}$", fontsize=6.5, labelpad=1)
            ax.set_ylabel(f"Est.\\ $\\hat{{\\beta}}_{k+1}$", fontsize=6.5, labelpad=1)
            ax.grid(True, alpha=0.2, lw=0.4)
            ax.tick_params(labelsize=6, pad=1)

    # Single legend below suptitle, above first row
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=6.5,
               bbox_to_anchor=(0.5, 0.90), framealpha=0.92, edgecolor="lightgray",
               handlelength=1.2, columnspacing=0.8)

    pgf_path = Path(output_path).with_suffix(".pgf")
    pgf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pgf_path)
    fig.savefig(pgf_path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"\nFigure saved: {pgf_path}")
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

    data_root = Path(cfg_deep.data.data_dir)
    if not data_root.is_absolute():
        data_root = Path(args.deepgpcm_config).parent.parent / data_root
    with (data_root / cfg_deep.data.dataset_name / "true_irt_parameters.json").open() as f:
        true_irt = json.load(f)

    models_data = {
        "Static GPCM":  stat_data,
        "Dynamic GPCM": dyn_data,
        "DEEP-GPCM":    deep_data,
    }
    make_figure(models_data,
                np.array(true_irt["theta"]),
                np.array(true_irt["alpha"]),
                np.array(true_irt["beta"]),
                Path(args.output))


if __name__ == "__main__":
    main()
