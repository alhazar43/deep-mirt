"""Plot learner state trajectories — overlays DEEP-GPCM, DKVMN+Softmax, and Dynamic GPCM.

Usage (from kt-gpcm/):
    PYTHONPATH=src python scripts/plot_learner_trajectories.py \\
        --deepgpcm-config    configs/deepgpcm_k5_s42.yaml \\
        --deepgpcm-checkpoint outputs/deepgpcm_k5_s42/best.pt \\
        --softmax-config     configs/softmax_k5_s42.yaml \\
        --softmax-checkpoint outputs/softmax_k5_s42/best.pt \\
        --dynamic-config     configs/dynamic_gpcm_k5_s42.yaml \\
        --dynamic-checkpoint outputs/dynamic_gpcm_k5_s42/best.pt \\
        --output-dir         outputs/deepgpcm_k5_s42/trajectory_plots
"""

from __future__ import annotations
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
import numpy as np
import torch

from kt_gpcm.config.loader import load_config
from kt_gpcm.models.kt_gpcm import DeepGPCM
from kt_gpcm.models.dkvmn_softmax import DKVMNSoftmax
from kt_gpcm.models.dynamic_gpcm import DynamicGPCM
from kt_gpcm.data.loaders import DataModule

COLORS = {
    "DEEP-GPCM":       "#E87722",
    "DKVMN+Softmax":   "#2CA02C",
    "Dynamic GPCM":    "#17BECF",
}


def gpcm_prob_true(theta, alpha, beta):
    K = len(beta) + 1
    eta = float(alpha) * float(theta)
    alpha_norm = abs(float(alpha))
    cum = np.zeros(K)
    for k in range(1, K):
        cum[k] = np.sum(eta - alpha_norm * beta[:k])
    cum -= cum.max()
    exp_l = np.exp(cum)
    return exp_l / exp_l.sum()


def expected_score(probs):
    return float(np.dot(np.arange(len(probs), dtype=float), probs))


def smooth(x, w=9):
    out = np.convolve(x, np.ones(w) / w, mode="same")
    for i in range(w // 2):
        out[i] = x[:2*i+1].mean()
        out[-(i+1)] = x[-(2*i+1):].mean()
    return out


def select_students(theta_true, questions_all, responses_all):
    n = len(theta_true)
    sorted_idx = np.argsort(theta_true)
    high_idx = int(sorted_idx[-1])
    low_idx  = int(sorted_idx[0])

    mid_long_idx = None
    for thr in [0.3, 0.6, 1.2]:
        cands = [i for i in range(n) if abs(theta_true[i]) < thr and len(questions_all[i]) > 80]
        if cands:
            mid_long_idx = int(max(cands, key=lambda i: len(questions_all[i])))
            break
    if mid_long_idx is None:
        mid_long_idx = int(sorted_idx[n // 2])

    lo40, hi60 = np.percentile(theta_true, [40, 60])
    mid_pool = [i for i in range(n) if lo40 <= theta_true[i] <= hi60] or list(range(n))
    ambig_idx = int(mid_pool[int(np.argmax([float(np.var(responses_all[i])) for i in mid_pool]))])

    return {"High-ability": high_idx, "Low-ability": low_idx,
            "Mid-ability": mid_long_idx, "Ambiguous": ambig_idx}


def select_reference_item(alpha_true, beta_true):
    mean_alpha = float(np.mean(alpha_true))
    for tol in [0.15, 0.30, 1.0]:
        mask = np.abs(alpha_true - mean_alpha) < tol
        if mask.any():
            cands = np.where(mask)[0]
            break
    return int(cands[np.argmin(np.abs(beta_true[cands, 0]))])


@torch.no_grad()
def run_deepgpcm(model, questions, responses, device):
    q = torch.tensor([questions], dtype=torch.long, device=device)
    r = torch.tensor([responses], dtype=torch.long, device=device)
    out = model(q, r)
    theta = out["theta"][0, :, 0].cpu().numpy()
    probs = out["probs"][0].cpu().numpy()
    return theta, probs


@torch.no_grad()
def run_softmax(model, questions, responses, device):
    q = torch.tensor([questions], dtype=torch.long, device=device)
    r = torch.tensor([responses], dtype=torch.long, device=device)
    out = model(q, r)
    probs = out["probs"][0].cpu().numpy()
    return probs


@torch.no_grad()
def run_dynamic(model, student_id, questions, responses, device):
    sid = torch.tensor([[student_id] * len(questions)], dtype=torch.long, device=device)
    q   = torch.tensor([questions], dtype=torch.long, device=device)
    r   = torch.tensor([responses], dtype=torch.long, device=device)
    out = model(sid, q, r)
    theta = out["theta"][0, :, 0].cpu().numpy()
    probs = out["probs"][0].cpu().numpy()
    return theta, probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepgpcm-config",      required=True)
    parser.add_argument("--deepgpcm-checkpoint",  required=True)
    parser.add_argument("--softmax-config",       required=True)
    parser.add_argument("--softmax-checkpoint",   required=True)
    parser.add_argument("--dynamic-config",       required=True)
    parser.add_argument("--dynamic-checkpoint",   required=True)
    parser.add_argument("--output-dir",           default=None)
    args = parser.parse_args()

    device = torch.device("cpu")
    cfg = load_config(args.deepgpcm_config)
    K   = cfg.model.n_categories

    out_dir = Path(args.output_dir) if args.output_dir else \
              Path(args.deepgpcm_checkpoint).parent / "trajectory_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(cfg.data.data_dir)
    if not data_root.is_absolute():
        data_root = Path(args.deepgpcm_config).parent.parent / data_root
    data_dir = data_root / cfg.data.dataset_name

    with (data_dir / "sequences.json").open() as f:
        seqs = json.load(f)
    with (data_dir / "true_irt_parameters.json").open() as f:
        params = json.load(f)

    questions_all = [s["questions"] for s in seqs]
    responses_all = [s["responses"] for s in seqs]
    theta_true = np.array(params["theta"])
    alpha_true = np.array(params["alpha"])
    beta_true  = np.array(params["beta"])

    _skip = {"model_type", "monotonic_betas"}
    model_kwargs = {k: v for k, v in vars(cfg.model).items() if k not in _skip}
    deep_kwargs  = {k: v for k, v in vars(cfg.model).items() if k != "model_type"}

    deep_model = DeepGPCM(**deep_kwargs).to(device)
    deep_model.load_state_dict(torch.load(args.deepgpcm_checkpoint, map_location=device)["model"])
    deep_model.eval()

    sm_model = DKVMNSoftmax(**model_kwargs).to(device)
    sm_model.load_state_dict(torch.load(args.softmax_checkpoint, map_location=device)["model"])
    sm_model.eval()

    cfg_dyn  = load_config(args.dynamic_config)
    dyn_ckpt = torch.load(args.dynamic_checkpoint, map_location=device)
    dyn_n_students = dyn_ckpt["model"]["theta_embed.weight"].shape[0] - 1
    dyn_kwargs = {k: v for k, v in vars(cfg_dyn.model).items() if k != "model_type"}
    dyn_model = DynamicGPCM(n_students=dyn_n_students, **dyn_kwargs).to(device)
    dyn_model.load_state_dict(dyn_ckpt["model"])
    dyn_model.eval()

    selected  = select_students(theta_true, questions_all, responses_all)
    ref_item  = select_reference_item(alpha_true, beta_true)
    alpha_ref = float(alpha_true[ref_item])
    beta_ref  = beta_true[ref_item]
    k_vals    = np.arange(K, dtype=float)

    true_er = {label: expected_score(gpcm_prob_true(theta_true[idx], alpha_ref, beta_ref))
               for label, idx in selected.items()}

    labels_order = ["High-ability", "Low-ability", "Mid-ability", "Ambiguous"]
    fig, axes = plt.subplots(2, 2, figsize=(4.6, 3.6), sharey=False)
    plt.subplots_adjust(hspace=0.52, wspace=0.28)

    for ax, label in zip(axes.flat, labels_order):
        idx = selected[label]
        qs  = questions_all[idx]
        rs  = responses_all[idx]
        T   = len(qs)
        t   = np.arange(T)

        theta_deep, probs_deep = run_deepgpcm(deep_model, qs, rs, device)
        er_deep = smooth(probs_deep @ k_vals)
        theta_deep_disp = smooth(np.clip((theta_deep + 3.0) / 6.0 * (K - 1), 0, K - 1))

        probs_sm = run_softmax(sm_model, qs, rs, device)
        er_sm = smooth(probs_sm @ k_vals)

        theta_dyn, _ = run_dynamic(dyn_model, idx + 1, qs, rs, device)
        theta_dyn_disp = smooth(np.clip((theta_dyn + 3.0) / 6.0 * (K - 1), 0, K - 1))

        ax.axhline(true_er[label], color="black", lw=1.5, ls="--",
                   label=r"True $E[r\,|\,\theta^*={:.2f}]={:.2f}$".format(
                       theta_true[idx], true_er[label]))

        ax.plot(t, er_deep,         color=COLORS["DEEP-GPCM"],    lw=2.0,
                label=r"DEEP-GPCM $\hat{s}_t$")
        ax.plot(t, theta_deep_disp, color=COLORS["DEEP-GPCM"],    lw=1.2, ls=":",
                alpha=0.75, label=r"DEEP-GPCM $\theta_t$")
        ax.plot(t, er_sm,           color=COLORS["DKVMN+Softmax"], lw=1.6, ls="--",
                alpha=0.6, label=r"DKVMN+Softmax $\hat{s}_t$")
        ax.plot(t, theta_dyn_disp,  color=COLORS["Dynamic GPCM"],  lw=1.6,
                alpha=0.6, label=r"Dynamic GPCM $\theta_t$")

        ax.set_title(r"{} ($\theta^*={:+.2f}$, $T={}$)".format(label, theta_true[idx], T),
                     fontweight="bold", fontsize=8, pad=3)
        ax.set_xlabel(r"Step $t$", fontsize=7.5, labelpad=2)
        ax.set_ylabel(r"$\hat{s}_t \;/\; \theta_t$ (rescaled)", fontsize=7.5, labelpad=2)
        ax.set_ylim(-0.1, K - 0.9)
        ax.grid(True, alpha=0.2, lw=0.5)
        ax.tick_params(labelsize=7, pad=2)

    # Shared legend below all subplots — avoids covering plot content
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="black",                  lw=1.5, ls="--",
               label=r"True $E[r|\theta^*]$"),
        Line2D([0], [0], color=COLORS["DEEP-GPCM"],      lw=2.0,
               label=r"DEEP-GPCM $\hat{s}_t$"),
        Line2D([0], [0], color=COLORS["DEEP-GPCM"],      lw=1.2, ls=":", alpha=0.75,
               label=r"DEEP-GPCM $\theta_t$"),
        Line2D([0], [0], color=COLORS["DKVMN+Softmax"],  lw=1.6, ls="--", alpha=0.6,
               label=r"DKVMN+Softmax $\hat{s}_t$"),
        Line2D([0], [0], color=COLORS["Dynamic GPCM"],   lw=1.6, alpha=0.6,
               label=r"Dynamic GPCM $\theta_t$"),
    ]
    Q = cfg.model.n_questions
    fig.suptitle(rf"Learner State Trajectories ($K={K}$, $Q={Q}$)",
                 fontsize=9, y=0.98)
    fig.legend(handles=legend_elements, loc="upper center", ncol=3, fontsize=7,
               bbox_to_anchor=(0.5, 0.92), framealpha=0.92, edgecolor="lightgray",
               handlelength=1.4, columnspacing=0.9)
    fig.tight_layout(rect=[0, 0, 1, 0.84], pad=0.2)

    out_path = out_dir / "learner_trajectories.pgf"
    fig.savefig(out_path)  # no bbox_inches="tight" — keeps figure at native width
    plt.close(fig)
    print(f"Figure saved: {out_path}")


if __name__ == "__main__":
    main()
