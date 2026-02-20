"""Plot learner state trajectories from a trained kt-gpcm DeepGPCM checkpoint.

Shows θ_t (DEEP-GPCM IRT ability) and E[r_t] = Σ k·p_k for four student
archetypes alongside the ground-truth E[r | θ*, α_ref, β_ref] reference line.

Usage (from repo root):
    PYTHONPATH=kt-gpcm/src python kt-gpcm/scripts/plot_learner_trajectories.py \\
        --config  kt-gpcm/configs/large_q5000_static.yaml \\
        --checkpoint kt-gpcm/outputs/large_q5000_static/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parents[2]
_src = _repo_root / "kt-gpcm" / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from kt_gpcm.config.loader import load_config
from kt_gpcm.models.kt_gpcm import DeepGPCM


# ---------------------------------------------------------------------------
# GPCM helpers (replicate data_gen formula for true E[r])
# ---------------------------------------------------------------------------

def gpcm_prob_true(theta: float, alpha: float, beta: np.ndarray) -> np.ndarray:
    """P(Y=k | theta, alpha, beta) for 1-trait GPCM.

    Formula (matches GPCMLogits.forward):
        eta   = alpha * theta
        phi_k = sum_{h=1}^{k} (eta - |alpha| * beta_h)   phi_0 = 0
        P(k)  = exp(phi_k) / sum_j exp(phi_j)
    """
    K = len(beta) + 1
    eta = float(alpha) * float(theta)
    alpha_norm = abs(float(alpha))
    cum_logits = np.zeros(K)
    for k in range(1, K):
        cum_logits[k] = np.sum(eta - alpha_norm * beta[:k])
    cum_logits -= cum_logits.max()  # numerical stability
    exp_l = np.exp(cum_logits)
    return exp_l / exp_l.sum()


def expected_score(probs: np.ndarray) -> float:
    """E[r] = sum_k k * p_k."""
    return float(np.dot(np.arange(len(probs), dtype=float), probs))


def smooth(x: np.ndarray, w: int = 9) -> np.ndarray:
    """Rolling mean, length-preserving."""
    out = np.convolve(x, np.ones(w) / w, mode="same")
    for i in range(w // 2):
        out[i] = x[: 2 * i + 1].mean()
        out[-(i + 1)] = x[-(2 * i + 1) :].mean()
    return out


# ---------------------------------------------------------------------------
# Student selection
# ---------------------------------------------------------------------------

def select_students(
    theta_true: np.ndarray,           # (N,)
    questions_all: List[List[int]],
    responses_all: List[List[int]],
) -> Dict[str, int]:
    """Select four representative student indices by signed ability."""
    n = len(theta_true)
    sorted_idx = np.argsort(theta_true)

    p5 = max(1, int(0.05 * n))
    high_idx = int(sorted_idx[-1])                          # highest ability
    low_idx  = int(sorted_idx[0])                           # lowest ability

    # Mid-ability with longest sequence
    mid_long_idx: Optional[int] = None
    for thr in [0.3, 0.6, 1.2]:
        cands = [i for i in range(n)
                 if abs(theta_true[i]) < thr and len(questions_all[i]) > 80]
        if cands:
            mid_long_idx = int(max(cands, key=lambda i: len(questions_all[i])))
            break
    if mid_long_idx is None:
        mid_long_idx = int(sorted_idx[n // 2])

    # Ambiguous: mid-ability (40-60 pct) with highest response variance
    lo40, hi60 = np.percentile(theta_true, [40, 60])
    mid_pool = [i for i in range(n) if lo40 <= theta_true[i] <= hi60]
    if not mid_pool:
        mid_pool = list(range(n))
    resp_vars = [float(np.var(responses_all[i])) for i in mid_pool]
    ambig_idx = int(mid_pool[int(np.argmax(resp_vars))])

    return {
        "High-ability":  high_idx,
        "Low-ability":   low_idx,
        "Mid-ability":   mid_long_idx,
        "Ambiguous":     ambig_idx,
    }


def select_reference_item(
    alpha_true: np.ndarray,  # (J,)
    beta_true: np.ndarray,   # (J, K-1)
) -> int:
    """Item with alpha near mean and beta_0 nearest 0 — most 'typical' item."""
    mean_alpha = float(np.mean(alpha_true))
    for tol in [0.15, 0.30, 1.0]:
        mask = np.abs(alpha_true - mean_alpha) < tol
        if mask.any():
            cands = np.where(mask)[0]
            break
    best = int(cands[np.argmin(np.abs(beta_true[cands, 0]))])
    return best


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_student(
    model: DeepGPCM,
    questions: List[int],
    responses: List[int],
    device: torch.device,
):
    """Single forward pass.  Returns theta_seq (T,) and probs_seq (T, K)."""
    model.eval()
    q = torch.tensor([questions], dtype=torch.long, device=device)  # (1, T)
    r = torch.tensor([responses], dtype=torch.long, device=device)  # (1, T)
    out = model(q, r)
    theta = out["theta"][0, :, 0].cpu().numpy()  # (T,)  — D=1, squeeze last dim
    probs = out["probs"][0].cpu().numpy()         # (T, K)
    return theta, probs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str, checkpoint_path: str, output_dir: Optional[str] = None):
    # ------------------------------------------------------------------
    # 1. Config
    # ------------------------------------------------------------------
    cfg = load_config(config_path)
    device = torch.device("cpu")

    out_dir = Path(output_dir) if output_dir else (
        Path(cfg.base.experiment_name and f"kt-gpcm/outputs/{cfg.base.experiment_name}")
        / "trajectory_plots"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    K = cfg.model.n_categories

    # ------------------------------------------------------------------
    # 2. Data
    # ------------------------------------------------------------------
    # Resolve data_dir relative to the config file if it is a relative path
    _data_root = Path(cfg.data.data_dir)
    if not _data_root.is_absolute():
        _data_root = Path(config_path).parent.parent / _data_root
    data_dir = _data_root / cfg.data.dataset_name
    print(f"Loading data from {data_dir}")

    with (data_dir / "sequences.json").open() as f:
        seqs = json.load(f)
    with (data_dir / "true_irt_parameters.json").open() as f:
        params = json.load(f)

    questions_all = [s["questions"] for s in seqs]
    responses_all = [s["responses"] for s in seqs]
    theta_true = np.array(params["theta"])   # (N,)  — 1D for D=1
    alpha_true = np.array(params["alpha"])   # (J,)
    beta_true  = np.array(params["beta"])    # (J, K-1)

    print(f"  {len(seqs)} students, theta range [{theta_true.min():.2f}, {theta_true.max():.2f}]")

    # ------------------------------------------------------------------
    # 3. Model
    # ------------------------------------------------------------------
    model = DeepGPCM(**vars(cfg.model)).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')}): {checkpoint_path}")

    # ------------------------------------------------------------------
    # 4. Student selection + reference item
    # ------------------------------------------------------------------
    selected = select_students(theta_true, questions_all, responses_all)
    ref_item = select_reference_item(alpha_true, beta_true)
    alpha_ref = float(alpha_true[ref_item])
    beta_ref  = beta_true[ref_item]          # (K-1,)

    print(f"\nReference item {ref_item}: alpha={alpha_ref:.3f}  beta_0={beta_ref[0]:.3f}")
    print("\nSelected students:")
    for label, idx in selected.items():
        T = len(questions_all[idx])
        print(f"  {label:15s}: student {idx:4d}  θ*={theta_true[idx]:+.3f}  T={T}")

    # True E[r] per student at reference item
    true_er: Dict[str, float] = {
        label: expected_score(gpcm_prob_true(theta_true[idx], alpha_ref, beta_ref))
        for label, idx in selected.items()
    }

    # ------------------------------------------------------------------
    # 5. Inference + expected score
    # ------------------------------------------------------------------
    k_vals = np.arange(K, dtype=float)
    student_data: Dict[str, dict] = {}

    for label, idx in selected.items():
        theta_seq, probs_seq = run_student(
            model, questions_all[idx], responses_all[idx], device
        )
        er_seq = probs_seq @ k_vals   # (T,)

        # Rescale θ_t to [0, K-1] for visual overlay:
        # IRT θ ~ N(0,1), map [-3, 3] → [0, K-1]
        theta_display = np.clip((theta_seq + 3.0) / 6.0 * (K - 1), 0, K - 1)

        student_data[label] = {
            "er":            smooth(er_seq),
            "theta_display": smooth(theta_display),
            "true_er":       true_er[label],
            "theta_true":    float(theta_true[idx]),
            "T":             len(questions_all[idx]),
        }

    # ------------------------------------------------------------------
    # 6. Plot
    # ------------------------------------------------------------------
    labels_order = ["High-ability", "Low-ability", "Mid-ability", "Ambiguous"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=False)

    for ax, label in zip(axes.flat, labels_order):
        d = student_data[label]
        T = d["T"]
        t = np.arange(T)

        ax.plot(t, d["er"],
                color="tab:blue", lw=1.8,
                label=r"DEEP-GPCM $\hat{s}_t = \Sigma\,k\,\hat{p}_k$")
        ax.plot(t, d["theta_display"],
                color="tab:orange", lw=1.4, ls="--",
                label=r"DEEP-GPCM $\theta_t$ (rescaled to $[0,K{-}1]$)")
        ax.axhline(d["true_er"],
                   color="black", lw=1.5, ls="--",
                   label=f"True $E[r\\,|\\,\\theta^*={d['theta_true']:+.2f}]$ = {d['true_er']:.2f}")

        ax.set_title(f"{label}  (θ*={d['theta_true']:+.2f}, T={T})",
                     fontweight="bold", fontsize=10)
        ax.set_xlabel("Interaction step t", fontsize=9)
        ax.set_ylabel("Expected response [0, K−1]", fontsize=9)
        ax.set_ylim(-0.1, K - 0.9)
        ax.grid(True, alpha=0.25, lw=0.7)
        ax.tick_params(labelsize=8)

    axes.flat[0].legend(fontsize=7.5, loc="lower right")

    fig.suptitle(
        f"Learner state trajectories — {cfg.data.dataset_name}  (K={K}, D={cfg.model.n_traits})\n"
        f"Reference item {ref_item}: α={alpha_ref:.3f}, β₀={beta_ref[0]:.3f}",
        fontsize=11,
    )
    fig.tight_layout()

    out_path = out_dir / "learner_trajectories.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")

    # Save metadata
    meta = {
        "config": config_path,
        "checkpoint": checkpoint_path,
        "reference_item": ref_item,
        "alpha_ref": alpha_ref,
        "beta_ref": beta_ref.tolist(),
        "students": {
            label: {
                "index": idx,
                "theta_true": float(theta_true[idx]),
                "true_er": true_er[label],
                "seq_len": len(questions_all[idx]),
            }
            for label, idx in selected.items()
        },
    }
    with (out_dir / "metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved: {out_dir / 'metadata.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot learner state trajectories")
    parser.add_argument("--config",     required=True,  help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True,  help="Path to .pt checkpoint")
    parser.add_argument("--output-dir", default=None,   help="Override output directory")
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.output_dir)
