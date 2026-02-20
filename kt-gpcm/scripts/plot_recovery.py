#!/usr/bin/env python3
"""IRT parameter recovery scatter plots.

Compares model-estimated parameters against the ground-truth values saved
by the synthetic data generator.  Only meaningful for synthetic datasets
that include ``true_irt_parameters.json``.

Usage::

    PYTHONPATH=src python scripts/plot_recovery.py \\
        --config   configs/smoke.yaml \\
        --checkpoint outputs/smoke/best.pt \\
        --output   outputs/smoke/recovery_plots

The script:
    1. Loads the trained model from ``--checkpoint``.
    2. Runs inference on the full dataset (train + test).
    3. Extracts item-level mean alpha and beta estimates.
    4. Loads ground-truth parameters from the dataset directory.
    5. Saves scatter plots for alpha (per trait dim) and beta (per threshold).
"""

from __future__ import annotations
import os; os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # Windows OpenMP conflict

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from kt_gpcm.config import load_config
from kt_gpcm.data.loaders import DataModule, SequenceDataset, collate_sequences
from kt_gpcm.models.kt_gpcm import DeepGPCM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_device(cfg_device: str) -> torch.device:
    if cfg_device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(cfg_device)


def build_model(cfg, device: torch.device) -> DeepGPCM:
    model = DeepGPCM(**vars(cfg.model))
    return model.to(device)


def link_alpha(vals: np.ndarray, target_std: float = 0.3) -> np.ndarray:
    """IRT linking for discrimination: z-score in log-space, rescale to target std.

    Equivalent to standard mean-sigma linking for lognormal(0, target_std) prior.
    Applied to both true and estimated alpha before plotting so both live on the
    same scale and the y=x diagonal is meaningful.
    """
    log_v = np.log(np.maximum(vals, 1e-6))
    std = log_v.std()
    if std < 1e-6:
        return np.ones_like(vals)
    return np.exp((log_v - log_v.mean()) / std * target_std)


def link_normal(vals: np.ndarray) -> np.ndarray:
    """IRT linking for theta / beta: z-score to N(0,1)."""
    std = vals.std()
    if std < 1e-6:
        return vals - vals.mean()
    return (vals - vals.mean()) / std


def scatter_plot(
    ax: plt.Axes,
    true_vals: np.ndarray,
    est_vals: np.ndarray,
    title: str,
    xlabel: str = "True",
    ylabel: str = "Estimated",
    color: str = "steelblue",
    link_fn=None,
) -> None:
    if link_fn is not None:
        true_vals = link_fn(true_vals)
        est_vals = link_fn(est_vals)
    r = float(np.corrcoef(true_vals, est_vals)[0, 1]) if len(true_vals) > 1 else float("nan")
    ax.scatter(true_vals, est_vals, alpha=0.5, s=18, color=color, edgecolors="none")
    lo = min(true_vals.min(), est_vals.min())
    hi = max(true_vals.max(), est_vals.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.0, label="y=x")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nr = {r:.3f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot IRT parameter recovery.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default=None,
                        help="Override data.dataset_name (reads n_questions/n_categories from metadata.json).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset:
        cfg.data.dataset_name = args.dataset
    device = resolve_device(cfg.base.device)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data (syncs n_questions/n_categories from metadata into cfg) --
    data_mgr = DataModule(cfg)
    train_loader, test_loader = data_mgr.build()

    # ---- Load model --------------------------------------------------------
    model = build_model(cfg, device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Load true IRT params
    irt_path = (
        Path(cfg.data.data_dir)
        / cfg.data.dataset_name
        / "true_irt_parameters.json"
    )
    if not irt_path.exists():
        print(f"No true_irt_parameters.json found at {irt_path}. Exiting.")
        return

    with irt_path.open("r", encoding="utf-8") as fh:
        true_irt = json.load(fh)

    true_alpha = np.array(true_irt["alpha"])   # (Q,)
    true_beta = np.array(true_irt["beta"])     # (Q, K-1)

    # ---- Run inference on ALL sequences ------------------------------------
    # Accumulate item-level estimates by averaging over appearances
    Q = cfg.model.n_questions
    D = cfg.model.n_traits
    K = cfg.model.n_categories

    alpha_sum = np.zeros((Q, D))
    alpha_count = np.zeros((Q,))
    beta_sum = np.zeros((Q, K - 1))
    beta_count = np.zeros((Q,))

    def process_loader(loader: DataLoader) -> None:
        for questions, responses, mask in loader:
            questions = questions.to(device)
            responses = responses.to(device)
            mask = mask.to(device)

            with torch.no_grad():
                out = model(questions, responses)

            alpha = out["alpha"].cpu().numpy()   # (B, S, D)
            beta = out["beta"].cpu().numpy()     # (B, S, K-1)
            q_np = questions.cpu().numpy()       # (B, S) — 1-based IDs
            m_np = mask.cpu().numpy()            # (B, S)

            B, S = q_np.shape
            for b in range(B):
                for t in range(S):
                    if m_np[b, t]:
                        qid = q_np[b, t] - 1   # back to 0-based
                        if 0 <= qid < Q:
                            alpha_sum[qid] += alpha[b, t]
                            alpha_count[qid] += 1
                            beta_sum[qid] += beta[b, t]
                            beta_count[qid] += 1

    process_loader(train_loader)
    process_loader(test_loader)

    # Compute per-item means (avoid division by zero)
    seen = alpha_count > 0
    alpha_est = np.where(seen[:, None], alpha_sum / np.maximum(alpha_count[:, None], 1), 0.0)
    beta_est = np.where(seen[:, None], beta_sum / np.maximum(beta_count[:, None], 1), 0.0)

    # ---- Plots --------------------------------------------------------------
    # Alpha recovery (one plot per trait dimension)
    for d in range(D):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        true_a = true_alpha[seen]
        est_a = alpha_est[seen, d]
        scatter_plot(ax, true_a, est_a, f"Alpha recovery (dim {d})",
                     xlabel="True alpha (linked)", ylabel="Estimated alpha (linked)",
                     color="steelblue", link_fn=link_alpha)
        plt.tight_layout()
        path = out_dir / f"alpha_dim{d}_recovery.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    # Beta recovery (one plot per threshold)
    for k in range(K - 1):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        true_b = true_beta[seen, k]
        est_b = beta_est[seen, k]
        scatter_plot(ax, true_b, est_b, f"Beta recovery (threshold {k})",
                     xlabel="True beta (linked)", ylabel="Estimated beta (linked)",
                     color="darkorange", link_fn=link_normal)
        plt.tight_layout()
        path = out_dir / f"beta_threshold{k}_recovery.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    print(f"\nAll recovery plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
