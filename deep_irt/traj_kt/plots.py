"""plots.py -- E2 result plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_rate_distribution(r_hat: np.ndarray, out_dir: Path) -> None:
    r_ok = r_hat[np.isfinite(r_hat)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(r_ok, bins=60, edgecolor="none", color="steelblue", alpha=0.8)
    ax.set_xlabel("Fitted learning rate r_hat")
    ax.set_ylabel("Learner count")
    ax.set_title("Distribution of recovered learning rates (E2, EdNet)")
    ax.axvline(np.median(r_ok), color="firebrick", linestyle="--",
               label=f"median={np.median(r_ok):.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "e2_rate_distribution.png", dpi=150)
    plt.close(fig)


def save_predictive_scatter(
    r_hat_first: np.ndarray,
    delta_acc: np.ndarray,
    rho: float,
    out_dir: Path,
) -> None:
    ok = np.isfinite(r_hat_first) & np.isfinite(delta_acc)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(r_hat_first[ok], delta_acc[ok],
               alpha=0.2, s=8, color="steelblue", rasterized=True)
    ax.set_xlabel("r_hat (first half)")
    ax.set_ylabel("delta_acc (2nd half - 1st half)")
    ax.set_title(f"Predictive validity  rho={rho:.3f}  n={ok.sum()}")
    fig.tight_layout()
    fig.savefig(out_dir / "e2_predictive_scatter.png", dpi=150)
    plt.close(fig)


def save_afm_scatter(
    afm_slopes: np.ndarray,
    r_hat: np.ndarray,
    rho: float,
    out_dir: Path,
) -> None:
    ok = np.isfinite(afm_slopes) & np.isfinite(r_hat)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(afm_slopes[ok], r_hat[ok],
               alpha=0.2, s=8, color="darkorange", rasterized=True)
    ax.set_xlabel("AFM slope")
    ax.set_ylabel("r_hat (aligned theta)")
    ax.set_title(f"AFM concurrent validity  rho={rho:.3f}  n={ok.sum()}")
    fig.tight_layout()
    fig.savefig(out_dir / "e2_afm_scatter.png", dpi=150)
    plt.close(fig)
