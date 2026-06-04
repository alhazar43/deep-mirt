"""Build the remaining v2 headline plots (M4-RL).

Plots produced.
  rl/results/v2/plots/m4rl_delta_j_distribution.png
  rl/results/v2/plots/m4rl_theta_recovery.png
  rl/results/v2/plots/m4rl_v1_vs_v2_baselines.png
  rl/results/v2/plots/m4rl_response_distribution.png

The baselines bar plot itself is emitted by ``eval_v2_baselines.py``.

Run.
  PYTHONPATH=ma-irt KMP_DUPLICATE_LIB_OK=TRUE \\
    python rl/scripts/build_v2_plots.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
V2_DEV_DIR = REPO / "rl" / "data" / "v2_dev"
V2_PLOT_DIR = REPO / "rl" / "results" / "v2" / "plots"
V2_DATA_DIR = REPO / "rl" / "results" / "v2" / "data"
V1_DATA_DIR = REPO / "rl" / "results" / "v1" / "data"


def _load(path: Path):
    with path.open() as fh:
        return json.load(fh)


def plot_delta_j_distribution() -> Path:
    jobs = _load(V2_DEV_DIR / "jobs.json")
    delta_j = np.asarray([j["delta_j"] for j in jobs], dtype=np.float64)
    meta = _load(V2_DEV_DIR / "oracle_metadata.json")
    stats = meta["delta_j_stats"]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.hist(delta_j, bins=40, color="#1f77b4", alpha=0.85, edgecolor="black", linewidth=0.4)
    ax.set_xlabel(r"$\delta_j$ (z-scored composite difficulty)")
    ax.set_ylabel("count")
    ax.set_title(
        f"v2 continuous $\\delta_j$ across {len(delta_j)} jobs\n"
        f"mean={stats['mean']:+.2f}, std={stats['std']:.2f}, "
        f"n_unique={stats['n_unique']}/{len(delta_j)}"
    )
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = V2_PLOT_DIR / "m4rl_delta_j_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_theta_recovery() -> Path:
    rec = _load(V2_DATA_DIR / "m4rl_theta_recovery.json")
    theta_true = np.asarray(rec["theta_true"])
    theta_hat = np.asarray(rec["theta_hat"])
    r = float(rec["pearson_r"])
    rmse = float(rec["rmse"])

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(theta_true, theta_hat, s=6, alpha=0.5, color="#ff7f0e", edgecolor="none")
    lo = float(min(theta_true.min(), theta_hat.min()))
    hi = float(max(theta_true.max(), theta_hat.max()))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xlabel(r"$\theta_{\text{true}}$")
    ax.set_ylabel(r"$\hat{\theta}$ (EAP)")
    ax.set_title(
        f"v2 theta recovery on user GPCM responses\n"
        f"N={len(theta_true)}, Pearson r = {r:.3f}, RMSE = {rmse:.3f}"
    )
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = V2_PLOT_DIR / "m4rl_theta_recovery.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_v1_vs_v2_baselines() -> Path:
    v1 = _load(V1_DATA_DIR / "m23_baselines.json")
    v2 = _load(V2_DATA_DIR / "m4rl_baselines.json")

    name_order = ["random", "popularity", "theta_hat", "theta_true"]
    pretty = {
        "random": "Random",
        "popularity": "Popularity",
        "theta_hat": r"1D ($\hat{\theta}$)",
        "theta_true": r"1D ($\theta$ oracle)",
    }
    v1_means = [v1["baselines"][n]["hit@10"]["mean"] for n in name_order]
    v1_lo = [v1["baselines"][n]["hit@10"]["mean"] - v1["baselines"][n]["hit@10"]["ci_lo"] for n in name_order]
    v1_hi = [v1["baselines"][n]["hit@10"]["ci_hi"] - v1["baselines"][n]["hit@10"]["mean"] for n in name_order]
    v2_means = [v2["baselines"][n]["hit@10"]["mean"] for n in name_order]
    v2_lo = [v2["baselines"][n]["hit@10"]["mean"] - v2["baselines"][n]["hit@10"]["ci_lo"] for n in name_order]
    v2_hi = [v2["baselines"][n]["hit@10"]["ci_hi"] - v2["baselines"][n]["hit@10"]["mean"] for n in name_order]

    x = np.arange(len(name_order))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(
        x - width / 2, v1_means, width, yerr=[v1_lo, v1_hi], capsize=3,
        label=f"v1 (n_eval={v1['n_eval_users']})", color="#7f7f7f",
        alpha=0.85, edgecolor="black", linewidth=0.5,
    )
    ax.bar(
        x + width / 2, v2_means, width, yerr=[v2_lo, v2_hi], capsize=3,
        label=f"v2 (n_eval={v2['n_eval_users']})", color="#2ca02c",
        alpha=0.85, edgecolor="black", linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([pretty[n] for n in name_order])
    ax.set_ylabel("Hit@10")
    ax.set_title("v1 vs v2 baselines, Hit@10 with 95% bootstrap CI")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper left")
    for i, (m1, m2) in enumerate(zip(v1_means, v2_means)):
        ax.annotate(f"{m1:.3f}", xy=(i - width / 2, m1), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=8)
        ax.annotate(f"{m2:.3f}", xy=(i + width / 2, m2), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=8)
    fig.tight_layout()
    out = V2_PLOT_DIR / "m4rl_v1_vs_v2_baselines.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_response_distribution() -> Path:
    meta = _load(V2_DEV_DIR / "oracle_metadata.json")
    counts = np.asarray(meta["response_stats"]["category_counts"], dtype=np.int64)
    n_total = int(counts.sum())
    K = len(counts)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    x = np.arange(K)
    bars = ax.bar(
        x, counts, color="#9467bd", alpha=0.85,
        edgecolor="black", linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(K)])
    ax.set_xlabel("ordinal response y")
    ax.set_ylabel("count")
    ax.set_title(
        f"v2 K=5 GPCM response distribution\n"
        f"n_responses = {n_total}, IsLiked rate = "
        f"{meta['response_stats']['is_liked_rate']:.3f} (y >= 3)"
    )
    ax.grid(alpha=0.3, axis="y")
    for b, c in zip(bars, counts):
        frac = c / n_total if n_total > 0 else 0.0
        ax.annotate(
            f"{int(c):,}\n({frac:.2%})",
            xy=(b.get_x() + b.get_width() / 2, b.get_height()),
            xytext=(0, 4), textcoords="offset points",
            ha="center", fontsize=9,
        )
    fig.tight_layout()
    out = V2_PLOT_DIR / "m4rl_response_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    V2_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    V2_DATA_DIR.mkdir(parents=True, exist_ok=True)
    outs = [
        plot_delta_j_distribution(),
        plot_theta_recovery(),
        plot_v1_vs_v2_baselines(),
        plot_response_distribution(),
    ]
    for p in outs:
        print(f"[m4rl-plots] wrote {p}")


if __name__ == "__main__":
    main()
