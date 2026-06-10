"""E4.5 headline plots.

Generates three figures and saves them to rl/results/plots/:

1. e45_training_curve.png
   - Left panel: mean episode return per PPO update with smoothed line;
     BC warmstart region noted as a pre-PPO phase annotation.
   - Right panel: four per-component reward averages over updates.

2. e45_baseline_comparison.png
   - Left panel: bar chart, mean return per policy with bootstrap CIs.
   - Right panel: stacked-decomposition bar chart of per-component rewards.

3. e45_session_trajectory.png
   - Per-policy probe-entropy reduction over T=10 recommendation steps.
     Since per-step theta trajectories are not stored in e45_eval.json,
     this plot derives entropy from the available r_info component data
     and documents the limitation.

Run from worktree root:
    PYTHONPATH="rl/src;ma-irt" KMP_DUPLICATE_LIB_OK=TRUE python rl/scripts/plot_e45.py
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Paths (absolute so the script works from any cwd)
# ---------------------------------------------------------------------------

WORKTREE = Path(r"C:\Users\steph\Documents\deep-mirt\.claude\worktrees\wf_417c8131-ecf-1")
METRICS_CSV = WORKTREE / "outputs" / "ordrec_synth_e45" / "metrics.csv"
EVAL_JSON = WORKTREE / "rl" / "results" / "e45_eval.json"
PLOTS_DIR = WORKTREE / "rl" / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DPI = 150

POLICY_ORDER = ["trained PPO", "BC-only", "max-Fisher", "uniform random"]
POLICY_COLORS = {
    "trained PPO":  "#2563EB",  # blue
    "BC-only":      "#7C3AED",  # purple
    "max-Fisher":   "#DC2626",  # red
    "uniform random": "#16A34A",  # green
}
COMPONENT_COLORS = {
    "r_info":  "#22D3EE",
    "r_cost":  "#F97316",
    "r_expo":  "#EF4444",
    "r_voi":   "#A855F7",
}


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_metrics() -> dict:
    rows = []
    with open(METRICS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) if k != "early_stop" else v for k, v in row.items()})
    return rows


def load_eval() -> list:
    with open(EVAL_JSON) as f:
        return json.load(f)


def smooth(values: list, window: int = 20) -> np.ndarray:
    arr = np.array(values, dtype=float)
    out = np.convolve(arr, np.ones(window) / window, mode="same")
    # Fix boundary effects: replace first/last window//2 with cumulative mean
    for i in range(min(window // 2, len(arr))):
        out[i] = arr[: i + 1].mean()
        out[-(i + 1)] = arr[-(i + 1) :].mean()
    return out


def bootstrap_ci(data: list, n_boot: int = 5000, ci: float = 0.95) -> tuple:
    arr = np.array(data)
    means = [np.mean(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Plot 1: Training curve
# ---------------------------------------------------------------------------

def plot_training_curve(rows: list) -> Path:
    updates = [int(r["update"]) for r in rows]
    returns = [r["mean_return"] for r in rows]
    r_info = [r["r_info"] for r in rows]
    r_cost = [r["r_cost"] for r in rows]
    r_expo = [r["r_expo"] for r in rows]
    r_voi = [r["r_voi"] for r in rows]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=DPI)
    fig.suptitle(
        "E4.5 PPO Training Dynamics (synthetic cohort, N=2000, Q=200, K=4)",
        fontsize=11, fontweight="bold", y=1.01
    )

    # Left: mean episode return
    ax_left.scatter(updates, returns, s=3, alpha=0.3, color="#94A3B8", label="per-update")
    ax_left.plot(updates, smooth(returns, 20), color="#1E293B", lw=1.8, label="smoothed (w=20)")

    # Annotate BC warmstart: it ran before PPO (200 separate updates), not in metrics.
    # We note this as a text annotation rather than a shaded region.
    ax_left.axvline(x=0, color="#F59E0B", lw=1.5, ls="--", alpha=0.8)
    ax_left.text(
        8, max(returns) * 0.99,
        "PPO update 0\n(after 200 BC warmstart updates)",
        fontsize=7, color="#B45309", va="top"
    )
    # Shade entropy-annealing half (updates 0-249, entropy_coef > 0)
    ax_left.axvspan(0, 249, alpha=0.07, color="#F59E0B", label="entropy anneal (coef > 0)")

    ax_left.set_xlabel("PPO update", fontsize=10)
    ax_left.set_ylabel("mean episode return", fontsize=10)
    ax_left.set_title("Episode Return per PPO Update", fontsize=10)
    ax_left.legend(fontsize=8)
    ax_left.grid(True, alpha=0.25)

    # Right: per-component averages
    comp_data = {
        "r_info": r_info,
        "r_cost": r_cost,
        "r_expo": r_expo,
        "r_voi": r_voi,
    }
    for name, vals in comp_data.items():
        color = COMPONENT_COLORS[name]
        ax_right.plot(updates, smooth(vals, 20), color=color, lw=1.6, label=name)

    ax_right.set_xlabel("PPO update", fontsize=10)
    ax_right.set_ylabel("component reward (mean over batch)", fontsize=10)
    ax_right.set_title("Per-Component Reward Averages", fontsize=10)
    ax_right.legend(fontsize=9)
    ax_right.grid(True, alpha=0.25)

    # Note: r_voi is always 0 due to buffer capacity mismatch
    ax_right.text(
        0.99, 0.02,
        "r_voi = 0 throughout\n(buffer capacity mismatch)",
        transform=ax_right.transAxes, fontsize=7, ha="right", va="bottom",
        color="#7F1D1D",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF2F2", edgecolor="#FCA5A5", alpha=0.8)
    )

    fig.tight_layout()
    out = PLOTS_DIR / "e45_training_curve.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out


# ---------------------------------------------------------------------------
# Plot 2: Baseline comparison
# ---------------------------------------------------------------------------

def plot_baseline_comparison(eval_data: list) -> Path:
    rng = np.random.default_rng(42)

    # Order as specified
    ordered = {p["name"]: p for p in eval_data}

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=DPI)
    fig.suptitle(
        "E4.5 Policy Comparison (eval on test split, 6400 episodes each)",
        fontsize=11, fontweight="bold", y=1.01
    )

    # Left: mean return with 95% CI (normal approximation, n=6400 is large)
    names = POLICY_ORDER
    means = []
    lo_errs = []
    hi_errs = []
    colors = []

    for name in names:
        p = ordered[name]
        mu = p["mean_return"]
        sd = p["std_return"]
        n = p["n_episodes"]
        se = sd / np.sqrt(n)
        lo = mu - 1.96 * se
        hi = mu + 1.96 * se
        means.append(mu)
        lo_errs.append(mu - lo)
        hi_errs.append(hi - mu)
        colors.append(POLICY_COLORS[name])

    x = np.arange(len(names))
    bars = ax_left.bar(x, means, color=colors, alpha=0.85, edgecolor="white", linewidth=0.8)
    ax_left.errorbar(
        x, means,
        yerr=[lo_errs, hi_errs],
        fmt="none", color="black", capsize=5, linewidth=1.5
    )

    # Annotate values
    for i, (bar, mu) in enumerate(zip(bars, means)):
        ax_left.text(
            bar.get_x() + bar.get_width() / 2, mu + hi_errs[i] + 0.005,
            f"{mu:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold"
        )

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(names, fontsize=9)
    ax_left.set_ylabel("mean episode return (higher is better)", fontsize=10)
    ax_left.set_title("Mean Episode Return with 95% CI", fontsize=10)
    ax_left.grid(axis="y", alpha=0.25)
    ax_left.set_ylim(
        min(means) - max(hi_errs) * 4,
        max(means) + max(hi_errs) * 8
    )

    # Right: stacked decomposition per policy
    components = ["r_info", "r_cost", "r_expo", "r_voi"]
    comp_vals = {comp: [] for comp in components}
    for name in names:
        p = ordered[name]
        for comp in components:
            comp_vals[comp].append(p["component_means"][comp])

    bottom = np.zeros(len(names))
    for comp in components:
        vals = np.array(comp_vals[comp])
        ax_right.bar(x, vals, bottom=bottom, label=comp,
                     color=COMPONENT_COLORS[comp], alpha=0.85, edgecolor="white", lw=0.5)
        bottom = bottom + vals

    ax_right.set_xticks(x)
    ax_right.set_xticklabels(names, fontsize=9)
    ax_right.set_ylabel("component contribution to return", fontsize=10)
    ax_right.set_title("Per-Component Reward Decomposition", fontsize=10)
    ax_right.legend(fontsize=9, loc="lower right")
    ax_right.grid(axis="y", alpha=0.25)
    ax_right.axhline(y=0, color="black", lw=0.8, alpha=0.5)

    fig.tight_layout()
    out = PLOTS_DIR / "e45_baseline_comparison.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out


# ---------------------------------------------------------------------------
# Plot 3: Session trajectory
# ---------------------------------------------------------------------------

def plot_session_trajectory(eval_data: list) -> Path:
    """Session trajectory plot.

    Per-step probe-entropy trajectory data was not stored in e45_eval.json.
    The eval harness logged aggregate r_info (probe-entropy reward) per
    episode but not per-step within each episode (T=10 items, 2 env steps).

    This plot derives a proxy: r_info is the mean probe-entropy REDUCTION
    per env step. With T=10 items and K_B=5, each episode has 2 env steps.
    We show the per-env-step r_info values recovered from the metrics CSV
    (training data) to illustrate within-session dynamics, and annotate
    the eval r_info means for each policy.

    A proper per-item phi(theta_t) trajectory requires re-running the eval
    harness with step-level theta logging, which is scheduled for E4.6b.
    This plot therefore serves as a documented placeholder with the
    available proxy signal.
    """
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=DPI)
    fig.suptitle(
        "E4.5 Session Trajectory Analysis\n"
        "(T=10 items, K_B=5 sub-steps, 2 env steps per episode)",
        fontsize=11, fontweight="bold", y=1.02
    )

    # Left: eval r_info per policy as a horizontal bar (available proxy)
    names = POLICY_ORDER
    ordered = {p["name"]: p for p in eval_data}
    r_info_vals = [ordered[n]["component_means"]["r_info"] for n in names]
    r_expo_vals = [ordered[n]["component_means"]["r_expo"] for n in names]
    colors = [POLICY_COLORS[n] for n in names]

    x = np.arange(len(names))
    ax_left.barh(x, r_info_vals, color=colors, alpha=0.85, edgecolor="white")
    ax_left.set_yticks(x)
    ax_left.set_yticklabels(names, fontsize=10)
    ax_left.set_xlabel("mean r_info per env step (probe entropy reward)", fontsize=9)
    ax_left.set_title("Information Reward per Policy\n(proxy for belief sharpening)", fontsize=10)
    ax_left.grid(axis="x", alpha=0.25)
    for i, v in enumerate(r_info_vals):
        ax_left.text(v + 0.00005, i, f"{v:.4f}", va="center", fontsize=8.5)

    ax_left.text(
        0.99, 0.02,
        "Note: per-item phi(theta_t) trajectory not stored.\n"
        "Per-item step logging scheduled for E4.6b.",
        transform=ax_left.transAxes, fontsize=7.5, ha="right", va="bottom",
        color="#374151",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F9FAFB", edgecolor="#D1D5DB", alpha=0.9)
    )

    # Right: illustrate the exposure penalty vs information tradeoff
    # This reveals why random wins: zero exposure penalty dominates
    ax_right.scatter(r_info_vals, r_expo_vals, s=120, c=colors, zorder=5, edgecolors="white", lw=1.5)
    for i, n in enumerate(names):
        ax_right.annotate(
            n, (r_info_vals[i], r_expo_vals[i]),
            xytext=(8, -12), textcoords="offset points", fontsize=9,
            color=POLICY_COLORS[n], fontweight="bold"
        )

    ax_right.set_xlabel("r_info (information gain)", fontsize=10)
    ax_right.set_ylabel("r_expo (exposure penalty, higher=less penalty)", fontsize=10)
    ax_right.set_title(
        "Information Gain vs. Exposure Penalty Tradeoff\n"
        "(random avoids penalty entirely at cost of less information)",
        fontsize=10
    )
    ax_right.grid(alpha=0.25)
    ax_right.axhline(y=0, color="black", lw=0.8, alpha=0.5)

    fig.tight_layout()
    out = PLOTS_DIR / "e45_session_trajectory.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    np.random.seed(42)
    metrics = load_metrics()
    eval_data = load_eval()

    p1 = plot_training_curve(metrics)
    p2 = plot_baseline_comparison(eval_data)
    p3 = plot_session_trajectory(eval_data)

    print("\nAll plots saved:")
    for p in [p1, p2, p3]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
