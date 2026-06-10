"""E4.7 plots: r_voi comparison, policy ranking bars, training curves.

Generates:
  rl/results/plots/e47_rvoi_compare.png
  rl/results/plots/e47_ranking_compare.png
  rl/results/plots/e47_training_curves.png

Usage::
    PYTHONPATH="rl/src;ma-irt" python rl/scripts/plot_e47.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path("C:/Users/steph/Documents/deep-mirt")
_WORKTREE = Path(__file__).parent.parent.parent.resolve()
_RESULTS = _WORKTREE / "rl/results"
_PLOTS = _RESULTS / "plots"
_PLOTS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load training metrics
# ---------------------------------------------------------------------------

def _load_metrics(path: Path) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            for k, v in row.items():
                out.setdefault(k, [])
                try:
                    out[k].append(float(v))
                except (ValueError, TypeError):
                    pass
    return out


# E4.6b static reported stats (no metrics.csv in this tree)
_STATIC_RVOI_MEAN = -0.043
_STATIC_RVOI_MIN = -0.146
_STATIC_RVOI_MAX = 0.066

stair_m = _load_metrics(_REPO / "rl/outputs/ordrec_synth_e47_stair/metrics.csv")
rw_m = _load_metrics(_REPO / "rl/outputs/ordrec_synth_e47_rw/metrics.csv")

# Detailed eval results
with (_WORKTREE / "rl/results/E47_eval_detailed.json").open("r") as f:
    eval_results = json.load(f)


def smooth(arr: List[float], window: int = 20) -> np.ndarray:
    a = np.array(arr, dtype=np.float64)
    if len(a) < window:
        return a
    kernel = np.ones(window) / window
    return np.convolve(a, kernel, mode="valid")


# ---------------------------------------------------------------------------
# Plot 1: r_voi training traces
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axes[0]
stair_rv = stair_m.get("r_voi", [])
rw_rv = rw_m.get("r_voi", [])

ax.axhline(_STATIC_RVOI_MEAN, color="#888888", lw=2.0, linestyle="--",
           label=f"static E4.6b (mean {_STATIC_RVOI_MEAN:+.3f}, range [{_STATIC_RVOI_MIN:.3f}, {_STATIC_RVOI_MAX:.3f}])")
ax.axhspan(_STATIC_RVOI_MIN, _STATIC_RVOI_MAX, color="#888888", alpha=0.08)

x_s = np.arange(len(stair_rv))
ax.plot(x_s, stair_rv, color="#1f77b4", alpha=0.2, lw=0.8)
sm_s = smooth(stair_rv)
ax.plot(x_s[len(x_s)-len(sm_s):], sm_s, color="#1f77b4", lw=2.0,
        label=f"staircase (mean +{np.mean(stair_rv):.3f}, 100% positive)")

x_r = np.arange(len(rw_rv))
ax.plot(x_r, rw_rv, color="#ff7f0e", alpha=0.2, lw=0.8)
sm_r = smooth(rw_rv)
ax.plot(x_r[len(x_r)-len(sm_r):], sm_r, color="#ff7f0e", lw=2.0,
        label=f"random-walk (mean +{np.mean(rw_rv):.3f}, 100% positive)")

ax.axhline(0, color="black", lw=0.8, linestyle=":")
ax.set_xlabel("PPO update")
ax.set_ylabel("r_voi (per training update)")
ax.set_title("r_voi during training: dynamic vs. static DGP")
ax.legend(fontsize=8.5)
ax.set_ylim(-0.18, 0.20)

# Right panel: episode return
ax2 = axes[1]
stair_ret = stair_m.get("mean_return", [])
rw_ret = rw_m.get("mean_return", [])
ax2.axhline(-0.537, color="#888888", lw=1.5, linestyle="--", label="static random (-0.537)")
ax2.axhline(-0.570, color="#888888", lw=1.5, linestyle=":", label="static PPO (-0.570)")
ax2.plot(np.arange(len(stair_ret)), stair_ret, color="#1f77b4", alpha=0.2, lw=0.8)
sm_sr = smooth(stair_ret)
ax2.plot(np.arange(len(stair_ret))[len(stair_ret)-len(sm_sr):], sm_sr, color="#1f77b4", lw=2.0,
         label=f"staircase PPO (best -0.243)")
ax2.plot(np.arange(len(rw_ret)), rw_ret, color="#ff7f0e", alpha=0.2, lw=0.8)
sm_rr = smooth(rw_ret)
ax2.plot(np.arange(len(rw_ret))[len(rw_ret)-len(sm_rr):], sm_rr, color="#ff7f0e", lw=2.0,
         label=f"randomwalk PPO (best -0.222)")
ax2.set_xlabel("PPO update")
ax2.set_ylabel("mean episode return")
ax2.set_title("Training return vs. static baseline")
ax2.legend(fontsize=8.5)

fig.suptitle("E4.7: Dynamic DGP training dynamics", fontsize=13)
fig.tight_layout()
fig.savefig(_PLOTS / "e47_rvoi_compare.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved e47_rvoi_compare.png")


# ---------------------------------------------------------------------------
# Plot 2: policy ranking grouped bars
# ---------------------------------------------------------------------------

POLICIES = ["ppo", "bc", "fisher", "random"]
POLICY_LABELS = ["PPO", "BC-only", "max-Fisher", "uniform random"]
COHORTS = ["static_e46b", "staircase", "randomwalk"]
COHORT_LABELS = ["static (E4.6b)", "staircase", "random-walk"]

static_returns = {
    "ppo":    {"mean_return": -0.5699, "ci_low": -0.5797, "ci_high": -0.5601},
    "bc":     {"mean_return": -0.5609, "ci_low": -0.5695, "ci_high": -0.5523},
    "fisher": {"mean_return": -0.5638, "ci_low": -0.5722, "ci_high": -0.5555},
    "random": {"mean_return": -0.5368, "ci_low": -0.5437, "ci_high": -0.5299},
}

colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
bar_w = 0.18
group_gap = 0.9
x = np.arange(len(COHORTS)) * group_gap

fig, ax = plt.subplots(figsize=(12, 5))
for pi, (pol, label, color) in enumerate(zip(POLICIES, POLICY_LABELS, colors)):
    xpos = x + (pi - (len(POLICIES) - 1) / 2) * bar_w
    means, errs_l, errs_h = [], [], []
    for cohort in COHORTS:
        d = static_returns[pol] if cohort == "static_e46b" else eval_results[cohort][pol]
        m = d["mean_return"]
        means.append(m)
        errs_l.append(m - d["ci_low"])
        errs_h.append(d["ci_high"] - m)
    ax.bar(xpos, means, bar_w, label=label, color=color, alpha=0.85,
           yerr=[errs_l, errs_h], capsize=3, error_kw={"elinewidth": 1.2})

ax.set_xticks(x)
ax.set_xticklabels(COHORT_LABELS, fontsize=11)
ax.set_ylabel("mean episode return (95% CI bars)")
ax.set_title("E4.7 policy ranking: static vs. staircase vs. random-walk\n(higher is better; RC3 test: does ranking flip under drift?)")
ax.axhline(0, color="black", lw=0.5, linestyle=":")
ax.legend(fontsize=9, loc="lower right")

# Add ordering annotation
for xi, cohort in zip(x, COHORTS):
    if cohort == "static_e46b":
        order = "random>BC>Fisher>PPO"
    else:
        order = "PPO>BC>Fisher>random"
    ax.text(xi, -0.62, order, ha="center", va="top", fontsize=7.5, color="#333333")

fig.tight_layout()
fig.savefig(_PLOTS / "e47_ranking_compare.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved e47_ranking_compare.png")


# ---------------------------------------------------------------------------
# Plot 3: training curves (return + r_voi per cohort)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(13, 7))

for row_i, (cohort_label, rv_data, ret_data) in enumerate([
    ("Staircase", stair_rv, stair_ret),
    ("Random-walk", rw_rv, rw_ret),
]):
    ax_r = axes[row_i, 0]
    ax_v = axes[row_i, 1]

    ax_r.plot(ret_data, color="#1f77b4", alpha=0.25, lw=0.8)
    sm = smooth(ret_data)
    ax_r.plot(np.arange(len(ret_data))[len(ret_data)-len(sm):], sm, color="#1f77b4", lw=2.0)
    ax_r.axhline(-0.537, color="gray", lw=1.0, linestyle="--", label="static random (-0.537)")
    ax_r.axhline(-0.570, color="gray", lw=1.0, linestyle=":", label="static PPO (-0.570)")
    ax_r.set_ylabel("mean return")
    ax_r.set_title(f"{cohort_label}: episode return")
    ax_r.legend(fontsize=8)

    ax_v.plot(rv_data, color="#d62728", alpha=0.25, lw=0.8)
    sm_v = smooth(rv_data)
    ax_v.plot(np.arange(len(rv_data))[len(rv_data)-len(sm_v):], sm_v, color="#d62728", lw=2.0, label=f"mean +{np.mean(rv_data):.3f}")
    ax_v.axhline(0, color="black", lw=0.8, linestyle=":")
    ax_v.axhline(_STATIC_RVOI_MEAN, color="gray", lw=1.0, linestyle="--", label=f"static mean {_STATIC_RVOI_MEAN:+.3f}")
    ax_v.set_ylabel("r_voi")
    ax_v.set_title(f"{cohort_label}: terminal VOI reward")
    ax_v.legend(fontsize=8)

    for ax in [ax_r, ax_v]:
        ax.set_xlabel("PPO update")

fig.suptitle("E4.7 training curves: dynamic DGP cohorts", y=1.01, fontsize=13)
fig.tight_layout()
fig.savefig(_PLOTS / "e47_training_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved e47_training_curves.png")
print(f"\nAll plots saved to {_PLOTS}/")
