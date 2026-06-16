"""
make_figures.py -- Generate slide-ready result figures from deep_irt experiments.

Run from the repo root:
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=deep_irt python docs/slides/figures/make_figures.py

Outputs (in this same directory):
    fig_trajectory.pdf
    fig_drift_sweep.pdf
    fig_crossformat.pdf
    fig_slam_transfer.pdf
    fig_cost.pdf
    fig_multiround.pdf
"""

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_SUBSTRATE_DIR = os.path.join(_REPO_ROOT, "deep_irt")
OUT_DIR = _HERE

# deep_irt/ is a package; import as deep_irt.core.* requires repo root on path.
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 5.5, 3.8
FONT_SIZE = 13
TICK_SIZE = 12
LABEL_SIZE = 13
LEGEND_SIZE = 11

# Palette: accessible, clean
C_BLUE   = "#2166AC"
C_RED    = "#D6604D"
C_GREEN  = "#4DAC26"
C_ORANGE = "#F4A582"
C_GREY   = "#999999"
C_DARK   = "#333333"
C_SHADE  = "#C8D9EE"  # light blue shade for bands

matplotlib.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,       # embed fonts as Type1 (editable in Illustrator)
    "ps.fonttype": 42,
})

SAVE_KW = dict(bbox_inches="tight", dpi=150)


def savefig(name: str):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, **SAVE_KW)
    plt.close()
    print(f"  Saved: {path}")
    return path


# ===========================================================================
# Figure 1: Trajectory recovery (dynamic random-walk, 3 learners)
# ===========================================================================

def fig_trajectory():
    print("\n[1/6] fig_trajectory.pdf -- fitting dynamic model in-process ...")

    import torch
    import torch.nn as nn
    import torch.optim as optim

    # Import DeepIRTModel (repo root on sys.path -> deep_irt is a package)
    from deep_irt.core import DeepIRTModel

    SEED = 42
    torch.manual_seed(SEED)
    DEVICE = torch.device("cpu")

    # --- Self-contained data generator (mirrors run_pipeline logic) ---
    def gpcm_probs(theta, a, b):
        K = len(b) + 1
        psi = np.zeros(K)
        for k in range(1, K):
            psi[k] = np.sum(a * (theta - b[:k]))
        psi -= psi.max()
        e = np.exp(psi)
        return e / e.sum()

    def sample_resp(theta, a, b, rng):
        probs = gpcm_probs(theta, a, b)
        return int(rng.choice(len(probs), p=probs))

    rng = np.random.default_rng(SEED)
    N_LEARNERS = 300
    N_BASE = 40
    N_CATS = 4
    DRIFT_SIGMA = 0.20

    # Item parameters
    a_items = rng.lognormal(0.0, 0.3, size=N_BASE)
    b_items = np.sort(rng.standard_normal((N_BASE, N_CATS - 1)), axis=1)

    # Trajectories
    theta_0 = rng.standard_normal(N_LEARNERS)
    steps = rng.normal(0.0, DRIFT_SIGMA, size=(N_LEARNERS, N_BASE))
    theta_traj = np.zeros((N_LEARNERS, N_BASE))
    theta_traj[:, 0] = theta_0
    theta_traj[:, 1:] = theta_0[:, None] + np.cumsum(steps, axis=1)[:, :-1]

    # Build sequences
    seqs_items = []
    seqs_resps = []
    for i in range(N_LEARNERS):
        item_order = rng.permutation(N_BASE).tolist()
        resps = [sample_resp(theta_traj[i, t], a_items[j], b_items[j], rng)
                 for t, j in enumerate(item_order)]
        seqs_items.append(item_order)
        seqs_resps.append(resps)

    items_t = torch.tensor(seqs_items, dtype=torch.long)
    resps_t = torch.tensor(seqs_resps, dtype=torch.long)

    # Fit model (small/fast for a slide demo)
    model = DeepIRTModel(
        num_items=N_BASE, emb_dim=8, hidden_dim=32, n_cats=N_CATS,
        decoder="gpcm", device=DEVICE, seed=SEED,
    )
    print("    Training ... (300 epochs, N=300, B=40)")
    t0 = time.time()
    model.fit(items_t, resps_t, n_epochs=300, verbose=False)
    print(f"    Done in {time.time()-t0:.1f}s")

    # Recover trajectories
    theta_rec = model.track(items_t, resps_t).cpu().numpy()  # (N, T)

    # Pick 3 representative learners: low/mid/high net-drift magnitude
    net_drift = np.abs(theta_traj[:, -1] - theta_traj[:, 0])
    sorted_idx = np.argsort(net_drift)
    n = N_LEARNERS
    # one from bottom third, one from middle, one from top third
    learner_ids = [sorted_idx[n // 6], sorted_idx[n // 2], sorted_idx[5 * n // 6]]

    # Sign-align recovered trajectories (overall)
    from scipy.stats import pearsonr
    r, _ = pearsonr(theta_rec.ravel(), theta_traj.ravel())
    if r < 0:
        theta_rec = -theta_rec

    colors = [C_BLUE, C_GREEN, C_RED]
    steps_x = np.arange(1, N_BASE + 1)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for idx, (li, col) in enumerate(zip(learner_ids, colors)):
        true_traj = theta_traj[li]
        rec_traj  = theta_rec[li]
        # Mean-center each learner's recovered trajectory to match true scale visually
        rec_traj_sc = rec_traj - rec_traj.mean() + true_traj.mean()
        lbl_true = "True ability" if idx == 0 else None
        lbl_rec  = "Recovered"    if idx == 0 else None
        ax.plot(steps_x, true_traj, color=col, lw=2.0, label=lbl_true)
        ax.plot(steps_x, rec_traj_sc, color=col, lw=1.5, ls="--",
                alpha=0.85, label=lbl_rec)

    ax.set_xlabel("Interaction step")
    ax.set_ylabel("Ability (theta)")
    ax.legend(frameon=False, loc="upper left")

    # Compute pooled Pearson for annotation
    pool_r, _ = sp_stats.pearsonr(theta_rec.ravel(), theta_traj.ravel())
    ax.annotate(f"Pooled Pearson r = {abs(pool_r):.3f}",
                xy=(0.98, 0.05), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=LEGEND_SIZE, color=C_DARK)

    fig.tight_layout()
    return savefig("fig_trajectory.pdf")


# ===========================================================================
# Figure 2: Drift magnitude sweep (net-drift Spearman)
# ===========================================================================

def fig_drift_sweep():
    print("\n[2/6] fig_drift_sweep.pdf")

    # Real values from deep_irt/robustness/outputs/robustness_report.txt
    sigmas    = [0.05, 0.12, 0.25, 0.50]
    spearman  = [0.3483, 0.7115, 0.8967, 0.9476]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    # Shade reliable region (sigma >= 0.12)
    ax.axvspan(0.12, 0.55, color=C_SHADE, alpha=0.55, zorder=0, label="Reliable region")

    ax.plot(sigmas, spearman, color=C_BLUE, lw=2.2, marker="o",
            ms=8, mec="white", mew=1.5, zorder=3)

    # Annotate each point
    for s, r in zip(sigmas, spearman):
        ax.annotate(f"{r:.2f}", xy=(s, r), xytext=(0, 8),
                    textcoords="offset points", ha="center",
                    fontsize=LEGEND_SIZE, color=C_DARK)

    ax.axhline(0.40, color=C_GREY, lw=1.2, ls="--", zorder=1)
    ax.text(0.51, 0.41, "reliability threshold", ha="right",
            fontsize=LEGEND_SIZE - 1, color=C_GREY)

    ax.set_xlabel("Drift magnitude (sigma)")
    ax.set_ylabel("Net-drift Spearman")
    ax.set_xlim(0.01, 0.56)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(sigmas)
    ax.set_xticklabels([str(s) for s in sigmas])
    ax.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    return savefig("fig_drift_sweep.pdf")


# ===========================================================================
# Figure 3: Cross-format transfer through a shared scale (the honest test)
#
# Two panels. Items are supervised in only ONE format and read out in the
# OTHER. With a shared difficulty scalar (left) the held-out-format items
# land on the true scale; with independent fits (right) they are noise,
# because an item never seen by a format has no position in that format.
# ===========================================================================

def fig_crossformat():
    print("\n[3/6] fig_crossformat.pdf -- loading jointfmt/outputs/item_arrays.npz")

    npz_path = os.path.join(_SUBSTRATE_DIR, "jointfmt", "outputs", "item_arrays.npz")
    d = np.load(npz_path)
    s_true  = d["s_true"]
    group   = d["group"]            # 0 BOTH, 1 DIRECT-ONLY, 2 PAIRWISE-ONLY
    joint_d = d["joint_d"]          # single shared scale
    ig_d    = d["indep_gpcm_d"]     # independent GPCM readout
    ib_d    = d["indep_bt_d"]       # independent BT readout

    def align(v, ref):
        return -v if sp_stats.pearsonr(v, ref)[0] < 0 else v

    def z(v):
        return (v - v.mean()) / (v.std() + 1e-8)

    joint_d = align(joint_d, s_true)
    ig_d    = align(ig_d, s_true)
    ib_d    = align(ib_d, s_true)

    both = group == 0
    do   = group == 1   # direct-only: GPCM-supervised, read out by the pairwise scale
    po   = group == 2   # pairwise-only: BT-supervised, read out by the ordinal scale

    sx = z(s_true)
    jz = z(joint_d)

    # Cross-format Spearman on held-out items
    sp_po = sp_stats.spearmanr(joint_d[po], s_true[po]).statistic   # ordinal head, BT-only items
    sp_do = sp_stats.spearmanr(joint_d[do], s_true[do]).statistic   # pairwise head, GPCM-only items
    sp_po_i = sp_stats.spearmanr(ig_d[po], s_true[po]).statistic    # independent, noise
    sp_do_i = sp_stats.spearmanr(ib_d[do], s_true[do]).statistic    # independent, noise
    print(f"    shared : pairwise-only ord={sp_po:.3f}  direct-only pair={sp_do:.3f}")
    print(f"    indep  : pairwise-only ord={sp_po_i:.3f}  direct-only pair={sp_do_i:.3f}")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.0, 3.9), sharex=True, sharey=True)
    lim = (-3.0, 3.0)

    # ---- Left: shared scale ----
    axL.plot(lim, lim, color=C_GREY, lw=1.2, ls="--", zorder=1)
    axL.scatter(sx[both], jz[both], s=16, alpha=0.30, color=C_GREY,
                edgecolors="none", zorder=2, label="seen in both (bridge)")
    axL.scatter(sx[po], jz[po], s=26, alpha=0.80, color=C_RED,
                edgecolors="none", zorder=3, label="pairwise-only → ordinal head")
    axL.scatter(sx[do], jz[do], s=26, alpha=0.80, color=C_BLUE,
                edgecolors="none", zorder=3, label="direct-only → pairwise head")
    axL.set_title("Shared scale (one model)", fontsize=FONT_SIZE)
    axL.set_xlabel("True item difficulty (z)")
    axL.set_ylabel("Recovered position (z)")
    axL.set_xlim(lim); axL.set_ylim(lim)
    axL.annotate(f"pairwise-only  $\\rho$={sp_po:.2f}\ndirect-only   $\\rho$={sp_do:.2f}",
                 xy=(0.04, 0.96), xycoords="axes fraction", ha="left", va="top",
                 fontsize=LEGEND_SIZE,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GREY, alpha=0.9))
    axL.legend(frameon=False, loc="lower right", fontsize=LEGEND_SIZE - 2)

    # ---- Right: independent fits (held-out items only) ----
    axR.plot(lim, lim, color=C_GREY, lw=1.2, ls="--", zorder=1)
    axR.scatter(sx[po], z(ig_d)[po], s=26, alpha=0.80, color=C_RED,
                edgecolors="none", zorder=3, label="pairwise-only → ordinal head")
    axR.scatter(sx[do], z(ib_d)[do], s=26, alpha=0.80, color=C_BLUE,
                edgecolors="none", zorder=3, label="direct-only → pairwise head")
    axR.set_title("Independent fits (no sharing)", fontsize=FONT_SIZE)
    axR.set_xlabel("True item difficulty (z)")
    axR.set_xlim(lim); axR.set_ylim(lim)
    axR.annotate(f"pairwise-only  $\\rho$={sp_po_i:.2f}\ndirect-only   $\\rho$={sp_do_i:.2f}",
                 xy=(0.04, 0.96), xycoords="axes fraction", ha="left", va="top",
                 fontsize=LEGEND_SIZE,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GREY, alpha=0.9))

    fig.tight_layout()
    # Report the held-out ordinal-from-pairwise number as the headline.
    return savefig("fig_crossformat.pdf"), sp_po


# ===========================================================================
# Figure 4: SLAM transfer scatter (human vs LLM difficulty)
# ===========================================================================

def fig_slam_transfer():
    print("\n[4/6] fig_slam_transfer.pdf -- loading tap_scaled_results.csv")

    import pandas as pd

    csv_path = os.path.join(_SUBSTRATE_DIR, "slam_pilot", "outputs",
                            "tap_scaled_results.csv")
    df = pd.read_csv(csv_path)
    human = df["human_difficulty"].values
    llm   = df["llm_difficulty"].values
    n     = len(df)

    sp_val = sp_stats.spearmanr(human, llm).statistic
    print(f"    Fig4 SLAM Spearman (human vs LLM difficulty): {sp_val:.4f}, n={n}")

    # Best-fit line
    m, b_coef = np.polyfit(human, llm, 1)
    x_line = np.linspace(human.min(), human.max(), 200)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    ax.scatter(human, llm, s=14, alpha=0.35, color=C_BLUE,
               edgecolors="none", zorder=2)
    ax.plot(x_line, m * x_line + b_coef, color=C_RED, lw=1.8,
            ls="--", zorder=3)

    ax.set_xlabel("Human difficulty")
    ax.set_ylabel("LLM difficulty")

    ax.annotate(f"Spearman = {sp_val:.3f}, n = {n}\n(Duolingo SLAM)",
                xy=(0.97, 0.07), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=LEGEND_SIZE,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))

    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)

    fig.tight_layout()
    return savefig("fig_slam_transfer.pdf"), sp_val


# ===========================================================================
# Figure 5: Cost comparison (parameters + wall-clock)
# ===========================================================================

def fig_cost():
    print("\n[5/6] fig_cost.pdf")

    labels        = ["Full\nrecalibration", "Anchored\nextension"]
    n_params      = [7541, 240]
    wall_seconds  = [92.1, 2.5]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))

    colors = [C_RED, C_BLUE]
    x = np.arange(len(labels))
    bar_w = 0.55

    # Panel 1: trainable parameters (log scale)
    bars1 = ax1.bar(x, n_params, width=bar_w, color=colors, zorder=3)
    ax1.set_yscale("log")
    ax1.set_ylabel("Trainable parameters")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=TICK_SIZE)
    ax1.set_ylim(10, n_params[0] * 3)
    for bar, val in zip(bars1, n_params):
        ax1.text(bar.get_x() + bar.get_width() / 2, val * 1.35,
                 f"{val:,}", ha="center", va="bottom",
                 fontsize=LEGEND_SIZE, color=C_DARK)
    ax1.yaxis.grid(True, which="both", alpha=0.3, zorder=0)
    ax1.set_axisbelow(True)

    # Panel 2: wall-clock seconds (log scale)
    bars2 = ax2.bar(x, wall_seconds, width=bar_w, color=colors, zorder=3)
    ax2.set_yscale("log")
    ax2.set_ylabel("Wall-clock time (seconds)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=TICK_SIZE)
    ax2.set_ylim(0.5, wall_seconds[0] * 4)
    for bar, val in zip(bars2, wall_seconds):
        ax2.text(bar.get_x() + bar.get_width() / 2, val * 1.4,
                 f"{val:.1f}s", ha="center", va="bottom",
                 fontsize=LEGEND_SIZE, color=C_DARK)
    ax2.yaxis.grid(True, which="both", alpha=0.3, zorder=0)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    return savefig("fig_cost.pdf")


# ===========================================================================
# Figure 6: Multi-round sequential extension
# ===========================================================================

def fig_multiround():
    print("\n[6/6] fig_multiround.pdf")

    # Real values from robustness_report.txt (Sweep 2, new_item_a_sp column)
    rounds     = [1, 2, 3, 4, 5]
    new_item_sp = [0.7504, 0.2887, 0.7744, 0.8947, 0.6767]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    # Reference line at 1.0
    ax.axhline(1.0, color=C_GREY, lw=1.5, ls="-", zorder=1)
    ax.text(5.05, 1.0, "Base scale\nstability (exact)",
            ha="left", va="center", fontsize=LEGEND_SIZE - 1, color=C_GREY)

    # Bars
    bar_colors = [C_BLUE if v >= 0.50 else C_ORANGE for v in new_item_sp]
    bars = ax.bar(rounds, new_item_sp, width=0.55, color=bar_colors,
                  zorder=3, edgecolor="white", linewidth=0.5)

    # Annotate bar values
    for bar, val in zip(bars, new_item_sp):
        ypos = val + 0.025 if val < 0.93 else val - 0.06
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=LEGEND_SIZE - 1, color=C_DARK)

    # Mean line
    mean_sp = float(np.mean(new_item_sp))
    ax.axhline(mean_sp, color=C_RED, lw=1.5, ls="--", zorder=2,
               label=f"Mean = {mean_sp:.3f}")

    ax.set_xlabel("Extension round")
    ax.set_ylabel("New-item recovery (Spearman)")
    ax.set_xticks(rounds)
    ax.set_xlim(0.3, 6.2)
    ax.set_ylim(0.0, 1.15)
    ax.legend(frameon=False, loc="lower right")

    # Legend patches for bar colors
    patch_ok  = mpatches.Patch(color=C_BLUE,   label="Spearman >= 0.50")
    patch_low = mpatches.Patch(color=C_ORANGE, label="Spearman < 0.50")
    ax.legend(handles=[patch_ok, patch_low,
                        plt.Line2D([0], [0], color=C_RED, lw=1.5, ls="--",
                                   label=f"Mean = {mean_sp:.3f}")],
              frameon=False, loc="lower right", fontsize=LEGEND_SIZE - 1)

    fig.tight_layout()
    return savefig("fig_multiround.pdf")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 60)
    print("DeepIRT slide figures -- generating 6 PDFs")
    print(f"Output directory: {OUT_DIR}")
    print("=" * 60)

    t0 = time.time()

    paths = {}
    spearmans = {}

    # Figure 1: trajectory (runs model in-process)
    paths["fig_trajectory"] = fig_trajectory()

    # Figure 2: drift sweep
    paths["fig_drift_sweep"] = fig_drift_sweep()

    # Figure 3: cross-format transfer through a shared scale (loads npz)
    path3, sp3 = fig_crossformat()
    paths["fig_crossformat"] = path3
    spearmans["crossformat_heldout_transfer"] = sp3

    # Figure 4: SLAM transfer (loads CSV)
    path4, sp4 = fig_slam_transfer()
    paths["fig_slam_transfer"] = path4
    spearmans["slam_human_vs_llm"] = sp4

    # Figure 5: cost comparison
    paths["fig_cost"] = fig_cost()

    # Figure 6: multi-round
    paths["fig_multiround"] = fig_multiround()

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("DONE  --  all 6 figures saved")
    print(f"Total time: {elapsed:.1f}s")
    print()
    print("Output PDFs:")
    for key, p in paths.items():
        print(f"  {p}")
    print()
    print("Computed Spearman values:")
    print(f"  Fig 3  Held-out cross-format transfer  : {spearmans['crossformat_heldout_transfer']:.4f}")
    print(f"  Fig 4  Human vs LLM difficulty (SLAM)  : {spearmans['slam_human_vs_llm']:.4f}")
    print("=" * 60)

    return paths, spearmans


if __name__ == "__main__":
    main()
