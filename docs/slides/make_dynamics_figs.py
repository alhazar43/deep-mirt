"""Render presentation figures for the finite-sample representation
learning-dynamics study. Plain-language labels only.

Reads existing bench outputs read-only and writes five PNGs into
docs/slides/figs/. Run from repo root with the research env active.
"""

import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(REPO, "docs", "slides", "figs")
BENCH = os.path.join(REPO, "deep_irt", "bench", "outputs")

plt.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 19,
        "axes.labelsize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

SHARED_C = "#c0392b"   # shared representation, red
SEP_C = "#2471a3"      # separate representation, blue
NARROW_C = "#7f8c8d"   # shared-narrow, grey
ABILITY_C = "#c0392b"  # ability pathway
SHARP_C = "#2471a3"    # sharpness pathway


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
# d1: capacity trade-off curve, from gate_table.md FRONTIER block
# ----------------------------------------------------------------------
def fig_d1():
    # (theta_static, alpha) on static_k4, mean over seeds (gate_table.md).
    shared = {
        8: (0.970, 0.609),
        16: (0.956, 0.823),
        24: (0.935, 0.906),
        32: (0.907, 0.918),
        48: (0.903, 0.922),
        64: (0.898, 0.924),
        96: (0.915, 0.918),
        128: (0.910, 0.924),
    }
    # separate point at matched total size; take W=48 (both above threshold).
    sep_point = (0.970, 0.904)  # DECOUPLED(W=48)

    widths = sorted(shared)
    sx = [shared[w][1] for w in widths]  # sharpness match (x)
    sy = [shared[w][0] for w in widths]  # score match (y)

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    ax.plot(sx, sy, "-o", color=SHARED_C, lw=2.5, ms=8,
            label="shared representation (width sweep)")
    ax.scatter([sep_point[1]], [sep_point[0]], s=260, marker="*",
               color=SEP_C, zorder=5, edgecolor="k", linewidth=0.8,
               label="separate representation (matched size)")

    # annotate the trade-off direction along the shared curve
    ax.annotate("narrow shared code\n(good score, poor sharpness)",
                xy=(sx[0], sy[0]),
                xytext=(sx[0] + 0.015, sy[0] - 0.028), fontsize=11.5,
                color=SHARED_C, va="top")
    ax.annotate("wide shared code\n(good sharpness, poor score)",
                xy=(sx[-1], sy[-1]),
                xytext=(sx[-1] - 0.30, sy[-1] + 0.006), fontsize=11.5,
                color=SHARED_C)
    ax.annotate("both high", xy=sep_point[::-1],
                xytext=(sep_point[1] - 0.135, sep_point[0] - 0.013),
                fontsize=13, color=SEP_C, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=SEP_C, lw=1.8))

    ax.set_xlabel("sharpness match (0 to 1)")
    ax.set_ylabel("score match (0 to 1)")
    ax.set_title("One shared code cannot get both\nseparate code sits above the curve",
                 pad=14)
    ax.legend(loc="lower left")
    ax.set_xlim(0.58, 0.96)
    ax.set_ylim(0.892, 0.978)
    return save(fig, "d1_capacity_tradeoff.png")


# ----------------------------------------------------------------------
# d2: sharpness-match trajectory, from trajectory_results.json (static_k4)
# ----------------------------------------------------------------------
def fig_d2():
    d = json.load(open(os.path.join(BENCH, "trajectory_results.json")))
    agg = d["static_k4"]["aggregated"]
    eps = [int(e) for e in d["checkpoints"]]

    def curve(cfg):
        return [agg[cfg][str(e)]["alpha_mean"] for e in eps]

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    ax.plot(eps, curve("SHARED-WIDE"), "-o", color=SHARED_C, lw=3, ms=9,
            label="shared, wide code")
    ax.plot(eps, curve("DECOUPLED"), "-s", color=SEP_C, lw=3, ms=9,
            label="separate code")
    ax.plot(eps, curve("SHARED-NARROW"), "-^", color=NARROW_C, lw=2.5, ms=8,
            label="shared, narrow code")

    # mark the shared-wide peak then decay
    sw = curve("SHARED-WIDE")
    peak_i = max(range(len(sw)), key=lambda i: sw[i])
    ax.scatter([eps[peak_i]], [sw[peak_i]], s=220, marker="*",
               color=SHARED_C, zorder=6, edgecolor="k", linewidth=0.8)
    ax.annotate("peaks early,\nthen decays",
                xy=(eps[peak_i], sw[peak_i]),
                xytext=(eps[peak_i] + 70, sw[peak_i] - 0.18),
                fontsize=14, color=SHARED_C, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=SHARED_C, lw=2))
    ax.annotate("rises and holds",
                xy=(eps[-1], curve("DECOUPLED")[-1]),
                xytext=(eps[-1] - 230, curve("DECOUPLED")[-1] + 0.03),
                fontsize=14, color=SEP_C, fontweight="bold")

    ax.set_xlabel("training step (epoch)")
    ax.set_ylabel("sharpness match (0 to 1)")
    ax.set_title("A good shared solution is visited, then abandoned")
    ax.legend(loc="lower right")
    ax.set_ylim(0.0, 1.0)
    return save(fig, "d2_trajectory.png")


# ----------------------------------------------------------------------
# d3: gradient capture, from gradient_conflict_table.md
# ----------------------------------------------------------------------
def fig_d3():
    md = open(os.path.join(BENCH, "gradient_conflict_table.md")).read()
    rows = []
    for line in md.splitlines():
        m = re.match(r"\|\s*(\d+)\s*\|", line)
        if m:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            ep = int(cells[0])
            g_lstm = float(cells[3])
            g_alpha = float(cells[4])
            rows.append((ep, g_lstm, g_alpha))
    rows.sort()
    eps = [r[0] for r in rows]
    g_ability = [r[1] for r in rows]
    g_sharp = [r[2] for r in rows]
    growth = g_ability[-1] / g_ability[0]

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    ax.plot(eps, g_ability, "-o", color=ABILITY_C, lw=3, ms=10,
            label="ability pathway")
    ax.plot(eps, g_sharp, "-s", color=SHARP_C, lw=3, ms=10,
            label="sharpness pathway")
    ax.annotate(f"grows about {growth:.0f}x",
                xy=(eps[-1], g_ability[-1]),
                xytext=(eps[-1] - 230, g_ability[-1] - 0.0015),
                fontsize=15, color=ABILITY_C, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ABILITY_C, lw=2))
    ax.annotate("stays flat",
                xy=(eps[-1], g_sharp[-1]),
                xytext=(eps[-1] - 200, g_sharp[-1] + 0.0028),
                fontsize=15, color=SHARP_C, fontweight="bold")

    ax.set_xlabel("training step (epoch)")
    ax.set_ylabel("pull on the shared representation")
    ax.set_title("The ability pathway captures the shared code")
    ax.legend(loc="upper left")
    return save(fig, "d3_gradient_capture.png")


# ----------------------------------------------------------------------
# d4: advantage vs answer levels with stiffness, from ksweep_table.md
# ----------------------------------------------------------------------
def fig_d4():
    md = open(os.path.join(BENCH, "ksweep_table.md")).read()
    # parse the "delta_K and stiffness side by side" block
    block = md.split("delta_K and stiffness side by side")[1]
    Ks, stiff, delta = [], [], []
    for line in block.splitlines():
        m = re.match(r"\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*\+?([\d.]+)\s*\|", line)
        if m:
            Ks.append(int(m.group(1)))
            stiff.append(float(m.group(2)))
            delta.append(float(m.group(3)))

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    l1 = ax.plot(Ks, delta, "-o", color=SEP_C, lw=3, ms=10,
                 label="separate-code advantage")
    ax.set_xlabel("answer levels")
    ax.set_ylabel("advantage in sharpness match", color=SEP_C)
    ax.tick_params(axis="y", labelcolor=SEP_C)
    ax.set_ylim(0, max(delta) * 1.25)

    ax2 = ax.twinx()
    ax2.grid(False)
    l2 = ax2.plot(Ks, stiff, "--D", color="#b9770e", lw=2.5, ms=8,
                  label="how lopsided the learning is")
    ax2.set_ylabel("how lopsided the learning is\n(ability vs sharpness)",
                   color="#b9770e")
    ax2.tick_params(axis="y", labelcolor="#b9770e")

    lines = l1 + l2
    ax.legend(lines, [ln.get_label() for ln in lines], loc="upper left")
    ax.set_title("The advantage grows with more answer levels,\ntracking how lopsided the learning is")
    ax.set_xticks(Ks)
    return save(fig, "d4_stiffness_Ksweep.png")


# ----------------------------------------------------------------------
# d5: advantage vs number of learners, from ndata_sweep_results.json
# ----------------------------------------------------------------------
def fig_d5():
    d = json.load(open(os.path.join(BENCH, "ndata_sweep_results.json")))
    N_vals = d["N_vals"]
    delta = d["delta"]
    # show a representative spread of answer levels for clarity
    show_K = [2, 4, 6, 8, 11]
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    for i, K in enumerate(show_K):
        ys = [delta[str(K)][str(N)] for N in N_vals]
        ax.plot(N_vals, ys, "-o", lw=2.5, ms=8,
                color=cmap(i / (len(show_K) - 1)),
                label=f"{K} answer levels")

    ax.set_xscale("log")
    ax.set_xticks(N_vals)
    ax.set_xticklabels([str(n) for n in N_vals])
    ax.set_xlabel("number of learners")
    ax.set_ylabel("advantage in sharpness match")
    ax.set_title("More data does not shrink the advantage\n(it is rate-limited, not data-limited)")
    ax.axhline(0, color="k", lw=1, alpha=0.5)
    ax.legend(loc="lower right", ncol=2)
    ax.set_ylim(bottom=0)
    return save(fig, "d5_finite_sample_Nsweep.png")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = [fig_d1(), fig_d2(), fig_d3(), fig_d4(), fig_d5()]
    for p in paths:
        print("wrote", p)


if __name__ == "__main__":
    main()
