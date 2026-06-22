"""Render the JOINT three-parameter trade-off figures (ability theta,
difficulty beta, discrimination alpha) under a shared item representation.

Reads bench outputs READ-ONLY and writes only new PNGs into docs/slides/figs/:
  j1_three_param_vs_width.png   theta/beta/alpha recovery vs shared width W
  j2_theta_overfit_vs_epoch.png theta recovery vs epoch, three configs
  j3_tradeoff_frontier.png      theta-vs-alpha Pareto frontier + decoupled point

Sources:
  deep_irt/bench/outputs/gate_results.json        (agg: theta/a/b vs W, by variant)
  deep_irt/bench/outputs/trajectory_results.json  (theta/alpha vs epoch, 3 configs)

Run from repo root with the research env active.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "deep_irt", "bench", "outputs")
FIGS = os.path.join(HERE, "figs")

DPI = 150
DATASET = "static_k4"  # the canonical parameter-recovery dataset

# Consistent, color-blind-aware palette and labels.
C_THETA = "#1f77b4"  # ability theta, blue
C_BETA = "#2ca02c"   # difficulty beta, green
C_ALPHA = "#d62728"  # discrimination alpha, red
C_DEC = "#7f3fbf"    # decoupled, purple

L_THETA = r"ability $\theta$"
L_BETA = r"difficulty $\beta$"
L_ALPHA = r"discrimination $\alpha$"

plt.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 19,
        "axes.labelsize": 17,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 2.6,
        "lines.markersize": 9,
    }
)


def load(name):
    with open(os.path.join(OUT, name), "r", encoding="utf-8") as fh:
        return json.load(fh)


def W_of(variant_key):
    return int(variant_key.split("_W")[1])


def gate_series(agg, variant_type):
    """Return (Ws, theta, theta_sd, alpha, alpha_sd, beta, beta_sd) sorted by W
    for the given variant_type ('shared' or 'decoupled') on DATASET."""
    rows = [
        a
        for a in agg
        if a["dataset"] == DATASET and a["variant_key"].startswith(variant_type)
    ]
    rows.sort(key=lambda a: W_of(a["variant_key"]))
    Ws = [W_of(a["variant_key"]) for a in rows]
    th = [a["theta_spearman_mean"] for a in rows]
    th_sd = [a["theta_spearman_std"] for a in rows]
    al = [a["a_spearman_mean"] for a in rows]
    al_sd = [a["a_spearman_std"] for a in rows]
    be = [a["b_spearman_mean"] for a in rows]
    be_sd = [a["b_spearman_std"] for a in rows]
    return Ws, th, th_sd, al, al_sd, be, be_sd


def fig_j1(agg):
    """Three-parameter recovery vs shared item-embedding width W."""
    Ws, th, th_sd, al, al_sd, be, be_sd = gate_series(agg, "shared")

    fig, ax = plt.subplots(figsize=(8.4, 6.0))

    ax.errorbar(Ws, th, yerr=th_sd, color=C_THETA, marker="o", capsize=4,
                label=L_THETA + "  (falls: overfit)")
    ax.errorbar(Ws, be, yerr=be_sd, color=C_BETA, marker="s", capsize=4,
                label=L_BETA + "  (flat, content-narrow)")
    ax.errorbar(Ws, al, yerr=al_sd, color=C_ALPHA, marker="^", capsize=4,
                label=L_ALPHA + "  (rises: needs width)")

    ax.set_xlabel("shared item embedding width $W$")
    ax.set_ylabel(r"recovery (Spearman $\rho$)")
    ax.set_title("Joint three-parameter trade-off on a shared item code")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ws)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(0.55, 1.0)

    # Annotate the conflict: theta down while alpha up.
    ax.annotate(
        r"widening for $\alpha$" + "\n" + r"overfits $\theta$",
        xy=(Ws[5], th[5]), xytext=(Ws[1], 0.66),
        arrowprops=dict(arrowstyle="->", color=C_THETA, lw=1.8),
        color=C_THETA, fontsize=13, ha="center",
    )
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()
    path = os.path.join(FIGS, "j1_three_param_vs_width.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path, (Ws, th, al, be)


def fig_j2(traj):
    """theta recovery vs epoch for shared-wide / decoupled / shared-narrow."""
    agg = traj[DATASET]["aggregated"]
    eps = sorted(int(k) for k in agg["SHARED-WIDE"].keys())

    def theta_curve(cfg):
        m = [agg[cfg][str(e)]["theta_mean"] for e in eps]
        s = [agg[cfg][str(e)]["theta_std"] for e in eps]
        return m, s

    fig, ax = plt.subplots(figsize=(8.4, 6.0))

    specs = [
        ("SHARED-WIDE", C_ALPHA, "o", "-", "shared (wide): overfits fastest"),
        ("SHARED-NARROW", C_THETA, "s", "--", "shared (narrow)"),
        ("DECOUPLED", C_DEC, "^", "-", "decoupled"),
    ]
    finals = {}
    peaks = {}
    for cfg, c, mk, ls, lab in specs:
        m, s = theta_curve(cfg)
        ax.errorbar(eps, m, yerr=s, color=c, marker=mk, linestyle=ls,
                    capsize=4, label=lab)
        finals[cfg] = (m[0], m[-1])
        peaks[cfg] = max(m)

    ax.set_xlabel("epoch")
    ax.set_ylabel(r"ability $\theta$ recovery (Spearman $\rho$)")
    ax.set_title(r"Wide shared code overfits ability $\theta$ with training")
    ax.set_xscale("log")
    ax.set_xticks(eps)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(0.70, 1.0)

    # Annotate the wide-vs-narrow theta gap (clearest near ep150).
    mw, _ = theta_curve("SHARED-WIDE")
    mn, _ = theta_curve("SHARED-NARROW")
    j = eps.index(150)
    ax.annotate(
        "", xy=(150, mw[j]), xytext=(150, mn[j]),
        arrowprops=dict(arrowstyle="<->", color="0.3", lw=1.6),
    )
    ax.text(165, 0.5 * (mw[j] + mn[j]),
            r"$\theta$ gap", color="0.2", fontsize=13, va="center")

    ax.legend(loc="lower left", framealpha=0.95)
    fig.tight_layout()
    path = os.path.join(FIGS, "j2_theta_overfit_vs_epoch.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path, finals, peaks


def fig_j3(agg):
    """theta-vs-alpha Pareto frontier: shared sweep as a curve, decoupled above."""
    sWs, sth, _, sal, _, _, _ = gate_series(agg, "shared")
    dWs, dth, _, dal, _, _, _ = gate_series(agg, "decoupled")

    fig, ax = plt.subplots(figsize=(8.4, 6.0))

    # Shared frontier as a connected curve, ordered by W.
    ax.plot(sal, sth, color="0.45", linestyle="-", marker="o", zorder=2,
            label="shared sweep (frontier)")
    for w, a, t in zip(sWs, sal, sth):
        ax.annotate(f"W={w}", (a, t), textcoords="offset points",
                    xytext=(6, -12), fontsize=11, color="0.35")

    # Decoupled cloud, sits above-and-right (the Pareto escape).
    ax.scatter(dal, dth, color=C_DEC, marker="*", s=240, zorder=3,
               edgecolor="white", linewidth=0.8, label="decoupled")

    ax.set_xlabel(r"discrimination $\alpha$ recovery (Spearman $\rho$)")
    ax.set_ylabel(r"ability $\theta$ recovery (Spearman $\rho$)")
    ax.set_title("Decoupling escapes the shared Pareto frontier")
    ax.set_xlim(0.55, 0.97)
    ax.set_ylim(0.86, 0.985)

    ax.annotate(
        "Pareto escape\n(both high)",
        xy=(min(dal), max(dth)),
        xytext=(0.72, 0.978),
        arrowprops=dict(arrowstyle="->", color=C_DEC, lw=1.8),
        color=C_DEC, fontsize=13, ha="center",
    )
    ax.legend(loc="lower left", framealpha=0.95)
    fig.tight_layout()
    path = os.path.join(FIGS, "j3_tradeoff_frontier.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def main():
    os.makedirs(FIGS, exist_ok=True)
    gate = load("gate_results.json")
    traj = load("trajectory_results.json")
    agg = gate["agg"]

    p1, d1 = fig_j1(agg)
    p2, finals, peaks = fig_j2(traj)
    p3 = fig_j3(agg)

    Ws, th, al, be = d1
    print("Wrote:")
    print(" ", p1)
    print(" ", p2)
    print(" ", p3)
    print()
    print("j1 (shared, %s): theta %.3f->%.3f, alpha %.3f->%.3f, beta %.3f->%.3f"
          % (DATASET, th[0], th[-1], al[0], al[-1], be[0], be[-1]),
          "over W", Ws[0], "->", Ws[-1])
    print("j2 theta (early->ep500):")
    for cfg in ["SHARED-WIDE", "SHARED-NARROW", "DECOUPLED"]:
        e, l = finals[cfg]
        print("   %-14s %.3f -> %.3f  (peak %.3f, drop %.3f)"
              % (cfg, e, l, peaks[cfg], e - l))


if __name__ == "__main__":
    main()
