"""make_slide_figs.py -- progress-deck figures for the kt-mirt program (read only).

Reads the result files of the kt-mirt campaign and renders large-font slide PNGs
into ``docs/slides/figs_ktmirt``, where ``docs/slides/ktmirt_progress.tex`` picks
them up. No training, no model imports, matplotlib Agg. Every number on every
axis comes from a file listed below; the script computes no quantity the
campaign did not measure.

Run from anywhere: ``python kt-mirt/scripts/make_slide_figs.py``. The deck lives
in the ``docs/slides`` Overleaf submodule, which ignores Python, so this
generator is kept in the code tree and writes across into it.

Sources
-------
  kt-mirt/outputs/a4/campaign/{kdd,ednet}_matched/{syn_ng,syn_kg,syn_ns,syn_sat}
      /slice_seed*.json      the A4 existence-gate certification battery
  kt-mirt/outputs/a1/r0a1/r0a1_study.json        the L1 ladder (tuning seeds)
  kt-mirt/outputs/a1/r0a1/r0a1_floor_cert.json   the dose ladder (seeds 0-4)
  kt-mirt/outputs/a1/r0a1/r0a1_kill_arms.json    shuffle-order and phantom arms
  kt-mirt/scripts/slide_real_summary.json        real-bed scalars with provenance

Terms used on the axes are the program's own: knowledge component (KC), bed
statistic (the held-out negative-log-likelihood advantage of the growth model
over the no-growth reference, summed over the bed), permutation p-value,
Benjamini-Hochberg false-discovery-rate threshold, transfer coefficient G.

Figures
-------
  f1_certification.png    bed permutation p-value per twin, both density
                          profiles; the saturated twin's false fire in red.
  f2_statistic_density.png  raw bed statistic per twin; a clean gap at
                          KDD-shaped density, overlapping bands at EdNet-shaped.
  f3_perkc_floor.png      sorted per-KC p-values against the BH threshold;
                          zero discoveries on either profile.
  f4_real_beds.png        real-bed statistics (KDD fires, Junyi silent) and the
                          practice-depth ladder with the pending deep cohort.
  f5_dose_floor.png       recovered negative transfer coefficient against
                          injected dose, with the matched-null band.
  f6_shuffle_kill.png     matched against order-shuffled coefficients and the
                          pre-registered 10 percent collapse bar.
  f7_l1_refuted.png       the L1 ladder: true edges shrink faster than the band.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent          # kt-mirt/scripts
ROOT = _HERE.parent.parent                       # repository root
OUT_DIR = ROOT / "docs" / "slides" / "figs_ktmirt"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CAMPAIGN = ROOT / "kt-mirt" / "outputs" / "a4" / "campaign"
R0A1 = ROOT / "kt-mirt" / "outputs" / "a1" / "r0a1"
REAL = _HERE / "slide_real_summary.json"

DPI = 150

C_NULL = "#2166ac"     # blue, the no-growth twin (must stay silent)
C_OK = "#1b7837"       # green, a correct detection
C_FAIL = "#b2182b"     # red, a false fire or a failed bar
C_GREY = "#9e9e9e"
C_BAND = "#767676"

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.linewidth": 1.2,
    "lines.linewidth": 3.0,
})

PROFILES = [("kdd_matched", "KDD-shaped density"), ("ednet_matched", "EdNet-shaped density")]
TWINS = [("syn_ng", "no growth"), ("syn_kg", "known growth"),
         ("syn_ns", "non-standard"), ("syn_sat", "saturated")]
TWIN_COLOR = {"syn_ng": C_NULL, "syn_kg": C_OK, "syn_ns": C_OK, "syn_sat": C_FAIL}


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _campaign_cells(profile: str, twin: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(str(CAMPAIGN / profile / twin / "slice_seed*.json"))):
        out.append(_load(Path(f)))
    return out


def _save(fig, name: str) -> str:
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return name


# ---------------------------------------------------------------------------
# f1 -- the certification battery
# ---------------------------------------------------------------------------

def fig_certification() -> str:
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.0), sharey=True)
    for ax, (profile, title) in zip(axes, PROFILES):
        for x, (twin, label) in enumerate(TWINS):
            cells = _campaign_cells(profile, twin)
            ps = [c["gate"]["bed_pvalue"] for c in cells]
            jitter = np.linspace(-0.13, 0.13, len(ps)) if len(ps) > 1 else [0.0]
            ax.scatter([x + j for j in jitter], ps, s=150, zorder=3,
                       color=TWIN_COLOR[twin], edgecolor="white", linewidth=1.5)
        ax.axhline(0.001, color=C_GREY, ls="--", lw=2, zorder=1)
        ax.axhline(0.05, color=C_BAND, ls=":", lw=2, zorder=1)
        ax.set_yscale("log")
        ax.set_xticks(range(len(TWINS)))
        ax.set_xticklabels([lb for _, lb in TWINS], rotation=20, ha="right")
        ax.set_title(title)
        ax.set_xlim(-0.5, len(TWINS) - 0.5)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("bed permutation p-value")
    axes[0].text(0.02, 0.0016, "permutation floor 0.001", fontsize=14, color=C_GREY)
    axes[0].text(0.02, 0.062, "0.05", fontsize=14, color=C_BAND)
    axes[1].annotate("false fire\n(pre-repair)", xy=(3, 0.001), xytext=(2.35, 0.012),
                     fontsize=15, color=C_FAIL, ha="center",
                     arrowprops=dict(arrowstyle="->", color=C_FAIL, lw=2))
    fig.suptitle("Existence gate: silent on the null twin, detecting both growth twins", y=1.02)
    return _save(fig, "f1_certification.png")


# ---------------------------------------------------------------------------
# f2 -- the raw statistic is density-dependent
# ---------------------------------------------------------------------------

def fig_statistic_density() -> str:
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.0))
    for ax, (profile, title) in zip(axes, PROFILES):
        ng_vals = [c["gate"]["bed_stat"] for c in _campaign_cells(profile, "syn_ng")]
        ax.axhspan(min(ng_vals), max(ng_vals), color=C_NULL, alpha=0.15, zorder=0)
        for x, (twin, label) in enumerate(TWINS[:3]):
            vals = [c["gate"]["bed_stat"] for c in _campaign_cells(profile, twin)]
            jitter = np.linspace(-0.13, 0.13, len(vals)) if len(vals) > 1 else [0.0]
            ax.scatter([x + j for j in jitter], vals, s=150, zorder=3,
                       color=TWIN_COLOR[twin], edgecolor="white", linewidth=1.5)
        ax.set_xticks(range(3))
        ax.set_xticklabels([lb for _, lb in TWINS[:3]], rotation=20, ha="right")
        ax.set_title(title, pad=12)
        ax.set_xlim(-0.5, 2.5)
        allv = [c["gate"]["bed_stat"] for t, _ in TWINS[:3] for c in _campaign_cells(profile, t)]
        span = max(allv) - min(allv)
        ax.set_ylim(min(allv) - 0.10 * span, max(allv) + 0.22 * span)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("bed statistic")
    axes[0].text(0.5, 0.93, "clean gap", transform=axes[0].transAxes, fontsize=17,
                 color=C_OK, ha="center")
    axes[1].text(0.5, 0.93, "bands overlap", transform=axes[1].transAxes, fontsize=17,
                 color=C_FAIL, ha="center")
    fig.suptitle("The raw statistic separates at one density only; the calibrated p-value transfers",
                 y=1.02)
    return _save(fig, "f2_statistic_density.png")


# ---------------------------------------------------------------------------
# f3 -- per-KC resolution floor
# ---------------------------------------------------------------------------

def fig_perkc_floor() -> str:
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.0))
    for ax, (profile, title) in zip(axes, PROFILES):
        cell = _campaign_cells(profile, "syn_kg")[0]
        ps = np.sort(np.asarray(cell["gate"]["kc_pvalue"], dtype=float))
        m = len(ps)
        ranks = np.arange(1, m + 1)
        bh = 0.05 * ranks / m
        ax.plot(ranks, ps, color=C_OK, lw=3, label="per-KC p-values (sorted)")
        ax.plot(ranks, bh, color=C_FAIL, lw=3, ls="--", label="BH threshold")
        ax.set_yscale("log")
        ax.set_xlabel("KC rank")
        ax.set_title(f"{title}  ({m} KCs)")
        ax.grid(alpha=0.3)
        n_rej = int(sum(cell["gate"]["bh_reject"]))
        ax.text(0.05, 0.90, f"{n_rej} of {m} discoveries", transform=ax.transAxes,
                fontsize=17, color=C_FAIL)
    axes[0].set_ylabel("p-value")
    axes[0].legend(loc="lower right", frameon=False)
    fig.suptitle("Per-KC resolution: the smallest p-value never reaches the threshold", y=1.02)
    return _save(fig, "f3_perkc_floor.png")


# ---------------------------------------------------------------------------
# f4 -- real beds and the practice-depth ladder
# ---------------------------------------------------------------------------

def fig_real_beds() -> str:
    real = _load(REAL)
    kdd, junyi, deep = real["kdd_real"], real["junyi_real"], real["deep_junyi_pending"]

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.0))

    ax = axes[0]
    labels = ["KDD Algebra\n(deep practice)", "Junyi 2015\n(thin practice)"]
    vals = [kdd["bed_stat"], junyi["bed_stat"]]
    ax.bar(labels, vals, color=[C_OK, C_NULL], width=0.55, edgecolor="white", linewidth=2)
    ax.axhline(0, color="black", lw=2)
    ax.set_ylabel("bed statistic")
    ax.set_title("Real beds, one seed each", pad=14)
    lo, hi = min(vals), max(vals)
    ax.set_ylim(lo * 1.32, hi * 1.55)
    ax.text(0, hi * 1.10, f"+{kdd['bed_stat']:.0f}", ha="center", fontsize=17, color=C_OK)
    ax.text(0, hi * 1.34, f"fires, p = {kdd['bed_pvalue']}", ha="center", fontsize=15, color=C_OK)
    ax.text(1, lo * 1.08, f"{junyi['bed_stat']:.0f}", ha="center", va="top",
            fontsize=17, color=C_NULL)
    ax.text(1, lo * 1.24, "no growth detected", ha="center", va="top",
            fontsize=15, color=C_NULL)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ladder = [
        ("random 40k\nsilent", junyi["mean_rows_per_student"], C_NULL),
        ("deepest 40k\npending", deep["mean_rows_per_student"], C_GREY),
        ("KDD Algebra\nfires", kdd["mean_rows_per_student"], C_OK),
    ]
    xs = list(range(len(ladder)))
    ax.bar(xs, [v for _, v, _ in ladder],
           color=[c for _, _, c in ladder], width=0.55, edgecolor="white", linewidth=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([lb for lb, _, _ in ladder], fontsize=15)
    ax.set_yscale("log")
    ax.set_ylim(60, 9000)
    ax.set_ylabel("mean rows per student")
    ax.set_title("The designed depth ladder", pad=14)
    for x, (_, v, _) in zip(xs, ladder):
        ax.text(x, v * 1.18, str(v), ha="center", fontsize=17)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("The detector fires on deep-practice data and stays silent on thin", y=1.04)
    return _save(fig, "f4_real_beds.png")


# ---------------------------------------------------------------------------
# f5 -- the detectable-dose floor
# ---------------------------------------------------------------------------

def fig_dose_floor() -> str:
    cert = _load(R0A1 / "r0a1_floor_cert.json")
    band = cert["band"]
    doses = sorted(float(d) for d in cert["doses"])
    mags = [abs(cert["doses"][str(d)]["Gneg"]) for d in doses]
    detected = [cert["doses"][str(d)]["detected"] for d in doses]
    floor = cert["certified_floor"]

    fig, ax = plt.subplots(figsize=(10.0, 6.4))
    ax.plot(doses, doses, color=C_GREY, ls=":", lw=2, label="truth", zorder=1)
    ax.plot(doses, mags, color=C_GREY, lw=2.5, zorder=2)
    det_x = [d for d, s in zip(doses, detected) if s]
    det_y = [m for m, s in zip(mags, detected) if s]
    und_x = [d for d, s in zip(doses, detected) if not s]
    und_y = [m for m, s in zip(mags, detected) if not s]
    ax.scatter(und_x, und_y, s=280, zorder=4, color=C_FAIL, edgecolor="white",
               linewidth=2, label="bar not met")
    ax.scatter(det_x, det_y, s=340, zorder=5, color=C_OK, edgecolor="white",
               linewidth=2, marker="D", label=f"certified floor, |g| = {floor}")
    ax.axhline(band, color=C_BAND, ls="--", lw=2.5, zorder=1,
               label=f"matched-null band {band:.4f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.008, 0.105)
    ax.set_ylim(0.008, 0.16)
    ax.set_xticks(doses)
    ax.set_xticklabels([str(d) for d in doses])
    ax.set_yticks([0.01, 0.02, 0.04, 0.08])
    ax.set_yticklabels(["0.01", "0.02", "0.04", "0.08"])
    ax.set_xlabel("injected interference dose |g|")
    ax.set_ylabel("recovered coefficient |G|")
    ax.set_title("Interference is detectable from twice the reference dose", pad=14)
    ax.legend(frameon=False, loc="upper left", fontsize=15)
    ax.grid(alpha=0.3)
    return _save(fig, "f5_dose_floor.png")


# ---------------------------------------------------------------------------
# f6 -- the order-shuffle kill
# ---------------------------------------------------------------------------

def fig_shuffle_kill() -> str:
    arms = _load(R0A1 / "r0a1_kill_arms.json")
    matched = arms["matched"]
    sh = arms["shuffle"]
    pos_sh = float(np.mean(sh["Gpos_shuffled_by_seed"]))
    neg_sh = float(np.mean(sh["Gneg_shuffled_by_seed"]))

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    x = np.arange(2)
    w = 0.34
    m_vals = [abs(matched["Gpos"]), abs(matched["Gneg"])]
    s_vals = [abs(pos_sh), abs(neg_sh)]
    ax.bar(x - w / 2, m_vals, w, label="matched order", color=C_OK,
           edgecolor="white", linewidth=2)
    ax.bar(x + w / 2, s_vals, w, label="order shuffled", color=C_FAIL,
           edgecolor="white", linewidth=2)
    ax.set_ylim(0, max(m_vals) * 1.30)
    for xi, (m, s) in enumerate(zip(m_vals, s_vals)):
        ax.text(xi + w / 2, s * 1.04, f"{100 * s / m:.0f}%\nsurvives", ha="center", fontsize=16,
                color=C_FAIL)
    bar_pos = [0.10 * m for m in m_vals]
    ax.plot([x[0] - w, x[0] + w], [bar_pos[0]] * 2, color="black", ls="--", lw=2.5)
    ax.plot([x[1] - w, x[1] + w], [bar_pos[1]] * 2, color="black", ls="--", lw=2.5,
            label="pre-registered bar (10%)")
    ax.set_xticks(x)
    ax.set_xticklabels(["facilitation edge", "interference edge"])
    ax.set_ylabel("recovered coefficient |G|")
    ax.set_title("Destroying the temporal order leaves the estimate standing")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, "f6_shuffle_kill.png")


# ---------------------------------------------------------------------------
# f7 -- the L1 hypothesis, refuted
# ---------------------------------------------------------------------------

def fig_l1_refuted() -> str:
    study = _load(R0A1 / "r0a1_study.json")
    ceil = 500
    keys = [k for k in study["phase1"] if k.endswith(f"|ceil={ceil}")]
    l1s = sorted(float(k.split("|")[0].split("=")[1]) for k in keys)
    gpos, gneg, band = [], [], []
    for l1 in l1s:
        row = study["phase1"][f"l1={l1}|ceil={ceil}"]
        gpos.append(row["Gpos"])
        gneg.append(abs(row["Gneg"]))
        band.append(row["band"])

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ax.plot(l1s, gpos, "o-", color=C_OK, ms=12, label="facilitation edge (true +0.05)")
    ax.plot(l1s, gneg, "s-", color=C_NULL, ms=12, label="interference edge (true 0.02)")
    ax.plot(l1s, band, "^--", color=C_FAIL, ms=12, label="false-edge band")
    ax.set_xscale("log")
    ax.set_ylim(0.008, 0.088)
    ax.set_xlabel("L1 penalty weight on the transfer matrix")
    ax.set_ylabel("magnitude")
    ax.set_title("Stronger sparsity shrinks the true edges faster than the noise", pad=14)
    ax.legend(frameon=False, loc="upper left", fontsize=15)
    ax.grid(alpha=0.3)
    return _save(fig, "f7_l1_refuted.png")


def main() -> None:
    made = [
        fig_certification(),
        fig_statistic_density(),
        fig_perkc_floor(),
        fig_real_beds(),
        fig_dose_floor(),
        fig_shuffle_kill(),
        fig_l1_refuted(),
    ]
    print(f"wrote {len(made)} figures to {OUT_DIR}")
    for m in made:
        print("  " + m)


if __name__ == "__main__":
    main()
