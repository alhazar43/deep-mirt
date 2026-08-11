"""make_slide_figs.py -- progress-deck figures for the kt-mirt program (read only).

Reads the result files of the kt-mirt experiments and renders large-font slide
PNGs into ``docs/slides/figs_ktmirt``, where ``docs/slides/ktmirt_progress.tex``
picks them up. No training, no model imports, matplotlib Agg. Every number on
every axis comes from a file listed below; the script computes no quantity the
experiments did not measure.

Run from anywhere: ``python kt-mirt/scripts/make_slide_figs.py``. The deck lives
in the ``docs/slides`` Overleaf submodule, which ignores Python, so this
generator is kept in the code tree and writes across into it.

Sources
-------
  kt-mirt/outputs/a4/campaign/{kdd,ednet}_matched/{syn_ng,syn_kg,syn_ns,syn_sat}
      /slice_seed*.json    simulated datasets, learning-detection results
  kt-mirt/outputs/a1/r0a1/r0a1_study.json        L1 penalty sweep, tuning seeds
  kt-mirt/outputs/a1/r0a1/r0a1_floor_cert.json   effect-size sweep, evaluation seeds 0-4
  kt-mirt/outputs/a1/r0a1/r0a1_kill_arms.json    order-shuffle and free-multiplier controls
  kt-mirt/scripts/slide_real_summary.json        real-dataset scalars with provenance

Vocabulary
----------
Directory names keep the original run codes. On the axes they read as

  syn_ng   no learning (null)          syn_ns   alternative learning curve
  syn_kg   learning injected           syn_sat  ceiling effects

Each of these is a simulated dataset with known ground truth. The pairs are
matched: same practice schedule, same random seed, differing only in whether
learning is injected.

``bed_stat`` is the held-out log-likelihood gain in nats, summed over all
learner-skill sequences, of the model that allows ability to move over the
baseline that holds it fixed. Positive means the learning model predicts
held-out responses better. ``bed_pvalue`` and ``kc_pvalue`` are permutation
p-values at the dataset and per-skill level. ``band`` is the 95th percentile of
the estimated transfer magnitude on control datasets that carry no true
transfer, used as a null threshold. A knowledge component (KC) is the skill tag
the dataset ships with; it is written "skill" on every axis.

Figures
-------
  f1_detection.png        dataset-level permutation p-value per condition, both
                          density profiles; the ceiling-effect false positive in red.
  f2_statistic_density.png  raw held-out log-likelihood gain per condition; the
                          null and learning ranges separate at one density only.
  f3_perskill_floor.png   sorted per-skill p-values against the Benjamini-Hochberg
                          threshold; no discovery at either density.
  f4_real_data.png        real-dataset gains (KDD detects, Junyi does not) and the
                          practice-depth comparison with the pending deep cohort.
  f5_min_detectable.png   estimated negative-transfer coefficient against injected
                          effect size, plus the false-positive rate that binds.
  f6_order_shuffle.png    original against order-shuffled coefficients and the
                          pre-registered 10 percent collapse criterion.
  f7_l1_refuted.png       the L1 sweep: true effects shrink faster than the noise.
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

C_NULL = "#2166ac"     # blue, a null condition that must stay undetected
C_OK = "#1b7837"       # green, a correct detection or a met criterion
C_FAIL = "#b2182b"     # red, a false positive or a missed criterion
C_GREY = "#9e9e9e"
C_BAND = "#767676"
C_NOTE = "#444444"

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

# Two simulated density profiles bracketing the range seen in real data.
DENSITIES = [
    ("kdd_matched", "KDD-shaped density\nfew skills, deep practice"),
    ("ednet_matched", "EdNet-shaped density\nmany skills, thin practice"),
]

# Run codes on disk, and how they read on an axis.
CONDITIONS = [
    ("syn_ng", "no learning\n(null)"),
    ("syn_kg", "learning\ninjected"),
    ("syn_ns", "alternative\nlearning curve"),
    ("syn_sat", "ceiling\neffects"),
]
COND_COLOR = {"syn_ng": C_NULL, "syn_kg": C_OK, "syn_ns": C_OK, "syn_sat": C_FAIL}

STAT_LABEL = "held-out log-likelihood gain (nats)"
COEF_LABEL = "estimated transfer coefficient $|G|$"


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _campaign_cells(profile: str, condition: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(str(CAMPAIGN / profile / condition / "slice_seed*.json"))):
        out.append(_load(Path(f)))
    return out


def _jitter(n: int) -> list[float]:
    return list(np.linspace(-0.13, 0.13, n)) if n > 1 else [0.0]


def _save(fig, name: str) -> str:
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return name


# ---------------------------------------------------------------------------
# f1 -- learning detection on simulated data
# ---------------------------------------------------------------------------

def fig_detection() -> str:
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 4.8), sharey=True)
    for ax, (profile, title) in zip(axes, DENSITIES):
        for x, (cond, _label) in enumerate(CONDITIONS):
            ps = [c["gate"]["bed_pvalue"] for c in _campaign_cells(profile, cond)]
            ax.scatter([x + j for j in _jitter(len(ps))], ps, s=150, zorder=3,
                       color=COND_COLOR[cond], edgecolor="white", linewidth=1.5)
        ax.axhline(0.001, color=C_GREY, ls="--", lw=2, zorder=1,
                   label="smallest attainable p-value, $1/(B{+}1)=0.001$")
        ax.axhline(0.05, color=C_BAND, ls=":", lw=2, zorder=1,
                   label="0.05 significance level")
        ax.set_yscale("log")
        ax.set_ylim(4e-4, 3.0)
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels([lb for _, lb in CONDITIONS], fontsize=15)
        ax.set_title(title, fontsize=20, pad=12)
        ax.set_xlim(-0.5, len(CONDITIONS) - 0.5)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("permutation p-value\n(whole dataset)")
    axes[1].annotate("false positive\n(before the fix)", xy=(3, 0.0012), xytext=(2.3, 0.014),
                     fontsize=15, color=C_FAIL, ha="center",
                     arrowprops=dict(arrowstyle="->", color=C_FAIL, lw=2))
    axes[0].legend(loc="center", bbox_to_anchor=(0.52, 0.42), frameon=False, fontsize=13)
    fig.suptitle("No detection on the null datasets, detection wherever learning was injected",
                 y=1.13)
    fig.text(0.5, -0.13, "One point per simulation seed, five seeds per condition. "
                         "Permutation test with $B=999$ relabelings.",
             ha="center", fontsize=15, color=C_NOTE)
    return _save(fig, "f1_detection.png")


# ---------------------------------------------------------------------------
# f2 -- the raw statistic depends on data density
# ---------------------------------------------------------------------------

def fig_statistic_density() -> str:
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.6))
    for ax, (profile, title) in zip(axes, DENSITIES):
        ng_vals = [c["gate"]["bed_stat"] for c in _campaign_cells(profile, "syn_ng")]
        ax.axhspan(min(ng_vals), max(ng_vals), color=C_NULL, alpha=0.15, zorder=0,
                   label="range across the five null datasets")
        for x, (cond, _label) in enumerate(CONDITIONS[:3]):
            vals = [c["gate"]["bed_stat"] for c in _campaign_cells(profile, cond)]
            ax.scatter([x + j for j in _jitter(len(vals))], vals, s=150, zorder=3,
                       color=COND_COLOR[cond], edgecolor="white", linewidth=1.5)
        ax.set_xticks(range(3))
        ax.set_xticklabels([lb for _, lb in CONDITIONS[:3]], fontsize=15)
        ax.set_title(title, fontsize=20, pad=12)
        ax.set_xlim(-0.5, 2.5)
        allv = [c["gate"]["bed_stat"] for t, _ in CONDITIONS[:3] for c in _campaign_cells(profile, t)]
        span = max(allv) - min(allv)
        ax.set_ylim(min(allv) - 0.14 * span, max(allv) + 0.30 * span)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel(STAT_LABEL)
    axes[0].text(0.5, 0.94, "no overlap", transform=axes[0].transAxes, fontsize=17,
                 color=C_OK, ha="center")
    axes[1].text(0.5, 0.94, "learning overlaps the null", transform=axes[1].transAxes,
                 fontsize=17, color=C_FAIL, ha="center")
    fig.suptitle("The raw statistic separates at one density only, the p-value at both", y=1.06)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 0.03), ncol=1, frameon=False, fontsize=15)
    fig.text(0.5, -0.08, "Higher is better prediction of held-out responses by the learning model. "
                         "Five seeds per condition.",
             ha="center", fontsize=15, color=C_NOTE)
    return _save(fig, "f2_statistic_density.png")


# ---------------------------------------------------------------------------
# f3 -- per-skill resolution floor
# ---------------------------------------------------------------------------

def fig_perskill_floor() -> str:
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.6))
    for ax, (profile, title) in zip(axes, DENSITIES):
        cell = _campaign_cells(profile, "syn_kg")[0]
        ps = np.sort(np.asarray(cell["gate"]["kc_pvalue"], dtype=float))
        m = len(ps)
        ranks = np.arange(1, m + 1)
        bh = 0.05 * ranks / m
        ax.plot(ranks, ps, color=C_OK, lw=3, label="per-skill p-values, sorted")
        ax.plot(ranks, bh, color=C_FAIL, lw=3, ls="--",
                label="Benjamini-Hochberg threshold, FDR $q=0.05$")
        ax.set_yscale("log")
        ax.set_xlabel("skills, ranked by p-value")
        ax.set_title(title.split("\n")[0] + f", {m} skills", fontsize=20, pad=12)
        ax.set_ylim(2e-5, 3.0)
        ax.grid(alpha=0.3)
        n_rej = int(sum(cell["gate"]["bh_reject"]))
        ax.text(0.04, 0.92, f"{n_rej} of {m} skills detected", transform=ax.transAxes,
                fontsize=17, color=C_FAIL)
        ax.text(0.40, 0.20, f"smallest p-value {ps[0]:.3f}\nagainst a threshold of {bh[0]:.1e}",
                transform=ax.transAxes, fontsize=15, color=C_NOTE)
    axes[0].set_ylabel("per-skill permutation p-value")
    fig.suptitle("Positive control, learning injected in every skill, and no skill is detected",
                 y=1.03)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 0.03), ncol=2, frameon=False, fontsize=15)
    return _save(fig, "f3_perskill_floor.png")


# ---------------------------------------------------------------------------
# f4 -- real datasets and practice depth
# ---------------------------------------------------------------------------

def fig_real_data() -> str:
    real = _load(REAL)
    kdd, junyi, deep = real["kdd_real"], real["junyi_real"], real["deep_junyi_pending"]

    fig, axes = plt.subplots(1, 2, figsize=(16.0, 4.9),
                             gridspec_kw={"wspace": 0.36})

    ax = axes[0]
    labels = ["KDD Algebra\n(deep practice)", "Junyi 2015\n(thin practice)"]
    vals = [kdd["bed_stat"], junyi["bed_stat"]]
    ax.bar(labels, vals, color=[C_OK, C_NULL], width=0.55, edgecolor="white", linewidth=2)
    ax.axhline(0, color="black", lw=2)
    ax.set_ylabel(STAT_LABEL)
    ax.set_title("Real data, one seed each", fontsize=20, pad=14)
    lo, hi = min(vals), max(vals)
    ax.set_ylim(lo * 1.36, hi * 2.02)
    ax.text(0, hi * 1.10, f"+{kdd['bed_stat']:.0f} nats\nlearning detected\np = {kdd['bed_pvalue']}",
            ha="center", va="bottom", fontsize=16, color=C_OK, linespacing=1.25)
    ax.text(1, lo * 1.08, f"{junyi['bed_stat']:.0f} nats\nno detection",
            ha="center", va="top", fontsize=16, color=C_NULL)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    depth = [
        ("random 40k\nno detection", junyi["mean_rows_per_student"], C_NULL),
        ("deepest 40k\npending", deep["mean_rows_per_student"], C_GREY),
        ("KDD Algebra\ndetected", kdd["mean_rows_per_student"], C_OK),
    ]
    xs = list(range(len(depth)))
    ax.bar(xs, [v for _, v, _ in depth],
           color=[c for _, _, c in depth], width=0.55, edgecolor="white", linewidth=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([lb for lb, _, _ in depth], fontsize=15)
    ax.set_yscale("log")
    ax.set_ylim(60, 9000)
    ax.set_ylabel("mean interactions\nper student (log scale)")
    ax.set_title("Practice depth, two Junyi cohorts and KDD", fontsize=19, pad=14)
    for x, (_, v, _) in zip(xs, depth):
        ax.text(x, v * 1.20, str(v), ha="center", fontsize=17)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Detection on deep-practice data, silence on thin", y=1.10)
    fig.text(0.5, -0.14, "A negative gain means the no-learning baseline predicts held-out "
                         "responses better than the learning model.",
             ha="center", fontsize=15, color=C_NOTE)
    return _save(fig, "f4_real_data.png")


# ---------------------------------------------------------------------------
# f5 -- the minimum detectable effect
# ---------------------------------------------------------------------------

def fig_min_detectable() -> str:
    cert = _load(R0A1 / "r0a1_floor_cert.json")
    band = cert["band"]
    sizes = sorted(float(d) for d in cert["doses"])
    mags = [abs(cert["doses"][str(d)]["Gneg"]) for d in sizes]
    fer = [cert["doses"][str(d)]["false_edge_rate"] for d in sizes]
    passed = [cert["doses"][str(d)]["detected"] for d in sizes]
    mde = cert["certified_floor"]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11.0, 8.2), sharex=True,
                                  gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.10})

    ax.plot(sizes, sizes, color=C_GREY, ls=":", lw=2, zorder=1, label="perfect recovery")
    ax.plot(sizes, mags, color=C_GREY, lw=2.5, zorder=2, label="estimate, mean of 5 seeds")
    ax.scatter([d for d, s in zip(sizes, passed) if not s],
               [m for m, s in zip(mags, passed) if not s],
               s=280, zorder=4, color=C_FAIL, edgecolor="white", linewidth=2,
               label="criteria not met")
    ax.scatter([d for d, s in zip(sizes, passed) if s],
               [m for m, s in zip(mags, passed) if s],
               s=340, zorder=5, color=C_OK, edgecolor="white", linewidth=2, marker="D",
               label=f"all criteria met, $|g|={mde}$")
    ax.axhline(band, color=C_BAND, ls="--", lw=2.5, zorder=1,
               label=f"null threshold {band:.4f}")
    ax.set_yscale("log")
    ax.set_ylim(0.008, 0.20)
    ax.set_yticks([0.01, 0.02, 0.04, 0.08])
    ax.set_yticklabels(["0.01", "0.02", "0.04", "0.08"])
    ax.set_ylabel(COEF_LABEL)
    ax.set_title("Minimum detectable effect for negative transfer", pad=14)
    ax.legend(frameon=False, loc="upper left", fontsize=14, ncol=1)
    ax.grid(alpha=0.3)

    ax2.plot(sizes, fer, color=C_GREY, lw=2.5, zorder=2)
    ax2.scatter([d for d, s in zip(sizes, passed) if not s],
                [f for f, s in zip(fer, passed) if not s],
                s=220, zorder=4, color=C_FAIL, edgecolor="white", linewidth=2)
    ax2.scatter([d for d, s in zip(sizes, passed) if s],
                [f for f, s in zip(fer, passed) if s],
                s=260, zorder=5, color=C_OK, edgecolor="white", linewidth=2, marker="D")
    ax2.axhline(0.05, color=C_BAND, ls="--", lw=2.5, zorder=1)
    ax2.text(0.0084, 0.062, "pre-registered criterion, 5 percent", fontsize=14, color=C_BAND)
    ax2.set_xscale("log")
    ax2.set_xlim(0.008, 0.105)
    ax2.set_ylim(-0.005, 0.21)
    ax2.set_xticks(sizes)
    ax2.set_xticklabels([str(d) for d in sizes])
    ax2.set_yticks([0.0, 0.05, 0.10, 0.15])
    ax2.set_yticklabels(["0", "5", "10", "15"])
    ax2.set_xlabel("injected effect size $|g|$")
    ax2.set_ylabel("false positives,\npercent of pairs\nwith no true effect", fontsize=15)
    ax2.grid(alpha=0.3)
    return _save(fig, "f5_min_detectable.png")


# ---------------------------------------------------------------------------
# f6 -- the temporal-order negative control
# ---------------------------------------------------------------------------

def fig_order_shuffle() -> str:
    arms = _load(R0A1 / "r0a1_kill_arms.json")
    matched = arms["matched"]
    sh = arms["shuffle"]
    pos_sh = float(np.mean(sh["Gpos_shuffled_by_seed"]))
    neg_sh = float(np.mean(sh["Gneg_shuffled_by_seed"]))

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    x = np.arange(2)
    w = 0.34
    m_vals = [abs(matched["Gpos"]), abs(matched["Gneg"])]
    s_vals = [abs(pos_sh), abs(neg_sh)]
    b1 = ax.bar(x - w / 2, m_vals, w, label="original order", color=C_OK,
                edgecolor="white", linewidth=2)
    b2 = ax.bar(x + w / 2, s_vals, w, label="cross-skill order shuffled", color=C_FAIL,
                edgecolor="white", linewidth=2)
    ax.set_ylim(0, max(m_vals) * 1.46)
    for xi, (m, s) in enumerate(zip(m_vals, s_vals)):
        ax.text(xi + w / 2, s * 1.05, f"{100 * s / m:.0f} percent\nsurvives", ha="center",
                fontsize=16, color=C_FAIL)
    bar_pos = [0.10 * m for m in m_vals]
    ax.plot([x[0] - w, x[0] + w], [bar_pos[0]] * 2, color="black", ls="--", lw=2.5, zorder=4)
    line, = ax.plot([x[1] - w, x[1] + w], [bar_pos[1]] * 2, color="black", ls="--", lw=2.5,
                    zorder=4, label="pre-registered criterion, 10 percent of the original")
    ax.set_xticks(x)
    ax.set_xticklabels(["positive transfer", "negative transfer"])
    ax.set_ylabel(COEF_LABEL)
    ax.set_title("Shuffling the cross-skill order\nbarely moves the estimate", pad=14)
    ax.legend([b1, b2, line], [b1.get_label(), b2.get_label(), line.get_label()],
              frameon=False, loc="upper center", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    fig.text(0.5, -0.05, "Each learner's cross-skill ordering is re-drawn, with every skill's own "
                         "sequence and every practice count held fixed. Five seeds.",
             ha="center", fontsize=14, color=C_NOTE)
    return _save(fig, "f6_order_shuffle.png")


# ---------------------------------------------------------------------------
# f7 -- the sparsity-penalty hypothesis, refuted
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

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.plot(l1s, gpos, "o-", color=C_OK, ms=12, label="positive transfer, true $+0.05$")
    ax.plot(l1s, gneg, "s-", color=C_NULL, ms=12, label="negative transfer, true $-0.02$")
    ax.plot(l1s, band, "^--", color=C_FAIL, ms=12,
            label="false-positive threshold, pairs with no true effect")
    ax.set_xscale("log")
    ax.set_ylim(0.008, 0.098)
    ax.set_xticks(l1s)
    ax.set_xticklabels([str(v) for v in l1s])
    ax.minorticks_off()
    ax.set_xlabel("L1 penalty weight on the transfer matrix (log scale)")
    ax.set_ylabel(COEF_LABEL)
    ax.set_title("Stronger sparsity shrinks the true effects\nfaster than the noise", pad=14)
    ax.legend(frameon=False, loc="upper left", fontsize=14)
    ax.grid(alpha=0.3)
    fig.text(0.5, -0.04, "Tuning seeds only, held out from every evaluation run.",
             ha="center", fontsize=14, color=C_NOTE)
    return _save(fig, "f7_l1_refuted.png")


def main() -> None:
    made = [
        fig_detection(),
        fig_statistic_density(),
        fig_perskill_floor(),
        fig_real_data(),
        fig_min_detectable(),
        fig_order_shuffle(),
        fig_l1_refuted(),
    ]
    print(f"wrote {len(made)} figures to {OUT_DIR}")
    for m in made:
        print("  " + m)


if __name__ == "__main__":
    main()
