"""make_claim_figs.py -- four presentation figures for the talk (read only).

Reads the existing bench result files under ``deep_irt/bench/outputs`` and the
fresh trajectory run, and renders four large-font slide PNGs into
``docs/slides/figs``.  No training, no model imports, matplotlib Agg.

Labels use precise psychometric/ML terms: ability (theta), difficulty (step
thresholds), discrimination (the IRT alpha), number of categories K, epoch, and
parameter recovery (Spearman rho) on a 0 to 1 axis.

Figures
-------
  s1_three_speeds.png      param_trajectory_K4.json (+K8 panel): three recovery
                           curves vs epoch; ability and difficulty rise fast,
                           discrimination rises slower.
  s2_sharpness_help.png    alpha_beta_asymmetry.json: grouped bars per number of
                           categories K -- gain in discrimination recovery from
                           the history-adjusted part vs the same change to
                           difficulty (control, ~0).
  s3_formula_distractor.png alpha_map_results.json: best discrimination recovery
                           per positive map, smooth group tied, non-smooth lower.
  s4_why_hard.png          fisher_ratio.json: the prediction's relative Fisher
                           leverage on discrimination, small and shrinking with
                           the number of categories K.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
OUT_DIR = _HERE / "figs"
OUT_DIR.mkdir(exist_ok=True)
SRC = _HERE.parent.parent / "deep_irt" / "bench" / "outputs"

DPI = 150

# Consistent parameter colours.
C_SCORE = "#1b7837"       # green (ability)
C_DIFF = "#2166ac"        # blue (difficulty)
C_SHARP = "#b2182b"       # red (discrimination)
C_CONTROL = "#9e9e9e"     # grey (control)

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "legend.fontsize": 17,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.linewidth": 1.2,
    "lines.linewidth": 3.0,
})


def _load(name: str) -> dict:
    with (SRC / name).open(encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# s1 -- three speeds of recovery
# ---------------------------------------------------------------------------

def fig_three_speeds() -> dict:
    k4 = _load("param_trajectory_K4.json")
    k8_path = SRC / "param_trajectory_K8.json"
    has_k8 = k8_path.exists()
    k8 = _load("param_trajectory_K8.json") if has_k8 else None

    panels = [("K = 4 categories", k4)]
    if has_k8:
        panels.append(("K = 8 categories", k8))

    fig, axes = plt.subplots(1, len(panels), figsize=(7.5 * len(panels), 6.2),
                             sharey=True)
    if len(panels) == 1:
        axes = [axes]

    headline = {}
    for ax, (title, blob) in zip(axes, panels):
        ep = blob["epochs"]
        for key, col, lab in (
            ("score", C_SCORE, "ability (theta)"),
            ("difficulty", C_DIFF, "difficulty (item)"),
            ("sharpness", C_SHARP, "discrimination (alpha)"),
        ):
            m = blob[f"{key}_mean"]
            ax.plot(ep, m, color=col, label=lab, marker="o", markersize=4)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)
        # record the K=4 panel headline: step where discrimination first reaches 0.8
        if "4" in title:
            sm = blob["sharpness_mean"]
            sh80 = next((e for e, v in zip(ep, sm) if v >= 0.8), ep[-1])
            sc80 = next((e for e, v in zip(ep, blob["score_mean"]) if v >= 0.8),
                        ep[-1])
            di80 = next((e for e, v in zip(ep, blob["difficulty_mean"])
                         if v >= 0.8), ep[-1])
            headline = {
                "score_reaches_0.8_at_step": sc80,
                "difficulty_reaches_0.8_at_step": di80,
                "sharpness_reaches_0.8_at_step": sh80,
                "sharpness_final": round(sm[-1], 3),
                "score_final": round(blob["score_mean"][-1], 3),
                "difficulty_final": round(blob["difficulty_mean"][-1], 3),
            }

    axes[0].set_ylabel("parameter recovery (Spearman rho)")
    axes[0].legend(loc="lower right", framealpha=0.95)
    fig.suptitle("Ability and difficulty are learned fast; discrimination lags",
                 fontsize=23, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT_DIR / "s1_three_speeds.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return {"path": out, "source": "param_trajectory_K4.json"
            + (" + param_trajectory_K8.json" if has_k8 else ""),
            "headline": headline}


# ---------------------------------------------------------------------------
# s2 -- the history-adjusted part helps sharpness, not difficulty
# ---------------------------------------------------------------------------

def fig_sharpness_help() -> dict:
    blob = _load("alpha_beta_asymmetry.json")
    rows = blob["summary"]
    Ks = [r["K"] for r in rows]
    d_sharp = [r["delta_alpha"] for r in rows]     # gain in discrimination recovery
    d_diff = [r["delta_beta"] for r in rows]       # same change to difficulty

    fig, ax = plt.subplots(figsize=(11, 6.2))
    x = range(len(Ks))
    w = 0.38
    b1 = ax.bar([i - w / 2 for i in x], d_sharp, w,
                color=C_SHARP, label="gain in discrimination recovery")
    b2 = ax.bar([i + w / 2 for i in x], d_diff, w,
                color=C_CONTROL, label="same change to difficulty (control)")
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(k) for k in Ks])
    ax.set_xlabel("number of categories K")
    ax.set_ylabel("change in recovery (Spearman rho)")
    ax.set_title("The history-adjusted part lifts discrimination, not difficulty")
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = OUT_DIR / "s2_sharpness_help.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)

    mean_sharp = sum(d_sharp) / len(d_sharp)
    mean_diff = sum(d_diff) / len(d_diff)
    # the largest answer-levels point makes the cleanest talk line
    kmax_i = Ks.index(max(Ks))
    return {"path": out, "source": "alpha_beta_asymmetry.json",
            "headline": {
                "mean_help_to_sharpness": round(mean_sharp, 3),
                "mean_change_to_difficulty_control": round(mean_diff, 3),
                f"help_to_sharpness_at_{max(Ks)}_levels": round(d_sharp[kmax_i], 3),
                f"control_at_{max(Ks)}_levels": round(d_diff[kmax_i], 3),
            }}


# ---------------------------------------------------------------------------
# s3 -- which positive formula recovers sharpness best
# ---------------------------------------------------------------------------

def fig_formula_distractor() -> dict:
    blob = _load("alpha_map_results.json")
    # best discrimination recovery (a_spearman_mean) per map across learning rates.
    best: dict[str, tuple[float, float]] = {}
    for r in blob["agg"]:
        m = r["map"]
        v = r.get("a_spearman_mean")
        if v is None:
            continue
        if m not in best or v > best[m][0]:
            best[m] = (v, r.get("a_spearman_std") or 0.0)

    # group exactly as the lead named them.
    smooth = ["exp", "clipped_exp", "softplus", "softplus_eps",
              "scaled_softplus", "temperature_softplus", "sigmoid",
              "scaled_sigmoid"]
    jagged = ["identity", "relu", "square"]
    pretty = {
        "exp": "exp", "clipped_exp": "exp (clipped)", "softplus": "softplus",
        "softplus_eps": "softplus+", "scaled_softplus": "softplus (scaled)",
        "temperature_softplus": "softplus (temp.)", "sigmoid": "sigmoid",
        "scaled_sigmoid": "sigmoid (scaled)", "identity": "raw",
        "relu": "relu", "square": "square",
    }

    order = [m for m in smooth if m in best] + [m for m in jagged if m in best]
    vals = [best[m][0] for m in order]
    errs = [best[m][1] for m in order]
    n_smooth = sum(1 for m in smooth if m in best)
    colors = ([C_SHARP] * n_smooth) + ([C_CONTROL] * (len(order) - n_smooth))
    labels = [pretty.get(m, m) for m in order]

    fig, ax = plt.subplots(figsize=(13, 6.4))
    x = range(len(order))
    ax.bar(x, vals, yerr=errs, color=colors, capsize=4,
           error_kw={"elinewidth": 1.5})
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("discrimination recovery (Spearman rho)")
    # zoom to where the action is
    lo = min(vals) - 0.02
    ax.set_ylim(max(0, lo), 1.0)
    ax.set_title("Smooth positive maps tie; non-smooth maps fall behind")
    ax.grid(True, axis="y", alpha=0.3)

    # group separator + legend
    sep = n_smooth - 0.5
    ax.axvline(sep, color="black", linestyle="--", linewidth=1.2, alpha=0.6)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_SHARP, label="smooth positive maps"),
                       Patch(color=C_CONTROL, label="non-smooth positive maps")],
              loc="lower left", framealpha=0.95)

    fig.tight_layout()
    out = OUT_DIR / "s3_formula_distractor.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)

    smooth_vals = [best[m][0] for m in smooth if m in best]
    jagged_vals = [best[m][0] for m in jagged if m in best]
    return {"path": out, "source": "alpha_map_results.json",
            "headline": {
                "best_smooth": round(max(smooth_vals), 3),
                "smooth_spread": round(max(smooth_vals) - min(smooth_vals), 3),
                "lowest_jagged": round(min(jagged_vals), 3),
                "lowest_jagged_name": pretty[
                    min((m for m in jagged if m in best),
                        key=lambda m: best[m][0])],
            }}


# ---------------------------------------------------------------------------
# s4 -- why sharpness is hard: small, shrinking prediction sensitivity
# ---------------------------------------------------------------------------

def fig_why_hard() -> dict:
    blob = _load("fisher_ratio.json")
    rows = blob["rows"]
    Ks = [r["K"] for r in rows]
    # relative Fisher leverage on discrimination = I_alpha / I_theta,
    # how much the prediction depends on discrimination compared with ability.
    rel = [r["I_alpha"] / r["I_theta"] for r in rows]

    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.plot(Ks, rel, color=C_SHARP, marker="o", markersize=10)
    for k, v in zip(Ks, rel):
        ax.annotate(f"{v:.2f}", (k, v), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=15)
    ax.set_xlabel("number of categories K")
    ax.set_ylabel("Fisher leverage on discrimination\nI(alpha)/I(theta)")
    ax.set_xticks(Ks)
    ax.set_ylim(0, max(rel) * 1.25)
    ax.set_title("The prediction has low Fisher leverage on discrimination,\nfalling "
                 "as the number of categories K grows")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = OUT_DIR / "s4_why_hard.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return {"path": out, "source": "fisher_ratio.json",
            "headline": {
                f"relative_sensitivity_at_{min(Ks)}_levels": round(rel[0], 3),
                f"relative_sensitivity_at_{max(Ks)}_levels": round(rel[-1], 3),
            }}


def main():
    print("source dir:", SRC)
    present = sorted(p.name for p in SRC.glob("*.json"))
    print(f"({len(present)} json files in outputs)")
    for need in ("param_trajectory_K4.json", "param_trajectory_K8.json",
                 "alpha_beta_asymmetry.json", "alpha_map_results.json",
                 "fisher_ratio.json"):
        print(f"  {'found' if need in present else 'MISSING'}: {need}")

    results = [fig_three_speeds(), fig_sharpness_help(),
               fig_formula_distractor(), fig_why_hard()]
    print("\n=== figures ===")
    for r in results:
        print(f"\n{r['path'].name}")
        print(f"  source : {r['source']}")
        for k, v in r["headline"].items():
            print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
