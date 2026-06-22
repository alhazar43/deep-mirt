"""make_map_convergence_fig.py -- wide convergence-profile figure for positive maps.

Reads deep_irt/bench/outputs/map_convergence_K{K}.json (written by
deep_irt/bench/run_map_convergence.py) and renders one curve per positive map
of discrimination recovery (Spearman rho) versus training epoch.  The smooth
maps (exp, softplus, sigmoid) are drawn in one warm family so their overlap is
visible at a glance; the non-smooth maps (relu, square) are drawn distinctly to
show that they lag early.

Headline the figure carries: smooth positive maps share one convergence
profile, non-smooth maps lag.

Usage
-----
  source ~/anaconda3/etc/profile.d/conda.sh && conda activate research
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=".;rl/src;ma-irt" \
      python docs/slides/make_map_convergence_fig.py [--K 4]

Output
------
  docs/slides/figs/j5_map_convergence.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_RESULTS = _REPO / "deep_irt" / "bench" / "outputs"
_FIGS = _HERE / "figs"
_FIGS.mkdir(exist_ok=True)

# Smooth family in one warm hue family (solid lines), non-smooth distinct
# (dashed/dotted, cool hues).  Pretty labels keep proper terms.
_STYLE = {
    "exp":      dict(color="#b2182b", ls="-", marker="o", label="exp (smooth)"),
    "softplus": dict(color="#ef8a62", ls="-", marker="s", label="softplus (smooth)"),
    "sigmoid":  dict(color="#f4a582", ls="-", marker="^", label="sigmoid (smooth)"),
    "relu":     dict(color="#2166ac", ls="--", marker="D", label="relu (non-smooth)"),
    "square":   dict(color="#4393c3", ls=":", marker="v", label="square (non-smooth)"),
}
_ORDER = ["exp", "softplus", "sigmoid", "relu", "square"]


def render(K: int) -> Path:
    src = _RESULTS / f"map_convergence_K{K}.json"
    if not src.exists():
        raise SystemExit(
            f"missing {src}; run deep_irt/bench/run_map_convergence.py first.")
    blob = json.loads(src.read_text(encoding="utf-8"))
    epochs = blob["epochs"]
    maps = blob["maps"]

    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 19,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })

    fig, ax = plt.subplots(figsize=(13, 5.5))

    for name in _ORDER:
        if name not in maps:
            continue
        st = _STYLE[name]
        mean = maps[name]["a_spearman_mean"]
        std = maps[name]["a_spearman_std"]
        lo = [m - s for m, s in zip(mean, std)]
        hi = [m + s for m, s in zip(mean, std)]
        ax.plot(epochs, mean, color=st["color"], ls=st["ls"],
                marker=st["marker"], markersize=6, linewidth=2.6,
                label=st["label"], zorder=3)
        ax.fill_between(epochs, lo, hi, color=st["color"], alpha=0.10, zorder=1)

    ax.axhline(0.8, color="#888888", ls=(0, (2, 4)), lw=1.2, zorder=0)
    ax.text(epochs[-1], 0.805, "rho = 0.8", ha="right", va="bottom",
            fontsize=12, color="#555555")

    ax.set_xlabel("training epoch")
    ax.set_ylabel("discrimination recovery (Spearman rho)")
    ax.set_title(
        "Smooth positive maps share one convergence profile; "
        "non-smooth maps lag")
    ax.set_xlim(min(epochs), max(epochs))
    ax.set_ylim(-0.05, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", ncol=2, frameon=True, framealpha=0.9)

    seeds = blob["meta"].get("seeds", [])
    ax.text(0.01, 0.98,
            f"K={blob['meta'].get('K', K)}, {len(seeds)} seeds, matched "
            f"effective-alpha init, lr={blob['meta'].get('lr')}",
            transform=ax.transAxes, ha="left", va="top", fontsize=12,
            color="#555555")

    fig.tight_layout()
    out = _FIGS / "j5_map_convergence.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=4)
    args = ap.parse_args()
    out = render(args.K)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
