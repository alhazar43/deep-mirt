"""analyze_e0.py -- diagnostics and plots for the E0 sweep.

Reads outputs/results_e0.json and asks three questions beyond the summary
table. How does recovery scale with density and response format. How does
the encoder compare to the per-respondent ML reference (oracle item
params). And, crucially, is the modest recovery a uniform limit or is it
concentrated in an unidentifiable regime (fast learners whose curve
saturates within the sequence, leaving the rate unobserved). Writes a
markdown summary and two plots to outputs/.
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "outputs")


def _arr(xs):
    return np.array([np.nan if x is None else x for x in xs], dtype=float)


def tercile_diagnostics(r_true, r_enc):
    """Recovery within low/mid/high true-rate terciles."""
    ok = np.isfinite(r_true) & np.isfinite(r_enc)
    r_true, r_enc = r_true[ok], r_enc[ok]
    order = np.argsort(r_true)
    thirds = np.array_split(order, 3)
    rows = []
    for name, idx in zip(("low", "mid", "high"), thirds):
        rt, re = r_true[idx], r_enc[idx]
        rho = spearmanr(rt, re)[0] if len(idx) > 3 else np.nan
        rows.append({
            "band": name,
            "r_range": (float(rt.min()), float(rt.max())),
            "median_abs_err": float(np.median(np.abs(re - rt))),
            "bias": float(np.median(re - rt)),
            "within_spearman": float(rho),
        })
    return rows


def main():
    with open(os.path.join(OUT, "results_e0.json")) as f:
        res = json.load(f)

    lines = ["# E0 analysis: ability-trajectory rate recovery", ""]
    cfg = res["config"]
    lines.append(f"Config: N={cfg['n_resp']}, items={cfg['n_items']}, "
                 f"epochs={cfg['n_epochs']}, seeds={cfg['seeds']}, "
                 f"device={cfg['device']}. Curve {cfg['curve']}.")
    lines.append("")

    # Density x format table
    lines.append("## Density and format")
    lines.append("")
    lines.append("| fmt | T | enc rho | enc std | oracle rho | per-step | enc MAE |")
    lines.append("|---|---|---|---|---|---|---|")
    by = {}
    for c in res["conditions"]:
        e = c["enc"]
        orc = c["oracle_seed0"]["spearman"] if c["oracle_seed0"] else float("nan")
        by[(c["format"], c["seq_len"])] = c
        lines.append(f"| {c['format']} | {c['seq_len']} | "
                     f"{e['spearman_mean']:.3f} | {e['spearman_std']:.3f} | "
                     f"{orc:.3f} | {e['perstep_corr_mean']:.3f} | "
                     f"{e['median_abs_err_mean']:.3f} |")
    lines.append("")

    # Tercile diagnostics on the richest condition with arrays
    lines.append("## Rate-band diagnostics (is recovery regime-dependent?)")
    lines.append("")
    for fmt in ("binary", "gpcm"):
        for T in (cfg["seq_lens"][-1],):  # highest density
            c = by.get((fmt, T))
            h = c.get("headline_arrays_seed0") if c else None
            if not h:
                continue
            rt = _arr(h["r_true"])
            re = _arr(h["r_enc"])
            rows = tercile_diagnostics(rt, re)
            lines.append(f"**{fmt}, T={T} (seed 0)** overall "
                         f"spearman={spearmanr(rt[np.isfinite(re)], re[np.isfinite(re)])[0]:.3f}")
            lines.append("")
            lines.append("| band | r range | median abs err | bias | within-band spearman |")
            lines.append("|---|---|---|---|---|")
            for r in rows:
                lines.append(f"| {r['band']} | "
                             f"[{r['r_range'][0]:.3f}, {r['r_range'][1]:.3f}] | "
                             f"{r['median_abs_err']:.3f} | {r['bias']:+.3f} | "
                             f"{r['within_spearman']:.3f} |")
            lines.append("")

    summary_md = os.path.join(OUT, "E0_analysis.md")
    with open(summary_md, "w") as f:
        f.write("\n".join(lines))

    # Plot 1: recovery vs density, both formats, enc + oracle
    fig, ax = plt.subplots(figsize=(5, 4))
    for fmt, marker in (("binary", "o"), ("gpcm", "s")):
        Ts, encs, encsd, orcs = [], [], [], []
        for T in cfg["seq_lens"]:
            c = by[(fmt, T)]
            Ts.append(T)
            encs.append(c["enc"]["spearman_mean"])
            encsd.append(c["enc"]["spearman_std"])
            orcs.append(c["oracle_seed0"]["spearman"] if c["oracle_seed0"] else np.nan)
        ax.errorbar(Ts, encs, yerr=encsd, marker=marker, capsize=3,
                    label=f"{fmt} encoder")
        ax.plot(Ts, orcs, marker=marker, linestyle="--", alpha=0.6,
                label=f"{fmt} ML ref")
    ax.set_xlabel("sequence density T")
    ax.set_ylabel("rate recovery (Spearman)")
    ax.set_title("E0: rate recovery vs density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "e0_recovery_vs_density.png"), dpi=130)
    plt.close(fig)

    # Plot 2: scatter r_true vs r_enc for the richest gpcm condition
    c = by[("gpcm", cfg["seq_lens"][-1])]
    h = c.get("headline_arrays_seed0")
    if h:
        rt, re = _arr(h["r_true"]), _arr(h["r_enc"])
        ok = np.isfinite(rt) & np.isfinite(re)
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.scatter(rt[ok], re[ok], s=12, alpha=0.5)
        lim = [0, max(rt[ok].max(), re[ok].max()) * 1.05]
        ax.plot(lim, lim, "k--", alpha=0.5)
        ax.set_xlabel("true rate r")
        ax.set_ylabel("recovered rate r_hat")
        ax.set_title(f"gpcm T={cfg['seq_lens'][-1]} (seed 0)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "e0_rate_scatter_gpcm.png"), dpi=130)
        plt.close(fig)

    print("\n".join(lines))
    print(f"\nwrote {summary_md}, e0_recovery_vs_density.png, e0_rate_scatter_gpcm.png")


if __name__ == "__main__":
    main()
