"""Build the slide figures from the on-disk JSON results.
  fig_convergence.png  RQ2: K=11 trajectory (acceleration) + endpoint gap vs K
  fig_rq3.png          RQ3: dose-response at K=4 + K-trend (corr flat, k falls)
  fig_endpoint.png     endpoint recovery Pearson r per parameter
The Fisher spine (fisher_ratio.png) is built by run_fisher_ratio.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "outputs"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "figure.dpi": 150})
C_STAT, C_DYN = "#888888", "#d62728"


def convergence():
    Ks = [4, 8, 11]
    data = {}
    for K in Ks:
        p = OUT / f"convergence_K{K}.json"
        if p.exists():
            data[K] = json.loads(p.read_text())
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # Left: K=11 trajectory, static vs dynamic, with std band
    d = data[11]
    ep = np.array(d["epochs"], float)
    sm, ss = np.array(d["alpha_static_mean"]), np.array(d["alpha_static_std"])
    dm, dd = np.array(d["alpha_dynamic_mean"]), np.array(d["alpha_dynamic_std"])
    axL.plot(ep, sm, color=C_STAT, lw=2, label="static alpha")
    axL.fill_between(ep, sm - ss, sm + ss, color=C_STAT, alpha=0.18)
    axL.plot(ep, dm, color=C_DYN, lw=2, label="dynamic alpha")
    axL.fill_between(ep, dm - dd, dm + dd, color=C_DYN, alpha=0.18)
    axL.set_xlabel("epoch"); axL.set_ylabel("alpha recovery (Pearson r)")
    axL.set_title("K=11: mid-training acceleration,\nstatic trapped by the stiffness ceiling")
    axL.legend(loc="lower right"); axL.set_ylim(0, 1)

    # Right: endpoint gap (dynamic - static) vs K
    gaps, ks = [], []
    for K in Ks:
        d = data[K]
        gaps.append(d["alpha_dynamic_mean"][-1] - d["alpha_static_mean"][-1])
        ks.append(K)
    axR.bar([str(k) for k in ks], gaps, color=C_DYN, alpha=0.85, width=0.55)
    for x, g in zip(range(len(ks)), gaps):
        axR.text(x, g + 0.004, f"+{g:.2f}", ha="center", fontsize=11)
    axR.set_xlabel("number of categories K")
    axR.set_ylabel("endpoint gap  (dynamic - static)")
    axR.set_title("Endpoint advantage grows with K")
    axR.set_ylim(0, max(gaps) * 1.25)
    fig.tight_layout(); fig.savefig(OUT / "fig_convergence.png"); plt.close(fig)
    print("wrote fig_convergence.png  gaps", [f"{g:+.3f}" for g in gaps])


def rq3():
    d = json.loads((OUT / "slides_rq3_ksweep.json").read_text())
    sigmas = d["sigmas"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # Left: dose-response at K=4
    k4 = d["per_K"]["4"]
    corrs = [k4[str(s)]["corr_signal"] for s in sigmas]
    nulls = [k4[str(s)]["corr_null"] for s in sigmas]
    x = np.arange(len(sigmas))
    axL.bar(x - 0.18, corrs, 0.36, color=C_DYN, label="corr(signal, gamma)")
    axL.bar(x + 0.18, nulls, 0.36, color=C_STAT, label="corr(null, gamma)")
    axL.axhline(0, color="0.6", lw=0.8)
    axL.set_xticks(x); axL.set_xticklabels([f"sigma={s}" for s in sigmas])
    axL.set_ylabel("correlation with planted gamma")
    axL.set_title("K=4: detection rises with planted strength,\nnull stays ~0")
    axL.legend(loc="upper left"); axL.set_ylim(-0.1, 0.8)

    # Right: K-trend at sigma=0.4, corr (flat) vs calib k (falls)
    Ks = sorted(int(k) for k in d["per_K"])
    s = str(max(sigmas))
    corr_k = [d["per_K"][str(K)][s]["corr_signal"] for K in Ks]
    calib = [d["per_K"][str(K)][s]["calib_k"] for K in Ks]
    axR.plot(Ks, corr_k, "o-", color=C_DYN, lw=2, label="detection corr (rank)")
    axR.set_xlabel("number of categories K")
    axR.set_ylabel("detection corr(signal, gamma)", color=C_DYN)
    axR.tick_params(axis="y", labelcolor=C_DYN); axR.set_ylim(0, 0.8)
    ax2 = axR.twinx()
    ax2.plot(Ks, calib, "s--", color="#1f77b4", lw=2, label="calibration k (magnitude)")
    ax2.set_ylabel("calibration k  (magnitude)", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4"); ax2.set_ylim(0, max(calib) * 1.4)
    axR.set_title(f"sigma={s}: rank detection holds,\nmagnitude calibration degrades with K")
    fig.tight_layout(); fig.savefig(OUT / "fig_rq3.png"); plt.close(fig)
    print("wrote fig_rq3.png  corr_k", [f"{c:.2f}" for c in corr_k],
          "calib", [f"{c:.3f}" for c in calib])


def endpoint():
    d = json.loads((OUT / "slides_endpoint.json").read_text())
    # Headline the regime the dynamic readout is indicated for (K>=4); the K=2
    # dynamic-alpha dip is the documented low-stiffness weak spot (use static there).
    Ks = [4]
    params = ["theta", "a", "b"]
    labels = {"theta": "ability theta", "a": "discrimination alpha",
              "b": "thresholds beta"}
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    x = np.arange(len(params))
    w = 0.8 / len(Ks)
    for i, K in enumerate(Ks):
        vals = [d["per_K"][str(K)][p]["pearson"] for p in params]
        bars = ax.bar(x + (i - (len(Ks) - 1) / 2) * w, vals, w, label=f"K={K}")
        for b_, v in zip(bars, vals):
            ax.text(b_.get_x() + b_.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([labels[p] for p in params])
    ax.set_ylabel("endpoint recovery (Pearson r)")
    ax.set_ylim(0, 1.12)
    ax.set_title("Endpoint recovery, K=4 (dynamic-alpha decoupled)")
    fig.tight_layout(); fig.savefig(OUT / "fig_endpoint.png"); plt.close(fig)
    print("wrote fig_endpoint.png")


if __name__ == "__main__":
    convergence()
    rq3()
    endpoint()
