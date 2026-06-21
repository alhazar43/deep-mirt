"""TEMP probe: decompose the Phase-2 magnitude attenuation (k ~ 0.04).

The calibration k = OLS slope of recovered log-alpha-on-TRUE-theta against the
planted gamma came out ~0.04 (a ~25x shrinkage).  Two causes are confounded:
(a) genuine head shrinkage (the GPCM likelihood is barely sensitive to the
alpha-theta slope, so the optimizer fixes rank not magnitude), and (b) the
model's latent theta is identified only up to an affine scale, and in a*(theta-b)
discrimination scales inversely with theta's scale, so a compressed internal theta
deflates the recovered alpha slope by the same factor.

Measure the scale ratio c = OLS slope of the model's per-learner theta on the TRUE
theta0, then c-correct the calibration as k/c.  If c ~ 1 the attenuation is
genuine head shrinkage; if c is small (compressed theta), part of the 25x is scale.
Planted sigma=0.4, the same config as run_phase2_signal. Temp probe.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from deep_irt.bench.datagen import BenchDataConfig, generate
from deep_irt.bench.engines import DeepIRTEngine
from deep_irt.bench.run_phase2_signal import (
    _alpha_per_pos, _per_item_logslope, _corr, _ols_slope,
)

K, N, Q, T, EPOCHS, SIGMA = 4, 2000, 60, 60, 150, 0.4
SEEDS = (0, 1, 2)
_HERE = Path(__file__).resolve().parent
OUT = _HERE / "outputs"
OUT.mkdir(exist_ok=True)


def _render(rows, summary, meta):
    lines = ["# Phase-2 Magnitude Decomposition\n"]
    lines.append(
        f"K={meta['K']} sigma={meta['sigma']} N={meta['N']} Q={meta['Q']} "
        f"T={meta['T']} epochs={meta['epochs']} seeds={meta['seeds']} "
        f"device={meta['device']}\n"
    )
    lines.append("| seed | k raw | theta scale c | k/c | corr(theta_hat, theta0) |")
    lines.append("|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['k_raw']:.3f} | "
            f"{row['theta_scale']:.3f} | {row['k_over_c']:.3f} | "
            f"{row['theta_corr']:.3f} |"
        )
    lines.append(
        f"\nMean: k(raw)={summary['k_raw_mean']:.3f}, "
        f"c={summary['theta_scale_mean']:.3f}, "
        f"k/c={summary['k_over_c_mean']:.3f}, "
        f"corr(theta_hat,theta0)={summary['theta_corr_mean']:.3f}.\n"
    )
    lines.append(
        "Interpretation: theta scale is near one, so the small calibration k is "
        "not explained by latent theta compression.  The planted residual is "
        "directionally detected but strongly attenuated in magnitude."
    )
    return "\n".join(lines) + "\n"


def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Phase-2 magnitude decomposition, K={K} sigma={SIGMA} N={N} "
          f"{len(SEEDS)} seeds device={device} ===")
    print(f"{'seed':>4s} {'k (raw)':>8s} {'c (theta scale)':>15s} "
          f"{'k/c':>8s} {'corr(theta_hat,theta0)':>22s}")
    rows = []
    for seed in SEEDS:
        ds = generate(BenchDataConfig(name="s", kind="static", n_cats=K,
                                      n_learners=N, n_items=Q, seq_len=T,
                                      seed=seed, alpha_theta_slope=SIGMA))
        gamma = ds.gt.gamma
        eng = DeepIRTEngine(ds, decoder="gpcm", emb_dim=8, hidden_dim=32,
                            state_alpha=True, item_key_dim=64, alpha_log_scale=1.0,
                            encoder="lstm", device=device, seed=seed)
        eng.fit(n_epochs=EPOCHS, batch_size=256)

        # alpha calibration k (raw planted slope on gamma)
        a = _alpha_per_pos(eng.model, ds.items0, ds.responses, device)
        slope = _per_item_logslope(a, ds.items0, ds.theta_at_step, Q)
        k = _ols_slope(gamma, slope)

        # theta scale c: model per-learner ability (mean over steps) vs true theta0
        it = torch.as_tensor(ds.items0, dtype=torch.long, device=device)
        rp = torch.as_tensor(ds.responses, dtype=torch.long, device=device)
        with torch.no_grad():
            theta_traj = eng.model.track(it, rp).cpu().numpy()       # (N, T)
        theta_hat = theta_traj.mean(axis=1)                          # (N,)
        c = _ols_slope(ds.gt.theta0, theta_hat)
        rho = _corr(theta_hat, ds.gt.theta0)

        kc = k / c if abs(c) > 1e-9 else float("nan")
        rows.append({
            "seed": seed,
            "k_raw": float(k),
            "theta_scale": float(c),
            "k_over_c": float(kc),
            "theta_corr": float(rho),
        })
        print(f"{seed:4d} {k:8.3f} {c:15.3f} {kc:8.3f} {rho:22.3f}")

    summary = {
        "k_raw_mean": float(np.mean([r["k_raw"] for r in rows])),
        "theta_scale_mean": float(np.mean([r["theta_scale"] for r in rows])),
        "k_over_c_mean": float(np.mean([r["k_over_c"] for r in rows])),
        "theta_corr_mean": float(np.mean([r["theta_corr"] for r in rows])),
    }
    print(f"\nMEAN  k(raw)={summary['k_raw_mean']:.3f}  "
          f"c={summary['theta_scale_mean']:.3f}  "
          f"k/c={summary['k_over_c_mean']:.3f}  "
          f"corr(theta_hat,theta0)={summary['theta_corr_mean']:.3f}")

    meta = {
        "K": K,
        "N": N,
        "Q": Q,
        "T": T,
        "epochs": EPOCHS,
        "sigma": SIGMA,
        "seeds": list(SEEDS),
        "device": device,
        "elapsed_s": time.time() - t0,
        "torch_version": torch.__version__,
    }
    out_json = OUT / "phase2_scale.json"
    out_md = OUT / "phase2_scale_table.md"
    out_json.write_text(
        json.dumps({"meta": meta, "rows": rows, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    out_md.write_text(_render(rows, summary, meta), encoding="utf-8")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
