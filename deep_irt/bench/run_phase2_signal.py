"""run_phase2_signal.py -- RQ3 Phase 2: can the state-conditioned discrimination
head DETECT genuine theta-dependent discrimination, above the Fisher-tail null bias?

Design (matched null, external theta)
-------------------------------------
At each seed a NULL dataset (alpha_theta_slope=0, static alpha) and one or more
PLANTED datasets (alpha_eff = a*exp(gamma_j*theta)) are generated.  Because gamma
is drawn from a SEPARATE rng (datagen.py), the null and every planted set at the
same seed share identical a, b, theta, item sequences and per-step choice draws;
only the responses differ.  So the null is a matched bias control, and at a given
seed the planted gammas at different sigmas are PROPORTIONAL (same standard-normal
draws scaled by sigma), a clean dose-response on the same items.

For each fit, every item's per-occurrence state-conditioned alpha_jt is read and
regressed (OLS) on the TRUE theta at that position -- external, so no
latent-variable circularity.  The per-item slope estimates log-alpha's linear
dependence on theta, which is EXACTLY the planted form (log alpha = log a +
gamma*theta).  The Fisher-tail null bias (ADYNAMIC_STUDY Result 2) is non-monotone
in theta and nearly cancels under this linear projection, so it barely contaminates
the slope; the matched-null difference removes what remains.

  signal_j = slope_planted_j - slope_null_j
  detection = corr(signal_j, gamma_true_j)        (rank/direction quality)
  calibration k = OLS slope of signal on gamma    (k=1 => exact magnitude)

Sanity: corr(slope_null, gamma) ~ 0 (the null bias is independent of the planted
gamma).  null_std reports the bias magnitude on this linear estimator.

Outputs: deep_irt/bench/outputs/phase2_signal.json (summary + per-item slopes).

Usage
-----
  source ~/anaconda3/etc/profile.d/conda.sh && conda activate research
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=".;rl/src;ma-irt" \
      python deep_irt/bench/run_phase2_signal.py \
          [--K 4] [--sigmas 0.2 0.4] [--seeds 0 1 2] [--N 2000] [--epochs 150]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr

from deep_irt.bench.datagen import BenchDataConfig, generate
from deep_irt.bench.engines import DeepIRTEngine

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "outputs"
OUT.mkdir(exist_ok=True)


@torch.no_grad()
def _alpha_per_pos(model, items, resp, device):
    """Per-(learner,step) state-conditioned alpha_jt, shape (N, T).

    The decoupled model reads discrimination from the prediction-aligned state
    plus the wide item key; this replays that read over every answered position.
    """
    enc, dec = model.encoder, model.decoder
    it = torch.as_tensor(items, dtype=torch.long, device=device)
    rp = torch.as_tensor(resp, dtype=torch.long, device=device)
    _, state = enc.aligned_theta_and_state(it, rp)
    Nn, Tt = it.shape
    H = state.shape[-1]
    val = enc.item_val_emb(it)
    key = enc.item_key_emb(it)
    p = dec.item_params(val.reshape(Nn * Tt, -1), state=state.reshape(Nn * Tt, H),
                        item_key=key.reshape(Nn * Tt, -1))
    return p["alpha"].reshape(Nn, Tt).cpu().numpy()


def _per_item_logslope(a, items, theta, Q):
    """OLS slope of log(alpha_jt) on the TRUE theta, per item.

    The planted signal log alpha = log a + gamma*theta is exactly linear, so the
    linear slope is its matched estimator; the nonlinear Fisher-tail null bias is
    non-monotone in theta and nearly cancels under this linear projection.
    """
    la = np.log(np.clip(a.ravel(), 1e-6, None))
    th = theta.ravel().astype(np.float64)
    fi = items.ravel()
    slope = np.full(Q, np.nan)
    for j in range(Q):
        m = fi == j
        if m.sum() < 10:
            continue
        x = th[m] - th[m].mean()
        denom = (x ** 2).sum()
        if denom < 1e-9:
            continue
        slope[j] = (x * (la[m] - la[m].mean())).sum() / denom
    return slope


def _train_slopes(ds, device, seed, K, Q, epochs):
    eng = DeepIRTEngine(ds, decoder="gpcm", emb_dim=8, hidden_dim=32,
                        state_alpha=True, item_key_dim=64, alpha_log_scale=1.0,
                        encoder="lstm", device=device, seed=seed)
    eng.fit(n_epochs=epochs, batch_size=256)
    a = _alpha_per_pos(eng.model, ds.items0, ds.responses, device)
    return _per_item_logslope(a, ds.items0, ds.theta_at_step, Q)


def _corr(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return float(pearsonr(x[m], y[m])[0])


def _ols_slope(x, y):
    """Slope k of y on x (k=1 => perfect magnitude recovery)."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.2, 0.4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--Q", type=int, default=60)
    ap.add_argument("--T", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    K, Q = args.K, args.Q

    print(f"=== RQ3 Phase 2 signal detection: K={K} N={args.N} Q={Q} T={args.T} "
          f"epochs={args.epochs} sigmas={args.sigmas} seeds={args.seeds} "
          f"device={device} ===")

    t0 = time.time()
    # One NULL train per seed (slope=0 is sigma-independent), reused across sigmas.
    per_sigma = {s: [] for s in args.sigmas}
    records = []
    for seed in args.seeds:
        common = dict(name="s", kind="static", n_cats=K, n_learners=args.N,
                      n_items=Q, seq_len=args.T, seed=seed)
        null = generate(BenchDataConfig(**common, alpha_theta_slope=0.0))
        s_null = _train_slopes(null, device, seed, K, Q, args.epochs)
        for sigma in args.sigmas:
            plant = generate(BenchDataConfig(**common, alpha_theta_slope=sigma))
            gamma = plant.gt.gamma
            s_plant = _train_slopes(plant, device, seed, K, Q, args.epochs)
            signal = s_plant - s_null
            row = dict(
                seed=seed, sigma=sigma,
                corr_plant=_corr(s_plant, gamma),
                corr_signal=_corr(signal, gamma),
                corr_null=_corr(s_null, gamma),
                calib_k=_ols_slope(gamma, signal),
                null_std=float(np.nanstd(s_null)),
            )
            per_sigma[sigma].append(row)
            records.append({**row, "gamma": gamma.tolist(),
                            "slope_plant": s_plant.tolist(),
                            "slope_null": s_null.tolist()})
            print(f"  seed {seed} sigma {sigma:.2f}: "
                  f"corr(plant,g)={row['corr_plant']:+.3f}  "
                  f"corr(signal,g)={row['corr_signal']:+.3f}  "
                  f"k={row['calib_k']:.3f}")

    print(f"\n{'sigma':>5s} {'corr(plant,g)':>13s} {'corr(signal,g)':>14s} "
          f"{'corr(null,g)':>12s} {'calib_k':>8s} {'null_std':>8s}")
    summary = {}
    for sigma in args.sigmas:
        rows = per_sigma[sigma]
        cp = float(np.mean([r["corr_plant"] for r in rows]))
        cs = float(np.mean([r["corr_signal"] for r in rows]))
        cn = float(np.mean([r["corr_null"] for r in rows]))
        ck = float(np.mean([r["calib_k"] for r in rows]))
        ns = float(np.mean([r["null_std"] for r in rows]))
        summary[str(sigma)] = dict(corr_plant=cp, corr_signal=cs, corr_null=cn,
                                   calib_k=ck, null_std=ns)
        print(f"{sigma:5.2f} {cp:+13.3f} {cs:+14.3f} {cn:+12.3f} "
              f"{ck:8.3f} {ns:8.3f}")

    blob = {"meta": {"K": K, "N": args.N, "Q": Q, "T": args.T,
                     "epochs": args.epochs, "sigmas": args.sigmas,
                     "seeds": args.seeds, "device": device},
            "summary": summary, "records": records}
    path = OUT / "phase2_signal.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)
    print(f"\nwall {time.time() - t0:.1f}s   wrote {path}")


if __name__ == "__main__":
    main()
