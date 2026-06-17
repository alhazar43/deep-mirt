"""Generate the remaining slide data in one GPU pass:
  Part 1  RQ3 detection K-sweep (corr(signal,gamma) + calibration k) at K in {2,4,8}.
  Part 2  endpoint recovery (Pearson + scale-aligned RMSE) for the dynamic-alpha
          decoupled model, the Tsutsumi-dimension defense.

Scale-aligned RMSE: the prediction-trained latent scale is unidentified, so raw RMSE
is dominated by an arbitrary affine. We report RMSE after the optimal affine map of
the recovered onto the truth, which equals std(true)*sqrt(1-r^2); that is the honest
best-case error on the truth's own scale and is itself on-message (prediction fixes
shape, not absolute scale).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from deep_irt.bench.datagen import BenchDataConfig, generate
from deep_irt.bench.engines import DeepIRTEngine
from deep_irt.bench import metrics_bench as M
from deep_irt.bench.run_phase2_signal import (
    _alpha_per_pos, _per_item_logslope, _corr, _ols_slope,
)

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "outputs"

EPOCHS = 150
N = 2000
Q = 60
T = 60
SEEDS = (0, 1, 2)


def _aligned_rmse(rec, true):
    """RMSE after optimal affine alignment = std(true)*sqrt(1-r^2)."""
    rec = np.asarray(rec, float); true = np.asarray(true, float)
    r = M._pearson(M._align_sign(rec, true), true)
    return float(np.std(true) * np.sqrt(max(0.0, 1.0 - r * r)))


def rq3_ksweep():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== RQ3 detection K-sweep (device={device}) ===")
    sigmas = [0.2, 0.4]
    out = {"sigmas": sigmas, "seeds": list(SEEDS), "per_K": {}}
    for K in (2, 4, 8):
        per_sigma = {s: {"corr_signal": [], "calib_k": [], "corr_null": []}
                     for s in sigmas}
        for seed in SEEDS:
            common = dict(name="s", kind="static", n_cats=K, n_learners=N,
                          n_items=Q, seq_len=T, seed=seed)
            null = generate(BenchDataConfig(**common, alpha_theta_slope=0.0))
            eng = DeepIRTEngine(null, decoder="gpcm", emb_dim=8, hidden_dim=32,
                                state_alpha=True, item_key_dim=64,
                                alpha_log_scale=1.0, encoder="lstm",
                                device=device, seed=seed)
            eng.fit(n_epochs=EPOCHS, batch_size=256)
            s_null = _per_item_logslope(
                _alpha_per_pos(eng.model, null.items0, null.responses, device),
                null.items0, null.theta_at_step, Q)
            for sigma in sigmas:
                plant = generate(BenchDataConfig(**common, alpha_theta_slope=sigma))
                gamma = plant.gt.gamma
                e2 = DeepIRTEngine(plant, decoder="gpcm", emb_dim=8, hidden_dim=32,
                                   state_alpha=True, item_key_dim=64,
                                   alpha_log_scale=1.0, encoder="lstm",
                                   device=device, seed=seed)
                e2.fit(n_epochs=EPOCHS, batch_size=256)
                s_plant = _per_item_logslope(
                    _alpha_per_pos(e2.model, plant.items0, plant.responses, device),
                    plant.items0, plant.theta_at_step, Q)
                signal = s_plant - s_null
                per_sigma[sigma]["corr_signal"].append(_corr(signal, gamma))
                per_sigma[sigma]["calib_k"].append(_ols_slope(gamma, signal))
                per_sigma[sigma]["corr_null"].append(_corr(s_null, gamma))
        agg = {}
        for sigma in sigmas:
            d = per_sigma[sigma]
            agg[str(sigma)] = {k: float(np.mean(v)) for k, v in d.items()}
        out["per_K"][str(K)] = agg
        print(f"  K={K}: " + "  ".join(
            f"sig{sigma} corr={agg[str(sigma)]['corr_signal']:+.3f} "
            f"k={agg[str(sigma)]['calib_k']:.3f}" for sigma in sigmas))
    (OUT / "slides_rq3_ksweep.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT/'slides_rq3_ksweep.json'}")


def endpoint_recovery():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== endpoint recovery (dynamic-alpha decoupled, device={device}) ===")
    out = {"seeds": list(SEEDS), "per_K": {}}
    for K in (2, 4):
        acc = {p: {"pearson": [], "rmse": []} for p in ("a", "b", "theta")}
        for seed in SEEDS:
            ds = generate(BenchDataConfig(name="s", kind="static", n_cats=K,
                                          n_learners=N, n_items=Q, seq_len=T,
                                          seed=seed))
            eng = DeepIRTEngine(ds, decoder="gpcm", emb_dim=8, hidden_dim=32,
                                state_alpha=True, item_key_dim=64,
                                alpha_log_scale=1.0, encoder="lstm",
                                device=device, seed=seed)
            eng.fit(n_epochs=EPOCHS, batch_size=256)
            rec = eng.recover()
            seen = rec.get("seen")
            a_hat, a_true = np.asarray(rec["a"]), ds.gt.a
            b_hat, b_true = np.asarray(rec["b"]), ds.gt.b
            th_hat, th_true = rec["theta_final"], ds.gt.theta0
            if seen is not None:
                a_hat, a_true = a_hat[seen], a_true[seen]
                b_hat, b_true = b_hat[seen], b_true[seen]
            acc["a"]["pearson"].append(M._pearson(M._align_sign(a_hat, a_true), a_true))
            acc["a"]["rmse"].append(_aligned_rmse(a_hat, a_true))
            acc["b"]["pearson"].append(
                M._pearson(M._align_sign(b_hat.ravel(), b_true.ravel()), b_true.ravel()))
            acc["b"]["rmse"].append(_aligned_rmse(b_hat.ravel(), b_true.ravel()))
            acc["theta"]["pearson"].append(M._pearson(M._align_sign(th_hat, th_true), th_true))
            acc["theta"]["rmse"].append(_aligned_rmse(th_hat, th_true))
        agg = {p: {"pearson": float(np.mean(acc[p]["pearson"])),
                   "rmse": float(np.mean(acc[p]["rmse"]))} for p in acc}
        out["per_K"][str(K)] = agg
        print(f"  K={K}: " + "  ".join(
            f"{p} r={agg[p]['pearson']:.3f} rmse={agg[p]['rmse']:.3f}" for p in agg))
    (OUT / "slides_endpoint.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT/'slides_endpoint.json'}")


if __name__ == "__main__":
    rq3_ksweep()
    endpoint_recovery()
