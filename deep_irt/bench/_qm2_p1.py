"""_qm2_p1.py -- C1: certify the frozen bank against MISSPECIFICATION.

The review's convergent demand (plan v1.1): matched-generator recovery is
self-fulfilling; certify calibration under generator MISMATCH instead.

Bed: mixed schedule (T=90, decouple 1.0), g_true=0 (no transfer; calibration
study), N=400, C=3, K=5, J=36. Generating twins:
  matched | nonmono | hetrate | forget | infosched   (see _qm2_datagen)
Calibration arms per twin:
  dynamic      stage-1 joint fit (items + moving person model)
  static_early anchor-first: static-person fit on the first W steps only
  oracle       per-item refit on TRUE theta (ceiling / reference)
Readouts per cell: alpha rank recovery + attenuation slope, d location
recovery, the position-bias detector (Debeer-Janssen artifact: admin position
vs location bias), and the refit discrepancy delta (dynamic arm; tau is
re-derived from the distribution across healthy vs mismatched cells).

JSON per cell -> outputs/qm2/p1/cell_<kind>_<calib>_<ds>.json, summary ->
p1_summary.json. Run from repo root: python deep_irt/bench/_qm2_p1.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _qm2_datagen import generate                                   # noqa: E402
from _qm2_model import SimpleStructureGPCM, refit_items             # noqa: E402
from _qm2_metrics import (d_recovery, item_recovery, position_bias,  # noqa: E402
                          seed_agg, spearman, within_learner)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "outputs", "qm2", "p1")
KINDS = ["matched", "nonmono", "hetrate", "forget", "infosched"]


def run_cell(kind, calib, ds_seed, N, T, W, e1, device):
    ds = generate(ds_seed, kind=kind, N=N, g_true=None, schedule="mixed",
                  T_mixed=T)
    items = ds["items"]
    seq_eff = torch.as_tensor(ds["seq_eff"], dtype=torch.long, device=device)
    resp = torch.as_tensor(ds["resp"], dtype=torch.long, device=device)

    torch.manual_seed(0)
    model = SimpleStructureGPCM(items["J"], ds["config"]["C"], items["K"], N,
                                items["concept"],
                                inert_items=ds["inert_items"]).to(device)

    out = {"kind": kind, "calib": calib, "data_seed": ds_seed}
    if calib == "dynamic":
        model.fit_stage1(seq_eff, resp, epochs=e1)
        log_alpha = model.log_alpha.detach().cpu().numpy()
        d_hat = model.d.detach().cpu().numpy()
        # refit discrepancy on the model's own states + refit-bank recovery
        # (paper-1 recipe test: joint fit for dynamics, refit for the bank)
        rf_log_alpha = refit_items(model, seq_eff, resp)
        out["delta"] = 1.0 - spearman(log_alpha, rf_log_alpha)
        out["alpha_spearman_refit"] = item_recovery(rf_log_alpha,
                                                    items)["alpha_spearman"]
        with torch.no_grad():
            z_hat = model.unroll(seq_eff).cpu().numpy()
        out["within_learner_mean"] = float(np.mean(
            [within_learner(z_hat, ds["theta"], c) for c in range(3)]))
    elif calib == "static_early":
        mask = torch.zeros_like(resp, dtype=torch.float)
        mask[:, :W] = 1.0
        _fit_static_masked(model, seq_eff, resp, mask, epochs=e1)
        log_alpha = model.log_alpha.detach().cpu().numpy()
        d_hat = model.d.detach().cpu().numpy()
    elif calib == "static_cohort":
        # measurement-regime calibration: a separate STATIC cohort (rate=0,
        # no transfer) answering the same bank; the operational
        # pre-calibrated-bank posture. Bank identity guaranteed by the
        # shared data_seed (make_items consumes the rng identically).
        ds_cal = generate(ds_seed, kind="matched", N=N, g_true=None,
                          schedule="mixed", T_mixed=60, rate_mean=0.0,
                          ref_inert=False)
        seq_c = torch.as_tensor(ds_cal["seq_eff"], dtype=torch.long,
                                device=device)
        resp_c = torch.as_tensor(ds_cal["resp"], dtype=torch.long,
                                 device=device)
        assert np.allclose(ds_cal["items"]["alpha"], items["alpha"])
        mask = torch.ones_like(resp_c, dtype=torch.float)
        _fit_static_masked(model, seq_c, resp_c, mask, epochs=e1)
        log_alpha = model.log_alpha.detach().cpu().numpy()
        d_hat = model.d.detach().cpu().numpy()
    elif calib == "oracle":
        # per-item refit on TRUE abilities (ceiling)
        theta_t = torch.as_tensor(ds["theta"], dtype=torch.float,
                                  device=device)
        log_alpha, d_hat = _oracle_refit(model, seq_eff, resp, theta_t)
    else:
        raise ValueError(calib)

    out.update(item_recovery(log_alpha, items))
    out.update(d_recovery(d_hat, items))
    out.update(position_bias(d_hat, items, ds["seq_eff"]))
    return out


def _fit_static_masked(model, seq_eff, resp, mask, epochs, lr=5e-3):
    from _qm2_model import _inv_softplus
    with torch.no_grad():
        model.gain_raw.fill_(_inv_softplus(1e-6))
        model.rho_raw.fill_(20.0)
    params = [model.log_alpha, model.d, model.z0]
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss, _ = model.nll(seq_eff, resp, step_mask=mask)
        loss.backward()
        opt.step()


def _oracle_refit(model, seq_eff, resp, theta_true, epochs=300, lr=2e-2):
    import torch.nn.functional as F
    log_a = torch.zeros(model.J, device=seq_eff.device,
                        requires_grad=True)
    d = model.d.detach().clone().requires_grad_(True)
    c = model.item_concept[seq_eff]
    z_tag = torch.gather(theta_true, 2, c.unsqueeze(-1)).squeeze(-1)
    opt = torch.optim.Adam([log_a, d], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        alpha = torch.exp(log_a)[seq_eff]
        dd = d[seq_eff]
        steps = alpha.unsqueeze(-1) * (z_tag.unsqueeze(-1) - dd)
        csum = torch.cumsum(steps, dim=-1)
        lg = torch.cat([torch.zeros_like(csum[..., :1]), csum], dim=-1)
        ll = F.log_softmax(lg, dim=-1)
        loss = -torch.gather(ll, 2, resp.unsqueeze(-1)).squeeze(-1).mean()
        loss.backward()
        opt.step()
    return log_a.detach().cpu().numpy(), d.detach().cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--t", type=int, default=90)
    p.add_argument("--w", type=int, default=30, help="static-early window")
    p.add_argument("--e1", type=int, default=150)
    p.add_argument("--data-seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--kinds", nargs="+", default=KINDS)
    p.add_argument("--calibs", nargs="+",
                   default=["dynamic", "static_early", "oracle"])
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    N = 100 if args.quick else args.n
    e1 = 30 if args.quick else args.e1
    seeds = args.data_seeds[:1] if args.quick else args.data_seeds

    os.makedirs(OUT, exist_ok=True)
    cells = []
    for kind in args.kinds:
        for calib in args.calibs:
            for ds_seed in seeds:
                t0 = time.time()
                r = run_cell(kind, calib, ds_seed, N, args.t, args.w, e1,
                             device)
                r["secs"] = round(time.time() - t0, 1)
                cells.append(r)
                with open(os.path.join(
                        OUT, f"cell_{kind}_{calib}_{ds_seed}.json"),
                        "w") as f:
                    json.dump(r, f, indent=1)
                extra = (f" delta={r['delta']:.3f} "
                         f"a_refit={r['alpha_spearman_refit']:.3f}"
                         if "delta" in r else "")
                print(f"[{kind}/{calib} ds={ds_seed}] "
                      f"a_rho={r['alpha_spearman']:.3f} "
                      f"a_slope={r['alpha_slope']:.3f} "
                      f"d_rho={r['d_loc_spearman']:.3f} "
                      f"posbias={r['pos_bias_r']:+.3f}{extra} "
                      f"({r['secs']}s)", flush=True)

    summary = {}
    for kind in args.kinds:
        summary[kind] = {}
        for calib in args.calibs:
            ks = [c for c in cells
                  if c["kind"] == kind and c["calib"] == calib]
            if not ks:
                continue
            s = {m: seed_agg([c[m] for c in ks])
                 for m in ("alpha_spearman", "alpha_slope",
                           "d_loc_spearman", "pos_bias_r")}
            for extra in ("delta", "alpha_spearman_refit"):
                if extra in ks[0]:
                    s[extra] = seed_agg([c[extra] for c in ks])
            summary[kind][calib] = s
    with open(os.path.join(OUT, "p1_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
