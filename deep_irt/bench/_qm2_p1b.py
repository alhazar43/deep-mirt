"""_qm2_p1b.py -- P1b SPINE BRIDGE (plan v1.1, the review's ONE THING).

Re-derive under FB-OFF simple structure at C=3, forecast schedule:
  - the forecast active gap on the canonical SPARSE A->B edge (g=+0.10),
  - the matched null twin (g=0, same seeds),
  - the interference twin (g=-0.10, sign recovery),
with per-cell: G_hat, active gaps B/U, within-learner recovery, observed vs
predicted growth scores on B/U reference items (conditioning M0 vs M3).

Protocol (venue-1 semantics, carried): train on conditioning [0, T_cond)
only; evaluate per-step NLL on masked forecast B/U measurement steps; arms =
with-G vs G-zeroed-from-T_cond. FB-OFF; per-learner z0 individuates.

Cells: kind x data_seed x model_seed. JSON per cell ->
outputs/qm2/p1b/cell_<kind>_<ds>_<ms>.json, summary -> p1b_summary.json.
Run from repo root: python deep_irt/bench/_qm2_p1b.py [--quick]
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

from _qm2_datagen import A, B, U, generate, sparse_g               # noqa: E402
from _qm2_model import SimpleStructureGPCM                          # noqa: E402
from _qm2_metrics import (forecast_gaps, obs_pred_growth, seed_agg,  # noqa: E402
                          within_learner)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "outputs", "qm2", "p1b")

KINDS = {"pos": +0.025, "null": 0.0, "neg": -0.025}   # matched-form dose


def run_cell(kind, g_val, ds_seed, m_seed, N, e1, e2, l1_g, device,
             oracle_bank=False, j_per_concept=12, per_learner_items=False):
    g = None if g_val == 0.0 else sparse_g(val=g_val)
    ds = generate(ds_seed, kind="matched", N=N, g_true=g,
                  schedule="forecast", j_per_concept=j_per_concept,
                  per_learner_items=per_learner_items,
                  theta0_mu_B=(0.5 if g_val < 0 else None))
    marks = ds["marks"]
    T_cond = marks["T_cond"]

    seq_eff = torch.as_tensor(ds["seq_eff"], dtype=torch.long, device=device)
    resp = torch.as_tensor(ds["resp"], dtype=torch.long, device=device)
    cond_mask = torch.zeros_like(resp, dtype=torch.float)
    cond_mask[:, :T_cond] = 1.0

    torch.manual_seed(m_seed)
    items = ds["items"]
    model = SimpleStructureGPCM(items["J"], 3, items["K"], N,
                                items["concept"],
                                inert_items=ds["inert_items"]).to(device)
    if oracle_bank:
        # freeze items at TRUE parameters (causal-chain test: does a good
        # bank heal the growth rendering?); stage 1 then fits dynamics only.
        with torch.no_grad():
            model.log_alpha.copy_(torch.as_tensor(
                np.log(items["alpha"]), dtype=torch.float, device=device))
            model.d.copy_(torch.as_tensor(items["d"], dtype=torch.float,
                                          device=device))
        model.freeze_items()

    # stage 1 / stage 2 on conditioning window only
    opt_nll = _fit_masked(model, seq_eff, resp, cond_mask,
                          stage=1, epochs=e1)
    s2_nll = _fit_masked(model, seq_eff, resp, cond_mask,
                         stage=2, epochs=e2, l1_g=l1_g)

    with torch.no_grad():
        _, per_withG = model.nll(seq_eff, resp)
        _, per_noG = model.nll(seq_eff, resp, g_off_from=T_cond)
        z_hat = model.unroll(seq_eff).cpu().numpy()
        z_noG = model.unroll(seq_eff, g_off_from=T_cond).cpu().numpy()
        G_hat = model.G_signed.cpu().numpy()
        log_alpha = model.log_alpha.cpu().numpy()
        d_hat = model.d.cpu().numpy()

    gaps = forecast_gaps(per_noG.cpu().numpy(), per_withG.cpu().numpy(),
                         marks)

    # stage 2b DEBIAS (rendering only; certification reads the L1 model):
    # refit G without L1, holding everything else, to undo shrinkage
    # attenuation in G magnitude and predicted growth.
    opt = torch.optim.Adam([model.G_raw], lr=5e-3)
    for _ in range(40):
        opt.zero_grad()
        loss, _ = model.nll(seq_eff, resp, step_mask=cond_mask)
        loss.backward()
        opt.step()
    with torch.no_grad():
        G_deb = model.G_signed.cpu().numpy()
        z_deb = model.unroll(seq_eff).cpu().numpy()

    # PRIMARY metric (plan v1.1 rendering contract): forecast growth-score
    # error per arm; gap = err(no-G) - err(with-G), positive = with-G wins.
    # Robust to probability-sharpness overshoot, unlike NLL (2026-07-04
    # pilot: matched-form NLL fragility).
    def score_err(z_arm, steps):
        errs = []
        K = items["K"]
        for t in steps:
            for i in range(0, resp.shape[0]):
                q = int(ds["seq_eff"][i, t])
                from _qm2_datagen import gpcm_probs
                p = gpcm_probs(np.exp(log_alpha[q]), d_hat[q],
                               z_arm[i, t, items["concept"][q]])
                pred = (p * np.arange(K)).sum() / (K - 1)
                obs = ds["resp"][i, t] / (K - 1)
                errs.append(abs(pred - obs))
        return float(np.mean(errs))

    e_wB = score_err(z_hat, marks["fc_B"])
    e_nB = score_err(z_noG, marks["fc_B"])
    e_wU = score_err(z_hat, marks["fc_U"])
    e_nU = score_err(z_noG, marks["fc_U"])
    sgB, sgU = e_nB - e_wB, e_nU - e_wU

    # conditioning measurement blocks for growth scores (B and U)
    concept_of = items["concept"][ds["seq"]]
    cond_steps = np.arange(T_cond)

    def blocks(c):
        steps = cond_steps[concept_of[:T_cond] == c]
        return steps[:4], steps[-4:]                     # M0 vs M3

    growth = {}
    for name, c in (("B", B), ("U", U)):
        early, late = blocks(c)
        r = obs_pred_growth(ds, z_hat, np.exp(log_alpha), d_hat, c,
                            early, late)
        rd = obs_pred_growth(ds, z_deb, np.exp(log_alpha), d_hat, c,
                             early, late)
        growth[name] = {k: r[k] for k in ("obs_rise", "pred_rise", "gap")}
        growth[name]["pred_rise_debias"] = rd["pred_rise"]
        growth[name]["gap_debias"] = rd["gap"]

    wl = {name: within_learner(z_hat[:, :T_cond, :],
                               ds["theta"][:, :T_cond, :], c)
          for name, c in (("A", A), ("B", B))}

    return {"kind": kind, "g_true": g_val, "data_seed": ds_seed,
            "model_seed": m_seed,
            "score_gap_B": sgB, "score_gap_U": sgU,
            "score_err_withG_B": e_wB, "score_err_noG_B": e_nB,
            "score_err_withG_U": e_wU, "score_err_noG_U": e_nU,
            "active_gap_B": gaps["active_gap_B"],
            "active_gap_U": gaps["active_gap_U"],
            "G_hat_BA": float(G_hat[B, A]),
            "G_hat_UA": float(G_hat[U, A]),
            "G_hat": G_hat.tolist(),
            "G_debias_BA": float(G_deb[B, A]),
            "G_debias": G_deb.tolist(),
            "growth": growth, "within_learner": wl,
            "s1_nll": opt_nll, "s2_nll": s2_nll}


def _fit_masked(model, seq_eff, resp, mask, stage, epochs, lr=5e-3,
                l1_g=1e-2):
    if stage == 1:
        params = ([model.log_alpha, model.d]
                  if model.log_alpha.requires_grad else []) \
                 + model.dyn_params()
    else:
        model.freeze_items()
        params = [model.G_raw] + model.dyn_params()
    opt = torch.optim.Adam(params, lr=lr)
    loss_v = float("nan")
    for _ in range(epochs):
        opt.zero_grad()
        loss, _ = model.nll(seq_eff, resp, step_mask=mask)
        total = loss if stage == 1 else \
            loss + l1_g * model.G_signed.abs().sum()
        total.backward()
        opt.step()
        loss_v = float(loss.item())
    return loss_v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--e1", type=int, default=120)
    p.add_argument("--e2", type=int, default=80)
    p.add_argument("--l1-g", type=float, default=3e-3)
    p.add_argument("--data-seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--model-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--kinds", nargs="+", default=list(KINDS))
    p.add_argument("--oracle-bank", action="store_true")
    p.add_argument("--j-per-concept", type=int, default=12)
    p.add_argument("--per-learner-items", action="store_true")
    p.add_argument("--tag", default=None,
                   help="route outputs to p1b_<tag>/ (scale runs must not "
                        "clobber the certification cells)")
    args = p.parse_args()

    global OUT
    if args.oracle_bank:
        OUT = OUT + "_oracle"          # never clobber certification cells
    if args.tag:
        OUT = OUT + "_" + args.tag
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N = 100 if args.quick else args.n
    e1 = 30 if args.quick else args.e1
    e2 = 20 if args.quick else args.e2
    ds_seeds = args.data_seeds[:1] if args.quick else args.data_seeds
    m_seeds = args.model_seeds[:1] if args.quick else args.model_seeds

    os.makedirs(OUT, exist_ok=True)
    cells = []
    for kind in args.kinds:
        for ds_seed in ds_seeds:
            for m_seed in m_seeds:
                t0 = time.time()
                r = run_cell(kind, KINDS[kind], ds_seed, m_seed, N, e1, e2,
                             args.l1_g, device,
                             oracle_bank=args.oracle_bank,
                             j_per_concept=args.j_per_concept,
                             per_learner_items=args.per_learner_items)
                r["secs"] = round(time.time() - t0, 1)
                cells.append(r)
                path = os.path.join(OUT,
                                    f"cell_{kind}_{ds_seed}_{m_seed}.json")
                with open(path, "w") as f:
                    json.dump(r, f, indent=1)
                print(f"[{kind} ds={ds_seed} ms={m_seed}] "
                      f"scoregapB={r['score_gap_B']:+.4f} "
                      f"scoregapU={r['score_gap_U']:+.4f} "
                      f"nllgapB={r['active_gap_B']:+.4f} "
                      f"G_BA={r['G_hat_BA']:+.4f} "
                      f"wlB={r['within_learner']['B']:.3f} "
                      f"growthB gap={r['growth']['B']['gap']:.4f} "
                      f"({r['secs']}s)", flush=True)

    summary = {}
    null_cells = {(c["data_seed"], c["model_seed"]): c for c in cells
                  if c["kind"] == "null"}
    for kind in args.kinds:
        ks = [c for c in cells if c["kind"] == kind]
        # matched-null PAIRED contrast (campaign correction, carried)
        paired = [c["score_gap_B"]
                  - null_cells[(c["data_seed"], c["model_seed"])]
                  ["score_gap_B"]
                  for c in ks
                  if (c["data_seed"], c["model_seed"]) in null_cells]
        summary[kind] = {
            "score_gap_B": seed_agg([c["score_gap_B"] for c in ks]),
            "score_gap_B_minus_null": seed_agg(paired) if paired else None,
            "score_gap_U": seed_agg([c["score_gap_U"] for c in ks]),
            "active_gap_B": seed_agg([c["active_gap_B"] for c in ks]),
            "active_gap_U": seed_agg([c["active_gap_U"] for c in ks]),
            "G_hat_BA": seed_agg([c["G_hat_BA"] for c in ks]),
            "sign_G_BA": int(sum(np.sign(c["G_hat_BA"]) ==
                                 np.sign(KINDS[kind]) for c in ks))
            if KINDS[kind] != 0 else None,
            "growth_B_gap": seed_agg([c["growth"]["B"]["gap"] for c in ks]),
            "growth_B_obs": seed_agg([c["growth"]["B"]["obs_rise"]
                                      for c in ks]),
            "growth_B_pred": seed_agg([c["growth"]["B"]["pred_rise"]
                                       for c in ks]),
            "growth_B_pred_debias": seed_agg(
                [c["growth"]["B"]["pred_rise_debias"] for c in ks]),
            "growth_B_gap_debias": seed_agg(
                [c["growth"]["B"]["gap_debias"] for c in ks]),
            "G_debias_BA": seed_agg([c["G_debias_BA"] for c in ks]),
            "within_learner_B": seed_agg([c["within_learner"]["B"]
                                          for c in ks]),
        }
    with open(os.path.join(OUT, "p1b_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
