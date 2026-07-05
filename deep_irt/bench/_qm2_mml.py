"""_qm2_mml.py -- the G1 mechanism adjudicator (editor morning-queue 1+2).

On the SAME static-cohort dataset (rate=0, mixed schedule, matched bank),
vary ONLY theta-knowledge and budget, never both:

  mml       per-concept marginal ML: z integrated over 61-pt quadrature on
            [-4,4] with a fixed N(0,1) prior (prior misspecification hits
            alpha SCALE, a common factor; rank recovery is scale-free).
            Budget 300 epochs @ lr 2e-2 (the oracle arm's budget).
  jml_eq    joint fit (free per-learner z0) at the SAME 300 @ 2e-2.
  jml_conv  joint fit converged: 1000 @ 5e-3.
  oracle    per-item fit on TRUE theta, 300 @ 2e-2 (unchanged reference).

Readout: within-concept alpha rank recovery (and d location) per arm.
  MML ~ oracle >> JML(both)  -> incidental-parameters mechanism confirmed.
  JML_conv ~ oracle          -> the earlier failure was optimization budget.
  all ~ fail                 -> range restriction / information absence.

JSON -> outputs/qm2/mml_race.json. Run from repo root:
    python deep_irt/bench/_qm2_mml.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _qm2_datagen import generate                                   # noqa: E402
from _qm2_model import SimpleStructureGPCM                          # noqa: E402
from _qm2_metrics import d_recovery, item_recovery, seed_agg        # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "outputs", "qm2", "mml_race.json")
QNODES = 61


def _gpcm_logp(log_alpha, d, theta):
    """log P(k | theta) for a block of items. theta: (Q,), log_alpha: (J,),
    d: (J, K-1) -> (Q, J, K)."""
    alpha = torch.exp(log_alpha)
    steps = alpha[None, :, None] * (theta[:, None, None] - d[None, :, :])
    csum = torch.cumsum(steps, dim=-1)
    logits = torch.cat([torch.zeros_like(csum[..., :1]), csum], dim=-1)
    return F.log_softmax(logits, dim=-1)


def fit_mml(ds, epochs=300, lr=2e-2, device="cpu"):
    """Per-concept quadrature marginal ML on the static cohort."""
    items = ds["items"]
    C = ds["config"]["C"]
    J, K = items["J"], items["K"]
    seq = ds["seq"]                       # shared schedule (matched kind)
    resp = ds["resp"]
    concept_of = items["concept"]

    nodes = torch.linspace(-4.0, 4.0, QNODES, device=device)
    logw = torch.log_softmax(-0.5 * nodes ** 2, dim=0)   # N(0,1) on grid

    log_alpha = torch.zeros(J, device=device, requires_grad=True)
    d0 = np.linspace(-0.8, 0.8, K - 1)
    d = torch.tensor(np.tile(d0, (J, 1)), dtype=torch.float, device=device,
                     requires_grad=True)
    opt = torch.optim.Adam([log_alpha, d], lr=lr)

    per_c = []
    for c in range(C):
        tsel = np.where(concept_of[seq] == c)[0]
        q_ids = torch.as_tensor(seq[tsel], dtype=torch.long, device=device)
        y = torch.as_tensor(resp[:, tsel], dtype=torch.long, device=device)
        per_c.append((q_ids, y))

    for _ in range(epochs):
        opt.zero_grad()
        loss = 0.0
        for q_ids, y in per_c:
            logp = _gpcm_logp(log_alpha[q_ids], d[q_ids], nodes)  # (Q,T,K)
            ll_t = logp.gather(
                2, y.T.unsqueeze(0).expand(QNODES, -1, -1)
            )                                                      # (Q,T,N)
            ll = ll_t.sum(dim=1)                                   # (Q,N)
            loss = loss - torch.logsumexp(logw[:, None] + ll, dim=0).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([log_alpha, d], 5.0)
        opt.step()
    return log_alpha.detach().cpu().numpy(), d.detach().cpu().numpy()


def fit_jml(ds, epochs, lr, device="cpu"):
    items = ds["items"]
    N = ds["config"]["N"]
    seq_eff = torch.as_tensor(ds["seq_eff"], dtype=torch.long, device=device)
    resp = torch.as_tensor(ds["resp"], dtype=torch.long, device=device)
    torch.manual_seed(0)
    m = SimpleStructureGPCM(items["J"], ds["config"]["C"], items["K"], N,
                            items["concept"]).to(device)
    from _qm2_model import _inv_softplus
    with torch.no_grad():
        m.gain_raw.fill_(_inv_softplus(1e-6))
        m.rho_raw.fill_(20.0)
    params = [m.log_alpha, m.d, m.z0]
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss, _ = m.nll(seq_eff, resp)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
    return (m.log_alpha.detach().cpu().numpy(),
            m.d.detach().cpu().numpy(), float(loss.item()))


def fit_oracle(ds, epochs=300, lr=2e-2, device="cpu"):
    import torch.nn.functional as F  # noqa: F811
    items = ds["items"]
    seq_eff = torch.as_tensor(ds["seq_eff"], dtype=torch.long, device=device)
    resp = torch.as_tensor(ds["resp"], dtype=torch.long, device=device)
    theta = torch.as_tensor(ds["theta"], dtype=torch.float, device=device)
    con = torch.as_tensor(items["concept"], dtype=torch.long, device=device)
    z_tag = torch.gather(theta, 2,
                         con[seq_eff].unsqueeze(-1)).squeeze(-1)
    J, K = items["J"], items["K"]
    log_a = torch.zeros(J, device=device, requires_grad=True)
    d = torch.tensor(np.tile(np.linspace(-0.8, 0.8, K - 1), (J, 1)),
                     dtype=torch.float, device=device, requires_grad=True)
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
        torch.nn.utils.clip_grad_norm_([log_a, d], 5.0)
        opt.step()
    return log_a.detach().cpu().numpy(), d.detach().cpu().numpy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    for seed in (42, 43, 44):
        ds = generate(seed, kind="matched", N=400, g_true=None,
                      schedule="mixed", T_mixed=60, rate_mean=0.0,
                      ref_inert=False)
        items = ds["items"]
        arms = {}
        t0 = time.time()
        la, dd = fit_mml(ds, device=device)
        arms["mml"] = {**item_recovery(la, items), **d_recovery(dd, items),
                       "secs": round(time.time() - t0, 1)}
        t0 = time.time()
        la, dd, nll = fit_jml(ds, epochs=300, lr=2e-2, device=device)
        arms["jml_eq"] = {**item_recovery(la, items),
                          **d_recovery(dd, items), "nll": nll,
                          "secs": round(time.time() - t0, 1)}
        t0 = time.time()
        la, dd, nll = fit_jml(ds, epochs=1000, lr=5e-3, device=device)
        arms["jml_conv"] = {**item_recovery(la, items),
                            **d_recovery(dd, items), "nll": nll,
                            "secs": round(time.time() - t0, 1)}
        t0 = time.time()
        la, dd = fit_oracle(ds, device=device)
        arms["oracle"] = {**item_recovery(la, items),
                          **d_recovery(dd, items),
                          "secs": round(time.time() - t0, 1)}
        results[str(seed)] = arms
        for a, r in arms.items():
            print(f"[seed {seed}] {a:9s} a_rho={r['alpha_spearman']:+.3f} "
                  f"a_slope={r['alpha_slope']:+.3f} "
                  f"d_rho={r['d_loc_spearman']:+.3f} ({r['secs']}s)",
                  flush=True)

    agg = {}
    for arm in ("mml", "jml_eq", "jml_conv", "oracle"):
        agg[arm] = {m: seed_agg([results[s][arm][m] for s in results])
                    for m in ("alpha_spearman", "alpha_slope",
                              "d_loc_spearman")}
    with open(OUT, "w") as f:
        json.dump({"cells": results, "agg": agg}, f, indent=1)
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
