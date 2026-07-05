"""_qm2_traj.py -- day item 5: trajectory exhibit, mechanism vs free tracker.

Consult-constrained design (2026-07-05 memo):
  - BOTH models read through the SAME frozen bank (oracle bank here; bank
    quality is orthogonal to this exhibit) on the expected-score scale.
  - BOTH legs shown: null twin (a free tracker reads response noise as
    learning; the mechanism stays flat) AND transfer twin (the mechanism
    also tracks true growth) -- one leg alone is a manufactured win.
  - Fair comparator: LSTM hidden 32 (paper-1 encoder size), trained on the
    same conditioning window with the same prediction objective through the
    same frozen link (= its own best calibration given the link).
  - Metrics: within-learner rank + RMSE (transfer twin), null wobble
    variance ratio, first-difference correlation, total variation.

The LSTM is response-driven BY DESIGN (input = previous item + previous
response); ours is schedule-driven with fitted z0 (R9). That asymmetry IS
the exhibit: responses can push the tracker's state, so noise becomes
"learning"; nothing can push ours except practice through the mechanism.

Outputs: outputs/qm2/traj/traj_summary.json + traj_exhibit.png (prototype
for the E-panel; user visual sign-off required before proliferation).
Run from repo root: python deep_irt/bench/_qm2_traj.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _qm2_datagen import A, B, U, generate, sparse_g, gpcm_probs   # noqa: E402
from _qm2_model import SimpleStructureGPCM                          # noqa: E402
from _qm2_p1b import _fit_masked                                    # noqa: E402
from _qm2_metrics import seed_agg, spearman                         # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "outputs", "qm2", "traj")


class LSTMTracker(nn.Module):
    """Free response-driven tracker, read through a FROZEN GPCM bank."""

    def __init__(self, J, C, K, item_concept, alpha, d, hidden=32, emb=16):
        super().__init__()
        self.C, self.K = C, K
        self.item_emb = nn.Embedding(J, emb)
        self.lstm = nn.LSTM(emb + K, hidden, batch_first=True)
        self.head = nn.Linear(hidden, C)
        self.register_buffer("item_concept",
                             torch.as_tensor(item_concept, dtype=torch.long))
        self.register_buffer("alpha", torch.as_tensor(alpha,
                                                      dtype=torch.float))
        self.register_buffer("d", torch.as_tensor(d, dtype=torch.float))

    def states(self, seq_eff, resp):
        """z_lstm (N, T, C); state at t uses history STRICTLY before t."""
        N, T = seq_eff.shape
        x_item = self.item_emb(seq_eff[:, :-1])
        x_resp = F.one_hot(resp[:, :-1], num_classes=self.K).float()
        x = torch.cat([x_item, x_resp], dim=-1)
        h, _ = self.lstm(x)
        z_tail = self.head(h)                        # states for t=1..T-1
        z0 = torch.zeros_like(z_tail[:, :1, :])
        return torch.cat([z0, z_tail], dim=1)

    def nll(self, seq_eff, resp, step_mask=None):
        z = self.states(seq_eff, resp)
        alpha = self.alpha[seq_eff]
        dd = self.d[seq_eff]
        c = self.item_concept[seq_eff]
        z_tag = torch.gather(z, 2, c.unsqueeze(-1)).squeeze(-1)
        steps = alpha.unsqueeze(-1) * (z_tag.unsqueeze(-1) - dd)
        csum = torch.cumsum(steps, dim=-1)
        lg = torch.cat([torch.zeros_like(csum[..., :1]), csum], dim=-1)
        ll = F.log_softmax(lg, dim=-1)
        per = -torch.gather(ll, 2, resp.unsqueeze(-1)).squeeze(-1)
        if step_mask is None:
            return per.mean()
        m = step_mask.to(per.dtype)
        return (per * m).sum() / m.sum()


def score_trace(items, z_c, ref_ids):
    """Expected score (prop-of-max) on reference items at abilities z_c
    (any shape); frozen-bank rendering shared by both models."""
    K = items["K"]
    es = np.zeros_like(z_c, dtype=float)
    for q in ref_ids:
        p = gpcm_probs(items["alpha"][q], items["d"][q], z_c)
        es += (p * np.arange(K)).sum(-1)
    return es / (len(ref_ids) * (K - 1))


def run_seed(kind, ds_seed, N, e_ours, e_lstm, device):
    g = sparse_g(val=0.025) if kind == "pos" else None
    ds = generate(ds_seed, kind="matched", N=N, g_true=g,
                  schedule="forecast")
    items, marks = ds["items"], ds["marks"]
    T_cond = marks["T_cond"]
    ref_B = marks["ref_ids"][B]
    seq_eff = torch.as_tensor(ds["seq_eff"], dtype=torch.long, device=device)
    resp = torch.as_tensor(ds["resp"], dtype=torch.long, device=device)
    mask = torch.zeros_like(resp, dtype=torch.float)
    mask[:, :T_cond] = 1.0

    # ours, oracle bank frozen
    torch.manual_seed(0)
    ours = SimpleStructureGPCM(items["J"], 3, items["K"], N,
                               items["concept"],
                               inert_items=ds["inert_items"]).to(device)
    with torch.no_grad():
        ours.log_alpha.copy_(torch.as_tensor(np.log(items["alpha"]),
                                             dtype=torch.float,
                                             device=device))
        ours.d.copy_(torch.as_tensor(items["d"], dtype=torch.float,
                                     device=device))
    ours.freeze_items()
    _fit_masked(ours, seq_eff, resp, mask, stage=1, epochs=e_ours)
    _fit_masked(ours, seq_eff, resp, mask, stage=2, epochs=int(e_ours * .6),
                l1_g=3e-3)
    with torch.no_grad():
        z_ours = ours.unroll(seq_eff).cpu().numpy()

    # free tracker, same frozen bank, same conditioning window
    torch.manual_seed(0)
    lstm = LSTMTracker(items["J"], 3, items["K"], items["concept"],
                       items["alpha"], items["d"]).to(device)
    opt = torch.optim.Adam(lstm.parameters(), lr=5e-3)
    for _ in range(e_lstm):
        opt.zero_grad()
        loss = lstm.nll(seq_eff, resp, step_mask=mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lstm.parameters(), 5.0)
        opt.step()
    with torch.no_grad():
        z_lstm = lstm.states(seq_eff, resp).cpu().numpy()

    # frozen-link score traces on B, conditioning window
    tgrid = np.arange(T_cond)
    s_true = score_trace(items, ds["theta"][:, :T_cond, B], ref_B)
    s_ours = score_trace(items, z_ours[:, :T_cond, B], ref_B)
    s_lstm = score_trace(items, z_lstm[:, :T_cond, B], ref_B)

    def wobble(s):        # mean per-learner variance of the trace
        return float(np.mean(np.var(s, axis=1)))

    def tv(s):            # mean total variation per step
        return float(np.mean(np.abs(np.diff(s, axis=1))))

    def wl_metrics(s):    # vs truth, per learner
        rks = [spearman(s[i], s_true[i]) for i in range(N)]
        rks = [r for r in rks if np.isfinite(r)]
        rmse = float(np.sqrt(np.mean((s - s_true) ** 2)))
        dcorr = [spearman(np.diff(s[i]), np.diff(s_true[i]))
                 for i in range(0, N, 4)]
        dcorr = [r for r in dcorr if np.isfinite(r)]
        return (float(np.mean(rks)) if rks else float("nan"), rmse,
                float(np.mean(dcorr)) if dcorr else float("nan"))

    out = {"kind": kind, "data_seed": ds_seed,
           "wobble_true": wobble(s_true), "wobble_ours": wobble(s_ours),
           "wobble_lstm": wobble(s_lstm),
           "tv_true": tv(s_true), "tv_ours": tv(s_ours),
           "tv_lstm": tv(s_lstm)}
    if kind == "pos":
        for name, s in (("ours", s_ours), ("lstm", s_lstm)):
            rk, rmse, dc = wl_metrics(s)
            out[f"wl_rank_{name}"] = rk
            out[f"rmse_{name}"] = rmse
            out[f"dcorr_{name}"] = dc
    # sample traces for the figure (first seed only used)
    out["_traces"] = {"t": tgrid.tolist(),
                      "true_mean": s_true.mean(0).tolist(),
                      "ours_mean": s_ours.mean(0).tolist(),
                      "lstm_mean": s_lstm.mean(0).tolist(),
                      "true_i": s_true[:3].tolist(),
                      "ours_i": s_ours[:3].tolist(),
                      "lstm_i": s_lstm[:3].tolist()}
    return out


def make_figure(cells, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, kind, title in ((axes[0], "null", "No learning (null twin)"),
                            (axes[1], "pos", "Transfer twin (B learns via A"
                                             " practice)")):
        tr = next(c["_traces"] for c in cells if c["kind"] == kind)
        t = tr["t"]
        for i in range(3):
            ax.plot(t, tr["lstm_i"][i], color="#d62728", alpha=0.25, lw=0.8)
            ax.plot(t, tr["ours_i"][i], color="#1f77b4", alpha=0.25, lw=0.8)
        ax.plot(t, tr["lstm_mean"], color="#d62728", lw=2,
                label="free tracker (LSTM)")
        ax.plot(t, tr["ours_mean"], color="#1f77b4", lw=2,
                label="mechanism (ours)")
        ax.plot(t, tr["true_mean"], color="black", lw=2, ls="--",
                label="truth")
        ax.set_title(title)
        ax.set_xlabel("practice step")
    axes[0].set_ylabel("expected score on B reference items (0-1)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print("figure ->", path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--data-seeds", type=int, nargs="+", default=[42, 43, 44])
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N = 100 if args.quick else args.n
    e_ours = 60 if args.quick else 500
    e_lstm = 80 if args.quick else 500
    seeds = args.data_seeds[:1] if args.quick else args.data_seeds

    os.makedirs(OUT, exist_ok=True)
    cells = []
    for kind in ("null", "pos"):
        for ds_seed in seeds:
            t0 = time.time()
            r = run_seed(kind, ds_seed, N, e_ours, e_lstm, device)
            r["secs"] = round(time.time() - t0, 1)
            cells.append(r)
            print(f"[{kind} ds={ds_seed}] wobble ours={r['wobble_ours']:.5f}"
                  f" lstm={r['wobble_lstm']:.5f} true={r['wobble_true']:.5f}"
                  + (f"  wl ours={r['wl_rank_ours']:.3f}"
                     f" lstm={r['wl_rank_lstm']:.3f}"
                     f"  rmse ours={r['rmse_ours']:.4f}"
                     f" lstm={r['rmse_lstm']:.4f}"
                     if kind == "pos" else "")
                  + f" ({r['secs']}s)", flush=True)

    summary = {}
    for kind in ("null", "pos"):
        ks = [c for c in cells if c["kind"] == kind]
        summary[kind] = {m: seed_agg([c[m] for c in ks])
                         for m in ks[0] if m.startswith(("wobble", "tv",
                                                         "wl_", "rmse",
                                                         "dcorr"))}
    ratio = [c["wobble_lstm"] / max(c["wobble_ours"], 1e-12)
             for c in cells if c["kind"] == "null"]
    summary["null_wobble_ratio_lstm_over_ours"] = seed_agg(ratio)
    with open(os.path.join(OUT, "traj_summary.json"), "w") as f:
        json.dump({"cells": [{k: v for k, v in c.items()
                              if k != "_traces"} for c in cells],
                   "summary": summary}, f, indent=1)
    make_figure(cells, os.path.join(OUT, "traj_exhibit.png"))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
