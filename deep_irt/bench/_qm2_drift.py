"""_qm2_drift.py -- day items 3+4: item-parameter drift check and
person-side reliability, per the 2026-07-05 psychometric consult.

PART 1 (item drift, consult Q1). Split the conditioning window into an
early and a late half; per half, refit ONLY item parameters on the model's
fitted states, starting from the frozen bank (FIPC posture). Statistic per
item: location displacement delta_b_j = b_j(late) - b_j(early), within-
concept centered. Items are static by construction on every synthetic twin,
so ALL displacement is false alarm; the empirical 95th percentile of
|delta_b| across twins x seeds is the critical value carried to real data.
MANDATORY control (consult's worst pitfall): the same refits on TRUE states
-- if the fitted-state displacement distribution is much wider than the
oracle-state one, the check is reading person misfit as item drift.
Re-exposure companion (R10): regress |delta_b_j| on administration count.

PART 2 (growth-score reliability, consult Q2c -- the statistic a referee
names first). Bed: noisy transfer twin (sigma_theta=0.15, g=+0.025),
j_per_concept=24, n_ref=8 so each half has 4 reference items. Per learner,
OBSERVED growth score (late minus early, prop-of-max) computed on item
half 1 vs half 2 of B's reference set; Spearman-Brown corrected split-half
correlation = reliability of the growth score. Model validity companion:
correlation of model-predicted growth with observed growth across learners.
Targets (consult): reliability >= 0.80 is the referee bar; report honestly
either way. Never computed on the deterministic bed.

Outputs: outputs/qm2/drift/drift_summary.json.
Run from repo root: python deep_irt/bench/_qm2_drift.py [--quick]
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

from _qm2_datagen import B, generate, sparse_g                      # noqa: E402
from _qm2_model import SimpleStructureGPCM                          # noqa: E402
from _qm2_p1b import _fit_masked                                    # noqa: E402
from _qm2_metrics import _center_wc, seed_agg, spearman             # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "outputs", "qm2", "drift")


# ------------------------------------------------------------ item drift
def _refit_window(model, seq_eff, resp, z_tag, wmask, epochs=250, lr=1e-2):
    """Per-item refit of (log_alpha, d) on FIXED states, window-masked,
    FIPC start from the model's current bank."""
    import torch.nn.functional as F
    log_a = model.log_alpha.detach().clone().requires_grad_(True)
    d = model.d.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([log_a, d], lr=lr)
    m = wmask
    for _ in range(epochs):
        opt.zero_grad()
        alpha = torch.exp(log_a)[seq_eff]
        dd = d[seq_eff]
        steps = alpha.unsqueeze(-1) * (z_tag.unsqueeze(-1) - dd)
        csum = torch.cumsum(steps, dim=-1)
        lg = torch.cat([torch.zeros_like(csum[..., :1]), csum], dim=-1)
        ll = F.log_softmax(lg, dim=-1)
        per = -torch.gather(ll, 2, resp.unsqueeze(-1)).squeeze(-1)
        loss = (per * m).sum() / m.sum()
        loss.backward()
        opt.step()
    return d.detach().cpu().numpy().mean(axis=1)      # item locations b_j


def item_drift_cell(kind, ds_seed, N, e1, device):
    ds = generate(ds_seed, kind=kind, N=N, g_true=None, schedule="mixed",
                  T_mixed=90)
    items = ds["items"]
    seq_eff = torch.as_tensor(ds["seq_eff"], dtype=torch.long, device=device)
    resp = torch.as_tensor(ds["resp"], dtype=torch.long, device=device)
    T = resp.shape[1]

    torch.manual_seed(0)
    model = SimpleStructureGPCM(items["J"], 3, items["K"], N,
                                items["concept"],
                                inert_items=ds["inert_items"]).to(device)
    mask = torch.ones_like(resp, dtype=torch.float)
    _fit_masked(model, seq_eff, resp, mask, stage=1, epochs=e1)

    con = model.item_concept[seq_eff]
    with torch.no_grad():
        z_fit = model.unroll(seq_eff)
    z_fit_tag = torch.gather(z_fit, 2, con.unsqueeze(-1)).squeeze(-1)
    theta = torch.as_tensor(ds["theta"], dtype=torch.float, device=device)
    z_true_tag = torch.gather(theta, 2, con.unsqueeze(-1)).squeeze(-1)

    early = torch.zeros_like(resp, dtype=torch.float)
    early[:, :T // 2] = 1.0
    late = 1.0 - early

    out = {"kind": kind, "data_seed": ds_seed}
    counts = np.bincount(np.asarray(ds["seq_eff"]).ravel(),
                         minlength=items["J"]).astype(float)
    for tag, z_tag in (("fitted", z_fit_tag), ("oracle", z_true_tag)):
        b_e = _refit_window(model, seq_eff, resp, z_tag, early)
        b_l = _refit_window(model, seq_eff, resp, z_tag, late)
        db = _center_wc(b_l, items["concept"]) \
            - _center_wc(b_e, items["concept"])
        # robust-z posture (Donoghue-Isham): standardize displacement by
        # its information, SE ~ 1/sqrt(n_responses); raw |db| conflates
        # low-information items with drifting items.
        out[f"absdb_{tag}"] = (np.abs(db) * np.sqrt(counts / 2.0)).tolist()
    out["exposure_r"] = spearman(counts, np.abs(
        np.array(out["absdb_fitted"])))
    return out


# ------------------------------------------------ growth-score reliability
def reliability_cell(ds_seed, N, e1, device, kind="matched",
                     pool_blocks=False):
    ds = generate(ds_seed, kind=kind, N=N,
                  g_true=sparse_g(val=0.025), schedule="forecast",
                  j_per_concept=24, n_ref=8, sigma_theta=0.15)
    items, marks = ds["items"], ds["marks"]
    T_cond = marks["T_cond"]
    ref_B = marks["ref_ids"][B]
    K = items["K"]

    concept_of = items["concept"][ds["seq"]]
    cond = np.arange(T_cond)
    b_steps = cond[concept_of[:T_cond] == B]
    n_blk = len(ref_B)
    if pool_blocks:      # M0+M1 vs M2+M3: doubles measurement per window
        early_steps, late_steps = b_steps[:2 * n_blk], b_steps[-2 * n_blk:]
    else:
        early_steps, late_steps = b_steps[:n_blk], b_steps[-n_blk:]

    def growth_on(items_subset):
        def win_score(steps):
            cols = [t for t in steps
                    if ds["seq"][t] in items_subset]
            return ds["resp"][:, cols].mean(axis=1) / (K - 1)
        return win_score(late_steps) - win_score(early_steps)

    half1, half2 = ref_B[0::2], ref_B[1::2]
    g1, g2 = growth_on(set(half1)), growth_on(set(half2))
    r12 = float(np.corrcoef(g1, g2)[0, 1])
    sb = 2 * r12 / (1 + r12) if r12 > -1 else float("nan")

    # model-validity companion: predicted growth vs observed growth
    seq_eff = torch.as_tensor(ds["seq_eff"], dtype=torch.long, device=device)
    resp = torch.as_tensor(ds["resp"], dtype=torch.long, device=device)
    mask = torch.zeros_like(resp, dtype=torch.float)
    mask[:, :T_cond] = 1.0
    torch.manual_seed(0)
    model = SimpleStructureGPCM(items["J"], 3, items["K"], N,
                                items["concept"],
                                inert_items=ds["inert_items"]).to(device)
    _fit_masked(model, seq_eff, resp, mask, stage=1, epochs=e1)
    _fit_masked(model, seq_eff, resp, mask, stage=2,
                epochs=int(e1 * 0.6), l1_g=3e-3)
    with torch.no_grad():
        z = model.unroll(seq_eff).cpu().numpy()
        la = model.log_alpha.cpu().numpy()
        dd = model.d.cpu().numpy()
    from _qm2_datagen import gpcm_probs

    def pred_score(steps):
        vals = np.zeros((len(steps), N))
        for j, t in enumerate(steps):
            q = ds["seq"][t]
            p = gpcm_probs(np.exp(la[q]), dd[q], z[:, t, B])
            vals[j] = (p * np.arange(K)).sum(-1) / (K - 1)
        return vals.mean(axis=0)

    g_pred = pred_score(late_steps) - pred_score(early_steps)
    g_obs = growth_on(set(ref_B))
    validity = float(np.corrcoef(g_pred, g_obs)[0, 1])
    return {"data_seed": ds_seed, "split_half_r": r12,
            "spearman_brown": sb, "pred_obs_validity": validity,
            "obs_growth_mean": float(g_obs.mean()),
            "obs_growth_sd_between": float(g_obs.std())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--e1", type=int, default=500)
    p.add_argument("--data-seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--rel-kind", default="matched")
    p.add_argument("--pool-blocks", action="store_true")
    p.add_argument("--rel-only", action="store_true")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N = 100 if args.quick else args.n
    e1 = 60 if args.quick else args.e1
    seeds = args.data_seeds[:1] if args.quick else args.data_seeds

    os.makedirs(OUT, exist_ok=True)
    drift_cells, rel_cells = [], []
    for kind in ([] if args.rel_only else ("matched", "nonmono")):
        for ds_seed in seeds:
            t0 = time.time()
            r = item_drift_cell(kind, ds_seed, N, e1, device)
            r["secs"] = round(time.time() - t0, 1)
            drift_cells.append(r)
            f95 = float(np.percentile(r["absdb_fitted"], 95))
            o95 = float(np.percentile(r["absdb_oracle"], 95))
            print(f"[drift {kind} ds={ds_seed}] |db| p95 fitted={f95:.4f} "
                  f"oracle={o95:.4f} exposure_r={r['exposure_r']:+.3f} "
                  f"({r['secs']}s)", flush=True)
    for ds_seed in seeds:
        t0 = time.time()
        r = reliability_cell(ds_seed, N, e1, device, kind=args.rel_kind,
                             pool_blocks=args.pool_blocks)
        r["secs"] = round(time.time() - t0, 1)
        rel_cells.append(r)
        print(f"[reliab ds={ds_seed}] split-half r={r['split_half_r']:.3f} "
              f"SB={r['spearman_brown']:.3f} "
              f"validity={r['pred_obs_validity']:.3f} ({r['secs']}s)",
              flush=True)

    summary = {}
    if drift_cells:
        all_fit = np.concatenate([c["absdb_fitted"] for c in drift_cells])
        all_ora = np.concatenate([c["absdb_oracle"] for c in drift_cells])
        summary.update({
            "item_drift_null_p95_fitted": float(np.percentile(all_fit, 95)),
            "item_drift_null_p95_oracle": float(np.percentile(all_ora, 95)),
            "circularity_ratio_fitted_over_oracle": float(
                np.percentile(all_fit, 95) / max(
                    np.percentile(all_ora, 95), 1e-9)),
            "exposure_r": seed_agg([c["exposure_r"]
                                    for c in drift_cells]),
        })
    summary.update({
        "rel_kind": args.rel_kind, "pool_blocks": args.pool_blocks,
        "split_half_r": seed_agg([c["split_half_r"] for c in rel_cells]),
        "spearman_brown": seed_agg([c["spearman_brown"]
                                    for c in rel_cells]),
        "pred_obs_validity": seed_agg([c["pred_obs_validity"]
                                       for c in rel_cells]),
    })
    out_name = "drift_summary.json" if not args.rel_only else \
        f"reliab_{args.rel_kind}{'_pooled' if args.pool_blocks else ''}.json"
    with open(os.path.join(OUT, out_name), "w") as f:
        json.dump({"drift_cells": [{k: v for k, v in c.items()
                                    if not k.startswith("absdb")}
                                   for c in drift_cells],
                   "reliability_cells": rel_cells,
                   "summary": summary}, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
