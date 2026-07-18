"""ACT repair verification rerun: `act_p0_fabrication_probe.py`'s machinery,
re-run after `train_active` gained its convergence-gated stopping rule
(`active.py` module docstring note 8; diagnosis
`_planning/research/act_p0_diagnosis.md`), at a reduced-but-meaningful
CPU scale (default C=32, N=1000, 1 generator seed x 2 model seeds, both
variants, both twins).

Differences from the original probe, all deliberate:
  - CPU-only by default (the local GPU belongs to the campaign worker).
  - One cell per invocation via ``--twin/--variant/--model-seed`` so the
    independent ACT fits can run as parallel processes (the original
    probe loops serially); a bare invocation with ``--aggregate DIR``
    instead collects the per-cell JSONs into one summary.
  - Reports ``epochs_run`` (the length of `train_active`'s loss trace),
    the number the repair makes meaningful: under the old fixed loop it
    was always ``n_epochs``; now it is the convergence-gated count, the
    input to the campaign cost multiplier.
Everything else -- twin generation, bank calibration, cohort splits, the
`_act_implied_rises` closed-form readout, seed derivation -- is imported
from the original probe / `run.py` unchanged.

Usage (repo root, `research` env):
    # one cell (parallelizable):
    python kt-mirt/scripts/probe/act_p0_repair_rerun.py \
        --twin syn_ng --variant act_p0 --model-seed 0 \
        --out kt-mirt/outputs/a4/probe_repair_rerun/ng_p0_s0.json
    # aggregate:
    python kt-mirt/scripts/probe/act_p0_repair_rerun.py \
        --aggregate kt-mirt/outputs/a4/probe_repair_rerun
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from kt_mirt.growth import active as active_mod
from kt_mirt.growth import tracker as tracker_mod
from kt_mirt.growth.run import _act_implied_rises, derive_seed

import act_p0_fabrication_probe as probe

GEN_SEED = 0  # single generator seed (reduced-scale concession)


def run_cell(twin: str, variant: str, model_seed: int, n_kcs: int, n_learners: int, device: str) -> dict:
    profile = probe.make_probe_profile(n_kcs, n_learners)
    t0 = time.time()
    ctx = probe.calibrate_one(twin, GEN_SEED, profile, device)
    t_bank = time.time() - t0

    fit, frozen, twin_data = ctx["fit"], ctx["frozen"], ctx["twin_data"]
    b_ref = float(np.median(fit.b_hat))
    m_init = active_mod.ceiling_init(fit.b_hat)

    acfg = active_mod.ActiveConfig(
        variant=variant, hidden_dim=probe.ACT_HIDDEN, emb_dim=probe.ACT_EMB, lr=probe.ACT_LR,
        n_epochs=probe.ACT_EPOCHS, seed=model_seed, device=device, m_fixed=False,
    )
    torch.manual_seed(derive_seed("torch", twin, GEN_SEED, model_seed, f"act_{variant}"))
    model = active_mod.ActiveModel(
        num_items=twin_data.item_bank.n_items, n_kcs=twin_data.n_kcs, cfg=acfg, m_init=m_init
    ).to(device)
    train_batch = tracker_mod.build_learner_batch(ctx["train_learners"], bank=frozen, device=device)
    t1 = time.time()
    loss_trace = active_mod.train_active(model, train_batch, acfg)
    t_act = time.time() - t1

    pop_mean, p95_abs, kc_rise, extra = _act_implied_rises(
        model, ctx["analysis_learners"], ctx["slices_analysis"], frozen, twin_data.n_kcs, b_ref, device=device,
    )
    true_rise = np.asarray(twin_data.truth.true_rise_per_kc)
    finite = np.isfinite(kc_rise) & np.isfinite(true_rise)
    from kt_mirt.growth import bank as bank_mod
    kc_rank_corr = (
        float(bank_mod.spearman_rank_correlation(kc_rise[finite], true_rise[finite]))
        if finite.sum() >= 2 else float("nan")
    )
    return {
        "twin": twin, "variant": variant, "gen_seed": GEN_SEED, "model_seed": model_seed,
        "n_kcs": n_kcs, "n_learners": n_learners, "device": device,
        "pop_mean_rise": pop_mean, "p95_abs_rise": p95_abs,
        "kc_rank_corr_vs_true": kc_rank_corr,
        "true_pop_rise": float(np.mean(true_rise)),
        "g_c_mean": float(extra["g_c"].mean()), "M": extra["M"],
        "epochs_run": len(loss_trace),
        "final_loss": loss_trace[-1] if loss_trace else float("nan"),
        "wall_s_bank": t_bank, "wall_s_act": t_act,
    }


def aggregate(out_dir: str) -> None:
    cells = []
    for path in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
        if os.path.basename(path) == "rerun_summary.json":
            continue
        with open(path) as f:
            cells.append(json.load(f))
    summary: dict[str, dict] = {}
    for twin in sorted({c["twin"] for c in cells}):
        for variant in sorted({c["variant"] for c in cells}):
            rows = [c for c in cells if c["twin"] == twin and c["variant"] == variant]
            if not rows:
                continue
            summary[f"{twin}.{variant}"] = {
                "pop_mean_rise_per_cell": [r["pop_mean_rise"] for r in rows],
                "p95_abs_rise_per_cell": [r["p95_abs_rise"] for r in rows],
                "kc_rank_corr_per_cell": [r["kc_rank_corr_vs_true"] for r in rows],
                "epochs_run_per_cell": [r["epochs_run"] for r in rows],
                "true_pop_rise": rows[0]["true_pop_rise"],
            }
    out = {"cells": cells, "summary": summary}
    with open(os.path.join(out_dir, "rerun_summary.json"), "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--twin", choices=("syn_ng", "syn_kg"))
    p.add_argument("--variant", choices=("act_p0", "act_p1"))
    p.add_argument("--model-seed", type=int, default=0)
    p.add_argument("--n-kcs", type=int, default=32)
    p.add_argument("--n-learners", type=int, default=1000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out")
    p.add_argument("--aggregate", metavar="DIR")
    args = p.parse_args()

    if args.aggregate:
        aggregate(args.aggregate)
        return
    if not (args.twin and args.variant and args.out):
        raise SystemExit("either --aggregate DIR, or all of --twin/--variant/--out")

    result = run_cell(args.twin, args.variant, args.model_seed, args.n_kcs, args.n_learners, args.device)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
