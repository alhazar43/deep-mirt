"""ACT-P0 fabrication probe: the A4 campaign gatekeeper, run BEFORE the full
synthetic certification campaign (`_planning/design/a4_design.md` v1.1
CG1/CG1a; LEDGER 2026-07-18 "P3 build EXECUTED").

Why this probe exists. The tiny (C=8, N=200) end-to-end dry run
(`outputs/a4/tiny_kdd/`) showed ACT-P0 implying a ~0.3 population
score-scale rise over opportunities 1-10 on EVERY twin, including SYN-NG
(no growth) -- CG1's active-silence bar is <= 0.01. Two explanations
compete:
  (a) a tiny-scale undertraining / underpowered-wiring artifact (the read
      should collapse toward ACT-P1's near-zero tiny-scale read, 0.0024,
      as scale grows), or
  (b) a genuine estimator or code defect (fabrication persists at a
      realistic scale, which would be a program-level finding, not a
      scale artifact).
This script decides between them empirically, at one honest intermediate
scale, and reports the numbers. It does NOT fix, tune, or otherwise touch
the estimator either way -- diagnosis only; any remediation is a separate,
deliberate step taken after reading this probe's output.

Scope (deliberately narrower than `kt_mirt.growth.run.run_neural_cell`,
per the harness instructions: "ACT-P0 AND ACT-P1 only, skip
trackers/gates you do not need"):
  - Twins: SYN-NG (no-growth twin; the fabrication question) and SYN-KG
    (known-growth twin; ACT's positive control -- does ACT-P0 detect real
    growth at all at this scale?).
  - Profile: KDD_MATCHED's density SHAPE (opportunity/rate quantiles,
    item arity, kcs_per_learner=40) at a downscaled size, C=64 KCs,
    N=2000 learners (compute concession vs the full C=515/N=3000 bed;
    `run.make_profile` is the same scale-override helper `run.py` already
    uses for its own TINY_PROFILE).
  - Seeds: 2 generator seeds x 2 model seeds (the harness instructions'
    explicit request for this probe; note this differs from A4's own
    compute concession of "neural arms run on generator seed 0 only",
    section 3.2 -- deliberate here, since varying the generator seed is
    exactly what distinguishes a data-realization fluke from a systematic
    estimator behavior).
  - Hyperparameters: RunConfig's own defaults (bank_epochs=40, i.e. the
    design's convergence-gated bank calibration cap; act_hidden=16,
    act_emb=8, act_lr=0.05, act_epochs=20) -- these ARE the "full
    pre-registered" neural budget already used, unmodified, by the tiny
    dry run (`a4_tiny_cert.sbatch` passes no epoch overrides), so running
    them here at a larger N/C is the requested apples-to-apples comparison
    with only scale changed.
  - Estimators trained: bank calibration (ACT reads through the frozen
    bank; unavoidable) and ACT-P0 + ACT-P1 only. No PAS-N1/PAS-N2
    trackers, no CG7/CG8/CG9/CG10, no gate/rate machinery, no permutation
    battery -- none of those bear on the fabrication question.

Reused, not re-implemented: `kt_mirt.growth.run`'s own private helpers
(`_train_held_split`, `_act_implied_rises`, `derive_seed`, `make_profile`)
are imported directly rather than copied, since they are exactly the
closed-form implied-rise / seeding / profile-scaling logic this probe
needs and `run.py` is this same program's own harness module (CLAUDE.md:
"Reusable functionality over one-off copies").

Usage (local 4060; `research` conda env active, KMP_DUPLICATE_LIB_OK=TRUE
set, run from the repo root):
    python kt-mirt/scripts/probe/act_p0_fabrication_probe.py \
        --device cuda --output kt-mirt/outputs/a4/probe_medium
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from kt_mirt.growth import active as active_mod
from kt_mirt.growth import bank as bank_mod
from kt_mirt.growth import synth as synth_mod
from kt_mirt.growth import tracker as tracker_mod
from kt_mirt.growth.run import _act_implied_rises, _train_held_split, derive_seed, make_profile
from kt_mirt.growth.slices import build_slices

GEN_SEEDS = [0, 1]
MODEL_SEEDS = [0, 1]
TWINS = ("syn_ng", "syn_kg")

# RunConfig's own defaults -- the design's "full pre-registered" neural
# budget (unchanged from the tiny dry run's hyperparameters; only scale
# changes here).
BANK_EPOCHS = 40
ACT_HIDDEN = 16
ACT_EMB = 8
ACT_LR = 0.05
ACT_EPOCHS = 20


def make_probe_profile(n_kcs: int, n_learners: int) -> synth_mod.DensityProfile:
    return make_profile(
        "probe_kdd_medium", n_kcs=n_kcs, n_learners=n_learners,
        kcs_per_learner=synth_mod.KDD_MATCHED.kcs_per_learner, base=synth_mod.KDD_MATCHED,
    )


def calibrate_one(twin: str, gen_seed: int, profile: synth_mod.DensityProfile, device: str):
    """Generates one (twin, gen_seed) draw and calibrates+freezes its bank
    ONCE (bank fit's own seed derives from (twin, gen_seed) only, not
    model_seed, so recomputing per model seed would be redundant and
    bit-identical -- reused across model seeds below)."""
    twin_data = synth_mod.generate_twin(twin, profile, gen_seed)
    acceptance = synth_mod.run_generator_acceptance_checks(twin_data)

    rows_all = bank_mod.build_calibration_rows(twin_data.learners)
    calib, analysis = bank_mod.split_learners(profile.n_learners, seed=derive_seed("cohort", twin, gen_seed))
    hierarchy = bank_mod.flat_hierarchy(twin_data.item_bank.n_items)
    bank_cfg = bank_mod.BankModelConfig(
        n_epochs_max=BANK_EPOCHS, lr=0.05, batch_size=4096, patience_epochs=2,
        seed=derive_seed("bank", twin, gen_seed), device=device,
    )
    fit = bank_mod.calibrate_bank(
        rows_all, profile.n_learners, twin_data.n_kcs, calib, hierarchy,
        bank_mod.FLAT_HIERARCHY_SPEC, growth_mode="blockwise", config=bank_cfg,
    )
    frozen = bank_mod.freeze_bank(fit)
    recovery = bank_mod.synthetic_bank_recovery_check(fit, twin_data.item_bank.b_true)

    analysis_set = set(int(a) for a in analysis.tolist())
    analysis_learners = [l for l in twin_data.learners if l.learner in analysis_set]
    train_learners, _held = _train_held_split(analysis_learners, seed=derive_seed("traineval", twin, gen_seed))

    rows_analysis = bank_mod.build_calibration_rows(analysis_learners)
    slices_analysis = build_slices(rows_analysis)

    return {
        "twin_data": twin_data, "fit": fit, "frozen": frozen, "recovery": recovery,
        "acceptance_passed": acceptance.all_passed,
        "analysis_learners": analysis_learners, "train_learners": train_learners,
        "slices_analysis": slices_analysis,
    }


def train_and_read_act(ctx: dict, twin: str, gen_seed: int, model_seed: int, variant: str, device: str) -> dict:
    fit = ctx["fit"]
    frozen = ctx["frozen"]
    twin_data = ctx["twin_data"]
    n_items = twin_data.item_bank.n_items
    n_kcs = twin_data.n_kcs
    b_ref = float(np.median(fit.b_hat))
    m_init = active_mod.ceiling_init(fit.b_hat)

    acfg = active_mod.ActiveConfig(
        variant=variant, hidden_dim=ACT_HIDDEN, emb_dim=ACT_EMB, lr=ACT_LR,
        n_epochs=ACT_EPOCHS, seed=model_seed, device=device, m_fixed=False,
    )
    torch.manual_seed(derive_seed("torch", twin, gen_seed, model_seed, f"act_{variant}"))
    model = active_mod.ActiveModel(num_items=n_items, n_kcs=n_kcs, cfg=acfg, m_init=m_init).to(device)
    train_batch = tracker_mod.build_learner_batch(ctx["train_learners"], bank=frozen, device=device)
    loss_trace = active_mod.train_active(model, train_batch, acfg)

    pop_mean, p95_abs, kc_rise, _extra = _act_implied_rises(
        model, ctx["analysis_learners"], ctx["slices_analysis"], frozen, n_kcs, b_ref, device=device,
    )
    true_rise = np.asarray(twin_data.truth.true_rise_per_kc)
    finite = np.isfinite(kc_rise) & np.isfinite(true_rise)
    kc_rank_corr = (
        float(bank_mod.spearman_rank_correlation(kc_rise[finite], true_rise[finite]))
        if finite.sum() >= 2 else float("nan")
    )
    true_pop_rise = float(np.mean(true_rise))
    return {
        "pop_mean_rise": pop_mean,
        "p95_abs_rise": p95_abs,
        "kc_rank_corr_vs_true": kc_rank_corr,
        "true_pop_rise": true_pop_rise,
        "ratio_vs_true": (pop_mean / true_pop_rise) if true_pop_rise not in (0.0, None) else float("nan"),
        "final_loss": loss_trace[-1] if loss_trace else float("nan"),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-kcs", type=int, default=64)
    p.add_argument("--n-learners", type=int, default=2000)
    p.add_argument("--output", default="kt-mirt/outputs/a4/probe_medium")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = make_probe_profile(args.n_kcs, args.n_learners)

    t0 = time.time()
    cells: dict[str, list[dict]] = {twin: [] for twin in TWINS}
    acceptance_by_cell: dict[str, bool] = {}
    recovery_by_cell: dict[str, float] = {}

    for twin in TWINS:
        for gen_seed in GEN_SEEDS:
            tcal0 = time.time()
            print(f"[{time.time()-t0:7.1f}s] calibrating bank: {twin} gen_seed={gen_seed} ...", flush=True)
            ctx = calibrate_one(twin, gen_seed, profile, args.device)
            print(
                f"    bank calib done in {time.time()-tcal0:.1f}s "
                f"(acceptance_passed={ctx['acceptance_passed']}, "
                f"b_recovery_rank_corr={ctx['recovery']['rank_corr']:.3f})",
                flush=True,
            )
            acceptance_by_cell[f"{twin}.seed{gen_seed}"] = bool(ctx["acceptance_passed"])
            recovery_by_cell[f"{twin}.seed{gen_seed}"] = float(ctx["recovery"]["rank_corr"])

            for model_seed in MODEL_SEEDS:
                for variant in ("act_p0", "act_p1"):
                    t1 = time.time()
                    read = train_and_read_act(ctx, twin, gen_seed, model_seed, variant, args.device)
                    read.update({"gen_seed": gen_seed, "model_seed": model_seed, "variant": variant})
                    cells[twin].append(read)
                    print(
                        f"[{time.time()-t0:7.1f}s] {twin} gen={gen_seed} model={model_seed} {variant}: "
                        f"pop_mean_rise={read['pop_mean_rise']:.5f} p95_abs_rise={read['p95_abs_rise']:.5f} "
                        f"kc_rank_corr={read['kc_rank_corr_vs_true']:.3f} true_pop_rise={read['true_pop_rise']:.5f} "
                        f"({time.time()-t1:.1f}s)",
                        flush=True,
                    )

    summary = {}
    for twin in TWINS:
        for variant in ("act_p0", "act_p1"):
            rows = [c for c in cells[twin] if c["variant"] == variant]
            pop = np.array([r["pop_mean_rise"] for r in rows])
            p95 = np.array([r["p95_abs_rise"] for r in rows])
            corr = np.array([r["kc_rank_corr_vs_true"] for r in rows])
            summary[f"{twin}.{variant}"] = {
                "pop_mean_rise_mean": float(np.nanmean(pop)),
                "pop_mean_rise_per_cell": pop.tolist(),
                "p95_abs_rise_mean": float(np.nanmean(p95)),
                "p95_abs_rise_per_cell": p95.tolist(),
                "kc_rank_corr_vs_true_mean": float(np.nanmean(corr)),
                "true_pop_rise": rows[0]["true_pop_rise"] if rows else float("nan"),
            }

    payload = {
        "profile": {"name": profile.name, "n_kcs": profile.n_kcs, "n_learners": profile.n_learners,
                    "kcs_per_learner": profile.kcs_per_learner},
        "gen_seeds": GEN_SEEDS, "model_seeds": MODEL_SEEDS, "device": args.device,
        "hyperparameters": {"bank_epochs": BANK_EPOCHS, "act_hidden": ACT_HIDDEN, "act_emb": ACT_EMB,
                             "act_lr": ACT_LR, "act_epochs": ACT_EPOCHS},
        "generator_acceptance_passed": acceptance_by_cell,
        "bank_recovery_rank_corr": recovery_by_cell,
        "wall_clock_s": time.time() - t0,
        "cells": cells,
        "summary": summary,
    }
    (out_dir / "probe_result.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))

    print("\n=== PROBE_DONE ===")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nCG1 silence bar (design 5.1): population mean <= 0.01, p95 <= 0.01 (KDD density)")
    print(f"wall clock: {payload['wall_clock_s']:.1f}s; written to {out_dir / 'probe_result.json'}")


if __name__ == "__main__":
    main()
