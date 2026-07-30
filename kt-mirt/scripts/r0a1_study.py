"""R0-A1 held-out trainer study for the L1-penalized G objective (avenue
A1; `_planning/design/a1_design.md` v1.1 section 2.3, and the CT0 report's
"disciplined next steps" 1-2 in `_planning/ct0_power_result.md`).

CT0's verdict: per-edge sign IS recoverable at D=3 on the KDD-shaped
density, but the negative/interference dose (|g|=0.02) is not discriminable
from the true-zero fabrication band (~0.02), and that band GROWS with
epochs at flat NLL under the default weak L1 (1e-3). This study re-derives
the L1 weight and checks epoch-ROBUSTNESS of the leak, on HELD-OUT tuning
seeds 100-102 -- never the 0-4 certification seeds (a1_design.md's
held-out-seed tuning discipline; tuning may never touch a test config).

Phase 1 -- L1 x ceiling grid at the CT0 reference cell (KDD-shaped, N=500,
decoupling=0.90, reference dose +0.05/-0.02): l1_weight in {1e-3, 3e-3,
1e-2, 3e-2} x epoch ceiling in {500, 1500}. Per config: pooled NG band,
true-zero leak, recovered Gpos/Gneg, sign metrics vs the config's own
band, epochs actually run. The ceiling axis tests whether a config's leak
is epoch-robust (an equilibrium held by the L1), not a tuned stopping
time -- the honest fix for the observed drift pathology.

Pre-registered winner rule (encoded in `pick_winner`, no hand tuning):
among configs with (a) the negative edge sign-correct in every seed at
BOTH ceilings, (b) seed-mean |Gneg| within [0.5x, 2x] truth at both
ceilings, (c) false-edge rate <= 0.05 at both ceilings, pick the max
epoch-robust separation score min_ceiling(|Gneg_mean| / band); ties break
to the SMALLER l1_weight (less shrinkage risk). If none qualifies the
study reports NO-WINNER and the negative half stays blocked at 1x dose.

Phase 2 -- dose-response (CT0 next-step 2): |g_neg| in {0.01, 0.02, 0.04,
0.08} at the reference l1 (1e-3) and at the phase-1 winner, ceiling 500.
The NG twin injects G_true=0 and its schedule comes from the KG support,
which is dose-invariant, so each config's phase-1 ceiling-500 band is
reused. Reports the minimal dose at which the negative edge separates.

Run:  python -u -m scripts.r0a1_study   (from kt-mirt/, research env)
Out:  outputs/a1/r0a1/r0a1_study.json + stdout tables.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from kt_mirt.growth.tracker import build_learner_batch
from kt_mirt.transfer.ct0 import pooled_null_band, sign_f1
from kt_mirt.transfer.model import TransferConfig, TransferModel, train_transfer
from kt_mirt.transfer.synth import generate_signed_twin

TUNING_SEEDS = [100, 101, 102]  # held-out; certification seeds 0-4 are untouchable here
DENSITY = "kdd"
N_LEARNERS = 500
DECOUPLING = 0.90
G_POS = 0.05
G_NEG_REF = 0.02
L1_LADDER = [1e-3, 3e-3, 1e-2, 3e-2]
CEILINGS = [500, 1500]
DOSES = [0.01, 0.02, 0.04, 0.08]
OUT_PATH = Path("outputs/a1/r0a1/r0a1_study.json")


def fit_with_trace(twin, cfg: TransferConfig, model_seed: int):
    """`ct0.fit_twin` (gate fixed at truth) with the epoch trace kept, so
    the study can report how long the stationarity gate actually ran."""
    batch = build_learner_batch(twin.learners, b_true=twin.truth.b_true, device=cfg.device)
    cfg_fit = replace(cfg, seed=model_seed, floor=twin.truth.floor, m_fixed=True)
    torch.manual_seed(model_seed)
    model = TransferModel(
        num_items=twin.item_bank.n_items, n_kcs=twin.n_kcs, cfg=cfg_fit, m_init=twin.truth.M_gen
    )
    n_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        trace = train_transfer(model, batch, cfg_fit)
    finally:
        torch.set_num_threads(n_threads)
    return model.recovered_G(), len(trace), float(trace[-1])


def edge_masks(G_true: np.ndarray):
    off = ~np.eye(G_true.shape[0], dtype=bool)
    return (G_true > 0) & off, (G_true < 0) & off, (G_true == 0) & off


def config_metrics(G_hats_kg, G_hats_ng, G_true, epochs_kg, band_quantile=0.95) -> dict:
    """Seed-pooled metrics for one (l1, ceiling) config, thresholded
    against the config's OWN pooled NG band (matched-null discipline)."""
    band = pooled_null_band(G_hats_ng, band_quantile)
    pos_m, neg_m, zero_m = edge_masks(G_true)
    per_seed = [sign_f1(Gk, G_true, band) for Gk in G_hats_kg]
    keys = ["sign_f1", "pos_f1", "neg_f1", "sign_accuracy_true_edges", "false_edge_rate", "neg_recall"]
    mean_metrics = {
        k: float(np.mean([m[k] for m in per_seed if np.isfinite(m.get(k, np.nan))] or [np.nan]))
        for k in keys
    }
    gpos_by_seed = [float(np.mean(G[pos_m])) for G in G_hats_kg]
    gneg_by_seed = [float(np.mean(G[neg_m])) for G in G_hats_kg]
    zleak_by_seed = [float(np.max(np.abs(G[zero_m]))) for G in G_hats_kg]
    gneg_mean = float(np.mean(gneg_by_seed))
    return {
        "band": band,
        **mean_metrics,
        "Gpos": float(np.mean(gpos_by_seed)),
        "Gneg": gneg_mean,
        "Gneg_by_seed": gneg_by_seed,
        "zLeak": float(np.mean(zleak_by_seed)),
        "neg_sign_all_seeds": bool(all(g < 0 for g in gneg_by_seed)),
        "neg_separation": float(abs(gneg_mean) / band) if band > 0 else float("inf"),
        "epochs_run": [int(e) for e in epochs_kg],
    }


def run_config(l1: float, ceiling: int, g_neg: float, kg_only: bool = False):
    """Fit the 3 tuning seeds at one config; returns (G_hats_kg,
    G_hats_ng, G_true, epochs_kg). With ``kg_only`` the NG fits are skipped
    (phase 2 reuses phase 1's dose-invariant band)."""
    cfg = TransferConfig(l1_weight=l1, n_epochs=ceiling, device="cpu")
    G_hats_kg, G_hats_ng, epochs_kg, G_true = [], [], [], None
    for s in TUNING_SEEDS:
        kg = generate_signed_twin("syn_t_kg", DENSITY, seed=s, n_learners=N_LEARNERS,
                                  decoupling=DECOUPLING, g_pos=G_POS, g_neg=g_neg)
        G_true = kg.truth.G_true
        t0 = time.time()
        Gk, ep_k, nll_k = fit_with_trace(kg, cfg, model_seed=s)
        G_hats_kg.append(Gk)
        epochs_kg.append(ep_k)
        print(f"    seed {s} KG: {ep_k} epochs, nll {nll_k:.4f}, {time.time() - t0:.0f}s", flush=True)
        if not kg_only:
            ng = generate_signed_twin("syn_t_ng", DENSITY, seed=s, n_learners=N_LEARNERS,
                                      decoupling=DECOUPLING, g_pos=G_POS, g_neg=g_neg)
            t0 = time.time()
            Gn, ep_n, nll_n = fit_with_trace(ng, cfg, model_seed=s)
            G_hats_ng.append(Gn)
            print(f"    seed {s} NG: {ep_n} epochs, nll {nll_n:.4f}, {time.time() - t0:.0f}s", flush=True)
    return G_hats_kg, G_hats_ng, G_true, epochs_kg


def pick_winner(phase1: dict) -> tuple:
    """The pre-registered rule. Returns (winner_l1 or None, reason)."""
    dose = G_NEG_REF
    qualified = []
    for l1 in L1_LADDER:
        rows = [phase1[f"l1={l1}|ceil={c}"] for c in CEILINGS]
        ok = all(
            r["neg_sign_all_seeds"]
            and 0.5 * dose <= abs(r["Gneg"]) <= 2.0 * dose
            and r["false_edge_rate"] <= 0.05
            for r in rows
        )
        if ok:
            qualified.append((min(r["neg_separation"] for r in rows), -l1, l1))
    if not qualified:
        return None, "no config met the sign/magnitude/FER qualification at both ceilings"
    qualified.sort(reverse=True)  # max score; ties -> larger -l1 == smaller l1
    score, _, l1 = qualified[0]
    return l1, f"epoch-robust separation {score:.2f}"


def main() -> None:
    t_start = time.time()
    results: dict = {"tuning_seeds": TUNING_SEEDS, "cell": {
        "density": DENSITY, "n_learners": N_LEARNERS, "decoupling": DECOUPLING,
        "g_pos": G_POS, "g_neg_ref": G_NEG_REF}, "phase1": {}, "phase2": {}}

    print("== Phase 1: L1 x ceiling grid ==", flush=True)
    for l1 in L1_LADDER:
        for ceiling in CEILINGS:
            key = f"l1={l1}|ceil={ceiling}"
            print(f"  config {key}", flush=True)
            Gk, Gn, G_true, ep = run_config(l1, ceiling, G_NEG_REF)
            m = config_metrics(Gk, Gn, G_true, ep)
            results["phase1"][key] = m
            print(f"    -> band {m['band']:.4f}  zLeak {m['zLeak']:.4f}  Gpos {m['Gpos']:.4f}  "
                  f"Gneg {m['Gneg']:.4f}  sep {m['neg_separation']:.2f}  negF1 {m['neg_f1']:.3f}  "
                  f"FER {m['false_edge_rate']:.3f}  epochs {m['epochs_run']}", flush=True)

    winner, reason = pick_winner(results["phase1"])
    results["winner_l1"] = winner
    results["winner_reason"] = reason
    print(f"== Winner: l1={winner} ({reason}) ==", flush=True)

    print("== Phase 2: dose-response on the negative edge ==", flush=True)
    phase2_l1s = [G for G in dict.fromkeys([1e-3, winner]) if G is not None]
    for l1 in phase2_l1s:
        band = results["phase1"][f"l1={l1}|ceil=500"]["band"]  # dose-invariant NG band
        for dose in DOSES:
            key = f"l1={l1}|dose={dose}"
            if dose == G_NEG_REF:
                src = results["phase1"][f"l1={l1}|ceil=500"]
                results["phase2"][key] = {**{k: src[k] for k in
                    ("band", "Gneg", "Gneg_by_seed", "neg_f1", "neg_sign_all_seeds",
                     "neg_separation", "zLeak")}, "reused_phase1": True}
                print(f"  {key}: reused phase-1 cell", flush=True)
                continue
            print(f"  config {key}", flush=True)
            Gk, _, G_true, ep = run_config(l1, 500, dose, kg_only=True)
            neg_m = edge_masks(G_true)[1]
            zero_m = edge_masks(G_true)[2]
            per_seed = [sign_f1(G, G_true, band) for G in Gk]
            gneg_by_seed = [float(np.mean(G[neg_m])) for G in Gk]
            gneg_mean = float(np.mean(gneg_by_seed))
            m2 = {
                "band": band,
                "Gneg": gneg_mean,
                "Gneg_by_seed": gneg_by_seed,
                "neg_f1": float(np.mean([m["neg_f1"] for m in per_seed])),
                "neg_sign_all_seeds": bool(all(g < 0 for g in gneg_by_seed)),
                "neg_separation": float(abs(gneg_mean) / band) if band > 0 else float("inf"),
                "zLeak": float(np.mean([float(np.max(np.abs(G[zero_m]))) for G in Gk])),
                "epochs_run": [int(e) for e in ep],
            }
            results["phase2"][key] = m2
            print(f"    -> Gneg {m2['Gneg']:.4f}  sep {m2['neg_separation']:.2f}  "
                  f"negF1 {m2['neg_f1']:.3f}", flush=True)

    results["elapsed_s"] = round(time.time() - t_start)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"== done in {results['elapsed_s']}s -> {OUT_PATH} ==", flush=True)


if __name__ == "__main__":
    main()
