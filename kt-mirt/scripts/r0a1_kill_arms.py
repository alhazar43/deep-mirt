"""The two cheapest A1 confound arms at the certified dose (follows
`scripts/r0a1_floor_cert.py`, certified negative-dose floor 0.04; design
`_planning/design/a1_design.md` v1.1 sections 4.4 (CT3-iii) and 4.6 (CT6)).

Arm 1 -- CT3-iii shuffle-order (a KILL arm): refit on SYN-T-KG (dose
0.04, certification seeds 0-4) whose per-learner event order has been
re-drawn by `growth.battery.permute_cross_kc_interleaving` (per-KC
internal order preserved, cross-KC interleaving randomized -- destroys
the causal lag, preserves practice counts). Pre-registered bar (design
section 5 table): the recovered |G| at the true edges must collapse to
<= 10% of the matched-form (unshuffled) magnitude, read from the floor-
cert JSON (same seeds, same config -> identical matched fits). A signed
read that SURVIVES the shuffle would be schedule-count driven, not
lag-driven, and kills the causal framing.

Arm 2 -- CT6 phantom-transfer sensitivity control (NON-kill, reported
either way): refit SYN-T-NG (null twin) with the free amortized
per-learner gamma head enabled (`TransferConfig.phantom_gamma=True`).
Pre-registered EXPECTATION: it fabricates -- the per-learner p95
transfer magnitude (p95 over learners of gamma_i, times the phantom
fit's max off-diagonal |G|) exceeds BOTH the pinned variant's magnitude
(max off-diagonal |G| of the matched pinned NG fit, gamma == 1) AND the
pooled null band. Fabrication confirms the tail metric is sensitive;
non-fabrication is an informative finding. The gamma pin is retained
regardless (v1.1 reframing).

Run:  python -u scripts/r0a1_kill_arms.py   (cwd kt-mirt/, research env)
Out:  outputs/a1/r0a1/r0a1_kill_arms.json + stdout.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace as dc_replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r0a1_study import DECOUPLING, DENSITY, G_POS, N_LEARNERS, edge_masks  # noqa: E402

from kt_mirt.growth.battery import permute_cross_kc_interleaving  # noqa: E402
from kt_mirt.growth.tracker import build_learner_batch  # noqa: E402
from kt_mirt.transfer.ct0 import pooled_null_band  # noqa: E402
from kt_mirt.transfer.model import (  # noqa: E402
    TransferConfig, TransferModel, seq_lens_from_mask, train_transfer,
)
from kt_mirt.transfer.synth import generate_signed_twin  # noqa: E402

CERT_SEEDS = [0, 1, 2, 3, 4]
DOSE = 0.04  # the certified floor (r0a1_floor_cert.json)
L1_DEFAULT = 1e-3
CEILING_DEFAULT = 500
CERT_PATH = Path("outputs/a1/r0a1/r0a1_floor_cert.json")
OUT_PATH = Path("outputs/a1/r0a1/r0a1_kill_arms.json")


def fit_model(twin, cfg: TransferConfig, model_seed: int):
    """One gate-fixed fit returning (model, batch) so arm 2 can read the
    per-learner gamma head afterwards. Mirrors `r0a1_study.fit_with_trace`."""
    batch = build_learner_batch(twin.learners, b_true=twin.truth.b_true, device=cfg.device)
    cfg_fit = dc_replace(cfg, seed=model_seed, floor=twin.truth.floor, m_fixed=True)
    torch.manual_seed(model_seed)
    model = TransferModel(
        num_items=twin.item_bank.n_items, n_kcs=twin.n_kcs, cfg=cfg_fit, m_init=twin.truth.M_gen
    )
    n_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        train_transfer(model, batch, cfg_fit)
    finally:
        torch.set_num_threads(n_threads)
    return model, batch


def main() -> None:
    t_start = time.time()
    cfg = TransferConfig(l1_weight=L1_DEFAULT, n_epochs=CEILING_DEFAULT, device="cpu")
    cert = json.loads(CERT_PATH.read_text(encoding="utf-8"))
    matched = cert["doses"][str(DOSE)]
    matched_gpos, matched_gneg = matched["Gpos"], matched["Gneg"]
    band = cert["band"]
    results = {"dose": DOSE, "cert_seeds": CERT_SEEDS, "band": band,
               "matched": {"Gpos": matched_gpos, "Gneg": matched_gneg}}

    print(f"== Arm 1: CT3-iii shuffle-order at dose {DOSE} ==", flush=True)
    gpos_sh, gneg_sh, maxoff_sh = [], [], []
    for s in CERT_SEEDS:
        kg = generate_signed_twin("syn_t_kg", DENSITY, seed=s, n_learners=N_LEARNERS,
                                  decoupling=DECOUPLING, g_pos=G_POS, g_neg=DOSE)
        shuffled = permute_cross_kc_interleaving(kg.learners, np.random.default_rng(1000 + s))
        twin_sh = dc_replace(kg, learners=shuffled)
        t0 = time.time()
        model, _ = fit_model(twin_sh, cfg, model_seed=s)
        G_sh = model.recovered_G()
        pos_m, neg_m, _ = edge_masks(kg.truth.G_true)
        off = ~np.eye(G_sh.shape[0], dtype=bool)
        gpos_sh.append(float(np.mean(G_sh[pos_m])))
        gneg_sh.append(float(np.mean(G_sh[neg_m])))
        maxoff_sh.append(float(np.max(np.abs(G_sh[off]))))
        print(f"  seed {s}: Gpos_sh {gpos_sh[-1]:+.4f}  Gneg_sh {gneg_sh[-1]:+.4f}  "
              f"maxoff {maxoff_sh[-1]:.4f}  {time.time() - t0:.0f}s", flush=True)
    ratio_pos = abs(float(np.mean(gpos_sh))) / abs(matched_gpos)
    ratio_neg = abs(float(np.mean(gneg_sh))) / abs(matched_gneg)
    shuffle_pass = bool(ratio_pos <= 0.10 and ratio_neg <= 0.10)
    results["shuffle"] = {
        "Gpos_shuffled_by_seed": gpos_sh, "Gneg_shuffled_by_seed": gneg_sh,
        "maxoff_by_seed": maxoff_sh, "ratio_pos": ratio_pos, "ratio_neg": ratio_neg,
        "bar": "<= 0.10 of matched magnitude on both true edges", "pass": shuffle_pass,
    }
    print(f"  -> collapse ratios: pos {ratio_pos:.3f}  neg {ratio_neg:.3f}  "
          f"PASS={shuffle_pass}", flush=True)

    print("== Arm 2: CT6 phantom-gamma sensitivity on the null twin ==", flush=True)
    per_seed = []
    for s in CERT_SEEDS:
        ng = generate_signed_twin("syn_t_ng", DENSITY, seed=s, n_learners=N_LEARNERS,
                                  decoupling=DECOUPLING, g_pos=G_POS, g_neg=DOSE)
        t0 = time.time()
        pinned_model, _ = fit_model(ng, cfg, model_seed=s)
        phantom_model, batch = fit_model(ng, dc_replace(cfg, phantom_gamma=True), model_seed=s)
        off = ~np.eye(ng.n_kcs, dtype=bool)
        pinned_mag = float(np.max(np.abs(pinned_model.recovered_G()[off])))
        with torch.no_grad():
            seq_lens = seq_lens_from_mask(batch.seq_mask)
            gamma = phantom_model._maybe_gamma(batch, seq_lens).cpu().numpy()
        phantom_maxoff = float(np.max(np.abs(phantom_model.recovered_G()[off])))
        p95 = float(np.quantile(gamma, 0.95)) * phantom_maxoff
        fabricates = bool(p95 > pinned_mag and p95 > band)
        per_seed.append({"seed": s, "pinned_maxoff": pinned_mag,
                         "phantom_maxoff": phantom_maxoff,
                         "gamma_p95": float(np.quantile(gamma, 0.95)),
                         "gamma_median": float(np.median(gamma)),
                         "perlearner_p95_magnitude": p95, "fabricates": fabricates})
        print(f"  seed {s}: pinned {pinned_mag:.4f}  phantom_p95_mag {p95:.4f} "
              f"(gamma_p95 {per_seed[-1]['gamma_p95']:.2f})  fabricates={fabricates}  "
              f"{time.time() - t0:.0f}s", flush=True)
    n_fab = sum(r["fabricates"] for r in per_seed)
    results["phantom"] = {"per_seed": per_seed, "n_fabricating": n_fab,
                          "expected": "fabricates (sensitivity confirmed); non-kill either way"}
    print(f"  -> {n_fab}/{len(CERT_SEEDS)} seeds fabricate", flush=True)

    results["elapsed_s"] = round(time.time() - t_start)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"== done in {results['elapsed_s']}s -> {OUT_PATH} ==", flush=True)


if __name__ == "__main__":
    main()
