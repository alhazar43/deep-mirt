"""Certification of the negative-transfer detectable-dose floor (avenue
A1; follows `scripts/r0a1_study.py`, whose held-out-seed verdict was
NO-WINNER: no L1 setting in the ladder holds true-zero cells at zero
without shrinking the true edges faster -- see the study JSON and
`_planning/ct0_power_result.md`).

Because no tuning changed the DEFAULT trainer config (l1=1e-3, ceiling
500), certification seeds 0-4 remain clean for it (the held-out-seed
discipline: seeds 0-4 were never used to select anything). This script
certifies the dose-response of the negative/interference edge at the
default config on those seeds, at the CT0 reference cell (KDD-shaped
density, D=3, N=500, decoupling=0.90).

Pre-registered per-dose detection bar (a dose is DETECTED iff all three
hold on the 5 certification seeds):
  (1) seed-mean negative-half sign-F1 >= 0.75 (the CT1 negative clause),
  (2) the negative edge's fitted sign is negative in EVERY seed,
  (3) seed-mean false-edge rate on true-zero cells <= 0.05.
The certified floor is the smallest dose in {0.01, 0.02, 0.04, 0.08}
meeting the bar. The NG twin is dose-invariant (G_true = 0; schedule from
the KG support), so ONE pooled 5-seed NG band serves every dose.

Run:  python -u scripts/r0a1_floor_cert.py   (cwd kt-mirt/, research env)
Out:  outputs/a1/r0a1/r0a1_floor_cert.json + stdout table.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r0a1_study import (  # noqa: E402
    DECOUPLING, DENSITY, G_NEG_REF, G_POS, N_LEARNERS,
    config_metrics, fit_with_trace,
)

from kt_mirt.transfer.ct0 import pooled_null_band  # noqa: E402
from kt_mirt.transfer.model import TransferConfig  # noqa: E402
from kt_mirt.transfer.synth import generate_signed_twin  # noqa: E402

CERT_SEEDS = [0, 1, 2, 3, 4]
DOSES = [0.01, 0.02, 0.04, 0.08]
L1_DEFAULT = 1e-3
CEILING_DEFAULT = 500
OUT_PATH = Path("outputs/a1/r0a1/r0a1_floor_cert.json")


def main() -> None:
    t_start = time.time()
    cfg = TransferConfig(l1_weight=L1_DEFAULT, n_epochs=CEILING_DEFAULT, device="cpu")

    print("== NG band (5 certification seeds, dose-invariant) ==", flush=True)
    G_hats_ng = []
    for s in CERT_SEEDS:
        ng = generate_signed_twin("syn_t_ng", DENSITY, seed=s, n_learners=N_LEARNERS,
                                  decoupling=DECOUPLING, g_pos=G_POS, g_neg=G_NEG_REF)
        t0 = time.time()
        Gn, ep, nll = fit_with_trace(ng, cfg, model_seed=s)
        G_hats_ng.append(Gn)
        print(f"  seed {s}: {ep} epochs, nll {nll:.4f}, {time.time() - t0:.0f}s", flush=True)
    band = pooled_null_band(G_hats_ng)
    print(f"  -> pooled band {band:.4f}", flush=True)

    results = {"cert_seeds": CERT_SEEDS, "band": band,
               "cell": {"density": DENSITY, "n_learners": N_LEARNERS,
                        "decoupling": DECOUPLING, "g_pos": G_POS,
                        "l1": L1_DEFAULT, "ceiling": CEILING_DEFAULT},
               "doses": {}}

    print("== dose ladder (KG per seed) ==", flush=True)
    floor = None
    for dose in DOSES:
        G_hats_kg, epochs, G_true = [], [], None
        for s in CERT_SEEDS:
            kg = generate_signed_twin("syn_t_kg", DENSITY, seed=s, n_learners=N_LEARNERS,
                                      decoupling=DECOUPLING, g_pos=G_POS, g_neg=dose)
            G_true = kg.truth.G_true
            t0 = time.time()
            Gk, ep, nll = fit_with_trace(kg, cfg, model_seed=s)
            G_hats_kg.append(Gk)
            epochs.append(ep)
            print(f"  dose {dose} seed {s}: {ep} epochs, nll {nll:.4f}, "
                  f"{time.time() - t0:.0f}s", flush=True)
        m = config_metrics(G_hats_kg, G_hats_ng, G_true, epochs)
        # config_metrics recomputes the same pooled band from G_hats_ng.
        detected = bool(m["neg_f1"] >= 0.75 and m["neg_sign_all_seeds"]
                        and m["false_edge_rate"] <= 0.05)
        results["doses"][str(dose)] = {**m, "detected": detected}
        if detected and floor is None:
            floor = dose
        print(f"  dose {dose}: Gneg {m['Gneg']:.4f}  sep {m['neg_separation']:.2f}  "
              f"negF1 {m['neg_f1']:.3f}  posF1 {m['pos_f1']:.3f}  "
              f"FER {m['false_edge_rate']:.3f}  detected={detected}", flush=True)

    results["certified_floor"] = floor
    results["elapsed_s"] = round(time.time() - t_start)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"== certified floor: {floor} (in {results['elapsed_s']}s) -> {OUT_PATH} ==", flush=True)


if __name__ == "__main__":
    main()
