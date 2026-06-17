"""run_nrm_bench_itemonly.py -- 3-way NRM engine head-to-head.

Follow-up to ``run_nrm_bench.py``.  The prior 2-way bench found a SPLIT: the
ma-irt NRM head (per-option slopes a_k STATE-CONDITIONED on [joint, item]) wins
prediction + ability but loses NRM item-parameter recovery, decisively and
unstably, to the deep_irt (item-only a_k).  This runner adds the MIDDLE PATH --
ma-irt's encoder + item-blind theta, but item-only NRM slopes a_k (and c_k) --
and runs the 3-way comparison on the SAME true-NRM data so we can isolate the
conditioning as the sole cause.

Engines (LSTM encoder, the one that matters)
--------------------------------------------
  deep_irt-LSTM/nrm        : plain LSTM + deep_irt item-only NRM decoder.
  ma-irt-lstm/nrm           : ma-irt encoder + STATE-CONDITIONED NRM head
                              (the existing result, re-run here for a clean
                              same-seed side-by-side).
  ma-irt-lstm/nrm-itemonly  : ma-irt encoder + ITEM-ONLY NRM head (NEW).

DKVMN is optional (--with-dkvmn) -- the prior run showed it strictly worse and
more unstable than LSTM, so it is off by default here.

Everything else (data, metrics, |theta| aggregation, 3 seeds, static+dynamic,
table) is identical to run_nrm_bench.py; only the engine roster + output paths
differ.  Existing nrm_bench_{results.json,table.md} are left untouched.

Outputs:
  deep_irt/bench/outputs/nrm_bench_itemonly_results.json
  deep_irt/bench/outputs/nrm_bench_itemonly_table.md

Usage
-----
  source ~/anaconda3/etc/profile.d/conda.sh && conda activate research
  KMP_DUPLICATE_LIB_OK=TRUE \
  PYTHONPATH="<repo>;<repo>/rl/src;<repo>/ma-irt" \
      python deep_irt/bench/run_nrm_bench_itemonly.py [--quick] [--device cuda] \
          [--epochs N] [--seeds 0 1 2] [--no-dynamic] [--with-dkvmn]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from deep_irt.bench.nrm_datagen import NRMDataConfig, generate
from deep_irt.bench.nrm_engines import DeepIRTNRMEngine, MaIrtNRMEngine
from deep_irt.bench.run_nrm_bench import run_cell, _agg, _fmt, _METRICS, _COST

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "outputs"
OUT.mkdir(exist_ok=True)

# Stable engine order for the table (3-way + optional dkvmn rows).
_ORDER = [
    "deep_irt-LSTM/nrm",
    "ma-irt-lstm/nrm",
    "ma-irt-lstm/nrm-itemonly",
    "ma-irt-dkvmn/nrm",
    "ma-irt-dkvmn/nrm-itemonly",
]


def build_engines(ds, device, seed, with_dkvmn):
    engines = [
        DeepIRTNRMEngine(ds, device=device, seed=seed),
        MaIrtNRMEngine(ds, encoder="lstm", device=device, seed=seed,
                       slopes_mode="state_conditioned"),
        MaIrtNRMEngine(ds, encoder="lstm", device=device, seed=seed,
                       slopes_mode="item_only"),
    ]
    if with_dkvmn:
        engines += [
            MaIrtNRMEngine(ds, encoder="dkvmn", device=device, seed=seed,
                           slopes_mode="state_conditioned"),
            MaIrtNRMEngine(ds, encoder="dkvmn", device=device, seed=seed,
                           slopes_mode="item_only"),
        ]
    return engines


def render_table(rows, device, epochs, N, Q, T, seeds, kinds) -> str:
    grouped = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for m in _METRICS + _COST:
            v = r.get(m)
            if m == "theta_headline" and v is not None and v == v:
                v = abs(v)   # |theta|: latent orientation unidentified
            grouped[(r["dataset"], r["label"])][m].append(v)

    lines = []
    lines.append("# NRM engine 3-way benchmark "
                 "(deep_irt vs ma-irt state-cond vs ma-irt item-only)\n")
    lines.append(f"device={device}  epochs={epochs}  N={N}  Q={Q}  T={T}  "
                 f"K=4  seeds={seeds}  (torch {torch.__version__})\n")
    lines.append("| engine/encoder | dataset | acc | NLL | macroAUC | |theta| | "
                 "a_k (sp) | c_k (sp) | params | train s |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    dataset_names = [f"nrm_{k}" for k in kinds]
    for dsname in dataset_names:
        for label in _ORDER:
            key = (dsname, label)
            if key not in grouped:
                continue
            g = grouped[key]
            cells = {m: _agg(g[m]) for m in _METRICS}
            np_mean = int(_agg(g["n_params"])[0])
            tt_mean = _agg(g["train_time"])[0]
            lines.append(
                f"| {label} | {dsname} | "
                f"{_fmt(*cells['acc'])} | {_fmt(*cells['nll'])} | "
                f"{_fmt(*cells['macro_auc'])} | {_fmt(*cells['theta_headline'])} | "
                f"{_fmt(*cells['a_spearman'])} | "
                f"{_fmt(*cells['c_spearman'])} | {np_mean} | {tt_mean:.1f} |"
            )

    lines.append(
        "\nNotes: |theta| = |Spearman| of recovered vs true ability (latent "
        "orientation unidentified, sign dropped; static: final-step level; "
        "dynamic: within-learner net drift). a_k / c_k = Spearman of recovered "
        "per-option slopes / intercepts vs truth, Bock mean-centered + "
        "sign-aligned. NLL = multiclass cross-entropy (lower better). macroAUC = "
        "macro OvR AUC. QWK omitted (options unordered). Engines: deep_irt "
        "a_k/c_k item-only; ma-irt/nrm a_k STATE-CONDITIONED on [joint,item]; "
        "ma-irt/nrm-itemonly a_k ITEM-ONLY (item_embed), theta item-blind in both "
        "ma-irt heads. Cells mean±sd over seeds.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--no-dynamic", action="store_true")
    ap.add_argument("--with-dkvmn", action="store_true")
    ap.add_argument("--rerender", action="store_true")
    args = ap.parse_args()

    if args.rerender:
        with (OUT / "nrm_bench_itemonly_results.json").open(encoding="utf-8") as fh:
            blob = json.load(fh)
        m = blob["meta"]
        table = render_table(blob["rows"], m["device"], m["epochs"],
                             m["N"], m["Q"], m["T"], m["seeds"], m["kinds"])
        (OUT / "nrm_bench_itemonly_table.md").write_text(table, encoding="utf-8")
        print(table)
        return

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if args.quick:
        N, Q, T = 200, 30, 30
        epochs = args.epochs or 40
    else:
        N, Q, T = 800, 60, 60
        epochs = args.epochs or 150

    kinds = ["static"] if args.no_dynamic else ["static", "dynamic"]
    print(f"=== NRM item-only bench: device={device} epochs={epochs} "
          f"N={N} Q={Q} T={T} seeds={args.seeds} kinds={kinds} "
          f"dkvmn={args.with_dkvmn} ===")

    rows = []
    t_start = time.time()
    for seed in args.seeds:
        data_cfgs = []
        if "static" in kinds:
            data_cfgs.append(NRMDataConfig(
                name="nrm_static", kind="static", n_options=4,
                n_learners=N, n_items=Q, seq_len=T, seed=seed))
        if "dynamic" in kinds:
            data_cfgs.append(NRMDataConfig(
                name="nrm_dynamic", kind="dynamic", n_options=4,
                n_learners=N, n_items=Q, seq_len=T, drift_sigma=0.15, seed=seed))
        datasets = {c.name: generate(c) for c in data_cfgs}

        for dsname, ds in datasets.items():
            print(f"\n--- seed {seed} | dataset {dsname} ---")
            for eng in build_engines(ds, device, seed, args.with_dkvmn):
                eng.fit(n_epochs=epochs)
                r = run_cell(eng, ds)
                r["seed"] = seed
                rows.append(r)
                print(f"   {r['label']:28s} acc={r['acc']:.3f} nll={r['nll']:.3f} "
                      f"auc={r['macro_auc']:.3f} | theta={r['theta_headline']:.3f} "
                      f"a_k={r['a_spearman']:.3f} "
                      f"c_k={r['c_spearman']:.3f} t={r['train_time']:.1f}s")

    with (OUT / "nrm_bench_itemonly_results.json").open("w", encoding="utf-8") as fh:
        json.dump({"meta": {"device": device, "epochs": epochs,
                            "N": N, "Q": Q, "T": T, "seeds": args.seeds,
                            "kinds": kinds, "with_dkvmn": args.with_dkvmn},
                   "rows": rows}, fh, indent=2)

    table = render_table(rows, device, epochs, N, Q, T, args.seeds, kinds)
    (OUT / "nrm_bench_itemonly_table.md").write_text(table, encoding="utf-8")
    print("\n" + table)
    print(f"\ntotal wall {time.time() - t_start:.1f}s")
    print(f"wrote {OUT/'nrm_bench_itemonly_results.json'} and "
          f"{OUT/'nrm_bench_itemonly_table.md'}")


if __name__ == "__main__":
    main()
