"""run_nrm_bench.py -- head-to-head NRM engine bench: ma-irt-NRM vs deep_irt-NRM.

Mirrors run_bench.py (the GPCM engine bench) for the NRM (Bock 1972 nominal /
multiple-choice) decoder.  Runs SYNCHRONOUSLY on CPU or CUDA, no detach.  For
each (engine x dataset x seed) cell it records:

  - prediction : top-1 acc, multiclass NLL, macro-averaged OvR AUC (NO QWK --
                 options are unordered).
  - recovery   : ability theta; per-option slopes a_k; per-option intercepts
                 c_k (Spearman, sign/center aligned).
  - cost       : trainable params, train wall-clock, infer wall-clock.

Engines
-------
  deep_irt-NRM   : plain LSTM + deep_irt item-only NRM decoder.
  ma-irt-{lstm,dkvmn}-NRM : ma-irt encoder (separated ability pathway) + the
                   state-conditioned NRM head.

Datasets
--------
  nrm_static  : NRM, ability fixed per learner.
  nrm_dynamic : NRM, ability a per-step random walk.

3 seeds; the table reports mean +/- sd per cell.

Usage
-----
  source ~/anaconda3/etc/profile.d/conda.sh && conda activate research
  KMP_DUPLICATE_LIB_OK=TRUE \
  PYTHONPATH="<repo>;<repo>/rl/src;<repo>/ma-irt" \
      python deep_irt/bench/run_nrm_bench.py [--quick] [--device cuda] \
          [--epochs N] [--seeds 0 1 2] [--no-dynamic]

Outputs:
  deep_irt/bench/outputs/nrm_bench_results.json   -- all numbers (per seed)
  deep_irt/bench/outputs/nrm_bench_table.md       -- mean+/-sd table + verdict
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
from deep_irt.bench import nrm_metrics as M
from deep_irt.bench.nrm_engines import DeepIRTNRMEngine, MaIrtNRMEngine

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "outputs"
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# one (engine, dataset) cell
# ---------------------------------------------------------------------------

def time_inference(engine, repeats=3):
    t = []
    for _ in range(repeats):
        t0 = time.time()
        engine.predict_heldout()
        t.append(time.time() - t0)
    return float(np.median(t))


def run_cell(engine, ds) -> dict:
    probs, targ = engine.predict_heldout()
    pred = M.prediction_metrics(probs, targ, ds.cfg.n_options)

    rec = engine.recover()
    seen = rec.get("seen", None)
    item = M.item_recovery(rec["a"], rec["c"], ds.gt.a, ds.gt.c, seen=seen)

    if ds.cfg.kind == "dynamic":
        theta = M.theta_recovery_dynamic(rec["theta_traj"], ds.gt.theta_traj)
        theta_headline = theta["theta_netdrift_spearman"]
    else:
        theta = M.theta_recovery_static(rec["theta_final"], ds.gt.theta0)
        theta_headline = theta["theta_spearman"]

    infer_t = time_inference(engine)
    cost = engine.cost()

    return {
        "label": engine.label,
        "encoder": engine.encoder_name,
        "decoder": "nrm",
        "dataset": ds.cfg.name,
        "kind": ds.cfg.kind,
        "n_options": ds.cfg.n_options,
        **pred,
        **theta,
        **item,
        "theta_headline": theta_headline,
        "n_params": cost["n_params"],
        "train_time": cost["train_time"],
        "infer_time": infer_t,
    }


# ---------------------------------------------------------------------------
# main grid
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small/fast smoke grid")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--no-dynamic", action="store_true",
                    help="skip the dynamic-theta NRM variant")
    ap.add_argument("--rerender", action="store_true",
                    help="re-render the table from the existing results JSON "
                         "without retraining")
    args = ap.parse_args()

    if args.rerender:
        with (OUT / "nrm_bench_results.json").open(encoding="utf-8") as fh:
            blob = json.load(fh)
        m = blob["meta"]
        table = render_table(blob["rows"], m["device"], m["epochs"],
                             m["N"], m["Q"], m["T"], m["seeds"], m["kinds"])
        (OUT / "nrm_bench_table.md").write_text(table, encoding="utf-8")
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
    print(f"=== NRM bench: device={device} epochs={epochs} N={N} Q={Q} T={T} "
          f"seeds={args.seeds} kinds={kinds} ===")

    rows = []  # one dict per (engine, dataset, seed)

    for seed in args.seeds:
        # Build datasets for this seed (shared ground truth + split per dataset).
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
            engines = [
                DeepIRTNRMEngine(ds, device=device, seed=seed),
                MaIrtNRMEngine(ds, encoder="lstm", device=device, seed=seed),
                MaIrtNRMEngine(ds, encoder="dkvmn", device=device, seed=seed),
            ]
            for eng in engines:
                eng.fit(n_epochs=epochs)
                r = run_cell(eng, ds)
                r["seed"] = seed
                rows.append(r)
                print(f"   {r['label']:22s} acc={r['acc']:.3f} nll={r['nll']:.3f} "
                      f"auc={r['macro_auc']:.3f} | theta={r['theta_headline']:.3f} "
                      f"a_k={r['a_spearman']:.3f} "
                      f"c_k={r['c_spearman']:.3f} t={r['train_time']:.1f}s")

    # ---- persist + render ----
    with (OUT / "nrm_bench_results.json").open("w", encoding="utf-8") as fh:
        json.dump({"meta": {"device": device, "epochs": epochs,
                            "N": N, "Q": Q, "T": T, "seeds": args.seeds,
                            "kinds": kinds},
                   "rows": rows}, fh, indent=2)

    table = render_table(rows, device, epochs, N, Q, T, args.seeds, kinds)
    (OUT / "nrm_bench_table.md").write_text(table, encoding="utf-8")
    print("\n" + table)
    print(f"\nwrote {OUT/'nrm_bench_results.json'} and {OUT/'nrm_bench_table.md'}")


# ---------------------------------------------------------------------------
# rendering: aggregate mean +/- sd over seeds per (label, dataset)
# ---------------------------------------------------------------------------

_METRICS = ["acc", "nll", "macro_auc", "theta_headline",
            "a_spearman", "c_spearman"]
_COST = ["n_params", "train_time", "infer_time"]


def _agg(vals):
    a = np.asarray([v for v in vals if v is not None and v == v], dtype=float)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std())


def _fmt(mean, sd, p=3):
    if mean != mean:
        return "  -  "
    if sd != sd or sd == 0:
        return f"{mean:.{p}f}"
    return f"{mean:.{p}f}±{sd:.{p}f}"


def render_table(rows, device, epochs, N, Q, T, seeds, kinds) -> str:
    # group by (dataset, label) preserving a stable engine order
    order = ["deep_irt-LSTM/nrm", "ma-irt-lstm/nrm", "ma-irt-dkvmn/nrm"]
    grouped = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for m in _METRICS + _COST:
            v = r.get(m)
            # theta_headline carries an unidentified latent ORIENTATION (its
            # sign flips freely across seeds), so the recovery quality is the
            # magnitude. Aggregate |theta| per seed; otherwise +0.9 and -0.9
            # runs would cancel to ~0. a_k/c_k are already sign-aligned
            # to truth inside item_recovery, so they are left as stored.
            if m == "theta_headline" and v is not None and v == v:
                v = abs(v)
            grouped[(r["dataset"], r["label"])][m].append(v)

    lines = []
    lines.append("# NRM engine head-to-head benchmark (ma-irt-NRM vs deep_irt-NRM)\n")
    lines.append(f"device={device}  epochs={epochs}  N={N}  Q={Q}  T={T}  "
                 f"K=4  seeds={seeds}  (torch {torch.__version__})\n")
    lines.append("| engine/encoder | dataset | acc | NLL | macroAUC | |theta| | "
                 "a_k (sp) | c_k (sp) | params | train s |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    dataset_names = [f"nrm_{k}" for k in kinds]
    for dsname in dataset_names:
        for label in order:
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
        "\nNotes: |theta| = |Spearman| of recovered vs true ability (the latent "
        "orientation is unidentified so the sign is dropped; static: "
        "final-step level; dynamic: within-learner net drift). a_k / c_k = "
        "Spearman of recovered per-option slopes / intercepts vs truth, both "
        "mean-centered (Bock) and sign-aligned before scoring. NLL = multiclass "
        "cross-entropy (lower better). macroAUC = macro-averaged one-vs-rest AUC. "
        "QWK omitted (options unordered). ma-irt-NRM: theta item-blind, slopes "
        "a_k state-conditioned on [joint,item]. deep_irt-NRM: a_k / c_k "
        "item-only. Cells are mean±sd over seeds.")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
