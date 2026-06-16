"""
run_bench.py -- FAIR head-to-head: deep_irt (vanilla LSTM) vs ma-irt (DKVMN++).

Runs SYNCHRONOUSLY on CPU or CUDA (torch only; no Ollama, no detach).  Produces
one table over (engine/encoder x decoder x dataset) with prediction (acc, QWK,
AUC), recovery (theta, discrimination a, step-thresholds b), and cost
(train/infer wall-clock, trainable params).

Usage
-----
  source ~/anaconda3/etc/profile.d/conda.sh && conda activate research
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="deep_irt;rl/src;ma-irt" \
      python deep_irt/bench/run_bench.py [--quick] [--device cuda] [--epochs N]

Outputs:
  deep_irt/bench/outputs/bench_results.json   -- all numbers
  deep_irt/bench/outputs/bench_table.md       -- the rendered table + verdict
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from deep_irt.bench.datagen import BenchDataConfig, generate
from deep_irt.bench import metrics_bench as M
from deep_irt.bench.engines import DeepIRTEngine, MaIrtEngine

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
    pred = M.prediction_metrics(probs, targ, ds.cfg.n_cats)

    rec = engine.recover()
    seen = rec.get("seen", None)
    item = M.item_recovery(rec["a"], rec["b"], ds.gt.a, ds.gt.b, seen=seen)

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
        "decoder": getattr(engine, "decoder", "gpcm"),
        "dataset": ds.cfg.name,
        "kind": ds.cfg.kind,
        "n_cats": ds.cfg.n_cats,
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
    ap.add_argument("--quick", action="store_true",
                    help="small/fast smoke grid")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if args.quick:
        N, Q, T = 200, 30, 30
        epochs = args.epochs or 40
    else:
        N, Q, T = 800, 60, 60
        epochs = args.epochs or 150

    print(f"=== bench: device={device} epochs={epochs} N={N} Q={Q} T={T} ===")

    # ---- datasets (shared ground truth + split per dataset) ----
    data_cfgs = [
        BenchDataConfig(name="static_k4", kind="static", n_cats=4,
                        n_learners=N, n_items=Q, seq_len=T, seed=args.seed),
        BenchDataConfig(name="dynamic_k4", kind="dynamic", n_cats=4,
                        n_learners=N, n_items=Q, seq_len=T, drift_sigma=0.15,
                        seed=args.seed),
        BenchDataConfig(name="binary_2pl", kind="binary", n_cats=2,
                        n_learners=N, n_items=Q, seq_len=T, seed=args.seed),
    ]
    datasets = {c.name: generate(c) for c in data_cfgs}
    for name, ds in datasets.items():
        seen = np.unique(ds.items0).size
        print(f"  {name}: {ds.items0.shape} train={len(ds.train_idx)} "
              f"val={len(ds.val_idx)} items_seen={seen}/{ds.cfg.n_items}")

    rows = []

    # ---- ENCODER AXIS + ENGINE AXIS on the ordinal datasets ----
    # deep_irt vanilla LSTM (GPCM) vs ma-irt {dkvmn, lstm_gpcm, transformer}.
    for dsname in ("static_k4", "dynamic_k4"):
        ds = datasets[dsname]
        print(f"\n--- dataset {dsname} (encoder axis, GPCM decoder) ---")

        eng = DeepIRTEngine(ds, decoder="gpcm", device=device, seed=args.seed)
        eng.fit(n_epochs=epochs); rows.append(run_cell(eng, ds))
        print(f"   {rows[-1]['label']:28s} acc={rows[-1]['acc']:.3f} "
              f"qwk={rows[-1]['qwk']:.3f} theta={rows[-1]['theta_headline']:.3f} "
              f"a={rows[-1]['a_spearman']:.3f} t={rows[-1]['train_time']:.1f}s")

        for enc in ("dkvmn", "lstm_gpcm", "transformer_gpcm"):
            eng = MaIrtEngine(ds, encoder=enc, device=device, seed=args.seed)
            eng.fit(n_epochs=epochs); rows.append(run_cell(eng, ds))
            print(f"   {rows[-1]['label']:28s} acc={rows[-1]['acc']:.3f} "
                  f"qwk={rows[-1]['qwk']:.3f} theta={rows[-1]['theta_headline']:.3f} "
                  f"a={rows[-1]['a_spearman']:.3f} t={rows[-1]['train_time']:.1f}s")

    # ---- DECODER AXIS on deep_irt (encoder fixed = vanilla LSTM) ----
    print("\n--- decoder axis (deep_irt LSTM held fixed) ---")
    # GPCM already covered on static_k4; add binary 2PL and NRM there.
    ds = datasets["binary_2pl"]
    eng = DeepIRTEngine(ds, decoder="binary", device=device, seed=args.seed)
    eng.fit(n_epochs=epochs); rows.append(run_cell(eng, ds))
    print(f"   {rows[-1]['label']:28s} acc={rows[-1]['acc']:.3f} "
          f"auc={rows[-1]['auc']:.3f} theta={rows[-1]['theta_headline']:.3f} "
          f"a={rows[-1]['a_spearman']:.3f}")
    # ma-irt binary via its GPCM decoder at K=2 (DKVMN encoder), same dataset.
    eng = MaIrtEngine(ds, encoder="dkvmn", device=device, seed=args.seed)
    eng.fit(n_epochs=epochs); rows.append(run_cell(eng, ds))
    print(f"   {rows[-1]['label']:28s} acc={rows[-1]['acc']:.3f} "
          f"auc={rows[-1]['auc']:.3f} theta={rows[-1]['theta_headline']:.3f} "
          f"a={rows[-1]['a_spearman']:.3f}")

    # NRM (multiple-choice) on the static set: deep_irt only (ma-irt has no NRM).
    # Treat the ordinal labels as unordered options so the decoder still trains;
    # recovery for NRM uses its correct-option beta location.
    ds = datasets["static_k4"]
    eng = DeepIRTEngine(ds, decoder="nrm", device=device, seed=args.seed)
    eng.fit(n_epochs=epochs)
    probs, targ = eng.predict_heldout()
    pred = M.prediction_metrics(probs, targ, ds.cfg.n_cats)
    rec = eng.recover()                  # nrm recover() returns a (K-wide), beta via decoder
    # For NRM, recover() exposes per-option a; collapse to the location beta for
    # difficulty recovery handled inside run_cell would mismatch shapes, so we
    # score only prediction + theta here and mark item recovery N/A.
    theta = M.theta_recovery_static(rec["theta_final"], ds.gt.theta0)
    rows.append({
        "label": eng.label, "encoder": eng.encoder_name, "decoder": "nrm",
        "dataset": ds.cfg.name, "kind": ds.cfg.kind, "n_cats": ds.cfg.n_cats,
        **pred, **theta, "theta_headline": theta["theta_spearman"],
        "a_spearman": float("nan"), "a_pearson": float("nan"),
        "b_spearman": float("nan"), "b_pearson": float("nan"),
        "n_params": eng.cost()["n_params"], "train_time": eng.cost()["train_time"],
        "infer_time": time_inference(eng),
    })
    print(f"   {rows[-1]['label']:28s} acc={rows[-1]['acc']:.3f} "
          f"theta={rows[-1]['theta_headline']:.3f} (item recovery N/A for NRM)")

    # ---- persist + render ----
    with (OUT / "bench_results.json").open("w", encoding="utf-8") as fh:
        json.dump({"meta": {"device": device, "epochs": epochs,
                            "N": N, "Q": Q, "T": T, "seed": args.seed},
                   "rows": rows}, fh, indent=2)

    table = render_table(rows, device, epochs, N, Q, T)
    (OUT / "bench_table.md").write_text(table, encoding="utf-8")
    print("\n" + table)
    print(f"\nwrote {OUT/'bench_results.json'} and {OUT/'bench_table.md'}")


def _f(x, p=3):
    if x is None or (isinstance(x, float) and x != x):
        return "  -  "
    return f"{x:.{p}f}"


def render_table(rows, device, epochs, N, Q, T) -> str:
    lines = []
    lines.append(f"# Engine head-to-head benchmark\n")
    lines.append(f"device={device}  epochs={epochs}  N={N}  Q={Q}  T={T}  "
                 f"(torch {torch.__version__})\n")
    lines.append("| engine/encoder | dec | dataset | acc | QWK | AUC | "
                 "theta | a (sp) | b (sp) | params | train s | infer s |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['decoder']} | {r['dataset']} | "
            f"{_f(r.get('acc'))} | {_f(r.get('qwk'))} | {_f(r.get('auc'))} | "
            f"{_f(r.get('theta_headline'))} | {_f(r.get('a_spearman'))} | "
            f"{_f(r.get('b_spearman'))} | {r['n_params']} | "
            f"{_f(r.get('train_time'),1)} | {_f(r.get('infer_time'),3)} |"
        )
    lines.append("\nNotes: theta = Spearman of recovered vs true ability "
                 "(static: final-step level; dynamic: within-learner net drift). "
                 "a/b = Spearman of recovered discrimination / step-thresholds vs "
                 "truth. AUC for K>2 is a top-half-vs-bottom-half proxy; for "
                 "binary it is the exact 2PL AUC. NRM item recovery is N/A "
                 "(unordered options).")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
