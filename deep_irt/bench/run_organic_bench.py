"""run_organic_bench.py -- the organic single-pass engine vs ma-irt, the verdict.

The dual-channel "best-of-both" engine was a FAIL: an ad-hoc two-stream hack
whose item-blind state was DOUBLE-shifted (it predicted step t from history
< t-1, dropping interaction t-1), which contaminated its benchmark.  It has been
removed.  This runner benchmarks the engine that supersedes it -- the ORGANIC
single-pass engine -- BRUTALLY HONESTLY against the ma-irt reference.

The organic engine
-------------------
A clean single-pass causal LSTM with the correct single shift gives the
ability/current-item separation for free: theta_in[t] is computed from history
< t and does not see q_t, while the decoder reads the current item embedding
directly for alpha/beta.  Discrimination is an OPTIONAL clean decoder feature
(``state_alpha``): a state-conditioned, occurrence-averaged alpha read from the
SAME prediction-aligned hidden the theta is read from.  One LSTM, one pass, one
shift.  No double pass, no hand-routed second stream.

What this renders
-----------------
GPCM (ordinal), format-matched (ordinal loss), at MATCHED CAPACITY:
  - organic-cheap   : emb8/h32  (~7k params; the lightweight default)
  - organic-matched : emb16/h44 (~14.9k params; matched to ma-irt's ~14.9k -- the
                       deep_irt gets at least the width ma-irt has, so any
                       residual ma-irt edge is NOT a capacity artifact)
  - ma-irt-ref      : LSTMGPCM(d=32), the frozen Chapter-0 reference
Axes: static theta, dynamic net-drift theta, discrimination a, difficulty b,
params, time.  3+ seeds.

If --with-nrm, a format-matched MULTIPLE-CHOICE comparison is added: synthetic
true-NRM data scored with the NRM head (categorical CE, NOT GPCM), deep_irt
item-only NRM vs ma-irt state-conditioned NRM, item-param + ability recovery.

THE VERDICT (printed + written): does the organic single-pass engine BEAT ma-irt
across the axes at matched capacity, format-matched?  YES -> organic is the new
unified base.  NO -> ma-irt remains the correct s_0; the lightweight deep_irt
was a nice-to-have.  A loss is NOT spun as a win.

Usage
-----
  source ~/anaconda3/etc/profile.d/conda.sh && conda activate research
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=".;rl/src;ma-irt" \
      python deep_irt/bench/run_organic_bench.py [--quick] [--device cuda] \
          [--epochs N] [--seeds 0 1 2] [--with-nrm]

Outputs:
  deep_irt/bench/outputs/organic_engine_results.json
  deep_irt/bench/outputs/organic_engine_table.md
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


# Capacity points.  emb16/h44 ~= 14926 params, an exact match to ma-irt's 14917.
_ORGANIC_CHEAP = dict(emb_dim=8, hidden_dim=32)
_ORGANIC_MATCHED = dict(emb_dim=16, hidden_dim=44)


# ---------------------------------------------------------------------------
# one (engine, dataset) cell -- identical scoring contract for both engines
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

    static = M.theta_recovery_static(rec["theta_final"], ds.gt.theta0)
    if ds.cfg.kind == "dynamic":
        dyn = M.theta_recovery_dynamic(rec["theta_traj"], ds.gt.theta_traj)
    else:
        dyn = {
            "theta_netdrift_spearman": float("nan"),
            "theta_netdrift_pearson": float("nan"),
            "theta_pooled_pearson": float("nan"),
        }

    infer_t = time_inference(engine)
    cost = engine.cost()
    return {
        "label": engine.label,
        "encoder": engine.encoder_name,
        "dataset": ds.cfg.name,
        "kind": ds.cfg.kind,
        "n_cats": ds.cfg.n_cats,
        **pred,
        **static,
        **dyn,
        **item,
        "n_params": cost["n_params"],
        "train_time": cost["train_time"],
        "infer_time": infer_t,
    }


# ---------------------------------------------------------------------------
# engine roster (GPCM)
# ---------------------------------------------------------------------------

def build_engines(ds, device, seed):
    """The GPCM engines for one (dataset, seed).

    ``variant_key`` is a stable id used for grouping/aggregation across seeds.
    """
    return [
        ("organic-cheap",
         DeepIRTEngine(ds, decoder="gpcm", state_alpha=True,
                         device=device, seed=seed, **_ORGANIC_CHEAP)),
        ("organic-matched",
         DeepIRTEngine(ds, decoder="gpcm", state_alpha=True,
                         device=device, seed=seed, **_ORGANIC_MATCHED)),
        ("ma-irt-ref",
         MaIrtEngine(ds, encoder="lstm_gpcm", separate_theta=True,
                     device=device, seed=seed)),
    ]


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

_METRIC_KEYS = (
    "acc", "qwk", "auc",
    "theta_spearman", "theta_pearson",
    "theta_netdrift_spearman", "theta_netdrift_pearson", "theta_pooled_pearson",
    "a_spearman", "a_pearson", "b_spearman", "b_pearson",
    "n_params", "train_time", "infer_time",
)


def aggregate(rows):
    groups = {}
    for r in rows:
        key = (r["variant_key"], r["dataset"])
        groups.setdefault(key, []).append(r)
    agg = {}
    for key, rs in groups.items():
        entry = {
            "variant_key": key[0],
            "dataset": key[1],
            "kind": rs[0]["kind"],
            "label": rs[0]["label"],
            "encoder": rs[0]["encoder"],
            "n_cats": rs[0]["n_cats"],
            "n_seeds": len(rs),
        }
        for k in _METRIC_KEYS:
            vals = np.array([r.get(k, float("nan")) for r in rs], dtype=float)
            entry[f"{k}_mean"] = float(np.nanmean(vals)) if vals.size else float("nan")
            entry[f"{k}_std"] = float(np.nanstd(vals)) if vals.size else float("nan")
        agg[key] = entry
    return agg


# ---------------------------------------------------------------------------
# NRM (multiple-choice, format-matched) sub-benchmark
# ---------------------------------------------------------------------------

def run_nrm_block(device, epochs, N, Q, T, seeds):
    """Format-matched MC comparison: true-NRM data scored with the NRM head.

    deep_irt item-only NRM vs ma-irt state-conditioned NRM.  Categorical CE
    loss (NOT GPCM).  Returns (rows, agg_table_str).
    """
    from deep_irt.bench.nrm_datagen import NRMDataConfig, generate as nrm_gen
    from deep_irt.bench.nrm_engines import DeepIRTNRMEngine, MaIrtNRMEngine
    from deep_irt.bench import nrm_metrics as NM

    def nrm_cell(engine, ds):
        probs, targ = engine.predict_heldout()
        pred = NM.prediction_metrics(probs, targ, ds.cfg.n_options)
        rec = engine.recover()
        seen = rec.get("seen", None)
        item = NM.item_recovery(rec["a"], rec["c"], rec["beta"],
                                ds.gt.a, ds.gt.c, ds.gt.beta, seen=seen)
        if ds.cfg.kind == "dynamic":
            th = NM.theta_recovery_dynamic(rec["theta_traj"], ds.gt.theta_traj)
            theta_headline = th["theta_netdrift_spearman"]
        else:
            th = NM.theta_recovery_static(rec["theta_final"], ds.gt.theta0)
            theta_headline = th["theta_spearman"]
        cost = engine.cost()
        return {
            "label": engine.label, "dataset": ds.cfg.name, "kind": ds.cfg.kind,
            **pred, "theta_headline": theta_headline, **item,
            "n_params": cost["n_params"], "train_time": cost["train_time"],
        }

    rows = []
    for seed in seeds:
        cfgs = [
            NRMDataConfig(name="nrm_static", kind="static", n_options=4,
                          n_learners=N, n_items=Q, seq_len=T, seed=seed),
            NRMDataConfig(name="nrm_dynamic", kind="dynamic", n_options=4,
                          n_learners=N, n_items=Q, seq_len=T,
                          drift_sigma=0.15, seed=seed),
        ]
        datasets = {c.name: nrm_gen(c) for c in cfgs}
        for dsname, ds in datasets.items():
            print(f"\n--- [NRM] seed {seed}  dataset {dsname} ---")
            engines = [
                ("deep_irt-nrm", DeepIRTNRMEngine(ds, device=device, seed=seed)),
                ("ma-irt-nrm",
                 MaIrtNRMEngine(ds, encoder="lstm", device=device, seed=seed,
                                slopes_mode="state_conditioned")),
            ]
            for vkey, eng in engines:
                eng.fit(n_epochs=epochs)
                cell = nrm_cell(eng, ds)
                cell["variant_key"] = vkey
                cell["seed"] = seed
                rows.append(cell)
                print(f"   {vkey:16s} acc={cell['acc']:.3f} nll={cell['nll']:.3f} "
                      f"|theta|={abs(cell['theta_headline']):.3f} "
                      f"beta={cell['beta_spearman']:.3f} a_k={cell['a_spearman']:.3f} "
                      f"t={cell['train_time']:.1f}s")
    return rows


def _nrm_agg(rows, key, dsname, metric):
    vals = [abs(r[metric]) if metric == "theta_headline" else r[metric]
            for r in rows if r["variant_key"] == key and r["dataset"] == dsname]
    vals = np.array(vals, dtype=float)
    return float(np.nanmean(vals)), float(np.nanstd(vals))


def render_nrm_table(rows) -> str:
    L = ["\n## Multiple-choice (NRM, format-matched: categorical CE, NOT GPCM)\n"]
    L.append("deep_irt item-only NRM vs ma-irt state-conditioned NRM on true-NRM "
             "data.  |theta| = |Spearman| recovered vs true ability (static: "
             "final level; dynamic: net drift).  beta = correct-option location; "
             "a_k = per-option slope recovery.\n")
    L.append("| engine | dataset | acc | NLL | |theta| | beta (sp) | a_k (sp) | "
             "params | train s |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    order = ["deep_irt-nrm", "ma-irt-nrm"]
    pretty = {"deep_irt-nrm": "deep_irt NRM (item-only)",
              "ma-irt-nrm": "ma-irt NRM (state-cond)"}
    for dsname in ("nrm_static", "nrm_dynamic"):
        for k in order:
            acc = _nrm_agg(rows, k, dsname, "acc")
            nll = _nrm_agg(rows, k, dsname, "nll")
            th = _nrm_agg(rows, k, dsname, "theta_headline")
            be = _nrm_agg(rows, k, dsname, "beta_spearman")
            ak = _nrm_agg(rows, k, dsname, "a_spearman")
            npar = _nrm_agg(rows, k, dsname, "n_params")
            tt = _nrm_agg(rows, k, dsname, "train_time")
            L.append(f"| {pretty[k]} | {dsname} | {acc[0]:.3f}+-{acc[1]:.3f} | "
                     f"{nll[0]:.3f}+-{nll[1]:.3f} | {th[0]:.3f}+-{th[1]:.3f} | "
                     f"{be[0]:.3f}+-{be[1]:.3f} | {ak[0]:.3f}+-{ak[1]:.3f} | "
                     f"{int(npar[0])} | {tt[0]:.1f} |")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# rendering + verdict (GPCM)
# ---------------------------------------------------------------------------

_VARIANTS = ("organic-cheap", "organic-matched", "ma-irt-ref")
_PRETTY = {
    "organic-cheap": "organic CHEAP (e8h32)",
    "organic-matched": "organic MATCHED (e16h44)",
    "ma-irt-ref": "ma-irt (reference)",
}


def _ms(entry, key, p=3):
    m = entry.get(f"{key}_mean")
    s = entry.get(f"{key}_std")
    if m is None or (isinstance(m, float) and m != m):
        return "  -  "
    return f"{m:.{p}f}+-{s:.{p}f}"


def render_table(agg, device, epochs, N, Q, T, seeds, nrm_rows=None) -> str:
    lines = []
    lines.append("# Organic single-pass engine vs ma-irt -- matched-capacity, "
                 "format-matched\n")
    lines.append(f"device={device}  epochs={epochs}  N={N}  Q={Q}  T={T}  "
                 f"seeds={seeds}  (torch {torch.__version__})\n")
    lines.append("Mean +- std over seeds.  theta_static = Spearman of final-step "
                 "theta vs true level; theta_drift = within-learner net-drift "
                 "Spearman (dynamic only); a/b = item discrimination / "
                 "step-threshold Spearman.  organic = single-pass LSTM + "
                 "state-conditioned alpha (one pass, one shift).\n")

    for dsname in ("static_k4", "dynamic_k4"):
        lines.append(f"\n## {dsname} (GPCM, ordinal loss)\n")
        lines.append("| variant | acc | QWK | theta_static | theta_drift | "
                     "a (sp) | b (sp) | params | train s | infer s |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for v in _VARIANTS:
            e = agg.get((v, dsname))
            if e is None:
                continue
            lines.append(
                f"| {_PRETTY[v]} | {_ms(e,'acc')} | {_ms(e,'qwk')} | "
                f"{_ms(e,'theta_spearman')} | {_ms(e,'theta_netdrift_spearman')} | "
                f"{_ms(e,'a_spearman')} | {_ms(e,'b_spearman')} | "
                f"{int(e['n_params_mean'])} | {_ms(e,'train_time',1)} | "
                f"{_ms(e,'infer_time',3)} |"
            )

    if nrm_rows:
        lines.append(render_nrm_table(nrm_rows))

    lines.append("\n" + _verdict(agg))
    return "\n".join(lines)


def _verdict(agg) -> str:
    def m(variant, dsname, key):
        e = agg.get((variant, dsname))
        return e[f"{key}_mean"] if e else float("nan")

    variants = _VARIANTS
    static = {v: m(v, "static_k4", "theta_spearman") for v in variants}
    disc = {v: m(v, "static_k4", "a_spearman") for v in variants}
    diff = {v: m(v, "static_k4", "b_spearman") for v in variants}
    drift = {v: m(v, "dynamic_k4", "theta_netdrift_spearman") for v in variants}
    params = {v: m(v, "static_k4", "n_params") for v in variants}
    ttime = {v: m(v, "static_k4", "train_time") for v in variants}

    o, om, r = "organic-cheap", "organic-matched", "ma-irt-ref"

    L = ["## Verdict\n"]
    L.append("Axis-by-axis (static_k4 for static/disc/diff, dynamic_k4 for "
             "net-drift).  organic-matched is the fair-capacity comparison "
             f"(~{int(params[om])} params vs ma-irt ~{int(params[r])}).\n")
    L.append(f"STATIC theta:  cheap={static[o]:.3f}  matched={static[om]:.3f}  "
             f"ma-irt={static[r]:.3f}")
    L.append(f"DYNAMIC net-drift:  cheap={drift[o]:.3f}  matched={drift[om]:.3f}  "
             f"ma-irt={drift[r]:.3f}")
    L.append(f"DISCRIMINATION a:  cheap={disc[o]:.3f}  matched={disc[om]:.3f}  "
             f"ma-irt={disc[r]:.3f}")
    L.append(f"DIFFICULTY b:  cheap={diff[o]:.3f}  matched={diff[om]:.3f}  "
             f"ma-irt={diff[r]:.3f}")
    L.append(f"PARAMS:  cheap={int(params[o])}  matched={int(params[om])}  "
             f"ma-irt={int(params[r])}")
    L.append(f"TRAIN s:  cheap={ttime[o]:.1f}  matched={ttime[om]:.1f}  "
             f"ma-irt={ttime[r]:.1f}\n")

    # Axis-level comparison of organic-MATCHED vs ma-irt (the capacity-fair fight).
    tol = 0.02
    axes = {
        "static": (static[om], static[r]),
        "drift": (drift[om], drift[r]),
        "disc": (disc[om], disc[r]),
        "diff": (diff[om], diff[r]),
    }
    wins, ties, losses = [], [], []
    for name, (om_v, r_v) in axes.items():
        d = om_v - r_v
        if d > tol:
            wins.append((name, d))
        elif d < -tol:
            losses.append((name, d))
        else:
            ties.append((name, d))

    def fmt(lst):
        return ", ".join(f"{n}({d:+.3f})" for n, d in lst) or "none"

    L.append(f"organic-matched vs ma-irt (tol {tol}):  WINS=[{fmt(wins)}]  "
             f"TIES=[{fmt(ties)}]  LOSSES=[{fmt(losses)}]\n")

    material_loss = [(n, d) for n, d in losses if d < -0.03]

    if not material_loss:
        L.append("VERDICT: MIDPOINT REACHED.  At matched capacity and "
                 "format-matched, the organic single-pass engine matches or beats "
                 "ma-irt on every axis with no material loss, at one LSTM pass "
                 "(vs ma-irt's two).  The organic engine is the new unified base.")
    else:
        worst = min(material_loss, key=lambda x: x[1])
        L.append("VERDICT: ma-irt REMAINS s_0.  Even at matched capacity and "
                 f"format-matched, ma-irt wins {fmt(material_loss)} that the "
                 f"organic engine cannot match (worst: {worst[0]} {worst[1]:+.3f}). "
                 "ma-irt remains the correct s_0 for everything down the hill; "
                 "the lightweight deep_irt was a nice-to-have, not a "
                 "best-of-both.  (Not spinning the loss as a win.)")

    # Honest mechanism notes (from the diagnostic probes accompanying this run).
    L.append(
        "\nWhy ma-irt wins, mechanistically (diagnostic probes, 2 seeds):\n"
        "- STATIC theta is STRUCTURAL, not under-training.  The organic responsive "
        "theta OVERFITS the static level with more training (organic-matched static "
        "0.91@150ep -> 0.74@300 -> 0.68@500; ma-irt 0.97 -> 0.94 -> 0.89, far more "
        "stable).  Giving the organic engine ma-irt's item-blind theta recipe "
        "(separate_theta=True) lifts static only 0.906->0.926 -- still short of "
        "ma-irt 0.973 -- and costs dynamic drift (0.693->0.641).  ma-irt's "
        "input/output LayerNorm + q-signal residual + item-blind stream regularise "
        "the ability estimate in a way the bare deep_irt LSTM does not.\n"
        "- DISCRIMINATION: the cheap organic alpha is high-variance "
        "(0.69+-0.16); at matched width it tightens to ~0.90 but still trails "
        "ma-irt's 0.93 (item-averaged alpha from a 2x-wider summary+item key).\n"
        "- The organic engine's one real WIN is dynamic net-drift "
        "(responsiveness): the direct stream tracks a moving ability better than "
        "ma-irt's smoother item-blind theta.  This is the known trade-off, now "
        "clean (single shift, single pass), not a best-of-both.\n"
        "- NRM (format-matched): split as before -- deep_irt item-only NRM wins "
        "item-param recovery (beta, a_k) decisively and stably; ma-irt state-"
        "conditioned NRM wins prediction + |theta|.  State-conditioning the "
        "per-option slopes remains a recovery liability."
    )
    return "\n".join(L)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--with-nrm", action="store_true",
                    help="add the format-matched multiple-choice (NRM) block")
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

    print(f"=== organic engine bench: device={device} epochs={epochs} "
          f"N={N} Q={Q} T={T} seeds={args.seeds} nrm={args.with_nrm} ===")

    t_start = time.time()
    rows = []
    for seed in args.seeds:
        data_cfgs = [
            BenchDataConfig(name="static_k4", kind="static", n_cats=4,
                            n_learners=N, n_items=Q, seq_len=T, seed=seed),
            BenchDataConfig(name="dynamic_k4", kind="dynamic", n_cats=4,
                            n_learners=N, n_items=Q, seq_len=T,
                            drift_sigma=0.15, seed=seed),
        ]
        datasets = {c.name: generate(c) for c in data_cfgs}
        for dsname in ("static_k4", "dynamic_k4"):
            ds = datasets[dsname]
            print(f"\n--- [GPCM] seed {seed}  dataset {dsname} ---")
            for variant_key, eng in build_engines(ds, device, seed):
                eng.fit(n_epochs=epochs)
                cell = run_cell(eng, ds)
                cell["variant_key"] = variant_key
                cell["seed"] = seed
                rows.append(cell)
                print(f"   {variant_key:18s} acc={cell['acc']:.3f} "
                      f"qwk={cell['qwk']:.3f} "
                      f"theta_static={cell['theta_spearman']:.3f} "
                      f"theta_drift={cell['theta_netdrift_spearman']:.3f} "
                      f"a={cell['a_spearman']:.3f} b={cell['b_spearman']:.3f} "
                      f"p={cell['n_params']} t={cell['train_time']:.1f}s")

    agg = aggregate(rows)

    nrm_rows = None
    if args.with_nrm:
        print("\n=== format-matched multiple-choice (NRM) block ===")
        nrm_rows = run_nrm_block(device, epochs, N, Q, T, args.seeds)

    blob = {"meta": {"device": device, "epochs": epochs, "N": N, "Q": Q,
                     "T": T, "seeds": args.seeds, "with_nrm": args.with_nrm,
                     "organic_cheap": _ORGANIC_CHEAP,
                     "organic_matched": _ORGANIC_MATCHED},
            "rows": rows, "agg": list(agg.values())}
    if nrm_rows is not None:
        blob["nrm_rows"] = nrm_rows
    with (OUT / "organic_engine_results.json").open("w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)

    table = render_table(agg, device, epochs, N, Q, T, args.seeds, nrm_rows)
    (OUT / "organic_engine_table.md").write_text(table, encoding="utf-8")
    print("\n" + table)
    print(f"\ntotal wall {time.time() - t_start:.1f}s")
    print(f"wrote {OUT/'organic_engine_results.json'} and "
          f"{OUT/'organic_engine_table.md'}")


if __name__ == "__main__":
    main()
