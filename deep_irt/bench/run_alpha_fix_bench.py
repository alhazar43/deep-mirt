"""run_alpha_fix_bench.py -- can we fix the deep_irt's alpha bottleneck?

The organic bench left ONE axis to ma-irt: discrimination (alpha).  At the cheap
config (e8h32) the deep_irt ties ma-irt on static theta + difficulty and WINS
dynamic drift, but its alpha is high-variance (0.687) against ma-irt's 0.935.
Widening the whole encoder to fix alpha OVERFITS the static theta (the bare
deep_irt LSTM lacks ma-irt's LayerNorm / item-blind / q-residual
regularisation): organic-matched (e16h44) gets alpha ~0.91 but static theta falls
0.968 -> 0.909 and degrades further with training.

This runner tests whether alpha can be lifted WITHOUT paying that theta tax, two
ways, brutally honestly against the frozen ma-irt reference:

  VARIANT 2  deep_irt 64+64 + state_alpha  (matched capacity + ma-irt's recipe).
      Hypothesis: alpha reaches ~0.93 but static theta overfits (no
      regularisation at 64 dims).  We report static theta AT CONVERGENCE and its
      trend across epochs to confirm / refute the overfit.

  VARIANT 3  deep_irt DECOUPLED.  The theta-encoder stays CHEAP (e8h32 -> static
      theta ~0.968, drift ~0.727 unchanged) but alpha gets its OWN wide item
      key: a separate emb_dim=64 item table that feeds ONLY the
      state-conditioned alpha head, NOT the LSTM input.  theta_proj reads the
      small h32 state (unchanged); beta reads the standard small item_emb
      (unchanged).  This decouples alpha's item capacity from theta's.

THE VERDICT (printed + written), no spin:
  - DECOUPLED keeps theta ~0.968/0.727 AND lifts alpha to ~0.93 -> the deep_irt
    (decoupled) BEATS ma-irt on every axis at a small encoder; the gate FLIPS,
    deep_irt becomes the candidate s_0.
  - 64+64 gets alpha but overfits theta (as predicted) -> theta and alpha want
    the same capacity in the bare deep_irt; the decoupling result decides it.
  - NEITHER lifts alpha without losing theta (even decoupled stalls because the
    small h32 state cannot condition alpha) -> alpha and theta are irreducibly
    coupled here; ma-irt's regularised coupling stays s_0, honestly.

Usage
-----
  source ~/anaconda3/etc/profile.d/conda.sh && conda activate research
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=".;rl/src;ma-irt" \
      python deep_irt/bench/run_alpha_fix_bench.py [--quick] [--device cuda] \
          [--epochs N] [--seeds 0 1 2] [--overfit-epochs 150 300 500]

Outputs:
  deep_irt/bench/outputs/alpha_fix_results.json
  deep_irt/bench/outputs/alpha_fix_table.md
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


# Capacity points.
_CHEAP = dict(emb_dim=8, hidden_dim=32)       # the theta-encoder we want to keep
_WIDE = dict(emb_dim=64, hidden_dim=64)       # variant 2: widen everything
_ALPHA_EMB_DIM = 64                           # variant 3: decoupled wide alpha key
# Exp discrimination transform = ma-irt's exp(log_scale * raw) with log_scale=1.0
# (plain exp(raw)).  ma-irt uses 1.0 and the MLP-driven alpha head absorbs any
# scale constant, so this is the apples-to-apples choice.  All three deep_irt
# variants use it so the alpha comparison isolates architecture (1-vs-2 pass,
# shared-vs-separate key), not the positivity transform (softplus was the confound).
_ALPHA_LOG_SCALE = 1.0


# ---------------------------------------------------------------------------
# one (engine, dataset) cell -- identical scoring contract for both engines
# ---------------------------------------------------------------------------

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
    }


# ---------------------------------------------------------------------------
# engine roster (GPCM): the 4-way fight
# ---------------------------------------------------------------------------

def build_engines(ds, device, seed):
    """cheap baseline + ma-irt reference + the two alpha-fix candidates."""
    return [
        ("cheap-baseline",
         DeepIRTEngine(ds, decoder="gpcm", state_alpha=True,
                         alpha_log_scale=_ALPHA_LOG_SCALE,
                         device=device, seed=seed, **_CHEAP)),
        ("ma-irt-ref",
         MaIrtEngine(ds, encoder="lstm_gpcm", separate_theta=True,
                     device=device, seed=seed)),
        ("deep_irt-64x64",
         DeepIRTEngine(ds, decoder="gpcm", state_alpha=True,
                         alpha_log_scale=_ALPHA_LOG_SCALE,
                         device=device, seed=seed, **_WIDE)),
        ("deep_irt-decoupled",
         DeepIRTEngine(ds, decoder="gpcm", state_alpha=True,
                         alpha_emb_dim=_ALPHA_EMB_DIM,
                         alpha_log_scale=_ALPHA_LOG_SCALE,
                         device=device, seed=seed, **_CHEAP)),
    ]


_VARIANTS = ("cheap-baseline", "ma-irt-ref", "deep_irt-64x64",
             "deep_irt-decoupled")
_PRETTY = {
    "cheap-baseline": "cheap baseline (e8h32 +sa)",
    "ma-irt-ref": "ma-irt (reference)",
    "deep_irt-64x64": "deep_irt 64x64 +sa",
    "deep_irt-decoupled": "deep_irt DECOUPLED (e8h32, alpha-ae64)",
}


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

_METRIC_KEYS = (
    "acc", "qwk",
    "theta_spearman", "theta_pearson",
    "theta_netdrift_spearman", "theta_netdrift_pearson",
    "a_spearman", "a_pearson", "b_spearman", "b_pearson",
    "n_params", "train_time",
)


def aggregate(rows):
    groups = {}
    for r in rows:
        key = (r["variant_key"], r["dataset"])
        groups.setdefault(key, []).append(r)
    agg = {}
    for key, rs in groups.items():
        entry = {
            "variant_key": key[0], "dataset": key[1], "kind": rs[0]["kind"],
            "label": rs[0]["label"], "encoder": rs[0]["encoder"],
            "n_cats": rs[0]["n_cats"], "n_seeds": len(rs),
        }
        for k in _METRIC_KEYS:
            vals = np.array([r.get(k, float("nan")) for r in rs], dtype=float)
            entry[f"{k}_mean"] = float(np.nanmean(vals)) if vals.size else float("nan")
            entry[f"{k}_std"] = float(np.nanstd(vals)) if vals.size else float("nan")
        agg[key] = entry
    return agg


# ---------------------------------------------------------------------------
# theta-vs-epochs overfit probe (variant 2 and the decoupled control)
# ---------------------------------------------------------------------------

def overfit_probe(ds_static, device, seed, epoch_grid):
    """Static-theta recovery as a function of training length, for the two
    deep_irt candidates that share the cheap-vs-wide question.

    Returns {variant_key: {epoch: static_theta_spearman}} for ONE seed.  Each
    point trains a FRESH model to that epoch budget (no warm restart), so the
    curve is an honest 'where does it land if you stop here' picture.
    """
    probe_specs = {
        "deep_irt-64x64": dict(state_alpha=True,
                                alpha_log_scale=_ALPHA_LOG_SCALE, **_WIDE),
        "deep_irt-decoupled": dict(state_alpha=True, alpha_emb_dim=_ALPHA_EMB_DIM,
                                    alpha_log_scale=_ALPHA_LOG_SCALE, **_CHEAP),
        "ma-irt-ref": None,   # reference curve via MaIrtEngine
    }
    out = {k: {} for k in probe_specs}
    for ep in epoch_grid:
        for vkey, spec in probe_specs.items():
            if vkey == "ma-irt-ref":
                eng = MaIrtEngine(ds_static, encoder="lstm_gpcm",
                                  separate_theta=True, device=device, seed=seed)
            else:
                eng = DeepIRTEngine(ds_static, decoder="gpcm", device=device,
                                      seed=seed, **spec)
            eng.fit(n_epochs=ep)
            rec = eng.recover()
            st = M.theta_recovery_static(rec["theta_final"], ds_static.gt.theta0)
            out[vkey][ep] = st["theta_spearman"]
            print(f"   [overfit] {vkey:20s} ep={ep:4d} "
                  f"static_theta={st['theta_spearman']:.3f}")
    return out


# ---------------------------------------------------------------------------
# rendering + verdict
# ---------------------------------------------------------------------------

def _ms(entry, key, p=3):
    m = entry.get(f"{key}_mean")
    s = entry.get(f"{key}_std")
    if m is None or (isinstance(m, float) and m != m):
        return "  -  "
    return f"{m:.{p}f}+-{s:.{p}f}"


def render_table(agg, device, epochs, N, Q, T, seeds, overfit) -> str:
    L = []
    L.append("# Alpha-fix bench -- can the deep_irt close the discrimination "
             "gap without losing theta?\n")
    L.append(f"device={device}  epochs={epochs}  N={N}  Q={Q}  T={T}  "
             f"seeds={seeds}  (torch {torch.__version__})\n")
    L.append("Mean +- std over seeds.  theta_static = Spearman of final-step "
             "theta vs true level; theta_drift = within-learner net-drift "
             "Spearman (dynamic only); a/b = discrimination / step-threshold "
             "Spearman.  All deep_irt variants use state-conditioned, "
             "occurrence-averaged alpha (+sa) with the EXP discrimination "
             "transform exp(raw), the same one ma-irt uses (log_scale=1.0; the "
             "MLP head absorbs the scale).  Unifying it with ma-irt removes the "
             "softplus-vs-exp confound so the alpha comparison isolates "
             "architecture, not the positivity transform.  DECOUPLED = cheap "
             "e8h32 theta-encoder + a separate emb=64 item key feeding ONLY the "
             "alpha head.\n")

    for dsname in ("static_k4", "dynamic_k4"):
        L.append(f"\n## {dsname} (GPCM, ordinal loss)\n")
        L.append("| variant | acc | QWK | theta_static | theta_drift | "
                 "a (sp) | b (sp) | params | train s |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for v in _VARIANTS:
            e = agg.get((v, dsname))
            if e is None:
                continue
            L.append(
                f"| {_PRETTY[v]} | {_ms(e,'acc')} | {_ms(e,'qwk')} | "
                f"{_ms(e,'theta_spearman')} | {_ms(e,'theta_netdrift_spearman')} | "
                f"{_ms(e,'a_spearman')} | {_ms(e,'b_spearman')} | "
                f"{int(e['n_params_mean'])} | {_ms(e,'train_time',1)} |"
            )

    if overfit is not None:
        L.append(render_overfit(overfit))

    L.append("\n" + _verdict(agg, overfit))
    return "\n".join(L)


def render_overfit(overfit) -> str:
    eps = sorted({ep for d in overfit.values() for ep in d})
    L = ["\n## Static-theta vs training length (the overfit check)\n"]
    L.append("Fresh model trained to each epoch budget, single seed.  The bare "
             "deep_irt LSTM has no LayerNorm / item-blind / q-residual "
             "regulariser, so a WIDE responsive theta tends to overfit the static "
             "level with more training; ma-irt is the stable reference.\n")
    header = "| variant | " + " | ".join(f"{ep} ep" for ep in eps) + " |"
    L.append(header)
    L.append("|---|" + "---|" * len(eps))
    pretty = {"deep_irt-64x64": "deep_irt 64x64 +sa",
              "deep_irt-decoupled": "deep_irt DECOUPLED (e8h32)",
              "ma-irt-ref": "ma-irt (reference)"}
    for vkey in ("deep_irt-64x64", "deep_irt-decoupled", "ma-irt-ref"):
        if vkey not in overfit:
            continue
        cells = " | ".join(f"{overfit[vkey].get(ep, float('nan')):.3f}"
                           for ep in eps)
        L.append(f"| {pretty[vkey]} | {cells} |")
    return "\n".join(L)


def _verdict(agg, overfit) -> str:
    def m(variant, dsname, key):
        e = agg.get((variant, dsname))
        return e[f"{key}_mean"] if e else float("nan")

    cheap, ref, wide, dec = (_VARIANTS[0], _VARIANTS[1], _VARIANTS[2],
                             _VARIANTS[3])
    static = {v: m(v, "static_k4", "theta_spearman") for v in _VARIANTS}
    disc = {v: m(v, "static_k4", "a_spearman") for v in _VARIANTS}
    diff = {v: m(v, "static_k4", "b_spearman") for v in _VARIANTS}
    drift = {v: m(v, "dynamic_k4", "theta_netdrift_spearman") for v in _VARIANTS}
    params = {v: m(v, "static_k4", "n_params") for v in _VARIANTS}

    L = ["## Verdict\n"]
    L.append("Axis-by-axis (static_k4 for static/disc/diff, dynamic_k4 for "
             "net-drift).  The question: can either candidate lift alpha to "
             "ma-irt's level WITHOUT losing the cheap baseline's theta?\n")
    for axis, d in (("STATIC theta", static), ("DYNAMIC net-drift", drift),
                    ("DISCRIMINATION a", disc), ("DIFFICULTY b", diff)):
        L.append(f"{axis}:  cheap={d[cheap]:.3f}  ma-irt={d[ref]:.3f}  "
                 f"64x64={d[wide]:.3f}  decoupled={d[dec]:.3f}")
    L.append(f"PARAMS:  cheap={int(params[cheap])}  ma-irt={int(params[ref])}  "
             f"64x64={int(params[wide])}  decoupled={int(params[dec])}\n")

    tol = 0.02
    # The decoupled fight: does it BEAT ma-irt on every axis?
    dec_axes = {"static": (static[dec], static[ref]),
                "drift": (drift[dec], drift[ref]),
                "disc": (disc[dec], disc[ref]),
                "diff": (diff[dec], diff[ref])}
    wins, ties, losses = [], [], []
    for name, (x, y) in dec_axes.items():
        d = x - y
        (wins if d > tol else losses if d < -tol else ties).append((name, d))

    def fmt(lst):
        return ", ".join(f"{n}({d:+.3f})" for n, d in lst) or "none"

    L.append(f"DECOUPLED vs ma-irt (tol {tol}):  WINS=[{fmt(wins)}]  "
             f"TIES=[{fmt(ties)}]  LOSSES=[{fmt(losses)}]")

    # Does decoupled keep theta vs the cheap baseline (no theta tax)?
    dec_keeps_static = abs(static[dec] - static[cheap]) <= 0.03
    dec_keeps_drift = abs(drift[dec] - drift[cheap]) <= 0.03
    dec_lifts_alpha = disc[dec] >= disc[ref] - 0.03
    dec_alpha_over_cheap = disc[dec] - disc[cheap]

    L.append(f"\nDECOUPLED keeps theta vs cheap baseline:  "
             f"static {static[cheap]:.3f}->{static[dec]:.3f} "
             f"({'KEPT' if dec_keeps_static else 'MOVED'}), "
             f"drift {drift[cheap]:.3f}->{drift[dec]:.3f} "
             f"({'KEPT' if dec_keeps_drift else 'MOVED'}).")
    L.append(f"DECOUPLED alpha:  {disc[cheap]:.3f}->{disc[dec]:.3f} over the "
             f"cheap baseline (Delta {dec_alpha_over_cheap:+.3f}); ma-irt "
             f"{disc[ref]:.3f}.")

    material_losses = [(n, d) for n, d in losses if d < -0.03]
    L.append("")
    if dec_keeps_static and dec_keeps_drift and dec_lifts_alpha and \
            not material_losses:
        L.append("VERDICT: THE GATE FLIPS.  The DECOUPLED deep_irt keeps the "
                 "cheap baseline's theta (static + drift) AND lifts "
                 "discrimination to ma-irt's level, at a small theta-encoder. "
                 "It matches or beats ma-irt on every axis -> the deep_irt "
                 "(decoupled) is the candidate s_0.")
    elif dec_keeps_static and dec_keeps_drift and dec_alpha_over_cheap > 0.05:
        L.append("VERDICT: PARTIAL.  Decoupling lifts alpha above the cheap "
                 f"baseline (Delta {dec_alpha_over_cheap:+.3f}) WITHOUT a theta "
                 "tax, but does not fully reach ma-irt's discrimination. The "
                 "decoupling mechanism is sound (the small h32 state partially "
                 "bottlenecks the alpha key); ma-irt keeps a residual disc edge.")
    elif not dec_keeps_static or not dec_keeps_drift:
        L.append("VERDICT: DECOUPLING PAYS A THETA TAX.  Even with a separate "
                 "alpha key the theta tracks moved off the cheap baseline, so "
                 "the wide alpha pathway leaks back into the ability estimate. "
                 "ma-irt remains s_0; alpha and theta are not cleanly separable "
                 "in this architecture.")
    else:
        L.append("VERDICT: DECOUPLING STALLS.  The separate wide alpha key does "
                 "not lift discrimination materially above the cheap baseline -- "
                 "the small h32 state cannot condition alpha well enough. ma-irt "
                 "remains the correct s_0 (its alpha rides a 2x-wider summary). "
                 "(Not spinning a null as a win.)")

    # 64x64 overfit confirmation note.
    if overfit is not None and "deep_irt-64x64" in overfit:
        eps = sorted(overfit["deep_irt-64x64"])
        curve = " -> ".join(f"{overfit['deep_irt-64x64'][e]:.3f}@{e}" for e in eps)
        ma_curve = " -> ".join(
            f"{overfit['ma-irt-ref'][e]:.3f}@{e}" for e in eps) \
            if "ma-irt-ref" in overfit else "n/a"
        dec_curve = " -> ".join(
            f"{overfit['deep_irt-decoupled'][e]:.3f}@{e}" for e in eps) \
            if "deep_irt-decoupled" in overfit else "n/a"
        L.append(
            f"\n64x64 overfit check (static theta vs epochs, 1 seed):\n"
            f"  64x64:     {curve}\n"
            f"  decoupled: {dec_curve}\n"
            f"  ma-irt:    {ma_curve}\n"
            "If 64x64 falls with more training while decoupled stays flat, the "
            "wide responsive theta overfits the static level -- the prior "
            "finding -- and decoupling is the way to buy alpha capacity without "
            "paying it.")
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
    ap.add_argument("--overfit-epochs", type=int, nargs="+",
                    default=[150, 300, 500],
                    help="epoch grid for the static-theta overfit probe "
                         "(empty list disables it)")
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if args.quick:
        N, Q, T = 200, 30, 30
        epochs = args.epochs or 40
        overfit_grid = args.overfit_epochs or [40, 80]
    else:
        N, Q, T = 800, 60, 60
        epochs = args.epochs or 150
        overfit_grid = args.overfit_epochs

    print(f"=== alpha-fix bench: device={device} epochs={epochs} "
          f"N={N} Q={Q} T={T} seeds={args.seeds} ===")

    t_start = time.time()
    rows = []
    static_ds_seed0 = None
    for seed in args.seeds:
        data_cfgs = [
            BenchDataConfig(name="static_k4", kind="static", n_cats=4,
                            n_learners=N, n_items=Q, seq_len=T, seed=seed),
            BenchDataConfig(name="dynamic_k4", kind="dynamic", n_cats=4,
                            n_learners=N, n_items=Q, seq_len=T,
                            drift_sigma=0.15, seed=seed),
        ]
        datasets = {c.name: generate(c) for c in data_cfgs}
        if static_ds_seed0 is None:
            static_ds_seed0 = datasets["static_k4"]
        for dsname in ("static_k4", "dynamic_k4"):
            ds = datasets[dsname]
            print(f"\n--- [GPCM] seed {seed}  dataset {dsname} ---")
            for variant_key, eng in build_engines(ds, device, seed):
                eng.fit(n_epochs=epochs)
                cell = run_cell(eng, ds)
                cell["variant_key"] = variant_key
                cell["seed"] = seed
                rows.append(cell)
                print(f"   {variant_key:20s} acc={cell['acc']:.3f} "
                      f"qwk={cell['qwk']:.3f} "
                      f"theta_static={cell['theta_spearman']:.3f} "
                      f"theta_drift={cell['theta_netdrift_spearman']:.3f} "
                      f"a={cell['a_spearman']:.3f} b={cell['b_spearman']:.3f} "
                      f"p={cell['n_params']} t={cell['train_time']:.1f}s")

    agg = aggregate(rows)

    overfit = None
    if overfit_grid:
        print("\n=== static-theta overfit probe (seed "
              f"{args.seeds[0]}, epochs {overfit_grid}) ===")
        overfit = overfit_probe(static_ds_seed0, device, args.seeds[0],
                                overfit_grid)

    blob = {"meta": {"device": device, "epochs": epochs, "N": N, "Q": Q, "T": T,
                     "seeds": args.seeds, "cheap": _CHEAP, "wide": _WIDE,
                     "alpha_emb_dim": _ALPHA_EMB_DIM,
                     "overfit_epochs": overfit_grid},
            "rows": rows, "agg": list(agg.values()), "overfit": overfit}
    with (OUT / "alpha_fix_results.json").open("w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)

    table = render_table(agg, device, epochs, N, Q, T, args.seeds, overfit)
    (OUT / "alpha_fix_table.md").write_text(table, encoding="utf-8")
    print("\n" + table)
    print(f"\ntotal wall {time.time() - t_start:.1f}s")
    print(f"wrote {OUT/'alpha_fix_results.json'} and {OUT/'alpha_fix_table.md'}")


if __name__ == "__main__":
    main()
