"""run_swap_bench.py -- is the decoupled deep_irt encoder-swappable?

The decoupling (a separate wide alpha item table feeding only the
state-conditioned alpha head) is a DECODER-side feature: it reads the encoder's
per-step state but is otherwise backbone-agnostic.  If that is true, the
decoupling pattern -- alpha lifts to ma-irt's level while theta is kept -- should
hold no matter which sequence backbone produces the state.

This runner tests exactly that.  For each backbone in {lstm, transformer, dkvmn}
it runs, at the SAME cheap theta-encoder (e8h32) and the SAME exp(raw) alpha
transform (log_scale=1.0, ma-irt's link):

  <backbone>-cheap       state_alpha, NO decouple (alpha rides the cheap item key)
  <backbone>-decoupled   state_alpha + a separate emb=64 alpha item key

and the frozen ma-irt LSTM reference for context.  The question per backbone:
does decoupling KEEP theta (static + drift) AND LIFT alpha, the same way it did
for the LSTM?  If yes for all three, the decoupling is genuinely a swappable
decoder feature, not an LSTM-specific trick.

Usage
-----
  source ~/anaconda3/etc/profile.d/conda.sh && conda activate research
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=".;rl/src;ma-irt" \
      python deep_irt/bench/run_swap_bench.py [--quick] [--device cuda] \
          [--epochs N] [--seeds 0 1 2]

Outputs:
  deep_irt/bench/outputs/swap_results.json
  deep_irt/bench/outputs/swap_table.md
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from deep_irt.bench.datagen import BenchDataConfig, generate
from deep_irt.bench.engines import DeepIRTEngine, MaIrtEngine
from deep_irt.bench.run_alpha_fix_bench import run_cell, aggregate, _ms

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "outputs"
OUT.mkdir(exist_ok=True)

_CHEAP = dict(emb_dim=8, hidden_dim=32)   # the theta-encoder held fixed everywhere
_ITEM_KEY_DIM = 64                        # decoupled wide alpha key
_ALPHA_LOG_SCALE = 1.0                    # exp(raw), ma-irt's link
_BACKBONES = ("lstm", "transformer", "dkvmn")


# ---------------------------------------------------------------------------
# engine roster: cheap + decoupled for each backbone, plus the ma-irt reference
# ---------------------------------------------------------------------------

def build_engines(ds, device, seed):
    engines = []
    for bk in _BACKBONES:
        engines.append((
            f"{bk}-cheap",
            DeepIRTEngine(ds, decoder="gpcm", state_alpha=True,
                            alpha_log_scale=_ALPHA_LOG_SCALE, encoder=bk,
                            device=device, seed=seed, **_CHEAP),
        ))
        engines.append((
            f"{bk}-decoupled",
            DeepIRTEngine(ds, decoder="gpcm", state_alpha=True,
                            item_key_dim=_ITEM_KEY_DIM,
                            alpha_log_scale=_ALPHA_LOG_SCALE, encoder=bk,
                            device=device, seed=seed, **_CHEAP),
        ))
    engines.append((
        "ma-irt-ref",
        MaIrtEngine(ds, encoder="lstm_gpcm", separate_theta=True,
                    device=device, seed=seed),
    ))
    return engines


_VARIANTS = tuple(f"{bk}-{v}" for bk in _BACKBONES for v in ("cheap", "decoupled")) \
            + ("ma-irt-ref",)
_PRETTY = {f"{bk}-cheap": f"{bk} cheap (e8h32 +sa)" for bk in _BACKBONES}
_PRETTY.update({f"{bk}-decoupled": f"{bk} DECOUPLED (alpha-ae64)" for bk in _BACKBONES})
_PRETTY["ma-irt-ref"] = "ma-irt (reference)"


# ---------------------------------------------------------------------------
# rendering + verdict
# ---------------------------------------------------------------------------

def render_table(agg, device, epochs, N, Q, T, seeds) -> str:
    L = []
    L.append("# Swap bench -- does the decoupling ride every encoder backbone?\n")
    L.append(f"device={device}  epochs={epochs}  N={N}  Q={Q}  T={T}  "
             f"seeds={seeds}  (torch {torch.__version__})\n")
    L.append("Mean +- std over seeds.  Cheap theta-encoder (e8h32) and exp(raw) "
             "alpha transform held fixed for all backbones; DECOUPLED adds a "
             "separate emb=64 alpha item key feeding only the alpha head.  The "
             "question: per backbone, does decoupling KEEP theta and LIFT alpha "
             "to ma-irt's level, as it did for the LSTM?\n")

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

    L.append("\n" + _verdict(agg))
    return "\n".join(L)


def _verdict(agg) -> str:
    def m(variant, dsname, key):
        e = agg.get((variant, dsname))
        return e[f"{key}_mean"] if e else float("nan")

    ref_static = m("ma-irt-ref", "static_k4", "theta_spearman")
    ref_drift = m("ma-irt-ref", "dynamic_k4", "theta_netdrift_spearman")
    ref_disc = m("ma-irt-ref", "static_k4", "a_spearman")

    L = ["## Verdict\n"]
    L.append("Per backbone: cheap -> decoupled on static theta / drift / "
             "discrimination, vs the ma-irt reference "
             f"(static {ref_static:.3f}, drift {ref_drift:.3f}, "
             f"disc {ref_disc:.3f}).\n")

    holds = []
    for bk in _BACKBONES:
        c, d = f"{bk}-cheap", f"{bk}-decoupled"
        st_c = m(c, "static_k4", "theta_spearman")
        st_d = m(d, "static_k4", "theta_spearman")
        dr_c = m(c, "dynamic_k4", "theta_netdrift_spearman")
        dr_d = m(d, "dynamic_k4", "theta_netdrift_spearman")
        a_c = m(c, "static_k4", "a_spearman")
        a_d = m(d, "static_k4", "a_spearman")
        keeps_static = abs(st_d - st_c) <= 0.03
        keeps_drift = abs(dr_d - dr_c) <= 0.03
        lifts_alpha = (a_d - a_c) > 0.05
        near_ref = a_d >= ref_disc - 0.03
        ok = keeps_static and keeps_drift and lifts_alpha
        holds.append(ok)
        L.append(
            f"{bk:11s}: static {st_c:.3f}->{st_d:.3f} "
            f"({'KEPT' if keeps_static else 'MOVED'}), "
            f"drift {dr_c:.3f}->{dr_d:.3f} "
            f"({'KEPT' if keeps_drift else 'MOVED'}), "
            f"alpha {a_c:.3f}->{a_d:.3f} "
            f"(Delta {a_d - a_c:+.3f}, {'LIFTED' if lifts_alpha else 'flat'}"
            f"{', ties ma-irt' if near_ref else ''})"
        )

    L.append("")
    n_hold = sum(holds)
    held = [b for b, h in zip(_BACKBONES, holds) if h]
    bad = [b for b, h in zip(_BACKBONES, holds) if not h]
    if n_hold == len(_BACKBONES):
        L.append("VERDICT: SWAPPABLE.  Decoupling keeps theta and lifts "
                 "discrimination on EVERY backbone -- it is a backbone-agnostic "
                 "decoder feature, not an LSTM-specific trick.  The candidate "
                 "s_0 architecture transfers across encoders.")
    elif n_hold >= 1:
        L.append(f"VERDICT: PARTIAL.  Decoupling holds for {held} but NOT for "
                 f"{bad} (theta moved or alpha did not lift past the tolerance). "
                 "Inspect the failing backbone, or check whether it is seed/"
                 "budget noise (the per-backbone alpha lift is the load-bearing "
                 "signal; small drift wobble at tolerance is often noise).")
    else:
        L.append("VERDICT: DOES NOT TRANSFER.  Decoupling fails on every "
                 "backbone; check the swap wiring before trusting any "
                 "cross-backbone claim.")
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

    print(f"=== swap bench: device={device} epochs={epochs} "
          f"N={N} Q={Q} T={T} seeds={args.seeds} ===")

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
                print(f"   {variant_key:22s} "
                      f"theta_static={cell['theta_spearman']:.3f} "
                      f"theta_drift={cell['theta_netdrift_spearman']:.3f} "
                      f"a={cell['a_spearman']:.3f} b={cell['b_spearman']:.3f} "
                      f"p={cell['n_params']} t={cell['train_time']:.1f}s")

    agg = aggregate(rows)
    blob = {"meta": {"device": device, "epochs": epochs, "N": N, "Q": Q, "T": T,
                     "seeds": args.seeds, "cheap": _CHEAP,
                     "item_key_dim": _ITEM_KEY_DIM,
                     "alpha_log_scale": _ALPHA_LOG_SCALE,
                     "backbones": _BACKBONES},
            "rows": rows, "agg": list(agg.values())}
    with (OUT / "swap_results.json").open("w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)

    table = render_table(agg, device, epochs, N, Q, T, args.seeds)
    (OUT / "swap_table.md").write_text(table, encoding="utf-8")
    print("\n" + table)
    print(f"\ntotal wall {time.time() - t_start:.1f}s")
    print(f"wrote {OUT/'swap_results.json'} and {OUT/'swap_table.md'}")


if __name__ == "__main__":
    main()
