"""run_ablation.py -- the wide-key item-parameter ablation for deep_irt.

ma-irt's LSTMGPCM recovers item parameters better than deep_irt's LSTM at the
same training budget.  Two architectural causes were identified:

  beta gap  -- deep_irt's step-threshold head ``fc_b`` read the NARROW item
               embedding (emb_dim=8); ma-irt reads beta from its WIDE item key
               (the same q_embed it feeds discrimination).  ``beta_wide=True``
               routes beta through the wide decoupled alpha key, mirroring it.
  alpha gap -- ma-irt's residual discrimination edge comes from its two-pass
               item-blind / item-conditioned split.  That does NOT port cleanly
               to deep_irt (its item-blindness is a shift-AFTER the LSTM, ma-irt's
               is a shift-BEFORE with a current-question residual); a flag cannot
               be confined to LSTMEncoder, so it is not included here.

The ablation isolates the beta fix against the frozen ma-irt reference.  Rows:

  1. deep_irt decoupled            (e8, alpha-ae64, beta_wide=False)   baseline
  2. deep_irt decoupled +wide-beta (e8, alpha-ae64, beta_wide=True)
  3. ma-irt lstm                   (lstm_gpcm, separate_theta=True)    reference

All deep_irt variants use the state-conditioned, occurrence-averaged alpha (+sa)
with the EXP discrimination transform exp(raw) -- the same positivity map ma-irt
uses (log_scale=1.0) -- so the comparison isolates the wide-key beta read, not
the softplus-vs-exp confound.

Every cell is scored by the SAME contract (``run_cell``): acc + QWK on the
held-out tail, alpha / beta / theta Pearson r against the known ground truth,
and the train wall-time + parameter count.

Usage
-----
  source ~/anaconda3/etc/profile.d/conda.sh && conda activate research
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=".;rl/src;ma-irt" \
      python deep_irt/bench/run_ablation.py [--K 4] [--seeds 0 1 2 3 4] \
          [--epochs 150] [--device cuda]

Outputs:
  deep_irt/bench/outputs/ablation_results.json
  deep_irt/bench/outputs/ablation_table.md
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
from deep_irt.bench.run_alpha_fix_bench import run_cell

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "outputs"
OUT.mkdir(exist_ok=True)

# Shared config points.
_CHEAP = dict(emb_dim=8, hidden_dim=32)   # the cheap theta-encoder
_ALPHA_EMB_DIM = 64                       # the wide decoupled alpha (and beta) key
_ALPHA_LOG_SCALE = 1.0                    # ma-irt's exp(raw) transform

# Row roster (order preserved in the table).
_VARIANTS = ("decoupled", "decoupled_wb", "ma-irt-lstm")
_PRETTY = {
    "decoupled": "deep_irt decoupled (e8, alpha-ae64)",
    "decoupled_wb": "deep_irt decoupled +wide-beta",
    "ma-irt-lstm": "ma-irt lstm (reference)",
}


def build_engines(ds, device, seed):
    """The three-way fight: decoupled, decoupled+wide-beta, ma-irt reference."""
    return [
        ("decoupled",
         DeepIRTEngine(ds, decoder="gpcm", state_alpha=True,
                       alpha_emb_dim=_ALPHA_EMB_DIM,
                       alpha_log_scale=_ALPHA_LOG_SCALE,
                       beta_wide=False, device=device, seed=seed, **_CHEAP)),
        ("decoupled_wb",
         DeepIRTEngine(ds, decoder="gpcm", state_alpha=True,
                       alpha_emb_dim=_ALPHA_EMB_DIM,
                       alpha_log_scale=_ALPHA_LOG_SCALE,
                       beta_wide=True, device=device, seed=seed, **_CHEAP)),
        ("ma-irt-lstm",
         MaIrtEngine(ds, encoder="lstm_gpcm", separate_theta=True,
                     device=device, seed=seed)),
    ]


# Metrics aggregated across seeds.  Pearson r for the recovery axes (the headline
# numbers the finding quotes); acc/QWK for prediction; cost columns last.
_METRIC_KEYS = (
    "acc", "qwk",
    "a_pearson", "b_pearson", "theta_pearson",
    "n_params", "train_time",
)


def aggregate(rows):
    groups = {}
    for r in rows:
        groups.setdefault(r["variant_key"], []).append(r)
    agg = {}
    for vkey, rs in groups.items():
        entry = {"variant_key": vkey, "label": rs[0]["label"],
                 "encoder": rs[0]["encoder"], "n_seeds": len(rs)}
        for k in _METRIC_KEYS:
            vals = np.array([r.get(k, float("nan")) for r in rs], dtype=float)
            entry[f"{k}_mean"] = float(np.nanmean(vals)) if vals.size else float("nan")
            entry[f"{k}_std"] = float(np.nanstd(vals)) if vals.size else float("nan")
        agg[vkey] = entry
    return agg


def _ms(entry, key, p=3):
    m = entry.get(f"{key}_mean")
    s = entry.get(f"{key}_std")
    if m is None or (isinstance(m, float) and m != m):
        return "  -  "
    return f"{m:.{p}f}+-{s:.{p}f}"


def render_table(agg, device, K, epochs, N, Q, T, seeds) -> str:
    L = []
    L.append("# Wide-key item-parameter ablation -- does routing beta through the "
             "wide key close the gap to ma-irt?\n")
    L.append(f"device={device}  K={K}  epochs={epochs}  N={N}  Q={Q}  T={T}  "
             f"seeds={seeds}  (torch {torch.__version__})\n")
    L.append("Static-K{0} GPCM, ordinal prediction loss.  Mean +- std over seeds. "
             "acc/QWK on the held-out tail; a/b/theta = Pearson r of recovered vs "
             "true discrimination / step-thresholds / final-step ability.  All "
             "deep_irt rows use state-conditioned occurrence-averaged alpha with "
             "the exp(raw) transform (ma-irt's positivity map) and the decoupled "
             "wide alpha key (ae=64); +wide-beta additionally reads beta from that "
             "same wide key instead of the cheap item_emb.\n".format(K))
    L.append("| variant | acc | QWK | a (r) | b (r) | theta (r) | params | "
             "train s |")
    L.append("|---|---|---|---|---|---|---|---|")
    for v in _VARIANTS:
        e = agg.get(v)
        if e is None:
            continue
        L.append(
            f"| {_PRETTY[v]} | {_ms(e,'acc')} | {_ms(e,'qwk')} | "
            f"{_ms(e,'a_pearson')} | {_ms(e,'b_pearson')} | "
            f"{_ms(e,'theta_pearson')} | {int(e['n_params_mean'])} | "
            f"{_ms(e,'train_time',1)} |"
        )
    L.append("\n" + _verdict(agg))
    return "\n".join(L)


def _verdict(agg) -> str:
    def m(vkey, key):
        e = agg.get(vkey)
        return e[f"{key}_mean"] if e else float("nan")

    base, wb, ref = _VARIANTS
    L = ["## Verdict\n"]
    for axis, key in (("alpha", "a_pearson"), ("beta", "b_pearson"),
                      ("theta", "theta_pearson")):
        L.append(f"{axis} (Pearson r):  decoupled={m(base,key):.3f}  "
                 f"+wide-beta={m(wb,key):.3f}  ma-irt={m(ref,key):.3f}")
    b_gain = m(wb, "b_pearson") - m(base, "b_pearson")
    b_to_ref = m(wb, "b_pearson") - m(ref, "b_pearson")
    a_to_ref = m(wb, "a_pearson") - m(ref, "a_pearson")
    L.append("")
    L.append(f"beta wide-key gain over narrow baseline:  {b_gain:+.3f}; "
             f"+wide-beta vs ma-irt beta:  {b_to_ref:+.3f}.")
    L.append(f"residual alpha gap to ma-irt (+wide-beta):  {a_to_ref:+.3f} "
             "(the two-pass item-blind theta split, deliberately not ported here).")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=4, help="response categories")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-learners", type=int, default=800)
    ap.add_argument("--n-items", type=int, default=60)
    ap.add_argument("--seq-len", type=int, default=60)
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    K, N, Q, T = args.K, args.n_learners, args.n_items, args.seq_len
    epochs = args.epochs

    print(f"=== ablation: device={device} K={K} epochs={epochs} "
          f"N={N} Q={Q} T={T} seeds={args.seeds} ===")

    t_start = time.time()
    rows = []
    for seed in args.seeds:
        cfg = BenchDataConfig(name=f"static_k{K}", kind="static", n_cats=K,
                              n_learners=N, n_items=Q, seq_len=T, seed=seed)
        ds = generate(cfg)
        print(f"\n--- seed {seed} ---")
        for variant_key, eng in build_engines(ds, device, seed):
            eng.fit(n_epochs=epochs)
            cell = run_cell(eng, ds)
            cell["variant_key"] = variant_key
            cell["seed"] = seed
            rows.append(cell)
            print(f"   {variant_key:14s} acc={cell['acc']:.3f} "
                  f"qwk={cell['qwk']:.3f} "
                  f"a={cell['a_pearson']:.3f} b={cell['b_pearson']:.3f} "
                  f"theta={cell['theta_pearson']:.3f} "
                  f"p={cell['n_params']} t={cell['train_time']:.1f}s")

    agg = aggregate(rows)
    table = render_table(agg, device, K, epochs, N, Q, T, args.seeds)

    blob = {"meta": {"device": device, "K": K, "epochs": epochs,
                     "N": N, "Q": Q, "T": T, "seeds": args.seeds,
                     "alpha_emb_dim": _ALPHA_EMB_DIM,
                     "alpha_log_scale": _ALPHA_LOG_SCALE, "cheap": _CHEAP},
            "rows": rows, "agg": list(agg.values())}
    with (OUT / "ablation_results.json").open("w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)
    (OUT / "ablation_table.md").write_text(table, encoding="utf-8")

    print("\n" + table)
    print(f"\ntotal wall {time.time() - t_start:.1f}s")
    print(f"wrote {OUT/'ablation_results.json'} and {OUT/'ablation_table.md'}")


if __name__ == "__main__":
    main()
