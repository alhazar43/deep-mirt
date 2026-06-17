"""run_alpha_beta_asymmetry.py -- the alpha/beta Fisher-asymmetry experiment (RQ1).

The learning-dynamics study (docs/LEARNING_DYNAMICS_STUDY.md) found that
DECOUPLING the discrimination representation buys a Fisher-CONDITIONING advantage
on alpha RANK recovery: alpha is the LOW-Fisher (capacity-hungry, hard-to-
estimate) parameter, so giving it its own state-conditioned, occurrence-averaged
readout helps.  Beta (the step thresholds / difficulty) is HIGH-Fisher: it is
well-determined from the response histogram, so making it dynamic should NOT help.

This runner makes that prediction falsifiable.  For each K it trains THREE configs
against the SAME all-static GPCM data, all decoupled (item_key_dim=64) so capacity
is comparable across configs:

  baseline       state_alpha=False, state_beta=False
  alpha-dynamic  state_alpha=True,  state_beta=False
  beta-dynamic   state_alpha=False, state_beta=True

and reports, per K, the alpha and beta recovery (Pearson r, via ``run_cell`` from
run_alpha_fix_bench) for each config, plus the two deltas:

  delta_alpha = alpha_recovery(alpha-dynamic) - alpha_recovery(baseline)
  delta_beta  = beta_recovery(beta-dynamic)  - beta_recovery(baseline)

PREDICTION (the hypothesis under test)
--------------------------------------
Making the LOW-Fisher alpha dynamic HELPS alpha recovery (delta_alpha > 0,
growing with K as the stiffness I(theta)/I(alpha) grows).  Making the HIGH-Fisher
beta dynamic does NOT help (delta_beta is neutral or NEGATIVE): beta is already
well-determined as a static item read, and a state-conditioned, occurrence-
averaged head only adds estimation noise.  An asymmetry delta_alpha > delta_beta
across K is the signature of the Fisher-conditioning mechanism.

NOTE on the baseline.  ``state_beta`` requires ``state_alpha`` (the head reads
[state, item_key]), so a pure "neither dynamic" decoupled config still uses the
static item-key alpha/beta heads.  The baseline here is exactly that: decoupled
wide key (item_key_dim=64) feeding the STATIC fc_a / fc_b readouts, no state
conditioning on either parameter.  The alpha-dynamic and beta-dynamic arms switch
ON state conditioning for exactly one parameter, so each delta isolates the
effect of making THAT parameter dynamic, capacity held fixed.

Usage
-----
  source ~/anaconda3/etc/profile.d/conda.sh && conda activate research
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=".;rl/src;ma-irt" \
      python deep_irt/bench/run_alpha_beta_asymmetry.py \
          [--K 2 4 6 8 11] [--seeds 0 1 2] [--epochs 150] [--device cuda]

Outputs:
  deep_irt/bench/outputs/alpha_beta_asymmetry.json
  deep_irt/bench/outputs/alpha_beta_asymmetry_table.md
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from deep_irt.bench.datagen import BenchDataConfig, generate
from deep_irt.bench.engines import DeepIRTEngine
from deep_irt.bench.run_alpha_fix_bench import run_cell

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "outputs"
OUT.mkdir(exist_ok=True)

# Decoupled cheap theta-encoder + a wide item key (the validated s_0 geometry).
_CHEAP = dict(emb_dim=8, hidden_dim=32)
_ITEM_KEY_DIM = 64
_ALPHA_LOG_SCALE = 1.0   # ma-irt's exp(raw) discrimination transform


def _build(ds, arm, device, seed):
    """Construct the engine for one experiment arm.

    Arms (all decoupled, item_key_dim=64; the heads are gated independently so
    each arm makes EXACTLY ONE parameter dynamic):
      baseline       static alpha, static beta   (state_alpha=False, state_beta=False)
      alpha-dynamic  dynamic alpha, static beta  (state_alpha=True,  state_beta=False)
      beta-dynamic   static alpha, dynamic beta  (state_alpha=False, state_beta=True)

    The state-conditioned beta head reads the SAME prediction-aligned state the
    alpha head would, but in the beta-dynamic arm alpha stays on its static
    item-key map (the decoder gates alpha and beta separately), so delta_alpha and
    delta_beta each isolate the effect of making THAT one parameter dynamic.
    """
    if arm == "baseline":
        kw = dict(state_alpha=False, state_beta=False)
    elif arm == "alpha-dynamic":
        kw = dict(state_alpha=True, state_beta=False)
    elif arm == "beta-dynamic":
        kw = dict(state_alpha=False, state_beta=True)
    else:
        raise ValueError(arm)
    return DeepIRTEngine(
        ds, decoder="gpcm", item_key_dim=_ITEM_KEY_DIM,
        alpha_log_scale=_ALPHA_LOG_SCALE, device=device, seed=seed,
        **kw, **_CHEAP,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, nargs="+", default=[2, 4, 6, 8, 11])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
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

    print(f"=== alpha/beta asymmetry: device={device} epochs={args.epochs} "
          f"K={args.K} seeds={args.seeds} "
          f"N={args.n_learners} Q={args.n_items} T={args.seq_len} ===")

    t_start = time.time()
    rows = []   # one row per (K, seed, arm)
    for K in args.K:
        for seed in args.seeds:
            cfg = BenchDataConfig(
                name=f"static_k{K}", kind="static", n_cats=K,
                n_learners=args.n_learners, n_items=args.n_items,
                seq_len=args.seq_len, seed=seed,
            )
            ds = generate(cfg)
            for arm in ("baseline", "alpha-dynamic", "beta-dynamic"):
                eng = _build(ds, arm, device, seed)
                eng.fit(n_epochs=args.epochs)
                cell = run_cell(eng, ds)
                rec = {
                    "K": K, "seed": seed, "arm": arm,
                    "a_pearson": cell["a_pearson"],
                    "a_spearman": cell["a_spearman"],
                    "b_pearson": cell["b_pearson"],
                    "b_spearman": cell["b_spearman"],
                    "theta_pearson": cell["theta_pearson"],
                    "n_params": cell["n_params"],
                }
                rows.append(rec)
                print(f"  K={K:2d} seed={seed} {arm:14s} "
                      f"a_r={rec['a_pearson']:.3f} b_r={rec['b_pearson']:.3f} "
                      f"theta_r={rec['theta_pearson']:.3f} p={rec['n_params']}")

    # --- aggregate over seeds, per K and arm ---
    def mean_over_seeds(K, arm, key):
        vals = [r[key] for r in rows if r["K"] == K and r["arm"] == arm]
        vals = np.array(vals, dtype=float)
        return float(np.nanmean(vals)) if vals.size else float("nan")

    table = []
    table.append("# Alpha/beta Fisher-asymmetry (RQ1)\n")
    table.append(f"device={device}  epochs={args.epochs}  seeds={args.seeds}  "
                 f"N={args.n_learners}  Q={args.n_items}  T={args.seq_len}  "
                 f"(torch {torch.__version__})\n")
    table.append(
        "All arms DECOUPLED (item_key_dim=64), exp discrimination transform, "
        "static GPCM data.  alpha-dynamic switches ON ONLY the state-conditioned, "
        "occurrence-averaged alpha head (beta static); beta-dynamic switches ON "
        "ONLY the state-conditioned, occurrence-averaged beta head (alpha static) "
        "-- the two heads are gated independently, so each arm makes exactly one "
        "parameter dynamic.  Pearson r vs ground truth.  delta_alpha = "
        "a_r(alpha-dynamic) - a_r(baseline); delta_beta = b_r(beta-dynamic) - "
        "b_r(baseline).\n")
    table.append("PREDICTION: delta_alpha > 0 and GROWS with K (low-Fisher alpha "
                 "benefits from a dynamic readout, more so as stiffness "
                 "I(theta)/I(alpha) grows); delta_beta <= 0 (high-Fisher beta is "
                 "already well-determined static, dynamic only adds noise).\n")
    table.append("| K | a_r base | a_r alpha-dyn | delta_alpha | "
                 "b_r base | b_r beta-dyn | delta_beta |")
    table.append("|---|---|---|---|---|---|---|")

    summary = []
    for K in args.K:
        a_base = mean_over_seeds(K, "baseline", "a_pearson")
        a_dyn = mean_over_seeds(K, "alpha-dynamic", "a_pearson")
        b_base = mean_over_seeds(K, "baseline", "b_pearson")
        b_dyn = mean_over_seeds(K, "beta-dynamic", "b_pearson")
        d_alpha = a_dyn - a_base
        d_beta = b_dyn - b_base
        table.append(
            f"| {K} | {a_base:.3f} | {a_dyn:.3f} | {d_alpha:+.3f} | "
            f"{b_base:.3f} | {b_dyn:.3f} | {d_beta:+.3f} |")
        summary.append({"K": K, "a_base": a_base, "a_dyn": a_dyn,
                        "delta_alpha": d_alpha, "b_base": b_base,
                        "b_dyn": b_dyn, "delta_beta": d_beta})

    # --- verdict ---
    da = np.array([s["delta_alpha"] for s in summary], dtype=float)
    db = np.array([s["delta_beta"] for s in summary], dtype=float)
    Ks = np.array(args.K, dtype=float)
    table.append("\n## Verdict\n")
    alpha_helps = np.nanmean(da) > 0.0
    beta_neutral = np.nanmean(db) <= 0.02   # neutral-or-hurts band
    asymmetry = np.nanmean(da) - np.nanmean(db)
    table.append(f"mean delta_alpha = {np.nanmean(da):+.3f}  "
                 f"mean delta_beta = {np.nanmean(db):+.3f}  "
                 f"asymmetry (alpha - beta) = {asymmetry:+.3f}")
    if len(Ks) >= 2 and np.std(Ks) > 0 and np.isfinite(da).sum() >= 2:
        finite = np.isfinite(da)
        trend = float(np.corrcoef(Ks[finite], da[finite])[0, 1]) \
            if finite.sum() >= 2 else float("nan")
        table.append(f"delta_alpha-vs-K correlation = {trend:+.3f} "
                     "(prediction: positive, the lead grows with K)")
    if alpha_helps and beta_neutral:
        table.append("VERDICT: ASYMMETRY CONFIRMED.  Making the low-Fisher alpha "
                     "dynamic helps recovery; making the high-Fisher beta dynamic "
                     "does not.  Consistent with the Fisher-conditioning "
                     "mechanism.")
    elif alpha_helps and not beta_neutral:
        table.append("VERDICT: PARTIAL.  Alpha benefits from a dynamic readout, "
                     "but beta also moves materially -- the clean alpha-only "
                     "asymmetry is not isolated (check the beta-dynamic arm's "
                     "co-conditioning of alpha).")
    else:
        table.append("VERDICT: NOT CONFIRMED.  Making alpha dynamic did not help "
                     "on average; the Fisher-asymmetry prediction is not borne "
                     "out at this budget.  (Not spinning a null.)")

    md = "\n".join(table)
    (OUT / "alpha_beta_asymmetry_table.md").write_text(md, encoding="utf-8")
    blob = {
        "meta": {"device": device, "epochs": args.epochs, "K": args.K,
                 "seeds": args.seeds, "n_learners": args.n_learners,
                 "n_items": args.n_items, "seq_len": args.seq_len,
                 "item_key_dim": _ITEM_KEY_DIM},
        "rows": rows, "summary": summary,
    }
    with (OUT / "alpha_beta_asymmetry.json").open("w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)

    print("\n" + md)
    print(f"\ntotal wall {time.time() - t_start:.1f}s")
    print(f"wrote {OUT/'alpha_beta_asymmetry.json'} and "
          f"{OUT/'alpha_beta_asymmetry_table.md'}")


if __name__ == "__main__":
    main()
