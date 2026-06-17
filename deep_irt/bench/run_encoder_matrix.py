"""Comprehensive matrix: ma-irt MAGPCM vs deep_irt {lstm, transformer, dkvmn}
x {shared-e8, shared-e72 (param-matched to decoupled), decoupled (e8+ae64)}.

Prediction (acc, QWK on held-out tail) + recovery (Pearson r: alpha, beta, theta)
+ wall-clock + params.  Same static-K data, fixed seeds, prediction loss (WOL).

deep_irt uses the canonical item-blind encoder (no separate_theta).  ma-irt uses
its real separated ability pathway (separate_theta=True).
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score

from deep_irt.bench.datagen import BenchDataConfig, generate
from deep_irt.bench.engines import DeepIRTEngine, MaIrtEngine
from deep_irt.bench.run_alpha_fix_bench import run_cell

# (label, kind, encoder, emb_dim, alpha_emb_dim, encoder_kwargs)
CONFIGS = [
    ("ma-irt MAGPCM", "mairt", "dkvmn", None, None, None),
]
for enc, ekw in [("lstm", None),
                 ("transformer", {"n_heads": 2, "n_layers": 2}),
                 ("dkvmn", {"memory_size": 20})]:
    CONFIGS += [
        (f"{enc} shared-e8", "deep", enc, 8, None, ekw),
        (f"{enc} shared-e72", "deep", enc, 72, None, ekw),
        (f"{enc} decoupled", "deep", enc, 8, 64, ekw),
    ]


def nparams(eng):
    m = eng.model
    if hasattr(m, "parameters"):
        return sum(p.numel() for p in m.parameters())
    return sum(p.numel() for p in
              (list(m.encoder.parameters()) + list(m.decoder.parameters())))


def predm(eng):
    probs, targ = eng.predict_heldout()
    pred = probs.argmax(1)
    acc = float((pred == targ).mean())
    try:
        qwk = float(cohen_kappa_score(targ, pred, weights="quadratic"))
    except Exception:
        qwk = float("nan")
    return acc, qwk


def build(kind, enc, emb, ae, ekw, ds, seed, device):
    if kind == "mairt":
        return MaIrtEngine(ds, encoder="dkvmn", separate_theta=True,
                           device=device, seed=seed)
    return DeepIRTEngine(ds, decoder="gpcm", emb_dim=emb, hidden_dim=32,
                         state_alpha=True, alpha_emb_dim=ae, alpha_log_scale=1.0,
                         encoder=enc, encoder_kwargs=ekw, device=device, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"\n=== encoder matrix, static K={args.K}, {len(args.seeds)} seeds, "
          f"{args.epochs} epochs, prediction loss ===")
    print(f"{'model':18s} {'acc':>6s} {'qwk':>6s} {'alpha':>6s} {'beta':>6s} "
          f"{'theta':>6s} {'sec':>6s} {'params':>8s}")
    for label, kind, enc, emb, ae, ekw in CONFIGS:
        a, b, th, ac, qw, tm, pr = [], [], [], [], [], [], None
        for seed in args.seeds:
            ds = generate(BenchDataConfig(name="s", kind="static", n_cats=args.K,
                                          n_learners=800, n_items=60, seq_len=60,
                                          seed=seed))
            eng = build(kind, enc, emb, ae, ekw, ds, seed, args.device).fit(
                n_epochs=args.epochs)
            c = run_cell(eng, ds)
            acc, qwk = predm(eng)
            a.append(c["a_pearson"]); b.append(c["b_pearson"])
            th.append(c["theta_pearson"]); ac.append(acc); qw.append(qwk)
            tm.append(eng._train_time); pr = nparams(eng)
        print(f"{label:18s} {np.mean(ac):6.3f} {np.mean(qw):6.3f} "
              f"{np.mean(a):6.3f} {np.mean(b):6.3f} {np.mean(th):6.3f} "
              f"{np.mean(tm):6.1f} {pr:8d}")


if __name__ == "__main__":
    main()
