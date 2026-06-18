"""run_convergence.py -- the alpha-recovery convergence-RATE experiment (RQ1).

Records the alpha-recovery trajectory for static-alpha vs state-conditioned
(dynamic) alpha to test whether the dynamic head wins through the learning
dynamics and how that tracks K.  FINDING (docs/LEARNING_DYNAMICS_STUDY.md section 3, RQ1/RQ2, 5 seeds):
it is NOT an earlier peel-up.  Both curves crawl early (dynamic is not ahead); the
dynamic head accelerates in mid training (ep20-40) and, at high K, keeps a
PERMANENT endpoint lead because static-alpha is trapped by the stiffness ceiling
I(theta)/I(alpha) (which grows with K).  So a mid-training acceleration plus a
K-growing endpoint gap, not an early lead.  The PREDICTION block below is the
original hypothesis, kept for the record and partly disconfirmed by this finding.

This runner records the alpha-recovery TRAJECTORY -- Pearson r vs the ground-truth
discrimination, logged every ``log_every`` epochs via the ``DeepIRTModel.fit``
callback (one optimizer, no warm-restart resets) -- for two configs, holding beta
static in both and the capacity fixed (both decoupled, item_key_dim=64):

  alpha-static   state_alpha=False   (alpha rides the static item-key map)
  alpha-dynamic  state_alpha=True    (alpha rides the state-conditioned head)

per K in the sweep, single seed by default.

PREDICTION (the hypothesis under test)
--------------------------------------
Dynamic-alpha recovery peels up EARLIER than static-alpha, and the lead
(alpha_dynamic - alpha_static at matched epoch) WIDENS with K, because the
stiffness I(theta)/I(alpha) -- which governs the conditioning advantage -- grows
with K.  At small K the two curves are close; at large K the dynamic curve
separates sooner and by more.

Outputs (one file per K):
  deep_irt/bench/outputs/convergence_K{K}.json
      {"alpha_static": [[epoch, a_r], ...], "alpha_dynamic": [[epoch, a_r], ...]}

Usage
-----
  source ~/anaconda3/etc/profile.d/conda.sh && conda activate research
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=".;rl/src;ma-irt" \
      python deep_irt/bench/run_convergence.py \
          [--K 4 8 11] [--seed 0] [--epochs 150] [--log-every 5] [--device cuda]
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
from deep_irt.core.model import DeepIRTModel

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "outputs"
OUT.mkdir(exist_ok=True)

_CHEAP = dict(emb_dim=8, hidden_dim=32)
_ITEM_KEY_DIM = 64
_ALPHA_LOG_SCALE = 1.0


def _alpha_recovery(model: DeepIRTModel, it, rp, a_true) -> float:
    """Pearson r of recovered alpha vs ground truth.

    The dynamic-alpha model recovers occurrence-averaged alpha from the
    state-conditioned head and needs the (N, T) sequence; the static-alpha model
    reads a per-item alpha and needs no sequence.  Dispatch on ``_uses_state``.
    Sign-aligned to the truth (the latent scale orientation is not identified),
    scored only on items the sequences actually visited (``seen``) when reported.
    """
    if model._uses_state:
        rec = model.recover_item_params(it, rp)
    else:
        rec = model.recover_item_params()           # per-item static read (Q,)
    a_hat = np.asarray(rec["alpha"], dtype=float)
    seen = rec.get("seen")
    if seen is not None:
        a_hat, a_ref = a_hat[seen], a_true[seen]
    else:
        a_ref = a_true
    a_hat = M._align_sign(a_hat, a_ref)
    return M._pearson(a_hat, a_ref)


def _train_with_trajectory(ds, state_alpha, K, seed, device, epochs, log_every):
    """Train one config, logging alpha-recovery every ``log_every`` epochs.

    The trajectory is gathered through the fit callback (single optimizer, Adam
    state preserved across the whole run -- NOT chunked fit calls, which would
    reset Adam).
    """
    it = torch.tensor(ds.items0, dtype=torch.long, device=device)
    rp = torch.tensor(ds.responses, dtype=torch.long, device=device)
    a_true = ds.gt.a

    model = DeepIRTModel(
        num_items=ds.cfg.n_items, n_cats=K, decoder="gpcm",
        state_alpha=state_alpha, item_key_dim=_ITEM_KEY_DIM,
        alpha_log_scale=_ALPHA_LOG_SCALE, device=torch.device(device),
        seed=seed, **_CHEAP,
    )

    traj: list[list[float]] = []

    def cb(epoch, loss, m):
        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            a_r = _alpha_recovery(m, it, rp, a_true)
            traj.append([int(epoch), float(a_r)])

    model.fit(it, rp, n_epochs=epochs, verbose=False, callback=cb)
    return traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, nargs="+", default=[4, 8, 11])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-learners", type=int, default=800)
    ap.add_argument("--n-items", type=int, default=60)
    ap.add_argument("--seq-len", type=int, default=60)
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print(f"=== convergence: device={device} epochs={args.epochs} "
          f"log_every={args.log_every} K={args.K} seeds={args.seeds} ===")

    t_start = time.time()
    written = []
    show = (1, 5, 10, 20, 40, 80, args.epochs)
    for K in args.K:
        static_runs, dynamic_runs = [], []
        for seed in args.seeds:
            ds = generate(BenchDataConfig(
                name=f"static_k{K}", kind="static", n_cats=K,
                n_learners=args.n_learners, n_items=args.n_items,
                seq_len=args.seq_len, seed=seed))
            static_runs.append(_train_with_trajectory(
                ds, state_alpha=False, K=K, seed=seed, device=device,
                epochs=args.epochs, log_every=args.log_every))
            dynamic_runs.append(_train_with_trajectory(
                ds, state_alpha=True, K=K, seed=seed, device=device,
                epochs=args.epochs, log_every=args.log_every))

        # All runs log the same epochs (deterministic schedule); align by index.
        epochs_logged = [e for e, _ in static_runs[0]]
        s = np.array([[v for _, v in r] for r in static_runs])   # (seeds, T)
        d = np.array([[v for _, v in r] for r in dynamic_runs])
        s_mean, s_std = s.mean(0), s.std(0)
        d_mean, d_std = d.mean(0), d.std(0)
        gap_mean = d_mean - s_mean

        print(f"\n--- K={K} ({len(args.seeds)} seeds) ---")
        for i, e in enumerate(epochs_logged):
            if e in show:
                print(f"  ep{e:3d}: static={s_mean[i]:.3f}+-{s_std[i]:.3f}  "
                      f"dynamic={d_mean[i]:.3f}+-{d_std[i]:.3f}  "
                      f"gap={gap_mean[i]:+.3f}")

        blob = {
            "meta": {"K": K, "seeds": args.seeds, "epochs": args.epochs,
                     "log_every": args.log_every, "device": device,
                     "n_learners": args.n_learners, "n_items": args.n_items,
                     "seq_len": args.seq_len, "item_key_dim": _ITEM_KEY_DIM},
            "epochs": epochs_logged,
            "alpha_static_mean": s_mean.tolist(),
            "alpha_static_std": s_std.tolist(),
            "alpha_dynamic_mean": d_mean.tolist(),
            "alpha_dynamic_std": d_std.tolist(),
            "alpha_static_runs": static_runs,
            "alpha_dynamic_runs": dynamic_runs,
        }
        path = OUT / f"convergence_K{K}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(blob, fh, indent=2)
        written.append(path)

    print(f"\ntotal wall {time.time() - t_start:.1f}s")
    print("wrote: " + ", ".join(str(p) for p in written))


if __name__ == "__main__":
    main()
