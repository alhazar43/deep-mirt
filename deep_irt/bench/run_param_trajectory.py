"""run_param_trajectory.py -- the three-parameter recovery trajectory (talk fig 1).

Trains ONE standard decoupled model and records, at every checkpoint, how well
each of the three recovered numbers matches the truth that generated the data:

  score       the learner ability   (theta)
  difficulty  the item step location (beta)
  sharpness   the item discrimination(alpha)

This MIRRORS the fit-and-measure pattern of ``run_convergence.py`` (single
optimizer, one fit call, trajectory gathered through the ``fit`` callback so Adam
state is preserved across the whole run) and reuses ``datagen`` and
``metrics_bench`` exactly -- the same sign-aligned rank-agreement metric the
bench already uses (Spearman of recovered vs true, sign aligned).  The data is
the SAME static synthetic setting run_convergence uses (same N, Q, T, epochs).

The model is the project's standard decoupled GPCM/LSTM
(``DeepIRTModel(decoder='gpcm', encoder='lstm', decouple=True)``), which auto-
enables the decoupled default (state-conditioned discrimination, wide item key,
exp link).  Recovery therefore runs through the state path: discrimination is
occurrence-averaged from the state-conditioned head, the step thresholds are the
item-only read, and the score is the per-learner final-step theta scored against
the constant true ability (the static-data recovery the bench uses).

The headline the figure should show: score and difficulty rise fast to near 1,
sharpness rises slower.

Outputs (one file per K):
  deep_irt/bench/outputs/param_trajectory_K{K}.json
      {"epochs": [...],
       "score_mean"/"score_std"/"difficulty_*"/"sharpness_*": [...],
       per-seed runs and meta}

Usage
-----
  source ~/anaconda3/etc/profile.d/conda.sh && conda activate research
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=".;rl/src;ma-irt" \
      python deep_irt/bench/run_param_trajectory.py \
          [--K 4 8] [--seeds 0 1 2] [--epochs 150] [--log-every 5] \
          [--device cuda] [--quick]
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


def _match_all(model: DeepIRTModel, it, rp, ds) -> tuple[float, float, float]:
    """Sign-aligned rank-agreement (Spearman) of the three recovered numbers.

    Returns (score_match, difficulty_match, sharpness_match), each the
    ``metrics_bench`` Spearman of recovered vs true, sign aligned to truth and
    scored only on items the sequences actually visited (``seen``).  The standard
    decoupled model uses the state path, so discrimination is occurrence-averaged
    from the state head and the step thresholds are the item-only read.
    """
    if model._uses_state:
        rec = model.recover_item_params(it, rp)
    else:
        rec = model.recover_item_params()
    a_hat = np.asarray(rec["alpha"], dtype=float)
    b_hat = np.asarray(rec["beta"], dtype=float)
    seen = rec.get("seen")

    # Item recovery (difficulty = step thresholds, sharpness = discrimination),
    # scored on the visited items, sign aligned -- exactly metrics_bench.
    item = M.item_recovery(a_hat, b_hat, ds.gt.a, ds.gt.b, seen=seen)
    sharpness_match = item["a_spearman"]
    difficulty_match = item["b_spearman"]

    # Score (ability): per-learner final-step theta vs constant true theta, the
    # static-data recovery metrics_bench/engines use.
    theta_traj = model.track(it, rp).cpu().numpy()      # (N, T)
    theta_final = theta_traj[:, -1]
    score_match = M.theta_recovery_static(
        theta_final, ds.gt.theta0)["theta_spearman"]

    return float(score_match), float(difficulty_match), float(sharpness_match)


def _train_with_trajectory(ds, K, seed, device, epochs, log_every):
    """Train the standard decoupled model, logging the three matches per checkpoint.

    Single optimizer, one fit call -- the trajectory is gathered through the fit
    callback (Adam state preserved across the whole run), mirroring
    run_convergence.
    """
    it = torch.tensor(ds.items0, dtype=torch.long, device=device)
    rp = torch.tensor(ds.responses, dtype=torch.long, device=device)

    # The project's standard model: decoupled GPCM/LSTM.  Passing only decouple
    # =True lets the constructor turn on the decoupled default (state alpha, wide
    # item key, exp link).
    model = DeepIRTModel(
        num_items=ds.cfg.n_items, n_cats=K, decoder="gpcm", encoder="lstm",
        decouple=True, device=torch.device(device), seed=seed, **_CHEAP,
    )

    score_t: list[float] = []
    diff_t: list[float] = []
    sharp_t: list[float] = []
    eps: list[int] = []

    def cb(epoch, loss, m):
        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            s, d, a = _match_all(m, it, rp, ds)
            eps.append(int(epoch))
            score_t.append(s)
            diff_t.append(d)
            sharp_t.append(a)

    model.fit(it, rp, n_epochs=epochs, verbose=False, callback=cb)
    return eps, score_t, diff_t, sharp_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-learners", type=int, default=800)
    ap.add_argument("--n-items", type=int, default=60)
    ap.add_argument("--seq-len", type=int, default=60)
    ap.add_argument("--quick", action="store_true",
                    help="tiny smoke: K=[4], 1 seed, 6 epochs, small data")
    args = ap.parse_args()

    if args.quick:
        args.K = [4]
        args.seeds = [0]
        args.epochs = 6
        args.log_every = 2
        args.n_learners = 120
        args.n_items = 20
        args.seq_len = 20

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print(f"=== param trajectory: device={device} epochs={args.epochs} "
          f"log_every={args.log_every} K={args.K} seeds={args.seeds} "
          f"{'QUICK' if args.quick else ''} ===")

    t_start = time.time()
    written = []
    show = (1, 5, 10, 20, 40, 80, args.epochs)
    for K in args.K:
        runs = []   # list of (eps, score, diff, sharp) per seed
        for seed in args.seeds:
            ds = generate(BenchDataConfig(
                name=f"static_k{K}", kind="static", n_cats=K,
                n_learners=args.n_learners, n_items=args.n_items,
                seq_len=args.seq_len, seed=seed))
            runs.append(_train_with_trajectory(
                ds, K=K, seed=seed, device=device,
                epochs=args.epochs, log_every=args.log_every))

        # All runs log the same epochs (deterministic schedule); align by index.
        epochs_logged = runs[0][0]
        score = np.array([r[1] for r in runs])      # (seeds, T)
        diff = np.array([r[2] for r in runs])
        sharp = np.array([r[3] for r in runs])

        def ms(arr):
            return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)

        s_mean, s_std = ms(score)
        d_mean, d_std = ms(diff)
        a_mean, a_std = ms(sharp)

        print(f"\n--- K={K} ({len(args.seeds)} seeds) ---")
        for i, e in enumerate(epochs_logged):
            if e in show:
                print(f"  ep{e:3d}: score={s_mean[i]:.3f}  "
                      f"difficulty={d_mean[i]:.3f}  sharpness={a_mean[i]:.3f}")

        blob = {
            "meta": {"K": K, "seeds": args.seeds, "epochs": args.epochs,
                     "log_every": args.log_every, "device": device,
                     "n_learners": args.n_learners, "n_items": args.n_items,
                     "seq_len": args.seq_len, "quick": bool(args.quick),
                     "metric": "spearman (sign-aligned), score=final-step theta",
                     "model": "DeepIRTModel(decoder=gpcm,encoder=lstm,"
                              "decouple=True)"},
            "epochs": epochs_logged,
            "score_mean": s_mean.tolist(), "score_std": s_std.tolist(),
            "difficulty_mean": d_mean.tolist(), "difficulty_std": d_std.tolist(),
            "sharpness_mean": a_mean.tolist(), "sharpness_std": a_std.tolist(),
            "score_runs": score.tolist(),
            "difficulty_runs": diff.tolist(),
            "sharpness_runs": sharp.tolist(),
        }
        path = OUT / f"param_trajectory_K{K}.json"
        if path.exists():
            raise SystemExit(
                f"REFUSING to overwrite existing file: {path}. Stop and report.")
        with path.open("w", encoding="utf-8") as fh:
            json.dump(blob, fh, indent=2)
        written.append(path)

    print(f"\ntotal wall {time.time() - t_start:.1f}s")
    print("wrote: " + ", ".join(str(p) for p in written))


if __name__ == "__main__":
    main()
