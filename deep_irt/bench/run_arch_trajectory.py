"""run_arch_trajectory.py -- architecture-independence of the alpha trajectory.

Replicates Phase-1 of the learning-dynamics trajectory (the SHARED-NARROW /
SHARED-WIDE / DECOUPLED comparison from ``run_trajectory_probe.py``) but with a
swappable ENCODER so the peak-then-decay claim can be tested on the transformer
and DKVMN backbones, not just the LSTM it was established on.

The three configs match ``run_trajectory_probe.py`` CONFIGS exactly:

  SHARED-NARROW  emb_dim=8,  item_key_dim=None   (state_alpha, exp link)
  SHARED-WIDE    emb_dim=64, item_key_dim=None   (state_alpha, exp link)
  DECOUPLED      emb_dim=8,  item_key_dim=64      (state_alpha, exp link)

Each (config, seed) is ONE continuous Adam run (warm checkpoints through the
``model.fit`` callback, so Adam state is preserved across the whole trajectory).
At every checkpoint we recover alpha (occurrence-averaged, state-conditioned) and
static theta and score the sign-aligned Spearman against ground truth -- the same
metrics ``run_trajectory_probe.py`` uses.

This runner does NOT edit any Codex-owned bench file.  It reuses ``datagen``,
``metrics_bench`` and ``DeepIRTModel`` directly, mirroring the recovery recipe of
``run_param_trajectory.py`` and ``run_trajectory_probe.py``.

The claim to reproduce (LSTM numbers): under SHARED-WIDE the alpha-Spearman rises
then DECAYS with training (peak ~0.906 ep50 -> ~0.787 ep500); DECOUPLED rises and
HOLDS (~0.912); SHARED-NARROW plateaus ~0.73 at K=4.

Usage
-----
  source ~/anaconda3/etc/profile.d/conda.sh && conda activate research
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=".;rl/src;ma-irt" \
      python deep_irt/bench/run_arch_trajectory.py \
          --encoder transformer [--K 4] [--seeds 0 1 2] \
          [--checkpoints 25 50 100 150 300 500] [--device cuda] [--quick]

Outputs
-------
  deep_irt/bench/outputs/arch_trajectory_{encoder}_K{K}.json
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

# Three configs -- identical to run_trajectory_probe.py CONFIGS, all
# state_alpha=True / exp link, hidden_dim=32, GPCM.
_BASE = dict(hidden_dim=32, state_alpha=True, alpha_log_scale=1.0, decoder="gpcm")
CONFIGS = {
    "SHARED-NARROW": dict(**_BASE, emb_dim=8, item_key_dim=None),
    "SHARED-WIDE":   dict(**_BASE, emb_dim=64, item_key_dim=None),
    "DECOUPLED":     dict(**_BASE, emb_dim=8, item_key_dim=64),
}


def _build_model(cfg, num_items, n_cats, encoder, device, seed):
    """DeepIRTModel with EXPLICIT alpha knobs (no reliance on the decouple default)."""
    return DeepIRTModel(
        num_items=num_items,
        emb_dim=cfg["emb_dim"],
        hidden_dim=cfg["hidden_dim"],
        n_cats=n_cats,
        decoder=cfg["decoder"],
        state_alpha=cfg["state_alpha"],
        item_key_dim=cfg.get("item_key_dim"),
        alpha_log_scale=cfg.get("alpha_log_scale"),
        encoder=encoder,
        device=device,
        seed=seed,
    )


def _recover_alpha_theta(model, it_all, rp_all, ds):
    """Sign-aligned Spearman of (alpha, static theta) vs ground truth."""
    rec = model.recover_item_params(it_all, rp_all)
    im = M.item_recovery(rec["alpha"], rec["beta"], ds.gt.a, ds.gt.b,
                         seen=rec.get("seen"))
    theta_traj = model.track(it_all, rp_all).cpu().numpy()
    tm = M.theta_recovery_static(theta_traj[:, -1], ds.gt.theta0)
    return float(im["a_spearman"]), float(tm["theta_spearman"])


def _train_with_checkpoints(model, it_train, rp_train, ds, checkpoints,
                            device, max_epoch):
    """One continuous fit (Adam preserved via callback) recovering at checkpoints."""
    it_all = torch.tensor(ds.items0, dtype=torch.long, device=device)
    rp_all = torch.tensor(ds.responses, dtype=torch.long, device=device)
    cps = set(checkpoints)
    results = {}

    def cb(epoch, loss, m):
        if epoch in cps:
            m.encoder.eval()
            m.decoder.eval()
            a, th = _recover_alpha_theta(m, it_all, rp_all, ds)
            results[int(epoch)] = {"epoch": int(epoch),
                                   "alpha_spearman": round(a, 4),
                                   "theta_static": round(th, 4),
                                   "loss": round(float(loss), 4)}
            m.encoder.train()
            m.decoder.train()

    model.fit(it_train, rp_train, n_epochs=max_epoch, lr=1e-2,
              verbose=False, callback=cb)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True,
                    choices=["lstm", "transformer", "dkvmn"])
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--checkpoints", type=int, nargs="+",
                    default=[25, 50, 100, 150, 300, 500])
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-learners", type=int, default=800)
    ap.add_argument("--n-items", type=int, default=60)
    ap.add_argument("--seq-len", type=int, default=60)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.seeds = [0]
        args.checkpoints = [10, 25, 50]
        args.n_learners, args.n_items, args.seq_len = 120, 20, 20

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    max_epoch = max(args.checkpoints)

    print(f"=== arch trajectory: encoder={args.encoder} K={args.K} "
          f"seeds={args.seeds} checkpoints={args.checkpoints} device={device} "
          f"{'QUICK' if args.quick else ''} ===")

    t_start = time.time()
    # config -> list over seeds of {epoch: metrics}
    raw = {name: [] for name in CONFIGS}

    for seed in args.seeds:
        ds = generate(BenchDataConfig(
            name=f"static_k{args.K}", kind="static", n_cats=args.K,
            n_learners=args.n_learners, n_items=args.n_items,
            seq_len=args.seq_len, seed=seed))
        it_train = torch.tensor(ds.items0[ds.train_idx], dtype=torch.long)
        rp_train = torch.tensor(ds.responses[ds.train_idx], dtype=torch.long)
        for name, cfg in CONFIGS.items():
            t0 = time.time()
            model = _build_model(cfg, ds.cfg.n_items, ds.cfg.n_cats,
                                 args.encoder, device, seed)
            res = _train_with_checkpoints(model, it_train, rp_train, ds,
                                          args.checkpoints, device, max_epoch)
            raw[name].append(res)
            last = args.checkpoints[-1]
            print(f"  seed{seed} {name:14s} "
                  f"alpha[{args.checkpoints[0]}->{last}]="
                  f"{res[args.checkpoints[0]]['alpha_spearman']:.3f}->"
                  f"{res[last]['alpha_spearman']:.3f}  "
                  f"theta_end={res[last]['theta_static']:.3f}  "
                  f"{time.time()-t0:.1f}s")

    # Aggregate over seeds per (config, epoch).
    agg = {}
    for name, seed_list in raw.items():
        per_epoch = {}
        for ep in args.checkpoints:
            a = [s[ep]["alpha_spearman"] for s in seed_list]
            th = [s[ep]["theta_static"] for s in seed_list]
            per_epoch[ep] = {
                "epoch": ep,
                "alpha_mean": round(float(np.mean(a)), 4),
                "alpha_std": round(float(np.std(a)), 4),
                "theta_mean": round(float(np.mean(th)), 4),
                "theta_std": round(float(np.std(th)), 4),
            }
        agg[name] = per_epoch

    # Peak-vs-final summary per config (the load-bearing claim numbers).
    print(f"\n--- alpha peak vs final (mean over {len(args.seeds)} seeds) ---")
    summary = {}
    for name in CONFIGS:
        means = [(ep, agg[name][ep]["alpha_mean"]) for ep in args.checkpoints]
        peak_ep, peak = max(means, key=lambda x: x[1])
        final_ep, final = means[-1]
        decay = peak - final
        summary[name] = {"peak_epoch": peak_ep, "peak_alpha": peak,
                         "final_epoch": final_ep, "final_alpha": final,
                         "decay": round(decay, 4)}
        print(f"  {name:14s} peak={peak:.3f}@ep{peak_ep}  "
              f"final={final:.3f}@ep{final_ep}  decay={decay:+.3f}")

    blob = {
        "meta": {"encoder": args.encoder, "K": args.K, "seeds": args.seeds,
                 "checkpoints": args.checkpoints, "device": device,
                 "n_learners": args.n_learners, "n_items": args.n_items,
                 "seq_len": args.seq_len, "quick": bool(args.quick),
                 "configs": {k: {kk: vv for kk, vv in v.items()}
                             for k, v in CONFIGS.items()},
                 "metric": "spearman (sign-aligned); alpha occurrence-averaged "
                           "state-conditioned; theta static final-step"},
        "agg": {name: {str(ep): m for ep, m in per.items()}
                for name, per in agg.items()},
        "summary": summary,
        "raw": {name: [{str(ep): m for ep, m in s.items()} for s in lst]
                for name, lst in raw.items()},
    }
    path = OUT / f"arch_trajectory_{args.encoder}_K{args.K}.json"
    if path.exists():
        raise SystemExit(f"REFUSING to overwrite existing file: {path}.")
    with path.open("w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)

    print(f"\ntotal wall {time.time()-t_start:.1f}s")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
