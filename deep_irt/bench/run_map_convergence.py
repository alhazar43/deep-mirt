"""run_map_convergence.py -- discrimination recovery PROFILE for positive maps.

The endpoint ablation (``run_alpha_map_bench.py`` -> outputs/alpha_map_results
.json) shows where each positive map ``g(.)`` for discrimination LANDS after a
full fit.  At K=4 every map ties near a_spearman ~ 0.90-0.93.  The endpoints do
not show how the maps GET there.  This runner records the per-epoch trajectory of
discrimination recovery for a chosen set of maps so the convergence PROFILE can
be read directly: smooth positive maps share one profile and the non-smooth maps
lag early before catching up.

It mirrors two existing runners.  The per-epoch recovery-logging pattern is
``run_param_trajectory.py`` (single optimizer, one fit call, trajectory gathered
through the fit callback so Adam state is preserved across the whole run).  The
engine construction and the MATCHED effective-alpha initialization are the fair-
comparison controls from ``run_alpha_map_bench.py``: alpha-head weights are
zeroed and the bias is set by the inverse map, so every map starts from exactly
the same effective discrimination for every item and occurrence.  The matched-
init helpers are replicated here (not imported) to keep this runner self-
contained and off the Codex-owned bench module.

For each map in {exp, softplus, sigmoid, relu, square} this trains the bench
model (LSTM/GPCM, state_alpha, wide item key) with matched init and logs
discrimination recovery (a_spearman, sign-aligned, scored on seen items) every
``--log-every`` epochs.

Learning rate
-------------
The endpoint file's best-LR view selects lr=0.01 for every K=4 map (a single
common best LR), so this runner uses lr=0.01 for all maps by default.  The load-
bearing fairness control is the matched effective-alpha init, not the LR.

Outputs
-------
  deep_irt/bench/outputs/map_convergence_K{K}.json
      {"epochs": [...],
       "maps": {map: {"a_spearman_mean": [...], "a_spearman_std": [...],
                      "runs": [[...] per seed]}},
       meta}

Usage
-----
  source ~/anaconda3/etc/profile.d/conda.sh && conda activate research
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=".;rl/src;ma-irt" \
      python deep_irt/bench/run_map_convergence.py \
          [--K 4] [--seeds 0 1 2] [--epochs 150] [--log-every 5] \
          [--device cuda] [--lr 0.01] [--quick]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from deep_irt.bench import metrics_bench as M
from deep_irt.bench.datagen import BenchDataConfig, generate
from deep_irt.bench.engines import DeepIRTEngine

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "outputs"
OUT.mkdir(exist_ok=True)

_CHEAP = dict(emb_dim=8, hidden_dim=32)
_ITEM_KEY_DIM = 64
_DEFAULT_ALPHA_INIT = 0.5
_DEFAULT_LR = 1e-2          # the single common best LR for every K=4 map.

# Two smooth-map representatives plus exp (smooth family) and the two non-smooth
# maps.  The kwargs mirror run_alpha_map_bench's _MAP_SPECS so the matched init
# uses the same constants.
_SMOOTH = ("exp", "softplus", "sigmoid")
_NONSMOOTH = ("relu", "square")
_MAP_SPECS: dict[str, dict[str, float]] = {
    "exp": {"scale": 1.0},
    "softplus": {},
    "sigmoid": {},
    "relu": {},
    "square": {"epsilon": 1e-6},
}
_DEFAULT_MAPS = list(_MAP_SPECS)


# ---------------------------------------------------------------------------
# Matched effective-alpha initialization (replicated, self-contained)
# ---------------------------------------------------------------------------

def _softplus_inverse(y: float) -> float:
    if y <= 0.0:
        raise ValueError("softplus inverse requires y > 0.")
    return math.log(math.expm1(y))


def _logit(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError("logit requires 0 < p < 1.")
    return math.log(p / (1.0 - p))


def _raw_alpha_for_map(alpha_target: float, name: str, kwargs: dict) -> float:
    """Raw head bias whose effective alpha equals ``alpha_target`` under ``g``."""
    name = name.lower().replace("-", "_")
    if name == "relu":
        if alpha_target <= 0.0:
            raise ValueError("relu matched init requires alpha_target > 0.")
        return float(alpha_target)
    if name == "softplus":
        return _softplus_inverse(alpha_target)
    if name == "sigmoid":
        return _logit(alpha_target)
    if name == "square":
        epsilon = float(kwargs.get("epsilon", 1e-6))
        base = alpha_target - epsilon
        if base <= 0.0:
            raise ValueError("square matched init requires alpha_target > epsilon.")
        return math.sqrt(base)
    if name == "exp":
        scale = float(kwargs.get("scale", 1.0))
        if alpha_target <= 0.0:
            raise ValueError("exp matched init requires alpha_target > 0.")
        return math.log(alpha_target) / scale
    raise ValueError(f"unsupported map {name!r} for matched init.")


def _base_gpcm_decoder(decoder):
    return decoder._gpcm if hasattr(decoder, "_gpcm") else decoder


def set_matched_alpha_init(model, alpha_target: float, name: str,
                           kwargs: dict) -> dict:
    """Zero the alpha heads and set the bias so every map starts at the same
    effective discrimination.  Mirrors run_alpha_map_bench.set_matched_alpha_init.
    """
    dec = _base_gpcm_decoder(model.decoder)
    if not hasattr(dec, "_alpha_pos"):
        raise ValueError("matched alpha init requires a GPCM-style alpha decoder.")

    raw = _raw_alpha_for_map(alpha_target, name, kwargs)
    heads = []
    with torch.no_grad():
        if hasattr(dec, "fc_a"):
            dec.fc_a.weight.zero_()
            dec.fc_a.bias.fill_(raw)
            heads.append("fc_a")
        if getattr(dec, "state_alpha", False) and hasattr(dec, "fc_a_state"):
            dec.fc_a_state.weight.zero_()
            dec.fc_a_state.bias.fill_(raw)
            heads.append("fc_a_state")

    probe = torch.tensor([[raw]], dtype=torch.float32, device=model.device)
    actual = float(dec._alpha_pos(probe).item())
    if not math.isclose(actual, alpha_target, rel_tol=1e-5, abs_tol=1e-6):
        raise RuntimeError(
            f"matched init failed for {name}: "
            f"target={alpha_target} actual={actual} raw={raw}")
    return {"alpha_init": float(alpha_target), "raw_alpha_init": float(raw),
            "alpha_init_check": actual, "initialized_heads": heads}


# ---------------------------------------------------------------------------
# Per-epoch discrimination recovery
# ---------------------------------------------------------------------------

def _a_spearman(engine, ds) -> float:
    """Sign-aligned discrimination recovery (Spearman) on seen items."""
    rec = engine.recover()
    seen = rec.get("seen")
    item = M.item_recovery(rec["a"], rec["b"], ds.gt.a, ds.gt.b, seen=seen)
    return float(item["a_spearman"])


def _train_with_profile(ds, map_name, kwargs, K, seed, device, epochs,
                        log_every, lr, alpha_init):
    """Train one map/seed, logging a_spearman through the fit callback."""
    eng = DeepIRTEngine(
        ds, decoder="gpcm", state_alpha=True, item_key_dim=_ITEM_KEY_DIM,
        alpha_pos_map=map_name, alpha_pos_kwargs=kwargs,
        device=device, seed=seed, **_CHEAP,
    )
    init_info = set_matched_alpha_init(eng.model, alpha_init, map_name, kwargs)

    eps: list[int] = []
    a_t: list[float] = []

    def cb(epoch, loss, model):
        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            eps.append(int(epoch))
            a_t.append(_a_spearman(eng, ds))

    eng.fit(n_epochs=epochs, lr=lr, callback=cb)
    return eps, a_t, init_info


def _epoch_to_threshold(epochs, vals, thr=0.8):
    """First logged epoch whose recovery reaches ``thr`` (None if never)."""
    for e, v in zip(epochs, vals):
        if v == v and v >= thr:
            return int(e)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, nargs="+", default=[4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--lr", type=float, default=_DEFAULT_LR)
    ap.add_argument("--alpha-init", type=float, default=_DEFAULT_ALPHA_INIT)
    ap.add_argument("--maps", type=str, nargs="+", default=_DEFAULT_MAPS)
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

    maps = {m: dict(_MAP_SPECS[m]) for m in args.maps}
    print(f"=== map convergence profile: device={device} epochs={args.epochs} "
          f"log_every={args.log_every} lr={args.lr} alpha_init={args.alpha_init} "
          f"K={args.K} seeds={args.seeds} maps={list(maps)} "
          f"{'QUICK' if args.quick else ''} ===")

    t_start = time.time()
    written = []
    show = (1, 5, 10, 20, 40, 80, args.epochs)

    for K in args.K:
        # Per-seed datasets reused across maps so all maps see identical data.
        datasets = {
            seed: generate(BenchDataConfig(
                name=f"static_k{K}", kind="static", n_cats=K,
                n_learners=args.n_learners, n_items=args.n_items,
                seq_len=args.seq_len, n_holdout=min(10, args.seq_len), seed=seed))
            for seed in args.seeds
        }

        map_results: dict[str, dict] = {}
        epochs_logged = None
        init_records: dict[str, dict] = {}

        for map_name, kwargs in maps.items():
            runs = []          # per seed: a_spearman list
            for seed in args.seeds:
                eps, a_t, init_info = _train_with_profile(
                    datasets[seed], map_name, kwargs, K=K, seed=seed,
                    device=device, epochs=args.epochs, log_every=args.log_every,
                    lr=args.lr, alpha_init=args.alpha_init)
                runs.append(a_t)
                if epochs_logged is None:
                    epochs_logged = eps
                init_records[map_name] = init_info
            arr = np.array(runs, dtype=float)              # (seeds, T)
            a_mean = np.nanmean(arr, axis=0)
            a_std = np.nanstd(arr, axis=0)
            map_results[map_name] = {
                "a_spearman_mean": a_mean.tolist(),
                "a_spearman_std": a_std.tolist(),
                "runs": arr.tolist(),
                "map_kwargs": dict(kwargs),
                "family": "smooth" if map_name in _SMOOTH else "non-smooth",
                "epoch_to_0.8": _epoch_to_threshold(epochs_logged, a_mean, 0.8),
                "final_a_spearman_mean": float(a_mean[-1]),
                "final_a_spearman_std": float(a_std[-1]),
                **init_records.get(map_name, {}),
            }
            e80 = map_results[map_name]["epoch_to_0.8"]
            print(f"  {map_name:10s} ({map_results[map_name]['family']:10s}) "
                  f"final a_sp={a_mean[-1]:.3f}+-{a_std[-1]:.3f}  "
                  f"ep->0.8={e80 if e80 is not None else 'never'}")

        # Compact trajectory printout.
        print(f"\n--- K={K} a_spearman trajectory ({len(args.seeds)} seeds) ---")
        header = "  epoch  " + "  ".join(f"{m:>10s}" for m in maps)
        print(header)
        for i, e in enumerate(epochs_logged):
            if e in show:
                row = "  ".join(
                    f"{map_results[m]['a_spearman_mean'][i]:10.3f}" for m in maps)
                print(f"  {e:5d}  {row}")

        blob = {
            "meta": {
                "K": K, "seeds": args.seeds, "epochs": args.epochs,
                "log_every": args.log_every, "lr": args.lr,
                "alpha_init": args.alpha_init, "device": device,
                "n_learners": args.n_learners, "n_items": args.n_items,
                "seq_len": args.seq_len, "quick": bool(args.quick),
                "cheap": _CHEAP, "item_key_dim": _ITEM_KEY_DIM,
                "smooth_maps": list(_SMOOTH), "nonsmooth_maps": list(_NONSMOOTH),
                "metric": "a_spearman (sign-aligned, seen items)",
                "lr_source": "best_lr view of outputs/alpha_map_results.json "
                             "(lr=0.01 for every K=4 map)",
                "model": "DeepIRTEngine(gpcm,lstm,state_alpha,item_key_dim=64)",
                "torch_version": torch.__version__,
            },
            "epochs": epochs_logged,
            "maps": map_results,
        }
        path = OUT / f"map_convergence_K{K}.json"
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
