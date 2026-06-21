"""run_neural_map_isolation.py -- neural controls after the E7 refutation.

E7 rejected the scalar alpha-space preconditioner-only explanation.  This runner
therefore stays inside the neural model and isolates the likely remaining
mechanism:

  * learned versus frozen sequence encoder pieces,
  * learned versus frozen item embeddings,
  * raw/ReLU maps with matched output range,
  * optional gradient clipping for the raw/ReLU controls.

The protocol deliberately reuses the E5/E6 alpha-map runner's matched effective
alpha initialization, scoring, and table format conventions.

Usage
-----
  conda run -n research python -m deep_irt.bench.run_neural_map_isolation --quick

Focused full starting point:
  conda run -n research python -m deep_irt.bench.run_neural_map_isolation \
      --device cuda --K 4 --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from deep_irt.bench.datagen import BenchDataConfig, generate
from deep_irt.bench.engines import DeepIRTEngine
from deep_irt.bench.run_alpha_map_bench import (
    FitMonitor,
    run_map_cell,
    set_matched_alpha_init,
)

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "outputs"
OUT.mkdir(exist_ok=True)

_CHEAP = dict(emb_dim=8, hidden_dim=32)
_ITEM_KEY_DIM = 64
_ALPHA_INIT = 0.5
_DEFAULT_SEEDS = [0, 1, 2, 3, 4]
_QUICK_SEEDS = [0]
_DEFAULT_LRS = [1e-2]
_QUICK_LRS = [1e-2]


@dataclass(frozen=True)
class MapSpec:
    name: str
    kwargs: dict
    grad_clip_norm: float | None = None


_MAP_SPECS: dict[str, MapSpec] = {
    "exp": MapSpec("exp", {"scale": 1.0}),
    "softplus": MapSpec("softplus", {}),
    "identity": MapSpec("identity", {}),
    "relu": MapSpec("relu", {}),
    # Range-matched raw/ReLU controls.  The cap covers the observed E5/E6 smooth
    # map range while preventing the raw heads from explaining failure through
    # simple alpha explosion.
    "clipped_identity": MapSpec(
        "clipped_identity",
        {"clip_min": -2.0, "clip_max": 2.0},
        grad_clip_norm=1.0,
    ),
    "clipped_relu": MapSpec(
        "clipped_relu",
        {"clip_min": 0.0, "clip_max": 2.0},
        grad_clip_norm=1.0,
    ),
}

_DEFAULT_MAPS = [
    "exp",
    "softplus",
    "identity",
    "relu",
    "clipped_identity",
    "clipped_relu",
]
_QUICK_MAPS = ["softplus", "relu", "clipped_relu"]

_REPRESENTATIONS = {
    "learned_all",
    "frozen_backbone",
    "frozen_item_embeddings",
    "frozen_encoder",
}
_DEFAULT_REPRESENTATIONS = [
    "learned_all",
    "frozen_backbone",
    "frozen_item_embeddings",
    "frozen_encoder",
]
_QUICK_REPRESENTATIONS = ["learned_all", "frozen_item_embeddings"]

_METRIC_KEYS = (
    "acc",
    "qwk",
    "auc",
    "theta_spearman",
    "a_spearman",
    "a_high_spearman",
    "b_spearman",
    "alpha_p50",
    "alpha_p95",
    "alpha_max",
    "alpha_grad_norm_mean",
    "decoder_grad_norm_mean",
    "total_grad_norm_mean",
    "final_loss",
    "train_time",
    "n_params",
    "n_trainable_params",
)


def _normalise(name: str) -> str:
    return name.lower().replace("-", "_")


def _set_requires_grad(module: torch.nn.Module, requires_grad: bool) -> list[str]:
    names = []
    for name, param in module.named_parameters():
        param.requires_grad_(requires_grad)
        names.append(name)
    return names


def _model_parameters(model):
    yield from model.encoder.parameters()
    yield from model.decoder.parameters()


def apply_representation_control(model, control: str) -> dict:
    """Freeze selected neural representation pieces for the isolation test."""
    control = _normalise(control)
    if control not in _REPRESENTATIONS:
        raise ValueError(
            f"unknown representation control {control!r}; "
            f"choices are {sorted(_REPRESENTATIONS)}"
        )

    enc = model.encoder
    frozen: list[str] = []

    if control in ("frozen_backbone", "frozen_encoder"):
        for attr in ("resp_emb", "lstm", "theta_proj"):
            if hasattr(enc, attr):
                names = _set_requires_grad(getattr(enc, attr), False)
                frozen.extend(f"encoder.{attr}.{name}" for name in names)

    if control in ("frozen_item_embeddings", "frozen_encoder"):
        for attr in ("item_val_emb", "item_key_emb"):
            if hasattr(enc, attr):
                names = _set_requires_grad(getattr(enc, attr), False)
                frozen.extend(f"encoder.{attr}.{name}" for name in names)

    params = list(_model_parameters(model))
    trainable = sum(p.numel() for p in params if p.requires_grad)
    total = sum(p.numel() for p in params)
    return {
        "representation_control": control,
        "frozen_parameter_names": frozen,
        "n_frozen_tensors": len(frozen),
        "n_trainable_params": int(trainable),
        "n_total_params": int(total),
    }


def _resolve_map_specs(names: list[str] | None, quick: bool) -> dict[str, MapSpec]:
    names = names or (_QUICK_MAPS if quick else _DEFAULT_MAPS)
    out = {}
    expanded = []
    for raw_name in names:
        if _normalise(raw_name) == "all":
            expanded.extend(_DEFAULT_MAPS)
        else:
            expanded.append(raw_name)
    for raw_name in expanded:
        name = _normalise(raw_name)
        if name == "raw":
            name = "identity"
        if name not in _MAP_SPECS:
            raise ValueError(
                f"unknown map {raw_name!r}; choices are {sorted(_MAP_SPECS)} or all"
            )
        out[name] = _MAP_SPECS[name]
    return out


def _resolve_representations(names: list[str] | None, quick: bool) -> list[str]:
    names = names or (_QUICK_REPRESENTATIONS if quick else _DEFAULT_REPRESENTATIONS)
    out = []
    for raw_name in names:
        if _normalise(raw_name) == "all":
            out.extend(_DEFAULT_REPRESENTATIONS)
            continue
        name = _normalise(raw_name)
        if name not in _REPRESENTATIONS:
            raise ValueError(
                f"unknown representation {raw_name!r}; "
                f"choices are {sorted(_REPRESENTATIONS)} or all"
            )
        out.append(name)
    return list(dict.fromkeys(out))


def _build_engine(ds, spec: MapSpec, device: str, seed: int):
    return DeepIRTEngine(
        ds,
        decoder="gpcm",
        state_alpha=True,
        item_key_dim=_ITEM_KEY_DIM,
        alpha_pos_map=spec.name,
        alpha_pos_kwargs=spec.kwargs,
        device=device,
        seed=seed,
        **_CHEAP,
    )


def aggregate_rows(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[
            (
                row["K"],
                row["representation_control"],
                row["map"],
                row["lr"],
                row["grad_clip_norm"],
            )
        ].append(row)

    out = []
    for (K, rep, map_name, lr, clip), rs in sorted(groups.items()):
        entry = {
            "K": K,
            "representation_control": rep,
            "map": map_name,
            "lr": lr,
            "grad_clip_norm": clip,
            "map_kwargs": rs[0]["map_kwargs"],
            "n_seeds": len(rs),
            "n_params": rs[0]["n_params"],
            "n_trainable_params": rs[0]["n_trainable_params"],
        }
        for key in _METRIC_KEYS:
            vals = np.array([r.get(key, float("nan")) for r in rs], dtype=float)
            finite = vals[np.isfinite(vals)]
            if finite.size:
                entry[f"{key}_mean"] = float(np.mean(finite))
                entry[f"{key}_std"] = float(np.std(finite))
            else:
                entry[f"{key}_mean"] = float("nan")
                entry[f"{key}_std"] = float("nan")
        out.append(entry)
    return out


def _score(value: float) -> float:
    return value if value == value else -1.0e12


def best_rows(agg: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in agg:
        groups[(row["K"], row["representation_control"], row["map"])].append(row)
    out = []
    for _key, rs in sorted(groups.items()):
        chosen = max(
            rs,
            key=lambda r: (
                _score(r.get("a_spearman_mean", float("nan"))),
                _score(r.get("a_high_spearman_mean", float("nan"))),
                -float(r["lr"]),
            ),
        )
        out.append(dict(chosen))
    return out


def _fmt(value: float, p: int = 3) -> str:
    return "-" if value != value else f"{value:.{p}f}"


def _fmt_ms(row: dict, key: str, p: int = 3) -> str:
    mean = row.get(f"{key}_mean", float("nan"))
    std = row.get(f"{key}_std", float("nan"))
    return "-" if mean != mean else f"{mean:.{p}f}+-{std:.{p}f}"


def _map_label(row: dict) -> str:
    kwargs = row.get("map_kwargs") or {}
    if not kwargs:
        return row["map"]
    bits = ",".join(f"{k}={v:g}" for k, v in sorted(kwargs.items()))
    return f"{row['map']}({bits})"


def render_table(meta: dict, agg: list[dict], best: list[dict]) -> str:
    lines = ["# Neural map-isolation controls\n"]
    lines.append(
        f"mode={meta['mode']} device={meta['device']} epochs={meta['epochs']} "
        f"N={meta['N']} Q={meta['Q']} T={meta['T']} K={meta['K_vals']} "
        f"seeds={meta['seeds']} lrs={meta['lrs']} maps={meta['maps']} "
        f"representations={meta['representations']}\n"
    )
    lines.append(
        "This E7 follow-up keeps the neural LSTM/GPCM model and changes only "
        "representation trainability, alpha map, matched range for raw/ReLU "
        "controls, and default-off gradient clipping.  All rows use matched "
        "effective-alpha initialization.\n"
    )

    for K in meta["K_vals"]:
        rows_k = [r for r in best if r["K"] == K]
        if not rows_k:
            continue
        lines.append(f"\n## K={K} best LR by alpha recovery\n")
        lines.append(
            "| representation | map | clip | trainable params | a_sp | high-a_sp | "
            "theta_sp | b_sp | QWK | alpha p95/max | alpha grad |"
        )
        lines.append("|---|---|---:|---:|---|---|---|---|---|---|---|")
        for row in rows_k:
            clip = row["grad_clip_norm"]
            clip_s = "-" if clip is None else f"{clip:g}"
            lines.append(
                f"| {row['representation_control']} | {_map_label(row)} | "
                f"{clip_s} | {row['n_trainable_params']} | "
                f"{_fmt_ms(row, 'a_spearman')} | "
                f"{_fmt_ms(row, 'a_high_spearman')} | "
                f"{_fmt_ms(row, 'theta_spearman')} | "
                f"{_fmt_ms(row, 'b_spearman')} | {_fmt_ms(row, 'qwk')} | "
                f"{_fmt(row.get('alpha_p95_mean', float('nan')))}/"
                f"{_fmt(row.get('alpha_max_mean', float('nan')))} | "
                f"{_fmt_ms(row, 'alpha_grad_norm_mean')} |"
            )

        lines.append(f"\n## K={K} map deltas within representation\n")
        lines.append("| representation | exp a_sp | softplus a_sp | best raw/ReLU | delta raw/ReLU - softplus |")
        lines.append("|---|---:|---:|---|---:|")
        for rep in meta["representations"]:
            rep_rows = [r for r in rows_k if r["representation_control"] == rep]
            exp = next((r for r in rep_rows if r["map"] == "exp"), None)
            soft = next((r for r in rep_rows if r["map"] == "softplus"), None)
            raw = [
                r for r in rep_rows
                if r["map"] in ("identity", "relu", "clipped_identity", "clipped_relu")
            ]
            best_raw = max(
                raw,
                key=lambda r: _score(r.get("a_spearman_mean", float("nan"))),
            ) if raw else None
            delta = float("nan")
            if best_raw is not None and soft is not None:
                delta = (
                    best_raw["a_spearman_mean"] -
                    soft["a_spearman_mean"]
                )
            lines.append(
                f"| {rep} | "
                f"{_fmt(exp['a_spearman_mean']) if exp else '-'} | "
                f"{_fmt(soft['a_spearman_mean']) if soft else '-'} | "
                f"{best_raw['map'] if best_raw else '-'} "
                f"{_fmt(best_raw['a_spearman_mean']) if best_raw else '-'} | "
                f"{_fmt(delta)} |"
            )

    lines.append("\n## Interpretation Rule\n")
    lines.append(
        "Treat this as a mechanism diagnostic, not a new architecture contest.  "
        "If clipped raw/ReLU plus gradient clipping closes the gap, the earlier "
        "weakness was mostly range or optimizer instability.  If frozen item "
        "embeddings erase the gap, the mechanism is item-representation learning.  "
        "If frozen backbone changes little but frozen item embeddings matter, "
        "the state encoder is less central than the item-key pathway."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--K", dest="k_vals", type=int, nargs="+", default=None)
    ap.add_argument("--maps", type=str, nargs="+", default=None)
    ap.add_argument("--representations", type=str, nargs="+", default=None)
    ap.add_argument("--lrs", type=float, nargs="+", default=None)
    ap.add_argument("--n-learners", type=int, default=None)
    ap.add_argument("--n-items", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--alpha-init", type=float, default=_ALPHA_INIT)
    ap.add_argument("--out-prefix", type=str, default=None)
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if args.quick:
        mode = "QUICK"
        K_vals = args.k_vals or [4]
        seeds = args.seeds or _QUICK_SEEDS
        lrs = args.lrs or _QUICK_LRS
        epochs = args.epochs or 25
        N = args.n_learners or 120
        Q = args.n_items or 24
        T = args.seq_len or 24
    else:
        mode = "FULL"
        K_vals = args.k_vals or [4]
        seeds = args.seeds or _DEFAULT_SEEDS
        lrs = args.lrs or _DEFAULT_LRS
        epochs = args.epochs or 150
        N = args.n_learners or 800
        Q = args.n_items or 60
        T = args.seq_len or 60

    map_specs = _resolve_map_specs(args.maps, quick=args.quick)
    reps = _resolve_representations(args.representations, quick=args.quick)
    prefix = args.out_prefix or (
        "neural_map_isolation_quick" if args.quick else "neural_map_isolation"
    )

    total_fits = len(K_vals) * len(seeds) * len(map_specs) * len(reps) * len(lrs)
    print(
        f"=== neural-map isolation [{mode}]: device={device} K={K_vals} "
        f"epochs={epochs} N={N} Q={Q} T={T} seeds={seeds} "
        f"maps={list(map_specs)} reps={reps} lrs={lrs} fits={total_fits} ==="
    )

    t0 = time.time()
    rows = []
    for K in K_vals:
        dsname = f"static_k{K}"
        for seed in seeds:
            cfg = BenchDataConfig(
                name=dsname,
                kind="static",
                n_cats=K,
                n_learners=N,
                n_items=Q,
                seq_len=T,
                n_holdout=min(10, T),
                seed=seed,
            )
            ds = generate(cfg)
            print(f"\n--- K={K} seed={seed} dataset={dsname} ---")
            for rep in reps:
                for map_name, spec in map_specs.items():
                    for lr in lrs:
                        eng = _build_engine(ds, spec, device, seed)
                        init_info = set_matched_alpha_init(eng.model, args.alpha_init)
                        freeze_info = apply_representation_control(eng.model, rep)
                        monitor = FitMonitor()
                        eng.fit(
                            n_epochs=epochs,
                            lr=lr,
                            batch_size=args.batch_size,
                            callback=monitor,
                            grad_clip_norm=spec.grad_clip_norm,
                        )
                        cell = run_map_cell(eng, ds)
                        cell.update(init_info)
                        cell.update(freeze_info)
                        cell.update(monitor.summary())
                        cell["K"] = K
                        cell["seed"] = seed
                        cell["map"] = map_name
                        cell["map_kwargs"] = dict(spec.kwargs)
                        cell["lr"] = float(lr)
                        cell["grad_clip_norm"] = spec.grad_clip_norm
                        cell["final_loss"] = cell["loss_final"]
                        rows.append(cell)
                        clip_s = "-" if spec.grad_clip_norm is None else spec.grad_clip_norm
                        print(
                            f"  {rep:24s} {map_name:18s} lr={lr:.4g} "
                            f"clip={clip_s} a={cell['a_spearman']:.3f} "
                            f"high-a={cell['a_high_spearman']:.3f} "
                            f"theta={cell['theta_spearman']:.3f} "
                            f"b={cell['b_spearman']:.3f} "
                            f"alpha95={cell['alpha_p95']:.3f} "
                            f"t={cell['train_time']:.1f}s"
                        )

    agg = aggregate_rows(rows)
    best = best_rows(agg)
    meta = {
        "mode": mode,
        "device": device,
        "epochs": epochs,
        "N": N,
        "Q": Q,
        "T": T,
        "K_vals": K_vals,
        "seeds": seeds,
        "lrs": lrs,
        "maps": list(map_specs),
        "map_specs": {
            name: {
                "map": spec.name,
                "kwargs": spec.kwargs,
                "grad_clip_norm": spec.grad_clip_norm,
            }
            for name, spec in map_specs.items()
        },
        "representations": reps,
        "alpha_init": args.alpha_init,
        "batch_size": args.batch_size,
        "cheap": _CHEAP,
        "item_key_dim": _ITEM_KEY_DIM,
        "total_fits": total_fits,
        "elapsed_s": time.time() - t0,
        "torch_version": torch.__version__,
    }

    out_json = OUT / f"{prefix}_results.json"
    out_md = OUT / f"{prefix}_table.md"
    blob = {"meta": meta, "rows": rows, "agg": agg, "best": best}
    out_json.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    out_md.write_text(render_table(meta, agg, best), encoding="utf-8")

    print(f"\nwrote {out_json}")
    print(f"wrote {out_md}")
    print(f"elapsed {meta['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
