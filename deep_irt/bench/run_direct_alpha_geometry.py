"""run_direct_alpha_geometry.py -- direct alpha-space optimizer control.

This is the E7 follow-up after the controlled positive-map ablation.  It is not
a neural model.  It freezes the true theta and beta from the synthetic GPCM data
and optimizes only one alpha_j per item from the response-level GPCM likelihood.

The simplification is deliberate:

  * no encoder,
  * no item embeddings,
  * no beta learning,
  * no theta learning,
  * no parameter-recovery loss,
  * no latent sign alignment when scoring alpha.

It asks whether alpha-space preconditioners and constraints alone reproduce the
smooth-map cluster seen in the neural positive-map ablation.

Outputs:
  deep_irt/bench/outputs/direct_alpha_geometry_results.json
  deep_irt/bench/outputs/direct_alpha_geometry_table.md
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deep_irt.bench import metrics_bench as M
from deep_irt.bench.datagen import BenchDataConfig, generate

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "outputs"
OUT.mkdir(exist_ok=True)

_DEFAULT_ALPHA_INIT = 0.5
_DEFAULT_K = [2, 4, 8]
_DEFAULT_SEEDS = [0, 1, 2, 3, 4]
_DEFAULT_LRS = [0.03, 0.1, 0.3, 1.0]
_QUICK_K = [4]
_QUICK_SEEDS = [0]
_QUICK_LRS = [0.1]

_CONDITIONS: dict[str, dict[str, Any]] = {
    "unconstrained": {"bounds": "symmetric", "preconditioner": "identity"},
    "projected": {"bounds": "positive", "preconditioner": "identity"},
    "exp_precond": {"bounds": "positive", "preconditioner": "exp"},
    "softplus_precond": {"bounds": "positive", "preconditioner": "softplus"},
    "scaled_softplus_precond": {
        "bounds": "positive",
        "preconditioner": "scaled_softplus",
        "scale": 1.5,
    },
    "temperature_softplus_precond": {
        "bounds": "positive",
        "preconditioner": "temperature_softplus",
        "temperature": 0.5,
    },
    "sigmoid_precond": {"bounds": "unit", "preconditioner": "sigmoid"},
    "scaled_sigmoid_precond": {
        "bounds": "scaled_unit",
        "preconditioner": "scaled_sigmoid",
        "scale": 2.0,
    },
    "square_precond": {"bounds": "positive", "preconditioner": "square"},
}

_METRIC_KEYS = (
    "a_spearman",
    "a_pearson",
    "a_high_spearman",
    "a_high_pearson",
    "final_loss",
    "loss_drop",
    "recovery_auc",
    "time_to_0_80",
    "time_to_0_90",
    "alpha_mean",
    "alpha_std",
    "alpha_min",
    "alpha_p05",
    "alpha_p50",
    "alpha_p95",
    "alpha_max",
    "grad_norm_mean",
    "grad_norm_final",
    "update_norm_mean",
    "hit_lower_frac",
    "hit_upper_frac",
    "train_time",
)


# ---------------------------------------------------------------------------
# Tensor preparation and likelihood
# ---------------------------------------------------------------------------

def flatten_training_observations(ds, device: torch.device) -> dict[str, torch.Tensor]:
    idx = ds.train_idx
    item = torch.tensor(ds.items0[idx].reshape(-1), dtype=torch.long, device=device)
    response = torch.tensor(
        ds.responses[idx].reshape(-1), dtype=torch.long, device=device
    )
    theta = torch.tensor(
        ds.theta_at_step[idx].reshape(-1), dtype=torch.float32, device=device
    )
    beta = torch.tensor(ds.gt.b, dtype=torch.float32, device=device)
    alpha_true = torch.tensor(ds.gt.a, dtype=torch.float32, device=device)
    return {
        "item": item,
        "response": response,
        "theta": theta,
        "beta": beta,
        "alpha_true": alpha_true,
    }


def gpcm_logits_from_alpha(
    alpha_by_item: torch.Tensor,
    item: torch.Tensor,
    theta: torch.Tensor,
    beta_by_item: torch.Tensor,
) -> torch.Tensor:
    """GPCM logits for flattened observations with true theta and beta."""
    alpha = alpha_by_item[item]
    beta = beta_by_item[item]
    step = alpha[:, None] * (theta[:, None] - beta)
    tail = torch.cumsum(step, dim=1)
    zero = torch.zeros((tail.shape[0], 1), dtype=tail.dtype, device=tail.device)
    return torch.cat([zero, tail], dim=1)


def direct_alpha_loss(alpha_by_item: torch.Tensor, obs: dict[str, torch.Tensor]):
    logits = gpcm_logits_from_alpha(
        alpha_by_item, obs["item"], obs["theta"], obs["beta"]
    )
    return torch.nn.functional.cross_entropy(logits, obs["response"])


# ---------------------------------------------------------------------------
# Alpha-space update rules
# ---------------------------------------------------------------------------

def alpha_bounds(spec: dict[str, Any], epsilon: float, alpha_max: float) -> tuple[float, float]:
    bounds = spec["bounds"]
    if bounds == "symmetric":
        return -alpha_max, alpha_max
    if bounds == "positive":
        return epsilon, alpha_max
    if bounds == "unit":
        return epsilon, 1.0 - epsilon
    if bounds == "scaled_unit":
        return epsilon, float(spec.get("scale", 1.0)) - epsilon
    raise ValueError(f"unknown bounds {bounds!r}")


def torch_preconditioner(alpha: torch.Tensor, spec: dict[str, Any]) -> torch.Tensor:
    name = spec["preconditioner"]
    if name == "identity":
        return torch.ones_like(alpha)
    if name == "exp":
        return alpha.square()
    if name == "softplus":
        return (1.0 - torch.exp(-alpha)).square()
    if name == "scaled_softplus":
        scale = float(spec.get("scale", 1.0))
        return (scale * (1.0 - torch.exp(-alpha / scale))).square()
    if name == "temperature_softplus":
        temperature = float(spec.get("temperature", 1.0))
        return (temperature * (1.0 - torch.exp(-alpha))).square()
    if name == "sigmoid":
        return (alpha * (1.0 - alpha)).square()
    if name == "scaled_sigmoid":
        scale = float(spec.get("scale", 1.0))
        return (alpha * (1.0 - alpha / scale)).square()
    if name == "square":
        return 4.0 * torch.clamp(alpha, min=0.0)
    raise ValueError(f"unknown preconditioner {name!r}")


def _array_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float).ravel()
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_p05": float(np.percentile(arr, 5)),
        f"{prefix}_p50": float(np.percentile(arr, 50)),
        f"{prefix}_p95": float(np.percentile(arr, 95)),
        f"{prefix}_max": float(np.max(arr)),
    }


def direct_alpha_recovery(alpha_hat, alpha_true) -> dict[str, float]:
    alpha_hat = np.asarray(alpha_hat, dtype=float).ravel()
    alpha_true = np.asarray(alpha_true, dtype=float).ravel()
    cutoff = float(np.quantile(alpha_true, 0.75))
    high = alpha_true >= cutoff
    return {
        "a_spearman": M._spearman(alpha_hat, alpha_true),
        "a_pearson": M._pearson(alpha_hat, alpha_true),
        "a_high_spearman": M._spearman(alpha_hat[high], alpha_true[high]),
        "a_high_pearson": M._pearson(alpha_hat[high], alpha_true[high]),
    }


def _time_to(history: list[dict[str, float]], threshold: float) -> float:
    for entry in history:
        val = entry.get("a_spearman", float("nan"))
        if val == val and val >= threshold:
            return float(entry["epoch"])
    return float("nan")


def _recovery_auc(history: list[dict[str, float]], epochs: int) -> float:
    if not history:
        return float("nan")
    xs = np.array([h["epoch"] for h in history], dtype=float)
    ys = np.array([h["a_spearman"] for h in history], dtype=float)
    finite = np.isfinite(ys)
    xs = xs[finite]
    ys = ys[finite]
    if len(xs) < 2:
        return float("nan")
    if len(xs) == 1:
        return float(ys[0])
    return float(np.trapezoid(ys, xs) / max(float(epochs), 1.0))


def optimize_alpha_cell(
    obs: dict[str, torch.Tensor],
    condition: str,
    lr: float,
    epochs: int,
    alpha_init: float,
    alpha_max: float,
    epsilon: float,
    checkpoints: list[int],
) -> dict[str, Any]:
    spec = dict(_CONDITIONS[condition])
    lo, hi = alpha_bounds(spec, epsilon=epsilon, alpha_max=alpha_max)
    Q = obs["beta"].shape[0]
    device = obs["beta"].device
    alpha = torch.full((Q,), float(alpha_init), dtype=torch.float32, device=device)
    alpha = torch.clamp(alpha, min=lo, max=hi)
    true_alpha_np = obs["alpha_true"].detach().cpu().numpy()

    grad_norms: list[float] = []
    update_norms: list[float] = []
    history: list[dict[str, float]] = []
    first_loss = float("nan")
    final_loss = float("nan")
    t0 = time.time()

    checkpoint_set = set(checkpoints)
    checkpoint_set.add(0)
    checkpoint_set.add(epochs)

    for epoch in range(0, epochs + 1):
        alpha_var = alpha.detach().clone().requires_grad_(True)
        loss = direct_alpha_loss(alpha_var, obs)
        loss_value = float(loss.item())
        if epoch == 0:
            first_loss = loss_value
        final_loss = loss_value

        if epoch in checkpoint_set:
            alpha_np = alpha.detach().cpu().numpy()
            rec = direct_alpha_recovery(alpha_np, true_alpha_np)
            history.append({"epoch": epoch, "loss": loss_value, **rec})

        if epoch == epochs:
            break

        loss.backward()
        grad = alpha_var.grad.detach()
        pre = torch_preconditioner(alpha.detach(), spec)
        update = pre * grad
        with torch.no_grad():
            alpha = alpha - float(lr) * update
            alpha = torch.clamp(alpha, min=lo, max=hi)
        grad_norms.append(float(torch.linalg.vector_norm(grad).item()))
        update_norms.append(float(torch.linalg.vector_norm(update).item()))

    alpha_np = alpha.detach().cpu().numpy()
    rec = direct_alpha_recovery(alpha_np, true_alpha_np)
    stats = _array_stats("alpha", alpha_np)
    lower_hits = float(np.mean(alpha_np <= lo + 10.0 * epsilon))
    upper_hits = float(np.mean(alpha_np >= hi - 10.0 * epsilon))
    return {
        "condition": condition,
        "lr": float(lr),
        "epochs": int(epochs),
        "alpha_init": float(alpha_init),
        "bounds": spec["bounds"],
        "preconditioner": spec["preconditioner"],
        "condition_kwargs": {
            k: v for k, v in spec.items() if k not in ("bounds", "preconditioner")
        },
        "lower_bound": float(lo),
        "upper_bound": float(hi),
        **rec,
        **stats,
        "final_loss": final_loss,
        "loss_drop": first_loss - final_loss,
        "recovery_auc": _recovery_auc(history, epochs),
        "time_to_0_80": _time_to(history, 0.80),
        "time_to_0_90": _time_to(history, 0.90),
        "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else float("nan"),
        "grad_norm_final": grad_norms[-1] if grad_norms else float("nan"),
        "update_norm_mean": float(np.mean(update_norms)) if update_norms else float("nan"),
        "hit_lower_frac": lower_hits,
        "hit_upper_frac": upper_hits,
        "history": history,
        "train_time": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Aggregation and rendering
# ---------------------------------------------------------------------------

def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["K"], row["condition"], row["lr"])].append(row)
    out = []
    for (K, condition, lr), rs in sorted(groups.items()):
        entry = {
            "K": K,
            "condition": condition,
            "lr": lr,
            "n_seeds": len(rs),
            "bounds": rs[0]["bounds"],
            "preconditioner": rs[0]["preconditioner"],
            "condition_kwargs": rs[0]["condition_kwargs"],
            "lower_bound": rs[0]["lower_bound"],
            "upper_bound": rs[0]["upper_bound"],
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


def best_lr_rows(agg_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in agg_rows:
        groups[(row["K"], row["condition"])].append(row)
    out = []
    for _key, rs in sorted(groups.items()):
        chosen = max(
            rs,
            key=lambda r: (
                _score(r.get("a_spearman_mean")),
                _score(r.get("recovery_auc_mean")),
                -float(r["lr"]),
            ),
        )
        out.append(dict(chosen))
    return out


def _score(value) -> float:
    value = float(value)
    return value if value == value else -1.0e12


def _fmt(value, p: int = 3) -> str:
    value = float(value)
    return "-" if value != value else f"{value:.{p}f}"


def _fmt_ms(row: dict[str, Any], key: str, p: int = 3) -> str:
    mean = row.get(f"{key}_mean", float("nan"))
    std = row.get(f"{key}_std", float("nan"))
    if float(mean) != float(mean):
        return "-"
    return f"{float(mean):.{p}f}+-{float(std):.{p}f}"


def render_table(meta: dict[str, Any], agg_rows: list[dict[str, Any]], best_rows) -> str:
    lines = ["# Direct Alpha Geometry Control\n"]
    lines.append(
        f"mode={meta['mode']}  device={meta['device']}  epochs={meta['epochs']}  "
        f"N={meta['N']}  Q={meta['Q']}  T={meta['T']}  K={meta['K_vals']}  "
        f"seeds={meta['seeds']}  lrs={meta['lrs']}  alpha_init={meta['alpha_init']}\n"
    )
    lines.append(
        "True theta and true beta are frozen.  Only one alpha per item is "
        "optimized from response-level GPCM likelihood.  Alpha recovery is "
        "scored without sign alignment because theta and beta fix the latent "
        "orientation.\n"
    )
    for K in meta["K_vals"]:
        rows = [r for r in agg_rows if r["K"] == K]
        if not rows:
            continue
        lines.append(f"\n## K={K} all LR cells\n")
        lines.append(
            "| condition | lr | a_sp | high-a_sp | rec AUC | t80 | loss drop | "
            "alpha p95 | hits low/high | update norm |"
        )
        lines.append("|---|---:|---|---|---|---:|---:|---:|---:|---:|")
        for row in rows:
            lines.append(
                f"| {row['condition']} | {row['lr']:.4g} | "
                f"{_fmt_ms(row, 'a_spearman')} | "
                f"{_fmt_ms(row, 'a_high_spearman')} | "
                f"{_fmt_ms(row, 'recovery_auc')} | "
                f"{_fmt(row.get('time_to_0_80_mean'))} | "
                f"{_fmt(row.get('loss_drop_mean'))} | "
                f"{_fmt(row.get('alpha_p95_mean'))} | "
                f"{_fmt(row.get('hit_lower_frac_mean'))}/"
                f"{_fmt(row.get('hit_upper_frac_mean'))} | "
                f"{_fmt(row.get('update_norm_mean_mean'))} |"
            )

        best = [r for r in best_rows if r["K"] == K]
        lines.append(f"\n## K={K} best LR by alpha recovery\n")
        lines.append(
            "| condition | best lr | preconditioner | bounds | a_sp | high-a_sp | "
            "rec AUC | alpha p95 | hits low/high |"
        )
        lines.append("|---|---:|---|---|---|---|---|---:|---:|")
        for row in best:
            lines.append(
                f"| {row['condition']} | {row['lr']:.4g} | "
                f"{row['preconditioner']} | {row['bounds']} | "
                f"{_fmt_ms(row, 'a_spearman')} | "
                f"{_fmt_ms(row, 'a_high_spearman')} | "
                f"{_fmt_ms(row, 'recovery_auc')} | "
                f"{_fmt(row.get('alpha_p95_mean'))} | "
                f"{_fmt(row.get('hit_lower_frac_mean'))}/"
                f"{_fmt(row.get('hit_upper_frac_mean'))} |"
            )
    lines.append("\n## Readout\n")
    lines.append(_readout(meta, best_rows))
    return "\n".join(lines) + "\n"


def _readout(meta: dict[str, Any], best_rows: list[dict[str, Any]]) -> str:
    if meta["mode"] == "QUICK":
        return (
            "This is a smoke run.  It checks the frozen-theta/frozen-beta "
            "direct-alpha protocol but should not be interpreted as evidence."
        )
    lines = []
    for K in meta["K_vals"]:
        rows = [r for r in best_rows if r["K"] == K]
        if not rows:
            continue
        best = max(rows, key=lambda r: _score(r.get("a_spearman_mean")))
        projected = [r for r in rows if r["condition"] == "projected"]
        exp = [r for r in rows if r["condition"] == "exp_precond"]
        soft = [r for r in rows if r["condition"] == "softplus_precond"]
        lines.append(
            f"K={K}: best condition is {best['condition']} at "
            f"a_sp={best['a_spearman_mean']:.3f}."
        )
        if projected and exp and soft:
            lines.append(
                f"  projected={projected[0]['a_spearman_mean']:.3f}, "
                f"exp_precond={exp[0]['a_spearman_mean']:.3f}, "
                f"softplus_precond={soft[0]['a_spearman_mean']:.3f}."
            )
    lines.append(
        "Interpretation rule: if direct alpha-space preconditioners do not "
        "reproduce the neural smooth-map cluster, the neural result should be "
        "attributed to representation or optimizer interaction rather than "
        "preconditioner magnitude alone."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_conditions(names: list[str] | None) -> list[str]:
    if names is None:
        return list(_CONDITIONS)
    out = []
    for raw in names:
        name = raw.lower().replace("-", "_")
        if name == "all":
            out.extend(_CONDITIONS)
        elif name in _CONDITIONS:
            out.append(name)
        else:
            raise ValueError(f"unknown condition {raw!r}; choices are {sorted(_CONDITIONS)}")
    return list(dict.fromkeys(out))


def _checkpoint_grid(epochs: int) -> list[int]:
    base = [0, 1, 2, 5, 10, 20, 50, 100, 200, epochs]
    return sorted({x for x in base if 0 <= x <= epochs})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--K", dest="k_vals", type=int, nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--lrs", type=float, nargs="+", default=None)
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--n-learners", type=int, default=None)
    ap.add_argument("--n-items", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--alpha-init", type=float, default=_DEFAULT_ALPHA_INIT)
    ap.add_argument("--alpha-max", type=float, default=20.0)
    ap.add_argument("--epsilon", type=float, default=1e-6)
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    torch_device = torch.device(device)

    if args.quick:
        mode = "QUICK"
        K_vals = args.k_vals or _QUICK_K
        seeds = args.seeds or _QUICK_SEEDS
        lrs = args.lrs or _QUICK_LRS
        epochs = args.epochs or 50
        N = args.n_learners or 200
        Q = args.n_items or 30
        T = args.seq_len or 30
        prefix = args.out_prefix or "direct_alpha_geometry_quick"
    else:
        mode = "FULL"
        K_vals = args.k_vals or _DEFAULT_K
        seeds = args.seeds or _DEFAULT_SEEDS
        lrs = args.lrs or _DEFAULT_LRS
        epochs = args.epochs or 300
        N = args.n_learners or 800
        Q = args.n_items or 60
        T = args.seq_len or 60
        prefix = args.out_prefix or "direct_alpha_geometry"

    conditions = _resolve_conditions(args.conditions)
    checkpoints = _checkpoint_grid(epochs)
    total_fits = len(K_vals) * len(seeds) * len(conditions) * len(lrs)
    print(
        f"=== direct-alpha geometry [{mode}]: device={device} K={K_vals} "
        f"epochs={epochs} N={N} Q={Q} T={T} seeds={seeds} "
        f"conditions={conditions} lrs={lrs} fits={total_fits} ==="
    )

    rows = []
    t_start = time.time()
    for K in K_vals:
        for seed in seeds:
            cfg = BenchDataConfig(
                name=f"static_k{K}",
                kind="static",
                n_cats=K,
                n_learners=N,
                n_items=Q,
                seq_len=T,
                seed=seed,
            )
            ds = generate(cfg)
            obs = flatten_training_observations(ds, torch_device)
            print(f"\n--- K={K} seed={seed} ---")
            for condition in conditions:
                for lr in lrs:
                    cell = optimize_alpha_cell(
                        obs=obs,
                        condition=condition,
                        lr=lr,
                        epochs=epochs,
                        alpha_init=args.alpha_init,
                        alpha_max=args.alpha_max,
                        epsilon=args.epsilon,
                        checkpoints=checkpoints,
                    )
                    cell["K"] = K
                    cell["seed"] = seed
                    rows.append(cell)
                    print(
                        f"  {condition:30s} lr={lr:.4g} "
                        f"a={cell['a_spearman']:.3f} "
                        f"high-a={cell['a_high_spearman']:.3f} "
                        f"loss_drop={cell['loss_drop']:.3f} "
                        f"hit={cell['hit_lower_frac']:.2f}/"
                        f"{cell['hit_upper_frac']:.2f} "
                        f"t={cell['train_time']:.2f}s"
                    )

    agg_rows = aggregate(rows)
    best_rows = best_lr_rows(agg_rows)
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
        "conditions": conditions,
        "alpha_init": args.alpha_init,
        "alpha_max": args.alpha_max,
        "epsilon": args.epsilon,
        "checkpoints": checkpoints,
        "total_fits": total_fits,
        "torch_version": torch.__version__,
    }
    blob = {"meta": meta, "rows": rows, "agg": agg_rows, "best_lr": best_rows}
    table = render_table(meta, agg_rows, best_rows)

    out_json = OUT / f"{prefix}_results.json"
    out_md = OUT / f"{prefix}_table.md"
    out_json.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    out_md.write_text(table, encoding="utf-8")
    print("\n" + table)
    print(f"total wall {time.time() - t_start:.1f}s")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
