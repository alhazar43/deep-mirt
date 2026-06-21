"""run_alpha_map_bench.py -- controlled positive-map ablation for alpha.

This is the E5/E6 runner for the learning-dynamics study.  It asks whether an
exponential discrimination map has an advantage after the obvious confounds are
removed:

  * same LSTM/GPCM architecture,
  * same state-conditioned alpha head and wide item key,
  * same data, seeds, parameter budget, and training objective,
  * matched effective alpha initialization across maps,
  * per-map learning-rate sweep,
  * recorded alpha-head gradient norms and final effective-alpha ranges.

The runner deliberately reports every LR cell.  Any "best LR" summary is only a
pre-registered view over those rows, not a replacement for the full table.

Usage
-----
  conda run -n research python -m deep_irt.bench.run_alpha_map_bench --quick

Full canonical starting point:
  conda run -n research python -m deep_irt.bench.run_alpha_map_bench \
      --device cuda --K 4 --seeds 0 1 2 3 4

Extensions after K=4 is stable:
  conda run -n research python -m deep_irt.bench.run_alpha_map_bench \
      --device cuda --K 2 4 8 --seeds 0 1 2 3 4

Outputs
-------
  quick: deep_irt/bench/outputs/alpha_map_quick_results.json
         deep_irt/bench/outputs/alpha_map_quick_table.md
  full:  deep_irt/bench/outputs/alpha_map_results.json
         deep_irt/bench/outputs/alpha_map_table.md
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

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
_DEFAULT_LRS = [3e-3, 1e-2, 3e-2]
_QUICK_LRS = [1e-2]
_DEFAULT_SEEDS = [0, 1, 2, 3, 4]
_QUICK_SEEDS = [0]


# Use one representative per map family.  "identity" is the unconstrained raw
# alpha control; the decoder also accepts the alias "raw".
_MAP_SPECS: dict[str, dict[str, float]] = {
    "identity": {},
    "relu": {},
    "softplus": {},
    "softplus_eps": {"epsilon": 1e-3},
    "scaled_softplus": {"scale": 1.5},
    "temperature_softplus": {"temperature": 0.5},
    "sigmoid": {},
    "scaled_sigmoid": {"scale": 2.0},
    "square": {"epsilon": 1e-6},
    "exp": {"scale": 1.0},
    "clipped_exp": {"scale": 1.0, "clip_min": -5.0, "clip_max": 5.0},
}
_DEFAULT_MAPS = list(_MAP_SPECS)
_QUICK_MAPS = ["softplus", "exp", "relu", "sigmoid"]


_METRIC_KEYS = (
    "acc",
    "qwk",
    "auc",
    "theta_spearman",
    "theta_pearson",
    "a_spearman",
    "a_pearson",
    "a_high_spearman",
    "a_high_pearson",
    "b_spearman",
    "b_pearson",
    "alpha_mean",
    "alpha_std",
    "alpha_min",
    "alpha_p05",
    "alpha_p50",
    "alpha_p95",
    "alpha_max",
    "alpha_grad_norm_mean",
    "alpha_grad_norm_max",
    "alpha_grad_norm_final",
    "decoder_grad_norm_mean",
    "total_grad_norm_mean",
    "final_loss",
    "n_params",
    "train_time",
)


# ---------------------------------------------------------------------------
# Matched effective-alpha initialization
# ---------------------------------------------------------------------------

def _normalise_map_name(name: str) -> str:
    return name.lower().replace("-", "_")


def _softplus_inverse(y: float) -> float:
    if y <= 0.0:
        raise ValueError("softplus inverse requires y > 0.")
    return math.log(math.expm1(y))


def _logit(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError("logit requires 0 < p < 1.")
    return math.log(p / (1.0 - p))


def raw_alpha_value_for_map(
    alpha_target: float,
    alpha_pos_map: str | None,
    alpha_pos_kwargs: dict | None = None,
    alpha_log_scale: float | None = None,
) -> float:
    """Return a raw head bias whose effective alpha equals ``alpha_target``.

    This is the fairness control for E5/E6.  The alpha-head weights are set to
    zero and the bias is set to this raw value, so every map starts from exactly
    the same effective discrimination for every item and occurrence.
    """
    if not math.isfinite(alpha_target):
        raise ValueError("alpha_target must be finite.")
    if alpha_pos_map is not None and alpha_log_scale is not None:
        raise ValueError("Use alpha_pos_map or alpha_log_scale, not both.")

    kwargs = dict(alpha_pos_kwargs or {})
    if alpha_pos_map is None:
        name = "softplus" if alpha_log_scale is None else "exp"
    else:
        name = _normalise_map_name(alpha_pos_map)

    if name in ("identity", "raw"):
        return float(alpha_target)
    if name == "clipped_identity":
        lo = float(kwargs.get("clip_min", -20.0))
        hi = float(kwargs.get("clip_max", 20.0))
        if not lo < alpha_target < hi:
            raise ValueError(
                "clipped_identity matched init requires "
                "clip_min < alpha_target < clip_max."
            )
        return float(alpha_target)
    if name == "relu":
        if alpha_target <= 0.0:
            raise ValueError("relu matched init requires alpha_target > 0.")
        return float(alpha_target)
    if name == "clipped_relu":
        lo = float(kwargs.get("clip_min", 0.0))
        hi = float(kwargs.get("clip_max", 20.0))
        if not lo < alpha_target < hi:
            raise ValueError(
                "clipped_relu matched init requires "
                "clip_min < alpha_target < clip_max."
            )
        return float(alpha_target)
    if name == "softplus":
        return _softplus_inverse(alpha_target)
    if name == "softplus_eps":
        epsilon = float(kwargs.get("epsilon", 1e-6))
        return _softplus_inverse(alpha_target - epsilon)
    if name == "scaled_softplus":
        scale = float(kwargs.get("scale", 1.0))
        return _softplus_inverse(alpha_target / scale)
    if name == "temperature_softplus":
        temperature = float(kwargs.get("temperature", 1.0))
        return _softplus_inverse(alpha_target) / temperature
    if name == "sigmoid":
        return _logit(alpha_target)
    if name == "scaled_sigmoid":
        scale = float(kwargs.get("scale", 1.0))
        return _logit(alpha_target / scale)
    if name == "square":
        epsilon = float(kwargs.get("epsilon", 1e-6))
        base = alpha_target - epsilon
        if base <= 0.0:
            raise ValueError("square matched init requires alpha_target > epsilon.")
        return math.sqrt(base)
    if name == "exp":
        scale = float(alpha_log_scale) if alpha_log_scale is not None else float(
            kwargs.get("scale", 1.0)
        )
        if alpha_target <= 0.0:
            raise ValueError("exp matched init requires alpha_target > 0.")
        return math.log(alpha_target) / scale
    if name == "clipped_exp":
        scale = float(alpha_log_scale) if alpha_log_scale is not None else float(
            kwargs.get("scale", 1.0)
        )
        if alpha_target <= 0.0:
            raise ValueError("clipped_exp matched init requires alpha_target > 0.")
        z = math.log(alpha_target)
        lo = float(kwargs.get("clip_min", -20.0))
        hi = float(kwargs.get("clip_max", 20.0))
        if not lo < z < hi:
            raise ValueError(
                "alpha_target lies in the clipped_exp plateau; choose a target "
                "with clip_min < log(alpha_target) < clip_max."
            )
        return z / scale

    raise ValueError(f"unknown alpha_pos_map {alpha_pos_map!r}.")


def _base_gpcm_decoder(decoder):
    return decoder._gpcm if hasattr(decoder, "_gpcm") else decoder


def set_matched_alpha_init(model, alpha_target: float = _DEFAULT_ALPHA_INIT) -> dict:
    """Set static and state alpha heads to a matched effective alpha constant."""
    dec = _base_gpcm_decoder(model.decoder)
    if not hasattr(dec, "_alpha_pos"):
        raise ValueError("matched alpha init requires a GPCM-style alpha decoder.")

    raw = raw_alpha_value_for_map(
        alpha_target,
        dec.alpha_pos_map,
        dec.alpha_pos_kwargs,
        dec.alpha_log_scale,
    )
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
            f"matched init failed for {dec.alpha_pos_map}: "
            f"target={alpha_target} actual={actual} raw={raw}"
        )

    return {
        "alpha_init": float(alpha_target),
        "raw_alpha_init": float(raw),
        "alpha_init_check": actual,
        "alpha_pos_map_resolved": dec.alpha_pos_map,
        "initialized_heads": heads,
    }


# ---------------------------------------------------------------------------
# Gradient monitor
# ---------------------------------------------------------------------------

def _grad_norm(named_params: Iterable[tuple[str, torch.nn.Parameter]], pred) -> float:
    total = 0.0
    found = False
    for name, param in named_params:
        if not pred(name, param) or param.grad is None:
            continue
        g = param.grad.detach()
        total += float(torch.sum(g * g).item())
        found = True
    return math.sqrt(total) if found else float("nan")


class FitMonitor:
    """Collect compact gradient and loss summaries through model.fit callback."""

    def __init__(self) -> None:
        self.n = 0
        self.loss_first = float("nan")
        self.loss_final = float("nan")
        self._alpha_sum = 0.0
        self._alpha_max = float("nan")
        self._alpha_final = float("nan")
        self._decoder_sum = 0.0
        self._decoder_max = float("nan")
        self._decoder_final = float("nan")
        self._total_sum = 0.0
        self._total_max = float("nan")
        self._total_final = float("nan")

    def __call__(self, epoch: int, loss_value: float, model) -> None:
        alpha_norm = _grad_norm(
            model.decoder.named_parameters(),
            lambda name, _param: "fc_a" in name,
        )
        decoder_norm = _grad_norm(
            model.decoder.named_parameters(),
            lambda _name, _param: True,
        )
        params = list(model.encoder.named_parameters()) + [
            (f"decoder.{name}", param)
            for name, param in model.decoder.named_parameters()
        ]
        total_norm = _grad_norm(params, lambda _name, _param: True)

        if self.n == 0:
            self.loss_first = float(loss_value)
        self.loss_final = float(loss_value)
        self.n += 1

        self._alpha_final = alpha_norm
        self._decoder_final = decoder_norm
        self._total_final = total_norm
        if alpha_norm == alpha_norm:
            self._alpha_sum += alpha_norm
            self._alpha_max = (
                alpha_norm if self._alpha_max != self._alpha_max
                else max(self._alpha_max, alpha_norm)
            )
        if decoder_norm == decoder_norm:
            self._decoder_sum += decoder_norm
            self._decoder_max = (
                decoder_norm if self._decoder_max != self._decoder_max
                else max(self._decoder_max, decoder_norm)
            )
        if total_norm == total_norm:
            self._total_sum += total_norm
            self._total_max = (
                total_norm if self._total_max != self._total_max
                else max(self._total_max, total_norm)
            )

    def summary(self) -> dict:
        denom = max(self.n, 1)
        return {
            "monitor_epochs": self.n,
            "loss_first": self.loss_first,
            "loss_final": self.loss_final,
            "alpha_grad_norm_mean": self._alpha_sum / denom,
            "alpha_grad_norm_max": self._alpha_max,
            "alpha_grad_norm_final": self._alpha_final,
            "decoder_grad_norm_mean": self._decoder_sum / denom,
            "decoder_grad_norm_max": self._decoder_max,
            "decoder_grad_norm_final": self._decoder_final,
            "total_grad_norm_mean": self._total_sum / denom,
            "total_grad_norm_max": self._total_max,
            "total_grad_norm_final": self._total_final,
        }


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _array_stats(prefix: str, values) -> dict:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_p05": float("nan"),
            f"{prefix}_p50": float("nan"),
            f"{prefix}_p95": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_mean": float(np.nanmean(arr)),
        f"{prefix}_std": float(np.nanstd(arr)),
        f"{prefix}_min": float(np.nanmin(arr)),
        f"{prefix}_p05": float(np.nanpercentile(arr, 5)),
        f"{prefix}_p50": float(np.nanpercentile(arr, 50)),
        f"{prefix}_p95": float(np.nanpercentile(arr, 95)),
        f"{prefix}_max": float(np.nanmax(arr)),
    }


def _high_alpha_recovery(a_hat, a_true, seen=None, top_frac: float = 0.25) -> dict:
    a_hat = np.asarray(a_hat, dtype=float).ravel()
    a_true = np.asarray(a_true, dtype=float).ravel()
    mask = np.ones_like(a_true, dtype=bool)
    if seen is not None:
        mask &= np.asarray(seen, dtype=bool).ravel()
    if mask.sum() < 4:
        return {"a_high_spearman": float("nan"), "a_high_pearson": float("nan")}

    ah = M._align_sign(a_hat[mask], a_true[mask])
    at = a_true[mask]
    cutoff = float(np.nanquantile(at, 1.0 - top_frac))
    high = at >= cutoff
    if high.sum() < 3:
        return {"a_high_spearman": float("nan"), "a_high_pearson": float("nan")}
    return {
        "a_high_spearman": M._spearman(ah[high], at[high]),
        "a_high_pearson": M._pearson(ah[high], at[high]),
    }


def run_map_cell(engine, ds) -> dict:
    probs, targ = engine.predict_heldout()
    pred = M.prediction_metrics(probs, targ, ds.cfg.n_cats)

    rec = engine.recover()
    seen = rec.get("seen", None)
    item = M.item_recovery(rec["a"], rec["b"], ds.gt.a, ds.gt.b, seen=seen)
    high = _high_alpha_recovery(rec["a"], ds.gt.a, seen=seen)

    static = M.theta_recovery_static(rec["theta_final"], ds.gt.theta0)
    if ds.cfg.kind == "dynamic":
        dyn = M.theta_recovery_dynamic(rec["theta_traj"], ds.gt.theta_traj)
    else:
        dyn = {
            "theta_netdrift_spearman": float("nan"),
            "theta_netdrift_pearson": float("nan"),
            "theta_pooled_pearson": float("nan"),
        }

    a_seen = np.asarray(rec["a"], dtype=float)
    if seen is not None:
        a_seen = a_seen[np.asarray(seen, dtype=bool)]
    alpha_stats = _array_stats("alpha", a_seen)

    cost = engine.cost()
    return {
        "label": engine.label,
        "encoder": engine.encoder_name,
        "dataset": ds.cfg.name,
        "kind": ds.cfg.kind,
        "n_cats": ds.cfg.n_cats,
        **pred,
        **static,
        **dyn,
        **item,
        **high,
        **alpha_stats,
        "n_params": cost["n_params"],
        "train_time": cost["train_time"],
    }


def aggregate_rows(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["K"], row["map"], row["lr"])].append(row)

    out = []
    for (K, map_name, lr), rs in sorted(groups.items()):
        entry = {
            "K": K,
            "map": map_name,
            "lr": lr,
            "dataset": rs[0]["dataset"],
            "n_cats": rs[0]["n_cats"],
            "n_seeds": len(rs),
            "map_kwargs": rs[0]["map_kwargs"],
            "alpha_init": rs[0]["alpha_init"],
            "raw_alpha_init": rs[0]["raw_alpha_init"],
            "n_params": rs[0]["n_params"],
        }
        for key in _METRIC_KEYS:
            vals = np.array([r.get(key, float("nan")) for r in rs], dtype=float)
            entry[f"{key}_mean"] = float(np.nanmean(vals)) if vals.size else float("nan")
            entry[f"{key}_std"] = float(np.nanstd(vals)) if vals.size else float("nan")
        out.append(entry)
    return out


def _nan_score(value: float) -> float:
    return value if value == value else -1.0e12


def best_lr_entries(agg_rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in agg_rows:
        groups[(row["K"], row["map"])].append(row)

    best = []
    for key, rows in sorted(groups.items()):
        chosen = max(
            rows,
            key=lambda r: (
                _nan_score(r.get("a_spearman_mean", float("nan"))),
                _nan_score(r.get("a_high_spearman_mean", float("nan"))),
                _nan_score(r.get("qwk_mean", float("nan"))),
                -float(r["lr"]),
            ),
        )
        best.append(dict(chosen))
    return best


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fmt_ms(entry: dict, key: str, p: int = 3) -> str:
    m = entry.get(f"{key}_mean", float("nan"))
    s = entry.get(f"{key}_std", float("nan"))
    if m != m:
        return "  -  "
    return f"{m:.{p}f}+-{s:.{p}f}"


def _fmt(value: float, p: int = 3) -> str:
    return "  -  " if value != value else f"{value:.{p}f}"


def _map_label(name: str, kwargs: dict) -> str:
    if not kwargs:
        return name
    bits = ",".join(f"{k}={v:g}" for k, v in sorted(kwargs.items()))
    return f"{name}({bits})"


def render_table(meta: dict, agg_rows: list[dict], best_rows: list[dict]) -> str:
    mode = meta["mode"]
    L = []
    L.append("# Alpha positive-map ablation\n")
    L.append(
        f"mode={mode}  device={meta['device']}  epochs={meta['epochs']}  "
        f"N={meta['N']}  Q={meta['Q']}  T={meta['T']}  K={meta['K_vals']}  "
        f"seeds={meta['seeds']}  lrs={meta['lrs']}  "
        f"alpha_init={meta['alpha_init']}  (torch {torch.__version__})\n"
    )
    L.append(
        "All cells use the same LSTM/GPCM architecture: emb_dim=8, hidden_dim=32, "
        "state_alpha=True, item_key_dim=64.  The only intended treatment is the "
        "raw-to-effective-alpha map.  Alpha-head weights are zeroed and biases "
        "are set by the inverse map so every condition starts at the same "
        "effective alpha.  The table reports every LR cell; the best-LR view "
        "selects the highest mean alpha Spearman, then high-alpha Spearman, then "
        "QWK.\n"
    )

    for K in meta["K_vals"]:
        rows = [r for r in agg_rows if r["K"] == K]
        if not rows:
            continue
        L.append(f"\n## K={K} all LR cells\n")
        L.append(
            "| map | lr | a_sp | high-a_sp | theta_sp | b_sp | QWK | "
            "alpha p50/p95/max | alpha grad | train s |"
        )
        L.append("|---|---:|---|---|---|---|---|---|---|---|")
        for row in rows:
            label = _map_label(row["map"], row["map_kwargs"])
            p50 = row.get("alpha_p50_mean", float("nan"))
            p95 = row.get("alpha_p95_mean", float("nan"))
            mx = row.get("alpha_max_mean", float("nan"))
            L.append(
                f"| {label} | {row['lr']:.4g} | {_fmt_ms(row, 'a_spearman')} | "
                f"{_fmt_ms(row, 'a_high_spearman')} | "
                f"{_fmt_ms(row, 'theta_spearman')} | {_fmt_ms(row, 'b_spearman')} | "
                f"{_fmt_ms(row, 'qwk')} | "
                f"{_fmt(p50)}/{_fmt(p95)}/{_fmt(mx)} | "
                f"{_fmt_ms(row, 'alpha_grad_norm_mean')} | "
                f"{_fmt_ms(row, 'train_time', 1)} |"
            )

        best = [r for r in best_rows if r["K"] == K]
        L.append(f"\n## K={K} best LR by alpha recovery\n")
        L.append(
            "| map | best lr | a_sp | high-a_sp | theta_sp | b_sp | QWK | "
            "alpha p95 | alpha grad |"
        )
        L.append("|---|---:|---|---|---|---|---|---|---|")
        for row in best:
            label = _map_label(row["map"], row["map_kwargs"])
            L.append(
                f"| {label} | {row['lr']:.4g} | {_fmt_ms(row, 'a_spearman')} | "
                f"{_fmt_ms(row, 'a_high_spearman')} | "
                f"{_fmt_ms(row, 'theta_spearman')} | {_fmt_ms(row, 'b_spearman')} | "
                f"{_fmt_ms(row, 'qwk')} | "
                f"{_fmt(row.get('alpha_p95_mean', float('nan')))} | "
                f"{_fmt_ms(row, 'alpha_grad_norm_mean')} |"
            )

    L.append("\n" + render_verdict(meta, best_rows))
    return "\n".join(L)


def render_verdict(meta: dict, best_rows: list[dict]) -> str:
    L = ["## Readout\n"]
    if meta["mode"] == "QUICK":
        L.append(
            "This is a smoke run.  It verifies that the matched-init and LR "
            "reporting machinery executes, but it is not evidence for or against "
            "the exp-specific claim."
        )
        return "\n".join(L)

    for K in meta["K_vals"]:
        rows = [r for r in best_rows if r["K"] == K]
        exp = [r for r in rows if r["map"] == "exp"]
        if not exp:
            L.append(f"K={K}: exp was not included, so no exp comparison is available.")
            continue
        exp_row = exp[0]
        exp_family = [r for r in rows if r["map"] in ("exp", "clipped_exp")]
        generic = [r for r in rows if r["map"] not in ("exp", "clipped_exp")]
        best_exp_family = max(
            exp_family,
            key=lambda r: _nan_score(r.get("a_spearman_mean", float("nan"))),
        )
        if generic:
            best_generic = max(
                generic,
                key=lambda r: _nan_score(r.get("a_spearman_mean", float("nan"))),
            )
            delta = exp_row["a_spearman_mean"] - best_generic["a_spearman_mean"]
            high_delta = (
                exp_row["a_high_spearman_mean"] -
                best_generic["a_high_spearman_mean"]
            )
            L.append(
                f"K={K}: exp best-lr a_sp={exp_row['a_spearman_mean']:.3f}; "
                f"best generic non-exp map is {best_generic['map']} at "
                f"{best_generic['a_spearman_mean']:.3f}; "
                f"delta={delta:+.3f}.  High-alpha delta={high_delta:+.3f}."
            )
        else:
            L.append(f"K={K}: no generic non-exp maps were included.")
        if best_exp_family["map"] != "exp":
            family_delta = (
                exp_row["a_spearman_mean"] -
                best_exp_family["a_spearman_mean"]
            )
            L.append(
                f"Within the exp family, best map is {best_exp_family['map']} "
                f"at {best_exp_family['a_spearman_mean']:.3f}; "
                f"exp delta={family_delta:+.3f}."
            )
        clipped = [r for r in exp_family if r["map"] == "clipped_exp"]
        if clipped:
            clip_row = clipped[0]
            clip_delta = exp_row["a_spearman_mean"] - clip_row["a_spearman_mean"]
            L.append(
                f"clipped_exp is an exp-family stability control, not a generic "
                f"non-exp map; exp minus clipped_exp={clip_delta:+.3f}."
            )
    L.append(
        "Interpretation rule: claim an exp advantage only if the full run shows "
        "positive deltas after LR tuning, no theta/beta tradeoff that explains "
        "the gain, and consistent high-alpha recovery."
    )
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_map_specs(names: list[str] | None, quick: bool) -> dict[str, dict]:
    if names is None:
        names = _QUICK_MAPS if quick else _DEFAULT_MAPS
    out = {}
    expanded = []
    for name in names:
        if _normalise_map_name(name) == "all":
            expanded.extend(_DEFAULT_MAPS)
        else:
            expanded.append(name)
    for raw_name in expanded:
        name = _normalise_map_name(raw_name)
        if name == "raw":
            name = "identity"
        if name not in _MAP_SPECS:
            raise ValueError(
                f"unknown map {raw_name!r}; choices are {sorted(_MAP_SPECS)} or all."
            )
        out[name] = dict(_MAP_SPECS[name])
    return out


def _build_engine(ds, map_name: str, kwargs: dict, device: str, seed: int):
    return DeepIRTEngine(
        ds,
        decoder="gpcm",
        state_alpha=True,
        item_key_dim=_ITEM_KEY_DIM,
        alpha_pos_map=map_name,
        alpha_pos_kwargs=kwargs,
        device=device,
        seed=seed,
        **_CHEAP,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--K", dest="k_vals", type=int, nargs="+", default=None)
    ap.add_argument("--maps", type=str, nargs="+", default=None)
    ap.add_argument("--lrs", type=float, nargs="+", default=None)
    ap.add_argument("--n-learners", type=int, default=None)
    ap.add_argument("--n-items", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--alpha-init", type=float, default=_DEFAULT_ALPHA_INIT)
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
        epochs = args.epochs or 40
        N = args.n_learners or 200
        Q = args.n_items or 30
        T = args.seq_len or 30
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
    prefix = args.out_prefix or ("alpha_map_quick" if args.quick else "alpha_map")

    total_fits = len(K_vals) * len(seeds) * len(map_specs) * len(lrs)
    print(
        f"=== alpha-map bench [{mode}]: device={device} K={K_vals} "
        f"epochs={epochs} N={N} Q={Q} T={T} seeds={seeds} "
        f"maps={list(map_specs)} lrs={lrs} fits={total_fits} ==="
    )

    t_start = time.time()
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
            for map_name, kwargs in map_specs.items():
                for lr in lrs:
                    eng = _build_engine(ds, map_name, kwargs, device, seed)
                    init_info = set_matched_alpha_init(eng.model, args.alpha_init)
                    monitor = FitMonitor()
                    eng.fit(
                        n_epochs=epochs,
                        lr=lr,
                        batch_size=args.batch_size,
                        callback=monitor,
                    )
                    cell = run_map_cell(eng, ds)
                    cell.update(init_info)
                    cell.update(monitor.summary())
                    cell["variant_key"] = f"{map_name}@lr{lr:g}"
                    cell["K"] = K
                    cell["seed"] = seed
                    cell["map"] = map_name
                    cell["map_kwargs"] = dict(kwargs)
                    cell["lr"] = float(lr)
                    cell["final_loss"] = cell["loss_final"]
                    rows.append(cell)
                    print(
                        f"  {map_name:20s} lr={lr:.4g} "
                        f"a={cell['a_spearman']:.3f} "
                        f"high-a={cell['a_high_spearman']:.3f} "
                        f"theta={cell['theta_spearman']:.3f} "
                        f"b={cell['b_spearman']:.3f} "
                        f"alpha95={cell['alpha_p95']:.3f} "
                        f"grad={cell['alpha_grad_norm_mean']:.3e} "
                        f"t={cell['train_time']:.1f}s"
                    )

    agg = aggregate_rows(rows)
    best = best_lr_entries(agg)
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
        "map_specs": map_specs,
        "cheap": _CHEAP,
        "item_key_dim": _ITEM_KEY_DIM,
        "alpha_init": args.alpha_init,
        "batch_size": args.batch_size,
        "total_fits": total_fits,
        "torch_version": torch.__version__,
    }

    table = render_table(meta, agg, best)
    blob = {"meta": meta, "rows": rows, "agg": agg, "best_lr": best}

    out_json = OUT / f"{prefix}_results.json"
    out_md = OUT / f"{prefix}_table.md"
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)
    out_md.write_text(table, encoding="utf-8")

    print("\n" + table)
    print(f"\ntotal wall {time.time() - t_start:.1f}s")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
