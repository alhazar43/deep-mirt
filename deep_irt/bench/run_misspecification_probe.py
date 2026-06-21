"""run_misspecification_probe.py -- E9 misspecification probes.

This runner covers controlled E9 violations where responses are generated from
the same known GPCM ground truth as the standard bench, then a single
misspecification is added while the fitted model remains a GPCM.

Implemented misspecifications:

* ``local_dependence``: response t receives an added logit bonus for repeating
  response t-1.
* ``noisy_thresholds``: each response is sampled from occurrence-level jittered
  thresholds while the fitted model assumes static item thresholds.
* ``learner_response_style``: each learner has a stable response style that
  favors extreme versus middle categories independent of item content.
* ``threshold_disorder``: item-level step thresholds are made non-monotone while
  the fitted model still assumes ordered/static thresholds.

The readout asks where the violation goes:

* prediction metrics,
* theta, alpha, and beta recovery,
* contextual log-alpha residual diagnostics from E8,
* contextual beta residual diagnostics for dynamic-beta controls,
* simple correlations with previous response.

Outputs
-------
  deep_irt/bench/outputs/misspec_localdep_results.json
  deep_irt/bench/outputs/misspec_localdep_table.md
  deep_irt/bench/outputs/misspec_noisy_thresholds_results.json
  deep_irt/bench/outputs/misspec_noisy_thresholds_table.md
  deep_irt/bench/outputs/misspec_response_style_results.json
  deep_irt/bench/outputs/misspec_response_style_table.md
  deep_irt/bench/outputs/misspec_threshold_disorder_results.json
  deep_irt/bench/outputs/misspec_threshold_disorder_table.md
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deep_irt.bench.datagen import (
    BenchDataConfig,
    BenchDataset,
    BenchGroundTruth,
    generate,
)
from deep_irt.bench.engines import DeepIRTEngine
from deep_irt.bench.run_alpha_map_bench import (
    FitMonitor,
    run_map_cell,
    set_matched_alpha_init,
)
from deep_irt.bench.run_alpha_residual_null import (
    _pearson,
    _spearman,
    extract_occurrence_params,
    item_demean,
    residual_diagnostics,
)

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "outputs"
OUT.mkdir(exist_ok=True)

_CHEAP = dict(emb_dim=8, hidden_dim=32)
_ITEM_KEY_DIM = 64
_DEFAULT_SEEDS = [0, 1, 2]
_QUICK_SEEDS = [0]
_DEFAULT_STRENGTHS = {
    "local_dependence": [0.0, 0.5, 1.0],
    "learner_response_style": [0.0, 0.5, 1.0],
    "noisy_thresholds": [0.0, 0.25, 0.5],
    "threshold_disorder": [0.0, 0.75, 1.5],
}
_QUICK_STRENGTHS = {
    "local_dependence": [0.0, 1.0],
    "learner_response_style": [0.0, 1.0],
    "noisy_thresholds": [0.0, 0.5],
    "threshold_disorder": [0.0, 1.5],
}
_DEFAULT_VARIANTS = ["static", "state_alpha", "state_beta"]
_QUICK_VARIANTS = ["static", "state_alpha"]

_MISSPEC_SPECS = {
    "local_dependence": {
        "title": "Local Dependence",
        "strength_label": "previous-response logit bonus",
        "description": (
            "Data are generated from the same known GPCM ground truth as the "
            "main bench, except response t receives an additive logit bonus for "
            "repeating response t-1."
        ),
        "default_prefix": "misspec_localdep",
    },
    "learner_response_style": {
        "title": "Learner Response Style",
        "strength_label": "learner style logit scale",
        "description": (
            "Data are generated from the same known GPCM ground truth as the "
            "main bench, except each learner has a stable style score. Positive "
            "style favors extreme categories and negative style favors middle "
            "categories, independent of item content."
        ),
        "default_prefix": "misspec_response_style",
    },
    "noisy_thresholds": {
        "title": "Noisy Thresholds",
        "strength_label": "threshold noise sigma",
        "description": (
            "Data are generated from the same known GPCM ground truth as the "
            "main bench, except each occurrence samples from thresholds with "
            "zero-mean Gaussian jitter before sorting. The fitted model still "
            "assumes static item thresholds."
        ),
        "default_prefix": "misspec_noisy_thresholds",
    },
    "threshold_disorder": {
        "title": "Threshold Disorder",
        "strength_label": "item-threshold disorder scale",
        "description": (
            "Data are generated from the same known GPCM alpha/theta/item "
            "sequences as the main bench, except item threshold vectors receive "
            "fixed item-level perturbations and are not sorted before sampling. "
            "The fitted model still assumes static ordered thresholds."
        ),
        "default_prefix": "misspec_threshold_disorder",
    },
}

_VARIANT_SPECS = {
    "static": {
        "state_alpha": False,
        "state_beta": False,
        "description": "static alpha and static beta",
    },
    "state_alpha": {
        "state_alpha": True,
        "state_beta": False,
        "description": "state-conditioned alpha, static beta",
    },
    "state_beta": {
        "state_alpha": False,
        "state_beta": True,
        "description": "static alpha, state-conditioned beta",
    },
    "state_alpha_beta": {
        "state_alpha": True,
        "state_beta": True,
        "description": "state-conditioned alpha and beta",
    },
}


# ---------------------------------------------------------------------------
# Local-dependence generator
# ---------------------------------------------------------------------------

def gpcm_logits(theta: float, alpha: float, beta: np.ndarray) -> np.ndarray:
    """Return unnormalized GPCM category scores for one response."""
    k = int(len(beta) + 1)
    logits = np.zeros(k, dtype=float)
    for cat in range(1, k):
        logits[cat] = float(np.sum(alpha * (theta - beta[:cat])))
    return logits


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def sample_probs(probs: np.ndarray, u: float) -> int:
    """Sample a categorical outcome using an explicit common random number."""
    cdf = np.cumsum(np.asarray(probs, dtype=float))
    return int(np.searchsorted(cdf, min(float(u), np.nextafter(1.0, 0.0)), side="right"))


def response_style_shape(n_cats: int) -> np.ndarray:
    """Category contrast for extreme-versus-middle response style."""
    if n_cats < 3:
        raise ValueError("response-style misspecification requires n_cats >= 3.")
    cats = np.arange(n_cats, dtype=float)
    center = 0.5 * (n_cats - 1)
    shape = np.abs(cats - center)
    shape = shape - np.mean(shape)
    scale = np.max(np.abs(shape))
    if scale <= 0.0:
        raise ValueError("degenerate response-style category contrast.")
    return shape / scale


def threshold_disorder_summary(beta: np.ndarray) -> tuple[float, float]:
    """Return item disorder rate and mean violation magnitude for thresholds."""
    beta = np.asarray(beta, dtype=float)
    if beta.ndim != 2 or beta.shape[1] < 2:
        return 0.0, 0.0
    diffs = np.diff(beta, axis=1)
    violations = np.maximum(0.0, -diffs)
    per_item = np.sum(violations, axis=1)
    return float(np.mean(per_item > 0.0)), float(np.mean(per_item))


def generate_local_dependence_dataset(
    cfg: BenchDataConfig,
    strength: float,
    name: str | None = None,
) -> BenchDataset:
    """Generate a matched GPCM dataset with previous-response persistence.

    ``strength`` is an additive logit bonus on the previous response category.
    At ``strength=0`` the data are correctly specified GPCM responses, sampled
    with the same explicit random-number stream used for the misspecified
    strengths.  Across strengths at the same seed, ground truth, item sequence,
    train/validation split, and response-draw uniforms are held fixed.
    """
    if strength < 0.0:
        raise ValueError("local-dependence strength must be nonnegative.")

    base = generate(cfg)
    rng = np.random.default_rng(cfg.seed + 20011)
    responses = np.zeros_like(base.responses)
    uniforms = rng.random(size=base.responses.shape)
    gamma = base.gt.gamma

    for learner in range(cfg.n_learners):
        for step in range(cfg.seq_len):
            item = int(base.items0[learner, step])
            theta = float(base.theta_at_step[learner, step])
            alpha = float(base.gt.a[item])
            if gamma is not None:
                alpha *= math.exp(float(gamma[item]) * theta)
            logits = gpcm_logits(theta, alpha, base.gt.b[item])
            if step > 0 and strength > 0.0:
                logits[int(responses[learner, step - 1])] += float(strength)
            responses[learner, step] = sample_probs(softmax(logits), uniforms[learner, step])

    cfg_name = name or f"{cfg.name}_localdep_{strength:g}"
    miss_cfg = replace(cfg, name=cfg_name)
    gt = BenchGroundTruth(
        theta0=base.gt.theta0,
        a=base.gt.a,
        b=base.gt.b,
        theta_traj=base.gt.theta_traj,
        gamma=base.gt.gamma,
    )
    return BenchDataset(
        cfg=miss_cfg,
        gt=gt,
        items0=base.items0,
        responses=responses,
        train_idx=base.train_idx,
        val_idx=base.val_idx,
        theta_at_step=base.theta_at_step,
    )


def generate_learner_response_style_dataset(
    cfg: BenchDataConfig,
    strength: float,
    name: str | None = None,
) -> BenchDataset:
    """Generate matched GPCM data with learner-level response style.

    ``strength`` scales a learner-specific logit contrast that is independent of
    item content and true ability.  Positive learner scores favor extreme
    categories; negative learner scores favor middle categories.  The fitted
    GPCM has no explicit learner response-style parameter, so this probes
    whether theta, alpha, beta, or contextual residuals absorb the nuisance.
    """
    if strength < 0.0:
        raise ValueError("response-style strength must be nonnegative.")
    if cfg.n_cats < 3:
        raise ValueError("response-style misspecification requires n_cats >= 3.")

    base = generate(cfg)
    rng = np.random.default_rng(cfg.seed + 40011)
    uniforms = rng.random(size=base.responses.shape)
    learner_style = rng.standard_normal(cfg.n_learners)
    shape = response_style_shape(cfg.n_cats)
    responses = np.zeros_like(base.responses)
    gamma = base.gt.gamma

    for learner in range(cfg.n_learners):
        style_shift = float(strength * learner_style[learner]) * shape
        for step in range(cfg.seq_len):
            item = int(base.items0[learner, step])
            theta = float(base.theta_at_step[learner, step])
            alpha = float(base.gt.a[item])
            if gamma is not None:
                alpha *= math.exp(float(gamma[item]) * theta)
            logits = gpcm_logits(theta, alpha, base.gt.b[item]) + style_shift
            responses[learner, step] = sample_probs(softmax(logits), uniforms[learner, step])

    cfg_name = name or f"{cfg.name}_response_style_{strength:g}"
    miss_cfg = replace(cfg, name=cfg_name)
    gt = BenchGroundTruth(
        theta0=base.gt.theta0,
        a=base.gt.a,
        b=base.gt.b,
        theta_traj=base.gt.theta_traj,
        gamma=base.gt.gamma,
    )
    return BenchDataset(
        cfg=miss_cfg,
        gt=gt,
        items0=base.items0,
        responses=responses,
        train_idx=base.train_idx,
        val_idx=base.val_idx,
        theta_at_step=base.theta_at_step,
        aux={
            "context_variables": {
                "response_style": learner_style,
                "abs_response_style": np.abs(learner_style),
            },
            "response_style_shape": shape,
        },
    )


def generate_threshold_disorder_dataset(
    cfg: BenchDataConfig,
    strength: float,
    name: str | None = None,
) -> BenchDataset:
    """Generate matched GPCM data with item-level non-monotone thresholds.

    ``strength`` scales a fixed item-level perturbation added to the sorted base
    thresholds.  The perturbed thresholds are deliberately not sorted before
    sampling responses, so the data-generating process violates the ordered-step
    threshold assumption.  The fitted model is still scored against the
    disordered generating thresholds to expose which parameters fail.
    """
    if strength < 0.0:
        raise ValueError("threshold-disorder strength must be nonnegative.")

    base = generate(cfg)
    rng = np.random.default_rng(cfg.seed + 50011)
    uniforms = rng.random(size=base.responses.shape)
    perturb = rng.standard_normal(size=base.gt.b.shape)
    beta_eff = base.gt.b + strength * perturb
    disorder_rate, disorder_magnitude = threshold_disorder_summary(beta_eff)
    disorder_per_item = np.sum(
        np.maximum(0.0, -np.diff(beta_eff, axis=1)),
        axis=1,
    ) if beta_eff.shape[1] >= 2 else np.zeros(cfg.n_items, dtype=float)
    responses = np.zeros_like(base.responses)
    gamma = base.gt.gamma

    for learner in range(cfg.n_learners):
        for step in range(cfg.seq_len):
            item = int(base.items0[learner, step])
            theta = float(base.theta_at_step[learner, step])
            alpha = float(base.gt.a[item])
            if gamma is not None:
                alpha *= math.exp(float(gamma[item]) * theta)
            logits = gpcm_logits(theta, alpha, beta_eff[item])
            responses[learner, step] = sample_probs(softmax(logits), uniforms[learner, step])

    cfg_name = name or f"{cfg.name}_threshold_disorder_{strength:g}"
    miss_cfg = replace(cfg, name=cfg_name)
    gt = BenchGroundTruth(
        theta0=base.gt.theta0,
        a=base.gt.a,
        b=beta_eff,
        theta_traj=base.gt.theta_traj,
        gamma=base.gt.gamma,
    )
    return BenchDataset(
        cfg=miss_cfg,
        gt=gt,
        items0=base.items0,
        responses=responses,
        train_idx=base.train_idx,
        val_idx=base.val_idx,
        theta_at_step=base.theta_at_step,
        aux={
            "ordered_b_reference": base.gt.b,
            "threshold_disorder_rate": disorder_rate,
            "threshold_disorder_magnitude_mean": disorder_magnitude,
            "item_variables": {
                "threshold_disorder": (disorder_per_item > 0.0).astype(float),
                "threshold_disorder_magnitude": disorder_per_item,
            },
        },
    )


def generate_noisy_threshold_dataset(
    cfg: BenchDataConfig,
    strength: float,
    name: str | None = None,
) -> BenchDataset:
    """Generate matched GPCM data with occurrence-level threshold jitter.

    ``strength`` is the standard deviation of Gaussian threshold noise.  The
    jittered thresholds are sorted per occurrence, so this is a noisy-threshold
    misspecification rather than a threshold-disorder misspecification.  Across
    strengths at the same seed, ground truth, item sequence, train/validation
    split, response-draw uniforms, and threshold-noise directions are held fixed.
    """
    if strength < 0.0:
        raise ValueError("threshold noise strength must be nonnegative.")

    base = generate(cfg)
    rng = np.random.default_rng(cfg.seed + 30011)
    uniforms = rng.random(size=base.responses.shape)
    noise_basis = rng.standard_normal(
        size=(cfg.n_learners, cfg.seq_len, cfg.n_cats - 1)
    )
    responses = np.zeros_like(base.responses)
    gamma = base.gt.gamma

    for learner in range(cfg.n_learners):
        for step in range(cfg.seq_len):
            item = int(base.items0[learner, step])
            theta = float(base.theta_at_step[learner, step])
            alpha = float(base.gt.a[item])
            if gamma is not None:
                alpha *= math.exp(float(gamma[item]) * theta)
            beta = np.sort(base.gt.b[item] + strength * noise_basis[learner, step])
            logits = gpcm_logits(theta, alpha, beta)
            responses[learner, step] = sample_probs(softmax(logits), uniforms[learner, step])

    cfg_name = name or f"{cfg.name}_noisy_thresholds_{strength:g}"
    miss_cfg = replace(cfg, name=cfg_name)
    gt = BenchGroundTruth(
        theta0=base.gt.theta0,
        a=base.gt.a,
        b=base.gt.b,
        theta_traj=base.gt.theta_traj,
        gamma=base.gt.gamma,
    )
    return BenchDataset(
        cfg=miss_cfg,
        gt=gt,
        items0=base.items0,
        responses=responses,
        train_idx=base.train_idx,
        val_idx=base.val_idx,
        theta_at_step=base.theta_at_step,
    )


def generate_misspecified_dataset(
    cfg: BenchDataConfig,
    misspecification: str,
    strength: float,
) -> BenchDataset:
    if misspecification == "local_dependence":
        return generate_local_dependence_dataset(cfg, strength)
    if misspecification == "learner_response_style":
        return generate_learner_response_style_dataset(cfg, strength)
    if misspecification == "noisy_thresholds":
        return generate_noisy_threshold_dataset(cfg, strength)
    if misspecification == "threshold_disorder":
        return generate_threshold_disorder_dataset(cfg, strength)
    raise ValueError(
        f"unknown misspecification {misspecification!r}; "
        f"choices are {sorted(_MISSPEC_SPECS)}"
    )


def response_repeat_rate(responses: np.ndarray) -> float:
    arr = np.asarray(responses)
    if arr.shape[1] < 2:
        return float("nan")
    return float(np.mean(arr[:, 1:] == arr[:, :-1]))


def response_abs_step_change(responses: np.ndarray) -> float:
    arr = np.asarray(responses, dtype=float)
    if arr.shape[1] < 2:
        return float("nan")
    return float(np.mean(np.abs(arr[:, 1:] - arr[:, :-1])))


def extreme_response_rate(responses: np.ndarray, n_cats: int) -> float:
    arr = np.asarray(responses)
    return float(np.mean((arr == 0) | (arr == n_cats - 1)))


# ---------------------------------------------------------------------------
# Context diagnostics
# ---------------------------------------------------------------------------

def _prev_response_flat(ds: BenchDataset) -> np.ndarray:
    prev = np.full(ds.responses.shape, np.nan, dtype=float)
    prev[:, 1:] = ds.responses[:, :-1]
    return prev.ravel()


def _hist_pos_flat(ds: BenchDataset) -> np.ndarray:
    return np.tile(np.arange(ds.cfg.seq_len, dtype=float), ds.cfg.n_learners)


def _aux_context_variables(ds: BenchDataset) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    aux = ds.aux if getattr(ds, "aux", None) else {}
    raw_context = aux.get("context_variables", {})
    for name, values in raw_context.items():
        arr = np.asarray(values, dtype=float)
        if arr.shape == (ds.cfg.n_learners,):
            out[name] = np.repeat(arr, ds.cfg.seq_len)
        elif arr.shape == (ds.cfg.n_learners, ds.cfg.seq_len):
            out[name] = arr.ravel()
        else:
            raise ValueError(
                f"aux context variable {name!r} has shape {arr.shape}; expected "
                f"({ds.cfg.n_learners},) or ({ds.cfg.n_learners}, {ds.cfg.seq_len})."
            )
    raw_items = aux.get("item_variables", {})
    for name, values in raw_items.items():
        arr = np.asarray(values, dtype=float)
        if arr.shape != (ds.cfg.n_items,):
            raise ValueError(
                f"aux item variable {name!r} has shape {arr.shape}; expected "
                f"({ds.cfg.n_items},)."
            )
        out[name] = arr[np.asarray(ds.items0, dtype=np.int64)].ravel()
    return out


def _var_corr(variables: list[dict[str, Any]], name: str, field: str = "pearson") -> float:
    for row in variables:
        if row["variable"] == name:
            return float(row.get(field, float("nan")))
    return float("nan")


def beta_context_diagnostics(ds: BenchDataset, occ: dict[str, np.ndarray]) -> dict[str, float]:
    """Item-demeaned occurrence variation for the beta mean."""
    beta = np.asarray(occ["beta"], dtype=float)
    beta_mean = beta.mean(axis=2).ravel()
    flat_items = np.asarray(ds.items0, dtype=np.int64).ravel()
    _means, delta, _counts = item_demean(beta_mean, flat_items, ds.cfg.n_items)
    prev = _prev_response_flat(ds)
    out = {
        "beta_delta_std": float(np.std(delta)),
        "beta_delta_abs_mean": float(np.mean(np.abs(delta))),
        "beta_delta_prev_response_corr": _pearson(delta, prev),
        "beta_delta_history_pos_corr": _pearson(delta, _hist_pos_flat(ds)),
    }
    for name, values in _aux_context_variables(ds).items():
        out[f"beta_delta_{name}_corr"] = _pearson(delta, values)
    return out


def theta_context_diagnostics(ds: BenchDataset, occ: dict[str, np.ndarray]) -> dict[str, float]:
    theta = np.asarray(occ["theta_model"], dtype=float).ravel()
    out = {
        "theta_prev_response_corr": _pearson(theta, _prev_response_flat(ds)),
        "theta_history_pos_corr": _pearson(theta, _hist_pos_flat(ds)),
    }
    for name, values in _aux_context_variables(ds).items():
        out[f"theta_{name}_corr"] = _pearson(theta, values)
    return out


def contextual_diagnostics(ds: BenchDataset, occ: dict[str, np.ndarray]) -> dict[str, Any]:
    alpha_summary, variables = residual_diagnostics(ds, occ)
    alpha = np.asarray(occ["alpha"], dtype=float)
    flat_items = np.asarray(ds.items0, dtype=np.int64).ravel()
    _means, alpha_delta, _counts = item_demean(
        np.log(np.clip(alpha.ravel(), 1e-8, None)),
        flat_items,
        ds.cfg.n_items,
    )
    for name, values in _aux_context_variables(ds).items():
        pear = _pearson(alpha_delta, values)
        spear = _spearman(alpha_delta, values)
        variables.append({
            "variable": name,
            "pearson": pear,
            "spearman": spear,
            "abs_pearson": abs(pear) if pear == pear else float("nan"),
            "abs_spearman": abs(spear) if spear == spear else float("nan"),
            "cubic_r2": float("nan"),
        })
    beta = beta_context_diagnostics(ds, occ)
    theta = theta_context_diagnostics(ds, occ)
    return {
        **alpha_summary,
        **beta,
        **theta,
        "corr_delta_prev_response": _var_corr(variables, "prev_response"),
        "corr_delta_info_model": _var_corr(variables, "info_model"),
        "corr_delta_theta_model": _var_corr(variables, "theta_model"),
        "corr_delta_history_pos": _var_corr(variables, "history_pos"),
        "corr_delta_exposure_count": _var_corr(variables, "exposure_count"),
        "corr_delta_response_style": _var_corr(variables, "response_style"),
        "corr_delta_abs_response_style": _var_corr(variables, "abs_response_style"),
        "corr_delta_threshold_disorder": _var_corr(variables, "threshold_disorder"),
        "corr_delta_threshold_disorder_magnitude": _var_corr(
            variables, "threshold_disorder_magnitude"
        ),
        "variables": variables,
    }


# ---------------------------------------------------------------------------
# Fitting and aggregation
# ---------------------------------------------------------------------------

def _build_engine(ds: BenchDataset, variant: str, device: str, seed: int) -> DeepIRTEngine:
    if variant not in _VARIANT_SPECS:
        raise ValueError(f"unknown variant {variant!r}; choices are {sorted(_VARIANT_SPECS)}")
    spec = _VARIANT_SPECS[variant]
    return DeepIRTEngine(
        ds,
        decoder="gpcm",
        state_alpha=bool(spec["state_alpha"]),
        state_beta=bool(spec["state_beta"]),
        item_key_dim=_ITEM_KEY_DIM,
        alpha_pos_map="exp",
        alpha_pos_kwargs={"scale": 1.0},
        encoder="lstm",
        device=device,
        seed=seed,
        **_CHEAP,
    )


def fit_cell(
    ds: BenchDataset,
    misspecification: str,
    strength: float,
    variant: str,
    args,
    seed: int,
) -> dict[str, Any]:
    eng = _build_engine(ds, variant, args.device, seed)
    init_info = set_matched_alpha_init(eng.model, args.alpha_init)
    monitor = FitMonitor()
    eng.fit(
        n_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        callback=monitor,
        grad_clip_norm=args.grad_clip_norm,
    )
    score = run_map_cell(eng, ds)
    occ = extract_occurrence_params(eng.model, ds.items0, ds.responses, args.device)
    ctx = contextual_diagnostics(ds, occ)
    row = {
        "misspecification": misspecification,
        "strength": float(strength),
        "variant": variant,
        "variant_description": _VARIANT_SPECS[variant]["description"],
        "seed": int(seed),
        "K": int(ds.cfg.n_cats),
        "N": int(ds.cfg.n_learners),
        "Q": int(ds.cfg.n_items),
        "T": int(ds.cfg.seq_len),
        "response_repeat_rate": response_repeat_rate(ds.responses),
        "extreme_response_rate": extreme_response_rate(ds.responses, ds.cfg.n_cats),
        "response_abs_step_change": response_abs_step_change(ds.responses),
        "threshold_disorder_rate": float(
            ds.aux.get("threshold_disorder_rate", threshold_disorder_summary(ds.gt.b)[0])
        ),
        "threshold_disorder_magnitude_mean": float(
            ds.aux.get(
                "threshold_disorder_magnitude_mean",
                threshold_disorder_summary(ds.gt.b)[1],
            )
        ),
        **score,
        **init_info,
        **monitor.summary(),
        **ctx,
    }
    row["final_loss"] = row["loss_final"]
    return row


def _finite_mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr))


def _mode(values: list[str]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[str(value)] += 1
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0] if counts else "-"


_AGG_KEYS = [
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
    "alpha_p50",
    "alpha_p95",
    "alpha_max",
    "alpha_grad_norm_mean",
    "delta_std",
    "delta_abs_mean",
    "strongest_abs_corr",
    "max_cubic_r2",
    "corr_delta_prev_response",
    "corr_delta_info_model",
    "corr_delta_theta_model",
    "corr_delta_history_pos",
    "corr_delta_exposure_count",
    "corr_delta_response_style",
    "corr_delta_abs_response_style",
    "corr_delta_threshold_disorder",
    "corr_delta_threshold_disorder_magnitude",
    "beta_delta_std",
    "beta_delta_prev_response_corr",
    "beta_delta_history_pos_corr",
    "beta_delta_response_style_corr",
    "beta_delta_abs_response_style_corr",
    "beta_delta_threshold_disorder_corr",
    "beta_delta_threshold_disorder_magnitude_corr",
    "theta_prev_response_corr",
    "theta_history_pos_corr",
    "theta_response_style_corr",
    "theta_abs_response_style_corr",
    "theta_threshold_disorder_corr",
    "theta_threshold_disorder_magnitude_corr",
    "response_repeat_rate",
    "extreme_response_rate",
    "response_abs_step_change",
    "threshold_disorder_rate",
    "threshold_disorder_magnitude_mean",
    "n_params",
    "train_time",
    "final_loss",
]


def aggregate(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[
            (str(row["misspecification"]), float(row["strength"]), str(row["variant"]))
        ].append(row)

    agg: list[dict[str, Any]] = []
    var_agg: list[dict[str, Any]] = []
    for (misspecification, strength, variant), rows in sorted(groups.items()):
        entry: dict[str, Any] = {
            "misspecification": misspecification,
            "strength": float(strength),
            "variant": variant,
            "variant_description": rows[0]["variant_description"],
            "K": rows[0]["K"],
            "N": rows[0]["N"],
            "Q": rows[0]["Q"],
            "T": rows[0]["T"],
            "n_seeds": len(rows),
            "strongest_corr_variable_mode": _mode(
                [r.get("strongest_corr_variable", "-") for r in rows]
            ),
        }
        for key in _AGG_KEYS:
            mean, std = _finite_mean_std([float(r.get(key, float("nan"))) for r in rows])
            entry[f"{key}_mean"] = mean
            entry[f"{key}_std"] = std
        agg.append(entry)

        variable_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            for variable in row.get("variables", []):
                variable_groups[variable["variable"]].append(variable)
        for variable, vs in sorted(variable_groups.items()):
            pear, pear_sd = _finite_mean_std([v["pearson"] for v in vs])
            spear, spear_sd = _finite_mean_std([v["spearman"] for v in vs])
            r2, r2_sd = _finite_mean_std([v["cubic_r2"] for v in vs])
            var_agg.append({
                "misspecification": misspecification,
                "strength": float(strength),
                "variant": variant,
                "variable": variable,
                "pearson_mean": pear,
                "pearson_std": pear_sd,
                "spearman_mean": spear,
                "spearman_std": spear_sd,
                "cubic_r2_mean": r2,
                "cubic_r2_std": r2_sd,
            })
    return agg, var_agg


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fmt(value: Any, p: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    return "-" if not math.isfinite(value) else f"{value:.{p}f}"


def _fmt_ms(row: dict[str, Any], key: str, p: int = 3) -> str:
    mean = row.get(f"{key}_mean", float("nan"))
    std = row.get(f"{key}_std", float("nan"))
    return "-" if not math.isfinite(float(mean)) else f"{mean:.{p}f}+-{std:.{p}f}"


def render_markdown(
    meta: dict[str, Any],
    agg: list[dict[str, Any]],
    var_agg: list[dict[str, Any]],
) -> str:
    spec = _MISSPEC_SPECS[meta["misspecification"]]
    lines = [f"# Misspecification Probe: {spec['title']}\n"]
    lines.append(
        f"mode={meta['mode']} device={meta['device']} K={meta['K']} "
        f"N={meta['N']} Q={meta['Q']} T={meta['T']} "
        f"misspecification={meta['misspecification']} "
        f"strength_label={spec['strength_label']} "
        f"strengths={meta['strengths']} variants={meta['variants']} "
        f"seeds={meta['seeds']} epochs={meta['epochs']} lr={meta['lr']}\n"
    )
    lines.append(
        f"{spec['description']}  This probes where the violation is absorbed.\n"
    )
    lines.append("## Summary\n")
    lines.append(
        "| strength | variant | disorder | repeat | QWK | theta_sp | alpha_sp | beta_sp | "
        "delta std | corr(delta, prev) | corr(delta, info) | corr(delta, style) | "
        "corr(delta, disorder) | beta delta std | corr(beta, prev) | "
        "corr(beta, style) | corr(beta, disorder) | strongest |"
    )
    lines.append(
        "|---:|---|---:|---:|---|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|"
    )
    for row in agg:
        lines.append(
            f"| {row['strength']:.2f} | {row['variant']} | "
            f"{_fmt(row['threshold_disorder_rate_mean'])} | "
            f"{_fmt(row['response_repeat_rate_mean'])} | "
            f"{_fmt_ms(row, 'qwk')} | {_fmt_ms(row, 'theta_spearman')} | "
            f"{_fmt_ms(row, 'a_spearman')} | {_fmt_ms(row, 'b_spearman')} | "
            f"{_fmt_ms(row, 'delta_std')} | "
            f"{_fmt(row['corr_delta_prev_response_mean'])} | "
            f"{_fmt(row['corr_delta_info_model_mean'])} | "
            f"{_fmt(row['corr_delta_response_style_mean'])} | "
            f"{_fmt(row['corr_delta_threshold_disorder_mean'])} | "
            f"{_fmt_ms(row, 'beta_delta_std')} | "
            f"{_fmt(row['beta_delta_prev_response_corr_mean'])} | "
            f"{_fmt(row['beta_delta_response_style_corr_mean'])} | "
            f"{_fmt(row['beta_delta_threshold_disorder_corr_mean'])} | "
            f"{row['strongest_corr_variable_mode']} |"
        )

    focus = {
        "prev_response",
        "theta_model",
        "info_model",
        "history_pos",
        "exposure_count",
        "response_style",
        "abs_response_style",
        "threshold_disorder",
        "threshold_disorder_magnitude",
    }
    lines.append("\n## Alpha Residual Correlations\n")
    lines.append("| strength | variant | variable | Pearson | Spearman | cubic R2 |")
    lines.append("|---:|---|---|---:|---:|---:|")
    for row in var_agg:
        if row["variable"] not in focus:
            continue
        lines.append(
            f"| {row['strength']:.2f} | {row['variant']} | {row['variable']} | "
            f"{_fmt(row['pearson_mean'])} | {_fmt(row['spearman_mean'])} | "
            f"{_fmt(row['cubic_r2_mean'])} |"
        )

    lines.append("\n## Interpretation Rule\n")
    lines.append(
        "Treat an absorption claim as supported only when the misspecification "
        "changes contextual residual structure beyond the strength-zero matched "
        "control, while prediction, recovery, and beta instability are reported "
        "side by side.  A prediction gain without stable recovery is a "
        "measurement-validity warning, not a psychometric success."
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_variants(names: list[str] | None, quick: bool) -> list[str]:
    names = names or (_QUICK_VARIANTS if quick else _DEFAULT_VARIANTS)
    out: list[str] = []
    for raw in names:
        name = raw.lower().replace("-", "_")
        if name == "all":
            out.extend(_VARIANT_SPECS)
        elif name in _VARIANT_SPECS:
            out.append(name)
        else:
            raise ValueError(f"unknown variant {raw!r}; choices are {sorted(_VARIANT_SPECS)}")
    return list(dict.fromkeys(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument(
        "--misspecification",
        choices=sorted(_MISSPEC_SPECS),
        default="local_dependence",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--N", type=int, default=None)
    ap.add_argument("--n-items", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--strengths", type=float, nargs="+", default=None)
    ap.add_argument("--variants", type=str, nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--grad-clip-norm", type=float, default=None)
    ap.add_argument("--alpha-init", type=float, default=0.5)
    ap.add_argument("--out-prefix", type=str, default=None)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    if args.quick:
        mode = "QUICK"
        args.N = args.N or 80
        args.n_items = args.n_items or 12
        args.seq_len = args.seq_len or 12
        args.epochs = args.epochs or 8
        strengths = args.strengths or _QUICK_STRENGTHS[args.misspecification]
        seeds = args.seeds or _QUICK_SEEDS
    else:
        mode = "FULL"
        args.N = args.N or 800
        args.n_items = args.n_items or 60
        args.seq_len = args.seq_len or 60
        args.epochs = args.epochs or 150
        strengths = args.strengths or _DEFAULT_STRENGTHS[args.misspecification]
        seeds = args.seeds or _DEFAULT_SEEDS

    variants = _resolve_variants(args.variants, args.quick)
    total = len(strengths) * len(seeds) * len(variants)
    print(
        f"=== misspec {args.misspecification} [{mode}]: device={args.device} "
        f"K={args.K} N={args.N} Q={args.n_items} T={args.seq_len} "
        f"strengths={strengths} variants={variants} seeds={seeds} "
        f"epochs={args.epochs} fits={total} ==="
    )

    t0 = time.time()
    records: list[dict[str, Any]] = []
    for seed in seeds:
        base_cfg = BenchDataConfig(
            name=f"misspec_{args.misspecification}_k{args.K}_seed{seed}",
            kind="static",
            n_learners=args.N,
            n_items=args.n_items,
            seq_len=args.seq_len,
            n_cats=args.K,
            n_holdout=min(10, args.seq_len),
            seed=seed,
            alpha_theta_slope=0.0,
        )
        datasets = {
            float(strength): generate_misspecified_dataset(
                base_cfg, args.misspecification, float(strength)
            )
            for strength in strengths
        }
        for strength, ds in datasets.items():
            print(
                f"\n--- seed={seed} strength={strength:g} "
                f"repeat={response_repeat_rate(ds.responses):.3f} ---"
            )
            for variant in variants:
                row = fit_cell(ds, args.misspecification, strength, variant, args, seed)
                records.append(row)
                print(
                    f"  {variant:16s} "
                    f"qwk={row['qwk']:.3f} theta={row['theta_spearman']:.3f} "
                    f"a={row['a_spearman']:.3f} b={row['b_spearman']:.3f} "
                    f"delta={row['delta_std']:.3f} "
                    f"corr_d_prev={row['corr_delta_prev_response']:.3f} "
                    f"beta_delta={row['beta_delta_std']:.3f} "
                    f"t={row['train_time']:.1f}s"
                )

    agg, var_agg = aggregate(records)
    meta = {
        "mode": mode,
        "device": args.device,
        "misspecification": args.misspecification,
        "strength_label": _MISSPEC_SPECS[args.misspecification]["strength_label"],
        "K": args.K,
        "N": args.N,
        "Q": args.n_items,
        "T": args.seq_len,
        "strengths": [float(s) for s in strengths],
        "variants": variants,
        "variant_specs": _VARIANT_SPECS,
        "seeds": seeds,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_clip_norm": args.grad_clip_norm,
        "alpha_init": args.alpha_init,
        "elapsed_s": time.time() - t0,
        "torch_version": torch.__version__,
    }
    base_prefix = _MISSPEC_SPECS[args.misspecification]["default_prefix"]
    prefix = args.out_prefix or (f"{base_prefix}_quick" if args.quick else base_prefix)
    out_json = OUT / f"{prefix}_results.json"
    out_md = OUT / f"{prefix}_table.md"
    blob = {"meta": meta, "records": records, "agg": agg, "var_agg": var_agg}
    out_json.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(meta, agg, var_agg), encoding="utf-8")
    print(f"\nwrote {out_json}")
    print(f"wrote {out_md}")
    print(f"elapsed {meta['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
