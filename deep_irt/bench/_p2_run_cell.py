"""
_p2_run_cell.py -- 5-fold learner-CV single-cell executor for Phase 2.

Fill-in of the scaffold (blueprint build order steps 3-5).  The replication unit
is the LOCKED 5-fold LEARNER CV (decision 1): partition the learners into folds,
rotate the held-out fold, fit on the training folds, recover item params on the
training folds, and score val-fold theta + held-out prediction.  Recovery is
reported as the fold-mean +/- bootstrap 95% CI (not a 5-sample significance test,
which decision 1 rules out as underpowered); toggle comparisons are adjudicated
downstream by paired fold-mean difference + bootstrap CI + rank-biserial.

Engine.  Cells build a :class:`_P2Engine` (proper GPCM ordinal CE, a halting
early-stop fit, masked variable-length prediction).  Option A (decision 2) is
realized inside ``_P2Model._make_decoder``: the GPCM/2PL DECOUPLED arms
(``item_key_dim`` set) route the STATIC slope to the wide item key via
``_p2_gpcm_alpha_key``, so building ``_P2Engine`` is all a cell needs.

NRM item recovery.  The NRM path pulls per-option slopes a_k AND intercepts c_k
from ``model.recover_item_params`` and scores them with
``nrm_metrics.item_recovery`` -- KEEPING the intercept, NOT forcing the numbers
through the GPCM-shaped (scalar a, K-1 thresholds) path.  It scores only when the
bed emits NRM ground truth (a_k, c_k); the clean GPCM bed does not, so those
cells report prediction + theta and leave item recovery empty (honest, matching
the base engine).

Reproducibility schema (results.json layout):
  {
    "schema_version": SCHEMA_VERSION,
    "config_sha256": <str>,
    "config": <full resolved P2Config dict>,
    "env": {git_sha, torch/numpy/scipy/sklearn versions, device, determinism},
    "per_seed_rows": [ {data_seed, init_seed, fold, fit_time_s, n_params,
                        <pred metrics>, <theta metrics>, <item metrics>}, ... ],
    "agg": { <metric>: {"mean", "ci_lo", "ci_hi", "n"}, ... }   # fold-mean + CI
  }
"""

from __future__ import annotations

import dataclasses
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from deep_irt.bench._p2_config import P2Config, config_sha256
from deep_irt.bench.datagen import BenchDataConfig, generate
from deep_irt.bench.engines import MaIrtEngine
from deep_irt.bench._p2_engine import _P2Engine
from deep_irt.bench import metrics_bench, nrm_metrics
from deep_irt.bench._p2_aggregate import aggregate_rows, SCORED_METRICS


SCHEMA_VERSION = "p2.1"


# ---------------------------------------------------------------------------
# Seed + determinism
# ---------------------------------------------------------------------------

def _set_seeds(data_seed: int, init_seed: int) -> dict:
    """Fix all RNG states; return a seed-log dict for the results manifest."""
    torch.manual_seed(init_seed)
    np.random.seed(data_seed)
    random.seed(data_seed ^ (init_seed << 16))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {
        "data_seed": data_seed,
        "init_seed": init_seed,
        "torch_seed": init_seed,
        "np_seed": data_seed,
    }


def _env_manifest(device: str) -> dict:
    """Snapshot env versions and determinism flags for provenance."""
    import sys

    manifest: dict = {
        "schema_version": SCHEMA_VERSION,
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": device,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }
    try:
        manifest["git_sha"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"],
                                    stderr=subprocess.DEVNULL)
            .decode().strip()
        )
    except Exception:
        manifest["git_sha"] = None
    try:
        import scipy
        manifest["scipy"] = scipy.__version__
    except Exception:
        pass
    try:
        import sklearn
        manifest["sklearn"] = sklearn.__version__
    except Exception:
        pass
    return manifest


# ---------------------------------------------------------------------------
# Dataset builder (clean bed now; realistic bed routed to a separate build)
# ---------------------------------------------------------------------------

def _generate_dataset(cfg: P2Config, data_seed: int):
    """Build the synthetic dataset for one data seed.

    ``data.regime == "clean"`` (default) -> the rectangular clean-bed
    ``datagen.generate``.  ``"realistic"`` -> the step-6a generator
    ``_p2_datagen_realistic`` (a SEPARATE build; imported lazily so the clean
    path has no dependency on it).
    """
    dc = cfg.data
    if dc.regime == "realistic":
        try:
            from deep_irt.bench import _p2_datagen_realistic as dgr
        except ImportError as exc:  # pragma: no cover - separate build
            raise RuntimeError(
                "data.regime='realistic' requires _p2_datagen_realistic "
                "(step-6a build); it is not present yet."
            ) from exc
        return dgr.generate_realistic(cfg, data_seed)

    bench_cfg = BenchDataConfig(
        name=cfg.report.cell_name,
        kind=dc.kind,
        n_learners=dc.n_learners,
        n_items=dc.n_items,
        seq_len=dc.seq_len,
        n_cats=dc.n_cats,
        drift_sigma=dc.drift_sigma,
        train_frac=dc.train_frac,
        n_holdout=dc.n_holdout,
        seed=data_seed,
        alpha_theta_slope=dc.alpha_theta_slope,
    )
    return generate(bench_cfg)


# ---------------------------------------------------------------------------
# Engine builder
# ---------------------------------------------------------------------------

def _build_engine(cfg: P2Config, ds, init_seed: int, device: str):
    """Instantiate the Phase-2 engine from the model config.

    ``engine == "deep_irt"`` -> :class:`_P2Engine` (the Phase-2 default: proper
    GPCM loss, halting early stop, Option-A decoupled routing, masked prediction).
    ``engine == "ma_irt"`` -> the frozen :class:`MaIrtEngine` (comparison only).
    """
    mc = cfg.model
    if mc.engine == "deep_irt":
        return _P2Engine(
            ds=ds,
            gpcm_loss=cfg.train.gpcm_loss,
            inner_val_frac=cfg.train.inner_val_frac,
            decoder=mc.decoder,
            emb_dim=mc.emb_dim,
            hidden_dim=mc.hidden_dim,
            state_alpha=mc.state_alpha,
            state_beta=mc.state_beta,
            item_key_dim=mc.item_key_dim,
            encoder=mc.encoder,
            device=device,
            seed=init_seed,
        )
    if mc.engine == "ma_irt":
        separate_theta = not mc.coupled_theta
        return MaIrtEngine(
            ds=ds,
            encoder=mc.encoder,
            device=device,
            seed=init_seed,
            separate_theta=separate_theta,
            **mc.ma_irt_kwargs,
        )
    raise ValueError(f"unknown engine: {mc.engine!r}")


# ---------------------------------------------------------------------------
# Gauge fix (honest affine; a no-op for the reported correlation metrics)
# ---------------------------------------------------------------------------

def _apply_gauge(a_hat: np.ndarray, a_true: np.ndarray,
                 b_hat: np.ndarray, b_true: np.ndarray, gauge: str):
    """Apply a global affine gauge map per parameter (slope+intercept OLS).

    ``gauge == "affine"`` fits ONE global (slope, intercept) per parameter
    (discrimination and thresholds separately -- they have different gauges) and
    applies it recovered->true.  Pearson and Spearman are invariant to an affine
    transform of one variable, so this is a NO-OP for the correlation metrics the
    recovery scoring reports; it is kept honest (not a silent pass-through) for
    the downstream magnitude diagnostic (linked-scale slope, RMSE).

    ``gauge == "none"`` returns the arrays unchanged.
    """
    if gauge == "none":
        return a_hat, b_hat

    def _affine(hat, true):
        h = np.asarray(hat, dtype=float)
        hf = h.ravel()
        tf = np.asarray(true, dtype=float).ravel()
        if hf.size < 2 or np.std(hf) < 1e-12 or not np.all(np.isfinite(hf)):
            return h
        slope, intercept = np.polyfit(hf, tf, 1)
        return slope * h + intercept

    return _affine(a_hat, a_true), _affine(b_hat, b_true)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _long(arr) -> torch.Tensor:
    return torch.as_tensor(np.asarray(arr), dtype=torch.long)


def _nrm_ground_truth(ds):
    """Return (a_k, c_k) NRM ground truth if the bed emits it, else (None, None).

    Looked up on ``gt.a_k``/``gt.c_k`` first, then ``ds.aux['nrm_a']``/
    ``ds.aux['nrm_c']`` -- the two places a future NRM generator may emit the
    per-option slopes/intercepts (both (Q, K), mean-centered, the Bock
    convention used by ``nrm_metrics.item_recovery``).
    """
    gt = ds.gt
    a = getattr(gt, "a_k", None)
    c = getattr(gt, "c_k", None)
    if a is None or c is None:
        aux = getattr(ds, "aux", None) or {}
        a = aux.get("nrm_a", a)
        c = aux.get("nrm_c", c)
    return a, c


def _score_theta(cfg: P2Config, ds, theta_traj_val: np.ndarray,
                 val_rows: np.ndarray) -> dict:
    """Theta recovery on the held-out fold learners (static level or net drift)."""
    gt = ds.gt
    if cfg.data.kind == "dynamic":
        return metrics_bench.theta_recovery_dynamic(
            theta_traj_val, gt.theta_traj[val_rows]
        )
    return metrics_bench.theta_recovery_static(
        theta_traj_val[:, -1], gt.theta0[val_rows]
    )


def _score_item_recovery(cfg: P2Config, ds, engine,
                         train_it: torch.Tensor, train_rp: torch.Tensor,
                         train_rows: np.ndarray) -> dict:
    """Item-parameter recovery on the TRAINING folds.

    GPCM/2PL: discrimination + step thresholds via ``metrics_bench.item_recovery``
    (static read is item-only; dynamic occurrence-averages over the training-fold
    sequences).  NRM: per-option a_k AND c_k via ``nrm_metrics.item_recovery``,
    keeping the intercept.  ``seen`` restricts scoring to items the training folds
    actually cover (the blueprint seen-mask fix: unseen items keep at-init
    embeddings and must not be scored).
    """
    mc = cfg.model
    model = engine.model
    uses_state = bool(mc.state_alpha) or bool(mc.state_beta)

    # Items the training folds cover (recovery is only meaningful for these).
    seen = np.zeros(ds.cfg.n_items, dtype=bool)
    seen[np.unique(np.asarray(ds.items0[train_rows]))] = True

    if mc.decoder == "nrm":
        # KEEP the intercept; do NOT route through the GPCM-shaped path.
        params = model.recover_item_params()          # {alpha (Q,K), intercept (Q,K)}
        a_hat = np.asarray(params["alpha"])
        c_hat = np.asarray(params["intercept"])
        a_true, c_true = _nrm_ground_truth(ds)
        if a_true is None or c_true is None:
            # Clean GPCM bed emits no NRM ground truth: report prediction + theta
            # only (honest; matches the base engine's NRM handling).
            return {}
        return nrm_metrics.item_recovery(
            a_hat, c_hat, np.asarray(a_true), np.asarray(c_true), seen=seen
        )

    # GPCM / 2PL.
    if uses_state:
        rec = model.recover_item_params(train_it, train_rp)   # {alpha, beta, seen}
        a_hat = np.asarray(rec["alpha"])
        b_hat = np.asarray(rec["beta"])
        seen = np.asarray(rec["seen"])
    else:
        rec = model.recover_item_params()                     # {alpha (Q,), beta (Q,K-1)}
        a_hat = np.asarray(rec["alpha"])
        b_hat = np.asarray(rec["beta"])
    a_hat, b_hat = _apply_gauge(a_hat, ds.gt.a, b_hat, ds.gt.b, gauge=cfg.eval.gauge)
    return metrics_bench.item_recovery(a_hat, b_hat, ds.gt.a, ds.gt.b, seen=seen)


# ---------------------------------------------------------------------------
# One fit-recover-score cycle on a given (train_rows, val_rows) partition
# ---------------------------------------------------------------------------

def _fit_and_score(
    cfg: P2Config,
    ds,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    data_seed: int,
    init_seed: int,
    device: str,
    fold: Optional[int] = None,
) -> dict:
    """Fit on ``train_rows``, score prediction + theta on ``val_rows``, recover
    item params on ``train_rows``.  Returns one results row."""
    seed_log = _set_seeds(data_seed, init_seed)
    train_rows = np.asarray(train_rows)
    val_rows = np.asarray(val_rows)

    # A fold view over the SAME arrays: only the train/val partition changes.
    fold_ds = dataclasses.replace(ds, train_idx=train_rows, val_idx=val_rows)
    engine = _build_engine(cfg, fold_ds, init_seed, device)

    tc = cfg.train
    t0 = time.time()
    if isinstance(engine, _P2Engine):
        engine.fit(
            n_epochs=tc.n_epochs, lr=tc.lr, batch_size=tc.batch_size,
            grad_clip_norm=tc.grad_clip_norm, patience=tc.early_stop_patience,
        )
    else:  # ma_irt comparison engine (no early stop / mask surface)
        engine.fit(n_epochs=tc.n_epochs, lr=tc.lr, batch_size=tc.batch_size)
    t_fit = time.time() - t0

    # Prediction on the held-out fold's last valid positions.
    probs, targets = engine.predict_heldout()
    is_nrm = cfg.model.decoder == "nrm"
    if is_nrm:
        pred_metrics = nrm_metrics.prediction_metrics(probs, targets, cfg.data.n_cats)
    else:
        pred_metrics = metrics_bench.prediction_metrics(probs, targets, cfg.data.n_cats)

    # Theta on the held-out fold learners.
    val_it, val_rp = _long(ds.items0[val_rows]), _long(ds.responses[val_rows])
    if hasattr(engine.model, "track"):
        theta_traj_val = engine.model.track(val_it, val_rp).cpu().numpy()
    else:  # ma_irt fallback: recover() returns theta over all fold_ds rows
        theta_traj_val = np.asarray(engine.recover()["theta_traj"])[val_rows]
    theta_metrics = _score_theta(cfg, ds, theta_traj_val, val_rows)

    # Item recovery on the training folds.
    train_it, train_rp = _long(ds.items0[train_rows]), _long(ds.responses[train_rows])
    item_metrics = _score_item_recovery(cfg, ds, engine, train_it, train_rp, train_rows)

    cost = engine.cost()
    row = {
        "data_seed": data_seed,
        "init_seed": init_seed,
        "fit_time_s": round(t_fit, 3),
        "n_params": cost["n_params"],
        "seed_log": seed_log,
        **pred_metrics,
        **theta_metrics,
        **item_metrics,
    }
    if fold is not None:
        row["fold"] = fold
    return row


# ---------------------------------------------------------------------------
# Single (data_seed, init_seed) run on the dataset's own train/val split
# ---------------------------------------------------------------------------

def run_one(cfg: P2Config, data_seed: int, init_seed: int,
            device: str = "cpu") -> dict:
    """One fit-recover-score cycle on the dataset's own train/val split.

    Retained for callers that want a single non-CV row (the ``run_one()`` schema
    referenced by ``_p2_oracle`` / ``_p2_coupled_theta``).  ``run_cell`` uses the
    5-fold CV path below.
    """
    _set_seeds(data_seed, init_seed)
    ds = _generate_dataset(cfg, data_seed)
    return _fit_and_score(cfg, ds, ds.train_idx, ds.val_idx,
                          data_seed, init_seed, device)


# ---------------------------------------------------------------------------
# Full cell: 5-fold learner CV -> fold-mean + bootstrap CI
# ---------------------------------------------------------------------------

def run_cell(
    cfg: P2Config,
    device: str = "cpu",
    force: bool = False,
) -> Path:
    """Run the 5-fold learner CV for one config cell and aggregate.

    For each data seed a fresh ground-truth bank is generated and its learners are
    partitioned into ``eval.n_folds`` folds (seeded, deterministic).  Rotating the
    held-out fold gives one fit per fold; each fit recovers item params on the
    training folds and scores val-fold theta + held-out prediction.  The fold rows
    (pooled across data seeds) are aggregated to the fold-mean + bootstrap 95% CI
    via ``_p2_aggregate.aggregate_rows``.

    Writes:
      <output_dir>/<cell_name>/results.json  -- full per-fold rows + agg
      <output_dir>/<cell_name>/summary.json  -- compact single-row summary

    Skips (returns existing path) if results.json already carries the matching
    config hash and force=False.
    """
    out_dir = Path(cfg.report.output_dir) / cfg.report.cell_name
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    cfg_hash = config_sha256(cfg)

    if not force and results_path.exists():
        try:
            existing = json.loads(results_path.read_text())
            if existing.get("config_sha256") == cfg_hash:
                return results_path  # already done, hash matches
        except Exception:
            pass  # corrupted file: re-run

    # Replication grid.  Quick/smoke: 1 data seed x 2 folds (exercises rotation
    # cheaply).  Full: eval.n_data_seeds banks x eval.n_folds folds.
    if cfg.report.quick:
        n_data = 1
        n_folds = min(2, cfg.eval.n_folds)
    else:
        n_data = cfg.eval.n_data_seeds
        n_folds = cfg.eval.n_folds
    if n_folds < 2:
        raise ValueError(f"CV needs n_folds >= 2, got {n_folds}")

    per_seed_rows = []
    for d_idx in range(n_data):
        data_seed = d_idx
        ds = _generate_dataset(cfg, data_seed)
        n_learners = int(ds.items0.shape[0])
        fold_rng = np.random.default_rng(2000 + data_seed)
        perm = fold_rng.permutation(n_learners)
        folds = np.array_split(perm, n_folds)
        # Fixed model init per bank; the fold rotation is the only thing that
        # varies within a bank (standard learner CV).
        init_seed = data_seed
        for f in range(n_folds):
            val_rows = np.sort(folds[f])
            train_rows = np.sort(
                np.concatenate([folds[j] for j in range(n_folds) if j != f])
            )
            row = _fit_and_score(cfg, ds, train_rows, val_rows,
                                 data_seed, init_seed, device, fold=f)
            per_seed_rows.append(row)

    # Fold-mean + bootstrap 95% CI (the LOCKED decision-1 uncertainty report).
    agg = aggregate_rows(
        per_seed_rows, metrics=SCORED_METRICS,
        n_boot=cfg.eval.n_bootstrap, seed=0,
    )

    results = {
        "schema_version": SCHEMA_VERSION,
        "config_sha256": cfg_hash,
        "config": dataclasses.asdict(cfg),
        "env": _env_manifest(device),
        "cv": {"n_data_seeds": n_data, "n_folds": n_folds,
               "n_rows": len(per_seed_rows)},
        "per_seed_rows": per_seed_rows,
        "agg": agg,
    }
    results_path.write_text(json.dumps(results, indent=2, default=_json_default))

    summary = {
        "cell_name": cfg.report.cell_name,
        "config_sha256": cfg_hash,
        "n_runs": len(per_seed_rows),
        "agg": agg,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default)
    )

    return results_path


def _json_default(obj):
    """Fallback JSON serialiser for numpy scalars and arrays."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from deep_irt.bench._p2_config import load_config

    parser = argparse.ArgumentParser(description="Phase 2 single-cell runner (5-fold CV)")
    parser.add_argument("config", help="path to a configs_p2/*.yaml file")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true",
                        help="re-run even if results.json hash matches")
    parser.add_argument("--quick", action="store_true",
                        help="override to 1 data seed x 2 folds for smoke testing")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.quick:
        cfg.report.quick = True
    out = run_cell(cfg, device=args.device, force=args.force)
    print(f"results written to: {out}")
