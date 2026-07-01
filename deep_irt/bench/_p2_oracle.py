"""
_p2_oracle.py -- Oracle-clamp arm for Phase 2 (E-oracle / oracle_clamp config).

SCAFFOLD: fill-in pending. Signatures and imports wired; bodies raise
NotImplementedError with explicit TODO(fill-in) notes.

Role:
  Teacher-force theta = theta* (ground truth), then recover item params only.
  Provides two readouts:
    (a) Causal Fisher-rate attribution: if the oracle recovers alpha well but
        the free model does not, the bottleneck is the theta channel.
    (b) MLE-oracle ceiling: upper bound on item-parameter recovery given
        perfect theta at every step.

  Blueprint: "Oracle-clamp: teacher-force theta = theta*, recover items only;
  causal Fisher-rate attribution + MLE-oracle ceiling."
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from deep_irt.bench._p2_config import P2Config
from deep_irt.bench.datagen import BenchDataset
from deep_irt.bench import metrics_bench


# ---------------------------------------------------------------------------
# MLE oracle: given true theta, optimise (a, b) per item
# ---------------------------------------------------------------------------

def mle_item_recovery(
    ds: BenchDataset,
    n_epochs: int = 500,
    lr: float = 5e-3,
    device: str = "cpu",
) -> Dict[str, Any]:
    """MLE oracle ceiling: recover (a, b) with true theta at every step.

    The GPCM log-likelihood is maximised jointly over all items, with:
      - a_j parameterised as exp(log_a_j) to enforce positivity.
      - b_j (K-1 thresholds) parameterised via cumulative sum to enforce ordering.

    Args:
        ds: BenchDataset (items0, responses, gt.theta_at_step used).
        n_epochs: gradient steps (Adam; usually converges in ~200).
        lr: learning rate for Adam.
        device: torch device string.

    Returns:
        Dict with keys:
          "a": np.ndarray (Q,) recovered discrimination
          "b": np.ndarray (Q, K-1) recovered step thresholds
          "item_metrics": metrics_bench.item_recovery() output dict
          "loss_history": list of per-epoch NLL values

    TODO(fill-in): implement the per-item GPCM NLL optimisation loop.
      Key steps:
        1. Build theta tensor from ds.gt.theta_at_step (N, T) and item index
           tensor from ds.items0 (N, T).
        2. Initialise log_a (Q,) and log_b_gaps (Q, K-1) as nn.Parameters.
        3. For each epoch: compute GPCM probs via the _gpcm_probs logic
           (vectorised), compute NLL = -mean log P(r_{it} | theta_{it}, a_j, b_j),
           Adam step.
        4. Recover a = exp(log_a), b = cumsum(exp(log_b_gaps)) + b_offset.
        5. Call metrics_bench.item_recovery(a_hat, b_hat, ds.gt.a, ds.gt.b).
    TODO(fill-in): import and reuse the GPCM log-prob helper from datagen or
      implement a vectorised torch version here.
    """
    raise NotImplementedError("mle_item_recovery not yet implemented -- fill-in pending")


# ---------------------------------------------------------------------------
# Engine-clamp oracle: use an engine with theta forced to ground truth
# ---------------------------------------------------------------------------

def clamp_theta_and_recover(
    cfg: P2Config,
    ds: BenchDataset,
    init_seed: int = 0,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Build a DeepIRTEngine, freeze theta to ground truth, recover item params.

    The clamping mechanism hooks the encoder output and replaces it with
    ds.gt.theta_at_step before it reaches the decoder. Item params are then
    learned to maximise the prediction likelihood given perfect theta.

    Returns a result dict in the run_one() schema.

    TODO(fill-in): implement the theta-clamp hook. Two approaches:
      (a) Monkey-patch model.encoder.theta_for_prediction() to return a fixed
          tensor wrapping ds.gt.theta_at_step. Restore after fit.
      (b) Subclass or wrap DeepIRTModel to accept an external theta_override
          tensor in its forward pass. Cleaner but more invasive.
    TODO(fill-in): decide whether to clamp during fit only, or also during
      recover() (which re-runs the encoder). Likely: clamp during fit so the
      decoder weights learn under oracle theta; recover() is then run free to
      compare oracle-fit vs oracle-recover.
    TODO(fill-in): import _p2_run_cell._build_engine and _set_seeds to avoid
      duplicating engine-construction logic.
    """
    raise NotImplementedError("clamp_theta_and_recover not yet implemented -- fill-in pending")


# ---------------------------------------------------------------------------
# Comparison helper: oracle vs free model
# ---------------------------------------------------------------------------

def compare_oracle_vs_free(
    oracle_results: Dict[str, Any],
    free_results: Dict[str, Any],
    metric: str = "a_spearman",
) -> Dict[str, float]:
    """Compute the oracle gain: oracle_metric - free_metric for each seed pair.

    Positive gain => the theta channel is the bottleneck for that metric.
    Returns {"oracle_mean", "free_mean", "gain_mean", "gain_std"}.

    TODO(fill-in): implement after run_one() and mle_item_recovery() both return.
    """
    raise NotImplementedError("oracle comparison not yet implemented -- fill-in pending")
