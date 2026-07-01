"""
_p2_datagen_budget.py -- E-budget grid generation for Phase 2.

SCAFFOLD: fill-in pending. Signatures and return contracts are specified;
bodies raise NotImplementedError with explicit TODO(fill-in) notes.

Role:
  Support three budget axes from the blueprint (E-budget experiment):
    (i)  Training-T sweep: vary n_epochs on PLAIN GD (not Adam), measure
         fit-gap(T) and fit A * exp(-T / kappa) + c. An Adam arm is included
         to show preconditioning compresses kappa.
    (ii) Sample-N sweep: vary n_learners, measure reliability gap ~ O(kappa/N)
         and EIV attenuation ~ O(1/N).
    (iii) I(theta)-at-fixed-K sweep: vary difficulty spread beta_sigma at fixed K
          to move kappa; crossed with a K-sweep to show and then break the
          K-kappa collinearity.

  This module produces grid-spec config lists. Actual training is dispatched
  via run_sweep() or run_cell() from _p2_sweep / _p2_run_cell.

NOTE: datagen.generate() currently hardcodes the b-spread to N(0,1). Until
BenchDataConfig exposes a beta_sigma field, the sigma_b axis requires either:
  (a) A post-generate rescale of ds.gt.b and ds.responses (fragile -- responses
      were sampled at sigma=1, so changing b post-hoc is inconsistent).
  (b) Requesting the beta_sigma field be added to BenchDataConfig (Codex-owned;
      do NOT edit datagen.py directly -- file a note in the blueprint tracker).
  The stubs below flag this open decision on each relevant function.
"""

from __future__ import annotations

import copy
from typing import List

from deep_irt.bench._p2_config import P2Config, TrainConfig


# ---------------------------------------------------------------------------
# Grid builders
# ---------------------------------------------------------------------------

def make_T_grid(base_cfg: P2Config, T_values: List[int]) -> List[P2Config]:
    """Produce a list of P2Configs varying n_epochs (training-T axis).

    Each config forces optimizer="sgd" (plain GD). Adam is handled by a
    separate grid with optimizer="adam" to show the preconditioning effect.
    cell_name is suffixed with "_T{value}" so each grid point writes to its
    own output directory.

    Args:
        base_cfg: baseline config (copied, not mutated).
        T_values: list of n_epochs values (e.g. [10, 20, 40, 80, 160]).

    Returns:
        One P2Config per T value, with train.n_epochs = T and
        train.optimizer = "sgd".

    TODO(fill-in): use copy.deepcopy(base_cfg) per grid point, then set
      cfg.train.n_epochs = T, cfg.train.optimizer = "sgd",
      cfg.report.cell_name = f"{base_cfg.report.cell_name}_T{T}".
    TODO(fill-in): validate that base_cfg.eval.n_data_seeds == 10 per blueprint.
    """
    raise NotImplementedError("make_T_grid not yet implemented -- fill-in pending")


def make_N_grid(base_cfg: P2Config, N_values: List[int]) -> List[P2Config]:
    """Produce a list of P2Configs varying n_learners (sample-N axis).

    cell_name is suffixed with "_N{value}".

    Args:
        base_cfg: baseline config (copied, not mutated).
        N_values: list of n_learners values (e.g. [200, 400, 800, 1600, 3200]).

    Returns:
        One P2Config per N value.

    TODO(fill-in): use copy.deepcopy, set cfg.data.n_learners = N,
      cfg.report.cell_name = f"{base_cfg.report.cell_name}_N{N}".
    """
    raise NotImplementedError("make_N_grid not yet implemented -- fill-in pending")


def make_sigma_b_grid(
    base_cfg: P2Config, sigma_values: List[float]
) -> List[P2Config]:
    """Produce a list of P2Configs varying difficulty spread (sigma_b axis).

    sigma_b controls I(theta) at fixed K by spreading the step thresholds.
    Larger sigma_b -> larger item information -> smaller kappa.

    OPEN DECISION: datagen.generate() does not yet expose beta_sigma as a
    BenchDataConfig field. Until it does, this grid builder writes the desired
    value into cfg.data.beta_sigma (the config field exists) but run_cell will
    silently ignore it. The TODO below flags the resolution path.

    TODO(fill-in): once BenchDataConfig.beta_sigma is wired in datagen, remove
      this warning and ensure _p2_run_cell._build_bench_cfg() threads it through.
    TODO(fill-in): validate that the K-sweep is crossed with this grid to break
      K-kappa collinearity (see make_K_grid below).
    """
    raise NotImplementedError("make_sigma_b_grid not yet implemented -- fill-in pending")


def make_K_grid(base_cfg: P2Config, K_values: List[int]) -> List[P2Config]:
    """Produce a list of P2Configs varying n_cats (K-sweep for kappa collinearity).

    Crossed with make_sigma_b_grid to show and then break K-kappa collinearity.

    TODO(fill-in): use copy.deepcopy, set cfg.data.n_cats = K,
      cfg.report.cell_name = f"{base_cfg.report.cell_name}_K{K}".
    TODO(fill-in): ensure decoder can handle the new K (nrm and gpcm do;
      binary is only valid for K=2).
    """
    raise NotImplementedError("make_K_grid not yet implemented -- fill-in pending")


# ---------------------------------------------------------------------------
# Curve fitting
# ---------------------------------------------------------------------------

def fit_exponential_decay(
    axis_values: List[float],
    gap_values: List[float],
) -> dict:
    """Fit A * exp(-x / kappa) + c to an empirical fit-gap curve.

    Returns a dict with keys: A, kappa, c, r_squared, fit_quality.

    HONEST LIMITATION (from blueprint): the exp(-T/kappa) coefficient is
    delicate. Full-batch GD may never reach the local-quadratic regime where
    the ODE approximation is tight. This function stays a form-and-ordering
    claim: kappa is the key comparative readout (larger sigma_b -> smaller
    kappa), not a precisely identified coefficient.

    Args:
        axis_values: x-axis values (e.g. T_values or N_values).
        gap_values: observed metric gap at each x-axis point.

    Returns:
        {"A": float, "kappa": float, "c": float, "r_squared": float,
         "fit_quality": "good" | "poor"}
        fit_quality is "poor" if r_squared < 0.90 (warn in the paper).

    TODO(fill-in): use scipy.optimize.curve_fit with p0=(max(gap)-min(gap),
      mean(axis), min(gap)) and bounds A>0, kappa>0, c>=0. Compute r_squared
      from residuals. Report honestly if convergence fails.
    """
    raise NotImplementedError("fit_exponential_decay not yet implemented -- fill-in pending")
