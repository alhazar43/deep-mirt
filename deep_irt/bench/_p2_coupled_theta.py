"""
_p2_coupled_theta.py -- Coupled-theta head for E2(a) identifiability experiment.

SCAFFOLD: fill-in pending. Both coupling vehicles are stubbed so the open
decision can be resolved by filling in one path without touching the other.

Role:
  E2(a) coupling -> IDENTIFIABILITY (structural): ability_coupling coupled vs
  decoupled, GPCM decoder, LSTM + DKVMN encoders. Beta rank recovery is the
  primary readout. Under coupling, a single global affine cannot rescue the
  per-item flat directions, so DEGRADED beta rank recovery IS the identifiability
  signature.

  *** OPEN DECISION (Blueprint §Open decisions #1) ***

  Vehicle A -- ma_irt engine, separate_theta toggle (cheapest path):
    MaIrtEngine already accepts a `separate_theta` kwarg that is forwarded
    to MAGPCM / LSTMGPCM / TransformerGPCM. When separate_theta=False,
    the theta stream is item-conditioned (coupled); when True, it is item-blind
    (decoupled). This covers DKVMN + LSTM + Transformer within ma-irt at zero
    new code cost.
    Risk: we are relying on ma-irt's notion of coupling rather than
    constructing our own -- need to verify this matches the blueprint's
    structural definition.

  Vehicle B -- deep_irt-native coupled-theta head (more code; fuller claim):
    Add a coupling flag to DeepIRTEngine that routes alpha through a
    [state | item_key] projection, making theta-conditioned alpha the channel
    that causes the structural flat direction in beta. state_alpha=True already
    conditions alpha on the encoder state; it may be the natural coupling toggle.
    Need to verify whether state_alpha=True IS the structural coupling intended
    by E2(a) or merely a capacity lever (E-levers).

  Resolution: fill in run_ma_irt_coupling_cell (Vehicle A) first; if the
  separate_theta toggle matches the blueprint's structural definition, Vehicle B
  is deferred to a follow-up. Document the resolution in the blueprint tracker.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from deep_irt.bench.datagen import BenchDataset
from deep_irt.bench._p2_config import P2Config
from deep_irt.bench import metrics_bench


# ---------------------------------------------------------------------------
# Vehicle A: ma_irt engine, separate_theta=False (item-conditioned / coupled)
# ---------------------------------------------------------------------------

def run_ma_irt_coupling_cell(
    cfg: P2Config,
    ds: BenchDataset,
    coupled: bool,
    init_seed: int = 0,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Run one E2(a) cell using the ma_irt engine's separate_theta flag.

    Args:
        cfg: P2Config with model.engine == "ma_irt".
        ds: BenchDataset (pre-generated; data seed already fixed by caller).
        coupled: True -> separate_theta=False (item-conditioned theta, the
                 coupled path that creates the structural flat direction in beta).
                 False -> separate_theta=True (item-blind, decoupled, the control).
        init_seed: weight-init RNG seed.
        device: torch device string.

    Returns:
        Result dict in run_one() schema (same keys as _p2_run_cell.run_one).

    TODO(fill-in): confirm direction: read MAGPCM.forward() to verify
      separate_theta=False -> theta is a function of the item key (coupled),
      separate_theta=True -> theta is item-blind (decoupled).
    TODO(fill-in): import and call _p2_run_cell._set_seeds(data_seed, init_seed).
    TODO(fill-in): build MaIrtEngine(ds, encoder=cfg.model.encoder, device=device,
      seed=init_seed, separate_theta=(not coupled), **cfg.model.ma_irt_kwargs).
    TODO(fill-in): fit -> predict_heldout -> recover -> score with
      metrics_bench.item_recovery. Return result row.
    TODO(fill-in): the key claim is that beta rank is degraded when coupled=True
      even after the global affine gauge fix; confirm this is scored via
      item_metrics["b_spearman"] from metrics_bench.item_recovery().
    """
    from deep_irt.bench.engines import MaIrtEngine
    raise NotImplementedError(
        "run_ma_irt_coupling_cell not yet implemented -- fill-in pending"
    )


# ---------------------------------------------------------------------------
# Vehicle B: deep_irt-native coupled-theta head (stub only)
# ---------------------------------------------------------------------------

class CoupledThetaHead(nn.Module):
    """Stub for a deep_irt-native coupled-theta head.

    In the DECOUPLED regime (E2a control):
      alpha is read from the item embedding alone (static, item-only).

    In the COUPLED regime (E2a treatment):
      alpha is read from a joint projection of [state, item_key], making
      theta-conditioned alpha the channel that opens a flat direction in beta
      (increasing alpha_i <-> decreasing b_i at fixed theta is equally good).

    OPEN: this may be equivalent to state_alpha=True in DeepIRTEngine. Verify
    before implementing -- if state_alpha=True already implements the structural
    coupling, CoupledThetaHead is not needed and Vehicle A suffices.

    TODO(fill-in): decide between:
      (a) state_alpha=True in DeepIRTEngine (zero new code; run E2a via the
          existing lever and check if it induces the beta flat direction).
      (b) A new CoupledThetaHead that explicitly projects [state | item_key] ->
          alpha with a learned coupling weight (more principled, more code).
    """

    def __init__(self, hidden_dim: int, item_emb_dim: int, coupled: bool = True):
        super().__init__()
        self.coupled = coupled
        # TODO(fill-in): define coupling projection layers when coupled=True.
        # Decoupled path: alpha = f(item_emb) only (static head, no new params).
        raise NotImplementedError("CoupledThetaHead not yet implemented -- fill-in pending")

    def forward(self, state: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        """Return per-step alpha (B,) for one sequence step.

        TODO(fill-in): coupled -> linear([state | item_emb]) -> softplus
                       decoupled -> linear(item_emb) -> softplus (matches static head)
        """
        raise NotImplementedError


def run_deep_irt_coupling_cell(
    cfg: P2Config,
    ds: BenchDataset,
    coupled: bool,
    init_seed: int = 0,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Run one E2(a) cell using the deep_irt native coupling toggle.

    TODO(fill-in): resolve the open decision first (see module docstring).
    If state_alpha=True is the coupling toggle, build DeepIRTEngine with
    state_alpha=coupled and run the standard fit -> recover -> score pipeline.
    If CoupledThetaHead is needed, patch the engine's decoder alpha head here.
    """
    raise NotImplementedError(
        "run_deep_irt_coupling_cell not yet implemented -- fill-in pending"
    )
