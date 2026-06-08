"""Reward-side primitives for OrdRec.

The E3 milestone lands the four-component reward composer
(:class:`OrdinalRewardCompute`), the vectorised GPCM predictive
entropy (:func:`phi_entropy`), the terminal NLL anchor on the
held-out probe (:func:`gpcm_nll`, :func:`terminal_anchor`), the
Sympson-Hetter (1985) exposure penalty (:func:`exposure_penalty`,
:func:`update_fleet_exposure`) and a Welford running-mean-std
normaliser with a freeze flag (:class:`RunningMeanStd`).

See ``docs/ordrec_impl_guide.md`` Section 3 for the strategic
design.
"""

from __future__ import annotations

from .config import RewardConfig
from .exposure import exposure_penalty, update_fleet_exposure
from .nll_anchor import gpcm_nll, terminal_anchor
from .ordinal_reward import OrdinalRewardCompute, RewardBreakdown
from .probe_entropy import phi_entropy
from .running_norm import RunningMeanStd, RunningStats


__all__ = [
    "OrdinalRewardCompute",
    "RewardBreakdown",
    "RewardConfig",
    "RunningMeanStd",
    "RunningStats",
    "exposure_penalty",
    "gpcm_nll",
    "phi_entropy",
    "terminal_anchor",
    "update_fleet_exposure",
]
