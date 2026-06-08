"""``OrdinalRewardCompute``, the four-component reward composer.

Combines the probe-entropy shaping (Lindley 1956, Owen 1975 lens),
the per-administered-item ask cost, the Sympson-Hetter (1985)
exposure penalty, and the terminal NLL anchor on the held-out probe
``H_probe`` into the single scalar ``r_total`` consumed by the RL
algorithm.

Formula::

    r_total = w_info * (phi(theta_t) - phi(theta_prev))
            - w_cost * K_B
            - w_expo * c_expo * sum_{q in action} max(0, expo_rate[q] - r_max)
            + 1[step == T // K_B] * w_voi * (nll_prior - nll_terminal)

The composer keeps per-component contributions in the returned dict
so logging can decompose the reward without re-running anything.

See ``docs/ordrec_impl_guide.md`` Sections 3.2 and 3.7 for the
contract surface, and Section 3.8 for numerical-stability notes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor

from .config import RewardConfig
from .exposure import exposure_penalty
from .nll_anchor import terminal_anchor
from .probe_entropy import phi_entropy
from .running_norm import RunningMeanStd


@dataclass
class RewardBreakdown:
    """Per-component reward contributions, all shaped ``(B,)``."""

    r_info: Tensor
    r_cost: Tensor
    r_expo: Tensor
    r_voi: Tensor
    r_total: Tensor
    phi_t: Tensor
    phi_prev: Tensor

    def as_dict(self) -> Dict[str, Tensor]:
        return {
            "r_info": self.r_info,
            "r_cost": self.r_cost,
            "r_expo": self.r_expo,
            "r_voi": self.r_voi,
            "r_total": self.r_total,
            "phi_t": self.phi_t,
            "phi_prev": self.phi_prev,
        }


class OrdinalRewardCompute:
    """Callable that composes the four-component OrdRec reward.

    Construction is cheap; the running-mean-std normaliser is the only
    stateful collaborator. The composer does not own the fleet exposure
    buffer; the env owns it and passes the current value via
    ``info["fleet_expo"]`` because the env updates the EMA at episode
    boundaries and persists it across resets.

    Args:
        cfg: Frozen :class:`~ordrec.reward.config.RewardConfig`.
        n_categories: K. Stored so the entropy bound assertion can fire
            in debug builds.
        normalise: When ``True`` ``r_total`` passes through a Welford
            running-mean-std normaliser frozen after
            ``cfg.running_norm_freeze_after`` samples. Default
            ``False`` so unit tests see un-normalised raw rewards.
    """

    def __init__(
        self,
        cfg: RewardConfig,
        n_categories: int,
        *,
        normalise: bool = False,
    ) -> None:
        if n_categories < 2:
            raise ValueError(
                f"n_categories must be >= 2, got {n_categories}."
            )
        self.cfg = cfg
        self.n_categories = int(n_categories)
        self._normalise = bool(normalise)
        self._running = RunningMeanStd(
            freeze_after=cfg.running_norm_freeze_after,
            eps=1e-8,
        )

    # ------------------------------------------------------------------
    # Property surface
    # ------------------------------------------------------------------

    @property
    def running_stats(self) -> RunningMeanStd:
        return self._running

    # ------------------------------------------------------------------
    # The four-component composer
    # ------------------------------------------------------------------

    def __call__(
        self,
        state: Mapping[str, Tensor],
        action: Tensor,
        next_state: Mapping[str, Tensor],
        info: Mapping[str, Tensor],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute ``(r_total (B,), breakdown_dict)``.

        Args:
            state: MAGPCM forward dict at position ``t - K_B``. Must
                expose at least ``theta``. Only ``theta[:, -1, :]`` is
                read; if ``theta`` is already shaped ``(B, D)`` it is
                accepted as-is.
            action: ``LongTensor (B, K_B)`` of 1-based item ids.
            next_state: MAGPCM forward dict at position ``t``.
            info: Probe context fixed at ``env.reset``. Required keys
                ``probe_C_ids (B, M)``, ``alpha_table (Q+1, D)``,
                ``beta_table (Q+1, K-1)``, ``fleet_expo (Q+1,)``,
                ``step_index`` (int). Optional keys
                ``probe_H_ids (B, H)``, ``probe_H_resp (B, H)``,
                ``theta_0 (B, D)`` (required when ``step_index ==
                T // K_B``).

        Returns:
            ``(r_total (B,), breakdown)`` where breakdown carries the
            four per-component tensors plus ``phi_t`` and ``phi_prev``.
        """
        cfg = self.cfg
        theta_t = _theta_last(next_state)
        theta_prev = _theta_last(state)

        alpha_table = info["alpha_table"]
        beta_table = info["beta_table"]
        probe_C_ids = info["probe_C_ids"]
        fleet_expo = info["fleet_expo"]

        # 1. Shaping. The probe is fixed at reset and identical for both
        # phi evaluations, so the difference is the Ng-Harada-Russell
        # potential-based shaping signal.
        phi_t = phi_entropy(
            theta_t, probe_C_ids, alpha_table, beta_table,
            eps=cfg.eps_prob,
        )
        phi_prev = phi_entropy(
            theta_prev, probe_C_ids, alpha_table, beta_table,
            eps=cfg.eps_prob,
        )
        # Sign convention. Higher entropy means more uncertainty.
        # The policy should drive uncertainty down, so r_info rewards
        # entropy reduction.
        r_info = cfg.w_info * (phi_prev - phi_t)

        # 2. Ask cost. Counts non-pad actions; padding (id == 0) does
        # not incur the cost.
        ask_count = (action > 0).sum(dim=-1).to(theta_t.dtype)
        r_cost = -cfg.w_cost * ask_count

        # 3. Sympson-Hetter exposure penalty.
        expo_mag = exposure_penalty(
            action, fleet_expo, r_max=cfg.r_max, c_expo=cfg.c_expo,
        )
        r_expo = -cfg.w_expo * expo_mag

        # 4. Terminal NLL anchor, only at horizon.
        T_over_KB = cfg.T // cfg.K_B
        step_index = int(info.get("step_index", 0))
        if step_index == T_over_KB:
            probe_H_ids = info.get("probe_H_ids")
            probe_H_resp = info.get("probe_H_resp")
            theta_0 = info.get("theta_0")
            if probe_H_ids is None or probe_H_resp is None or theta_0 is None:
                raise KeyError(
                    "Terminal step requires probe_H_ids, probe_H_resp and "
                    "theta_0 in info."
                )
            r_voi = cfg.w_voi * terminal_anchor(
                theta_t, theta_0, probe_H_ids, probe_H_resp,
                alpha_table, beta_table,
            )
        else:
            r_voi = torch.zeros_like(r_info)

        r_total = r_info + r_cost + r_expo + r_voi

        if self._normalise:
            self._running.update(r_total)
            r_total_normed = self._running.normalise(r_total)
        else:
            r_total_normed = r_total

        breakdown = RewardBreakdown(
            r_info=r_info,
            r_cost=r_cost,
            r_expo=r_expo,
            r_voi=r_voi,
            r_total=r_total_normed,
            phi_t=phi_t,
            phi_prev=phi_prev,
        ).as_dict()
        return r_total_normed, breakdown


def _theta_last(state: Mapping[str, Tensor]) -> Tensor:
    """Read ``theta`` from a MAGPCM forward dict, supporting both shapes.

    Accepts either a 3D ``(B, S, D)`` tensor (MAGPCM forward output)
    or an already-summarised ``(B, D)`` tensor (handy for the env's
    state cache).
    """
    theta = state["theta"]
    if theta.ndim == 3:
        return theta[:, -1, :]
    if theta.ndim == 2:
        return theta
    raise ValueError(
        f"theta must be 2D or 3D, got shape {tuple(theta.shape)}"
    )


__all__ = ["OrdinalRewardCompute", "RewardBreakdown"]
