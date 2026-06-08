"""Sympson-Hetter (1985) exposure penalty.

Reference. Sympson and Hetter, "Controlling item-exposure rates in
computerized adaptive testing", 1985 Proceedings of the Military
Testing Association. The Sympson-Hetter rule caps the fraction of
test takers who see any one item. Our adaptation is a soft hinge,
items with a fleet-wide EMA exposure ``expo_rate[q]`` above
``r_max`` incur a linear penalty per administration. Below
``r_max`` the penalty is zero so early exploration is unconstrained.

See ``docs/ordrec_impl_guide.md`` Section 3.6.
"""

from __future__ import annotations

import torch
from torch import Tensor


def exposure_penalty(
    action: Tensor,
    fleet_expo: Tensor,
    *,
    r_max: float,
    c_expo: float = 1.0,
) -> Tensor:
    """Sympson-Hetter linear-above-threshold exposure penalty.

    Args:
        action: ``LongTensor (B, K_B)`` of 1-based item ids administered
            in the current batch. ``0`` denotes padding and contributes
            ``0.0`` penalty as long as ``fleet_expo[0] <= r_max``.
        fleet_expo: ``Tensor (Q + 1,)`` EMA fleet exposure rates,
            indexed by item id.
        r_max: Hinge threshold. ``0.20`` per the strategic defaults.
        c_expo: Slope multiplier applied to the hinge.

    Returns:
        ``Tensor (B,)`` penalty values (non-negative).

    Notes:
        The composer multiplies this by ``-w_expo`` when adding it to
        the reward sum so callers get the magnitude here and the
        sign is fixed in one place.
    """
    if action.ndim != 2:
        raise ValueError(
            f"action must be 2D (B, K_B), got shape {tuple(action.shape)}"
        )
    if fleet_expo.ndim != 1:
        raise ValueError(
            f"fleet_expo must be 1D (Q + 1,), got shape {tuple(fleet_expo.shape)}"
        )

    fleet_expo = fleet_expo.to(dtype=torch.float32)
    selected = fleet_expo[action.long()]  # (B, K_B)
    hinge = (selected - float(r_max)).clamp_min(0.0)
    penalty = float(c_expo) * hinge.sum(dim=-1)  # (B,)
    return penalty


def update_fleet_exposure(
    fleet_expo: Tensor,
    admin_counts: Tensor,
    *,
    decay: float,
) -> Tensor:
    """In-place EMA update of the fleet exposure buffer.

    ``fleet_expo[q] <- decay * fleet_expo[q] + (1 - decay) * admin_counts[q]``
    where ``admin_counts[q]`` is the per-episode normalised count of
    administrations of item q (fraction of sessions in the just-completed
    rollout that saw item q at least once).

    Args:
        fleet_expo: ``Tensor (Q + 1,)``. Updated in-place and returned.
        admin_counts: ``Tensor (Q + 1,)``. Per-episode normalised
            session counts in ``[0, 1]``.
        decay: EMA decay. ``0.99`` per the strategic defaults.

    Returns:
        ``fleet_expo`` after the update.
    """
    if fleet_expo.shape != admin_counts.shape:
        raise ValueError(
            f"fleet_expo {tuple(fleet_expo.shape)} and admin_counts "
            f"{tuple(admin_counts.shape)} must share shape."
        )
    with torch.no_grad():
        fleet_expo.mul_(float(decay)).add_(
            admin_counts.to(dtype=fleet_expo.dtype), alpha=1.0 - float(decay)
        )
    return fleet_expo


__all__ = ["exposure_penalty", "update_fleet_exposure"]
