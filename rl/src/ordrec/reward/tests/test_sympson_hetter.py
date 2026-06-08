"""Sympson-Hetter (1985) hinge.

The exposure penalty is zero whenever ``fleet_expo[q] <= r_max`` and
linear above. The hinge slope is ``c_expo``, so the penalty per
administration of an item with exposure ``rho`` above ``r_max`` is
``c_expo * (rho - r_max)``.
"""

from __future__ import annotations

import torch

from ordrec.reward.exposure import exposure_penalty, update_fleet_exposure


def test_below_threshold_penalty_zero() -> None:
    Q = 16
    fleet = torch.full((Q + 1,), 0.10)
    action = torch.tensor([[1, 2, 3]])
    pen = exposure_penalty(action, fleet, r_max=0.20, c_expo=1.0)
    assert torch.allclose(pen, torch.tensor([0.0]))


def test_above_threshold_linear_in_excess() -> None:
    Q = 16
    fleet = torch.zeros(Q + 1)
    fleet[5] = 0.30  # 0.10 above threshold
    fleet[6] = 0.50  # 0.30 above threshold
    fleet[7] = 0.10  # below threshold, contributes zero
    action = torch.tensor([[5, 6, 7]])
    pen = exposure_penalty(action, fleet, r_max=0.20, c_expo=1.0)
    expected = torch.tensor([0.10 + 0.30 + 0.0])
    assert torch.allclose(pen, expected, atol=1e-6)


def test_c_expo_scales_slope() -> None:
    Q = 8
    fleet = torch.zeros(Q + 1)
    fleet[1] = 0.40
    action = torch.tensor([[1]])
    pen_a = exposure_penalty(action, fleet, r_max=0.20, c_expo=1.0)
    pen_b = exposure_penalty(action, fleet, r_max=0.20, c_expo=2.5)
    assert torch.allclose(pen_b, 2.5 * pen_a, atol=1e-6)


def test_per_session_penalty_aggregates_over_K_B() -> None:
    Q = 16
    fleet = torch.zeros(Q + 1)
    fleet[5] = 0.40
    action = torch.tensor([[5, 5, 5]])  # K_B = 3, all hits on item 5
    pen = exposure_penalty(action, fleet, r_max=0.20, c_expo=1.0)
    assert torch.allclose(pen, torch.tensor([0.60]), atol=1e-6)


def test_fleet_ema_update() -> None:
    Q = 8
    fleet = torch.zeros(Q + 1)
    admin = torch.zeros(Q + 1)
    admin[2] = 1.0
    update_fleet_exposure(fleet, admin, decay=0.99)
    # decay 0.99 means new value = 0.01 * admin
    assert torch.allclose(fleet[2], torch.tensor(0.01), atol=1e-6)

    # Repeated updates accumulate toward the steady-state.
    for _ in range(99):
        update_fleet_exposure(fleet, admin, decay=0.99)
    # After 100 updates, geometric series with ratio 0.99 sums to ~0.634
    assert fleet[2].item() > 0.5 and fleet[2].item() < 0.7
