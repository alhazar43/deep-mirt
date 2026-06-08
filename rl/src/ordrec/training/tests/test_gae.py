"""Tests for ``compute_gae``.

Validates the formula against a hand-written numpy reference at
``T = 8`` with one mid-trajectory done. Also covers the
zero-bootstrap-at-terminal invariant.
"""

from __future__ import annotations

import numpy as np
import torch

from ordrec.training.gae import compute_gae


def _numpy_gae(rewards, values, dones, *, gamma, lambda_, last_value=0.0):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float64)
    gae = 0.0
    next_value = float(last_value)
    for t in reversed(range(T)):
        non_terminal = 1.0 - float(dones[t])
        delta = float(rewards[t]) + gamma * next_value * non_terminal - float(values[t])
        gae = delta + gamma * lambda_ * non_terminal * gae
        advantages[t] = gae
        next_value = float(values[t])
    returns = advantages + np.asarray(values, dtype=np.float64)
    return advantages, returns


def test_matches_numpy_reference_with_mid_trajectory_done() -> None:
    rewards = torch.tensor(
        [0.1, 0.0, 0.4, 0.2, 0.5, -0.1, 0.3, 0.8], dtype=torch.float32,
    )
    values = torch.tensor(
        [0.05, 0.1, 0.0, 0.2, 0.4, 0.1, 0.0, 0.3], dtype=torch.float32,
    )
    dones = torch.tensor(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32,
    )
    adv, ret = compute_gae(
        rewards, values, dones, gamma=0.95, lambda_=0.95, last_value=0.0,
    )

    ref_adv, ref_ret = _numpy_gae(
        rewards.numpy(), values.numpy(), dones.numpy(),
        gamma=0.95, lambda_=0.95, last_value=0.0,
    )
    assert torch.allclose(adv, torch.from_numpy(ref_adv).float(), atol=1e-6)
    assert torch.allclose(ret, torch.from_numpy(ref_ret).float(), atol=1e-6)


def test_terminal_zero_bootstrap() -> None:
    """At ``done[t] = True`` the future-value contribution must vanish."""
    rewards = torch.tensor([1.0, 0.0], dtype=torch.float32)
    values = torch.tensor([0.5, 0.5], dtype=torch.float32)
    dones = torch.tensor([0.0, 1.0], dtype=torch.float32)
    adv, _ = compute_gae(
        rewards, values, dones, gamma=0.99, lambda_=0.99, last_value=1e6,
    )
    # last_value should be ignored because the final transition is done.
    # delta_1 = 0.0 + 0.99 * 0 * 1e6 - 0.5 = -0.5; A_1 = -0.5
    assert float(adv[1].item()) == np.float32(-0.5)


def test_shape_mismatch_raises() -> None:
    rewards = torch.zeros(4)
    values = torch.zeros(3)
    dones = torch.zeros(4)
    try:
        compute_gae(rewards, values, dones)
    except ValueError:
        return
    raise AssertionError("compute_gae did not reject mismatched shapes")


def test_bootstrap_used_when_not_done() -> None:
    """If the final step is not done, last_value enters the delta."""
    rewards = torch.tensor([1.0], dtype=torch.float32)
    values = torch.tensor([0.5], dtype=torch.float32)
    dones = torch.tensor([0.0], dtype=torch.float32)
    adv, _ = compute_gae(rewards, values, dones, gamma=0.9, lambda_=0.95,
                          last_value=2.0)
    # delta_0 = 1.0 + 0.9 * 2.0 * 1 - 0.5 = 2.3
    assert float(adv[0].item()) == np.float32(2.3)
