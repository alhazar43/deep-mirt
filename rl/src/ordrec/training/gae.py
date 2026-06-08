"""Generalised Advantage Estimation (Schulman et al. 2016).

Computes the GAE advantages and bootstrapped returns for a single
contiguous trajectory of length ``T``. Done masking sets the bootstrap
to zero at terminal transitions.

Formula::

    delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
    A_hat_t = delta_t + (gamma * lambda) * (1 - done_t) * A_hat_{t+1}
    R_t     = A_hat_t + V(s_t)

See ``docs/ordrec_impl_guide.md`` Section 4.3.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    *,
    gamma: float = 0.95,
    lambda_: float = 0.95,
    last_value: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    """Compute GAE advantages and bootstrapped returns.

    Args:
        rewards: ``Tensor (T,)`` scalar rewards per step.
        values: ``Tensor (T,)`` critic value estimates per step.
        dones: ``Tensor (T,)`` Boolean or float-typed terminal flags.
            ``1`` at the last step of an episode, ``0`` otherwise.
        gamma: Discount factor in ``[0, 1]``. OrdRec default ``0.95``.
        lambda_: GAE bias-variance trade-off in ``[0, 1]``. OrdRec
            default ``0.95`` (close to TD(1) Monte Carlo).
        last_value: ``V(s_T)`` bootstrap used when the final transition
            is not terminal. Defaults to ``0``, which matches the
            "always reach done" v1 OrdRec assumption.

    Returns:
        ``(advantages (T,), returns (T,))`` both float-typed. The
        returns are ``advantages + values`` and feed the value loss.
    """
    if rewards.ndim != 1 or values.ndim != 1 or dones.ndim != 1:
        raise ValueError(
            "rewards, values and dones must each be 1D. Got "
            f"{tuple(rewards.shape)}, {tuple(values.shape)}, "
            f"{tuple(dones.shape)}."
        )
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError(
            "rewards, values and dones must share length. Got "
            f"{tuple(rewards.shape)}, {tuple(values.shape)}, "
            f"{tuple(dones.shape)}."
        )

    T = rewards.shape[0]
    device = rewards.device
    dtype = rewards.dtype if rewards.is_floating_point() else torch.float32
    rewards_f = rewards.to(dtype=dtype)
    values_f = values.to(dtype=dtype)
    dones_f = dones.to(dtype=dtype)

    advantages = torch.zeros(T, dtype=dtype, device=device)
    gae = torch.zeros((), dtype=dtype, device=device)
    next_value = torch.tensor(float(last_value), dtype=dtype, device=device)
    for t in reversed(range(T)):
        non_terminal = 1.0 - dones_f[t]
        delta = rewards_f[t] + gamma * next_value * non_terminal - values_f[t]
        gae = delta + gamma * lambda_ * non_terminal * gae
        advantages[t] = gae
        next_value = values_f[t]

    returns = advantages + values_f
    return advantages, returns


__all__ = ["compute_gae"]
