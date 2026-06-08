"""``OrdinalEnvBase`` ABC and the :class:`OrdinalState` dataclass.

Defines the Gym-style contract every OrdRec environment satisfies and
the dataclass the env returns to the RL algorithm. The state carries
``theta_t``, optional posterior precision ``sigma_t``, the per-batch
action mask, exposure feature vector, an episode-step counter, the
``terminated`` and ``truncated`` flags, and a free-form ``raw_info``
dict that is passed straight through to the reward composer.

See ``docs/ordrec_impl_guide.md`` Section 5.1 for the envelope spec
and Section 6 for the contract surface between modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class OrdinalState:
    """One-step ordinal-env observation envelope.

    The dataclass keeps the policy-facing tensors near the metadata the
    reward function reads so the env's ``step`` return type is a
    single-arg pass to either side.

    Attributes:
        theta_t: ``Tensor (B, D)`` latest ability estimate from the
            world model. Always present.
        sigma_t: Optional posterior precision ``Tensor (B, D, D)`` when
            the env computes a Laplace summary. ``None`` in the
            point-estimate path.
        history_questions: ``LongTensor (B, S_so_far)`` of administered
            item ids (1-based), where ``S_so_far = step_index * K_B``.
        history_responses: ``LongTensor (B, S_so_far)`` of observed
            ordinal labels.
        exposure_counts: ``LongTensor (B, Q + 1)`` per-session
            administration counts. Per-row sums match
            ``len(history_questions)`` so the state can be checked at
            tests.
        episode_step: ``int`` current batch index ``b in [0, T // K_B]``.
            ``0`` after reset, increments by 1 per ``step``.
        terminated: ``bool``. True iff the episode reached its horizon.
        truncated: ``bool``. Reserved for future variable-length
            episodes; always ``False`` in v1.
        action_mask: ``Tensor (B, Q + 1)``. Combined admin/probe/no-repeat
            mask. Allowed items have ``True``; disallowed have
            ``False``. Bool dtype.
        raw_info: Free-form dict the env populates and the reward
            consumes verbatim. Must carry the probe context
            (``probe_C_ids``, ``probe_H_ids``, ``probe_H_resp``),
            the per-item ``(alpha_table, beta_table)``,
            ``fleet_expo``, ``step_index``, ``theta_0`` and any
            additional logging fields.
    """

    theta_t: Tensor
    history_questions: Tensor
    history_responses: Tensor
    exposure_counts: Tensor
    episode_step: int
    terminated: bool
    truncated: bool
    action_mask: Tensor
    raw_info: Dict[str, Any] = field(default_factory=dict)
    sigma_t: Optional[Tensor] = None

    def to_tensor(self, device: Optional[torch.device] = None) -> Tensor:
        """Flatten the policy-relevant fields into a single observation.

        The default flattening concatenates ``theta_t`` (``(B, D)``)
        and a one-hot-of-episode-step prefix into ``(B, D + T_over_KB)``.
        Algorithms that need the exposure feature can read it from
        ``self.raw_info``; this method gives PPO a stable 2D vector.

        Args:
            device: Optional torch device override; defaults to
                ``self.theta_t.device``.

        Returns:
            ``Tensor (B, D + horizon)`` float-typed observation.
        """
        theta = self.theta_t
        if theta.ndim != 2:
            raise ValueError(
                f"theta_t must be 2D (B, D), got shape {tuple(theta.shape)}"
            )
        target_dev = device if device is not None else theta.device
        B = theta.shape[0]
        horizon = int(self.raw_info.get("horizon_steps", 0))
        step_one_hot = torch.zeros(
            B, max(1, horizon), device=target_dev, dtype=theta.dtype
        )
        idx = min(self.episode_step, step_one_hot.shape[1] - 1)
        step_one_hot[:, idx] = 1.0
        return torch.cat([theta.to(target_dev), step_one_hot], dim=-1)


class OrdinalEnvBase(ABC):
    """Abstract base for any ordinal-recommendation Gym-style env.

    Lifecycle::

        env = ConcreteEnv(...)
        state = env.reset(seed=0)
        for _ in range(T // K_B):
            action = policy(state)
            state, reward, done, info = env.step(action)
            if done: break

    Subclasses must implement :meth:`reset` and :meth:`step`. The
    ``observation_dim`` and ``action_dim`` properties expose static
    shape information so the policy can be built before the first
    reset.
    """

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> OrdinalState:
        """Start a new episode and return the initial observation.

        The env is responsible for sampling the per-episode probe
        sets, capturing ``theta_0`` and resetting the per-session
        exposure counters.
        """

    @abstractmethod
    def step(
        self, action: Tensor
    ) -> Tuple[OrdinalState, Tensor, bool, Dict[str, Any]]:
        """Advance the env by one batch of ``K_B`` items.

        Args:
            action: ``LongTensor (B, K_B)`` of 1-based item ids.

        Returns:
            ``(next_state, reward, done, info)``. ``reward`` is a
            ``Tensor (B,)``. ``done`` is a python bool that is true at
            the episode horizon. ``info`` carries the four-component
            reward breakdown plus any env-side logging fields.
        """

    # Static shape accessors -------------------------------------------------

    @property
    @abstractmethod
    def observation_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def action_dim(self) -> int:
        ...


__all__ = ["OrdinalEnvBase", "OrdinalState"]
