"""Regression test for per-component reward metric averaging.

The rollout loop aggregates per-component reward means (``r_info``,
``r_cost``, ``r_expo``, ``r_voi``) from the env's info dict. The
components can appear at different frequencies, for example ``r_voi``
fires only at the terminal step on some envs. Each reported average
must divide a component's sum by that component's own observation
count, never by a counter shared across components.

The original implementation divided every sum by
``component_count // n_components`` where ``component_count`` was
shared, which inflated the reported averages whenever the per-key
frequencies differed. This test pins the exact per-key arithmetic
with two synthetic components logged at different frequencies.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch import Tensor

from ordrec.training import PPO


class _TwoComponentEnv:
    """Two-step toy env emitting components at different frequencies.

    ``r_info`` is emitted at every step with constant per-row value
    ``2.0``. ``r_voi`` is emitted only at the terminal step with
    constant value ``6.0``. ``r_cost`` and ``r_expo`` are never
    emitted. The exact expected averages are therefore ``2.0``,
    ``6.0``, ``0.0`` and ``0.0``.
    """

    K_B = 1
    B = 1
    n_actions = 2
    obs_dim = 2
    horizon_steps = 2

    def __init__(self) -> None:
        self._step = 0

    @property
    def observation_dim(self) -> int:
        return self.obs_dim

    @property
    def action_dim(self) -> int:
        return self.n_actions

    def reset(self) -> Dict[str, Tensor]:
        self._step = 0
        return self._state()

    def step(
        self, action: Tensor
    ) -> Tuple[Dict[str, Tensor], Tensor, bool, Dict[str, Any]]:
        self._step += 1
        done = self._step >= self.horizon_steps
        reward = torch.tensor([1.0])
        info: Dict[str, Any] = {"r_info": torch.tensor([2.0])}
        if done:
            info["r_voi"] = torch.tensor([6.0])
        return self._state(), reward, done, info

    def _state(self) -> Dict[str, Tensor]:
        obs = torch.zeros(1, self.obs_dim)
        obs[0, min(self._step, self.obs_dim - 1)] = 1.0
        return {
            "obs": obs,
            "action_mask": torch.ones(1, self.n_actions, dtype=torch.bool),
        }


def test_component_averages_use_per_key_counts() -> None:
    """Averages are exact when components fire at different frequencies.

    Four 2-step episodes give 8 ``r_info`` observations (sum 16.0) and
    4 ``r_voi`` observations (sum 24.0). The correct averages are
    exactly 2.0 and 6.0. A shared-counter denominator (the old bug)
    would report 16/3 and 24/3 instead.
    """
    torch.manual_seed(0)
    env = _TwoComponentEnv()
    ppo = PPO(
        observation_dim=env.observation_dim,
        action_dim=env.action_dim,
        seed=0,
        hidden_dim=16,
        n_hidden_layers=1,
        n_episodes_per_update=4,
        max_steps_per_episode=env.horizon_steps,
        minibatch_size=4,
        n_epochs=1,
        total_updates=1,
    )
    stats = ppo.rollout(env, n_episodes=4)
    assert stats.n_transitions == 8
    assert stats.info["r_info"] == 2.0, (
        f"r_info average must be exactly 2.0, got {stats.info['r_info']}"
    )
    assert stats.info["r_voi"] == 6.0, (
        f"r_voi average must be exactly 6.0, got {stats.info['r_voi']}"
    )


def test_unobserved_components_report_zero() -> None:
    """Components that never appear in info must report 0.0, not NaN."""
    torch.manual_seed(0)
    env = _TwoComponentEnv()
    ppo = PPO(
        observation_dim=env.observation_dim,
        action_dim=env.action_dim,
        seed=0,
        hidden_dim=16,
        n_hidden_layers=1,
        n_episodes_per_update=2,
        max_steps_per_episode=env.horizon_steps,
        minibatch_size=4,
        n_epochs=1,
        total_updates=1,
    )
    stats = ppo.rollout(env, n_episodes=2)
    assert stats.info["r_cost"] == 0.0
    assert stats.info["r_expo"] == 0.0
