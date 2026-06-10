"""Regression tests for the B0+B4 buffer rework.

Verifies that the buffer capacity accounting is exact (one entry per
env-step per episode row, not one per sub-step item) and that terminal
entries containing ``r_voi`` actually enter the buffer.

Root cause of E4.5 failure: ``_insert_transition`` was writing
``B * K_B = 160`` entries per env step into a buffer of capacity 64.
The buffer filled on the first env step so terminal rewards (the only
place ``r_voi`` fires) were dropped. These tests pin the corrected
behaviour.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ordrec.training.ppo import PPO
from ordrec.training.rollout import RolloutBuffer


# ---------------------------------------------------------------------------
# Buffer construction
# ---------------------------------------------------------------------------


def test_buffer_capacity_exact_with_kB_1() -> None:
    """K_B=1: capacity == n_eps * max_steps, unchanged."""
    ppo = PPO(
        observation_dim=4, action_dim=10,
        n_episodes_per_update=32, max_steps_per_episode=2,
        seed=0,
    )
    assert ppo.buffer.capacity == 64


def test_buffer_kB_gt1_action_shape() -> None:
    """K_B=5: action storage is 2D (capacity, K_B)."""
    buf = RolloutBuffer(capacity=64, observation_dim=4, n_actions=201, K_B=5)
    assert buf.actions.ndim == 2
    assert buf.actions.shape == (64, 5)


def test_buffer_kB_1_action_shape() -> None:
    """K_B=1: action storage is 1D (capacity,) for backward compat."""
    buf = RolloutBuffer(capacity=64, observation_dim=4, n_actions=201, K_B=1)
    assert buf.actions.ndim == 1
    assert buf.actions.shape == (64,)


# ---------------------------------------------------------------------------
# Insertion counting
# ---------------------------------------------------------------------------


class _BatchedEnv:
    """Tiny batched env with configurable B, K_B and horizon_steps.

    Emits a reward of 0.0 at non-terminal steps and ``r_voi=1.0`` at
    the terminal step.  The action mask allows any action in
    [1, n_actions - 1].
    """

    def __init__(self, B: int = 4, K_B: int = 5, horizon_steps: int = 2,
                 n_actions: int = 201) -> None:
        self.B = B
        self.K_B = K_B
        self.horizon_steps = horizon_steps
        self.n_actions = n_actions
        self._step = 0
        self.obs_dim = 3

    @property
    def observation_dim(self) -> int:
        return self.obs_dim

    @property
    def action_dim(self) -> int:
        return self.n_actions

    def reset(self) -> dict:
        self._step = 0
        return self._state()

    def step(self, action: Tensor):
        self._step += 1
        done = self._step >= self.horizon_steps
        # Terminal step emits r_voi=1.0 so we can verify it lands in buffer.
        r_voi = torch.ones(self.B) if done else torch.zeros(self.B)
        reward = r_voi  # (B,)
        info = {
            "r_info": torch.zeros(self.B),
            "r_cost": torch.zeros(self.B),
            "r_expo": torch.zeros(self.B),
            "r_voi": r_voi,
        }
        return self._state(), reward, done, info

    def _state(self) -> dict:
        obs = torch.zeros(self.B, self.obs_dim)
        obs[:, min(self._step, self.obs_dim - 1)] = 1.0
        mask = torch.ones(self.B, self.n_actions, dtype=torch.bool)
        mask[:, 0] = False  # pad slot forbidden
        return {"obs": obs, "action_mask": mask}


def test_rollout_32_eps_2_steps_fills_buffer_exactly() -> None:
    """32 episodes x 2 steps fills to exactly 64 entries (B=1 env row)."""
    torch.manual_seed(0)
    B, K_B, n_eps, max_steps = 1, 5, 32, 2
    env = _BatchedEnv(B=B, K_B=K_B, horizon_steps=max_steps, n_actions=21)
    ppo = PPO(
        observation_dim=env.observation_dim,
        action_dim=env.action_dim,
        n_episodes_per_update=n_eps,
        max_steps_per_episode=max_steps,
        seed=0,
        hidden_dim=16, n_hidden_layers=1,
        minibatch_size=8,
    )
    stats = ppo.rollout(env, n_episodes=n_eps)
    # Capacity = n_eps * max_steps = 64.  Buffer should be exactly full.
    assert ppo.buffer.capacity == n_eps * max_steps == 64
    assert ppo.buffer.size == 64, (
        f"Expected 64 entries, got {ppo.buffer.size}"
    )
    assert stats.n_transitions == 64


def test_terminal_r_voi_in_buffer_rewards() -> None:
    """Terminal entries (done=True) must carry the non-zero r_voi reward.

    With the old B*K_B-entries-per-step bug the buffer filled before
    the terminal step so r_voi was always 0 in the buffer.
    """
    torch.manual_seed(0)
    B, K_B, n_eps, max_steps = 1, 5, 4, 2
    env = _BatchedEnv(B=B, K_B=K_B, horizon_steps=max_steps, n_actions=21)
    ppo = PPO(
        observation_dim=env.observation_dim,
        action_dim=env.action_dim,
        n_episodes_per_update=n_eps,
        max_steps_per_episode=max_steps,
        seed=0,
        hidden_dim=16, n_hidden_layers=1,
        minibatch_size=4,
    )
    ppo.rollout(env, n_episodes=n_eps)
    # Terminal entries are those with done=True.  The toy env emits
    # reward=1.0 (from r_voi) at the terminal step, so terminal rewards
    # must be 1.0 in the buffer.
    done_mask = ppo.buffer.dones[: ppo.buffer.size]
    assert done_mask.any(), "No terminal entries in buffer."
    terminal_rewards = ppo.buffer.rewards[: ppo.buffer.size][done_mask]
    assert (terminal_rewards > 0.0).all(), (
        f"Terminal rewards should be 1.0 (r_voi), got {terminal_rewards.tolist()}"
    )


def test_rollout_b4_capacity_matches_n_eps_times_steps() -> None:
    """B=4, K_B=5, n_eps=8, max_steps=2 -> capacity=16, buffer fills to 16.

    The capacity is n_episodes_per_update * max_steps_per_episode = 16.
    Each outer-loop episode call inserts B=4 entries per step (B rows
    per env step), so the buffer fills after 16 / 4 / 2 = 2 outer
    episodes. The rollout exits early via the buffer.full guard and
    total entries is exactly 16.
    """
    torch.manual_seed(0)
    B, K_B, max_steps, n_eps = 4, 5, 2, 8
    env = _BatchedEnv(B=B, K_B=K_B, horizon_steps=max_steps, n_actions=21)
    ppo = PPO(
        observation_dim=env.observation_dim,
        action_dim=env.action_dim,
        n_episodes_per_update=n_eps,
        max_steps_per_episode=max_steps,
        seed=0,
        hidden_dim=16, n_hidden_layers=1,
        minibatch_size=4,
    )
    ppo.rollout(env, n_episodes=n_eps)
    # Capacity = 8 * 2 = 16.  B=4 rows per step, 2 steps per ep -> 8 rows per
    # ep.  Buffer fills after 2 outer episodes (16 rows).
    assert ppo.buffer.capacity == n_eps * max_steps == 16
    assert ppo.buffer.size == 16, (
        f"Expected 16 entries, got {ppo.buffer.size}"
    )


def test_action_shape_in_buffer_for_kB5() -> None:
    """After rollout with K_B=5, buffer.actions is (size, 5)."""
    torch.manual_seed(0)
    B, K_B, max_steps = 2, 5, 2
    env = _BatchedEnv(B=B, K_B=K_B, horizon_steps=max_steps, n_actions=21)
    ppo = PPO(
        observation_dim=env.observation_dim,
        action_dim=env.action_dim,
        n_episodes_per_update=1,
        max_steps_per_episode=max_steps,
        seed=0,
        hidden_dim=16, n_hidden_layers=1,
        minibatch_size=4,
    )
    ppo.rollout(env, n_episodes=1)
    size = ppo.buffer.size
    assert ppo.buffer.actions.shape == (ppo.buffer.capacity, K_B), (
        f"Expected actions shape ({ppo.buffer.capacity}, {K_B}), "
        f"got {tuple(ppo.buffer.actions.shape)}"
    )
    # All stored actions should be legal (in [1, n_actions-1]).
    stored = ppo.buffer.actions[:size]
    assert (stored >= 1).all() and (stored < env.n_actions).all()


def test_update_runs_after_kB5_rollout() -> None:
    """PPO update must complete without error when K_B=5."""
    torch.manual_seed(0)
    B, K_B, max_steps = 2, 5, 2
    env = _BatchedEnv(B=B, K_B=K_B, horizon_steps=max_steps, n_actions=21)
    ppo = PPO(
        observation_dim=env.observation_dim,
        action_dim=env.action_dim,
        n_episodes_per_update=4,
        max_steps_per_episode=max_steps,
        seed=0,
        hidden_dim=16, n_hidden_layers=1,
        minibatch_size=4, n_epochs=2,
    )
    ppo.rollout(env, n_episodes=4)
    stats = ppo.update()
    assert stats.n_grad_steps > 0
    import math
    assert math.isfinite(stats.policy_loss)
    assert math.isfinite(stats.value_loss)
