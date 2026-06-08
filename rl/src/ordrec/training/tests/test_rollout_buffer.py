"""Tests for ``RolloutBuffer``.

Covers insert, reset, mini-batch iteration and the GAE compute
contract. The buffer must visit every transition exactly once during
minibatch iteration.
"""

from __future__ import annotations

import pytest
import torch

from ordrec.training.rollout import RolloutBuffer


def _filled_buffer(
    capacity: int = 4, obs_dim: int = 3, n_actions: int = 5,
    rewards=(0.0, 0.0, 1.0, 0.5),
):
    buf = RolloutBuffer(
        capacity=capacity, observation_dim=obs_dim, n_actions=n_actions,
        gamma=0.9, gae_lambda=0.95,
    )
    for t in range(capacity):
        state = torch.zeros(obs_dim)
        state[t % obs_dim] = 1.0
        action = torch.tensor(t % n_actions, dtype=torch.long)
        mask = torch.ones(n_actions, dtype=torch.bool)
        mask[0] = False
        buf.insert(
            state=state, action=action,
            reward=rewards[t], log_prob=-0.5, value=0.1 * t,
            done=(t == capacity - 1),
            action_mask=mask,
            episode_start=(t == 0),
        )
    return buf


def test_insert_and_reset() -> None:
    buf = _filled_buffer()
    assert buf.size == 4
    assert buf.full is True
    buf.reset()
    assert buf.size == 0
    assert buf.full is False
    # Reuse after reset is allowed.
    buf.insert(
        state=torch.zeros(3), action=torch.tensor(0, dtype=torch.long),
        reward=0.0, log_prob=0.0, value=0.0, done=True,
        action_mask=torch.ones(5, dtype=torch.bool),
        episode_start=True,
    )
    assert buf.size == 1


def test_insert_rejects_wrong_shapes() -> None:
    buf = RolloutBuffer(capacity=2, observation_dim=3, n_actions=4)
    with pytest.raises(ValueError):
        buf.insert(
            state=torch.zeros(5),  # wrong obs dim
            action=torch.tensor(0, dtype=torch.long),
            reward=0.0, log_prob=0.0, value=0.0, done=False,
            action_mask=torch.ones(4, dtype=torch.bool),
        )
    with pytest.raises(ValueError):
        buf.insert(
            state=torch.zeros(3),
            action=torch.tensor(0, dtype=torch.long),
            reward=0.0, log_prob=0.0, value=0.0, done=False,
            action_mask=torch.ones(7, dtype=torch.bool),  # wrong mask size
        )


def test_overflow_raises() -> None:
    buf = _filled_buffer(capacity=2, rewards=(0.0, 1.0))
    with pytest.raises(IndexError):
        buf.insert(
            state=torch.zeros(3), action=torch.tensor(0, dtype=torch.long),
            reward=0.0, log_prob=0.0, value=0.0, done=False,
            action_mask=torch.ones(5, dtype=torch.bool),
        )


def test_iter_minibatches_visits_every_transition_once() -> None:
    buf = _filled_buffer(capacity=8, rewards=(0.1,) * 8)
    buf.compute_advantages()
    seen = set()
    n_batches = 0
    for batch in buf.iter_minibatches(minibatch_size=3, shuffle=True):
        n_batches += 1
        for action in batch.actions.tolist():
            seen.add(int(action))
    # All four unique actions in {0..4} appeared. With capacity=8 and
    # action = t % 5, we expect actions {0, 1, 2, 3, 4}.
    assert seen == {0, 1, 2, 3, 4}
    assert n_batches == 3  # ceil(8 / 3)


def test_iter_minibatches_requires_compute_advantages() -> None:
    buf = _filled_buffer()
    with pytest.raises(RuntimeError):
        for _ in buf.iter_minibatches(minibatch_size=2):
            pass


def test_compute_advantages_zero_bootstrap_at_done() -> None:
    """When the buffer ends with done=True the bootstrap must be 0."""
    buf = _filled_buffer(
        capacity=2,
        rewards=(0.0, 1.0),
    )
    buf.compute_advantages(last_value=100.0)  # huge bootstrap, should be ignored
    # Last value used in delta_T = r_T + gamma * 0 - V_T; advantage is
    # delta_T directly because there is no future. r_1 = 1.0, V_1 = 0.1.
    assert torch.isclose(buf.advantages[1], torch.tensor(1.0 - 0.1), atol=1e-6)


def test_advantage_normalization_off_when_n_eq_1() -> None:
    """A buffer with a single transition must not normalise advantages."""
    buf = RolloutBuffer(capacity=1, observation_dim=2, n_actions=3)
    buf.insert(
        state=torch.zeros(2), action=torch.tensor(0, dtype=torch.long),
        reward=1.0, log_prob=0.0, value=0.0, done=True,
        action_mask=torch.ones(3, dtype=torch.bool), episode_start=True,
    )
    buf.compute_advantages(last_value=0.0)
    batches = list(buf.iter_minibatches(minibatch_size=1, shuffle=False,
                                        normalize_advantages=True))
    assert len(batches) == 1
    # Advantage equals delta_0 = 1.0 - 0.0 = 1.0; no /0 standardisation.
    assert torch.isclose(batches[0].advantages, torch.tensor([1.0]), atol=1e-6)


def test_advantage_sign_matches_reward() -> None:
    """A trajectory with one positive reward must produce positive A on that step."""
    buf = _filled_buffer(capacity=4, rewards=(0.0, 0.0, 1.0, 0.0))
    buf.compute_advantages()
    assert buf.advantages[2].item() > 0


def test_episode_starts_split_segments() -> None:
    """Two episodes in the buffer use two independent GAE segments."""
    buf = RolloutBuffer(capacity=4, observation_dim=2, n_actions=3,
                        gamma=0.9, gae_lambda=0.95)
    # Episode 1, length 2 ending with done.
    buf.insert(state=torch.zeros(2), action=torch.tensor(0, dtype=torch.long),
               reward=0.0, log_prob=0.0, value=0.0, done=False,
               action_mask=torch.ones(3, dtype=torch.bool), episode_start=True)
    buf.insert(state=torch.zeros(2), action=torch.tensor(1, dtype=torch.long),
               reward=1.0, log_prob=0.0, value=0.0, done=True,
               action_mask=torch.ones(3, dtype=torch.bool), episode_start=False)
    # Episode 2, length 2.
    buf.insert(state=torch.zeros(2), action=torch.tensor(0, dtype=torch.long),
               reward=0.0, log_prob=0.0, value=0.0, done=False,
               action_mask=torch.ones(3, dtype=torch.bool), episode_start=True)
    buf.insert(state=torch.zeros(2), action=torch.tensor(2, dtype=torch.long),
               reward=-1.0, log_prob=0.0, value=0.0, done=True,
               action_mask=torch.ones(3, dtype=torch.bool), episode_start=False)
    buf.compute_advantages()
    # Episodes are independent: the negative reward in episode 2 must
    # not bleed back to episode 1.
    assert buf.advantages[1].item() == pytest.approx(1.0, abs=1e-6)
    assert buf.advantages[3].item() == pytest.approx(-1.0, abs=1e-6)
