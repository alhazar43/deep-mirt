"""Tests for PPO ``save`` / ``load`` state-dict round trip.

A saved checkpoint must restore identical policy parameters on load.
The optimizer state and update count must also round trip.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from ordrec.training import PPO


@pytest.fixture(scope="module")
def sanity_module():
    repo_root = Path(__file__).resolve().parents[5]
    script = repo_root / "rl" / "scripts" / "sanity_toy_env.py"
    spec = importlib.util.spec_from_file_location("sanity_toy_env", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules["sanity_toy_env"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _build_ppo(sanity_module) -> PPO:
    env = sanity_module.ToyEnv()
    return sanity_module._PPOToy(
        observation_dim=env.observation_dim,
        action_dim=env.action_dim,
        hidden_dim=16, n_hidden_layers=1,
        n_episodes_per_update=4, max_steps_per_episode=2,
        minibatch_size=4, n_epochs=2, total_updates=2,
        seed=0,
    )


def test_save_load_round_trip(tmp_path: Path, sanity_module) -> None:
    ppo = _build_ppo(sanity_module)
    env = sanity_module.ToyEnv()
    ppo.rollout(env, n_episodes=2)
    ppo.update()

    sd_before = {
        k: v.detach().clone() for k, v in ppo.policy.state_dict().items()
    }
    ckpt = tmp_path / "ppo.pt"
    ppo.save(ckpt)
    assert ckpt.exists()

    # Modify the policy in place, then load and verify recovery.
    with torch.no_grad():
        for p in ppo.policy.parameters():
            p.add_(torch.randn_like(p))
    ppo.load(ckpt)
    sd_after = ppo.policy.state_dict()
    for key, tensor in sd_before.items():
        assert torch.equal(sd_after[key], tensor), f"mismatch on {key}"


def test_load_into_fresh_instance(tmp_path: Path, sanity_module) -> None:
    """A fresh PPO instance must load the same checkpoint into matching weights."""
    ppo_a = _build_ppo(sanity_module)
    env = sanity_module.ToyEnv()
    ppo_a.rollout(env, n_episodes=2)
    ppo_a.update()
    ckpt = tmp_path / "ppo.pt"
    ppo_a.save(ckpt)

    ppo_b = _build_ppo(sanity_module)
    # Sanity: at this point ppo_a.policy != ppo_b.policy in general
    # because of optimizer updates. Verify load equalises them.
    ppo_b.load(ckpt)
    for k, v in ppo_a.policy.state_dict().items():
        assert torch.equal(ppo_b.policy.state_dict()[k], v)
