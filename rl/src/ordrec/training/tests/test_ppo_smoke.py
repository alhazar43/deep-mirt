"""PPO smoke test on a tiny two-state two-action env.

Confirms PPO update mechanics by checking that mean episode return
increases between the first window of 5 updates and the last window
of 5 updates over a 20-update run. The toy env rewards action 1 and
penalises nothing, so the optimum is +2 per episode and the random
baseline is +1.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def sanity_module():
    """Load ``rl/scripts/sanity_toy_env.py`` as a module under tests."""
    repo_root = Path(__file__).resolve().parents[5]
    script = repo_root / "rl" / "scripts" / "sanity_toy_env.py"
    spec = importlib.util.spec_from_file_location(
        "sanity_toy_env", script,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["sanity_toy_env"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_ppo_increases_return_on_toy_env(sanity_module) -> None:
    returns = sanity_module.run(n_updates=20, seed=0)
    assert len(returns) == 20
    first = sum(returns[:5]) / 5.0
    last = sum(returns[-5:]) / 5.0
    assert last > first + 0.3, (
        f"PPO failed smoke. first_5_mean={first:.3f} last_5_mean={last:.3f}"
    )


def test_ppo_act_deterministic_argmax(sanity_module) -> None:
    """``act(deterministic=True)`` must return the argmax under the mask."""
    import torch

    env = sanity_module.ToyEnv(seed=42)
    ppo = sanity_module._PPOToy(
        observation_dim=env.observation_dim,
        action_dim=env.action_dim,
        hidden_dim=16, n_hidden_layers=1,
        n_episodes_per_update=4, max_steps_per_episode=2,
        minibatch_size=4, n_epochs=2, total_updates=2,
    )
    state = env.reset()
    action_det, extras = ppo.act(state, deterministic=True)
    # masked argmax under all-true mask is the plain argmax of the logits
    with torch.no_grad():
        logits, _ = ppo.policy(state.obs.to(ppo.device))
    expected = torch.argmax(logits, dim=-1)
    assert torch.equal(action_det.cpu(), expected.cpu())


def test_ppo_mask_respected(sanity_module) -> None:
    """1000 samples under a mask permitting one action must all equal it."""
    import torch

    env = sanity_module.ToyEnv(seed=7)
    ppo = sanity_module._PPOToy(
        observation_dim=env.observation_dim,
        action_dim=env.action_dim,
        hidden_dim=16, n_hidden_layers=1,
        n_episodes_per_update=4, max_steps_per_episode=2,
        minibatch_size=4, n_epochs=2, total_updates=2,
    )
    state = env.reset()
    # Build a state mask that only permits action 0.
    forced_mask = torch.tensor([[True, False]], dtype=torch.bool)
    forced_state = sanity_module._ToyState(obs=state.obs, action_mask=forced_mask)
    for _ in range(1000):
        action, _ = ppo.act(forced_state, deterministic=False)
        assert int(action.item()) == 0
