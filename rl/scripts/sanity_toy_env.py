"""PPO smoke runner on a tiny two-state two-action env.

Confirms PPO update mechanics work before plugging into the
OrdRecEnv. The env is a deterministic Bernoulli "always pick action
1 to get a positive reward" task. PPO should drive the mean episode
return monotonically up over 20 updates.

Usage::

    PYTHONPATH="rl/src;ma-irt" KMP_DUPLICATE_LIB_OK=TRUE \
        python rl/scripts/sanity_toy_env.py

Prints the per-update mean episode return. Exit code 0 when the
mean of the last five updates exceeds the mean of the first five
by more than 0.5; nonzero otherwise.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
from torch import Tensor

from ordrec.training import PPO, set_seed


@dataclass
class _ToyState:
    obs: Tensor
    action_mask: Tensor


class ToyEnv:
    """Two-state two-action env. action == 1 rewards +1, action == 0 rewards 0.

    The state is a one-hot of the current step index. Episodes last
    two steps. The optimal return is +2; the random-policy mean
    return is +1.

    Attributes:
        K_B: ``1`` (single-action env), exposed for PPO compatibility.
        B: ``1`` (single concurrent agent).
        horizon_steps: ``2``.
    """

    K_B = 1
    B = 1
    n_actions = 2
    obs_dim = 2
    horizon_steps = 2

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = int(seed)
        self._step = 0

    @property
    def observation_dim(self) -> int:
        return self.obs_dim

    @property
    def action_dim(self) -> int:
        return self.n_actions

    def reset(self, seed: int | None = None) -> _ToyState:
        if seed is not None:
            self.seed = int(seed)
        self._step = 0
        return self._state()

    def step(self, action: Tensor) -> Tuple[_ToyState, Tensor, bool, Dict[str, Any]]:
        a = int(action.reshape(-1)[0].item())
        if a not in (0, 1):
            raise ValueError(f"toy env expects action in {{0, 1}}, got {a}")
        # Both actions are valid; the rewarding action is index 1.
        reward = torch.tensor([1.0 if a == 1 else 0.0])
        self._step += 1
        done = self._step >= self.horizon_steps
        return self._state(), reward, done, {}

    def _state(self) -> _ToyState:
        obs = torch.zeros(1, self.obs_dim)
        obs[0, min(self._step, self.obs_dim - 1)] = 1.0
        mask = torch.ones(1, self.n_actions, dtype=torch.bool)
        return _ToyState(obs=obs, action_mask=mask)


def _state_to_ppo(state: _ToyState) -> Dict[str, Any]:
    return {"obs": state.obs, "action_mask": state.action_mask}


class _PPOToy(PPO):
    """PPO subclass that unpacks the toy env's lightweight state."""

    def _unpack_state(self, state: Any) -> Tuple[Tensor, Tensor]:  # type: ignore[override]
        if isinstance(state, _ToyState):
            return state.obs.to(self.device), state.action_mask.to(self.device)
        return super()._unpack_state(state)


def run(n_updates: int = 20, seed: int = 0) -> list[float]:
    set_seed(seed)
    env = ToyEnv(seed=seed)
    ppo = _PPOToy(
        observation_dim=env.observation_dim,
        action_dim=env.action_dim,
        device=torch.device("cpu"),
        seed=seed,
        hidden_dim=32,
        n_hidden_layers=1,
        n_episodes_per_update=16,
        max_steps_per_episode=2,
        minibatch_size=8,
        n_epochs=4,
        total_updates=n_updates,
        learning_rate=3e-3,
        entropy_coef_initial=0.01,
        entropy_coef_final=0.0,
    )
    returns: list[float] = []
    for it in range(int(n_updates)):
        rstats = ppo.rollout(env, n_episodes=ppo.n_episodes_per_update)
        ppo.update()
        returns.append(rstats.mean_episode_return)
        print(
            f"update={it:3d} mean_return={rstats.mean_episode_return:.3f} "
            f"n_transitions={rstats.n_transitions}",
            flush=True,
        )
    return returns


def main() -> int:
    parser = argparse.ArgumentParser(description="PPO toy-env smoke runner.")
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    returns = run(n_updates=args.updates, seed=args.seed)
    n_window = max(1, len(returns) // 4)
    first_window = sum(returns[:n_window]) / n_window
    last_window = sum(returns[-n_window:]) / n_window
    delta = last_window - first_window
    print(
        f"\nfirst_window_mean={first_window:.3f} "
        f"last_window_mean={last_window:.3f} delta={delta:+.3f}"
    )
    return 0 if delta > 0.3 else 1


if __name__ == "__main__":
    sys.exit(main())
