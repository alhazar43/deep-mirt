"""``RLAlgorithm`` ABC plus ``RolloutStats`` and ``UpdateStats`` dataclasses.

The abstract base captures the minimal surface every RL algorithm in
OrdRec needs to expose to the training loop and evaluation harness. The
buffer is owned by the algorithm because PPO and DQN need fundamentally
different buffer semantics (on-policy ``RolloutBuffer`` versus
off-policy ``ReplayBuffer``).

See ``docs/ordrec_impl_guide.md`` Sections 4.1 and 4.2 for the
contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn


@dataclass
class RolloutStats:
    """Summary of one ``rollout(env, n_episodes)`` call.

    Attributes:
        mean_episode_return: Per-row episode return averaged over the
            sampled episodes. The buffer logs per-trajectory returns so
            the algorithm can compute this without re-stepping.
        mean_episode_length: Average number of ``step`` calls per
            episode. For OrdRec this is constant ``T // K_B`` in v1 but
            kept here for the variable-horizon future.
        n_transitions: Total number of transitions written to the
            buffer.
        info: Free-form dict for per-component reward averages
            (``r_info``, ``r_cost``, ``r_expo``, ``r_voi``) and any
            algorithm-specific diagnostics.
    """

    mean_episode_return: float
    mean_episode_length: float
    n_transitions: int
    info: Dict[str, float] = field(default_factory=dict)


@dataclass
class UpdateStats:
    """Summary of one ``update(buffer)`` call.

    Attributes:
        policy_loss: Mean clipped-surrogate policy loss across all
            mini-batch gradient steps actually run.
        value_loss: Mean value-clipped MSE.
        entropy: Mean policy entropy across mini-batches.
        approx_kl: Mean ``KL(pi_old || pi_new)`` proxy, ``mean(old_lp
            - new_lp)``. Used by the KL early-stop check.
        clipfrac: Fraction of samples for which the ratio fell outside
            ``[1 - eps, 1 + eps]``. Schulman 2017 diagnostic.
        n_grad_steps: Number of mini-batch updates that ran before any
            early-stop trigger.
        extras: Free-form dict for additional algorithm-specific
            diagnostics (e.g. ``early_stop=True`` flag).
    """

    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clipfrac: float
    n_grad_steps: int
    extras: Dict[str, Any] = field(default_factory=dict)


class RLAlgorithm(ABC):
    """Abstract base for every OrdRec RL algorithm.

    The interface is deliberately small. Pedagogy follows CleanRL
    (Huang et al. 2022) and Spinning Up (Achiam 2018): the only shared
    surface is ``rollout``, ``update``, ``act``, ``save`` and ``load``.
    Each concrete algorithm owns its own buffer (PPO uses an on-policy
    ``RolloutBuffer``, DQN/SAC would use a ``ReplayBuffer``) so the
    base class refuses to commit to a buffer type.

    Attributes (set by subclasses):
        policy: ``nn.Module`` whose parameters the algorithm trains.
        device: Torch device on which the policy lives.
    """

    policy: nn.Module

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        device: Optional[torch.device] = None,
        seed: int = 0,
    ) -> None:
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.seed = int(seed)

    # ------------------------------------------------------------------
    # Abstract surface
    # ------------------------------------------------------------------

    @abstractmethod
    def rollout(self, env: Any, n_episodes: int) -> RolloutStats:
        """Drive ``env.reset``/``env.step`` and write to ``self.buffer``."""

    @abstractmethod
    def update(self, buffer: Any = None) -> UpdateStats:
        """Run one gradient-update cycle on ``self.buffer``."""

    @abstractmethod
    def act(
        self, state: Any, *, deterministic: bool = False
    ) -> Tuple[Any, Dict[str, Any]]:
        """Select an action from a state. Returns ``(action, extras)``."""

    # ------------------------------------------------------------------
    # Save / load default implementation
    # ------------------------------------------------------------------

    def save(self, path: Path) -> Path:
        """Persist the policy state-dict plus an algorithm tag.

        Subclasses should call ``super().save(path)`` then extend the
        archive if they have additional state (e.g. value-head buffers,
        optimizer state).
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": type(self).__name__,
                "policy_state_dict": self.policy.state_dict(),
                "observation_dim": self.observation_dim,
                "action_dim": self.action_dim,
                "seed": self.seed,
            },
            p,
        )
        return p

    def load(self, path: Path) -> None:
        """Restore the policy state-dict from ``path``.

        Performs a strict shape/key match. Subclasses may override to
        restore optimizer state or any additional buffers they persist.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"No checkpoint at {p}")
        ckpt = torch.load(p, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy_state_dict"])


__all__ = ["RLAlgorithm", "RolloutStats", "UpdateStats"]
