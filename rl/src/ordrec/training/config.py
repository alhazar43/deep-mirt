"""``PPOConfig`` dataclass for the OrdRec PPO trainer.

Centralises every numeric knob passed to :class:`~ordrec.training.ppo.PPO`
so the training script can be constructed from a single config object
loaded from YAML rather than scattering keyword arguments across
``_build_ppo`` helper functions.

See ``docs/ordrec_impl_guide.md`` Section 4.4 for the hyperparameter
table and rationale behind each default.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class PPOConfig:
    """Frozen configuration for :class:`~ordrec.training.ppo.PPO`.

    Attributes:
        observation_dim: Policy input dimension. Set at build time from
            the env's observation space.
        action_dim: ``Q + 1`` item bank size plus padding slot.
        hidden_dim: Actor-critic trunk width. Default 128.
        n_hidden_layers: Number of trunk hidden layers. Default 2.
        learning_rate: Adam learning rate. Default ``3e-4``.
        adam_eps: Adam epsilon. Default ``1e-5``.
        clip_eps: Policy and value clip range. Default ``0.2``.
        gamma: GAE discount factor. Default ``0.95``.
        gae_lambda: GAE lambda. Default ``0.95``.
        entropy_coef_initial: Starting entropy coefficient ``ent0``.
        entropy_coef_final: Final entropy coefficient ``ent_T``.
        entropy_anneal_fraction: Fraction of ``total_updates`` over
            which entropy linearly decays. Default ``0.5``.
        value_coef: Weight on the value loss. Default ``0.5``.
        max_grad_norm: Gradient clipping norm. Default ``0.5``.
        n_epochs: Update epochs per rollout. Default ``4``.
        minibatch_size: PPO mini-batch size. Default ``32``.
        target_kl: KL early-stop threshold. Default ``0.02``.
        n_episodes_per_update: Episodes collected before each update.
            Default ``32``.
        max_steps_per_episode: Horizon in env steps used to size the
            rollout buffer. Equals ``T / K_B``. Default ``2``.
        total_updates: Total number of ``update`` calls for the entropy
            anneal schedule. Default ``1000``.
        seed: RNG seed for action sampling. Default ``0``.
    """

    # Network
    observation_dim: int = 0
    action_dim: int = 0
    hidden_dim: int = 128
    n_hidden_layers: int = 2

    # Optimiser
    learning_rate: float = 3e-4
    adam_eps: float = 1e-5

    # PPO objective
    clip_eps: float = 0.2
    gamma: float = 0.95
    gae_lambda: float = 0.95
    entropy_coef_initial: float = 0.01
    entropy_coef_final: float = 0.0
    entropy_anneal_fraction: float = 0.5
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.02

    # Rollout geometry
    n_epochs: int = 4
    minibatch_size: int = 32
    n_episodes_per_update: int = 32
    max_steps_per_episode: int = 2
    total_updates: int = 1000

    # Misc
    seed: int = 0

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PPOConfig":
        """Construct from a plain dict, ignoring unknown keys.

        Only the keys that match :class:`PPOConfig` field names are
        forwarded; extra keys (from a broader YAML section) are silently
        dropped. Missing keys fall back to the dataclass defaults.

        Args:
            d: Mapping from field name to value. Typically the ``ppo``
               sub-dict loaded from a YAML config file.

        Returns:
            A frozen :class:`PPOConfig` instance.

        Example::

            cfg = PPOConfig.from_dict(yaml_data.get("ppo", {}))
        """
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


__all__ = ["PPOConfig"]
