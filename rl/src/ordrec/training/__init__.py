"""Training package for OrdRec, RL algorithm library.

E4 ships ``base`` (the ``RLAlgorithm`` ABC plus stats dataclasses),
``rollout`` (the on-policy buffer), ``gae`` (GAE with done masking),
``utils`` (seeding and schedule helpers) and ``ppo`` (the only
concrete algorithm at v1). DQN and SAC live as sketches inside
``ppo.py`` per the impl guide and will land in later experiments.

See ``docs/ordrec_impl_guide.md`` Sections 4.1 to 4.4.
"""

from __future__ import annotations

from .base import RLAlgorithm, RolloutStats, UpdateStats
from .gae import compute_gae
from .ppo import PPO, ActorCritic
from .rollout import RolloutBatch, RolloutBuffer
from .utils import (
    cosine_anneal,
    linear_anneal,
    pick_device,
    polyak_update,
    set_seed,
)


__all__ = [
    "ActorCritic",
    "PPO",
    "RLAlgorithm",
    "RolloutBatch",
    "RolloutBuffer",
    "RolloutStats",
    "UpdateStats",
    "compute_gae",
    "cosine_anneal",
    "linear_anneal",
    "pick_device",
    "polyak_update",
    "set_seed",
]
