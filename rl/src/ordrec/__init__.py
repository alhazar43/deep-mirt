"""OrdRec, ordinal item recommendation built on deep MA-IRT.

This package is the active development surface for the OrdRec project.
The E1 milestone landed the data layer (see ``ordrec.data``). E2 added
the frozen world-model wrapper, the per-item ``(alpha, beta)`` cache
and a forward-pass timing harness (see ``ordrec.envs``). E3 adds the
Gym-style env contract and the four-component reward composer
(see ``ordrec.envs.OrdRecEnv`` and ``ordrec.reward``). The PPO
training loop arrives in E4.
"""

from __future__ import annotations

__version__ = "0.0.3"

# Re-export the three subpackages so ``import ordrec`` is enough at the
# call site. We do not eagerly pull every symbol up to the package root
# to keep import-time cost low for callers that only need one of the
# layers.
from . import data, envs, reward

__all__ = ["data", "envs", "reward"]
