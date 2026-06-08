"""Environment-side primitives for OrdRec.

The E2 milestone lands the world-model freeze wrapper, the per-item
``(alpha, beta)`` cache and a forward-pass timing harness. The Gym-style
``OrdinalEnvBase`` plus ``OrdRecEnv`` arrive in E3 along with the
reward and the policy-side action mask.

See ``docs/ordrec_impl_guide.md`` Section 5.1 for the env-layer spec
and Section 7 for the milestone scope.
"""

from __future__ import annotations

from .frozen_magpcm import FrozenMAGPCM, freeze_magpcm


__all__ = [
    "FrozenMAGPCM",
    "freeze_magpcm",
]
