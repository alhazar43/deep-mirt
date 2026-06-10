"""Environment-side primitives for OrdRec.

The E2 milestone landed the world-model freeze wrapper, the per-item
``(alpha, beta)`` cache and a forward-pass timing harness. E3 adds the
Gym-style :class:`OrdinalEnvBase` ABC, the concrete :class:`OrdRecEnv`
that wraps a frozen MAGPCM and a data adapter, and the three-source
:func:`compose_action_mask` policy mask composer.

See ``docs/ordrec_impl_guide.md`` Section 5.1 for the env-layer spec.
"""

from __future__ import annotations

from .action_mask import (
    build_admin_mask,
    build_probe_mask,
    compose_action_mask,
    update_no_repeat_mask,
)
from .base import OrdinalEnvBase, OrdinalState
from .bench_forward import (
    BenchConfig,
    BenchResult,
    bench_forward_pass,
    no_grad_invariance_check,
    write_bench_artifacts,
)
from .frozen_magpcm import FrozenMAGPCM, MAIRTOutput, freeze_magpcm
from .probe_sampler import (
    ProbeSampler,
    StratifiedProbeSampler,
    UniformProbeSampler,
    make_probe_sampler,
)
from .item_cache import (
    ITEM_CACHE_DTYPE,
    ItemCache,
    build_item_cache,
    checkpoint_sha7,
    config_hash,
    item_cache_path,
    load_item_cache,
    save_item_cache,
    validate_provenance,
    world_model_git_sha_from_repo,
)
from .ordrec_env import OrdRecEnv


__all__ = [
    "BenchConfig",
    "BenchResult",
    "FrozenMAGPCM",
    "ITEM_CACHE_DTYPE",
    "MAIRTOutput",
    "ItemCache",
    "OrdRecEnv",
    "OrdinalEnvBase",
    "OrdinalState",
    "bench_forward_pass",
    "build_admin_mask",
    "build_item_cache",
    "build_probe_mask",
    "checkpoint_sha7",
    "compose_action_mask",
    "config_hash",
    "freeze_magpcm",
    "item_cache_path",
    "load_item_cache",
    "no_grad_invariance_check",
    "save_item_cache",
    "update_no_repeat_mask",
    "ProbeSampler",
    "StratifiedProbeSampler",
    "UniformProbeSampler",
    "make_probe_sampler",
    "validate_provenance",
    "world_model_git_sha_from_repo",
    "write_bench_artifacts",
]
