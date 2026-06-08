"""Environment-side primitives for OrdRec.

The E2 milestone lands the world-model freeze wrapper, the per-item
``(alpha, beta)`` cache and a forward-pass timing harness. The Gym-style
``OrdinalEnvBase`` plus ``OrdRecEnv`` arrive in E3 along with the
reward and the policy-side action mask.

See ``docs/ordrec_impl_guide.md`` Section 5.1 for the env-layer spec
and Section 7 for the milestone scope.
"""

from __future__ import annotations

from .bench_forward import (
    BenchConfig,
    BenchResult,
    bench_forward_pass,
    no_grad_invariance_check,
    write_bench_artifacts,
)
from .frozen_magpcm import FrozenMAGPCM, freeze_magpcm
from .item_cache import (
    ITEM_CACHE_DTYPE,
    ItemCache,
    build_item_cache,
    checkpoint_sha7,
    item_cache_path,
    load_item_cache,
    save_item_cache,
)


__all__ = [
    "BenchConfig",
    "BenchResult",
    "FrozenMAGPCM",
    "ITEM_CACHE_DTYPE",
    "ItemCache",
    "bench_forward_pass",
    "build_item_cache",
    "checkpoint_sha7",
    "freeze_magpcm",
    "item_cache_path",
    "load_item_cache",
    "no_grad_invariance_check",
    "save_item_cache",
    "write_bench_artifacts",
]
