# T7 Refactor Readiness - 2026-06-02

This note records a markdown-only adjustment to Plan v2 before any cleanup
refactor touches core model, training, config, or metric code.

## Reason

The cleanup plan originally listed T7 as an immediate code architecture
refactor. A status scan showed pre-existing uncommitted edits in core pipeline
files, including:

- `ma-irt/scripts/train.py`
- `ma-irt/config/loader.py`
- `ma-irt/config/types.py`
- `ma-irt/models/__init__.py`
- `ma-irt/dataloading/loaders.py`
- `ma-irt/models/components/irt.py`
- `ma-irt/utils/metrics.py`

These edits may affect CV behavior, early stopping, model exports, data
loading, IRT component semantics, and metric behavior. Cleanup-owned code
refactors should not overwrite or silently absorb them.

## Markdown Changes

- Updated `CLEANUP_PLAN_2026.md` so T6 distinguishes completed public pipeline
  tests from still-pending model output and CLI ergonomics tests.
- Changed T7 from immediate code refactor to architecture refactor readiness.
- Added a T7 guardrail requiring the dirty core edits to be reconciled or
  explicitly incorporated before cleanup-owned source refactor commits.
- Tightened T8 so artifact hygiene starts with generated runtime artifacts and
  avoids deleting `ma-irt/outputs/` wholesale.

## Verification

No source code, config YAML, model, training, data-loading, evaluation, or
plotting files were intentionally edited in this tier.

Because this tier is markdown-only, no pytest run is required for behavioral
verification. The previous full test gate remains:

```text
68 passed in 9.58s
```
