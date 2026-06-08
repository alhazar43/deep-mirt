# rl/, OrdRec, Active Development Surface

This directory hosts the OrdRec (Ordinal-IRT Exercise Recommendation)
project. It is a clean restart, isolated from the archived
job-recommendation work at `archive/rl_jobrec/`.

## Canonical plan

[`docs/exrec_ordinal_plan.md`](../docs/exrec_ordinal_plan.md).

## Status

- 2026-06-04, plan locked. The directory is intentionally empty pending
  E1 (the Eedi loader and ordinal validation), tracked on a feature
  branch `feat/ordrec-e1-eedi`.

## Isolation rule

Do not import from `archive/rl_jobrec/`. The archived tree is frozen
prior-direction code preserved for traceability. Any pattern from
there must be re-derived clean in this tree, with clear notes on what
was borrowed and why.

## Reusable from outside this directory

- `ma-irt/` provides the deep IRT belief tracker. M1 step API on
  `feat/online-step-api` is the per-step belief surface OrdRec
  consumes.

## Coming with E1

```
rl/
  pyproject.toml
  README.md                          (this file)
  src/
    ordrec/
      __init__.py
      belief/
      retrieval/
      policy/
      env/
      reward/
      training/
  scripts/
    prepare_eedi.py                  (E1)
    validate_eedi_ordering.py        (E1)
  tests/
    test_eedi_loader.py              (E1)
  configs/
    e1_eedi_dev.yaml                 (E1)
  data/                              (gitignored)
  results/                           (gitignored except plots/)
```
