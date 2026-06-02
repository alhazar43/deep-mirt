# T6 Reproducibility Tests - 2026-06-02

This note records verification for Plan v2 T6, Reproducibility tests.

## Files Changed

- `ma-irt/tests/test_public_pipeline.py`
- `docs/pipeline.md`
- `CLEANUP_VERIFICATION_2026.md`

No core model, data-loading, training, evaluation, plotting, or config YAML
files were intentionally edited in this tier.

## What Changed

Added public pipeline tests that cover:

- `scripts/data_gen.py` emits the documented dataset contract:
  `sequences.json`, `metadata.json`, `true_irt_parameters.json`.
- All public smoke configs load through `config.load_config`.
- A tiny MA-GPCM subprocess path trains for one epoch and then runs
  `evaluate.py single`, proving that `best.pt`, `last.pt`, `metrics.csv`,
  and `recovery_metrics.json` are produced.

The tiny subprocess test uses a unique experiment name and removes its
temporary `outputs/pytest_pipeline_*` directory in a `finally` block.

## Contract Correction

The new test exposed that `train.py` writes `test_metrics.json` only in CV
mode. `docs/pipeline.md` now states:

- non-CV smoke training writes checkpoints and `metrics.csv`;
- `evaluate.py single` writes `recovery_metrics.json`;
- `test_metrics.json` is a CV-mode training artifact.

## Test Commands

New test file only:

```powershell
cd C:\Users\steph\Documents\deep-mirt\ma-irt
$env:PYTHONPATH='.'
pytest tests\test_public_pipeline.py -q
```

Result:

```text
3 passed in 7.95s
```

Full suite:

```powershell
cd C:\Users\steph\Documents\deep-mirt\ma-irt
$env:PYTHONPATH='.'
pytest tests -q
```

Result:

```text
68 passed in 9.58s
```

## Artifact Cleanup

Post-test check:

```powershell
Get-ChildItem outputs -Directory -Filter 'pytest_pipeline_*'
```

Result: no remaining `pytest_pipeline_*` output directories.
