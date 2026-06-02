# T2 Pipeline Documentation - 2026-06-02

This note records verification for Plan v2 T2, Scientific pipeline
documentation.

## Files Changed

- `docs/pipeline.md`

The change is documentation-only. No core model, training, data-loading,
evaluation, plotting, or config files were intentionally edited in this tier.

## What Changed

`docs/pipeline.md` now documents the executable MA-GPCM research contract:

- environment and `PYTHONPATH` requirements
- dataset layout and sequence schema
- loader padding, masking, train/test split, and CV split behavior
- config overlay behavior and CLI overrides
- training flow and artifact names
- model-family prediction/recovery semantics
- single-checkpoint evaluation and recovery outputs
- paper reproduction flow and key orchestrators
- cleanup artifact policy for temporary smoke runs

## Reference Verification

Command:

```powershell
$paths = @(
  'ma-irt/scripts/data_gen.py',
  'ma-irt/scripts/train.py',
  'ma-irt/scripts/evaluate.py',
  'ma-irt/scripts/_run_k4_cv_recovery.sh',
  'ma-irt/scripts/_run_k356_cv_recovery.sh',
  'ma-irt/scripts/_run_pykt_sweep.sh',
  'ma-irt/scripts/_aggregate_pykt_results.py',
  'ma-irt/scripts/aggregate_recovery_v5.py',
  'ma-irt/scripts/mirt_baseline_all_k.R',
  'ma-irt/scripts/mirt_predict.R',
  'ma-irt/configs/smoke.yaml',
  'CLEANUP_VERIFICATION_2026.md'
)
foreach ($p in $paths) {
  if (-not (Test-Path $p)) { Write-Output "MISSING $p" } else { Write-Output "OK $p" }
}
```

Result:

```text
OK ma-irt/scripts/data_gen.py
OK ma-irt/scripts/train.py
OK ma-irt/scripts/evaluate.py
OK ma-irt/scripts/_run_k4_cv_recovery.sh
OK ma-irt/scripts/_run_k356_cv_recovery.sh
OK ma-irt/scripts/_run_pykt_sweep.sh
OK ma-irt/scripts/_aggregate_pykt_results.py
OK ma-irt/scripts/aggregate_recovery_v5.py
OK ma-irt/scripts/mirt_baseline_all_k.R
OK ma-irt/scripts/mirt_predict.R
OK ma-irt/configs/smoke.yaml
OK CLEANUP_VERIFICATION_2026.md
```

Stale/mojibake reference scan:

```powershell
rg -n "鈥|鈹|胃|伪|尾|卤|脳|v2_q200_k4|run_all_experiments|plot_recovery.py|plot_recovery_figure.py" README.md ma-irt\README.md docs\pipeline.md
```

Result: no matches.

## Unit Tests

Command:

```powershell
cd C:\Users\steph\Documents\deep-mirt\ma-irt
$env:PYTHONPATH='.'
pytest tests -q
```

Result:

```text
65 passed in 3.19s
```
