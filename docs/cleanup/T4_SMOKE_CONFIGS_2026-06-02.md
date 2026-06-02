# T4 Smoke Configs - 2026-06-02

This note records verification for Plan v2 T4, Canonical smoke and CLI
ergonomics.

## Files Changed

- `ma-irt/configs/smoke_dkvmn_gpcm.yaml`
- `ma-irt/configs/smoke_dkvmn_softmax.yaml`
- `ma-irt/configs/smoke_static_gpcm.yaml`
- `ma-irt/configs/smoke_dynamic_gpcm.yaml`
- `ma-irt/README.md`

No core model, data-loading, training, or evaluation source files were
intentionally edited in this tier.

## What Changed

Added small CPU smoke configs for the ordinal model family paths that were
missing from the public smoke surface:

- DKVMN+GPCM ablation: `magpcm` with `separate_theta: false`
- DKVMN+Softmax
- Static GPCM
- Dynamic GPCM

The existing smoke configs already cover MA-GPCM and the binary DKT/DKVMN/
Deep-IRT baselines.

## Ordinal Smoke Dataset

Command:

```powershell
python scripts\data_gen.py --name smoke_test --n_students 120 --n_questions 20 --n_cats 4 --min_seq 10 --max_seq 25 --output_dir data --seed 42
```

Result:

```text
Wrote 120 sequences -> data\smoke_test\sequences.json
Wrote metadata       -> data\smoke_test\metadata.json
Wrote IRT parameters -> data\smoke_test\true_irt_parameters.json
```

## New Ordinal Smoke Config Training

Command:

```powershell
$env:PYTHONPATH='.'
$configs = @(
  'configs\smoke_dkvmn_gpcm.yaml',
  'configs\smoke_dkvmn_softmax.yaml',
  'configs\smoke_static_gpcm.yaml',
  'configs\smoke_dynamic_gpcm.yaml'
)
foreach ($cfg in $configs) {
  python scripts\train.py --config $cfg --epochs 1
}
```

Results:

```text
smoke_dkvmn_gpcm:    parameters=24901, best_qwk=0.1237
smoke_dkvmn_softmax: parameters=22756, best_qwk=0.1506
smoke_static_gpcm:   parameters=205,   best_qwk=-0.2445
smoke_dynamic_gpcm:  parameters=1359,  best_qwk=0.0305
```

These one-epoch results are only path checks, not performance targets.

## IRT Smoke Evaluation

Command:

```powershell
$env:PYTHONPATH='.'
python scripts\evaluate.py single --config configs\smoke_dkvmn_gpcm.yaml --checkpoint outputs\smoke_dkvmn_gpcm\best.pt --data-dir data\smoke_test --batch-size 32
python scripts\evaluate.py single --config configs\smoke_static_gpcm.yaml --checkpoint outputs\smoke_static_gpcm\best.pt --data-dir data\smoke_test --batch-size 32
python scripts\evaluate.py single --config configs\smoke_dynamic_gpcm.yaml --checkpoint outputs\smoke_dynamic_gpcm\best.pt --data-dir data\smoke_test --batch-size 32
```

Results:

```text
smoke_dkvmn_gpcm:   ACC=0.1749, QWK=0.1237, r_theta=0.8462, recovery_metrics.json written
smoke_static_gpcm:  ACC=0.1659, QWK=-0.2445, r_theta=0.8763, recovery_metrics.json written
smoke_dynamic_gpcm: ACC=0.2152, QWK=0.0305, r_theta=0.0714, recovery_metrics.json written
```

## Binary Smoke Check

`outputs/smoke_dkt` already existed, so it was not overwritten. Instead, a
temporary K=2 dataset was generated and used with `--dataset`, which routes
artifacts to a temporary output directory.

Commands:

```powershell
python scripts\data_gen.py --name cleanup_t4_bin_codex --n_students 120 --n_questions 20 --n_cats 2 --min_seq 10 --max_seq 25 --output_dir data --seed 42
$env:PYTHONPATH='.'
python scripts\train.py --config configs\smoke_dkt.yaml --dataset cleanup_t4_bin_codex --epochs 1
```

Result:

```text
cleanup_t4_bin_codex: parameters=86920, best_qwk=0.0064, auc=0.5003
```

## Artifact Cleanup

Removed after verification:

```text
data\smoke_test
outputs\smoke_dkvmn_gpcm
outputs\smoke_dkvmn_softmax
outputs\smoke_static_gpcm
outputs\smoke_dynamic_gpcm
data\cleanup_t4_bin_codex
outputs\cleanup_t4_bin_codex
```

## Unit Tests

Command:

```powershell
cd C:\Users\steph\Documents\deep-mirt\ma-irt
$env:PYTHONPATH='.'
pytest tests -q
```

Result:

```text
65 passed in 3.65s
```
