# T1 README Entry Surface - 2026-06-02

This note records the verification for Plan v2 T1, Public entry surface.

## Files Changed

- `README.md`
- `ma-irt/README.md`

The changes are documentation-only. No core model, training, data-loading,
evaluation, plotting, or config files were intentionally edited in this tier.

## What Changed

- Removed README encoding corruption and stale quickstart references.
- Reframed the project around the verified MA-GPCM pipeline:
  dataset generation, training, prediction evaluation, and IRT recovery.
- Documented the key model-family distinction:
  - MA-GPCM is `model_type: magpcm` with `separate_theta: true`.
  - DKVMN+GPCM is the same class with `separate_theta: false`.
- Documented that DKVMN+Softmax and binary KT baselines are prediction
  baselines, not IRT recovery models.
- Made the public quickstart use the exact smoke path verified below.

## Stale Reference Scan

Command:

```powershell
rg -n "鈥|鈹|胃|伪|尾|卤|脳|v2_q200_k4|run_all_experiments|plot_recovery.py|plot_recovery_figure.py" README.md ma-irt\README.md
```

Result: no matches.

## README Smoke Verification

The documented commands were run exactly with the public `smoke_test` paths.
Those paths did not exist before the check.

Data generation:

```powershell
python scripts\data_gen.py --name smoke_test --n_students 120 --n_questions 20 --n_cats 4 --min_seq 10 --max_seq 25 --output_dir data --seed 42
```

Result:

```text
Wrote 120 sequences -> data\smoke_test\sequences.json
Wrote metadata       -> data\smoke_test\metadata.json
Wrote IRT parameters -> data\smoke_test\true_irt_parameters.json
```

Training:

```powershell
$env:PYTHONPATH='.'
python scripts\train.py --config configs\smoke.yaml --dataset smoke_test --epochs 1
```

Result:

```text
Experiment: smoke_test | device: cpu | seed: 0
Dataset: smoke_test | train batches: 6 | test batches: 2
Training observations: 1598
Model parameters: 24901
Training complete. Best qwk: 0.2193 at epoch 1
Artifacts saved to: outputs\smoke_test
```

Evaluation:

```powershell
$env:PYTHONPATH='.'
python scripts\evaluate.py single --config configs\smoke.yaml --checkpoint outputs\smoke_test\best.pt --data-dir data\smoke_test --batch-size 32
```

Result:

```text
Model: magpcm, Q=20, K=4, D=1
ACC=0.1794  QWK=0.2193  tau=0.2955  MAE=1.0090
r_alpha=-0.1220  r_beta_mean=-0.0633
r_theta=0.9151  (static DGP, forward-pass)
Wrote outputs\smoke_test\recovery_metrics.json
```

The generated `data/smoke_test` and `outputs/smoke_test` directories were
removed after verification.

## Unit Tests

Command:

```powershell
cd C:\Users\steph\Documents\deep-mirt\ma-irt
$env:PYTHONPATH='.'
pytest tests -q
```

Result:

```text
65 passed in 4.62s
```
