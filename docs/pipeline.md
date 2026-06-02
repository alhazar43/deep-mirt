# MA-GPCM Pipeline

This document describes the executable research pipeline for `ma-irt/`.
It is the contract cleanup work must preserve.

The shortest path is:

```text
synthetic data -> train model -> evaluate prediction -> evaluate recovery
```

For synthetic datasets, prediction and IRT parameter recovery are both
first-class outputs. For proxy-ordinal ASSISTments datasets, only prediction
metrics and learned-parameter diagnostics are directly observable because
ground-truth IRT parameters are not available.

## Environment Contract

Scripts currently require the `ma-irt/` package root on `PYTHONPATH`.

From inside `ma-irt/`:

```bash
export PYTHONPATH=.
```

PowerShell:

```powershell
$env:PYTHONPATH = "."
```

From the repository root:

```bash
export PYTHONPATH=ma-irt
```

This is a documented current-state requirement. Making scripts runnable
without setting `PYTHONPATH` is a later cleanup tier because it touches
executable import behavior.

## Dataset Contract

Each dataset lives in:

```text
ma-irt/data/<dataset_name>/
```

Required file:

```text
sequences.json
```

Recommended file:

```text
metadata.json
```

Synthetic-only recovery file:

```text
true_irt_parameters.json
```

`sequences.json` schema:

```json
[
  {
    "questions": [1, 7, 12],
    "responses": [0, 2, 1]
  }
]
```

Rules:

- `questions` and `responses` must have the same length for each sequence.
- Item IDs are 1-based in real data.
- Item ID 0 is reserved for padding/unknown inside the model.
- Responses are ordinal category IDs in `0..K-1`.
- The loader drops sequences shorter than `data.min_seq_len`.
- If `data.max_seq_len > 0`, sequences are truncated unless
  `data.chunk_long_sequences` is enabled.
- In chunking mode, long sequences are split into non-overlapping chunks.

`metadata.json` should include:

```json
{
  "n_questions": 20,
  "n_categories": 4
}
```

When present, `metadata.json` overrides `cfg.model.n_questions` and
`cfg.model.n_categories` at loader construction time. This is why
`scripts/train.py --dataset <name>` can safely point one smoke config at a
newly generated dataset.

## Data Loading Contract

`dataloading.loaders.DataModule` builds PyTorch loaders.

Non-CV path:

```text
DataModule.build() -> train_loader, test_loader
```

CV path:

```text
DataModule.build_cv() -> train_loader, valid_loader, test_loader
```

Batch collation returns:

```text
questions:   LongTensor, shape (B, S_max)
responses:   LongTensor, shape (B, S_max)
mask:        BoolTensor, shape (B, S_max)
student_ids: LongTensor, shape (B, S_max)
```

Padding convention:

- padded `questions` are 0
- padded `responses` are 0
- `mask[b, t]` is true only for real timesteps
- `student_ids` are broadcast across the row

Train/test split:

- non-CV uses `data.train_split`
- optional `data.shuffle_before_split` uses `base.seed`

CV split:

- enabled by `data.cv.enabled`
- uses `data.cv.split_seed`
- holds out `data.cv.test_frac` as test
- splits the remainder into `data.cv.n_folds`
- uses `data.cv.fold_id` as validation fold

## Config Contract

Config files are YAML overlays over dataclass defaults in `config/types.py`.
The loader recognizes these top-level sections:

```text
base
model
training
data
```

Unknown keys are ignored during loading. Semantic validation currently checks:

- `model.n_categories >= 2`
- `model.n_traits >= 1`
- `0 < data.train_split < 1`

Important CLI overrides in `scripts/train.py`:

```bash
python scripts/train.py --config <yaml> --dataset <dataset_name> --epochs <n>
```

`--dataset` sets both:

```text
cfg.data.dataset_name = <dataset_name>
cfg.base.experiment_name = <dataset_name>
```

That means training artifacts are written to:

```text
outputs/<dataset_name>/
```

## Training Contract

Training entry point:

```bash
python scripts/train.py --config <yaml> [--dataset <name>] [--epochs <n>] [--resume]
```

Training flow:

1. Load YAML config.
2. Apply CLI overrides.
3. Resolve device, falling back to CPU if CUDA is unavailable.
4. Seed Python and PyTorch with `base.seed`.
5. Build train/test or train/valid/test loaders.
6. Compute class weights when `training.weighted_ordinal_weight > 0`.
7. Build model from `model.model_type`.
8. Train with `CombinedLoss`, Adam, gradient clipping, and
   `ReduceLROnPlateau`.
9. Save checkpoints and metrics under `outputs/<experiment_name>/`.

Training artifacts:

```text
outputs/<experiment_name>/best.pt
outputs/<experiment_name>/last.pt
outputs/<experiment_name>/metrics.csv
outputs/<experiment_name>/test_metrics.json
```

`best.pt` is saved when the selected early-stop metric improves.
`last.pt` is saved every epoch. `test_metrics.json` is written by
`train.py` after loading `best.pt` and evaluating on the test loader.

## Model Contract

Models are selected by `model.model_type`.

| Model family | Config setting | Prediction | Recovery |
|---|---|---|---|
| MA-GPCM | `magpcm`, `separate_theta: true` | yes | yes |
| DKVMN+GPCM ablation | `magpcm`, `separate_theta: false` | yes | yes |
| Static GPCM | `static_gpcm` | yes | yes |
| Dynamic GPCM | `dynamic_gpcm` | yes | yes |
| DKVMN+Softmax | `dkvmn_softmax` | yes | no |
| DKT | `dkt` | yes | no |
| DKVMN | `dkvmn` | yes | no |
| Deep-IRT | `deep_irt` | yes | no |

For MA-GPCM, the main model output dictionary contains:

```text
theta:  (B, S, D)
alpha:  (B, S, D)
beta:   (B, S, K-1)
logits: (B, S, K)
probs:  (B, S, K)
```

Prediction-only baselines may expose placeholder parameter fields for trainer
compatibility. Public analysis must not interpret those placeholders as IRT
recovery.

## Evaluation Contract

Single-checkpoint evaluation:

```bash
python scripts/evaluate.py single \
  --config <yaml> \
  --checkpoint outputs/<experiment_name>/best.pt \
  --data-dir data/<dataset_name> \
  --batch-size 256
```

Prediction metrics include:

- categorical accuracy
- quadratic weighted kappa
- Kendall tau
- mean absolute error
- AUC for binary-compatible cases

Recovery metrics are emitted when `true_irt_parameters.json` exists. The
single-checkpoint evaluator writes:

```text
outputs/<experiment_name>/recovery_metrics.json
```

Recovery metrics include linked and raw comparisons for:

- `theta`
- `alpha`
- `beta_mean`

The evaluator uses linking transforms to account for IRT scale
indeterminacy:

- mean/sigma linking for normal-scale parameters
- log-space mean/sigma linking for positive discrimination (`alpha`)

## Paper Experiment Flow

Synthetic-static recovery and prediction:

```text
data_gen.py
configs/bulk/bench_<model>_static_q200_k<K>_pykt_fold<F>.yaml
scripts/_run_k4_cv_recovery.sh
scripts/_run_k356_cv_recovery.sh
scripts/aggregate_recovery_v5.py
```

Binary K=2 and ASSISTments prediction:

```text
scripts/_run_pykt_sweep.sh
scripts/_aggregate_pykt_results.py
```

R GPCM baseline:

```text
scripts/mirt_baseline_all_k.R
scripts/mirt_predict.R
```

The authoritative mapping from paper tables to configs, datasets, and
verification tolerances is `CLEANUP_VERIFICATION_2026.md`.

## Artifact Policy During Cleanup

Cleanup work should not leave generated smoke datasets, checkpoints, logs, or
diagnostic scripts behind. Temporary runs should use unique names and remove:

```text
ma-irt/data/<temporary_name>/
ma-irt/outputs/<temporary_name>/
```

Core files that can affect MA-GPCM performance require modular tests and,
where relevant, recovery/prediction checks before committing.
