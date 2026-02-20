# kt-gpcm

Knowledge tracing with the Generalized Partial Credit Model (GPCM) and a Dynamic Key-Value Memory Network (DKVMN). Trains on student response sequences and recovers ground-truth IRT parameters (θ = ability, α = discrimination, β = thresholds).

---

## Table of contents

1. [Install](#install)
2. [Quick start](#quick-start)
3. [Generate data](#generate-data)
4. [Train](#train)
5. [Long training runs](#long-training-runs)
6. [Monitor training](#monitor-training)
7. [Plot IRT recovery](#plot-irt-recovery)
8. [Multi-dimensional MIRT](#multi-dimensional-mirt)
9. [Configuration reference](#configuration-reference)
10. [Project layout](#project-layout)
11. [Tests](#tests)

---

## Install

```
pip install -r requirements.txt
```

> PyTorch must be installed separately to match your CUDA version — see the comment at the top of `requirements.txt`.

There are no sklearn or scipy dependencies in `src/`.

### Set PYTHONPATH once per session

All scripts require `src/` on the Python path. Set it once before running anything.

**PowerShell (Windows)**
```powershell
$env:PYTHONPATH = "src"
```

**bash / zsh / Git Bash**
```bash
export PYTHONPATH=src
```

**cmd.exe**
```cmd
set PYTHONPATH=src
```

The examples throughout this README assume PYTHONPATH is already set. No inline prefix is used.

---

## Quick start

```powershell
# 1. Generate a smoke dataset
python scripts/data_gen.py `
    --name smoke_test --n_questions 20 --n_cats 4 --output_dir data

# 2. Train for 2 epochs (CPU, fast)
python scripts/train.py --config configs/smoke.yaml --dataset smoke_test

# 3. Plot training curves
python scripts/plot_metrics.py `
    --metrics outputs/smoke/metrics.csv `
    --output  outputs/smoke/metric_plots

# 4. Plot IRT parameter recovery
python scripts/plot_recovery.py `
    --config     configs/smoke.yaml `
    --checkpoint outputs/smoke/best.pt `
    --output     outputs/smoke/recovery_plots
```

> **bash users**: replace the backtick `` ` `` line continuation with `\`.

---

## Generate data

`scripts/data_gen.py` generates synthetic student response sequences using the standard GPCM (Muraki 1992) with D = 1 latent trait. Ground-truth IRT parameters are saved alongside the data for recovery analysis.

```powershell
python scripts/data_gen.py `
    --name       my_dataset `
    --n_students 2000 `
    --n_questions 200 `
    --n_cats     5 `
    --min_seq    20 `
    --max_seq    80 `
    --output_dir data `
    --seed       42
```

| Flag | Default | Description |
|------|---------|-------------|
| `--name` | required | Dataset subdirectory name |
| `--n_students` | 500 | Number of simulated students |
| `--n_questions` | 100 | Item bank size Q |
| `--n_cats` | 5 | Ordinal response categories K |
| `--min_seq` | 10 | Minimum sequence length |
| `--max_seq` | 50 | Maximum sequence length |
| `--output_dir` | `data` | Root directory for datasets |
| `--seed` | 42 | Random seed |

**Output** — written to `<output_dir>/<name>/`:

```
sequences.json           — list of {questions, responses} dicts
metadata.json            — dataset parameters
true_irt_parameters.json — ground-truth theta, alpha, beta arrays
```

Replace `my_dataset` with any name you like. The `data.dataset_name` key in your YAML must match the `--name` value used here.

---

## Train

Pass `--dataset` with the name you used in `data_gen.py`. `n_questions` and `n_categories` are read from the dataset's `metadata.json` automatically — no manual config editing needed.

```powershell
python scripts/train.py --config configs/base.yaml --dataset my_dataset
```

Checkpoints and logs are written to `outputs/my_dataset/`:

```
outputs/base/
├── best.pt      — checkpoint with highest validation QWK
├── last.pt      — checkpoint after the most recent epoch
└── metrics.csv  — per-epoch metrics (see columns below)
```

`metrics.csv` columns:

| Column | Description |
|--------|-------------|
| `epoch` | Epoch number (1-based) |
| `train_loss` | Mean training loss |
| `train_accuracy` | Categorical accuracy on training set |
| `train_grad_norm` | L2 norm of gradients before clipping |
| `val_loss` | Mean validation loss |
| `val_categorical_accuracy` | Exact-match accuracy on validation set |
| `val_ordinal_accuracy` | Within-±1 accuracy on validation set |
| `val_qwk` | Quadratic Weighted Kappa (primary metric) |
| `val_mae` | Mean Absolute Error |
| `val_spearman` | Spearman rank correlation |
| `lr` | Learning rate at end of epoch |
| `epoch_time_s` | Wall-clock seconds for the epoch |

The learning rate scheduler (`ReduceLROnPlateau`) tracks `val_qwk`. Best model is saved whenever QWK improves.

---

## Long training runs

### Resuming an interrupted run

Pass `--resume` to continue from `last.pt`. The epoch counter picks up where training stopped; the CSV log is appended (no header duplication).

```powershell
python scripts/train.py --config configs/base.yaml --resume
```

`last.pt` is overwritten every epoch so a crash loses at most one epoch. `best.pt` is only overwritten when QWK improves, so it always holds the best weights found so far.

### Running more epochs than the config specifies

`--resume` adds `training.epochs` on top of the saved `start_epoch`, so you can extend a run by keeping the epoch count the same and re-running with `--resume`:

```yaml
# configs/base.yaml — first run: epochs 1–100
training:
  epochs: 100
```

```powershell
# First run (epochs 1–100)
python scripts/train.py --config configs/base.yaml

# Continue for another 100 epochs (epochs 101–200)
python scripts/train.py --config configs/base.yaml --resume

# Continue again (epochs 201–300)
python scripts/train.py --config configs/base.yaml --resume
```

To run a different total, just change `epochs` before resuming:

```yaml
training:
  epochs: 50   # will run 50 more epochs from wherever last.pt left off
```

### Isolating experiments

Copy `configs/base.yaml` and set `base.experiment_name` to a unique value. Each name gets its own `outputs/<name>/` directory — runs never overwrite each other.

**PowerShell**
```powershell
Copy-Item configs/base.yaml configs/large_d3.yaml
# edit: experiment_name, n_traits, epochs, etc.
python scripts/train.py --config configs/large_d3.yaml
```

**bash**
```bash
cp configs/base.yaml configs/large_d3.yaml
python scripts/train.py --config configs/large_d3.yaml
```

### NaN / Inf safety

The trainer skips any batch that produces non-finite loss and logs a warning. Training continues; the skipped batch is excluded from epoch statistics. If entire epochs produce NaN (e.g. exploding gradients), lower `training.lr` or raise `training.grad_clip`.

### Learning rate decay

`ReduceLROnPlateau` fires when QWK has not improved for `lr_patience` consecutive epochs, multiplying LR by `lr_factor`. Both are visible in `metrics.csv` under the `lr` column.

```yaml
training:
  lr_patience: 5    # wait longer before decaying
  lr_factor: 0.5    # more aggressive decay
```

---

## Monitor training

Plot all six metric panels from `metrics.csv`:

```powershell
python scripts/plot_metrics.py `
    --metrics outputs/base/metrics.csv `
    --output  outputs/base/metric_plots
```

Saves `outputs/base/metric_plots/training_metrics.png` — a 2×3 grid showing:

- Loss (train vs. val)
- Accuracy (categorical + ordinal, train vs. val)
- Quadratic Weighted Kappa
- Mean Absolute Error
- Spearman correlation
- Learning rate and gradient norm (dual-axis)

---

## Plot IRT recovery

For synthetic datasets (those with `true_irt_parameters.json`), compare model estimates against ground truth:

```powershell
python scripts/plot_recovery.py `
    --config     configs/base.yaml `
    --checkpoint outputs/base/best.pt `
    --output     outputs/base/recovery_plots
```

Runs inference over the full dataset (train + test), averages item-level estimates across all appearances, then saves one scatter plot per parameter group:

- `alpha_dim0_recovery.png` — item discrimination (one file per trait dim D)
- `beta_threshold0_recovery.png` — first threshold (one file per threshold, K−1 total)
- `beta_threshold1_recovery.png`, etc.

Each plot shows estimated vs. true values with Pearson r in the title and a y = x reference line.

---

## Multi-dimensional MIRT

Setting `model.n_traits > 1` enables multi-dimensional IRT with no code changes. The dot-product interaction `sum(theta * alpha, dim=-1)` degenerates correctly for D = 1 and scales to any D:

```yaml
# configs/mirt_d3.yaml
base:
  experiment_name: "mirt_d3"

model:
  n_traits: 3   # only change needed for MIRT
  memory_size: 50
  key_dim: 64
  value_dim: 128
  summary_dim: 50
```

```powershell
python scripts/train.py --config configs/mirt_d3.yaml
```

The model returns `theta` and `alpha` of shape `(B, S, D)`. Recovery plots produce D alpha scatter plots (`alpha_dim0_recovery.png`, `alpha_dim1_recovery.png`, …).

> **Note**: The data generator produces D = 1 ground truth. For D > 1, train on real data or extend the generator.

---

## Configuration reference

All fields carry defaults defined in `src/kt_gpcm/config/types.py`. A YAML only needs to specify fields that differ from those defaults.

### `base`

| Field | Default | Description |
|-------|---------|-------------|
| `experiment_name` | `"base"` | Subdirectory name under `outputs/` |
| `device` | `"cuda"` | `"cuda"` or `"cpu"` — falls back to CPU if CUDA unavailable |
| `seed` | `42` | Random seed for Python, NumPy, and PyTorch |

### `model`

| Field | Default | Description |
|-------|---------|-------------|
| `n_questions` | `100` | Item bank size Q (must match dataset) |
| `n_categories` | `5` | Ordinal response categories K (must match dataset) |
| `n_traits` | `1` | Latent trait dimension D |
| `memory_size` | `50` | DKVMN memory slots M |
| `key_dim` | `64` | Key/query dimension d_k |
| `value_dim` | `128` | Value memory dimension d_v |
| `summary_dim` | `50` | Summary (FC) hidden dimension d_s |
| `ability_scale` | `1.0` | Global scale on raw theta output |
| `dropout_rate` | `0.0` | Dropout probability in summary network |
| `memory_add_activation` | `"tanh"` | Activation for DKVMN add gate |
| `init_value_memory` | `true` | Use a learned initial value memory |

### `training`

| Field | Default | Description |
|-------|---------|-------------|
| `epochs` | `100` | Epochs to train; added on top of `start_epoch` when resuming |
| `batch_size` | `64` | |
| `lr` | `0.001` | Initial Adam learning rate |
| `grad_clip` | `1.0` | L2 gradient clip norm |
| `focal_weight` | `0.5` | Weight for FocalLoss component |
| `weighted_ordinal_weight` | `0.5` | Weight for WeightedOrdinalLoss component |
| `ordinal_penalty` | `0.5` | Ordinal distance penalty inside WeightedOrdinalLoss |
| `lr_patience` | `3` | Epochs without QWK improvement before LR decay |
| `lr_factor` | `0.8` | LR multiplication factor on plateau |
| `attention_entropy_weight` | `0.0` | Penalty encouraging focused memory reads (off by default) |
| `theta_norm_weight` | `0.0` | Penalty keeping theta ~ N(0,1) (off by default) |
| `alpha_prior_weight` | `0.0` | Penalty keeping log(alpha) ~ N(0, 0.3) (off by default) |
| `beta_prior_weight` | `0.0` | Penalty keeping beta ~ N(0, 1) (off by default) |

### `data`

| Field | Default | Description |
|-------|---------|-------------|
| `data_dir` | `"data"` | Root directory containing datasets |
| `dataset_name` | `"synthetic"` | Dataset subdirectory name (must match `--name` from data_gen) |
| `train_split` | `0.8` | Fraction of sequences used for training |
| `min_seq_len` | `10` | Sequences shorter than this are filtered out |

---

## Project layout

```
kt-gpcm/
├── src/kt_gpcm/
│   ├── config/
│   │   ├── types.py          # BaseConfig, ModelConfig, TrainingConfig, DataConfig, Config
│   │   └── loader.py         # load_config(yaml_path) → Config
│   ├── models/
│   │   ├── kt_gpcm.py        # DeepGPCM — main model (dict return)
│   │   ├── components/
│   │   │   ├── embeddings.py # LinearDecayEmbedding (nn.Module, vectorised)
│   │   │   ├── memory.py     # DKVMN (flat class, learned_init)
│   │   │   └── irt.py        # IRTParameterExtractor, GPCMLogits
│   │   └── heads/
│   │       └── gpcm.py       # GPCMHead (softmax)
│   ├── training/
│   │   ├── losses.py         # FocalLoss, QWKLoss, WeightedOrdinalLoss, CombinedLoss
│   │   └── trainer.py        # Trainer (train_epoch, evaluate_epoch)
│   ├── data/
│   │   └── loaders.py        # SequenceDataset, DataModule, collate_sequences
│   └── utils/
│       └── metrics.py        # compute_metrics — pure PyTorch, no sklearn
├── scripts/
│   ├── data_gen.py           # GPCMGenerator — synthetic data
│   ├── train.py              # Training entry point
│   ├── plot_metrics.py       # Training curves from metrics.csv
│   └── plot_recovery.py      # IRT parameter recovery scatter plots
├── configs/
│   ├── base.yaml             # Validated Deep-GPCM defaults
│   └── smoke.yaml            # Tiny config for rapid iteration
├── tests/
│   ├── test_shapes.py        # Forward-pass tensor shapes for D=1 and D=3
│   ├── test_heads.py         # GPCMLogits baseline, prob normalisation
│   ├── test_losses.py        # CombinedLoss finite output
│   └── test_config_loader.py # YAML loading and validation
├── data/                     # Generated datasets (gitignored)
├── outputs/                  # Training outputs — checkpoints, metrics, plots (gitignored)
├── requirements.txt
└── README.md
```

### Model forward pass

```
(questions, responses)              # (B, S) integer tensors
  → LinearDecayEmbedding            # triangular-kernel ordinal embedding → (B, S, E)
  → per-timestep DKVMN loop:
      attention(query)              # softmax over M memory slots → (B, M)
      read(value_memory, weights)   # weighted sum → context vector (B, d_v)
      summary FC + tanh + dropout   # → (B, d_s)
      IRTParameterExtractor         # theta (B,S,D)  alpha (B,S,D)  beta (B,S,K-1)
      GPCMLogits                    # cumulative logits → (B,S,K)
      write(value_memory, ...)      # erase-add update
  → GPCMHead                        # softmax → probs (B,S,K)

Returns dict:
  "theta"  (B, S, D)    student ability
  "alpha"  (B, S, D)    item discrimination
  "beta"   (B, S, K-1)  item thresholds
  "logits" (B, S, K)    GPCM logits  (used by loss)
  "probs"  (B, S, K)    category probabilities  (used by metrics)
```

---

## Tests

```powershell
# from kt-gpcm/ with PYTHONPATH=src already set
pytest tests/ -v
```

| File | What it checks |
|------|----------------|
| `test_shapes.py` | Output tensor shapes for D=1 and D=3, all model components |
| `test_heads.py` | Category-0 logit = 0, probabilities sum to 1, D=1 equivalence |
| `test_losses.py` | Each loss produces finite output; `compute_class_weights` normalises correctly |
| `test_config_loader.py` | `smoke.yaml` loads correctly; missing sections get defaults; invalid values raise |
