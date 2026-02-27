# kt-gpcm

Knowledge tracing with the Generalized Partial Credit Model (GPCM) and a Dynamic Key-Value Memory Network (DKVMN). Trains on student response sequences and recovers ground-truth IRT parameters (θ = ability, α = discrimination, β = thresholds).

Three model variants are supported via `model.model_type` in the config:

| `model_type` | Class | Description |
|---|---|---|
| `deepgpcm` (default) | `DeepGPCM` | DKVMN backbone + GPCM head; dynamic θ_t per step |
| `dkvmn_softmax` | `DKVMNSoftmax` | DKVMN backbone + softmax head; no IRT parameters |
| `static_gpcm` | `StaticGPCM` | Lookup-table IRT baseline; static per-student θ |
| `dynamic_gpcm` | `DynamicGPCM` | Recurrent IRT baseline; dynamic θ_t, no memory |

---

## Table of contents

1. [Install](#install)
2. [Quick start](#quick-start)
3. [Generate data](#generate-data)
4. [Train](#train)
5. [Long training runs](#long-training-runs)
6. [Monitor training](#monitor-training)
7. [Plot IRT recovery](#plot-irt-recovery)
8. [Combined recovery figure](#combined-recovery-figure)
9. [Learner trajectory plot](#learner-trajectory-plot)
10. [Multi-dimensional MIRT](#multi-dimensional-mirt)
11. [Configuration reference](#configuration-reference)
12. [Project layout](#project-layout)
13. [Tests](#tests)

---

## Install

```
pip install -r requirements.txt
```

> PyTorch must be installed separately to match your CUDA version — see the comment at the top of `requirements.txt`.

### Set PYTHONPATH once per session

All scripts require `src/` on the Python path.

**bash / zsh / Git Bash**
```bash
export PYTHONPATH=src
```

**PowerShell (Windows)**
```powershell
$env:PYTHONPATH = "src"
```

**cmd.exe**
```cmd
set PYTHONPATH=src
```

The examples throughout this README assume PYTHONPATH is already set.

---

## Quick start

```bash
# 1. Generate a smoke dataset
python scripts/data_gen.py \
    --name smoke_test --n_questions 20 --n_cats 4 --output_dir data

# 2. Train for 2 epochs (CPU, fast)
python scripts/train.py --config configs/smoke.yaml --dataset smoke_test

# 3. Plot training curves
python scripts/plot_metrics.py \
    --metrics outputs/smoke/metrics.csv \
    --output  outputs/smoke/metric_plots

# 4. Plot IRT parameter recovery
python scripts/plot_recovery.py \
    --config     configs/smoke.yaml \
    --checkpoint outputs/smoke/best.pt \
    --output     outputs/smoke/recovery_plots
```

---

## Generate data

`scripts/data_gen.py` generates synthetic student response sequences using the M-GPCM (Muraki 1992) with D = 1 latent trait. Ground-truth IRT parameters are saved alongside the data for recovery analysis.

```bash
python scripts/data_gen.py \
    --name       my_dataset \
    --n_students 2000 \
    --n_questions 200 \
    --n_cats     5 \
    --min_seq    20 \
    --max_seq    80 \
    --output_dir data \
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

The `data.dataset_name` key in your YAML must match the `--name` value used here.

---

## Train

Pass `--dataset` with the name you used in `data_gen.py`. `n_questions` and `n_categories` are read from the dataset's `metadata.json` automatically.

```bash
python scripts/train.py --config configs/deepgpcm_k5_s42.yaml --dataset my_dataset
```

Checkpoints and logs are written to `outputs/<experiment_name>/`:

```
outputs/<name>/
├── best.pt      — checkpoint with highest validation QWK
├── last.pt      — checkpoint after the most recent epoch
└── metrics.csv  — per-epoch metrics
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

```bash
python scripts/train.py --config configs/deepgpcm_k5_s42.yaml --resume
```

`last.pt` is overwritten every epoch so a crash loses at most one epoch. `best.pt` is only overwritten when QWK improves.

### Running more epochs than the config specifies

`--resume` adds `training.epochs` on top of the saved `start_epoch`:

```bash
# First run (epochs 1–100)
python scripts/train.py --config configs/deepgpcm_k5_s42.yaml

# Continue for another 100 epochs (epochs 101–200)
python scripts/train.py --config configs/deepgpcm_k5_s42.yaml --resume
```

### Isolating experiments

Each config's `base.experiment_name` determines the output directory. Copy a config and change the name to keep runs separate:

```bash
cp configs/deepgpcm_k5_s42.yaml configs/my_experiment.yaml
# edit: experiment_name, epochs, etc.
python scripts/train.py --config configs/my_experiment.yaml
```

### NaN / Inf safety

The trainer skips any batch that produces non-finite loss and logs a warning. If entire epochs produce NaN, lower `training.lr` or raise `training.grad_clip`.

### Learning rate decay

`ReduceLROnPlateau` fires when QWK has not improved for `lr_patience` consecutive epochs, multiplying LR by `lr_factor`.

---

## Monitor training

```bash
python scripts/plot_metrics.py \
    --metrics outputs/deepgpcm_k5_s42/metrics.csv \
    --output  outputs/deepgpcm_k5_s42/metric_plots
```

Saves a 2×3 grid showing loss, accuracy, QWK, MAE, Spearman, and learning rate / gradient norm.

---

## Plot IRT recovery

For synthetic datasets (those with `true_irt_parameters.json`), compare model estimates against ground truth. Applies proper IRT linking: log-space z-score rescaled to `target_std=0.3` for α; z-score for β.

```bash
python scripts/plot_recovery.py \
    --config     configs/deepgpcm_k5_s42.yaml \
    --checkpoint outputs/deepgpcm_k5_s42/best.pt \
    --output     outputs/deepgpcm_k5_s42/recovery_plots
```

Saves one scatter plot per parameter group:

- `alpha_dim0_recovery.png` — item discrimination (one file per trait dim D)
- `beta_threshold0_recovery.png` — first threshold (one file per threshold, K−1 total)

Each plot shows linked estimated vs. true values with Pearson r and a y = x reference line.

---

## Combined recovery figure

`scripts/plot_recovery_figure.py` produces the paper's 3×6 recovery figure comparing all three model types side by side. Requires trained checkpoints for DeepGPCM, StaticGPCM, and DynamicGPCM.

```bash
python scripts/plot_recovery_figure.py \
  --deepgpcm-config    configs/deepgpcm_k5_s42.yaml \
  --deepgpcm-checkpoint outputs/deepgpcm_k5_s42/best.pt \
  --static-config      configs/static_gpcm_k5_s42.yaml \
  --static-checkpoint  outputs/static_gpcm_k5_s42/best.pt \
  --dynamic-config     configs/dynamic_gpcm_k5_s42.yaml \
  --dynamic-checkpoint outputs/dynamic_gpcm_k5_s42/best.pt \
  --output             outputs/deepgpcm_k5_s42/recovery_figure
```

Saves `recovery_figure.pgf` and `recovery_figure.png`. The PGF file is `\input`-ed directly into `paper.tex`.

---

## Learner trajectory plot

`scripts/plot_learner_trajectories.py` produces the paper's Fig 3 comparing θ_t trajectories across DeepGPCM, DynamicGPCM, and DKVMN+Softmax for four learner archetypes.

```bash
python scripts/plot_learner_trajectories.py \
  --deepgpcm-config    configs/deepgpcm_k5_s42.yaml \
  --deepgpcm-checkpoint outputs/deepgpcm_k5_s42/best.pt \
  --softmax-config     configs/softmax_k5_s42.yaml \
  --softmax-checkpoint outputs/softmax_k5_s42/best.pt \
  --dynamic-config     configs/dynamic_gpcm_k5_s42.yaml \
  --dynamic-checkpoint outputs/dynamic_gpcm_k5_s42/best.pt \
  --output-dir         outputs/deepgpcm_k5_s42/trajectory_plots
```

Saves `learner_trajectories.pgf` and `learner_trajectories.png`.

---

## Multi-dimensional MIRT

Setting `model.n_traits > 1` enables multi-dimensional IRT with no code changes:

```yaml
model:
  n_traits: 3
  model_type: deepgpcm
```

The model returns `theta` and `alpha` of shape `(B, S, D)`. Recovery plots produce D alpha scatter plots.

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
| `model_type` | `"deepgpcm"` | `"deepgpcm"` \| `"dkvmn_softmax"` \| `"static_gpcm"` \| `"dynamic_gpcm"` |
| `n_questions` | `100` | Item bank size Q (must match dataset) |
| `n_categories` | `5` | Ordinal response categories K (must match dataset) |
| `n_traits` | `1` | Latent trait dimension D |
| `memory_size` | `50` | DKVMN memory slots M |
| `key_dim` | `64` | Key/query dimension d_k |
| `value_dim` | `128` | Value memory dimension d_v |
| `summary_dim` | `50` | Summary (FC) hidden dimension d_s |
| `embedding_type` | `"linear_decay"` | `"linear_decay"` \| `"static_item"` \| `"separable"` |
| `item_embed_dim` | `0` | Static item embedding dim H (`static_item` only); 0 = auto |
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
│   │   ├── types.py              # BaseConfig, ModelConfig, TrainingConfig, DataConfig, Config
│   │   └── loader.py             # load_config(yaml_path) → Config
│   ├── models/
│   │   ├── kt_gpcm.py            # DeepGPCM — DKVMN + GPCM head
│   │   ├── static_gpcm.py        # StaticGPCM — lookup-table IRT baseline
│   │   ├── dynamic_gpcm.py       # DynamicGPCM — recurrent IRT baseline
│   │   ├── dkvmn_softmax.py      # DKVMNSoftmax — DKVMN + softmax head baseline
│   │   ├── components/
│   │   │   ├── embeddings.py     # LinearDecayEmbedding, StaticItemEmbedding
│   │   │   ├── memory.py         # DKVMN (flat class, learned_init)
│   │   │   └── irt.py            # IRTParameterExtractor, GPCMLogits
│   │   └── heads/
│   │       └── gpcm.py           # GPCMHead (softmax over GPCM logits)
│   ├── training/
│   │   ├── losses.py             # FocalLoss, WeightedOrdinalLoss, CombinedLoss
│   │   └── trainer.py            # Trainer (train_epoch, evaluate_epoch)
│   ├── data/
│   │   └── loaders.py            # SequenceDataset, DataModule, collate_sequences
│   └── utils/
│       └── metrics.py            # compute_metrics — pure PyTorch, no sklearn
├── scripts/
│   ├── data_gen.py               # GPCMGenerator — synthetic data with ground-truth IRT params
│   ├── train.py                  # Training entry point (all model_types)
│   ├── plot_metrics.py           # Training curves from metrics.csv
│   ├── plot_recovery.py          # Per-model IRT recovery scatter plots
│   ├── plot_recovery_figure.py   # Combined 3×6 paper figure (all three models)
│   ├── plot_learner_trajectories.py  # Learner state trajectory figure (Fig 3)
│   ├── eval_metrics.py           # Standalone evaluation on a saved checkpoint
│   └── prepare_assistments.py    # ASSISTments → ordinal proxy pipeline
├── configs/
│   ├── smoke.yaml                # Tiny config for rapid iteration
│   ├── deepgpcm_k{2,3,4,5}_s42.yaml   # DeepGPCM experiments
│   ├── static_gpcm_k{2,3,4,5}_s42.yaml
│   ├── dynamic_gpcm_k{2,3,4,5}_s42.yaml
│   ├── softmax_k{2,3,4,5}_s42.yaml
│   ├── ordinal_k{2,3,4,5}_s42.yaml
│   └── large_q5000_static.yaml   # Scalability experiment
├── tests/
│   ├── test_shapes.py            # Forward-pass tensor shapes for D=1 and D=3
│   ├── test_heads.py             # GPCMLogits baseline, prob normalisation
│   ├── test_losses.py            # CombinedLoss finite output
│   └── test_config_loader.py     # YAML loading and validation
├── data/                         # Generated datasets (gitignored)
├── outputs/                      # Training outputs — checkpoints, metrics, plots (gitignored)
├── requirements.txt
└── README.md
```

### DeepGPCM forward pass

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

```bash
# from kt-gpcm/ with PYTHONPATH=src already set
pytest tests/ -v
```

| File | What it checks |
|------|----------------|
| `test_shapes.py` | Output tensor shapes for D=1 and D=3, all model components |
| `test_heads.py` | Category-0 logit = 0, probabilities sum to 1, D=1 equivalence |
| `test_losses.py` | Each loss produces finite output; `compute_class_weights` normalises correctly |
| `test_config_loader.py` | `smoke.yaml` loads correctly; missing sections get defaults; invalid values raise |
