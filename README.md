# MA-IRT / MA-GPCM

Memory-Augmented Item Response Theory for Polytomous Knowledge Tracing. The model trains on synthetic student response sequences and recovers ground-truth IRT parameters (theta = ability, alpha = discrimination, beta = step thresholds).

**Paper**: "MA-IRT: A Memory-Augmented Framework for Polytomous Knowledge Tracing with IRT Parameter Recovery" (target: IJAIED)

## Environment

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate research
```

## Quick Start

```bash
cd kt-gpcm
export PYTHONPATH=src

# Generate a synthetic dataset (5000 students, 200 items, 4 categories)
python scripts/data_gen.py \
  --name v2_q200_k4 --n_students 5000 --n_questions 200 --n_cats 4 \
  --min_seq 20 --max_seq 80 --output_dir data

# Train MA-GPCM
python scripts/train.py --config configs/smoke.yaml

# Evaluate (prediction metrics + IRT parameter recovery)
KMP_DUPLICATE_LIB_OK=TRUE python scripts/evaluate.py single \
  --config configs/smoke.yaml \
  --checkpoint outputs/smoke/best.pt \
  --data-dir data/v2_q200_k4

# Plot training curves
python scripts/plot_metrics.py \
  --metrics outputs/smoke/metrics.csv \
  --output outputs/smoke/metric_plots
```

## Models

| Model | Paper name | `model_type` | Description |
|-------|-----------|-------------|-------------|
| `DeepGPCM` | MA-GPCM | `deepgpcm` | DKVMN + separated ability pathway + GPCM head |
| `DeepGPCM` | DKVMN+GPCM | `deepgpcm` | Same but `separate_theta: false` |
| `DKVMNSoftmax` | DKVMN+Softmax | `dkvmn_softmax` | DKVMN + K-way softmax (no IRT) |
| `DynamicGPCM` | Dynamic GPCM | `dynamic_gpcm` | Gated recurrent theta + per-item lookup |
| `StaticGPCM` | GPCM (SGD) | `static_gpcm` | Static theta embedding + per-item params |
| R `mirt` | GPCM (EM) | N/A | EM calibration via `mirt_baseline_all_k.R` |

## Item Representations

| `embedding_type` | Description |
|-----------------|-------------|
| `static_item` (default) | Frozen random unit-norm vectors + learned projection |
| `linear_decay` | Triangular ordinal kernel (Kronecker product) |
| `separable` | Learned item embedding + ordinal weights |

## Data Generators

| Script | DGP type | Ability dynamics |
|--------|----------|-----------------|
| `data_gen.py` | Static | theta fixed per student |
| `data_gen_staircase.py` | Staircase | 3-level discrete shifts |
| `data_gen_randomwalk.py` | Random walk | Continuous drift |
| `data_gen_block.py` | Block change | Pretest-posttest |
| `data_gen_imbalanced.py` | Imbalanced | Skewed ability prior |

## Repository Layout

```
kt-gpcm/
  src/kt_gpcm/         # Library code (models, training, data, config)
  scripts/              # Core pipeline scripts (train, evaluate, plot, data gen)
  configs/              # YAML experiment configs
    experiments/        # Per-RQ multi-seed configs
    dynamic_seeds/      # Multi-seed block/rw configs
  data/                 # Generated datasets
  outputs/              # Checkpoints, metrics, figures
  tests/                # Unit tests
  archive/              # Archived scripts, configs, shell wrappers
overleaf-sync/          # Paper LaTeX source (synced to Overleaf)
```

## Notes

- `base.device` in configs controls GPU usage; falls back to CPU if CUDA unavailable.
- Use `KMP_DUPLICATE_LIB_OK=TRUE` on Windows to avoid MKL conflicts.
- Loss: Weighted Ordinal Loss (WOL) only. `training.weighted_ordinal_weight: 1.0`.
- Step thresholds are unconstrained (no monotonic enforcement).
