# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Use the `research` conda environment and set `PYTHONPATH` before running anything:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate research
export PYTHONPATH=kt-gpcm/src
```

Use `KMP_DUPLICATE_LIB_OK=TRUE` on Windows when running torch-based scripts.

## Commands

**Run tests:**
```bash
cd kt-gpcm && PYTHONPATH=src pytest tests/ -v
```

**Run a single test file:**
```bash
cd kt-gpcm && PYTHONPATH=src pytest tests/test_shapes.py -v
```

**Generate synthetic data:**
```bash
cd kt-gpcm && PYTHONPATH=src python scripts/data_gen.py \
  --name large_q200_k4 --n_students 5000 --n_questions 200 --n_cats 4 \
  --min_seq 20 --max_seq 80 --output_dir data
```

**Train:**
```bash
cd kt-gpcm && PYTHONPATH=src python scripts/train.py \
  --config configs/generated/q200_k4_static_item.yaml
```

**Compute recovery correlations:**
```bash
cd kt-gpcm && KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python scripts/compute_all_recovery.py \
  --output_csv outputs/recovery_correlations.csv
```

**Plot split recovery (requires all 3 model checkpoints):**
```bash
cd kt-gpcm && PYTHONPATH=src python scripts/plot_recovery_split.py \
  --deepgpcm-config configs/generated/q200_k4_static_item.yaml \
  --deepgpcm-checkpoint outputs/q200_k4_static_item/best.pt \
  --static-config configs/baselines/large_q200_k4_static_gpcm.yaml \
  --static-checkpoint outputs/large_q200_k4_static_gpcm/best.pt \
  --dynamic-config configs/baselines/large_q200_k4_dynamic_gpcm.yaml \
  --dynamic-checkpoint outputs/large_q200_k4_dynamic_gpcm/best.pt \
  --output outputs/q200_k4_static_item/recovery
```

**Plot training curves:**
```bash
cd kt-gpcm && python scripts/plot_metrics.py \
  --metrics outputs/q200_k4_static_item/metrics.csv \
  --output outputs/q200_k4_static_item/metric_plots
```

**Compile paper:**
```bash
pdflatex paper.tex
```

## Architecture

The active project is `kt-gpcm/`. Legacy directories (`mirt-dkvmn/`, `deep-gpcm/`) are archived.

**Goal**: Train a neural network on synthetic student response sequences and recover ground-truth IRT parameters (θ = ability, α = discrimination, β = thresholds).

### Data flow

```
(question_ids, responses)
→ Embedding (LinearDecay / Separable / StaticItem)
→ DKVMN memory (attention + read + write)
→ summary network
→ IRTParameterExtractor    # produces θ, α, β
→ GPCMLogits               # K-1 cumulative logits
→ categorical probabilities
```

### Key source files

| File | Role |
|------|------|
| `kt-gpcm/src/kt_gpcm/models/kt_gpcm.py` | Main DeepGPCM model |
| `kt-gpcm/src/kt_gpcm/models/components/memory.py` | DKVMN key/value memory |
| `kt-gpcm/src/kt_gpcm/models/components/irt.py` | IRT parameter extraction + GPCM logits |
| `kt-gpcm/src/kt_gpcm/models/components/embeddings.py` | LinearDecay, Separable, StaticItem embeddings |
| `kt-gpcm/src/kt_gpcm/models/heads/gpcm.py` | GPCM head |
| `kt-gpcm/src/kt_gpcm/training/trainer.py` | Training loop, metric logging |
| `kt-gpcm/src/kt_gpcm/training/losses.py` | FocalLoss, WeightedOrdinalLoss, CombinedLoss |
| `kt-gpcm/src/kt_gpcm/data/loaders.py` | SequenceDataset, DataModule, collate_sequences |
| `kt-gpcm/src/kt_gpcm/config/types.py` | Config dataclasses (Base/Model/Training/Data) |

### Configuration

Experiments are driven by YAML configs in `kt-gpcm/configs/`. Key parameters:

- `model.n_questions` — item bank size Q
- `model.n_categories` — ordinal response categories K
- `model.n_traits` — latent dimensions (1 for IRT)
- `model.embedding_type` — `"linear_decay"`, `"separable"`, or `"static_item"`
- `model.model_type` — `"deepgpcm"`, `"static_gpcm"`, `"dynamic_gpcm"`, `"dkvmn_softmax"`
- `model.monotonic_betas` — enforce β₁ < β₂ < ... < β_{K-1}
- `training.focal_weight` / `training.weighted_ordinal_weight` — loss composition
- `base.device` — `"cuda"` or `"cpu"`; falls back to CPU if CUDA unavailable

### Data and artifacts

- Datasets: `kt-gpcm/data/large_q<items>_k<cats>/`
  - `sequences.json`, `metadata.json`, `true_irt_parameters.json`
- Training outputs: `kt-gpcm/outputs/<experiment_name>/`
  - `metrics.csv`, `best.pt`, `last.pt`
- Recovery correlations: `kt-gpcm/outputs/recovery_correlations.csv`
