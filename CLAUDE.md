# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Use the `research` conda environment and set `PYTHONPATH` before running anything:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate research
export PYTHONPATH=ma-irt
```

Use `KMP_DUPLICATE_LIB_OK=TRUE` on Windows when running torch-based scripts.

## Commands

**Run tests:**
```bash
cd ma-irt && PYTHONPATH=. pytest tests/ -v
```

**Generate synthetic data (static DGP):**
```bash
cd ma-irt && python scripts/data_gen.py \
  --name v2_q200_k4 --n_students 5000 --n_questions 200 --n_cats 4 \
  --min_seq 20 --max_seq 80 --output_dir data
```

**Generate dynamic data:**
```bash
# Staircase (3-level discrete ability change)
cd ma-irt && python scripts/data_gen_staircase.py \
  --name staircase_q200_k4 --n_students 5000 --n_questions 200 --n_cats 4 --output_dir data

# Random walk (continuous drift)
cd ma-irt && python scripts/data_gen_randomwalk.py \
  --name rw_q200_k4 --n_students 5000 --n_questions 200 --n_cats 4 --output_dir data

# Block change (pretest-posttest)
cd ma-irt && python scripts/data_gen_block.py \
  --name block_q200_k4 --n_students 5000 --n_questions 200 --n_cats 4 --output_dir data
```

**Train:**
```bash
cd ma-irt && PYTHONPATH=. python scripts/train.py \
  --config configs/staircase_q200_k4.yaml
```

**Evaluate (unified, handles all model types and DGPs):**
```bash
cd ma-irt && KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python scripts/evaluate.py single \
  --config configs/staircase_q200_k4.yaml \
  --checkpoint outputs/staircase_q200_k4/best.pt \
  --data-dir data/staircase_q200_k4
```

**Plot training curves:**
```bash
cd ma-irt && python scripts/plot_metrics.py \
  --metrics outputs/staircase_q200_k4/metrics.csv \
  --output outputs/staircase_q200_k4/metric_plots
```

**Compile paper:**
```bash
cd overleaf-sync && pdflatex main.tex
```

## Architecture

The active project is `ma-irt/`. Legacy directories (`mirt-dkvmn/`, `deep-gpcm/`) are archived.
Archived scripts and configs are in `ma-irt/archive/`.

**Goal**: Train a neural network on synthetic student response sequences and recover ground-truth IRT parameters (theta = ability, alpha = discrimination, beta = step thresholds).

### Models

Six models are used in the paper:

| Model | Code | Config `model_type` | IRT params | Notes |
|-------|------|-------------------|------------|-------|
| **MA-GPCM** (ours) | `MAGPCM` | `magpcm` | theta, alpha, beta | Separated ability pathway + SIE |
| **DKVMN+GPCM** | `MAGPCM` | `magpcm` | theta, alpha, beta | `separate_theta: false` (shared pathway) |
| **DKVMN+Softmax** | `DKVMNSoftmax` | `dkvmn_softmax` | none | No IRT structure |
| **Dynamic GPCM** | `DynamicGPCM` | `dynamic_gpcm` | theta, alpha, beta | Gated recurrence, no memory |
| **GPCM (SGD)** | `StaticGPCM` | `static_gpcm` | theta, alpha, beta | Static theta per student |
| **GPCM (EM)** | R `mirt` package | N/A | theta, alpha, beta | `scripts/mirt_baseline_all_k.R` |

### Data flow (MA-GPCM)

```
(question_ids, responses)
-> StaticItemEmbedding       # frozen random item vectors + ordinal kernel
-> DKVMN memory              # attention + read + write
-> separated ability summary # f_theta from read vector only (no item key)
-> IRTParameterExtractor     # produces theta, alpha, beta (unconstrained)
-> GPCMLogits                # K-1 cumulative logits
-> categorical probabilities
```

### Key source files

| File | Role |
|------|------|
| `models/magpcm.py` | MAGPCM (MA-GPCM and DKVMN+GPCM) |
| `models/static_gpcm.py` | StaticGPCM baseline |
| `models/dynamic_gpcm.py` | DynamicGPCM baseline |
| `models/dkvmn_softmax.py` | DKVMN+Softmax baseline |
| `models/components/memory.py` | DKVMN key/value memory |
| `models/components/irt.py` | IRT parameter extraction + GPCM logits |
| `models/components/embeddings.py` | LinearDecay, Separable, StaticItem embeddings |
| `models/heads/gpcm.py` | GPCM head (softmax over cumulative logits) |
| `training/trainer.py` | Training loop, metric logging |
| `training/losses.py` | WeightedOrdinalLoss, CombinedLoss |
| `dataloading/loaders.py` | SequenceDataset, DataModule, collate_sequences |
| `config/types.py` | Config dataclasses (Base/Model/Training/Data) |

### Core scripts

| Script | Purpose |
|--------|---------|
| `scripts/train.py` | Train any model type |
| `scripts/evaluate.py` | Unified eval: prediction metrics + IRT recovery |
| `scripts/data_gen.py` | Generate static-theta GPCM data |
| `scripts/data_gen_staircase.py` | Generate 3-staircase DGP |
| `scripts/data_gen_randomwalk.py` | Generate random walk DGP |
| `scripts/data_gen_block.py` | Generate block-change DGP |
| `scripts/data_gen_imbalanced.py` | Generate imbalanced data |
| `scripts/plot_trajectory_comparison.py` | 3-model trajectory figures |
| `scripts/plot_recovery_split.py` | Item/student recovery scatter |
| `scripts/plot_theta_temporal.py` | Temporal theta convergence |
| `scripts/plot_metrics.py` | Training curve plots |
| `scripts/mirt_baseline_all_k.R` | GPCM(EM) baseline via R mirt |

### Configuration

Experiments are driven by YAML configs in `ma-irt/configs/`. Key parameters:

- `model.n_questions` — item bank size Q
- `model.n_categories` — ordinal response categories K
- `model.n_traits` — latent dimensions (1 for IRT)
- `model.embedding_type` — `"onehot"`, `"learned"`, or `"static_item"` (default)
- `model.model_type` — `"magpcm"`, `"static_gpcm"`, `"dynamic_gpcm"`, `"dkvmn_softmax"`
- `model.separate_theta` — `true` (MA-GPCM) or `false` (DKVMN+GPCM)
- `training.weighted_ordinal_weight` — WOL weight (default 1.0)
- `base.device` — `"cuda"` or `"cpu"`; falls back to CPU if CUDA unavailable

### Data naming conventions

- Static DGP: `data/v2_q{Q}_k{K}/`
- Block change: `data/block_q{Q}_k{K}/`
- Random walk: `data/rw_q{Q}_k{K}/`
- Staircase: `data/staircase_q{Q}_k{K}/`
- Imbalanced: `data/v2_q{Q}_k{K}_{mild,severe,extreme}_imb/`

Each dataset contains: `sequences.json`, `metadata.json`, `true_irt_parameters.json`

### Training outputs

- `ma-irt/outputs/<experiment_name>/`
  - `metrics.csv`, `best.pt`, `last.pt`

### Paper

The paper source is in `overleaf-sync/` (synced via git to Overleaf).
- `overleaf-sync/main.tex` — main paper
- `overleaf-sync/ref.bib` — bibliography
- `overleaf-sync/figures/` — PDF/PGF figures
