# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Use the `vrec-env` conda environment and set `PYTHONPATH` before running anything:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate vrec-env
export PYTHONPATH=mirt-dkvmn/src
```

## Commands

**Run tests:**
```bash
cd mirt-dkvmn && PYTHONPATH=src pytest tests/ -v
```

**Run a single test file:**
```bash
cd mirt-dkvmn && PYTHONPATH=src pytest tests/test_shapes.py -v
```

**Generate synthetic data:**
```bash
PYTHONPATH=mirt-dkvmn/src python mirt-dkvmn/scripts/data_gen.py \
  --name synthetic_5000_1000_5_d3 --n_traits 3 --min_seq 120 --max_seq 150 \
  --output_dir mirt-dkvmn/data
```

**Train:**
```bash
PYTHONPATH=mirt-dkvmn/src python mirt-dkvmn/scripts/train.py \
  --config mirt-dkvmn/configs/large_d3_opt3.yaml
```

**Use `smoke.yaml` for rapid iteration** (small dataset, few epochs).

**Plot recovery metrics:**
```bash
PYTHONPATH=mirt-dkvmn/src python mirt-dkvmn/scripts/plot_recovery.py \
  --config mirt-dkvmn/configs/large_d3_opt3.yaml \
  --checkpoint mirt-dkvmn/artifacts/large_d3_opt3/last.pt \
  --output mirt-dkvmn/artifacts/large_d3_opt3/recovery_plots
```

**Plot training metrics:**
```bash
python mirt-dkvmn/scripts/plot_metrics.py \
  --metrics mirt-dkvmn/artifacts/large_d3_opt3/metrics.csv \
  --output mirt-dkvmn/artifacts/large_d3_opt3/metric_plots
```

## Architecture

The active project is `mirt-dkvmn/`. All other top-level directories (`deep-gpcm/`, `dkvmn-ori/`, `dkvmn-torch/`, `akt/`, `deep-1pl/`) are legacy baselines.

**Goal**: Train a neural network on synthetic student response sequences and recover ground-truth MIRT parameters (θ = traits, α = discrimination, β = difficulty).

### Data flow

```
(question_ids, responses)
→ LinearDecayEmbedding      # triangular weights over ordinal categories
→ DKVMN memory (attention + read + write)
→ summary network
→ MIRTParameterExtractor    # produces θ, α, β
→ MIRTGPCMLogits            # K-1 cumulative logits (not K-step)
→ categorical probabilities
```

### Key source files

| File | Role |
|------|------|
| `src/mirt_dkvmn/models/implementations/dkvmn_mirt.py` | Main model |
| `src/mirt_dkvmn/models/components/memory.py` | DKVMN key/value memory |
| `src/mirt_dkvmn/models/components/irt.py` | MIRT parameter extraction + GPCM logits |
| `src/mirt_dkvmn/models/components/embeddings.py` | `LinearDecayEmbedding` |
| `src/mirt_dkvmn/models/heads/gpcm.py` | Active GPCM head |
| `src/mirt_dkvmn/training/trainer.py` | Training loop, metric logging |
| `src/mirt_dkvmn/training/losses.py` | `CombinedOrdinalLoss` (cross-entropy + QWK + ordinal MAE) |
| `src/mirt_dkvmn/utils/data_gen.py` | `MirtGpcmGenerator` — synthetic data with ground-truth IRT params |
| `src/mirt_dkvmn/data/loaders.py` | `DataLoaderManager`, `SequenceDataset`, `collate_sequences` |
| `src/mirt_dkvmn/config/types.py` | `AppConfig` dataclass (Model/Training/Data/Base sub-configs) |

### Configuration

All experiments are driven by YAML configs in `mirt-dkvmn/configs/`. Key parameters:

- `model.n_traits` — number of latent dimensions
- `model.theta_source` — `"summary"` or `"memory"` (where θ is read from)
- `model.concept_aligned_memory` — aligns memory slots to trait dimensions
- `training.qwk_weight`, `training.ordinal_weight` — loss composition
- `training.alpha_norm_weight` / `training.alpha_ortho_weight` — identifiability regularizers
- `base.device` — `"cuda"` or `"cpu"`; falls back to CPU if CUDA unavailable

### Data and artifacts

- Datasets: `mirt-dkvmn/data/synthetic_<students>_<items>_<cats>_d<traits>/`
  - `sequences.json`, `metadata.json`, `true_irt_parameters.json`
- Training outputs: `mirt-dkvmn/artifacts/<config_name>/`
  - `metrics.csv`, `epoch_N.pt`, `last.pt`, confusion matrices, plots

Dataset regeneration must match the current GPCM formulation (K-1 cumulative logits) to keep recovery plots meaningful.
