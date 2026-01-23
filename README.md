# Deep MIRT DKVMN

This repo hosts multiple knowledge tracing/IRT prototypes. The active work is in
`mirt-dkvmn/`, which implements a multi-dimensional DKVMN with a GPCM head and
synthetic MIRT data generation.

## Environment

Use the `vrec-env` conda environment for all runs:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate vrec-env
```

Most scripts expect `PYTHONPATH=mirt-dkvmn/src`.

## Quick Start

Generate a synthetic dataset (example: 5000 students, 1000 items, 5 categories, 3 traits):

```bash
PYTHONPATH=mirt-dkvmn/src python mirt-dkvmn/scripts/data_gen.py \
  --name synthetic_5000_1000_5_d3 --n_traits 3 --min_seq 120 --max_seq 150 \
  --output_dir mirt-dkvmn/data
```

Train with a config:

```bash
PYTHONPATH=mirt-dkvmn/src python mirt-dkvmn/scripts/train.py \
  --config mirt-dkvmn/configs/large_d3_opt3.yaml
```

Plot recovery/metrics:

```bash
PYTHONPATH=mirt-dkvmn/src python mirt-dkvmn/scripts/plot_recovery.py \
  --config mirt-dkvmn/configs/large_d3_opt3.yaml \
  --checkpoint mirt-dkvmn/artifacts/large_d3_opt3/last.pt \
  --output mirt-dkvmn/artifacts/large_d3_opt3/recovery_plots

python mirt-dkvmn/scripts/plot_metrics.py \
  --metrics mirt-dkvmn/artifacts/large_d3_opt3/metrics.csv \
  --output mirt-dkvmn/artifacts/large_d3_opt3/metric_plots
```

## Repository Layout

- `mirt-dkvmn/`: Current MIRT-DKVMN implementation, configs, data tools, plots.
- `deep-gpcm/`: Reference implementation and legacy scripts.
- `dkvmn-ori/`, `dkvmn-torch/`, `akt/`, `deep-1pl/`: Legacy or comparative baselines.
- `updated_plan.tex`: Math/architecture notes for the MIRT-DKVMN design.

## Data and Artifacts

- Synthetic datasets live under `mirt-dkvmn/data/` and follow
  `synthetic_<students>_<items>_<cats>_d<traits>`.
- Training artifacts (metrics, checkpoints, plots) go in `mirt-dkvmn/artifacts/`.
- Older datasets and results are archived in `mirt-dkvmn/archive/`.

## Notes

- GPU usage is controlled by `base.device` in each config; training falls back to CPU if CUDA is unavailable.
- Dataset regeneration should match the current GPCM formulation to keep recovery plots meaningful.
