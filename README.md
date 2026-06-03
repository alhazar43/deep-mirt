# deep-mirt

Memory-augmented item response theory for ordinal knowledge tracing.

The flagship model, **MA-GPCM**, is a DKVMN-based architecture that predicts
ordinal student responses *and* recovers interpretable IRT parameters
(`theta`, `alpha`, `beta`) in a single forward pass. The repository also
includes five baselines (DKVMN+GPCM, DKVMN+Softmax, Dynamic GPCM, Static
GPCM, DKT, DKVMN, Deep-IRT) and five synthetic data generators (static,
block, random-walk, staircase, imbalanced) for parameter-recovery
experiments.

The paper *MA-GPCM: A Memory-Augmented Model for Interpretable Ordinal
Knowledge Tracing* reproduces from the configs and scripts in this
repository. Cell-by-cell metric reproduction is in
[`benchmarks.md`](benchmarks.md).

## Project structure

```
deep-mirt/
├── README.md               # this file
├── benchmarks.md           # paper benchmark tables
└── ma-irt/                 # working directory; cd here, then run anything
    ├── train.py            # entry point: train a model
    ├── evaluate.py         # entry point: prediction metrics + IRT recovery
    ├── data_gen.py         # entry point: generate a synthetic dataset
    ├── requirements.txt    # Python dependencies
    ├── configs/            # YAML recipes (1 smoke config + bulk sweep)
    ├── models/             # MA-GPCM + baselines, components, trainer
    ├── utils/              # config schema, dataloader, metrics, losses, datagen/
    ├── tests/              # pytest, regression snapshots
    ├── data/               # datasets (gitignored, populated by data_gen.py)
    └── outputs/            # training artifacts (gitignored)
```

Three folders carry the library: `models/`, `utils/`, `data/`. Everything
else is either an input recipe (`configs/`), test plumbing (`tests/`), or
generated at runtime.

## Install

No `pip install` step. Clone the repo, install the listed dependencies into
whatever Python environment you use, and run scripts directly. Python adds
the script's directory to `sys.path` automatically, so imports resolve
without `PYTHONPATH` or an editable install.

```bash
git clone https://github.com/alhazar43/deep-mirt
cd deep-mirt/ma-irt
pip install -r requirements.txt
```

`requirements.txt` pins `torch`, `numpy`, `pyyaml`, `scipy`, and
`scikit-learn`. Pick a PyTorch build matching your CUDA from
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)
if you need GPU.

## Quick start

A one-epoch smoke run on a tiny generated dataset. Run from `ma-irt/`.

```bash
# 1. Generate a small synthetic dataset (writes data/smoke_test/)
python data_gen.py static --name smoke_test \
    --n_students 120 --n_questions 20 --n_cats 4 \
    --min_seq 10 --max_seq 25 --output_dir data --seed 42

# 2. Train for one epoch (writes outputs/smoke_test/)
python train.py --config configs/smoke.yaml --dataset smoke_test --epochs 1

# 3. Evaluate (prediction metrics + IRT recovery; writes recovery_metrics.json)
python evaluate.py single --config configs/smoke.yaml \
    --checkpoint outputs/smoke_test/best.pt \
    --data-dir data/smoke_test
```

## Usage

### Train

```bash
python train.py --config <yaml> [--dataset <name>] [--epochs <N>]
```

- `--config` points at a YAML file under `configs/` (e.g. `configs/smoke.yaml`
  for the smoke check, or any of the 1 632 files under `configs/bulk/` for
  the paper sweep).
- `--dataset <name>` overrides `data.dataset_name` and re-uses the dataset
  directory `data/<name>/`.
- `--epochs <N>` overrides `training.epochs`.
- Output: `outputs/<experiment_name>/{best.pt, last.pt, metrics.csv}`.

### Evaluate

```bash
python evaluate.py single \
    --config <yaml> \
    --checkpoint <best.pt> \
    --data-dir <data/dataset_name> \
    [--batch-size <N>]
```

`evaluate.py single` runs forward inference, computes the prediction
metrics (ACC, AUC for K=2, QWK, MAE, Kendall tau), and, if the dataset has
ground-truth IRT parameters at `data/<name>/true_irt_parameters.json`,
also reports the parameter-recovery numbers (`r_alpha`, `r_beta_mean`,
`r_theta`, `RMSE_theta`) using Kolen & Brennan 2014 mean-sigma linking.
Writes `recovery_metrics.json` next to the checkpoint.

### Generate data

```bash
python data_gen.py <generator> \
    --name <output-dir-name> \
    --n_students <N> --n_questions <Q> --n_cats <K> \
    --min_seq <S_min> --max_seq <S_max> \
    --output_dir data --seed <seed>
```

`<generator>` is one of:

| Generator | Ability dynamics |
|---|---|
| `static` | Fixed `theta` per student (paper-default recovery experiments). |
| `block` | Pretest / posttest with a discrete block change. |
| `randomwalk` | Continuous Brownian drift in `theta`. |
| `staircase` | Three-level discrete ability shifts. |
| `imbalanced` | Static-theta variant with a skewed response distribution. |

Writes `data/<name>/{sequences.json, metadata.json, true_irt_parameters.json}`.

### Run tests

```bash
pytest tests/
```

No `PYTHONPATH` or editable install is needed; `tests/__init__.py` makes
`tests/` a package and pytest sets the project root automatically. The
104-test suite
covers shape/contract checks, the YAML loader, the 5 migration regression
snapshots, the GPCM head, and an end-to-end public-pipeline smoke
(data_gen → train → evaluate).

A subset of tests under `test_baseline_reproduction.py` verifies the
cached MA-GPCM K=4 and ASSIST2009 K=2 metrics against the published
tolerance band. They skip cleanly when the matching `outputs/.../*.pt`
checkpoints are missing.

## Models

Select via `model.model_type` in any YAML config.

| Paper name | Config setting | IRT params | Notes |
|---|---|---|---|
| MA-GPCM | `model_type: magpcm`, `separate_theta: true` | θ, α, β | Main model |
| DKVMN+GPCM | `model_type: magpcm`, `separate_theta: false` | θ, α, β | Shared-pathway ablation |
| DKVMN+Softmax | `model_type: dkvmn_softmax` | none | Prediction baseline |
| Dynamic GPCM | `model_type: dynamic_gpcm` | θ, α, β | Sequential IRT baseline |
| Static GPCM | `model_type: static_gpcm` | θ, α, β | Per-student IRT baseline |
| DKT | `model_type: dkt` | none | Binary K=2 baseline |
| DKVMN | `model_type: dkvmn` | none | Binary K=2 baseline |
| Deep-IRT | `model_type: deep_irt` | none | Binary K=2 baseline |

The MA-GPCM contribution is the **separated ability pathway**
(`separate_theta: true`), which estimates `theta` from the memory read
state only while item parameters remain item-conditioned. Setting
`separate_theta: false` reverts to the shared pathway and gives the
DKVMN+GPCM ablation cell.

## Paper reproduction

The 240-run 5-fold cross-validation sweep and the K=3,4,5,6 recovery
sweeps run from configs under `configs/bulk/`. The aggregator scripts and
the R `mirt` GPCM(EM) baseline that produced the rest of the paper tables
are kept under `ma-irt/archive/` (gitignored) — see
[`benchmarks.md`](benchmarks.md) for the cell-by-cell mapping.

The R2 invariant for the headline MA-GPCM K=4 fold-0 cell is checked by
`tests/test_baseline_reproduction.py` against the cached
`outputs/magpcm_static_q200_k4_fold0/{test_metrics, recovery_metrics}.json`.
Reproduction passes within the published tolerance for `ACC`, `QWK`,
`MAE`, `r_alpha`, `r_beta`, `r_theta`, and `RMSE_theta`.

## License

[MIT](LICENSE).
