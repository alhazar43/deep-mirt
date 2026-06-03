# deep-mirt

Memory-augmented item response theory for ordinal knowledge tracing. The
flagship model is **MA-GPCM**, a DKVMN-based architecture that predicts
ordinal student responses and recovers interpretable IRT parameters
(`theta`, `alpha`, `beta`) in a single forward pass.

Paper: *MA-GPCM: A Memory-Augmented Model for Interpretable Ordinal
Knowledge Tracing*. Benchmarks reproducing the paper tables are in
[`benchmarks.md`](benchmarks.md).

## Install

```bash
git clone https://github.com/alhazar43/deep-mirt
cd deep-mirt/ma-irt
pip install -e .
```

That puts the `models` and `utils` packages on `sys.path`, so the three
entry-point scripts run by name from this directory. No `PYTHONPATH`
needed.

PyTorch is listed in the dependencies generically; pick the CUDA build
that matches your driver from the [PyTorch install guide](https://pytorch.org/get-started/locally/)
if you need GPU.

## Quick start

A 1-epoch smoke run on a tiny generated dataset, validating the full
data-gen, train, evaluate loop end-to-end.

```bash
cd ma-irt
python data_gen.py static --name smoke_test \
    --n_students 120 --n_questions 20 --n_cats 4 \
    --min_seq 10 --max_seq 25 --output_dir data --seed 42
python train.py --config configs/smoke.yaml --dataset smoke_test --epochs 1
python evaluate.py single --config configs/smoke.yaml \
    --checkpoint outputs/smoke_test/best.pt --data-dir data/smoke_test
```

`outputs/smoke_test/best.pt`, `metrics.csv`, and `recovery_metrics.json`
appear at the end. Both `data/` and `outputs/` are gitignored — the
public repo ships only the source and the configs.

## Models

Selected via `model.model_type` in YAML configs.

| Paper name | Config | IRT params | Notes |
|---|---|---|---|
| MA-GPCM | `magpcm`, `separate_theta: true` | θ, α, β | Main model |
| DKVMN+GPCM | `magpcm`, `separate_theta: false` | θ, α, β | Shared-pathway ablation |
| DKVMN+Softmax | `dkvmn_softmax` | none | Prediction baseline |
| Dynamic GPCM | `dynamic_gpcm` | θ, α, β | Sequential IRT baseline |
| Static GPCM | `static_gpcm` | θ, α, β | Per-student IRT baseline |
| DKT, DKVMN, Deep-IRT | `dkt`, `dkvmn`, `deep_irt` | none | Binary K=2 baselines |
| GPCM (EM) | R `mirt` package | θ, α, β | Archived under `archive/scripts/` |

The MA-GPCM contribution is the separated ability pathway, set with
`separate_theta: true`, which estimates `theta` from the memory read
state only while item parameters remain item-conditioned. The
`separate_theta: false` ablation reverts to the shared pathway.

## Entry points

| Script | Purpose |
|---|---|
| `python data_gen.py <name> ...` | Generate a synthetic dataset (`static`, `block`, `randomwalk`, `staircase`, `imbalanced`) |
| `python train.py --config <yaml>` | Train any model on any dataset |
| `python evaluate.py single --config <yaml> --checkpoint <pt> --data-dir <dir>` | Prediction metrics + IRT recovery (when a ground-truth file is present) |

All three are at `ma-irt/` top level. They read from `configs/` and
write to `outputs/<experiment_name>/`.

## Repository layout

The portable library is exactly three folders. Everything else is
either an input recipe (`configs/`), test plumbing (`tests/`), or
generated at runtime (`data/`, `outputs/`).

```
deep-mirt/
├── README.md
├── benchmarks.md           # paper benchmark tables
├── CLAUDE.md               # codebase guidance
└── ma-irt/
    ├── pyproject.toml      # pip install -e . target
    ├── LICENSE             # MIT
    ├── README.md           # short pointer back here
    ├── requirements.txt
    ├── train.py            # entry point
    ├── evaluate.py         # entry point
    ├── data_gen.py         # entry point
    ├── models/             # MA-GPCM + baselines, components, trainer
    ├── utils/              # config, dataloader, metrics, losses, datagen
    ├── configs/            # YAML recipes (smoke + bulk sweep)
    ├── data/               # datasets, gitignored
    ├── outputs/            # training artifacts, gitignored
    ├── tests/              # pytest
    └── archive/            # paper-only sweeps, plotting, R baseline, gitignored
```

After `pip install -e .`, `import models`, `import utils`,
`from utils.datagen.static import GPCMDataGenerator`, etc. resolve from
anywhere.

## Tests

```bash
cd ma-irt
pytest tests/
```

The R2 baseline gate (`tests/test_baseline_reproduction.py`) verifies
the MA-GPCM K=4 and ASSIST2009 K=2 cached metrics within the published
tolerance band when those checkpoint directories are present under
`outputs/`. Without them, the test cleanly skips.

## Reproducing the paper benchmarks

The 240-run 5-fold CV sweep and the K=3,4,5,6 recovery sweeps were
driven by shell scripts that now live under
`ma-irt/archive/scripts/`. The aggregated table generators and the R
`mirt` GPCM(EM) baseline are archived alongside. See
[`benchmarks.md`](benchmarks.md) for the cell-by-cell mapping of
configs to table cells.

## License

[MIT](LICENSE).
