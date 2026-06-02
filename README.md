# deep-mirt

Memory-augmented item response theory for ordinal knowledge tracing. The
active project is **MA-GPCM**, a DKVMN-based model that predicts ordinal
student responses and recovers interpretable IRT parameters (`theta`,
`alpha`, `beta`) in a single forward pass.

Paper: **MA-GPCM: A Memory-Augmented Model for Interpretable Ordinal
Knowledge Tracing**.

The active codebase is [`ma-irt/`](ma-irt/). See
[`ma-irt/README.md`](ma-irt/README.md) for the full usage guide. Benchmark
numbers reproducing the paper tables are in [`benchmarks.md`](benchmarks.md).

## Environment

From the repository root:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate research
export PYTHONPATH=ma-irt
```

PowerShell:

```powershell
$env:PYTHONPATH = "ma-irt"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
```

## Quick Start

This smoke path generates synthetic GPCM data, trains MA-GPCM for one epoch,
and evaluates both prediction metrics and IRT parameter recovery. It is a
functionality check, not a paper-performance run.

```bash
cd ma-irt
export PYTHONPATH=.

python scripts/data_gen.py \
  --name smoke_test \
  --n_students 120 \
  --n_questions 20 \
  --n_cats 4 \
  --min_seq 10 \
  --max_seq 25 \
  --output_dir data \
  --seed 42

python scripts/train.py \
  --config configs/smoke.yaml \
  --dataset smoke_test \
  --epochs 1

python scripts/evaluate.py single \
  --config configs/smoke.yaml \
  --checkpoint outputs/smoke_test/best.pt \
  --data-dir data/smoke_test \
  --batch-size 32
```

## Models

Models are selectable via `model.model_type` in YAML configs.

| Paper name | Config setting | IRT params | Notes |
|---|---|---|---|
| MA-GPCM | `model_type: magpcm`, `separate_theta: true` | `theta`, `alpha`, `beta` | Main model |
| DKVMN+GPCM | `model_type: magpcm`, `separate_theta: false` | `theta`, `alpha`, `beta` | Shared-pathway ablation |
| DKVMN+Softmax | `model_type: dkvmn_softmax` | none | Prediction baseline |
| Dynamic GPCM | `model_type: dynamic_gpcm` | `theta`, `alpha`, `beta` | Dynamic IRT baseline |
| Static GPCM | `model_type: static_gpcm` | `theta`, `alpha`, `beta` | Static IRT baseline |
| DKT / DKVMN / Deep-IRT | `dkt`, `dkvmn`, `deep_irt` | none | Binary K=2 baselines |
| GPCM (EM) | R `mirt` package | `theta`, `alpha`, `beta` | Offline batch baseline |

The central MA-GPCM contribution is the separated ability pathway:
`separate_theta: true` estimates `theta` from the memory read state only,
while item parameters remain item-conditioned. The DKVMN+GPCM ablation turns
that separation off.

## Data Generators

| Script | DGP | Ability dynamics |
|---|---|---|
| `data_gen.py` | Static GPCM | Fixed per student |
| `data_gen_staircase.py` | Staircase | Discrete shifts |
| `data_gen_randomwalk.py` | Random walk | Continuous drift |
| `data_gen_block.py` | Block change | Pretest-posttest |
| `data_gen_imbalanced.py` | Imbalanced | Shifted/skewed ability prior |

Real-data evaluation uses proxy-ordinal ASSISTments 2009 and 2017 datasets.

## Repository Layout

```text
deep-mirt/
  ma-irt/          # Active codebase
  overleaf-sync/   # Paper LaTeX source
  docs/            # Cleanup evidence and archived planning notes
  benchmarks.md    # Paper benchmark tables
  CLAUDE.md        # Agent/codebase guidance
  README.md        # This file
```

Legacy directories such as `mirt-dkvmn/`, `deep-gpcm/`, `deep-1pl/`,
`dkt-ori/`, `dkvmn-ori/`, `akt/`, and `pykt/` are inactive references kept
for archival or data-source reasons until cleanup verification proves they can
be moved.

## Research Roadmap

This repository ships MA-GPCM. The next committed research step is multidim
MA-IRT: generalizing scalar `theta_t` to a vector-valued latent state with
multidimensional IRT decoders and explicit identifiability constraints.

## See Also

- [`ma-irt/README.md`](ma-irt/README.md), full usage guide and project layout
- [`benchmarks.md`](benchmarks.md), paper benchmark tables
- [`CLEANUP_PLAN_2026.md`](CLEANUP_PLAN_2026.md), public-repo cleanup plan
- [`CLEANUP_VERIFICATION_2026.md`](CLEANUP_VERIFICATION_2026.md), paper-critical verification contract
- [`docs/cleanup/`](docs/cleanup/), cleanup evidence notes
