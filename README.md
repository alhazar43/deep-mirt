# deep-mirt

Memory-Augmented Item Response Theory for polytomous knowledge tracing. Trains deep neural networks on student response sequences and recovers ground-truth IRT parameters ($\theta$ ability, $\alpha$ discrimination, $\beta$ step thresholds) in a single forward pass.

**Paper**: MA-GPCM, a memory-augmented model for interpretable ordinal knowledge tracing (manuscript under review at IJAIED 2026).

The active codebase is [`ma-irt/`](ma-irt/). See [`ma-irt/README.md`](ma-irt/README.md) for the full usage guide and config reference. Benchmark numbers reproducing the paper tables are in [`benchmarks.md`](benchmarks.md).

## Environment

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate research
export PYTHONPATH=ma-irt
# Windows only
export KMP_DUPLICATE_LIB_OK=TRUE
```

## Quick start

```bash
cd ma-irt

# Generate a synthetic dataset (5000 students, 200 items, 4 categories)
python scripts/data_gen.py \
    --name v2_q200_k4 --n_students 5000 --n_questions 200 --n_cats 4 \
    --min_seq 20 --max_seq 80 --output_dir data

# Train MA-GPCM
PYTHONPATH=. python scripts/train.py --config configs/v2_q200_k4.yaml

# Evaluate (prediction metrics + IRT parameter recovery)
PYTHONPATH=. python scripts/evaluate.py single \
    --config configs/v2_q200_k4.yaml \
    --checkpoint outputs/v2_q200_k4/best.pt \
    --data-dir data/v2_q200_k4

# Plot training curves
python scripts/plot_metrics.py \
    --metrics outputs/v2_q200_k4/metrics.csv \
    --output outputs/v2_q200_k4/plots
```

## Models in the paper

Six models, selectable via `model.model_type` in the config.

| Paper name | `model_type` | IRT params | Notes |
|---|---|---|---|
| **MA-GPCM** (ours) | `magpcm` | $\theta, \alpha, \beta$ | DKVMN encoder + separated ability pathway + GPCM head |
| DKVMN+GPCM | `magpcm` (`separate_theta: false`) | $\theta, \alpha, \beta$ | Shared pathway ablation |
| DKVMN+Softmax | `dkvmn_softmax` | none | DKVMN + $K$-way softmax, no IRT structure |
| Dynamic GPCM | `dynamic_gpcm` | $\theta, \alpha, \beta$ | Gated recurrence, no memory |
| Static GPCM | `static_gpcm` | $\theta, \alpha, \beta$ | Static per-student $\theta$ embedding |
| GPCM (EM) | (R `mirt` package) | $\theta, \alpha, \beta$ | Offline batch baseline via `scripts/mirt_baseline_all_k.R` |

## Data generators

| Script | DGP | $\theta$ dynamics |
|---|---|---|
| `data_gen.py` | Static | Fixed per student |
| `data_gen_staircase.py` | Staircase | 3-level discrete shifts |
| `data_gen_randomwalk.py` | Random walk | Continuous drift |
| `data_gen_block.py` | Block change | Pretest-posttest |
| `data_gen_imbalanced.py` | Imbalanced | Skewed $\theta$ prior |

Real-data evaluation uses ASSISTments 2009 and 2017 with five-fold cross-validation.

## Repository layout

```
deep-mirt/
├── ma-irt/          # Active codebase (see ma-irt/README.md)
├── overleaf-sync/   # Paper LaTeX source
├── benchmarks.md    # Paper benchmark tables
├── CLAUDE.md        # Guidance for Claude Code
└── README.md        # This file
```

Legacy directories (`mirt-dkvmn/`, `deep-gpcm/`, `deep-1pl/`, `dkt-ori`, `dkvmn-ori`, `akt`, `pykt`) are inactive references kept for archival reasons.

## Research roadmap

This codebase ships MA-GPCM. The doctoral program building on it has one committed first step (**multidim MA-IRT**, generalizing scalar $\theta_t$ to $\boldsymbol{\theta}_t \in \mathbb{R}^D$ with multidim 2PL/GPCM/DINA decoders, addressing rotational identifiability under streaming and sparse-attention Q-matrix recovery) plus four follow-up directions presented as options rather than as a fixed pipeline.

- **A.** Joint multidim MA-IRT and DRL recommender. Shared encoder, joint loss, partial-identification-aware off-policy evaluation. Puts MA-IRT inside the decision loop.
- **B.** LLM-mediated assessment. LLM raters and item generators as structured contributions to MA-IRT estimates, not as oracles. Identifiability for trait-vs-rater-bias separation.
- **C.** Learner world model on the MA-IRT state. Long-horizon planning in trait units, causal identifiability for learning dynamics, lifelong measurement under drift.
- **D.** MA-IRT for AI capability evaluation. AI as test-taker, multidim capability profiling, cross-fine-tune tracking, cross-model DIF.

Cross-cutting theoretical thread, identifiability of streaming neural IRT, runs through all four. Two-year scope realistically delivers multidim MA-IRT plus two to three of A through D. Full proposal in `docs/archive/2026-06-cleanup/phd_research_proposal.md`.

## See also

- [`ma-irt/README.md`](ma-irt/README.md), full code-usage guide, config reference, project layout, tests
- [`CLAUDE.md`](CLAUDE.md), commands and architecture summary for Claude Code
- [`benchmarks.md`](benchmarks.md), paper results tables
