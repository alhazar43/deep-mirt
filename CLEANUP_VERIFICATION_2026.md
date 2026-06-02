# Cleanup verification specification

Date written, 2026-06-02. This document is the safety net for a planned cleanup of the deep-mirt repository. It enumerates what must keep working so the three tables in `benchmarks.md` and the corresponding paper rows in `overleaf-sync/main.tex` can still be reproduced after the cleanup. Nothing in this document executes a cleanup. It defines the verification suite the cleanup planner will run before, during, and after structural changes.

The repository root is `C:\Users\steph\documents\deep-mirt`. The active codebase is `ma-irt/`. The PYTHONPATH the rest of this document assumes is `ma-irt` (set from the repo root) or `.` (set from inside `ma-irt/`).

---

## 1. Paper-critical experiments

This section enumerates, for each table in `benchmarks.md`, the exact configs, scripts, models, and datasets that produced the reported numbers. Note that no `run_all_experiments.sh` exists in the repo as of today. The orchestration lives in `ma-irt/scripts/_run_*_sweep.sh` and `_run_*_recovery.sh` shell scripts, plus the per-table aggregators in `_aggregate_*.py`.

### 1.1 Table 1, ordinal prediction on Synthetic-Static, K in {3, 4, 5, 6}

**Source**, `benchmarks.md` section "Ordinal prediction on Synthetic-Static". Corresponds to `tab:comp_results` in `overleaf-sync/main.tex`.

**Reproduction protocol**, five-fold pyKT-style CV (five folds, not five seeds), `test_frac=0.2`, `split_seed=42`, fold ids 0..4. Standard deviations are over folds.

**Configs**, located in `ma-irt/configs/bulk/`. The naming pattern is `bench_<model>_static_q200_k<K>_pykt_fold<F>.yaml`.

| Model in table | model_type | Config glob |
|---|---|---|
| Static GPCM | `static_gpcm` | `bench_static_gpcm_static_q200_k{3,4,5,6}_pykt_fold{0,1,2,3,4}.yaml` |
| Dynamic GPCM | `dynamic_gpcm` | `bench_dynamic_gpcm_static_q200_k{3,4,5,6}_pykt_fold{0,1,2,3,4}.yaml` |
| DKVMN+Softmax | `dkvmn_softmax` | `bench_dkvmn_softmax_static_q200_k{3,4,5,6}_pykt_fold{0,1,2,3,4}.yaml` |
| DKVMN+GPCM | `magpcm` with `separate_theta=false` | `bench_dkvmn_gpcm_static_q200_k{3,4,5,6}_pykt_fold{0,1,2,3,4}.yaml` |
| MA-GPCM | `magpcm` with `separate_theta=true` | `bench_magpcm_static_q200_k{3,4,5,6}_pykt_fold{0,1,2,3,4}.yaml` |

Total configs, 5 models x 4 K-values x 5 folds = 100.

**Scripts invoked**

- Training and per-fold eval, `ma-irt/scripts/_run_k4_cv_recovery.sh` (K=4 block) and `ma-irt/scripts/_run_k356_cv_recovery.sh` (K=3, 5, 6 block). Each script calls `python scripts/train.py --config ...` then `python scripts/evaluate.py single --config ... --checkpoint ... --data-dir ...`.
- Aggregator, `ma-irt/scripts/aggregate_recovery_v5.py` (parses `outputs/bench_<model>_static_q200_k<K>_pykt_fold<F>/recovery_metrics.json` and `test_metrics.json`).

**Model files required**

- `ma-irt/models/magpcm.py` (MA-GPCM and DKVMN+GPCM)
- `ma-irt/models/static_gpcm.py`
- `ma-irt/models/dynamic_gpcm.py`
- `ma-irt/models/dkvmn_softmax.py`
- `ma-irt/models/components/memory.py`, `ma-irt/models/components/irt.py`, `ma-irt/models/components/embeddings.py`
- `ma-irt/models/heads/gpcm.py`

**Datasets required**

- `ma-irt/data/static_q200_k3/`
- `ma-irt/data/static_q200_k4/`
- `ma-irt/data/static_q200_k5/`
- `ma-irt/data/static_q200_k6/`

Each contains `sequences.json`, `metadata.json`, `true_irt_parameters.json`.

**Regeneration recipe if datasets are missing**

```
cd ma-irt
python scripts/data_gen.py --name static_q200_k3 --n_students 5000 --n_questions 200 --n_cats 3 --min_seq 20 --max_seq 80 --output_dir data --seed 42
python scripts/data_gen.py --name static_q200_k4 --n_students 5000 --n_questions 200 --n_cats 4 --min_seq 20 --max_seq 80 --output_dir data --seed 42
python scripts/data_gen.py --name static_q200_k5 --n_students 5000 --n_questions 200 --n_cats 5 --min_seq 20 --max_seq 80 --output_dir data --seed 42
python scripts/data_gen.py --name static_q200_k6 --n_students 5000 --n_questions 200 --n_cats 6 --min_seq 20 --max_seq 80 --output_dir data --seed 42
```

**Seeds and folds**, fold_id in {0, 1, 2, 3, 4} (the pyKT CV split), with `split_seed=42`. The reported sd is over the five folds.

### 1.2 Table 2, binary prediction at K=2 on Synthetic-Static, Synthetic-5, ASSIST2009, ASSIST2017

**Source**, `benchmarks.md` section "Binary prediction (K = 2), five-fold CV". Corresponds to `tab:combined_perf` in `overleaf-sync/main.tex`.

**Reproduction protocol**, five-fold pyKT-style CV. For Synthetic-5, five dataset versions are each run with five folds. Per-version means are averaged, sd is computed across the five per-version means (matches `_aggregate_pykt_results.py`, lines 71..83).

**Configs**, located in `ma-irt/configs/bulk/`. Naming pattern, `bench_<model>_<dataset>_pykt_fold<F>.yaml`.

| Model in table | model_type | Configs |
|---|---|---|
| DKT | `dkt` | `bench_dkt_<dataset>_pykt_fold{0..4}.yaml` |
| DKVMN | `dkvmn` | `bench_dkvmn_<dataset>_pykt_fold{0..4}.yaml` |
| Deep-IRT | `deep_irt` | `bench_deep_irt_<dataset>_pykt_fold{0..4}.yaml` |
| DKVMN+GPCM | `magpcm` with `separate_theta=false` | `bench_dkvmn_gpcm_<dataset>_pykt_fold{0..4}.yaml` |
| MA-GPCM | `magpcm` with `separate_theta=true` | `bench_magpcm_<dataset>_pykt_fold{0..4}.yaml` |

Datasets in column order, `<dataset>` is one of `static_q200_k2`, `synthetic5_v{0..4}`, `assist2009_bin`, `assist2017_bin`. Total configs, 5 models x (1 + 5 + 1 + 1) datasets x 5 folds = 200.

**Scripts invoked**

- `ma-irt/scripts/_run_pykt_sweep.sh` (the 240-run sweep that also covers the ordinal ASSIST K=4 rows). Phase 1 covers Synthetic-5 and Synthetic-Static binary, Phase 3 covers ASSIST2009/2017 binary.
- Aggregator, `ma-irt/scripts/_aggregate_pykt_results.py`.

**Model files required (additional beyond Section 1.1)**

- `ma-irt/models/dkt.py`
- `ma-irt/models/dkvmn.py`
- `ma-irt/models/deep_irt.py`

**Datasets required**

- `ma-irt/data/static_q200_k2/`
- `ma-irt/data/synthetic5_v0/`, `synthetic5_v1/`, `synthetic5_v2/`, `synthetic5_v3/`, `synthetic5_v4/`
- `ma-irt/data/assist2009_bin/`
- `ma-irt/data/assist2017_bin/`

**Regeneration recipes**

- `static_q200_k2`, run `python scripts/data_gen.py --name static_q200_k2 --n_students 5000 --n_questions 200 --n_cats 2 --min_seq 20 --max_seq 80 --output_dir data --seed 42`.
- `synthetic5_v{0..4}`, run `python scripts/_build_pykt_synthetic5.py` (reads `dkvmn-ori/data/synthetic/naive_c5_q50_s4000_v{V}_*.csv`). External raw files must exist at `C:\Users\steph\documents\deep-mirt\dkvmn-ori\data\synthetic\`.
- `assist2009_bin`, derived from `assisstment-raw/skill_builder_data_2009.csv` via `python scripts/convert_assistments_2009.py` (binary variant, see script for the binary flag).
- `assist2017_bin`, derived from `assisstment-raw/anonymized_full_release_competition_dataset.csv` via `python scripts/convert_assistments.py`.

**Seeds and folds**, `fold_id` in {0..4}, `split_seed=42`. For Synthetic-5 the version index also varies, so the cell is over 25 runs per model, summarised as mean of 5 per-version means with sd over those means.

### 1.3 Table 3, IRT parameter recovery on Synthetic-Static, K in {3, 4, 5, 6}

**Source**, `benchmarks.md` section "IRT parameter recovery on Synthetic-Static". Corresponds to `tab:irt_recovery_k` in `overleaf-sync/main.tex`.

**Reproduction protocol**, identical configs to Section 1.1 (same `bench_<model>_static_q200_k<K>_pykt_fold<F>.yaml` files). The recovery numbers come from `outputs/<run>/recovery_metrics.json`, which `scripts/evaluate.py single` writes after the model is trained. The GPCM (EM) ceiling row is produced separately by R.

**Additional script**, `ma-irt/scripts/mirt_baseline_all_k.R`, run via Rscript with `mirt` package installed. This produces the italicised GPCM (EM) row.

**Model files**, same as Section 1.1. Note the Static GPCM model contributes only `r_theta` and `RMSE_theta` rows since DKT/DKVMN/Deep-IRT are not run on this table.

**Datasets required**, same as Section 1.1 (the static K=3..6 datasets).

**Linking**, log-space z-score with target std 0.3 for alpha, z-score for beta, implemented in `scripts/evaluate.py` (`link_zscore`, `mean_sigma_link`). The R baseline applies the equivalent `mean_sigma_link` from `mirt_baseline_all_k.R`.

**Seeds and folds**, same five folds as Section 1.1. The `_aggregate_recovery_v5.py` script pools by `(dgp, model, K)` and emits mean +- sd.

### 1.4 Shared file inventory (intersection of sections 1.1 to 1.3)

These files must remain functional after cleanup. See Section 3 for the hard-stop list.

- Models, `magpcm.py`, `static_gpcm.py`, `dynamic_gpcm.py`, `dkvmn_softmax.py`, `dkt.py`, `dkvmn.py`, `deep_irt.py`, plus components and heads.
- Library, `config/{types.py,loader.py,__init__.py}`, `dataloading/loaders.py`, `training/{trainer.py,losses.py}`, `utils/metrics.py`.
- Scripts, `train.py`, `evaluate.py`, `data_gen.py`, `convert_assistments.py`, `convert_assistments_2009.py`, `_build_pykt_synthetic5.py`, `_aggregate_pykt_results.py`, `_aggregate_bench.py`, `aggregate_recovery_v5.py`, `mirt_baseline_all_k.R`, `mirt_predict.R`.
- Sweep drivers, `_run_pykt_sweep.sh`, `_run_k4_cv_recovery.sh`, `_run_k356_cv_recovery.sh`.

---

## 2. Minimal verification suite

Three layers, each with its own pass criteria. The cleanup is acceptable if Layer 1 and Layer 2 both pass; Layer 3 is the optional regression net.

### 2.1 Smoke layer, target wall time under 30 minutes on a single GPU

Verifies every model class trains for one or two epochs and produces metrics in the right ballpark. Catches import breakage, dataclass field removal, checkpoint key mismatches, model registry changes, dataloader regressions.

**Configs to run**, one per model type, all from `ma-irt/configs/`.

| # | Config | Model | Purpose |
|---|---|---|---|
| 1 | `configs/smoke.yaml` | MA-GPCM (default) | sanity of MAGPCM forward + GPCM head |
| 2 | `configs/smoke_dkt.yaml` | DKT | DKT model intact |
| 3 | `configs/smoke_dkvmn.yaml` | DKVMN | DKVMN baseline intact |
| 4 | `configs/smoke_deep_irt.yaml` | Deep-IRT | Deep-IRT baseline intact |
| 5 | A copy of `smoke.yaml` with `model_type: static_gpcm` | Static GPCM | Static IRT baseline forward |
| 6 | A copy of `smoke.yaml` with `model_type: dynamic_gpcm` | Dynamic GPCM | Dynamic IRT baseline forward |
| 7 | A copy of `smoke.yaml` with `model_type: dkvmn_softmax` | DKVMN+Softmax | softmax-head DKVMN intact |

Configs 5..7 are generated on the fly by the verification driver; they are not paper-critical, just smoke fixtures. Two epochs is enough; `smoke.yaml` already specifies `epochs: 2`.

**Prerequisites**, the `smoke_test` dataset must exist. If not, run

```
cd ma-irt
python scripts/data_gen.py --name smoke_test --n_questions 20 --n_cats 4 --n_students 200 --output_dir data --seed 0
```

**Tests**

- `cd ma-irt && PYTHONPATH=. pytest tests/ -v`, all five test files must pass (`test_config_loader.py`, `test_heads.py`, `test_losses.py`, `test_optimization_equivalence.py`, `test_shapes.py`).
- For each of the 7 configs above, `python scripts/train.py --config <cfg>` must exit 0 and write `outputs/<exp_name>/best.pt` and `outputs/<exp_name>/metrics.csv` with at least one row.

**Acceptance**, exit code 0 from pytest, and the seven smoke trainings must each produce a `best.pt` plus a `metrics.csv` whose `val_categorical_accuracy` is finite and strictly positive. No regression threshold here, since two-epoch numbers are uninformative.

### 2.2 Regression layer, target wall time 2 to 4 hours on a single GPU

Verifies that one representative slice of the paper actually reproduces. K=4 is chosen because it appears in all three tables and in every figure. One fold is used (`fold0`) so the runtime is one fifth of the full K=4 column.

**Configs to run** (15 configs, all from `ma-irt/configs/bulk/`)

| Config | Maps to row in benchmarks.md |
|---|---|
| `bench_static_gpcm_static_q200_k4_pykt_fold0.yaml` | Tables 1 and 3, Static GPCM at K=4 |
| `bench_dynamic_gpcm_static_q200_k4_pykt_fold0.yaml` | Tables 1 and 3, Dynamic GPCM at K=4 |
| `bench_dkvmn_softmax_static_q200_k4_pykt_fold0.yaml` | Table 1, DKVMN+Softmax at K=4 |
| `bench_dkvmn_gpcm_static_q200_k4_pykt_fold0.yaml` | Tables 1 and 3, DKVMN+GPCM at K=4 |
| `bench_magpcm_static_q200_k4_pykt_fold0.yaml` | Tables 1 and 3, MA-GPCM at K=4 |
| `bench_dkt_static_q200_k2_pykt_fold0.yaml` | Table 2, DKT Synthetic-Static |
| `bench_dkvmn_static_q200_k2_pykt_fold0.yaml` | Table 2, DKVMN Synthetic-Static |
| `bench_deep_irt_static_q200_k2_pykt_fold0.yaml` | Table 2, Deep-IRT Synthetic-Static |
| `bench_dkvmn_gpcm_static_q200_k2_pykt_fold0.yaml` | Table 2, DKVMN+GPCM Synthetic-Static |
| `bench_magpcm_static_q200_k2_pykt_fold0.yaml` | Table 2, MA-GPCM Synthetic-Static |
| `bench_dkt_assist2009_bin_pykt_fold0.yaml` | Table 2, DKT ASSIST2009 |
| `bench_dkvmn_assist2009_bin_pykt_fold0.yaml` | Table 2, DKVMN ASSIST2009 |
| `bench_deep_irt_assist2009_bin_pykt_fold0.yaml` | Table 2, Deep-IRT ASSIST2009 |
| `bench_dkvmn_gpcm_assist2009_bin_pykt_fold0.yaml` | Table 2, DKVMN+GPCM ASSIST2009 |
| `bench_magpcm_assist2009_bin_pykt_fold0.yaml` | Table 2, MA-GPCM ASSIST2009 |

**Execution**

```
cd ma-irt
export PYTHONPATH=. KMP_DUPLICATE_LIB_OK=TRUE
for cfg in <the 15 configs above>; do
  python scripts/train.py --config "configs/bulk/$cfg"
  python scripts/evaluate.py single --config "configs/bulk/$cfg" \
    --checkpoint "outputs/$(basename $cfg .yaml)/best.pt"
done
```

**Metrics compared and acceptance thresholds**

The reference is the column for fold 0 from each row, drawn from `benchmarks.md`. Since `benchmarks.md` reports mean +- sd across five folds, we cannot assume the single-fold-0 value equals the mean. To make this layer meaningful but not flaky, the acceptance threshold is

- the metric must fall within `mean +/- max(2 * sd, 0.01)` of the published cell, on the headline metric for that row.

Headline metric per row,

| Row | Headline metric | Cell (mean +- sd from benchmarks.md) | Acceptance interval |
|---|---|---|---|
| Static GPCM K=4 ordinal | QWK | 0.305 ± 0.000 | [0.295, 0.315] |
| Dynamic GPCM K=4 ordinal | QWK | 0.628 ± 0.001 | [0.618, 0.638] |
| DKVMN+Softmax K=4 ordinal | QWK | 0.647 ± 0.001 | [0.637, 0.657] |
| DKVMN+GPCM K=4 ordinal | QWK | 0.680 ± 0.000 | [0.670, 0.690] |
| MA-GPCM K=4 ordinal | QWK | 0.681 ± 0.001 | [0.671, 0.691] |
| DKT K=2 Static | AUC | 77.35 ± 0.09 | [77.16, 77.54] |
| DKVMN K=2 Static | AUC | 78.31 ± 0.07 | [78.16, 78.46] |
| Deep-IRT K=2 Static | AUC | 78.28 ± 0.03 | [78.21, 78.35], widened to [78.18, 78.38] for noise |
| DKVMN+GPCM K=2 Static | AUC | 78.18 ± 0.05 | [78.07, 78.29] |
| MA-GPCM K=2 Static | AUC | 78.11 ± 0.03 | [78.04, 78.18], widened to [78.01, 78.21] |
| DKT K=2 ASSIST2009 | AUC | 83.70 ± 0.14 | [83.41, 83.99] |
| DKVMN K=2 ASSIST2009 | AUC | 83.19 ± 0.07 | [83.04, 83.34] |
| Deep-IRT K=2 ASSIST2009 | AUC | 83.09 ± 0.29 | [82.50, 83.68] |
| DKVMN+GPCM K=2 ASSIST2009 | AUC | 83.47 ± 0.32 | [82.82, 84.12] |
| MA-GPCM K=2 ASSIST2009 | AUC | 83.50 ± 0.23 | [83.03, 83.97] |

Additionally, for the five K=4 ordinal rows on MA-GPCM-family models (rows 1..5), also check the recovery metrics from `recovery_metrics.json`,

| Row | r_alpha cell | r_beta cell | r_theta cell |
|---|---|---|---|
| Static GPCM K=4 | 0.447 ± 0.028 | 0.947 ± 0.002 | 0.968 ± 0.000 |
| Dynamic GPCM K=4 | 0.842 ± 0.007 | 0.964 ± 0.001 | 0.936 ± 0.001 |
| DKVMN+GPCM K=4 | 0.880 ± 0.013 | 0.631 ± 0.015 | 0.938 ± 0.001 |
| MA-GPCM K=4 | 0.894 ± 0.009 | 0.967 ± 0.002 | 0.957 ± 0.001 |

For each, accept if `|fold0_value - mean| <= max(2 * sd, 0.02)`. DKVMN+Softmax is not in this table.

**Pass criterion for the regression layer**, all 15 headline metrics inside their intervals, and all 12 recovery metrics inside their intervals. One failure means the layer fails.

### 2.3 Full regression layer (optional, ~42 hours)

Triggered only if Layer 2 flagged a discrepancy. This regenerates every cell in `benchmarks.md` from scratch.

**Configs**, the full set in `ma-irt/configs/bulk/` matching the patterns from Section 1, in three sweep scripts.

| Sweep script | Coverage |
|---|---|
| `_run_pykt_sweep.sh` | Table 2 (all binary cells), plus the ordinal ASSIST K=4 rows used by Table at `tab:assistments_pred` |
| `_run_k4_cv_recovery.sh` | K=4 column of Tables 1 and 3 across all three DGPs (static, discrete, continuous) |
| `_run_k356_cv_recovery.sh` | K=3, 5, 6 columns of Tables 1 and 3 across all three DGPs |

**Execution**

```
cd ma-irt
export PYTHONPATH=. KMP_DUPLICATE_LIB_OK=TRUE
bash scripts/_run_pykt_sweep.sh
bash scripts/_run_k4_cv_recovery.sh
bash scripts/_run_k356_cv_recovery.sh
python scripts/_aggregate_pykt_results.py > outputs/agg/pykt_table.tex
python scripts/aggregate_recovery_v5.py
"/c/Program Files/R/R-4.5.0/bin/Rscript.exe" scripts/mirt_baseline_all_k.R
```

**Acceptance**, every cell in every table of `benchmarks.md` must reproduce within 1% relative on the headline metric, or within 0.01 absolute on probabilities and correlations, whichever is tighter. Specifically, for each cell with published value `m`, the recomputed value `m'` must satisfy `|m' - m| <= max(0.01 * |m|, 0.01)` for accuracies and `|m' - m| <= max(0.005 * |m|, 0.005)` for correlations and RMSE.

The five per-fold runs already build sd in, so if the recomputed mean is within 1% of the published mean and the recomputed sd is within 50% of the published sd, the cell passes.

---

## 3. Files that must never be deleted

Deletion of any item below is a stop-the-line event. These are the files that, if removed, break the reproduction of the three benchmark tables.

### 3.1 Library code (under `ma-irt/`)

```
ma-irt/config/__init__.py
ma-irt/config/loader.py
ma-irt/config/types.py
ma-irt/dataloading/__init__.py
ma-irt/dataloading/loaders.py
ma-irt/training/__init__.py
ma-irt/training/losses.py
ma-irt/training/trainer.py
ma-irt/utils/__init__.py
ma-irt/utils/metrics.py
ma-irt/models/__init__.py
ma-irt/models/magpcm.py
ma-irt/models/static_gpcm.py
ma-irt/models/dynamic_gpcm.py
ma-irt/models/dkvmn_softmax.py
ma-irt/models/dkt.py
ma-irt/models/dkvmn.py
ma-irt/models/deep_irt.py
ma-irt/models/components/__init__.py
ma-irt/models/components/memory.py
ma-irt/models/components/irt.py
ma-irt/models/components/embeddings.py
ma-irt/models/heads/__init__.py
ma-irt/models/heads/gpcm.py
```

### 3.2 Scripts (under `ma-irt/scripts/`)

```
ma-irt/scripts/train.py
ma-irt/scripts/evaluate.py
ma-irt/scripts/data_gen.py
ma-irt/scripts/convert_assistments.py
ma-irt/scripts/convert_assistments_2009.py
ma-irt/scripts/_build_pykt_synthetic5.py
ma-irt/scripts/_aggregate_pykt_results.py
ma-irt/scripts/_aggregate_bench.py
ma-irt/scripts/aggregate_recovery_v5.py
ma-irt/scripts/mirt_baseline_all_k.R
ma-irt/scripts/mirt_predict.R
ma-irt/scripts/_run_pykt_sweep.sh
ma-irt/scripts/_run_k4_cv_recovery.sh
ma-irt/scripts/_run_k356_cv_recovery.sh
```

### 3.3 Configs (under `ma-irt/configs/`)

Required globs, every file matching these patterns,

```
ma-irt/configs/bulk/bench_static_gpcm_static_q200_k{3,4,5,6}_pykt_fold{0,1,2,3,4}.yaml
ma-irt/configs/bulk/bench_dynamic_gpcm_static_q200_k{3,4,5,6}_pykt_fold{0,1,2,3,4}.yaml
ma-irt/configs/bulk/bench_dkvmn_softmax_static_q200_k{3,4,5,6}_pykt_fold{0,1,2,3,4}.yaml
ma-irt/configs/bulk/bench_dkvmn_gpcm_static_q200_k{2,3,4,5,6}_pykt_fold{0,1,2,3,4}.yaml
ma-irt/configs/bulk/bench_magpcm_static_q200_k{2,3,4,5,6}_pykt_fold{0,1,2,3,4}.yaml
ma-irt/configs/bulk/bench_{dkt,dkvmn,deep_irt}_static_q200_k2_pykt_fold{0,1,2,3,4}.yaml
ma-irt/configs/bulk/bench_{dkt,dkvmn,deep_irt,dkvmn_gpcm,magpcm}_synthetic5_v{0,1,2,3,4}_pykt_fold{0,1,2,3,4}.yaml
ma-irt/configs/bulk/bench_{dkt,dkvmn,deep_irt,dkvmn_gpcm,magpcm}_assist2009_bin_pykt_fold{0,1,2,3,4}.yaml
ma-irt/configs/bulk/bench_{dkt,dkvmn,deep_irt,dkvmn_gpcm,magpcm}_assist2017_bin_pykt_fold{0,1,2,3,4}.yaml
```

Smoke configs (Layer 1 prerequisite),

```
ma-irt/configs/smoke.yaml
ma-irt/configs/smoke_dkt.yaml
ma-irt/configs/smoke_dkvmn.yaml
ma-irt/configs/smoke_deep_irt.yaml
ma-irt/configs/base.yaml
```

### 3.4 Datasets (under `ma-irt/data/`)

```
ma-irt/data/static_q200_k2/{sequences.json,metadata.json,true_irt_parameters.json}
ma-irt/data/static_q200_k3/{sequences.json,metadata.json,true_irt_parameters.json}
ma-irt/data/static_q200_k4/{sequences.json,metadata.json,true_irt_parameters.json}
ma-irt/data/static_q200_k5/{sequences.json,metadata.json,true_irt_parameters.json}
ma-irt/data/static_q200_k6/{sequences.json,metadata.json,true_irt_parameters.json}
ma-irt/data/synthetic5_v0/{sequences.json,metadata.json}
ma-irt/data/synthetic5_v1/{sequences.json,metadata.json}
ma-irt/data/synthetic5_v2/{sequences.json,metadata.json}
ma-irt/data/synthetic5_v3/{sequences.json,metadata.json}
ma-irt/data/synthetic5_v4/{sequences.json,metadata.json}
ma-irt/data/assist2009_bin/{sequences.json,metadata.json}
ma-irt/data/assist2017_bin/{sequences.json,metadata.json}
```

All static synthetic datasets are deterministically regenerable from `scripts/data_gen.py` with seed 42 (see Section 1.1). The Synthetic-5 family requires the external `dkvmn-ori/data/synthetic/` raw CSVs. The ASSIST datasets require the raw CSVs under `assisstment-raw/` (sibling of `deep-mirt/`).

### 3.5 Tests (under `ma-irt/tests/`)

```
ma-irt/tests/__init__.py
ma-irt/tests/test_config_loader.py
ma-irt/tests/test_heads.py
ma-irt/tests/test_losses.py
ma-irt/tests/test_optimization_equivalence.py
ma-irt/tests/test_shapes.py
```

### 3.6 Documentation, paper sources, and headline tables

```
CLAUDE.md
benchmarks.md
ma-irt/README.md
ma-irt/requirements.txt
overleaf-sync/main.tex
overleaf-sync/ref.bib
overleaf-sync/figures/
```

### 3.7 Existing outputs that should not be discarded without thought

If `ma-irt/outputs/bench_*_pykt_fold*/test_metrics.json` and `recovery_metrics.json` exist, they are the cached results behind the published tables. Deleting them forces a 42-hour rerun. Treat them as artifacts, not intermediate files.

---

## 4. Files that look paper-critical but are not

This is the trickier list. Surface-level scans will flag these as in-use, but they can be removed without affecting paper reproduction.

### 4.1 Scripts that are superseded or one-off

These were used during development but no longer contribute to the published numbers. Listed in `ma-irt/CLEANUP.md` and `ma-irt/CLEANUP_PLAN.md` as "to archive".

```
ma-irt/scripts/_bench_table_draft.tex                  # draft, not the active LaTeX
ma-irt/scripts/_bench_writeup_draft.md                 # draft writeup
ma-irt/scripts/_convert_yeung_synthetic.py             # synthetic5_yeung dataset not in benchmarks.md
ma-irt/scripts/_emit_k4_tables.py                      # superseded by aggregate_recovery_v5.py
ma-irt/scripts/_extract_row.py                         # one-off table digest
ma-irt/scripts/_gen_chunked_bench_configs.py           # chunked30 sweeps not in benchmarks.md
ma-irt/scripts/_gen_imb_scale_pykt_configs.py          # imbalance/scaling tables are separate sections, not the three headline tables
ma-irt/scripts/_gen_pykt_configs.py                    # configs already generated, frozen in repo
ma-irt/scripts/_gen_table_rows.py                      # draft table emitter
ma-irt/scripts/_k4_digest.sh                           # one-off digest log
ma-irt/scripts/_linking_learned.py                     # linking experiment, not in headline tables
ma-irt/scripts/_orbit_align_static_experiment.py       # one-off experiment
ma-irt/scripts/_profile_dkvmn.py                       # profiling script
ma-irt/scripts/_profile_dkvmn_report.md                # profiling report
ma-irt/scripts/_reeval_discrete.sh                     # one-off re-eval
ma-irt/scripts/_run_assist2017_binary.sh               # superseded by _run_pykt_sweep.sh
ma-irt/scripts/_run_chunked_bench_seeds.sh             # chunked30 variants not in benchmarks.md
ma-irt/scripts/_run_chunked_bench_sweep.sh             # ditto
ma-irt/scripts/_run_imb_scale_cv.sh                    # imbalance section is separate
ma-irt/scripts/_run_softmax_cv.sh                      # one-off softmax CV
ma-irt/scripts/_run_synthetic5_shuf_sweep.sh           # shuffled variant, not headline
ma-irt/scripts/_run_synthetic5_yeung_sweep.sh          # Yeung variant, not headline
ma-irt/scripts/_verify_bench_configs.py                # config audit tool, useful but not on the path
ma-irt/scripts/_verify_datasets.py                     # dataset audit tool
ma-irt/scripts/_verify_fix_behavior.py                 # one-off
ma-irt/scripts/aggregate_recovery_v4.py                # superseded by v5
ma-irt/scripts/analyze_threshold_ordering.py           # one-off analysis
ma-irt/scripts/compare_alpha1.py                       # alpha=1 ablation, not in headline tables
ma-irt/scripts/compute_linking.py                      # used inline by evaluate.py now
ma-irt/scripts/convert_assistments.py is critical (see Section 3) but convert_dkvmn_format.py is not
ma-irt/scripts/convert_dkvmn_format.py                 # historical converter, raw inputs no longer used
ma-irt/scripts/data_gen_block.py                       # discrete DGP, used for tab:block_* not for benchmarks.md
ma-irt/scripts/data_gen_imbalanced.py                  # imbalance section, separate from benchmarks.md
ma-irt/scripts/data_gen_randomwalk.py                  # continuous DGP, used for tab:rw_* not for benchmarks.md
ma-irt/scripts/data_gen_staircase.py                   # staircase DGP, not in benchmarks.md
ma-irt/scripts/diag_alpha_collapse.py                  # diagnostic
ma-irt/scripts/eval_retrained.py                       # one-off
ma-irt/scripts/eval_all_collect.sh                     # one-off collector
ma-irt/scripts/eval_and_compare_learned.sh             # one-off
ma-irt/scripts/eval_remaining.sh                       # one-off
ma-irt/scripts/gen_ablation_configs.py                 # ablation, separate
ma-irt/scripts/gen_ablation_data.sh                    # ablation, separate
ma-irt/scripts/gen_all_configs.py                      # configs already frozen in repo
ma-irt/scripts/gen_alpha1_configs.py                   # alpha=1 ablation
ma-irt/scripts/gen_assist_learned_configs.py           # learned-emb ASSIST variants, not headline
ma-irt/scripts/gen_bench_configs.py                    # configs already frozen in repo
ma-irt/scripts/gen_raw_alpha_configs.py                # variant configs
ma-irt/scripts/gen_table_updates.py                    # draft table emitter
ma-irt/scripts/generate_tables.py                      # superseded by _aggregate_pykt_results.py
ma-irt/scripts/investigate_wol_threshold.py            # diagnostic
ma-irt/scripts/monitor.py, monitor_retrain.sh          # monitoring utilities, not on reproduction path
ma-irt/scripts/plot_*.py                               # paper figures, see Section 4.2 caveat
ma-irt/scripts/rerun_*.sh, resweep_eval_all.sh         # one-off re-evals
ma-irt/scripts/retrain_baselines.sh                    # one-off
ma-irt/scripts/run_after_*.sh                          # one-off chain runners
ma-irt/scripts/run_assist2009.sh, run_assist2009_ord.sh # superseded by _run_pykt_sweep.sh
ma-irt/scripts/run_bench_phase*.sh, run_bench_*chain.sh # superseded
ma-irt/scripts/run_bench_sweep.sh                      # superseded
ma-irt/scripts/run_bulk_retrain.sh                     # one-off
ma-irt/scripts/run_imbalance_extension.sh              # imbalance section
ma-irt/scripts/run_learned_sweep.sh                    # learned-emb sweep, not headline
ma-irt/scripts/run_remaining.sh                        # one-off
ma-irt/scripts/train_ablations.sh, train_alpha1.sh, train_learned_repr.sh  # ablation variants
```

**Caveat on plot scripts**, the paper figures in `overleaf-sync/figures/*.pdf` are produced by some of `scripts/plot_*.py`. Those scripts are not required to reproduce the three numerical tables in `benchmarks.md`, but they are required to regenerate the paper figures. If a regression triggers a figure regen, keep them. Otherwise they are post-hoc.

### 4.2 Configs that look in-use but are not on the reproduction path

These are not in the path to the three headline tables.

```
ma-irt/configs/bulk/*_chunked_s42.yaml          # chunked30 with single seed, draft only
ma-irt/configs/bulk/*_chunked30_s{0,1,7,42,123}.yaml   # superseded by pykt_fold variants
ma-irt/configs/bulk/*_s{0,1,7,42,123}.yaml      # seeded variants of pykt CV configs (older protocol). Older _aggregate_bench.py uses some of these for the Synthetic-Static K=2 row, but the published Table 2 uses the pykt_fold variants per _aggregate_pykt_results.py (the active aggregator).
ma-irt/configs/bulk/assist2009_*_s*.yaml        # ASSIST runs at K=4 ordinal, used for ASSIST tables in the paper (tab:assistments_pred) but NOT for the K=2 binary Table 2 cells. Keep if the paper figures still cite them; remove if cleaning to bench tables only.
ma-irt/configs/bulk/assistments_*_s*.yaml       # ASSIST K=4 ordinal, same caveat
ma-irt/configs/bulk/continuous_*_s*.yaml        # continuous DGP, used for tab:rw_* not for benchmarks.md
ma-irt/configs/bulk/discrete_*_s*.yaml          # discrete DGP, used for tab:block_* not for benchmarks.md
ma-irt/configs/bulk/imbalance_*_s*.yaml         # imbalance section
ma-irt/configs/bulk/imbalanced_*.yaml           # imbalance section
ma-irt/configs/bulk/scalability_*.yaml          # scaling section
ma-irt/configs/bulk/scaling_*.yaml              # scaling section
ma-irt/configs/bulk/static_*_a1_*.yaml          # alpha=1 ablation
ma-irt/configs/bulk/static_*_noprior*.yaml      # prior ablation
ma-irt/configs/bulk/static_*_wd*.yaml           # weight-decay ablation
ma-irt/configs/bulk/static_*_learned_s*.yaml    # seeded learned-emb variants (older protocol). The Table 2 cells use pykt_fold variants now; the seeded variants are kept for backward compatibility but are not on the active reproduction path.
ma-irt/configs/bulk/static_*_s{0,1,7,42,123}.yaml  # seeded variants (older protocol)
ma-irt/configs/block_*.yaml, rw_*.yaml, staircase_*.yaml  # root-level dynamic DGP configs, used for tab:block_*/tab:rw_* not for benchmarks.md
ma-irt/configs/experiments/                     # legacy RQ folders, predate the bulk/ directory
ma-irt/configs/_archive_s0p5/                   # archive
ma-irt/configs/tmp_alpha1/                      # temp
ma-irt/configs/dynamic_seeds/                   # legacy dynamic seed configs
```

**Important warning**, the older seeded variant configs (`static_magpcm_q200_k4_s42` and similar) are NOT what produced the headline tables. The active path is the pykt_fold variants. Removing the seeded variants does not affect benchmarks.md but may break some plot scripts that hardcode the seeded naming.

### 4.3 Source-file dead code from the existing cleanup plan

`ma-irt/CLEANUP.md` and `ma-irt/CLEANUP_PLAN.md` list dead code paths that can be removed without breaking benchmarks (subject to checkpoint compatibility caveats),

- `FocalLoss` class in `training/losses.py` (`focal_weight` is 0 in every active config)
- `monotonic_betas` branch in `models/components/irt.py` (deprecated path, but removing it breaks any old checkpoint with `threshold_base.weight` keys)
- `response_dim`, `use_separable_embed` fields in `config/types.py` (never read)
- `memory_add_activation` field, DKVMN hardcodes tanh
- Regularization penalty methods in `training/trainer.py` (always off in every config)

Caveat, the `monotonic_betas` removal is destructive to old checkpoints. If existing `outputs/bench_*/best.pt` files were trained with `monotonic_betas=True` and the cleanup removes the branch, loading those checkpoints will fail. Cross-check the relevant `outputs/<run>/best.pt` state_dict keys against the post-cleanup `IRTParameterExtractor` before declaring the cleanup safe.

### 4.4 Documentation and planning files

```
ma-irt/CLEANUP.md            # historical plan
ma-irt/CLEANUP_PLAN.md       # historical plan
ma-irt/NOTES_linking_appendix.md
ma-irt/PLAN_sigma05_bulk_retrain.md
ma-irt/RETRAIN_PLAN.md
ma-irt/REVIEW_*.md
ma-irt/TODO_alternating_optim.md
```

These are notes, not source. The cleanup planner can archive or remove them without breaking reproduction.

---

## 5. Verification record format

After each cleanup tier is applied, the planner must append one row to a `cleanup_log.md` file at the repo root, using the schema below. The schema is fixed, not free-form, so the records are diff-able and machine-parseable.

### 5.1 Per-tier record

```markdown
| Date | Tier | Before git ref | After git ref | Smoke pytest | Smoke trainings | Reg headline deltas | Reg recovery deltas | Full reg (if run) | Pass/fail | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
```

Column definitions,

- **Date**, ISO 8601 (YYYY-MM-DD). Always absolute.
- **Tier**, one of `A_dead_code`, `B_renames`, `C_restructure`, `D_datasets`, or a free-form short tag.
- **Before git ref**, the commit SHA (short, 7 chars) immediately before the tier was applied.
- **After git ref**, the commit SHA after the tier landed.
- **Smoke pytest**, `pass` or `fail (N tests failed)`.
- **Smoke trainings**, `K/7 pass` where K is the number of the seven smoke configs that produced a valid `best.pt`.
- **Reg headline deltas**, the max absolute delta across the 15 headline metrics in Section 2.2, written as `max=<delta> on <row_name>`.
- **Reg recovery deltas**, the max absolute delta across the 12 recovery metrics in Section 2.2.
- **Full reg (if run)**, either `skipped` or `max=<delta> on <cell>`.
- **Pass/fail**, `pass` if both smoke and regression layers pass per their criteria, `fail` otherwise.
- **Notes**, brief free text. Must mention any acceptance interval widened from the defaults in Section 2.

### 5.2 Example row (illustrative)

```markdown
| 2026-06-10 | A_dead_code | 40f26f1 | a1b2c3d | pass | 7/7 pass | max=0.003 on MA-GPCM K=4 QWK | max=0.008 on DKVMN+GPCM r_alpha | skipped | pass | FocalLoss removal, no checkpoint impact |
| 2026-06-12 | B_renames | a1b2c3d | e4f5678 | fail (1 tests failed) | 7/7 pass | not run | not run | not run | fail | test_shapes::test_separable_embedding failed, fix pending |
```

### 5.3 Record placement

Append-only file at `C:\Users\steph\documents\deep-mirt\cleanup_log.md`. New rows go at the bottom. The header row stays at the top. Each tier produces exactly one row, even if the tier required iteration to land.

### 5.4 Per-cell metric dump (optional but recommended)

When a regression run is executed, also dump the raw metrics to `C:\Users\steph\documents\deep-mirt\cleanup_metrics_<tier>.json` with the schema,

```json
{
  "tier": "A_dead_code",
  "date": "2026-06-10",
  "git_ref": "a1b2c3d",
  "regression": {
    "bench_magpcm_static_q200_k4_pykt_fold0": {
      "qwk": 0.6810,
      "acc": 0.5284,
      "r_alpha": 0.892,
      "r_beta": 0.965,
      "r_theta": 0.957
    }
  }
}
```

This lets future cleanup tiers compare against the previous baseline directly, without rerunning the full regression.

---

## 6. Final pre-flight checklist for the cleanup planner

Before starting any tier, verify,

1. Git working tree is clean, or all uncommitted state is documented.
2. `ma-irt/data/` contains all 12 dataset directories listed in Section 3.4.
3. `cd ma-irt && PYTHONPATH=. pytest tests/ -v` passes on the current commit. If not, the baseline is broken and the cleanup is premature.
4. At least one cached `outputs/bench_*_pykt_fold0/test_metrics.json` exists for each Section 1 table row, so the Layer 2 baseline is verifiable.
5. The conda `research` environment is active and `KMP_DUPLICATE_LIB_OK=TRUE` is set.

If any of those five fails, fix the precondition before starting the cleanup.
