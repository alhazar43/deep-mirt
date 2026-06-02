# MA-GPCM Codebase Cleanup and Unified Pipeline Plan

**Date**: 2026-03-29
**Status**: PLAN (no files modified)
**Scope**: Full audit of `kt-gpcm/` source, scripts, configs, tests, and data flow.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Source Code Audit](#2-source-code-audit)
3. [Scripts Audit](#3-scripts-audit)
4. [Configs Audit](#4-configs-audit)
5. [Model Architecture Audit](#5-model-architecture-audit)
6. [Test Audit](#6-test-audit)
7. [Data Flow Audit](#7-data-flow-audit)
8. [Files to Remove or Archive](#8-files-to-remove-or-archive)
9. [Code to Remove from Kept Files](#9-code-to-remove-from-kept-files)
10. [Unified Pipeline Design](#10-unified-pipeline-design)
11. [Config Simplification](#11-config-simplification)
12. [Test Plan](#12-test-plan)
13. [Risk Assessment](#13-risk-assessment)
14. [Execution Order](#14-execution-order)

---

## 1. Executive Summary

The codebase has 22 source files (library), 42 Python scripts, 5 R scripts, 10 shell scripts, 779 YAML configs, and 4 test files. Accumulated over many development iterations, it contains:

- **Dead code paths**: `monotonic_betas=True` branch (no longer used), `use_separable_embed` legacy flag, `response_dim` config field, `dkvmn_ordinal` model type (falls through to DeepGPCM)
- **Redundant loss code**: FocalLoss is still in the library and the default config (0.5 weight), but the paper uses WOL only (`focal_weight: 0.0`)
- **Duplicated scripts**: `compute_all_recovery.py` vs `compute_all_recovery_v3.py`, multiple plot scripts for the same figure type, 5+ data gen scripts with duplicated GPCM probability code
- **Config explosion**: 779 YAML files across 7 directories, many from superseded experiments (large_5000_v2 through v8c, ordinal_*, softmax_*, deepgpcm_*), plus 80 `dkvmn_ordinal` configs in baselines/ that use a nonexistent model type
- **Inconsistent loss defaults**: TrainingConfig defaults still set `focal_weight: 0.5` even though the paper switched to WOL-only long ago. Newer configs (block/rw/staircase) correctly set `focal_weight: 0.0` but generated/ and baselines/ configs are stale.

### Key Numbers

| Category | Count | Notes |
|----------|-------|-------|
| src/ Python files | 22 | Clean; main bloat is dead code paths, not dead files |
| scripts/ Python | 42 | ~15 are core, ~27 are diagnostic/one-off/superseded |
| scripts/ R | 5 | 2 core (mirt_baseline_all_k.R, mirt_predict.R), 3 debug |
| scripts/ shell | 10 | All one-off orchestration; superseded by unified pipeline |
| configs/ total | 779 | ~94 root-level, 260 generated/, 80 baselines/, 185 experiments/, 160 dynamic_seeds/ |
| tests/ | 4 | Good unit coverage for DeepGPCM; no coverage for baselines, trainer, data loading |

---

## 2. Source Code Audit

### 2.1 config/types.py (ModelConfig)

**Dead fields to remove**:
- `response_dim: int = 16` -- never read by any model. The separable embedding computes its own dimensions from `key_dim + n_categories`. Only passed through kwargs and ignored.
- `use_separable_embed: bool = False` -- legacy toggle. The `embedding_type` field fully supersedes it. All configs use `embedding_type` explicitly.

**Fields whose defaults are wrong**:
- `focal_weight: float = 0.5` -- paper uses WOL only. Should be `0.0`.
- `weighted_ordinal_weight: float = 0.5` -- paper uses `1.0`.
- `monotonic_betas` default was already changed to `False` (correct).
- `embedding_type` default was already changed to `"static_item"` (correct).

**Fields to keep but document better**:
- `model_type` -- valid values are `"deepgpcm"`, `"static_gpcm"`, `"dynamic_gpcm"`, `"dkvmn_softmax"`. The value `"dkvmn_ordinal"` exists in baselines/ configs but falls through to DeepGPCM in `build_model()`, which is a silent bug. Should be removed from all configs.

### 2.2 config/loader.py

Clean. The `_merge()` function silently ignores unknown keys, which is correct for forward compatibility but means stale config keys (like `response_dim`, `use_separable_embed`) will never raise an error. Consider adding a warning for unrecognized keys.

### 2.3 models/kt_gpcm.py (DeepGPCM)

**Dead code to remove**:
- `use_separable_embed` parameter and all conditional branches checking it (lines 77, 98, 108, 172, 216). The `embedding_type == "separable"` check is sufficient.
- `response_dim` parameter (line 76). Never used inside the class.
- The entire `separable` embedding branch (lines 108-115, 216-221) can be debated. It IS still a valid embedding_type for RQ4 experiments. Keep it, but remove the `use_separable_embed` fallback conditions.

**`monotonic_betas` parameter**: Passed through to `IRTParameterExtractor`. The constructor default is `True` (line 80) but `ModelConfig` default is `False`. These are mismatched. The config value always wins because `build_model()` passes all model kwargs. Still, the constructor default should match the config default for clarity.

**`separate_theta` parameter**: Active and needed. Controls theta source (ability_summary vs summary). Keep.

### 2.4 models/components/irt.py (IRTParameterExtractor)

**Dead code to remove (monotonic_betas=True path)**:
- `threshold_base` and `threshold_gaps` layers (lines 110-113)
- The entire `if self.monotonic_betas:` branch in `_init_weights()` (lines 145-151)
- The entire `if self.monotonic_betas:` branch in `forward()` (lines 197-207)
- The `monotonic_betas` parameter and attribute

Once removed, only the unconstrained path remains:
- Single `threshold_unconstrained = nn.Linear(question_dim, n_categories - 1)` layer
- Initialization with evenly-spaced bias

**Impact**: This is the highest-risk removal. All existing checkpoints were trained with specific `monotonic_betas` settings. Checkpoints trained with `monotonic_betas=True` will fail to load if the layer names change. The plan must include a checkpoint migration strategy or accept that old checkpoints become incompatible.

### 2.5 models/components/embeddings.py

Clean. Both `LinearDecayEmbedding` and `StaticItemEmbedding` are actively used. No dead code.

### 2.6 models/components/memory.py (DKVMN)

Clean. No dead code. Well-documented.

### 2.7 models/heads/gpcm.py (GPCMHead)

Clean. Trivial softmax wrapper. Keep as-is.

### 2.8 models/static_gpcm.py (StaticGPCM)

**`monotonic_betas` issue**: Uses hardcoded monotonic gap parameterization (always). Unlike DeepGPCM/IRTParameterExtractor, there is no `monotonic_betas` flag here. The beta construction is always gap-based:
```python
beta_0 = self.beta_base[questions]
gaps = F.softplus(self.beta_gaps[questions])
```
This means StaticGPCM always uses monotonic betas regardless of what the config says. This is a discrepancy. The cleanup should either:
1. Add an unconstrained path to match DeepGPCM, or
2. Document that StaticGPCM always uses monotonic betas (which is the standard psychometric practice for GPCM with SGD estimation).

**Recommendation**: Option 2. Static GPCM is the traditional IRT baseline, and monotonic thresholds are standard in that context. Add a comment documenting this design choice.

### 2.9 models/dynamic_gpcm.py (DynamicGPCM)

Same `monotonic_betas` issue as StaticGPCM: hardcoded gap-based beta. Same recommendation: document rather than change.

### 2.10 models/dkvmn_softmax.py (DKVMNSoftmax)

**Dead code to remove**:
- `use_separable_embed` parameter (line 44) and its fallback conditional (line 60)
- `response_dim` parameter (line 43)
- `monotonic_betas` parameter (line 45)
- `model_type` parameter (line 46)
- `separate_theta` parameter (line 47)

These are all accepted-and-ignored for config compatibility. After cleanup, use `**kwargs` to swallow them instead, with a comment explaining why.

### 2.11 training/losses.py

**FocalLoss**: Still in the library and exported from `__init__.py`. The paper uses WOL only (`focal_weight: 0.0`). However, `CombinedLoss` uses it when `focal_weight > 0`, and some old configs still set `focal_weight: 0.5`. Two approaches:

1. **Conservative**: Keep FocalLoss in the library, change the TrainingConfig default to `focal_weight: 0.0`. This preserves backward compatibility with old configs.
2. **Aggressive**: Remove FocalLoss entirely, simplify CombinedLoss to only WeightedOrdinalLoss.

**Recommendation**: Option 1. FocalLoss is small, well-tested, and may be useful for future experiments. The real fix is changing the config default.

**QWKLoss**: Only used when `qwk_weight > 0`, which is never the case in any config. It is a valid component to keep for potential future use, but it is never exercised in practice.

### 2.12 training/trainer.py

Clean. Handles both DeepGPCM and baseline model types via `_forward()` dispatch. The `_model_type` attribute check is a bit fragile but functional.

### 2.13 data/loaders.py

Clean. `SequenceDataset` and `DataModule` are well-structured. The `all_train_targets()` method is only used for class weight computation.

### 2.14 utils/metrics.py

Clean. All metrics are used in evaluation.

### 2.15 models/__init__.py

Only exports `DeepGPCM`. Should also export `StaticGPCM`, `DynamicGPCM`, `DKVMNSoftmax` since they are all used by `train.py` via `build_model()`.

---

## 3. Scripts Audit

### 3.1 Core Pipeline Scripts (KEEP)

| Script | Role | Notes |
|--------|------|-------|
| `train.py` | Training entry point for all models | Clean. Has `build_model()` for all 4 model types. |
| `data_gen.py` | Static theta DGP | Core. Used for Ordinal-Static datasets. |
| `data_gen_block.py` | Block-change DGP | Core. Used for Ordinal-Block datasets. |
| `data_gen_randomwalk.py` | Random walk DGP | Core. Used for Ordinal-RW datasets. |
| `data_gen_staircase.py` | Staircase DGP | Core. Used for Ordinal-Staircase datasets. |
| `data_gen_imbalanced.py` | Imbalanced class distribution DGP | Core. Used for RQ5. |
| `compute_all_recovery.py` | Recovery computation (original) | Core. Comprehensive, includes theta. |
| `eval_block_and_rw.py` | Dynamic DGP evaluation | Core. Staircase + block + RW eval. |
| `plot_recovery_split.py` | Recovery scatter plots (student + item) | Core. Paper figure script. |
| `plot_metrics.py` | Training curve plots | Core. Diagnostic utility. |
| `plot_trajectory_comparison.py` | 3-model trajectory figures | Core. Paper figure script. |
| `generate_tables.py` | LaTeX table generation | Core. Paper table script. |
| `mirt_baseline_all_k.R` | GPCM(EM) baseline via mirt package | Core. R baseline. |
| `mirt_predict.R` | GPCM(EM) theta prediction | Core. R baseline. |

### 3.2 Scripts to Keep (Secondary)

| Script | Role | Notes |
|--------|------|-------|
| `eval_metrics.py` | Quick checkpoint evaluation | Useful diagnostic. |
| `train_and_eval_dynamic.py` | Multi-model training on block/RW | Useful for bulk train. |
| `compare_architectures.py` | Sep-theta vs shared comparison | Paper figure (if renamed). |
| `plot_block_and_rw.py` | Block/RW trajectory figures | Paper figure. |
| `plot_dynamic_trajectories.py` | Dynamic trajectory comparison | Paper figure. |
| `merge_all_metrics.py` | Metrics + recovery merge | Table utility. |
| `train_binary_baselines.py` | DKT/DKVMN binary baselines | Used for RQ1 K=2 baseline. |
| `train_benchmark.py` | ASSIST2015/Synthetic-5 benchmarks | Used for external validation. |
| `prepare_benchmark_data.py` | Benchmark data preparation | Data prep utility. |
| `estimate_theta_eap.py` | EAP theta estimation | Used for Static GPCM theta recovery. |
| `gen_dynamic_seed_configs.py` | Config generator for multi-seed | Useful automation. |

### 3.3 Scripts to ARCHIVE (one-off/diagnostic/superseded)

| Script | Reason |
|--------|--------|
| `compute_all_recovery_v3.py` | Superseded by `compute_all_recovery.py` which has been updated. The v3 was a parallel development that duplicates linking functions and adds tasks that should be in the main script. |
| `data_gen_dynamic.py` | LINEAR growth DGP. Superseded by block-change (discrete) and random-walk (continuous) DGPs. Not used in the final paper. |
| `eval_dynamic_recovery.py` | Superseded by `eval_block_and_rw.py` which handles all dynamic DGPs. |
| `eval_dynamic_seeds.py` | Multi-seed eval, can be folded into unified pipeline. |
| `infer_old_arch_dynamic.py` | One-off diagnostic for old architecture inference. |
| `plot_dynamic_comparison.py` | One-off diagnostic, superseded by `plot_trajectory_comparison.py`. |
| `plot_dynamic_old_vs_new.py` | One-off diagnostic, superseded by `plot_trajectory_comparison.py`. |
| `plot_recovery.py` | Original recovery plots, superseded by `plot_recovery_split.py`. |
| `plot_recovery_figure.py` | Combined recovery figure, superseded by `plot_recovery_split.py`. |
| `plot_rq2_trajectories_pdf.py` | One-off PDF version of trajectory plots. |
| `plot_rq3_recovery.py` | One-off RQ3 recovery scatter. |
| `plot_learner_trajectories.py` | Superseded by `plot_trajectory_comparison.py` and `plot_dynamic_trajectories.py`. |
| `plot_theta_temporal.py` | One-off temporal theta convergence plot. |
| `quick_alpha_recovery.py` | One-off diagnostic. |
| `run_recovery_abs.py` | One-off wrapper for absolute paths. |
| `run_all_dynamic_k.py` | One-off pipeline for dynamic K sweep. Folded into unified pipeline. |
| `merge_metrics_recovery.py` | Superseded by `merge_all_metrics.py`. |
| `prepare_assistments.py` | One-off data prep for ASSISTments. |
| `mirt_baseline.R` | Superseded by `mirt_baseline_all_k.R`. |
| `mirt_check_k2.R` | Debug script. |
| `mirt_k6_fix.R` | Debug/fix script. |
| `bulk_train.sh` | Superseded by unified pipeline. |
| `bulk_run_comprehensive.sh` | Superseded by unified pipeline. |
| `run_baselines_comprehensive.sh` | Superseded by unified pipeline. |
| `regenerate_plots.sh` | Superseded by unified pipeline. |
| `regenerate_all_recovery_plots.sh` | Superseded by unified pipeline. |
| `run_all_experiments.sh` | Superseded by unified pipeline. |
| `run_recovery.sh` | Superseded by unified pipeline. |
| `progress.sh` | Diagnostic. |
| `train_all_dynamic.sh` | Superseded by unified pipeline. |
| `train_all_dynamic_seeds.sh` | Superseded by unified pipeline. |

### 3.4 Duplicated Code Across Scripts

Linking functions (`link_alpha`, `link_normal`, `link_theta_irt`) are copy-pasted in at least 4 scripts:
- `compute_all_recovery.py`
- `compute_all_recovery_v3.py`
- `eval_block_and_rw.py`
- `plot_recovery_split.py`

**Action**: Extract to `src/kt_gpcm/utils/linking.py`.

Batched inference logic is duplicated in:
- `compute_all_recovery.py`
- `eval_block_and_rw.py`
- `plot_trajectory_comparison.py`

**Action**: Extract to `src/kt_gpcm/utils/inference.py`.

The `build_model()` function in `train.py` is also duplicated in several eval scripts.

**Action**: Move `build_model()` into `src/kt_gpcm/models/__init__.py`.

---

## 4. Configs Audit

### 4.1 Directory Structure (779 total YAML files)

| Directory | Count | Purpose | Status |
|-----------|-------|---------|--------|
| `configs/` (root) | 94 | Mix of active and legacy | ~50 are legacy (large_5000_*, deepgpcm_*, softmax_*, ordinal_*, ablation_*, seqlen_*) |
| `configs/generated/` | 260 | RQ1/RQ4 multi-seed MA-GPCM configs | Active for static-theta experiments |
| `configs/baselines/` | 80 | Baseline model configs for static data | 20 are `dkvmn_ordinal` (broken model_type), rest active |
| `configs/experiments/` | 185 | Multi-seed experiment configs (rq1/rq4/rq5/ablation) | Active |
| `configs/dynamic_seeds/` | 160 | Multi-seed configs for dynamic DGPs | Active |

### 4.2 Legacy Configs to Archive

**Root-level configs to archive** (superseded by generated/ or experiments/):
- `large_5000_v2.yaml` through `large_5000_v8c.yaml` (7 files) -- iterative development
- `large_q500_v9a.yaml`, `large_q500_v9b.yaml` -- iterative
- `large_q1000_v9a.yaml`, `large_q1000_v9b.yaml` -- iterative
- `large_q5000_linear.yaml`, `large_q5000_separable.yaml`, `large_q5000_static.yaml` -- large-Q experiments
- `deepgpcm_k2_s42.yaml` through `deepgpcm_k5_s99.yaml` (5 files) -- superseded by experiments/rq1/
- `softmax_k2_s42.yaml` through `softmax_q5000_k5_s42.yaml` (5 files) -- superseded by experiments/rq1/
- `ordinal_k2_s42.yaml` through `ordinal_k5_s42.yaml` (4 files) -- unknown purpose, not referenced by any active script
- `static_gpcm_k2_s42.yaml` through `static_gpcm_k5_s42.yaml` (4 files) -- superseded by experiments/rq1/
- `dynamic_gpcm_k2_s42.yaml` through `dynamic_gpcm_k5_s42.yaml` (4 files) -- superseded by experiments/rq1/
- `ablation_focal_k4_s42.yaml`, `ablation_wol_k4_s42.yaml` -- focal vs WOL ablation, no longer in paper
- `ablation_nomonot_*.yaml` (4 files) -- superseded by experiments/ablation/
- `seqlen_*.yaml` (2 files) -- sequence-length ablation, not in paper
- `dynamic_q200_k4*.yaml` (5 files) -- early dynamic experiments, superseded by block/rw/staircase

**Total root-level to archive**: ~45 of 94

**Baselines configs to fix**: 20 `dkvmn_ordinal` configs have `model_type: "dkvmn_ordinal"` which falls through to DeepGPCM in `build_model()`. These should either be removed or changed to `model_type: "deepgpcm"`.

### 4.3 Config Field Inconsistencies

Across all active configs, these inconsistencies exist:

| Field | Old configs | New configs | Target default |
|-------|------------|-------------|---------------|
| `focal_weight` | 0.5 | 0.0 | 0.0 |
| `weighted_ordinal_weight` | 0.5 | 1.0 | 1.0 |
| `monotonic_betas` | true | false | false (but needs retrain) |
| `init_value_memory` | true (old) / false (new) | false | false |
| `value_dim` | 128 (old) / 64 (new) | 64 | 64 |

The bulk retrain will standardize all configs. Until then, existing checkpoints are tied to their specific config values.

---

## 5. Model Architecture Audit

### 5.1 MA-GPCM (DeepGPCM, separated theta)

**Config**: `model_type: "deepgpcm"`, `separate_theta: true`, `embedding_type: "static_item"`, `monotonic_betas: false`
**Code path**: `kt_gpcm.models.kt_gpcm.DeepGPCM`
**Forward signature**: `forward(questions: Tensor, responses: Tensor) -> dict`
**Key fields used**: `n_questions`, `n_categories`, `n_traits`, `memory_size`, `key_dim`, `value_dim`, `summary_dim`, `ability_scale`, `dropout_rate`, `memory_add_activation`, `init_value_memory`, `embedding_type`, `item_embed_dim`, `monotonic_betas`, `separate_theta`
**Fields ignored**: `response_dim` (separable-only), `use_separable_embed` (legacy)
**Theta source**: `ability_summary(read_t)` -- pure student state, no item contamination.

### 5.2 DKVMN+GPCM (DeepGPCM, shared theta)

**Config**: `model_type: "deepgpcm"`, `separate_theta: false`, `embedding_type: "static_item"`, `monotonic_betas: false`
**Code path**: Same `DeepGPCM` class, `separate_theta=False` branch
**Forward signature**: Same
**Theta source**: `summary([read_t, q_t])` -- includes item identity.
**Note**: Only difference from MA-GPCM is theta source. All IRT parameter extraction is identical.

### 5.3 DKVMN+Softmax (DKVMNSoftmax)

**Config**: `model_type: "dkvmn_softmax"`
**Code path**: `kt_gpcm.models.dkvmn_softmax.DKVMNSoftmax`
**Forward signature**: `forward(questions: Tensor, responses: Tensor) -> dict`
**Key fields used**: `n_questions`, `n_categories`, `memory_size`, `key_dim`, `value_dim`, `summary_dim`, `dropout_rate`, `memory_add_activation`, `init_value_memory`, `embedding_type`, `item_embed_dim`
**Fields ignored**: `n_traits`, `ability_scale`, `response_dim`, `use_separable_embed`, `monotonic_betas`, `model_type`, `separate_theta`
**Returns**: `theta/alpha/beta` are dummy zeros/ones for API compatibility. `logits` and `probs` are real.

### 5.4 GPCM(SGD) (StaticGPCM)

**Config**: `model_type: "static_gpcm"`
**Code path**: `kt_gpcm.models.static_gpcm.StaticGPCM`
**Forward signature**: `forward(student_ids: Tensor, questions: Tensor, responses: Tensor) -> dict`
**Key fields used**: `n_questions`, `n_categories`, `n_traits`, `ability_scale`
**Extra init arg**: `n_students` (passed from DataModule)
**Fields ignored**: All DKVMN-related fields, via `**kwargs`
**Note**: Forward signature differs from DeepGPCM (takes `student_ids`). Trainer handles this via `_forward()` dispatch with `_model_type` attribute.
**Beta**: Always monotonic (gap-based). No `monotonic_betas` flag.

### 5.5 Dynamic GPCM (DynamicGPCM)

**Config**: `model_type: "dynamic_gpcm"`
**Code path**: `kt_gpcm.models.dynamic_gpcm.DynamicGPCM`
**Forward signature**: `forward(student_ids: Tensor, questions: Tensor, responses: Tensor) -> dict`
**Key fields used**: `n_questions`, `n_categories`, `n_traits`, `ability_scale`
**Extra init arg**: `n_students` (passed from DataModule)
**Fields ignored**: All DKVMN-related fields, via `**kwargs`
**Note**: Same forward signature issue as StaticGPCM.
**Beta**: Always monotonic. No flag.

### 5.6 GPCM(EM) (External R, mirt package)

**Script**: `scripts/mirt_baseline_all_k.R` and `scripts/mirt_predict.R`
**Not a Python model**. Reads `sequences.json` and `true_irt_parameters.json`, runs EM via the R `mirt` package, writes recovery CSV and `estimated_parameters.rds`.
**Theta recovery**: Uses MLE with fixed item parameters for test students.

---

## 6. Test Audit

### 6.1 Current Coverage

| Test file | What it tests | Coverage |
|-----------|--------------|----------|
| `test_shapes.py` | DeepGPCM forward shapes (D=1, D=3, K=2) | Good for DeepGPCM only |
| `test_config_loader.py` | Config loading, validation, defaults | Good |
| `test_losses.py` | FocalLoss, QWKLoss, WeightedOrdinalLoss, CombinedLoss | Good |
| `test_heads.py` | GPCMLogits, GPCMHead | Good |

### 6.2 Coverage Gaps

**Not tested**:
- `StaticGPCM` forward pass (shapes, output keys)
- `DynamicGPCM` forward pass (shapes, output keys, theta update)
- `DKVMNSoftmax` forward pass (shapes, output keys, dummy IRT fields)
- `Trainer` (train_epoch, evaluate_epoch, _forward dispatch)
- `DataModule` / `SequenceDataset` / `collate_sequences`
- `DKVMN` (attention, read, write separately)
- `LinearDecayEmbedding` (shapes, output values)
- `StaticItemEmbedding` (shapes, output values, frozen embedding)
- `IRTParameterExtractor` (separate from GPCMLogits)
- `compute_metrics` in isolation
- Gradient flow through full model
- Checkpoint save/load round-trip
- The `separate_theta` True/False code paths produce different theta values

### 6.3 Fragile Points

- The `_model_type` attribute-based dispatch in `trainer.py._forward()` is fragile. If someone forgets to set `model._model_type` in `build_model()`, the trainer silently calls the wrong forward signature.
- The monotonic_betas True/False branches in `IRTParameterExtractor` produce different parameter names in `state_dict`, so checkpoint loading fails silently with mismatched keys when `strict=False`.

---

## 7. Data Flow Audit

### 7.1 Data Generation Scripts and Output Format

All data generators output the same 3-file format:

```
data/<dataset_name>/
    sequences.json            -- [{questions: [1-based IDs], responses: [0-based]}]
    metadata.json             -- {n_students, n_questions, n_categories, theta_type, ...}
    true_irt_parameters.json  -- {theta, alpha, beta, ...DGP-specific fields}
```

**DGP-specific fields in `true_irt_parameters.json`**:

| DGP | Theta field | Extra fields |
|-----|------------|--------------|
| Static (`data_gen.py`) | `theta` (N,) | -- |
| Block (`data_gen_block.py`) | `theta_0` (N,), `theta_trajectories` (N, 60) | `delta` (N,) |
| Staircase (`data_gen_staircase.py`) | `theta_0` (N,), `theta_trajectories` (N, 60) | `delta_1` (N,), `delta_2` (N,) |
| Random Walk (`data_gen_randomwalk.py`) | `theta_0` (N,), `theta_trajectories` (N, var_len) | `mu_drift` (N,), `sigma_innov` |
| Dynamic Linear (`data_gen_dynamic.py`) | `theta_0` (N,), `theta_trajectories` (N, var_len) | `gamma` (N,) |
| Imbalanced (`data_gen_imbalanced.py`) | `theta` (N,) | `target_dist`, `actual_dist` |

### 7.2 DataModule Load Path

1. `DataModule.__init__()`: Sets `dataset_dir = data_dir / dataset_name`
2. `DataModule.build()`:
   - Loads `sequences.json` -> list of {questions, responses} dicts
   - Loads `metadata.json` -> syncs `n_questions` and `n_categories` into `cfg.model`
   - Splits into train/test by `train_split` (sequential, not shuffled)
   - Creates `SequenceDataset` with 1-based student IDs (train: 1..n_train, test: n_train+1..n_total)
   - Returns `DataLoader` with `collate_sequences`

### 7.3 Collate Function

`collate_sequences()` returns `(questions, responses, mask, student_ids)` where:
- `questions`: (B, S_max) long, 0-padded
- `responses`: (B, S_max) long, 0-padded
- `mask`: (B, S_max) bool, True for valid positions
- `student_ids`: (B, S_max) long, same ID broadcast across all timesteps

### 7.4 Key Observation About Data Splits

The train/test split is **sequential** (first 80% = train, last 20% = test). This is fine for synthetic data (students are exchangeable) but would be problematic for real data. No shuffling before split.

---

## 8. Files to Remove or Archive

### 8.1 Move to `archive/scripts/`

All scripts listed in Section 3.3 (27 Python, 3 R, 10 shell scripts).

### 8.2 Move to `archive/configs/`

- All root-level legacy configs (~45 files listed in Section 4.2)
- All `dkvmn_ordinal` configs in baselines/ (20 files)
- All dynamic_q200_k4* configs (5 files)

### 8.3 Files Already Deleted (per git status)

- `scripts/mirt_check_param.R` -- already deleted
- `scripts/mirt_debug_theta.R` -- already deleted

---

## 9. Code to Remove from Kept Files

### 9.1 High Priority (Dead Code)

| File | What to remove | Lines |
|------|---------------|-------|
| `config/types.py` | `response_dim` field | 40 |
| `config/types.py` | `use_separable_embed` field | 41 |
| `config/types.py` | Change `focal_weight` default to `0.0` | 70 |
| `config/types.py` | Change `weighted_ordinal_weight` default to `1.0` | 71 |
| `models/kt_gpcm.py` | `response_dim` parameter | 76 |
| `models/kt_gpcm.py` | `use_separable_embed` parameter and all `or use_separable_embed` checks | 77, 98, 108, 172, 216 |
| `models/dkvmn_softmax.py` | `response_dim`, `use_separable_embed`, `monotonic_betas`, `model_type`, `separate_theta` parameters | 43-47 |
| `models/dkvmn_softmax.py` | `or use_separable_embed` check | 60 |

### 9.2 Medium Priority (monotonic_betas removal)

| File | What to remove | Impact |
|------|---------------|--------|
| `config/types.py` | `monotonic_betas` field entirely | All configs explicitly set it; removing from defaults OK |
| `models/kt_gpcm.py` | `monotonic_betas` parameter, attribute storage, passthrough to IRT | Constructor signature change |
| `models/components/irt.py` | Entire `if self.monotonic_betas:` branch in `__init__` | Removes threshold_base, threshold_gaps layers |
| `models/components/irt.py` | Entire `if self.monotonic_betas:` branch in `_init_weights()` | |
| `models/components/irt.py` | Entire `if self.monotonic_betas:` branch in `forward()` | |
| `models/components/irt.py` | `monotonic_betas` parameter and attribute | |

**CHECKPOINT COMPATIBILITY WARNING**: Removing the monotonic_betas=True path means old checkpoints trained with `monotonic_betas=True` will fail to load because their state_dict contains `irt.threshold_base.weight`, `irt.threshold_base.bias`, `irt.threshold_gaps.weight`, `irt.threshold_gaps.bias` instead of `irt.threshold_unconstrained.weight`, `irt.threshold_unconstrained.bias`.

**Mitigation**: All remaining paper experiments will be retrained in the bulk retrain with `monotonic_betas: false`. Old checkpoints can be archived alongside their configs. Add a checkpoint migration utility if backward compatibility is needed.

### 9.3 Low Priority (Cleanup)

| File | What to change | Notes |
|------|---------------|-------|
| `models/kt_gpcm.py` | Align constructor default `monotonic_betas=True` to config default `False` | Cosmetic if config always overrides |
| `models/__init__.py` | Export all 4 model classes | Cleaner imports |
| `training/__init__.py` | Consider not exporting FocalLoss/QWKLoss if unused | Minor |
| `config/loader.py` | Add warning for unrecognized YAML keys | Quality of life |

---

## 10. Unified Pipeline Design

### 10.1 Shared Utilities (new files in src/)

**`src/kt_gpcm/utils/linking.py`**:
```python
link_alpha(vals, target_std=0.3) -> np.ndarray
link_normal(vals) -> np.ndarray
link_theta_irt(true_theta, est_theta, true_beta, est_beta) -> tuple
```

**`src/kt_gpcm/utils/inference.py`**:
```python
run_batched_inference(model, records, batch_size, device, model_type=None) -> dict
    # Returns: {est_theta: (N, S, D), est_alpha: (N, S, D), est_beta: (N, S, K-1), ...}
```

**`src/kt_gpcm/models/__init__.py`** (extended):
```python
def build_model(cfg, device, n_students=0) -> nn.Module:
    # Moved from train.py. Single source of truth for model construction.
```

### 10.2 Data Generation

**Single entry point**: `scripts/data_gen.py` with `--dgp` flag.

```bash
python scripts/data_gen.py --dgp static --name ordinal_static_q200_k4 \
    --n_students 5000 --n_questions 200 --n_cats 4 --output_dir data

python scripts/data_gen.py --dgp block --name ordinal_block_q200_k4 \
    --n_students 5000 --n_questions 200 --n_cats 4 --output_dir data

python scripts/data_gen.py --dgp staircase --name ordinal_staircase_q200_k4 \
    --n_students 5000 --n_questions 200 --n_cats 4 --output_dir data

python scripts/data_gen.py --dgp randomwalk --name ordinal_rw_q200_k4 \
    --n_students 5000 --n_questions 200 --n_cats 4 --output_dir data

python scripts/data_gen.py --dgp imbalanced --name ordinal_static_q200_k4_mild \
    --n_students 5000 --n_questions 200 --n_cats 4 \
    --target_dist 0.10 0.20 0.30 0.40 --output_dir data
```

**Implementation**: The unified script imports the specific generator class based on `--dgp`. DGP-specific arguments are passed as extra kwargs. Internally, each generator class stays separate (no monolithic class).

**Alternative** (lower risk): Keep the 5 separate scripts but refactor them to share GPCM probability code via a common base class or utility function.

### 10.3 Training

**Entry point**: `scripts/train.py` (already handles all 4 model types). No change needed beyond importing `build_model` from the library.

### 10.4 Evaluation

**Entry point**: `scripts/evaluate.py` (new unified script)

```bash
# Static-theta recovery (alpha, beta, theta)
python scripts/evaluate.py recovery --config configs/block_q200_k4.yaml \
    --checkpoint outputs/block_q200_k4/best.pt --output outputs/block_q200_k4/recovery.json

# Dynamic-theta evaluation (trajectories, deltas, RMSE)
python scripts/evaluate.py trajectory --config configs/rw_q200_k4.yaml \
    --checkpoint outputs/rw_q200_k4/best.pt --output outputs/rw_q200_k4/trajectory.json

# Prediction metrics only (ACC, QWK, MAE)
python scripts/evaluate.py predict --config configs/block_q200_k4.yaml \
    --checkpoint outputs/block_q200_k4/best.pt --output outputs/block_q200_k4/metrics.json

# Multi-seed aggregation
python scripts/evaluate.py aggregate --pattern "outputs/block_q200_k4_s*/recovery.json" \
    --output outputs/block_q200_k4_summary.csv
```

**Implementation**: Uses subcommands via argparse. Each subcommand calls the appropriate evaluation function from `src/kt_gpcm/utils/`. Linking functions, inference, and metrics are all library code.

### 10.5 Plotting

Consolidate into 3 scripts:

1. **`scripts/plot_recovery.py`**: Recovery scatter plots (static + dynamic items).
   Replaces: `plot_recovery.py`, `plot_recovery_figure.py`, `plot_recovery_split.py`, `plot_rq3_recovery.py`

2. **`scripts/plot_trajectories.py`**: Theta trajectory comparison figures.
   Replaces: `plot_trajectory_comparison.py`, `plot_dynamic_trajectories.py`, `plot_block_and_rw.py`, `plot_dynamic_comparison.py`, `plot_dynamic_old_vs_new.py`, `plot_learner_trajectories.py`, `plot_rq2_trajectories_pdf.py`, `plot_theta_temporal.py`

3. **`scripts/plot_metrics.py`**: Training curve plots (already exists, keep as-is).

### 10.6 Config Generation

**Entry point**: `scripts/gen_configs.py` (consolidates `gen_dynamic_seed_configs.py`)

```bash
python scripts/gen_configs.py --rq rq1 --seeds 0 1 7 42 123 --output configs/experiments/rq1/
python scripts/gen_configs.py --rq rq4 --seeds 0 1 7 42 123 --output configs/experiments/rq4/
python scripts/gen_configs.py --rq dynamic --model all --k 3 4 5 6 --dgp block rw staircase \
    --seeds 0 1 7 42 123 --output configs/dynamic_seeds/
```

### 10.7 End-to-End Pipeline

**Entry point**: `scripts/run_experiment.py`

```bash
# Run a complete RQ experiment
python scripts/run_experiment.py --rq rq1 --k 4 --seeds 0 1 7 42 123 \
    --steps data train eval plot

# Run only evaluation (data and checkpoints exist)
python scripts/run_experiment.py --rq rq1 --k 4 --seeds 0 1 7 42 123 \
    --steps eval plot
```

This orchestrates: data generation, config generation, training all models, evaluation, and figure generation.

---

## 11. Config Simplification

### 11.1 Fields to Keep

| Field | Purpose | Default |
|-------|---------|---------|
| `model.n_questions` | Item bank size | 200 |
| `model.n_categories` | Response categories | 4 |
| `model.n_traits` | Latent dimensions | 1 |
| `model.memory_size` | DKVMN slots | 50 |
| `model.key_dim` | Key/query dim | 64 |
| `model.value_dim` | Value dim | 64 |
| `model.summary_dim` | Summary hidden dim | 50 |
| `model.embedding_type` | Embedding strategy | "static_item" |
| `model.item_embed_dim` | SIE dimension (0=auto) | 0 |
| `model.ability_scale` | Theta scale | 1.0 |
| `model.dropout_rate` | Dropout | 0.0 |
| `model.memory_add_activation` | DKVMN add gate | "tanh" |
| `model.init_value_memory` | Learned init memory | false |
| `model.model_type` | Model selector | "deepgpcm" |
| `model.separate_theta` | Theta pathway | true |
| `training.epochs` | Epoch count | 30 |
| `training.batch_size` | Batch size | 64 |
| `training.lr` | Learning rate | 1e-3 |
| `training.grad_clip` | Gradient clip norm | 1.0 |
| `training.weighted_ordinal_weight` | WOL weight | 1.0 |
| `training.ordinal_penalty` | WOL internal penalty | 0.5 |
| `training.lr_patience` | LR scheduler patience | 5 |
| `training.lr_factor` | LR scheduler factor | 0.8 |
| All `data.*` fields | | |

### 11.2 Fields to Drop

| Field | Reason |
|-------|--------|
| `model.response_dim` | Never used |
| `model.use_separable_embed` | Legacy, superseded by embedding_type |
| `model.monotonic_betas` | Always false after cleanup; remove from config and code |
| `training.focal_weight` | Always 0 in paper configs; keep in code (CombinedLoss) but default to 0 |
| `training.attention_entropy_weight` | Always 0 in all configs |
| `training.theta_norm_weight` | Always 0 in all active configs |
| `training.alpha_prior_weight` | Always 0 in all active configs |
| `training.beta_prior_weight` | Always 0 in all active configs |

**Note on regularization weights**: These are all 0 in active configs but were used during development. They can be kept in the code and config schema for future experimentation, but removed from YAML files that explicitly set them to 0 (to reduce noise).

### 11.3 Post-Cleanup Config Template

```yaml
base:
  experiment_name: "block_q200_k4"
  device: "cuda"
  seed: 42

model:
  model_type: "deepgpcm"
  n_questions: 200
  n_categories: 4
  n_traits: 1
  memory_size: 50
  key_dim: 64
  value_dim: 64
  summary_dim: 50
  embedding_type: "static_item"
  init_value_memory: false
  separate_theta: true

training:
  epochs: 30
  batch_size: 64
  lr: 0.001
  grad_clip: 1.0
  weighted_ordinal_weight: 1.0
  ordinal_penalty: 0.5
  lr_patience: 5
  lr_factor: 0.8

data:
  data_dir: "data"
  dataset_name: "block_q200_k4"
  train_split: 0.8
  min_seq_len: 10
```

Note: fields not listed (ability_scale, dropout_rate, etc.) fall to their defaults. Config is ~30% shorter than the current verbose version.

---

## 12. Test Plan

### 12.1 Tests to Add

**Model shape tests** (extend `test_shapes.py`):
- `TestStaticGPCM`: shapes, theta from embedding, alpha positive, beta monotonic
- `TestDynamicGPCM`: shapes, theta updates over time, alpha positive
- `TestDKVMNSoftmax`: shapes, dummy IRT fields are zeros/ones
- `TestDeepGPCMSeparateTheta`: verify separate_theta=True produces different theta than False

**Integration tests** (new `test_integration.py`):
- `test_trainer_one_epoch`: smoke test train_epoch + evaluate_epoch for each model type
- `test_trainer_model_dispatch`: verify _forward() calls correct forward signature per model_type
- `test_checkpoint_save_load`: save, load, verify identical outputs

**Data tests** (new `test_data.py`):
- `test_collate_sequences_shapes`: verify padding, mask, student_ids
- `test_data_module_build`: verify train/test split with synthetic data
- `test_data_module_all_train_targets`: verify flat target tensor

**Component tests** (new `test_components.py`):
- `test_dkvmn_attention_sums_to_one`
- `test_dkvmn_read_write_shapes`
- `test_static_item_embedding_frozen`: verify item_embed is a buffer, not a parameter
- `test_linear_decay_embedding_shapes`
- `test_irt_extractor_shapes`: theta, alpha, beta dimensions
- `test_irt_extractor_alpha_positive`: exp mapping

**Gradient flow tests** (new `test_gradients.py`):
- `test_deepgpcm_gradient_flow`: loss.backward() produces non-zero gradients on all parameters
- `test_static_gpcm_gradient_flow`: same
- `test_dynamic_gpcm_gradient_flow`: same

### 12.2 Tests to Update

- `test_shapes.py`: After removing `monotonic_betas` parameter, update `make_model()` to not pass it. Test both K=2 and K>2 for the single unconstrained path.
- `test_losses.py`: Update CombinedLoss default test to expect `focal_weight=0.0`.
- `test_config_loader.py`: Update default value assertions for `focal_weight`, `weighted_ordinal_weight`. Remove assertions on removed fields.

---

## 13. Risk Assessment

### 13.1 High Risk: Checkpoint Incompatibility

**Risk**: Removing `monotonic_betas=True` path changes layer names in IRTParameterExtractor state_dict. Existing checkpoints trained with `monotonic_betas=True` will fail to load.

**Impact**: All generated/, baselines/, and experiments/ checkpoints from pre-bulk-retrain era.

**Mitigation**:
1. Archive all pre-cleanup checkpoints and their configs together.
2. Run bulk retrain with unified configs AFTER the cleanup.
3. Optionally provide a `scripts/migrate_checkpoint.py` that maps old layer names to new ones.

### 13.2 Medium Risk: Config Default Changes

**Risk**: Changing `focal_weight` default from 0.5 to 0.0 affects any config that omits this field.

**Impact**: `smoke.yaml` and any future configs that rely on defaults. Existing experiments all explicitly set this field.

**Mitigation**: Verify all active configs explicitly set `focal_weight`. The `smoke.yaml` does not set it, so it will switch from focal+WOL to WOL-only. This is the desired behavior.

### 13.3 Medium Risk: Removing dkvmn_ordinal Model Type

**Risk**: 20 baselines/ configs use `model_type: "dkvmn_ordinal"`. Currently this silently falls through to DeepGPCM in `build_model()`.

**Impact**: If anyone runs these configs, they get DeepGPCM instead of the intended model. This is already a bug.

**Mitigation**: Delete or fix these configs. If they were intended as "DeepGPCM with ordinal output" (which is just DeepGPCM), change model_type to "deepgpcm". If they were experiments that are no longer needed, archive them.

### 13.4 Low Risk: Script Consolidation

**Risk**: Archived scripts may be referenced by someone. Consolidated scripts may miss edge cases from the originals.

**Mitigation**: Create the archive directory before deleting. The unified scripts should be tested against the same outputs as the originals before switching.

### 13.5 Verification Checklist

After cleanup, verify no performance regression:

- [ ] `cd kt-gpcm && PYTHONPATH=src pytest tests/ -v` passes (all tests)
- [ ] `smoke.yaml` trains 2 epochs without error on CPU
- [ ] Each model type (deepgpcm, static_gpcm, dynamic_gpcm, dkvmn_softmax) trains 1 epoch without error
- [ ] Recovery computation on existing checkpoints produces same r_alpha, r_beta values
- [ ] At least one config from each experiment category loads and validates

---

## 14. Execution Order

### Phase 1: Foundation (no training, no checkpoint changes)

```
1a. Create archive/ directory structure
1b. Extract shared utilities to src/ (linking.py, inference.py)
1c. Move build_model() to models/__init__.py
1d. Add tests for all model types (shapes, gradients)
1e. Verify all tests pass
```

### Phase 2: Dead Code Removal (source only)

```
2a. Remove response_dim, use_separable_embed from types.py
2b. Remove use_separable_embed from kt_gpcm.py and dkvmn_softmax.py
2c. Change focal_weight default to 0.0, weighted_ordinal_weight to 1.0
2d. Update test assertions for new defaults
2e. Verify all tests pass
```

### Phase 3: Monotonic Betas Removal (BREAKING)

```
3a. Remove monotonic_betas from types.py, kt_gpcm.py, irt.py
3b. Rename threshold_unconstrained -> threshold (now the only path)
3c. Update all tests
3d. Archive all pre-cleanup checkpoints
3e. Verify all tests pass
3f. Verify smoke.yaml trains successfully
```

### Phase 4: Script and Config Cleanup

```
4a. Archive legacy scripts to archive/scripts/
4b. Archive legacy configs to archive/configs/
4c. Fix or archive dkvmn_ordinal configs
4d. Consolidate plotting scripts
4e. Create unified evaluate.py
```

### Phase 5: Bulk Retrain

```
5a. Generate unified configs for all experiments
5b. Generate data for any missing datasets
5c. Train all models (all seeds, all K, all DGPs)
5d. Run evaluation on all checkpoints
5e. Generate all paper figures and tables
5f. Verify paper results match or improve
```

### Phase 6: Documentation

```
6a. Update CLAUDE.md with new commands and file structure
6b. Update config documentation
6c. Clean up RETRAIN_PLAN.md and CODE_CHANGES_2026-03-29.md
```
