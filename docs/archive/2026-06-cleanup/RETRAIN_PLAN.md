# Bulk Retrain Plan

**Date**: 2026-03-29
**Status**: APPROVED DECISIONS
**Prerequisite**: CLEANUP.md Phase A (dead code), Phase B (renames), Phase C (restructure)

---

## Decisions

| # | Decision |
|---|----------|
| D1 | Discrete DGP uses 3-level staircase (data_gen_staircase.py) |
| D2 | Compute K=2 for dynamic but skip figures for now |
| D3 | Generate all figures for K=2..6, use K=4 in main body, rest in appendix |
| D4 | Scalability experiment: MA-GPCM only (3 embeddings) |
| D5 | Binary benchmarks (DKT/DKVMN): keep unchanged, reuse existing results |
| D6 | Same as D5, defer DKT implementation |
| Loss | WOL only (weighted_ordinal_weight=1.0, focal_weight=0.0). FocalLoss will be removed in cleanup |

---

## 1. Datasets

### 1.1 Static (Ordinal-Static)

| Name | K | Q | N | obs/item |
|------|---|---|---|----------|
| `static_q200_k2` | 2 | 200 | 5,000 | ~1,250 |
| `static_q200_k3` | 3 | 200 | 5,000 | ~1,250 |
| `static_q200_k4` | 4 | 200 | 5,000 | ~1,250 |
| `static_q200_k5` | 5 | 200 | 5,000 | ~1,250 |
| `static_q200_k6` | 6 | 200 | 5,000 | ~1,250 |
| `static_q500_k4` | 4 | 500 | 5,000 | ~500 |
| `static_q1000_k4` | 4 | 1,000 | 10,000 | ~500 |
| `static_q2000_k4` | 4 | 2,000 | 10,000 | ~250 |

Generator: `data_gen.py`
Parameters: `--min_seq 20 --max_seq 80`
**Total: 8 datasets**

### 1.2 Discrete (Ordinal-Staircase, 3-level)

| Name | K | Q | N |
|------|---|---|---|
| `discrete_q200_k2` | 2 | 200 | 5,000 |
| `discrete_q200_k3` | 3 | 200 | 5,000 |
| `discrete_q200_k4` | 4 | 200 | 5,000 |
| `discrete_q200_k5` | 5 | 200 | 5,000 |
| `discrete_q200_k6` | 6 | 200 | 5,000 |

Generator: `data_gen_staircase.py`
Parameters: `--seq_len 60 --block_size 20 --delta1_mean 0.5 --delta1_std 0.3 --delta2_mean 0.4 --delta2_std 0.3`
**Total: 5 datasets**

### 1.3 Continuous (Ordinal-RW)

| Name | K | Q | N |
|------|---|---|---|
| `continuous_q200_k2` | 2 | 200 | 5,000 |
| `continuous_q200_k3` | 3 | 200 | 5,000 |
| `continuous_q200_k4` | 4 | 200 | 5,000 |
| `continuous_q200_k5` | 5 | 200 | 5,000 |
| `continuous_q200_k6` | 6 | 200 | 5,000 |

Generator: `data_gen_randomwalk.py`
Parameters: `--min_seq 40 --max_seq 80 --drift_mean 0.02 --drift_std 0.01 --noise_std 0.1`
**Total: 5 datasets**

### 1.4 Imbalanced (Static with shifted ability prior)

| Name | K | Q | N | Ability prior | Approx distribution |
|------|---|---|---|---------------|---------------------|
| `imbalanced_q200_k4_mild` | 4 | 200 | 5,000 | N(0.5, 1) | 23/19/21/37 |
| `imbalanced_q200_k4_severe` | 4 | 200 | 5,000 | N(1.0, 1) | 15/16/22/48 |
| `imbalanced_q200_k4_extreme` | 4 | 200 | 5,000 | N(1.5, 1) | 9/12/20/58 |

Generator: `data_gen_imbalanced.py`
**Total: 3 datasets**

### 1.5 Binary Benchmarks (existing, no regeneration)

| Name | Source |
|------|--------|
| `assist2015` | External, already exists |
| `synthetic5` | External, already exists |

**Grand total: 22 datasets to generate + 2 existing = 24**

---

## 2. Models to Train

### 2.1 Model Specifications

| Paper name | Code class | model_type | Key config | Epochs |
|-----------|-----------|------------|------------|--------|
| **MA-GPCM** | MAGPCM (DeepGPCM) | `magpcm` | separate_theta=true, embedding=static_item | 30 |
| **DKVMN+Softmax** | DKVMNSoftmax | `dkvmn_softmax` | (no IRT params) | 30 |
| **DKVMN+GPCM** | MAGPCM (DeepGPCM) | `magpcm` | separate_theta=false | 30 |
| **GPCM (SGD)** | StaticGPCM | `static_gpcm` | static theta embed | 150 |
| **Dynamic GPCM** | DynamicGPCM | `dynamic_gpcm` | gated recurrence | 50 |
| **GPCM (EM)** | R mirt | N/A | EM calibration | N/A |

### 2.2 Training Matrix

#### Static DGP (Tables 1, 2 in paper)

| Model | K values | Q | Seeds | Runs |
|-------|----------|---|-------|------|
| MA-GPCM | 2,3,4,5,6 | 200 | 5 | 25 |
| DKVMN+Softmax | 2,3,4,5,6 | 200 | 5 | 25 |
| Dynamic GPCM | 2,3,4,5,6 | 200 | 5 | 25 |
| GPCM (SGD) | 2,3,4,5,6 | 200 | 5 | 25 |
| GPCM (EM) | 2,3,4,5,6 | 200 | 1 | 5 |
| **Subtotal** | | | | **105** |

#### Discrete DGP (dynamic tracking section)

| Model | K values | Seeds | Runs |
|-------|----------|-------|------|
| MA-GPCM | 3,4,5,6 | 5 | 20 |
| DKVMN+Softmax | 3,4,5,6 | 5 | 20 |
| Dynamic GPCM | 3,4,5,6 | 5 | 20 |
| GPCM (SGD) | 3,4,5,6 | 5 | 20 |
| DKVMN+GPCM | 4 | 1 | 1 |
| **Subtotal** | | | **81** |

(K=2 computed but not in tables. K=4 DKVMN+GPCM for trajectory figure only.)

#### Continuous DGP (dynamic tracking section)

| Model | K values | Seeds | Runs |
|-------|----------|-------|------|
| MA-GPCM | 3,4,5,6 | 5 | 20 |
| DKVMN+Softmax | 3,4,5,6 | 5 | 20 |
| Dynamic GPCM | 3,4,5,6 | 5 | 20 |
| GPCM (SGD) | 3,4,5,6 | 5 | 20 |
| DKVMN+GPCM | 4 | 1 | 1 |
| **Subtotal** | | | **81** |

#### Scalability (item representation comparison)

| Model | Embedding | Q values | Seeds | Runs |
|-------|-----------|----------|-------|------|
| MA-GPCM | onehot | 200,500,1000,2000 | 5 | 20 |
| MA-GPCM | learned | 200,500,1000,2000 | 5 | 20 |
| MA-GPCM | static_item | 200,500,1000,2000 | 5 | 20 |
| **Subtotal** | | | | **60** |

#### Imbalanced

| Model | Condition | Seeds | Runs |
|-------|-----------|-------|------|
| MA-GPCM | mild | 5 | 5 |
| MA-GPCM | severe | 5 | 5 |
| MA-GPCM | extreme | 5 | 5 |
| **Subtotal** | | | **15** |

#### Binary compatibility (reuse existing or retrain)

| Model | Dataset | Seeds | Runs |
|-------|---------|-------|------|
| MA-GPCM | static_q200_k2 | 5 | (counted above) |
| DKT | assist2015, synthetic5 | 5 | 10 (deferred) |
| DKVMN | assist2015, synthetic5 | 5 | 10 (deferred) |
| MA-GPCM | assist2015, synthetic5 | 5 | 10 |

**Grand total: ~357 training runs** (excluding deferred DKT/DKVMN)

---

## 3. Evaluation

All via unified `evaluate.py`:

| DGP | Metrics (all models) | Recovery (IRT models only) |
|-----|---------------------|---------------------------|
| Static | ACC, QWK, tau, MAE | r_alpha, r_beta, r_theta, RMSE |
| Discrete | ACC, QWK, tau, MAE | r_alpha, r_beta, r_block (3 levels), staircase_acc, traj_RMSE, median_traj_r |
| Continuous | ACC, QWK, tau, MAE | r_alpha, r_beta, r_endpoint, median_traj_r, traj_RMSE |
| Imbalanced | ACC, QWK, tau, MAE | r_alpha, r_beta, r_theta |
| Binary | ACC, AUC | (none) |

Seeds: {0, 1, 7, 42, 123}. Report mean +/- std.

---

## 4. Figures

### Main body (K=4)

| Figure | Script | Datasets | Models |
|--------|--------|----------|--------|
| Architecture diagram | TikZ (inline) | N/A | N/A |
| Temporal theta convergence | `plot_theta_temporal.py` | static_q200_k4 | MA-GPCM, Dynamic GPCM |
| Item parameter scatter | `plot_recovery_split.py` | static_q200_k4 | MA-GPCM, Dynamic GPCM, GPCM(SGD) |
| Discrete trajectory comparison | `plot_trajectory_comparison.py` | discrete_q200_k4 | MA-GPCM, DKVMN+GPCM, Dynamic GPCM |
| Continuous trajectory comparison | `plot_trajectory_comparison.py` | continuous_q200_k4 | MA-GPCM, DKVMN+GPCM, Dynamic GPCM |

### Appendix (K=2,3,5,6)

| Figure type | K values | Script |
|-------------|----------|--------|
| Temporal theta | 2,3,5,6 | `plot_theta_temporal.py` |
| Item scatter | 2,3,5,6 | `plot_recovery_split.py` |
| Learner trajectories | 2,3,5,6 | `plot_learner_trajectories.py` |

---

## 5. Tables

| Table | Content | Source data |
|-------|---------|------------|
| 1 | Ordinal prediction (static, K=3..6) | evaluate.py static |
| 2 | Parameter recovery (static, K=3..6) | evaluate.py static |
| 3 | Binary compatibility (K=2) | evaluate.py static (K=2) + binary benchmarks |
| 4 | Discrete prediction (K=3..6) | evaluate.py dynamic (discrete) |
| 5 | Discrete recovery (K=3..6) | evaluate.py dynamic (discrete) |
| 6 | Continuous prediction (K=3..6) | evaluate.py dynamic (continuous) |
| 7 | Continuous recovery (K=3..6) | evaluate.py dynamic (continuous) |
| 8 | Item representation (Q=200..2000) | evaluate.py static (scalability) |
| 9 | Imbalance robustness (K=4) | evaluate.py static (imbalanced) |

---

## 6. Execution Order

```
Phase 0: Codebase cleanup (CLEANUP.md Phase A + B + C)
  - Delete dead scripts, remove dead code
  - Rename classes (DeepGPCM -> MAGPCM, etc.)
  - Flatten directory structure (kt-gpcm/ -> ma-irt/)
  - Verify tests pass

Phase 1: Dataset generation (~10 min)
  - 8 static datasets
  - 5 discrete datasets
  - 5 continuous datasets
  - 4 imbalanced datasets

Phase 2: Config generation
  - gen_all_configs.py produces all ~350 configs

Phase 3: Static training (~6h GPU)
  - 100 runs: 4 models x 5K x 5 seeds
  - 5 R mirt calibrations

Phase 4: Dynamic training (~8h GPU)
  - 80 discrete + 80 continuous: 4 models x 4K x 5 seeds each
  - 2 DKVMN+GPCM single-seed for trajectory figures

Phase 5: Scalability + imbalanced (~6h GPU)
  - 60 scalability: 3 embeddings x 4Q x 5 seeds
  - 20 imbalanced: 4 conditions x 5 seeds

Phase 6: Evaluation (~30 min)
  - evaluate.py static on all static checkpoints
  - evaluate.py dynamic on all dynamic checkpoints

Phase 7: Figures (~10 min)
  - All main body + appendix figures

Phase 8: Tables
  - generate_tables.py reads eval CSVs, produces LaTeX
```

**Total GPU time: ~20h on RTX 4060**
**Wall clock (sequential): ~24h**
**Wall clock (if phases 3-5 parallelized with 2nd GPU): ~12h**

---

## 7. Naming Conventions (post-rename)

### Dataset directories
```
datasets/static_q{Q}_k{K}/
datasets/discrete_q{Q}_k{K}/
datasets/continuous_q{Q}_k{K}/
datasets/imbalanced_q{Q}_k{K}_{severity}/
```

### Output directories
```
outputs/{dgp}_{model}_q{Q}_k{K}_s{seed}/
outputs/{dgp}_{model}_q{Q}_k{K}_{variant}_s{seed}/

Examples:
outputs/static_magpcm_q200_k4_s42/
outputs/static_dkvmn_softmax_q200_k4_s42/
outputs/static_dynamic_gpcm_q200_k4_s42/
outputs/static_static_gpcm_q200_k4_s42/
outputs/discrete_magpcm_q200_k4_s42/
outputs/continuous_magpcm_q200_k4_s42/
outputs/scaling_magpcm_q500_k4_onehot_s42/
outputs/imbalanced_magpcm_q200_k4_mild_s42/
```

### Config files
```
configs/{dgp}_{model}_q{Q}_k{K}_s{seed}.yaml

Examples:
configs/static_magpcm_q200_k4_s42.yaml
configs/discrete_dkvmn_softmax_q200_k3_s7.yaml
configs/scaling_magpcm_q1000_k4_learned_s123.yaml
```

---

## 8. Verification Checklist

After bulk retrain, verify these approximate ranges (from current paper):

| Metric | Expected range | Table |
|--------|---------------|-------|
| MA-GPCM QWK (static K=4) | 0.68-0.70 | Table 1 |
| r_theta (MA-GPCM, static) | > 0.93 | Table 2 |
| r_alpha (MA-GPCM, static) | 0.65-0.80 | Table 2 |
| r_beta (all GPCM models) | > 0.96 | Table 2 |
| Binary ACC (K=2) | ~70% | Table 3 |
| Discrete GDA (MA-GPCM) | > 95% | Table 5 |
| Continuous endpoint r | > 0.91 | Table 7 |
