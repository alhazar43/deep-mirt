# Experiment Plan v3 (FINAL): DEEP-GPCM Paper

**Status: FINALIZED 2026-03-10**

## Motivation

The v1 experiments had critical design flaws:
1. **Confounded seq_len**: Q=2000 used seq_len [100,200] while Q=200/500/1000 used [20,80], making cross-Q comparisons invalid.
2. **Insufficient obs/item at large Q**: Q=2000 with N=5000 gives only 125 obs/item.
3. **Single-seed results**: no standard deviations reported; LinDecay r_alpha showed high seed sensitivity (.198--.335 at Q=1000).
4. **RQ1 conflated embedding and architecture**: should compare model architectures with a fixed embedding.

All existing datasets and checkpoints will be archived. New datasets use `v2_` prefix.

## Key Decisions

1. **seq_len [20, 80] for ALL datasets** -- confirmed by literature (Piech synthetic: 50 fixed, ASSIST2015: avg 34, ASSIST2009: avg 78, KT standard max: 200).
2. **N=10,000 for Q=2000** -- 250 obs/item is marginal but defensible; saves compute.
3. **RQ4 is encoding ablation** (not "scalability") -- compares LinDecay, Separable, SIE across Q in {200, 500, 1000, 2000}.
4. **5 seeds** {42, 123, 7, 0, 1} for all table results.
5. **Loss: weighted_ordinal_weight=1.0, focal_weight=0.0** for ALL models (deepgpcm and dkvmn_softmax).
6. **No loss ablation study**.

## Agent Responsibilities

| Agent | Responsibilities |
|-------|-----------------|
| **research-scientist** | Tracks experiment status, manages training queue, monitors convergence, verifies recovery correlations |
| **psychometric-researcher** | Updates paper tables and prose whenever new results are ready, ensures psychometric correctness of IRT formulations |
| **ml-system-architect** | Writes the bulk training script, handles GPU efficiency, parallelization, and config generation |

## Hyperparameter Specification

### Resolved: value_dim discrepancy

The paper (line 417) states d_v=128. The `ModelConfig` default is `value_dim: int = 128`. However, all v1 YAML configs explicitly override to `value_dim: 64`. The v1 results were generated with value_dim=64.

**Decision: Use value_dim=64 for v2 experiments.** This matches v1 results and keeps compute reasonable. Update paper line 417 to say d_v=64, or run experiments with d_v=128 if psychometric-researcher determines recovery improves. For now, lock to 64.

### Resolved: No early stopping in training script

The current `train.py` has no early stopping logic. It runs all configured epochs and saves the best checkpoint by QWK. The LR scheduler (`ReduceLROnPlateau`) provides implicit convergence pressure. This is acceptable -- we just need enough epochs.

### Resolved: Epoch counts from convergence analysis

Evidence from existing metrics.csv files:

| Model | Observed behavior | Conclusion |
|-------|------------------|------------|
| **deepgpcm** (Q=200, K=4, 15 ep) | val_loss: 1.114 -> 1.042 (still decreasing at ep 15); QWK: 0.648 -> 0.680 (still improving) | 15 epochs is NOT enough. Train loss gradient at ep 15: -0.0012/ep. Need ~30 epochs. |
| **dkvmn_softmax** (Q=200, K=4, 15 ep) | Similar architecture, same training dynamics as deepgpcm | Same epoch count as deepgpcm: 30. |
| **dynamic_gpcm** (Q=200, K=4, 30 ep) | val_loss: 1.529 -> 1.115 (still decreasing at ep 30); QWK: 0.280 -> 0.644 (still improving) | 30 epochs is marginal. Train 50 epochs. |
| **static_gpcm** (Q=200, K=4, 200 ep) | val_loss plateaus around epoch 145 at ~1.505; QWK converges to ~0.322; LR decays triggered at ep 185 | 200 epochs is adequate but wasteful. 150 epochs with LR schedule is sufficient. |

### Canonical hyperparameters (ALL runs)

```yaml
# Model (shared across deepgpcm/dkvmn_softmax where applicable)
memory_size: 50
key_dim: 64
value_dim: 64
summary_dim: 50
n_traits: 1
ability_scale: 1.0
dropout_rate: 0.0
memory_add_activation: "tanh"
init_value_memory: false
monotonic_betas: true  # except ablation

# Training (all models)
batch_size: 64
lr: 0.001
grad_clip: 1.0
focal_weight: 0.0
weighted_ordinal_weight: 1.0
ordinal_penalty: 0.5
lr_patience: 5
lr_factor: 0.8
attention_entropy_weight: 0.0
theta_norm_weight: 0.0
alpha_prior_weight: 0.0
beta_prior_weight: 0.0

# Data
train_split: 0.8
min_seq_len: 10
```

**Model-specific overrides:**

| Model | epochs | Extra params |
|-------|--------|-------------|
| deepgpcm | 30 | embedding_type per experiment |
| dkvmn_softmax | 30 | embedding_type: "static_item" |
| static_gpcm | 150 | (no memory/embedding params) |
| dynamic_gpcm | 50 | hidden_dim: 128 |

**Note on lr_patience/lr_factor change from v1:** v1 used patience=10/factor=0.9 which was too conservative -- LR barely decayed. Switching to patience=5/factor=0.8 for faster convergence and more aggressive LR schedule.

## Datasets

### Core datasets (RQ1, RQ2, RQ3, Ablation)

All at Q=200, N=5000, seq_len [20, 80] => 1250 obs/item.

| Name | Q | K | N | seq_len |
|------|---|---|---|---------|
| `v2_q200_k2` | 200 | 2 | 5,000 | [20, 80] |
| `v2_q200_k3` | 200 | 3 | 5,000 | [20, 80] |
| `v2_q200_k4` | 200 | 4 | 5,000 | [20, 80] |
| `v2_q200_k5` | 200 | 5 | 5,000 | [20, 80] |
| `v2_q200_k6` | 200 | 6 | 5,000 | [20, 80] |

### Encoding ablation datasets (RQ4)

Fixed K=4, seq_len [20, 80]. N chosen to balance obs/item and compute.

| Name | Q | K | N | seq_len | obs/item |
|------|---|---|---|---------|----------|
| `v2_q200_k4` | 200 | 4 | 5,000 | [20, 80] | 1,250 |
| `v2_q500_k4` | 500 | 4 | 5,000 | [20, 80] | 500 |
| `v2_q1000_k4` | 1,000 | 4 | 10,000 | [20, 80] | 500 |
| `v2_q2000_k4` | 2,000 | 4 | 10,000 | [20, 80] | 250 |

Note: Q=2000 uses N=10,000 (250 obs/item) -- marginal but defensible; saves compute.

### Imbalance datasets (RQ5)

Q=200, K=4, N=5000, seq_len [20, 80]. Shift theta prior upward.

| Name | theta prior | Expected distribution |
|------|---------|----------------------|
| `v2_q200_k4` | N(0, 1) | Balanced (shared) |
| `v2_q200_k4_mild_imb` | N(0.5, 1) | Mild skew |
| `v2_q200_k4_severe_imb` | N(1.0, 1) | Severe skew |
| `v2_q200_k4_extreme_imb` | N(1.5, 1) | Extreme skew |

**Total unique datasets: 10**

## Prerequisite: Code Changes Required

Before running experiments:

1. **Add `--theta_mean` argument to `data_gen.py`** -- currently not supported. Required for RQ5 imbalance datasets.
2. **Update paper line 417** to match d_v=64 (or decide to run with 128 and update configs).
3. **Update config defaults in `types.py`** to align with v2 recipe:
   - `lr_patience: 5` (currently 3 in defaults, 10 in v1 yamls)
   - `lr_factor: 0.8` (currently 0.8 in defaults, 0.9 in v1 yamls)
   - `focal_weight: 0.0` (currently 0.5)
   - `weighted_ordinal_weight: 1.0` (currently 0.5)

## Data Generation Commands

```bash
cd kt-gpcm && export PYTHONPATH=src

# Core (Q=200, K=3/4/5/6)
python scripts/data_gen.py --name v2_q200_k3 --n_students 5000 --n_questions 200 --n_cats 3 --min_seq 20 --max_seq 80 --output_dir data
python scripts/data_gen.py --name v2_q200_k4 --n_students 5000 --n_questions 200 --n_cats 4 --min_seq 20 --max_seq 80 --output_dir data
python scripts/data_gen.py --name v2_q200_k5 --n_students 5000 --n_questions 200 --n_cats 5 --min_seq 20 --max_seq 80 --output_dir data
python scripts/data_gen.py --name v2_q200_k6 --n_students 5000 --n_questions 200 --n_cats 6 --min_seq 20 --max_seq 80 --output_dir data

# Encoding ablation (K=4, varying Q and N)
python scripts/data_gen.py --name v2_q500_k4  --n_students 5000  --n_questions 500  --n_cats 4 --min_seq 20 --max_seq 80 --output_dir data
python scripts/data_gen.py --name v2_q1000_k4 --n_students 10000 --n_questions 1000 --n_cats 4 --min_seq 20 --max_seq 80 --output_dir data
python scripts/data_gen.py --name v2_q2000_k4 --n_students 10000 --n_questions 2000 --n_cats 4 --min_seq 20 --max_seq 80 --output_dir data

# Imbalance (Q=200, K=4, shifted theta priors) -- requires --theta_mean flag in data_gen.py
python scripts/data_gen.py --name v2_q200_k4_mild_imb    --n_students 5000 --n_questions 200 --n_cats 4 --min_seq 20 --max_seq 80 --output_dir data --theta_mean 0.5
python scripts/data_gen.py --name v2_q200_k4_severe_imb  --n_students 5000 --n_questions 200 --n_cats 4 --min_seq 20 --max_seq 80 --output_dir data --theta_mean 1.0
python scripts/data_gen.py --name v2_q200_k4_extreme_imb --n_students 5000 --n_questions 200 --n_cats 4 --min_seq 20 --max_seq 80 --output_dir data --theta_mean 1.5
```

## Training Matrix

### RQ1: Ordinal Prediction (architecture comparison)

All use SIE embedding on Q=200 data. Compare 4 architectures at 4 category levels.

| Model | model_type | embedding | Datasets | Seeds | Runs | Epochs/run |
|-------|-----------|-----------|----------|-------|------|------------|
| DEEP-GPCM (SIE) | deepgpcm | static_item | v2_q200_k{2,3,4,5,6} | 42,123,7,0,1 | 25 | 30 |
| Static GPCM | static_gpcm | -- | v2_q200_k{2,3,4,5,6} | 42,123,7,0,1 | 25 | 150 |
| Dynamic GPCM | dynamic_gpcm | -- | v2_q200_k{2,3,4,5,6} | 42,123,7,0,1 | 25 | 50 |
| DKVMN+Softmax | dkvmn_softmax | static_item | v2_q200_k{2,3,4,5,6} | 42,123,7,0,1 | 25 | 30 |
| **Total** | | | | | **100** | |

Metrics reported: Categorical Accuracy, Ordinal Accuracy, QWK, MAE, Spearman rho.

### RQ2: Learner Trajectory Visualization

No extra training. Use best DEEP-GPCM K=4 checkpoint from RQ1. Single seed, qualitative figure.

**Framing**: "ability estimation trajectories" (not "learning trajectories"), since the DGP has static theta.

### RQ3: Parameter Recovery

No extra training. Compute r_alpha, r_beta, r_theta from RQ1 checkpoints. Report mean +/- std across 5 seeds.

Models: DEEP-GPCM, Static GPCM, Dynamic GPCM (DKVMN+Softmax has no IRT params).

### RQ4: Encoding Ablation

All deepgpcm, K=4, 30 epochs. Tests how 3 embedding strategies behave across item bank sizes.

| Encoding | embedding_type | Datasets | Seeds | Runs |
|----------|---------------|----------|-------|------|
| SIE | static_item | v2_q{200,500,1000,2000}_k4 | 42,123,7,0,1 | 20 |
| Separable | separable | v2_q{200,500,1000,2000}_k4 | 42,123,7,0,1 | 20 |
| LinDecay | linear_decay | v2_q{200,500,1000,2000}_k4 | 42,123,7,0,1 | 20 |
| **Total** | | | | **60** |

Note: Q=200 SIE runs are shared with RQ1 K=4 (5 runs saved). Net new: 55 runs.

**Risk**: LinDecay at Q=2000 may OOM (8000-dim Kronecker input). If so, report as "did not scale" -- this is itself a finding.

### RQ5: Class Imbalance

DEEP-GPCM (SIE) only, Q=200, K=4, 30 epochs.

| Condition | Dataset | Seeds | Runs |
|-----------|---------|-------|------|
| Balanced | v2_q200_k4 | 42,123,7,0,1 | (shared with RQ1) |
| Mild | v2_q200_k4_mild_imb | 42,123,7,0,1 | 5 |
| Severe | v2_q200_k4_severe_imb | 42,123,7,0,1 | 5 |
| Extreme | v2_q200_k4_extreme_imb | 42,123,7,0,1 | 5 |
| **Total** | | | **15 new** |

### Ablation: Monotonicity Constraint

DEEP-GPCM (SIE), Q=200, K=4, monotonic_betas=false, 30 epochs.

| Condition | Seeds | Runs |
|-----------|-------|------|
| Unconstrained | 42,123,7,0,1 | 5 |

## Run Count Summary

| Source | Runs | Notes |
|--------|------|-------|
| RQ1 | 80 | 4 models x 4 K-values x 5 seeds |
| RQ4 | 55 | 3 encodings x 4 Q-values x 5 seeds, minus 5 shared with RQ1 |
| RQ5 | 15 | 3 imbalance conditions x 5 seeds (balanced shared with RQ1) |
| Ablation | 5 | monotonicity off x 5 seeds |
| **Grand total** | **155** | |

## Time Estimates

Timing based on v1 observations. Times are per-epoch, multiplied by epoch count.

### RQ1 (80 runs)

| Model | ep/run | sec/ep (Q=200) | time/run | total |
|-------|--------|----------------|----------|-------|
| deepgpcm | 30 | ~24s | ~12 min | 20 x 12 = 4.0h |
| dkvmn_softmax | 30 | ~24s | ~12 min | 20 x 12 = 4.0h |
| static_gpcm | 150 | ~0.7s | ~2 min | 20 x 2 = 0.7h |
| dynamic_gpcm | 50 | ~15s | ~12.5 min | 20 x 12.5 = 4.2h |
| **RQ1 subtotal** | | | | **~13h** |

### RQ4 (55 new runs, all deepgpcm 30 epochs)

| Q | N | sec/ep (est.) | time/run | runs | total |
|---|---|--------------|----------|------|-------|
| 200 | 5,000 | ~24s | ~12 min | 10 | 2.0h |
| 500 | 5,000 | ~28s | ~14 min | 15 | 3.5h |
| 1,000 | 10,000 | ~60s | ~30 min | 15 | 7.5h |
| 2,000 | 10,000 | ~90s | ~45 min | 15 | 11.3h |
| **RQ4 subtotal** | | | | | **~24h** |

### RQ5 + Ablation (20 runs, all deepgpcm 30 epochs Q=200)

20 x 12 min = **~4h**

### Total Compute Budget

| Phase | Runs | Est. Time |
|-------|------|-----------|
| Data generation | 10 datasets | < 10 min |
| RQ1 | 80 | ~13h |
| RQ4 | 55 | ~24h |
| RQ5 + Ablation | 20 | ~4h |
| Recovery computation | all | < 1h |
| **Grand total** | **155** | **~42 hours** |

## Execution Priority

### Phase 0: Prerequisites (before any training)
1. Add `--theta_mean` to `data_gen.py` **(ml-system-architect)**
2. Update `types.py` defaults to match v2 recipe **(ml-system-architect)**
3. Write bulk training script with config generation **(ml-system-architect)**
4. Generate all 10 datasets (< 10 min) **(research-scientist)**

### Phase 1: Smoke Test (~15 min)
5. Single run: DEEP-GPCM SIE, K=4, seed=42 on v2_q200_k4 -- verify loss/QWK trajectory over 30 epochs **(research-scientist)**
6. Single run: Static GPCM, K=4, seed=42 -- verify convergence by epoch 150 **(research-scientist)**

### Phase 2: Core Results (overnight, ~13h)
7. **RQ1 full**: 80 runs across 4 models x 4 K x 5 seeds **(ml-system-architect runs, research-scientist monitors)**
8. **Recovery computation** for RQ1/RQ3 **(research-scientist)**
9. **Paper table updates** for RQ1/RQ3 **(psychometric-researcher)**

### Phase 3: Encoding Ablation (~24h, can be split across 2 nights)
10. **RQ4 Q=200/500** (5.5h) -- run first, get early results
11. **RQ4 Q=1000** (7.5h) -- overnight
12. **RQ4 Q=2000** (11.3h) -- overnight (check LinDecay OOM first with single test run)
13. **Paper table updates** for RQ4 **(psychometric-researcher)**

### Phase 4: Imbalance + Ablation (~4h)
14. **RQ5**: 15 runs **(research-scientist)**
15. **Ablation monotonic**: 5 runs **(research-scientist)**
16. **Paper table updates** for RQ5 + ablation **(psychometric-researcher)**

### Phase 5: Finalization
17. **RQ2 trajectory figure** from best RQ1 checkpoint **(psychometric-researcher)**
18. Full recovery correlation table **(research-scientist)**
19. Paper revision and consistency check **(all)**

## Naming Convention

Configs: `v2_q{Q}_k{K}_{model_type}_{embedding}_s{seed}.yaml`
Outputs: `outputs/v2_q{Q}_k{K}_{model_type}_{embedding}_s{seed}/`

Examples:
- `v2_q200_k4_deepgpcm_sie_s42.yaml`
- `v2_q1000_k4_deepgpcm_lindecay_s123.yaml`
- `v2_q200_k4_static_gpcm_s7.yaml`
- `v2_q200_k4_deepgpcm_sie_nomonot_s0.yaml`

For static/dynamic GPCM (no embedding type): `v2_q200_k4_static_gpcm_s42.yaml`

## Changes from v1 / v2-draft

| Aspect | v1 | v2-draft | v3-final |
|--------|----|----|-----|
| seq_len | Variable per Q | Constant [20, 80] | Constant [20, 80] |
| n_students Q=2000 | 5,000 | 20,000 | 10,000 |
| obs/item Q=2000 | 125 | 500 | 250 (defensible) |
| Seeds | 1 (42) | 3 {42,123,7} | 5 {42,123,7,0,1} |
| Loss | focal=0.5 + wol=0.5 | wol=1.0 | wol=1.0, focal=0.0 |
| deepgpcm epochs | 15 | 15 | 30 |
| dynamic_gpcm epochs | 30 | 200 | 50 |
| static_gpcm epochs | 200 | 200 | 150 |
| dkvmn_softmax epochs | 15 | 15 | 30 |
| value_dim | 64 (config) / 128 (paper) | unresolved | 64 (align paper) |
| lr_patience | 10 | 10 | 5 |
| lr_factor | 0.9 | 0.9 | 0.8 |
| Total runs | ~30 | ~93 | 155 |
| RQ4 framing | Scalability | Scalability | Encoding ablation |
| Loss ablation | yes | planned | removed |
