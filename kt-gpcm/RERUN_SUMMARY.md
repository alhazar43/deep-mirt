# Bulk Re-run Summary (2026-03-04)

## Problem Identified
- Previous experiments used K=5 for Q-size ablation (RQ4), but standard should be K=4
- Results were inconsistent between paper and actual outputs
- Need to standardize all experiments to K=4 (except RQ1 which tests K∈{2,3,4,5})

## Actions Taken

### 1. Archived Previous Results
- Moved `outputs/` → `outputs_archive_k5_20260304/`
- Preserves all K=5 results for reference

### 2. Regenerated Datasets with K=4
Created new datasets:
- `large_5000_q500_k4`: N=5000, Q=500, K=4, seq=[20,80]
- `large_5000_q1000_k4`: N=5000, Q=1000, K=4, seq=[20,80]
- `large_q5000_k4`: N=1000, Q=5000, K=4, seq=[400,600]

### 3. Updated Configs
Updated all `large_q*.yaml` configs to:
- Set `n_categories: 4`
- Point to new K=4 datasets

### 4. Created Bulk Training Script
`scripts/bulk_train.sh` will re-run all experiments:

**RQ1: Ordinal Prediction (K∈{2,3,4,5})**
- deepgpcm, softmax, ordinal, static_gpcm, dynamic_gpcm
- For each K∈{2,3,4,5}

**RQ2: Learner Trajectories (K=4)**
- Covered by deepgpcm_k4_s42, softmax_k4_s42, dynamic_gpcm_k4_s42

**RQ3: IRT Parameter Recovery (K=4)**
- Covered by deepgpcm_k4_s42, static_gpcm_k4_s42, dynamic_gpcm_k4_s42

**RQ4: Scalable Item Encoding (K=4, varying Q)**
- Q=500: large_q500_v9a (LinearDecay), large_q500_v9b (StaticItem)
- Q=1000: large_q1000_v9a (LinearDecay), large_q1000_v9b (StaticItem)
- Q=5000: large_q5000_linear, large_q5000_separable, large_q5000_static

**RQ5: Ablation Studies (K=4)**
- Loss ablations: ablation_focal_k4_s42, ablation_wol_k4_s42
- Monotonicity: ablation_nomonot_k4_s42
- Sequence length: seqlen_s20_40_k4_s42, seqlen_s20_100_k4_s42

## Standard Model Configuration
- Model: DeepGPCM
- Embedding: static_item
- K: 4 (except RQ1 which varies K)
- Q: 200 (except RQ4 which varies Q)
- Memory size: 50
- Key/Value dim: 64
- Epochs: 15
- Batch size: 64
- Loss: 0.5×Focal + 0.5×WeightedOrdinal

## How to Run

```bash
cd kt-gpcm
export PYTHONPATH=src
bash scripts/bulk_train.sh
```

This will take several hours. Monitor progress in the terminal.

## After Training

1. Generate recovery plots for all models
2. Generate trajectory plots for K=4 models
3. Update paper.tex with new results
4. Verify all tables match actual outputs

## Paper Updates Needed

After re-running, update these sections:
- Table 1 (RQ1): K∈{2,3,4,5} results
- Figure 2 (RQ2): Trajectories at K=4
- Figure 3 & Table 2 (RQ3): Recovery at K=4
- Table 3 (RQ4): Q-size ablation at K=4
- Table 4 & 5 (RQ5): Ablations at K=4
