# Recovery Investigation Summary

## 1. Recovery Computation Differences

### Three Scripts Analyzed

**`compute_all_recovery.py` (CORRECT)**
- Lines 132-133: Uses first dimension only for D=1 experiments
```python
true_a_linked = link_alpha(true_alpha[seen])
est_a_linked = link_alpha(alpha_est[seen, 0])  # First dimension
```
- Applies linking to 1D arrays (Q,) after dimension selection
- **Status**: Working correctly for D=1 experiments

**`compute_deepgpcm_recovery.py` (BUGGY - REMOVED)**
- Lines 112-113: Applied linking to 2D arrays
```python
est_alpha_linked = link_alpha(est_alpha)  # BUG: 2D array (Q, D)
true_alpha_linked = link_alpha(true_alpha)
```
- Bug: `link_alpha()` computes `log_v.std()` across ALL elements, not per-dimension
- This gives wrong scaling for multi-dimensional alpha
- **Status**: Removed

**`extract_recovery_fast.py` (BUGGY - REMOVED)**
- Lines 60-61: Same bug as compute_deepgpcm_recovery.py
- **Status**: Removed

### Key Difference
The bug in scripts 2 & 3 is that `link_alpha()` at line 23 computes:
```python
std = log_v.std()  # Computes std across ALL elements of 2D array
```

For a (Q, D) array, this mixes dimensions and produces incorrect scaling. The correct approach (script 1) is to select dimension first, then apply linking to 1D array.

## 2. Archived "DKVMN+Ordinal" Model Investigation

### Finding: It's DeepGPCM with Ordinal-Only Loss

**Evidence from checkpoint analysis:**
```python
# outputs_archive_20260304_205653/ordinal_k5_s42/best.pt
Model keys: 23
Has IRT parameter heads: YES
  irt.ability_network.weight: torch.Size([1, 50])
  irt.discrimination_network.weight: torch.Size([1, 114])
  irt.threshold_base.weight: torch.Size([1, 64])
  irt.threshold_gaps.weight: torch.Size([3, 64])

# Comparison with deepgpcm_k5_s42
Keys in DeepGPCM but not Ordinal: set()
Keys in Ordinal but not DeepGPCM: set()
Are they the same architecture? True
```

**Config analysis (`configs/ordinal_k5_s42.yaml`):**
- **NO `model_type` field** - defaults to DeepGPCM via fallback
- Loss configuration:
  - `focal_weight: 0.0`
  - `weighted_ordinal_weight: 1.0`
- This is a **loss ablation**, not a separate model

**Training script fallback (`scripts/train.py` lines 62-75):**
```python
def build_model(cfg, device, n_students=0):
    model_type = getattr(cfg.model, "model_type", "deepgpcm")
    if model_type == "dkvmn_softmax":
        model = DKVMNSoftmax(**model_kwargs)
    elif model_type == "static_gpcm":
        model = StaticGPCM(n_students=n_students, **model_kwargs)
    elif model_type == "dynamic_gpcm":
        model = DynamicGPCM(n_students=n_students, **model_kwargs)
    else:
        model = DeepGPCM(**model_kwargs)  # ← Fallback
```

### What "ordinal" Actually Was

The archived experiments named "ordinal_k*_s42" were:
- **Architecture**: DeepGPCM (full IRT parameterization)
- **Loss**: Ordinal-only (no focal loss)
- **Purpose**: Loss ablation study

This explains why:
1. The checkpoint contains IRT parameters
2. Recovery metrics show r_α and r_β values (not NaN)
3. The architecture is identical to deepgpcm

### Current "dkvmn_ordinal" Configs

The new baseline configs (`configs/baselines/*_dkvmn_ordinal.yaml`) specify:
```yaml
model_type: "dkvmn_ordinal"  # Does not exist!
```

This model type doesn't exist in the codebase, so it falls back to DeepGPCM. This means:
- Current "dkvmn_ordinal" experiments are actually DeepGPCM
- They are NOT a separate baseline
- RQ1 comparisons are invalid (comparing model to itself)

## 3. Static GPCM Recovery Issue

Static GPCM shows anomalous recovery:
- r_α = 0.094 (essentially random)
- r̄_β = 0.890 (good)

**Possible causes:**
1. Alpha collapsed to constant values (identifiability issue)
2. Wrong recovery script was used (if buggy scripts 2/3 were used)
3. Data mismatch between true parameters and dataset

**Action**: Recompute with correct script (`compute_all_recovery.py`) to verify.

## 4. Actions Taken

1. ✅ Removed buggy recovery scripts:
   - `scripts/compute_deepgpcm_recovery.py`
   - `scripts/extract_recovery_fast.py`

2. ✅ Identified correct recovery script:
   - `scripts/compute_all_recovery.py`

3. ⏳ Running recovery recomputation:
   - Command: `PYTHONPATH=src python scripts/compute_all_recovery.py`
   - Output: `outputs/recovery_all.csv`

4. ✅ Created plot regeneration script:
   - `scripts/regenerate_all_recovery_plots.sh`
   - Generates split plots (student/item) for all Q and K combinations

## 5. Next Steps

1. Wait for recovery recomputation to complete
2. Run plot regeneration script for all experiments
3. Verify Static GPCM alpha recovery with correct computation
4. Update paper with correct recovery metrics
5. Address the "dkvmn_ordinal" baseline issue (separate concern)
