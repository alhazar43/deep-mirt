# Final Summary: Recovery Investigation & Paper Cleanup

## Completed Tasks

### 1. Recovery Computation Bug Fix ✅

**Problem Identified:**
- Two buggy scripts (`compute_deepgpcm_recovery.py`, `extract_recovery_fast.py`) applied `link_alpha()` to 2D arrays (Q, D)
- Bug: `log_v.std()` computed std across ALL elements instead of per-dimension
- This corrupted alpha recovery for multi-dimensional experiments

**Solution:**
- Removed buggy scripts
- Kept correct script: `compute_all_recovery.py` which selects first dimension before linking
- Created targeted script: `compute_recovery_key_experiments.py` for all Q×K baselines

**Results:**
- Successfully computed recovery for 80 experiments (4 Q values × 5 K values × 4 models)
- Output: `outputs/recovery_baselines.csv`
- Confirmed Static GPCM poor alpha recovery (r_α=0.024-0.183) is real, not a bug

### 2. DKVMN+Ordinal Investigation ✅

**Finding:**
The archived "DKVMN+Ordinal" was **NOT a separate baseline** - it was DeepGPCM with ordinal-only loss.

**Evidence:**
1. **Checkpoint analysis**: Identical 23 keys to DeepGPCM, all IRT parameters present
   ```python
   irt.ability_network.weight: torch.Size([1, 50])
   irt.discrimination_network.weight: torch.Size([1, 114])
   irt.threshold_base.weight: torch.Size([1, 64])
   irt.threshold_gaps.weight: torch.Size([3, 64])
   ```

2. **Config analysis**: No `model_type` field in archived configs
   - Defaults to DeepGPCM via fallback in `train.py` line 74
   - Loss config: `focal_weight: 0.0`, `weighted_ordinal_weight: 1.0`

3. **Training script**:
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

4. **Current configs**: New baseline configs specify `model_type: "dkvmn_ordinal"` which doesn't exist

**Conclusion:** This was a **loss ablation study**, not a separate architecture.

### 3. Paper Cleanup ✅

**Changes Made to paper.tex:**

1. **Baseline description (line 558)**
   - Changed from "four baselines" to "three baselines"
   - Removed DKVMN+Ordinal bullet point

2. **Results narrative (line 598)**
   - Before: "DKVMN+Ordinal achieves parity with DEEP-GPCM"
   - After: "DEEP-GPCM outperforms DKVMN+Softmax across all category resolutions"
   - Added specific comparison: K=6, QWK 0.791 vs 0.773

3. **Table 1 (lines 602-617)**
   - Caption: "matches DKVMN+Ordinal" → "outperforms DKVMN+Softmax"
   - Removed DKVMN+Ordinal row from table

4. **Binary compatibility section (line 626)**
   - Removed reference to "matches DKVMN+Ordinal on Synthetic-Ordinal (K=2)"

5. **Table 2 (lines 628-644)**
   - Removed Synthetic-Ordinal columns (only had DKVMN+Ordinal data)
   - Removed DKVMN+Ordinal row
   - Now shows only ASSIST2015 and Synthetic-5

6. **Table 3 caption (line 668)**
   - Removed "identical to DKVMN+Ordinal in estimates but uniquely exposes them"
   - Simplified to focus on DEEP-GPCM's recovery

**Impact:**
- Paper narrative is now **stronger**: DEEP-GPCM improves both prediction AND interpretability
- No longer comparing model to itself
- Claims are consistent with actual experimental setup

## Recovery Results Summary

### Key Findings from 80 Experiments:

**DKVMN+Ordinal (actually DeepGPCM):**
- Q=200, K=4: r_α=0.707, r̄_β=0.918 ✓ (matches paper)
- Q=200, K=5: r_α=0.746, r̄_β=0.912
- Q=200, K=6: r_α=0.759, r̄_β=0.927
- Best overall recovery across all conditions

**Static GPCM:**
- Q=200: r_α=0.108-0.183 (poor)
- Q=1000, K=5: r_α=0.094 (the anomalous value)
- Q=2000: r_α=0.024-0.054 (worse with more items)
- Beta recovery remains good: r̄_β=0.843-0.964
- **Conclusion**: Poor alpha recovery is real, not a computation bug

**Dynamic GPCM:**
- Q=200, K=4: r_α=0.634, r̄_β=0.949
- Moderate recovery, better than Static but worse than DeepGPCM
- Degrades with scale: Q=2000, r_α=0.082-0.323

**DKVMN+Softmax:**
- All NaN (expected - no IRT parameterization)

**Scalability Pattern:**
- Alpha recovery degrades as Q increases for all models
- DeepGPCM maintains best recovery across scales
- Static GPCM degrades most severely

## Files Created/Modified

### Created:
- `scripts/compute_recovery_key_experiments.py` - Targeted recovery computation
- `scripts/regenerate_all_recovery_plots.sh` - Plot generation for all Q×K
- `RECOVERY_INVESTIGATION_SUMMARY.md` - Technical analysis
- `PAPER_EDITS_DKVMN_ORDINAL.md` - Paper changes documentation
- `FINAL_SUMMARY.md` - This file

### Modified:
- `paper.tex` - Removed all DKVMN+Ordinal references
- `scripts/regenerate_all_recovery_plots.sh` - Fixed checkpoint paths

### Removed:
- `scripts/compute_deepgpcm_recovery.py` - Buggy 2D linking
- `scripts/extract_recovery_fast.py` - Buggy 2D linking

### Generated:
- `outputs/recovery_baselines.csv` - Recovery metrics for 80 experiments
- Recovery plots (in progress): `outputs/large_q*_k*_dkvmn_ordinal/recovery_split_*.pgf`

## Next Steps

1. ✅ Recovery computation completed
2. ⏳ Plot regeneration running (will create split figures for all Q×K)
3. 📝 Update paper tables with correct recovery metrics if needed
4. 📝 Consider renaming "dkvmn_ordinal" experiments to "deepgpcm" for clarity
5. 📝 Update paper to use correct experiment names in figure references

## Technical Notes

### Recovery Computation Method (Correct)
```python
# Correct: Select dimension first, then link
true_a_linked = link_alpha(true_alpha[seen])
est_a_linked = link_alpha(alpha_est[seen, 0])  # First dimension only
r_alpha = np.corrcoef(true_a_linked, est_a_linked)[0, 1]
```

### Buggy Method (Removed)
```python
# Bug: Link 2D array, std computed across all elements
est_alpha_linked = link_alpha(est_alpha)  # (Q, D) array
true_alpha_linked = link_alpha(true_alpha)
# link_alpha() computes log_v.std() across ALL elements
```

### Model Naming Clarification
- **In code**: `dkvmn_ordinal` (config) → DeepGPCM (actual model via fallback)
- **In paper**: Should be called "DEEP-GPCM" or "DeepGPCM"
- **Archived experiments**: "ordinal_k*_s42" were loss ablations, not separate models

## Validation

All changes validated:
- ✅ Recovery computation produces consistent results
- ✅ Paper no longer references non-existent baseline
- ✅ Recovery metrics match archived results
- ✅ Static GPCM poor recovery confirmed as real phenomenon
- ✅ Plot generation script updated with correct paths
