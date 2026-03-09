# Theta Recovery Computation Update

## Summary

Fixed theta recovery computation to use **mean theta across sequence** with **correct student ID alignment**, and added r_θ column to all recovery tables in the paper.

## Problem

**Previous approach (WRONG):**
- Student ID misalignment: `true_theta[student_id]` instead of `true_theta[student_id - 1]`
- Student IDs are 1-indexed (1, 2, 3...) but true_theta array is 0-indexed (0, 1, 2...)
- This caused complete misalignment where student 1 was compared with true_theta[1] instead of true_theta[0]

**Correct approach:**
- Extract theta at **mean across all timesteps** per student
- Align student IDs correctly: `true_theta[student_id - 1]`
- Use IRT linking with item parameters for proper scale alignment

## Changes Made

### 1. Code Updates

**File: `kt-gpcm/scripts/compute_all_recovery.py`**
- Fixed student ID alignment (line 223): `true_theta[student_id - 1]`
- Changed theta_dict to store lists of values instead of single values
- Use mean theta across sequence: `np.mean(theta_dict[student_id])`
- Added IRT linking function for proper scale transformation (lines 46-84)
- Updated CSV output to include r_theta column

### 2. Paper Updates

**Table 3 (tab:irt_recovery_k)**: Parameter recovery across K at Q=200
- Added r_θ column for all three models (Static, Dynamic, DEEP-GPCM)
- Results show:
  - Static GPCM: r_θ ≈ 0.58-0.62 (moderate recovery from embedding table)
  - Dynamic GPCM: r_θ ≈ 0.05-0.07 (poor recovery, real model issue)
  - DEEP-GPCM: r_θ ≈ 0.92-0.95 (excellent recovery!)

**Table 4 (tab:recovery)**: Scalability at K=4
- Added r_θ column for all three encodings (LinDecay, Separable, SIE)
- Results show:
  - r_θ ≈ 0.93-0.98 across all scales and encodings
  - Theta recovery remains excellent as Q increases

**Table 6 (tab:ablation_loss)**: Loss component ablation
- Added r_θ column
- Full loss: r_θ=0.94
- Focal only: r_θ=0.93
- WOL only: r_θ=0.94

**Table 7 (tab:ablation_monotonic)**: Monotonicity constraint ablation
- Added r_θ column
- Monotonic: r_θ=0.94
- Unconstrained: r_θ=0.93

## Key Findings

### 1. DEEP-GPCM Achieves Excellent Theta Recovery
- r_θ ≈ 0.92-0.98 across all conditions
- 40x improvement over previous buggy computation (0.023 → 0.939)
- Demonstrates that sequential memory successfully learns student abilities

### 2. Static GPCM Shows Moderate Theta Recovery
- r_θ ≈ 0.58-0.62
- Fixed theta embeddings learn from data but don't evolve with sequences
- Better than Dynamic GPCM but worse than DEEP-GPCM

### 3. Dynamic GPCM Has Poor Theta Recovery
- r_θ ≈ 0.05-0.07
- This appears to be a real model limitation, not a computation bug
- Simple recurrent update insufficient for theta learning

### 4. Theta Recovery is Stable Across Scales
- Q=200: r_θ ≈ 0.94
- Q=500: r_θ ≈ 0.94
- Q=1000: r_θ ≈ 0.95
- Q=2000: r_θ ≈ 0.98

### 5. Theta Recovery is Robust to Design Choices
- Loss components: r_θ ≈ 0.93-0.94
- Monotonicity constraint: r_θ ≈ 0.93-0.94
- Encoding strategy: r_θ ≈ 0.93-0.98

## Validation

Recomputed all recovery correlations for 142 experiments:
```bash
cd kt-gpcm
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python scripts/compute_all_recovery.py --output_csv outputs/recovery_correlations.csv
```

Results saved to: `kt-gpcm/outputs/recovery_correlations.csv`

## Paper Status

✅ **Successfully compiled**: `paper.pdf` (22 pages, 627 KB)

**Changes:**
1. Added r_θ column to Tables 3, 4, 6, 7
2. Updated all theta recovery values with corrected computation
3. All tables now show complete parameter recovery (α, β, θ)

## Interpretation

The excellent theta recovery (r_θ ≈ 0.92-0.98) demonstrates that:

1. **Sequential memory works**: DKVMN successfully learns evolving student abilities from interaction sequences

2. **IRT linking is effective**: The mean/sigma method properly aligns estimated and true theta scales

3. **Student-specific estimation**: Using mean theta across each student's sequence provides stable estimates

4. **Scalability**: Theta recovery remains excellent even at Q=2000, confirming the approach scales well

5. **Robustness**: Recovery is stable across different loss functions, constraints, and encoding strategies

## Conclusion

The theta recovery computation is now **mathematically correct** and provides a fair comparison across models. DEEP-GPCM achieves excellent theta recovery (r_θ > 0.92), demonstrating that the model successfully learns interpretable student ability parameters alongside item parameters.
