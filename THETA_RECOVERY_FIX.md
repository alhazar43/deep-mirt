# Theta Recovery Fix - Student ID Alignment Bug

## The Bug

**Problem**: Student IDs in the dataloader are 1-indexed, but true_theta array is 0-indexed.

**Incorrect alignment**:
```python
for student_id in sorted(theta_dict.keys()):
    if student_id < len(true_theta):
        theta_est_aligned.append(theta_dict[student_id])
        true_theta_aligned.append(true_theta[student_id])  # WRONG!
```

This compared:
- student_id=1 with true_theta[1] (should be true_theta[0])
- student_id=2 with true_theta[2] (should be true_theta[1])
- etc.

**Correct alignment**:
```python
for student_id in sorted(theta_dict.keys()):
    if student_id >= 1 and student_id <= len(true_theta):
        theta_est_aligned.append(np.mean(theta_dict[student_id]))
        true_theta_aligned.append(true_theta[student_id - 1])  # CORRECT!
```

## The Fix

1. **Corrected indexing**: `true_theta[student_id - 1]`
2. **Use mean theta**: Average across all timesteps (more stable than last timestep)
3. **Store all theta values**: Changed theta_dict from storing single value to list of values

## Results

### Before Fix (WRONG):
- Static GPCM: r_θ = 0.580 (correct by luck - uses embedding table)
- Dynamic GPCM: r_θ = 0.057
- DEEP-GPCM: r_θ = 0.023 ❌

### After Fix (CORRECT):
- Static GPCM: r_θ = 0.580 (unchanged - uses embedding table)
- Dynamic GPCM: r_θ = 0.057 (unchanged - real issue with this model)
- DEEP-GPCM: r_θ = 0.939 ✅ (40x improvement!)

## Why This Happened

The bug only affected DEEP-GPCM because:
- **Static/Dynamic GPCM**: Extract theta from embedding table (1-indexed with padding at 0), so alignment was correct
- **DEEP-GPCM**: Extract theta from forward pass using student_ids from batch, which are 1-indexed but were incorrectly mapped to 0-indexed true_theta

## Validation

Checked first 10 students with corrected alignment:
```
Student 1: true=0.305, est_mean=0.626  (was comparing with true_theta[1]=-1.040)
Student 2: true=-1.040, est_mean=-0.683 (was comparing with true_theta[2]=0.750)
Student 3: true=0.750, est_mean=1.251  (was comparing with true_theta[3]=0.941)
...
```

Correlation improved from 0.023 to 0.939!

## Action Required

Recompute ALL recovery correlations with the fixed script.
