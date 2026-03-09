# Recovery Script Comparison - Detailed Analysis

**Investigation Date**: 2026-03-08
**File Timestamps**: All recovery computation scripts modified 2026-03-06 (2 days ago)

---

## CRITICAL FINDING: Three Different Recovery Methods

### Script 1: `compute_all_recovery.py` (CURRENT STANDARD)
**Last Modified**: 2026-03-06 01:36
**Purpose**: Compute recovery for ALL experiments (baselines + DEEP-GPCM)

#### Alpha Recovery Method:
```python
# Lines 89-90, 118-119, 128-134
alpha_sum = np.zeros((Q, D))           # Accumulator: (Q, D)
alpha_count = np.zeros((Q,))           # Counter: (Q,)

# Accumulation loop
alpha_sum[qid] += alpha[b, t]          # alpha[b,t] is (D,) - adds full vector
alpha_count[qid] += 1

# Averaging
alpha_est = np.where(seen[:, None],
                     alpha_sum / np.maximum(alpha_count[:, None], 1),
                     0.0)                # Result: (Q, D)

# Correlation - ONLY USES FIRST DIMENSION
true_a_linked = link_alpha(true_alpha[seen])      # Shape: (Q_seen,) if true_alpha is 1D
est_a_linked = link_alpha(alpha_est[seen, 0])    # Shape: (Q_seen,) - ONLY DIM 0
r_alpha = np.corrcoef(true_a_linked, est_a_linked)[0, 1]
```

**KEY ISSUE**: Line 133 uses `alpha_est[seen, 0]` - **ONLY the first dimension**
- For D=1: This is correct
- For D>1: This **ignores all other dimensions**
- True alpha shape assumption: **1D array** (scalar per item)

#### Beta Recovery Method:
```python
# Lines 91-92, 120-121, 129, 137-144
beta_sum = np.zeros((Q, K - 1))
beta_count = np.zeros((Q,))

# Accumulation
beta_sum[qid] += beta[b, t]            # beta[b,t] is (K-1,)
beta_count[qid] += 1

# Averaging
beta_est = np.where(seen[:, None],
                    beta_sum / np.maximum(beta_count[:, None], 1),
                    0.0)                # Result: (Q, K-1)

# Correlation - PER THRESHOLD
r_beta_list = []
for k in range(K - 1):
    true_b = link_normal(true_beta[seen, k])
    est_b = link_normal(beta_est[seen, k])
    r = np.corrcoef(true_b, est_b)[0, 1]
    r_beta_list.append(r)
r_beta_mean = np.mean(r_beta_list)
```

**Correct**: Computes correlation per threshold, then averages

---

### Script 2: `compute_deepgpcm_recovery.py` (DEEP-GPCM ONLY)
**Last Modified**: 2026-03-06 01:48
**Purpose**: Only process DEEP-GPCM experiments (skip baselines)

#### Alpha Recovery Method:
```python
# Lines 69-70, 93-96, 105-128
alpha_sum = np.zeros((Q, D))
alpha_count = np.zeros((Q,))

# Accumulation (IDENTICAL to Script 1)
alpha_sum[q] += alpha[b, s]
alpha_count[q] += 1

# Averaging
est_alpha = np.zeros_like(alpha_sum)   # (Q, D)
est_alpha[mask_alpha] = alpha_sum[mask_alpha] / alpha_count[mask_alpha, None]

# Linking - APPLIES TO FULL ARRAY
est_alpha_linked = link_alpha(est_alpha)        # Links (Q, D) array
true_alpha_linked = link_alpha(true_alpha)      # Links true_alpha (shape?)

# Correlation - PER DIMENSION, THEN AVERAGE
r_alpha_list = []
for d in range(D):
    mask = mask_alpha & ~(np.isnan(est_alpha_linked[:, d]) | np.isnan(true_alpha_linked[:, d]))
    if mask.sum() > 1:
        r = np.corrcoef(est_alpha_linked[mask, d], true_alpha_linked[mask, d])[0, 1]
        r_alpha_list.append(r if not np.isnan(r) else 0.0)
    else:
        r_alpha_list.append(0.0)
r_alpha = np.mean(r_alpha_list)
```

**KEY DIFFERENCE**:
- Applies `link_alpha()` to **FULL (Q, D) array** (line 112)
- Computes correlation **per dimension** (lines 120-126)
- **Averages across dimensions** (line 128)

**PROBLEM**: `link_alpha()` expects 1D input but receives 2D!
```python
def link_alpha(vals: np.ndarray, target_std: float = 0.3) -> np.ndarray:
    log_v = np.log(np.maximum(vals, 1e-6))
    std = log_v.std()  # <-- This computes std across ALL elements (Q*D)
    # ...
```
This will compute std across the **flattened array**, not per-dimension!

#### Beta Recovery Method:
```python
# Lines 71-72, 95-96, 108-139
# IDENTICAL to Script 1 - per threshold correlation
```

---

### Script 3: `extract_recovery_fast.py` (CHECKPOINT-BASED)
**Last Modified**: 2026-03-06 01:46
**Purpose**: Read pre-saved estimates from checkpoint (no inference)

#### Alpha Recovery Method:
```python
# Lines 56-77
# Get estimates from checkpoint
est_alpha = ckpt["alpha_estimates"].cpu().numpy()  # Assumes (Q, D)
est_beta = ckpt["beta_estimates"].cpu().numpy()    # Assumes (Q, K-1)

# Link parameters - FULL ARRAYS
est_alpha_linked = link_alpha(est_alpha)           # Links (Q, D) - SAME BUG AS SCRIPT 2
true_alpha_linked = link_alpha(true_alpha)

# Compute correlations - PER DIMENSION
r_alpha_list = []
for d in range(est_alpha.shape[1]):
    mask = ~(np.isnan(est_alpha_linked[:, d]) | np.isnan(true_alpha_linked[:, d]))
    if mask.sum() > 1:
        r = np.corrcoef(est_alpha_linked[mask, d], true_alpha_linked[mask, d])[0, 1]
        r_alpha_list.append(r if not np.isnan(r) else 0.0)
    else:
        r_alpha_list.append(0.0)
r_alpha = np.mean(r_alpha_list)
```

**IDENTICAL BUG to Script 2**: Applies `link_alpha()` to 2D array

#### Beta Recovery Method:
```python
# Lines 63-64, 79-89
# IDENTICAL to Scripts 1 & 2
```

---

## CRITICAL DIFFERENCES SUMMARY

| Aspect | Script 1 (compute_all) | Script 2 (deepgpcm) | Script 3 (fast) |
|--------|------------------------|---------------------|-----------------|
| **Alpha Linking** | Per-dimension (implicit) | Full array (BUG) | Full array (BUG) |
| **Alpha Correlation** | First dim only | Per-dim, averaged | Per-dim, averaged |
| **True Alpha Shape** | Assumes 1D | Assumes 2D | Assumes 2D |
| **Data Source** | Forward pass | Forward pass | Checkpoint |
| **Scope** | All experiments | DEEP-GPCM only | All experiments |

---

## THE SMOKING GUN: Shape Mismatch

### True Alpha Format (from data files):
```json
"alpha": [
    0.9255328734148558,
    0.9140721542734216,
    0.8113968797333475,
    ...
]
```
**Shape**: `(Q,)` - **1D array, scalar per item**

### Estimated Alpha Format (from models):
```python
# DeepGPCM.forward() returns:
"alpha": alpha  # Shape: (B, S, D)

# After accumulation:
alpha_est  # Shape: (Q, D)
```
**Shape**: `(Q, D)` - **2D array, D-dimensional vector per item**

### The Problem:

**Script 1** (`compute_all_recovery.py`):
```python
true_a_linked = link_alpha(true_alpha[seen])      # Input: (Q_seen,) - 1D
est_a_linked = link_alpha(alpha_est[seen, 0])    # Input: (Q_seen,) - 1D (first dim only)
```
✅ **Correct for D=1**: Both are 1D, correlation is valid
❌ **Wrong for D>1**: Ignores dimensions 1..D-1

**Scripts 2 & 3** (`compute_deepgpcm_recovery.py`, `extract_recovery_fast.py`):
```python
true_alpha_linked = link_alpha(true_alpha)        # Input: (Q,) - 1D
est_alpha_linked = link_alpha(est_alpha)          # Input: (Q, D) - 2D
```
❌ **BROKEN**: `link_alpha()` computes std across flattened array
- For true_alpha: std of Q values
- For est_alpha: std of Q*D values
- **Linking scales are incompatible!**

Then:
```python
for d in range(D):
    r = np.corrcoef(est_alpha_linked[mask, d], true_alpha_linked[mask, d])
```
❌ **BROKEN**: `true_alpha_linked` is 1D, indexing `[:, d]` will fail or give wrong results

---

## WHICH SCRIPT WAS USED?

### Evidence from Output Files:

**Current `outputs/recovery_correlations.csv`**:
```csv
large_q1000_k5_dkvmn_ordinal,0.5482587543673634,0.9026836862247193,"0.8939,0.9189,0.9142,0.8837"
large_q1000_k5_static_gpcm,0.09407938640347963,0.8902804870678973,"0.8775,0.8916,0.8979,0.8942"
```

**Checking which script could produce these results**:

Script 1 would produce:
- Single r_alpha value (not per-dimension)
- Uses first dimension only
- ✅ Matches output format

Scripts 2 & 3 would produce:
- Averaged r_alpha across dimensions
- Would crash or give wrong results due to shape mismatch
- ❌ Cannot produce valid output

**Conclusion**: `compute_all_recovery.py` (Script 1) was used for current results

---

## HISTORICAL ANALYSIS

### File Timestamps:
- `compute_all_recovery.py`: 2026-03-06 01:36
- `compute_deepgpcm_recovery.py`: 2026-03-06 01:48
- `extract_recovery_fast.py`: 2026-03-06 01:46

**All modified 2 days ago** - suggests recent refactoring

### Archived Results:
Need to check if archived experiments used a different script version

---

## IMPLICATIONS FOR STATIC GPCM ALPHA RECOVERY

### Current Results:
```
Static GPCM (K=5): r_α = 0.094, r̄_β = 0.890
```

### Hypothesis: Script 1 is Correct, Model is Wrong

If Script 1 is correct (uses first dimension only), then:
1. True alpha is 1D: `(Q,)` with scalar values
2. Estimated alpha is 2D: `(Q, D)` with D=1
3. Correlation uses `alpha_est[seen, 0]` - the only dimension

**This should work correctly for D=1!**

So why is r_α so low?

### Diagnostic: Check True Alpha Shape

```python
import json
import numpy as np

with open('data/large_q1000_k5/true_irt_parameters.json') as f:
    irt = json.load(f)

true_alpha = np.array(irt['alpha'])
print(f"True alpha shape: {true_alpha.shape}")
print(f"True alpha ndim: {true_alpha.ndim}")
print(f"First 10 values: {true_alpha[:10]}")
```

If `true_alpha.shape = (Q,)` → Script 1 is correct
If `true_alpha.shape = (Q, D)` → Script 1 has a bug

### Diagnostic: Check Estimated Alpha Values

```python
import torch
ckpt = torch.load('outputs/large_q1000_k5_static_gpcm/best.pt')

# Check if estimates are saved
if 'alpha_estimates' in ckpt:
    alpha_est = ckpt['alpha_estimates']
    print(f"Saved alpha shape: {alpha_est.shape}")
    print(f"Alpha std: {alpha_est.std()}")
    print(f"Alpha range: [{alpha_est.min()}, {alpha_est.max()}]")
else:
    # Need to run inference
    print("No saved estimates - need to run compute_all_recovery.py with debug")
```

If `alpha_est.std() < 0.01` → Alpha has collapsed to constant (identifiability issue)
If `alpha_est.std() > 0.1` → Alpha is varying, correlation should be higher

---

## RECOMMENDATIONS

### Immediate Actions:

1. **Fix Scripts 2 & 3** - They have a critical bug:
```python
# WRONG (current):
est_alpha_linked = link_alpha(est_alpha)  # 2D input

# CORRECT (should be):
est_alpha_linked = np.zeros_like(est_alpha)
for d in range(D):
    est_alpha_linked[:, d] = link_alpha(est_alpha[:, d])
```

2. **Verify True Alpha Shape**:
```bash
python -c "import json, numpy as np; irt=json.load(open('data/large_q1000_k5/true_irt_parameters.json')); print(np.array(irt['alpha']).shape)"
```

3. **Add Debug Output to Script 1**:
```python
# After line 128
print(f"DEBUG: true_alpha shape: {true_alpha.shape}")
print(f"DEBUG: alpha_est shape: {alpha_est.shape}")
print(f"DEBUG: true_alpha[seen] shape: {true_alpha[seen].shape}")
print(f"DEBUG: alpha_est[seen, 0] shape: {alpha_est[seen, 0].shape}")
print(f"DEBUG: alpha_est std: {alpha_est[seen, 0].std():.4f}")
print(f"DEBUG: true_alpha std: {true_alpha[seen].std():.4f}")
```

4. **Standardize on One Script**:
- Use Script 1 (`compute_all_recovery.py`) as canonical
- Fix the first-dimension-only issue for D>1
- Deprecate Scripts 2 & 3 or fix their bugs

### For Paper:

5. **Document Recovery Method**:
```latex
\footnote{Parameter recovery computed using Pearson correlation after IRT linking
transformations: $\alpha' = \exp((\log\alpha - \mu_{\log\alpha})/\sigma_{\log\alpha} \cdot 0.3)$
and $\beta' = (\beta - \mu_\beta)/\sigma_\beta$. For multidimensional $\alpha$,
we report correlation for the first dimension.}
```

6. **Add to Methods Section**:
> "Recovery correlations were computed by accumulating parameter estimates across
> all training and test sequences, averaging per item, applying IRT linking
> transformations to account for scale indeterminacy, and computing Pearson
> correlations with ground truth parameters."

---

## NEXT STEPS FOR INVESTIGATION

1. ✅ Identified three recovery scripts with different methods
2. ✅ Found critical bug in Scripts 2 & 3 (2D linking)
3. ✅ Determined Script 1 is currently used
4. ⏳ Need to verify true alpha shape in data files
5. ⏳ Need to check if Static GPCM alpha has collapsed
6. ⏳ Need to examine archived results for comparison
7. ⏳ Need to trace model calling for DKVMN+Ordinal

**Proceeding to Task 2: Archived Results Analysis**
