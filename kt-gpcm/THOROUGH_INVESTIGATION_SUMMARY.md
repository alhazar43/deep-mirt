# THOROUGH INVESTIGATION SUMMARY - kt-gpcm Critical Issues

**Investigation Date**: 2026-03-08
**Investigator**: Research Scientist Agent
**Status**: CRITICAL - Paper submission blocker

---

## EXECUTIVE SUMMARY

After thorough investigation of the kt-gpcm codebase, I have identified **critical flaws** that invalidate key paper claims:

1. **The "DKVMN+Ordinal" baseline has NEVER existed** - it's actually DeepGPCM
2. **Three different recovery computation methods exist** with incompatible implementations
3. **Static GPCM alpha recovery is anomalously low** (r_α=0.094) - requires investigation
4. **All baseline configs use identical loss weights** - no true ablation

**Paper Impact**: RQ1 conclusions are invalid. Tables 1-2 contain incorrect baseline comparisons. Recovery metrics are misleading.

---

## TASK 1: Recovery Computation Differences - COMPLETE ANALYSIS

### Three Scripts Identified:

| Script | Purpose | Alpha Method | Beta Method | Status |
|--------|---------|--------------|-------------|--------|
| `compute_all_recovery.py` | All experiments | First dim only | Per-threshold | ✅ WORKING |
| `compute_deepgpcm_recovery.py` | DEEP-GPCM only | Per-dim averaged | Per-threshold | ❌ BROKEN |
| `extract_recovery_fast.py` | From checkpoint | Per-dim averaged | Per-threshold | ❌ BROKEN |

### CRITICAL BUG in Scripts 2 & 3:

**Problem**: They apply `link_alpha()` to 2D arrays, but the function expects 1D input.

```python
# BROKEN CODE (Scripts 2 & 3):
est_alpha_linked = link_alpha(est_alpha)  # Input: (Q, D) - 2D array
true_alpha_linked = link_alpha(true_alpha)  # Input: (Q,) - 1D array

def link_alpha(vals: np.ndarray, target_std: float = 0.3) -> np.ndarray:
    log_v = np.log(np.maximum(vals, 1e-6))
    std = log_v.std()  # ← Computes std across ALL elements (Q*D)
    # This gives WRONG scaling for 2D input!
```

**Impact**:
- `link_alpha()` computes std across flattened array (Q*D elements)
- For true_alpha: std of Q values
- For est_alpha: std of Q*D values
- **Linking scales are incompatible** → correlations are meaningless

**Which Script Was Used?**

Evidence from output format:
```csv
large_q1000_k5_dkvmn_ordinal,0.5482587543673634,0.9026836862247193,"0.8939,0.9189,0.9142,0.8837"
```
- Single r_alpha value (not per-dimension) ✓
- Per-threshold r_beta values ✓
- **Matches Script 1 output format**

**Conclusion**: `compute_all_recovery.py` (Script 1) was used for current results.

### Script 1 Implementation Details:

```python
# Lines 89-134 of compute_all_recovery.py

# Accumulation: (Q, D) arrays
alpha_sum = np.zeros((Q, D))
alpha_count = np.zeros((Q,))

for batch in loader:
    # ... forward pass ...
    alpha = out["alpha"].cpu().numpy()  # (B, S, D)
    for b, t in valid_positions:
        qid = questions[b, t] - 1
        alpha_sum[qid] += alpha[b, t]  # Adds (D,) vector
        alpha_count[qid] += 1

# Averaging
alpha_est = alpha_sum / alpha_count[:, None]  # (Q, D)

# Correlation - ONLY FIRST DIMENSION
true_a_linked = link_alpha(true_alpha[seen])      # (Q_seen,) - 1D
est_a_linked = link_alpha(alpha_est[seen, 0])    # (Q_seen,) - 1D (first dim)
r_alpha = np.corrcoef(true_a_linked, est_a_linked)[0, 1]
```

**KEY ISSUE**: Line 133 uses `alpha_est[seen, 0]` - **ONLY the first dimension**

For D=1: This is correct (only one dimension exists)
For D>1: This **ignores dimensions 1..D-1**

### True Alpha Format:

From `data/large_q1000_k5/true_irt_parameters.json`:
```json
"alpha": [
    0.9255328734148558,
    0.9140721542734216,
    0.8113968797333475,
    ...
]
```

**Shape**: `(Q,)` - **1D array, scalar per item**

This confirms the data generation uses **unidimensional IRT** (D=1), so Script 1's first-dimension-only approach is correct for current experiments.

### Recommendations:

1. **Fix Scripts 2 & 3**:
```python
# CORRECT implementation:
est_alpha_linked = np.zeros_like(est_alpha)
for d in range(D):
    est_alpha_linked[:, d] = link_alpha(est_alpha[:, d])

# Then compute per-dimension correlations
```

2. **Standardize on Script 1** as canonical
3. **Add unit tests** for linking functions
4. **Document** which script was used for paper

---

## TASK 2: Archived Results Analysis - SMOKING GUN

### Key Finding: "ordinal" Was a Loss Ablation, Not a Baseline

**Archived Experiments** (2026-03-04):
- `deepgpcm_k5_s42/` - DeepGPCM with balanced loss (focal=0.5, ordinal=0.5)
- `ordinal_k5_s42/` - DeepGPCM with pure ordinal loss (focal=0.0, ordinal=1.0)
- `softmax_k5_s42/` - DKVMNSoftmax
- `static_gpcm_k5_s42/` - StaticGPCM
- `dynamic_gpcm_k5_s42/` - DynamicGPCM

### Config Comparison:

**deepgpcm_k5_s42.yaml**:
```yaml
model:
  # NO model_type field → defaults to DeepGPCM
  n_questions: 200
  # ... standard params ...

training:
  focal_weight: 0.5
  weighted_ordinal_weight: 0.5
  ordinal_penalty: 0.5
```

**ordinal_k5_s42.yaml**:
```yaml
model:
  # NO model_type field → defaults to DeepGPCM
  n_questions: 200
  # ... IDENTICAL params ...

training:
  focal_weight: 0.0              # ← ONLY DIFFERENCE
  weighted_ordinal_weight: 1.0   # ← ONLY DIFFERENCE
  ordinal_penalty: 0.5
```

**BOTH ARE DeepGPCM** - only difference is loss weights!

### Metrics Comparison:

| Experiment | Q | train_loss | val_qwk | Model |
|------------|---|------------|---------|-------|
| deepgpcm_k5_s42 (archived) | 200 | 1.2303 | 0.7584 | DeepGPCM |
| ordinal_k5_s42 (archived) | 200 | 1.8063 | 0.7609 | DeepGPCM |
| large_q1000_k5_dkvmn_ordinal (current) | 1000 | 1.2308 | 0.7515 | DeepGPCM |

**SMOKING GUN**: Current "dkvmn_ordinal" has train_loss=1.2308, nearly identical to archived "deepgpcm" (1.2303), confirming both are DeepGPCM with balanced loss.

### Checkpoint Size Analysis:

```
outputs_archive_20260304_205653/deepgpcm_k5_s42/best.pt:  280K  (Q=200)
outputs_archive_20260304_205653/ordinal_k5_s42/best.pt:   280K  (Q=200)
outputs/large_q1000_k5_dkvmn_ordinal/best.pt:              2.5M  (Q=1000)
```

Scaling: 2500K / 280K = 8.9x for 5x increase in Q
Expected: ~5-10x due to embedding tables + IRT parameters
**Confirms all checkpoints contain IRT parameters (DeepGPCM architecture)**

### Conclusion:

The archived "ordinal" experiments were **loss ablations**, not architecture baselines:
- Testing pure ordinal loss (weighted_ordinal_weight=1.0) vs balanced loss
- Both used DeepGPCM architecture
- Difference in QWK: 0.7609 vs 0.7584 (0.0025) - negligible

**There has NEVER been a true "DKVMN+Ordinal" baseline in this codebase.**

---

## TASK 3: Model Calling Investigation - COMPLETE TRACE

### Training Pipeline:

```
scripts/train.py
  ↓
main()
  ↓
cfg = load_config(args.config)  # Loads YAML
  ↓
model = build_model(cfg, device, n_students)
  ↓
build_model():
    model_type = getattr(cfg.model, "model_type", "deepgpcm")

    if model_type == "dkvmn_softmax":
        return DKVMNSoftmax(**kwargs)
    elif model_type == "static_gpcm":
        return StaticGPCM(n_students, **kwargs)
    elif model_type == "dynamic_gpcm":
        return DynamicGPCM(n_students, **kwargs)
    else:
        return DeepGPCM(**kwargs)  # ← CATCHES "dkvmn_ordinal"
```

### The Fatal Flaw:

**Silent Fallback**: The `else` clause catches ANY unrecognized model_type:
- `"deepgpcm"` → DeepGPCM ✓ (intended)
- `"dkvmn_ordinal"` → DeepGPCM ✗ (BUG)
- `"foobar"` → DeepGPCM ✗ (BUG)
- Missing field → DeepGPCM ✓ (default)

**NO ERROR, NO WARNING, NO VALIDATION**

### Bulk Training Script:

`scripts/run_baselines_comprehensive.sh` generates configs:

```bash
BASELINES=("static_gpcm" "dynamic_gpcm" "dkvmn_softmax" "dkvmn_ordinal")

for BASELINE in "${BASELINES[@]}"; do
    create_baseline_config ${EXP_NAME} ${Q} ${K} ${DATASET} ${BASELINE}
done
```

Creates `configs/baselines/large_q1000_k5_dkvmn_ordinal.yaml`:
```yaml
model:
  model_type: "dkvmn_ordinal"  # ← Does not exist!
  monotonic_betas: true        # ← DeepGPCM-specific parameter
```

### What Actually Runs:

1. Config specifies `model_type: "dkvmn_ordinal"`
2. Training script loads config
3. `build_model()` doesn't recognize "dkvmn_ordinal"
4. Falls through to `else` clause
5. **Instantiates DeepGPCM**
6. Trains with full IRT parameterization
7. Saves checkpoint with IRT parameters
8. Recovery script extracts valid IRT correlations
9. Paper reports as "DKVMN+Ordinal" baseline

### Evidence:

**1. Source Code Has No DKVMNOrdinal Class**:
```bash
$ grep -r "class DKVMNOrdinal" src/
# No results
```

**2. Recovery Shows IRT Parameters**:
```csv
large_q1000_k5_dkvmn_ordinal,0.548,0.903,"0.894,0.919,0.914,0.884"
```
If truly non-IRT, these would be NaN or 0.0.

**3. Training Loss Matches DeepGPCM**:
```
deepgpcm (archived):     1.2303
dkvmn_ordinal (current): 1.2308
```

**4. Checkpoint Size Matches DeepGPCM**:
```
Q=200 → 280K
Q=1000 → 2.5M
Scaling matches IRT parameter count
```

### Loss Configuration:

**ALL models use the SAME loss**:
```yaml
training:
  focal_weight: 0.5
  weighted_ordinal_weight: 0.5
  ordinal_penalty: 0.5
```

This includes:
- DeepGPCM ✓
- "DKVMN+Ordinal" (actually DeepGPCM) ✓
- DKVMNSoftmax ✓
- StaticGPCM ✓
- DynamicGPCM ✓

**Even DKVMNSoftmax uses ordinal loss!**

The only architectural difference is the head:
- DKVMNSoftmax: Linear classifier (K-way softmax)
- DeepGPCM: IRT extractor + GPCM logits
- Intended DKVMN+Ordinal: Linear ordinal head (never implemented)

---

## STATIC GPCM ALPHA RECOVERY ANOMALY

### Current Results:
```
Static GPCM (K=5): r_α = 0.094, r̄_β = 0.890
```

### Analysis:

**Good beta recovery** (0.890) suggests the model learned item difficulties correctly.
**Terrible alpha recovery** (0.094) suggests discrimination is essentially random.

### Possible Causes:

1. **Alpha Collapse**: Static GPCM may learn constant alpha (identifiability issue)
   - Without sequential constraints, discrimination may not be identifiable
   - Check: Load checkpoint and verify `alpha.std() > 0.1`

2. **Recovery Script Bug**: Shape mismatch or incorrect indexing
   - Script 1 uses `alpha_est[seen, 0]` for first dimension
   - For D=1, this should work correctly
   - Need to verify true_alpha shape matches expected format

3. **Data Mismatch**: True IRT parameters may not match dataset
   - Dataset may have been regenerated without updating true parameters
   - Check: metadata.json vs true_irt_parameters.json consistency

### Diagnostic Steps:

```python
# 1. Check true alpha shape
import json, numpy as np
with open('data/large_q1000_k5/true_irt_parameters.json') as f:
    irt = json.load(f)
true_alpha = np.array(irt['alpha'])
print(f"Shape: {true_alpha.shape}, ndim: {true_alpha.ndim}")
print(f"Std: {true_alpha.std():.4f}, Range: [{true_alpha.min():.4f}, {true_alpha.max():.4f}]")

# 2. Check estimated alpha from checkpoint
# (Requires loading model and running inference)

# 3. Add debug output to compute_all_recovery.py
print(f"DEBUG: alpha_est std: {alpha_est[seen, 0].std():.4f}")
print(f"DEBUG: true_alpha std: {true_alpha[seen].std():.4f}")
```

**Hypothesis**: If `alpha_est.std() < 0.01`, alpha has collapsed to constant (identifiability problem). This would explain low correlation despite correct beta recovery.

---

## PAPER IMPACT ASSESSMENT

### Invalid Claims:

**RQ1**: "Does coupling a DKVMN backbone with an ordinal, psychometrically constrained response head improve prediction?"

**Current Comparison**:
- DKVMN+Softmax (real) vs "DKVMN+Ordinal" (actually DeepGPCM) vs DEEP-GPCM (real)
- Comparing DeepGPCM to itself!

**Claim**: "DKVMN+Ordinal achieves parity with DEEP-GPCM in prediction, yet produces no trait estimates"

**Reality**: Both are DeepGPCM, both produce IRT parameters.

### Invalid Tables:

**Table 1 (Prediction)**:
```
Model              | K=5 QWK | Reality
-------------------|---------|------------------
DKVMN+Softmax      | 0.740   | DKVMNSoftmax ✓
DKVMN+Ordinal      | 0.759   | DeepGPCM ✗
DEEP-GPCM          | 0.759   | DeepGPCM ✓
```

"DKVMN+Ordinal" and "DEEP-GPCM" show identical performance because they're the same model.

**Table 2 (Recovery)**:
```
Model              | r_α   | r̄_β  | Reality
-------------------|-------|------|------------------
DKVMN+Ordinal      | 0.548 | 0.903| DeepGPCM ✗ (should be NaN)
DEEP-GPCM          | 0.746 | 0.912| DeepGPCM ✓
```

"DKVMN+Ordinal" shows valid IRT recovery because it's actually DeepGPCM.

### What Needs Fixing:

1. **Implement DKVMNOrdinal** class
2. **Re-run all baseline experiments** (~30 configs × 15 epochs)
3. **Update Table 1** with true baseline results
4. **Update Table 2** (DKVMN+Ordinal should show NaN)
5. **Revise RQ1 conclusions** based on new results
6. **Add model validation** to prevent future silent failures

---

## RECOMMENDATIONS

### Immediate (Before Paper Submission):

1. **Implement DKVMNOrdinal**:
```python
class DKVMNOrdinal(nn.Module):
    def __init__(self, ...):
        # Same DKVMN backbone as DeepGPCM
        self.memory = DKVMN(...)
        self.summary = nn.Sequential(...)
        # Different head - no IRT
        self.ordinal_head = nn.Linear(summary_dim, n_categories)

    def forward(self, questions, responses):
        # ... DKVMN operations ...
        logits = self.ordinal_head(summary)
        return {
            "logits": logits,
            "probs": torch.softmax(logits, dim=-1),
            "theta": torch.zeros(...),  # Dummy
            "alpha": torch.ones(...),   # Dummy
            "beta": torch.zeros(...),   # Dummy
        }
```

2. **Add Model Validation**:
```python
def build_model(cfg, device, n_students=0):
    MODEL_REGISTRY = {
        "deepgpcm": DeepGPCM,
        "dkvmn_softmax": DKVMNSoftmax,
        "dkvmn_ordinal": DKVMNOrdinal,
        "static_gpcm": StaticGPCM,
        "dynamic_gpcm": DynamicGPCM,
    }

    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type: '{model_type}'")

    model = MODEL_REGISTRY[model_type](...)
    log.info(f"Instantiated {model.__class__.__name__}")
    return model
```

3. **Re-run Experiments**:
```bash
# Re-train all DKVMN+Ordinal baselines
python scripts/run_baselines_comprehensive.sh

# Re-compute recovery
python scripts/compute_all_recovery.py

# Update paper tables
```

4. **Fix Recovery Scripts**:
- Standardize on Script 1
- Fix Scripts 2 & 3 or deprecate them
- Add unit tests for linking functions

5. **Investigate Static GPCM**:
- Check if alpha has collapsed
- Verify data consistency
- Add debug output to recovery script

### Short-term (Paper Revision):

6. **Update Paper**:
- Revise RQ1 claims
- Update Tables 1-2 with correct results
- Add footnote explaining baseline implementation
- Document recovery computation method

7. **Add Documentation**:
- Document which recovery script was used
- Add model architecture validation
- Create integration tests

### Long-term (Codebase Health):

8. **Improve Robustness**:
- Add model registry pattern
- Add checkpoint inspection tools
- Add architecture validation tests
- Improve error messages

---

## TIMELINE ESTIMATE

### Critical Path (Paper Submission):

- Day 1: Implement DKVMNOrdinal (4 hours)
- Day 1-2: Re-train baselines (20-30 hours compute)
- Day 3: Re-compute recovery (2 hours)
- Day 3: Investigate Static GPCM (4 hours)
- Day 4: Update paper (6 hours)
- Day 5: Final review and submission

**Total**: 5 days

### Risk Factors:

- **High Risk**: DKVMN+Ordinal may show worse performance than DEEP-GPCM
  - Would weaken RQ1 claims about "no tradeoff"
  - May require reframing as "small cost for interpretability"

- **Medium Risk**: Static GPCM issue may be unfixable
  - Could be fundamental identifiability problem
  - May need to remove from paper or add caveats

- **Low Risk**: Recovery script fix is straightforward
  - If bug confirmed, fix is simple
  - Re-computation is fast (no retraining)

---

## FILES CREATED

This investigation produced four detailed analysis documents:

1. **`RECOVERY_SCRIPT_ANALYSIS.md`** - Complete comparison of three recovery methods
2. **`ARCHIVED_RESULTS_ANALYSIS.md`** - Analysis of archived experiments and configs
3. **`MODEL_CALLING_INVESTIGATION.md`** - Complete trace of model instantiation
4. **`THOROUGH_INVESTIGATION_SUMMARY.md`** - This executive summary

All documents are in: `/c/Users/steph/documents/deep-mirt/kt-gpcm/`

---

## CONCLUSION

**The "DKVMN+Ordinal" baseline has never existed in this codebase.**

Every experiment labeled "dkvmn_ordinal" has actually run DeepGPCM due to a silent fallback in the model factory. This invalidates key paper claims about the contribution of IRT parameterization.

**The paper cannot be submitted without**:
1. Implementing the true DKVMN+Ordinal baseline
2. Re-running all baseline experiments
3. Updating tables and claims with correct results

**Estimated time to resolution**: 5 days for critical fixes + paper revision.

---

**Investigation Complete**
