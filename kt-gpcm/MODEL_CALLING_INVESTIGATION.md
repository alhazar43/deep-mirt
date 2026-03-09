# Model Calling Investigation - Complete Trace

**Investigation Date**: 2026-03-08
**Focus**: How models are instantiated and which architecture actually runs

---

## EXECUTIVE SUMMARY

**CONFIRMED**: The "dkvmn_ordinal" model type does NOT exist. All experiments with `model_type: "dkvmn_ordinal"` silently fall back to `DeepGPCM` due to the else clause in `build_model()`.

**IMPACT**: Every "DKVMN+Ordinal" experiment in the current codebase is actually running DeepGPCM with full IRT parameterization.

---

## TRAINING PIPELINE TRACE

### Entry Point: `scripts/train.py`

```python
# Line 91-114
def main() -> None:
    parser = argparse.ArgumentParser(description="Train DeepGPCM model.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    # ...
    args = parser.parse_args()

    cfg = load_config(args.config)  # Load YAML config
    # ...
    model = build_model(cfg, device, n_students=data_mgr.n_students)  # ← KEY CALL
```

### Model Factory: `build_model()`

```python
# Lines 62-75
def build_model(cfg, device: torch.device, n_students: int = 0) -> nn.Module:
    # Extract all model parameters EXCEPT model_type
    model_kwargs = {k: v for k, v in vars(cfg.model).items() if k != "model_type"}

    # Get model_type with DEFAULT fallback
    model_type = getattr(cfg.model, "model_type", "deepgpcm")  # ← DEFAULT = "deepgpcm"

    # EXPLICIT CASES
    if model_type == "dkvmn_softmax":
        model = DKVMNSoftmax(**model_kwargs)
    elif model_type == "static_gpcm":
        model = StaticGPCM(n_students=n_students, **model_kwargs)
        model._model_type = "static_gpcm"
    elif model_type == "dynamic_gpcm":
        model = DynamicGPCM(n_students=n_students, **model_kwargs)
        model._model_type = "dynamic_gpcm"
    else:
        # SILENT FALLBACK - NO ERROR, NO WARNING
        model = DeepGPCM(**model_kwargs)  # ← CATCHES EVERYTHING ELSE

    return model.to(device)
```

**CRITICAL FLAW**: The `else` clause catches:
- `model_type = "deepgpcm"` (intended)
- `model_type = "dkvmn_ordinal"` (BUG - should error)
- `model_type = "foobar"` (BUG - should error)
- Missing `model_type` field (defaults to "deepgpcm")

**NO VALIDATION** - Any typo or non-existent model type silently becomes DeepGPCM.

---

## CONFIG GENERATION: Bulk Training Script

### Script: `scripts/run_baselines_comprehensive.sh`

```bash
# Lines 82-94
BASELINES=("static_gpcm" "dynamic_gpcm" "dkvmn_softmax" "dkvmn_ordinal")

for DATASET in "${DATASETS[@]}"; do
    Q=$(echo $DATASET | sed 's/large_q\([0-9]*\)_k.*/\1/')
    K=$(echo $DATASET | sed 's/large_q[0-9]*_k\([0-9]*\)/\1/')

    for BASELINE in "${BASELINES[@]}"; do
        EXP_NAME="${DATASET}_${BASELINE}"
        echo "Creating config: ${EXP_NAME}"
        create_baseline_config ${EXP_NAME} ${Q} ${K} ${DATASET} ${BASELINE}
    done
done
```

### Config Template: `create_baseline_config()`

```bash
# Lines 29-79
create_baseline_config() {
    local EXP_NAME=$1
    local Q=$2
    local K=$3
    local DATASET=$4
    local MODEL_TYPE=$5  # ← "dkvmn_ordinal" passed here

    cat > configs/baselines/${EXP_NAME}.yaml <<EOF
base:
  experiment_name: "${EXP_NAME}"
  device: "cuda"
  seed: 42

model:
  model_type: "${MODEL_TYPE}"  # ← "dkvmn_ordinal" written to config
  n_questions: ${Q}
  n_categories: ${K}
  n_traits: 1
  memory_size: 50
  key_dim: 64
  value_dim: 64
  summary_dim: 50
  embedding_type: "static_item"
  ability_scale: 1.0
  dropout_rate: 0.0
  memory_add_activation: "tanh"
  init_value_memory: false
  monotonic_betas: true  # ← IRT-specific parameter!

training:
  epochs: 15
  batch_size: 64
  lr: 0.001
  grad_clip: 1.0
  focal_weight: 0.5
  weighted_ordinal_weight: 0.5
  ordinal_penalty: 0.5
  # ... more training params ...

data:
  data_dir: "data"
  dataset_name: "${DATASET}"
  train_split: 0.8
  min_seq_len: 10
EOF
}
```

**OBSERVATION**: The config includes `monotonic_betas: true`, which is a **DeepGPCM-specific parameter**. This suggests the config was designed for DeepGPCM, not a separate ordinal baseline.

---

## LOSS CONFIGURATION

### Loss Weights in Generated Configs:

```yaml
training:
  focal_weight: 0.5
  weighted_ordinal_weight: 0.5
  ordinal_penalty: 0.5
```

### Loss Computation: `CombinedLoss`

From `src/kt_gpcm/training/losses.py`:

```python
# Lines 19-23 (docstring)
"""
Default loss recipe (Architecture Decision A6):
    L = 0.5 * FocalLoss(logits, targets)
      + 0.5 * WeightedOrdinalLoss(logits, targets)

``WeightedOrdinalLoss`` internally uses ``ordinal_penalty=0.5``.
"""
```

**Effective Loss**:
```
L_total = 0.5 * L_focal + 0.5 * L_weighted_ordinal
L_weighted_ordinal = L_CE + 0.5 * L_ordinal_penalty
```

**This is the SAME loss used for DeepGPCM!**

---

## WHAT ACTUALLY RUNS

### Execution Flow for "dkvmn_ordinal":

1. **Config Created**: `configs/baselines/large_q1000_k5_dkvmn_ordinal.yaml`
   ```yaml
   model:
     model_type: "dkvmn_ordinal"  # Non-existent model
   ```

2. **Training Invoked**:
   ```bash
   python scripts/train.py --config configs/baselines/large_q1000_k5_dkvmn_ordinal.yaml
   ```

3. **Config Loaded**:
   ```python
   cfg = load_config(args.config)
   # cfg.model.model_type = "dkvmn_ordinal"
   ```

4. **Model Built**:
   ```python
   model_type = getattr(cfg.model, "model_type", "deepgpcm")
   # model_type = "dkvmn_ordinal"

   if model_type == "dkvmn_softmax":
       # NO - skip
   elif model_type == "static_gpcm":
       # NO - skip
   elif model_type == "dynamic_gpcm":
       # NO - skip
   else:
       # YES - FALLBACK
       model = DeepGPCM(**model_kwargs)  # ← INSTANTIATES DeepGPCM
   ```

5. **Model Architecture**:
   ```python
   DeepGPCM(
       n_questions=1000,
       n_categories=5,
       n_traits=1,
       memory_size=50,
       key_dim=64,
       value_dim=64,
       summary_dim=50,
       embedding_type="static_item",
       monotonic_betas=True,  # ← IRT parameter
       # ... all other DeepGPCM params ...
   )
   ```

6. **Forward Pass**:
   ```python
   # DeepGPCM.forward() returns:
   {
       "theta": theta,      # (B, S, D) - student ability
       "alpha": alpha,      # (B, S, D) - item discrimination
       "beta": beta,        # (B, S, K-1) - item thresholds
       "logits": logits,    # (B, S, K) - GPCM logits
       "probs": probs,      # (B, S, K) - GPCM probabilities
   }
   ```

7. **Loss Computed**:
   ```python
   loss = 0.5 * focal_loss(logits, targets) + 0.5 * weighted_ordinal_loss(logits, targets)
   ```

8. **Checkpoint Saved**:
   ```python
   torch.save({
       "model": model.state_dict(),  # Contains IRT parameters
       "epoch": epoch,
       # ...
   }, "outputs/large_q1000_k5_dkvmn_ordinal/best.pt")
   ```

9. **Recovery Computed**:
   ```python
   # compute_all_recovery.py extracts:
   alpha = out["alpha"]  # (B, S, D) - EXISTS because it's DeepGPCM
   beta = out["beta"]    # (B, S, K-1) - EXISTS because it's DeepGPCM

   # Computes correlations:
   r_alpha = 0.548  # Valid correlation
   r_beta_mean = 0.903  # Valid correlation
   ```

**RESULT**: "dkvmn_ordinal" produces valid IRT recovery metrics because it's actually DeepGPCM.

---

## COMPARISON: What SHOULD Happen

### Intended "DKVMN+Ordinal" Architecture:

```python
class DKVMNOrdinal(nn.Module):
    """DKVMN backbone with linear ordinal head (no IRT)."""

    def __init__(self, n_questions, n_categories, ...):
        super().__init__()

        # SAME DKVMN backbone as DeepGPCM
        self.q_embed = nn.Embedding(n_questions + 1, key_dim, padding_idx=0)
        self.embedding = LinearDecayEmbedding(n_questions, n_categories)
        self.value_proj = nn.Linear(self.embedding.output_dim, value_dim)
        self.memory = DKVMN(...)
        self.summary = nn.Sequential(...)

        # DIFFERENT HEAD - no IRT extraction
        self.ordinal_head = nn.Linear(summary_dim, n_categories)

    def forward(self, questions, responses):
        # ... DKVMN memory operations (IDENTICAL to DeepGPCM) ...

        # DIFFERENT - direct linear projection
        logits = self.ordinal_head(summary)  # (B, S, K)

        # Return dummy IRT fields for compatibility
        return {
            "logits": logits,
            "probs": torch.softmax(logits, dim=-1),
            "theta": torch.zeros(B, S, 1, device=logits.device),   # NO ability
            "alpha": torch.ones(B, S, 1, device=logits.device),    # NO discrimination
            "beta": torch.zeros(B, S, K-1, device=logits.device),  # NO thresholds
        }
```

### Expected Recovery for True DKVMN+Ordinal:

```python
# compute_all_recovery.py would extract:
alpha = out["alpha"]  # (B, S, 1) - all ones (dummy)
beta = out["beta"]    # (B, S, K-1) - all zeros (dummy)

# After accumulation:
alpha_est = np.ones((Q, 1))  # Constant
beta_est = np.zeros((Q, K-1))  # Constant

# Correlation with true parameters:
r_alpha = NaN or 0.0  # No variation
r_beta_mean = NaN or 0.0  # No variation
```

**This is what the paper CLAIMS but never achieved.**

---

## ORDINAL LOSS APPLICATION

### Question: Is ordinal loss actually applied?

**YES** - for ALL models including DKVMNSoftmax:

```python
# scripts/train.py, lines 156-165
loss_fn = CombinedLoss(
    n_categories=cfg.model.n_categories,
    class_weights=class_weights,
    focal_weight=t.focal_weight,
    weighted_ordinal_weight=t.weighted_ordinal_weight,
    ordinal_penalty=t.ordinal_penalty,
)
```

The loss is configured from `cfg.training`, not from model type.

### DKVMNSoftmax with Ordinal Loss:

```yaml
# configs/baselines/large_q1000_k5_dkvmn_softmax.yaml
model:
  model_type: "dkvmn_softmax"  # ← Correct model type

training:
  focal_weight: 0.5
  weighted_ordinal_weight: 0.5
  ordinal_penalty: 0.5  # ← ORDINAL LOSS APPLIED
```

**DKVMNSoftmax ALSO uses ordinal loss!**

This means:
- DKVMNSoftmax = DKVMN + softmax head + ordinal loss
- "DKVMN+Ordinal" (intended) = DKVMN + linear head + ordinal loss
- "DKVMN+Ordinal" (actual) = DKVMN + GPCM head + ordinal loss = DeepGPCM

**The only difference between DKVMNSoftmax and intended DKVMN+Ordinal is the head architecture (softmax vs linear), NOT the loss.**

---

## MODEL REGISTRY - What Exists

### Implemented Models:

1. **`DeepGPCM`** (`src/kt_gpcm/models/kt_gpcm.py`)
   - DKVMN backbone
   - IRT parameter extractor
   - GPCM logit layer
   - Returns: theta, alpha, beta, logits, probs

2. **`DKVMNSoftmax`** (`src/kt_gpcm/models/dkvmn_softmax.py`)
   - DKVMN backbone
   - Linear classifier (K-way softmax)
   - Returns: logits, probs, dummy IRT fields

3. **`StaticGPCM`** (`src/kt_gpcm/models/static_gpcm.py`)
   - No DKVMN (pure IRT)
   - Embedding tables for theta, alpha, beta
   - GPCM logit layer
   - Returns: theta, alpha, beta, logits, probs

4. **`DynamicGPCM`** (`src/kt_gpcm/models/dynamic_gpcm.py`)
   - No DKVMN (recurrent IRT)
   - GRU-based theta update
   - Embedding tables for alpha, beta
   - GPCM logit layer
   - Returns: theta, alpha, beta, logits, probs

### Missing Models:

5. **`DKVMNOrdinal`** - DOES NOT EXIST
   - Should have: DKVMN backbone + linear ordinal head
   - Should return: logits, probs, dummy IRT fields
   - **This is the model the paper describes but was never implemented**

---

## EVIDENCE SUMMARY

### 1. Config Files Show Non-Existent Model Type:
```yaml
# configs/baselines/large_q1000_k5_dkvmn_ordinal.yaml
model:
  model_type: "dkvmn_ordinal"  # ← Does not exist in codebase
```

### 2. Training Script Has Silent Fallback:
```python
# scripts/train.py, line 74
else:
    model = DeepGPCM(**model_kwargs)  # ← Catches "dkvmn_ordinal"
```

### 3. Checkpoint Contains IRT Parameters:
```bash
$ ls -lh outputs/large_q1000_k5_dkvmn_ordinal/best.pt
-rw-r--r-- 1 steph 197609 2.5M Mar  5 07:07 best.pt

# Size matches DeepGPCM (Q=1000), not DKVMNSoftmax
```

### 4. Recovery Metrics Show IRT Parameters:
```csv
large_q1000_k5_dkvmn_ordinal,0.548,0.903,"0.894,0.919,0.914,0.884"
```
If it were truly non-IRT, these would be NaN or 0.0.

### 5. Training Metrics Match DeepGPCM:
```
deepgpcm_k5_s42 (archived):           train_loss=1.2303
large_q1000_k5_dkvmn_ordinal (current): train_loss=1.2308
```
Nearly identical despite different Q (200 vs 1000).

### 6. Source Code Has No DKVMNOrdinal Class:
```bash
$ grep -r "class DKVMNOrdinal" src/
# No results
```

---

## ROOT CAUSE ANALYSIS

### How Did This Happen?

1. **Paper Design Phase**:
   - Authors planned "DKVMN+Ordinal" as a baseline
   - Described it in paper as "DKVMN with linear ordinal head"
   - Created config files with `model_type: "dkvmn_ordinal"`

2. **Implementation Phase**:
   - Implemented DeepGPCM, DKVMNSoftmax, StaticGPCM, DynamicGPCM
   - **Never implemented DKVMNOrdinal**
   - Training script has `else: DeepGPCM` fallback

3. **Experiment Phase**:
   - Ran bulk training script
   - "dkvmn_ordinal" configs silently ran DeepGPCM
   - No error, no warning, no validation

4. **Analysis Phase**:
   - Recovery script extracted IRT parameters
   - Produced valid correlations (because it's DeepGPCM)
   - Results looked reasonable, no red flags

5. **Paper Writing Phase**:
   - Reported "DKVMN+Ordinal" results
   - Claimed it produces no IRT parameters
   - **Never verified the actual model architecture**

### Why Wasn't It Caught?

- **No model validation**: Training script doesn't check if model_type is valid
- **No architecture logging**: Doesn't print model class name
- **No checkpoint inspection**: Never verified what's in the saved model
- **No integration tests**: No tests that verify model_type → architecture mapping
- **Silent fallback**: `else` clause masks the error

---

## RECOMMENDATIONS

### Immediate Fix:

```python
# scripts/train.py
def build_model(cfg, device, n_students=0):
    model_kwargs = {k: v for k, v in vars(cfg.model).items() if k != "model_type"}
    model_type = getattr(cfg.model, "model_type", "deepgpcm")

    # EXPLICIT REGISTRY
    MODEL_REGISTRY = {
        "deepgpcm": DeepGPCM,
        "dkvmn_softmax": DKVMNSoftmax,
        "dkvmn_ordinal": DKVMNOrdinal,  # ← Must implement first
        "static_gpcm": StaticGPCM,
        "dynamic_gpcm": DynamicGPCM,
    }

    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. "
            f"Valid options: {list(MODEL_REGISTRY.keys())}"
        )

    model_class = MODEL_REGISTRY[model_type]

    # Special handling for student-indexed models
    if model_type in ("static_gpcm", "dynamic_gpcm"):
        model = model_class(n_students=n_students, **model_kwargs)
        model._model_type = model_type
    else:
        model = model_class(**model_kwargs)

    # LOG ARCHITECTURE
    log.info(f"Instantiated {model.__class__.__name__} (model_type={model_type})")

    return model.to(device)
```

### Validation:

```python
# At end of main()
assert model.__class__.__name__ in ["DeepGPCM", "DKVMNSoftmax", "DKVMNOrdinal", "StaticGPCM", "DynamicGPCM"], \
    f"Unexpected model class: {model.__class__.__name__}"

if cfg.model.model_type == "dkvmn_ordinal":
    assert not hasattr(model, "irt"), "DKVMN+Ordinal should not have IRT extractor"
elif cfg.model.model_type == "deepgpcm":
    assert hasattr(model, "irt"), "DEEP-GPCM must have IRT extractor"
```

---

## CONCLUSION

**The "dkvmn_ordinal" model type does NOT exist.**

Every experiment with this model_type silently runs DeepGPCM due to the fallback in `build_model()`. This invalidates:
- RQ1 comparison (comparing DeepGPCM to itself)
- Table 1 results (DKVMN+Ordinal is actually DeepGPCM)
- Table 2 recovery (shows IRT parameters because it's DeepGPCM)
- Paper claims about ordinal structure without IRT

**Action Required**:
1. Implement `DKVMNOrdinal` class
2. Add model validation to prevent silent fallbacks
3. Re-run all baseline experiments
4. Update paper with correct results

**Estimated Impact**: 3-5 days to implement, re-train, and revise paper.
