# Paper Edits Summary: DKVMN+Ordinal Removal

## Changes Made

All references to "DKVMN+Ordinal" as a separate baseline have been removed from paper.tex.

### Rationale

Investigation revealed that the archived "DKVMN+Ordinal" experiments were actually **DeepGPCM with ordinal-only loss** (focal_weight=0.0, weighted_ordinal_weight=1.0), not a separate architecture. Evidence:

1. **Checkpoint analysis**: Identical architecture to DeepGPCM (23 keys, all IRT parameters present)
2. **Config analysis**: No `model_type` field in archived configs, defaults to DeepGPCM via fallback
3. **Training script**: Line 73-74 in `scripts/train.py` shows `else: model = DeepGPCM(**model_kwargs)`
4. **Current configs**: New baseline configs specify `model_type: "dkvmn_ordinal"` which doesn't exist in codebase

This was a **loss ablation study**, not a separate baseline model.

### Specific Edits

1. **Section: Models and Baselines (line 558)**
   - Changed from "four baselines" to "three baselines"
   - Removed DKVMN+Ordinal bullet point

2. **Section: Ordinal Prediction Results (line 598)**
   - Removed claim that "DKVMN+Ordinal achieves parity with DEEP-GPCM"
   - Updated to show DEEP-GPCM outperforms DKVMN+Softmax
   - Added specific performance comparison at K=6

3. **Table 1 caption (line 602)**
   - Changed from "matches DKVMN+Ordinal" to "outperforms DKVMN+Softmax"

4. **Table 1 content (lines 614-615)**
   - Removed DKVMN+Ordinal row from results table
   - Now shows: Static GPCM, Dynamic GPCM, DKVMN+Softmax, then DEEP-GPCM

5. **Section: Binary Compatibility (line 626)**
   - Removed reference to "matches DKVMN+Ordinal on Synthetic-Ordinal (K=2)"
   - Simplified to focus on ASSIST2015 and Synthetic-5 benchmarks

6. **Table 2 (lines 628-644)**
   - Removed Synthetic-Ordinal columns (which only had DKVMN+Ordinal data)
   - Removed DKVMN+Ordinal row
   - Now shows only ASSIST2015 and Synthetic-5 results

7. **Table 3 caption (line 668)**
   - Removed phrase "identical to DKVMN+Ordinal in estimates but uniquely exposes them"
   - Simplified to focus on DEEP-GPCM's recovery capabilities

### Remaining Valid Uses

The following uses of "DKVMN" with "ordinal" remain and are correct:
- "DKVMN backbone with ordinal encoding" (architectural description)
- "ordinal interaction encoding, DKVMN memory update" (pipeline stages)
- "DKVMN backbone with ordinal loss" (DEEP-GPCM description)
- "ordinal categories" (response format)

These refer to ordinal structure in the data/loss, not to a separate "DKVMN+Ordinal" baseline.

## Impact on Paper Claims

### Before
- Claimed DKVMN+Ordinal as a separate baseline
- Showed DEEP-GPCM "matches" DKVMN+Ordinal in prediction
- Implied interpretability comes at no cost by matching a non-IRT baseline

### After
- Three baselines: DKVMN+Softmax, Static GPCM, Dynamic GPCM
- Shows DEEP-GPCM **outperforms** DKVMN+Softmax
- Demonstrates that IRT parameterization improves both prediction and interpretability

The revised narrative is stronger: DEEP-GPCM doesn't just match baselines while adding interpretability—it actually improves prediction performance.

## Files Modified

- `/c/Users/steph/documents/deep-mirt/paper.tex`

## Related Work

- Recovery computation scripts cleaned up (buggy scripts removed)
- Recovery recomputation running in background
- Plot regeneration script created for all Q×K combinations
