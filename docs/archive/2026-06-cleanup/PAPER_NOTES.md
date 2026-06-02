# Paper Update Notes

Procedural record of code/model changes that affect paper text. Reference this when updating the paper after bulk retrain.

---

## Changes Already Applied to Paper (2026-03-29)

1. **Unconstrained beta is default** — methodology section rewritten, equations swapped
2. **"shared" renamed to "DKVMN+GPCM"** — throughout dynamic sections and baselines list
3. **Monotonic ablation removed** — from results, intro, roadmap
4. **"ordinal inductive bias" -> "cumulative logit structure"** — 4 locations
5. **Contribution list rewritten** — SIE + separated pathway elevated
6. **Dynamic sections merged** — single "Dynamic Ability Tracking" with 2 arguments
7. **New trajectory figures** — 2-row layout (examples + population error), 3 models

## Changes Pending for Paper (after bulk retrain)

### Naming changes (Phase B of cleanup)

These code renames need corresponding paper updates:

| Code change | Paper impact |
|-------------|-------------|
| `DeepGPCM` -> `MAGPCM` | No paper impact (paper already says MA-GPCM) |
| `"deepgpcm"` -> `"magpcm"` | No paper impact (config internals) |
| `"linear_decay"` -> `"onehot"` | Paper already says "One-hot" in Table 8 |
| `"separable"` -> `"learned"` | Paper already says "Learned" in Table 8 |
| `kt-gpcm/` -> `ma-irt/` | No paper impact |

**Conclusion: Phase B renames align code WITH paper. No paper text changes needed.**

### Dataset naming (Phase 1 of retrain)

| Old name | New name | Paper name |
|----------|----------|------------|
| `v2_q200_k4` | `static_q200_k4` | "Ordinal-Static" |
| `block_q200_k4` | `discrete_q200_k4` | Currently "Ordinal-Block", needs update to "Ordinal-Staircase" or "Ordinal-Discrete" |
| `rw_q200_k4` | `continuous_q200_k4` | "Ordinal-RW" is fine, or update to "Ordinal-Continuous" |

**Paper updates needed:**
- [ ] Ordinal-Block paragraph -> describe 3-staircase DGP instead of 2-block

### Loss simplification

| Change | Paper impact |
|--------|-------------|
| FocalLoss removed from code | Paper never mentions focal loss. No change needed. |
| CombinedLoss simplified to WOL-only | Paper already describes WOL only (Eq. 12). No change needed. |
| Regularization penalties removed | Paper mentions "dropout 0.05" in training setup but not reg penalties. Verify training setup paragraph. |

### Recovery table changes (Phase 6-8 of retrain)

| Table | Change |
|-------|--------|
| Tables 5,7 (dynamic recovery) | Replace GDA with traj RMSE + median traj r |
| All recovery tables | Numbers will change (new defaults, new data) |
| Table 9 (imbalance) | Three conditions (mild, severe, extreme) |

### Static GPCM theta recovery

evaluate.py needs MLE theta estimation for Static GPCM test students. The old estimate_theta_eap.py was archived. This needs to be reimplemented in evaluate.py before bulk retrain. Without it, GPCM(SGD) r_theta will show -0.006 instead of ~0.967.

---

## Model Configuration Summary (for paper's Training Setup section)

After cleanup, the actual training config is:

| Parameter | Value | Paper reference |
|-----------|-------|----------------|
| Loss | WOL only (weighted_ordinal_weight=1.0) | Eq. 12 |
| Batch size | 64 | Section 4.1.4 |
| Optimizer | Adam, lr=1e-3 | Section 4.1.4 |
| LR scheduler | ReduceLROnPlateau, patience=5, factor=0.8 | Section 4.1.4 |
| Grad clipping | max_norm=1.0 | (not mentioned in paper) |
| Epochs | 30 (MA-GPCM, DKVMN+Softmax), 50 (Dynamic GPCM), 150 (Static GPCM) | (not mentioned in paper) |
| Embedding | StaticItem (default for MA-GPCM) | Section 3.2 |
| Thresholds | Unconstrained (no monotonic enforcement) | Section 3.4 |
| Separated theta | True (MA-GPCM), False (DKVMN+GPCM) | Section 3.4 |
| Memory slots | 50 | Section 4.1.4 |
| Key/value dim | 64/64 | Section 4.1.4 |

**Note:** Paper says "dropout 0.05" but code uses `dropout_rate: 0.0` in all active configs. This discrepancy should be resolved (either add dropout or update paper).
