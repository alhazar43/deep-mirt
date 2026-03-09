# Paper Revisions Summary

## Changes Made

### 1. RQ1 (Ordinal Prediction) - Lines 595-620
**Before:** Mentioned alpha recovery (r_α = 0.707) in RQ1 results, mixed prediction and recovery discussion.

**After:**
- Focused entirely on prediction performance from Table 1
- Emphasized DEEP-GPCM beating all baselines across all K values
- Highlighted that the gap widens as K increases (0.592 vs 0.578 at K=3, 0.791 vs 0.773 at K=6)
- Explained that psychometric structure improves prediction without trading off performance
- Removed all mentions of parameter recovery (saved for RQ3)

### 2. RQ2 (Learner State Dynamics) - Lines 647-658
**Before:** Brief description that GPCM head preserves dynamics, Dynamic GPCM remains flat.

**After:**
- Detailed explanation of what the figure shows: four learner archetypes (high/low ability, improving/declining)
- Three key findings:
  1. DEEP-GPCM and DKVMN+Softmax produce identical trajectories (GPCM head preserves dynamics)
  2. DEEP-GPCM successfully differentiates archetypes (high ≈ 1.5, low ≈ -1.5, clear growth/decline trends)
  3. Dynamic GPCM fails to differentiate (all trajectories flat near 0)
- Explained why this matters: distributed memory necessary for learner-specific state representations
- Updated figure caption to be more descriptive

### 3. IRT Analysis Setup - Lines 585-592
**Before:** Brief mention of normalization without explanation.

**After:**
- Detailed explanation of IRT linking transformations
- Z-score normalization for θ and β: (x - μ) / σ
- Log-transformation for α: exp((log α - μ_log) / σ_log · 0.3)
- Explained why: removes arbitrary scale/location differences while preserving correlation structure
- Noted this is standard practice in IRT parameter recovery studies

### 4. RQ3 (Parameter Recovery) - Lines 659-680
**Before:** Did not explain why Static/Dynamic GPCM have lower recovery.

**After:**
- Added paragraph explaining that lower recovery is NOT deliberate suppression
- Static/Dynamic GPCM are architecturally optimized for prediction:
  - Static: Fixed embeddings prioritize stable associations over dynamic tracking
  - Dynamic: Recurrent updates blend ability and item signals
- Both sacrifice parameter identifiability for prediction performance
- DEEP-GPCM demonstrates that distributed memory enables BOTH:
  - Strong prediction (Table 1)
  - High parameter recovery (r_θ = 0.94)
- Key insight: DKVMN architecture naturally separates item characteristics (key memory) from learner state (value memory)

### 5. RQ4 (Scalability) - Lines 707-735
**Before:** Simple statement that SIE achieves best recovery across scales.

**After:**
- Emphasized interaction between encoding strategy and item bank size
- At Q=200: All encodings competitive (differences modest)
- At Q=2000: Context-aware encoding critical (SIE achieves 0.856)
- Noted LinDecay erratic behavior:
  - Q=1000: r_α = 0.198 (very poor)
  - Q=2000: r_α = 0.802 (excellent)
  - Suggests Kronecker encoding requires sufficient item diversity
- Key finding: Context importance increases with scale
- SIE tops at larger Q, indicating frozen random projections maintain distinctiveness
- Theta recovery excellent across all encodings (r_θ > 0.93)
- Prediction quality stable (QWK varies < 0.02)

### 6. Removed Content

**Removed from Introduction (line 227):**
- RQ5 (Ecological Validity) about proxy-ordinality

**Removed from Datasets (lines 546-548):**
- Proxy-Ordinality paragraph

**Removed from Results (lines 779-823):**
- Sequence length sensitivity section (paragraph + table)
- Proxy-Ordinality on Binary Benchmarks (RQ5) section (entire subsection + table + pending box)

## Summary Statistics

- **Pages:** 23 (down from 23, content reorganized)
- **File size:** 629 KB
- **Research questions:** 4 (down from 5)
- **Tables removed:** 2 (sequence length, proxy-ordinality)
- **Sections removed:** 2 (sequence length paragraph, RQ5 subsection)

## Key Improvements

1. **RQ1:** Now focuses purely on prediction performance, making it clear DEEP-GPCM beats all baselines
2. **RQ2:** Provides detailed interpretation of trajectory figure, explaining what patterns mean
3. **IRT Setup:** Explains normalization procedure clearly for reproducibility
4. **RQ3:** Addresses potential concern about "making baselines look bad" by explaining architectural tradeoffs
5. **RQ4:** Emphasizes context importance at scale, addresses LinDecay anomaly
6. **Streamlined:** Removed incomplete/pending sections (RQ5, sequence length)

## Compilation Status

✅ Paper compiles successfully
✅ All references intact
✅ All figures referenced correctly
✅ No orphaned labels or citations
