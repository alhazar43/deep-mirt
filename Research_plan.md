# Research plan: Multidimensional DKVMN with polytomous IRT heads (psychometrics-facing)

**Goal:** Position the model as an *amortized dynamic MIRT measurement model* for polytomous responses—usable for psychometric research, not only prediction.

---

## 1. Problem statement and motivation

Knowledge Tracing (KT) models such as DKVMN and AKT achieve strong predictive performance, but their evaluation is typically classification-centric (e.g., AUC/ACC/QWK). For psychometric research with **polytomous items** (partial credit / graded responses) and **multidimensional constructs**, the primary concerns shift to:

- probabilistic **calibration** and **model fit**,
- interpretable **latent trait structure** (dimensions, loadings, stability),
- reliability / information and uncertainty,
- invariance and robustness under multi-KC items and data sparsity.

This work extends DKVMN to **natively accept multidimensional latent traits** and supports **polytomous IRT heads** (e.g., GPCM/GRM), enabling psychometric-style analysis and reporting.

---

## 2. Core model (high-level)

### 2.1 Dynamic latent trait view
- Student state is a time-varying latent trait vector: \(\theta_t \in \mathbb{R}^D\).
- DKVMN-style memory provides an amortized update rule for \(\theta_t\) given interactions \((q_t, x_t)\).

### 2.2 Multi-KC / multi-dimensional structure
- Item-to-slot attention \(w_t\) acts as a soft multi-KC mapping.
- Slot/value memory stores dimension-relevant components of \(\theta_t\).
- Optional anchoring with a Q-matrix prior when available.

### 2.3 Polytomous measurement model
Replace binary heads with ordinal/polytomous IRT heads:
- **GPCM** (partial credit) or **GRM** (graded response) as primary.
- Optional NRM for unordered categories.

Training uses maximum likelihood (cross-entropy / NLL) under the chosen head.

---

## 3. Primary research questions (RQs)

### RQ1 — Measurement validity beyond prediction
**RQ1:** Does the model behave like a *well-calibrated probabilistic measurement model* for graded responses?

Focus: calibration and proper scoring rules (psychometric fit > agreement).

### RQ2 — Interpretable multidimensionality and identifiability
**RQ2:** When \(D\) increases (and performance improves), does the learned multidimensional structure remain *stable and interpretable*, or does it reflect capacity/overparameterization?

Focus: dimension stability across seeds/folds, identifiability constraints, and capacity–measurement trade-offs.

### RQ3 — Multi-KC items as MIRT (loadings / Q-structure)
**RQ3:** Does item-to-slot attention correspond to a meaningful loading/Q-structure, especially for multi-KC items?

Focus: attention ↔ discrimination support; attention ↔ Q-matrix alignment (if labels exist); internal consistency if labels do not exist.

### RQ4 — Dynamic measurement and trajectories
**RQ4:** Does the evolving \(\theta_t\) yield coherent trajectories and uncertainty that align with psychometric expectations (learning, forgetting, evidence accumulation)?

Focus: trajectory interpretability, uncertainty behavior, time-conditional calibration.

---

## 4. Evaluation plan (what matters in polytomous psychometrics)

### 4.1 Metrics (prioritize these over QWK)
**Core probabilistic / ordinal metrics**
- Negative log-likelihood (NLL) / cross-entropy
- Ranked Probability Score (RPS) for ordinal outcomes
- Ordinal calibration error (e.g., ECE on cumulative thresholds \(P(X\ge k)\))
- Optional: Earth Mover’s Distance (EMD) between predicted and empirical category distributions

**Secondary agreement metric**
- Quadratic Weighted Kappa (QWK) (report, but do not center contributions on it)

### 4.2 Calibration and fit checks
- Reliability diagrams for **cumulative thresholds** \(P(X\ge k)\), \(k=1..K-1\)
- Predicted vs observed category proportions (overall + per item)
- Posterior predictive checks: replicate score/category distributions

### 4.3 Structure / interpretability / stability
- Dimension/slot stability across seeds (matching similarity matrix)
- Attention entropy and sparsity vs item discrimination strength
- If Q-matrix/KC labels exist: attention–Q alignment (precision/recall for top-attended slots)

### 4.4 Dynamic behavior
- \(\theta_{t,d}\) trajectories for representative students
- Evidence markers: show how high-information interactions shift predicted distributions
- If uncertainty is approximated (ensembles/MC dropout): uncertainty vs sequence length and novelty

---

## 5. Baselines (minimum credible set)

### Psychometric baselines
- Static polytomous IRT: GPCM/GRM (uni- and multi-dimensional if feasible)
- Simple dynamic IRT baseline: random-walk \(\theta_t\) with GPCM/GRM head (even approximate)

### KT baselines (polytomous-adapted)
- DKVMN with multinomial/ordinal head (no explicit MIRT structure)
- AKT with multinomial/ordinal head
- Any strong public KT baselines you can reliably reproduce under the same split

---

## 6. What to present (figures/tables that are actually meaningful)

### 6.1 Core tables (must-have)
**Table A: Ordinal probabilistic performance**
- NLL, RPS, calibration error (threshold-based), EMD (optional), QWK (secondary)

**Table B: Measurement structure**
- stability index across seeds, attention sparsity/entropy, attention–Q alignment (if available)

**Table C: Baseline comparisons**
- Psychometric baselines + KT baselines with the same splits

### 6.2 Core figures (high-yield set)
1. **Ordinal calibration plots** (per-threshold \(P(X\ge k)\))
2. **D-scaling curves**: performance (NLL/RPS/QWK) + stability/interpretability index vs \(D\)
3. **Polytomous item characteristic curves** (model-implied \(P(X=k\mid \theta)\) for selected items)
4. **Multi-KC loading visualization**: top-attended slots + discrimination magnitudes for selected multi-KC items
5. **Student trajectories**: \(\theta_{t,d}\) over time with evidence markers
6. *(Optional)* Concept interaction map if using concept-to-concept refinement attention

---

## 7. How to make attention plots count (psychometric standard)

Raw attention heatmaps are not sufficient. Tie attention to *measurement evidence*:

1. **Attention ↔ discrimination**: dimensions with high attention should have high discrimination magnitude for that item.
2. **Attention ↔ differential functioning**: stratify students by \(\theta_d\) quantiles and show category frequencies shift primarily for attended dimensions.
3. **Attention ↔ Q-matrix**: quantify alignment if labels exist (not just a visual).

---

## 8. Suggested paper contributions (tight and defensible)

1. **Model:** A multidimensional DKVMN that functions as an amortized dynamic MIRT model for polytomous outcomes.
2. **Bridge:** Item-to-slot attention provides a learned multi-KC structure that maps to MIRT loadings/discriminations.
3. **Psychometric evaluation:** Demonstrate calibration, fit, structure stability, and dynamic interpretability beyond prediction.

---

## 9. Practical next steps (execution checklist)

1. Decide primary polytomous head: **GPCM vs GRM**
2. Confirm whether a Q-matrix/KC labels exist (even noisy)
3. Run a **D-sweep** with multiple seeds and collect:
   - NLL/RPS/calibration, QWK
   - stability index and attention statistics
4. Produce the 5–6 core plots above
5. Write results around RQ1–RQ4 with psychometric language (fit, calibration, invariance, structure)

---

## 10. Deliverables

- A psychometrics-facing results package: calibration + fit + structure stability + dynamic trajectories
- A minimal reproducibility bundle: splits, config, evaluation scripts for NLL/RPS/calibration and stability analysis
- A narrative that centers on **measurement**, not only prediction
