# Paper Plan: DEEP-GPCM

## Target Venue
International Journal of Artificial Intelligence in Education (IJAIED)
- Education-first framing mandatory
- Real-world data strongly expected
- 20-35 pages Springer single-column
- 40-70 references typical (currently ~36, need ~10 more)
- Springer Nature LaTeX template required

## Framing: DEEP-MIRT as Framework, GPCM as Instantiation
- **DEEP-MIRT**: any differentiable observation model mapping (theta_t, k_t) -> P(r | theta_t, item)
- Alternative heads: GRM (cumulative logistic), NRM (unordered), 2PL/3PL (binary)
- GPCM chosen for: adjacent-category model, natural for rubric progressions, monotone thresholds
- Contributions must separate: (1) framework, (2) GPCM instantiation, (3) SIE encoding

### Missing framework substantiation
- No interface description paragraph in Methodology (what contract must a response head satisfy?)
- Architecture diagram is GPCM-specific, not framework-level — caption should note alternatives
- Contributions conflate framework and instantiation — rewrite to separate cleanly

## Key Messages
1. Implicit psychometric inference: neural networks recover IRT discrimination WITHOUT supervision
2. No interpretability-prediction tradeoff: DEEP-GPCM achieves both simultaneously
3. Architectural tradeoffs: threshold recovery gap (r_beta ~0.91 vs ~0.98) is a design consequence
4. Scalability with interpretability: SIE maintains recovery while reducing parameters by 80%
5. Dual purpose: predicts performance AND recovers interpretable parameters simultaneously

## Research Questions
- **RQ1**: Ordinal prediction (DEEP-GPCM vs baselines across K) — DONE
- **RQ2**: Learner state dynamics (trajectory archetypes) — DONE
- **RQ3**: Parameter recovery (alpha, beta, theta correlations) — DONE
- **RQ4**: Scalability (Q scaling, encoding strategies) — DONE
- **RQ5**: Class imbalance robustness — DONE

## Current Status
- 5 RQs complete with results
- 3 baselines: Static GPCM, Dynamic GPCM, DKVMN+Softmax
- Recovery: r_alpha > 0.70, r_beta > 0.80, r_theta > 0.92 for DEEP-GPCM
- Paper compiles, ~24 pages

## Open Weaknesses
- **Synthetic-only evaluation** — no real educational data yet (high rejection risk for IJAIED)
- **No standard deviations** — run >= 3 seeds per condition
- **Single simulation DGP** — vary alpha spread, beta spacing for stronger generalizability
- **Framework claim not fully substantiated** — needs interface description paragraph
- **Bibliography thin** — ~36 refs, target 40-70

## Writing Principles
- Lead with psychometric insight (what does this mean for measurement theory?)
- Explain theoretical significance (why does it matter?)
- Use numbers as supporting evidence (not the narrative)
- Connect to educational practice (what can practitioners do?)
