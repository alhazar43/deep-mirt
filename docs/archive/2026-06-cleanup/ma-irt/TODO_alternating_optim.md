# TODO: Alternating Optimization for Static GPCM Baseline

## Problem
Joint SGD on theta embeddings + item parameters (alpha, beta) leads to poor alpha recovery (r ≈ 0.35) because of alpha-theta compensation. Theta embeddings absorb item-level variance and vice versa. MLE theta with fixed items gives r_theta ≈ 0.97, confirming items are functionally adequate but individually poorly identified.

## Proposed Fix
Implement alternating optimization that approximates EM:
1. **Phase A (M-step):** Freeze theta embeddings, update alpha/beta only for N epochs
2. **Phase B (E-step):** Freeze alpha/beta, update theta embeddings only for M epochs
3. Repeat alternation until convergence

This separates the person estimation from item calibration, which is standard in IRT software (mirt uses EM, flexMIRT uses MH-RM).

## Implementation
In `trainer.py` or a new `alternating_trainer.py`:
- Accept `alternating_phases` config parameter
- Each phase freezes one set of parameters via `requires_grad_(False)`
- Log which phase is active

## Expected Outcome
- Alpha recovery should improve significantly (from r ≈ 0.35 to potentially r > 0.7)
- Theta recovery should remain high
- Beta recovery should remain near-perfect

## Status
**Deferred.** Currently using MLE theta for test students as a fair comparison. The mirt R package is also being evaluated as an alternative gold-standard baseline.
