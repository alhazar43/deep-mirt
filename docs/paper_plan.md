# Paper plan: "Not All Parameters Learn Alike"

Venue: JEDM full article, **acmtrans class, two-column, ~12-15 pp** (the EDM Journal Track carries it to the
conference). Home: knowledge tracing, a **KT encoder plus an interpretable IRT decoder**. Do NOT use the term
"neural KT" (banned). IRT is the readable flavor, not the subject.

## Scope: only two things
- **(a) A modular tracer, benchmarked by recovery.** An interchangeable sequence encoder feeding an
  interpretable IRT decoder; both swap freely. The contribution here is the side-by-side **parameter-recovery
  benchmark across the encoder-by-decoder matrix**. Recovery is the yardstick; it shows the modularity is real
  and that the ordering below is not an artifact of one encoder or decoder.
- **(b) A finite-data, finite-budget law.** Trained on prediction loss alone, the readout parameters recover
  unevenly: the parameter that MULTIPLIES ability recovers worst, this is a multiplicative (scale-gauge) effect,
  and it is separable from a low-Fisher rate penalty.

## Thesis (one sentence)
A modular KT-encoder + IRT-decoder tracer, trained only on response prediction, recovers the coefficient that
multiplies ability worst, a finite-data multiplicative-coupling effect that decoupling fixes, distinct from a
low-Fisher rate penalty that a state-conditioned head fixes.

## The mechanism (the sharp point, worked out from the NRM control)
Look at where each item parameter sits relative to ability.
- GPCM logit `alpha*theta - alpha*beta`: the only bilinear term is `alpha*theta`.
- NRM logit `a_k*theta + c_k`: the only bilinear term is `a_k*theta`; `c_k` is additive.

Two gauges follow:
- **SCALE gauge (multiplicative):** `alpha*theta` / `a_k*theta` is invariant under `alpha -> alpha/s, theta -> s*theta`.
  Prediction loss pins only the product, so under finite data and a shared representation the split of scale between
  the slope and ability is under-determined. That is the slope-vs-ability trade-off. **Decoupling the representation
  is the fix.** The additive `c_k` never multiplies ability, so no scale coupling, no trade-off.
- **LOCATION gauge (additive):** `theta - beta`. Threshold identifiability; it breaks if the ability readout sees the
  current item (the ability-item coupling). Structural, does not vanish with data.

**NRM is the control that dissociates two things the deck was blurring:**
- a multiplicative coupling (the trade-off, common to GPCM `alpha` and NRM `a_k`, cured by decoupling), and
- a low-Fisher RATE penalty (only GPCM `alpha`, cured by the state-conditioned head).
`a_k` is multiplicative so it trades off with ability, but it is NOT low information, so it recovers at a normal rate
and the dynamic head does nothing for it. `alpha` is both. That is the paper's sharpest result.

## Story arc (the deck, told as a paper)
1. KT bolts interpretable IRT readouts on for interpretability, but are the parameters actually recovered?
2. A modular framework to ask cleanly: swap the encoder, swap the IRT decoder; benchmark by recovery. (a)
3. Recovery is uneven: the coefficient that multiplies ability comes back worst.
4. Why: the multiplicative coupling (scale gauge). Decoupling the representation escapes it.
5. It is a finite-data effect, the gap closes as data and training grow.
6. A separate, second penalty, low Fisher, lands only on GPCM `alpha`; a state-conditioned head is the targeted fix.
7. NRM is the control that separates the two (multiplicative-but-not-low-Fisher `a_k`).
8. Holds across the matrix and on real data (reliability). Practice: decouple always, state-condition only the
   low-Fisher parameter.

## Experiments
**(a) Recovery matrix.** {DKVMN, LSTM, transformer} x {GPCM, NRM, binary}, recovery side-by-side, seeds + CIs,
ordering-concordance across the matrix. (Salvage the old paper's recovery benchmark machinery.)
**(b) The finite-data law:**
- recovery follows the structure (the multiplicative coefficient is worst);
- finite-vs-asymptotic sweep (the gap vanishes with data and training budget);
- **multiplicative-vs-additive ablation (core, new):** hold Fisher information fixed and vary only whether a parameter
  enters multiplicatively or additively with ability; the trade-off should track multiplicative entry, not information.
  This is what turns "multiplicative" from best-explanation into necessary;
- embedding configuration is the lever (shared entangles, decoupling separates);
- the dynamic head (targeted fix for the low-Fisher `alpha`);
- the NRM control (the dissociation);
- real data (recovery where possible, reliability otherwise).

## Salvage from the old paper (overleaf-sync/main_magpcm_ijaied.tex, archived, do not discard)
KEEP: the recovery-benchmark methodology, metrics, baselines, the architecture diagram, and the real-data
(ASSISTments) recovery, all for the (a) matrix. DROP: the ordinal-KT framing, the separated-pathway-as-hero,
weighted-ordinal-loss as a contribution, imbalance robustness, binary compatibility, the deployment discussion.

## Related work (brief, not three rings)
Converse (AIED 2021) and beta4-IRT (2023) as the symptom-vs-mechanism neighbors (they saw discrimination recover
badly, diagnosed as a data quirk or a sign symmetry; we give the multiplicative-coupling mechanism). Ghosh option
tracing (AIED 2021) as the NRM-readout precedent. The anti-variational point: prior neural-IRT recovers via
variational inference (VIBO, VTIRT); we use plain prediction loss, and pre-empt VTIRT's "discrimination recovered
best" as well-specified generative inference, not prediction loss. Do NOT cite Growing Pains (arXiv 2604.12843).
Cite-and-distinguish Ma et al. 2024 [VERIFY].

## Template and terminology
JEDM acmtrans class, two-column, NOT elsarticle. Never "neural KT". No decorative or invented jargon; exact names
(low Fisher, Fisher information, the slope a_k, the intercept c_k, multiplicative coupling, scale gauge). No em- or
en-dashes, no colons in flowing prose, American English, grant-then-qualify.

## Honest caveats
Finite-vs-asymptotic is a within-model statement (real data is reliability plus calibration, not recovery-vs-truth).
The recovery-order law is per-eigenmode; below discrimination ~1 the order inverts (state it). Multiplicativity is
earned by the ablation, not asserted. See docs/theory_memo.md for the derivations.

## Status
Plan approved 2026-07-01. Previous elsarticle draft (glued MA-GPCM in) is SCRAPPED; restart on the JEDM class,
deck-anchored, tight. Overleaf push is blocked (403, see docs/paper_workflow.md); paper is written and previewed
from local builds until the Overleaf access is restored.
