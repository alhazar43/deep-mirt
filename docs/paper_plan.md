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
- GPCM logit `alpha*theta - alpha*beta`: the only bilinear term is `alpha*theta` (`alpha` is the discrimination,
  `beta` the difficulty / step thresholds).
- NRM logit `a_k*theta + c_k`: the only bilinear term is `a_k*theta` (`a_k` the slope); `c_k` (the intercept) is additive.

Two mechanisms follow, with two different fixes.
- **The multiplicative scale gauge (representation).** `alpha*theta` / `a_k*theta` is invariant under
  `alpha -> alpha/s, theta -> s*theta`, so prediction loss pins only the product. Under finite data and a shared
  embedding, the split of scale between the multiplicative parameter and ability is under-determined. That is the
  trade-off. **Decoupling** (the multiplicative parameter gets its own item key) is the fix. It fires for any slope on
  ability, GPCM `alpha` or NRM `a_k`, whatever its Fisher. The additive `c_k` never multiplies ability, so it carries
  no scale gauge.
- **Fisher information (recovery rate).** Low Fisher curvature makes a coordinate the slow direction, recovering last
  and least reliably. The targeted fix is the **dynamic (state-conditioned) head**, which helps only a low-Fisher
  coordinate. GPCM `alpha` is low-Fisher (its leverage `(theta-beta)^2` vanishes where responses concentrate); the NRM
  slope `a_k` is not.

**NRM is the control that dissociates the two:**
- the representation trade-off (common to GPCM `alpha` and NRM `a_k`, cured by decoupling), and
- a low-Fisher rate penalty (only GPCM `alpha`, cured by the dynamic head).
`a_k` shares ability's representation so it pays the trade-off, but it is NOT low information, so it recovers at a
normal rate and the dynamic head does nothing for it (it hurts it). `alpha` has both problems at once. That is the
paper's sharpest result. The full 10-setup NRM run decides which NRM parameter needs its own item key; do not presume
it is the slope (the deck's evidence points to the intercept `c_k`).

## Story arc (the deck, told as a paper)
1. KT bolts interpretable IRT readouts on for interpretability, but are the parameters actually recovered?
2. A modular framework to ask cleanly: swap the encoder, swap the IRT decoder; benchmark by recovery. (a)
3. Recovery is uneven: the coefficient that multiplies ability comes back worst.
4. Why: the multiplicative coupling (scale gauge). Decoupling the representation escapes it.
5. It is a finite-budget effect, vanishing only asymptotically; at realistic data and training sizes the decoupling
   advantage holds and does not wash out.
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
(low Fisher, Fisher information, the discrimination alpha, the difficulty / step thresholds beta, the slope a_k, the
intercept c_k, multiplicative coupling, scale gauge). No em- or
en-dashes, no colons in flowing prose, American English, grant-then-qualify.

## Honest caveats
Finite-vs-asymptotic is a within-model statement (real data is reliability plus calibration, not recovery-vs-truth).
The recovery-order law is per-eigenmode; below discrimination ~1 the order inverts (state it). Multiplicativity is
earned by the ablation, not asserted. See docs/theory_memo.md for the derivations.

## Status
Plan approved 2026-07-01. Previous elsarticle draft (glued MA-GPCM in) is SCRAPPED; restart on the JEDM class,
deck-anchored, tight. Overleaf push is blocked (403, see docs/paper_workflow.md); paper is written and previewed
from local builds until the Overleaf access is restored.
