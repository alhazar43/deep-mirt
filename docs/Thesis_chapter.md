# First Paper (Thesis Chapter)

This document is the concrete realization of `Thesis_overview.md` and
`Engineering_overview.md`. It was first written as a forward plan. It is
now rewritten as the verified narrative, organized around the result the
experiments converged on. The scattered record is in
`substrate/RESULTS.md`. This is the argument.

## Working title

Anchoring as the identifiability mechanism for a learned ability scale.
(Placeholder, refine later. The earlier framing, a versatile measurement
framework for human and machine respondents, is still the setting. The
contribution below is the load-bearing part.)

## Claim

We build an encoder and decoder measurement pack. A sequence encoder
turns a response history into a latent ability. Swappable item-response
decoders read that ability into any response format on one shared item
scale. The item response flavor is a readout, not a constraint, and the
encoder also exposes its internal knowledge-tracing state.

The load-bearing finding is about identifiability, not about any single
component. A freely optimized learned ability scale is unstable and
non-transferable. A structural constraint, anchoring in the sense of
fixed-parameter calibration, restores it. We show this on three
independent axes, on real data for two of them, and the result matches
what identifiability theory predicts a deep latent scale needs.

## Why this is the open problem

Deep models that recover interpretable item-response parameters exist and
we cite them rather than claim them (Yeung 2019, Wu et al. 2020). Placing
mixed response formats on one scale is classical and not neural (Kim and
Lee 2006), and reading pairwise comparison on the same logit scale rests
on Andrich 1978, with the forced-choice to item-response bridge in Brown
and Maydeu-Olivares 2012. Fixed-parameter calibration that holds a scale
fixed while new content links onto it is decades old (Kolen and Brennan
2004) and has a recent machine claimant for frozen models (Habba et al.
2026).

What is open is whether a single shared neural item representation read
at once by an ordinal decoder and a pairwise decoder, on a sequence
encoder, recovers one coherent scale, and whether that learned scale is
separable and invariant. Theory says a deep latent scale carries
indeterminacies that do not resolve with more data unless structure, a
prior or an anchor, is imposed (Xi and Bloem-Reddy 2023), that forcing
independence collapses recovery (Wu et al. 2020), and that a standard
deep item-response model yields item-dependent ability (Tsutsumi et al.
2024). No prior work audits whether a learned ability scale stays
invariant across item banks or instruments. That audit, and the fix, are
the contribution.

## The spine, three legs

The same pattern appears three times. Free optimization destabilizes the
learned scale. A structural constraint preserves it.

**Leg 1, re-estimation axis, real Duolingo SLAM.** New items attach to a
trained scale by anchored extension, freezing the encoder and the
existing item embeddings and fitting only the new item embeddings. On
real data the anchored estimate of new-item difficulty reproduces a full
recalibration at about 0.85 of the recalibration's own seed-to-seed
reliability ceiling, robust to a stress split that makes the new items
the hardest ones, at roughly one hundred times lower cost, and it leaves
the existing scale exactly fixed. The deeper observation is the ceiling
itself. Two full recalibrations of the same data agree only at about
0.80 on difficulty, and that agreement does not rise with more training,
it drifts down, because each run settles into its own optimum. The freely
re-estimated scale is weakly identified. Anchoring is not a lossy
shortcut, it is a stabilizer for an estimation that is otherwise not
unique. Discrimination does not transfer reliably, which we report as a
real limit.

**Leg 2, cross-instrument axis, real EdNet.** We audit whether the
learned ability scale is separable across TOEIC sections. The ability
axis itself is instrument-agnostic, a probe that decodes which section a
learner is on from the full representation succeeds, but a control that
conditions on scalar ability falls to chance, so the apparent leakage is
item content from disjoint banks, not the ability encoding. Whether a
learner sits at a consistent position on that axis across sections
depends entirely on how the per-instrument ability is read. Reading it by
running the frozen encoder on instrument-restricted subsequences, which
is out of distribution, gives weak cross-instrument consistency, about
0.41, and no construct-distance gradient. Reading it by anchored
fixed-parameter estimation, holding item parameters fixed and estimating
each learner's per-instrument ability directly, raises consistency to
about 0.72, makes person the dominant variance component, and a
construct-distance gradient emerges, within-section consistency above
cross-section, which is exactly what a valid scale should show. The same
anchoring mechanism that stabilized the scale on the re-estimation axis
raises cross-instrument consistency by about 0.32 here. Anchoring does
double duty.

**Leg 3, training-time axis, synthetic.** In the encoder-driven joint
pack, cross-format transfer to held-out items degrades with longer
training. The ordinal head keeps reshaping the shared scale through its
extra free parameters while the pairwise head saturates, so the shared
placement drifts away from the geometry that generalizes to items seen in
only one format. Early stopping recovers the transfer, from about 0.87
back to about 0.95, with no cost to the other targets. Loss reweighting,
the obvious fix, does not help and slightly hurts, which we verified by
ablation rather than assuming. The lever is how much the scale is allowed
to be reshaped, not the instantaneous balance of the two losses.

**Reading the three together.** One disease, free or over-optimization
destabilizing the learned scale, one cure, a structural constraint that
holds part of the model fixed. Three independent axes, two on real data.
A method lesson falls out, read per-subset ability by anchored
fixed-parameter estimation, not by running a sequence encoder on a
restricted subsequence, which is noisier and understates consistency.

## The artifact

The pack is one model. A recurrent encoder produces a time-varying
ability. A shared item embedding feeds a shared difficulty reader, which
feeds both an ordinal decoder and a pairwise decoder, so an item seen in
one format lands on the scale defined by the other. On synthetic data the
encoder-driven pack holds both properties at once, dynamic tracking and
cross-format placement, with held-out pairwise-only items recovered at
about 0.95 to 0.97 against an independent-fit noise floor, and the
encoder's internal state is exposed as a knowledge-tracing readout. The
item-response flavor is a swappable decoder, and richer side information
is allowed on both ends, item content into the encoder and a richer
ability unpacking out of the decoder.

## Supporting and boundary results

Respondent transfer is real but construct-bounded. On SLAM, item
difficulty derived from a pool of small language models that genuinely
attempt the items rank-predicts human item difficulty at Spearman about
0.34. Widening the model pool across families and sizes fixed the error
variance but did not move the correlation, which falsifies the hypothesis
that pool homogeneity was the limit and locates the limit in the exact-
match grading construct of the data. The number is moderate and now
shown robust to pool composition, and it will not climb on this testbed.

The synthetic cross-format mechanism is strong, held-out-format items at
about 0.98 to 0.99 against a noise floor, which isolates the mechanism
under shared ground truth and is the clean precursor to the real-data
test on graded essays.

The cost of anchored extension over full recalibration is large, about
one hundred times in wall clock on the real bank, which is the practical
half of the continual claim.

## Open

Format-agnosticism on real graded data is the next real-data leg. We have
the ASAP essay corpus, with human holistic scores as the graded format
and model-generated pairwise comparisons as the second format, run
through the shared scale. The honest qualifier is that the comparisons
are model-generated, a respectable but weaker form than human comparative
judgment, and the test must be content-mediated so that transfer is the
shared representation doing work rather than both formats independently
recovering true quality.

Unpacking ability into per-skill dimensions is a refinement, not a
necessity. The verification on the cross-instrument axis showed the
scalar scale is more adequate than the readout artifact suggested, so the
multidimensional readout is for capturing the residual rather than for
rescuing the scalar.

A policy that chooses what to ask, closing a measurement and selection
loop, is the capstone and is deferred.

## Honest caveats

On the cross-instrument axis the fixed item parameters still come from the
joint fit, a milder residual confound than the out-of-distribution
readout it replaced, which caps how strongly the positive is stated. On
the real essay test the comparisons are model-generated. Two of the three
legs are real data and one is synthetic. The respondent-transfer number
is bounded by the grading construct of the chosen corpus. Each of these
is stated where the result is reported, and each has a concrete way to be
strengthened.

## Contributions

A measurement pack whose response format is a swappable decoder on one
shared neural item scale, with the encoder's tracking state exposed. The
first audit of whether a learned neural ability scale is separable and
invariant across instruments. The finding, on three axes and two on real
data, that anchoring is the identifiability mechanism that makes a learned
ability scale stable, separable, and extensible, where free estimation
does not. A method, read per-subset ability by anchored fixed-parameter
estimation.

## References

Citation details are confirmed against primary sources before submission.
Shared references with the thesis overview are not repeated here.

- Andrich, D. 1978. Relationships between the Thurstone and Rasch
  approaches to item scaling. Applied Psychological Measurement 2(3).
- Brown, A., and Maydeu-Olivares, A. 2012. Fitting a Thurstonian
  item-response model to forced-choice data. Behavior Research Methods 44.
- Habba, et al. 2026. Growing Pains, fixed-parameter calibration for
  benchmark populations of models. arXiv 2604.12843.
- Kim, S., and Lee, W.-C. 2006. An extension of four item-response
  linking methods for mixed-format tests. Journal of Educational
  Measurement 43(1).
- Kolen, M., and Brennan, R. 2004. Test Equating, Scaling, and Linking.
  Springer.
- Paassen, B., et al. 2022. Sparse factor autoencoders for item response
  theory. EDM.
- Tsutsumi, E., Kinoshita, R., and Ueno, M. 2024. Deep item-response
  theory with independent student and item networks. IEEE TLT 17.
- Wu, M., et al. 2020. Variational item response theory. EDM.
- Xi, Q., and Bloem-Reddy, B. 2023. Indeterminacy in generative models,
  characterization and strong identifiability. AISTATS.
- Yeung, C.-K. 2019. Deep item-response theory. EDM.
