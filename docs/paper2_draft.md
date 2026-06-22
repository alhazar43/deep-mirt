# Representation sharing under-recovers low-leverage readouts in amortized encoders

Working draft. A learning-dynamics / representation paper; neural IRT is the
closed-form instance, not the subject. Register: general claim primary, low jargon
in both directions, honest scope. Sections 1-3 are drafted prose; 4-6 are structured
against the evidence and will be filled as the writing proceeds.

---

## Abstract (draft)

When a model infers a representation from data and reads several quantities off it,
the quantities are not recovered equally. We study a sharp version of this: an
amortized encoder produces a per-instance state, and two parameters are read from a
shared learned representation, one with high leverage on the prediction and one with
low. The low-leverage parameter is recovered poorly, not because the data lack
information about it (a per-instance oracle recovers it well), but because the shared
representation is allocated to the high-leverage parameter during training. Giving the
low-leverage parameter its own representation, or access to the inferred state,
restores its recovery. The effect is parameter-specific, vanishes at the joint
infinite-data-and-training limit, and is a property of the amortized-encoder
arrangement rather than of any task: we show it in a non-psychometric encoder model
and in neural item response theory, where the leverage has a closed form and we can
predict which parameter is lost and see its under-allocation in the representation
geometry. The practical consequence is a validity caution for amortized parameter
estimation: a parameter read off a shared representation can be an artifact of how the
representation was allocated, removable by decoupling.

## 1. Introduction (draft)

A growing class of models replaces direct parameter estimation with an amortized
encoder: a network reads a data instance and emits both a latent state and a set of
interpretable parameters, trained end to end on a prediction loss. Neural knowledge
tracing and neural item response theory are examples; so is any encoder-decoder that
exposes named quantities as readouts. The appeal is that one model serves every
instance and every parameter at once.

This paper asks what that convenience costs the *parameters*. We find a specific and
avoidable failure. When two parameters are read off the same learned representation
and one has much weaker leverage on the prediction than the other, the weak one is
recovered poorly even when the data determine it. The shortfall is not statistical.
The data carry enough information (a per-instance estimator that does not share a
representation recovers the parameter well), and the model would recover it at
convergence. The shortfall is that, at any practical training budget, gradient descent
shapes the shared representation to serve the high-leverage parameter, and the
low-leverage one is left with whatever capacity remains.

We make the claim general and then concrete. In a minimal encoder model with no
psychometric content, the effect appears exactly: a low-leverage readout sharing the
encoder's representation is under-recovered, the deficit is specific to that parameter,
it is not information-limited, and decoupling the readout removes it. The same effect
appears in neural item response theory, where the leverage of each parameter has a
closed form. There the weak parameter is discrimination, whose information vanishes
where responses concentrate, and we can both predict that it is the one lost and see,
in the geometry of the learned item representation, that it occupies a thin slice while
difficulty dominates.

**Contributions.**
1. We identify a recovery failure specific to amortized encoders: a low-leverage
   readout sharing the inferred representation is under-recovered, and the cause is
   representational allocation, not the data's information (a per-instance oracle
   recovers what the shared model misses).
2. We show it is general to the amortized-encoder arrangement, not to any task, by
   reproducing it in a non-psychometric encoder model and in neural IRT.
3. We characterize it: parameter-specific, removable by two representational
   interventions (a dedicated channel, or access to the inferred state), visible in
   the representation geometry, and vanishing only at the joint infinite limit.
4. We draw the consequence for measurement: parameters read off a shared amortized
   representation can be allocation artifacts unless decoupled.

**What this is not.** It is not the observation that low-information parameters are
estimated with high variance; that is classical and is about the data, whereas our
deficit persists where the data are sufficient and is fixed by changing the
representation at fixed data. It is not a claim that any reparameterization changes the
solution; we show smooth positive maps are interchangeable, so the link function is not
the lever. It is not a conditioning-number law; we tested whether the gap tracks the
Fisher conditioning number and it does not once that number is decorrelated from the
task's category count. And it is not a challenge to estimation theory: the model is
identified and the estimator is consistent. The new content is a property of the
amortized estimator's representation, which classical estimation does not have because
it has no learned representation to allocate.

## 2. Setup (draft)

**The general model.** An encoder reads an instance (a sequence of observations) and
produces a per-instance latent state. Item-level parameters are read by linear
readouts off a learned item representation that also feeds the encoder. Two readouts
matter: a high-leverage one and a low-leverage one, where leverage means the
sensitivity of the prediction to the parameter. Training minimizes a prediction loss
only; no parameter is ever a supervised target. Recovery is measured by rank
correlation between recovered and true parameters, because the latent scale is
identified only up to a gauge.

**Two representational arrangements** are the knobs we turn:
- *Shared vs decoupled:* the low-leverage readout reads the same representation that
  feeds the encoder, or its own separate one.
- *Static vs dynamic:* the low-leverage readout is a function of the item alone, or it
  may also read the inferred per-instance state.

**The closed-form instance (neural IRT).** A 2PL/GPCM decoder makes leverage explicit.
With logit z = alpha (theta - beta), prediction p = sigmoid(z), and cross-entropy loss,
the residual is r = p - y and the per-parameter gradients are
`d L / d theta = r alpha`, `d L / d beta = -r alpha`, `d L / d alpha = r (theta - beta)`.
Ability (theta) and difficulty (beta) enter through alpha; discrimination (alpha)
enters through the separation theta - beta. The Fisher informations are
`I(theta) = I(beta) = alpha^2 w` and `I(alpha) = (theta - beta)^2 w` with w = p(1-p),
so discrimination is the low-leverage parameter: its information is suppressed by the
squared lever arm and vanishes where responses concentrate (theta ~ beta). This is why
IRT is the right instrument: it tells us, before any experiment, which readout is the
weak one.

## 3. Theory: the differential rate, and that recovery is not information (draft)

Two results, both honest about their reach.

**The low-leverage readout is the slow mode.** Near an identifiable optimum, each
parameter's recovery proceeds at a rate set by its information (the standard
linearization of gradient flow, as in the deep-linear-network dynamics literature).
Discrimination, with the smallest information, is the slowest and noisiest. This is the
per-parameter analog of the classical result that low-signal modes learn last, and on
its own it is not new; it is the starting observation, not the contribution.

**Recovery is gated by representation, not information (the linchpin).** When the item
parameters form a free table that can reach the optimal prediction, every gradient pull
is linear in the residual and they vanish together, so the optimum is unbiased and the
truth is recoverable. We verify this to machine precision in a tractable model. It has a
strong consequence: the under-recovery we observe at a finite budget is *not* an
information limit, because the same data recover the parameter at the optimum. The
binding constraint is the finite-budget path through a *shared* representation. This is
the dissociation the paper turns on: in the amortized estimator, recovery is gated by
representational allocation even where the data's information is sufficient, and the
manipulations of Section 4 confirm it by holding the data fixed and moving recovery.

**The positivity map is not the lever.** Discrimination is read through a positive map.
One might suspect the map (for example, the exponential) is responsible. It is not: any
smooth strictly-monotone positive map induces a flow that is a time-reparameterization
of one canonical trajectory, so the fixed point and the rank recovery are
map-invariant; only non-smooth or non-monotone maps (a dead zone, a sign fold) lag.
Positivity is necessary, the link function is not the lever.

**What we do not claim.** We do not claim the gap is predicted by the Fisher
conditioning number. In neural IRT the category count and that conditioning number are
collinear, and when we decorrelate them (vary the conditioning number at fixed category
count over a wide range) the gap does not track it. So we describe the growth of the
gap with category count directly, as shared-channel capacity (more categories strain
one narrow item channel), not as a conditioning law. The mechanism is allocation, shown
empirically and geometrically in Section 4, not a closed-form rate theorem.

## 4. Experiments (to draft; structured against the evidence)

- **4.1 Generality (the non-IRT encoder model).** A GRU infers a per-instance state
  sharing its item representation with a low-leverage readout. Decoupling helps the
  low-leverage readout (gap grows as leverage falls; variance collapses), the
  high-leverage readout shows no benefit (parameter-specific), and a per-item oracle
  recovers what the shared model misses (not information-limited). The static-code
  control (no encoder) shows zero benefit, locating the effect in the encoder
  contention.
- **4.2 Allocation, not capacity (neural IRT gate).** Sweeping the shared width traces
  a frontier; the decoupled model sits above it at matched total capacity.
- **4.3 The under-allocation is geometric (the literal evidence).** In the shared item
  code, discrimination rides ~11% of the variance; difficulty dominates the code; the
  item code carries no respondent-ability axis. This is the allocation made visible.
- **4.4 Two levers, one control.** Decoupling and dynamic (state-conditioned)
  discrimination each relieve recovery (the dynamic result at 12 seeds: helps for
  K>=4, paired tests significant, grows with K); difficulty is the indifferent control.
- **4.5 The distortion is attenuation, not a confound.** Recovered discrimination is
  attenuated and seed-unstable under sharing, but not contaminated by ability or
  difficulty (leakage null); decoupling de-noises it.
- **4.6 A finite-budget effect.** It vanishes at the joint infinite limit; the precise
  data-vs-training dependence differs across the toy and IRT and is reported as such,
  not as a single law.

## 5. Discussion (to draft)

- Positioning: build on the rate-from-curvature dynamics (Saxe) and the amortization-gap
  idea (amortized inference is worse than per-instance), of which this is the
  parameter-recovery, non-uniform version; affirm classical IRT estimation; distinguish
  from multi-task gradient conflict (the pathway gradients are orthogonal, the cost is
  geometric under-allocation, not interference); closest structural analog is weight
  normalization, which decouples a gain from a shared direction.
- The measurement consequence and the practical prescription (decouple low-leverage
  readouts; do not read their magnitude off a shared amortized representation).
- Honest scope: synthetic; qualitative generality, not a quantitative law; rate vs data
  dependence model-dependent; the theory is the differential rate plus the free-table
  invariant, not a conditioning theorem.

## 6. Honest record (appendix)

Retired along the way and reported as such: the exponential map is not special; there
is no population-limit dynamics law (the effect is finite-budget); the conditioning
number is not the lever (collinear with category count, flat when decorrelated); there
is no ability-into-discrimination confound (leakage null); the effect is not a generic
shared-readout property (a static-code model shows nothing) but an amortized-encoder
one.
