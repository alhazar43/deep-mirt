# Paper 2 outline (draft)

A learning-dynamics paper about representation learning, with neural IRT as the
concrete instantiation. The subject is *how a parameterized readout on a shared
representation learns*; discrimination in IRT is where we instantiate it. Register:
learning-dynamics centric, IRT as flavor, low jargon in both directions.

**Working title directions**
- *Representation conditioning, not parameterization, sets recovery speed in
  prediction-trained encoder-decoders.*
- *Why a gain learns last: a gradient-flow account of shared-representation readouts.*

## The one-line novelty

Not a new parameterization trick. A sharpening: among smooth parameterizations the
choice does not change what is recovered (a time-reparameterization), so the real
lever is the conditioning of the shared representation, which decoupling removes.
This locates the lever that three existing lineages each leave implicit.

## The claim chain (what we show, in order)

1. A readout parameter's recovery speed is set by its information (curvature) on the
   prediction, not by the loss value. Discrimination is the low-information mode, so
   it recovers last. (Build on Saxe 2014.)
2. Reading discrimination off a *shared* representation that a high-information
   parameter (ability) dominates throttles its recovery rate by the conditioning
   number kappa of that representation. A finite-budget rate effect, not a bias: the
   endpoint is unchanged.
3. The deficit scales with a task knob (number of answer categories K), because
   kappa grows with K.
4. The choice of positive map is *not* a lever: among smooth strictly-monotone maps
   the recovered ranking is identical (the map is a time-reparameterization). Only
   non-smooth or non-monotone maps break this.
5. Decoupling the representation (a separate readout code for discrimination) removes
   the conditioning bottleneck and restores fast recovery. Architecture-independent.

## Section-by-section

**Section 1, Introduction.**
- The puzzle, stated as a training-dynamics question: with one shared representation
  and one loss, why does the discrimination readout recover far slower and noisier
  than the ability and difficulty readouts?
- Name the minimal model and the tool up front (a shared-representation encoder-decoder;
  gradient flow plus the curvature/conditioning of the shared code).
- Mechanism in one sentence (claim 2 above).
- Contributions as the five falsifiable claims.
- One paragraph of "what this is not": not a lazy/NTK story (this is the feature-learning
  regime), not an information-bottleneck argument (no mutual-information claims), not a
  worst-case bound (average-case gradient flow; the conditioning is a task property,
  not a pathology).

**Section 2, Setup (the minimal model).**
- The encoder-decoder: a shared item representation feeds both a high-information readout
  (ability, difficulty) and a low-information readout (discrimination, a positive slope).
- The IRT instantiation as the concrete case (2PL, GPCM for K categories).
- The objects we need: the prediction loss (cross-entropy) and its residual; the
  per-parameter gradients; the information each parameter carries on the prediction;
  the conditioning number kappa of the shared code; how kappa grows with K.
- Recovery measured by rank (the scale is identified only up to a gauge).

**Section 3, Gradient-flow theory (leads, before any experiment).**
- The per-parameter gradient-flow equations; discrimination is the slow mode because its
  information vanishes where responses concentrate.
- The shared-code conditioning number and the recovery-time consequence (a factor kappa
  more steps under one learning rate); decoupling cuts it back.
- The map result: a smooth positive map induces a preconditioned flow that only
  reparameterizes the clock, so the recovered ranking is map-invariant; non-smooth and
  non-monotone maps break a stated condition.
- Name the phases: capture, slow-recovery, joint-limit.
- Calibration discipline: state the rate and the finite-data floor as consequences of
  this model, not as new general theorems.

**Section 4, Synthetic experiments (one per claim, theory overlaid on data).**
- Trade-off frontier: it is allocation, not capacity (decoupled dominates at matched size).
- Stiffness vs K: the decoupling advantage tracks the conditioning number and grows with K.
- Peak-then-decay trajectory: the good solution is reached then left (verify it survives a
  continuous progress metric, not just a thresholded one, per Schaeffer 2023).
- Gradient-pathway split: the mechanism ablation, showing the shared code is captured by the
  high-information direction (orthogonal, not a tug of war).
- Finite-sample sweep: at a fixed budget the gap does not shrink with more data (rate-limited).
- Map convergence: smooth maps share one profile, non-smooth lag (claim 4 confirmed).

**Section 5, Decoupling removes it (the ablation and the prescription).**
- Coupled vs decoupled, theory overlaid; decoupling is the fix and the prescription.
- Difficulty is the indifferent control (high information, untouched).
- Ability is a *different kind* of effect (a finite-data overfit on the encoder readout,
  not a rate effect), which sharpens that discrimination is the rate effect.

**Section 6, Lift to other encoders (honest scope).**
- The mechanism is architecture-independent: on a recurrent, an attention, and a memory
  encoder, sharing slows discrimination and decoupling restores it.
- Two distinguished subtleties, not exceptions: the K-scaling law replicates cleanly on the
  recurrent and memory encoders and is optimization-limited (unreadable as run) on the
  attention encoder at high K; the decay magnitude varies by encoder, which is the
  ability-overfit term and is expected to depend on encoder capacity.
- No exact rates claimed here. Real-data ground-truth recovery is future (needs an
  error-rich, graded corpus; the data proposal targets it).

**Section 7, Discussion and positioning.**
- The three lineages (see positioning map below).
- External corroboration of the one prediction (discrimination is the fragile parameter):
  a misspecification taxonomy (better prediction does not imply valid recovery; discrimination
  collapses first) and real-data scale stability (difficulty holds more of its reliability
  ceiling than discrimination). Both confirm the ordering independently.
- Failure modes: the attention-encoder instability, the ability-overfit closed form, the
  encoder-dependence of the decay magnitude.
- Open: deeper stacks, normalization, the attention instability at high K.

**Appendix.** Full derivations, hyperparameters, and the honest record of the two retired
claims (the exponential-map-is-special hypothesis, refuted; the population-limit law,
downgraded to a finite-budget rate effect).

## Positioning map (build on vs contrast)

- **Build on, rate-from-curvature.** Saxe-McClelland-Ganguli 2014; Amari 1998 (natural
  gradient, conditioning sets the rate); Martens-Grosse 2015 (K-FAC: ill-conditioning is
  the bottleneck; our decoupling is the architectural block-diagonal analog).
- **Instantiate, reparameterization as preconditioning.** Li-Wang-Lee-Arora 2022 (any
  reparameterization is mirror descent); Amid-Warmuth 2020 (multiplicative updates);
  Chou-Maly-Stoger 2023 (smooth monotone map is a time-reparameterization). Our scalar
  preconditioned flow is the concrete instance; the map-invariance is the known
  fixed-point guarantee.
- **Contrast, implicit bias of parameterization.** Woodworth 2020, Gunasekar 2018,
  Vaskevicius 2019, Azulay 2021, Yun 2021. There the parameterization is the lever, but
  the levers are structural (depth, the u^2-v^2 Hadamard form, init scale or shape) under
  underdetermination. Ours is smooth-monotone and fully determined, so the map cannot be
  the lever; we draw that boundary precisely.
- **Closest structural analog.** Salimans-Kingma 2016 (weight normalization decouples a
  gain from the shared direction); van Laarhoven 2017, Wan 2021, Heo 2021 (a scale's
  effective learning rate is diluted by the shared representation norm). They found the
  symptom and an engineering fix; we give the structural reason (representation
  conditioning) and tie it to a specific low-information parameter.

## Honest scope

- All dynamics evidence is synthetic; the real-data leg is a stability proxy, not
  ground-truth recovery.
- No population-limit law; the effect is finite-budget.
- The ability-overfit side is the softest formalization (proved for a linear readout,
  argued for the encoder).
- The attention-encoder K-scaling readout is optimization-limited and needs a targeted
  rerun before any claim.

## Discipline the genre enforces (carry into the writing)

- Write the theory and its predictions as if they preceded the experiments; do not fit the
  theory curves to the data.
- Check any threshold or peak claim against a continuous progress metric.
- Keep the register learning-dynamics centric with IRT as flavor; gloss every term; no
  jargon pile-up in either direction.
