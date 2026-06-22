# Paper 2 story and evidence ledger

Consolidated after the overnight runs (the N by K by budget sweep, the powered
representation probe, the formal leverage proposition, the stable-config decoupling
re-run, and the dynamic re-test) and the venue exploration. This supersedes the earlier
version of this file, which was written before those runs and overstated the decoupling
and dynamic effects.

## The core, as the evidence supports it

In neural IRT and amortized knowledge tracing trained on response prediction, the
interpretable item parameters are recovered at different rates and qualities, ordered,
under a stated condition, by their Fisher leverage on the prediction. Discrimination,
the low-leverage parameter for typically discriminating items, is recovered slowly and
non-monotonically, is under-encoded in the shared item embedding relative to the step
thresholds, is recovered below what the data allow by a margin that grows as data become
scarce, and is attenuated in magnitude. State-conditioned (dynamic) discrimination gives
a modest, parameter-specific improvement. Giving discrimination its own static embedding
does not. The framing is representation learning. The infinite-data limit is not the
object, we affirm the estimator's consistency and do not contest IRT identifiability.

The contribution is the characterization, not a fix. The robust result is that an
amortized predictor under-serves its low-leverage interpretable parameter in a specific,
measurable, and predictable way. The intervention is a secondary and modest lever.

## Intro arc

Observation. Train an amortized neural IRT model to minimize prediction loss and read
back the item parameters it has learned. Ability and step thresholds come back well.
Discrimination comes back compressed in magnitude and degraded in rank, and the deficit
worsens as training continues even while the prediction loss keeps falling. The shared
model recovers discrimination at 0.872 in rank against ground truth where a per-item
maximum-likelihood fit given the true ability reaches 0.968. The data are informative.
The learned representation is the bottleneck.

The leverage explanation, made precise and conditional. The Fisher information for
discrimination is I(alpha) = (theta - beta)^2 w, with the squared lever arm suppressed
exactly where the response weight w concentrates, near theta = beta. This suppression
holds unconditionally (Lemma 1, a Chebyshev correlation inequality). The ordering in
which discrimination carries the least information holds for typically discriminating
items, alpha at least about one, and inverts for under-discriminating items, because
I(alpha) carries no alpha^2 prefactor while I(theta) and I(beta) do (Proposition 2). So
the response model names, conditionally and in closed form, which parameter the shared
predictor will under-serve. This is real, and on its own it is a statement about a
parameter in isolation. It says nothing about the shared learned representation that
actually carries discrimination.

The representation contribution. The deficit is a property of the amortized encoder's
shared representation, established four ways. The oracle dissociation isolates it from
data information, the per-item fit reaches 0.968 where the shared model sits at 0.872.
It scales with data scarcity, the oracle gap in rank is +0.097 at N=800, +0.182 at
N=400, and +0.371 at N=200, and it persists at +0.085 at N=4000 with a five-hundred item
bank, every interval excluding zero. The shared embedding under-encodes discrimination,
a powered cross-validated probe over five hundred items decodes the step-threshold
location at R^2 0.96 but discrimination at only 0.69, and only 1.4% of the embedding
variance lies along the discrimination direction. And it is attenuated in magnitude, the
gauge-fixed linking slope of recovered against true discrimination is 0.81 for the shared
model against 0.96 for the oracle. The effect is not specific to psychometrics, a
non-psychometric amortized encoder with a deliberately low-leverage readout reproduces
it while a static encoder-free variant does not, which locates the cause in the
amortized-encoder architecture.

The non-monotone dissociation. The recovery does not merely lag, it degrades with
training. Shared discrimination recovery rises to about 0.93 near epoch 100 then decays
to about 0.87 by epoch 500 while the prediction loss is still falling, in every seed,
with a LayerNorm control separating the decay from generic overfitting. Prediction
improvement and parameter recovery diverge, so stopping early is not a free repair when
the practitioner trains to a prediction criterion.

The intervention. State-conditioned discrimination, which lets the discrimination
readout depend on the inferred ability, improves recovery modestly and parameter
specifically, by +0.035 to +0.043 in rank across category counts and up to +0.081 at
eleven categories on the decoupled architecture, with the step thresholds unmoved.
Giving discrimination its own static embedding does not improve rank recovery at any
sample size. The fix is small and the dynamic lever carries a training-instability cost.

The study. We characterize this across eight to ten seeds with bootstrap confidence
intervals on a GPCM decoder with an LSTM encoder on the live prediction-loss path. The
leverage ordering is stated as a formal conditional proposition. The amortization
deficit, its scaling with data scarcity, the under-encoding asymmetry, the attenuation,
the non-monotone decay with its regularization control, the modest dynamic intervention,
and the non-psychometric generality control are each established.

## Claim to evidence map

| Claim | Evidence | Tag |
|---|---|---|
| Recovery is leverage-ordered (conditional, alpha at least about one) | Formal proposition (Lemma 1 unconditional, Proposition 2 conditional) plus 8 to 10 seed simulations | Robust |
| The data are informative, the representation is the bottleneck | Oracle dissociation, per-item MLE 0.968 vs shared 0.872, parameter-specific | Robust |
| The deficit scales with data scarcity | Oracle gap rank +0.097, +0.182, +0.371 at N=800, 400, 200, and +0.085 at N=4000, all CIs exclude 0 | Robust |
| Discrimination is under-encoded in the shared embedding, an asymmetry | Powered CV probe at Q=500, threshold R^2 0.96 vs discrimination 0.69, 1.4% of embedding variance | Robust |
| Discrimination is attenuated in magnitude | Gauge-fixed linking slope shared 0.81 vs oracle 0.96 | Robust |
| Recovery worsens with training while the loss falls | Non-monotone curve 0.93 to 0.87, 10/10 seeds, continuous metric, LayerNorm control | Robust |
| The effect is a property of the amortized-encoder architecture, not of IRT | Non-psychometric encoder reproduces it, static no-encoder variant does not | Robust |
| Dynamic conditioning modestly strengthens recovery | +0.035 to +0.043 rank across K, +0.081 at K=11, parameter-specific, fragile | Modest |
| Decoupling via a separate static embedding strengthens rank recovery | Null at every N at 8 seeds, +0.027, +0.021, -0.000, none exclude 0 | Dropped |
| The deficit grows with category count | Largest at K=2, flat above | Dropped |
| Geometric under-allocation as an absence of encoding | Q=60 in-sample artifact, the honest result is the asymmetry above | Reframed |
| A Fisher-conditioning-number law for the gap | Gap flat in kappa at fixed K over a 45x range | Dropped |
| A universal representation law independent of the encoder | Static toy null | Dropped |
| The +0.27 decoupling and +0.19 dynamic magnitudes | Inflated baselines, metric, and seed counts, honest figures above | Retired |

## Honest scope

We affirm consistency. The estimator is not claimed to be inconsistent and IRT
identifiability is not contested. The contribution lives in the finite-data,
finite-epoch regime that real training occupies, and we argue that regime is the
operative one, supported by the non-monotone decay that makes convergence to oracle
recovery unlikely under a prediction stopping criterion.

The leverage result is a conditional predictive ordering, not a magnitude law. It ranks
which parameter suffers, for typically discriminating items, and inverts for
under-discriminating ones. It does not predict by how much or when the decay peaks, and
the gap is flat in the conditioning number kappa at fixed K, so the strongest
conditioning-number form of the story does not hold and is not claimed.

The intervention is modest and is dynamic, not decoupling. Decoupling a separate static
embedding does not improve rank recovery at any sample size once the configuration is
stable and the seed count is adequate. Dynamic conditioning helps by a small,
parameter-specific, fragile amount. The earlier +0.27 decoupling and +0.19 dynamic
figures are retired, they came from narrow baselines, the slope rather than rank metric,
and three-seed short-training runs.

The evidence is synthetic GPCM ground truth on a single architecture family, an LSTM
encoder with a GPCM decoder, with a small backbone swap rather than a full ablation.
Real-data confirmation is not yet in hand.

## The venue decision, a fork

The evidence supports two framings, and the choice is a positioning decision rather than
a ranked list.

Measurement framing. Psychometrika is the best pure fit, with Behaviormetrika as a
faster lower-bar backup whose 2025 special feature actively wants the machine learning
and measurement intersection. The under-encoding asymmetry, the attenuation, the oracle
dissociation, and the conditional leverage proposition read as a measurement-validity
result in the audience's own language, and the modest intervention is not penalized.
This framing conflicts with the standing project rule that the work should be presented
prediction-home with IRT as flavor and not as a psychometrics-theory contribution.

Machine-learning framing. The ICLR amortized-inference and representation-learning
thread is the natural ML home, and the sweep strengthens it, the deficit's lawful
scaling with data scarcity is the dynamics-respectable spine that was missing, and the
non-monotone prediction-recovery dissociation and the under-encoding asymmetry are
representation findings. This framing respects the standing rule. The modest effect and
the synthetic-only evidence place the near-term home at a representation-learning or
learning-dynamics workshop, building toward an ICLR main-track paper once the decay has a
mechanistic account and generality is shown at scale.

Recommendation. Lead with the machine-learning framing, since it respects the standing
rule and the scaling law now anchors it. The near-term target is a representation
learning or learning dynamics workshop, with the same study converting to an ICLR
main-track submission once the decay mechanism and the at-scale generality land.
Behaviormetrika is the journal backup if a measurement venue is preferred and the
framing rule is relaxed. EDM and AIED need a real graded dataset with item-parameter
ground truth and are revisited then. See docs/paper2_venues.md for the full comparison.

## Evidence gaps to close

1. A mechanistic account of the non-monotone decay, beyond the LayerNorm control. Track
   the gradient or the representation over training and show why the shared embedding
   turns against discrimination late in training while the loss is still falling. This
   is the single highest-value gap for the ML framing.

2. Generality at scale. The non-psychometric encoder result and a second backbone, the
   transformer encoder, each at adequate seeds with intervals that exclude zero, to lift
   the architecture claim above a small backbone swap.

3. A real graded dataset with item-parameter ground truth, required for EDM, AIED, and
   the measurement framing, and a useful robustness check for the ML framing.
