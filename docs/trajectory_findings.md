# Recovering ability trajectories and their rates: findings

A short, paper-style summary of the trajectory-recovery program. The full
plan, prior work, and per-experiment detail are in
`docs/trajectory_program.md` and the `deep_irt/traj_*/RESULTS_*.md` files.
This is the honest state of what has and has not been shown.

## Abstract

Item response theory estimates a fixed trait and knowledge tracing
predicts the next response. We study a third quantity, the latent ability
written as a function of accumulated evidence and its rate, recovered from
a prediction-trained encoder-decoder, for human learners (a learning
curve) and language models (an in-context adaptation curve) on one shared
scale. Across six experiments we find the estimand is recoverable when the
data-generating process genuinely contains a trajectory, the measure
correctly registers its absence otherwise, and the trajectory's shape is
respondent-specific. The clean positive results are on controllable data
and on language-model in-context adaptation. The real-data educational
fronts, human learning-rate recovery and machine-to-human difficulty
transfer, did not produce positive results on the datasets available, and
we report why.

## Estimand and method

For a respondent we recover theta(e), ability over accumulated evidence e,
and its rate. A sequence encoder trained under prediction loss exposes a
per-step ability, a parametric curve is fit to it, and the rate is read
off. The rate is affine-invariant in theta, so the encoder's arbitrary
scale does not bias it. Identifiability requires enough density, a window
that spans the curve's elbow, and a generating process that actually
contains a curve.

## Results

| exp | respondent, data | result |
|---|---|---|
| E0 | synthetic, human-like | positive, recovery corr(r_hat, r_true) 0.46, weak and density-limited |
| E1 | LLM, ARC (known task) | null, flat theta(k), true equals shuffled, correctly no adaptation |
| E1b | LLM, synthetic remapping | positive, rises under true, chance under shuffled, magnitude scales with model size, threshold at k about 10, robust across mappings |
| MT | LLM, EN to Dinka translation | real-task positive (gate), chrF rises +5.6 with shots under true and is flat under shuffled (gap +7.1), full IRT ladder compute-bound |
| E2 / E2b / E2c / E2d | human, EdNet / ASSISTments / KDD (binary + graded) | rate-magnitude recovery not established; a trajectory EXISTS but the per-student rate is unreliable because the KT response signal is intrinsically near-saturated (graded K=4 still 80% top-category in E2d), a data-property limit, not the coding or the method |
| E3 | transfer, SciEx graded | null, graded 0.07 does not beat binary 0.10 |

## Three findings

1. The measure recovers a trajectory when the generating process contains
one (E0, E1b) and registers its absence without fabricating a curve when it
does not (E1 on a task the model already knows). The shuffled-label and
permuted-order controls make these nulls interpretable.

2. The trajectory's shape is respondent-specific. Human and human-like
learning approaches an asymptote smoothly, so a single rate captures it.
Language-model in-context adaptation is threshold-like, a phase transition
near ten shots, so the comparable quantity is the adaptation magnitude and
its threshold, not a smooth rate. One latent object, respondent-specific
parameterization.

3. Validating a recovered rate needs the right criterion, and we
established one. Gain over a fixed window is invalid when rates are
heterogeneous, because fast learners plateau inside the window, a positive
control showed recovery works on synthetic data (0.46) while that gain
metric scores negative even for the true rate. The replacement splits in
two. Held-out predictive improvement of a dynamic model over a
constant-ability null is a valid, saturation-robust, ground-truth-free GATE
for whether a trajectory EXISTS (with-rate versus no-rate separation at
p = 5e-11), and it is the test to license a dynamic-ability claim on real
data. It is NOT valid for magnitude, the per-learner margin does not rank
learners by rate, so the rate itself is read from a parametric curve fit on
the model's estimated item parameters (recovery ceiling about 0.41 with
estimated items). Existence gate first, then the parametric rate.

## Honest scope and limitations

The demonstrated contributions are the estimand and its identifiability
conditions (E0), the machine in-context adaptation curve with a clean
learning-versus-priming separation and a size-scaled, mapping-robust
threshold (E1b, E1), and the cross-respondent shape difference. The
real-data educational fronts are unsupported. EdNet is a single-pass
stream with no learning curve. ASSISTments has repeated practice but gives
only a small concurrent signal confounded by skill-id-as-item, and its
predictive number is discarded as a metric artifact. KDD Cup 2010, the
decisive non-circular test with the validated pipeline, is more
informative, a learning trajectory genuinely EXISTS (model-free, accuracy
rises 6.1 points, 74 percent improve) and the encoder tracks it stably, but
the per-student rate is unreliable (split-half 0.17) because the binary
response is near-saturated, so the AFM concurrent null is a measurement-
floor artifact, not a refutation. E2d tested whether a graded K=4 response
rescues it and it does not, the graded signal is still 80 percent
top-category, so the saturation is intrinsic to these mastery-oriented
tutoring logs rather than an artifact of binary coding. The human
RATE-magnitude claim is thus unestablished with a fully diagnosed blocker,
the data response distribution itself, which points to a genuinely error-
rich or partial-credit corpus (E0 recovers the rate when the signal carries
it). SciEx transfer is a real null,
machine difficulty does not track the coarse three-level human label and
graded does not beat binary. The strong positives are on synthetic and
machine data, and the real-data human claim is gated on richer responses
rather than refuted.

## Next steps

The decisive human test uses problem-level KDD Cup 2010 data, now on disk,
for a non-circular concurrent-validity check with a well-posed criterion
(in progress). A real-task language-model front (low-resource translation)
would extend the adaptation result beyond a synthetic task. Two flagged
near-duplicates (D-BIRD, the ZPD paper) need a full read before any
submission.
