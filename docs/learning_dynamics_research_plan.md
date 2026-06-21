# Learning Dynamics Research Plan

Working title: Learning Geometry and Parameter-Specific Dynamics of Discrimination in Neural IRT Models

Version: canonical execution plan

Status: planning document. The live empirical status is tracked separately in
`docs/learning_dynamics_progress.md`.

## Goal

Develop a rigorous learning-dynamics study of neural knowledge-tracing models
with IRT-structured decoders. The study asks how psychometric quantities emerge
when the model is trained only by response-level likelihood, with no direct
parameter supervision on `theta`, `beta`, or `alpha`.

The central claim to test is that psychometric parameters are not generic neural
outputs. They have different statistical roles and therefore different finite
sample recovery dynamics.

Expected taxonomy:

| Parameter | Role | Preferred treatment | Core risk |
|---|---|---|---|
| `theta` | learner state or ability | dynamic, narrow learner state | item identity leakage and overfit |
| `beta` | item location, difficulty, threshold, intercept | static item parameter | dynamic treatment adds noise |
| `alpha` | discrimination, sensitivity, gain, slope | positive map plus separate, wider, or contextual channel | low-information slow recovery |

The study should not be framed as a generic architecture improvement. It should
be framed as a finite-time psychometric recoverability study.

## Operating Rules

- Treat prediction and recovery as separate objectives.
- Treat synthetic recovery as mechanism evidence, not real-world proof.
- Treat real data as ecological support through prediction, calibration,
  stability, and external-reference agreement.
- Preserve negative, mixed, and refuted results. They are part of the study.
- Do not overwrite the historical consolidated study in
  `docs/LEARNING_DYNAMICS_STUDY.md`.
- Use `docs/learning_dynamics_progress.md` after every run to record
  expectation versus reality.
- Do not edit frozen `ma-irt` except additive configs if a run requires them.

Preferred language:

- Use "response-level IRT likelihood", not "non-IRT loss".
- Use "no parameter-level supervision", not "unsupervised".
- Use "near-linear recovery", not "exact recovery".
- Use "finite-time recoverability", not vague convergence.
- Use "state-conditioned effective discrimination", not only "dynamic alpha".
- Use "contextual discrimination residual" for the dynamic component.
- Use "selective state-conditioning", not "make parameters dynamic".
- Use "parameter-specific representation and learning geometry", not
  "activation trick".

Claims to avoid:

- Do not claim `exp` is universally optimal.
- Do not claim `alpha` is lognormal, therefore `exp` is required.
- Do not claim dynamic `alpha` is theoretically necessary in all IRT models.
- Do not claim prediction improvement proves psychometric recovery.
- Do not claim synthetic recovery proves real-world validity.
- Do not claim high Pearson correlation is exact metric recovery.
- Do not claim response-level cross entropy through an IRT decoder is not IRT
  likelihood.

## Core Framing

The model class is:

```text
h_t = E_phi(i_<=t, y_<=t)
p_psi(y_t | h_t, i_t) = IRTDecoder_psi(h_t, i_t)
L(phi, psi) = - sum_t log p_psi(y_t | h_t, i_t)
```

The loss supervises observed responses only. It does not directly supervise:

```text
theta
beta
alpha
```

There is no recovery loss and no explicit prior-matching term forcing learned
parameters to match generating distributions.

The research question is:

> Which psychometric quantity becomes recoverable, when does it become
> recoverable, through what gradient signal, and under which representation and
> parameterization constraints.

The serious contribution is not that correctly specified synthetic IRT data can
be recovered by an IRT decoder. The serious contribution is that `theta`,
`beta`, and `alpha` have different finite-time recovery paths under a neural
sequence encoder, and those paths depend on architectural factorization and
positive-map geometry.

## Research Questions

### RQ1. Prediction versus recovery

Does good response prediction imply meaningful recovery of `theta`, `beta`, and
`alpha`, or can prediction and psychometric recovery diverge.

Evidence required:

- prediction metrics and recovery metrics reported separately,
- prediction-recovery phase plots,
- cases where prediction improves but recovery does not, if present.

### RQ2. Parameter-specific learning dynamics

Do `theta`, `beta`, and `alpha` recover at different speeds and with different
stability under the same response-level likelihood.

Expected pattern:

- `theta` and `beta` recover quickly and near-linearly under correct
  specification.
- `alpha` is slower, lower-information, more sensitive to architecture, and
  more affected by category complexity.

Evidence required:

- checkpointed recovery curves,
- recovery area under the curve,
- time-to-threshold,
- high-alpha versus low-alpha recovery.

### RQ3. Parameter-specific representation

Does each parameter prefer a different static or dynamic representation.

Expected pattern:

```text
theta: dynamic learner state
beta: static item or threshold location
alpha: static item component plus contextual residual, or separate/wider channel
```

Evidence required:

- capacity-matched static versus dynamic controls,
- dynamic `beta` negative control,
- all-wide and all-dynamic controls,
- shuffled-state alpha control.

### RQ4. Positivity versus parameterization geometry

Is `exp(raw)` useful merely because it enforces `alpha > 0`, or because it
induces a different effective learning geometry in `alpha` space.

Evidence required:

- positive-map ablation,
- matched effective alpha initialization,
- per-map learning-rate controls,
- gradient and induced-preconditioner diagnostics,
- high-alpha recovery split.

### RQ5. Finite-time recoverability versus asymptotic identifiability

Are the representation and positive-map advantages strongest under finite data,
finite training, sparse exposure, noise, and high category complexity.

Evidence required:

- data-size sweeps,
- training-time sweeps,
- category-count sweeps,
- explicit distinction among sample, optimization, category, and capacity
  asymptotics.

### RQ6. Generalization across IRT heads

Does the same taxonomy generalize from 2PL and GPCM to GRM and NRM.

Expected pattern:

- slope-like parameters behave like `alpha`,
- threshold, location, and intercept parameters behave like `beta`,
- latent state parameters behave like `theta`.

## Formal Mechanism Layer

The theory should be local and mechanism-focused. It should not promise global
convergence or universal optimality.

### Positive maps are not optimization-equivalent

Let:

```text
alpha = g(a)
```

where `a` is the raw neural output and `g` is a differentiable positive map.
For a loss `L(alpha)`, gradient flow in raw space gives:

```text
dL/da = dL/dalpha * g'(a)
dot(a) = - dL/da
dot(alpha) = g'(a) dot(a)
dot(alpha) = - [g'(a)]^2 dL/dalpha
```

Since `a = g^{-1}(alpha)`:

```text
dot(alpha) = - m_g(alpha) dL/dalpha
m_g(alpha) = [g'(g^{-1}(alpha))]^2
```

Interpretation:

Different positive maps define different effective preconditioners in
`alpha` space. They are not merely interchangeable positivity constraints.

### Induced preconditioners

Exponential:

```text
g(a) = exp(a)
g'(a) = exp(a) = alpha
m_exp(alpha) = alpha^2
```

Softplus:

```text
g(a) = log(1 + exp(a))
g'(a) = sigmoid(a)
g'(g^{-1}(alpha)) = 1 - exp(-alpha)
m_softplus(alpha) = (1 - exp(-alpha))^2
```

Sigmoid:

```text
g(a) = sigmoid(a)
0 < alpha < 1
m_sigmoid(alpha) = alpha^2 (1 - alpha)^2
```

ReLU:

```text
g(a) = max(0, a)
m_relu(alpha) = 1 when active
m_relu(alpha) = 0 in the inactive region
```

Square:

```text
g(a) = a^2 + epsilon
```

This is positive but non-injective, introducing raw-space sign symmetry and
ambiguous raw dynamics.

Core statement:

> Positivity is a constraint. Positive maps define optimization geometries.
> Exponential is one such geometry, not a universal optimum and not sufficient
> by itself to explain the observed neural recovery ordering.

### Local convergence rate

Near an identifiable optimum `alpha_star`, assume:

```text
dL/dalpha approx H (alpha - alpha_star)
H > 0
e(t) = alpha(t) - alpha_star
```

Then:

```text
dot(e) approx - m_g(alpha_star) H e
e(t) approx e(0) exp[-m_g(alpha_star) H t]
```

The local recovery rate is:

```text
rate_g = m_g(alpha_star) H
rate_exp = alpha_star^2 H
```

Interpretation:

Exp can accelerate local recovery for sufficiently high true discrimination
items in the scalar alpha-space approximation. This is conditional on local
identifiability and comparable optimization conditions. It does not prove that
exp always wins, and it does not imply that scalar preconditioning alone explains
the neural model's positive-map behavior.

### IRT gradient coupling

For binary 2PL:

```text
p_ij = sigmoid(alpha_i (theta_j - beta_i))
x_ij = theta_j - beta_i
```

Binary cross entropy gradients have the form:

```text
dL/dtheta_j = (p_ij - y_ij) alpha_i
dL/dbeta_i = -(p_ij - y_ij) alpha_i
dL/dalpha_i = (p_ij - y_ij) x_ij
```

If `alpha_i = exp(a_i)`, then:

```text
dL/da_i = alpha_i sum_j (p_ij - y_ij) x_ij
```

Interpretation:

- `theta` and `beta` are location-like.
- `alpha` is slope-like.
- `alpha` depends on current learner-item separation.
- `alpha` recovery is coupled to the quality and alignment of `theta` and
  `beta`.

Fisher-style local blocks:

```text
I_alpha_alpha = E[x^2 p(1-p)]
I_theta_theta = E[alpha^2 p(1-p)]
I_beta_beta = E[alpha^2 p(1-p)]
I_alpha_theta = E[alpha x p(1-p)]
I_alpha_beta = - E[alpha x p(1-p)]
```

This supports the low-information and coupling interpretation of
discrimination recovery.

## Canonical Alpha Decomposition

Use a defensible decomposition for state-conditioned discrimination:

```text
log alpha_hat[j,t] = a_j + delta[j,t]
E_t[delta[j,t] | j] = 0
```

Definitions:

- `a_j` is the static item discrimination component.
- `delta[j,t]` is the contextual discrimination residual.
- `exp(a_j)` is the stable item-level discrimination.
- `exp(delta[j,t])` is a context-dependent modulation factor.

This avoids saying that classical item discrimination is simply dynamic. The
model estimates a stable item component plus an effective contextual slope.

## Experiment Ladder

### E1. Baseline recovery trajectories

Purpose:

Show that a learning-dynamics study needs trajectories, not endpoint tables.

Default setup:

- synthetic 2PL and GPCM data,
- known `theta`, `beta`, and `alpha`,
- LSTM encoder,
- response-level prediction loss,
- GPCM as the primary ordinal setting,
- start with `N=800`, `Q=60`, `T=60`, `K=4`,
- seeds at least `[0, 1, 2, 3, 4]`.

At each checkpoint compute:

- prediction NLL or CE,
- accuracy,
- AUC for binary,
- QWK for ordinal,
- Brier score where applicable,
- ECE or category calibration where applicable,
- `r_theta(s)`,
- `r_beta(s)`,
- `r_alpha(s)`,
- Pearson and Spearman,
- linked RMSE,
- slope and bias after linking.

Learning-dynamics summaries:

```text
T_tau_alpha = min{s : r_alpha(s) >= tau}
AUC_rec_alpha = mean_s r_alpha(s)
```

Required plots:

- recovery curves for `theta`, `beta`, and `alpha`,
- prediction versus recovery phase plot,
- NLL versus `r_alpha`,
- high-alpha versus low-alpha recovery,
- linked endpoint scatterplots.

Success criterion:

Show whether `alpha` is slower, stiffer, more variable, or more
architecture-sensitive than `theta` and `beta`.

### E2. Static and dynamic channel ablation

Purpose:

Test whether parameters need different static or dynamic representations.

Conditions:

- static-only `theta`, `beta`, `alpha`,
- dynamic-only variants,
- static plus dynamic concatenation,
- separate static and dynamic channels,
- static narrow alpha,
- static wide alpha,
- dynamic narrow alpha,
- dynamic wide alpha,
- decoupled wide-key alpha,
- state-conditioned alpha,
- shuffled-state alpha,
- dynamic beta negative control,
- all heads wide,
- all heads dynamic,
- parameter-specific channels.

Critical comparisons:

- dynamic or separate alpha versus capacity-matched static alpha,
- dynamic beta versus static beta,
- all-wide versus selective alpha wide channel,
- all-dynamic versus selective dynamic alpha.

Expected result:

- `theta` prefers a dynamic state but is harmed by item identity leakage.
- `beta` prefers static item-only representation.
- `alpha` benefits from a hybrid, decoupled, or state-conditioned channel.
- Dynamic beta should be neutral or harmful.

### E3. Positive-map ablation

Purpose:

Test whether any apparent exp or smooth-map advantage is positivity,
alpha-space geometry, neural representation interaction, or optimizer behavior.

Maps to compare:

- direct unconstrained raw alpha,
- ReLU,
- softplus,
- softplus plus epsilon,
- scaled softplus,
- temperature softplus,
- sigmoid,
- scaled sigmoid,
- square plus epsilon,
- exponential,
- clipped exponential,
- direct positive parameter with projection if feasible.

Required controls:

- matched effective `alpha` initialization,
- comparable initial effective alpha range,
- per-map learning-rate sweep,
- same encoder and decoder except alpha map,
- same parameter budget,
- same seeds,
- same regularization,
- monitored gradient norms,
- monitored effective alpha ranges during training.

Metrics:

- prediction NLL or CE,
- accuracy,
- AUC,
- QWK,
- calibration,
- `alpha`, `beta`, and `theta` recovery,
- linked RMSE,
- slope,
- bias,
- high-alpha rank recovery,
- recovery AUC,
- time-to-threshold.

Predictions:

- Exp should especially help high true-alpha items.
- Exp should improve recovery trajectories, not only endpoint scores.
- Positivity alone should not be sufficient.
- If scaled or temperature softplus matches exp under fair controls, revise the
  claim to "positive-map geometry matters" rather than "exp is uniquely best".

Execution gate after the controlled K=4 run:

- If exp beats smooth non-exp maps by a meaningful margin after matched
  effective initialization and LR tuning, continue to the exp-specific
  geometry-matched control.
- If smooth non-exp maps nearly match exp, do not write an exp-only claim.
  Reframe the result around smooth positive-map geometry, then run the planned
  category extension only to test whether the exp margin grows with `K`.
- If raw, ReLU, or bounded maps fail while smooth positive maps cluster, treat
  the finding as evidence about smoothness, saturation, and effective-alpha
  range before claiming an exponential preconditioner mechanism.

### E4. Initialization and gradient-magnitude controls

Purpose:

Rule out the objections that exp wins only through initialization or larger
gradients.

Tests:

- matched effective alpha initialization,
- per-map learning-rate sweep,
- gradient-norm matched training,
- effective alpha range monitoring,
- clipped exp for unboundedness,
- scaled sigmoid for range restriction,
- temperature softplus for slope tuning.

Key diagnostic:

Compare empirical recovery speed with:

```text
m_g(alpha) = [g'(g^{-1}(alpha))]^2
```

### E5. Map-geometry diagnostic

Purpose:

Explain why smooth positive maps cluster after matched effective initialization
and LR tuning, while raw, ReLU-like, or non-monotone maps are less stable.

Gate:

Run this as an exp-specific diagnostic only if the positive-map ablation leaves
a stable exp advantage worth explaining. If the ablation shows that smooth
positive maps nearly match exp, rewrite this experiment as a broader
map-geometry control rather than a proof of exponential optimality.

No-training diagnostic:

```text
1. Read the controlled alpha-map outputs.
2. Group maps by geometry family:
   exp family
   smooth softplus family
   bounded sigmoid family
   unconstrained raw
   nonsmooth ReLU
   nonmonotone square
3. Compute induced preconditioner values:
   m_g(alpha_init)
   m_g(alpha_p50)
   m_g(alpha_p95)
4. Compare recovery with:
   map family
   effective alpha range
   alpha-head gradient norms
   high-alpha recovery
5. Decide whether m_g alone explains recovery, or whether smoothness,
   monotonicity, saturation, and range are also needed.
```

Prospective direct-alpha conditions:

```text
1. direct alpha with positivity projection
2. direct alpha with exp-equivalent preconditioner
   dot(alpha) = - alpha^2 grad_alpha L
3. direct alpha with softplus-equivalent preconditioner
   dot(alpha) = - (1 - exp(-alpha))^2 grad_alpha L
4. direct alpha with scaled-softplus-equivalent preconditioner
5. direct alpha with bounded-sigmoid-equivalent preconditioner
6. raw/ReLU-like low-smoothness controls
```

The original exp-specific condition remains only as a reference:

```text
alpha = exp(a)
```

Identifiability simplification:

```text
Use the same synthetic GPCM generator.
Freeze theta to the true learner ability used at each response.
Freeze beta to the true item thresholds.
Optimize only one scalar alpha_j per item from response likelihood.
Do not use parameter-recovery loss.
Use the same train split and response-level GPCM likelihood.
Initialize all alpha_j to the same effective alpha.
Compare alpha-space update rules under the same LR grid.
Score direct alpha recovery without latent sign alignment because theta and beta
are fixed to their generating orientation.
```

This simplification removes encoder learning, beta learning, theta drift, item
embeddings, and representation capacity. It answers only whether alpha-space
preconditioning and constraints can reproduce the smooth-map cluster observed
in the neural model. If it cannot, the smooth-map cluster is likely tied to
neural representation or optimizer interaction rather than alpha-space geometry
alone.

Expected result:

If the direct-alpha preconditioned controls reproduce the smooth-map cluster,
the mechanism is strengthened. If only the neural maps cluster, revise toward
architecture, optimizer interaction, finite-time saturation, or learned
representation. If raw/ReLU/non-monotone controls fail in both settings, the
finding supports the broader smooth positive-map geometry claim.

Post-E7 neural isolation, if the direct-alpha control does not reproduce the
neural map separation:

```text
1. learned encoder and learned item embeddings
2. frozen sequence backbone with learned item embeddings
3. learned sequence backbone with frozen item embeddings
4. fully frozen encoder representation
5. raw and ReLU maps with matched effective-alpha initialization
6. clipped raw and clipped ReLU maps with matched output range
7. gradient clipping only as an explicit raw/ReLU stability control
```

This is not a replacement for the positive-map ablation. It tests whether the
remaining neural map effect is caused by representation learning, item-key
learning, raw-head range instability, or optimizer interaction.

Original exp-specific contrast, now downgraded:

```text
If a future condition creates a stable exp gap:
1. alpha = exp(a)
2. direct alpha with positivity projection
3. direct alpha with exp-equivalent preconditioner
   dot(alpha) = - alpha^2 grad_alpha L
4. direct alpha with softplus-equivalent preconditioner
   dot(alpha) = - (1 - exp(-alpha))^2 grad_alpha L
```

### E6. Finite-sample and training-time scaling

Purpose:

Separate finite-time recoverability from asymptotic identifiability.

Sweeps:

- number of learners `N`,
- number of items `Q`,
- sequence length `T`,
- responses per item,
- response sparsity,
- item exposure imbalance,
- noise level,
- true alpha distribution,
- overlap between theta and beta distributions,
- training budget,
- model capacity.

Report:

```text
Delta_alpha(N) = r_alpha_variant(N) - r_alpha_baseline(N)
Delta_alpha(s) = r_alpha_variant(s) - r_alpha_baseline(s)
```

Distinguish:

- sample asymptotics,
- optimization-time asymptotics,
- category-complexity scaling,
- capacity scaling.

Expected result:

The advantage should be strongest under realistic finite-data and finite-training
conditions. If it shrinks with abundant data or long training, that supports the
finite-time framing rather than weakening it.

### E7. Category-complexity sweep

Purpose:

Test whether alpha recovery becomes harder as response-category complexity
grows.

Models:

- 2PL or GPCM with `K=2`,
- GPCM with `K=3,4,5,6,8,11`,
- optional GRM,
- optional NRM.

Metrics:

- alpha recovery,
- beta or threshold recovery,
- theta recovery,
- prediction,
- Fisher ratio or conditioning proxy,
- selective-alpha delta,
- positive-map delta.

Expected result:

As `K` grows, alpha stiffness should grow and selective alpha treatment should
matter more. The exp geometry advantage may also become clearer when high-gain
recovery is the bottleneck.

### E8. Contextual alpha residual diagnostics

Purpose:

Treat `delta[j,t]` as an object of study.

Regress or correlate `delta[j,t]` against:

- `theta_t`,
- `theta_t - beta_j`,
- `abs(theta_t - beta_j)`,
- `p_jt (1 - p_jt)`,
- history length,
- recent correctness,
- exposure count,
- learner ability band,
- item difficulty band.

Interpretation:

| Residual pattern | Interpretation |
|---|---|
| tracks `p(1-p)` | local informativeness correction |
| tracks `theta - beta` | learner-item location modulation |
| tracks history length | exposure or learning effect |
| tracks recent correctness | local dependence or momentum |
| high residual variance | possible identifiability leakage |
| persists under null | finite-data bias or measurement artifact |
| detects planted theta-dependence | directional signal, not necessarily calibrated magnitude |

Required negative control:

Run null synthetic data where true alpha is static. If `delta[j,t]` still has
structure, record the null artifact before interpreting real data.

### E9. Misspecification studies

Purpose:

Find which parameter absorbs which violation when the fitted model is wrong.

Misspecifications:

- generate with GRM and fit GPCM,
- generate with GPCM and fit NRM,
- local dependence on previous response,
- learner response styles,
- item exposure imbalance,
- drifting `theta_t`,
- noisy thresholds,
- differential item functioning,
- threshold disorder.

Report:

- prediction,
- theta recovery,
- beta or threshold recovery,
- alpha recovery,
- contextual residual behavior,
- whether alpha absorbs misspecification,
- whether beta becomes unstable,
- whether prediction improves while recovery worsens.

### E10. GRM and NRM extension

Purpose:

Test whether the parameter taxonomy generalizes beyond GPCM.

Expected taxonomy:

| Type | Examples | Expected treatment |
|---|---|---|
| latent state | `theta` | dynamic state tracking |
| location or threshold | `beta`, GPCM thresholds, GRM thresholds, NRM intercepts `c_k` | static |
| gain or slope | `alpha`, NRM slopes `a_k` | wider, separate, positive or slope-aware geometry |

NRM model:

```text
P(Y = k | theta) = softmax_k(a_ik theta + c_ik)
```

Test whether:

- `a_k` behaves like alpha,
- `c_k` behaves like beta or intercept,
- state-conditioned `a_k` helps prediction but may hurt item recovery,
- item-only slopes stabilize recovery.

GRM test:

- thresholds should behave like static beta,
- discrimination should remain the hard positive gain parameter.

### E11. Real-data evaluation

Purpose:

Support ecological validity without claiming true recovery.

Candidate datasets:

- ASSISTments,
- KDD Cup 2010,
- EdNet when the data-generating process is appropriate,
- ordinal or partial-credit datasets,
- questionnaire-style data if compatible.

Metrics:

- prediction NLL or CE,
- accuracy,
- AUC,
- QWK,
- calibration,
- split-half item-parameter stability,
- random learner-split stability,
- early versus late stability,
- short-history versus long-history stability,
- agreement with classical IRT estimates where available.

Rules:

- Do not claim ground-truth recovery on real data.
- Classical IRT estimates are external references, not truth.
- Real data supports stability and validity, not the core synthetic mechanism.
- If prediction improves while item stability worsens, report it as a
  measurement-validity failure.

## Metrics and Reporting Standards

Prediction metrics:

- NLL or cross entropy,
- accuracy,
- AUC,
- QWK for ordinal responses,
- Brier score,
- ECE,
- category calibration for ordinal or nominal outcomes.

Recovery metrics:

- Pearson correlation,
- Spearman correlation,
- linked RMSE,
- slope after linking,
- bias after linking,
- rank recovery of high-discrimination items,
- Procrustes alignment for multidimensional variants.

Learning-dynamics metrics:

- recovery curves over checkpoints,
- recovery AUC,
- time-to-threshold,
- prediction-recovery phase plots,
- high-alpha versus low-alpha recovery curves,
- raw `a(t)` and effective `alpha(t)` trajectories,
- gradient norms,
- induced preconditioner values.

Real-data validity metrics:

- split-half item-parameter stability,
- random-split stability,
- early/late stability,
- exposure-band stability,
- agreement with classical IRT as external reference.

## Minimum Viable Paper

The smallest serious paper contains:

1. A synthetic GPCM setup with known `theta`, `beta`, and `alpha`.
2. Checkpointed recovery curves for all three parameters.
3. Static, dynamic, selective, and decoupled alpha channel ablations.
4. Capacity-matched static-wide alpha control.
5. Dynamic beta negative control.
6. Positive-map ablation with at least exp, softplus, scaled softplus, ReLU,
   sigmoid, square, and clipped exp.
7. Matched effective alpha initialization.
8. Category sweep across at least `K=2,4,8`.
9. High-alpha recovery diagnostic.
10. Formal positive-map gradient-flow proposition.

## Expanded Thesis Version

The thesis chapter adds:

- GRM and NRM extensions,
- misspecification studies,
- real-data split-half stability,
- Fisher and Hessian diagnostics,
- geometry-matched alpha-space optimizer control,
- contextual residual analysis,
- finite-data versus long-training asymptotic separation,
- external classical IRT comparison.

## Paper Structure

1. Introduction. Prediction is not measurement.
2. Problem setup. Response-level IRT likelihood and no parameter-level
   supervision.
3. Empirical phenomenon. `theta` and `beta` recover quickly, while `alpha` is
   slow, stiff, and sensitive.
4. Theory. IRT gradients, Fisher coupling, and positive-map geometry.
5. Experiments. Recovery trajectories, representation controls, positive maps,
   finite-sample and K-sweeps.
6. Residual analysis. Contextual alpha residuals and null artifacts.
7. Real data. Prediction, calibration, stability, and external references.
8. Discussion. Parameter-specific learning geometry as a measurement-validity
   condition.
9. Limitations. Synthetic recovery, local theory, real-data ground truth, and
   exp non-universality.
10. Conclusion. Neural psychometric models need parameter-specific temporal and
    optimization treatment.

## Acceptance Criteria

The study is ready to write when the repo has:

- checkpointed recovery curves for `theta`, `beta`, and `alpha`,
- capacity-matched representation controls,
- dynamic beta and all-dynamic negative controls,
- positive-map ablation with matched effective initialization,
- per-map learning-rate protocol,
- high-alpha versus low-alpha recovery diagnostics,
- finite-data sweep,
- category-complexity sweep,
- at least one geometry diagnostic testing induced preconditioning against
  neural representation interaction,
- real-data stability checks or a documented reason for deferring them,
- formal positive-map gradient-flow propositions,
- explicit separation of finite-time recoverability from asymptotic
  identifiability.
