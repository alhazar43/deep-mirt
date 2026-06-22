# Theory support for the joint parameter-recovery dynamics (theta, beta, alpha)

A self-contained mathematical support layer for the empirical learning-dynamics
results in `docs/LEARNING_DYNAMICS_STUDY.md`. The role of this document is
modest and local: it derives the information-geometric structure that EXPLAINS
why prediction training recovers the three psychometric parameters at different
rates and qualities under a SHARED item representation, and it states precisely
what that structure does and does not imply. It is a support layer for a
deep-learning paper, not a standalone theorem. Every claim below is marked
Proved, Argued (heuristic), or Empirical.

Sections 2 through 7.5 (P1 to P9) analyze the discrimination parameter alpha:
why it is the low-information slow mode and why giving it its own readout code
buys a convergence-rate advantage. Section 7.6 (P10, P11, P12) completes the
picture into the JOINT three-parameter trade-off the architecture actually
poses. The single item code feeds two consumers, the ability ENCODER and the
alpha/beta READOUTS, and the three parameters want that code's width pulled in
opposite directions: alpha wants it WIDE (P9), ability theta wants it NARROW
(P10), difficulty beta is the indifferent control (P11). A shared code ties the
encoder-input width to the readout width, so the two pressures collide along a
Pareto frontier; decoupling unties them and is Pareto-dominant (P12). The alpha
side (P9) and the theta side (P10) are DISTINCT mechanisms, a transient rate /
conditioning effect on a free item table versus a finite-data
capacity / generalization effect on an amortized readout, and the document keeps
them separate.

This document does not edit or supersede the theory appendix
(`docs/learning_dynamics_toy.md`). The appendix builds an analytically tractable
toy ladder and reports its verdicts; this document extracts the general
information-geometric statements that hold independently of any single toy and
ties each to a settled empirical finding. Where the two overlap, the appendix is
the source of the numerics and this document is the source of the derivations.

Notation. We use standard IRT notation in the math and the plain-word map from
the study throughout: ability `theta` (score), item location `beta`
(difficulty), item discrimination `alpha > 0` (sharpness), number of ordered
response categories `K` (answer levels). The amortized neural model reads
`(theta, alpha, beta)` off a learned encoder of a learner's response history and
emits a response distribution; training minimizes a response-prediction loss
(cross-entropy family) on that distribution, with NO direct supervision on
`(theta, alpha, beta)` and no parameter priors. "Recovery" is sign-aligned rank
agreement (Spearman) between learned and data-generating parameters, hence
invariant to the standard IRT scale and location indeterminacy.

---

## 0. Summary of propositions and their empirical anchors

| # | Proposition | Status | Empirical finding supported |
|---|---|---|---|
| P1 | The 2PL per-parameter gradients are `dL/dtheta = r alpha`, `dL/dbeta = -r alpha`, `dL/dalpha = r (theta - beta)`, with residual `r = p - p*`. The alpha gradient carries the separation lever arm `(theta - beta)`, so it vanishes where `theta ~ beta`. | Proved | Finding 1 (alpha is the hard parameter); the coupling of alpha's gradient to `theta - beta` alignment. |
| P2 | The single-response Fisher informations are `I(theta) = I(beta) = alpha^2 w`, `I(alpha) = (theta - beta)^2 w`, with `w = p(1-p)`. `I(alpha)` is suppressed by the squared lever arm and vanishes at `theta = beta`, exactly where `w` peaks. Alpha is the low-information parameter. | Proved | Finding 1, Finding 2 (prediction depends weakly on alpha). |
| P2-full (full Fisher, alpha is the slow eigen-mode) | The per-response score is the single vector `s = (alpha, theta - beta, -alpha)`, so the per-response Fisher `F_resp = w s s^T` is RANK ONE. The population `(theta, alpha)` block has exact eigenvalues `lambda_pm = (1/2)[(I_tt + I_aa) +- sqrt((I_tt - I_aa)^2 + 4 I_ta^2)]`; the SMALL one is alpha-aligned, `lambda_- ~ I_aa(1 - rho^2) <= I_aa`, since `I_aa = E[w x^2]` is suppressed (`w` peaks where `x^2` vanishes), `I_tt/I_aa ~ alpha^2/Var_w(x)`. Off-diagonal coupling `rho` deepens it. | Proved (matrix, eigenvalues exact; suppression Argued in the constant) | Finding 1, Finding 2; the DERIVED basis for "alpha is the slow mode" (Section 3.5). |
| P3 | Prediction sensitivity to alpha is `dp/dalpha = (theta - beta) w`, which is `I(alpha)^{1/2}`-scaled. The prediction is first-order blind to alpha wherever responses concentrate near `theta = beta`. | Proved | Finding 2 (weak alpha dependence of the prediction). |
| P4 (rate, not endpoint) | Near an identifiable optimum, gradient flow on each direction decays at a rate set by that direction's curvature, which is its Fisher information. Low Fisher therefore sets the RECOVERY RATE. At the population limit, every reachable zero-residual direction still converges to truth, so low Fisher does NOT set an endpoint bias. | Proved (local) + Argued (global, via the free-table invariant) | Finding 3 (a finite-data SPEED/rank effect with no endpoint advantage; ties at convergence). |
| P5 (finite-data bias is errors-in-variables) | At finite repetitions the amortizer input is a noisy encoding of `theta*`; through the bilinear `z = alpha(theta - beta)` this contaminates alpha. The bias scales with the amortizer input noise and vanishes as repetitions to infinity. | Argued + Empirical (toy) | Finding 3, Finding 4 (the shortfall behaves like a finite-data errors-in-variables effect that vanishes as repetitions grow). |
| P6 (K worsens conditioning) | For GPCM, `I(alpha)` rises in absolute terms with K but `I(theta)` rises faster, so the stiffness ratio `I(theta)/I(alpha)` grows monotonically with K. The shared-code flow becomes more ill-conditioned, so alpha's rate disadvantage grows with K. | Proved (Fisher forms) + Empirical (ratio table) | Finding 3 (the benefit of an alpha-specific readout GROWS with K) and the K=2 sign flip. |
| P7 (smooth-map recovery equivalence) | Any `C^1` map induces the preconditioner `m_g(alpha) = [g'(g^{-1}(alpha))]^2` (Prop P7-0, research-plan Prop 1). For smooth strictly-monotone positive `g`, `m_g > 0` makes the flow a strictly-increasing TIME-REPARAMETERIZATION of one canonical curve, so (i) endpoint and (ii) Spearman rank are map-INVARIANT with no tuning, and (iii) the residual SPEED is one scalar `eta_g m_g(alpha*)` absorbed by per-map LR. So exp is not special. The two exceptions break a hypothesis of the theorem: ReLU has `m_g = 0` on a dead zone (flow halts, unrescalable), square `a^2` is non-injective (two-valued `g^{-1}`, sign-fold saddle). | Proved (endpoint + rank invariance via ODE uniqueness; both exceptions exact) + Argued (residual speed constant) | Finding 4 (exp is not special; smooth maps tie; only non-smooth/non-monotone maps lag); the j5 convergence profiles. |
| P8 (scalar preconditioning is insufficient) | A scalar `alpha`-space preconditioner acts on a single direction and cannot reproduce the recovery effect that lives in the coupling between the shared representation directions. With `theta, beta` frozen and only the scalar `alpha_j` optimized, all reasonable `alpha`-space update rules converge to the same band. | Argued + Empirical (direct-alpha control) | Finding 4 (the scalar preconditioner-only explanation is refuted by the direct-alpha control). |
| P9 (representation coupling is the mechanism) | The validated accelerator is REPRESENTATION DECOUPLING, not scalar reparameterization. The shared item-code Gauss-Newton block is the explicit sum `H^sh_j = (g'_j)^2 I(alpha_j)(a a^T) + alpha_j^2 sum_i w_ij v_{ij}v_{ij}^T + alpha_j^2 I_w (b b^T)`; its two eigen-directions have curvatures `c_theta ~ I(theta_j)` (steep) and `c_alpha = (g'_j)^2 I(alpha_j)` (flat), so `kappa^sh_j = [I(theta)/I(alpha)][|v|^2/(g'_j)^2] ~ I(theta)/I(alpha) >> 1`. Plain GD with one step size (`eta < 2/lambda_max`) resolves alpha in `t^sh = kappa log(1/epsilon)` iterations; the decoupled alpha block drops the `v_{ij}` term, so `kappa^dc = O(1)` and `t^dc = O(log(1/epsilon))`. Speedup `= O(kappa)`, tolerance-independent, growing with K via `kappa(K)`. Same fixed point (free-table invariant), a pure rate / early-stopping effect. | Proved (block Hessian, condition number, GD iteration count exact under A1-A6) + Argued (global endpoint via P4b, A5/A6) + Empirical (rung-7, K-sweep, N-sweep) | Replaces the REFUTED `alpha^2`-preconditioner proposition; anchors the gate (d1), trajectory (d2), 28x gradient (d3), stiffness rank 0.89 (d4), N-sweep (d5). |
| P10 (theta wants the encoder input narrow) | The amortized ability readout pools item codes through the encoder. Widening the code that enters the encoder raises the VARIANCE of `theta_hat` by giving the encoder degrees of freedom to absorb item-specific idiosyncrasy (item identity, learner-by-item response noise) into the ability state. So theta recovery DEGRADES with encoder-input width and degrades FURTHER along the training trajectory as the optimizer drives the over-parameterized encoder deeper into interpolation. This is a finite-data capacity / generalization effect on an amortized readout, DISTINCT in kind from alpha's conditioning (P9), and OUTSIDE the free-table invariant (P4b) because theta is read by an encoder, not a free per-item table. | Argued (linear-amortizer bias-variance `O(W/n)`) + Argued (lift to the LSTM) + Empirical (theta 0.97->0.88 with width; 0.97->0.68 with training; theta-pathway gradient on the shared code grows ~28x) | Finding (gate Pareto theta side); the d2 theta-overfit trajectory; the d3 28x gradient growth as the theta side of the curvature asymmetry. |
| P11 (beta is the indifferent control) | Difficulty beta is a high-Fisher location parameter (`I(beta) = alpha^2 w`, same order as I(theta)), read DIRECTLY off the item code with no amortized-pooling pathway. So beta neither NEEDS width (high Fisher pins it from a thin code, unlike alpha) nor is HURT by it (no learner-by-item leakage channel into a per-learner latent, unlike theta). Beta is therefore the negative control on BOTH pressures: a richer / wider / state-conditioned treatment is a no-op for beta. | Proved (Fisher form + readout structure) + Empirical (delta_beta ~ 0 across K, beta ~0.98 everywhere) | Finding (the beta negative control: decoupling and dynamic treatment lift alpha but do essentially nothing for beta, delta_beta ~ +0.003). |
| P12 (shared ties the widths -> Pareto frontier; decoupling unties them -> Pareto-dominant) | A SHARED code uses ONE width W for both the encoder input and the readout code. Alpha's pressure (P9) wants W large, theta's pressure (P10) wants W small, beta is flat, so sweeping the single knob W traces a Pareto frontier in (theta, alpha): theta down, alpha up, beta flat. DECOUPLING introduces a SECOND knob, a narrow encoder-input width and a separate wide readout width, so alpha gets its capacity without widening the encoder. The decoupled point dominates the shared frontier (high theta AND high alpha): it ESCAPES the frontier rather than moving along it. | Proved (two independent knobs dominate one coupled knob given opposing single-knob gradients) + Empirical (the gate: shared frontier theta 0.97->0.88 / alpha 0.66->0.91; decoupled sits above it; matched-total-capacity controlled) | Finding (the Phase-0 gate Pareto and its decoupled dominance at matched total capacity). |

The free-table invariant (Section 4.3) is the single load-bearing structural
fact behind P4 through P9: as long as the per-item parameters form a free table
that can reach the zero-residual fit `p = p*`, every gradient pull is linear in
the residual and all pulls vanish together at the reachable optimum, so the
endpoint is invariant to representation choice and to smooth reparameterization.
Everything interesting is therefore a transient (rate) effect, not an endpoint
(bias) effect.

---

## 1. Setup and the prediction objective

A single response is indexed by learner `i` and item `j`. The 2PL (binary,
`K = 2`) response model is

```
z_ij = alpha_j (theta_i - beta_j),     p_ij = sigma(z_ij) = 1 / (1 + e^{-z_ij}),
```

with `alpha_j > 0`. The training objective is the response-prediction loss, the
cross-entropy between the observed response and the model probability. Writing
`p*_ij` for the data-generating probability and taking the population objective
(expectation over responses), the loss is

```
L = E_{i,j} [ -p*_ij log p_ij - (1 - p*_ij) log(1 - p_ij) ].
```

`L` is minimized exactly when `p_ij = p*_ij` for every `(i,j)`; its value there
is the irreducible Bernoulli entropy floor. This is a PREDICTION objective: the
triple `(theta, alpha, beta)` is the structured route to the predicted
probability, never a supervised target. The loss can only pin a parameter
through that parameter's leverage on `p_ij`, which is the bridge to Fisher
information used throughout.

We use the logit residual

```
r_ij = p_ij - p*_ij = dL/dz_ij    (up to the averaging constant),
```

the central object: every per-parameter gradient and every shared-representation
pull below is linear in `r`. The form `r = p - p*` (per sample `r = p - y`) is the
canonical-link identity, sigmoid/softmax paired with cross-entropy gives
`dL/dz = p - y`; a non-matched loss (squared error, or a reweighted ordinal loss)
replaces `r` by a reweighted residual `s(z)(p - y)`, which rescales the constants
below but leaves the residual-linear structure intact. The clean `r = p - y` form,
and hence the tidy Fisher identities, are specific to the cross-entropy objective.

---

## 2. The 2PL gradient structure (P1)

**Proposition P1 (per-parameter gradients).** With `dz/dtheta = alpha`,
`dz/dbeta = -alpha`, `dz/dalpha = theta - beta`, the chain rule gives

```
dL/dtheta_i = (1/J)  sum_j  r_ij alpha_j                       (ability,        location-like)
dL/dbeta_j  = -(1/N) sum_i  r_ij alpha_j                       (difficulty,     location-like)
dL/dalpha_j =  (1/N) sum_i  r_ij (theta_i - beta_j).           (discrimination, slope-like)
```

*Proof.* Direct differentiation of `L` through `z_ij`, using `dL/dz_ij = r_ij`.
∎

Three structural readings, all consequential for the dynamics.

1. **Location vs slope.** `theta` and `beta` enter through the multiplier
   `alpha_j`, which is positive and `O(1)`. `alpha` enters through the SEPARATION
   `theta_i - beta_j`, the signed lever arm between learner and item. `theta` and
   `beta` are location-like; `alpha` is slope-like.

2. **Alpha's gradient couples to alignment.** The discrimination gradient is a
   residual-weighted sum of `theta_i - beta_j`. A response from a learner well
   matched to an item (`theta_i ~ beta_j`) contributes almost nothing to
   `dL/dalpha_j`, regardless of how large its residual is. So the alpha signal is
   carried only by MISMATCHED responses, and its magnitude is set by the current
   quality of the `theta` and `beta` estimates (a mis-located `theta` or `beta`
   feeds the wrong lever arm into the alpha update). This is the precise sense in
   which "alpha recovery is coupled to the alignment of theta and beta."

3. **A reparameterization detail.** If `alpha_j = g(a_j)` for a positive map `g`,
   then `dL/da_j = g'(a_j) dL/dalpha_j`. For `g = exp`, `dL/da_j = alpha_j
   dL/dalpha_j`, so the raw-space update is the alpha-space gradient scaled by
   `alpha_j`. This scaling is exactly the preconditioner analyzed in Section 6,
   and it is the reason a positive map is not a neutral choice in raw space (P7).

---

## 3. Fisher information and prediction sensitivity (P2, P3)

The connection between "alpha is hard to recover" and the dynamics runs through
Fisher information, which here means the curvature of the prediction loss in each
parameter, equivalently the squared sensitivity of the predicted probability.

**Proposition P2 (single-response Fisher informations, 2PL).** With Bernoulli
variance weight `w_ij = p_ij(1 - p_ij)`,

```
I(theta_i) = sum_j w_ij alpha_j^2          (sum over the learner's items)
I(beta_j)  = sum_i w_ij alpha_j^2          (sum over the item's learners)
I(alpha_j) = sum_i w_ij (theta_i - beta_j)^2.
```

*Proof.* The Fisher information of a Bernoulli logit model in a parameter `phi`
is `E[(dz/dphi)^2 w]`, with `w = p(1-p)` the Bernoulli variance. Substitute
`dz/dtheta = alpha`, `dz/dbeta = -alpha`, `dz/dalpha = theta - beta`. ∎

**The alpha suppression (the heart of the matter).** Two facts combine.

- `w_ij = p_ij(1 - p_ij)` is maximized at `p_ij = 1/2`, i.e. at `z_ij = 0`, i.e.
  at `theta_i = beta_j`. The MOST informative responses for everything else are
  the well-matched ones.
- `I(alpha_j)` carries the factor `(theta_i - beta_j)^2`, which VANISHES at
  exactly those responses. So `I(alpha)` is a product of a large weight and a
  small lever arm. `I(theta)` and `I(beta)` carry the weight `w` with the `O(1)`
  factor `alpha^2` and no vanishing companion.

The targeted, adaptively-selected responses an assessment concentrates near
`theta ~ beta` (where the outcome is most uncertain and most diagnostic of
ability) are precisely the responses that carry the least information about
sharpness. This is the textbook reason discrimination is the hardest 2PL
parameter to estimate, restated as a property of the prediction loss curvature.
It is classical IRT, not a new claim; we reuse it as the input to the dynamics.

**Proposition P3 (prediction sensitivity to alpha).** The first-order
sensitivity of the predicted probability to alpha is

```
dp_ij/dalpha_j = (theta_i - beta_j) w_ij,        (dp/dz) = w,
```

so `|dp/dalpha| = |theta - beta| w = I(alpha)^{1/2}` per response (up to sign).

*Proof.* `dp/dalpha = (dp/dz)(dz/dalpha) = w (theta - beta)`. ∎

P3 is the support for Finding 2 ("the prediction depends weakly on alpha"). The
prediction is first-order INSENSITIVE to alpha wherever `theta ~ beta`; a finite
change in sharpness barely moves `p` there. A prediction-trained model has no
gradient pressure to pin alpha in that regime, because moving alpha does not
reduce the prediction loss. The dependence does not merely "shrink as K grows" by
assumption; Section 5 derives the K-scaling of this sensitivity from the GPCM
Fisher forms.

**Caveat on accumulation.** Each alpha PARAMETER is informed by the item's whole
learner population (`sum_i`), so with enough learners and enough spread in
`theta - beta` the per-item alpha is well determined at the OPTIMUM. P2/P3 are
statements about the per-response information density and the prediction
sensitivity, which govern the RATE at which the optimizer resolves alpha and the
finite-sample regime, not an in-principle non-identifiability. This distinction
is the subject of Section 4.

### 3.5 The full per-response Fisher matrix and the alpha slow mode (P2-full)

P2 gives the DIAGONAL of the Fisher information, one number per parameter. The
diagonal alone cannot say which DIRECTION in `(theta, alpha, beta)` space is the
slow mode, because the parameters are coupled: the off-diagonal blocks tilt the
eigenvectors away from the coordinate axes. The claim "alpha is the slow mode" is
a statement about the eigenstructure of the FULL Fisher matrix, so we derive that
matrix and its eigenvalues with every step shown. This matches the full-Fisher
form (including the off-diagonals `I_atheta = E[alpha x w]`,
`I_abeta = -E[alpha x w]`) carried by the integrated research plan, Section 5.4.

**The per-response score is a single vector (rank-one Fisher).** Fix one response
of learner `i` to item `j` and drop the indices. Write `x = theta - beta` for the
lever arm and `z = alpha x` for the logit. The per-response log-likelihood is
`ell = y log p + (1-y) log(1-p)` with `p = sigma(z)`, and `dell/dz = y - p =
-r`, where `r = p - p*` is the residual (here `p*` is the realized outcome `y` for
a single draw; in expectation `E[y] = p*`). The score in the parameter triple
`phi = (theta, alpha, beta)` is, by the chain rule `dell/dphi = (dell/dz)(dz/dphi)`,

```
dz/dphi = (dz/dtheta, dz/dalpha, dz/dbeta) = (alpha, x, -alpha) =: s,
grad_phi ell = (y - p) s = (y - p) (alpha, x, -alpha).
```

So the score is `(y - p)` times the SINGLE fixed vector `s = (alpha, x, -alpha)`.
Every per-response gradient points along `s` (up or down by the sign of the
residual): one informative direction per response.

**Proposition P2-full (the per-response Fisher is rank one).** The per-response
Fisher information is the variance of the score, `F_resp = E_y[(grad ell)(grad
ell)^T]` at the true `p`. Since `grad ell = (y - p) s` with `s` deterministic
given the parameters, and `Var_y(y - p) = p(1-p) = w`,

```
F_resp = E_y[(y - p)^2] s s^T = w * s s^T
       = w (alpha, x, -alpha)(alpha, x, -alpha)^T

         [  alpha^2     alpha x    -alpha^2  ]
       = [  alpha x      x^2       -alpha x  ] * w.
         [ -alpha^2    -alpha x     alpha^2  ]
```

*Proof.* `F_resp = E[(y-p)^2] s s^T` because `s` is non-random given `(theta,
alpha, beta)`; `E[(y-p)^2] = Var(y) = w` for a Bernoulli with mean `p`. Expand the
outer product `s s^T` with `s = (alpha, x, -alpha)`. ∎

The matrix `s s^T` is rank ONE by construction (an outer product of a single
vector), so `F_resp` has exactly one nonzero eigenvalue, `w |s|^2 = w (2 alpha^2 +
x^2)`, with eigenvector `s`. A single response informs exactly one direction in
parameter space; the other two directions get ZERO information from it. This is
the sharpest possible statement of why per-response information is the binding
constraint: a response is a rank-one measurement.

**The population Fisher is the GPT matrix.** Summing (averaging) `F_resp` over the
responses of the population gives the population Fisher `F = E[w s s^T]`. Reading
off the entries (the expectation now couples `alpha`, `x`, `w` across responses),

```
I_tt = E[alpha^2 w]    I_ta = E[alpha x w]    I_tb = -E[alpha^2 w] = -I_tt
I_ta = E[alpha x w]    I_aa = E[x^2 w]        I_ab = -E[alpha x w] = -I_ta
I_tb = -E[alpha^2 w]   I_ab = -E[alpha x w]   I_bb = E[alpha^2 w] = I_tt
```

These are EXACTLY the off-diagonal forms `I_atheta = E[alpha x w]`,
`I_abeta = -E[alpha x w]` of the research plan, plus the diagonal of P2 (`I_tt =
I_bb = E[alpha^2 w]`, `I_aa = E[x^2 w]`). The `theta` and `beta` rows are negatives
of each other in the `alpha` coupling and equal on the diagonal because
`dz/dtheta = +alpha` and `dz/dbeta = -alpha` differ only in sign; this sign
structure is the gauge direction of Section 8 and is why the full `3x3 F` is itself
rank-deficient along the `(1, 0, 1)`-type gauge combination. We analyze the
NON-gauge content, the `(theta, alpha)` block, which is what the dynamics resolve.

**The `(theta, alpha)` 2x2 block: explicit eigenvalues and condition number.**
Take the leading `2x2` block (the `(beta)` direction is the location partner of
`theta`, treated identically in Section 4; isolating `(theta, alpha)` is the
minimal coupled subsystem),

```
F_2 = [ I_tt   I_ta ]      I_tt = E[alpha^2 w],  I_aa = E[x^2 w],  I_ta = E[alpha x w].
      [ I_ta   I_aa ]
```

The eigenvalues solve `det(F_2 - lambda I) = 0`, i.e. `lambda^2 - (I_tt + I_aa)
lambda + (I_tt I_aa - I_ta^2) = 0`, giving exactly

```
lambda_pm = (1/2) [ (I_tt + I_aa) +- sqrt( (I_tt - I_aa)^2 + 4 I_ta^2 ) ].
```

Both eigenvalues are real and positive (`F_2` is a sum of PSD rank-one terms `w s
s^T` and is positive definite as long as the population spans both `theta` and
`alpha` directions, i.e. there is at least one response with `x != 0`). The
condition number is

```
kappa_2 = lambda_+ / lambda_-
        = [ (I_tt + I_aa) + sqrt((I_tt - I_aa)^2 + 4 I_ta^2) ]
          / [ (I_tt + I_aa) - sqrt((I_tt - I_aa)^2 + 4 I_ta^2) ].
```

**Why the small eigenvalue is the alpha-aligned one.** Consider the regime that
holds for IRT, `I_aa << I_tt` (the lever-arm information is suppressed, quantified
below), and `|I_ta|` bounded by Cauchy-Schwarz, `I_ta^2 <= I_tt I_aa` (with
equality only if `alpha` and `x` are perfectly correlated under the `w`-weighted
measure, which they are not). Then to leading order in the small ratio `I_aa/I_tt`,
expand the square root: `sqrt((I_tt - I_aa)^2 + 4 I_ta^2) = (I_tt - I_aa)
sqrt(1 + 4 I_ta^2/(I_tt - I_aa)^2) = (I_tt - I_aa) + 2 I_ta^2/(I_tt - I_aa) +
O((I_aa/I_tt)^2)`. Substituting,

```
lambda_+ = I_tt + I_ta^2 / I_tt + O(I_aa^2 / I_tt)         (the theta-aligned mode)
lambda_- = I_aa - I_ta^2 / I_tt + O(I_aa^2 / I_tt) = I_aa (1 - rho^2) + ...
```

where `rho^2 = I_ta^2 / (I_tt I_aa) <= 1` is the squared `w`-weighted correlation
between `alpha` and `x`. So:

- The LARGE eigenvalue is `lambda_+ ~ I_tt = E[alpha^2 w]`, the ability/location
  curvature. Its eigenvector is the `theta`-aligned direction (tilted by the small
  `I_ta`).
- The SMALL eigenvalue is `lambda_- ~ I_aa (1 - rho^2) <= I_aa = E[x^2 w]`, the
  lever-arm curvature, FURTHER reduced by the coupling factor `(1 - rho^2)`. Its
  eigenvector is the `alpha`-aligned direction. Coupling never helps alpha: it
  drains a further `rho^2` fraction of alpha's already-small curvature into the
  theta mode.

Hence the condition number is, to leading order,

```
kappa_2 = lambda_+ / lambda_-  =  I_tt / [ I_aa (1 - rho^2) ]  >=  I_tt / I_aa  =  I(theta) / I(alpha),
```

so `kappa_2` is at least the diagonal stiffness `I(theta)/I(alpha)` and exceeds it
whenever the parameters are correlated (`rho != 0`). The diagonal ratio used
elsewhere in this document is the OPTIMISTIC (decoupling-free) bound on the true
coupled stiffness; the off-diagonals make the real flow stiffer, not less.

**Quantifying the lever-arm suppression `I_aa << I_tt`.** The suppression is not
assumed; it follows from the structure of `w` and `x`. Write `x = theta - beta`
and consider `I_aa = E[w x^2]`. Because `w` depends on `x` through `z = alpha x`,
a product-of-expectations split is unavailable, so we bound `I_aa` directly from
the shape of `w`. The weight `w(z) = sigma(z)(1 - sigma(z))` is a bell
peaked at `z = 0` (`w = 1/4`) and decaying like `e^{-|z|}` in the tails, so as a
function of `x` it is sharply peaked at `x = 0` with width `~ 1/alpha`. Therefore
`w x^2` is the product of a function peaked at `x = 0` and a function (`x^2`)
ZEROED at `x = 0`: the integrand is suppressed exactly where the measure
concentrates. Concretely, treating the `w`-weight as concentrating mass near
`x = 0`,

```
I_aa = E[w x^2] ~ E[w] * Var_w(x) ,      I_tt = E[w alpha^2] = alpha^2 E[w]  (alpha ~ const),
=> I_tt / I_aa ~ alpha^2 / Var_w(x),
```

where `Var_w(x)` is the `w`-weighted spread of the lever arm. The suppression is
therefore exactly the smallness of the `w`-weighted lever-arm spread relative to
the squared discrimination. For an adaptively administered assessment that targets
`x = theta - beta ~ 0` (maximally informative items), `Var_w(x)` is driven small
ON PURPOSE, so `I_tt / I_aa` is large by design: the data-collection policy that
maximizes ability information minimizes discrimination information. For a fixed
broad item bank `Var_w(x)` is `O(1)` and the ratio is `O(alpha^2)`, still `> 1`
for `alpha > Var_w(x)^{1/2}`. Either way `I_aa < I_tt` and alpha is the slow mode.

*Status: Proved* (the rank-one per-response Fisher and the population matrix are
exact; the `2x2` eigenvalues are exact; the leading-order `kappa_2` expansion is
exact to `O(I_aa/I_tt)`; the `I_tt/I_aa ~ alpha^2/Var_w(x)` form is the
concentration estimate, *Argued* in the constant, exact in the suppression
mechanism). This is the derived replacement for the diagonal-only "alpha is slow"
assertion: the slow eigen-DIRECTION is alpha-aligned because the lever-arm
information `I_aa = E[w x^2]` is suppressed by `w` concentrating where `x^2`
vanishes, and the coupling `rho` only deepens the suppression.

---

## 4. Information sets the rate, not the endpoint (P4)

This is the central and most carefully delimited claim. We separate a local
linear statement (proved) from the global endpoint statement (the free-table
invariant), and we state explicitly where each holds.

### 4.1 Local convergence rate

Let `phi` collect the free parameters and `phi*` an identifiable optimum (gauge
fixed; see Section 8). Near `phi*` the population prediction loss is locally
quadratic with Hessian `H`, and at a well-specified optimum the Gauss-Newton
identity makes `H` equal to the Fisher information matrix `F` to leading order
(both equal `E[J^T diag(w) J]` for the per-response Jacobian `J` of `z` in
`phi`). Gradient flow `d phi/dt = -grad L` linearizes to

```
d(delta)/dt = -F delta,    delta = phi - phi*,
```

so in the eigenbasis of `F` each mode `c` decays as `delta_c(t) = delta_c(0)
e^{-lambda_c t}`, with `lambda_c` the eigenvalue (the curvature, the Fisher
information of that direction).

**Proposition P4a (rate is set by Fisher).** A direction's recovery time scale is
`tau_c = 1/lambda_c`. Low-Fisher directions are slow modes; high-Fisher
directions are fast modes. For the IRT parameters this orders the modes as
`theta, beta` fast and `alpha` slow, since `I(alpha)` is the suppressed quantity
of P2. The condition number of the flow is `kappa = lambda_max/lambda_min ~
I(theta)/I(alpha)`; a larger `kappa` means a stiffer flow and a longer wait for
the slow (alpha) mode to resolve. The `~` here is the DIAGONAL value; Section 3.5
gives the exact `(theta, alpha)`-block eigenvalue ratio `kappa_2 = I_tt/[I_aa(1 -
rho^2)] >= I_tt/I_aa`, so the diagonal `I(theta)/I(alpha)` is the optimistic lower
bound and off-diagonal coupling makes the true flow stiffer, never less.

*Status: Proved* (it is the standard linearization of gradient flow at a
quadratic minimum; the Gauss-Newton/Fisher equality is exact at a well-specified
zero-residual optimum and approximate otherwise).

This is exactly the scalar law `t = O(tau/s)` the methodology survey flagged as
the surviving result: recovery time is inversely proportional to the parameter's
information. Note this is a statement about HOW FAST each mode reaches `phi*`,
not about WHERE the flow goes. Both arise from the SAME limit `t -> infinity`:
every mode with `lambda_c > 0` reaches `phi*`, just at different speeds.

### 4.2 Why the slow mode is not a biased mode

A slow mode decays slowly but it still decays to zero. Slowness becomes bias only
if the slow direction cannot reach truth, i.e. if the optimum itself is displaced
from truth. P2's suppression of `I(alpha)` lowers `lambda_alpha` and thus
lengthens `tau_alpha`; it does not move `phi*`. So at infinite training time, a
low-Fisher alpha still converges to its true value (up to gauge). The shortfall
is a finite-TRAINING-time phenomenon.

This is the mathematical content of "low Fisher sets the rate, not the endpoint."
It is exactly consistent with Finding 3: an alpha-specific readout buys a
finite-budget SPEED and RANK advantage, and the configurations TIE when trained
to convergence.

### 4.3 The free-table invariant (the endpoint statement)

The local argument shows the optimum is reached if it is identifiable. The global
statement is why the optimum is the SAME regardless of representation choice or
smooth reparameterization.

**Proposition P4b (free-table invariant).** Suppose the per-item parameters are
produced by a free table (or any readout) rich enough to express the
zero-residual fit `p_ij = p*_ij` for all `i,j`. Then every gradient pull on the
representation is linear in the residual `r_ij = p_ij - p*_ij`, and at the
zero-residual point `r = 0` all pulls vanish simultaneously. Hence the
zero-residual set is a stationary point of the flow, it is the global minimizer
(the entropy floor), and it is invariant to:

- which readout vectors share the code (sharing vs decoupling), and
- any smooth, strictly monotone reparameterization of the parameters
  (`alpha = g(a)` for smooth invertible `g`), since the chain-rule factor
  multiplies a pull that is already zero at `r = 0`.

*Proof sketch.* By construction the loss gradient with respect to any internal
quantity factors through `dL/dz_ij = r_ij` and a Jacobian, so it is linear and
homogeneous in `r`. At `r = 0` every such gradient is zero. The expressivity
hypothesis guarantees `r = 0` is attainable, so it is the minimizer and is
stationary. Reparameterization by smooth invertible `g` multiplies each gradient
by a nonsingular factor that does not change its zero set. ∎

This is the structural reason the endpoint carries no representation-choice or
positive-map advantage (the negative results of Section 6 and 7), while the
transient does (P4a). The invariant is what forces every interesting effect to be
a rate effect. It is verified to machine precision and across many toy variants
in `docs/learning_dynamics_toy.md` (Sections 2.4, 9, 10), where shared and
decoupled configurations are byte-identical at the population optimum.

**Where the invariant can break (honest limits).** The invariant requires (i)
enough representational rank to express `p = p*` (an expressivity wall appears
below rank 2 for 2PL, rank K for GPCM; Section 5), and (ii) reaching the
zero-residual set, which a stiff flow may not do within a finite budget (that is
the whole point of P4a). It says nothing about the transient, and it is the
transient the real, early-stopped model lives in.

### 4.4 The finite-data errors-in-variables mechanism (P5)

Beyond finite training time there is a finite-DATA effect, and it has a distinct,
cleanly identified mechanism.

**Proposition P5 (errors-in-variables on alpha).** In the amortized model
`theta_i` is read from an encoding of the learner's responses. At finite
repetitions that encoding is a noisy estimate `theta_hat_i = theta*_i +
epsilon_i`. Because `z_ij = alpha_j(theta_i - beta_j)` is BILINEAR in `alpha` and
`theta`, noise in `theta_hat` propagates multiplicatively into the alpha
estimate: fitting `z` with a noisy `theta` regressor is a classical
errors-in-variables problem, which attenuates and biases the slope `alpha`. The
bias scales with the variance of `epsilon` (the amortizer input noise) and
vanishes as repetitions grow, since `theta_hat -> theta*`.

*Status: Argued* (the errors-in-variables attenuation of a bilinear slope is
classical) *and Empirical* (the toy measures `corr(amortizer input, theta*)`
rising `0.36 -> 1.00` over `reps = 1..inf` with the gauge-fixed alpha bias
decaying in lockstep `-3.33 -> -0.003`; `docs/learning_dynamics_toy.md` Sec 9.3).

P5 supports Finding 4's "errors-in-variables effect that vanishes as repetitions
grow." Two points keep it honest.

- It is a FINITE-DATA bias, distinct from the finite-TRAINING-time rate effect of
  P4a. They are different axes (data vs steps) and they vanish at different
  limits (`reps -> inf` vs `steps -> inf`). Real models sit at finite-both, so
  both contribute.
- In the tractable toys this finite-data bias is the SAME for shared and
  decoupled representations, because both feed the amortizer the same noisy
  input. The empirical decoupling RANK benefit is therefore NOT this bias; it is
  the rate effect of P4a, surfaced because the stiffer flow lags more at a fixed
  budget. The toy ladder is explicit that the strong "population-limit dynamics
  law" does not exist in tractable form, and this document does not claim one.

---

## 5. The GPCM generalization and why K worsens alpha's conditioning (P6)

### 5.1 GPCM gradient and Fisher structure

The graded model (GPCM, `K` ordered categories `k = 0..K-1`) has item
discrimination `alpha_j > 0` and step thresholds `beta_{j,1..K-1}`. With the
partial sums `B_{jk} = sum_{c<=k} beta_{j,c}` the category log-odds are

```
psi_{ijk} = alpha_j (k theta_i - B_{jk}),   psi_{ij0} = 0,
P(Y_ij = k) = exp(psi_{ijk}) / sum_m exp(psi_{ijm}).
```

Writing the per-category residual `R_{ijk} = P_model(k) - P_true(k)`, the
population gradients are

```
dL/dtheta_i    = (1/NJ) sum_j alpha_j sum_k R_{ijk} k
dL/dalpha_j    = (1/NJ) sum_i sum_k R_{ijk} (k theta_i - B_{jk})
dL/dbeta_{j,c} = -(1/NJ) alpha_j sum_i sum_{k>=c} R_{ijk}.
```

The alpha gradient again carries a separation-like lever arm `(k theta - B_k)`,
the polytomous analog of `(theta - beta)`: alpha's signal is carried by the
spread of the natural statistic `k theta - B_k`, suppressed where the category
distribution concentrates. The structure of P1 transfers with one alpha pathway
and `K-1` threshold pathways. (The shared-representation decomposition into `K`
pathway directions, and the resulting expressivity wall at code rank `< K`, are
derived and verified in `docs/learning_dynamics_toy.md` Sec 11.3-11.4; they are
the polytomous version of the 2PL rank-2 wall and are not repeated here.)

**Proposition P6a (GPCM single-response Fisher).** From the natural-parameter
score with category distribution `p`,

```
I(theta)   = alpha^2 Var_p(k)                     (variance of the category index)
I(alpha)   = Var_p(k theta - B_k)                 (variance of the natural statistic)
I(beta_c)  = alpha^2 P_{>=c}(1 - P_{>=c}).
```

*Proof.* The GPCM is an exponential family in the natural statistics; the Fisher
information in a parameter is the variance of the corresponding score, which for
`theta` is `alpha k`, for `alpha` is `(k theta - B_k)`, and for the cumulative
threshold gives the binary-type form. ∎

**Consistency check (Proved).** At `K = 2`, `Var_p(k) = p(1-p)` and these reduce
EXACTLY to the 2PL forms `I(theta) = I(beta_1) = alpha^2 p(1-p)`, `I(alpha) =
(theta - beta)^2 p(1-p)`. The GPCM forms are a strict generalization.

### 5.2 The two faces of K, and why the conditioning worsens

Computing the per-response Fisher means at matched truth as K increases gives two
trends that point opposite ways (numbers from `docs/learning_dynamics_toy.md`
Sec 11.5, reproduced here as the empirical anchor for P6):

```
 K   I(theta)   I(alpha)   I(theta)/I(alpha)
 2    0.187      0.181        1.03
 3    0.409      0.344        1.19
 4    0.600      0.414        1.45
 5    0.832      0.429        1.94
 6    1.098      0.479        2.29
```

- **Absolute (the standalone view).** `I(alpha)` RISES with K. More categories
  give each response more discrimination signal, so alpha's own identifiability
  improves and the per-response finite-sample MLE bias on alpha is milder. The
  naive expectation "more categories make alpha easier" is correct in this sense.

- **Relative (the dynamics view, P6).** `I(theta)` rises FASTER, because the
  category index `k` spreads over a wider range and `Var_p(k)` grows. Hence the
  stiffness ratio

  ```
  kappa(K) = I(theta) / I(alpha)
  ```

  climbs monotonically with K. By P4a this is the condition number of the
  shared-representation flow, so the flow gets STIFFER with K and alpha (the slow
  mode) falls further behind theta at any fixed training budget.

**Proposition P6 (K worsens alpha's conditioning).** The recovery-rate
disadvantage of alpha relative to theta, governed by `kappa(K) =
I(theta)/I(alpha)`, increases with K. Therefore a representational change that
relieves alpha's rate disadvantage (giving alpha its own readout) yields a
benefit that GROWS with K, and at `K = 2`, where `kappa ~ 1` and alpha is
relatively well conditioned, the benefit is smallest and can be net negative once
the extra readout's added variance is counted.

*Status: Proved* (the Fisher forms P6a) *plus Empirical* (the monotone `kappa(K)`
table, and the measured `delta_alpha` vs K correlation in the study, Pearson
`+0.87` on log-stiffness).

This supports Finding 3's K-growth pattern and Finding 3's "slightly negative at
K = 2" exactly: at `K = 2` the stiffness is near 1, alpha is not the bottleneck,
and the additional flexibility only adds estimation variance, so the change is
neutral-to-harmful; as `kappa(K)` climbs the rate relief switches on.

**Honest caveat (per-category dilution).** The classical finding is that K is
NEGLIGIBLE or HARMFUL for STATIC discrimination recovery, attributed to
per-category information dilution: the threshold information `I(beta_c) = alpha^2
P_{>=c}(1 - P_{>=c})` per category FALLS with K (the table above:
`0.187 -> 0.111`), spreading a fixed amount of response information across more
cut points. P6 is not in tension with this; it is a statement about the
theta-vs-alpha CONDITIONING of the training dynamics, a different quantity from
the static per-parameter estimability. The dynamics result and the static
estimation result genuinely point in different directions, and the document
claims only the former.

---

## 6. Negative result I: smooth positive maps are recovery-neutral (P7)

The empirical study refuted the claim that the exponential positivity map is
special. The math below is CONSISTENT WITH that refutation and explains it; it
does not argue exp is optimal.

### 6.1 The induced preconditioner (Proposition 1, reparameterization flow)

We adopt verbatim the reparameterization-flow result of the integrated research
plan (Section 5.1, "Proposition 1: positive maps are not optimization-equivalent")
and restate it cleanly here, since the corrected proposition of 6.2 is built on it.

**Proposition P7-0 (reparameterization gradient flow; research plan Prop 1).** Let
`alpha = g(a)` for a differentiable map `g`, with `a` the raw neural output and `L`
a loss seen as a function of `alpha`. Gradient flow on the RAW parameter `a` is
`da/dt = -dL/da`. By the chain rule `dL/da = (dL/dalpha) g'(a)`, so

```
da/dt = -g'(a) (dL/dalpha),
```

and the induced flow of the EFFECTIVE parameter `alpha = g(a)` is

```
d(alpha)/dt = g'(a) (da/dt) = -[g'(a)]^2 (dL/dalpha)
            = -m_g(alpha) (dL/dalpha),     m_g(alpha) := [g'(g^{-1}(alpha))]^2 >= 0,
```

using `a = g^{-1}(alpha)` to write the factor purely in terms of `alpha`. ∎

So each map `g` induces a SCALAR (diagonal) preconditioner `m_g(alpha) >= 0` on the
effective-alpha gradient flow. The exact preconditioners for the common maps:

```
exp:       g(a) = e^a,            g'(a) = e^a = alpha,          m_exp(alpha)      = alpha^2
softplus:  g(a) = log(1+e^a),     g'(a) = sigma(a),            m_softplus(alpha) = (1 - e^{-alpha})^2
sigmoid:   g(a) = sigma(a),       g'(a) = alpha(1-alpha),      m_sigmoid(alpha)  = alpha^2 (1-alpha)^2   (0<alpha<1)
```

(For softplus, `alpha = log(1+e^a)` gives `e^{-alpha} = 1/(1+e^a) = 1 - sigma(a)`,
so `g'(a) = sigma(a) = 1 - e^{-alpha}`, hence `m_softplus = (1 - e^{-alpha})^2`.)
These are genuinely different functions of `alpha`, which is why a positive map is
not a neutral positivity constraint in raw space. The plan's Prop 1 stops here; the
plan's Prop 2 then concluded that the `m_exp = alpha^2` map is UNIQUELY FASTER for
large alpha. That conclusion is empirically REFUTED here (the smooth maps tie under
matched effective-alpha init and per-map LR; the per-epoch convergence profiles
confirm it). We retain Prop 1 (the flow) and CORRECT the conclusion in 6.2.

### 6.2 The corrected proposition: smooth positive maps are recovery-equivalent

This is the central positive result that REPLACES the refuted "exp is special"
conclusion. Where the plan's Prop 2 read the `m_g`-dependence of the LOCAL rate as
proof that `m_exp = alpha^2` wins, the correct reading is that `m_g` is a strictly
positive TIME-REPARAMETERIZATION of one and the same trajectory. We prove this in
full, then derive the three consequences (endpoint invariance, rank invariance,
matched-init-and-LR speed equivalence), then prove the two genuine exceptions.

Throughout this subsection HOLD `theta` and `beta` fixed and treat `alpha` as a
scalar flow, `L = L(alpha)` (the direct-alpha setting; the joint case is Section
4 and 7.5). Let `g` be `C^1`, STRICTLY MONOTONE, and POSITIVE on its range, so
`g' != 0` everywhere and `m_g(alpha) = [g'(g^{-1}(alpha))]^2 > 0` strictly.

**Theorem P7 (smooth-map recovery equivalence).** Let `g_1, g_2` be two `C^1`
strictly-monotone positive maps. Consider the two induced effective-alpha flows
from Prop P7-0,

```
d(alpha)/dt = -m_{g_1}(alpha) L'(alpha),       alpha(0) = alpha_0,        (flow 1)
d(beta)/dt  = -m_{g_2}(beta)  L'(beta),         beta(0)  = alpha_0,        (flow 2)
```

started at the SAME effective initial value `alpha_0` (matched init). Then:

(i) **Same ordered path and same fixed point.** The two flows trace the IDENTICAL
ordered set of effective-alpha values, from `alpha_0` to the same limit
`alpha_inf`, in the same order; they differ only by a strictly increasing
reparameterization of time. In particular both have the same fixed point(s)
(`L'(alpha*) = 0`) and converge to the same one.

(ii) **Rank invariance.** Any metric that is a function of the ORDER of the
effective-alpha values across items (Spearman rank correlation against truth) is
identical for `g_1` and `g_2` at corresponding points of the path, and in
particular at the endpoint; it is invariant to the monotone time
reparameterization.

(iii) **Speed equivalence under matched init and LR.** The only remaining
difference is traversal SPEED, governed by the local rate `m_g(alpha*) H` of the
plan's Prop 2 (`H = L''(alpha*) = I(alpha*)`). Introducing a per-map scalar
learning rate `eta_g` and choosing `eta_{g_1} m_{g_1}(alpha*) = eta_{g_2}
m_{g_2}(alpha*)` makes the leading-order local rates coincide; the speed
difference is then a single absorbable constant. So no smooth map is special.

*Proof.*

(i) Because `g` is strictly monotone and `C^1`, the map `t -> alpha(t)` along flow
1 is a continuous curve in `alpha`-space. Since `m_{g_1}(alpha) > 0` strictly, the
sign of `d(alpha)/dt` equals the sign of `-L'(alpha)` at every point: the
preconditioner rescales the speed but NEVER reverses the direction of motion.
Therefore flow 1 moves monotonically along `-L'`, exactly as the un-preconditioned
flow `d(alpha)/dt = -L'(alpha)` does, and visits the same ordered sequence of
`alpha`-values. Formally, define the time change `tau` by `d tau/dt =
m_{g_1}(alpha(t)) > 0`; then in `tau`-time flow 1 becomes `d(alpha)/d tau = -L'(
alpha)`, the canonical (un-preconditioned) flow, INDEPENDENT of `g_1`. The same
construction with `m_{g_2}` reduces flow 2 to the SAME canonical flow `d(alpha)/d
tau = -L'(alpha)` with the SAME initial value `alpha_0`. By uniqueness of the
solution of an ODE with `C^1` (locally Lipschitz) right-hand side, the two
canonical trajectories are IDENTICAL as functions of `tau`. Hence flow 1 and flow
2 are the same curve, each a strictly-increasing time-reparameterization (`t ->
tau`) of it, with the same image, the same ordering, the same fixed points (where
`L'(alpha) = 0`, unchanged by multiplying by `m_g > 0`), and the same limit
`alpha_inf = lim_{tau -> inf} alpha(tau)`. ∎(i)

(ii) A Spearman rank metric depends only on the ORDER STATISTICS of the learned
effective alphas across the item population, compared to the order of the true
alphas. Apply the per-item flow to every item with its own `L_j` but a common map
`g`; by (i), at any common stopping rule expressed in canonical time `tau` each
item sits at the SAME effective-alpha value regardless of which smooth map drives
it, so the cross-item ordering is identical for `g_1` and `g_2`. A rank metric is
a function of that ordering alone, hence equal. (Even comparing at matched
WALL-CLOCK time `t` rather than canonical `tau`, the ordering is preserved as long
as the per-item time changes are co-monotone, which they are when one map is used
for all items; the endpoint, where all flows are at `alpha_inf`, is unconditionally
equal.) ∎(ii)

(iii) Near a fixed point `alpha*` with `L'(alpha*) = 0` and `H = L''(alpha*) =
I(alpha*) > 0`, linearize flow `g` with a learning rate `eta_g`:
`e(t) = alpha(t) - alpha*` obeys `de/dt = -eta_g m_g(alpha*) H e + O(e^2)`, so
`e(t) = e(0) exp[-eta_g m_g(alpha*) H t]` to leading order (this is the plan's
Prop 2 with the LR made explicit). The map enters ONLY through the scalar product
`eta_g m_g(alpha*)`. Choosing `eta_g` per map so that `eta_g m_g(alpha*)` is the
same constant `c` for both maps gives the identical local rate `c H`; the per-map
LR sweep is exactly the one-dimensional degree of freedom that absorbs the
constant `m_g(alpha*)`. ∎(iii)

**Conclusion.** (i) endpoint and (ii) rank are map-invariant for ALL `C^1`
strictly-monotone positive maps WITHOUT tuning anything; (iii) the residual SPEED
difference is a single scalar absorbed by the learning rate after matched init. The
exponential is one such map; nothing in the analysis distinguishes it from
softplus, scaled-sigmoid, or any other smooth strictly-monotone positive map. This
is the precise content of the refutation "exp is not special."

*Status: Proved* for (i) and (ii) (exact: positivity of `m_g` gives the time
change; ODE uniqueness gives one trajectory; rank reads the invariant ordering).
*Argued* for (iii) only in that the constant absorption is exact at a single
`alpha*` and approximate across an item population with a spread of true alphas (a
single scalar `eta` cannot match `m_g(alpha*)` simultaneously at every item's
`alpha*` because `m_g` varies with `alpha`). This second-order, population-spread
residual is the source of the empirical `+-0.002` clustering: smooth maps cluster
tightly but not byte-identically, and the residual is too small and inconsistent
to be load-bearing, exactly as Theorem P7 predicts.

**Relation to the free-table invariant.** P7(i)-(ii) are stronger, parameter-level
restatements of P4b for the scalar-alpha flow: P4b says smooth reparameterization
multiplies a residual-linear pull and so leaves the zero set (the endpoint)
unchanged; Theorem P7 says the WHOLE trajectory is one canonical curve up to a
positive time change, so not only the endpoint but the entire ordered path and
every rank metric are map-invariant. The endpoint invariance and the rank/path
invariance are the same fact at two resolutions.

**Direct-alpha control, predicted.** Theorem P7 with `theta, beta` frozen IS the
direct-alpha control of Section 7 (P8): a scalar preconditioner changes only the
SPEED of a single canonical 1-D flow to a fixed point, never the fixed point and
never the rank, so all reasonable update rules converge to the same band. The
control found exactly this; Theorem P7 is the derivation of why it had to.

### 6.3 The two genuine exceptions break a hypothesis of Theorem P7

Theorem P7 has two hypotheses on `g`: `C^1` SMOOTH (so `m_g` is defined and the
time change is `C^1`) and STRICTLY MONOTONE (so `m_g > 0` strictly and `g^{-1}` is
single-valued). Exactly the maps that fail one of these are the ones the study
finds genuinely lag. We prove each breakage.

**ReLU / clipped raw breaks `m_g > 0` (dead zone, vanishing preconditioner).** For
`g(a) = max(0, a)`, `g'(a) = 1` for `a > 0` and `g'(a) = 0` for `a < 0`, so

```
m_relu(alpha) = [g'(g^{-1}(alpha))]^2 = 1   on the active set (a > 0),
m_relu        = 0                            on the inactive set (a <= 0).
```

*Claim.* An item whose raw value `a_j` sits in the inactive set `a_j <= 0` receives
ZERO alpha-flow and cannot move under the induced flow. *Proof.* By Prop P7-0 the
induced flow is `d(alpha)/dt = -m_relu(alpha) L'(alpha) = 0` whenever
`m_relu = 0`. The time-change construction of Theorem P7(i) requires `m_g > 0` to
define `d tau/dt = m_g > 0` as a valid (invertible) reparameterization; on the dead
zone `d tau/dt = 0`, the reparameterization is SINGULAR, canonical time stops, and
the equivalence to the canonical flow fails. The fixed point set is no longer just
`{L'(alpha) = 0}`: every inactive item is ALSO a spurious fixed point of the raw
flow (`da/dt = -g'(a) L'(alpha) = 0` since `g'(a) = 0`). This is a genuine, NON-
absorbable penalty: a learning rate `eta` multiplies a quantity that is exactly
zero, `eta * 0 = 0`, so no LR choice revives a dead item. The item recovers only if
some OTHER force (noise, coupling to other parameters) pushes `a_j` back across
`a = 0`, which is outside the scalar flow. *Status: Proved* (the preconditioner is
exactly zero on the inactive set; LR cannot rescale a zero). This is the derivation
of the empirical "clipped raw / ReLU lags and is not rescued by gradient clipping
across `K = 2, 4, 8`."

**Square `g(a) = a^2 + epsilon` breaks strict monotonicity (non-injective, sign
folding).** Here `g'(a) = 2a`, which CHANGES SIGN at `a = 0`, so `g` is not
monotone and not injective: `g(a) = g(-a)`, so `g^{-1}(alpha)` is two-valued,
`a = +-sqrt(alpha - epsilon)`. *Claim.* The raw flow has a spurious unstable fixed
point and sign ambiguity that Theorem P7 excludes. *Proof.* The induced
preconditioner `m_sq(alpha) = [g'(g^{-1}(alpha))]^2 = 4(alpha - epsilon)` vanishes
as `alpha -> epsilon` (i.e. `a -> 0`), so the EFFECTIVE flow slows to a halt as it
approaches the minimum of `g`; the time change `d tau/dt = m_sq -> 0` is again
singular at `a = 0`. Worse, the RAW flow `da/dt = -2a L'(alpha)` has `a = 0` as a
fixed point regardless of `L'`, and its stability flips with the sign of
`L'(alpha)`: linearizing, `d(da)/dt = -2 L'(alpha) da`, so `a = 0` is a SADDLE-type
sign-ambiguous point. Two raw trajectories `+a` and `-a` map to the SAME effective
`alpha`, so the raw dynamics carry a spurious `Z_2` symmetry and the canonical-flow
uniqueness argument of Theorem P7(i) (which needs single-valued `g^{-1}`) does not
apply. *Status: Proved* (non-injectivity is explicit; the `a = 0` sign-flip fixed
point is a direct linearization). This is why a squared map is unstable near
`a = 0` even though the effective `alpha = a^2 + epsilon > 0` is always positive.

**Summary.** Theorem P7 (smooth strictly-monotone tie) plus these two
exact-breakage exceptions is the precise mathematical content of "exp is not
special; all smooth strictly-monotone positive maps tie in endpoint and rank;
only non-smooth (ReLU dead zone, `m_g = 0`) or non-monotone (square, two-valued
`g^{-1}`) maps genuinely lag." Both exceptions fail by making the preconditioner
non-positive or the inverse multi-valued, which is exactly the hypothesis Theorem
P7 requires; the failure is a vanishing or folded preconditioner, NOT a rescalable
constant, which is why a per-map learning rate (which fixes the smooth-map speed
difference) cannot fix them.

---

## 7. Negative result II: scalar alpha-space preconditioning is insufficient (P8)

The study also refuted the explanation that the neural representation effect is
just a scalar `alpha`-space preconditioner, via a direct-alpha control that
freezes true `theta, beta` and optimizes only the scalar `alpha_j` per item. The
math below is CONSISTENT WITH that refutation.

**Proposition P8 (scalar preconditioning cannot reproduce the representation
effect).** Freeze `theta = theta*` and `beta = beta*` and optimize only the
scalar `alpha_j` per item from the prediction loss. Then:

1. The problem is a set of `J` DECOUPLED scalar optimizations, one per item, each
   strictly convex in `z_j` and well-conditioned in `alpha_j` given correct
   `theta*, beta*` (the curvature is `I(alpha_j) = sum_i w_ij (theta_i -
   beta_j)^2 > 0` whenever the item sees any mismatched learners).

2. Any positive smooth `alpha`-space preconditioner `m(alpha_j)` only rescales the
   per-item scalar flow `d alpha_j/dt = -m(alpha_j) dL/dalpha_j`. By the
   one-dimensional version of P7, after LR tuning these converge to the SAME band;
   the preconditioner choice changes only the (rescalable) rate of a 1-D convex
   problem.

3. Therefore the recovery effect observed in the FULL neural model cannot be
   reproduced by scalar `alpha`-space preconditioning, because the full effect
   lives in the COUPLING among the shared representation directions (the
   stiffness `kappa = I(theta)/I(alpha)` of the JOINT flow, P4a/P6), which the
   direct-alpha control has removed by freezing `theta, beta`. With the coupling
   gone, there is nothing left for a preconditioner to fix, and all update rules
   tie.

*Status: Argued* (the frozen-`theta, beta` problem is convex-per-item and
single-direction) *plus Empirical* (the direct-alpha control: with a wide LR grid
the central `alpha`-space update rules converge to the same recovery band;
`docs/learning_dynamics_progress.md` E7).

The takeaway is a relocation, not an erasure. The recovery phenomenon is a
property of the JOINT, coupled flow on the shared representation (where alpha is
the low-Fisher slow mode competing with high-Fisher theta), not of the scalar
alpha subproblem in isolation. The same `kappa = I(theta)/I(alpha)` that governs
the rate effect (P4a) is the quantity a per-parameter-preconditioned optimizer
(Adam) partially cancels, which is why the study finds the empirical effect
COMPRESSED but not removed under Adam (a per-parameter preconditioner partially
substitutes for an alpha-specific readout, but cannot fully, because it
preconditions parameters, not the shared representation directions).

---

## 7.5 The positive result: representation coupling, not scalar reparameterization, sets the rate (P9)

This section replaces a proposition from an earlier plan that is now REFUTED. The
earlier claim was that the exponential positivity map induces an
`alpha^2`-preconditioned gradient flow that ACCELERATES discrimination recovery,
i.e. that a scalar reparameterization is the mechanism. Two empirical controls
kill it: (i) after matched effective-alpha init and per-map learning-rate tuning,
the exponential ties all smooth strictly-monotone positive maps (the content of
P7, Section 6); (ii) a direct-alpha control with true `theta, beta` frozen finds
all `alpha`-space update rules and scalar preconditioners converging to the same
recovery band (the content of P8, Section 7). A scalar reparameterization is
therefore NOT the mechanism. P9 states the mechanism that survives both controls:
the acceleration is a property of the COUPLED two-block representation, the
shared item code, and is relieved by giving discrimination its own code block
(decoupling). P9 builds directly on the rate law P4a, the K-conditioning P6, and
the insufficiency argument P8, and inherits the endpoint invariance of P4b; it
does not restate them.

### 7.5.1 Setup: the item-code block in the local quadratic model

Write the amortized model so the item-code block is explicit. Each item `j`
carries a code `e_j in R^m`. The discrimination and location readouts are linear
in the code, `alpha_j = g(a^T e_j)`, `beta_j = b^T e_j` (2PL; the GPCM threshold
readouts `B_c^T e_j` add `K-1` location directions, treated below). The same code
also feeds the ability pathway: the encoder pools item codes across a learner's
history to form `theta_i`, so a perturbation of `e_j` moves `theta_i` for every
learner who saw item `j`, with a person-dependent sensitivity vector `v_{ij} =
d theta_i / d e_j` (for a fixed linear pool `v_{ij} = (1/J) s_{ij} u`; for a
learned encoder `v_{ij} = (1/J) s_{ij} W^T diag(g'(.)) u`, the rung-5 and rung-6
forms, sum-checked to machine precision in `docs/learning_dynamics_toy.md` Secs
9, 10). The two architectures differ only in which readouts share a code block:

- SHARED: one code block `e_j` feeds the ability pathway AND the alpha/beta
  readouts.
- DECOUPLED: a separate block `e_j^al` feeds ONLY the alpha (and beta) readouts;
  the ability pathway reads its own block `e_j^th`.

**Assumptions (stated up front, used throughout 7.5).**

- (A1) Local regime. We work in a neighborhood of a gauge-fixed identifiable
  optimum `phi*` (gauge per Section 8); claims are local-quadratic and concern
  the transient, not the basin globally.
- (A2) Gauss-Newton / Fisher-Hessian. Near `phi*` we use the Gauss-Newton
  Hessian `H ~ E[J^T diag(w) J]`, which equals the Fisher information `F` exactly
  at a zero-residual well-specified optimum and approximately otherwise (the same
  identity used in P4a). The residual-curvature term is dropped; it is `O(r)` and
  vanishes at the reachable zero-residual optimum (P4b).
- (A3) Single shared step size. Plain gradient flow / gradient descent with ONE
  scalar learning rate across the code block (no per-coordinate preconditioning).
  This is the regime where conditioning bites; Adam relaxes it (Section 7.5.5).
- (A4) Free-table expressivity. The code has rank `>= 2` for 2PL, `>= K` for
  GPCM, so the zero-residual fit is reachable and P4b applies (this is the
  expressivity wall of Section 5; below it the effect is a wall, not a rate).
- (A5) Block-diagonal reduction. Cross-item and cross-person off-diagonal Hessian
  couplings are subdominant to the within-code-block curvature we analyze; the
  per-item code block is the relevant slow subsystem. *Status: Argued* (it is the
  block the gradient-flow lag is measured on in the toy; a full off-diagonal
  treatment is not attempted).

### 7.5.2 The shared code block is ill-conditioned; the decoupled block is not

We derive the item-`j` code-block Gauss-Newton Hessian explicitly, in terms of the
readout Jacobians and the Fisher entries of Section 3, then compute its two
eigen-directions and its condition number in closed form. No "schematically."

**The per-response Jacobian into the code.** For one response of learner `i` to
item `j`, the logit `z = alpha_j(theta_i - beta_j)` depends on `e_j` through THREE
readouts. With `alpha_j = g(a^T e_j)`, `beta_j = b^T e_j`, and `theta_i` produced
by the encoder pooling (sensitivity `v_{ij} = d theta_i / d e_j`), the chain rule
gives the per-response Jacobian of `z` into the code,

```
dz/de_j = (dz/dalpha) (dalpha/de_j) + (dz/dbeta) (dbeta/de_j) + (dz/dtheta) (dtheta_i/de_j)
        = x_ij * g'_j * a          +  (-alpha_j) * b          +  alpha_j * v_{ij},
```

where `x_ij = theta_i - beta_j`, `g'_j = g'(a^T e_j)`, and the three vector
directions are `a` (alpha-readout), `b` (beta-readout), `v_{ij}` (ability-pathway).
This is the per-response score `s = (alpha, x, -alpha)` of Section 3.5 pushed
through the readout Jacobian `J_j = [g'_j a, v_{ij}, -b]` (columns = the
`(alpha, theta, beta)` directions into the code): `dz/de_j = J_j s`.

**Proposition P9a (block Gauss-Newton Hessian, explicit).** The Gauss-Newton
Hessian on the item-`j` code is the `w`-weighted sum of the rank-one outer products
`w (dz/de_j)(dz/de_j)^T` over the responses touching item `j` (A2):

```
H^sh_j = sum_i w_ij (dz/de_j)(dz/de_j)^T
       = sum_i w_ij [ x_ij g'_j a - alpha_j b + alpha_j v_{ij} ]
                     [ x_ij g'_j a - alpha_j b + alpha_j v_{ij} ]^T.
```

Expanding and collecting by direction (using `I(alpha_j) = sum_i w_ij x_ij^2`,
`I(beta_j) = sum_i w_ij alpha_j^2 = I(theta)-scale`, and the cross terms), the
block is EXACTLY

```
H^sh_j =  (g'_j)^2 I(alpha_j) (a a^T)                         [A] alpha-readout, curvature (g'_j)^2 I(alpha_j)
        +  alpha_j^2 (sum_i w_ij v_{ij} v_{ij}^T)             [B] ability-pathway, curvature ~ alpha_j^2 sum_i w_ij |v_{ij}|^2
        +  alpha_j^2 I_w (b b^T)                              [C] beta-readout, curvature alpha_j^2 I_w,  I_w = sum_i w_ij
        +  cross terms (A,B,C off-diagonal, carried below).
```

*Proof.* Substitute `dz/de_j = x_ij g'_j a - alpha_j b + alpha_j v_{ij}` into
`sum_i w_ij (dz/de_j)(dz/de_j)^T` and expand the square. The pure squares give
the three displayed diagonal blocks with coefficients `sum_i w_ij x_ij^2 (g'_j)^2 =
(g'_j)^2 I(alpha_j)` (term A), `alpha_j^2 sum_i w_ij v_{ij} v_{ij}^T` (term B), and
`alpha_j^2 sum_i w_ij = alpha_j^2 I_w` (term C, since `b` is item-fixed). The cross
terms are the off-diagonal `(A,B), (A,C), (B,C)` outer products with coefficients
`sum_i w_ij x_ij g'_j alpha_j` etc.; they are the code-space image of the Fisher
off-diagonals `I_ta, I_ab` of Section 3.5 and they only INCREASE the spread (Section
3.5: coupling deepens the suppression). *Status: Proved* (exact expansion under A2).
∎

**The two relevant eigen-directions and the condition number.** Restrict attention
to the two-dimensional subspace spanned by the alpha-readout direction `a` and the
ability-pathway direction (the aggregate `v_j := (sum_i w_ij v_{ij} v_{ij}^T)`'s top
eigenvector; `b` is the location partner, fast like the ability mode, grouped with
it). Write the curvatures along these two directions as

```
c_alpha = (g'_j)^2 I(alpha_j)                                    (curvature in the alpha direction)
c_theta = alpha_j^2 * lambda_top(sum_i w_ij v_{ij} v_{ij}^T)     (curvature in the ability direction).
```

By Section 3.5 (the diagonal Fisher, lifted through the Jacobian) and the standard
encoder normalization (A6 below, the top eigenvalue is a constant fraction of the
trace, not the generic `1/m`), `lambda_top(sum_i w_ij v_{ij} v_{ij}^T) =
(sum_i w_ij |v_{ij}|^2) * O(1)` under A6, and `sum_i w_ij = I_w` so `c_theta = alpha_j^2
I_w |v|^2 ~ I(theta_j-aggregate)` where `I(theta_j-aggregate) = sum_i w_ij
alpha_j^2 = alpha_j^2 I_w` is the item's accumulated ability curvature. Then for the
shared block the two eigenvalues are `lambda_max = max(c_theta, c_alpha)` and
`lambda_min = min(c_theta, c_alpha)` (up to the cross-term tilt of 3.5, which only
spreads them further), and the condition number is

```
kappa^sh_j = lambda_max / lambda_min
           = c_theta / c_alpha                                  (since c_theta >> c_alpha)
           = [ alpha_j^2 I_w |v|^2 ] / [ (g'_j)^2 I(alpha_j) ]
           = [ I(theta_j) / I(alpha_j) ] * [ |v|^2 / (g'_j)^2 ]   (Jacobian-norm correction)
           = I(theta_j) / I(alpha_j) * (1 + O(coupling))   under A6 (matched Jacobian norms),
```

so to leading order `kappa^sh_j = I(theta_j)/I(alpha_j) = kappa` of P4a/P6, with an
EXPLICIT Jacobian-norm prefactor `|v|^2/(g'_j)^2` that A6 normalizes to `O(1)`. The
condition number is the Fisher stiffness times the ratio of the squared Jacobian
norms of the two pathways into the code.

**The decoupled block.** The decoupled alpha code `e_j^al` is read ONLY by the
alpha (and beta) heads, NOT by the encoder, so its per-response Jacobian is
`dz/de_j^al = x_ij g'_j a - alpha_j b` with NO `v_{ij}` term (the encoder reads a
separate `e_j^th`). Its Gauss-Newton block is

```
H^dc_j = (g'_j)^2 I(alpha_j) (a a^T) + alpha_j^2 I_w (b b^T),
```

whose two curvatures are `c_alpha = (g'_j)^2 I(alpha_j)` (alpha) and
`alpha_j^2 I_w` (beta). Beta is high-Fisher (P11) but it is a SEPARATE location
readout; the ALPHA-relevant conditioning, the eigenvalue spread the alpha component
must traverse, has NO ability-pathway high-curvature direction, so

```
kappa^dc_j (alpha-relevant) = O(1),    independent of I(theta).
```

*Status: Proved* (the block Hessians are exact under A2; the condition numbers
follow from the curvature magnitudes and A6).

**(A6) Matched Jacobian regularity (the stated assumption).** The clean reduction
`kappa^sh_j = I(theta_j)/I(alpha_j)` requires the squared Jacobian-norm prefactor
`|v|^2/(g'_j)^2 = O(1)`, i.e. the alpha-readout vector `a`, the beta-readout vector
`b`, and the pooled ability-sensitivity `v_{ij}` have comparable norms and are not
degenerate (the encoder does not amplify or annihilate the code direction it pools).
This is the regularity condition that lets the Fisher stiffness, not an architectural
norm artifact, govern the conditioning. *Status: Argued* (a normalization /
well-conditioned-readout condition; it holds in the toy where the readouts are
unit-initialized and the empirical `kappa` tracks `I(theta)/I(alpha)` at Spearman
0.891, d4). If A6 fails the prefactor reweights `kappa` but does not change its
SIGN or its monotone-in-K trend (which come from `I(theta)/I(alpha)`, P6).

The directions need not be orthogonal for this to hold; `kappa` is an eigenvalue
ratio of a symmetric PSD matrix, not an angle between gradients. We return to the
orthogonality point in 7.5.4 because it is exactly the Phase-2 subtlety the user
flagged.

### 7.5.3 The shared block throttles the alpha component; the decoupled block does not

Linearize gradient flow on the code block (A1-A3). With `delta_j = e_j - e_j*`,

```
d(delta_j)/dt = -H_j delta_j,
```

and in the eigenbasis of `H_j` each mode `c` decays as `e^{-lambda_c t}` (this is
P4a applied to the block). Decompose `delta_j` along the alpha-aligned eigenvector.

**Proposition P9b (rate throttling and the speedup factor, explicit GD).** We
derive the iteration count to a fixed tolerance `epsilon` for plain gradient
descent on the code block, every step shown. Diagonalize the symmetric PSD block
`H_j = sum_c lambda_c u_c u_c^T` in its eigenbasis; the eigenvalues are
`lambda_max = c_theta` (ability-aligned) and `lambda_min = c_alpha = (g'_j)^2
I(alpha_j)` (alpha-aligned) from P9a. Project the error `delta_j = e_j - e_j^*`
onto each eigenvector, `delta_c = u_c^T delta_j`.

*Gradient-descent recursion.* On the local quadratic `L = (1/2) delta^T H_j delta`,
one GD step with rate `eta` is `delta <- delta - eta H_j delta = (I - eta H_j)
delta`, which in the eigenbasis decouples to

```
delta_c(t+1) = (1 - eta lambda_c) delta_c(t),    so    delta_c(t) = (1 - eta lambda_c)^t delta_c(0).
```

*Stability bound (sets the usable step size).* The mode `c` decays iff
`|1 - eta lambda_c| < 1`, i.e. `0 < eta < 2/lambda_c`. For ALL modes to be stable
simultaneously under a SINGLE shared `eta` (A3) the binding constraint is the
LARGEST eigenvalue:

```
eta < 2 / lambda_max = 2 / c_theta.
```

The fastest stable choice (minimizing the slow mode's contraction factor subject to
the fast mode staying stable) is `eta = 1/lambda_max` (a standard GD optimum; at
`eta = 1/lambda_max` the fast mode contracts by `0` per step and the slow mode by
`1 - lambda_min/lambda_max = 1 - 1/kappa`).

*Slow-mode contraction and iteration count.* With `eta = 1/lambda_max`, the
alpha-aligned (slow) component contracts as

```
delta_alpha(t) = (1 - eta lambda_min)^t delta_alpha(0) = (1 - 1/kappa)^t delta_alpha(0).
```

To reach tolerance `|delta_alpha(t)| <= epsilon |delta_alpha(0)|` requires
`(1 - 1/kappa)^t <= epsilon`, i.e.

```
t  >=  log(1/epsilon) / ( -log(1 - 1/kappa) ).
```

Since `-log(1 - x) = x + x^2/2 + ... ~ x` for small `x`, and `1/kappa` is small
(stiff block, `kappa >> 1`), `-log(1 - 1/kappa) = 1/kappa + O(1/kappa^2)`, so

```
t^sh_alpha  =  log(1/epsilon) / ( 1/kappa + O(1/kappa^2) )  =  kappa * log(1/epsilon) * (1 + O(1/kappa))
           =  O( kappa log(1/epsilon) ).
```

This is the EXACT step count, not a scaling claim: the iterations to resolve the
alpha rank to tolerance `epsilon` in the shared block are `kappa log(1/epsilon)` to
leading order, with `kappa = I(theta)/I(alpha)` from P9a.

*Decoupled block.* The decoupled alpha block (P9a) has NO ability-aligned
high-curvature direction; its largest alpha-relevant eigenvalue IS `lambda_alpha =
c_alpha` itself (the beta direction is a separate readout, not a constraint on
alpha's step). The stability bound becomes `eta < 2/lambda_alpha`, and the optimal
`eta = 1/lambda_alpha` gives the alpha component a contraction `(1 - eta
lambda_alpha)^t = 0` in one step in the ideal isotropic case, or more honestly an
`O(1)` per-step contraction `(1 - 1/kappa^dc)^t` with `kappa^dc = O(1)`:

```
t^dc_alpha  =  log(1/epsilon) / ( -log(1 - 1/kappa^dc) )  =  O( log(1/epsilon) ),    kappa^dc = O(1).
```

*The speedup factor.* The ratio of iteration counts to the same tolerance is

```
speedup  =  t^sh_alpha / t^dc_alpha  =  [ kappa log(1/epsilon) ] / [ O(1) log(1/epsilon) ]
         =  O(kappa)  =  O( I(theta) / I(alpha) ).
```

The `log(1/epsilon)` cancels: the speedup is the condition-number factor `kappa`,
INDEPENDENT of the tolerance. The shared block pays `kappa` extra iterations on the
alpha mode purely because one shared step size must stay stable on the
high-curvature ability direction, throttling the alpha mode to a contraction of
`1/kappa` per step; decoupling removes that high-curvature direction from alpha's
block, lifting the step-size ceiling the alpha mode was paying.

*Status: Proved* (local quadratic, single step size, A1-A6; the recursion, the
stability bound `eta < 2/lambda_max`, the contraction `(1-1/kappa)^t`, and the
iteration count `kappa log(1/epsilon)` are exact for the linearized flow). ∎

**K-scaling (Proved + Empirical).** The speedup `O(kappa)` inherits the K-growth of
`kappa` directly through the GPCM Fisher forms of P6a. There `I(theta) = alpha^2
Var_p(k)` and `I(alpha) = Var_p(k theta - B_k)`, so

```
kappa(K) = I(theta)/I(alpha) = alpha^2 Var_p(k) / Var_p(k theta - B_k),
```

and P6 shows `Var_p(k)` (the spread of the category index, which `theta` reads)
grows FASTER in K than `Var_p(k theta - B_k)` (the spread of the natural statistic,
which `alpha` reads), because the category index `k` ranges over `{0..K-1}` and its
variance climbs while the natural-statistic variance saturates. Hence `kappa(K)` is
monotone increasing (toy table `0.96 -> 5.22` for `K = 2..11`), and the EXPLICIT
iteration count `t^sh_alpha = kappa(K) log(1/epsilon)` grows with K while
`t^dc_alpha` stays `O(log(1/epsilon))`. So the derived speedup `O(kappa(K))` grows
with K: decoupling buys more iterations saved as the number of answer levels rises.
This is the derived form of Finding 3's K-growth and of the `K = 2` near-tie (at
`kappa ~ 1` the two blocks have the same conditioning, the GD counts `t^sh ~ t^dc`,
so the only difference is the decoupled block's extra estimation variance, which is
why `K = 2` is net-neutral or slightly negative). The empirical `kappa(K)`-vs-
`delta_K` tracking (Spearman 0.891 over `K = 2..11`, d4) is the direct test of this
derived `speedup = O(kappa(K))` law.

### 7.5.4 Orthogonal gradients are consistent with the rate penalty

The Phase-2 instrumentation found the shared-code alpha-pathway and theta-pathway
gradients NEAR-ORTHOGONAL (`cos ~ 0`, measured `0.05..0.20` in the toy), which
refutes a gradient-CONFLICT (tug-of-war) story. P9 does not need a conflict.

**Why orthogonality does not rescue the rate.** The penalty in P9b is the
EIGENVALUE SPREAD of `H_j`, not the inner product of the two pathway gradients.
Two facts make this precise.

- The condition number `kappa = lambda_max / lambda_min` is a property of the
  Hessian's SPECTRUM. Orthogonal eigenvectors are the GENERIC case for a symmetric
  PSD Hessian (its eigenvectors are exactly orthogonal); orthogonality of the
  alpha and theta directions is therefore what you EXPECT, and it leaves `kappa`
  untouched. A stiff symmetric system with orthogonal eigenvectors is still stiff.
- Gradient conflict would require `cos < 0` (pulls partially cancel). `cos ~ 0`
  means the pulls do not cancel, the alpha direction is simply a low-curvature
  direction of the SAME block that a single step size cannot service quickly while
  staying stable on the high-curvature direction. The alpha signal is not
  overwritten; it is under-resolved.

*Status: Argued* (it is the standard reading of a stiff linear flow; the empirical
`cos ~ 0` and the empirical rate gap co-occur in rung 7, which is the consistency
check). This is the resolution of the "honest open subtlety" recorded in study
Section 2.3: the mechanism is magnitude/curvature dominance in an orthogonal
subspace, formalized as eigenvalue spread, not a directional fight.

### 7.5.5 Same fixed point: a pure rate / early-stopping effect

**Proposition P9c (endpoint invariance).** The shared and decoupled architectures
share the SAME fixed point, so P9 is a transient-only effect with no endpoint or
bias difference.

*Proof.* By P4b (the free-table invariant), under A4 every readout gradient
factors through `dL/dz = r` and a Jacobian, so every pull is linear and
homogeneous in the residual `r` and all pulls vanish simultaneously at the
reachable zero-residual optimum `r = 0`. Which readouts share a code block changes
the off-optimum Jacobian geometry (hence `H_j` and `kappa`, hence the RATE) but
not the zero set of the gradient (hence not the optimum). So both architectures
have the same stationary, globally-minimal, zero-residual fixed point. *Status:
Proved* (inherits P4b). ∎

Consequences, each matching an empirical control.

- The rank advantage CLOSES at convergence (rung 7: per-step rank gap `+0.07..
  +0.21` through the transient, `+0.001` by step 8000, both reach rank `1.0`). A
  transient advantage that vanishes at convergence is the signature of a pure rate
  effect, exactly as P9c requires.
- The advantage is on RANK, not bias (rung 7: steps to `|logbias| < 0.10`
  identical, `2252` vs `2301`). The endpoint magnitude is gauge-fixed identical
  (P4b); the only thing decoupling moves is the SPEED of the rank ordering, the
  metric the tracking model is judged on.
- Adam compresses but does not erase the gap (rung 7: GD `2.2x` -> Adam `1.6x`). A
  per-parameter preconditioner partially cancels `kappa` by rescaling the slow
  coordinate, so it PARTIALLY substitutes for decoupling; it cannot fully, because
  it preconditions parameters, not the shared code DIRECTIONS along which alpha and
  theta mix. This is the same `kappa` that P8's Adam remark referenced; here it is
  the explicit reason Adam helps and decoupling helps more.

### 7.5.6 Why the scalar direct-alpha control found nothing (the distinguishing check)

This is the consistency check that separates P9 (correct) from the refuted scalar
claim. The direct-alpha control freezes true `theta, beta` and optimizes only the
scalar `alpha_j`.

**Proposition P9d (the scalar control removes the mechanism by construction).**
With `theta, beta` frozen, there is no code block coupling two pathways: the
problem is `J` independent scalar optimizations, one per item, each a 1-D
strictly-convex problem in `alpha_j` with curvature `I(alpha_j) > 0` (P8.1). A
single direction has condition number exactly `1`; there is NO eigenvalue spread
for a preconditioner to fix. Hence every scalar `alpha`-space update rule and every
scalar preconditioner converges to the same band (P8.2). The acceleration P9
describes is intrinsically a property of the COUPLED two-block structure
(`kappa = I(theta)/I(alpha)` of the JOINT flow), which freezing `theta, beta`
deletes.

*Proof.* Freezing `theta, beta` removes the ability-pathway and beta-readout
contributions to the code-block Hessian of P9a, leaving a single alpha direction
with curvature `I(alpha_j)`. A 1-D PSD Hessian has `kappa = 1`, so P9b's speedup
`~ kappa = 1`: no rate gap exists to relieve. By P7's 1-D version any smooth
positive preconditioner is absorbed by the learning rate. *Status: Argued +
Empirical* (the 1-D convexity is exact under A1-A2; the direct-alpha control E7
confirms the tie). ∎

The contrast is the whole point. The REFUTED claim predicted that a scalar
`alpha`-space preconditioner (induced by the `exp` map, `m_exp = alpha^2`) would
accelerate alpha; the direct-alpha control isolates exactly a scalar
`alpha`-space problem and finds NO acceleration, because a scalar problem has no
conditioning to fix. P9 predicts that acceleration requires the coupled two-block
structure and appears precisely when the alpha direction must share a code with
the high-curvature ability pathway; the rung-7, K-sweep, and N-sweep experiments,
which DO have the coupled structure, find exactly that acceleration. The two
results are not in tension: they are the negative and positive halves of the same
mechanism. The scalar control is the load-bearing falsifier that says "the
mechanism is the coupling, not the reparameterization."

### 7.5.7 Scope and limits of P9

- (Proved) The block-curvature condition number `kappa = I(theta)/I(alpha)` for
  the shared block and its `O(1)` value for the decoupled alpha block (P9a); the
  `O(kappa)` resolution-time penalty and the `~ kappa` speedup factor under a
  single step size (P9b); the K-growth via P6; the endpoint invariance via P4b
  (P9c); the `kappa = 1` collapse of the scalar control (P9d). All are LOCAL
  quadratic statements under A1-A5.
- (Argued) The block-diagonal reduction A5 (the per-item code block is the
  relevant slow subsystem; a full off-diagonal Hessian treatment is not done);
  the orthogonality-consistency reading (7.5.4); the partial-substitution reading
  of Adam (7.5.5).
- (Empirical, supported not derived) The QUANTITATIVE speedup (rung 7: `2.2x`
  fewer steps to rank `0.95`, `8x` lower seed variance; the K-sweep `delta_K`
  magnitudes) and its exact K-profile (the magnitudes are empirical; the theory
  gives the SIGN, the parameter-specificity, and the monotone-in-K trend). The
  architecture-independence (LSTM / Transformer / DKVMN) of the empirical fix and
  the across-seed variance collapse (a stiffer shared flow is more seed-sensitive
  in its slow mode, consistent but not proved).
- (Explicitly NOT claimed) No claim that a scalar reparameterization or scalar
  preconditioner accelerates alpha (P9d and P7/P8 say the opposite; this is the
  refuted proposition). No endpoint or bias advantage for decoupling (P9c: same
  fixed point). No claim of a closed-form global trajectory; P9 is a local-rate
  plus global-endpoint composition, like P4, not a population-limit dynamics law.
  No variational content.

### 7.5.8 One-line mapping to each empirical anchor

- d1 (gate). Decoupling adding a separate alpha code is a structural change to
  WHICH readouts share a block (P9a), not a capacity change; the capacity gate is
  controlled and the effect persists, consistent with P9 locating the effect in
  the block STRUCTURE, not parameter count.
- d2 (trajectory). Alpha is reachable then resolved last because it is the slow
  (`lambda_alpha ~ I(alpha)`) eigendirection of the shared block (P9b); the code
  is shaped first along the high-curvature ability/location directions.
- d3 (gradient, 28x). The theta-pathway gradient on the shared code growing ~28x
  while the alpha pathway stays flat is the curvature asymmetry of P9a made
  visible: the ability direction accumulates the high `I(theta)` curvature and
  dominates the block's variation, so the linear alpha readout sees the alpha
  direction at shrinking SNR (study Sec 2.3), the magnitude-dominance reading of
  P9b/P9d.
- d4 (stiffness, rank 0.89). The decoupling advantage tracking `kappa =
  I(theta)/I(alpha)` at Spearman `0.891` over `K = 2..11` is the direct empirical
  signature of P9b's speedup `~ kappa` and P6's `kappa(K)` growth.
- d5 (N-sweep). The advantage NOT shrinking with N at a fixed budget (flat-to-
  widening) is P9c read on the data axis: at a fixed step budget the stiff shared
  flow is rate-limited (P9b), so more data leaves it more under-resolved on the
  slow alpha mode while the decoupled block exploits the data at its own rate; the
  "gap narrows with data" holds only AT convergence, which the fixed-budget model
  does not reach.

---

## 7.6 The joint three-parameter trade-off and the Pareto escape (P10, P11, P12)

Sections 2 through 7.5 analyze one parameter, alpha. That is half the story. The
architecture poses a THREE-WAY trade-off, and the alpha-only narrative hides the
parameter that actually drives the decoupling decision, ability theta. This
section formalizes the theta side (P10), the beta control (P11), and the
trade-off synthesis (P12), and cross-references P9 for the alpha half.

### 7.6.1 The architecture: one code, two consumers

The item code `e_j` feeds TWO consumers with opposite width needs.

- The ability ENCODER (its INPUT). The encoder pools the per-step item codes
  across a learner's history to form the amortized ability `theta_hat_i`. In the
  real model the LSTM input at step t is `[e_{q_t}, r_t]`, so `e_j` is the
  encoder's per-item INPUT signal (`deep_irt/core/encoder.py`, `_direct_hidden`).
- The alpha/beta READOUTS (the readout code). Linear heads read
  `alpha_j = g(a^T e_j)`, `beta_j = b^T e_j` (and the GPCM threshold readouts)
  directly off `e_j`.

The two architectures differ ONLY in whether one width serves both consumers.

- SHARED: a single width `W` is the item-code width = encoder input = readout
  code. Widening to give the readouts capacity simultaneously widens the encoder
  input.
- DECOUPLED: a NARROW encoder input (`emb_dim = 8`) and a SEPARATE WIDE readout
  code (`item_key_dim = W - 8`) that feeds ONLY the alpha/beta readouts, never
  the encoder input (`item_val_emb` feeds the LSTM; `item_key_emb` feeds the
  readouts). `state_alpha = True` in both, so the comparison isolates WHICH width
  serves the encoder, not whether alpha is state-conditioned.

This is the precise structure behind the gate (Section 2.1) and is the lever the
whole study turns on.

### 7.6.2 Alpha wants the readout code WIDE (recap of P9)

The alpha half is P9 and is not repeated. In one line: alpha is the
low-Fisher (`I(alpha) = (theta - beta)^2 w`, P2) hard-to-read parameter; it needs
readout CAPACITY to be expressed, and within a shared width it is RATE-throttled
by the block conditioning `kappa = I(theta)/I(alpha)` (P9a, P9b). So alpha's
pressure is "make the readout code wide," and the benefit of doing so grows with
K (P6). This pressure acts on the READOUT consumer.

### 7.6.3 Theta wants the encoder input NARROW (P10)

The new content. Theta's pressure acts on the ENCODER consumer and points the
opposite way. The mechanism is NOT conditioning; it is capacity / generalization
of an amortized readout, and it is honestly the softest formalization in the
document, so the status markers are explicit at every step.

**The dynamical picture.** Theta is a low-rank latent (one scalar per learner).
The only history content that should inform `theta_hat_i` is each answered item's
location and discrimination, NOT its identity. Widening the encoder INPUT code
gives the encoder degrees of freedom to route item-IDENTITY (and the
learner-by-item response noise that rides on it) into the ability state, fitting
training responses through a channel that does not generalize to `theta*`. So the
variance of `theta_hat` rises with encoder-input width, and the optimizer drives
it higher along the trajectory.

**Setup (linear-amortizer surrogate).** To make the variance statement rigorous,
replace the LSTM with a linear amortizer, the standard tractable surrogate (the
same move the toy ladder uses for the fixed-pool rungs). A learner answers a set
of items; stack their item codes as rows of `X_i in R^{n_i x W}` (each row an
answered item's `W`-dim code), and let the amortizer read ability as a linear
functional of the pooled, response-weighted codes, fit by least squares against
the response signal. The estimable ability is the projection of the response
signal onto the `W`-dimensional code span.

**Proposition P10a (variance of the amortized theta grows with encoder-input
width).** For the linear amortizer fitting `theta_hat` from a `W`-dimensional
per-item code at finite data `n` per learner, the excess risk decomposes as

```
E[(theta_hat_i - theta*_i)^2]  =  bias(W)^2  +  Var(W),     Var(W) ~ sigma_r^2 * (W / n),
```

with `sigma_r^2` the per-response noise variance. The variance term is MONOTONE
INCREASING in the number of fitted code directions `W` (the effective degrees of
freedom of the readout): each extra code direction the encoder is allowed to read
is an extra direction along which it can fit response noise into `theta_hat`. The
bias term is non-increasing in `W` (more directions can only improve the best
achievable fit), but for a low-rank latent like theta the bias saturates after a
few directions (the location/discrimination-relevant subspace is small), so past
that point widening `W` is PURE variance.

*Proof sketch.* Ordinary least squares of a target on a `W`-column design has
estimator variance `sigma^2 tr[(X^T X)^{-1} X^T X] / n = sigma^2 W / n` in the
isotropic case (effective-degrees-of-freedom = number of fitted directions); this
is the textbook variance / d.o.f. identity. The bias is the approximation error
of the best `W`-direction fit, non-increasing in `W` and saturating once the span
contains the theta-relevant subspace. *Status: Proved* for the linear amortizer
(it is the standard bias-variance / d.o.f. decomposition); the `W/n` rate is the
isotropic-design special case. ∎

**The lift to the LSTM (Argued).** The real encoder is a nonlinear sequence model,
not a linear least-squares readout, so P10a is a surrogate. The lifted claim is:
a wider input code raises the encoder's capacity to absorb item-identity nuisance
into the hidden state, so `theta_hat`'s variance / item-identity leakage rises
with `emb_dim`, with the same sign as P10a. *Status: Argued* (the nonlinear,
sequential generalization of the d.o.f. argument; not proved). The plain-words
version is already in `deep_irt/core/encoder.py` ("a fat encoder input lets the
LSTM memorize items into the ability state").

**The training-time face (Argued + Empirical, the weakest link).** P10a is a
finite-data statement at a fixed fit. The empirical signature is also a
TRAJECTORY effect: theta recovery PEAKS then DECAYS with training (study Sec 2.2:
0.97 at ep150 to 0.68 at ep500 for the bare wide encoder), and decays FASTEST for
the widest shared encoder. The dynamical reading: gradient flow on an
over-parameterized encoder does not stop at the generalizing fit; it keeps
descending the training loss by fitting residual training-response noise through
the surplus code directions, so the variance term of P10a INFLATES along the
trajectory. *Status: Argued* (the over-parameterized-interpolation reading of the
trajectory) *+ Empirical* (the peak-then-decay curve, and its width-ordering).
This is the honest soft spot: there is no closed-form trajectory for the theta
variance under the LSTM, only the surrogate's static `W/n` plus the empirical
decay shape. We do not claim a proved over-training law for theta.

**Why this is OUTSIDE the free-table invariant (the key distinction from P9).**
P4b protects the endpoint only for parameters produced by a FREE per-item table.
Theta is NOT such a parameter: it is read by a shared ENCODER, an amortized
function, not a free per-learner table. So the invariant does not apply to theta,
and theta CAN be displaced from truth at finite data by the variance term of P10a
even at the loss optimum of the prediction objective (the encoder minimizes
prediction loss by overfitting, which is not the same as recovering `theta*`).
This is exactly why P10 is a different KIND of mechanism from P9: P9 is a
transient RATE effect on an invariant endpoint (alpha, a free item parameter);
P10 is a finite-data ENDPOINT / generalization effect on an amortized readout
(theta, not a free parameter). Naming them as one "trade-off" must not collapse
this distinction.

**The 28x gradient as the theta-side signature.** Study Sec 2.3 measures the
theta-pathway gradient on the shared code growing ~28x over training while the
alpha-pathway gradient stays flat. P9 reads this as the alpha side (alpha at
shrinking SNR). P10 reads the SAME number as the theta side: the encoder keeps
finding ability-explaining (and, past the generalizing fit, noise-explaining)
variation in the wide shared code to exploit, so its pull on the code keeps
growing; the growing pull IS the encoder routing more of the code into theta over
training, the dynamical visible form of the variance term inflating. One
measurement, two consistent readings, which is why it sits under both P9 (d3) and
P10.

### 7.6.4 Beta is the indifferent control (P11)

Beta closes the three-way picture by being neither side.

**Proposition P11 (beta needs no width and is not hurt by it).** Two facts.

1. Beta is HIGH-Fisher. `I(beta) = alpha^2 w` (P2), the same order as `I(theta)`
   and far above `I(alpha)`. So beta is well determined from a THIN code; it does
   not need the readout width alpha needs. *Status: Proved* (the Fisher form P2).
2. Beta has NO amortized-pooling pathway. Beta is read DIRECTLY off the item's own
   code by a linear head (one location vector per item), not pooled across a
   learner's history into a per-learner latent. There is therefore no
   learner-by-item nuisance channel through which code width could leak into a
   beta estimate, so beta does not suffer theta's variance inflation (P10). beta
   is a free per-item readout, so it is also fully protected at the endpoint by the
   free-table invariant (P4b). *Status: Proved* (readout structure + P4b).

Hence beta is the NEGATIVE CONTROL on both pressures: it does not need width
(unlike alpha) and is not degraded by it (unlike theta). A wider / richer /
state-conditioned treatment of beta should do essentially nothing.

*Empirical anchor.* The dynamic-beta arm moves beta by `delta_beta ~ +0.003`
mean across K (study Sec 3.1), versus `delta_alpha ~ +0.042` for the matched
dynamic-alpha arm, and beta recovery sits at ~0.98 everywhere regardless of
width. Beta is flat along the gate frontier. This is the parameter-specificity
control that rules out "generic flexibility helps": the same architectural
change that lifts the low-Fisher pooled parameter (alpha) is a no-op on the
high-Fisher direct parameter (beta). *Status: Empirical.*

A subtlety to keep honest. In the DECOUPLED config beta is read from the WIDE
item key (it sides with alpha on the readout side, since a wide readout does not
hurt it and the high Fisher means it recovers cleanly either way). The precise
statement is therefore: beta does not DRIVE the demand for width (P11.1) and is
not on the HARMED side of width (P11.2). It is content with whatever the readout
width is, narrow or wide, and it is the encoder-input width that beta, like
theta, has no reason to inflate.

### 7.6.5 The trade-off and the Pareto escape (P12)

Now assemble the three pressures on the single shared knob.

**The collision.** In SHARED, one width `W` is the encoder input AND the readout
code. The three single-knob gradients are:

```
d(alpha recovery) / dW  >  0    (P9: alpha needs readout capacity; throttled by kappa)
d(theta recovery) / dW  <  0    (P10: wider encoder input inflates theta variance)
d(beta  recovery) / dW  ~  0    (P11: beta indifferent)
```

Two of the three pull in OPPOSITE directions on the SAME knob. Sweeping `W` can
therefore only trade theta for alpha; it cannot raise both.

**Proposition P12 (shared traces a Pareto frontier; decoupling is
Pareto-dominant).**

(a) Under the shared architecture the achievable `(theta recovery, alpha
recovery)` pairs, as `W` varies, form a Pareto FRONTIER: theta monotone down,
alpha monotone up, beta flat. No single `W` attains both the high-theta value of
small `W` and the high-alpha value of large `W`, because the two objectives are
monotone in opposite directions in the one free variable.

*Proof.* A single scalar control with two objectives whose gradients have
OPPOSITE signs everywhere admits no interior point dominating both endpoints:
increasing `W` strictly improves one and strictly worsens the other, so the
image of the map `W -> (theta(W), alpha(W))` is a monotone curve, the definition
of a Pareto frontier for two opposed monotone objectives. *Status: Proved* (given
the signs of 7.6.5, which are P9 / P10 / P11; the SIGNS are Proved-for-alpha,
Argued-for-theta, so the frontier inherits the theta side's Argued status). ∎

(b) The DECOUPLED architecture has TWO independent knobs, `W_enc` (encoder input)
and `W_read` (separate readout code). Set `W_enc` narrow (theta keeps its low
variance, P10) and `W_read` wide (alpha gets its capacity, P9; beta unaffected,
P11). The resulting `(theta, alpha)` point lies ABOVE the shared frontier:
high theta from small `W_enc` AND high alpha from large `W_read`, simultaneously.
Decoupling is Pareto-DOMINANT; it ESCAPES the frontier rather than moving along
it.

*Proof.* With two independent controls each objective is optimized by its OWN
knob without coupling: `theta` by `W_enc`, `alpha` by `W_read`, so the pair
`(theta(W_enc^*), alpha(W_read^*))` attains each coordinate's best value
independently. This pair dominates every shared point, where one knob serves both
and the opposed gradients forbid attaining both bests at once. *Status: Proved*
(two independent controls dominate one coupled control under opposed single-knob
gradients) *+ Empirical* (the gate at matched TOTAL capacity: the decoupled point
sits above the shared frontier even when the shared table is widened to 6x the
decoupled budget, so the dominance is ALLOCATION, not budget). ∎

**The honest scope of P12.** What is Proved is the STRUCTURE: opposed single-knob
gradients force a frontier under sharing, and two independent knobs dominate it.
What is Argued is the theta-side SIGN that the structure rests on (P10 is the
Argued/Empirical part). What is Empirical is the MAGNITUDE and the matched-budget
control (the gate numbers). The claim is not that decoupling improves any
parameter's ENDPOINT for free; theta's protection is a finite-data / generalization
gain (P10) and alpha's is a finite-budget rate gain (P9), and beta is unchanged.
At the joint infinite-data-and-training limit alpha's rate gain vanishes (P9c)
and theta's variance vanishes (`W/n -> 0` in P10a), so the frontier and its
escape are a FINITE-regime phenomenon, exactly the regime the real model trains
in.

### 7.6.6 One-line mapping to the gate / trajectory / gradient empirics

- GATE (Section 2.1, the Pareto). The shared frontier (theta 0.97->0.88, alpha
  0.66->0.91, beta flat as `W` grows) is P12(a): opposed single-knob gradients
  (P9 up, P10 down, P11 flat). The decoupled point above it at matched total
  capacity is P12(b): two knobs dominating one. The theta DOWN-slope specifically
  is P10 (encoder-input variance); the alpha UP-slope is P9 (readout capacity +
  conditioning relief).
- TRAJECTORY (Section 2.2, theta overfits with training). The peak-then-decay of
  theta, fastest for the widest shared encoder, is P10's training-time face: the
  over-parameterized encoder descends into interpolation, inflating the theta
  variance term along the trajectory. Decoupling does NOT fix the theta
  over-training decay (it is a separate regularization problem, study Sec 2.2),
  consistent with P10 locating it in the encoder INPUT width, which decoupling
  keeps narrow but does not regularize.
- GRADIENT (Section 2.3, theta-pathway grows 28x). The theta side of the same
  curvature / capacity asymmetry as d3: the encoder's growing pull on the shared
  code IS the encoder routing more code into theta over training (P10), the dual
  reading of the alpha-SNR-shrinking story (P9).
- BETA NULL (Section 3.1, delta_beta ~ 0). P11: the high-Fisher direct-readout
  parameter is indifferent to the architectural change that lifts the low-Fisher
  pooled parameter, the parameter-specificity control.

---

## 7.7 Results and insights (the derived picture, crisply)

This section states the corrected, fully-derived results in one place, separating
the alpha RATE effect from the theta ENDPOINT effect, and listing the explicit
quantitative laws. It is the synthesis of Sections 3.5, 6.2, 6.3, 7.5; nothing here
is new, it is the consolidated answer to "what is right" that replaces the refuted
"exp is special / scalar preconditioner accelerates alpha."

### 7.7.1 The four mechanisms

**(a) The positivity map is a preconditioner; smooth maps are recovery-equivalent.**
Any `C^1` map induces the scalar preconditioner `m_g(alpha) = [g'(g^{-1}(alpha))]^2`
on the effective-alpha flow (Prop P7-0, the research plan's Prop 1). For SMOOTH
STRICTLY-MONOTONE POSITIVE maps `m_g > 0`, so the flow is a strictly-increasing
TIME-REPARAMETERIZATION of one canonical curve `d alpha/d tau = -L'(alpha)` (Theorem
P7). Consequences, all derived: (i) the fixed point and endpoint are map-invariant;
(ii) any rank metric (Spearman) is invariant to the monotone time change, so rank
recovery is map-invariant WITHOUT tuning; (iii) the only residual difference is
traversal SPEED, a single scalar `eta_g m_g(alpha*)` absorbed by a per-map learning
rate after matched effective-alpha init. So no smooth map is special; exp is not.
The two genuine exceptions break a HYPOTHESIS of Theorem P7, proved exactly: ReLU
has `m_g = 0` on a dead zone (the flow halts; `eta * 0 = 0` is unrescalable), and
square `g = a^2` is non-injective (two-valued `g^{-1}`, a sign-folding `a = 0`
saddle). It is positivity-plus-smoothness-plus-strict-monotonicity that matters, not
the exponential. This derives the j5 convergence-profile result (smooth maps tie,
non-smooth/non-monotone lag) and the direct-alpha control (a scalar preconditioner
with `theta, beta` frozen changes only speed, never the fixed point or the rank).

**(b) Alpha is the slow mode because the per-response Fisher is rank one with a
suppressed lever arm.** The per-response score is the single vector `s = (alpha, x,
-alpha)` (`x = theta - beta`), so the per-response Fisher `F_resp = w s s^T` is
RANK ONE: one informative direction per response (Prop P2-full). The population
Fisher `F = E[w s s^T]` has the off-diagonal entries `I_ta = E[alpha x w]`,
`I_ab = -E[alpha x w]` of the research plan. The `(theta, alpha)` block eigenvalues
are exactly `lambda_pm = (1/2)[(I_tt + I_aa) +- sqrt((I_tt - I_aa)^2 + 4 I_ta^2)]`;
the SMALL eigenvalue is alpha-aligned, `lambda_- ~ I_aa(1 - rho^2) <= I_aa`, because
`I_aa = E[w x^2]` is suppressed: `w = p(1-p)` peaks where `x = 0`, exactly where the
lever arm `x^2` vanishes, giving `I_tt/I_aa ~ alpha^2 / Var_w(x)`. Coupling
(`rho != 0`) only deepens the suppression. This is WHY alpha is slow, derived from
the full Fisher, not assumed from the diagonal.

**(c) Sharing a code creates one ill-conditioned Hessian block; one learning rate
costs `O(kappa)`; decoupling removes it.** The item-code Gauss-Newton block is the
explicit sum `H^sh_j = (g'_j)^2 I(alpha_j)(a a^T) + alpha_j^2 sum_i w_ij v_{ij}
v_{ij}^T + alpha_j^2 I_w (b b^T)` (Prop P9a), with two eigen-directions: a steep
ability-aligned one (curvature `c_theta ~ I(theta_j)`) and a flat alpha-aligned one
(curvature `c_alpha = (g'_j)^2 I(alpha_j)`). Its condition number is `kappa^sh_j =
c_theta/c_alpha = [I(theta_j)/I(alpha_j)] [|v|^2/(g'_j)^2]`, the Fisher stiffness
times a Jacobian-norm prefactor that A6 normalizes to `O(1)`. The DECOUPLED alpha
block omits the `v_{ij}` (ability) term entirely, so its alpha-relevant condition
number is `O(1)`. Plain GD with one step size must satisfy `eta < 2/lambda_max`, so
the alpha mode contracts as `(1 - 1/kappa)^t` and reaches tolerance `epsilon` in
`t^sh_alpha = kappa log(1/epsilon)` iterations; the decoupled block reaches it in
`O(log(1/epsilon))`. The `log(1/epsilon)` cancels, leaving `speedup = O(kappa)`.

**(d) The theta side is a DISTINCT endpoint effect, not the same as alpha's rate.**
This must not be collapsed. Alpha's effect (c) is a TRANSIENT RATE effect on an
INVARIANT endpoint (P9c, P4b): both architectures share the same fixed point, the
rank gap CLOSES at convergence (rung 7: `+0.07..+0.21` through the transient,
`+0.001` by step 8000), so decoupling buys SPEED and early-stopped rank, not a
better endpoint, on alpha. Theta's effect (P10) is a FINITE-DATA ENDPOINT /
generalization effect on an amortized readout, OUTSIDE the free-table invariant
(theta is read by an encoder, not a free per-item table): a wide encoder input
inflates `Var(theta_hat) ~ sigma_r^2 W/n` and the over-parameterized encoder drives
it higher along the trajectory (theta `0.97 -> 0.68` with training). These are
different axes (steps vs data) vanishing at different limits (`steps -> inf` vs
`reps -> inf`). The joint Pareto escape (P12) works because the two pressures act on
two different consumers of the code; decoupling gives each its own width.

### 7.7.2 The explicit quantitative laws (the numbers the theory now delivers)

- **Condition-number formula (Proved).** `kappa^sh_j = [I(theta_j)/I(alpha_j)] *
  [|v|^2/(g'_j)^2]`, reducing to `kappa = I(theta)/I(alpha)` under matched Jacobian
  norms (A6). The full-Fisher version is strictly larger: `kappa_2 = I_tt /
  [I_aa(1 - rho^2)] >= I_tt/I_aa`, so the diagonal ratio used elsewhere is the
  optimistic bound; off-diagonal coupling makes the real flow stiffer.

- **Iteration count to tolerance (Proved).** `t^sh_alpha = log(1/epsilon) /
  (-log(1 - 1/kappa)) = kappa log(1/epsilon) (1 + O(1/kappa))` for the shared block
  at `eta = 1/lambda_max`; `t^dc_alpha = O(log(1/epsilon))` for the decoupled block.
  Speedup `= t^sh/t^dc = O(kappa)`, tolerance-independent.

- **K-growth law (Proved, Fisher forms; Empirical, magnitudes).** `kappa(K) =
  alpha^2 Var_p(k) / Var_p(k theta - B_k)` is monotone increasing in K (toy `0.96
  -> 5.22` for `K = 2..11`), so `t^sh_alpha = kappa(K) log(1/epsilon)` and the
  speedup `O(kappa(K))` grow with K. Empirically `kappa(K)` tracks the decoupling
  advantage `delta_K` at Spearman 0.891 over `K = 2..11` (d4), the direct test of
  the derived `speedup = O(kappa(K))` law. At `K = 2`, `kappa ~ 1`, `t^sh ~ t^dc`,
  net-neutral; the advantage switches on as `kappa(K)` climbs.

- **Map equivalence (Proved for endpoint and rank; Argued for the residual speed
  constant).** Endpoint and rank are map-invariant for all smooth strictly-monotone
  positive maps; the residual speed difference is the single absorbable constant
  `m_g(alpha*)`, exact at one alpha, approximate across an item population (the
  empirical `+-0.002` clustering).

### 7.7.3 The one-paragraph corrected statement

Under a prediction objective the per-response Fisher is rank one along `s = (alpha,
theta - beta, -alpha)`, so a response informs one direction; alpha's direction is
the slow eigen-mode because its information `I(alpha) = E[w(theta - beta)^2]` is
suppressed where `w` concentrates (`theta ~ beta`), and the off-diagonal coupling
deepens the suppression. A positivity map is a positive scalar preconditioner on
alpha's flow, a time-reparameterization that leaves the endpoint and the rank
invariant for every smooth strictly-monotone positive map (so exp is not special;
only the ReLU dead zone and the square sign-fold genuinely lag, by breaking the
positivity or injectivity hypothesis). The accelerator that works is REPRESENTATION
decoupling: sharing one item code forces alpha's flat (`I(alpha)`) direction to
share a Hessian block with the steep ability (`I(theta)`) direction, so a single
learning rate, stable on the steep direction, resolves alpha in `kappa log(1/
epsilon)` iterations; giving alpha its own code removes the steep direction and
resolves it in `O(log(1/epsilon))`, an `O(kappa)` speedup that grows with K via the
GPCM Fisher, with the SAME fixed point in both cases (a pure rate / early-stopping
effect on alpha rank). This is distinct from theta's finite-data endpoint overfit,
which lives outside the free-table invariant and is the OTHER half of the Pareto
escape, not the same mechanism.

---

## 8. The gauge, and why claims are on rank

The 2PL and GPCM losses are invariant under the joint reparameterization

```
theta -> s theta + t,   beta -> s beta + t,   alpha -> alpha / s   (s > 0),
```

which leaves every `alpha(theta - beta)` and every category probability
unchanged (verified to `3e-15` in `docs/learning_dynamics_toy.md`). A uniform
alpha shrinkage paired with a theta inflation is therefore a COORDINATE choice,
not a recovery bias. Any raw "alpha biased low" magnitude must be quotiented out
(regress learned `theta` on true `theta` for the scale `s`, rescale `alpha`)
before it is interpreted. In the toys the entire raw apparent shrinkage collapsed
to `0.0000` after gauge fixing. This is why the study reports recovery as
sign-aligned RANK (Spearman), which is gauge-invariant, and why all propositions
above concern rank/rate quantities, not raw magnitudes. (The methodological order
gauge -> expressivity -> finite-sample -> dynamics is the discipline used
throughout the study; this document inherits it.)

---

## 9. Scope and limits

What this document establishes, and as importantly what it does not.

**Established (Proved).**

- The 2PL and GPCM per-parameter gradient structure (P1, Section 5.1) and the
  single-response Fisher informations (P2, P6a), including the consistency of the
  GPCM forms with the 2PL forms at `K = 2`.
- Alpha is the low-information parameter: `I(alpha)` carries the squared
  separation lever arm and vanishes where targeted responses concentrate
  (`theta ~ beta`); the prediction sensitivity `dp/dalpha = I(alpha)^{1/2}`
  vanishes there too (P2, P3).
- The full per-response Fisher is RANK ONE, `F_resp = w s s^T` with score `s =
  (alpha, theta - beta, -alpha)`; the population `(theta, alpha)` block eigenvalues
  are exact, and the SMALL (alpha-aligned) one `lambda_- ~ I_aa(1 - rho^2) <= I_aa`
  is the derived reason alpha is the slow eigen-DIRECTION, with the off-diagonals
  `I_ta = E[alpha x w]`, `I_ab = -E[alpha x w]` deepening the suppression
  (P2-full, Section 3.5).
- The local linear rate law: a direction's recovery time scale is the inverse of
  its Fisher information, ordering alpha as the slow mode with flow condition
  number `kappa ~ I(theta)/I(alpha)` (P4a).
- Smooth-map recovery EQUIVALENCE at the endpoint AND on rank: a positive smooth
  strictly-monotone map gives `m_g > 0`, so the effective-alpha flow is one
  canonical curve up to a strictly-increasing time change; endpoint and Spearman
  rank are map-invariant with no tuning, only the speed constant `m_g(alpha*)`
  differs and is absorbed by per-map LR (Theorem P7). The two exceptions are exact:
  ReLU `m_g = 0` dead zone, square non-injective sign-fold (Section 6.2-6.3).
- The free-table invariant: a reachable zero-residual optimum is stationary,
  globally minimal, and invariant to representation sharing and to smooth
  reparameterization, so the endpoint carries no representation-choice or
  positive-map advantage (P4b).
- The GPCM stiffness `kappa(K) = I(theta)/I(alpha)` grows monotonically with K
  (P6a, P6).
- The shared item-code Gauss-Newton block is the explicit sum `H^sh_j = (g'_j)^2
  I(alpha_j)(a a^T) + alpha_j^2 sum_i w_ij v_{ij}v_{ij}^T + alpha_j^2 I_w (b b^T)`,
  with condition number `kappa^sh_j = [I(theta)/I(alpha)][|v|^2/(g'_j)^2] ~
  I(theta)/I(alpha)` under matched Jacobian norms (A6); the decoupled alpha block's
  is `O(1)`. The EXPLICIT gradient-descent count under one step size (`eta <
  2/lambda_max`) is `t^sh_alpha = kappa log(1/epsilon)` versus `t^dc_alpha =
  O(log(1/epsilon))`, so the speedup is `O(kappa)`, tolerance-independent and
  growing with K via `kappa(K) = alpha^2 Var_p(k)/Var_p(k theta - B_k)`, with the
  SAME fixed point in both cases (P9, the validated mechanism, replacing the refuted
  scalar-preconditioner claim).
- Beta is the indifferent control: high Fisher (`I(beta) = alpha^2 w`) so it
  recovers from a thin code and does not need width, and a direct (non-pooled)
  readout protected by the free-table invariant so width does not hurt it (P11).
- The Pareto STRUCTURE: opposed single-knob gradients (alpha up, theta down, beta
  flat) force the shared architecture onto a frontier in `(theta, alpha)`, and two
  independent widths (decoupling) dominate one coupled width (P12). The structure
  is Proved given the gradient signs; the theta-side sign it rests on is Argued.

**Argued (heuristic, locally rigorous but not a global theorem).**

- That low Fisher therefore sets the RECOVERY RATE and not the endpoint at
  infinite data and training (P4); this combines the proved local rate law with
  the proved endpoint invariant, but a single closed-form global trajectory is
  not available (and the study's own verdict is that no clean population-limit
  dynamics law exists in tractable form).
- The finite-data errors-in-variables mechanism for the residual alpha bias (P5).
- The RESIDUAL SPEED constant in smooth-map equivalence: Theorem P7 PROVES the
  endpoint and rank are map-invariant and that the speed difference is the single
  scalar `eta_g m_g(alpha*)`; the Argued part is only that one scalar `eta`
  absorbs it across an item population with a spread of true alphas (exact at one
  alpha, approximate across the population, the source of the `+-0.002` clustering).
  The two exceptions (ReLU, square) are Proved, not Argued.
- Scalar alpha-space preconditioning insufficiency (P8); it is the direct-alpha
  case of Theorem P7 (a scalar preconditioner changes only the canonical-flow
  speed, never the fixed point or rank).
- That the validated accelerator is REPRESENTATION decoupling, a coupled-two-block
  conditioning effect, not a scalar reparameterization (P9). The block Hessian, its
  condition number, and the GD iteration count `t^sh = kappa log(1/epsilon)` are
  PROVED under A1-A6; the Argued parts are the block-diagonal reduction (A5), the
  matched-Jacobian-norm regularity (A6), and the orthogonality-consistency reading.
- That theta wants the encoder input NARROW because encoder-input width inflates
  the variance of the amortized `theta_hat` (item-identity leakage), DISTINCT from
  alpha's conditioning and OUTSIDE the free-table invariant (P10). The clean part
  is the linear-amortizer `O(W/n)` variance / d.o.f. decomposition (P10a, Proved);
  the lift to the LSTM and the over-training INFLATION of that variance are Argued
  (this is the document's softest formalization, flagged as such in 7.6.3).

**Empirical only (this document supports but does not derive).**

- The QUANTITATIVE size of the decoupling / state-conditioning rank benefit and
  its exact K-profile (the study's `delta_alpha` table). The theory predicts the
  SIGN, the parameter-specificity (alpha not beta), and the monotone-in-K trend
  via `kappa(K)`; it does not predict the magnitudes.
- The architecture-independence of the empirical fix (LSTM / Transformer /
  DKVMN) and the across-seed variance collapse. These are robust empirical
  observations; the theory is consistent with them (a stiffer shared flow is
  more seed-sensitive in its slow mode) but does not prove them.
- That the empirical decoupling fix depends on a LEARNED, theta-specific encoder
  pathway that the fixed-pool toys omit. The tractable toys reproduce the rate
  asymmetry and the finite-data rank advantage but not the full
  decoupling-fixability, so the encoder-dependence remains an empirical result.
- The state-conditioned discrimination diagnostics (the residual is a
  directional, not calibrated, detector of context-dependent discrimination; the
  ~30x magnitude shrinkage). The theory explains WHY magnitude is
  under-identified (alpha's low prediction leverage where `theta ~ beta`, P3) but
  the calibration constant is empirical.
- The MAGNITUDE of the theta degradation with width (0.97->0.88) and with
  training (0.97->0.68), and the matched-total-capacity gate dominance. The theory
  (P10, P12) gives the SIGN, the parameter-specificity, and the allocation-not-
  budget structure; the magnitudes and the over-training decay SHAPE are empirical.

**Explicitly NOT claimed (consistent with the study's refutations).**

- No claim that the exponential map is special or optimal. P7 argues the opposite
  (smooth maps tie).
- No claim that a scalar alpha-space preconditioner explains the neural effect.
  P8 argues the opposite.
- No claim that a scalar reparameterization (the exponential's `alpha^2` map, or
  any positive-map preconditioner) accelerates discrimination recovery. This was
  the earlier-plan proposition; it is REFUTED (P7 smooth-map tie, P8 direct-alpha
  control), and P9 relocates the accelerator to the coupled two-block
  representation structure (decoupling), which the scalar control deletes by
  construction.
- No claim of a clean population-limit learning-DYNAMICS LAW. P4b shows the
  endpoint is invariant, so the entire effect is a finite-data plus
  finite-training RATE phenomenon, not an endpoint law.
- No claim that decoupling or a richer alpha readout improves the ENDPOINT.
  Configurations tie at convergence (Finding 3); the benefit is speed and rank at
  a finite budget. (Theta is the exception that proves the rule: its protection is
  a finite-DATA generalization gain, P10, not an endpoint identity, because theta
  is amortized and falls outside the free-table invariant. The joint
  infinite-data-and-training limit erases both alpha's rate gain and theta's
  variance, P12.)
- No claim of a PROVED over-training law for theta. P10's training-time
  inflation of the theta variance is Argued plus Empirical (the peak-then-decay
  curve), not a closed-form trajectory.
- No claim that beta is helped or harmed by the architecture (P11: it is the
  null control).
- No variational content. The whole analysis is a point estimate on a likelihood,
  so it is distinct from the GVEM / IW-GVEM posterior-approximation
  discrimination bias; that boundary (Section 7 of the toy appendix) is untouched.

The single sentence. Under a prediction objective, the model pins each parameter
in proportion to that parameter's information leverage on the predicted response;
alpha's leverage is suppressed by the squared learner-item separation and
vanishes where targeted responses concentrate, so alpha is the low-information,
slow mode. Low information sets the RATE of alpha's recovery, not its endpoint;
the endpoint is invariant to representation choice and to smooth positivity maps
(the free-table invariant), and the conditioning that drives the rate worsens
with the number of answer levels K. The accelerator that works is not a scalar
reparameterization but REPRESENTATION decoupling: giving discrimination its own
item-code block removes the high-curvature ability direction from alpha's block,
cutting the shared block's `kappa = I(theta)/I(alpha)` rate penalty to `O(1)` and
speeding alpha's rank recovery by a factor `~ kappa` that grows with K, with no
change to the fixed point.

The joint sentence (the three-parameter picture). The single item code feeds two
consumers, the ability encoder and the alpha/beta readouts, and the three
parameters pull its width in opposite directions: alpha wants the readout WIDE
(low Fisher, needs capacity, throttled by `kappa`, P9), theta wants the encoder
input NARROW (a wide input inflates the variance of the amortized `theta_hat` by
letting the encoder absorb item identity, a finite-data generalization effect
outside the free-table invariant, P10), and beta is indifferent (high Fisher,
direct readout, the null control, P11). Because a SHARED code ties the
encoder-input width to the readout width, these pressures collide on one knob and
the architecture is forced onto a Pareto frontier, theta down as alpha up, beta
flat (P12a, the gate). DECOUPLING gives each consumer its own width, a narrow
encoder input for theta and beta and a separate wide readout for alpha, so it
ESCAPES the frontier rather than trading along it and is Pareto-dominant (P12b).
The alpha gain is a finite-budget rate effect and the theta gain is a finite-data
variance effect; both vanish only at the joint infinite limit, so the real model,
finite in both, gets both.
