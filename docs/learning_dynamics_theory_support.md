# Theory support for the discrimination-recovery dynamics

A self-contained mathematical support layer for the empirical learning-dynamics
results in `docs/LEARNING_DYNAMICS_STUDY.md`. The role of this document is
modest and local: it derives the information-geometric structure that EXPLAINS
why the discrimination parameter (sharpness, alpha) recovers slowly and less
reliably under prediction training, and it states precisely what that structure
does and does not imply. It is a support layer for a deep-learning paper, not a
standalone theorem. Every claim below is marked Proved, Argued (heuristic), or
Empirical.

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
| P3 | Prediction sensitivity to alpha is `dp/dalpha = (theta - beta) w`, which is `I(alpha)^{1/2}`-scaled. The prediction is first-order blind to alpha wherever responses concentrate near `theta = beta`. | Proved | Finding 2 (weak alpha dependence of the prediction). |
| P4 (rate, not endpoint) | Near an identifiable optimum, gradient flow on each direction decays at a rate set by that direction's curvature, which is its Fisher information. Low Fisher therefore sets the RECOVERY RATE. At the population limit, every reachable zero-residual direction still converges to truth, so low Fisher does NOT set an endpoint bias. | Proved (local) + Argued (global, via the free-table invariant) | Finding 3 (a finite-data SPEED/rank effect with no endpoint advantage; ties at convergence). |
| P5 (finite-data bias is errors-in-variables) | At finite repetitions the amortizer input is a noisy encoding of `theta*`; through the bilinear `z = alpha(theta - beta)` this contaminates alpha. The bias scales with the amortizer input noise and vanishes as repetitions to infinity. | Argued + Empirical (toy) | Finding 3, Finding 4 (the shortfall behaves like a finite-data errors-in-variables effect that vanishes as repetitions grow). |
| P6 (K worsens conditioning) | For GPCM, `I(alpha)` rises in absolute terms with K but `I(theta)` rises faster, so the stiffness ratio `I(theta)/I(alpha)` grows monotonically with K. The shared-code flow becomes more ill-conditioned, so alpha's rate disadvantage grows with K. | Proved (Fisher forms) + Empirical (ratio table) | Finding 3 (the benefit of an alpha-specific readout GROWS with K) and the K=2 sign flip. |
| P7 (positive-map neutrality) | Any smooth, strictly monotone positive map induces an `alpha`-space preconditioner `m_g(alpha) = [g'(g^{-1}(alpha))]^2`. After matching the effective initial alpha and tuning the learning rate per map, the leading-order local rate is map-independent up to a constant absorbed by the learning rate. Only non-smooth or non-monotone maps (ReLU dead zone, square sign-folding) genuinely lag. | Argued | Finding 4 (exp is not special; smooth maps tie; only non-smooth/non-monotone maps lag). |
| P8 (scalar preconditioning is insufficient) | A scalar `alpha`-space preconditioner acts on a single direction and cannot reproduce the recovery effect that lives in the coupling between the shared representation directions. With `theta, beta` frozen and only the scalar `alpha_j` optimized, all reasonable `alpha`-space update rules converge to the same band. | Argued + Empirical (direct-alpha control) | Finding 4 (the scalar preconditioner-only explanation is refuted by the direct-alpha control). |
| P9 (representation coupling is the mechanism) | The validated accelerator is REPRESENTATION DECOUPLING, not scalar reparameterization. A shared item code gives the code's Hessian block a two-direction structure with condition number `kappa = I(theta)/I(alpha) >> 1`, so a single-step-size flow resolves the alpha-aligned component at a rate throttled by `kappa`. A separate alpha code has curvature `~ I(alpha)` only and resolves alpha at its own uncontested rate. The speedup factor is `O(kappa)`, growing with K. Same fixed point in both cases (free-table invariant), so it is a pure rate / early-stopping effect. | Proved (local quadratic rate) + Argued (global endpoint via P4b) + Empirical (rung-7, K-sweep, N-sweep) | Replaces the REFUTED `alpha^2`-preconditioner proposition; anchors the gate (d1), trajectory (d2), 28x gradient (d3), stiffness rank 0.89 (d4), N-sweep (d5). |

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
pull below is linear in `r`.

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
the slow (alpha) mode to resolve.

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

### 6.1 Induced preconditioner

Let `alpha = g(a)` for a differentiable strictly increasing positive map `g`,
with `a` the raw neural output. Gradient flow in raw space induces a flow in
alpha space:

```
da/dt = -dL/da = -g'(a) dL/dalpha,
d(alpha)/dt = g'(a) da/dt = -[g'(a)]^2 dL/dalpha = -m_g(alpha) dL/dalpha,
m_g(alpha) = [g'(g^{-1}(alpha))]^2 >= 0.
```

So each smooth positive map is a different DIAGONAL PRECONDITIONER `m_g` on the
alpha-space gradient flow. Examples:

```
exp:       g(a) = e^a,            m_exp(alpha)      = alpha^2
softplus:  g(a) = log(1+e^a),     m_softplus(alpha) = (1 - e^{-alpha})^2
sigmoid:   g(a) = sigma(a),       m_sigmoid(alpha)  = alpha^2 (1-alpha)^2   (0<alpha<1)
```

These are genuinely different functions of alpha, which is why a positive map is
not a neutral positivity constraint in raw space. The research plan's mechanism
section records the same `m_g` derivation; we use it to make the neutrality
argument precise.

### 6.2 Why smooth maps tie after matched init and LR

**Proposition P7 (smooth-map neutrality, to first order).** Consider two smooth
strictly-monotone positive maps `g_1, g_2`, each initialized so the EFFECTIVE
alpha starts at the same value `alpha_0` (matched-init), and each with its own
tuned learning rate. Near an identifiable optimum `alpha*`, the local recovery
rate under map `g` is

```
e(t) = alpha(t) - alpha*,   d e/dt ~ -eta * m_g(alpha*) * H * e,
e(t) ~ e(0) exp[ -eta m_g(alpha*) H t ],
```

where `H = I(alpha*)` is the local curvature (P2) and `eta` the learning rate.
The map enters ONLY through the scalar product `eta m_g(alpha*)`. Tuning `eta`
per map absorbs the constant `m_g(alpha*)`, so after matched init and per-map LR
the leading-order local rates COINCIDE. Smooth maps are first-order
recovery-equivalent.

*Status: Argued* (local linearization; the constant `m_g(alpha*)` is exactly the
degree of freedom the per-map learning-rate sweep removes).

Two honest qualifications, both consistent with the empirical "mixed" verdict.

- `m_g` varies with alpha, so the absorption by a single scalar `eta` is exact
  only at one alpha and approximate across an item population with a spread of
  true alphas. This is why the smooth maps cluster tightly but not identically
  (the study's `+-0.002` spread), and why the residual differences are too small
  and too inconsistent to be load-bearing. The argument predicts a tie up to
  this second-order term, which is what the data show.
- The endpoint is map-independent by P4b regardless (smooth reparameterization
  multiplies a residual-linear pull). So any map difference can live only in the
  transient, and the transient difference is absorbed by LR to first order. Both
  the endpoint invariance and the rate near-equivalence point to a tie.

### 6.3 Why non-smooth or non-monotone maps genuinely lag

The neutrality argument requires `g` smooth and strictly monotone, the conditions
that make `m_g(alpha) > 0` everywhere and the chain-rule absorption valid. Two
failure modes break it, matching the empirical "only non-smooth or non-monotone
maps lag":

- **ReLU / clipped raw (dead zone).** `m_relu = 1` where active and `0` where
  inactive. The preconditioner is zero on the inactive set, so an item whose raw
  output is in the dead region receives NO alpha gradient and cannot recover until
  some other dynamics pushes it active. This is a genuine, non-absorbable rate
  penalty (a vanishing gradient, not a rescalable one), and no LR choice fixes a
  zero.

- **Square `g(a) = a^2 + epsilon` (sign folding).** `g` is non-injective:
  `g^{-1}` is two-valued and the raw dynamics carry a spurious sign symmetry,
  giving ambiguous trajectories and saddle structure near `a = 0` that a strictly
  monotone map does not have.

*Status: Argued.* These are the maps the study finds genuinely lag, and the
mechanism (a literally vanishing or sign-folded preconditioner, not a rescalable
constant) is why they fall outside P7's smooth-monotone hypothesis. P7 plus these
two exclusions is the precise mathematical content of "exp is not special; all
smooth positive maps tie; only non-smooth or non-monotone maps lag."

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

Restrict the Gauss-Newton Hessian to the item-`j` code block. The block collects
the curvature contributed by every readout that uses `e_j`. Each readout
contributes a rank-one (per response) outer product `w (d z / d e_j)(d z / d e_j)^T`
with `d z / d e_j` the readout's Jacobian into the code (A2).

**Proposition P9a (block curvature).** Summing the per-response Gauss-Newton
contributions over the responses that touch item `j`, the SHARED code block has
Hessian

```
H^sh_j  ~  I(alpha_j) (a a^T)            (alpha-readout direction)
        +  I(beta_j)  (b b^T)            (beta-readout direction)
        +  sum_i I(theta_i)-weighted (v_{ij} v_{ij}^T)   (ability-pathway direction)
```

(schematically; the alpha direction carries curvature `~ I(alpha_j) = sum_i w_ij
(theta_i - beta_j)^2`, the ability direction carries curvature `~ I(theta)` summed
over the learners who saw item `j`, from P2). The block therefore has an
ability-aligned direction with curvature `~ I(theta)` and an alpha-aligned
direction with curvature `~ I(alpha)`, so its condition number is

```
kappa = lambda_max / lambda_min  ~  I(theta) / I(alpha)  >> 1,
```

the same stiffness ratio as P4a/P6. The DECOUPLED alpha block `e_j^al` receives
contributions ONLY from the alpha (and beta) readouts, so its largest curvature
along the alpha direction is `~ I(alpha)` with NO ability-aligned high-curvature
direction; its alpha-relevant condition number is `O(1)`.

*Proof.* Each readout's Gauss-Newton contribution is `E[w (dz/de_j)(dz/de_j)^T]`
(A2); summing the contributions of the readouts that share the block gives the
displayed sum. The curvature magnitudes are the per-direction Fisher informations
of P2 (2PL) / P6a (GPCM). The shared block contains both the ability-pathway term
(magnitude `I(theta)`) and the alpha-readout term (magnitude `I(alpha)`), so its
eigenvalue spread is bounded below by their ratio; the decoupled alpha block omits
the ability-pathway term by construction. *Status: Proved* (local, under A1-A5).
∎

The directions need not be orthogonal for this to hold; `kappa` is an eigenvalue
ratio, not an angle. We return to the orthogonality point in 7.5.4 because it is
exactly the Phase-2 subtlety the user flagged.

### 7.5.3 The shared block throttles the alpha component; the decoupled block does not

Linearize gradient flow on the code block (A1-A3). With `delta_j = e_j - e_j*`,

```
d(delta_j)/dt = -H_j delta_j,
```

and in the eigenbasis of `H_j` each mode `c` decays as `e^{-lambda_c t}` (this is
P4a applied to the block). Decompose `delta_j` along the alpha-aligned eigenvector.

**Proposition P9b (rate throttling and the speedup factor).** Under a single
shared step size (A3):

- SHARED block. The alpha-aligned component decays at rate `lambda_alpha ~
  I(alpha)`, while the fast ability-aligned component decays at `lambda_theta ~
  I(theta)`. A single step size `eta` is bounded by stability on the FAST mode,
  `eta < 2 / lambda_max ~ 2 / I(theta)`. The slow alpha mode then contracts per
  unit time by at most `eta lambda_alpha ~ I(alpha) / I(theta) = 1 / kappa`. So the
  time to resolve the alpha-aligned component to a fixed tolerance scales as

  ```
  tau_alpha^sh  ~  1 / (eta lambda_alpha)  ~  kappa / lambda_theta-scale  =  O(kappa).
  ```

  Alpha's ordering is the LAST thing the shared block resolves: the code is shaped
  first along the high-curvature ability/location directions, and the alpha
  direction is dragged along the slow mode at a rate divided by `kappa`.

- DECOUPLED block. The alpha block has no high-curvature ability direction (P9a),
  so its step size is bounded by `lambda_alpha` itself and the alpha component
  contracts at `O(1)` per unit time:

  ```
  tau_alpha^dc  ~  1 / lambda_alpha,    independent of I(theta).
  ```

The speedup factor is the ratio of resolution times,

```
speedup  =  tau_alpha^sh / tau_alpha^dc  ~  kappa  =  I(theta) / I(alpha).
```

*Proof.* Standard stiff-flow argument. Under one step size the stability ceiling
is set by `lambda_max` (else the fast mode diverges), so the slow mode's per-step
contraction is `eta lambda_min <= 2 lambda_min / lambda_max = 2/kappa`; the number
of steps to a fixed tolerance is `O(kappa)`. Removing the high-curvature direction
(decoupling) removes the ceiling on `eta` that the slow mode pays, giving an
`O(1)` resolution time. *Status: Proved* (local quadratic, single step size,
A1-A5). ∎

**K-scaling (Proved + Empirical).** By P6, `kappa(K) = I(theta)/I(alpha)` climbs
monotonically with K (the toy table `1.03 -> 2.29` for `K = 2..6`, extended
`0.96 -> 5.22` for `K = 2..11`). Therefore the speedup factor grows with K:
decoupling buys more as the number of answer levels rises. This is the derived
form of Finding 3's K-growth and of the K=2 near-tie (at `kappa ~ 1` the two
blocks have the same conditioning, so the only difference is the decoupled block's
extra estimation variance, which is why K=2 is net-neutral or slightly negative).

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
- The local linear rate law: a direction's recovery time scale is the inverse of
  its Fisher information, ordering alpha as the slow mode with flow condition
  number `kappa ~ I(theta)/I(alpha)` (P4a).
- The free-table invariant: a reachable zero-residual optimum is stationary,
  globally minimal, and invariant to representation sharing and to smooth
  reparameterization, so the endpoint carries no representation-choice or
  positive-map advantage (P4b).
- The GPCM stiffness `kappa(K) = I(theta)/I(alpha)` grows monotonically with K
  (P6a, P6).
- The shared item code block's condition number is `kappa = I(theta)/I(alpha)`
  while the decoupled alpha block's is `O(1)`; under a single step size this gives
  an `O(kappa)` rate penalty on alpha in the shared case and a `~ kappa` decoupling
  speedup that grows with K, with the SAME fixed point in both cases (P9, the
  validated mechanism, replacing the refuted scalar-preconditioner claim).

**Argued (heuristic, locally rigorous but not a global theorem).**

- That low Fisher therefore sets the RECOVERY RATE and not the endpoint at
  infinite data and training (P4); this combines the proved local rate law with
  the proved endpoint invariant, but a single closed-form global trajectory is
  not available (and the study's own verdict is that no clean population-limit
  dynamics law exists in tractable form).
- The finite-data errors-in-variables mechanism for the residual alpha bias (P5).
- Smooth-map recovery neutrality after matched init and LR, with the non-smooth /
  non-monotone exclusions (P7).
- Scalar alpha-space preconditioning insufficiency (P8).
- That the validated accelerator is REPRESENTATION decoupling, a coupled-two-block
  conditioning effect, not a scalar reparameterization (P9); the block-diagonal
  reduction (A5) and the orthogonality-consistency reading are the Argued parts.

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
  a finite budget.
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
```
