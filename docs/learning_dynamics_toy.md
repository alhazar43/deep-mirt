# Learning-dynamics toy for the alpha-theta coupling: derivation and verdict

> THEORY APPENDIX of `docs/LEARNING_DYNAMICS_STUDY.md`. Standalone derivations
> (gauge proof, rank-wall proofs, free-table invariant Sec 10, GPCM Fisher table
> Sec 11.5, rung-7 GD-vs-Adam controls Sec 12). The main study links here.

Phase 3 of `LEARNING_DYNAMICS_STUDY.md`. Goal: build the simplest analytically
tractable 2PL model that could exhibit the alpha-recovery degradation under
representation sharing, and decide via the global-optimum-vs-gradient-flow
discriminator whether the cause is expressivity, dynamics, or neither.

Opened 2026-06-16. Status: the minimal model is SOLVED. Verdict below.

---

## Verdict first

**The minimal model does NOT exhibit the coupling.** In a point-estimate 2PL
with a free per-person ability `theta_i`, population gradient flow on a shared
item representation recovers `alpha` without bias, once two trivial confounds
are removed:

1. **The 2PL gauge.** The loss is invariant under `theta -> s theta + t`,
   `beta -> s beta + t`, `alpha -> alpha / s`. A uniform `alpha` shrinkage paired
   with a `theta` inflation is a coordinate choice, not bias. Every raw "alpha
   biased low" number we measured (mean log-ratio -0.13 to -0.34) collapsed to
   `0.0000 +- 0.0001` once `theta`'s scale was quotiented out. This confound
   alone explains the entire apparent shrinkage in the toy.

2. **Embedding rank.** A scalar code (`d=1`, or a rank-1 shared encoder) locks
   `(alpha_j, beta_j)` onto a one-parameter curve and genuinely cannot express
   independent discriminations. That is an EXPRESSIVITY wall, and it shows up in
   the fixed-point algebra as collinearity of the two gradient pulls (Sec 3).
   It is not dynamics. With `d >= 2` the wall is gone.

With the gauge fixed and `d >= 2`, the discriminator returns the cleanest
possible answer:

- **(a) Expressivity:** the global minimizer of the shared architecture recovers
  unbiased `alpha` (gauge-fixed log-bias `0.0000`, item-wise corr `1.0000`). The
  architecture is fully capable. NLL reaches the irreducible entropy floor.
- **(b) Dynamics:** gradient flow from random init CONVERGES to that global
  minimizer on every seed (8/8, NLL at floor to 6 decimals, gauge-fixed bias
  `0.0000 +- 0.0000`). The good optimum is not merely reachable, it is reached.

There is no gap between (a) and (b). **In the minimal model the flow is not
captured.** Shared and decoupled are statistically indistinguishable at the
optimum. Reported honestly: at this altitude the effect is neither expressivity
nor dynamics; the toy is well-conditioned.

**Why, in one line:** the free per-person `theta_i` absorbs all the gauge
freedom and carries enormous Fisher information, so the item channel is never
forced to choose between serving ability and serving discrimination. The
competition the real model suffers from is structurally absent here.

**The minimal missing ingredient (the climb-up, Sec 6).** The real model does
not have a free `theta_i`. It reads `theta` from the SAME shared encoder state
that feeds the `alpha` head, and it trains by SGD over interaction streams. The
coupling needs `theta` to compete with `alpha` FOR the shared code. The toy
deliberately gave `theta` its own private parameter, which is exactly the
decoupling that the real model lacks on the ability side. The minimal model that
should exhibit capture is therefore: **shared bottleneck `h_i` of width `m`,
read by BOTH a theta-head and an alpha-head, trained by SGD.** Specification and
the predicted gradient-capture mechanism are in Sec 6.

A secondary, separate effect we did find and ruled out: finite-data MLE bias on
`alpha` (Sec 5). It is real, it shrinks like `O(1/reps)`, and it is IDENTICAL
for shared and decoupled, so it is not the phenomenon either.

---

## 1. The minimal model and its gradient flow

### 1.1 Setup

Persons `i = 1..N` with ability `theta_i in R` (free scalar each). Items
`j = 1..J` with a shared embedding `e_j in R^d`. Shared linear read-out maps
with shared weights `a, b in R^d`:

```
alpha_j = g(a . e_j),     beta_j = b . e_j,     g = exp  (positivity, see note)
z_ij    = alpha_j (theta_i - beta_j),     p_ij = sigma(z_ij).
```

Note on `g`. We use `g = exp` so `alpha > 0` automatically and `log alpha` is
linear in the code, which is the natural scale for the multiplicative gauge and
for the Fisher analysis. `softplus` gives the same verdict numerically. Identity
is admissible only if positivity is not enforced; it makes `alpha` and `beta`
exactly proportional in `d=1` (a sharper version of the curve-locking in Sec 3),
so it is a worse, not better, minimal choice.

Data are generated from known true `(alpha*, beta*, theta*)`. The population
loss is the expected NLL, i.e. the cross-entropy between the true response
probability `p*_ij` and the model `p_ij`:

```
L = (1/NJ) sum_{i,j} [ -p*_ij log p_ij - (1 - p*_ij) log(1 - p_ij) ].
```

`L` is minimized iff `p_ij = p*_ij` for all `i,j`; its value there is the
irreducible Bernoulli entropy floor.

### 1.2 Per-parameter gradients

Let the logit-residual be `r_ij = p_ij - p*_ij` (this is `dL/dz_ij` up to the
`1/NJ`). Using `dz/dtheta = alpha`, `dz/dbeta = -alpha`, `dz/dalpha = theta-beta`:

```
dL/dtheta_i = (1/J)  sum_j  r_ij alpha_j                         (ability)
dL/dbeta_j  = -(1/N) sum_i  r_ij alpha_j                         (difficulty)
dL/dalpha_j =  (1/N) sum_i  r_ij (theta_i - beta_j)              (discrimination)
```

The gradient flow is `d/dt of each parameter = -` its gradient above.

### 1.3 The shared-embedding gradient splits into two pathways

Chain through `alpha_j = exp(a . e_j)` and `beta_j = b . e_j`:

```
dL/de_j = (dL/dalpha_j)(dalpha_j/de_j) + (dL/dbeta_j)(dbeta_j/de_j)
        = [ alpha_j  dL/dalpha_j ] a   +   [ dL/dbeta_j ] b
        =       G^alpha_j         a   +        G^beta_j   b.                (*)
```

This is exactly the requested decomposition: the gradient on the shared code is
**an alpha-pathway pull `G^alpha_j a` plus a (beta/theta)-pathway pull
`G^beta_j b`**, with scalar magnitudes

```
G^alpha_j = alpha_j (1/N) sum_i r_ij (theta_i - beta_j),
G^beta_j  = -alpha_j (1/N) sum_i r_ij.
```

The shared stationary code solves `G^alpha_j a + G^beta_j b = 0`: the embedding
sits where the two pulls cancel, a **magnitude-weighted blend of the alpha
direction `a` and the difficulty direction `b`**.

---

## 2. The discriminator (expressivity vs dynamics)

### 2.1 (a) Expressivity: is the global optimum unbiased?

For `d >= 2` with `a, b` linearly independent, the global minimizer drives every
`p_ij` to `p*_ij`, hence (after fixing the gauge) every `(alpha_j, beta_j)` to
truth. Numerically (J=10, near-population N): NLL reaches the entropy floor
`0.532804` to six decimals, item-wise `alpha` corr `= 1.0000`, gauge-fixed
log-bias `= 0.0000`. **The shared architecture is expressively capable.** So the
limit, if any, is not expressivity (for `d >= 2`).

### 2.2 (b) Dynamics: where does the flow converge?

From standard random init (`0.3 * N(0,1)`), Adam-to-convergence then L-BFGS
polish, 8 seeds: every seed lands at the floor (`max excess 0.000000`) with
gauge-fixed `alpha` bias `0.0000 +- 0.0000` and `alpha` corr `1.0000`. **The
flow reaches the capable optimum.** No capture.

### 2.3 The discriminator's reading

(a) capable AND (b) reaches it => the failure is NEITHER expressivity NOR
dynamics in the minimal model. The honest conclusion is that the minimal model
is well-conditioned, and the real-model coupling requires an ingredient the
minimal model omits (Sec 6). This is the skeptic's outcome the study plan
anticipated, and it is informative: it rules out the two simplest stories and
localizes the cause.

### 2.4 Why the free table cannot be captured (the proof, not just the numerics)

From (*), the shared stationary condition is `G^alpha_j a + G^beta_j b = 0`. For
`d >= 2` with `a, b` linearly independent, this vector equation forces BOTH
scalar coefficients to vanish:

```
G^alpha_j = 0   AND   G^beta_j = 0   for every item j.
```

`G^beta_j = 0` is `sum_i r_ij = 0` and `G^alpha_j = 0` is
`sum_i r_ij(theta_i - beta_j) = 0`; together with the `theta_i` stationary
conditions these are exactly the score equations whose unique interior solution
(modulo gauge) is `p_ij = p*_ij`. **The alpha and beta stationary conditions are
imposed separately**, so there is no trade-off: a free per-item `e_j` with
`d >= 2` has enough degrees of freedom to zero both pulls at once. The two
pathways share the code but do not have to compromise on it.

The compromise only appears when `a` and `b` are collinear, which is the `d=1`
case (Sec 3): then `G^alpha_j a + G^beta_j b = 0` is ONE equation that cannot
zero two independent residual functionals, and the code settles at a forced
blend that biases `alpha`. That blend is real, but it is an EXPRESSIVITY
artifact of the rank-1 code, not gradient capture.

---

## 3. The shared fixed point and the alpha bias (the d=1 / rank-1 case)

Set `d = 1`: `e_j, a, b` scalars, `alpha_j = exp(a e_j)`, `beta_j = b e_j`.
Eliminating `e_j = beta_j / b` gives the **curve-locking constraint**

```
alpha_j = exp( (a/b) beta_j ),
```

so `alpha` is a deterministic function of `beta`: the model has ONE free item
scalar masquerading as two parameters. Unless the true `(alpha*_j, beta*_j)` lie
on this single exponential curve (they generically do not), the optimum CANNOT
match truth and `alpha` is biased.

The stationary code is the single-equation blend
`G^alpha_j a + G^beta_j b = 0`, i.e.

```
(a/b) = - G^beta_j / G^alpha_j  =  ( sum_i r_ij ) / ( sum_i r_ij (theta_i - beta_j) ).
```

The right side is set by the difficulty/ability residuals (the high-Fisher
pathway, Sec 4), so `a/b` is pinned by `beta`-fitting and `alpha` is whatever
the curve then delivers, which **shrinks `alpha` toward a common value**.
Numerically (d=1 shared): item-wise `alpha` corr collapses to `0.07-0.54`,
`alpha^` is squashed to a near-constant, and NLL sits ABOVE the floor
(`floor + 0.004`), the signature of an expressivity wall, not a captured-but-
reachable optimum.

**This is the cleanest in-model illustration of the gradient blend biasing
alpha, but it is expressivity, not dynamics.** A learning-dynamics claim cannot
rest on `d=1`.

---

## 4. The Fisher bridge

The two pathway magnitudes in (*) are governed by the Fisher information each
parameter carries, which ties the classical statistical fact (alpha is the
low-information IRT parameter) to the gradient picture.

Per response `(i,j)`, with weight `w_ij = p_ij(1-p_ij)` (the Bernoulli
variance), the single-observation Fisher informations are

```
I(theta_i) = sum_j w_ij alpha_j^2            ~  alpha^2 * (# items)
I(beta_j)  = sum_i w_ij alpha_j^2            ~  alpha^2 * (# persons)
I(alpha_j) = sum_i w_ij (theta_i - beta_j)^2.
```

Two facts make `alpha` the weak pathway:

1. **The `(theta - beta)^2` factor.** `w_ij = p(1-p)` is largest exactly where
   `theta_i ~ beta_j`, i.e. where `(theta_i - beta_j)^2 ~ 0`. The responses that
   are most informative about everything else carry almost NO information about
   `alpha`. `I(alpha)` is a product of a large weight and a small lever arm,
   suppressed relative to `I(beta) ~ alpha^2 sum_i w_ij`, which has the weight
   with no vanishing factor. This is the textbook reason discrimination is the
   hardest 2PL parameter to estimate.

2. **The pathway gradients scale with these informations.** Near the solution,
   linearize the residual `r_ij ~ w_ij dz_ij`. Then

   ```
   G^beta_j  = -alpha_j sum_i r_ij             ~ alpha_j  sum_i w_ij (.)   (carries I(beta)/alpha)
   G^alpha_j =  alpha_j sum_i r_ij(theta_i-beta_j) ~ alpha_j sum_i w_ij (theta_i-beta_j)(.)   (carries I(alpha))
   ```

   so the ratio of pathway strengths on the shared code is, to leading order,

   ```
   |G^beta_j| / |G^alpha_j|   ~   I(beta_j) / [alpha_j sqrt(I(alpha_j))]-type scale,
   ```

   i.e. **the difficulty/ability pathway dominates the alpha pathway in exactly
   the ratio by which their Fisher informations differ.** The shared code is
   pulled hardest along `b` (high Fisher) and only weakly along `a` (low Fisher).

This is the precise statement of "alpha's known low Fisher information is what
makes its gradient pull weak and lose the shared embedding." The crucial caveat
the discriminator forces: in the FREE-table model the weak pull still reaches
`G^alpha_j = 0` because there is no competitor on `e_j`'s alpha-direction (Sec
2.4). Low Fisher slows alpha down, it does not bias it, as long as alpha owns a
direction of the code. **The Fisher asymmetry becomes a BIAS only when alpha
must share its direction with a higher-Fisher parameter** (rank-1 collinearity
in Sec 3, or theta-on-the-bottleneck in Sec 6). The bridge therefore predicts
exactly where the coupling will and will not appear, which the numerics confirm.

---

## 5. Finite-data MLE bias (found, measured, ruled out)

Replacing the population objective with `reps` Bernoulli draws per `(i,j)` and
refitting (shared rank-2 encoder vs decoupled, gauge-fixed):

```
reps   shared gfix-bias        decoupled gfix-bias       alpha-corr
   1   +0.831 +- 0.094         +0.829 +- 0.097           0.89
   4   +0.061 +- 0.031         +0.061 +- 0.031           0.998
  20   +0.015 +- 0.008         +0.015 +- 0.008           0.999
 200   -0.000 +- 0.004         -0.000 +- 0.004           0.999
```

There is a real positive `alpha` bias at small samples, it decays like
`O(1/reps)` toward zero, and it is **identical for shared and decoupled at every
sampling level.** This is the classical finite-sample bias of the 2PL MLE (the
score is nonlinear in `alpha`), NOT a representational-coupling effect:
decoupling does nothing to it. We record it so it is not mistaken for the
phenomenon. The real-model alpha degradation is fixed BY decoupling; this one is
not, so they are different effects.

---

## 6. Climb-up: the minimal ingredient that produces the coupling

The minimal model is well-conditioned because `theta_i` is a free per-person
scalar. That gives `theta` (i) its own private parameter, never competing with
`alpha` for the code, and (ii) ownership of the entire gauge orbit, which it
traverses for free. The real model has NEITHER property: `theta` is amortized,
read from the shared encoder state that also feeds the heads. So the ranked
candidates and the verdict on each:

1. **`d=1` / rank-1 code** -- produces a bias, but it is EXPRESSIVITY (curve
   locking, Sec 3). Rejected as the dynamics mechanism.
2. **`J > 2` items, `d >= 2`** -- no change, still well-conditioned (tested
   J=8,10,12). Rejected.
3. **Shared linear encoder `e_j = W x_j` (rank >= 2) instead of a free table**
   -- still reaches the floor with unbiased gauge-fixed alpha (rank-2 encoder:
   NLL at floor, corr 0.999, bias 0.000). Cross-item parameter sharing through
   `W` is NOT sufficient. Rejected. (Rank-1 encoder fails, but for the same
   expressivity reason as 1.)
4. **Finite data / SGD sampling** -- adds the `O(1/reps)` MLE bias, identical
   shared vs decoupled (Sec 5). Not the phenomenon by itself. Rejected as the
   coupling, retained as a nuisance to control.
5. **Amortized theta sharing the bottleneck with alpha (the prediction).** Give
   each person raw features and read ability through a SHARED code of width `m`
   that ALSO feeds the alpha head:

   ```
   h = phi(W_shared * input)  in R^m         (the shared bottleneck)
   theta = u . h                              (ability read-out)
   alpha-relevant signal also read from h     (discrimination read-out)
   ```

   Now the theta read-out vector `u` and the alpha read-out compete for the SAME
   `m` directions of `h`. The Sec-4 Fisher bridge then bites with full force:
   theta carries `I(theta) ~ alpha^2 * (#items per person)`, accumulated over
   every interaction in the person's stream, while alpha carries the suppressed
   `I(alpha)`. The shared `h` is pulled to align its dominant directions with
   the high-Fisher ability signal first; alpha is left to whatever low-variance
   residual directions of `h` remain, UNDER-RESOLVED and seed-sensitive. This is
   the gradient capture. Decoupling (a separate `h_alpha` for the alpha head)
   removes `u` from alpha's directions and lets the alpha read-out own its
   subspace, exactly the fix the real model shows.

**Prediction to test on rung 5 (matched-total-capacity, gauge-fixed):** shared
`h` of width `m` shows a gauge-fixed alpha bias and high across-seed variance
that PERSIST at the population limit (distinguishing it from the Sec-5 finite-
sample bias, which vanishes); the decoupled model at matched total width does
not. If that holds at `reps -> inf`, it is dynamics. If it too vanishes at the
population limit, the real-model effect is finite-data + SGD interacting with
the bottleneck, a weaker but still honest claim.

The single structural difference between rung 5 and the well-conditioned minimal
model is whether `theta` owns its own read-out subspace. That isolates the
mechanism to: **amortized ability captures the shared representation from the
low-Fisher discrimination signal.**

---

## 7. The variational boundary

This model is a pure POINT ESTIMATE. There is no posterior, no variational
family, no ELBO, no amortized inference network over a distribution. Whatever
`alpha` behavior it shows (gauge, expressivity, finite-sample, or the rung-5
capture) arises from gradient flow on a likelihood, full stop.

The variational-IRT literature (GVEM, IW-GVEM) documents a discrimination-
estimate bias whose origin is the POSTERIOR APPROXIMATION: a factorized or
Gaussian `q` mis-weights the likelihood and the resulting `alpha` estimate is
biased even at the variational optimum, before any optimization difficulty. That
is a STATISTICAL property of the inference family.

The boundary is therefore clean and testable: **our toy exhibits (or, for the
minimal version, fails to exhibit) the alpha effect with no variational
approximation anywhere in it.** If rung 5 produces a persistent, decoupling-
fixable, population-limit alpha bias, it CANNOT be the variational bias, because
there is no `q`. The mechanism is gradient capture of a shared point-estimate
representation, which the variational story does not cover and cannot explain.
The minimal model already establishes half of this boundary: it shows that
point-estimate 2PL with adequate rank has NO intrinsic alpha bias (gauge-fixed),
so any bias the full model shows is attributable to a specific added structure
(the shared bottleneck), not to maximum-likelihood point estimation per se.

---

## 8. What is proved, argued, conjectured

- **Proved (closed form + 8-seed numerics):** the free-table `d >= 2` minimal
  model has its alpha/beta stationary conditions decoupled on the shared code
  (Sec 2.4); the global optimum is unbiased (gauge-fixed) and gradient flow
  reaches it. No capture.
- **Proved (algebra):** `d=1`/rank-1 forces `alpha = exp((a/b) beta)`, an
  expressivity wall; the resulting alpha bias is the single-equation blend of
  Sec 3, not dynamics.
- **Measured:** the raw "alpha biased low" signal in the toy is 100% the 2PL
  gauge; finite-data adds an `O(1/reps)` MLE bias identical across shared and
  decoupled.
- **Argued (Fisher bridge):** the pathway-strength ratio scales with
  `I(beta)/I(alpha)`-type quantities, so alpha is the weak pull; low Fisher
  causes a BIAS only when alpha shares its code direction with a higher-Fisher
  parameter.
- **Conjectured (rung 5, with a concrete test):** amortized theta sharing a
  width-`m` bottleneck with the alpha head captures the representation from
  alpha, producing a population-limit, decoupling-fixable bias. This is the
  minimal structure that should reproduce the real-model phenomenon, and it is
  the next thing to build.

---

## Numerical evidence (scripts were scratch, deleted after this run)

All runs: 2PL, `g = exp`, LogNormal(0,0.4) alpha*, N(0,1) beta*, N(0,1) theta*,
init `0.3 * N(0,1)`, gauge fixed by least-squares regression of `theta_hat` on
`theta_star`. Population objective unless `reps` stated.

- d=1 shared: NLL `floor + 0.004`, alpha-corr 0.07-0.54 (expressivity wall).
- d=2 shared, 8 seeds, Adam+LBFGS: NLL at floor `0.532804` (max excess 0.000000),
  gauge-fixed alpha bias `0.0000 +- 0.0000`, corr `1.0000`.
- d=2 decoupled: identical to shared at the optimum.
- rank-1 shared encoder: alpha-corr -0.11, NLL above floor (expressivity).
- rank-2 shared encoder: NLL at floor, corr 0.999, gauge-fixed bias 0.000.
- finite data: bias `+0.83 -> +0.06 -> +0.015 -> 0.000` for reps `1,4,20,200`,
  identical shared vs decoupled.
```

---

## 9. Rung 5: amortized theta (the decisive test)

Built and run 2026-06-16. This is the model Sec 6 specified: `theta` is no
longer a free per-person scalar, it is AMORTIZED off the same shared item code
that feeds the `alpha` head, so ability and discrimination compete for the
code's directions. Verdict at the head, derivation and the population-limit
table below.

### 9.0 Verdict

**Not dynamics capture. The shared rung-5 effect is FINITE-DATA, and at this
capacity it is not even decoupling-fixable.** The discriminator returns clean on
both axes at the population limit, and the persistence sweep shows the gauge-
fixed `alpha` bias and the across-seed spread BOTH vanish as `reps -> inf`,
identically for shared and decoupled. Concretely:

- **(a) Global optimum unbiased.** The shared amortized architecture's population
  minimizer reaches the entropy floor (excess `0.00003`) with gauge-fixed
  log-bias `-0.003`, item-wise `alpha` rank `1.0000` and pearson `1.0000`, on all
  8 init seeds. No amortization-identifiability wall.
- **(b) Gradient flow reaches it.** Every seed lands at that same optimum. No gap
  between (a) and (b), so no capture.
- **Population-limit persistence: the bias DOES NOT persist.** Sweeping
  `reps = 1, 4, 20, 200, inf`, the shared gauge-fixed bias decays `-3.33 -> -2.12
  -> -0.58 -> -0.18 -> -0.003` and the across-seed spread collapses `1.21 -> 0.22
  -> 0.09 -> 0.04 -> 0.0004`. It is `O(noise in the amortizer input)`, gone at the
  population limit.
- **Decoupling does not remove it (this toy).** Shared and decoupled are
  statistically IDENTICAL at every `reps`, byte-identical at `reps >= 20`, under
  BOTH capacity conventions (total-split and matched-each-full-width). So in this
  minimal amortized model the finite-data bias is architecture-independent, the
  same conclusion as the Sec-5 MLE bias, not the decoupling-fixable phenomenon.

This is the third of the three outcomes the brief named: **finite-data + SGD
interacting with the bottleneck, the honest weaker claim.** It is NOT a
population-limit gradient-flow law, and in the toy it is not even a sharing
effect. The strong "dynamics capture" claim is refuted for the minimal amortized
model.

A control pins the mechanism: clamp `theta = theta*` (oracle, amortizer bypassed)
on the shared item table and `alpha` is recovered exactly (excess `0.000000`,
gauge `s = 1`, bias `0.0000`) on every seed. The shared ITEM TABLE is never the
problem (consistent with Sec 2.4). Everything rung 5 shows enters through the
AMORTIZED-THETA pathway, and only at finite data.

### 9.1 The model

Items `j = 1..J` share a free code `e_j in R^m`. Readouts:

```
alpha_j = exp(a . e_j),   beta_j = b . e_j,        a, b in R^m.
```

Amortized ability: person `i`'s ability is read from a response-weighted pool of
the SAME shared codes,

```
theta_i = u . ebar_i,    ebar_i = (1/J) sum_j s_ij e_j,    u in R^m,
```

with `s_ij` a FIXED person-centered response coding (`s_ij = phat_ij - mean_k
phat_ik`, not learned). The competition is explicit: the single table `{e_j}` is
read by `u` (after pooling), by `a`, and by `b`; the three readout vectors share
the `m` directions of the code.

`z_ij = alpha_j (theta_i - beta_j)`, `p_ij = sigma(z_ij)`, loss = cross-entropy
against the response data. Population objective: replace Bernoulli draws by the
true `p*_ij`; then `s_ij = p*_ij - mean_k p*_ik` is deterministic and the loss is
exactly the `reps -> inf` cross-entropy. Finite `reps`: `reps` Bernoulli draws
per `(i,j)`, `phat` the empirical rate (the Bernoulli sufficient statistic) used
BOTH as the fit target and as the amortizer input.

Decoupled comparison: `theta`-amortizer + `beta` read a code `E_th`; `alpha`
reads its OWN code `E_al`. Two capacity conventions, total-split (`m1 + m2 = m`,
same item-code params as shared) and matched-each (`m1 = m2 = m`, alpha owns a
full-width table). Both tested.

### 9.2 The three-pathway shared gradient (the key equation)

Chaining through the three readouts, the gradient on the shared code now carries
a THIRD pull, the ability pathway, because every person's `theta_i` reads `e_j`
through `ebar_i`:

```
dL/de_j = G^alpha_j a  +  G^beta_j b  +  G^theta_j u,                       (**)

G^alpha_j = alpha_j (1/N) sum_i r_ij (theta_i - beta_j)         (low Fisher)
G^beta_j  = -alpha_j (1/N) sum_i r_ij
G^theta_j = (1/NJ) sum_i rho_i s_ij,   rho_i = sum_k r_ik alpha_k = J dL/dtheta_i
```

`G^theta_j` is the HIGH-Fisher pull: `rho_i` is person `i`'s aggregate ability
residual, summed over all of that person's items, and `s_ij` spreads it back onto
the code. This is the amortized analog of the Sec-1.3 split, with the ability
pathway promoted from a private parameter to a competitor on `e_j`.

**Why there is no capture at the population limit (the rank argument, extended).**
The stationary code solves `(**) = 0`. For `m >= 3` with `a, b, u` linearly
independent, the single vector equation forces all THREE scalar pulls to vanish
SEPARATELY:

```
G^alpha_j = 0   AND   G^beta_j = 0   AND   G^theta_j = 0   for every j.
```

A free per-item code of rank `>= 3` has enough degrees of freedom to zero all
three at once, exactly as in Sec 2.4 with one extra pathway. There is no
trade-off at the optimum, so the optimum is unbiased and (b) reaches it. Capture
would require either rank deficiency (`m < 3`, the `d = 1` curve-locking analog)
or a finite-data mechanism. We used `m = 4 >= 3`, so the optimum is clean, which
the numerics confirm.

**The Fisher bridge sets the RATE, not the endpoint.** The three pulls have
wildly different magnitudes. `G^theta` carries ability Fisher accumulated over
all `N` persons (`~ alpha^2 * #items`), the dominant pull; `G^alpha` is the
suppressed low-Fisher pull (`I(alpha_j) = sum_i w_ij (theta_i - beta_j)^2`, the
`(theta - beta)^2` lever-arm suppression of Sec 4). So the shared code is dragged
FAST along `u` and SLOW along `a`. This is a stiff flow, condition number
`~ I(theta)/I(alpha)`, the reason the optimizer needs Adam plus an L-BFGS polish
to actually reach the floor (gauge scale `s ~ 0.07-0.13`, heavily compressed
theta axis, while gauge-fixed alpha is identical across seeds). But a slow mode
is not a biased mode. With `a` un-contested at the fixed point (rank `>= 3`),
`G^alpha_j -> 0` regardless of how slowly. **Low Fisher slows alpha; it does not
bias it, exactly as Sec 4 predicted, because alpha still owns a code direction.**

### 9.3 The finite-data mechanism (what the bias actually is)

The amortizer input is `S = coding(phat)`, built from the empirical rates. At
finite `reps`, `S` is a NOISY encoding of `theta*`. Measuring how well the column
space of `S` can even represent `theta*` (best linear fit `theta_hat = (1/J) S c`,
`c_j = u . e_j`):

```
reps     1     4     20    200    inf
corr   0.365 0.579 0.812 0.947  1.000   (best achievable corr(col(S), theta*))
```

So `theta_hat` is an errors-in-variables estimate of `theta*`, and since
`z_ij = alpha_j (theta_i - beta_j)` is BILINEAR in `alpha` and `theta`,
attenuation/noise in `theta_hat` contaminates `alpha_hat`. That is the entire
finite-`reps` bias: it tracks the amortizer's input noise and vanishes as
`corr(col(S), theta*) -> 1`. Because both shared and decoupled feed the amortizer
the SAME `S` and inherit the SAME theta-estimation error, decoupling the item
tables does nothing to it. The bottleneck is the amortizer INPUT, not the
item-code sharing, which is why the toy's decoupling has no effect here.

### 9.4 The population-limit table (J=12, N=2000, m=4, gauge-fixed)

6 init seeds; finite-`reps` rows averaged over 3 data seeds; `inf` is the
deterministic population objective. `gfix-logbias` = mean over items of
`log(alpha_hat_gauged / alpha*)`; `rank` = Spearman(`alpha_hat`, `alpha*`);
`spread` = std of the gauge-fixed bias across seeds (the across-seed-variance
signature); `excess` = NLL above the entropy floor.

```
                          gfix-logbias        rank            excess     spread
shared (cap=total)
  reps    1            -3.3342 +- 1.2107   0.7428 +- 0.16   0.042310    1.2107
  reps    4            -2.1229 +- 0.2193   0.8780 +- 0.09   0.057387    0.2193
  reps   20            -0.5794 +- 0.0907   0.9301 +- 0.06   0.032537    0.0907
  reps  200            -0.1847 +- 0.0394   0.9977 +- 0.003  0.009677    0.0394
  reps  inf            -0.0026 +- 0.0004   1.0000 +- 0.000  0.000030    0.0004
decoupled (cap=total: theta+beta read E_th width 2, alpha reads E_al width 2)
  reps    1            -3.8667 +- 1.3259   0.6985 +- 0.32   0.041829    1.3259
  reps    4            -2.1612 +- 0.2134   0.8819 +- 0.09   0.057387    0.2134
  reps   20            -0.5794 +- 0.0907   0.9301 +- 0.06   0.032537    0.0907
  reps  200            -0.1847 +- 0.0394   0.9977 +- 0.003  0.009677    0.0394
  reps  inf            -0.0025 +- 0.0000   1.0000 +- 0.000  0.000030    0.0000
decoupled (cap=matched_each: alpha owns its OWN full width-4 table)
  reps    1            -3.6009 +- 0.9737   0.7661 +- 0.12   0.042135    0.9737
  reps    4            -2.1498 +- 0.2162   0.8765 +- 0.09   0.057387    0.2162
  reps   20            -0.5794 +- 0.0907   0.9301 +- 0.06   0.032537    0.0907
  reps  200            -0.1847 +- 0.0394   0.9977 +- 0.003  0.009677    0.0394
  reps  inf            -0.0030 +- 0.0011   1.0000 +- 0.000  0.000038    0.0011
```

Read it two ways. DOWN each block: the bias and the across-seed spread both decay
toward zero as `reps -> inf`, the finite-data signature. ACROSS blocks at any
fixed `reps`: shared and both decoupled variants coincide (byte-identical at
`reps >= 20`), so decoupling does not touch the bias. The decisive cell is
`reps = inf`: gauge-fixed bias `-0.003` and spread `0.0004` for shared, the same
for decoupled. The phenomenon does not survive the population limit.

### 9.5 Relation to the empirical model and the variational boundary

The real model's `alpha` degradation is gauge-invariant (measured by Spearman)
and IS removed by decoupling, on finite data (`N = 800`). The toy reproduces a
finite-data amortizer bias but NOT its decoupling-fixability, because the toy's
amortizer is a fixed linear pool over the SAME `S` for both architectures, so
decoupling cannot change the theta-estimation error. The real encoder is a
trained sequence model whose theta-pathway and alpha-pathway are decoupled by
giving alpha its own learned read, which can change the effective signal-to-noise
on the alpha direction in a way this fixed-pool toy cannot. So the honest reading
is: the toy localizes the rung-5 bias to the amortizer's finite-data theta error
(a real, point-estimate, non-variational effect, so the Sec-7 boundary still
holds, there is no `q` anywhere), but it does NOT reproduce the decoupling fix,
which therefore depends on structure the minimal amortized model omits, the
LEARNED, theta-specific encoder pathway rather than a fixed shared pool. That is
the next ingredient if the decoupling mechanism itself is to be derived, not just
the bias.

### 9.6 What is proved, argued, measured (rung 5)

- **Proved (rank argument + 8-seed numerics):** for `m >= 3` the amortized shared
  optimum zeroes `G^alpha = G^beta = G^theta` separately, is unbiased
  (gauge-fixed), and gradient flow reaches it. No capture at the population limit.
- **Measured:** the finite-`reps` gauge-fixed bias decays `-3.33 -> -0.003` and
  the across-seed spread `1.21 -> 0.0004` as `reps -> inf`; identical for shared
  and both decoupled conventions; the amortizer-input expressivity
  `corr(col(S), theta*)` rises `0.365 -> 1.000` over the same range, the mechanism.
- **Argued (Fisher bridge):** amortization makes ability the fast mode and
  discrimination the slow mode on the shared code (stiff flow, cond `~
  I(theta)/I(alpha)`), but slowness is not bias when alpha owns a code direction.
- **Verdict:** finite-data + SGD interacting with the bottleneck, not
  population-limit dynamics; in this toy not even decoupling-fixable. The strong
  dynamics-capture claim is refuted for the minimal amortized model.

---

## 10. Rung 6: learned encoder (does decoupling finally bite?)

Built and run 2026-06-16. The single structural change from rung 5: `theta` is read
through a LEARNED, trainable encoder of the response-weighted pool of the shared item
code, `theta_i = scale * u . tanh(W ebar_i + c)`, with `W, c, u` trainable, instead of
rung 5's fixed linear pool `theta_i = u . ebar_i`. The same code feeds `alpha, beta`.
The hypothesis (Sec 9.5): a learned encoder can REALLOCATE the shared code's directions
to serve `theta`, opening a population-limit decoupling gap that the fixed pool could not.

### 10.0 Verdict

**Rung 6 ALSO fails to reproduce the decoupling-fixable effect. Shared == decoupled at
the population limit, at every encoder width, in both the lazy and rich regimes.** The
learned nonlinear encoder reshapes the TRANSIENT flow (the theta pull on the code is now
person-dependent, no longer a fixed vector), but it does not bias the optimum and it does
not make decoupling bite. The third-of-three honest outcome from the brief, again:
**finite-data x amortizer interaction, NOT population-limit dynamics, and not decoupling-
fixable in the toy.** The reason is sharper than rung 5's and is the real lesson:

> A free per-item code `e_j` of any rank `m >= 1` has a zero-residual optimum (`p = p*`),
> and EVERY pathway pull on `e_j` is linear in the residual `r_ij = p_ij - p*_ij`. At
> `r = 0` all pulls vanish at once, whatever their direction structure. Making the theta
> pathway learned and nonlinear changes the SHAPE of the pulls off-optimum, hence the
> trajectory, but not the existence or reachability of the common zero. Capture would need
> the optimum itself to be unreachable (an expressivity wall), and the bilinear 2PL has no
> such wall here: `alpha, beta` come from the free table and `theta` need only be monotone
> in `theta*`, which even a rank-1 tanh encoder achieves (rank 1.0000 at hidden = 1).

### 10.1 The model and the linear-W sanity gate

```
e_j in R^m,   alpha_j = exp(a . e_j),   beta_j = b . e_j,    a, b in R^m
ebar_i = (1/J) sum_j s_ij e_j,   s_ij = phat_ij - mean_k phat_ik   (fixed coding)
theta_i = scale * u . g(W ebar_i + c),   W in R^{h x m}, c, u in R^h, g = tanh
z_ij = alpha_j (theta_i - beta_j),  p = sigma(z).
```

Decoupled: theta-encoder + beta read `E_th`; alpha reads its OWN code `E_al` (matched
each-full-width). Gauge-fixed exactly as before (regress `theta_hat` on `theta_star` for
scale `s`, report `s`-corrected log-bias and Spearman rank).

**Linear-W gate (g = id).** With `g = id`, `theta_i = (u^T W) ebar_i = utilde . ebar_i`:
a linear learned encoder COLLAPSES to rung 5 with effective readout `utilde = W^T u`. It
must reproduce rung 5, and it does: shared == decoupled at every reps, pop-limit clean
(`logbias -0.007/-0.0015`, rank `1.0000`, NLL at floor). This confirms the implementation
and isolates the NONLINEARITY as the only new ingredient rung 6 adds over rung 5.

### 10.2 The mechanism equation (verified to machine precision)

Chaining through all three readouts, the shared-code gradient is (sum-checked against
autograd, max abs diff `1.1e-16`):

```
dL/de_j = G^alpha_j a  +  G^beta_j b  +  sum_i (dL/dtheta_i) (1/J) s_ij v_i,           (***)

G^alpha_j = alpha_j sum_i r_ij (theta_i - beta_j)/(NJ)        (low Fisher, as before)
G^beta_j  = -alpha_j sum_i r_ij/(NJ)
v_i = W^T diag(g'(W ebar_i + c)) u   in R^m                   (PERSON-DEPENDENT direction)
dL/dtheta_i = sum_k r_ik alpha_k /(NJ).
```

The one structural difference from rung 5's (**) is the theta term. In rung 5 every person
pulled `e_j` along the SAME fixed vector `u`, so the rank-`>= 3` argument forced
`G^alpha = G^beta = G^theta = 0` separately. Here each person pulls along her own
`v_i = W^T diag(g') u`, a STATE-DEPENDENT direction set by the encoder Jacobian. The fixed-
three-vector rank argument no longer applies. This is exactly the "reallocation" the
hypothesis wanted, and it is real: the effective theta-readout direction on the code is
trainable and varies across persons.

**Why it still does not bias the optimum.** Every term in (***) is linear in `r_ij`. The
population minimizer drives `p_ij = p*_ij` so `r_ij = 0` for all `i, j` (the free table at
`m >= 1` plus a monotone theta-encoder can express truth), and at `r = 0` all three terms
vanish identically regardless of the `v_i`. The learned encoder changes WHERE the flow goes
before it reaches the zero-residual set and how stiff that flow is, but the zero-residual
set is still the global optimum and is still reachable. Reshaping the off-optimum pulls is
a transient effect, not an endpoint bias. This is the rung-6 form of "low Fisher sets the
RATE not the ENDPOINT" (Sec 4): the nonlinear encoder is an even stiffer, more curved flow,
but a stiffer flow that still converges is not a biased flow.

### 10.3 Q1+Q2 The reps table (J=12, N=2000, m=4, hidden=8, gauge-fixed)

4 init x 3 data seeds (finite reps); `inf` is the deterministic population objective.
Shared vs decoupled (alpha owns its own full-width table).

```
                gfix-logbias        rank      excess     spread
shared
  reps   1    -1.3432 +- 1.1444   0.8357   -0.018994    1.1444
  reps   4    -0.6230 +- 0.0973   0.9295    0.024200    0.0973
  reps  20    -0.2143 +- 0.0321   0.9837    0.012563    0.0321
  reps 200    -0.0307 +- 0.0038   1.0000    0.001860    0.0038
  reps inf    -0.0124 +- 0.0008   1.0000    0.000188    0.0008
decoupled (alpha owns its own table)
  reps   1    -0.7384 +- 1.1616   0.8403   -0.036103    1.1616
  reps   4    -0.6357 +- 0.1217   0.9429    0.023706    0.1217
  reps  20    -0.2113 +- 0.0295   0.9889    0.012252    0.0295
  reps 200    -0.0301 +- 0.0030   1.0000    0.001856    0.0030
  reps inf    -0.0126 +- 0.0005   1.0000    0.000196    0.0005
```

DOWN each block: the bias and across-seed spread decay toward zero as `reps -> inf`, the
finite-data signature (the `reps = 1` spread `+- 1.14` swamps the mean, so that row carries
almost no signal). ACROSS blocks at any fixed `reps`: shared and decoupled coincide within
their seed spread (e.g. `reps = 4` shared `-0.623 +- 0.097` vs decoupled `-0.636 +- 0.122`,
same rank `0.93/0.94`), and at `reps = inf` they are statistically identical (`-0.0124` vs
`-0.0126`). **Decoupling does not bite. Q1: NO.**

### 10.4 Q4 The discriminator (hard-optimized population optimum, 6 seeds)

Driving HARD to the floor (long Adam + 4 L-BFGS rounds, float64) at `reps = inf` removes
the residual under-optimization the light-opt runs leave:

```
shared              logbias -0.0008 +- 0.0005   rank 1.0000   excess 0.000008
decoupled(match)    logbias -0.0006 +- 0.0005   rank 1.0000   excess 0.000005
```

**(a) the global optimum is unbiased** (`-0.0008`, at the entropy floor to 5 decimals);
**(b) gradient flow reaches it** on all 6 seeds; **shared == decoupled**. The `-0.012`
in the light-opt `reps = inf` rows above was UNDER-OPTIMIZATION of the stiff nonlinear
flow, not a property of the optimum. No gap between (a) and (b), so no capture, and the
optimum is not biased: rules out BOTH the dynamics and the identifiability branches of Q4.

### 10.5 Q3 Lazy/rich and width sweeps (population limit, gauge-fixed)

If the effect were a rich feature-learning phenomenon it would appear at small output
scale (rich) and vanish at large scale (lazy/linearized). It appears NOWHERE.

```
out_scale (rich -> lazy)   shared logbias    decoup logbias    GAP(sh-dec)
  0.1                       -0.0025           -0.0019           -0.0005
  1.0                       -0.0124           -0.0126           +0.0002
  5.0                       -0.0126           -0.0128           +0.0002
 20.0                       -0.0126           -0.0128           +0.0002
```

Width sweep (encoder hidden width, hard-opt, `reps = inf`; tests whether a BOTTLENECK
narrower than `m = 4` forces a compromise):

```
hidden     shared logbias    decoup logbias    GAP(sh-dec)   rank
   1        -0.0104           -0.0103           -0.0001       1.0000  (rank-1 encoder)
   2        -0.0065           -0.0061           -0.0004       1.0000
   3        -0.0065           -0.0058           -0.0006       1.0000
   4        -0.0073           -0.0070           -0.0003       1.0000
   8        -0.0105           -0.0107           +0.0001       1.0000
  16        -0.0110           -0.0110           +0.0000       1.0000
```

The GAP stays within `+- 0.0006` of zero across the entire lazy/rich axis and across every
encoder width from a rank-1 bottleneck to `4x` over-capacity, with rank `1.0000` throughout.
**Q3: there is no rich-regime gap and no bottleneck-induced gap.** A rank-1 nonlinear
encoder (the maximal squeeze) still recovers alpha exactly at the population limit, because
a monotone scalar `tanh(w . ebar + c)` suffices to order `theta` and the free item table
carries `alpha, beta` independently.

### 10.6 Relation to rung 5, the empirical model, and the variational boundary

Rung 6 reaches the same verdict as rung 5 by a CLEANER argument. Rung 5 leaned on a rank-
`>= 3` "three fixed vectors" condition; rung 6 shows that condition was not essential, the
nonlinear encoder breaks it (the theta pull is person-dependent, eq. ***) and the optimum is
STILL unbiased, because the zero-residual set annihilates all residual-linear pulls at once.
The deeper invariant: **as long as the per-item code is a free table that can hit `p = p*`,
no readout structure on top of it, fixed pool or learned nonlinear encoder, biases the
population optimum or makes decoupling matter.** The empirical decoupling fix therefore does
NOT come from the learned-encoder property in isolation. The remaining structural
differences between rung 6 and the real model, any of which could be the missing ingredient:

1. **Genuinely sparse / incomplete data with a SHARED amortizer input.** Both toys feed the
   amortizer the same `S` for shared and decoupled, so decoupling cannot change theta's
   estimation error (Sec 9.3). The real encoder is a sequence model whose theta-read and
   alpha-read see DIFFERENT learned projections of the stream; giving alpha its own learned
   read can change the effective SNR on alpha's direction in a way no shared-`S` toy can.
   This is the leading candidate and the next rung if one is built: a learned amortizer
   whose INPUT projection is decoupled, not just the item table.
2. **No free per-item alpha table in the sparse regime.** With few responses per item the
   alpha table itself can starve (the Sec-2.4 "alpha owns a direction" premise fails per
   item); this is a finite-data x sparsity effect the dense toy (`N = 2000`, every (i,j))
   cannot show.
3. **Joint SGD over a single stream with shared optimizer state / no L-BFGS polish.** The
   toy reaches the floor with a hard polish; the real model is early-stopped Adam on a stiff
   landscape, so it lives in the under-optimized regime where the `-0.01..-0.2` transient
   bias is real and where the stiffer SHARED flow (theta-fast, alpha-slow) genuinely lags
   the decoupled flow at a fixed step budget. This is the honest residual story: the
   empirical fix is a RATE / early-stopping effect on a stiff flow, not an endpoint law,
   consistent with the trajectory result (alpha peaks then decays, Phase 1) and the
   gradient-growth result (theta pull grows 28x, Phase 2).

The variational boundary (Sec 7) is untouched: rung 6 is a pure point estimate, no `q`, no
ELBO. Any alpha behavior is gradient flow on a likelihood.

### 10.7 What is proved, argued, measured (rung 6)

- **Proved (sum-checked to 1e-16 + discriminator numerics):** the three-pathway shared
  gradient (***) has a person-dependent theta pull `v_i = W^T diag(g') u`; nonetheless the
  population optimum drives `r = 0`, zeroing all pulls at once, so the optimum is unbiased
  (`-0.0008 +- 0.0005`, floor to 5 decimals) and gradient flow reaches it on all seeds.
- **Measured:** shared == decoupled at every `reps` (table 10.3), across the full lazy/rich
  axis (`out_scale 0.1..20`, GAP `<= 0.0005`), and at every encoder width (`hidden 1..16`,
  GAP `<= 0.0006`), with rank `1.0000` at the population limit throughout. The finite-`reps`
  bias and seed spread decay to zero as `reps -> inf`, the finite-data signature.
- **Argued:** a learned nonlinear encoder reshapes the off-optimum flow (stiffer, curved,
  person-dependent pulls) but cannot bias a zero-residual optimum reachable by a free item
  table; "learned encoder" alone is not the missing ingredient.
- **Verdict:** DECOUPLING STILL DOES NOT BITE. Rung 6 reproduces neither the population-limit
  dynamics law nor the empirical decoupling fix. The honest remaining mechanism is a
  finite-data / sparsity / early-stopping RATE effect on a stiff flow (Sec 10.6), and the
  next ingredient to test is a learned amortizer whose INPUT projection (not just the item
  table) is decoupled, so that alpha's read can carry a different signal-to-noise than
  theta's, which is the only structure all toys so far have withheld from the decoupled arm.

---

## 11. GPCM replication (does the toy hold beyond 2PL?)

Built and run 2026-06-16. Everything above is BINARY 2PL. The real model and data
are GPCM with `K = 4` ordered categories, so the conclusions must be CHECKED there,
not assumed. This section rebuilds every decisive 2PL check in GPCM and reports
gauge-fixed numbers plus Spearman. No argument by analogy; the numbers are below.

### 11.0 Verdict

**The 2PL conclusions TRANSFER to GPCM, with one quantitative refinement and one
new finite-data effect.** Point by point:

1. **Gauge: identical.** GPCM is invariant under the SAME `theta -> s theta + t`,
   `beta -> s beta + t`, `alpha -> alpha / s` (verified to `3e-15`). A raw "alpha
   biased low" is again pure gauge: a planted `s = 1.8` reads as raw log-bias
   `-0.588 = -log 1.8` and collapses to `-0.0000` (rank `1.0000`) after quotienting
   theta's scale. Same artifact, same fix.
2. **Minimal model: clean, with a SHARPER rank wall.** Free `theta_i`, shared code
   `e_j in R^d`, `alpha = exp(a.e)`, `beta_c = B_c.e`. The global optimum is gauge-
   fixed-unbiased (`logbias -0.0000`, rank `1.0000`) and gradient flow reaches it on
   every seed, for `d >= K`. The 2PL wall was at `d < 2`; the GPCM wall is at
   `d < K = 4`, because there are now `K` pathway directions `(a, B_1..B_{K-1})` on
   the code, not two. This is the only structural change to the discriminator.
3. **Rung 5: identical.** The gauge-fixed alpha bias VANISHES at the population limit
   (`-0.82 -> -0.003`) with the across-seed spread collapsing (`0.045 -> 0.0001`),
   and decoupling is INERT (shared == decoupled, byte-identical at `reps >= 4`, under
   both capacity conventions). Finite-data, not dynamics, not sharing. Same as 2PL Sec 9.
4. **Rate asymmetry: alpha is still the slow mode, and K makes the stiffness WORSE,
   not better.** `I(alpha)` rises in ABSOLUTE terms with K (more categories per
   response carry more discrimination signal, so alpha's own identifiability improves,
   the conjecture is right that direction). But `I(theta)` rises FASTER, so the
   stiffness ratio `I(theta)/I(alpha)` climbs `1.03 -> 2.29` as `K: 2 -> 6`. The
   shared-code flow is MORE stiff in GPCM, not less.
5. **Decoupling-advantage curve (the headline): a robust finite-data-ONLY advantage on
   alpha RANK that VANISHES at the population limit.** With a learned encoder, decoupling
   raises alpha Spearman by `+0.15` at `reps = 1` (positive on all 5 data seeds) and
   ALSO de-noises it (smaller across-seed std), `+0.06` and sign-variable at `reps = 4`,
   `~0` by `reps = 200`, exactly `0` at `reps -> inf`. Decoupling buys alpha
   sample-efficiency, and the advantage is finite-data-only, NOT persistent. This is the
   one place GPCM shows a real decoupling signal where 2PL rung-6 showed none.

The honest bottom line is unchanged from 2PL: **no population-limit dynamics law, no
persistent decoupling fix; the effect is a finite-data / stiffness phenomenon.** GPCM
sharpens two things: the expressivity wall moves to `d < K`, and the rate asymmetry that
makes alpha slow gets WORSE with more categories, which is the dynamical reason a real
finite-data decoupling advantage on alpha rank is now visible where 2PL's was buried in
seed noise.

### 11.1 GPCM setup and notation

Item `j`: discrimination `alpha_j > 0`, step thresholds `beta_{j,1..K-1}`. Person `i`:
ability `theta_i`. Category probabilities (the standard GPCM, k = 0..K-1):

```
psi_{ijk} = sum_{c=1..k} alpha_j (theta_i - beta_{j,c}) = alpha_j (k theta_i - B_{jk}),
            B_{jk} = sum_{c<=k} beta_{j,c},   psi_{ij0} = 0
P(Y_ij = k) = exp(psi_{ijk}) / sum_{m=0..K-1} exp(psi_{ijm}).
```

Truth: `alpha* ~ LogNormal(0, 0.4)`, `beta*` = sorted `N(0,1)` per item (ordered
thresholds), `theta* ~ N(0,1)`. Population objective = expected cross-entropy to the
true category probabilities `p*_{ijk}`; finite-data = `reps` categorical draws per
`(i,j)`, `phat` the empirical category rates. Gauge-fixed exactly as in 2PL (regress
`theta_hat` on `theta_star` for scale `s`, report `s`-corrected alpha log-bias and
Spearman). `K = 4` throughout unless a K-sweep is stated.

### 11.2 Check 1: the gauge is the same, and the raw bias is again pure gauge

`theta -> s theta + t`, `beta -> s beta + t`, `alpha -> alpha/s` leaves
`alpha_j(theta_i - beta_{j,c})` invariant termwise, hence every `psi` and every
category probability. Verified at three `(s,t)`: `max|dP| = 3.4e-15`. Planting a pure
gauge on truth (`s = 1.8`, theta inflated, alpha shrunk `/1.8`):

```
raw logbias = -0.5878  (= -log 1.8, exactly)
recovered s = 0.5556 (= 1/1.8),  gauge-fixed logbias = -0.0000,  Spearman = 1.0000
```

Same conclusion as 2PL Sec 1-2: a uniform alpha shrinkage paired with theta inflation is
a coordinate choice, not bias. Quotient it before any claim.

### 11.3 The shared-code gradient splits into K pathways (sum-checked to 1e-16)

The per-parameter population gradients (derived, autograd-checked to `<= 2e-17`):

```
dL/dtheta_i   = (1/NJ) sum_j alpha_j sum_k R_{ijk} k                  (k-moment residual)
dL/dalpha_j   = (1/NJ) sum_i sum_k R_{ijk} (k theta_i - B_{jk})
dL/dbeta_{j,c}= -(1/NJ) alpha_j sum_i sum_{k>=c} R_{ijk},   R = p_model - p_true.
```

Chaining through `alpha_j = exp(a.e_j)`, `beta_{j,c} = B_c.e_j` gives the GPCM analog of
the 2PL split (***):

```
dL/de_j = G^alpha_j a + sum_{c=1..K-1} G^{beta_c}_j B_c,    (sum-check vs autograd: 9.7e-17)
G^alpha_j = alpha_j (dL/dalpha_j),    G^{beta_c}_j = dL/dbeta_{j,c}.
```

The structural difference from 2PL: there is ONE alpha-pathway but now `K-1` threshold
pathways, so `K` pathway directions `(a, B_1, ..., B_{K-1})` compete on the code. The
rank argument of Sec 2.4 carries over verbatim with `K` vectors instead of two: for
`d >= K` with the `K` directions independent, the stationary equation
`G^alpha a + sum_c G^{beta_c} B_c = 0` forces ALL `K` scalar pulls to vanish separately,
so the optimum zeroes the alpha pull without compromise. For `d < K` the directions are
forced collinear and alpha is squeezed onto a low-rank curve, the GPCM curve-locking.

### 11.4 Check 2: minimal model discriminator (population limit, K=4, J=10, N=800)

Free `theta_i`, shared code rank `d`, 5 init seeds, Adam + L-BFGS, gauge-fixed:

```
 d   excess              gfix-logbias        Spearman
 1   0.017520            -0.0309             0.4424   (rank wall: d<K)
 2   0.004252            -0.0068             0.9273   (rank wall lifting)
 3   0.000416            +0.0001             1.0000   (essentially cleared)
 4   0.000000            -0.0000             1.0000   (d = K: clean)
 6   0.000000            -0.0000             1.0000   (over-capacity: clean)
```

**(a) Expressivity:** for `d >= K` the global minimizer reaches the entropy floor
(`excess 0.000000`) with gauge-fixed log-bias `-0.0000` and Spearman `1.0000`. **(b)
Dynamics:** every seed reaches it. No gap, no capture. Same clean answer as 2PL, with
the wall relocated to `d < K = 4` (in 2PL it was `d < 2`). The `d = 1,2,3` rows are the
GPCM expressivity wall: NLL sits above the floor and alpha rank degrades, the polytomous
version of Sec 3's curve-locking. It is expressivity, not dynamics.

### 11.5 Check 4: GPCM Fisher information (the rate asymmetry, and the K story)

Per single response, the GPCM Fisher informations are (derived from the natural-parameter
score, `Var` over the category distribution `p`):

```
I(theta)  = alpha^2 Var_p(k)                         (variance of the category index)
I(alpha)  = Var_p(k theta - B_k)                     (variance of the natural statistic)
I(beta_c) = alpha^2 P_{>=c}(1 - P_{>=c}).
```

At `K = 2` these reduce EXACTLY to the 2PL forms `Var(k) = p(1-p)`,
`I(theta) = I(beta_1) = alpha^2 p(1-p)` (consistency check passes). The alpha
suppression of Sec 4 survives: `I(alpha)` still carries a `(k theta - B_k)` lever-arm
that is small exactly where the response is most informative about ability. Per-response
means at matched truth, varying K:

```
 K   I(theta)   I(alpha)   I(beta_c)   I(alpha)/I(theta)   I(alpha)/I(beta_c)
 2    0.1865     0.1808     0.1865          0.9696              0.9696
 3    0.4092     0.3442     0.1545          0.8413              2.2281
 4    0.6000     0.4138     0.1296          0.6896              3.1925
 5    0.8319     0.4287     0.1194          0.5153              3.5911
 6    1.0981     0.4792     0.1108          0.4364              4.3230
```

Two readings, and they point opposite ways:

- **Absolute:** `I(alpha)` RISES with K (`0.18 -> 0.48`). More categories give each
  response more discrimination signal, so alpha's OWN identifiability improves and the
  finite-data MLE bias on alpha should be MILDER per response. The conjecture is correct
  in this sense.
- **Relative:** `I(theta)` rises FASTER (`0.19 -> 1.10`, the category index spreads out),
  so `I(theta)/I(alpha)` CLIMBS `1.03 -> 2.29`. The shared-code flow's stiffness, which
  is set by exactly this ratio (Sec 9.2), gets WORSE with K. Alpha is still the slow mode
  and becomes RELATIVELY slower against theta.

So K does not simply make alpha "more identifiable" in the dynamics. It improves alpha's
standalone information but worsens the theta-vs-alpha conditioning that drives any sharing
effect. This is the dynamical reason the finite-data decoupling advantage (11.7) is
visible in GPCM where it was buried in 2PL: the stiffer flow lags more at a fixed budget,
and decoupling relieves exactly that lag on alpha's direction.

### 11.6 Check 3: rung 5 persistence sweep (amortized fixed pool, K=4, J=12, N=800, m=5)

`theta_i = u . ebar_i`, `ebar_i = (1/J) sum_j s_ij e_j`, `s_ij` = person-centered
k-moment coding of `phat` (fixed). `m = 5 >= K = 4`. 4 init x 2 data seeds; `inf` is the
deterministic population objective.

```
                       gfix-logbias   spread    Spearman   excess
shared (m=5)
  reps   1             -0.8186        0.0445     0.7762     1.193540
  reps   4             -0.2375        0.0084     0.8706     0.459917
  reps  20             -0.1313        0.0063     0.9860     0.127940
  reps 200             -0.0887        0.0008     1.0000     0.035118
  reps inf             -0.0032        0.0001     1.0000     0.000117
decoupled (m_alpha=2, total-split-ish)
  reps   1             -0.8291        0.0553     0.7762     1.193540
  reps   4             -0.2375        0.0084     0.8706     0.459917   (== shared)
  reps  20             -0.1313        0.0063     0.9860     0.127940   (== shared)
  reps 200             -0.0887        0.0008     1.0000     0.035118   (== shared)
  reps inf             -0.0034        0.0003     1.0000     0.000125
decoupled (m_alpha=5, matched-each-full-width)
  reps   1             -0.8381        0.0640     0.7762     1.193539
  reps   4             -0.2375        0.0084     0.8706     0.459917   (== shared)
  reps  20             -0.1313        0.0063     0.9860     0.127940   (== shared)
  reps 200             -0.0887        0.0008     1.0000     0.035118   (== shared)
  reps inf             -0.0034        0.0003     1.0000     0.000121
```

DOWN each block: gauge-fixed bias and across-seed spread BOTH decay to ~0 as
`reps -> inf` (`-0.82 -> -0.003`, spread `0.045 -> 0.0001`), the finite-data signature.
ACROSS blocks: shared and both decoupled variants coincide (byte-identical at
`reps >= 4`). **Identical conclusion to 2PL Sec 9: finite-data, not dynamics, and
decoupling is inert.** Reason is the same: the fixed linear pool feeds the SAME `S` to
both arms, so decoupling the item tables cannot change theta's estimation error.
Oracle control (clamp `theta = theta*` on the shared table, K=4): `excess 0.000000`,
gauge `s = 1.0000`, gfix-logbias `-0.0000`, Spearman `1.0000`. The shared ITEM TABLE is
never the problem; everything enters through the amortized-theta pathway. Same as 2PL.

### 11.7 Check 5: the decoupling-advantage curve (LEARNED encoder, rung-6 style)

`theta_i = scale * u . tanh(W ebar_i + c)`, `W, c, u` trainable; same code feeds
`alpha, beta`. Decoupled: alpha reads its OWN code `E_al`; theta-encoder + beta read
`E_th`. `K=4, J=12, N=800, m=5`, 4 init x 2 data seeds, gauge-fixed.

```
 reps |   shared bias    shared rank |   decoup bias    decoup rank | rank ADV (dec-sh)
    1 | -0.0033 +-0.039      0.7072   | +0.0283 +-0.039      0.8820  |   +0.1748
    4 | -0.0530 +-0.012      0.9336   | -0.0518 +-0.011      0.9545  |   +0.0210
   20 | -0.0132 +-0.003      0.9895   | -0.0147 +-0.002      0.9948  |   +0.0052
  200 | -0.0003 +-0.002      0.9991   | -0.0005 +-0.002      0.9965  |   -0.0026
  inf | -0.0000 +-0.000      1.0000   | +0.0000 +-0.000      1.0000  |   +0.0000
```

Two findings:

- **Gauge-fixed BIAS: no decoupling advantage.** The learned encoder fits theta well
  enough that the bilinear contamination of alpha is small for BOTH arms at every reps;
  the bias difference is within seed noise and is zero at the population limit. On the
  bias axis, GPCM agrees with 2PL rung-6: decoupling does not bite.
- **Spearman RANK: a small, positive, finite-data-only advantage.** Decoupling raises
  alpha rank by `+0.175` at `reps = 1`, `+0.021` at `reps = 4`, `+0.005` at `reps = 20`,
  then `0` at `reps >= 200` and EXACTLY `0` at the population limit. **Decoupling buys
  alpha sample-efficiency (ranking), and the advantage is finite-data-ONLY: it vanishes
  as `reps -> inf`.**

Robustness of the headline rank advantage (5 data seeds x 4 init seeds each,
re-run independently of the sweep above):

```
 reps    shared rank        decoup rank        rank-adv (dec-sh)   per-seed advs
   1    0.7241 +- 0.0370   0.8745 +- 0.0279        +0.1503         all 5 positive:
                                                                   +.147 +.203 +.128 +.199 +.075
   4    0.8983 +- 0.0592   0.9587 +- 0.0183        +0.0605         4/5 positive:
                                                                   -.019 +.061 +.003 +.100 +.157
```

The `reps = 1` advantage is large and POSITIVE ON EVERY DATA SEED (`+0.15` mean), not a
fluke. The `reps = 4` advantage is smaller (`+0.06`) and SIGN-VARIABLE across seeds, and
it keeps shrinking to zero by `reps = 200`. A second, robust effect: decoupling also
STABILIZES alpha rank, its across-seed std is consistently smaller than shared's
(`0.028` vs `0.037` at `reps = 1`, `0.018` vs `0.059` at `reps = 4`). So decoupling both
raises and de-noises alpha ranking at small data.

This is the headline curve the PI asked for, and the answer is unambiguous: in GPCM,
decoupling DOES help alpha recovery at finite data (on rank, where it matters for an
ability-tracking model), and the advantage is NOT persistent. It is a sample-efficiency
effect on a stiff flow, exactly the rate / early-stopping mechanism Sec 10.6 named as the
honest residual story, now made VISIBLE by GPCM's worse theta-vs-alpha conditioning
(11.5). The population optimum is unbiased and decoupling-neutral, identical to 2PL.

### 11.8 Where GPCM genuinely DIFFERS from 2PL

1. **The expressivity wall is at `d < K`, not `d < 2`.** Polytomous responses need a
   rank-K code to express `K` independent pathway directions; below that, alpha is
   curve-locked (11.4). For the real `K = 4` model the relevant minimum item-code rank
   is 4, not 2.
2. **The theta-vs-alpha stiffness WORSENS with K** (`I(theta)/I(alpha): 1.03 -> 2.29`,
   11.5). More categories help alpha's standalone information but hurt the conditioning
   of the shared flow. The naive conjecture "K > 2 makes alpha more identifiable, so
   milder bias" is HALF right: milder finite-sample MLE bias per response, but a stiffer
   shared-code flow.
3. **A finite-data decoupling advantage on alpha RANK is now visible** (`+0.175` at
   `reps = 1`, 11.7), where 2PL rung-6 saw none (its `reps = 1` rank was `0.836/0.840`,
   no gap). It is small and it still vanishes at the population limit, so it does not
   change the qualitative verdict, but it is a real GPCM-specific sample-efficiency signal
   and it is consistent with (2): the stiffer flow lags more, and decoupling relieves the
   lag on alpha's direction at finite budget.

Everything else transfers exactly: the gauge artifact, the population-limit unbiasedness,
the rung-5 finite-data-vanishing bias with inert decoupling, the oracle control, and the
variational boundary (this is a pure point estimate, no `q`).

### 11.9 What is proved, argued, measured (GPCM)

- **Proved (sum-checked + discriminator):** the GPCM shared-code gradient splits into
  `K` pathways; for `d >= K` the population optimum zeroes the alpha pull separately,
  is gauge-fixed-unbiased (`-0.0000`, rank `1.0000`), and gradient flow reaches it on
  all seeds. Gauge invariance verified to `3e-15`; gradient decomposition to `1e-16`.
- **Measured:** rung-5 gauge-fixed bias `-0.82 -> -0.003` with spread `0.045 -> 0.0001`
  as `reps -> inf`, decoupling byte-identical at `reps >= 4`; rung-6 learned-encoder
  decoupling RANK advantage `+0.150 (reps 1, all 5 seeds +) -> +0.060 (reps 4) -> 0
  (reps >= 200)`, plus a smaller decoupled across-seed std (rank stabilization); oracle
  control exact (`s = 1`, bias `-0.0000`, rank `1.0`).
- **Argued (Fisher):** alpha remains the low-information slow mode; K raises `I(alpha)`
  absolutely but raises `I(theta)` faster, so the shared-flow stiffness `I(theta)/I(alpha)`
  WORSENS with K, the dynamical reason the finite-data decoupling advantage surfaces in
  GPCM.
- **Verdict:** 2PL conclusions transfer. No population-limit dynamics law, no persistent
  decoupling fix. GPCM adds a sharper rank wall (`d < K`), worse stiffness with K, and a
  small finite-data-only decoupling advantage on alpha rank that vanishes at the
  population limit.

---

## 12. Rung 7: training-time / convergence-rate test (does the transient bite?)

Built and run 2026-06-16. Every rung so far measured the OPTIMUM (or close to it) and
found the FREE-TABLE INVARIANT: as long as the per-item code can hit `p = p*`, every
pathway pull is linear in the residual `r = p - p*`, all pulls vanish together at `r = 0`,
so the optimum is unbiased and decoupling-neutral. At the optimum decoupling is provably
inert. But the real model trains a FINITE number of epochs and never converges: it lives
in the TRANSIENT. Rung 7 asks the one question the optimum cannot answer: on the
TRAINING-TIME axis, does decoupling help alpha as a convergence-RATE effect?

Reuses the rung-6 learned-encoder amortized GPCM setup (`theta_i = scale * u . tanh(W
ebar_i + c)`, shared item code feeds alpha/beta/theta; decoupled gives alpha its own
full-width code `E_al`, matched capacity). Two design rules make it clean: (i) the
POPULATION objective (reps -> inf, true category probabilities, NO sampling noise), which
removes the finite-data errors-in-variables confound entirely, so any effect here is PURE
optimization dynamics; (ii) do NOT polish to the optimum -- train with a fixed-lr optimizer
and CHECKPOINT across training time, gauge-fixing at each checkpoint and reporting
gauge-fixed alpha log-bias AND Spearman rank vs step.

### 12.0 Verdict

**YES, the transient bites, and it is a clean population-limit convergence-RATE advantage
for DECOUPLING -- on RANK, exposed by plain GD, that closes at convergence.** This is the
WIN scenario: the "alpha learns faster when decoupled" curve, on the correct (training-time)
axis, with no finite-data confound and no variational approximation. Precisely:

1. **Q1 rate advantage (YES, on rank).** Steps for alpha Spearman to first reach `0.95`,
   plain GD, `K = 4`, population limit, 3 truth x 4 init: **shared `1025 +- 1006`,
   decoupled `457 +- 123`.** Decoupled reaches good alpha ranking in under HALF the steps
   and roughly `8x` more reliably (the seed spread collapses from `+-1006` to `+-123`). The
   per-checkpoint rank gap (decoupled - shared) is POSITIVE throughout the transient,
   `+0.07..+0.21`, largest around steps `95..673` where decoupled sits at rank `0.93..0.99`
   while shared drags at `0.82..0.88`.
2. **Q2 vanishes at convergence (YES).** Both reach rank `1.0000` and gauge-fixed bias
   `~ -0.026` by step `8000`; the rank gap decays `+0.21 -> +0.001` and the bias gap stays
   `~ 0` the whole way. A transient advantage that closes at convergence is the signature
   of a pure RATE effect, exactly consistent with the free-table invariant: both arms reach
   the SAME clean optimum, decoupled just gets there sooner.
3. **Q3 transient shape.** No clean shared peak-then-decay; instead a persistent shared LAG
   -- shared alpha rank climbs slowly while the high-curvature theta/beta directions
   dominate the single shared code, and the alpha ordering is the last thing to resolve.
   Decoupled alpha, reading its own code, resolves its ordering early and saturates. (The
   one transient shared rank DIP at step ~100 is a single truth-seed's gauge `s` swinging
   through its near-singular start, `s: 0.01 -> 0.23`, not alpha capture; it is gone after
   averaging the gauge-invariant rank.)
4. **The advantage is on RANK, not BIAS.** Steps for gauge-fixed `|logbias| < 0.10`:
   shared `2252 +- 615`, decoupled `2301 +- 833` -- statistically identical, and the bias
   gap is `~0` at every checkpoint. The scale of alpha (bias) converges at the same rate
   for both; what decoupling accelerates is alpha's ORDERING across items, which is the
   metric the ability-tracking model is actually judged on (Spearman).

### 12.1 The decisive control: it is OWNERSHIP, not a cold-start artifact

The decoupled arm carries a second item table `E_al`. A skeptic's first objection: maybe
decoupled is faster only because `E_al` gets an independent random draw, or because the
shared arm's single table is merely a worse init. The INIT CONTROL refutes both. Warm-start
`E_al = E_th`'s init (the two tables start byte-identical) and re-run GD:

```
steps -> (rank > 0.95), GD, K=4, pop limit, 3 truth x 4 init
                     shared            decoupled
  cold E_al        1025 +- 1006        457 +- 123
  warm E_al=E_th   1025 +- 1006        491 +- 174     <- control
```

Decoupled keeps its advantage from the IDENTICAL init (`491` vs `1025`). The only remaining
difference between the arms is WHICH gradients pull each table: in shared, the one table
`E_th` receives the SUM of the alpha, beta and theta pulls; in decoupled, `E_al` receives
ONLY the alpha pull. So the advantage is structural -- alpha's code, when it owns it,
converges at alpha's own (uncontested) rate; when alpha shares the code, that code is
shaped first for the high-curvature theta/beta directions and alpha's ordering resolves
later. (At the identical-init start the warm control even shows shared briefly AHEAD,
steps `44..209`, gap `-0.02..-0.09`: the shared table moves faster early under the larger
combined gradient. The lead reverses by step `308` and decoupled is well ahead by the time
either reaches good rank. The transient detail matters; the steps-to-rank conclusion does
not change.)

### 12.2 Q4 Rate vs Fisher: it is a CONDITIONING effect, and the orthogonality caveat is resolved

The mechanism is the Fisher stiffness of Secs 4 and 9.2, now read on the TIME axis. Near
the optimum gradient flow linearizes to `d(delta)/dt = -H delta`; each mode decays
`exp(-lambda t)` with `lambda` the curvature of that direction, set by its Fisher
information. The decisive evidence that this is a CURVATURE (Hessian-conditioning) effect,
not a gradient-magnitude effect, is the optimizer comparison:

```
steps -> (rank > 0.95), K=4, pop limit, 3 truth x 4 init
                  shared            decoupled       sh/dc ratio
  plain GD       1025 +- 1006        457 +- 123        2.2x
  Adam            119 +-   68         75 +-  22        1.6x
```

Plain GD does NOT precondition, so it pays the full `cond ~ I(theta)/I(alpha)` stiffness and
the decoupling advantage is large (`2.2x`, with an `8x` reliability gain). Adam's
per-parameter preconditioner cancels most of the curvature asymmetry, so both arms converge
fast and the advantage compresses to `1.6x` (and closes by step ~320 vs ~3000 for GD).
**The advantage is largest exactly when the optimizer fails to precondition the
stiffness** -- the defining signature of a conditioning effect.

This also RESOLVES the Phase-2 skeptic caveat (`cos(g_theta, g_alpha) ~ 0`, gradients
orthogonal not conflicting). On the shared code we measure `cos(g_alpha-pathway,
g_theta-pathway) ~ 0.05..0.20` (small, near-orthogonal), consistent with Phase 2. But
orthogonal gradients do NOT predict a null rate effect: the lag comes from the EIGENVALUE
SPREAD of the shared-code Hessian block (a curvature property), not from the gradient inner
product. Two pathways can have orthogonal gradients and still impose mutual slowing through
shared curvature. So the orthogonality finding and the rate advantage are CONSISTENT, and
the rung-7 result is precisely the conditioning-driven lag that gradient orthogonality alone
could not rule out.

Fisher numbers at truth (per response, then accumulated): `I(theta) = 0.73`,
`I(alpha) = 0.33`, ratio `I(theta)/I(alpha) = 2.18` at `K = 4`. The accumulated per-item
counts differ (`alpha_j` informed by `N = 800` persons, `theta_i` by `J = 12` items), which
is why each alpha PARAMETER is well-determined and the OPTIMUM is clean; the transient lag
is about how fast the shared CODE's directions get allocated, which the stiff combined pull
resolves for theta/beta first and alpha last.

### 12.3 Relation to the empirical model, the prior rungs, and the variational boundary

Rung 7 is the first rung to produce a decoupling advantage at the POPULATION limit. It does
not contradict the free-table invariant: the invariant is a statement about the OPTIMUM
(both arms reach the same clean `p = p*`), and rung 7 confirms it (Q2, the gap closes). The
advantage lives entirely in the TRANSIENT, on the training-time axis the optimum cannot see.
This is the same rate / early-stopping mechanism named as the honest residual story in Secs
10.6 and 11.7, now isolated and measured directly:

- It matches the GPCM finite-data RANK advantage (Sec 11.7, `+0.175` at `reps = 1`): both
  are the SAME stiffness lag on alpha's direction, one surfaced by finite data, the other by
  finite TRAINING. Rung 7 shows the time-axis version survives at `reps = inf`, so it is
  pure optimization dynamics, not the errors-in-variables bias of rung 5.
- It matches the empirical Phase-1 (alpha peaks/lags) and Phase-2 (theta gradient grows,
  dominates the shared code) observations: the shared code is dragged fast along the
  high-curvature theta/beta directions and alpha's ordering lags at a fixed step budget.
- The empirical decoupling fix is therefore a CONVERGENCE-RATE / early-stopping effect on a
  stiff shared-code flow, not an endpoint identifiability law. At infinite training shared
  and decoupled agree; at the realistic finite budget the stiffer SHARED flow has not yet
  resolved alpha's ordering, and decoupling -- giving alpha an uncontested code -- buys the
  rate. This is the analytically clean, population-limit, variational-free statement of the
  empirical benefit.

Variational boundary (Sec 7) untouched: rung 7 is a pure point estimate, no `q`, no ELBO.
The advantage is gradient-descent dynamics on a likelihood.

### 12.4 What is proved, argued, measured (rung 7)

- **Measured (population limit, gauge-fixed, 3 truth x 4 init):** under plain GD, decoupled
  reaches alpha rank `> 0.95` in `457 +- 123` steps vs shared `1025 +- 1006` (`2.2x` faster,
  `8x` more reliable); the per-step rank gap is positive through the transient (`+0.07..
  +0.21`) and decays to `+0.001` at convergence (both reach rank `1.0`, bias `~ -0.026`).
  The advantage is on RANK; the bias-convergence rate is identical across arms.
- **Controlled:** warm-starting the decoupled alpha table to the shared init (byte-identical
  start) preserves the advantage (`491` vs `1025`), so it is OWNERSHIP of an uncontested
  code, not a cold-start or lucky-draw artifact.
- **Argued (conditioning, not conflict):** the advantage is largest under plain GD and
  compresses under Adam (`2.2x -> 1.6x`), the signature of a Hessian-conditioning (stiffness)
  effect that preconditioning cancels. Shared-code pathway gradients are near-orthogonal
  (`cos ~ 0.05..0.20`), which is consistent with -- not contradicted by -- a rate lag driven
  by the eigenvalue spread of the shared block. The Phase-2 orthogonality caveat is resolved.
- **Predicted, not yet measured (K-dependence):** since the shared-flow stiffness
  `I(theta)/I(alpha)` WORSENS with K (Sec 11.5, `1.03 -> 2.29` for `K: 2 -> 6`), the
  steps-to-rank decoupling gap should WIDEN with K. The direct GD K-sweep (`K = 2` vs `K = 4`)
  was set up but not completed this session; the analytic prediction is unambiguous and the
  `K = 4` result here already sits at the stiffer end. Worth confirming if a clean
  K-dependence curve is wanted for the paper.
- **Verdict:** a clean POPULATION-LIMIT convergence-RATE decoupling advantage on alpha
  ranking, exposed by un-preconditioned GD and closing at convergence. The analytical toy
  ladder bottoms out HERE: the empirical decoupling benefit is a rate / early-stopping effect
  on a stiff shared-code flow (Fisher conditioning `I(theta)/I(alpha)`), NOT an endpoint
  bias and NOT a finite-data artifact. Decoupling does not change WHERE training converges;
  it changes HOW FAST alpha's ordering gets there, which is what matters under a finite epoch
  budget.
