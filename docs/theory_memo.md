# Theory memo (Phase 1, reconciled): "Not All Parameters Learn Alike"

Status: reconciled from the Phase 1 math derivation (ml-math-researcher) and the psychometric
validation (psychometric-researcher), 2026-07-01. AWAITING USER APPROVAL before Phase 2 and any
paper writing. No hard math errors were found; the validation confirmed the spine and added
precision. Home is representation learning in a KT sequence model; IRT supplies readable
coordinates, not the subject.

## The picture
A prediction-trained KT encoder with an IRT readout relaxes each readout coordinate at a rate set by
that coordinate's Fisher curvature, so the flat direction (discrimination) is learned last. That
penalty is a finite-budget transient plus a finite-sample variance term; both vanish as gradient
steps and observations grow. A separate, structural failure occurs if the ability readout is allowed
to see the current item, which enlarges the gauge group and destroys the ability-difficulty location
split at every budget.

## Setup
Causal state h_t = enc(y_{1..t-1}). Readout exposes per-item coordinates xi_i and ability
theta_t = phi(h_t [, e_i]). Categorical response, logits eta = R(xi_i; theta_t), loss L = E[CE],
vector residual r_k = p_k - y_k, covariance Sigma = diag(p) - p p^T. Decoder-generic; GPCM
eta_k = alpha_i(k theta - B_{i,k}) and NRM eta_k = a_{i,k} theta + c_{i,k} are two instances.

Gradient framing (validation precision): write the gradients honestly. The scalar r = p - y and
d_theta L = r alpha etc. are the BINARY (2PL) case. For GPCM, d_theta L is alpha times the
expected-score residual (E[X] - x_obs) and d_beta L a cumulative residual, generalized by the vector
r_k. Carry this framing (binary setup + r_k vector) into the paper, not a compressed scalar-r as the
literal GPCM gradient. The STRUCTURE the Fisher argument uses (theta and beta scale with alpha; alpha
carries the (theta - beta) leverage; sign flip d_beta = -d_theta) is correct.

## Claims (each labeled FINITE or STRUCTURAL)

**C1, RATE (FINITE).** Near a gauge-fixed optimum, gradient flow d(delta)/dt = -H delta with H = Fisher
F at a well-specified zero-residual optimum. Mode m decays exp(-lambda_m t); a coordinate recovers in
time ~ 1/I(xi). For a generic readout coordinate I(xi) = (d eta / d xi)^T Sigma (d eta / d xi). The
person-item separation lever (theta - beta) gives I(alpha) = E[w (theta - beta)^2], w = p(1-p),
suppressed where response mass concentrates (theta ~ beta); the multiplier lever gives
I(theta) = I(beta) = alpha^2 w, not suppressed. Discrimination is the slow direction. Vanishes as T grows.

**C2, STATISTICAL (FINITE).** Budget = (T steps, N observations). kappa = lambda_max/lambda_min
~ I(theta)/I(alpha) is the readout conditioning number.
- Optimization axis (N = infinity): with a stability-bounded shared step size, the slow-mode residual
  after T steps is ~ exp(-T/kappa); the recovery GAP ~ exp(-T/kappa) -> 0 (geometric in steps).
- Statistical axis (T = infinity): a free per-item table (rank >= K) reaches the unbiased endpoint; the
  residual reliability gap between alpha and theta is O(kappa/N), plus an O(1/N) errors-in-variables bias
  on alpha (noisy amortized theta_hat attenuates the bilinear slope).
- Joint: G(T, N) <~ exp(-T/kappa) + O(kappa/N). Both terms carry kappa; both -> 0 under min(T, N) growing.
  THIS licenses the honest headline: the penalty is real at finite budget and vanishes asymptotically.

**C3, COUPLING / IDENTIFIABILITY (STRUCTURAL).** The logit alpha_i(theta - beta_i) carries the global
affine gauge (theta, beta) -> (s theta + c, s beta + c), alpha -> alpha/s (2 parameters, fixed by one
normalization). DECOUPLED theta_t = phi(h_t) (excludes item i): one theta_t faces every item at step t,
cannot re-center per item, so beta_i is identified. COUPLED theta_{t,i} = phi(h_t, e_i): per item,
(theta_{t,i}, beta_i) -> (theta_{t,i} + delta_i, beta_i + delta_i) is a likelihood symmetry, and phi
(already a function of e_i) absorbs delta_i; the gauge orbit enlarges from 2 to 2 + J (one location shift
per item). Those J directions are exact flat directions of the POPULATION loss, so no estimator at any N
identifies beta_i within its orbit. This is MA-GPCM's threshold collapse read correctly as identifiability,
not rate; it persists at all budgets. (Validation caveat: regularization or finite width turns exact
flatness into near-flatness; the does-not-vanish-with-data property still holds.) The general statement is
a measurability condition: theta_t must be measurable with respect to a sigma-field EXCLUDING e_i. DKVMN's
separated pathway (theta from the memory-read alone) and the LSTM's narrow item-value embedding plus a
separate wide item-key are two enforcements; neither architecture's trick is claimed encoder-generic.

**C4, NRM CONTROL (consistency).** NRM's slope a_k is the genuine discrimination analogue (GPCM is a
constrained NRM with a_k = alpha k; Thissen & Steinberg 1986). Its leverage is theta itself, centered at
theta = 0 (away from the response mode), so I(a_k)/I(c_k) ~ 0.90, NOT low, unlike GPCM's alpha whose
(theta - beta)^2 centers on the mode. NRM separates a REPRESENTATION effect (decoupling fires regardless
of Fisher) from a FISHER effect (the dynamic head helps only a low-information coordinate). Report 0.90 as
MEASURED; "~1" (E[theta^2] = 1) is intuition only, since p(1-p) co-varies with theta.

## The key refinement (both passes flagged it): two "decouplings," keep them distinct
The plan section 5 runs together two mechanisms on two different readouts:
- (a) ABILITY-readout coupling -> the location-gauge IDENTIFIABILITY failure (C3, STRUCTURAL).
- (b) any readout SHARING EMBEDDING WIDTH -> the capacity/representation trade-off (rate/conditioning, FINITE).
Both are called "decoupling" but sit on different axes with different asymptotics. The NRM control (C4)
adjudicates only (b); it says NOTHING about (a). Do not let NRM evidence support the location-gauge claim.
E2 is the vehicle for (a); keep the evidence separate. The paper must not let one decoupling borrow the
other's credit.

## Honest caveats and reviewer attacks (keep on the record)
1. Per-eigenmode, not per-parameter. F has off-diagonal F_{alpha,beta} = E[w(theta-beta)(-alpha)], so the
   law is per-eigenvector; the slow eigenvector is discrimination-dominated only for alpha >= a_star ~ 1.
   Below a_star the ordering INVERTS (I(alpha) > I(beta)). State the inversion.
2. Well-specification. Gauss-Newton = Fisher only at a zero-residual well-specified optimum. On misspecified
   real data H != F and a residual gap need not vanish; "vanishes asymptotically" is a WITHIN-MODEL statement.
3. kappa not identified from K. The exp(-T/kappa) shape is a form-and-sign claim, not a coefficient: over the
   K-sweep Spearman(K, kappa) = 1 (collinear), the contraction fits R^2 ~ 0.34 and loses to an
   ordinal-information ceiling ~ 1/K. Isolating kappa needs a knob that varies I(theta) at fixed K.
4. Co-learned theta (the central gap). theta_t is amortized and co-learned, so its effective rate is Fisher
   attenuated by pooling variance; the load-bearing object is the Hessian restricted to the SHARED CODE block,
   and the slow direction is a direction in code space, not the bare alpha axis. Needs the oracle-clamp control.

## Assumptions for leverage-sets-rate under a shared encoder
(a) Local, near a gauge-fixed, identifiable, well-specified optimum, small residual. (b) Encoder timescale
separation: theta_t settles fast relative to readout relaxation, or is oracle-clamped (the oracle-clamp
control, alpha exact under theta = theta*, is the check). (c) Rank adequacy: shared code rank >= K, else an
expressivity wall (endpoint), not rate, controls recovery. (d) Eigenmode statement with alpha >= a_star.
(e) Single step size for the raw kappa penalty (Adam compresses it by preconditioning parameters, not code
directions). (f) softmax-CE-family loss (plain, class-weighted, WOL): a detached positive per-sample weight
enters as a scale and preserves kappa's structure; breaks for EMD, margin, or regression-on-E[k] losses.

## Implications for Phase 2/3 experiments
- E2 must isolate mechanism (a) (ability-item coupling -> identifiability) SEPARATELY from mechanism (b)
  (width sharing -> rate). Do not let NRM stand in for (a).
- NRM confound (C4): NRM has more parameters and its own identification constraints (sum-to-zero or a
  reference category). Hold K, N, Q, seeds fixed; score slope-VECTOR rank recovery vs truth; check a_k recovery
  is information-geometry-driven, not parameter-count/identification-driven. ("Decoupling a_k alone breaks c_k
  to 0.392" shows NRM's own internal readout coupling, control it.)
- E8 (adaptive testing): frame alpha as the DEGRADED channel within a JOINT (alpha, beta) selection rule (item
  info peaks at theta ~ beta, so location matters too), not the sole determinant. Score with recovered params
  against the oracle at fixed test length.
- Add the oracle-clamp control (theta = theta*) to bridge the co-learned-theta gap.
- To claim the exp(-T/kappa) rate COEFFICIENT (not just form and sign), add a knob that varies I(theta) at fixed K.

## IRT-centrism guard
The three most psychometric pieces (the Fisher/rate theory, the gauge/identifiability formalization, the CAT
downstream) must read as PROPERTIES OF THE KT TRACER'S READOUT and A KT DEPLOYMENT DECISION OUR READOUTS
CHANGE, never as new psychometrics or a CAT advance. Keep the gauge section at the readout level
(phi(h_t, e_i) versus phi(h_t)).

## Source derivations (agent-committed)
docs/learning_dynamics_theory_support.md (P1-P12), docs/paper2_leverage_proposition.md (the conditional
ordering and a_star), docs/learning_dynamics_toy.md (the 2PL/GPCM rung ladder, rung 7 = the training-time
rate result), and the ml-math agent memory combined-paper-two-axis-spine.md.
