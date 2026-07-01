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
steps and observations grow. A separate, representation effect follows from the multiplicative scale gauge: alpha
multiplies theta, so prediction loss pins only their product, and when discrimination and ability read from one shared
embedding width their split is under-determined. Decoupling (a separate item key for the multiplicative parameter)
clears it; the dynamic head is a second, targeted fix for the low-Fisher parameter.

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
  on alpha (noisy amortized theta_hat attenuates the recovered discrimination alpha).
- Joint: G(T, N) <~ exp(-T/kappa) + O(kappa/N). Both terms carry kappa; both -> 0 under min(T, N) growing.
  THIS licenses the honest headline: the penalty is real at finite budget and vanishes asymptotically.

**C3, SCALE GAUGE / REPRESENTATION (FINITE).** The bilinear term alpha_i theta carries a multiplicative scale gauge
alpha -> alpha/s, theta -> s theta (one parameter, fixed by one normalization), so prediction loss pins only the
product alpha theta, never the split. Magnitude is unidentified, rank survives (hence Spearman rho for the
multiplicative parameter). This is not a structural failure. It becomes a REPRESENTATION trade-off when discrimination
and ability read from one shared embedding of width W. Widening W reallocates that width between the two coordinates,
so the discrimination rank rises while the ability rank falls, and no single width serves both. Giving the
multiplicative parameter its own item key (decoupling) removes the shared-width constraint and clears the trade-off.
This fires for any slope on ability, GPCM alpha or NRM a_k, independent of Fisher curvature; it is the mechanism the
NRM control (C4) isolates from the Fisher-rate effect. The additive intercept c_k never multiplies theta, so it carries
no scale gauge.

**C4, NRM CONTROL (consistency).** NRM's slope a_k is the genuine discrimination analogue (GPCM is a
constrained NRM with a_k = alpha k; Thissen & Steinberg 1986). Its leverage is theta itself, centered at
theta = 0 (away from the response mode), so I(a_k)/I(c_k) ~ 0.90, NOT low, unlike GPCM's alpha whose
(theta - beta)^2 centers on the mode. NRM separates a REPRESENTATION effect (decoupling fires regardless
of Fisher) from a FISHER effect (the dynamic head helps only a low-information coordinate). Report 0.90 as
MEASURED; "~1" (E[theta^2] = 1) is intuition only, since p(1-p) co-varies with theta.

## The key refinement: two mechanisms, two fixes, keep them distinct
Two mechanisms govern how well a readout parameter recovers, and each has its own fix.
- REPRESENTATION (C3, FINITE): sharing one embedding width between a slope and ability forces a trade-off. The fix is
  DECOUPLING (a separate item key for the multiplicative parameter). Fires for any slope on ability, independent of Fisher.
- FISHER INFORMATION (C1/C2, FINITE): low Fisher curvature makes a coordinate the slow direction, recovering last and
  least reliably. The targeted fix is the DYNAMIC (state-conditioned) head, which helps ONLY a low-Fisher coordinate
  and hurts one whose Fisher is not low.
GPCM discrimination has BOTH problems at once (low Fisher AND a slope sharing ability's representation); the NRM slope
a_k has only the second. The NRM control (C4) is what dissociates them. The paper must not let the decoupling result
borrow the dynamic head's credit, or vice versa.

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
- Keep the two fixes separate in the experiments: decoupling (own item key) tests the REPRESENTATION mechanism, the
  dynamic (state-conditioned) head tests the FISHER mechanism. The NRM control adjudicates both; do not let one fix's
  evidence stand in for the other's.
- NRM confound (C4): NRM has more parameters and its own identification constraints (sum-to-zero or a
  reference category). Hold K, N, Q, seeds fixed; score slope-VECTOR rank recovery vs truth; check a_k recovery
  is information-geometry-driven, not parameter-count/identification-driven. ("Decoupling a_k alone breaks c_k
  to 0.392" shows NRM's own a_k / c_k entanglement, control it.)
- E8 (adaptive testing, downstream / open): frame discrimination as the DEGRADED channel within a JOINT
  (discrimination, difficulty) selection rule (item info peaks at theta ~ difficulty, so difficulty matters too),
  not the sole determinant. Score with recovered params against the oracle at fixed test length.
- Add the oracle-clamp control (theta = theta*) to bridge the co-learned-theta gap.
- To claim the exp(-T/kappa) rate COEFFICIENT (not just form and sign), add a knob that varies I(theta) at fixed K.

## IRT-centrism guard
The psychometric pieces (the Fisher/rate theory and the scale-gauge/representation analysis) must read as PROPERTIES
OF THE KT TRACER'S READOUT, never as new psychometrics. Keep the scale-gauge argument at the readout level (the
product alpha theta and which embedding feeds each readout), a property of the tracer, not a psychometric claim.

## Source derivations (agent-committed)
docs/learning_dynamics_theory_support.md (P1-P12), docs/paper2_leverage_proposition.md (the conditional
ordering and a_star), docs/learning_dynamics_toy.md (the 2PL/GPCM rung ladder, rung 7 = the training-time
rate result), and the ml-math agent memory combined-paper-two-axis-spine.md.
