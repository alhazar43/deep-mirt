# Learning dynamics of discrimination recovery in amortized neural IRT

The consolidated study. It folds the motivation and prior-art sweep, the
empirical Pareto/trajectory/gradient/K-sweep/N-sweep spine, the RQ1/RQ2
Fisher-asymmetry and convergence-rate results, the RQ3 state-conditioned
discrimination diagnostics, and the training-loss robustness checks into one
line. The standalone derivations live in the theory appendix,
`docs/learning_dynamics_toy.md` (pointer at the end). Opened 2026-06-16,
converged 2026-06-17. Preserve every number.

---

## 1. Intro and motivation

### The claim in one sentence

Amortized neural IRT couples its psychometric parameters through the shared
learned representation, and this coupling silently degrades discrimination
(alpha) recovery. The degradation is a learning-dynamics phenomenon, not a
statistical one, and it is a measurement-validity condition the field has not
examined.

> DOWNGRADE BANNER (2026-06-16, Phases 0 to 3b run). The strong version of the
> sentence above is REFUTED. The gate confirmed a real theta-vs-alpha trade-off
> (capacity killed) and the empirical decoupling fix is robust and
> architecture-independent. But the toy ladder refuted the population-limit
> learning-DYNAMICS-LAW framing. In the tractable toys the alpha bias is
> FINITE-DATA errors-in-variables (it vanishes as reps to infinity) and is NOT
> decoupling-fixable. Low Fisher sets the RATE, not the ENDPOINT. The empirical
> decoupling fix needs a learned encoder the toys omit, and a clean
> population-limit dynamics law does not exist in tractable form. Honest
> framing, a finite-data plus learned-representation recovery effect, not a
> fundamental dynamics law. The sentence is retained as the original hypothesis,
> downgraded. The decoupled deep-irt ENGINE decision is unaffected, it stands on
> its empirical merits. The final converged contribution is in Section 2.7.

> ADDENDUM (2026-06-21, positive-map and E7/E7a controls). The later
> positive-map study further narrows the mechanism. The exp-specific story is
> not supported: across K=2,4,8, exp does not beat the best smooth non-exp map
> by a meaningful margin under matched effective-alpha initialization and LR
> controls. A direct-alpha optimizer control with true theta and beta frozen
> rejects the scalar alpha-space preconditioner-only explanation: with a wide LR
> grid, direct alpha-space update rules mostly converge to the same recovery
> band. The live wording is therefore smooth positive-map stability under neural
> representation learning, not exp optimality and not scalar alpha-space
> preconditioning alone. The post-E7 neural isolation supports this: clipped
> raw/ReLU plus gradient clipping does not rescue the learned-model gap across
> K=2,4,8, while freezing the backbone or encoder breaks recovery. See
> `docs/learning_dynamics_progress.md`, E5 to E7a, and outputs
> `alpha_map_*`, `direct_alpha_geometry_*`, and `neural_map_isolation_*`.

### The phenomenon (precise)

Neural and amortized IRT reads ability (theta), discrimination (alpha), and
difficulty (beta) off a SHARED learned representation (a shared item embedding
and a shared encoder state). Theta and alpha then have OPPOSING capacity needs.

- Theta wants the item channel NARROW. It is a low-rank latent. Widen the item
  representation and the encoder routes item identity into the ability state and
  overfits theta, and it overfits MORE with more training.
- Alpha wants the item channel WIDE. It is the lowest-information,
  hardest-to-recover IRT parameter. A narrow shared key STARVES alpha recovery,
  which comes out biased low AND high-variance across seeds.

One shared representation cannot be both narrow and wide, so sharing forces a
compromise that degrades alpha. Beta is content narrow (it recovers well from a
thin item code), so it sits with theta.

The fix is to decouple. Give alpha its own separate, wider item-embedding table
that feeds ONLY the discrimination head, keep the ability encoder narrow. This
lifts alpha recovery to a strong baseline's level and collapses its across-seed
variance, with no cost to theta.

### First evidence (synthetic, the seed of the study)

- 4-way recovery (GPCM, 3 seeds, 150ep), the decoupled model ties the
  carefully-engineered ma-irt baseline on alpha (0.929 vs 0.935), keeps theta
  (static 0.967, drift 0.729), at fewer params and one encoder pass.
- ARCHITECTURE-INDEPENDENT (swap bench, LSTM / Transformer / DKVMN, 3 seeds),
  decoupling keeps theta and lifts alpha to ma-irt's level on every backbone.
  alpha cheap to decoupled, LSTM 0.654 to 0.929, Transformer 0.650 to 0.925,
  DKVMN 0.708 to 0.916.
- VARIANCE COLLAPSE, cheap-alpha is high-variance on every backbone (+-0.11 to
  +-0.19 across seeds), decoupling collapses it to +-0.02 to +-0.035 as well as
  lifting the mean. The variance signal is cleaner than the mean lift.
- One dynamics signature already in hand, the bare encoder's responsive theta
  overfits the static level WITH MORE TRAINING (0.97 at 150ep to 0.68 at 500ep),
  while the regularised baseline is stable. Overfitting that grows with training
  is a trajectory property, not a capacity ceiling.

Sources, deep_irt/bench/outputs/alpha_fix_table.md,
deep_irt/bench/outputs/swap_table.md. Caveats, all synthetic, dense, K=4, Q=60,
LogNormal(0,0.3) alpha. The decoupled theta still softens under long training
(it lacks the baseline's LayerNorm / q-residual regularisation).

### The honest fork, capacity or dynamics

The empirical result alone is an ablation ("alpha needs more parameters"). It
becomes a contribution only if the cause is the COUPLING (a learning-dynamics
property of the shared parameterisation) and not mere CAPACITY (alpha simply
wants more degrees of freedom, which is trivial). Two different claims that must
be separated.

- CAPACITY (static, trivial), a narrow shared embedding lacks the expressivity
  to encode both theta-relevant and alpha-relevant item variation. Independent
  of optimisation. A bigger embedding fixes it.
- DYNAMICS (optimisation, interesting), even with enough total capacity, the
  gradient flow drives the shared embedding toward theta's optimum and alpha is
  left under-fit. The fix is SEPARATION, not size.

THE CRUX EXPERIMENT, compare a shared embedding of width (w_theta + w_alpha)
against a decoupled model with the SAME TOTAL width split across the two tables.
If decoupling wins at matched total capacity, the effect is the SEPARATION (the
interesting claim). If the matched-total shared model catches up, it is just
capacity. This is the Phase-0 gate in Section 2.

### Prior-art verdict (deep-research, 2026-06-16; 99 agents, 17 primary sources, 22/25 claims adversarially confirmed)

All three layers came back GENUINELY OPEN, and no single work or combination
covers the conjunction.

- MECHANISM (per-IRT-parameter embeddings), open. Nearest is Tsutsumi,
  Kinoshita and Ueno (EDM 2021, "Deep-IRT with independent student and item
  networks"), but it splits STUDENT vs ITEM, not theta vs alpha, and is
  Rasch-style with discrimination HARDCODED at 3.0, so it recovers no alpha at
  all. The recsys "multi-embedding" work was explicitly refuted as a true analog
  (it assigns no embedding to an interpretable parameter).
- PHENOMENON (alpha-theta capacity coupling degrading recovery,
  architecture-independent), open. No neural-IRT paper reports or fixes a
  discrimination-recovery degradation from representation sharing. VIBO (EDM
  2020) measures recovery ONLY as ability correlation and reports no
  discrimination metric. Architecture-independence is addressed by nobody.
- FRAMING (learning-dynamics / measurement-validity gap between classical IRT,
  DKT, neural IRT), open. No work contrasts the three settings and locates a
  coupling or validity problem unique to the third.

Nearest neighbors to cite and distinguish.

- Tsutsumi et al. EDM 2021, student/item split, Rasch, no alpha.
- VIBO, EDM 2020 (arXiv 2002.00276), amortized variational IRT, recovery is
  ability-only.
- Urban and Bauer 2021 (arXiv 2109.09500), amortized IFA, the sharpest neighbor.
  It DOES recover discrimination (loadings) via an amortized autoencoder, but no
  coupling analysis, no per-parameter decoupling.
- JE-IRT (arXiv 2509.22888, Sept 2025), the structural antithesis, everything
  read off one maximally-shared geometric space, no alpha scalar.
- PCGrad / Standley / embedding-collapse, the MTL-interference machinery, but
  prediction-only. PCGrad's fix is gradient surgery that leaves the shared
  representation intact, the opposite of decoupling.

White space, the CONJUNCTION of the three layers.

### The variational-bias boundary (the sharpest reviewer risk)

GVEM / IW-GVEM variational-IRT works document discrimination-estimate BIAS under
variational inference. Their bias is a property of the POSTERIOR APPROXIMATION
(an inference-quality / statistical effect). Ours is a property of GRADIENT-FLOW
behavior of a shared POINT-ESTIMATE representation in an amortised model. The
toy (Section 2.4 and the appendix) draws the line explicitly, we exhibit the
alpha behavior in a NON-variational, point-estimate model purely from gradient
competition, which the variational story does not cover. The toy shows the
phenomenology without any variational approximation, so the boundary is clean,
distinct from GVEM / IW-GVEM posterior-approximation bias.

### Why this matters to the thesis (not a side-quest)

If recovered parameters are representation-coupled artifacts, then "a stable
measurement scale, invariant under extension" is meaningless, the scale would be
an optimisation byproduct. Decoupling is a VALIDITY CONDITION for the learned
scale being a real measurement instrument. The gap is load-bearing for the
thesis's central measurement claim, not a detour.

---

## 2. Empirical spine: the Pareto/trajectory/gradient/K-sweep ladder

The study is a sequence of experiments each designed to kill one of three
competing explanations for alpha degradation under sharing, in increasing order
of interest.

1. TRIVIAL CAPACITY. The cheap config was simply too small. A wide enough SHARED
   table recovers alpha AND keeps theta. Decoupling is then unnecessary.
2. EXPRESSIVITY (static). One shared table, even arbitrarily wide, cannot encode
   the theta-optimal and alpha-optimal directions simultaneously. A property of
   the model class / global optimum, independent of optimisation.
3. DYNAMICS (optimisation). A shared table WITH enough rank to express both is
   still driven by gradient flow to the theta-compromise. The good optimum is
   reachable but not reached.

### 2.1 Phase 0, the gate. Is there a trade-off at all, or just size

The Pareto experiment. Sweep the SHARED item-table width W small to large, at
each W record alpha-recovery and theta-recovery, place the decoupled point on
the same axes. Controls, match TOTAL item-embedding parameters and total model
params between the shared sweep and the decoupled point, so the comparison is
about ALLOCATION not budget. Fix decoder, optimiser, epochs, seeds (>=5).

RESULT (2026-06-16), TRADE-OFF CONFIRMED, proceed. Full sweep (LSTM/GPCM,
hidden=32 FIXED, 5 seeds, 150ep, N=800, Q=60, W in {8,16,24,32,48,64,96,128}).
The SHARED family traces a clean Pareto frontier on static_k4, theta falls
monotonically 0.970 (W=8) to ~0.88 (W>=32) while alpha climbs 0.658 to ~0.91. NO
shared width clears both thresholds (theta>=0.95 AND alpha>=0.90), every
alpha>=0.90 point has paid theta down to ~0.88, and the high-theta point (W=8)
has alpha only 0.658. The DECOUPLED family at MATCHED total item-embedding
capacity (Q*W, hidden fixed) sits ABOVE the frontier, theta pinned at ~0.97 AND
alpha 0.90 to 0.94 at every W>=16, at fewer total params. Widening the shared
table to W=128 (6x the decoupled budget) still cannot get both. The gap is
ALLOCATION, not budget. EXPLANATION 1 (trivial capacity) is ELIMINATED.

Reframe (corrects the earlier "sharing starves alpha" story), alpha needs WIDTH
(a wide key, shared OR separate, lifts it, the cheap config's alpha 0.65 was a
width artifact at W=8). But buying that width THROUGH the shared table costs
theta (the wide encoder loses its bottleneck and overfits, an effect that grows
with training). Decoupling delivers alpha's width while keeping the encoder
narrow, so it gets both. The interesting dynamics now points at the THETA side,
why a wide encoder degrades theta, and the open question is EXPLANATION 2 vs 3
(static expressivity vs learning dynamics). Source,
deep_irt/bench/outputs/gate_table.md.

### 2.2 Phase 1, the trajectory. What do the dynamics look like

Checkpoint training every K epochs, recover alpha and theta at each checkpoint,
plot recovery vs epoch for shared (narrow) and decoupled, across seeds.

RESULT (2026-06-16, static_k4, 3 seeds, theta_static and alpha vs epoch).

- SHARED-WIDE (emb=64) alpha PEAKS at 0.906 (ep50) then DECAYS to 0.787 (ep500).
  The architecture CAN express good alpha (it reaches it), continued training
  ABANDONS it. And at ep50 SHARED-WIDE is theta 0.959 AND alpha 0.906, BOTH high.
  So the gate's "shared never gets both" is a 150-epoch SNAPSHOT, the trajectory
  shows the both-high solution is VISITED then LEFT. Evidence for DYNAMICS
  (explanation 3) over expressivity (2), the good solution is reachable and
  reached, then the trajectory leaves it.
- DECOUPLED alpha rises monotonically to 0.912 and HOLDS (no decay).
  SHARED-NARROW alpha rises slowly, plateaus ~0.73.
- theta, all three degrade past ep150 (the bare substrate has no LayerNorm,
  known), SHARED-WIDE fastest (-0.21 vs -0.12). DECOUPLED does NOT fix
  theta-overfit (degrades like NARROW), a SEPARATE problem (regularisation),
  distinct from the alpha protection decoupling provides.

Source, deep_irt/bench/outputs/trajectory_table.md.

### 2.3 Phase 2, the mechanism. Why does theta win the shared embedding

Instrument the shared item embedding e_q. At each step decompose dL/de_q into the
theta-pathway gradient g_theta (through encoder to theta to loss, alpha head's
use of e_q detached) and the alpha-pathway gradient g_alpha (through the alpha
head, theta path detached). Track the magnitude ratio, the cosine, and the
alignment of the net update.

RESULT (gradient conflict on SHARED-WIDE, sum-check PASSED at machine
precision).

- The theta-pathway (LSTM) gradient on the shared embedding GROWS ~28x over
  training (0.0006 to 0.0170) while the alpha-head gradient stays flat (~0.001),
  the ratio reaches ~11x by ep500. Theta progressively dominates the shared
  embedding.
- cos(g_theta, g_alpha) ~ 0 at all epochs, the pathways are ORTHOGONAL, not
  anti-correlated. H5 (tug-of-war) is REFUTED, the effect is MAGNITUDE DOMINANCE
  in an orthogonal subspace, not a directional fight.

HONEST OPEN SUBTLETY (resolved later by the Hessian-conditioning mechanism in
2.6), orthogonality complicates the simple swamping story. If g_theta is
perpendicular to g_alpha, theta updates do not directly overwrite alpha's
embedding direction. The likely mechanism is relative scale, as the
theta-direction variation of E grows ~28x, alpha's fixed orthogonal signal
becomes a shrinking FRACTION of E's variation and the linear alpha readout
cannot isolate the orthogonal alpha-direction at low SNR. Decoupling localises
it to the item-embedding INPUT of alpha (decoupled alpha reads its own table and
does NOT decay), so the decay is via the shared E, not the shared state. Source,
deep_irt/bench/outputs/gradient_conflict_table.md.

NET after Phases 0 to 2, explanation 1 (capacity) killed by the gate,
explanation 2 (expressivity) refuted by the trajectory (alpha reachable then
abandoned), explanation 3 (dynamics) supported by the trajectory AND the
gradient growth. The toy formalises the mechanism.

### 2.4 Phase 3, the theory. Prove dynamics, not expressivity

A minimal analytically-solvable surrogate, with the
global-optimum-vs-gradient-flow discriminator at its centre. Binary 2PL, a
LINEAR encoder, a per-item embedding e_q fed to linear heads, per-person theta_n
learned, P = sigma(alpha_q (theta_n - beta_q)), static ability, population NLL,
continuous-time gradient flow. THE DISCRIMINATOR, (a) compute the GLOBAL
minimiser for the shared architecture wide enough to express both directions,
does it recover unbiased alpha (no => expressivity), (b) from standard init,
where does gradient flow CONVERGE (if (a) recovers but (b) does not =>
dynamics).

RESULT (2026-06-16), minimal model is CLEAN, the driver is AMORTIZED theta. Full
derivation in docs/learning_dynamics_toy.md. The minimal point-estimate 2PL with
a FREE per-person theta and adequate rank (d>=2) does NOT exhibit the coupling,
the global optimum recovers unbiased alpha (gauge-fixed) AND gradient flow
reaches it on 8/8 seeds. No gap, no capture. It cleared three confounds.

- GAUGE, the raw "alpha biased low" magnitude is 100% the 2PL gauge
  (theta -> s.theta+t, alpha -> alpha/s), it collapses to 0.0000 once theta's
  scale is quotiented out. LESSON, report only gauge-fixed or RANK (Spearman)
  alpha, magnitude bias is a coordinate trap. Our empirical metric IS Spearman
  (gauge-invariant), so the empirical phenomenon survives this.
- EXPRESSIVITY, the d>=2 global optimum is unbiased (corr 1.0), confirming the
  trajectory's "alpha reachable". Refuted as the limit.
- FINITE-DATA MLE bias, real, O(1/reps), but IDENTICAL shared vs decoupled, so
  not the phenomenon decoupling fixes.

PROOF (Sec 2.4 of the appendix), for d>=2 with independent readouts the shared
stationary code forces G^alpha=0 AND G^beta=0 separately, no trade-off. The
biasing blend appears only under collinearity (d=1, expressivity) or when theta
competes for the bottleneck. FISHER BRIDGE sharpened, low Fisher SLOWS alpha but
BIASES it only when alpha must SHARE its code direction with a higher-Fisher
parameter, the minimal model gives theta its own free parameter, so no sharing,
no bias.

THE REFRAME (isolated ingredient), the coupling needs AMORTIZED theta read from
the SAME bottleneck that feeds the alpha head, so high-Fisher ability captures
the shared code from low-Fisher alpha. So the phenomenon is NOT item-embedding
sharing per se, it is AMORTIZED-ABILITY CAPTURE. The real model's item code
feeds the LSTM (amortized theta) AND the alpha head, so the gate / trajectory /
gradient results are manifestations of this. VARIATIONAL BOUNDARY clean, pure
point estimate, no posterior anywhere.

### 2.5 Phase 3b, the strong dynamics claim is REFUTED, the bias is FINITE-DATA

Full derivation in docs/learning_dynamics_toy.md Sec 9. Rung 5 (amortized theta
read from a shared item code that also feeds the alpha head) returned the third,
weakest of the three named outcomes.

- DISCRIMINATOR clean at the POPULATION limit, shared global optimum unbiased
  (gauge-fixed bias -0.003, alpha rank 1.0, 8 seeds) AND gradient flow reaches
  it. No capture, no identifiability wall.
- POPULATION-LIMIT PERSISTENCE, the gauge-fixed alpha bias and the across-seed
  spread VANISH as reps to infinity (-3.33 at reps=1 to -0.003 at reps=inf), and
  shared == decoupled at every reps (byte-identical at reps>=20). The bias is
  FINITE-DATA, and DECOUPLING does NOTHING in this toy.
- KEY INSIGHT (survives), for m>=3 with independent readouts the three pathways
  (theta, alpha, beta) zero SEPARATELY at the optimum. Amortization makes theta
  the FAST mode and alpha the SLOW mode (stiff flow, cond ~ I(theta)/I(alpha)),
  but a slow mode is not a biased mode while alpha owns a code direction. LOW
  FISHER SETS THE RATE, NOT THE ENDPOINT.
- MECHANISM of the finite-data bias, the amortizer input is a noisy encoding of
  theta* (errors-in-variables), through the bilinear z = alpha(theta - beta) it
  contaminates alpha. Both architectures share the same noisy amortizer input,
  so decoupling cannot help. Oracle control (clamp theta = theta*) recovers
  alpha exactly.

THE GAP (honest), the toy's effect is NOT the empirical effect. The real model's
alpha degradation IS removed by decoupling on finite data, this fixed-linear-pool
toy's bias is NOT decoupling-fixable. So the empirical decoupling-fix is STILL
unexplained, it depends on a LEARNED, theta-specific encoder pathway the
fixed-pool toy omits.

### 2.6 Phase 3c, rung 6 also clean, the free-table invariant and the wrong-axis insight

Detail in docs/learning_dynamics_toy.md Sec 10. The learned-encoder rung returned
the third negative, decoupling does NOT bite. Shared == decoupled at every reps
and at the population limit (gauge-fixed bias ~-0.001 both, rank 1.0), no
lazy/rich gap (out_scale 0.1 to 20, width 1 to 16), discriminator clean.

GENERAL INVARIANT (the theorem the ladder produced), as long as the per-item
parameters are a FREE table that can hit p = p*, NO readout on top of it (fixed
pool or learned nonlinear encoder) biases the population optimum or makes
decoupling matter, every gradient pull is linear in the residual r = p - p* and
all pulls vanish together at the reachable zero-residual optimum. This explains
why ALL toys (minimal / rung 5 / rung 6) are clean at the optimum.

THE WRONG-AXIS INSIGHT (reframes "alpha learns faster when decoupled"), the
decoupling benefit is NOT an optimum / asymptotic effect (the invariant forbids
it), it is a TRAINING-TIME / EARLY-STOPPING / RATE effect. The toys polish to
the optimum and sweep DATA (reps), which structurally ERASES a transient effect.
The real model trains a FINITE number of epochs (not polished), so it lives in
the transient, where the rate asymmetry (alpha slow / theta fast) bites. This
UNIFIES the evidence, rate asymmetry + Phase 1 (alpha peak-then-decay in shared)
+ Phase 2 (theta-grad grows 28x). "Alpha learns faster when decoupled" is a
CONVERGENCE-RATE / early-stopping claim, and it is exactly the regime real models
train in.

### 2.7 The K-sweep ladder: GPCM transfer, the extended K=2..11 table, the rung-7 mechanism, the N-sweep

GPCM RESULT (2026-06-16, docs/learning_dynamics_toy.md Sec 11). Built the GPCM
(K=4) version of every decisive check (gauge invariance to 3e-15, gradient
sum-check to 1e-16). TRANSFERS EXACTLY, the gauge artifact (a planted s=1.8 reads
as -log 1.8 and collapses to 0 gauge-fixed), the minimal-model clean optimum, the
rung-5 finite-data bias vanishing at the population limit with decoupling INERT
(byte-identical reps>=4). Two sharpenings specific to K>2.

- The rank wall is at d < K, not d < 2 (2PL). GPCM has K pathway directions on
  the shared code, so the free optimum needs item-code rank >= K to zero the
  alpha pull (d=1/2/3 to alpha rank 0.44/0.93/1.00, d>=4 exact). For the real
  K=4 model the minimum item-code rank is 4.
- K WORSENS the theta-vs-alpha stiffness, I(theta)/I(alpha) climbs 1.03 (K=2) to
  2.29 (K=4). I(alpha) rises absolutely with K (milder per-response MLE bias)
  but I(theta) rises faster.

THE NEW FINDING (reproduces the empirical effect), with a LEARNED encoder in
GPCM, decoupling gives a FINITE-DATA advantage on alpha SPEARMAN RANK where 2PL
rung 6 showed NONE.

  reps=1:   shared 0.724 to decoupled 0.875  (+0.150, all 5 seeds positive)
  reps=4:   shared 0.898 to decoupled 0.959  (+0.060)
  reps=200: ~0;  reps=inf: 0.

Decoupling raises alpha rank AND de-noises it, finite-data-only, vanishing
asymptotically. On gauge-fixed MAGNITUDE bias there is still no advantage (the
invariant holds), the effect is purely on RANK. Testing GPCM rather than assuming
2PL transfer was decisive, 2PL hid this.

EMPIRICAL K-SWEEP (5 seeds, the first pass). LSTM/GPCM, e8h32, state_alpha, exp,
150ep, K=2..6, static, shared (alpha_emb=None) vs decoupled (alpha_emb=64).
delta_K = decoupled - shared alpha Spearman, stiffness = analytical
I(theta)/I(alpha) by Monte Carlo over the data priors.

| K | shared a_sp | decoupled a_sp | delta_K | stiffness |
|---|---|---|---|---|
| 2 | 0.626 | 0.807 | +0.181 | 0.96 |
| 3 | 0.694 | 0.918 | +0.224 | 1.42 |
| 4 | 0.658 | 0.928 | +0.270 | 1.87 |
| 5 | 0.640 | 0.933 | +0.293 | 2.34 |
| 6 | 0.711 | 0.943 | +0.232 | 2.80 |

Decoupled beats shared on ALL 25 runs, no theta tax, decoupled alpha std SHRINKS
monotonically (0.049 to 0.020), decoupled alpha itself is MONOTONE in K (0.807 to
0.943). delta_K rises K=2..5 (0.181 to 0.293) tracking stiffness
(Spearman(stiffness, delta)=0.70), then DIPS at K=6 (0.232) in the SHARED arm
(its alpha is high-variance, std up to 0.16). At 5 seeds the K-trend is
UNDER-POWERED.

K-SWEEP EXTENDED (K=2..11, 10 seeds, supersedes the 5-seed run). delta_K =
decoupled - shared alpha Spearman, stiffness = analytical I(theta)/I(alpha).

| K        | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|
| decoup a | .798 | .910 | .926 | .937 | .942 | .945 | .940 | .952 | .949 | .957 |
| shared a | .682 | .734 | .734 | .711 | .755 | .721 | .689 | .651 | .668 | .687 |
| delta_K  | .116 | .175 | .192 | .226 | .187 | .224 | .251 | .300 | .282 | .270 |
| stiffness| 0.96 | 1.42 | 1.87 | 2.34 | 2.80 | 3.28 | 3.75 | 4.23 | 4.75 | 5.22 |

- ROBUST, decoupled wins 9/10 seeds at K=2 and 10/10 at K=3..11, no theta tax at
  any K, decoupled std uniformly low (0.005 to 0.042), de-noising holds to K=11.
- STIFFNESS TRACKING, Spearman 0.891, Pearson 0.907 over K=2..11 (vs 0.70 on
  K=2..5). The advantage tracks the Fisher stiffness across the full practical
  range of real assessment scales.
- SHAPE, RISE-THEN-SATURATE. delta_K grows K=2..9 (peak +0.300) then plateaus,
  the decoupled arm CEILINGS ~0.95 by K=7, so the plateau is a ceiling effect,
  NOT the arms converging (shared stays noisy 0.65 to 0.76). The K=6 dip persists
  (shared-arm variance), so it is NOT strictly monotone.
- RECONCILES WITH IKEDA 2026, the shared arm IS K-messy (matches "no monotonic
  K-vs-recovery"), the decoupled arm is clean, we add the decoupling dimension
  classical IRT cannot see.

Source, deep_irt/bench/outputs/ksweep_table.md.

PHASE 3D, RUNG 7 (2026-06-16, docs/learning_dynamics_toy.md Sec 12), the
transient bites, convergence-RATE win, mechanism = Hessian conditioning. At the
population optimum decoupling is inert (free-table invariant), but in the
TRANSIENT the shared-code flow is STIFF and alpha (low-Fisher) resolves LAST,
decoupling gives alpha an uncontested code that converges at its own rate.

- RATE ADVANTAGE (on RANK), steps for alpha Spearman to reach 0.95, shared
  1025+-1006, decoupled 457+-123. ~2.2x faster, ~8x more reliable.
- VANISHES AT CONVERGENCE, rank gap to +0.001 by step 8000, both reach rank 1.0.
  A transient advantage that closes = a pure RATE effect.
- RANK not BIAS, steps to |logbias|<0.10 identical (2252 vs 2301). The advantage
  is on the ranking metric the tracking model is judged on.
- OWNERSHIP control, warm-starting decoupled E_alpha = E_theta's init
  (byte-identical start) PRESERVES the advantage (491 vs 1025). Structural
  ownership of an uncontested code, not an init/cold-start artifact.
- MECHANISM = HESSIAN CONDITIONING (resolves the Phase-2 orthogonality puzzle),
  the shared-code pathway gradients ARE near-orthogonal (cos 0.05 to 0.20,
  matching Phase 2), but orthogonal gradients do not imply a null rate effect,
  the lag comes from the EIGENVALUE SPREAD of the shared-code Hessian block
  (~ I(theta)/I(alpha) = 2.18 at K=4), a curvature property. Orthogonality kills
  only a gradient-CONFLICT mechanism, the actual mechanism is conditioning.
- OPTIMIZER interaction, largest under plain GD (pays the stiffness in full),
  COMPRESSES under Adam (2.2x to 1.6x), whose preconditioner partially cancels
  the stiffness. The real model uses Adam, so the empirical effect is the (still
  real) compressed version. A per-parameter-preconditioned optimizer partially
  substitutes for decoupling.

N-SWEEP (2026-06-16), at a fixed budget the gap does NOT narrow with data, it is
RATE-limited (confirms rung 7). Empirical N-data sweep (N=100..1600 x K=2..11, 5
seeds, 150ep). delta_K(N) = decoupled - shared alpha Spearman. At the fixed
150-epoch budget the advantage does NOT narrow with more data, it is
flat-to-WIDENING (K=2, 0.03 -> 0.12 -> 0.15 -> 0.18 -> 0.23 over N=100..1600,
K=8, 0.23 -> 0.36, K=9, 0.25 -> 0.38). Higher-K curves sit higher (stiffness).
More learners with the same 150 steps leaves the model more under-trained
relative to the data, the stiff shared flow (slow alpha) cannot catch up, while
decoupled exploits the extra data faster, so the gap persists or widens. The
toy's "gap narrows with data" holds only AT CONVERGENCE, the real model at 150ep
is in the rate-limited regime, so the two faces (data / training) are ENTANGLED
at a fixed budget and the RATE face dominates. Figures, ksweep_plot.png,
ndata_sweep_plot.png. Source, ndata_sweep_results.json.

### 2.8 The converged verdict

STUDY STATUS, COMPLETE / CONVERGED (2026-06-16). The analytical ladder bottomed
out at rung 7, the mechanism is pinned (Hessian conditioning ~ I(theta)/I(alpha),
grows with K), the empirical benches (gate, trajectory, gradient, K-sweep,
N-sweep) and the toys (minimal / rung 5 / 6 / 7, GPCM) all agree.

FINAL CONTRIBUTION, decoupling the discrimination representation in amortized
neural IRT buys a Fisher-conditioning-governed CONVERGENCE-RATE advantage on
discrimination RANK recovery, scaling with K, dominant at any practical training
budget, with NO endpoint cost (the optimum is invariant). A workshop-shaped
methods result, prior-art-cleared, variational-clean, reproduced in tractable
toys AND empirically.

The decoupling benefit has TWO faces, both governed by the Fisher stiffness
I(theta)/I(alpha) (which grows with K), both on RANK. (1) FINITE DATA, a
sample-efficiency advantage (GPCM toy Sec 11, vanishes as data to infinity).
(2) FINITE TRAINING, a convergence-rate advantage (rung 7, vanishes as training
to infinity). Both vanish only at the JOINT (infinite data + infinite training)
limit. Real models live at finite-data + finite-training, so they get the
benefit. Mechanism, Hessian conditioning, not gradient conflict.

What it is NOT, a population-limit learning-dynamics law. The honest contribution
is (a) a robust empirical phenomenon (decoupling fixes alpha recovery,
architecture-independent, finite-data), (b) a rigorous ladder of what it is NOT,
gauge, expressivity, population-limit dynamics, plain errors-in-variables all
ruled out, (c) the clean "low Fisher sets the rate not the endpoint" result plus
the errors-in-variables finite-data mechanism, (d) a clean variational boundary.

METHODOLOGY MAPPING (deep-research survey, Saxe deep-linear, Pesme / Jacot
saddle-to-saddle, Atanasov silent-alignment, Chizat lazy-vs-rich, PCGrad, 103
agents, 21 primary sources). R1 Saxe Fig 3 per-parameter recovery vs step, WE
HAVE IT (Phase 1). R2 the scalar law t = O(tau/s), the SURVIVING result ("low
Fisher sets the rate not the endpoint"), the toy gives cond ~ I(theta)/I(alpha).
R3 population vs finite-data, the field-standard control, WE RAN IT (Phase 3b)
and got a clean honest NEGATIVE on the bias. R4 silent-alignment / gradient-cosine
vs time, WE HAVE A VERSION (Phase 2, theta-gradient grows 28x, orthogonal). R5
lazy-vs-rich controls if we pursue the learned-encoder rung. The honest
contribution is presentable with canonical artifacts we mostly already have, the
RATE asymmetry (R1 + R2) + the population-vs-finite control (R3, the honest
negative) + the gradient-alignment diagnostic (R4). Framing, "the learning
dynamics of amortized IRT recovery," NOT a gradient-capture law.

---

## 3. The Fisher asymmetry and the convergence-rate dynamics (RQ1, RQ2)

Status PRELIMINARY (3 to 5 seeds, single architecture, synthetic static-alpha
data). Branch feat/prediction-loss. 2026-06-17. This section reframes the spine's
shared-vs-decoupled result onto the state-conditioned head (let the decoder read
the encoder hidden state when producing the parameter, occurrence-averaged at
recovery), the lever the spine's toy ladder pointed to.

DKT-home framing (the translation gap). That discrimination is low-information
and ill-conditioned is classical IRT (see Classical grounding below), this study
does not reclaim it. The contribution is the DL-native question IRT's static
theory never poses, what the TRAINING DYNAMICS do with it, whether and how fast
SGD pins each parameter under a PREDICTION loss, and how a representational choice
(decoupling, state-conditioning) changes that. Fisher information appears only as
the backstage bridge (a parameter's leverage on the prediction yhat), the
front-stage story and the claims are learning dynamics, for a DKT / DL / EduAI
audience. The objective throughout is the prediction loss on y vs yhat, not a
model-wise likelihood, the IRT triple is the route to yhat, not the estimand.

The conjecture, from the per-response Fisher of the 2PL/GPCM. Discrimination
alpha is LOW information, I(alpha) ~ (theta - beta)^2, which vanishes at
theta = beta where targeted responses concentrate, so alpha is hard and
ill-conditioned at finite data. Difficulty beta and ability theta are higher
information. So a richer (state-conditioned) readout should help the low-Fisher
alpha and do little for the high-Fisher beta. The stiffness I(theta)/I(alpha)
grows with K, so the alpha benefit should grow with K. All runs decoupled
(item_key_dim=64), exp transform, static GPCM data, N=800, Q=60, T=60, 150
epochs, Adam. Pearson r vs ground truth.

### 3.1 RQ1, the alpha-vs-beta asymmetry (3 seeds)

`alpha-dynamic` switches ON only the state-conditioned alpha head (beta static),
`beta-dynamic` switches ON only the state-conditioned beta head (alpha static),
the two heads gate independently, so each arm makes exactly one parameter
dynamic. Same architectural change, applied to a low-Fisher and a high-Fisher
parameter.

| K | a base | a dyn | delta_alpha | b base | b dyn | delta_beta |
|---|---|---|---|---|---|---|
| 2 | 0.798 | 0.703 | -0.095 | 0.977 | 0.987 | +0.010 |
| 4 | 0.876 | 0.933 | +0.057 | 0.983 | 0.986 | +0.002 |
| 6 | 0.928 | 0.941 | +0.014 | 0.982 | 0.983 | +0.002 |
| 8 | 0.914 | 0.951 | +0.037 | 0.982 | 0.983 | +0.001 |
| 11 | 0.755 | 0.951 | +0.196 | 0.980 | 0.979 | -0.001 |

mean delta_alpha = +0.042, mean delta_beta = +0.003, Pearson(delta_alpha, K) =
+0.877 (this is np.corrcoef, a PEARSON, the rank Spearman(delta_alpha, K) =
+0.70, lower because of the noisy non-monotone mid-K, an earlier draft mislabeled
the Pearson as "Spearman").

Mechanism figure (deep_irt/bench/run_fisher_ratio.py, fisher_ratio.png). The GPCM
stiffness E[I(theta)]/E[I(alpha)] computed analytically under the bench priors
(K=2 cross-checked against the closed-form 2PL) GROWS monotonically with K,
0.98, 1.94, 2.81, 3.77, 5.34 for K = 2,4,6,8,11, because E[I(alpha)] plateaus
(0.20 to 0.49) while E[I(theta)] keeps climbing (0.20 to 2.61). delta_alpha
tracks it (Pearson on log-stiffness +0.87). At K=2 stiffness ~ 1 and delta_alpha
< 0 (the dynamic head only adds noise), the benefit switches on as stiffness
climbs.

Classical grounding (lit pass, 2026-06-17). That discrimination is the
LOW-information parameter is classical (I(a) ~ (theta-b)^2 P(1-P), vanishing at
theta=b, confirmed by a primary GRM Monte Carlo). But the K-GROWTH of the dynamic
benefit is NOT anticipated by classical estimation theory and runs COUNTER to it,
the best primary evidence (PLOS One 2024 GRM) finds K NEGLIGIBLE (or, Frontiers
2019, harmful) for STATIC discrimination recovery, attributing it to per-category
information dilution. The dynamic, occurrence-pooled readout escapes that
dilution. So RQ1 is the prediction-side, learning-dynamics counterpart to a
result static estimation theory gets the opposite way.

CONFIRMED. Making the low-Fisher alpha dynamic helps and the help GROWS with K,
making the high-Fisher beta dynamic does essentially nothing at any K. The K=2
sign-flip (delta_alpha = -0.095) is consistent with the mechanism, not against
it, at K=2 the stiffness is lowest so alpha is relatively well determined and the
dynamic head only adds noise (it behaves like beta), and the benefit switches on
as stiffness grows. So the effect is parameter-specific and Fisher-governed,
which rules out "generic flexibility helps."

### 3.2 RQ2, the dynamics (5 seeds)

Per-epoch alpha-recovery trajectory for alpha-static vs alpha-dynamic (beta
static in both), via the fit callback (one optimizer, no warm restart). Gap =
dynamic - static, mean over 5 seeds.

```
K     ep1    ep20   ep40   ep80   ep150     static@150   dynamic@150
4    -0.07  +0.26  +0.48  +0.27  +0.03       0.914         0.944
8    -0.04  +0.18  +0.38  +0.31  +0.06       0.902         0.966
11   +0.03  +0.10  +0.42  +0.42  +0.20       0.775         0.970
```

The predicted story ("dynamic peels up EARLIER from the start, lead widens with
K") is HALF WRONG and the correction is the real finding.

- Both curves crawl for the first ~10 to 20 epochs, dynamic is NOT ahead early
  (it is neck-and-neck or slightly behind). So there is no early peel-up.
- A MID-TRAINING surge (ep20 to 40) is where the dynamic head accelerates past
  static, the gap peaks around ep40 to 80 at +0.4 or so, at every K.
- At the ENDPOINT the gap grows monotonically with K (K=4 +0.03, K=8 +0.06,
  K=11 +0.20) because at low K static catches up by ep150, while at high K static
  is trapped (0.775) and dynamic stays far ahead (0.970).

So there IS a convergence-rate advantage, but it is a mid-training ACCELERATION
after a shared slow start (dynamic reaches any given recovery level sooner once
the encoder state organizes, e.g. it hits ~0.78 by ep40 where static needs
~ep90), NOT an earlier start. And at high K that rate advantage converts into a
PERMANENT endpoint ceiling gap, because static-alpha cannot escape the stiffness
bottleneck the dynamic head breaks through.

Mechanism. The static head can fit alpha from the item embedding immediately, so
it is competitive early but caps low against the stiffness ceiling. The dynamic
head must wait for the encoder to organize a useful state before its conditioning
means anything, so it lags early, then breaks the ceiling once the state matures.
It trades early speed for a higher ceiling, and at high K the ceiling difference
is the lasting effect.

### 3.3 Combined verdict (preliminary)

The state-conditioned discrimination advantage is real, parameter-specific, and
Fisher-governed (RQ1). Its dynamics are a shared slow start, a mid-training
acceleration of the low-Fisher parameter, and a K-growing permanent endpoint gap
where static-alpha is trapped by the stiffness ceiling (RQ2). The earlier "peels
up from epoch one" framing (inherited from the retracted width study) does not
hold for the static-vs-dynamic head, correct it to "mid-training acceleration
plus K-growing ceiling escape."

Caveats. 3 to 5 seeds, one architecture (LSTM/GPCM, emb=8/hidden=32/key=64), one
data regime (static-ability synthetic). The mid-K delta_alpha values in RQ1 are
noisy, the robust signals are the endpoints, the beta null, and the
K-correlation.

Reproduce.

```
python deep_irt/bench/run_alpha_beta_asymmetry.py --device cuda          # RQ1
python deep_irt/bench/run_convergence.py --seeds 0 1 2 3 4 --device cuda  # RQ2
```
Outputs under deep_irt/bench/outputs/ (alpha_beta_asymmetry.*,
convergence_K*.json).

---

## 4. State-conditioned discrimination as a signal: real or artifact (RQ3)

Status PRELIMINARY (3 seeds, needs replication). Branch feat/prediction-loss.
2026-06-17. Phase 1 (is the wiggle real or artifact) and Phase 2 (does it detect
planted theta-dependence) both DONE, the verdict is directional detection, not
calibrated magnitude.

The object. The state-conditioned discrimination head produces a per-occurrence
alpha, alpha_jt = exp(fc_a_state([state_t, item_key_j])). Decompose it as
a_static(j) = mean_t alpha_jt (the would-be fixed item discrimination) and
a_dynamic(j,t) = alpha_jt - a_static(j) (the per-occurrence wiggle). a_dynamic is
the neural-IRT-native quantity, classical IRT has one scalar alpha per item and
nothing corresponding to the wiggle. The question is whether the wiggle is a real
signal (discrimination that genuinely varies with the respondent's state) or an
estimation artifact. beta is item-key-only so it has no dynamic part by
construction, theta is purely dynamic, the decomposition is alpha-specific.

Test design. Train on STATIC-alpha synthetic GPCM data, where the true
discrimination is occurrence-invariant, so the true a_dynamic is exactly zero.
Anything the model produces is therefore artifact, this is the clean null.

### 4.1 Result 1, null probe, N-sweep (3 seeds, static K=4)

| N | a_static recovery | dyn_CV | corr(a_dyn, theta_true) | corr(a_dyn, theta_model) |
|---|---|---|---|---|
| 200 | 0.756 | 0.640 | +0.140 | +0.121 |
| 800 | 0.923 | 0.837 | +0.020 | +0.025 |
| 3200 | 0.933 | 0.631 | +0.054 | +0.035 |

- a_static recovers the true alpha and sharpens with N (0.76 to 0.93).
- a_dynamic does NOT vanish with data, dyn_CV stays ~0.6 to 0.8 across a 16x
  increase in N. So it is not finite-data scaffold, it is structural, the head
  injects a fixed amount of per-occurrence wobble because it reads the full
  hidden state, not just theta. The "more data kills it" branch is ruled out.
- Linear corr(a_dyn, theta) is near zero. This turned out to be MISLEADING (see
  Result 2), the relation is non-monotone and nearly cancels in a single Pearson.

### 4.2 Result 2, the a_dynamic vs theta relation (single model, N=2000, K=4)

```
a_static recovery 0.946   a_dynamic CV 0.773
linear  corr(a_dyn, theta_model) = -0.266   corr(a_dyn, theta_true) = -0.188
gap     corr(a_dyn, theta-beta)  = -0.150   corr(a_dyn, |gap|) = +0.146   corr(a_dyn, gap^2) = +0.156
R^2 of a_dyn from cubic(theta_model) = 0.179   cubic(theta_true) = 0.087
```

Binned a_dynamic across theta_model deciles.

```
theta decile   a_dyn mean   a_dyn std
  -1.91         +0.638        1.06      low-ability tail: alpha inflated and wild
  -1.20         +0.137        0.68
  -0.78         ~0            0.57
  -0.05         -0.168        0.47
  +0.59         -0.154        0.39
  +0.96         -0.140        0.38      bulk: alpha slightly deflated, tight
  +2.09         -0.006        0.43
```

Reading. On data where true discrimination is theta-independent, a_dynamic
carries a real NONLINEAR theta-structure, a cubic in theta_model explains ~18% of
its variance. The shape is a Fisher-tail estimation bias. At the low-theta tail
the head inflates alpha (+0.64) and its variance explodes (std 1.06), through the
bulk it deflates slightly and tightens. The positive |gap| and gap^2 correlations
confirm the geometry, the wobble concentrates where (theta-beta)^2 is small,
exactly where alpha is least identified (I(alpha) ~ (theta-beta)^2 vanishes
there). So most of the wiggle is the head over-estimating and destabilizing
discrimination where the data is uninformative, not a genuine context-dependence.
Analogy, judging a ruler's precision by measuring only babies (all the same
height), no height range, so the readings are nonsense, and the nonsense is not
"the ruler behaves differently for babies."

Phase 1 verdict (preliminary). a_dynamic is not finite-data scaffold (persists
with N) and not clean noise (carries nonlinear theta-structure, R^2 ~0.18). Out
of the box it is CONTAMINATED by an information-starved estimation bias
concentrated at the low-ability / low-Fisher tail. The naive detector
corr(a_dynamic, theta) is therefore non-zero and structured on the null, so a
positive correlation on real data cannot be read as genuine theta-dependent
discrimination without controlling this bias first. Linear correlation is a bad
summary (the non-monotone shape cancels), the structure is only visible
nonlinearly. The bias itself is worth reporting, anyone reading state-conditioned
alpha as "context-dependent discrimination" is partly reading a low-ability
estimation artifact, which the neural-IRT line has not flagged.

### 4.3 Phase 2, the decisive signal-detection test (3 seeds, K=4)

Plant genuine theta-dependent discrimination, alpha_eff(i,t) = a_j *
exp(gamma_j * theta_it), with gamma_j ~ N(0, sigma) drawn from a SEPARATE rng
(datagen.py). At a fixed seed the null (sigma=0) and every planted set share the
same a, b, theta, item sequences and per-step choice draws, only the responses
differ, so the null is a matched bias control and (because the gamma rng is
seeded once) the planted gammas at different sigmas are proportional, a clean
dose-response on the same items.

Readout. For each fit, read every per-occurrence alpha_jt and take the per-item
OLS slope of log(alpha_jt) on the TRUE theta (external, no latent circularity).
This linear slope is the matched estimator for the planted form (log alpha =
log a + gamma*theta is exactly linear) AND it sidesteps the Phase-1 contamination
(the Fisher-tail bias is non-monotone in theta and nearly cancels under a linear
projection). The detector is signal_j = slope_planted_j - slope_null_j, scored as
corr(signal_j, gamma_j), calibration is the OLS slope k of signal on gamma (k = 1
is exact magnitude).

| sigma | corr(slope_planted, g) | corr(signal, g) | corr(slope_null, g) | calib k | null slope std |
|---|---|---|---|---|---|
| 0.20 | +0.427 | +0.438 | -0.040 | 0.039 | 0.015 |
| 0.40 | +0.649 | +0.666 | -0.040 | 0.040 | 0.015 |

Findings.

1. DETECTION CONFIRMED. The head recovers the planted theta-dependence, and it is
   a genuine dose-response, corr rises 0.43 to 0.67 as the planted slope doubles.
   The sanity corr(slope_null, gamma) ~ 0 holds (gamma is independent of item
   a/b), so the positive correlation is detection, not a spurious pathway.
2. THE BIAS PROBLEM EVAPORATES ON THE RIGHT READOUT. On the linear slope the null
   bias is tiny and gamma-independent (std 0.015 vs a planted signal an order
   larger, corr_null ~ 0), so the matched-null correction is nearly a no-op
   (0.649 to 0.666). The contamination was specific to the NONLINEAR
   per-occurrence wiggle, the linear log-alpha-on-theta slope is the clean
   instrument and needs almost no bias control. (Result 2's "linear correlation
   cancels" cut both ways, it kills the naive detector but it also kills the
   bias, leaving the linear PLANTED signal clean.)
3. MAGNITUDE IS NOT RECOVERED, ONLY RANK, AND THE SHRINKAGE IS GENUINE (not a
   scale artifact). Calibration k ~ 0.04 at both sigmas (stable), the recovered
   slope is ~4% of the planted magnitude, a ~25x shrinkage. A natural objection
   is identifiability, the model's latent theta is fixed only up to an affine
   scale and in a*(theta-b) discrimination scales inversely with theta's scale,
   so a compressed internal theta would deflate the alpha slope by the same
   factor. The decomposition rules this out (3 seeds, sigma=0.4), the model's
   per-learner theta recovers true theta0 at corr 0.96 on essentially the right
   scale, c = OLS(theta_hat, theta0) = 1.14 (slightly INFLATED, which would only
   deflate k further), so the scale-corrected calibration k/c = 0.034 barely
   differs from k. The ~30x attenuation is therefore genuine HEAD SHRINKAGE, not
   a scale mismatch. Mechanism, and this is the IRT-as-FLAVOR point, the training
   objective is a PREDICTION loss on y vs yhat (WOL, the ordinal-penalised
   cross-entropy on the GPCM logits), the triple (theta, alpha, beta) is the
   structured ROUTE to yhat, never the estimand. So the loss pins a parameter
   only through that parameter's leverage on the prediction, NOT by estimating
   it. Alpha's leverage on the predicted response distribution IS the GPCM Fisher
   information I(alpha) ~ (theta-beta)^2, which vanishes at theta=beta where
   targeted responses concentrate, there the prediction is nearly blind to alpha.
   So the prediction objective cannot determine alpha's theta-slope MAGNITUDE, it
   pins only the sign and rank (which still nudge yhat) and leaves the size to the
   optimization dynamics, which shrink it. The magnitude is not mis-estimated, it
   is UNDER-IDENTIFIED by prediction wherever alpha barely moves yhat, the same
   low-Fisher leverage that limits alpha recovery overall (RQ1).
   State-conditioned alpha is therefore a RANK / direction detector of
   theta-dependent discrimination, not a calibrated estimate of it, reading the
   size of a_dynamic as the strength of context-dependence is wrong by a large
   factor.

Verdict. The neural-IRT-native quantity a_dynamic does carry real signal about
genuine context-dependent discrimination, recoverable in rank once read as a
linear log-alpha-on-theta slope, which simultaneously dodges the Fisher-tail
bias. Its magnitude is heavily and genuinely attenuated (~30x head shrinkage, not
a scale artifact), so the honest claim is DIRECTIONAL detection, not calibrated
measurement.

Prior art (2026-06-17 lit pass). No neural KT state-conditions DISCRIMINATION,
the closest (SAD-IRT, NCDM) state-condition DIFFICULTY, the classical relatives
are D2PMM person-discrimination (a per-person constant, theta-INDEPENDENT) and
non-uniform DIF (a group-level slope-by-ability interaction, not within-person).
So both the modeling (state-conditioned alpha) and the rank-vs-magnitude
diagnostic are novel. See memory dynamics-recovery-prior-art.

### 4.4 K-robustness (K in {2, 4, 8}, 3 seeds)

Repeating the detection test across K confirms it generalizes and splits the two
predictions cleanly, one held, one did not.

| K | corr(signal,g) sigma=0.2 | corr(signal,g) sigma=0.4 | calib k sigma=0.4 | null slope std |
|---|---|---|---|---|
| 2 | +0.421 | +0.492 | 0.058 | 0.039 |
| 4 | +0.438 | +0.666 | 0.040 | 0.015 |
| 8 | +0.358 | +0.621 | 0.028 | 0.013 |

- Detection is robustly positive and dose-responsive at EVERY K, but does NOT
  rise monotonically with K (a modest K=2 to K=4 lift, then a noisy plateau at
  K=8). The "more categories sharpen the state so detection improves" prediction
  does not hold, detection rank is largely K-insensitive within seed noise. So
  whatever caps the rank is not state sharpness.
- Calibration k FALLS monotonically with K (sigma=0.4, 0.058 to 0.040 to 0.028,
  same direction at sigma=0.2), the magnitude shrinkage WORSENS as K grows. That
  matches the low-Fisher prediction, higher K lowers alpha's leverage on yhat
  relative to the other parameters, so prediction pins its theta-slope even less.
  Caveat, the scale c was only measured at K=4 (1.14), so the cross-K k trend is
  suggestive of the Fisher mechanism, not a scale-clean proof.
- The null-bias slope std also shrinks with K (0.039, 0.015, 0.013), the linear
  slope is estimated more cleanly with more categories, which is why differencing
  matters least at high K.

Net, the directional-detection-not-magnitude verdict holds across K, and the one
quantitative K-trend (calibration degrading with K) points the same way as RQ1,
the low Fisher of alpha is the binding constraint.

### 4.5 Implications and the real-data validation design (future)

Phase 2 is DONE and turned the instrument question on its head. The naive
per-occurrence wiggle does need the regularized split or a bias model, but the
LINEAR log-alpha-on-theta slope already IS the clean instrument, it both matches
the planted signal and dodges the nonlinear Fisher-tail bias, so no extra bias
control is needed for directional detection. Magnitude is genuinely shrunk ~30x,
RESOLVED as head shrinkage not a scale artifact (theta recovers at corr 0.96,
scale c=1.14, so k/c=0.034 ~ k). Lifting k off ~0.04 toward a calibrated estimate
needs an objective that targets the slope DIRECTLY (a regularized static+dynamic
split, or an auxiliary term that rewards the alpha-theta slope), because the
prediction loss only constrains alpha through its leverage on yhat, which is near
zero where the magnitude lives. This is the open MAGNITUDE problem, detection
(rank) is solved.

Real-data validation, A/B design. Use a dataset where the same student answers
items across different subjects or skills (cross-subject), which gives
within-student ability spread. Two characterizations, one clean and one hard.

A. a_static is the test/subject-defining property, invariant across students.
   This is measurement invariance and it is the clean confirmatory test. Estimate
   a_static(item) from different student splits (cohort, ability band) and check
   it agrees item by item, across subjects a_static should cluster by subject.
   Doable on real data today.

B. a_dynamic of one student tracks that student's theta within noise. The
   ambitious test, two confounds block it. Theta is latent, so testing a_dynamic
   against the model's OWN theta is circular and co-contaminated by the same
   Fisher-tail bias. And cross-subject mixes item effects (different a_static)
   with theta effects. The cleanest handle is to make theta external, estimate a
   student's subject ability from a HELD-OUT set of that subject's items, then ask
   whether their wiggle on different items of that subject tracks the held-out
   ability and not their other-subject ability. Still model-estimated, so a
   validation, not a proof.

Ordering. Establish the mechanism on synthetic Phase 2 first, where theta and the
planted theta-dependence are known and there is no latent-variable circularity,
THEN validate on real cross-subject data with A as the clean confirmatory test
and B as the ambitious one. Running B before A, or before Phase 2, is how the
Fisher-tail bias gets published as a discovery.

Reproduce.

```
python deep_irt/bench/run_adynamic_probe.py          # Result 1 (N-sweep null probe)
python deep_irt/bench/run_adyn_theta_relation.py     # Result 2 (relation study)
python deep_irt/bench/run_phase2_signal.py --device cuda   # Phase 2 (signal detection)
python deep_irt/bench/run_phase2_scale.py            # Phase 2 magnitude decomposition
```

---

## 5. Robustness to the training loss (historical loss-invariance evidence)

Date 2026-06-16. Branch feat/prediction-loss (off feat/duolingo-mini). Question,
the deep_irt training loop once optimized the GPCM likelihood NLL directly.
ma-irt never did that (it trains on a prediction loss, weighted ordinal loss),
Deep-IRT (Yeung) never did either (BCE on the next response). IRT is meant to be
a readout FLAVOR, not the objective. So re-fit everything on a pure prediction
loss and ask whether the decoupling result survives. This section is the
historical record of that gate.

> RETRACTION BANNER. The lightweight s_0 WIDTH-DECOUPLING verdict that this
> document originally carried (the SHARED-width vs DECOUPLED-width result in
> Section 5.1) was LATER RETRACTED as a CAPACITY ARTIFACT. Do NOT cite the s_0
> width split as a live result. The real lever is ma-irt's key/value split (the
> wide item_key table feeding the static readout, the thin item_val table feeding
> the LSTM/theta) and the state-conditioned head studied in Sections 3 and 4. The
> tables below are retained ONLY as evidence that the decoupling phenomenology and
> its K-stiffness tracking are INVARIANT to whether the categorical loss is the
> GPCM NLL or ma-irt's WOL, not as a claim that the s_0 width decoupling itself
> stands.

### 5.1 Result 1, the gate (theta-vs-alpha frontier, K=4, 5 seeds, matched params)

SHARED(W) spends one item table of width W on both the LSTM input and the alpha
head. DECOUPLED(W) uses a narrow encoder (emb=8) + a separate alpha table of
width W-8, matched in total item params.

The qualitative frontier dominance is unchanged across the loss switch. SHARED
trades theta away as it widens to chase alpha and never reaches both high,
DECOUPLED holds theta high AND reaches the both-high corner SHARED cannot.

| static_k4, W=48 (matched) | theta_static | alpha (Spearman) |
|---|---|---|
| NLL  DECOUPLED | 0.968 | 0.936 |
| WOL  DECOUPLED | 0.970 | 0.904 |
| NLL  SHARED    | 0.881 | 0.900 |
| WOL  SHARED    | 0.903 | 0.922 |

- theta-protection (the large, robust part), unchanged. DECOUPLED theta stays
  ~0.967 to 0.971 under both losses, SHARED still falls to ~0.90 at high W.
- alpha-recovery level, attenuated ~0.03 to 0.05 under WOL (DECOUPLED alpha ~0.93
  to ~0.88 across W), about one seed-std. DECOUPLED still uniquely reaches
  theta >= 0.95 AND alpha >= 0.9 (W=48, 0.970/0.904), no SHARED cell does under
  either loss.

Tables, deep_irt/bench/outputs/gate_table_{NLL,WOL}_full.md.

### 5.2 Result 2, the K-sweep (does the advantage scale with Fisher stiffness)

The dynamics headline was that the decoupled-minus-shared alpha gap delta_K grows
with K and tracks the Fisher stiffness ratio I(theta)/I(alpha). Recorded NLL
value rho = 0.89 (10 seeds). 5-seed same-machine reruns.

| loss | Spearman(stiffness, delta_K) | delta_K trend (K=2 -> 11) | sign consistency |
|---|---|---|---|
| NLL (5 seeds) | 0.745 | +0.03 -> +0.32 | K>=4 mostly 5/5 |
| WOL (5 seeds) | 0.903 | +0.03 -> +0.42 | K>=4 unanimous 5/5 |

The K-scaling survives under WOL, indistinguishable from (and at matched 5 seeds,
numerically above) the NLL baseline. The 5-seed rho is noisy, the 10-seed
confirmation gives the headline rho = 0.89. Tables,
deep_irt/bench/outputs/ksweep_table_{NLL,WOL}.md.

### 5.3 Mechanism (analytical), why theta-protection is untouched but alpha attenuates

1. The detached per-sample weight w_i = w_class[y_i] * (1 + 0.5*|argmax - y_i|)
   enters the Gauss-Newton/Fisher matrix linearly in front of every block. A
   GLOBAL constant weight is pure scale and leaves the stiffness
   kappa = I(theta)/I(alpha) exactly invariant, only the sample-dependence of
   w_i can move it, and a CPU toy shows it moves kappa slightly DOWN (favorable
   to alpha), 0.94x at K=4 to 0.89x at K=11. The population endpoint stays
   loss-invariant (the free item-param table can still hit p=p*), so any WOL
   effect is rate/transient, not identifiability.
2. theta-protection is an ownership/curvature property invariant to any positive
   per-sample scalar, hence untouched. The alpha attenuation is NOT a
   conditioning change (conditioning shifted the favorable way), it is a mild
   "target tilt", the ordinal-distance weight up-weights the model's own
   argmax-error samples, which while alpha is under-set are the flat-model tail
   responses, applying a small anti-alpha pressure that lowers the alpha ceiling
   roughly uniformly without reintroducing the shared trade-off.
3. Predicted rho_WOL ~ 0.85 to 0.93 (slight upward lean), since stiffness and the
   WOL tilt both scale monotonically with K in the same direction. The observed
   5-seed 0.903 confirms the prediction. Falsifier would be rho < 0.80.

### 5.4 Loss-invariance verdict and CE-family scope

The decoupling phenomenology and its K-scaling are NOT an artifact of optimizing
the GPCM likelihood. They survive the switch to ma-irt's pure prediction loss,
theta-protection intact, K-stiffness tracking intact (rho ~ 0.9), alpha level
modestly attenuated by a benign target tilt. The cleanest defensible claim,
decoupling is a property of an IRT-structured bilinear readout under any
softmax-CE-family prediction loss, inherited regardless of which categorical loss
scores the logits. It is expected to break only for non-CE-family losses (EMD on
the expected category, margin/ranking) or an aggressive penalty lambda ~ O(K),
the clean adversarial test if a reviewer pushes back. ma-irt remains the
canonical measurement kernel, its prediction-loss recipe (WOL) is now the
deep_irt training loss too, so the two are consistent rather than the deep_irt
side silently optimizing a different objective.

---

## 6. Theory appendix pointer

The standalone derivations are in `docs/learning_dynamics_toy.md`. It is the
THEORY APPENDIX of this study and is not inlined here. It contains the 2PL gauge
proof (theta -> s.theta+t, alpha -> alpha/s, the raw shrinkage collapsing to
0.0000), the rank-wall proofs (d < 2 for 2PL, d < K for GPCM), the free-table
invariant (Sec 10, every gradient pull linear in the residual r = p - p* so all
vanish together at the reachable optimum), the GPCM Fisher table (Sec 11.5,
I(theta)/I(alpha) climbing 1.03 at K=2 to 2.29 at K=4), and the rung-7
GD-vs-Adam controls (Sec 12, the 2.2x GD / 1.6x Adam rate advantage and the
byte-identical warm-start ownership control).
