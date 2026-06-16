# Study plan -- learning dynamics of representational capacity coupling in neural IRT

Operationalises the open core of `docs/RESEARCH_GAP.md`. Goal: decide whether
the alpha-recovery degradation under representation sharing is trivial capacity,
static expressivity, or learning dynamics, and if dynamics, explain the
mechanism at the math level. Opened 2026-06-16.

---

## The three competing explanations (what we must separate)

For a neural IRT model that reads theta, alpha, beta off a shared item
representation, the observed alpha degradation under sharing has three candidate
causes, in increasing order of interest:

1. TRIVIAL CAPACITY. The cheap config was simply too small. A wide enough SHARED
   table recovers alpha AND keeps theta. Decoupling is then unnecessary.
2. EXPRESSIVITY (static). One shared table, even arbitrarily wide, cannot encode
   the theta-optimal and alpha-optimal directions simultaneously. A property of
   the model class / global optimum, independent of optimisation.
3. DYNAMICS (optimisation). A shared table WITH enough rank to express both is
   still driven by gradient flow to the theta-compromise. The good optimum is
   reachable but not reached. This is the contribution.

The study is a sequence of experiments each designed to kill one explanation.

## What we already have (honest inventory)

- A CONFOUNDED capacity probe. The "64x64 trap" widened the shared table, but
  that also widened the LSTM input, so theta capacity grew too and theta broke.
  This conflates alpha-DOF with theta-DOF; it does NOT cleanly test capacity.
- TWO dynamics glimpses, unanalysed: theta overfits MORE with training length
  (0.97 at 150ep to 0.68 at 500ep), and alpha is high-variance across seeds.
- The decoupled-vs-shared outcome at one operating point, architecture-
  independent. An endpoint, not a mechanism.

Neither capacity nor dynamics has been studied properly. Everything below is new.

---

## Phase 0 -- THE GATE. Is there a trade-off at all, or just size? (kills explanation 1)

The Pareto experiment. Sweep the SHARED item-table width W from small to large;
at each W record alpha-recovery and theta-recovery. Plot the (theta, alpha)
frontier the shared family traces. Place the decoupled point on the same axes.

- Controls: match TOTAL item-embedding parameters and total model parameters
  between the shared sweep and the decoupled point, so the comparison is about
  ALLOCATION not budget. Fix decoder, optimiser, epochs, seeds (>=5).
- Outcomes:
  - Some shared W reaches BOTH high alpha AND high theta (no frontier) -> TRIVIAL
    CAPACITY. Decoupling is unnecessary; report honestly and stop.
  - Shared traces a frontier (alpha up <=> theta down, never both) and the
    decoupled point sits ABOVE it (dominates) -> a real trade-off that separation
    removes. Proceed. The frontier-dominance is a stronger result than any single
    matched-point comparison.
- This is cheap (reuse the bench, sweep one knob). It MUST run first; if it says
  "trivial capacity," there is no study.

### Phase 0 RESULT (2026-06-16): TRADE-OFF CONFIRMED -- proceed

Full sweep (LSTM/GPCM, hidden=32 FIXED, 5 seeds, 150ep, N=800, Q=60, W in
{8,16,24,32,48,64,96,128}). The SHARED family traces a clean Pareto frontier on
static_k4: theta falls monotonically 0.970 (W=8) -> ~0.88 (W>=32) while alpha
climbs 0.658 -> ~0.91. NO shared width clears both thresholds (theta>=0.95 AND
alpha>=0.90): every alpha>=0.90 point has paid theta down to ~0.88, and the
high-theta point (W=8) has alpha only 0.658. The DECOUPLED family at MATCHED
total item-embedding capacity (Q*W, hidden fixed) sits ABOVE the frontier: theta
pinned at ~0.97 AND alpha 0.90-0.94 at every W>=16, at fewer total params.
Widening the shared table to W=128 (6x the decoupled budget) still cannot get
both. The gap is ALLOCATION, not budget. EXPLANATION 1 (trivial capacity) is
ELIMINATED.

Reframe (corrects the earlier "sharing starves alpha" story): alpha needs WIDTH
(a wide key, shared OR separate, lifts it; the cheap config's alpha 0.65 was a
width artifact at W=8). But buying that width THROUGH the shared table costs theta
(the wide encoder loses its bottleneck and overfits, an effect that grows with
training). Decoupling delivers alpha's width while keeping the encoder narrow, so
it gets both. The interesting dynamics now points at the THETA side -- why a wide
encoder degrades theta -- and the open question is EXPLANATION 2 vs 3 (static
expressivity vs learning dynamics), which Phase 3 (toy) and Phases 1-2 resolve.
Source: deep_irt/bench/outputs/gate_table.md.

## Phase 1 -- THE TRAJECTORY. What do the dynamics look like? (characterise, layer C)

Checkpoint training every K epochs; recover alpha and theta at each checkpoint;
plot recovery vs epoch for shared (narrow) and decoupled, across seeds.

- Hypotheses:
  - H1 (late alpha): alpha is recovered late; theta saturates first.
  - H2 (crowding): in the shared model, after theta saturates/overfits, alpha
    plateaus or DECAYS; in decoupled it keeps improving.
  - H3 (early bifurcation): the across-seed alpha variance in shared traces to
    sensitivity to which compromise the embedding falls into early; decoupling
    removes the bifurcation.
- Output: the dynamics as an object, not an endpoint. Extends the existing
  theta-overfit-vs-epoch curve to alpha.

## Phase 2 -- THE MECHANISM. Why does theta win the shared embedding? (gradient conflict, layer B)

Instrument the shared item embedding e_q. At each step decompose dL/de_q into the
theta-pathway gradient g_theta (through encoder -> theta -> loss, alpha head's use
of e_q detached) and the alpha-pathway gradient g_alpha (through the alpha head,
theta path detached). Track over training, per item and aggregated:

- magnitude ratio ||g_theta|| / ||g_alpha|| on e_q,
- cosine(g_theta, g_alpha),
- alignment of the NET update on e_q with g_theta (is the embedding captured by
  theta?).
- Hypotheses:
  - H4 (theta dominance): ||g_theta|| >> ||g_alpha|| (theta enters the loss at
    every occurrence; alpha enters multiplicatively, scaled by (theta - beta),
    and is low-Fisher).
  - H5 (conflict): cosine(g_theta, g_alpha) <= 0 (the pathways pull e_q in
    different directions).
  - H6 (decoupling removes it): the alpha-only table receives only g_alpha; its
    update is clean.
- Output: direct evidence that sharing couples them via gradient competition and
  theta wins.

### Phase 1+2 RESULT (2026-06-16): the real model already points to DYNAMICS

PHASE 1 (trajectory, static_k4, 3 seeds, theta_static & alpha vs epoch):
- SHARED-WIDE (emb=64) alpha PEAKS at 0.906 (ep50) then DECAYS to 0.787 (ep500).
  The architecture CAN express good alpha (it reaches it); continued training
  ABANDONS it. And at ep50 SHARED-WIDE is theta 0.959 AND alpha 0.906 -- BOTH
  high. So the gate's "shared never gets both" is a 150-epoch SNAPSHOT; the
  trajectory shows the both-high solution is VISITED then LEFT. This is evidence
  for DYNAMICS (explanation 3) over expressivity (2): the good solution is
  reachable and reached, then the trajectory leaves it.
- DECOUPLED alpha rises monotonically to 0.912 and HOLDS (no decay).
  SHARED-NARROW alpha rises slowly, plateaus ~0.73.
- theta: all three degrade past ep150 (the bare substrate has no LayerNorm;
  known), SHARED-WIDE fastest (-0.21 vs -0.12). DECOUPLED does NOT fix
  theta-overfit (degrades like NARROW) -- a SEPARATE problem (regularisation),
  distinct from the alpha protection decoupling provides.

PHASE 2 (gradient conflict on SHARED-WIDE; sum-check PASSED at machine precision):
- The theta-pathway (LSTM) gradient on the shared embedding GROWS ~28x over
  training (0.0006 -> 0.0170) while the alpha-head gradient stays flat (~0.001);
  the ratio reaches ~11x by ep500. Theta progressively dominates the shared
  embedding.
- cos(g_theta, g_alpha) ~ 0 at all epochs: the pathways are ORTHOGONAL, not
  anti-correlated. H5 (tug-of-war) is REFUTED; the effect is MAGNITUDE DOMINANCE
  in an orthogonal subspace, not a directional fight.

HONEST OPEN SUBTLETY (for the toy): orthogonality complicates the simple
"swamping" story -- if g_theta perp g_alpha, theta updates do not directly
overwrite alpha's embedding direction. Likely mechanism is relative scale: as the
theta-direction variation of E grows ~28x, alpha's fixed orthogonal signal
becomes a shrinking FRACTION of E's variation and the linear alpha readout cannot
isolate the orthogonal alpha-direction at low SNR. Decoupling localises it to the
item-embedding INPUT of alpha (decoupled alpha reads its own table and does NOT
decay, though it still reads the same overfitting state) -- so the decay is via
the shared E, not the shared state. The PRECISE mechanism is exactly what Phase 3
must pin down.

NET after Phases 0-2: explanation 1 (capacity) killed by the gate; explanation 2
(expressivity) refuted by the trajectory (alpha reachable then abandoned);
explanation 3 (dynamics) supported by the trajectory AND the gradient growth. The
toy formalises the mechanism. Source: deep_irt/bench/outputs/trajectory_table.md,
gradient_conflict_table.md.

## Phase 3 -- THE THEORY. Prove dynamics, not expressivity. (the rigorous core, layer A)

A minimal analytically-solvable surrogate, with the global-optimum-vs-gradient-
flow discriminator at its centre.

- Model: binary 2PL; LINEAR encoder = a per-item embedding e_q in R^d fed
  directly to linear heads alpha_q = f(a . e_q), beta_q = b . e_q; per-person
  ability theta_n learned; P = sigma(alpha_q (theta_n - beta_q)). Static ability.
  Population loss = expected NLL under true 2PL data. Continuous-time gradient
  flow.
- THE DISCRIMINATOR (kills explanation 2 vs 3):
  - (a) EXPRESSIVITY TEST: compute the GLOBAL minimiser of the population loss for
    the SHARED architecture with e_q wide enough to express both directions. Does
    it recover unbiased alpha? If NO, the limit is expressivity (explanation 2).
  - (b) DYNAMICS TEST: from standard init, where does gradient flow CONVERGE? If
    (a) recovers alpha but (b) does not, the limit is DYNAMICS (explanation 3),
    cleanly: the good optimum exists but gradient flow is captured by the
    theta-compromise.
- Derive: the coupled gradient-flow ODEs for (e_q, a, b, theta_n); the shared
  stationary e_q as a gradient-magnitude-weighted blend of the alpha-optimal and
  theta-optimal directions; show the blend biases alpha (shrinks it); show the
  decoupled system's e_q^alpha converges to the alpha-optimal direction. Start at
  2 items, d=1 (closed form), then generalise.
- FISHER BRIDGE: show the per-pathway gradient-magnitude ratio is governed by the
  Fisher information of each parameter, tying the statistical fact (alpha is
  low-information) to the dynamics fact (alpha loses the shared embedding). This
  is also how we separate our claim from the variational-bias literature (below).

### Phase 3 RESULT (2026-06-16): minimal model is CLEAN; the driver is AMORTIZED theta

Full derivation in docs/learning_dynamics_toy.md. The minimal point-estimate 2PL
with a FREE per-person theta and adequate rank (d>=2) does NOT exhibit the
coupling: the global optimum recovers unbiased alpha (gauge-fixed) AND gradient
flow reaches it on 8/8 seeds. No gap, no capture. The skeptic's outcome, and it
cleared three confounds:

- GAUGE: the raw "alpha biased low" magnitude is 100% the 2PL gauge
  (theta->s.theta+t, alpha->alpha/s); it collapses to 0.0000 once theta's scale
  is quotiented out. LESSON: report only gauge-fixed or RANK (Spearman) alpha;
  magnitude bias is a coordinate trap. Our empirical metric IS Spearman
  (gauge-invariant), so the empirical phenomenon survives this.
- EXPRESSIVITY: the d>=2 global optimum is unbiased (corr 1.0), confirming the
  trajectory's "alpha reachable". Refuted as the limit.
- FINITE-DATA MLE bias: real, O(1/reps), but IDENTICAL shared vs decoupled, so
  not the phenomenon decoupling fixes.

PROOF (Sec 2.4): for d>=2 with independent readouts the shared stationary code
forces G^alpha=0 AND G^beta=0 separately -> no trade-off. The biasing blend
appears only under collinearity (d=1, expressivity) or when theta competes for
the bottleneck (rung 5). FISHER BRIDGE sharpened: low Fisher SLOWS alpha but
BIASES it only when alpha must SHARE its code direction with a higher-Fisher
parameter; the minimal model gives theta its own free parameter, so no sharing,
no bias.

THE REFRAME (isolated ingredient): the coupling needs AMORTIZED theta read from
the SAME bottleneck that feeds the alpha head, so high-Fisher ability (accumulated
over the stream) captures the shared code from low-Fisher alpha. So the
phenomenon is NOT item-embedding sharing per se (a free table at d>=2 is fine) --
it is AMORTIZED-ABILITY CAPTURE. The real model's item code feeds the LSTM
(amortized theta) AND the alpha head, so the gate/trajectory/gradient results are
rung-5 manifestations; decoupling works by keeping the amortized-theta bottleneck
narrow while giving alpha its own capacity.

VARIATIONAL BOUNDARY clean: pure point estimate, no posterior anywhere; distinct
from GVEM/IW-GVEM posterior-approximation bias.

OPEN -- THE DECISIVE TEST (Sec 6 = Phase 3b): is the effect a true
POPULATION-LIMIT dynamics law, or finite-data + SGD interacting with the
bottleneck? Our empirical data is finite (N=800) and cannot distinguish. Build
rung 5 (amortized theta + a shared width-m bottleneck also feeding the alpha
head, matched-total-capacity, gauge-fixed) and test whether the alpha bias and
seed-variance PERSIST at reps -> inf. Persist => dynamics (strong claim). Vanish
=> finite-data + SGD (weaker but honest). This is the next and decisive math
step.

---

### Phase 3b RESULT (2026-06-16): the strong dynamics claim is REFUTED -- the bias is FINITE-DATA

Full derivation in docs/learning_dynamics_toy.md Sec 9. Rung 5 (amortized theta
read from a shared item code that also feeds the alpha head) returned the third,
weakest of the three named outcomes.

- DISCRIMINATOR clean at the POPULATION limit: shared global optimum unbiased
  (gauge-fixed bias -0.003, alpha rank 1.0, 8 seeds) AND gradient flow reaches
  it. No capture, no identifiability wall.
- POPULATION-LIMIT PERSISTENCE: the gauge-fixed alpha bias and the across-seed
  spread VANISH as reps -> inf (-3.33 at reps=1 -> -0.003 at reps=inf), and
  shared == decoupled at every reps (byte-identical at reps>=20). The bias is
  FINITE-DATA, and DECOUPLING does NOTHING in this toy.
- KEY INSIGHT (survives): for m>=3 with independent readouts the three pathways
  (theta, alpha, beta) zero SEPARATELY at the optimum. Amortization makes theta
  the FAST mode and alpha the SLOW mode (stiff flow, cond ~ I(theta)/I(alpha)),
  but a slow mode is not a biased mode while alpha owns a code direction. LOW
  FISHER SETS THE RATE, NOT THE ENDPOINT.
- MECHANISM of the finite-data bias: the amortizer input is a noisy encoding of
  theta* (errors-in-variables); through the bilinear z = alpha(theta - beta) it
  contaminates alpha. Both architectures share the same noisy amortizer input,
  so decoupling cannot help. Oracle control (clamp theta = theta*) recovers
  alpha exactly.

THE GAP (honest): the toy's effect is NOT the empirical effect. The real model's
alpha degradation IS removed by decoupling on finite data; this fixed-linear-pool
toy's bias is NOT decoupling-fixable. So the empirical decoupling-fix is STILL
unexplained; it depends on a LEARNED, theta-specific encoder pathway (which can
reallocate SNR on alpha's direction) that the fixed-pool toy omits. The clean
population-limit DYNAMICS LAW we hoped for does not exist in the tractable toys;
the next rung (a learned encoder) approaches real-model complexity and loses
tractability.

NET after Phases 0-3b: the contribution is NOT a population-limit learning-
dynamics law. It is (a) a robust empirical phenomenon (decoupling fixes alpha
recovery, architecture-independent, finite-data); (b) a rigorous ladder of what
it is NOT -- gauge, expressivity, population-limit dynamics, plain
errors-in-variables all ruled out; (c) the clean "low Fisher sets the rate not
the endpoint" result + the errors-in-variables finite-data mechanism; (d) a clean
variational boundary. Source: docs/learning_dynamics_toy.md Sec 9.

---

## Phase 4 -- VALIDATION. Real data + mechanism-level architecture independence.

- Real sparse bank: does the decoupling fix hold where the per-item alpha table
  can starve (the open caveat)? If not, the authorised fallback is a learned
  state-projection feeding only the alpha head.
- Mechanism architecture-independence: confirm the Phase-2 gradient-conflict
  signature (not just the Phase-0 outcome) reproduces on Transformer and DKVMN.

---

## The variational-bias boundary (the sharpest reviewer risk, addressed by design)

GVEM / IW-GVEM variational-IRT works document discrimination-estimate BIAS under
variational inference. Their bias is a property of the POSTERIOR APPROXIMATION (an
inference-quality / statistical effect). Ours is a property of GRADIENT-FLOW
CAPTURE of a shared POINT-ESTIMATE representation in an amortised model. Phase 3
draws the line explicitly: we exhibit the alpha bias in a NON-variational,
point-estimate model purely from gradient competition, which the variational
story does not cover. If the toy shows the bias without any variational
approximation, the boundary is clean.

## Decision tree (kill criteria, honest)

- Phase 0 says trivial capacity -> STOP. Report as a capacity result.
- Phase 0 passes, Phase 2 shows no theta-dominance / no conflict -> the gradient-
  competition mechanism is wrong; reframe (conditioning? identifiability?).
- Phase 3 (a) already fails to recover alpha (global optimum biased) -> it is
  EXPRESSIVITY not dynamics; a weaker, still-honest claim, reframe.
- Phase 3 (a) recovers and (b) does not -> DYNAMICS confirmed; the contribution
  stands.

## Sequencing and effort

1. Phase 0 (gate): cheap, decisive, reuses the bench. Days.
2. Phase 1 + Phase 2: cheap, instrument the existing model. Days.
3. Phase 3 (toy): the real work; the math. Weeks. The discriminator (global vs
   gradient-flow) is the single most important result.
4. Phase 4: after the mechanism is established.

Minimal credible package for a contribution: Phase 0 (trade-off real) + Phase 3
(dynamics not expressivity, via the discriminator) + Phase 2 (the mechanism
operates in the real model). Phases 1 and 4 strengthen and validate.

---

### Phase 3c RESULT (2026-06-16): rung 6 ALSO clean -- a GENERAL INVARIANT + the wrong-axis insight

Detail in docs/learning_dynamics_toy.md Sec 10. The learned-encoder rung returned
the third negative: decoupling does NOT bite. Shared == decoupled at every reps
and at the population limit (gauge-fixed bias ~-0.001 both, rank 1.0); no
lazy/rich gap (out_scale 0.1-20, width 1-16); discriminator clean (optimum
unbiased AND gradient flow reaches it).

GENERAL INVARIANT (the theorem the ladder produced): as long as the per-item
parameters are a FREE table that can hit p = p*, NO readout on top of it (fixed
pool or learned nonlinear encoder) biases the population optimum or makes
decoupling matter -- every gradient pull is linear in the residual r = p - p* and
all pulls vanish together at the reachable zero-residual optimum. This explains
why ALL toys (minimal / rung 5 / rung 6) are clean at the optimum and predicts
they always will be on the data and structure axes.

THE WRONG-AXIS INSIGHT (reframes "alpha learns faster when decoupled"): the
decoupling benefit is NOT an optimum / asymptotic effect (the invariant forbids
it) -- it is a TRAINING-TIME / EARLY-STOPPING / RATE effect. The toys polish to
the optimum and sweep DATA (reps), which structurally ERASES a transient effect.
The real model trains a FINITE number of epochs (not polished), so it lives in
the transient, where the rate asymmetry (alpha slow / theta fast, established)
bites. This UNIFIES the evidence: rate asymmetry + Phase 1 (alpha peak-then-decay
in shared) + Phase 2 (theta-grad grows 28x) = at finite training the shared code
is transiently dominated by fast theta, degrading slow alpha; decoupling gives
alpha its own code so its slow convergence is not disrupted. "Alpha learns faster
when decoupled" is a CONVERGENCE-RATE / early-stopping claim, and it is exactly
the regime real models train in.

NEXT (rung 7): the EARLY-STOPPING / training-time toy. Do NOT polish to the
optimum; sweep TRAINING TIME (epochs); plot decoupled-vs-shared alpha at each
budget; expect a decoupling advantage at finite training that VANISHES at
convergence -- the "alpha learns faster when decoupled" curve on the correct
(time) axis. This targets the transient the invariant exempts, so it should
finally be POSITIVE. Source: docs/learning_dynamics_toy.md Sec 10.

---

### GPCM RESULT (2026-06-16): 2PL conclusions transfer -- and GPCM reveals the finite-data decoupling advantage 2PL hid

Detail in docs/learning_dynamics_toy.md Sec 11. Built the GPCM (K=4) version of
every decisive check (gauge invariance to 3e-15, gradient sum-check to 1e-16).

TRANSFERS EXACTLY: the gauge artifact (a planted s=1.8 reads as -log 1.8 and
collapses to 0 gauge-fixed); the minimal-model clean optimum with gradient flow
reaching it; rung-5 finite-data bias vanishing at the population limit with
decoupling INERT on the data axis (byte-identical reps>=4).

TWO SHARPENINGS specific to K>2:
- The rank wall is at d < K, not d < 2 (2PL). GPCM has K pathway directions on
  the shared code, so the free optimum needs item-code rank >= K to zero the
  alpha pull (d=1/2/3 -> alpha rank 0.44/0.93/1.00; d>=4 exact). For the real K=4
  model the minimum item-code rank is 4.
- K WORSENS the theta-vs-alpha stiffness: I(theta)/I(alpha) climbs 1.03 (K=2) ->
  2.29 (K=4). I(alpha) rises absolutely with K (milder per-response MLE bias, the
  conjecture confirmed) but I(theta) rises faster.

THE NEW FINDING (reproduces the empirical effect): with a LEARNED encoder in
GPCM, decoupling gives a FINITE-DATA advantage on alpha SPEARMAN RANK -- the
metric that matters for tracking and the one the deep_irt benches use -- where
2PL rung 6 showed NONE:
  reps=1:  shared 0.724 -> decoupled 0.875  (+0.150, all 5 seeds positive)
  reps=4:  shared 0.898 -> decoupled 0.959  (+0.060)
  reps=200: ~0 ;  reps=inf: 0.
Decoupling raises alpha rank AND de-noises it (lower across-seed std),
finite-data-only, vanishing asymptotically. Predicted by the worse GPCM
stiffness: the stiffer shared flow lags more at a fixed budget and decoupling
relieves the lag on alpha's direction. On gauge-fixed MAGNITUDE bias there is
still no advantage (the invariant holds) -- the effect is purely on RANK.

BOTTOM LINE: GPCM (the real model's K=4 setting) REPRODUCES the empirical
decoupling advantage as a FINITE-DATA SAMPLE-EFFICIENCY effect on discrimination
RANK, driven by the Fisher-stiffness asymmetry amplified by K, vanishing at
infinite data. The tractable-toy reproduction + mechanism for the empirical
0.65 -> 0.93 (sign/shape/metric and mechanism, not exact magnitude). NOT an
asymptotic law; a low-data / many-category sample-efficiency advantage. Testing
GPCM rather than assuming 2PL transfer was decisive -- 2PL hid this. Source:
docs/learning_dynamics_toy.md Sec 11.

---

### K-sweep RESULT (2026-06-16): decoupling advantage robust + de-noising; K-scaling suggestive but noisy

Empirical K-sweep on the deep_irt benches (LSTM/GPCM, e8h32, state_alpha, exp,
5 seeds, 150ep, K=2..6, static), shared (alpha_emb=None) vs decoupled
(alpha_emb=64). delta_K = decoupled - shared alpha Spearman; analytical Fisher
stiffness I(theta)/I(alpha) by Monte Carlo over the data priors.

| K | shared a_sp | decoupled a_sp | delta_K | stiffness |
|---|---|---|---|---|
| 2 | 0.626 | 0.807 | +0.181 | 0.96 |
| 3 | 0.694 | 0.918 | +0.224 | 1.42 |
| 4 | 0.658 | 0.928 | +0.270 | 1.87 |
| 5 | 0.640 | 0.933 | +0.293 | 2.34 |
| 6 | 0.711 | 0.943 | +0.232 | 2.80 |

ROBUST: decoupled beats shared on ALL 25 runs (5 seeds x 5 K); no theta tax at
any K; decoupled alpha std SHRINKS monotonically (0.049 -> 0.020) -- decoupling
de-noises; decoupled alpha itself is MONOTONE in K (0.807 -> 0.943).

K-SCALING: delta_K rises K=2..5 (0.181 -> 0.293) tracking stiffness
(Spearman(stiffness, delta)=0.70), then DIPS at K=6 (0.232). STRICT monotonicity
NOT confirmed. The K=6 dip is in the SHARED arm (its alpha is high-variance,
std up to 0.16; seeds happened to recover better at K=6), not a decoupled
ceiling. At 5 seeds the delta_K K-trend is UNDER-POWERED.

RECONCILES WITH PRIOR-ART: the classical result (Ikeda 2026: "no monotonic
relationship between K and relative estimability of alpha vs theta") matches our
SHARED arm's messy K-behavior; the DECOUPLED arm is clean. Classical IRT has no
decoupling, so it cannot see the decoupling dimension we add.

HONEST CLAIM: solid = "decoupling robustly improves AND de-noises discrimination
recovery across K, no theta tax" (25/25). suggestive = "the advantage scales with
K / Fisher stiffness" (Spearman 0.70, clean K=2..5, K=6 dip). To firm the K-law:
10-20 seeds to tame shared-arm variance and resolve the K=6 dip. Source:
deep_irt/bench/outputs/ksweep_table.md.

---

### K-sweep EXTENDED (K=2..11, 10 seeds) -- supersedes the 5-seed run above

Re-ran K=2..11 at 10 seeds (real ordinal scales are commonly 7/9/10/11). delta_K
= decoupled - shared alpha Spearman; stiffness = analytical I(theta)/I(alpha).

| K        | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|
| decoup a | .798 | .910 | .926 | .937 | .942 | .945 | .940 | .952 | .949 | .957 |
| shared a | .682 | .734 | .734 | .711 | .755 | .721 | .689 | .651 | .668 | .687 |
| delta_K  | .116 | .175 | .192 | .226 | .187 | .224 | .251 | .300 | .282 | .270 |
| stiffness| 0.96 | 1.42 | 1.87 | 2.34 | 2.80 | 3.28 | 3.75 | 4.23 | 4.75 | 5.22 |

- ROBUST: decoupled wins 9/10 seeds at K=2 and 10/10 at K=3..11; no theta tax at
  any K; decoupled std uniformly low (0.005-0.042) -- de-noising holds to K=11.
- STIFFNESS TRACKING: Spearman 0.891, Pearson 0.907 over K=2..11 (vs 0.70 on
  K=2..5). The decoupling advantage tracks the Fisher stiffness across the full
  practical range of real assessment scales.
- SHAPE: RISE-THEN-SATURATE. delta_K grows K=2..9 (peak +0.300) then plateaus;
  the decoupled arm CEILINGS ~0.95 by K=7, so the plateau is a ceiling effect,
  NOT the arms converging (shared stays noisy 0.65-0.76). The K=6 dip persists
  (shared-arm variance), so it is NOT strictly monotone.
- RECONCILES WITH IKEDA 2026: the shared arm IS K-messy (matches "no monotonic
  K-vs-recovery"); the decoupled arm is clean; we add the decoupling dimension
  classical IRT cannot see.
- WORKSHOP CLAIM (now well-supported): decoupling robustly improves AND de-noises
  discrimination recovery across K=2..11 with no ability tax, the advantage
  tracking the ability-vs-discrimination Fisher stiffness (Spearman 0.89) and
  growing with K until the decoupled arm saturates near ceiling. Finite-data, the
  regime real assessment lives in. Source: deep_irt/bench/outputs/ksweep_table.md.

---

### Phase 3d RESULT (rung 7, 2026-06-16): the transient bites -- convergence-RATE win; mechanism = Hessian conditioning

Detail in docs/learning_dynamics_toy.md Sec 12. The WIN scenario, qualified
precisely. At the population optimum decoupling is inert (free-table invariant),
but in the TRANSIENT the shared-code flow is STIFF and alpha (low-Fisher) resolves
LAST; decoupling gives alpha an uncontested code that converges at its own rate.

- RATE ADVANTAGE (on RANK): steps for alpha Spearman to reach 0.95 -- shared
  1025+-1006, decoupled 457+-123. ~2.2x faster, ~8x more reliable.
- VANISHES AT CONVERGENCE: rank gap -> +0.001 by step 8000; both reach rank 1.0.
  A transient advantage that closes = a pure RATE effect (consistent with the
  invariant).
- RANK not BIAS: steps to |logbias|<0.10 identical (2252 vs 2301). The advantage
  is on the ranking metric the tracking model is judged on.
- OWNERSHIP control: warm-starting decoupled E_alpha = E_theta's init
  (byte-identical start) PRESERVES the advantage (491 vs 1025). Structural
  ownership of an uncontested code, not an init/cold-start artifact.
- MECHANISM = HESSIAN CONDITIONING (resolves the Phase-2 orthogonality puzzle):
  the shared-code pathway gradients ARE near-orthogonal (cos 0.05-0.20, matching
  Phase 2), but orthogonal gradients do not imply a null rate effect -- the lag
  comes from the EIGENVALUE SPREAD of the shared-code Hessian block
  (~ I(theta)/I(alpha) = 2.18 at K=4), a curvature property. Orthogonality kills
  only a gradient-CONFLICT mechanism; the actual mechanism is conditioning.
- OPTIMIZER interaction: largest under plain GD (pays the stiffness in full);
  COMPRESSES under Adam (2.2x -> 1.6x), whose preconditioner partially cancels
  the stiffness. The real model uses Adam, so the empirical effect is the (still
  real) compressed version. Implication: a per-parameter-preconditioned optimizer
  partially substitutes for decoupling.

THE UNIFIED PICTURE (study converged): the endpoint is INVARIANT (decoupling does
NOT change WHERE training converges -- proven). The decoupling benefit lives in
not-fully-resolved regimes and has TWO faces, both governed by the Fisher
stiffness I(theta)/I(alpha) (which grows with K), both on RANK:
  (1) FINITE DATA -> a sample-efficiency advantage (GPCM toy Sec 11; vanishes as
      data -> inf). [N-sweep quantifies the data-axis rate, the 11 curves.]
  (2) FINITE TRAINING -> a convergence-rate advantage (rung 7; vanishes as
      training -> inf).
Both vanish only at the JOINT (infinite data + infinite training) limit. Real
models live at finite-data + finite-training, so they get the benefit. Mechanism:
Hessian conditioning, not gradient conflict.

UNFINISHED: the direct K=2-vs-K=4 GD rate sweep (rate widens with K) ran too slow
on CPU and was stopped; the prediction is analytical and the fixed-budget
K-dependence is already shown by the K-sweep (rho=0.89). Source:
docs/learning_dynamics_toy.md Sec 12.

---

### N-sweep RESULT (2026-06-16): at a fixed budget the gap does NOT narrow with data -- it is RATE-limited (confirms rung 7)

Empirical N-data sweep (N=100..1600 x K=2..11, 5 seeds, 150ep). delta_K(N) =
decoupled - shared alpha Spearman. At the fixed 150-epoch budget the decoupling
advantage does NOT narrow with more data -- it is flat-to-WIDENING (K=2:
0.03->0.12->0.15->0.18->0.23 over N=100..1600; K=8: 0.23->0.36; K=9: 0.25->0.38).
Higher-K curves sit higher (stiffness).

This CONFIRMS rung 7 empirically: at a fixed budget the advantage is RATE-limited
(training steps), not data-limited. More learners with the same 150 steps leaves
the model more under-trained relative to the data; the stiff shared flow (slow
alpha) cannot catch up, while decoupled exploits the extra data faster, so the
gap persists or widens. The toy's "gap narrows with data" holds only AT
CONVERGENCE; the real model at 150ep is in the rate-limited regime, so the two
faces (data / training) are ENTANGLED at a fixed budget and the RATE face
dominates. Figures: ksweep_plot.png, ndata_sweep_plot.png. Source:
ndata_sweep_results.json.

---

## STUDY STATUS: COMPLETE / CONVERGED (2026-06-16)

The analytical ladder bottomed out at rung 7; the mechanism is pinned (Hessian
conditioning ~ I(theta)/I(alpha), grows with K); the empirical benches (gate,
trajectory, gradient, K-sweep, N-sweep) and the toys (minimal / rung 5 / 6 / 7,
GPCM) all agree. FINAL CONTRIBUTION: decoupling the discrimination representation
in amortized neural IRT buys a Fisher-conditioning-governed CONVERGENCE-RATE
advantage on discrimination RANK recovery -- scaling with K, dominant at any
practical training budget, with NO endpoint cost (the optimum is invariant). A
workshop-shaped methods result, prior-art-cleared, variational-clean, reproduced
in tractable toys AND empirically. NEXT: write it up as a workshop note (figures
in hand) OR bank and return to the thesis program (the engine = decoupled
deep-irt was settled long ago; continual tracking / content channel / bank-growth
transfer await). Do NOT spawn further rungs.

---

## Presentation artifacts (methodology survey, 2026-06-16)

Deep-research survey of the learning-dynamics literature (Saxe deep-linear; Pesme
/ Jacot saddle-to-saddle; Atanasov silent-alignment; Chizat lazy-vs-rich; PCGrad;
103 agents, 21 primary sources). The field's standard repertoire for a "staged /
heterogeneous / competitive parameter learning" story, mapped to what we already
have:

- R1 -- Saxe Fig 3 template: per-parameter recovery vs training step, theta and
  alpha, shared vs decoupled; expect staged theta-fast / alpha-slow in shared.
  WE HAVE THIS (Phase 1 trajectory).
- R2 -- the canonical scalar law learning-time ~ 1/(information/curvature),
  t = O(tau/s): a learning-speed-vs-Fisher scatter, high-Fisher theta fast,
  low-Fisher alpha slow. This is the SURVIVING result ("low Fisher sets the rate
  not the endpoint"); the toy gives the stiffness cond ~ I(theta)/I(alpha).
- R3 -- population vs finite-data: THE field-standard control for "true dynamics
  vs finite-sample." WE RAN IT (Phase 3b reps sweep), and it returned a clean
  honest NEGATIVE on the bias (finite-sample). The survey confirms running R3 was
  the correct, load-bearing rigorous move, not a methodological miss.
- R4 -- silent-alignment / gradient-cosine vs time ("direction captured early"):
  WE HAVE A VERSION (Phase 2 gradient decomposition: theta-gradient grows 28x,
  orthogonal to alpha).
- R5 -- controls if we pursue the learned-encoder rung: lazy-vs-rich init-scale
  sweep + balanced-vs-coupled init, to show capture is a rich feature-learning
  effect not a lazy/linearized artifact.

IMPLICATION: the honest contribution is presentable with canonical artifacts we
MOSTLY ALREADY HAVE -- the RATE asymmetry (R1 + R2, genuine dynamics) + the
population-vs-finite control (R3, the honest negative on the bias) + the
gradient-alignment diagnostic (R4). Framing: "the learning dynamics of amortized
IRT recovery" (rate asymmetry + finite-sample bias + empirical decoupling fix),
NOT a gradient-capture law.
