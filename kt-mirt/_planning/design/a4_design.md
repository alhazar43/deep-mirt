# A4 pre-registered design: per-KC growth posture matrix (G2)

Status: PRE-REGISTERED, v1.1, 2026-07-18. v1 was registered earlier the
same day and revised after two independent pre-run reviews; NO RUN HAS
STARTED, so threshold changes here are within the registration rule
(thresholds may change now, never after runs begin; several were
tightened, none loosened). This document freezes the A4 estimands,
estimators, generator, battery, thresholds, and kill conditions before
any run. Mechanism fixes after runs begin are allowed only within the
revision budget in section 5.6, and every revision is logged in
`_planning/LEDGER.md`. Review dispositions are cataloged in section 10;
decisions escalated to the program lead are in section 11.
Inputs: PLAN.md, THINKING.md (2026-07-17 posture directive), LEDGER.md
(stage-0 triage), vendor_report.md, triage/triage_report.md,
research/{avenue_map, growth-methodology, trajectory-dynamics-archaeology,
qmirt-archaeology, interpretability_critiques_read}.md, plus the two
pre-run review reports (findings R1-*, R2-* in section 10).

## 0. Scope and claim boundary

Goal G2: a per-learner digital twin shows ability growth or decline
beyond noise, read on a trustworthy scale. A4 decides whether growth is
detectable and readable per KC; A5 (frozen-anchor twin) builds on it
only if A4 passes on at least one bed.

Three postures, per the user directive. Posture disagreement is itself
a diagnostic and is reported, never averaged away.

- ACTIVE. The model imposes growth structure. Characteristic error is
  fabricated growth (Lemma 1, free-asymptote; Lemma 3, gain-form misfit
  laundering). Certified by silence on no-growth twins AND by bounded
  behavior under shape misfit and saturation (CG1 family; the
  mismatched-generator arm the qmirt record flagged and never ran is
  run here, section 3.2).
- PASSIVE. An unconstrained tracker (and model-free curves) read against
  noise. Characteristic errors are missed growth and the reconstruction
  artifact (ability moving against the observed response, the Deep-IRT
  admission). Direction audit mandatory.
- MIXED. Existence gate first, then a parametric rate whose growth
  channel is testable against zero.

Boundaries, fixed for v1:

- Measurement-validity claims only. No pedagogical decision-utility
  claim, stated or implied (Khalid et al. 2025 scope rule).
- No cross-KC transfer term anywhere in A4 (that is A1). No
  multidimensional theta, no rotation freedom; per-KC attribution uses
  the pure-anchor discipline where it is claimed.
- Both real beds are monotone-rising in triage AT THE POPULATION LEVEL,
  so v1 has no decay or forgetting channel (Lemma 2 policy, rho pinned
  at 1). Population monotonicity does not establish that no individual
  learner-KC slice declines; the pin therefore structurally bars ACT
  from ever representing an individual decline, while the passive and
  mixed machinery are free-signed (displacement, blockwise profiles,
  and r all carry sign). Consequence, stated rather than buried: Tier-3
  posture comparison is partitioned by sign (section 5.5); on
  decline-side slices ACT abstains by construction and only PAS/MIX
  vote. This is a recorded scope limit with a named comparability cost;
  whether v1 must instead carry a decline-capable ACT variant is an
  open ruling (section 11, ruling 2).
- Binary responses only in v1 (both real beds are binary; the K>2
  decoders stay unused here).
- No state-inert measurement-item class in v1; every interaction is
  both practice and measurement. On synthetic twins the generator's
  ground truth carries the separating load the qmirt ref-inert device
  carried. On real beds no such device exists in v1: existence-of-
  dynamics is read there conditional on the frozen bank absorbing
  item-selection-within-slice effects UP TO DIFFICULTY; item-selection
  effects beyond difficulty are a stated, unresolved v1 limitation of
  every real-bed claim. The M0-bootstrap twin (battery arm 2), which
  preserves the realized item schedule, is the partial in-situ control.

## 1. Estimands

Notation. Bed = a dataset with learners i, interactions ordered per
learner t = 1..T_i, items j carrying KC tag sets K(j) under the bed's
stated Q-matrix expansion policy (section 6). The slice (i,c) is the
ordered subsequence of learner i's interactions on items tagged c. The
opportunity index n = 1, 2, ... counts a learner's interactions within
the slice. Response y in {0,1} (KDD: Correct First Attempt per step;
EdNet: chosen option equals correct_answer). theta is always read
against the frozen calibrated bank of section 2.1 (the trustworthy
scale); rates are additionally affine-invariant (section 2.4).

Density strata, fixed: D0 = slices with 4-5 opportunities, D1 = 6-9,
D2 = 10-19, D3 = 20+. Slices with fewer than 4 opportunities enter
population curves only. Per-slice (per-learner) estimands are defined
only on D2+.

PASSIVE estimands.

- E-P1 (population curve, per KC): the opportunity-indexed first-attempt
  success curve p_c(n) = mean over learners of y at opportunity n on KC
  c, n = 1..20, and its summaries, the early rise Delta_c = p_c(10) -
  p_c(1) and the AFM-style pooled slope beta_c from
  logit P(y=1) = theta_ic + beta_c (n-1) - b_j with per-slice constants
  theta_ic and frozen item difficulties b_j. Population existence claim
  = beta_c > 0 beyond the permutation null.
- E-P2 (existence verdicts): binary verdicts, per KC (pooled over that
  KC's slices), per bed (pooled over KCs), and per slice on D2+, that a
  dynamic-ability model beats the constant-ability null on held-out
  interactions beyond the permutation null (gate spec in 2.2).
- E-P3 (tracker displacement, per slice): Delta theta_ic = mean of the
  tracker's causal theta_hat_c over the last quarter of the slice minus
  the mean over the first quarter. Free-signed, shape-free. Per-KC
  profile = mean of Delta theta_ic over slices of c.

ACTIVE estimands.

- E-A1 (per-KC gain, population): g_c, the practice-gated gain
  coefficient of the pre-registered transition (2.3), reported both in
  state units and as the model-implied first-attempt success rise over
  opportunities 1..10 on a median-difficulty item (score scale, so it
  is directly comparable to E-P1). Whether an E-A1 read on real data
  counts as ACT "firing" is operationally defined in RB-A (5.2).
- E-A2 (per-learner rate multiplier): lambda_i, amortized, ACT-P1 only.

MIXED estimands.

- E-M1 (KC rate): r_c from the bounded-exponential family
  theta(n) = m_c - (m_c - theta0_ic) exp(-r_c (n-1)), fit per KC with
  shared (m_c, r_c) and free per-slice theta0_ic, only on KCs whose
  existence gate passed.
- E-M2 (per-learner rate): r_ic from the same family with per-slice
  (theta0, r) and shared m_c, only on gate-passing D2+ slices.
- E-M3 (family-misfit flag, per KC): fires when the shape-agnostic
  blockwise model beats the bounded-exponential on held-out data
  (2.4); a fired flag withholds r and the reported object becomes
  "growth exists, shape outside the family".

Claim tiers (the G2 ladder): Tier 1 = population per-KC growth beyond
noise (E-P1/E-P2). Tier 2 = KC-level rates on the trustworthy scale
(E-M1). Tier 3 = per-learner rates (E-M2, E-A2). G2 minimal success is
Tier 1 on one real bed under the full battery; full success is Tier 3.

## 2. Estimators and mapping onto the vendored core

### 2.1 Shared measurement layer: the frozen bank

All theta-referenced estimators read through item difficulties frozen
before any dynamics fit, per the anchoring posture. Because every
posture reads through this one bank, the bank is itself audited
(battery arm 10, RB0): it is the single shared artifact whose
undetected bias would propagate uniformly into every verdict.

- Split each real bed's learners at random into a calibration cohort
  (50%) and an analysis cohort (50%). Random learner split supplies the
  ability spread the qmirt calibration mechanism requires.
- Calibration model: item difficulties with learner intercepts and a
  per-KC BLOCKWISE opportunity profile,
  logit P(y=1) = a_i + u_c(B(n)) - b_j,
  where u_c is a free per-KC piecewise-constant profile over the
  pre-registered opportunity blocks {1-3, 4-7, 8-15, 16+} (the same
  block edges as M1b). Freeze b_j; discard a_i and u_c. The growth
  profile is included so difficulty is not confounded with
  when-in-learning an item tends to appear; it is BLOCKWISE rather
  than linear (v1 had a linear gamma_c) because this design itself
  asserts growth need not be linear (SYN-NS), and a linear-only
  correction would leave nonlinear trend in the residual, biasing b_j
  wherever item difficulty covaries with curriculum position (review
  finding R1-B4). Freezing only b_j means no growth information leaks
  into the analysis cohort (disjoint learners).
- Identification under sparsity (KDD). The KDD step bank is extremely
  sparse: 1,313,276 distinct steps over 8.9M rows, median item
  frequency 1, 74.3% of steps occurring exactly once in the whole file
  (measured directly during review, R2-B1). A free b_j per step is not
  identified. The bank is therefore hierarchical:
  b_j = b_H + e_P + d_j, with H the step's Problem Hierarchy, P its
  problem (Problem Hierarchy + Problem Name), and j the step;
  b_H ~ N(0, 1.5^2), e_P ~ N(0, 1.0^2), and the step offset d_j ~
  N(0, 0.5^2) fit ONLY for steps with >= 20 calibration-cohort
  responses (otherwise d_j = 0). All scales in logits; estimation is
  MAP. Every parameter has a proper prior, so no b diverges on
  all-correct or all-incorrect exposure. Item = step is RETAINED as
  the estimand (section 6); the hierarchy identifies the bank without
  redefining the item (the alternative, item = problem, is an estimand
  change escalated as ruling 1, section 11). EdNet: 13,169 items,
  dense; a single-level b_j ~ N(0, 1.5^2) suffices, same code path.
- Optimizer. This fit is NOT the slice-based batched Newton of 2.2 (a
  dense Hessian over ~1.3M difficulty parameters is unstorable,
  R2-B2): it is Adam over parameter embeddings, minibatched over rows,
  on GPU; convergence = relative NLL change < 1e-4 over 3 consecutive
  epochs plus a parameter-drift check. Costed in section 7.
- 1PL only in v1. Rationale: item location recovers robustly under the
  qmirt record while discrimination is the fragile parameter under
  joint calibration; the gate and the rate need only a
  difficulty-adjusted margin. 2PL is a pre-registered extension, not
  v1.
- Analysis-cohort read rule. A step unseen in calibration whose
  problem WAS calibrated reads b_H + e_P (zero step offset); this
  retains most row mass despite the singleton-heavy bank. Interactions
  on items whose problem (KDD) or item (EdNet) is entirely unseen in
  calibration are excluded from gate/rate likelihoods but still
  increment opportunity counters.
- Bank-robustness audit (battery arm 10). (i) Synthetic: calibrated b
  must rank-correlate >= 0.9 with the generator's true b on SYN-KG-KDD
  AND on SYN-NS (the misfit twin, exactly where a growth-misspecified
  calibration would fail). (ii) Real beds: the bank is refit under
  three growth specs (no-growth, linear gamma_c, blockwise u_c);
  RB0 (5.2) requires pairwise rank correlation >= 0.95 of calibrated
  difficulties (computed at the finest fitted level, restricted to
  units with >= 50 calibration responses). Note the permutation null
  does NOT protect against a difficulty-vs-curriculum-position
  confound (permutation destroys the item-position coupling in the
  null replicates but not in the observed statistic), which is why
  this audit exists.
- The saturation flag (used in 5.2) is a RAW calibration-cohort per-KC
  correct rate, model-free; it involves no fitted parameter, so it is
  not entangled with the bank fit beyond sharing the cohort, which is
  deliberate (no selection on analysis data; learner-disjoint cohorts
  keep analysis inference clean). As a diagnostic, flags are
  recomputed on the analysis cohort and agreement is reported (R1-I13;
  reported, not gated).
- All gates, rates, trackers, and the active model run on the analysis
  cohort only.

### 2.2 PASSIVE estimators

PAS-C (model-free curves). Per-KC opportunity curves and Delta_c
exactly as the triage computed them, now on the analysis cohort with
learner-clustered uncertainty. Known limitation, stated up front:
curves confound ability growth with curriculum-driven item-difficulty
drift; the gate (which conditions on frozen b_j) is therefore the
primary passive read and PAS-C is descriptive corroboration.

PAS-G (existence gate). The two models, on learner-KC slices, both
using frozen b_j:

- M0 (constant-ability null): logit P(y=1 at opportunity n, item j)
  = theta_ic - b_j, one free constant theta_ic per slice.
- M1 (dynamic alternative), a two-member pre-registered family:
  - M1a (linear trend): M0 plus beta (n-1), with beta = beta_c shared
    across slices within KC (pooled gate) or beta = delta_ic per slice
    (per-slice gate, D2+ only).
  - M1b (blockwise): M0 with theta_ic replaced by theta_ic + u_c(B(n)),
    a per-KC free profile over opportunity blocks with pre-registered
    edges {1-3, 4-7, 8-15, 16+} (the bed's density quartiles). Shape-
    robust, free-signed, catches non-standard and non-monotone growth.

Numerical safeguards, pre-registered (R2-B3): all slice-level logistic
fits (M0, M1a, M1b, and the MIX-L rate stage) are PENALIZED, Gaussian
prior N(0, 2.0^2 logits) on theta_ic, beta/delta, block offsets, and
the rate stage's (theta0, m) deviations, fit by damped, bounded Newton
(step-norm clamp 1.0, max 25 iterations, backtracking on any NLL
increase). Quasi-complete separation is ROUTINE here, not exotic: a
10-opportunity slice from a p = 0.85 KC is all-correct with
probability about 0.2, and the per-KC saturation screen cannot catch
per-slice separation. The penalty guarantees finite MAP estimates and
finite held-out log-likelihoods on separated slices; it is shared by
M0 and M1 (no asymmetric advantage), and the permutation null runs
under the identical penalized machinery, so the gate statistic's
calibration is preserved. A unit test asserts finite parameters and
finite held-out NLL on an all-correct and an all-incorrect slice.

Evaluation: within each slice, fit on odd opportunity indices, evaluate
held-out log-likelihood on even indices (interpolative split). The gate
statistic is the summed held-out improvement of M1 over M0, with the
M1a/M1b selection handled by taking the max and permuting the max
statistic. Pooling levels: per KC, per bed, per slice (D2+).
Significance comes from the permutation null (section 4.1); per-KC
discoveries are controlled at BH-FDR q = 0.05 within bed; the bed-level
pooled test uses alpha 0.01. The interpolative (odd/even) rather than
forecast (early/late) split is a deliberate choice: a forecast split
conflates trend detection with extrapolation and is biased toward the
dynamic model whenever growth continues; interpolation tests whether
time-alignment of ability explains held-out responses.

Dependence note on the per-KC family (R1-I1): multi-KC interactions
enter every tagged KC's slice, so per-KC gate tests are positively
dependent (KTracedSkills decoupling 0.80, EdNet mean arity 2.18; the
dependency is measured, not hypothetical). Two protections, both
pre-registered: (i) BH's realized false-discovery behavior is itself
certified on SYN-NG at BOTH density profiles (CG2), and the
EdNet-matched twin carries matched multi-tag arity, so it is the
dependence stress; (ii) every per-KC discovery list is additionally
reported under Benjamini-Yekutieli q = 0.05 (valid under arbitrary
dependence), and discoveries that vanish under BY are flagged
dependence-sensitive in the report. BH remains the primary control;
the BY read is report content, not a gate.

Validation regimes, fixed (R1-I11): the interpolative odd/even split
governs all SLICE-BASED model comparisons (M0 vs M1; bounded-
exponential vs blockwise), which is where the extrapolation-bias
argument bites. NEURAL arms (PAS-N, ACT) are trained on a seeded 80%
learner split of the analysis cohort, and every neural evaluation
statistic (RB2 trained-vs-frozen NLL, ACT NLL reads) is causal
next-step forecast NLL on the held-out 20% of learners. A per-position
odd/even loss mask is ill-defined for neural arms on multi-tag beds (a
position can be odd in one tagged KC's slice and even in another's).
No statistic in this design ever compares a neural NLL with a
slice-based NLL, so the two regimes never meet inside any comparison;
each comparison is apples-to-apples within its regime (judgment call,
section 9).

PAS-N (neural passive trackers). Two configs, both trained on
prediction NLL, no growth structure, per-KC theta heads read against
the frozen bank scale (the decoder consumes frozen b_j; only the
ability path trains):

- PAS-N1 (shared-state tracker, the field-representative object): stock
  encoder (lstm primary; dkvmn as the pre-registered alternate) plus a
  per-KC ability head reading `state_for_prediction`, output dimension
  C, gathered by the current item's KC ids (multi-tag gather = mean
  over the item's tagged slots, section 2.5). This is the config Ding
  and Larson predict will fail contamination; running it is the point.
- PAS-N2 (factorized per-KC tracker, contamination-proof by
  construction): the same shared LSTM cell applied independently to
  each (i,c) slice's subsequence, so KC c's state sees only KC c's
  interactions. Order-invariant to cross-KC interleaving and
  contamination-free by construction. Cost note (R2-M1): the
  factorized pass consumes each interaction once per tagged KC, so its
  token count is 1.0-1.8x the bed on KDD and about 2.2x on EdNet, not
  exactly bed-sized; priced accordingly in section 7.

The passive growth read from either tracker is E-P3 (displacement),
judged only through the battery (section 4). Tracker reads never
substitute for PAS-G; they are the digital-twin leg the audits target,
and the A5 go/no-go input.

### 2.3 ACTIVE estimator

ACT (minimal gain-gated growth model), honoring the identification
lemmas exactly:

- State: per-learner per-KC scalar z_ic,t on the frozen bank's theta
  scale.
- Transition (the only state-moving term):
  z_ic,t+1 = z_ic,t + lambda_i g_c (M - z_ic,t)+ 1[item at t is tagged c].
  Practice-gated (moves only when c is practiced; Lemma 1 rule: no
  always-on channel). Ceiling-gated ((M - z)+ diminishing returns;
  Lemma 3 rule: gain must be gap-scaled). Response-blind (the
  transition never reads y; the qmirt R9 rule, which also makes the
  active model AFM-flavored rather than PFA-flavored in v1). Note the
  transition makes each z_ic trajectory, as a function of its own
  opportunity index, invariant to cross-KC interleaving by
  construction; order sensitivity can enter ONLY through the amortized
  recognition inputs, which is exactly what battery arm 5 stresses
  (CG9-ACT).
- Pins: mu = 0 (no mean-reversion target exists; the OU channel is
  absent). rho = 1 (no decay; both beds monotone in triage; the
  individual-decline consequence is recorded in section 0). gamma-type
  per-learner transfer multipliers do not exist here by construction
  (no transfer term at all).
- Ceiling M: one global fitted scalar shared by all KCs and learners,
  initialized at the 95th percentile of calibrated b_j plus 2 (an
  arbitrary anchoring constant, now flagged as a judgment call,
  section 9 item 10); if CG1 (active silence) fails with fitted M, the
  pre-registered fallback is M fixed at that initialization.
- Per-learner quantities: never free parameters (Gate B). ACT-P0
  (primary): lambda_i pinned at 1; initial state z0_ic = u_i + v_c with
  u_i amortized by a recognition network over the learner's full
  conditioning window and v_c a free population per-KC offset. ACT-P1
  (extension): adds amortized scalar lambda_i, full-window only. Both
  variants must clear the no-growth twin; real-data claims use the
  richest variant that stayed silent.
- Readout and loss: frozen 1PL bank, Bernoulli forecast NLL,
  two-stage training (calibrate and freeze the bank, then fit dynamics
  and recognition heads). Training/evaluation split: the neural
  regime of 2.2 (80/20 learner split, forecast NLL on held-out
  learners).
- Fabrication read (the CG1 statistic): on a no-growth twin, the model-
  implied score change over 10 opportunities, population mean and p95
  per-learner (the qmirt lesson that the population mean alone hides
  per-learner fabrication).
- Twin coverage (R1-B1, R1-I10): ACT is certified on ALL FOUR twins,
  not only SYN-NG. SYN-KG is its positive control (CG1a: the firing
  definition must trigger on real growth). SYN-NS is the program's
  mismatched-generator robustness arm, finally run (CG1b: silence on
  the embedded silent-KC subset, detection under shape misfit, bounded
  overshoot; this is Lemma 3's laundering channel measured directly).
  SYN-SAT is the ceiling-fabrication probe (CG1c: no confident gains
  manufactured from saturated observations).
- Reporting rule under saturation (pre-registered): ACT latent-scale
  gains g_c are reported only on unsaturated KCs; on saturation-
  flagged KCs only score-scale implied changes are reported, and the
  ACT verdict there follows the saturation flag ("insufficient dynamic
  range for gain calibration"), mirroring CG6.
- Real-bed firing: RB-A (5.2) defines operationally when an E-A1 read
  counts as ACT firing; without RB-A no disagreement-matrix row
  involving "active fires" is evaluable (R1-B2).

### 2.4 MIXED estimator

MIX-L (gate-then-rate ladder):

1. Existence gate = PAS-G verbatim (shared machinery, one
   implementation, ONE result). Structural consequence, stated
   plainly (R1-B3): at the existence tier, PASSIVE and MIXED do not
   cast independent votes; "passive fires, mixed flat at existence" is
   impossible by construction. This sharing is deliberate (two
   implementations of the same gate would let implementation variance
   masquerade as posture disagreement), and its price, one fewer
   independent Tier-1 vote, is carried into the disagreement matrix
   (5.5), which now names its actually-independent inputs.
2. On gate-passing KCs (and gate-passing D2+ slices for E-M2), fit the
   bounded-exponential theta(n) = m - (m - theta0) exp(-r (n-1)) by
   Bernoulli NLL against frozen b_j, parameter sharing as in E-M1/E-M2,
   under the penalized bounded-step machinery of 2.2.
   r >= 0 is not imposed; r is free and its sign is reported.
3. Family-misfit flag E-M3: fit both bounded-exponential and M1b
   (blockwise) on odd indices, compare held-out even-index
   log-likelihood per KC; if blockwise beats bounded-exponential
   (paired bootstrap over slices, one-sided alpha 0.05), the flag fires
   and r is withheld for that KC.
4. Growth channel testable against zero: the family nests no-growth at
   r = 0 (equivalently m = theta0), so the rate stage carries its own
   within-family test in addition to the gate.

Affine-invariance argument, stated: if the ability scale is remapped
theta' = a theta + b with a > 0 (the encoder's or the bank's arbitrary
gauge), then m' = a m + b and theta0' = a theta0 + b reproduce the same
trajectory with the SAME r. The bounded-exponential family is closed
under affine maps with r invariant, so the rate estimand cannot be
biased by scale gauge; displacements and asymptotes are gauge-dependent
and are therefore only ever reported on the frozen anchored scale,
while r is the magnitude estimand of choice. The frozen bank pins the
gauge anyway; invariance of r is the second, independent protection.

### 2.5 Attachment-point mapping and new modules

Mapping onto the five per-KC attachment points of vendor_report.md:

| Attachment point | Use in A4 |
|---|---|
| #1 ability readout (`theta_proj`, `state_for_prediction`) | PAS-N1's per-KC head is a Linear(hidden_dim, C) reading `state_for_prediction`, gathered by KC id (the vendor report's named pattern, no backbone change). PAS-N2 reuses the stock scalar head per slice. ACT's recognition network reads the encoder's final state / `state_for_prediction` over the full window to emit (u_i, lambda_i). |
| #2 item-to-KC mapping (absent in core) | New fixed lookup buffer (Q-matrix) threaded through data and model calls. `collate_adapter_items` is extended with a RAGGED `kc_ids` field of shape (N, T, A_max) with its own per-slot mask, where A_max is the bed's maximum item-KC arity (small on KDD, 6 on EdNet). One scalar per position cannot carry EdNet's genuinely multi-tag items (mean arity 2.18), so this is more machinery than a clone of `item_ids` (R2-I1, correcting v1's "exactly like item_ids"). PAS-N1's per-KC gather averages over the item's tagged slots; each tagged KC's head trains on the interaction. `kc_data.py` owns this bridge. No learned KC embedding table in v1. |
| #3 item parameters (`item_params`, `item_key_dim` pattern) | The frozen 1PL bank feeds `Binary2PLDecoder` unchanged (difficulties frozen, ability path free). No per-KC alpha/beta structure in v1, so no new decoder inputs. |
| #4 loss (`CombinedLoss`, per-KC weights) | Prediction NLL home, binary BCE via the existing decoder `nll`; no ordinal penalty, no per-KC class weights in v1 (binary responses). |
| #5 anchoring (item-axis only) | Deliberately NOT extended. A4 introduces no learnable KC table, so no KC-axis anchoring primitive is needed; that gap is A5's problem and is recorded as such. |

New modules (names final; no code in this document):

- `kt-mirt/src/kt_mirt/growth/__init__.py`
- `kt-mirt/src/kt_mirt/growth/qmatrix.py` (Q-matrix buffer, expansion policies, pure-anchor counts)
- `kt-mirt/src/kt_mirt/growth/kc_data.py` (ragged kc_ids collate bridge extending `core.realdata`, per-slot masking)
- `kt-mirt/src/kt_mirt/growth/slices.py` (slice construction, opportunity indexing, density strata, saturation flags)
- `kt-mirt/src/kt_mirt/growth/curves.py` (PAS-C)
- `kt-mirt/src/kt_mirt/growth/bank.py` (hierarchical MAP calibration via Adam, blockwise growth absorption, tri-spec RB0 refits, freeze, cohort split)
- `kt-mirt/src/kt_mirt/growth/gate.py` (PAS-G: M0, M1a, M1b, penalized bounded Newton, odd/even evaluation, batched fits)
- `kt-mirt/src/kt_mirt/growth/rate.py` (MIX-L stage 2, penalized bounded-exponential fits, misfit flag)
- `kt-mirt/src/kt_mirt/growth/tracker.py` (PAS-N1, PAS-N2 on the vendored core)
- `kt-mirt/src/kt_mirt/growth/recognition.py` (full-window amortized u_i, lambda_i heads)
- `kt-mirt/src/kt_mirt/growth/active.py` (ACT transition and two-stage trainer)
- `kt-mirt/src/kt_mirt/growth/synth.py` (generator and the four twins, incl. the SYN-NS silent subset)
- `kt-mirt/src/kt_mirt/growth/battery.py` (section-4 battery; doubles as the A6 core)
- `kt-mirt/src/kt_mirt/growth/report.py` (verdict assembly, posture-disagreement matrix, BY sensitivity)
- Scripts: `kt-mirt/scripts/a4/{prep_kdd,prep_ednet,run_synth,run_bed,run_battery,make_report}.py`
- Tests: `kt-mirt/tests/test_growth_{qmatrix,slices,gate,rate,active,synth,battery}.py`

Convention reminders: no runtime import from `deep_irt`; results files
never embed dataset copies; datasets stay under `data/` and kt-irt
caches (path exceptions for the two raw corpora noted in section 6).

## 3. Synthetic generator and the four certification twins

### 3.1 Generator family

Ground truth per learner-KC, standard shape: theta*_ic(n) = m_ic -
(m_ic - theta0_ic) exp(-r_c lambda_i (n-1)). Responses y ~
Bernoulli(sigmoid(theta*(n) - b_j)) with items drawn from the KC's item
bank.

- Distinct per-KC rates (fixing the old M3 generator flaw of
  near-identical rates): r_c log-uniform on [0.02, 0.40] per
  opportunity, a 20-fold spread, so rate RANK recovery is a meaningful
  target.
- Learner heterogeneity per the PNAS pattern (wide start, narrow rate):
  theta0_ic = xi_i + eta_c + noise with xi_i ~ N(0,1); lambda_i ~
  LogNormal(0, 0.2).
- Magnitude anchored to the 0.1 log-odds/opportunity population number:
  gaps (m - theta0) scaled so the median model-implied opportunity-1
  slope lands in [0.05, 0.15] logits (acceptance check below).
- Learner-KC incidence: each learner practices a Zipf-weighted subset
  of KCs (approximately 40 KCs/learner in KDD-matched configs, 25 in
  EdNet-matched), producing the long-tailed density the real beds show.
- Learner counts are compute-chosen (they were not a triage-matched
  quantity); density and KC counts ARE triage-matched.

Two density/KC-count profiles, matched to triage:

| Profile | C (KCs) | N learners | opp per slice (target quantiles) | per-KC overall rate (target quartiles) | item arity |
|---|---|---|---|---|---|
| KDD-matched (KTracedSkills) | 515 | 3000 | q1 4 / median 8 / q3 16, clipped [1, 60], frac >= 10 near 0.42 | 0.75 / 0.855 / 0.925 (saturated mass included on purpose) | 1 |
| EdNet-matched | 189 | 6000 | q1 1 / median 2 / q3 5, frac >= 10 near 0.14 | 0.61 / 0.68 / 0.72 | mean 2.2, max 6 (multi-tag) |

Generator acceptance checks, pre-registered (the build agent has
implementation freedom; these make the spec verifiable): realized
per-KC rate quartiles within 0.03 of target; pooled opportunity-1-to-10
rise within [0.08, 0.16] (KDD-matched; triage 0.734 to 0.839) and
[0.09, 0.17] (EdNet-matched; triage 0.569 to 0.697); realized density
quantiles within 1 opportunity of target; median implied opportunity-1
slope in [0.05, 0.15] logits. A generator run failing acceptance is
regenerated, never analyzed.

A small development config SYN-DEV (C = 50, N = 500, KDD density) exists
for iteration and smoke tests; no certification number comes from it.

### 3.2 The four certification twins

Twins share seeds and practice schedules with their known-growth
sibling wherever the dynamics allow (matched-twin discipline). ACT now
runs on all four twins (v1 ran it on SYN-NG only, R1-B1); trackers run
on SYN-NG and SYN-KG plus the probe arms.

- SYN-NG (no-growth twin; both density profiles): r_c = 0 for all c,
  everything else identical. The ACTIVE posture must stay silent (CG1)
  and the gate must not fire above its nominal false-positive rate
  (CG2; the EdNet-density variant, with matched multi-tag arity,
  doubles as the BH dependence stress per 2.2).
- SYN-KG (known-growth twin; both density profiles): the standard
  generator above. The PASSIVE posture must detect at the
  density-predicted reliability (CG3, CG4a/b): the Spearman-Brown-style
  response-sampling arithmetic is run on the generator config FIRST to
  predict split-half reliability and detectability, and observed values
  must match the prediction within tolerance, in both directions. ACT
  must FIRE here under the RB-A definition (CG1a, the active positive
  control; a firing rule that never fires on true growth is
  uninterpretable on real data).
- SYN-NS (non-standard-shape twin; KDD density): growth exists but the
  bounded-exponential family is wrong. Among the growing KCs, half
  follow step growth (jump of 0.8-1.5 logits at a learner-specific
  changepoint drawn uniform on opportunities 3-8, insight-like); half
  follow dip-then-recover (a 0.5 logit dip over opportunities 2-4,
  then bounded-exponential rise, interference-like and non-monotone).
  NEW in v1.1: a pre-registered 20% of KCs (seed-chosen) are held at
  r_c = 0 (the SILENT SUBSET, mechanically identical to SYN-NG KCs),
  so that misfit laundering into channels that should carry zero
  signal, Lemma 3's exact signature, is directly measurable. The MIXED
  ladder must fire the existence gate AND fire the misfit flag AND
  withhold r on the non-standard KCs (CG5). ACT runs on this twin as
  the program's MISMATCHED-GENERATOR ROBUSTNESS ARM (the "live C3
  threat" the qmirt archaeology flagged repeatedly as never run, now
  run): its gain family is wrong by construction, and certification
  (CG1b) requires silence on the silent subset, detection under
  misfit, and bounded overshoot.
- SYN-SAT (saturated twin; KDD density): theta0 raised so every per-KC
  start rate is at least 0.90 (q1 at least 0.88), true growth present
  with the same r_c. The existence gate must FAIL, reproducing the E2d
  lesson that saturation is a data-property limit; the pipeline's
  verdict must be "insufficient dynamic range", driven by the
  saturation flag, not a silent null (CG6). ACT must not launder
  saturation into confident gains (CG1c): with observations pinned
  near the ceiling there is no information to calibrate latent gains,
  and manufacturing well-resolved g_c there is the ceiling-driven
  fabrication mode (R1-B1).

Seeds: 5 generator seeds per config for all slice-based machinery;
neural arms (trackers, active, frozen-null, probes) run on generator
seed 0 with 3 model seeds (compute concession, pre-registered).

## 4. Null and audit battery

The battery is one module (`growth/battery.py`) applied uniformly to
synthetic and real beds; it doubles as the A6 certification core. Ten
arms.

1. Permutation null (primary reference for PAS-C and PAS-G). Scheme:
   permute each learner's full interaction order, then rebuild slices
   and opportunity indices. This destroys within-slice trend while
   preserving slice membership, item multisets, marginal rates, and
   (on EdNet) multi-tag co-occurrence, so it is coherent on both beds
   with one construction. Implementation constraint, binding (R2-I4):
   permutation replicates operate on the pre-extracted compact slice
   tensors (npz) in memory; the raw file is parsed exactly once per
   bed (a naive re-parse per replicate would alone cost ~37 CPU-h on
   KDD at the triage's measured 110 s/pass, blowing the budget).
   B = 199 replicates for per-KC empirical p (BH q = 0.05 across KCs;
   BY sensitivity reported per 2.2); B = 999 for the bed-level pooled
   statistic (empirical p floor 0.001). B may be raised, never lowered.
2. Static twin. Synthetic: SYN-NG. Real beds additionally get an
   M0-PARAMETRIC-BOOTSTRAP twin: within each slice, responses are
   redrawn y ~ Bernoulli(sigmoid(theta_hat_ic - b_j)) with theta_hat_ic
   the slice's penalized constant-ability (M0) fit and b_j frozen;
   schedule and item sequence intact; the few positions excluded from
   likelihoods (uncalibrated items) are redrawn from the slice mean.
   This null preserves the item-difficulty structure that the gate's
   own null model assumes, so a pipeline returning non-null on it is
   fabricating relative to its own null family. v1's i.i.d.
   slice-mean resample is DROPPED (R1-B5): erasing b_j structure made
   it a compound null (no growth AND no difficulty structure) whose
   direction of error on the gate was uncharacterized. The full
   pipeline must return null on the bootstrap twin (RB1). Trackers
   refit on it at 1 seed (compute concession, pre-registered).
3. Untrained/frozen-encoder null (Ding and Larson arm 1). Freeze the
   tracker's encoder at seeded random initialization; train only the
   readout head(s) against the frozen bank; run the identical pipeline.
   Synthetic requirement CG7: the trained tracker must beat the frozen
   null on KC-level growth recovery by a pre-registered margin, and the
   frozen null must not itself clear the passive-detection gate. The
   trained-vs-frozen per-KC growth-profile CORRELATION is also
   recorded on synthetic runs and calibrates RB2's real-bed bar
   (bridge rule in 5.1/5.2; R1-I4). Real-bed requirement RB2: trained
   must beat frozen on held-out forecast NLL (seed-clustered, 2.2
   regime), and tracker growth claims are made only if the
   trained-vs-frozen per-KC growth-profile correlation is below the
   RB2 bar; otherwise the tracker read is labeled not attributable to
   learned state and excluded.
4. Single-KC-drill contamination probe (Ding and Larson arm 2).
   Construct synthetic drill sequences: one KC practiced 100 times all
   correct (oracle) and all incorrect (anti-oracle), other KCs
   untouched; replay through the fitted tracker; contamination ratio =
   max over off-target KCs of |displacement| divided by on-target
   |displacement|, averaged over 5 drilled KCs and both directions.
   The GATE is the ratio alone: pass at ratio <= 0.10 (CG8). The
   off-target ABSOLUTE movement is additionally reported against the
   length-matched p95 displacement band of SYN-NG slices, as a
   diagnostic, NOT a gate (demoted in v1.1, R1-I3): a 100-repeat
   single-KC drill is far outside the Zipf-interleaved practice
   distribution, and holding an out-of-distribution stress to an
   in-distribution band conflates two properties; the ratio is
   self-normalized under the same input and carries the certification
   load. PAS-N1 is expected to fail (that failure is a finding, not a
   bug); PAS-N2 passes by construction and the probe verifies the
   implementation.
5. Order-invariance stress (Ding and Larson arm 3). Hold within-KC
   order fixed, permute cross-KC interleaving (5 reshuffles); per-KC
   tracker trajectories must be stable: median per-KC trajectory
   correlation >= 0.8 and displacement sign flips on fewer than 10% of
   KCs (CG9). PAS-C, PAS-G, MIX-L, and PAS-N2 are invariant to this
   permutation by construction (they see only the slice). The stress
   binds on PAS-N1 and on ACT's recognition inputs; ACT now has its
   own quantified bar (CG9-ACT, closing the silent gap R1-I12):
   across reshuffles, median per-learner correlation of u_i >= 0.9
   (and of lambda_i >= 0.9 for ACT-P1), and population per-KC
   implied-rise profile correlation >= 0.95 (ACT's transition is
   interleaving-invariant given the recognition outputs, per 2.3, so
   instability can enter only through them).
6. Direction audit (the Deep-IRT reconstruction check). For the
   practiced KC at step t, using causally aligned theta, a violation is
   d theta < -0.01 after y = 1 or d theta > +0.01 after y = 0.
   Tracker certification requires violation fraction <= 10% (CG10);
   the full distribution is reported. ACT is response-blind by
   construction, so the audit is vacuous there (stated, not celebrated);
   it binds on PAS-N trackers.
7. Split-half reliability. Odd/even opportunity split within slice;
   per-learner reliability = Spearman-Brown-corrected correlation of
   half-sample rates (or displacements) across D2+ slices; KC-level
   reliability = split learners into halves, fit r_c on each, correlate
   across KCs. Observed reliability is compared against the
   density-predicted value (the qmirt no-fitting arithmetic,
   generalized by parametric bootstrap of the fitted curves);
   agreement tolerance 0.10 (tightened from v1's unjustified 0.15;
   the qmirt precedent achieved ~0.03 agreement in a cleaner setting,
   and 0.10 already grants 3x that slack for the sparser per-KC
   setting; R1-I8, judgment call, section 9).
8. Truncation stress. Re-estimate beta_c and r at opportunity cutoffs
   {5, 10, 20, full}. Claimed rates must be truncation-stable (rank
   correlation of r_c between cutoff 10 and full >= 0.8), and no
   rate-heterogeneity magnitude claim is made unless BOTH hold: the
   IQR of r moves less than 25% between consecutive cutoffs AND the
   CUMULATIVE ratio IQR(cutoff 10)/IQR(full) lies in [0.75, 1.33]
   (the consecutive bound alone lets three ~24% moves compound to the
   very ~75% inflation Lee et al. warn about, R1-I7; the cumulative
   band binds the quantity their result is actually about).
9. Seed clustering. Neural fits at 3 model seeds minimum; slice-based
   results across 5 generator seeds on synthetic. All confirmatory
   statistics reported with seed treated as a cluster; claims must be
   sign-consistent in every seed and significant in the seed-pooled
   analysis.
10. Bank robustness (new in v1.1, R1-B4). Synthetic: calibrated b vs
    generator-true b rank correlation >= 0.9 on SYN-KG-KDD and SYN-NS
    (clauses in CG3/CG5). Real beds: tri-spec recalibration
    (no-growth / linear / blockwise growth absorption) with pairwise
    rank correlation >= 0.95 of calibrated difficulties at the finest
    fitted level, restricted to units with >= 50 calibration responses
    (RB0). Rationale in 2.1: every posture reads through this one
    frozen artifact, and the permutation null cannot detect a
    difficulty-vs-curriculum-position confound.

Ordering discipline (positive-control-first): no real-bed null or
verdict is interpreted before the known-growth twin certifies the
pipeline (CG3), because the trajectory program's own history shows a
null read from an uncertified metric is uninterpretable.

## 5. Pre-registered thresholds and kill conditions

### 5.1 Synthetic certification gates (ALL must pass before any real-bed interpretation)

| Gate | Twin / target | Pass condition |
|---|---|---|
| CG1 active silence | SYN-NG, ACT-P0 and ACT-P1, both densities | Model-implied score change over 10 opportunities: population mean <= 0.01 proportion-of-max; p95 per-learner <= 0.01 (KDD density) / <= 0.02 (EdNet density) |
| CG1a active positive control | SYN-KG, both densities | RB-A firing definition triggers at bed level; population implied rise within [0.5, 1.5] x the true score-scale rise; rank corr(per-KC implied rise, true per-KC rise) >= 0.6 (KDD density) / >= 0.5 (EdNet density) over unsaturated KCs |
| CG1b active misfit robustness (mismatched generator) | SYN-NS | Silent subset (r_c = 0 KCs): CG1 silence bars hold; growing KCs: rank corr(implied rise, true total rise) >= 0.5; bounded laundering: implied rise exceeds 1.5x true rise on <= 10% of growing KCs |
| CG1c active saturation refusal | SYN-SAT | Population implied score rise <= true rise + 0.02 proportion-of-max; p95 per-learner <= true p95 + 0.03; ACT verdict = "insufficient dynamic range for gain calibration" driven by the saturation flag (latent g_c withheld per the 2.3 reporting rule) |
| CG2 gate false positives | SYN-NG, both densities | Bed-level pooled p > 0.01; per-KC BH discoveries <= 2% of KCs (EdNet-density twin = the multi-tag dependence stress; BY-sensitivity list reported) |
| CG3 passive detection, KDD density | SYN-KG-KDD | Bed pooled p < 0.001; >= 60% of unsaturated KCs discovered at BH q = 0.05; rank corr(r_hat_c, r_c) >= 0.7 over discovered KCs; observed split-half within 0.10 of the density-predicted value; bank recovery rank corr(b_hat, b_true) >= 0.9 |
| CG4a passive existence detection, EdNet density | SYN-KG-EDNET | Bed pooled p < 0.001 (this clause licenses the RB-E1 machinery); per-learner detection reported but NOT gated (pre-registered exclusion: no per-learner claims at median-2 density, ever) |
| CG4b KC-rate recovery at EdNet density | SYN-KG-EDNET | KC-level pooled rate rank corr >= 0.6. FAILURE IS A FINDING, NOT A KILL: with median 2 opportunities a 2-parameter curve sits at the identifiability floor (the archaeology's elbow-spanning window requirement cannot be met), so failure = pre-registered honest verdict "EdNet-class density is below the ladder's rate-recovery floor" (K7). EdNet real-bed claims are capped at Tier 1 in v1 REGARDLESS of CG4b (5.3), so no real-bed license rides on this gate |
| CG5 non-standard shape | SYN-NS | Pooled gate fires (p < 0.001); misfit flag fires on >= 80% of non-standard KCs; <= 20% of non-standard KCs receive an unflagged rate; bank recovery rank corr(b_hat, b_true) >= 0.9 under misfit |
| CG6 saturation refusal | SYN-SAT | Pooled gate p > 0.05; saturation flag marks >= 95% of KCs; pipeline verdict is "insufficient dynamic range" |
| CG7 frozen-encoder margin | SYN-KG-KDD, trackers | Trained KC-growth-profile rank recovery >= 0.6; trained minus frozen >= 0.2; frozen must not clear CG3's detection bars; decertify the tracker if frozen is within 0.1 of trained. BRIDGE CLAUSE (R1-I4): record the trained-vs-frozen profile correlation across seeds; at R2 close, RB2's real-bed bar is set to min(0.9, synthetic p95 + 0.1), floored at 0.7, and frozen before any real run (bars may tighten, never loosen) |
| CG8 contamination | trained trackers | Ratio <= 0.10 (the gate). Off-target absolute movement vs the length-matched SYN-NG p95 band is reported as a diagnostic only (see battery arm 4) |
| CG9 order stress, trackers | trackers | Median per-KC trajectory correlation >= 0.8; sign flips < 10% of KCs |
| CG9-ACT order stress, active | ACT recognition inputs | Median per-learner corr(u_i) >= 0.9 across 5 reshuffles (and corr(lambda_i) >= 0.9 for ACT-P1); population per-KC implied-rise profile correlation >= 0.95 |
| CG10 direction audit | trackers on SYN-KG | Violation fraction <= 10% |

If the gate fires robustly on SYN-SAT (CG6 inverted), the twin or the
gate is broken; that is a certification failure requiring diagnosis,
not a power bonus.

### 5.2 Real-bed licensing conditions, KDD-KTracedSkills (primary bed)

Unsaturated subset = KCs with calibration-cohort correct rate <= 0.85
(a raw, model-free statistic computed on the calibration cohort so no
selection on analysis data; see 2.1 for the dependency note).

- RB0 (bank robustness, new in v1.1): tri-spec recalibration stability
  per battery arm 10 (pairwise rank corr >= 0.95). Failure means the
  frozen scale itself moves with the growth assumption used to
  calibrate it; since a curriculum-position-confounded bank can push
  opportunity-correlated residual into the gate (a Tier-1 threat, not
  merely a magnitude threat), RB0 failure quarantines the bed pending
  diagnosis, same severity as RB1 (K6).
- RB1 (in-situ fabrication check): the M0-parametric-bootstrap twin
  (battery arm 2) must return null through the whole pipeline: pooled
  gate p > 0.05; ACT population implied score change <= 0.01
  proportion-of-max AND p95 per-learner <= 0.02 (p95 clause added in
  v1.1; the population mean alone hides per-learner fabrication).
- RB2 (frozen-null distinguishability): as battery arm 3, with the bar
  set by the CG7 bridge clause (min(0.9, synthetic p95 + 0.1), floor
  0.7, frozen at R2 close).
- RB-A (active firing definition, new in v1.1, R1-B2). ACT "fires" at
  bed level iff ALL of: (i) population implied score rise over
  opportunities 1-10 >= 0.05 proportion-of-max; (ii) that rise is
  >= 5x the same model's RB1-twin read (the in-situ null reference);
  (iii) the rise is sign-consistent across all model seeds. ACT
  "fires" on KC c iff the bed fired and KC c's implied rise is
  >= 0.05 with all-seed sign consistency. The definition is validated
  on synthetic data before use: it must trigger on SYN-KG (CG1a) and
  its 0.05 bar sits 2.5-5x above the CG1 silence bars, so silence and
  firing cannot overlap. Active-only firing is never claimed as
  growth (5.5); RB-A exists so the disagreement matrix's
  active-involving rows are evaluable at all.
- RB3 (Tier 1, existence): bed-level pooled gate p < 0.01 on the
  unsaturated subset, surviving RB0 and RB1. Pre-registered internal
  replication: the gate failure rate on the saturated subset is
  predicted to be materially higher (direction reported, not gated).
- RB4 (Tier 2, KC rates): KC-level split-half >= 0.8 AND truncation
  stability (battery arm 8, both clauses) AND misfit flag silent on
  the claimed KCs.
- RB5 (Tier 3, per-learner rates): per-slice split-half >= 0.7 within
  D2+ strata AND truncation stability. If RB5 fails at every stratum,
  the verdict is the E2c pattern at per-KC granularity: existence yes,
  individual rates unreliable at this density. That is a reportable
  honest outcome, not a program failure.

### 5.3 Real-bed licensing, EdNet KT1 (population corroboration only)

- Pre-registered exclusions: no per-learner claims and no G1-style
  causal reads at median-2 density with the bundle confound,
  regardless of results. EdNet claims are capped at Tier 1 in v1
  regardless of CG4b (the cap follows from the bundle confound and
  density, not from certification outcomes; opening a conditional
  Tier-2 path is ruling 3, section 11). ACT does not run on EdNet.
- RB-E1: pooled gate p < 0.01 plus the RB0/RB1-equivalent checks
  (tri-spec bank stability; M0-bootstrap null through the slice
  machinery and trackers) = population corroboration achieved.
  Failure = corroboration dead; EdNet then contributes nothing to G2.

### 5.4 XES3G5M (conditional bed)

Enters only after the stage-0 triage script has produced its density,
saturation, decoupling, and anchor numbers. Certification is
DENSITY-SPECIFIC (v1.1, closing R1-I6): XES3G5M gets its own
triage-matched generator profile and a fresh CG2/CG3/CG5/CG6-
equivalent twin certification at that profile before any real-bed
interpretation; positive-control-first is not transferable across
density regimes. It inherits KDD's numeric bars unless its triage
exposes a regime those bars cannot express, in which case new bars are
pre-registered in a design addendum BEFORE any run on it. No number in
this design is assumed for it.

### 5.5 Posture-disagreement matrix (diagnostic, pre-registered readings)

Structural preamble (v1.1, R1-B3): at the existence tier there are
exactly THREE independent inputs, the shared slice-based gate (PAS-G,
which is also MIX-L's stage 1: one implementation, ONE vote), the
tracker displacement read (PAS-N, judged through the battery), and the
ACT firing verdict (RB-A). "Passive fires, mixed flat at existence" is
structurally impossible and appears in no row. Passive-vs-mixed
disagreement is defined only at the rate stage (gate fired, rate
machinery disagrees). Tier-3 comparisons are additionally partitioned
by sign (section 0): on decline-side slices ACT abstains by
construction, so decline diagnostics are two-posture (PAS/MIX) reads;
an ACT gain on a slice where PAS-N and MIX-L agree on decline is
recorded as the rho = 1 pin operating, not as independent evidence
either way, and a high rate of such slices is reported under the
scope limit.

| Pattern (independent inputs) | Reading |
|---|---|
| ACT fires (RB-A), gate flat | Fabrication suspect; active quarantined; recheck CG1/CG1b config parity |
| Gate fires, ACT flat | Gain family or gating too rigid; consult E-M3 misfit flags |
| Gate fires, tracker flat | Tracker insensitivity; consult CG7 margin and contamination probe before blaming the data |
| Tracker fires, gate flat | Reconstruction/contamination artifact suspect; direction audit and CG8 consulted; tracker read quarantined |
| Gate fires, rate unreliable (RB4/RB5 fail) | Existence-only claim (Tier 1); the E2c pattern |
| Gate and rate fire, shapes disagree (E-M3 fired) | Growth real, family wrong; report blockwise profile, withhold r |
| All three independent inputs agree | Strongest claim tier available at that density |

Active-only signals are never claimed as growth.

### 5.6 Kill conditions and revision budget

Revision budget: at most TWO revision rounds on the synthetic
certification matrix; each revision may fix mechanisms and bugs but may
never loosen a threshold; every revision is a LEDGER entry stating what
changed and why. Clarification (v1.1, R1-M2): invoking a fallback
PRE-REGISTERED in this document (e.g. CG1's fixed-M refit) is NOT a
revision round, but each named fallback may be invoked at most ONCE
and is logged in the LEDGER like a revision; any change not
pre-registered here consumes a revision round. This closes the
scope-creep vector of unbounded "contingency" reruns.

- K1 (posture machinery): CG1-family/CG2/CG5/CG6 still failing after
  the budget kills the failing posture program-wide for G2. CG1/CG1b/
  CG1c failure in both ACT variants = active posture dead
  (fabrication), itself a reportable diagnostic. CG3 failure = the
  ladder cannot detect at realistic density; G2 retreats to synthetic
  certification plus an honest data-property verdict (the
  exhaust-venues discipline).
- K2 (KDD existence): RB3 fails = G2 existence dead on the primary
  bed; G2 then rests on XES3G5M if landed, else the program-level
  fallback in K1 applies.
- K3 (KDD magnitude): RB4 fails at all strata = Tier 2+ dead on KDD;
  Tier 1 claim stands if RB3 passed.
- K4 (EdNet): RB-E1 fails = population corroboration dead.
- K5 (tracker leg): CG7 and CG8 both fail for every tracker config =
  neural per-KC readout decertified; A5 is blocked pending
  architecture work; A4's G2 verdict rides on the slice-based
  machinery alone, which remains valid because its scale is the frozen
  bank, not the encoder.
- K6 (in-situ integrity): RB0 or RB1 fails on a real bed = every
  readout on that bed is quarantined until diagnosed; no claim of any
  tier.
- K7 (density floor, informative non-kill): CG4b failing while CG3
  passes = the honest verdict "KC-rate recovery is below the
  identifiability floor at EdNet-class density"; it kills nothing
  (EdNet is Tier-1-capped regardless) and is itself reportable
  methodology.

## 6. Bed plan and Q-matrix policy

Order: synthetic first (nothing real is interpreted before the full CG
matrix passes), then KDD-KTracedSkills (primary), then EdNet
(population corroboration), then XES3G5M when landed. TIMSS is
structurally not a growth bed (single occasion) and SLAM is absent and
untagged; both are excluded from A4.

| Bed | Role | Data | KC model and expansion policy (restating avenue_map section 4 for A4) |
|---|---|---|---|
| Synthetic | Certification | `growth/synth.py` | As generated; 1-to-1 (KDD-matched) and multi-tag (EdNet-matched) |
| KDD Cup 2010 Algebra 2008-2009 | PRIMARY (density: median 8, frac >= 10 = 0.423; decoupling and anchors pass) | `data/kdd/algebra_2008_2009_train.txt`, full 8.9M steps | KC model = KTracedSkills (515 KCs; the triage's recommended model, avoiding SubSkills' co-scheduling failure and Rules' catch-all-bug artifact). Item = step (Problem Hierarchy + Problem Name + Step Name), RETAINED under the hierarchical bank of 2.1 (1.31M distinct steps, 74% singletons; see ruling 1); response = Correct First Attempt. Steps are practice opportunities, never compared 1:1 with item-level beds. Multi-KC steps: the interaction enters every tagged KC's slice and increments each tagged KC's opportunity counter (explicit multi-KC opportunity policy, stated because this is the Algebra file, not near-1-to-1 Bridge; the induced cross-KC test dependence is handled per 2.2). Per-KC attribution claims only on KCs with >= 3 pure steps (85.2% qualify). Saturation policy: per-KC flags at 0.85; primary verdicts on the unsaturated subset (48.3% of KCs) |
| EdNet KT1 | Population corroboration ONLY (median-2 density; bundle confound) | `EdNet-KT1/KT1/` per-user files + `EdNet-Contents/contents/questions.csv`. PATH NOTE (R2-M2): both corpora live at the REPO ROOT, outside `data/`; the kt-irt caches hold only small capped npz samples, NOT the required draws, so `prep_ednet.py` must read the repo-root corpus. Seeded samples: 50k users for slice-based, 20k for trackers (disjoint from the 4k triage sample) | Tags as KCs (189). 1-to-many: each attempt enters all its tags' slices and increments all their counters (ragged kc_ids per 2.5); bundle exposure noted as a scheduling confound on co-movement (harmless to population growth, fatal to causal reads, hence the exclusions). Anchors moderate (40% of tags with >= 3 pure items): attribution-restricted. Tag ids are numeric only; no content naming without a crosswalk |
| XES3G5M | Conditional (learning-heavy, expert KC tree) | pending landed download | Question-level sequences; multi-KC leaf tags as Q-row loadings; NEVER split train/test at KC-expanded rows (leakage); follow the benchmark's own split; count pure questions per leaf KC before any attribution claim; stage-0 triage first, then OWN-DENSITY twin certification per 5.4 |

## 7. Compute plan

Principles: slice-based machinery is vectorized (penalized batched
Newton steps over hundreds of thousands of slices, torch tensors; KDD
KTracedSkills has 335,430 learner-KC slices), so it runs on CPU and
goes 10-50x faster on GPU; the BANK CALIBRATION is a different
numerical problem (Adam over ~1.3M hierarchical difficulty parameters,
2.1) and is costed separately; neural fits need GPU; permutation
batteries and seed farms are embarrassingly parallel and go to SLURM
(code/guest-research partition, 2-GPU cap; CPU array jobs for
permutations). Preprocessing produces compact slice tensors (npz,
order 100 MB) that are rsynced to the cluster; raw beds never leave
local disk; no credentials stored in the repo. Runs assume the
`research` conda env (torch 2.7.1+cu126 verified on the 4060).

| Workload | Where | Wall-clock estimate |
|---|---|---|
| Build + unit tests (modules of 2.5) | local CPU | 5-8 working days agent time (revised from 3-5, R2-I5: the bank calibrator, penalized Newton with separation safeguards, ragged multi-tag bridge, two-density generator with acceptance loop, two trackers, and the two-stage ACT trainer are net-new statistical code; the vendored core covers only the neural backbone) |
| Bank calibration (Adam, MAP): KDD ~4.45M calibration rows, ~1.3M difficulty params, x 3 growth specs (RB0) | local 4060 | 1-3 GPU-h per spec, 3-9 GPU-h total; EdNet minutes (new line, R2-B2) |
| Generator + acceptance checks, all configs x 5 seeds | local CPU | < 1 h |
| Slice-based pipeline, one pass (gate + rate + curves), synthetic or KDD | local CPU 10-30 min; local GPU minutes | -- |
| Permutation battery, per bed (B = 999 pooled + 199 per-KC), cached-tensor replicates | local 4060 overnight (2-8 h) or SLURM CPU array (10 x ~1 h) | <= 1 day per bed |
| Tracker fits, synthetic (1M-token configs): PAS-N1 + PAS-N2 x {trained, frozen} x 3 seeds x 2 density configs | local 4060 + SLURM 2 GPUs | 24 fits x 20-40 min = ~8-16 GPU-h; ~1 day |
| ACT fits, synthetic: 2 variants x 6 twin-density configs (NG/KG at both densities, NS/SAT at KDD) x 3 seeds = 36 fits | local 4060 + SLURM | ~18-36 GPU-h; 2-3 nights (was 6-12 GPU-h in v1, which under-counted the twin matrix, R1-B1/R2-I2) |
| KDD trackers (8.9M steps): 2 configs x {trained, frozen} x 3 seeds + 1 bootstrap-twin seed | SLURM 2 GPUs + local 4060 | 13 fits x 2-6 h = 26-78 GPU-h; 1.5-2.5 days at 3 concurrent |
| KDD ACT: 2 variants x 3 seeds + bootstrap twin | local 4060 | 12-21 GPU-h; ~1 day |
| EdNet slice-based (50k users) + trackers (20k users; PAS-N2 at ~2.2x tokens) | CPU hours + ~35 GPU-h | ~1.5 days |
| Probes, stress arms, report assembly | local CPU/GPU | hours |

Sequence lengths: KDD per-learner sequences can be long; cap encoder
input at the most recent 2048 interactions for PAS-N1/ACT recognition
(cap recorded in outputs; PAS-N2 and all slice-based machinery are
uncapped since slices are short). Calendar (v1.1, R2-I3): about 2.5 to
3 weeks ONE-PASS from build start to the KDD + EdNet verdicts, plus a
reserved week covering the up-to-two revision rounds section 5.6
permits (which re-run neural certification fits); 3.5 to 4 weeks is
the honest envelope. v1's 2-2.5 weeks priced one pass only.

## 8. Deliverables and run order

Run order (each run gets a LEDGER entry with expectation before,
reality after, verdict):

- R0 build: modules + tests green (`python -m pytest kt-mirt/tests`),
  including the separation, ragged-collate, and bank-prior unit tests
  named in sections 2.1-2.2.
- R1 generator bring-up: acceptance checks on all configs; SYN-DEV
  smoke of gate and rate (positive control first).
- R2 synthetic certification matrix: the full CG set (CG1, CG1a-c,
  CG2, CG3, CG4a/b, CG5-CG8, CG9, CG9-ACT, CG10) across the four
  twins, two density profiles, 5 generator seeds (slice-based) and 3
  model seeds (neural). STOP/GO checkpoint; RB2 bar frozen here per
  the CG7 bridge; revision budget per 5.6.
- R3 KDD slice-based: calibration (tri-spec, RB0) + freeze, PAS-C,
  PAS-G, MIX-L, permutation battery, M0-bootstrap twin (RB1),
  split-half, truncation.
- R4 KDD neural: PAS-N1/N2 trained + frozen-null, contamination,
  order stress, direction audit, RB2.
- R5 KDD active: ACT-P0/P1, in-situ silence check on the bootstrap
  twin, RB-A read.
- R6 KDD verdict assembly: posture-by-bed matrix, tier claims,
  disagreement diagnostics; A5 go/no-go recommendation.
- R7 EdNet population leg: slice-based + trackers per section 6 (no
  ACT).
- R8 XES3G5M: triage first, then own-density twin certification, then
  the same ladder (conditional).

Deliverables:

- D1 `kt-mirt/src/kt_mirt/growth/` + tests (battery module doubles as
  the A6 core).
- D2 `kt-mirt/_planning/a4/synth_certification.md` + JSON results (the
  full CG matrix).
- D3 `kt-mirt/_planning/a4/kdd_matrix.md` (primary G2 verdict, three
  postures, tiers, disagreement matrix).
- D4 `kt-mirt/_planning/a4/ednet_population.md`.
- D5 PLAN/LEDGER updates and the A5 recommendation.

## 9. Judgment calls made in this design (flagged for review)

1. Gate evaluation uses an interpolative odd/even split rather than a
   forecast split (rationale in 2.2); the archaeology did not specify
   its split, so this is a design decision, not an inheritance.
2. The frozen bank is 1PL, calibrated inside a blockwise-growth model
   with hierarchical difficulty shrinkage (KDD), freezing difficulties
   only. Chosen to sidestep the discrimination-collapse mechanism
   entirely; 2PL is a pre-registered extension. The prior scales
   (1.5 / 1.0 / 0.5 logits), the d_j exposure floor (>= 20), and the
   RB0 restriction (>= 50 responses) are judgment numbers.
3. The permutation null permutes each learner's whole interaction
   order (not within-slice only) so one scheme is coherent on both
   beds, including EdNet's multi-tag sharing.
4. The active model is response-blind (AFM-flavored); a
   success/failure-count (PFA-flavored) gain is an extension only if
   v1 clears the twins. Consequence: the direction audit is vacuous
   for ACT and binds on the passive trackers.
5. Ceiling M is a single global fitted scalar with a fixed-M fallback;
   per-KC ceilings were rejected for v1 as a fabrication surface.
6. PAS-N2 (factorized per-KC tracker) was added beyond the brief so
   that a contamination-proof-by-construction tracker exists alongside
   the field-representative PAS-N1; the N1-fails/N2-passes contrast is
   itself measurement-audit material.
7. Numeric bets not derivable from the record: CG3's 60% detection bar
   is a POWER FLOOR chosen by design judgment (v1 dressed it in the
   73-77% positive-slope triage fraction, but that is a model-free OLS
   statistic over all KCs on real data, a different method, population,
   and source; the transplant was a non sequitur, R1-I2, and the bar
   now stands on its own as a pre-registered bet); likewise the 0.10
   contamination ratio, the 0.8/0.7 reliability bars (qmirt's 0.80 bar
   and the archaeology's too-low 0.17/0.19 bracket them), the 10%
   direction-violation bar, and the RB2 bar's 0.9 cap and 0.7 floor.
   All are pre-registered here so they cannot drift after results
   exist.
8. Learner counts and seed splits (5 generator / 3 model seeds, neural
   on generator seed 0 only, bootstrap-twin trackers at 1 seed) are
   compute concessions, stated as such.
9. Neural arms are validated under a learner-level 80/20 forecast
   regime while slice-based comparisons are interpolative (2.2); the
   two regimes never meet inside a comparison. Chosen because a
   per-position odd/even loss mask is ill-defined on multi-tag beds.
10. Ceiling M initialization (95th percentile of calibrated b_j plus
    2 logits) is an arbitrary anchoring constant (R1-M1).
11. RB-A firing numbers (0.05 proportion-of-max; 5x the bootstrap-twin
    read) and the CG1a band ([0.5, 1.5]x truth) are fresh judgment
    numbers, validated for non-overlap with the silence bars but not
    derivable from the record.
12. CG1b's laundering bounds (silent-subset silence at CG1 bars;
    <= 10% of growing KCs above 1.5x overshoot; rank corr >= 0.5 under
    misfit) and CG1c's margins (+0.02 population, +0.03 p95) are
    judgment numbers.
13. Split-half agreement tolerance 0.10 (tightened from v1's 0.15;
    rationale in battery arm 7).
14. The cumulative truncation band [0.75, 1.33] on IQR(10)/IQR(full)
    is a judgment number sized against Lee et al.'s 75% inflation.
15. CG9-ACT bars (0.9 recognition-stability, 0.95 profile) are
    judgment numbers.
16. BY-sensitivity reporting is a reporting rule, not a gate; BH
    remains primary, certified under measured dependence via CG2.

## 10. Review dispositions (v1.1)

Two independent pre-run reviews. R1 = methodology review, R2 =
feasibility review. B/I/M = blocking/important/minor, numbered in the
reviews' own order. Every blocking item is fixed (none rebutted
outright); dispositions one line each.

- R1-B1 (ACT never run on SYN-NS/SYN-SAT): FIXED. ACT now certified on
  all four twins; CG1a/CG1b/CG1c added; SYN-NS gains the 20% silent
  subset; the mismatched-generator arm is named and run (3.2, 5.1).
- R1-B2 (no operational ACT firing test): FIXED. RB-A defined,
  synthetic-validated via CG1a, wired into 5.5.
- R1-B3 (shared existence gate breaks the three-vote framing): FIXED
  by restructuring, not by duplicating machinery: 5.5 now names the
  three actually-independent inputs, states the impossible cell, and
  2.4 records the deliberate trade (implementation variance vs one
  fewer vote).
- R1-B4 (linear-only calibration growth term can bias frozen b_j):
  FIXED. Calibration growth absorption is now blockwise; battery arm
  10 + RB0 audit the bank (synthetic truth recovery + tri-spec
  stability); rationale incl. why the permutation null cannot catch
  this (2.1).
- R1-B5 (i.i.d. resample twin is a compound null): FIXED. Replaced by
  the M0-parametric-bootstrap twin, which preserves item-difficulty
  structure and is the gate's own null family (arm 2, RB1).
- R2-B1 (KDD 1PL bank unidentified, 74% singleton steps): FIXED.
  Hierarchical MAP bank with proper priors and exposure-floored step
  offsets; item = step retained; escalated as ruling 1 because the
  alternative changes the estimand.
- R2-B2 (bank fit is not batched-Newton-shaped; uncosted): FIXED. Adam
  MAP spec in 2.1, dedicated compute line in 7.
- R2-B3 (separation in slice fits corrupts the gate statistic): FIXED.
  Penalized bounded Newton for all slice fits, shared by M0/M1 and the
  permutation null; unit test required (2.2).
- R1-I1 (BH under cross-KC dependence): FOLDED. CG2 at both densities
  certifies realized FDR under measured arity; BY sensitivity
  reported (2.2).
- R1-I2 (CG3 60% justification non sequitur): FOLDED. Justification
  rewritten as a stated power-floor bet (9.7); bar unchanged.
- R1-I3 (CG8 conflates OOD response with contamination): FOLDED. Ratio
  is the sole gate; absolute band demoted to a length-matched
  diagnostic (arm 4).
- R1-I4 (CG7/RB2 thresholds not matched translations): FOLDED. CG7
  bridge clause calibrates RB2's bar from synthetic runs; tighten-only,
  frozen at R2 close.
- R1-I5 (CG4 lacks an honest-floor path and a kill mapping): FOLDED.
  CG4 split into CG4a (licenses RB-E1) and CG4b (density-floor
  finding, K7); asymmetry against EdNet removed by making the cap
  explicit and CG4b non-load-bearing.
- R1-I6 (uneven density-stratified certification; XES3G5M inherits
  KDD thresholds): FOLDED. 5.4 now mandates own-density twin
  certification for XES3G5M; EdNet's narrower certification matches
  its Tier-1 cap (Tier-2 conditional path escalated as ruling 3).
- R1-I7 (consecutive-only truncation bound compounds): FOLDED.
  Cumulative band added (arm 8).
- R1-I8 (0.15 split-half tolerance unjustified): FOLDED. Tightened to
  0.10, flagged as judgment call 13.
- R1-I9 (rho = 1 forecloses individual decline; Tier-3 comparability):
  FOLDED as an explicit structural statement (0, 5.5) plus escalation
  (ruling 2). The pin itself is retained in v1: freeing rho on
  monotone beds is the Lemma-2 fabrication surface, so "test for
  decline inside ACT" is not free; the free-signed PAS/MIX machinery
  is the v1 decline detector.
- R1-I10 (no mismatched-generator arm for ACT): FIXED via R1-B1
  (CG1b IS that arm, named as such).
- R1-I11 (ACT evaluation split unspecified): FOLDED. Validation
  regimes fixed in 2.2 (neural 80/20 forecast; slice-based
  interpolative; no cross-regime comparison exists), judgment call 9.
- R1-I12 (no CG9 equivalent for ACT): FIXED. CG9-ACT added.
- R1-I13 (saturation flag / bank joint-estimation dependency):
  PARTIALLY REBUTTED, partially folded. The flag is a raw cohort
  statistic, not a fit output, so it does not inherit the growth
  term's estimation error; the shared cohort is deliberate
  (no selection on analysis data). Folded: flags recomputed on the
  analysis cohort as a reported diagnostic (2.1).
- R2-I1 (kc_ids cannot be a scalar clone of item_ids): FOLDED. Ragged
  (N, T, A_max) spec with per-slot mask (2.5).
- R2-I2 (CG1 EdNet-density ACT run unscheduled): FOLDED. ACT twin
  matrix now explicit (NG/KG both densities), compute row updated.
- R2-I3 (revision-budget compute not in calendar): FOLDED. Reserved
  week added; one-pass vs envelope stated (7).
- R2-I4 (permutation arm could be read as re-parsing raw files):
  FOLDED. Cached-tensor constraint written into arm 1 with the
  37-CPU-h arithmetic.
- R2-I5 (R0 build estimate optimistic): FOLDED. 5-8 days with the
  reason stated (7).
- R1-M1 (ceiling init unflagged): NOTED, judgment call 10.
- R1-M2 (fallback vs revision-budget ambiguity): NOTED and closed
  (5.6: fallbacks are once-only, logged, non-consuming).
- R1-M3 (no state-inert class on real beds): NOTED as an explicit v1
  limitation (0), with the M0-bootstrap twin named as the partial
  in-situ control.
- R2-M1 (PAS-N2 token-count equality overstated): NOTED, corrected
  (2.2, 7).
- R2-M2 (raw corpora live at repo root, not data/): NOTED in the bed
  table (6) for prep_ednet.py.
- R2-M3 (env verified: research conda env, torch cu126, 4060): NOTED
  in 7.

## 11. Open rulings for the orchestrator

1. KDD item granularity. v1.1 keeps item = step and identifies the
   bank by hierarchical shrinkage (steps under problems under
   hierarchies; step offsets only at >= 20 calibration responses).
   The alternative, redefining item = problem, would also solve the
   sparsity but CHANGES THE ESTIMAND (what "item" means in every
   readout and in the CFA response field). Rule: step-plus-shrinkage
   (default) or the estimand change.
2. ACT decline asymmetry. v1.1 keeps ACT growth-only (rho = 1) and
   partitions Tier-3 posture comparison into gain-side (three
   postures) and decline-side (PAS/MIX only). If the posture mandate
   is read as requiring an ACTIVE model that can represent individual
   decline, a signed-gain or free-rho ACT variant must be designed and
   certified now (new twins, new silence gates, the Lemma-2
   fabrication surface reopened), roughly +1 week. Rule: recorded
   scope limit (default) or decline-capable ACT in v1.
3. EdNet Tier-2 ambition. v1.1 caps EdNet at Tier 1 regardless of
   CG4b (CG4b becomes a density-floor finding, not a license). Opening
   a conditional EdNet Tier-2 path would require EdNet-density SYN-NS
   and split-half certification (~2-4 extra GPU-days) and sits
   uneasily with the bundle-confound rationale for the cap. Rule: keep
   the cap (default) or open the conditional path.
4. Budget approval. The revised plan adds bank calibration (3-9
   GPU-h), a tripled ACT synthetic matrix (18-36 GPU-h), a build
   re-estimate (5-8 days vs 3-5), and a reserved revision week;
   calendar moves from 2-2.5 weeks to 2.5-3 one-pass, 3.5-4 envelope.
   Approve, or name the cut (the candidate cut is ACT at EdNet
   density, saving about a GPU-day at the cost of an uncertified CG1
   EdNet bar and a weaker basis for XES3G5M-class reuse).
