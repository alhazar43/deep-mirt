# A4 pre-registered design: per-KC growth posture matrix (G2)

Status: PRE-REGISTERED, v1, 2026-07-18. This document freezes the A4
estimands, estimators, generator, battery, thresholds, and kill
conditions before any run. Thresholds may never be loosened after
registration; mechanism fixes are allowed within the revision budget in
section 5.6, and every revision is logged in `_planning/LEDGER.md`.
Inputs: PLAN.md, THINKING.md (2026-07-17 posture directive), LEDGER.md
(stage-0 triage), vendor_report.md, triage/triage_report.md,
research/{avenue_map, growth-methodology, trajectory-dynamics-archaeology,
qmirt-archaeology, interpretability_critiques_read}.md.

## 0. Scope and claim boundary

Goal G2: a per-learner digital twin shows ability growth or decline
beyond noise, read on a trustworthy scale. A4 decides whether growth is
detectable and readable per KC; A5 (frozen-anchor twin) builds on it
only if A4 passes on at least one bed.

Three postures, per the user directive. Posture disagreement is itself
a diagnostic and is reported, never averaged away.

- ACTIVE. The model imposes growth structure. Characteristic error is
  fabricated growth (Lemma 1, free-asymptote). Certified by silence on
  no-growth twins.
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
- Both real beds are monotone-rising in triage, so v1 has no decay or
  forgetting channel (Lemma 2 policy, rho pinned at 1). Decline can
  still be DETECTED by the passive and mixed machinery (displacement
  and blockwise profiles are free-signed); the active model is
  growth-only and this asymmetry is a recorded scope limit.
- Binary responses only in v1 (both real beds are binary; the K>2
  decoders stay unused here).
- No state-inert measurement-item class in v1; every interaction is
  both practice and measurement. Split-half, truncation, and the twins
  carry the load the qmirt ref-inert device carried there.

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
  is directly comparable to E-P1).
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
before any dynamics fit, per the anchoring posture.

- Split each real bed's learners at random into a calibration cohort
  (50%) and an analysis cohort (50%). Random learner split supplies the
  ability spread the qmirt calibration mechanism requires.
- Calibration model: AFM-with-item-difficulties on the calibration
  cohort, logit P(y=1) = a_i + gamma_c (n-1) - b_j (learner intercepts,
  per-KC opportunity slope, item difficulty). Freeze b_j; discard a_i
  and gamma_c. The growth term is included so difficulty is not
  confounded with when-in-learning an item tends to appear; freezing
  only b_j means no growth information leaks into the analysis cohort
  (disjoint learners).
- 1PL only in v1. Rationale: item location recovers robustly under the
  qmirt record while discrimination is the fragile parameter under
  joint calibration; the gate and the rate need only a
  difficulty-adjusted margin. 2PL is a pre-registered extension, not
  v1.
- Interactions on items unseen in calibration are excluded from
  gate/rate likelihoods but still increment opportunity counters.
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

PAS-N (neural passive trackers). Two configs, both trained on
prediction NLL, no growth structure, per-KC theta heads read against
the frozen bank scale (the decoder consumes frozen b_j; only the
ability path trains):

- PAS-N1 (shared-state tracker, the field-representative object): stock
  encoder (lstm primary; dkvmn as the pre-registered alternate) plus a
  per-KC ability head reading `state_for_prediction`, output dimension
  C, gathered by the current item's KC ids. This is the config Ding and
  Larson predict will fail contamination; running it is the point.
- PAS-N2 (factorized per-KC tracker, contamination-proof by
  construction): the same shared LSTM cell applied independently to
  each (i,c) slice's subsequence, so KC c's state sees only KC c's
  interactions. Order-invariant to cross-KC interleaving and
  contamination-free by construction; total token count equals the bed,
  so cost is comparable.

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
  active model AFM-flavored rather than PFA-flavored in v1).
- Pins: mu = 0 (no mean-reversion target exists; the OU channel is
  absent). rho = 1 (no decay; both beds monotone in triage). gamma-type
  per-learner transfer multipliers do not exist here by construction
  (no transfer term at all).
- Ceiling M: one global fitted scalar shared by all KCs and learners,
  initialized at the 95th percentile of calibrated b_j plus 2; if CG1
  (active silence) fails with fitted M, the pre-registered fallback is
  M fixed at that initialization.
- Per-learner quantities: never free parameters (Gate B). ACT-P0
  (primary): lambda_i pinned at 1; initial state z0_ic = u_i + v_c with
  u_i amortized by a recognition network over the learner's full
  conditioning window and v_c a free population per-KC offset. ACT-P1
  (extension): adds amortized scalar lambda_i, full-window only. Both
  variants must clear the no-growth twin; real-data claims use the
  richest variant that stayed silent.
- Readout and loss: frozen 1PL bank, Bernoulli forecast NLL,
  two-stage training (calibrate and freeze the bank, then fit dynamics
  and recognition heads).
- Fabrication read (the CG1 statistic): on a no-growth twin, the model-
  implied score change over 10 opportunities, population mean and p95
  per-learner (the qmirt lesson that the population mean alone hides
  per-learner fabrication).

### 2.4 MIXED estimator

MIX-L (gate-then-rate ladder):

1. Existence gate = PAS-G verbatim (shared machinery, one
   implementation).
2. On gate-passing KCs (and gate-passing D2+ slices for E-M2), fit the
   bounded-exponential theta(n) = m - (m - theta0) exp(-r (n-1)) by
   Bernoulli NLL against frozen b_j, parameter sharing as in E-M1/E-M2.
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
| #2 item-to-KC mapping (absent in core) | New fixed lookup buffer (Q-matrix) threaded through data and model calls; `collate_adapter_items` is extended with a parallel `kc_ids` field padded and masked exactly like `item_ids` (the vendor report's stated contract gap). No learned KC embedding table in v1. |
| #3 item parameters (`item_params`, `item_key_dim` pattern) | The frozen 1PL bank feeds `Binary2PLDecoder` unchanged (difficulties frozen, ability path free). No per-KC alpha/beta structure in v1, so no new decoder inputs. |
| #4 loss (`CombinedLoss`, per-KC weights) | Prediction NLL home, binary BCE via the existing decoder `nll`; no ordinal penalty, no per-KC class weights in v1 (binary responses). |
| #5 anchoring (item-axis only) | Deliberately NOT extended. A4 introduces no learnable KC table, so no KC-axis anchoring primitive is needed; that gap is A5's problem and is recorded as such. |

New modules (names final; no code in this document):

- `kt-mirt/src/kt_mirt/growth/__init__.py`
- `kt-mirt/src/kt_mirt/growth/qmatrix.py` (Q-matrix buffer, expansion policies, pure-anchor counts)
- `kt-mirt/src/kt_mirt/growth/kc_data.py` (kc_ids-bearing collate bridge extending `core.realdata`)
- `kt-mirt/src/kt_mirt/growth/slices.py` (slice construction, opportunity indexing, density strata, saturation flags)
- `kt-mirt/src/kt_mirt/growth/curves.py` (PAS-C)
- `kt-mirt/src/kt_mirt/growth/bank.py` (AFM-style calibration, freeze, cohort split)
- `kt-mirt/src/kt_mirt/growth/gate.py` (PAS-G: M0, M1a, M1b, odd/even evaluation, batched fits)
- `kt-mirt/src/kt_mirt/growth/rate.py` (MIX-L stage 2, bounded-exponential fits, misfit flag)
- `kt-mirt/src/kt_mirt/growth/tracker.py` (PAS-N1, PAS-N2 on the vendored core)
- `kt-mirt/src/kt_mirt/growth/recognition.py` (full-window amortized u_i, lambda_i heads)
- `kt-mirt/src/kt_mirt/growth/active.py` (ACT transition and two-stage trainer)
- `kt-mirt/src/kt_mirt/growth/synth.py` (generator and the four twins)
- `kt-mirt/src/kt_mirt/growth/battery.py` (section-4 battery; doubles as the A6 core)
- `kt-mirt/src/kt_mirt/growth/report.py` (verdict assembly, posture-disagreement matrix)
- Scripts: `kt-mirt/scripts/a4/{prep_kdd,prep_ednet,run_synth,run_bed,run_battery,make_report}.py`
- Tests: `kt-mirt/tests/test_growth_{qmatrix,slices,gate,rate,active,synth,battery}.py`

Convention reminders: no runtime import from `deep_irt`; results files
never embed dataset copies; datasets stay under `data/` and kt-irt
caches.

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
sibling wherever the dynamics allow (matched-twin discipline).

- SYN-NG (no-growth twin; both density profiles): r_c = 0 for all c,
  everything else identical. The ACTIVE posture must stay silent (CG1)
  and the gate must not fire above its nominal false-positive rate
  (CG2).
- SYN-KG (known-growth twin; both density profiles): the standard
  generator above. The PASSIVE posture must detect at the
  density-predicted reliability (CG3, CG4): the Spearman-Brown-style
  response-sampling arithmetic is run on the generator config FIRST to
  predict split-half reliability and detectability, and observed values
  must match the prediction within tolerance, in both directions.
- SYN-NS (non-standard-shape twin; KDD density): growth exists but the
  bounded-exponential family is wrong. Half the growing KCs follow step
  growth (jump of 0.8-1.5 logits at a learner-specific changepoint
  drawn uniform on opportunities 3-8, insight-like); half follow
  dip-then-recover (a 0.5 logit dip over opportunities 2-4, then
  bounded-exponential rise, interference-like and non-monotone). The
  MIXED ladder must fire the existence gate AND fire the misfit flag
  AND withhold r (CG5). This is the twin that catches what the
  parametric family misses.
- SYN-SAT (saturated twin; KDD density): theta0 raised so every per-KC
  start rate is at least 0.90 (q1 at least 0.88), true growth present
  with the same r_c. The existence gate must FAIL, reproducing the E2d
  lesson that saturation is a data-property limit; the pipeline's
  verdict must be "insufficient dynamic range", driven by the
  saturation flag, not a silent null (CG6).

Seeds: 5 generator seeds per config for all slice-based machinery;
neural arms (trackers, active, frozen-null, probes) run on generator
seed 0 with 3 model seeds (compute concession, pre-registered).

## 4. Null and audit battery

The battery is one module (`growth/battery.py`) applied uniformly to
synthetic and real beds; it doubles as the A6 certification core. Nine
arms.

1. Permutation null (primary reference for PAS-C and PAS-G). Scheme:
   permute each learner's full interaction order, then rebuild slices
   and opportunity indices. This destroys within-slice trend while
   preserving slice membership, item multisets, marginal rates, and
   (on EdNet) multi-tag co-occurrence, so it is coherent on both beds
   with one construction. B = 199 replicates for per-KC empirical p
   (BH q = 0.05 across KCs); B = 999 for the bed-level pooled statistic
   (empirical p floor 0.001). B may be raised, never lowered.
2. Static twin. Synthetic: SYN-NG. Real beds additionally get a
   rate-matched i.i.d. resample twin: within each slice, responses
   redrawn i.i.d. Bernoulli(slice mean), schedule and items intact.
   The full pipeline must return null on it (RB1). Trackers refit on
   the resample at 1 seed (compute concession, pre-registered).
3. Untrained/frozen-encoder null (Ding and Larson arm 1). Freeze the
   tracker's encoder at seeded random initialization; train only the
   readout head(s) against the frozen bank; run the identical pipeline.
   Synthetic requirement CG7: the trained tracker must beat the frozen
   null on KC-level growth recovery by a pre-registered margin, and the
   frozen null must not itself clear the passive-detection gate.
   Real-bed requirement RB2: trained must beat frozen on held-out NLL
   (seed-clustered), and tracker growth claims are made only if the
   trained-vs-frozen per-KC growth-profile correlation is below 0.9;
   otherwise the tracker read is labeled not attributable to learned
   state and excluded.
4. Single-KC-drill contamination probe (Ding and Larson arm 2).
   Construct synthetic drill sequences: one KC practiced 100 times all
   correct (oracle) and all incorrect (anti-oracle), other KCs
   untouched; replay through the fitted tracker; contamination ratio =
   max over off-target KCs of |displacement| divided by on-target
   |displacement|, averaged over 5 drilled KCs and both directions.
   Pass at ratio <= 0.10 with off-target absolute movement inside the
   no-growth p95 band (CG8). PAS-N1 is expected to fail (that failure
   is a finding, not a bug); PAS-N2 passes by construction and the
   probe verifies the implementation.
5. Order-invariance stress (Ding and Larson arm 3). Hold within-KC
   order fixed, permute cross-KC interleaving (5 reshuffles); per-KC
   tracker trajectories must be stable: median per-KC trajectory
   correlation >= 0.8 and displacement sign flips on fewer than 10% of
   KCs (CG9). PAS-C, PAS-G, MIX-L, and PAS-N2 are invariant to this
   permutation by construction (they see only the slice); the stress
   binds on PAS-N1 and on ACT's recognition inputs.
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
   agreement tolerance 0.15.
8. Truncation stress. Re-estimate beta_c and r at opportunity cutoffs
   {5, 10, 20, full}. Claimed rates must be truncation-stable (rank
   correlation of r_c between cutoff 10 and full >= 0.8), and no
   rate-heterogeneity magnitude claim is made unless the IQR of r moves
   less than 25% between consecutive cutoffs (the Lee et al. 75%
   inflation warning made operational).
9. Seed clustering. Neural fits at 3 model seeds minimum; slice-based
   results across 5 generator seeds on synthetic. All confirmatory
   statistics reported with seed treated as a cluster; claims must be
   sign-consistent in every seed and significant in the seed-pooled
   analysis.

Ordering discipline (positive-control-first): no real-bed null or
verdict is interpreted before the known-growth twin certifies the
pipeline (CG3), because the trajectory program's own history shows a
null read from an uncertified metric is uninterpretable.

## 5. Pre-registered thresholds and kill conditions

### 5.1 Synthetic certification gates (ALL must pass before any real-bed interpretation)

| Gate | Twin / target | Pass condition |
|---|---|---|
| CG1 active silence | SYN-NG, ACT-P0 and ACT-P1 | Model-implied score change over 10 opportunities: population mean <= 0.01 proportion-of-max; p95 per-learner <= 0.01 (KDD density) / <= 0.02 (EdNet density) |
| CG2 gate false positives | SYN-NG | Bed-level pooled p > 0.01; per-KC BH discoveries <= 2% of KCs |
| CG3 passive detection, KDD density | SYN-KG-KDD | Bed pooled p < 0.001; >= 60% of unsaturated KCs discovered at BH q = 0.05; rank corr(r_hat_c, r_c) >= 0.7 over discovered KCs; observed split-half within 0.15 of the density-predicted value |
| CG4 passive detection, EdNet density | SYN-KG-EDNET | Bed pooled p < 0.001; KC-level pooled rate rank corr >= 0.6; per-learner detection reported but NOT gated (pre-registered exclusion: no per-learner claims at median-2 density, ever) |
| CG5 non-standard shape | SYN-NS | Pooled gate fires (p < 0.001); misfit flag fires on >= 80% of non-standard KCs; <= 20% of non-standard KCs receive an unflagged rate |
| CG6 saturation refusal | SYN-SAT | Pooled gate p > 0.05; saturation flag marks >= 95% of KCs; pipeline verdict is "insufficient dynamic range" |
| CG7 frozen-encoder margin | SYN-KG-KDD, trackers | Trained KC-growth-profile rank recovery >= 0.6; trained minus frozen >= 0.2; frozen must not clear CG3's detection bars; decertify the tracker if frozen is within 0.1 of trained |
| CG8 contamination | trained trackers | Ratio <= 0.10 and off-target movement inside the SYN-NG p95 band |
| CG9 order stress | trackers | Median per-KC trajectory correlation >= 0.8; sign flips < 10% of KCs |
| CG10 direction audit | trackers on SYN-KG | Violation fraction <= 10% |

If the gate fires robustly on SYN-SAT (CG6 inverted), the twin or the
gate is broken; that is a certification failure requiring diagnosis,
not a power bonus.

### 5.2 Real-bed licensing conditions, KDD-KTracedSkills (primary bed)

Unsaturated subset = KCs with calibration-cohort correct rate <= 0.85
(computed on the calibration cohort so no selection on analysis data).

- RB1 (in-situ fabrication check): the rate-matched i.i.d. resample
  twin must return null through the whole pipeline (pooled gate
  p > 0.05; active population score change <= 0.01).
- RB2 (frozen-null distinguishability): as battery arm 3.
- RB3 (Tier 1, existence): bed-level pooled gate p < 0.01 on the
  unsaturated subset, surviving RB1. Pre-registered internal
  replication: the gate failure rate on the saturated subset is
  predicted to be materially higher (direction reported, not gated).
- RB4 (Tier 2, KC rates): KC-level split-half >= 0.8 AND truncation
  stability (battery arm 8) AND misfit flag silent on the claimed KCs.
- RB5 (Tier 3, per-learner rates): per-slice split-half >= 0.7 within
  D2+ strata AND truncation stability. If RB5 fails at every stratum,
  the verdict is the E2c pattern at per-KC granularity: existence yes,
  individual rates unreliable at this density. That is a reportable
  honest outcome, not a program failure.

### 5.3 Real-bed licensing, EdNet KT1 (population corroboration only)

- Pre-registered exclusions: no per-learner claims and no G1-style
  causal reads at median-2 density with the bundle confound,
  regardless of results.
- RB-E1: pooled gate p < 0.01 plus RB1-equivalent resample null =
  population corroboration achieved. Failure = corroboration dead;
  EdNet then contributes nothing to G2.

### 5.4 XES3G5M (conditional bed)

Enters only after the stage-0 triage script has produced its density,
saturation, decoupling, and anchor numbers; the KDD thresholds apply
with the density strata recomputed from its own quartiles. No number in
this design is assumed for it.

### 5.5 Posture-disagreement matrix (diagnostic, pre-registered readings)

| Pattern | Reading |
|---|---|
| Active fires, passive flat | Fabrication suspect; active quarantined; recheck CG1 config parity |
| Passive fires, active flat | Gain family or gating too rigid; consult E-M3 misfit flags |
| Passive fires, mixed rate unreliable | Existence-only claim (Tier 1); the E2c pattern |
| Passive and mixed fire, shapes disagree (E-M3 fired) | Growth real, family wrong; report blockwise profile, withhold r |
| All three agree | Strongest claim tier available at that density |

Active-only signals are never claimed as growth.

### 5.6 Kill conditions and revision budget

Revision budget: at most TWO revision rounds on the synthetic
certification matrix; each revision may fix mechanisms and bugs but may
never loosen a threshold; every revision is a LEDGER entry stating what
changed and why.

- K1 (posture machinery): CG1/CG2/CG5/CG6 still failing after the
  budget kills the failing posture program-wide for G2. CG1 failure in
  both ACT variants = active posture dead (fabrication), itself a
  reportable diagnostic. CG3 failure = the ladder cannot detect at
  realistic density; G2 retreats to synthetic certification plus an
  honest data-property verdict (the exhaust-venues discipline).
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
- K6 (in-situ fabrication): RB1 fails on a real bed = every readout on
  that bed is quarantined until diagnosed; no claim of any tier.

## 6. Bed plan and Q-matrix policy

Order: synthetic first (nothing real is interpreted before CG1-CG10
pass), then KDD-KTracedSkills (primary), then EdNet (population
corroboration), then XES3G5M when landed. TIMSS is structurally not a
growth bed (single occasion) and SLAM is absent and untagged; both are
excluded from A4.

| Bed | Role | Data | KC model and expansion policy (restating avenue_map section 4 for A4) |
|---|---|---|---|
| Synthetic | Certification | `growth/synth.py` | As generated; 1-to-1 (KDD-matched) and multi-tag (EdNet-matched) |
| KDD Cup 2010 Algebra 2008-2009 | PRIMARY (density: median 8, frac >= 10 = 0.423; decoupling and anchors pass) | `data/kdd/algebra_2008_2009_train.txt`, full 8.9M steps | KC model = KTracedSkills (515 KCs; the triage's recommended model, avoiding SubSkills' co-scheduling failure and Rules' catch-all-bug artifact). Item = step (Problem Hierarchy + Problem Name + Step Name); response = Correct First Attempt. Steps are practice opportunities, never compared 1:1 with item-level beds. Multi-KC steps: the interaction enters every tagged KC's slice and increments each tagged KC's opportunity counter (explicit multi-KC opportunity policy, stated because this is the Algebra file, not near-1-to-1 Bridge). Per-KC attribution claims only on KCs with >= 3 pure steps (85.2% qualify). Saturation policy: per-KC flags at 0.85; primary verdicts on the unsaturated subset (48.3% of KCs) |
| EdNet KT1 | Population corroboration ONLY (median-2 density; bundle confound) | `EdNet-KT1/` per-user files + `EdNet-Contents/contents/questions.csv`; seeded samples: 50k users for slice-based, 20k for trackers (disjoint from the 4k triage sample) | Tags as KCs (189). 1-to-many: each attempt enters all its tags' slices and increments all their counters; bundle exposure noted as a scheduling confound on co-movement (harmless to population growth, fatal to causal reads, hence the exclusions). Anchors moderate (40% of tags with >= 3 pure items): attribution-restricted. Tag ids are numeric only; no content naming without a crosswalk |
| XES3G5M | Conditional (learning-heavy, expert KC tree) | pending landed download | Question-level sequences; multi-KC leaf tags as Q-row loadings; NEVER split train/test at KC-expanded rows (leakage); follow the benchmark's own split; count pure questions per leaf KC before any attribution claim; stage-0 triage first |

## 7. Compute plan

Principles: slice-based machinery is vectorized (batched Newton steps
over hundreds of thousands of slices, torch tensors), so it runs on CPU
and goes 10-50x faster on GPU; neural fits need GPU; permutation
batteries and seed farms are embarrassingly parallel and go to SLURM
(code/guest-research partition, 2-GPU cap; CPU array jobs for
permutations). Preprocessing produces compact slice tensors (npz, order
100 MB) that are rsynced to the cluster; raw beds never leave local
disk; no credentials stored in the repo.

| Workload | Where | Wall-clock estimate |
|---|---|---|
| Build + unit tests (modules of 2.5) | local CPU | 3-5 working days agent time |
| Generator + acceptance checks, all configs x 5 seeds | local CPU | < 1 h |
| Slice-based pipeline, one pass (gate + rate + curves), synthetic or KDD | local CPU 10-30 min; local GPU minutes | -- |
| Permutation battery, per bed (B = 999 pooled + 199 per-KC) | local 4060 overnight (2-8 h) or SLURM CPU array (10 x ~1 h) | <= 1 day per bed |
| Tracker fits, synthetic (1M-token configs): PAS-N1 + PAS-N2 x {trained, frozen} x 3 seeds x 2 density configs | local 4060 + SLURM 2 GPUs | 24 fits x 20-40 min = ~8-16 GPU-h; ~1 day |
| ACT fits, synthetic: 2 variants x twins x 3 seeds | local 4060 | ~6-12 GPU-h; overnight |
| KDD trackers (8.9M steps): 2 configs x {trained, frozen} x 3 seeds + 1 resample-twin seed | SLURM 2 GPUs + local 4060 | 13 fits x 2-6 h = 26-78 GPU-h; 1.5-2.5 days at 3 concurrent |
| KDD ACT: 2 variants x 3 seeds + resample | local 4060 | 12-21 GPU-h; ~1 day |
| EdNet slice-based (50k users) + trackers (20k users) | CPU hours + ~30 GPU-h | ~1.5 days |
| Probes, stress arms, report assembly | local CPU/GPU | hours |

Sequence lengths: KDD per-learner sequences can be long; cap encoder
input at the most recent 2048 interactions for PAS-N1/ACT recognition
(cap recorded in outputs; PAS-N2 and all slice-based machinery are
uncapped since slices are short). Total calendar estimate: about 2 to
2.5 weeks from build start to the KDD + EdNet verdicts, consistent with
the avenue map's 1-2 week harness plus runs.

## 8. Deliverables and run order

Run order (each run gets a LEDGER entry with expectation before,
reality after, verdict):

- R0 build: modules + tests green (`python -m pytest kt-mirt/tests`).
- R1 generator bring-up: acceptance checks on all configs; SYN-DEV
  smoke of gate and rate (positive control first).
- R2 synthetic certification matrix: CG1-CG10 across the four twins,
  two density profiles, 5 generator seeds (slice-based) and 3 model
  seeds (neural). STOP/GO checkpoint; revision budget per 5.6.
- R3 KDD slice-based: calibration + freeze, PAS-C, PAS-G, MIX-L,
  permutation battery, resample twin (RB1), split-half, truncation.
- R4 KDD neural: PAS-N1/N2 trained + frozen-null, contamination,
  order stress, direction audit, RB2.
- R5 KDD active: ACT-P0/P1, in-situ silence check on the resample
  twin.
- R6 KDD verdict assembly: posture-by-bed matrix, tier claims,
  disagreement diagnostics; A5 go/no-go recommendation.
- R7 EdNet population leg: slice-based + trackers per section 6.
- R8 XES3G5M: triage first, then the same ladder (conditional).

Deliverables:

- D1 `kt-mirt/src/kt_mirt/growth/` + tests (battery module doubles as
  the A6 core).
- D2 `kt-mirt/_planning/a4/synth_certification.md` + JSON results (the
  CG1-CG10 matrix).
- D3 `kt-mirt/_planning/a4/kdd_matrix.md` (primary G2 verdict, three
  postures, tiers, disagreement matrix).
- D4 `kt-mirt/_planning/a4/ednet_population.md`.
- D5 PLAN/LEDGER updates and the A5 recommendation.

## 9. Judgment calls made in this design (flagged for review)

1. Gate evaluation uses an interpolative odd/even split rather than a
   forecast split (rationale in 2.2); the archaeology did not specify
   its split, so this is a design decision, not an inheritance.
2. The frozen bank is 1PL, calibrated inside an AFM-style model with a
   growth term, freezing difficulties only. Chosen to sidestep the
   discrimination-collapse mechanism entirely; 2PL is deferred.
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
   (grounded loosely on the 73-77% positive-slope triage fraction),
   the 0.10 contamination ratio, the 0.8/0.7 reliability bars
   (qmirt's 0.80 bar and the archaeology's too-low 0.17/0.19 bracket
   them), the 10% direction-violation bar, and the 0.9
   frozen-profile-correlation bar. All are pre-registered here so they
   cannot drift after results exist.
8. Learner counts and seed splits (5 generator / 3 model seeds, neural
   on generator seed 0 only, resample-twin trackers at 1 seed) are
   compute concessions, stated as such.
