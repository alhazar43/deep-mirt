# DRL-MAIRT Path A. Adaptive Assessment with a Deep IRT Belief State

Date, 2026-06-04.

## 1. Positioning

Path A treats the project as **computerized adaptive testing (CAT) with a
deep-IRT belief state**. The action is the next assessment item, the
observation is the student's response to that item, and both live entirely
inside the assessment-side data that ma-irt is already trained on. The
"recommendation" is not a downstream course or a career label, it is the
next question the testlet should serve.

Path A is the strongest answer to the one-sided data constraint surfaced by
the user. The constraint is that response data exists but logged
recommendation outcomes do not. Path A defines the action and the response
on the same side of the loop, so no recommendation log is needed at any
stage of training or evaluation. The reward is intrinsic psychometric
efficiency, computable from quantities the GPCM decoder already exposes
(Fisher information at theta_hat, posterior entropy reduction, expected
score gain). Path A is also the cleanest match for the existing literature,
the four publishability hooks identified in
[`drl_mairt_evidence.md`](drl_mairt_evidence.md) all sit naturally inside a
CAT framing.

What Path A is **not**. It is not a course or curriculum recommender, it is
not a vocational pipeline, and it is not a tutoring intervention study.
Those use cases are addressed by Path B (CDM remediation with a Q-matrix
skill readout) and Path C (off-policy implicit-reward learning), which are
written up separately.

## 2. Theoretical Foundations

### 2.1 IRT measurement and GPCM

The classical reference is Lord 1980 (*Applications of Item Response Theory
to Practical Testing Problems*, Erlbaum), with the standard textbook
synthesis in Hambleton, Swaminathan & Rogers 1991 (*Fundamentals of Item
Response Theory*, Sage). Under a unidimensional IRT model, a student's
ability is a scalar latent trait theta, each item is parameterized by
discrimination alpha and one or more difficulty parameters beta, and the
probability of any observed response is a function only of theta and the
item parameters.

The ma-irt decoder is a GPCM head (Muraki 1992, *A Generalized Partial
Credit Model, Application of an EM Algorithm*, Applied Psychological
Measurement 16(2), 159 to 176). The GPCM models polytomous ordinal
responses across K categories using K-1 step-difficulty thresholds. The
category response function is

```
P(Y = k | theta, alpha, beta_1..K-1) = exp( sum_{j=1..k} alpha * (theta - beta_j) )
                                       / sum_{m=0..K-1} exp( sum_{j=1..m} alpha * (theta - beta_j) )
```

with the empty inner sum equal to zero by convention. Path A treats this
decoder as the measurement model. The encoder produces a belief over
theta, and the GPCM head turns that belief into a likelihood over
candidate responses for any item in the bank. The likelihood is what makes
intrinsic information rewards computable in closed form.

### 2.2 Information-theoretic item selection

The psychometric anchor for Path A's reward is **Owen 1975** (*A Bayesian
Sequential Procedure for Quantal Response in the Context of Adaptive
Mental Testing*, JASA 70, 351 to 356). Owen formalized adaptive testing as
sequential Bayesian inference, with each item chosen to minimize the
posterior variance of theta after the response is observed. Modern
neural-CAT methods inherit this framing. The reward in Path A is built
from four classical item-selection criteria.

**Maximum Fisher Information (MFI), Lord 1980.** For dichotomous IRT,
Fisher information at theta is

```
I(theta, item) = alpha^2 * P(theta) * (1 - P(theta))
```

For GPCM, the per-item Fisher information is (Muraki 1993,
*Information functions of the generalized partial credit model*, Applied
Psychological Measurement 17, 351 to 363)

```
I_GPCM(theta, item) = alpha^2 * ( E[X^2 | theta] - (E[X | theta])^2 )
                    = alpha^2 * Var( X | theta )
```

where X is the ordinal score in {0..K-1}. MFI is asymptotically optimal
once theta_hat is near the truth, and is the standard adult-CAT operating
point.

**KL Information, Chang & Ying 1996** (*A Global Information Approach to
Computerized Adaptive Testing*, Applied Psychological Measurement 20,
213 to 229). KL information selects items by

```
KL_item(theta_hat) = integral over theta of  P(X | theta) * log( P(X | theta) / P(X | theta_hat) )  d theta
```

KL info is more robust than MFI at the cold start because it integrates
over a region around theta_hat rather than evaluating only at the point
estimate. Path A uses KL for the first three to five items, then switches
to MFI.

**Maximum Posterior Weighted Information, van der Linden 1998** (*Bayesian
Item Selection Criteria for Adaptive Testing*, Psychometrika 63, 201 to
216). MPWI weights Fisher information by the current posterior over
theta,

```
MPWI_item = integral over theta of  pi(theta | history) * I(theta, item)  d theta
```

This is the Bayes-optimal selection criterion under a quadratic loss and
collapses to MFI in the limit of a delta posterior.

**Maximum Likelihood Weighted Information, Veerkamp & Berger 1997**
(*Selecting the Maximum Information Item in the Sequential Adaptive Test
Procedure*, Applied Psychological Measurement 21, 357 to 368). MLWI is
the likelihood-weighted analogue,

```
MLWI_item = integral over theta of  L(theta | history) * I(theta, item)  d theta
```

useful when a prior is intentionally diffuse.

The Path A reward is anchored on MFI (since ma-irt exposes alpha, theta,
and the GPCM probabilities directly) with KL as the cold-start
substitute.

### 2.3 The deep encoder hidden state as a sufficient statistic

A standard concern raised by psychometric reviewers is whether a deep
encoder hidden state is a *measurement* at all, as opposed to a
prediction. Khajah, Lindsey & Mozer 2016 (*How Deep is Knowledge
Tracing?*, EDM 2016) make this distinction explicit, deep KT models
optimize next-step prediction loss, classical IRT optimizes a measurement
likelihood. The two objectives do not have to coincide.

The route out of this concern is the architecture pattern in **Yeung 2019
Deep-IRT** (*Deep-IRT, Make Deep Learning Based Knowledge Tracing
Explainable Using Item Response Theory*, EDM 2019). Yeung routes the deep
encoder state through a parametric IRT head whose outputs are theta,
alpha, beta, and forces the decoder likelihood to be the IRT response
function. The deep hidden state then plays the role of a sufficient
statistic for theta given the response history, and the downstream
inference is psychometrically legible.

`ma-irt` already follows this pattern. The DKVMN encoder produces an
attention-summarized student state, the IRT parameter extractor maps it
to theta / alpha / beta, and the GPCM decoder is the response function.
This is documented in
`ma-irt/models/components/irt.py` and is the project's central
architectural commitment. For Path A this means the belief state used by
the policy is psychometrically anchored by construction, the policy
consumes theta and (alpha, beta) of candidate items, both of which have
the standard IRT interpretation. The "is this a measurement" question
becomes a calibration question (Section 2.5) rather than an architectural
question.

### 2.4 Item exposure control

Any CAT system that selects items by information criteria will
over-expose a small set of high-information items at any given theta.
This is not a theoretical curiosity, it is the dominant operational risk
in deployed CAT, and it is mandatory reporting in every psychometrics
venue. Path A's exposure-control toolkit draws on four references.

**Sympson & Hetter 1985** (*Controlling Item Exposure Rates in
Computerized Adaptive Testing*, Proc. Military Testing Association),
maintains a target maximum exposure rate r_max per item by attaching a
probabilistic acceptance gate to the information-maximizing item. Sampled
control parameters are calibrated on simulated test administrations to
hit the rate target.

**Stocking & Lewis 1995/1998** (*A New Method of Controlling Item
Exposure in Computerized Adaptive Testing*, Applied Psychological
Measurement 22, 57 to 75), conditional Sympson-Hetter, exposure rates are
controlled conditional on theta strata so high-ability and low-ability
items are not over-served at their respective tails.

**Randomesque, Kingsbury & Zara 1989** (*Procedures for Selecting Items
for Computerized Adaptive Tests*, Applied Measurement in Education 2,
359 to 375), pick uniformly at random from the top-N
information-maximizing items. Cheaper than Sympson-Hetter and equally
effective for small banks.

**a-stratified, Chang & Ying 1999** (*A Global Information Approach to
Computerized Adaptive Testing*, Applied Psychological Measurement 23,
211 to 222), and Chang, Qian & Ying 2001 (*a-stratified Multistage CAT
Design with Step-by-Step Considerations*, Educational and Psychological
Measurement 61, 720 to 735), the item bank is stratified by
discrimination alpha into K bands, and the test is administered in
phases that draw from increasing alpha bands so high-discrimination items
are reserved for the later, better-calibrated phases. This both controls
exposure and improves theta estimation when starting from a diffuse
prior.

Path A includes either randomesque or Sympson-Hetter from day one (the
choice is driven by bank size, see Section 3). a-stratified is added as
an ablation factor.

### 2.5 Validation when ground truth is unobserved

Real CAT data does not contain true theta. The validation toolkit
therefore consists of internal consistency tests rather than ground-truth
comparisons.

**Simulation-based calibration, Talts et al. 2018**
(*Validating Bayesian Inference Algorithms with Simulation-Based
Calibration*, arXiv 1804.06788). SBC draws theta from the prior,
simulates a response sequence, runs the posterior inference, and checks
that the rank of theta_true within the posterior samples is uniform
across draws. Non-uniform rank histograms diagnose miscalibrated
posteriors. For Path A, SBC is the headline calibration check of the
ma-irt belief over theta.

**Person-fit l_z, Drasgow, Levine & Williams 1985**
(*Appropriateness Measurement with Polychotomous Item Response Models
and Standardized Indices*, British Journal of Mathematical and
Statistical Psychology 38, 67 to 86). l_z standardizes the log-likelihood
of an observed response string under the fitted IRT model. Values below
-2 flag aberrant responding (cheating, careless responding, multiple
proficiency states), which validates that the model's residuals look
like its theoretical residuals.

**Classification consistency, Livingston & Lewis 1995**
(*Estimating the Consistency and Accuracy of Classifications Based on
Test Scores*, Journal of Educational Measurement 32, 179 to 197). If the
CAT is used to assign a discrete proficiency category, classification
consistency is the probability that two independent administrations
assign the same category. Reported as a Cohen kappa.

**Cross-form rank-order correlation**. Train two policies on the same
ma-irt encoder seed but with two different RL seeds, administer each to
a held-out simulated cohort, and report Spearman correlation between the
two theta_hat trajectories. A high correlation indicates that the
inferred theta is not an artifact of a particular policy sample path.

**Marginal reliability**. The standard psychometric scalar,

```
rho_xx = 1 - E[ SE( theta_hat )^2 ] / Var( theta_hat )
```

reported per dataset and per simulated cohort.

These five together form the validation gate. The first three are
mandatory for the v1 paper, the last two strengthen it.

## 3. Practical Implementation

### 3.1 The ma-irt online step API

Path A requires that ma-irt expose a true online interface, where the
encoder state can be carried forward step by step and the decoder can
score candidates against the current state without writing the
memory. This is the H1 milestone in
[`drl_mairt_synthesis.md`](drl_mairt_synthesis.md). It is the first
deliverable for the entire program and is gated on a numerical parity
test against the existing whole-sequence forward.

Surface, named to match the synthesis,

```
EncoderDecoderModel.step(state, q_t, r_t)             -> new_state
encoder.forward_with_state(state, q_t, r_t)           -> new_state, summaries
decoder.compute_logits_from_state(state, q_t)          -> logits, probs
StepState dataclass: { value_memory, theta, alpha, beta, attention, t, q_history }
```

The parity test is the gating test, iterated `step` across a sequence of
length T must equal `forward(full_seq)` on logits, probs, theta, alpha,
beta, attention to a numerical tolerance of 1e-5 single precision and
1e-8 double precision. This test lives in `ma-irt/tests/` and is part of
CI. No Path A code lands until parity passes.

### 3.2 The IRTBridge surface

The boundary between ma-irt and the policy is documented in
[`drl_mairt_synthesis.md`](drl_mairt_synthesis.md) Section H2. Path A
inherits the contract without restating it, the policy sees the
`StepBundle` (StepState plus candidate features alpha, beta, expected
score, KL contribution, Fisher contribution), and emits a discrete item
index. The bridge is implemented in `deep-mirt-rl/src/bridge/` and
imports nothing from the policy side, the policy is free to use the
bridge but never reaches into ma-irt internals.

### 3.3 Baseline ladder

The Path A baseline ladder, in execution order. Each rung is added only
if the next-better rung shows a measurable gap.

1. **Random selection.** Uniformly random item from the bank. Floor.
2. **Popularity selection.** Most-frequently-administered item conditional
   on test position. Detects whether any deep state is contributing.
3. **KL at cold start, MFI mid-test.** Per Chang & Ying 1996 and Lord
   1980 respectively. Switch happens at posterior std < tau (default
   tau = 0.5 in standard normal theta scale). Strong classical baseline.
4. **Sympson-Hetter or randomesque wrapper** around step 3. Exposure
   control. Mandatory for psychometrics venues, so this is the first
   "submission-ready" rung.
5. **BanditCAT.** Thompson sampling over Fisher information per Mukherjee
   et al. 2024 (*BanditCAT and AutoIRT, Machine Learning Approaches to
   Computerized Adaptive Testing and Item Calibration*, arXiv 2410.21033).
   The strongest non-deep CAT baseline.
6. **theta-only DQN, CaRReL replication.** A DQN whose state is theta_t
   only. The negative control. This is hypothesis H1 from Codex's
   plan, the deep belief state must beat this.
7. **Full-state DQN.** The Path A v1 reference, DQN over the full
   StepBundle from Section 3.2.
8. **PPO.** Only if a measurable gap remains above full-state DQN.
   Otherwise PPO is reported as a non-result in the ablation section.

The ladder is executed in order. Per the evidence synthesis, the gap from
step 5 to step 7 is the expected publishability margin. If step 7 does
not beat step 5 robustly, PPO is unlikely to help, and the v1 paper
becomes a calibration / methodology paper rather than an RL paper.

### 3.4 Reward composition

Per Owen 1975 the dominant reward term is information gain. Path A v1
uses

```
R_t = w_info   * Fisher_info( theta_hat_t, item_t )
    + w_unc    * ( H( theta | history_{<t} ) - H( theta | history_{<=t} ) )
    + w_exp    * expected_score_gain
    - w_repeat * repeat_penalty
```

with weights `w_info = 1.0`, `w_unc = 0.5`, `w_exp = 0.25`,
`w_repeat = 1.0` as starting values, all four components logged
separately per the discipline in
[`drl_mairt_evidence.md`](drl_mairt_evidence.md). Repeat penalty applies
when item_t has been administered earlier in the current session.

**Learning gain is excluded in v1.** The simulator's predicted
correctness change under counterfactual item selection is the simulator's
own forward model, and the policy is trained against that same
simulator. Including learning gain in the reward creates a closed loop
where the policy maximizes a quantity the simulator was already
predicting, which is the failure mode flagged by both Codex's
proposal and the evidence synthesis. Learning gain returns only in Path C
under an off-policy estimator.

### 3.5 Validation pipeline

For each trained policy the v1 paper reports

- Sympson-Hetter exposure rates (per-item exposure r_i and bank-level
  test overlap S),
- SBC rank histograms over simulated cohorts of n = 500,
- l_z distributions on held-out simulated response sequences,
- ECI per-item (Expected Conditional Information) under the policy,
- cross-form rank-order correlation between two independent policy
  rollouts,
- marginal reliability rho_xx,
- classification consistency kappa, if a threshold decision is included.

Exposure and SBC are non-negotiable, the rest are added on a journal-fit
basis.

## 4. Datasets

### 4.1 Primary, ASSISTments 2009

(`assistments_2009_skill_builder`, Feng, Heffernan & Koedinger 2009.) The
de facto reviewer expectation for any KT or CAT paper. Roughly 350K
interactions across about 100 skills. ma-irt already supports this
pipeline, the loader, the GPCM head, and the IRT recovery diagnostics
have all been validated on ASSISTments. Mastery-gated within-skill
random item order gives genuine variation in administered items, which
is what makes off-policy evaluation possible later in Path C and gives
the baseline ladder a meaningful CAT signal.

### 4.2 Scale companion, EdNet KT1

(Choi et al. 2020, *EdNet, A Large-Scale Hierarchical Dataset in
Education*, AIED 2020.) About 131 million interactions across roughly
13K items. EdNet is the scale benchmark that supports two artifacts the
v1 paper needs, the held-out-item generalization experiment (publishable
hook 2) and the sample-efficiency curve that argues against PPO when
data is plentiful but the simulator-real gap is dominant.

### 4.3 Ordinal angle, NeurIPS 2020 Eedi (Task 4)

(Wang et al. 2020, *Diagnostic Questions, The NeurIPS 2020 Education
Challenge*, NeurIPS Datasets & Benchmarks.) Four-option multiple choice
with public Task 4 train/test splits. The natural K=4 GPCM target is
what unlocks publishability hook 1 (ordinal-reward CAT). Eedi is the
dataset that demonstrates the strict ordinal advantage of the MA-GPCM
decoder over binary-correctness CAT.

### 4.4 Excluded, Statics2011 and KDD Cup 2010

Both have a fixed curriculum order with very little administrative
variation. The CAT signal in such data is structurally absent, fitting a
CAT policy to a curriculum-locked dataset is a measurement of the
curriculum, not of the policy. The evidence synthesis flags both as
skip.

## 5. Comparison Matrix

The seventeen baselines, organized by family. The complete reference list
is in [`drl_mairt_evidence.md`](drl_mairt_evidence.md).

| Family | Baseline | Algorithm class | Role in Path A |
|---|---|---|---|
| Random / popularity | Random | Uniform | Floor |
| Random / popularity | Popularity | Frequency | Does deep state help at all? |
| Classical CAT | MFI (Lord 1980) | Fisher info | Standard CAT floor |
| Classical CAT | KLI (Chang & Ying 1996) | KL info | Cold-start hedge |
| Classical CAT | MPWI (van der Linden 1998) | Posterior-weighted Fisher | Bayes-optimal classical |
| Classical CAT | a-stratified (Chang & Ying 1999) | Stratified Fisher | Exposure-aware classical |
| Classical CAT | Sympson-Hetter wrapper (1985) | Exposure-controlled MFI | Submission-ready classical |
| Classical bandit | BanditCAT (Mukherjee 2024) | Thompson on Fisher | Bandit ceiling |
| Neural CAT | BOBCAT (Ghosh 2021) | Bilevel meta-CAT | Direct neural competitor |
| Neural CAT | NCAT (Zhuang 2022) | End-to-end RL CAT | Direct neural competitor |
| Neural CAT | MAAT (Bi 2020) | Attentive RL CAT | Direct neural competitor |
| Neural CAT | GMOCAT (Wang 2023) | Multi-objective neural CAT | Direct neural competitor |
| Neural CAT | CCAT | Calibrated neural CAT | Direct neural competitor |
| RL with KT state | ExRec-best (2025) | KT-sim RL | Closest positive template |
| Behavior cloning | BC over MFI | Supervised | Cheap off-policy floor |
| Negative control | CaRReL-stripped theta-only DQN | DQN | H1 negative control |
| Path A v1 | Full-state DQN with KLI cold-start + Sympson-Hetter | DQN + exposure wrapper | This paper |

PPO is run as an ablation but does not appear in the matrix until the
DQN result is settled, this matches the evidence-synthesis recommendation
to walk the ladder rather than commit early.

## 6. Publishability Hooks

The four hooks from
[`drl_mairt_evidence.md`](drl_mairt_evidence.md) are inherited in full.
One sentence each on why Path A advances the literature on that hook.

1. **Ordinal-reward CAT with MA-GPCM.** Path A computes the reward as
   GPCM-Fisher (with the Muraki 1993 formula), which is strictly more
   information per item than binary-correctness rewards under any K > 2,
   no published neural-CAT paper exploits this.
2. **Held-out item generalization.** The 80/10/10 item-bank split with
   the pointer-network scorer is novel against BOBCAT / NCAT / MAAT /
   GMOCAT, all of which assume a closed item set, Path A reports the
   held-out 10% as either a primary metric or a Section 6 ablation
   depending on the open scoping decision in Section 9.2.
3. **Separated ability pathway as policy input.** The
   `separate_theta = true / false` ablation is a built-in test of whether
   the right policy state is the pure-ability summary or the
   item-conditioned interaction summary, nobody else has this lever
   because nobody else has a separated ability head.
4. **Cross-simulator validation.** Train the policy inside the
   DKVMN-based ma-irt and evaluate it inside a Transformer-based ma-irt
   trained on the same data with a different seed, the standard
   simulator-real-gap mitigation that the published CAT literature
   uniformly ignores.

## 7. Risks and Mitigations Specific to Path A

### 7.1 Circular reward from simulator-only training

**Risk.** The policy is trained inside an ma-irt-based simulator. If the
reward depends on quantities the simulator itself is predicting (such as
predicted correctness change after an action), the policy can exploit
simulator artifacts. **Mitigation.** Use GPCM Fisher information as the
dominant reward term. Fisher info is decoder-derived but
simulator-invariant in a specific sense, it depends only on the
parametric form of the response function and on (theta_hat, alpha,
beta), all of which the policy treats as observations from the encoder
rather than as forward predictions from a transition model. Excluding
learning gain in v1 is the architectural side of this mitigation.

### 7.2 Cold-start error

**Risk.** Early in a session, theta_hat is a draw from a diffuse prior.
MFI evaluated at a wrong theta_hat is uninformative or worse, the policy
locks onto items whose Fisher information is high near the wrong theta.
**Mitigation.** Use KL information for the first 3 to 5 items. KL info
integrates over a region around theta_hat (Chang & Ying 1996) and is
empirically robust to early miscalibration. Optionally hot-mix with a
UCB term on per-item posterior variance for the first item only.

### 7.3 Item exposure

**Risk.** Without an exposure controller, the bank degrades into a small
working set of high-information items, security and fairness violations
follow, and the policy is not deployable. **Mitigation.** Wrap the policy
in randomesque (top-N=5 from day one) or Sympson-Hetter (target r_max =
0.2, calibrated on a simulated cohort of n = 1000 administrations).
Exposure rates are a reported metric, not an internal tuning knob, the
paper shows distributions, not only means.

### 7.4 Measurement invariance across simulated cohorts

**Risk.** The architectural claim that the deep encoder state is a
measurement (Section 2.3) is defensible only if the inferred theta
behaves invariantly across subgroups. If the simulator generates two
synthetic cohorts with different latent-trait priors and the policy
performs differently across them, the paper cannot claim measurement
status. **Mitigation.** Run DIF (differential item functioning) or
score-based DIF (Mantel-Haenszel and logistic regression DIF, Holland &
Wainer 1993) over simulated subgroups before claiming the deep state is
a measurement. Subgroups defined by (latent prior mean, latent prior
variance, response noise level) at a minimum. DIF passes are reported as
a calibration-section table.

## 8. Ten-Week Implementation Plan

Milestone-level, mapped to the H1 to H4 numbering in
[`drl_mairt_synthesis.md`](drl_mairt_synthesis.md). Weekly granularity is
indicative, the gating is the milestone, not the calendar.

| Week | Milestone | Deliverable |
|---|---|---|
| 1 | H1.a | `EncoderDecoderModel.step` and `StepState` on the DKVMN encoder. Parity test scaffolding. |
| 2 | H1.b | Parity test green to 1e-5 / 1e-8 tolerance across ASSISTments, EdNet KT1 subset, Eedi. Microbenchmark CI test (p95 latency budget per the synthesis doc). |
| 3 | H2.a | `deep-mirt-rl/` sibling repo skeleton with `vendor/ma-irt/` submodule and `IRTBridge` smoke test. |
| 4 | H2.b | StepBundle, candidate scorer, MFI, KLI, MPWI policies as registered baselines. |
| 5 | H3.a | Sympson-Hetter and randomesque exposure wrappers. BanditCAT. BC. CaRReL-stripped theta-only DQN. |
| 6 | H3.b | Full-state DQN baseline. First end-to-end training run on the SimStudent backed by frozen ma-irt. |
| 7 | H4.a | Reward components instrumented (Fisher, entropy reduction, expected score, repeat penalty). SBC and exposure validation pipeline. |
| 8 | H4.b | Held-out item generalization experiment on EdNet KT1 (pointer-network scorer over the GPCM decoder). |
| 9 | H4.c | Cross-simulator validation (DKVMN policy evaluated in Transformer ma-irt). |
| 10 | Writeup | Ordinal-vs-binary headline comparison on Eedi. Paper draft. |

PPO is deferred. It is added as a stretch milestone after week 10 only
if Section 5's full-state DQN does not close the gap to BanditCAT.

## 9. Open Scoping Decisions

Two scoping decisions are still on the table. They are the user's calls,
this document does not pre-commit either way.

### 9.1 Ordinal-first or binary-first headline

**Option A (ordinal-first).** Eedi 2020 as the primary dataset, K = 4
GPCM as the headline reward target. The ordinal-reward CAT hook becomes
the single methodological contribution. Cleaner story, smaller reviewer
community.

**Option B (binary-first).** ASSISTments 2009 as the primary, EdNet KT1
as the scale companion, Eedi as the ordinal angle in one section. Binary
correctness drives the primary tables, the ordinal hook is hook 1 of 4
rather than the headline. Larger reviewer community, four hooks rather
than one.

The evidence synthesis prefers Option B for venue fit at IJAIED. Path A
inherits that recommendation but does not lock it.

### 9.2 Held-out item generalization, primary metric or Section 6 ablation

**As a primary metric.** The 80/10/10 split appears in the main results
table, ranked alongside test-length efficiency and exposure. This makes
hook 2 the central contribution and reframes the paper as a
generalization paper.

**As a Section 6 ablation.** The split appears in a separate ablation
section. The main tables remain test-length efficiency and exposure
control, hook 2 is one of four supporting contributions.

The synthesis-doc default is the ablation route, on the grounds that the
main tables should be familiar to CAT reviewers. The user has not
committed.

## File reference

- [`drl_mairt_background.md`](drl_mairt_background.md), Codex feasibility
  dossier.
- [`drl_mairt_recommender_plan.md`](drl_mairt_recommender_plan.md),
  Codex proposal.
- [`drl_mairt_synthesis.md`](drl_mairt_synthesis.md), plan-level
  synthesis with the H1 to H11 hybrid phasing.
- [`drl_mairt_evidence.md`](drl_mairt_evidence.md), evidence synthesis
  with the seventeen baselines and four publishability hooks.
- `docs/cleanup/_drl_research_digest.md`, raw research outputs from the
  two literature surveys.
