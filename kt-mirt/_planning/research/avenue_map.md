# kt-mirt avenue map

Synthesis of the eight-topic research sweep, 2026-07-17. Sources are the eight
reports in `kt-mirt/_planning/research/` (qmirt-archaeology,
trajectory-dynamics-archaeology, psi-kt, graph-kt, interpretable-kt,
transfer-forgetting, growth-methodology, datasets) plus the adversarial
claim-verification pass run over them.

Flag conventions. [CONFIRMED] / [REFUTED] / [UNCLEAR] mark claims that went
through the verification pass. [UNVERIFIED] marks claims sourced from search
snippets or secondary summaries only (network egress was blocked in most
report sessions). Unflagged claims from the two internal archaeologies are
internal record from this lab's own docs, not independently re-verified;
treat them as hypotheses to re-test on the kt-mirt harness, not inherited
results. Where verification REFUTED a detail in a report, the corrected fact
is stated here and the error is listed in section 6 so it does not propagate.

---

## 1. State of the field

**Signed cross-KC influence is near white space.** Directionality is well
established in graph-KT (GKT, SKT, PKT, PSI-KT, COMMAND all use directed
prerequisite structure), but sign, meaning facilitation versus interference,
is essentially absent. PSI-KT's edges resolve direction only, a_ik in [0,1]
with reciprocal normalization; it does not model negative transfer anywhere
[CONFIRMED via convergent secondary evidence]. The classical toolkit cannot
express it either. AFM has per-KC intercept and slope, no cross-KC term
[CONFIRMED]; LFTA searches for transfer relations but its hypothesis space is
structurally non-negative [method CONFIRMED; note the Thorndike/Faculty-
Transfer framing attributed to it was REFUTED, that framing appears only
loosely in Pavlik, Yudelson and Koedinger's 2015 IJAIED paper, which built
CPFA precisely because Q-matrix models cannot capture negative or asymmetric
transfer]. Exactly one KT paper claims signed transfer, LTKT (Tsinghua
Science and Technology 2024), and its primary text was unreachable in every
session [UNVERIFIED, must be read before any novelty claim is finalized].
HawkesKT has a signed cross-effect mechanism but its sign validation is a
qualitative visualization as far as could be determined [UNVERIFIED]. No
paper reporting an externally corroborated negative-transfer instance between
two named KCs in real logs was found (a gap statement bounded by search
depth, not proof of absence).

**Readout auditing is confirmed white space.** Across eleven interpretable-KT
lines, only Deep-IRT quantitatively checks a readout against external
references (Pearson r = 0.56 item analysis, 0.58 traditional IRT, 0.69 PFA;
the report's 0.60/0.45 figures were REFUTED and are corrected here), and its
own authors admit the model inherits the DKT reconstruction problem, ability
moving against the observed response [CONFIRMED]. No surveyed paper combines
external validation with a faithfulness or stability audit, and none runs a
pre-registered null against a provably uninformed encoder. A 2024 survey
states outright that evaluation methods for explainable KT are lacking. This
is the program's constraint (c) territory and its sharpest defensible claim
of white space.

**Growth is anchored at the population level, open at the individual and
per-KC level.** The population anchor is about 0.1 log-odds (about 2.5
accuracy points) per practice opportunity, from 1.3M observations across 27
ITS datasets [CONFIRMED], replicated on MATHia and Campus AI data
[UNVERIFIED]. But the estimate is sequence-length sensitive, truncation to
the first 10 opportunities inflates rate heterogeneity by about 75%
[UNVERIFIED, Lee et al. 2026], and no per-skill rate exists for
Junyi/ASSISTments-class logs at all. Individual-level slope reliability needs
much denser data than group-level slope [UNCLEAR, and the cited two-timepoint
literature contains a direct published rebuttal showing elapsed time span,
not wave count, drives individual-slope precision]. Simulated-learner
validation has two established criteria, population-curve match and blinded
Turing-like behavioral match; neither tests individual-trajectory growth
fidelity. G2 as scoped has no established test to borrow.

**The lab's own archaeology is the main methodological asset.** The parked
Q-MIRT thread already built and exercised what the field lacks: matched-null
paired contrasts, a four-part confound battery, per-learner 95th-percentile
tail reads on null twins, shuffle-order and reverse-direction controls,
practice-gating rules, three identification lemmas (free asymptote, free
persistence, gain-form misfit), the positivity condition, and the Gate C
anchor recipe (frozen cross-loading discriminations recoverable via marginal
ML with at least 3 pure anchor items per KC). The trajectory thread adds the
existence-gate-then-parametric-rate ladder, split-half reliability, the
positive-control-first discipline, and ceiling normalization against
seed-to-seed refit noise. All internal record, none applied at per-KC
granularity, none on real transfer data.

**Constraint nuances the sweep sharpened.** (a) The fabrication risk
concentrated specifically on the per-learner transfer-multiplier trait
(gamma), which failed certification under every estimation posture; ability
level z0 and rate multiplier lambda passed under full-window amortized
encoding. The constraint is narrower than "shared encoders are unsafe."
PSI-KT's learner-specific transfer trait is exactly the object that failed.
(b) The rate failure is a data property (KDD split-half 0.17 binary, 0.19
graded, both at about 80% top-category), not a coding artifact; per-KC
decomposition was never attempted, so it remains a genuine untested escape
route. (c) The discrimination collapse under joint dynamic calibration was
mechanism-resolved into optimization budget, marginalization benefit, and
cohort ability spread, with the fix being marginal ML on a spread cohort,
then freeze. This is the anchoring recipe kt-mirt already believes in, now
with a mechanism.

**Data reality.** Expert prerequisite edges exist only in Junyi15 (the 2015
DataShop release) [UNCLEAR, corroborated by secondary sources only, and
annotation may cover only 370 of 722 exercises per the apparent origin
paper]. The 2020 Junyi Kaggle release does NOT carry the prerequisite field,
contrary to the dataset report [REFUTED, Info_Content.csv has hierarchy IDs
only; prerequisite and map coordinates live in the older release's
junyi_Exercise_table.csv]. Misconception labels exist only in Eedi. Item-KC
arity is many-to-many nearly everywhere. No raw correct-rate statistic was
located for ANY dataset, only model accuracies, so the saturation bet is
currently unmeasured on every candidate bed.

---

## 2. Avenues

### A1. Explicit-route signed transfer state (G1, primary)

**Mechanism.** Port the qmirt-certified skeleton onto kt-mirt: per-KC scalar
state z_{t,c}; the only cross-KC route is a fitted zero-diagonal signed
matrix G, driven by practice indicators only (never responses); own-gain is
ceiling-gated, (ceiling - z)+ for positive transfer, (z - floor)+ for
negative; mu pinned at 0, rho pinned at 1 on monotone beds, OU with fixed mu
on non-monotone beds; per-learner z0 and lambda amortized by the encoder over
the full conditioning window; per-learner transfer multiplier gamma pinned at
1. Readout through frozen anchored 2PL/GPCM heads calibrated by marginal ML
on a spread cohort.

**Architecture.** Surgery required. The existing LSTM/transformer/DKVMN
hidden state cannot be the knowledge carrier (passive transfer mimicry,
constraint a); it is demoted to a recognition network producing (z0, lambda).
New work: a structured per-KC transition module holding z, a two-stage
training loop (calibrate bank, freeze, fit dynamics), and the gate harness.
Decoder heads are reusable as-is.

**Beds.** Synthetic first, D = 3 (already certified internally) scaled to
D = 5 and 8 (the scale-up was built but never executed, an infra failure not
a science failure). Real: KDD Cup Bridge-to-Algebra 2006-07 (about 1.01
KCs/step, near one-to-one, ITS scheduling likely gives decoupled practice
slots) [UNVERIFIED figures]; Junyi15 for the external direction check.

**Failure modes.** Positivity failure on real curricula (co-scheduled
practice makes G unidentifiable; the internal threshold was at least 0.75
source-only slots for a clean read, and no real bed's decoupling fraction has
been measured). Gain-form misfit laundering into G (Lemma 3) when the real
gain law is unknown; the mismatched-generator robustness arm was flagged
repeatedly as the live threat and never run. Any per-learner multiplier
reintroduced anywhere resurrects phantom transfer.

**Synthetic gate.** The full ported battery: matched-null paired contrast on
the score scale, four-part confound battery (correlated-no-transfer,
co-scheduling, shuffle-order, reverse-direction), per-learner p95 tail on the
null twin, state-inert measurement items, plus a new mandatory
mismatched-generator arm (fit the model's gain family against a different
true gain family and require the null twin to stay clean) before any real
run.

**Cost.** Build 3-5 weeks (transition module, two-stage loop, gate port,
D-scaling). GPU light: these models are tiny, D = 8 synthetic runs on CPU or
the 4060; real-leg seed farms are embarrassingly parallel on SLURM within the
2-GPU cap. Most build of the G1 trio, most derisked scientifically.

### A2. Misconception-channel negative transfer on Eedi (G1, flagship for the negative half)

**Mechanism.** Interference via shared misconception, the one a priori
negative-transfer mechanism with labels independent of any model. Practicing
KC A while repeatedly committing misconception m (Eedi tags each distractor
with a misconception ID) predicts elevated m-consistent wrong-option
selection on KC B items whose distractors carry m, relative to
exposure-matched controls. The estimand is signed by construction and does
not ride on a fitted K x K matrix.

**Architecture.** Two legs. A model-free first leg (conditional
logistic/event-history analysis on option choices) needs no neural core at
all and can be certified with the null battery alone. The neural leg uses the
existing NRM head (Eedi is 4-option multiple choice, categorical response is
NRM's home) plus a practice-gated per-learner per-misconception evidence
channel with no free multiplier traits. The NRM a_k/c_k leverage-split idea
is dead (Fisher information near-symmetric, internal record), so no
leverage-based shortcut; treat NRM parameters as jointly anchored.

**Beds.** Eedi only. Access via the Azure blob zip and scale (about 119k
students, roughly 2e7 answers) are [UNVERIFIED, no live HTTP check
succeeded]; license unconfirmed.

**Failure modes.** Adaptive item routing creates selection effects (who sees
which B items is not random). Weak students pick all wrong options more, so a
base-rate ability confound can mimic misconception coupling; the null must be
exposure-AND-ability matched. Misconception labels are expert annotation of
options, not ground truth of causal interference. Saturation is a minor risk
here (categorical responses carry more range than binary).

**Synthetic gate.** A multiple-choice generator with injected
misconception-coupling edges; recovery scored as signed-edge F1 (PKT-style)
plus the matched-null contrast; on real data, pre-registered
exposure-and-ability-matched null, KC-pair identity permutation, and seed
robustness.

**Cost.** Data wrangling heavy, acquisition risk real. Model-free leg cheap
(CPU, 1-2 weeks once data is in hand). Neural leg moderate. Highest novelty
of any avenue: no paper was found using Eedi misconception labels for
cross-KC negative transfer (bounded by search depth).

### A3. Signed cross-effect readout audit on the stock core (G1, cheap probe)

**Mechanism.** Add a low-rank signed KC-pair excitation term (HawkesKT-shaped
exponential-decay kernel) to the decoder input of the existing prediction-
trained core, then audit whether the fitted signs survive certification. The
contribution is the audit, not the mechanism; this is the natural extension
of the measurement-audit paper into the cross-KC readout.

**Architecture.** Minimal. Decoder-side addition, anchored item path per
constraint (c), existing encoders untouched.

**Beds.** Any many-to-many bed; Junyi15 for the direction cross-check; avoid
EdNet (bundle exposure confound).

**Failure modes.** This is the stable-and-wrong disease's natural habitat.
Shared-encoder passive mimicry is unmitigated here, only audited. Lone
negative cells in a weakly identified K x K matrix are exactly the fitting
artifact the transfer-forgetting report warns about. The honest prior is that
this avenue FAILS certification, and that failure is itself a publishable
audit result.

**Synthetic gate.** Injected-edge recovery with known signs, uninformed-
encoder null (the audit paper's own tool), KC-identity permutation, seed
clustering.

**Cost.** Cheapest build (1-2 weeks on the existing pipeline), moderate GPU
(full-bed training runs). Highest interpretive risk. Run only with the full
battery attached.

### A4. Per-KC growth existence ladder (G2, primary)

**Mechanism.** The untested escape route from constraint (b), executed with
the trajectory program's validated ladder at per-KC granularity. Per
learner-KC slice: (1) existence gate, held-out predictive improvement of a
dynamic-ability model over a constant-ability null (ground-truth-free, worked
at p about 5e-11 on KDD aggregate, correctly failed on the saturated
recoding); (2) only where the gate passes, a bounded-exponential rate read
(affine-invariant in theta, so encoder scale does not bias it); (3)
split-half reliability of the rate; (4) truncation stress test (re-estimate
at multiple opportunity cutoffs; the 75% heterogeneity-inflation warning);
(5) the 0.1 log-odds/opportunity anchor as a prior expectation, with the
caveat it comes from mastery-managed ITS platforms, not classroom logs.

**Architecture.** The existing core suffices. Per-KC theta readouts already
exist (DKVMN/Deep-IRT style); the build is the gate harness and slicing, not
the model. Anchored item path required before any magnitude claim.

**Beds.** Chosen by triage (section 3, stage 0), because no raw correct rate
is known for any bed. Candidates in rough order of promise: XES3G5M (5.5M
interactions, 865 leaf KCs, learning-heavy K-12 math) [UNVERIFIED figures],
KDD Cup per-KC slices (aggregate is saturated at about 80%, but per-KC
distributions may not be uniform), Junyi15 (long-tail warning, most students
under 50 interactions), Eedi. Avoid single-pass streams (EdNet-KT1 was the
wrong data-generating process outright).

**Failure modes.** Per-KC slices as saturated as the aggregate (the whole
bet). Per-learner-KC opportunity counts too short for any rate read (the
truncation artifact then manufactures spurious heterogeneity). Circularity if
the skill ID doubles as the model's item key (the ASSISTments lesson; use
problem-level item keys wherever the bed allows).

**Synthetic gate.** A positive control with genuinely DISTINCT per-KC rates
(fixing the internal generator flaw that made the old M3 rate test
inconclusive), recovery required before any real null is believed; a
static-ability twin on which the existence gate must fail.

**Cost.** 1-2 weeks harness, cheap fits, embarrassingly parallel on SLURM.
The cheapest decisive scientific result available to the program.

### A5. Frozen-anchor digital twin with growth-beyond-noise certification (G2, measurement leg)

**Mechanism.** The trustworthy-scale half of G2. Calibrate the item bank by
marginal ML on a measurement-regime cohort with adequate ability spread,
freeze it, verify at least 3 pure anchor items per KC (Gate C), then read
per-learner per-KC theta trajectories against the frozen scale. Certify
growth beyond noise with (i) the existence gate, (ii) ceiling normalization,
judging trajectory stability against the seed-to-seed noise floor of
independent full refits (the SLAM protocol reached 0.885/0.826 of ceiling for
difficulty/discrimination; porting it to trajectories is unproven), (iii) the
Spearman-Brown density arithmetic, which predicts growth-score reliability
from measurement density with no model fitting and yields deployment rules of
the form "N times the reference-item density to reach 0.80," and (iv) a
reconstruction-direction audit, ability must not move against the observed
response, the failure Deep-IRT admits.

**Architecture.** Mostly exists (anchored/separated paths from the
measurement-audit work). New: anchor-selection tooling, the reliability
arithmetic, and occasion-to-occasion scale-stability checks borrowed from
longitudinal IRT (a literature not yet imported into neural KT by anyone,
per the growth-methodology report).

**Beds.** Whatever A4 licenses. Do not build twins on a bed that failed the
existence gate.

**Failure modes.** Anchor scarcity (real Q-matrices may not have 3 pure items
per KC; count first). Item drift and re-exposure effects (the internal drift
detector was judged underpowered and queued for refinement). Vertical-scaling
drift across occasions. High between-KC correlation degrades cross-loading
attribution (ratio recovery 0.833 at r = 0.6, internal record).

**Synthetic gate.** A no-growth twin must read flat at the per-learner p95
tail, not just the mean; a known-growth twin must read growth at the
reliability the density arithmetic predicts; seed stability normalized by
the refit ceiling.

**Cost.** Moderate build (2-3 weeks on top of A4), light GPU.

### A6. Portable certification battery (serves G1 and G2, publishable methodology)

**Mechanism.** Package the certification machinery as a bed-agnostic module
in kt-irt: matched-null paired contrasts, the confound quartet, per-learner
tail statistics on null twins, practice-gating enforcement checks, the
existence gate, split-half and ceiling normalization, the opportunity-order
permutation null (no EDM paper applies a shuffle null to KT opportunity
sequences, so this piece is itself a contribution), and a
Griffiths-Tenenbaum-style causal-support statistic computed from held-out
consecutive interactions with an explicit null-graph comparison (the PSI-KT
template, the strongest beyond-null validation precedent found).

**Architecture.** No model surgery. Assembled incrementally inside A4 and A1;
small standalone overhead to make it bed-agnostic and documented.

**Failure modes.** Scope creep; the battery is scaffolding for the two
claims, not a product in itself, until a claim survives it.

**Gate.** Self-hosting: every component must catch a planted defect (a
fabricating trait, a saturated bed, an ill-posed metric) that the archaeology
already documented.

**Cost.** 1-2 weeks incremental. High reuse value; also the thesis's
certification spine.

---

## 3. Build order (ranked recommendation)

**Stage 0, bed triage (before any avenue; about 1 week, CPU only).** Compute
directly from raw files: per-KC and overall raw correct rates (located for NO
bed in this sweep), per-learner-KC opportunity distributions, KC-pair
decoupling fractions (the positivity condition), pure-anchor counts per KC,
and category usage for Eedi options. Verify Eedi and Junyi15 access and
license terms live. Every avenue's bed choice currently rests on unmeasured
numbers; this is the single highest-information-per-cost step available.

**1. A4, per-KC growth existence ladder (A6's core built inside it).** First
scientific deliverable. Reasoning: cheapest decisive result; it directly
tests the program's central untested bet (the per-KC escape from saturation);
the existing core suffices, so no surgery risk; a negative result is cheap
now and expensive later (it would redirect G2 before twin engineering); and
its harness is most of A6, so the battery falls out nearly free.

**2. A1, explicit-route signed transfer.** The G1 primary. Reasoning: its
synthetic side is the most derisked asset the program owns (certified 9/9
seeds at D = 3 internally), so the port has low science risk; the two open
blocks (D-scaling, mismatched-generator arm) are exactly what a first
milestone should close; the real-data leg waits on stage-0 decoupling numbers
because positivity, not modeling, is the likely binding constraint on real
curricula. Start the synthetic port in parallel with stage 0.

**3. A2, Eedi acquisition now, science third.** Reasoning: highest-novelty
claim in the map (a certified negative-transfer instance with an a priori
mechanism and independent labels), but it has the largest data risk and needs
the battery to exist first. Acquire and triage the data during 1-2; run the
model-free leg once A1's battery is ported; the NRM leg after.

**4. A5, frozen-anchor twin.** Strictly conditional on A4 passing on at least
one bed. Building the trustworthy scale before knowing growth exists on the
bed would repeat the field's error in reverse.

**5. A3, readout audit.** Cheapest but riskiest read; run only with the full
battery, and frame it as measurement-audit-paper territory (either a
certified quick win or a demonstrated failure, both publishable in that
line). Deliberately last among G1 routes so its likely failure cannot be
mistaken for G1's failure.

**6. A6 standalone write-up** once battle-tested by 1-3.

Why not A1 first overall: A1's real-data feasibility hinges on stage-0
positivity numbers, and its synthetic result is already believed internally.
A4 buys genuinely new information fastest, and its outcome gates more
downstream decisions (bed choice for G2 AND the learning-heavy bed
identification G1 wants) than any other single run.

---

## 4. Q-matrix policy per bed

The sanctioned one-to-many device is the model-side Q-row loading: vector
discriminations with support restricted to the item's Q-row (the internal qm3
emission), which kills rotation freedom by pinning each KC's scale to its own
pure-anchor set. It requires at least 3 pure anchor items per KC (Gate C) and
degrades at high between-KC correlation. Where anchors are missing, do not
make per-KC attribution claims on that bed.

| Bed | Native arity | Policy |
|---|---|---|
| Junyi15 (DataShop 2015) | ~1 exercise to 1 topic-KC (722 exercises, 41 KCs) [UNCLEAR, secondary-corroborated] | 1-to-1 at topic level. The only bed with an external prerequisite graph; check on export whether annotation covers all 722 or only 370 exercises. |
| Junyi 2020 (Kaggle) | exercise to 4-level hierarchy IDs; NO prerequisite field [report claim REFUTED] | 1-to-1 at a declared hierarchy level. Never claim external-graph validation on this release. Different cohort/window than Junyi15; never merge. |
| ASSISTments 2009 | many-to-many; raw file ~25% duplicated rows [UNCLEAR, strongly corroborated] | Use skill_builder_data_corrected_collapsed.csv only (one row per student-problem). For prediction, compound labels may stand as distinct KCs; for G1/G2 readouts, expand to Q-row loadings and check anchor counts. State the exact file in every result. |
| ASSISTments 2012 | many-to-many | Same policy as 2009. Exact URL path is soft; confirm from the site root. Scale figures vary by filtering; name the file. |
| ASSISTments 2017 | action-level longitudinal STEM-outcome data [CONFIRMED] | Exclude. Not a KT log natively; pyKT-style repackagings discard its design. |
| XES3G5M | question to multiple leaf KCs of an expert tree [UNVERIFIED] | Keep question-level sequences; represent multi-KC as Q-row loadings; NEVER split train/test at KC-expanded rows (leakage); follow the benchmark's own split. Count pure questions per leaf KC before attribution claims. |
| Eedi | question to multiple ontology nodes, 4 levels [UNVERIFIED] | Fix one ontology level (leaf) as the KC layer and state it. Misconception IDs are a separate channel, never KC tags. Responses are categorical; use the NRM head, not binarized correctness, when the distractor identity matters. |
| EdNet KT1 | multi-tag per question, average arity unknown; bundle-level exposure | Avoid for G1 causal reads (bundle confound) and for G2 rate reads (single-pass DGP, internal finding). If used at all, 1-to-many with a bundle covariate. |
| KDD Cup 2010 | steps, multi-KC with ~~-separated opportunity counts; Algebra I 05-06 ~1.35 KCs/step, Bridge 06-07 ~1.01 [UNVERIFIED] | Prefer Bridge-to-Algebra 2006-07 as the near-1-to-1 bed. Algebra I requires an explicit multi-KC opportunity policy. Steps are practice opportunities, not test items; never compare item counts 1:1 with item-level beds. |

General rule for the map: fix and state an explicit item-to-KC expansion
policy per bed before any run; the duplicate-row AUC inflation on ASSISTments
2009 is the documented cost of not doing so.

---

## 5. Must-cite references

G1 precedent and structure:

- PSI-KT, Zhou, Bamler, Wu, Tejero-Cantero, ICLR 2024, arXiv:2403.13179
  [identity, OU transition, amortized inference all CONFIRMED]. The
  beyond-null validation template (causal support vs null graph, four-axis
  trait battery where only held-out behavioral regression counts as
  grounding); also the cautionary object, its per-learner transfer trait is
  exactly what internal Gate B killed. Note: the report's Eq. 5 description
  was partially REFUTED; the reversion-target baseline is a LEARNER-level
  trait mu^l (not a KC baseline) and the transfer trait is gamma^l_n (not
  a_n). Code is AGPL-3.0; do not derive code from it.
- LTKT, Xu et al., Tsinghua Science and Technology, DOI
  10.26599/TST.2024.9010201 [UNVERIFIED]. Sole claimant to signed
  positive-and-negative transfer in KT; must be read in primary text before
  any novelty statement.
- HawkesKT, Wang et al., WSDM 2021, DOI 10.1145/3437963.3441802
  [UNVERIFIED]. Nearest signed cross-KC mechanism; sign validation apparently
  qualitative; verify the CMI section first-hand.
- GKT, Nakagawa, Iwasawa, Matsuo, WI 2019, DOI 10.1145/3350546.3352513
  [partially REFUTED: two evaluation datasets, not three; sign-absence
  unverified]. Statistics-based vs learned KC graphs, typed directed edges.
- GIKT, arXiv:2009.05991 [CONFIRMED]. Given Q-matrix bipartite graph + GCN,
  structure never validated against ground truth; the field's default.
- SKT, Tong et al., ICDM 2020, IEEE 9338285 [UNCLEAR, stated PDF link is
  dead]. Dual similarity/prerequisite propagation, unsigned.
- PKT, Annabi and Nguyen, ICDL 2023, arXiv:2402.01672. Synthetic-simulator
  F1 validation of discovered structure, plus the explicit admission that
  public datasets lack structural ground truth.
- COMMAND, Chen, Gonzalez-Brenes, Tian, EDM 2016. Joint prerequisite-graph +
  student-model discovery, validated against textbook ordering.
- Prerequisite Relation Learning survey, ACM Computing Surveys, DOI
  10.1145/3733593 [UNVERIFIED]. Ground-truth prerequisite structure is
  pedagogy-dependent; caution for any single-true-graph validation plan.

Classical growth/transfer toolkit:

- LFA/AFM, Cen, Koedinger, Junker, ITS 2006, DOI 10.1007/11774303_17
  [CONFIRMED]. Per-KC intercept+slope, no cross-KC term; "mixed-effects" is
  imprecise for the base 2006 model (fixed-effects logistic; iAFM adds
  random slopes).
- PFA, Pavlik, Cen, Koedinger, AIED 2009 [UNCLEAR, primary unreachable].
  Success/failure-count logistic; per-KC, no pair terms.
- LFTA, Pavlik, Cen, Koedinger, EDM 2009 [method CONFIRMED, framing
  REFUTED]. Pairwise BIC learning-curve test building a Q-matrix; validated
  by cross-validated fit only.
- Pavlik, Yudelson, Koedinger, IJAIED 2015, 25:346-379 (CPFA). Found during
  verification: built because Q-matrix models cannot capture negative or
  asymmetric transfer; the better classical citation for signed transfer.
- Koedinger, Carvalho, Liu, McLaughlin, PNAS 2023, DOI
  10.1073/pnas.2221311120 [CONFIRMED]. The 0.1 log-odds/opportunity anchor;
  wide initial-performance variance, narrow rate variance.
- Simpson, Norberg, Fancsali, EDM 2024 (MATHia replication) and Beauchesne
  et al. 2026, arXiv:2604.03246 (Campus AI) [both UNVERIFIED]. Replications.
- Lee et al., L@S 2026, arXiv:2605.01690 [UNVERIFIED]. Truncation to 10
  opportunities inflates rate-heterogeneity IQR by 75%; mandatory stress
  test for any slope estimate.

Readout auditing and interpretability:

- Deep-IRT, Yeung, EDM 2019, arXiv:1904.11738 [equations and
  reconstruction-admission CONFIRMED; report's correlation values REFUTED,
  corrected: 0.56 item analysis / 0.58 IRT / 0.69 PFA]. The central
  IRT-flavored KT model and its self-admitted directional failure.
- Yeung and Yeung, L@S 2018, arXiv:1806.02180 [venue REFUTED as cited in the
  report; it is L@S 2018, not CSEDU]. Reconstruction and waviness problems,
  loss-side regularizers.
- Scruggs, Baker, McLaren, ICCE 2020, arXiv:1910.12597. The lone external
  post-test validation of a KT knowledge estimate; never became field norm.
- LPKT, Shen et al., KDD 2021, DOI 10.1145/3447548.3467237. Explicit
  learning-gain gate; validation is one qualitative case study, the closest
  and still-insufficient G2 analog.
- Bai et al., Applied Intelligence 2024, arXiv:2403.07279. Third-party
  statement that xKT evaluation methods are lacking.
- On the Interpretability of DKT, arXiv:2101.11335 [UNVERIFIED, unread
  primary]. Untrained RNNs match trained DKT on some diagnostics; aggregate-
  ability warning.
- Does KT Interpretability Support Teacher Decision Making, arXiv:2511.02718
  [UNVERIFIED, unread primary]. Read before finalizing the validation story.
- pyKT, arXiv:2206.11460, NeurIPS 2022 D&B. Benchmark hygiene; use its
  protocols to keep prediction numbers comparable.
- DKT-Forget, Nagatani et al., WWW 2019, DOI 10.1145/3308558.3313565
  [report's feature description partially REFUTED: the sequence time gap is
  KC-agnostic, not same-KC]. Engineered forgetting features, no pair-level
  transfer.

G2 methodology:

- Käser and Alexandron, IJAIED 2024, DOI 10.1007/s40593-023-00337-2
  [UNVERIFIED]. Turing-like simulated-learner test; population-level only,
  no individual-trajectory criterion exists.
- Koedinger, Matsuda, MacLellan, McLaughlin, SimStudent evaluation methods
  [UNVERIFIED]. Population learning-curve match as fidelity criterion.
- Wang and Nydick, longitudinal IRT overview, DOI 10.3102/1076998619882026
  [UNVERIFIED]. Scale-stability diagnostics to import for A5.
- Parsons and McCormick 2024 + Brandmaier, Lindenberger, McCormick 2024
  (Dev. Cogn. Neurosci.) [UNCLEAR, pair of paper and rebuttal]. Two-timepoint
  individual-slope limits; the rebuttal shows elapsed time span, not wave
  count, drives precision; design lesson for measurement-occasion spacing.

Datasets:

- Junyi15, PSLC DataShop dataset 1198 [UNCLEAR] + Chang, Hsu, Chen, EDM 2015
  (annotation origin; 370 exercises annotated). The one expert prerequisite
  graph.
- XES3G5M, NeurIPS 2023 D&B [UNVERIFIED]. Learning-heavy bed with an expert
  KC tree.
- Eedi / NeurIPS 2020 Education Challenge, arXiv:2007.12061 and 2104.04034
  [UNVERIFIED]. Misconception-per-distractor labels, the negative-transfer
  resource.
- EdNet, arXiv:1912.03072 [UNVERIFIED]. Scale, bundle confound.
- Griffiths and Tenenbaum causal-support formalism (via PSI-KT Appendix
  A.7.3). The independent behavioral statistic for A6.

From project memory, outside this sweep: Growing Pains, arXiv 2604.12843,
recorded as the nearest thesis-level prior-art analog; re-check its overlap
with G1/G2 before the thesis framing is fixed.

---

## 6. Open risks the sweep could not resolve

1. **LTKT unread.** The only prior signed-transfer claimant in KT. If its
   mechanism and validation are stronger than the abstract suggests, A1/A2
   novelty framing changes. Highest-priority read.
2. **Saturation unmeasured everywhere.** No raw correct rate was located for
   any candidate bed; only model accuracies, which are not the same number.
   The per-KC escape route (A4) is a bet on numbers nobody has computed.
3. **Positivity on real curricula unknown.** If real logs co-schedule KC
   pairs (decoupling fraction near 0), A1's real leg is unidentifiable
   regardless of modeling quality. Stage-0 must measure this before A1's
   real leg is promised.
4. **Mismatched-generator robustness never run.** All internal transfer
   certifications are matched-form, best-case identifiability. The "live C3
   threat" is still live; A1's gate adds the arm, but it has never passed.
5. **D-scaling never executed.** The D = 5/8 harness crashed on infra
   (output-token limits in background agents), not science. Unknown whether
   certification survives realistic KC counts, let alone K in the hundreds;
   low-rank G factorization at real K is unvalidated.
6. **No negative-transfer ground truth exists anywhere.** Junyi edges are
   positive prerequisite/similarity only. Any G1 negative claim
   self-certifies through the battery plus the Eedi misconception channel;
   there is no external benchmark to fall back on.
7. **Per-KC growth has zero direct evidence either way.** The archaeology
   only shows the aggregate rate fails on saturated binary logs; per-KC was
   never attempted by anyone, internally or in the literature found.
8. **HawkesKT sign validation, and the two interpretability critiques
   (arXiv:2101.11335, 2511.02718), unread at primary depth.** All three bear
   directly on whether A3-style readouts can be trusted at all.
9. **PSI-KT referee record unretrieved** (Cloudflare block). Reviewer-known
   weaknesses of the closest methodological neighbor are unknown.
10. **Junyi15 access and annotation coverage.** DataShop gating unconfirmed;
    prerequisite annotation may cover 370 of 722 exercises. Eedi's Azure
    blob and license likewise unverified. Both are single points of failure
    for their avenues.
11. **Report errata now corrected here** (do not propagate from the topic
    reports): Junyi-Kaggle 2020 has no prerequisite field; Deep-IRT's
    correlations are 0.56/0.58/0.69; Yeung and Yeung 2018 is L@S, not CSEDU;
    GKT's conference version reports two datasets; LFTA does not contain the
    Thorndike/Faculty framing; DKT-Forget's sequence gap is KC-agnostic.
12. **Rotation freedom stays out of scope only by construction.** Any
    multi-KC readout must use the pure-anchor pinning device (Gate C) or it
    silently reopens the MIRT rotation question the program excludes.
13. **Network egress was blocked in most report sessions.** A large share of
    external facts above are snippet-sourced; anything marked [UNVERIFIED]
    that becomes load-bearing in a paper must be re-verified against primary
    text first.
