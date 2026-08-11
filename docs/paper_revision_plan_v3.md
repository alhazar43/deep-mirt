# Paper revision plan v3 (2026-08-11)

Plan of record for the rewrite. Self-contained: written so an outside
reader (human or AI) can follow without repo access. Supersedes
paper_plan_v2 where they conflict; the author writes the manuscript
personally; agents supply evidence and analysis only.

## 0. Context in one paragraph

The paper studies knowledge-tracing (KT) models — sequence models
trained only to predict a student's next answer — whose output layer is
shaped like item response theory (IRT): each question gets learned
parameters (difficulty b, discrimination a; per-option slopes for
multiple choice). People read those parameters out as measurement. We
show the readout fails silently and architecture-dependently, explain
why with a gradient-routing theory verified by three kinds of
experiment, repair it with a one-embedding architectural change, and
supply a truth-free audit that tells a practitioner when readouts
cannot be trusted. An earlier draft was criticized as "math and
datasets glued together with no storyline"; this plan is the storyline,
with every number already computed and committed.

## 1. The claim (final wording)

Prediction-trained KT models learn item parameters only as a byproduct
of what their sequence dynamics need. Where the readout fails depends
on the architecture and is unpredictable in practice; the failure
decomposes into two mechanisms (head displacement, universal;
representation crowding, architecture-dependent, worst under global
attention); a one-embedding separated item key — trained in, fully
online at inference — repairs both at zero accuracy cost (the shared
design keeps a small held-out log-likelihood edge, .005-.043 nats:
the quantified price prediction pays for exactly the entanglement that
corrupts measurement); and a truth-free refit-discrepancy audit
detects untrustworthy readouts without ground truth. Unpredictable
disease, one cheap trained-in cure, a meter that says when you need
it.

Vocabulary discipline: "recovery" only where truth exists (synthetic);
"concordance" on real data. Never "the audit is invariance testing"
(it is a refit-discrepancy diagnostic in the parameter-comparison
family). "Zero cost" always scoped to accuracy. No refit-style method
is ever proposed (real-time operating assumption; the audit measures,
never modifies).

## 2. Core evidence (all computed, committed)

### 2.1 Synthetic recovery grid (truth known; N=2000 learners, Q=200
items, 25 fits/cell; SH = shared item embedding feeds dynamics AND
parameter heads; SK = separate wide key feeds heads only)

| encoder | fmt | acc SH/SK | a: SH->SK | b: SH->SK | laggard |
|---|---|---|---|---|---|
| lstm | 2PL | .712/.714 | .553->.898 | .723->.957 | a |
| transformer | 2PL | .708/.713 | .373->.806 | .604->.955 | both |
| dkvmn | 2PL | .716/.715 | .752->.914 | .652->.950 | b |
| lstm | GPCM | .487/.502 | .719->.941 | .826->.965 | a |
| transformer | GPCM | .459/.492 | .438->.900 | .768->.947 | both |
| dkvmn | GPCM | .496/.502 | .879->.952 | .849->.966 | (mild) |

The lagging family FLIPS with architecture; accuracy identical; SK
lands every cell at the same plateau. Correct statistics: paired
per-seed differences, t between 3 and 47, all cells positive.

### 2.2 Mechanism evidence (three independent kinds)

(a) THEORY (gradient routing): the shared embedding receives gradients
via a parameter route and a dynamics route; the key receives the
parameter route only. Mechanism A (displacement): at stationarity the
heads sit an inverse-Fisher-amplified distance from the per-item
conditional-MLE readout — proved in a linear model; explains why
widening the shared embedding closes only ~half the gap (measured).
Mechanism B (crowding): slope information survives in the embedding
only where dynamics demand leaves slack; demand breadth orders
dkvmn < lstm << transformer. SK's stationary point makes the heads
exact conditional M-estimators given the model's own abilities —
hence the encoder-free plateau.

(b) LINEAR PROBES of trained embeddings vs truth (item-CV ridge):
difficulty decodable at .97-.99 from EVERY shared embedding even where
the trained head reads .60-.65 (displacement = misread, not lost);
slope decodable .75-.87 for lstm/dkvmn (under-extracted) but .364 for
transformer-2PL ~= its recovery .373 (genuinely absent = crowding);
under SK the value embedding is PURGED of slope (.06-.44) while
retaining a difficulty residue that orders dkvmn > lstm > transformer
— each architecture's own demand for difficulty, measured in
isolation.

(c) FALSIFICATION TESTS (the theory's own pre-registered checks):
- MLP probe on the transformer table: .14 — nothing the linear probe
  missed; the absence is real. PASSED.
- Held-out NLL between arms: REFUTED AS ORIGINALLY STATED — the
  shared arm WINS NLL slightly (.005-.043 nats, paired t 3.0-19.8,
  accuracy tied; the one tie is dkvmn-GPCM). Corrected account,
  consistent with the routing core: the shared channel optimizes a
  larger effective function class, so displacement is
  likelihood-PROFITABLE — the corruption is purchased by prediction
  training, not incidental. Reported honestly; sharpens the
  dissociation (both sides of the trade now have prices).
- DKVMN summary-key stop-gradient ablation (the flip's load-bearing
  test): CONFIRMED, stronger than predicted. On the spiraled bed,
  cutting only the gradient through the key-to-summary path lifts SH
  difficulty from .652 to .904 (past the lstm reference .723) and SH
  discrimination .752 to .853, accuracy unchanged, SK control flat.
  Bonus finding: the displacement is BED-DEPENDENT -- under
  uniform-random item draws the lag never appears (.989 unablated);
  structured (spiraled) sequencing is required to generate systematic
  gradient pressure through the path. Store: p2_toggle_sg.

DKVMN is the mechanism's fingerprint, not an anomaly: it internally
separates addressing (static key memory) from state (dynamic value
memory), the contention moves, and the failure moves with it —
while SK (an orthogonal cut: readout vs dynamics) still helps it at
full size.

### 2.3 Robustness battery (2500 fits; 7 violations x doses x
encoders x formats, paired seeds)

SK-minus-SH recovery advantage: positive in 49/50 cells (t 2.3-18.4),
never reverses, GROWS under local dependence and threshold disorder
(the two violations predicted to kill it); accuracy within .011
everywhere. The single non-significant cell is extreme exposure
starvation, where both arms fail together — the exposure law's
boundary, stated as such. The audit's discrepancy tracks true
corruption at Spearman .93 (shared arm; .72 separated — low dynamic
range because the SK head already is the estimator the audit compares
against; never print .93 unqualified) and rises with dose within
every family.

### 2.4 Real data — binary/ordinal (frozen pre-registered panel,
3 encoders x EdNet/KDD/TIMSS, 25 folds lstm/transformer, 5-6 dkvmn)

- Locations/difficulty: robust under BOTH arms everywhere (readout vs
  empirical p-values -.975; vs MML ~.73; TIMSS raw threshold ORDER vs
  the classical calibration rho .98 with all 12 classically non-modal
  items contained — the eval-time sort had erased this true signal;
  the ordered-fraction claim of the old draft is withdrawn and
  replaced by this stronger result).
- Discrimination concordance with MML: EdNet .68-.74 (best), KDD ~.44,
  TIMSS .32-.56; SK-vs-SH is encoder-conditional even here (SK wins
  EdNet on lstm +.051 CI-excl-0 and dkvmn +.015, loses on transformer
  -.040). No real-data SK-wins claim is made on this criterion.
- The audit fires on real data exactly where the synthetic analysis
  says risk lives: TIMSS 4-5x threshold (with SH~SK inter-agreement
  .976 = the stable-and-wrong signature), KDD flagged; EdNet mildest.
- MML's own defects disclosed: EdNet-2PL run converged FALSE; NRM
  anchor covers 395/4220 items; leakage vs fold-trained models.

### 2.5 Real data — the nominal (multiple-choice) story, new

Diagnosis: the original EdNet-NRM cell had 5.1 responses per raw item
parameter (4,220 items, 8 params each) vs 300-530 in every other cell
— below every published adequacy figure (De Ayala & Sava-Bolesta 10:1
examinee-to-parameter benchmark; DeMars) — and an MML anchor covering
9.4% of items. Everything weird about that cell follows; it stays in
the paper only as the labeled exposure floor.

Matched-exposure rebuild (top-250 option-rich items, 8,493 learners,
191 responses/raw-param; same routed protocol; 150 fits, 0 failures):

| encoder | anchor SH->SK | audit delta SH/SK | stability SH/SK | SH~SK |
|---|---|---|---|---|
| lstm | .437 -> .705 | .239/.325 | .762/.780 | .600 |
| dkvmn | .462 -> .696 | .203/.281 | .771/.807 | .582 |
| transformer | .264 -> .364 | .174/.285 | .660/.706 | .836 |

The anchor = per-item keyed contrast of per-option point-biserials
against a leave-one-out correctness score — the CLASSICAL
distractor-validation standard (Gierl et al. 2017 review), computed
from raw data with no model. Every encoder: the anchor sides with the
separated key; lstm and dkvmn are near-identical (+.25); the
transformer is weakest with either arm (consistent with its crowding).
FULL MATCHED-BANK BENCHMARK REBUILT (kt-irt/results/p2_matched_bench.md;
figure overleaf-sync/figures/fig_matched_anchor.pdf): per encoder,
guardrailed anchors on ednet250 -- lstm .548->.632, dkvmn .562->.609,
transformer .378->.447 (every encoder SK-positive); eedi250 lstm
.756->.723 (the level-not-direction outcome). Floors: item-majority
.584 (ednet250) / .569 (eedi250); IRT arms clear the ednet floor
(.591-.597) while direct heads do not (.563-.585); on eedi the IRT
arms sit AT the floor (.568-.573) -- the cell's value is anchor level,
not prediction, and the table says so. Classical MML: estimable on
eedi250 only (concordance .626/.623 with the full fit for BOTH arms),
and its own split-half self-agreement is r=.163 (ceiling ~.28) --
weaker than either arm's stability, the ceiling argument made
concrete; on ednet250 the campaign mirt harness fails on real option
data (documented, results/p2_matched_mml) -- the point-biserial is
the anchor of record on both banks. Guardrails NOW APPLIED (option-count floor 50, frequency-weighted
distractor means, anchor split-half reliability, paired seed stats):
the EdNet headline becomes SH .548 -> SK .632, +.094 [t(4)=10.5, 5/5
seeds], anchor reliability .857 -- passes every pre-set criterion;
the unguarded .437/.705 is retired to the record. The pre-registered
Eedi cross-platform replication EXECUTED (12,299 learners, 380k obs,
190 resp/param, 50 fits, 0 failures): replicates in LEVEL, not
DIRECTION -- both arms high (.756/.723, anchor reliability .912),
small seed-consistent SH edge (-.023). Mechanism-consistent reading:
Eedi distractors are expert-designed diagnostics (strong option
signal; displacement is inverse-Fisher, so both arms read it); EdNet
lures are weak-signal, where separation pays. Licensed sentence in
docs/framing_review/eedi_replication.md; never claim a universal
real-data SK win on options.

### 2.6 The audit, precisely

delta = 1 - Spearman(readout slopes, per-item MLE re-estimate with
abilities frozen at the model's own last-step estimates). Lineage: a
refit-discrepancy diagnostic in the parameter-comparison family —
cite Smith & Suh (2003): split-recalibration catches what fit
statistics miss (53/60 items); deployment posture = the operational
norm (drift screens recalibrate in shadow; the deployed scale is
never modified; Bock, Muraki & Pfeiffenberger 1988; FIPC). Blind
spots in print: shared corruption of abilities+parameters is
invisible; at starvation its own re-estimate is noise; it compares
two estimators and certifies neither; cross-arm comparisons are out
of scope (the shared arm self-grades: its abilities and parameters
are co-trained into coherence). Empirical calibration (from the
battery, truth known): delta <.25 -> recovery ~.8; .25-.40 -> ~.7;
.40-.60 -> ~.55; >.60 -> ~.35. On real data every cell exceeds the
synthetic alarm threshold .152; the audit provides the RANKING
(EdNet mildest, TIMSS worst).

## 3. Positioning (from the external landscape sweep)

(a) ARCHITECTURES: a 23-model code-verified survey (2019-2026) shows
the field has climbed toward separation since AKT-2020 (Rasch
difficulty embeddings everywhere) but NO model puts item parameters
behind a gradient boundary; "discrimination" in this lineage is
absent, a fixed constant, or a mislabeled difficulty scalar; and NONE
validate parameters beyond AUC. The separated key is the missing rung
of a visible six-year ladder — generalization/completion, not
invention. Honesty: five of the surveyed models share one pykt-toolkit
code block; the independent lines are AKT, DIMKT, Deep-IRT, DisKT.
Cite Vie & Kashima 2023 (plain DKT is an implicit IRT model with
parameters smeared across shared weights).

(b) EVALUATION PRECEDENTS: only four papers correlate learned
parameters against a classical estimator on real data; the two
well-executed ones (ML2P-VAE: r~.98, ECPE n=2,922; Tabak et al.
Behaviormetrika 2025: r=.88-.96, ENEM n=5,000) amortize STATIC IRT on
complete-matrix exam data — the estimator sees exactly what MML sees.
Our object (sequential, causal, sparse logs) has never been validated
at all; where regimes match (our dense static synthetic bed) SK sits
at .90-.95, inside the precedent range. The paper carries a 2x2
regime table (static/sequential x matrix/logs) placing everyone, and
adds split-sample MML-vs-MML self-agreement per dataset — the
achievable ceiling on our own data — so .68-.74 is read against the
right bar. Print the pre-registered primary in every cell FIRST, then
argue the criterion (anything else is outcome switching).

(c) PRIOR ART, full treatment.

The nearest-sounding paper — verified against the source
(arXiv:2606.14123): Yan, Tang & Shimada, "Recovering Stranded
Discrimination in Knowledge Tracing: Per-Item Bias Correction via
Empirical-Bayes Shrinkage," ECML PKDD 2026. What it actually is:
their "discrimination" is PREDICTION QUALITY (per-item AUC), not the
IRT slope parameter; nothing in the paper estimates or evaluates item
parameters in the measurement sense. Their finding: KT backbones
carry a systematic per-item logit bias — attributed, in their own
words, to "limited per-item expressivity in backbone architectures"
and post-deployment shift — and global monotone calibrators (Platt
etc.) cannot recover the lost AUC because monotone score-only
transforms preserve ranks ("a structural consequence"; recovery
"requires conditioning on item identity"). Their fix, SLC
(State-space Logit Correction): binary observations to Gaussian
pseudo-observations via Laplace/IRLS, empirical-Bayes shrinkage
through a Kalman smoother, an offset-Platt link — a POST-DEPLOYMENT
corrector, validated on AUC/NLL only, binary responses only, with
the gains concentrated on sparse items.

Construct mapping (write this into related work; disambiguate the
word "discrimination" in the first sentence that cites them):

| theirs (prediction side) | ours (measurement side) |
|---|---|
| per-item logit bias from limited per-item backbone expressivity | the contested item channel: displacement + crowding of item PARAMETERS (mechanism, probes, theory) |
| monotone post-hoc calibrators cannot recover per-item structure (their impossibility result) | why the repair must condition on item identity AT TRAINING TIME: the separated key; post-hoc paths additionally excluded by the real-time premise |
| gains concentrate on sparse items | the exposure law (battery boundary cell; the 5.1-resp/param starved cell) |
| fix = post-deployment Kalman shrinkage for prediction quality | fix = trained-in separation for measurement quality; shrinkage is precisely the SH-style medicine (stability bought at information cost) and repairs prediction, not parameters |

How to use them: cite FIRST and generously — their impossibility
result is independent prediction-side support for our central
architectural claim (per-item structure needs a dedicated per-item
pathway), and their sparse-item concentration independently
replicates the exposure law from the prediction side. Differences to
state plainly: binary-only vs our three response formats;
AUC/NLL-only validation vs parameter recovery + external anchors;
post-deployment correction vs trained-in separation under a
real-time constraint; no architecture-dependence analysis vs our
three-encoder law. The two papers are complementary sides of one
phenomenon; ours supplies the mechanism, the measurement
consequences, and the audit.

Also anticipate and answer: "Deep-IRT already split the heads" (its
key also drives addressing — no gradient boundary; discrimination is
a fixed 3.0; no validation); "SK is two-stage refit rediscovered"
(SK repairs the deployed heads themselves, online, no second
calibration pass — and frees re-estimation to serve as the audit).

(d) OPTION-LEVEL KT: Option Tracing (AIED 2021) and successors
validate options with accuracy only; the classical per-option
point-biserial standard has never been attempted there. Meeting it is
a stated contribution.

## 4. Paper structure (the rewrite skeleton)

1. INTRODUCTION. Object = a deployed, real-time KT predictor whose
   IRT-shaped readouts get used as measurement. Question = are those
   parameters trustworthy, and how would anyone know without truth?
   The arc: disease (invisible, architecture-dependent) -> mechanism
   (two proven components) -> cure (trained-in separation) -> meter
   (truth-free audit) -> real-data instantiation.
2. RELATED WORK. The pathway table (C-survey) + the 2x2 regime table
   (D-precedents) + the option-validation gap. Three paragraphs, two
   tables.
3. METHODS. One divide-by-total family, three doses of slope
   structure (2PL c GPCM c NRM; constraint table). SH vs SK with the
   gradient boundary stated. The audit defined with lineage +
   blind spots. The anchor defined with guardrails. Exposure reported
   as responses per identified parameter + option skew (convention
   stated: 6 identified vs 8 raw for K=4).
4. RESULTS, ordered by claim: (i) the dissociation at slope-dim 1
   (grid + paired stats + capacity controls: width closes ~half);
   (ii) locations are robust everywhere (GPCM rung; TIMSS raw-order
   .98 — the sort-artifact story told as an exhibit of the thesis);
   (iii) slopes are the fragile family and separation repairs them
   (NRM rung: starved cell as floor, matched cells x3 encoders,
   anchor sides with SK everywhere); (iv) the mechanism section
   (probes + theory summary + falsification outcomes INCLUDING the
   honest NLL reversal = purchased corruption); (v) robustness
   battery; (vi) real-data reading protocol: primary printed
   everywhere, audit ranking, MML ceilings, regime table; (vii) the
   decision cost (CAT invoice: the shared readout stops testing at 8
   items certifying SE .29 when truth is .69; banked + real-bank
   replication).
5. DISCUSSION. The encoder-conditional law (whatever is pooled lags);
   DKVMN as fingerprint; the purchased-corruption reading; honest
   ceiling (MML dominates where it can be run and its regime holds);
   limits (datasets/architectures not exhausted — the claim is
   failure-location-agnostic by construction; anchor is a proxy;
   audit blind spots).

## 5. Metric discipline (curation, defensible)

HEADLINE (clean, convergent): synthetic recovery with paired stats;
battery dose-response; hardened point-biserial anchor; TIMSS
raw-order .98; accuracy ties; audit as within-arm dose-response +
real-data ranking; MML concordance only with ceiling + regime table +
full per-cell reporting.
APPENDIX/DISCLOSURE (mixed-signal producers): cross-arm delta
comparisons (scope-ruled, with the self-grading explanation);
one stability construct only (across-fit), split-half where the
frozen table requires; SH~SK agreement as symptom exhibit only;
starved-NRM as one labeled floor row; person-side ability as symptom
+ the unrouted .595 disclosure; KDD non-interpretive.
OUT (kept in repo): load-axis curves, monotone-ladder wording, pooled
option correlations, "newly discovered disordered items", any
refit-style method (banned; real-time assumption).

## 6. Reviewer threats, prepared answers

1. "Precedents hit .9+ vs MML; you get .7." — The regime table +
   ceiling analysis + our .90-.95 where regimes match; the question
   is the deployed sequential object nobody has validated.
2. "Isn't the audit just item fit?" — Smith & Suh: fit statistics
   miss what recalibration-comparison catches; ours is the
   parameter-comparison family's neural analog, simulation-warranted.
3. "Deep-IRT/AKT already do this." — Pathway table: no gradient
   boundary anywhere, no learned discrimination, no validation. And
   "isn't this the ECML stranded-discrimination paper?" — different
   construct (their discrimination = per-item AUC; ours = the IRT
   slope), different side (prediction vs measurement), different fix
   class (post-deployment corrector vs trained-in separation); their
   impossibility result and sparse-item concentration SUPPORT us;
   see the 3(c) mapping table.
4. "Why not just refit?" — Refit-as-method violates the real-time
   premise; SK is the trained-in, online equivalent (its endpoint IS
   the conditional MLE), and re-estimation serves as the meter.
5. "NRM cell looks broken." — Starvation diagnosis in the
   literature's own currency + the matched rebuild x3 encoders.
6. "49/50, not 50/50." — Stated: the boundary cell is the exposure
   law, and it is the same law the battery and the starved cell
   demonstrate.

## 7. Remaining work

Compute: COMPLETE. Executed and committed: the full falsification
package (P7 passed; P1 sign-refuted with the purchased-corruption
account; P3 confirmed with bed-dependence), the Eedi pre-registered
replication, guardrailed anchors for every encoder on both banks,
direct baselines and floors on both matched banks, classical-MML
anchors/ceilings where estimable, the rebuilt benchmark table
(p2_matched_bench.md), and the anchor figure. Optional follow-ups
that remain genuinely open: the DKVMN memory-size sweep, the
gradient-geometry ordering, PIRLS as a revision-stage second
official-anchor bed if a reviewer demands one.
Writing (author): the rewrite per section 4 from the committed
evidence; every table exists; the two must-not sentences respected;
title to be chosen.
Do-not list (standing): no measurement-journal/TMLR repackaging, no
NAEP/PISA acquisition, PIRLS only if a reviewer demands it, no
width-matched real rerun, no tuned direct-NRM baseline, no new
encoders, no MIRT, no refit-style methods ever.

## 8. Source map (for verification)

Master audit: docs/results_critique.md. Matched-bank benchmark:
kt-irt/results/p2_matched_bench.md (+ stores p2_matched_direct,
p2_matched_mml, p2_toggle_sg; figure
overleaf-sync/figures/fig_matched_anchor.pdf). Theory:
docs/framing_review/theory_contention.md. Probes/exhibits:
docs/framing_review/E6-E8, P1_nll_ece.md, nrm_matched_exposure.md,
format_unification.md. External landscape + expert reads:
docs/framing_review/external_landscape.md. Battery:
kt-irt/results/p2_misspec/battery_report.md. Stores: p2_toggle,
p2_realstudy, p2_v3_arm1r, p2_realstudy_rawbeta, p2_nrm250,
p2_misspec, p2_probe_dkvmn (+ p2_dkvmn_ablation, p2_eedi250 when
jobs land).
