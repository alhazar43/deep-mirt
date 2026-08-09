# CAEAI framing review (overnight program, opened 2026-07-17)

Working dossier for the storyline criticism. The author rewrites the
paper; this document carries the diagnosis, the evidence, the framing
options with a recommendation, and pointers to everything produced
overnight. Built incrementally; each section is dated.

**The criticism.** The paper lacks "the main storyline"; it "reads like
the author is trying to glue math (theory) and random code/datasets
(empirical) to prove a point that was never there." Author's own
diagnosis: no "aha"/"wow"; GPCM and NRM feel glued; EdNet and TIMSS feel
unrelated.

**Program.** (A) seven-lens constructive reading of the draft, plan,
results inventory, venue, and the archived prior consult; (B) synthesis
into candidate framings with evidence-gap matrices; (C) judged
selection; (D) findings package + any framing-required experiments run
on the local/cluster GPUs. No paper edits.

---

## A0. First-hand spine read (coordinator, 2026-07-17 late)

Three structural faults visible from the abstract/intro/skeleton alone:

1. **The front matter sells half the paper.** Abstract + all three
   stated contributions describe the synthetic amortization study
   (recovery-as-validity, shared-vs-separated head, amortization-gap
   diagnosis) plus a CAT payoff. TIMSS, EdNet, GPCM-vs-NRM, the
   stability/agreement instruments -- the entire real-data half of the
   body (sections at tex lines 1033/1128/1178) -- are never announced or
   motivated up front. A reader reaches TIMSS and EdNet as unannounced
   excursions; hence "random datasets."
2. **A dangling promise.** The CAT simulation is promised (abstract
   line 51, contribution 3 line 93, design table ~892/922) and asserted
   in the discussion (~1381-1385), but its methods paragraph (872-880)
   and figure are commented out; the p2_cat results were archived out of
   the port keep-set. The draft currently claims results it does not
   present -- the sharpest instance of "a point that was never there."
3. **Known-incomplete tail.** Discussion and Conclusion carry
   [FULL-REWORK] markers; the stale "exposure regimes" phrase survives
   in the abstract though the exposure grid left the paper.

Working hypothesis for synthesis: the draft contains TWO internally
coherent half-papers -- (i) the synthetic amortization story the front
matter tells, and (ii) a real-data audit story (truth-free instruments:
stability, slack, cross-reading agreement, MML reference; rising
response resolution binary -> ordinal -> nominal; SH/SK as the design
contrast) that the front matter never introduces. The missing storyline
is the bridge that makes (ii) the necessary continuation of (i), and
the response-format ladder + designed-assessment (TIMSS) vs log-data
(EdNet) archetypes the principled reason for the dataset/format pairs.

---

## A1-A7. The seven lens reports (filed 2026-07-17)

Full reports in `docs/framing_review/A1_storyline-arc.md` ... `A7_prior-consult.md`.
One-line verdicts:

- **A1 storyline-arc**: the draft is a half-completed transplant between two
  paper architectures; the complete ancestor (title, disease/invoice/slack
  arc, figures) survives in `overleaf-sync/submission/`; dangling refs
  `sec:diagnostics`/`sec:downstream` would compile as "??".
- **A2 theory-empirics**: orphan theory = the refit-ladder stationarity
  result (defined, never exhibited), the Fisher operating-point claim
  (sharper than what is tested), the incidental-parameter analogy (never
  consumed). Strongest chain = NRM routing math -> reversal-bridge panels,
  term-for-term. The direct-predictor baseline is the one orphan experiment.
- **A3 aha-audit**: the paper deleted the aha it was built around and
  half-installed another without rewriting the frame; ranked candidates:
  (1) two-resolution EdNet elevated by a robustness-hierarchy frame,
  (2) the truth-free slack product (high ceiling, worst constraint fit),
  (3) the SK repair alone (current de facto spine -- exactly what was
  criticized), (4) stable-and-wrong (banned vocabulary, synthetic-only).
- **A4 psychometric**: the 2PL/GPCM/NRM pairing is PRINCIPLED -- the
  complete nested taxonomy; the unifying citation (Thissen & Steinberg
  1986) is already in the bib on the wrong sentence; Bock 1972 missing.
  TIMSS/EdNet are the two canonical archetypes (designed assessment vs
  platform log) one sentence each away from locking. The buried wow: the
  NRM head BEATS unconstrained option predictors by +.08-.12 on EdNet.
- **A5 inventory**: synthetic grid complete (72 cells); real panel frozen
  at 4 pairs; KDD is the sharpest single glue instance (an orphan column);
  extension menu costed (EdNet-GPCM/KDD-GPCM feasible but construct-caveated,
  TIMSS-2PL needs a new dichotomization rule, TIMSS/KDD-NRM impossible);
  every real-data figure is LSTM-only; the un-reported NxQ exposure grid
  exists archived; CAT data banked, driver lost (scaffold only in git).
- **A6 venue**: CAEAI rewards caution + tool + DECISION COST + fix; the CAT
  invoice is the campaign's only practitioner stake (196.8% [190,204] test
  length, +2.3pp misclassification, SK roughly halving both) -- banked in
  `docs/exposure_rerun_results.md`, absent from the manuscript. The
  SH-edges-SK-on-EdNet-NRM prediction result must be foreshadowed as the
  fix's honest boundary, or it reads as a bolted-on contradiction.
- **A7 prior-consult**: the archived revision plan lost the fight it is
  remembered for (centerpiece demotion) and won a quieter one -- its
  stability-not-recovery logic for real data IS in the draft; what was
  never adopted is its connective frame, which is a mechanical account of
  the current symptom.

---

## B. Synthesis (coordinator, 2026-07-17 night)

**The mechanical diagnosis, unanimous across lenses.** The criticism is not
about missing substance; it is the signature of a half-finished pivot. The
front matter and discussion still promise the plan-of-record paper (refit
ladder, CAT invoice, truth-free product); the body delivers the grafted
case studies (TIMSS, EdNet) that the front matter never announces; three
cross-references dangle; the one sentence that would make the decoder trio
principled (draft line 181) and the one result that would make a KT reader
sit up (+.08-.12, line 1092) are both buried. "Glue" is what a reader calls
a body under the wrong frame.

**What the material already contains, once framed.**

1. *One decoder family, not three decoders.* 2PL -> GPCM -> NRM is the
   complete nested response-format taxonomy (binary -> ordinal -> nominal);
   each rung retains strictly more of the raw response. One paragraph +
   moving the Thissen & Steinberg citation + adding Bock 1972 makes GPCM
   and NRM rungs of one ladder. The theory section already climbs this
   ladder (one vulnerable-parameter mechanism per rung); the glue feeling
   is the shift of organizing axis between sections.
2. *Two archetypes, not two random datasets.* TIMSS is the designed-
   assessment archetype (rubric-built ordinal structure; GPCM is its
   platform-native model) and EdNet is the platform-log archetype
   (dichotomization is an analyst convention the log does not force; the
   NRM reads what the convention discards). KDD is the binary control --
   say so, or drop it. TIMSS-stable vs EdNet-fragile then stops being a
   contradiction and becomes the two poles of one prediction.
3. *One finding: the readout is stratified.* Item locations port across
   design, resolution, and dataset (synthetic beta .72-.85 even under SH;
   EdNet beta design-robust .998, cross-resolution .95/.97; 31/31 TIMSS
   items keep ordered thresholds under both designs). Discrimination-family
   parameters are fragile under the shared readout and conditionally
   recoverable -- the separated key is the one lever, at unchanged
   accuracy (18/18 cells). Person ability is recoverable synthetically
   (new exhibit E1: SH .61-.96 -> SK .86-.97, SK better in all nine
   N=2000 cells) but is the weakest tier on real data across designs and
   resolutions (.18-.64) -- the tier where the synthetic-to-real gap is
   largest. The two-resolution EdNet study is the capstone: same learners,
   two readings; items agree, slopes transfer weakly, persons partially
   (disattenuated .59/.63).
4. *The KT-audience wow.* The measurement head WINS at prediction on the
   richest format: +.08-.12 accuracy over unconstrained option predictors,
   all three encoders -- Bock's 1972 motivation (wrong answers carry
   information) realized inside a KT model. Currently a subordinate clause.
5. *The practitioner stake.* The banked CAT invoice: a system reusing the
   shared head's parameters tests roughly twice as long as oracle and
   misclassifies +2.3pp at fixed length; the separated key halves both;
   repairing difficulty alone is the WORST arm at its own stop. This is
   the venue's required decision-cost beat and it resolves the draft's
   dangling promises by fulfillment.

**The three candidate framings** (all zero new training):

- **F1 -- ladder + stratified-readout core.** Framing-only; fixes both
  glue complaints; leaves the CAT promises to be deleted.
- **F2 -- F1 + the invoice restored.** The CAEAI caution+tool shape;
  dangling promises fulfilled; exhibit writable from the banked clustered
  summary (the lost simulator is only a from-scratch-replication concern).
- **F3 -- F2 + the truth-free slack product restored.** The frozen plan's
  own centerpiece; highest wow, methods-primary risk, invented-label
  tension; the TLT-flavored swing.

Coordinator recommendation going into the panel: **F2**, with F3 presented
as the author's conscious fork (the draft abandoned plan-A by default;
that abandonment should be a decision, not an accident).

---

## C. Judged selection (three-judge panel, 2026-07-17 night)

Scores (0-10): original-critic F1 7 / **F2 9** / F3 6; CAEAI-editor F1 5 /
**F2 8** / F3 6; constraints-guardian F1 8 / **F2 8.5** / F3 4. Unanimous
pick: **F2** -- the ladder + stratified-readout spine with the banked
adaptive-testing invoice restored by fulfillment.

Why F3 died: beyond the methods-primary desk risk, the guardian surfaced
that "slack" was personally ordered out of the prose by the author
(replacement: refit discrepancy) -- the branding F3 sells is the exact
vocabulary the ruling forbids. Why F1 is only the floor: deleting the CAT
promise makes the front matter honest but ends the paper on a
descriptive map, under-serving the venue's decision-cost beat.

Three amendments adopted into the recommendation ("F2+"):

1. **The axis sentence (critic).** State prediction-vs-measurement as the
   paper's axis once in the abstract/intro, and scope the invoice section
   to the measurement use in one bridge sentence. Then the capstone's
   honest boundary (SH edges SK on EdNet-NRM prediction by .002-.012) and
   the invoice (SK halves the measurement cost) become two coordinates of
   one finding: prediction is not the decision. Name the same
   discrimination parameter in both places (the a whose fragility the
   ladder establishes is the a-error driving ~85% of the length
   inflation).
2. **The staging + fence (guardian).** The author drafts the F1 spine as a
   complete paper first, then adds the invoice as ONE additive section
   written strictly from the banked artifacts (outputs/p2_cluster/
   cat_clustered.json + outputs/p2_cat*/ fold JSONs). Committed fence: the
   lost simulator is NOT rebuilt inside the paper's scope; the manuscript
   is never again in a half-pivoted state; if the invoice section stalls,
   the paper degrades gracefully to F1, not to a broken pivot.
3. **The real-bank defensibility check (editor), executed dossier-side
   tonight.** The invoice's headline rests on a simulation whose driver
   was never tracked. Tonight's run: a small tracked harness replaying the
   comparison on the REAL EdNet 250-item 2PL bank under a textbook
   protocol (maximum-Fisher selection, SE stopping; examinees simulated
   from the mirt reference; arms = mirt / neural SH / neural SK, raw and
   mean-sigma-linked). No model training -- frozen fitted parameters only.
   Outcome lands as exhibit E2; the author decides whether it enters the
   paper. This satisfies the editor's reproducibility demand without
   violating the guardian's fence (the harness is post-campaign tooling,
   not the lost simulator).

---

## D. Findings package (2026-07-18 early; program complete)

### D.1 Exhibit E2 -- real-bank adaptive-testing replication (run tonight)

Tracked harness `kt-irt/src/deep_irt/bench/_p2_cat_realbank.py`; results
`kt-irt/results/p2_cat_realbank/realbank_cat.{json,md}`. Textbook
protocol (mirt-reference generator, max-Fisher selection, EAP, 4000
simulees, per-seed neural parameters, raw and mean-sigma-linked arms) on
the real EdNet 250-item 2PL bank (207 kept).

| arm | rmse@20 | miscls@20 | stop len | claimed SE | true RMSE at stop |
|---|---|---|---|---|---|
| mirt | .412 | .120 | 49.9 | .318 | .317 |
| SH raw | .603 | .187 | 7.9 | .290 | .693 |
| SH linked | .484 | .158 | 46.8 | .307 | .382 |
| SK raw | .567 | .195 | 25.8 | .298 | .547 |
| SK linked | .482 | .152 | 50.0 | .378 | .393 |

Reading, honest and citable: the real EdNet bank is information-poor, so
the synthetic invoice's LENGTH RATIOS do not transport. The harm changes
form instead of disappearing: the shared head's raw parameters stop the
adaptive test at 8 items while certifying SE .29 against a true error of
.69 -- the test is confidently wrong; the separated key is three times
more conservative at the same claimed precision with a materially
smaller calibration gap; after honest scale linking, SK is modestly
better at every fixed length and essentially calibrated (+.015 vs SH's
+.075), and the mirt arm is well-calibrated throughout (claimed .318 vs
true .317 -- the harness's internal-consistency check passes). The
synthetic finding's ORDER (SK over SH for measurement use) transports;
its magnitude and form are bank-dependent. This is the defensibility
answer the editor-judge demanded, produced without touching the frozen
campaign or rebuilding the lost simulator.

### D.2 The recommended frame, beat by beat (for the author's own prose)

F2+ in ten beats, each with its exhibit. This is a skeleton, not prose.

1. **Setup + the axis.** Platforms log more than right/wrong -- partial
   credit, option choice (the signals this paper actually studies; fix
   the intro's current attempts/hints/revisions list). Systems read
   named parameters off prediction-trained models. Axis sentence:
   prediction quality and measurement fidelity are different objects,
   and this paper prices the difference.
2. **One family, not three decoders.** The nested response-format
   taxonomy 2PL -> GPCM -> NRM; each rung keeps strictly more of the raw
   response. Move Thissen & Steinberg 1986 to this sentence; add Bock
   1972. [fig:arch]
3. **Climbing pays.** Prediction never loses across the family, and wins
   at the top: the NRM head beats unconstrained option predictors by
   +.08-.12 on EdNet, all three encoders. Promote from subordinate
   clause to a named result. [tab:real_prediction]
4. **The catch, with truth in hand.** Synthetically, the information the
   upper rungs add rides in exactly the parameters the shared readout
   corrupts (one mechanism per rung; compress the gradient section per
   the A2 orphan list); the separated key repairs recovery at unchanged
   accuracy, 18/18 cells. [tab:mass, fig:dd, fig:scatter]
5. **The finding is a stratification.** Item locations port; the
   discrimination family is fragile-under-SH / repairable-under-SK;
   persons recover synthetically (exhibit E1: SK better in all nine
   N=2000 cells) yet are the weakest tier on real data -- the tier where
   the synthetic-to-real gap is largest. [tab:mass + E1 column]
6. **Real data, real instruments.** No truth exists, so the checks
   become stability, design-agreement, and the offline mirt reference.
   Archetype sentences: TIMSS is the designed-assessment archetype
   (rubric-built ordinal structure; GPCM platform-native); EdNet is the
   platform-log archetype (dichotomization is an analyst convention);
   KDD is the binary control, named as such. [transition prose]
7. **The benign pole.** TIMSS: ordered structure holds under both
   designs, 31/31 items, SH ~ SK -- design barely matters where
   structure is built in by the instrument. [fig:timss_case_shsk +
   threshold appendix]
8. **The fragile pole and the capstone.** EdNet: design matters exactly
   where the theory says (slopes, persons). Two-resolution study: same
   learners read at binary and nominal grain -- item locations port
   (.95/.97), slopes transfer weakly (.21 -> .46 under SK), persons
   partially (.18/.33 raw; .59/.63 disattenuated, fingerprint-matched).
   [fig:ednet_2pl_shsk, tab:ednet_two_resolution, fig:reversal_bridge,
   fig:ednet_case_shsk]
9. **The invoice, measurement-scoped.** Bridge sentence scoping to
   measurement reuse. Banked synthetic costs: shared 196.8% [190,204] of
   oracle test length vs separated 157.2% [151,163]; +2.3pp vs +0.6-0.8pp
   misclassification; repairing difficulty alone is the WORST arm at its
   own stop. Optional E2 sentence: on the real information-poor bank the
   harm surfaces as miscalibrated confidence (SH certifies SE .29 at
   true error .69; SK three times more conservative; order transports).
   [restored CAT exhibit from outputs/p2_cluster/cat_clustered.json +
   fold JSONs; E2 optional appendix]
10. **Practice guidance + the boundary.** Read parameters off a KT model
    only through the stratification; use the separated key wherever
    parameters are reused; prediction alone can prefer the shared head
    (EdNet-NRM edge, .002-.012) -- which is the axis sentence again:
    prediction is not the decision. The person tier stays the standing
    caution.

### D.3 Decision menu (author rulings requested)

- **R-A. Adopt F2+ staged per the guardian:** draft the F1 spine as a
  complete paper first; add the invoice as one additive section written
  strictly from the banked artifacts; committed fence -- the lost
  simulator is not rebuilt inside the paper's scope. [recommended]
- **R-B. E2 placement:** appendix/footnote in the paper, or dossier-only
  evidence. Either is safe; the harness is tracked and reproducible.
- **R-C. KDD:** keep as the named binary control (one sentence) or drop
  the column. [keep-with-sentence recommended]
- **R-D. Orphan-theory pruning (A2 list):** compress `sec:gradient`;
  either delete the refit-ladder promise or fulfill it from the archived
  oracle rungs; soften the Fisher operating-point claim to the marginal
  statement or add the conditional binning exhibit (computable from
  stored arrays). [author picks per item]
- **R-E. Dangling promises:** under F2+ all three resolve by fulfillment
  (invoice) or deletion (refit sentence in the abstract if R-D deletes
  the ladder). Must be resolved either way -- they compile as "??" today.
- **R-F. Disclosure sentences:** every real-data figure is LSTM-only
  (breadth lives in tab:real_prediction) -- one sentence; the Methods
  NxQ grid overclaim -- delete the Q clause or surface one archived
  Q-robustness line. [delete recommended]
- **R-G. New cells (EdNet-GPCM / KDD-GPCM / TIMSS-2PL):** NOT
  recommended -- the ladder is already complete with TIMSS as the
  ordinal rung, and the EdNet speed-ordinalization carries a construct
  caveat. Costed menu in A5 if ever wanted.
- **R-H. Title:** re-rank the plan-v2 slate under F2+ (author picks; the
  slate's truth-free noun is F3 vocabulary -- avoid).

### D.4 What was produced tonight (all committed)

- Dossier: this file + seven lens reports (`docs/framing_review/A1-A7`).
- Exhibits: E1 synthetic person-tier table
  (`docs/framing_review/E1_synthetic_theta_recovery.md`); E2 real-bank
  CAT replication (`kt-irt/results/p2_cat_realbank/` + tracked harness).
- Verifications: CAT artifacts banked and located (driver lost -- scaffold
  only in git; exhibit writable from `outputs/p2_cluster/
  cat_clustered.json`); the invoice numbers' source confirmed
  (`docs/exposure_rerun_results.md`); ancestor draft with the full
  plan-A arc confirmed at `overleaf-sync/submission/`.
- No paper files were edited. The campaign remains frozen; no model was
  trained.

---

## E. Re-think under corrected premises (2026-08-09, after author feedback)

**The correction.** The author: the draft has only had its experimental
results updated -- nothing else -- so the staleness diagnosis is true but
trivial, and the front matter still describes the INTENDED paper. Two
consequences for the analysis above: (1) the "three incompatible
conceptions" story over-reads editorial lag as architectural conflict --
there is one intended conception (the plan-v2 arc) with modernized
results and unwritten connective tissue; (2) the useful question is not
"finish the pivot" but: assume the mechanical repair done -- does the
glue criticism still bite against the best-completed version, and what
substantive (not editorial) unification does the rewrite need?

**Two seams survive any front-matter repair.**

1. *The inferential contract.* The synthetic half answers recovery
   against truth; the real half answers stability and cross-design
   agreement. Different questions. The completed paper must state the
   contract in one place: synthetic data establishes the causal
   mechanism (the shared path corrupts specific parameter groups; the
   separated key repairs them -- truth in hand); real data cannot show
   truth, so it is examined for the mechanism's fingerprint --
   design-sensitivity concentrated exactly where the mechanism predicts.
   Without that sentence pair, "two case studies" stays glued no matter
   how good the archetype prose is.
2. *Why the fragility lives where it lives.* The archetype framing
   (designed assessment vs platform log) is venue-friendly narrative but
   post hoc as an explanation. The mechanistic axis is the amortization
   load: responses per item parameter.

**A claim in section D falsified by the paper's own numbers.** The
ladder wording "fidelity risk climbs with information content" is wrong
synthetically: SH hard-parameter recovery IMPROVES up the rungs (LSTM
.553 -> .719 -> .812; the SH-SK gap shrinks +.344 -> +.223 -> +.148;
same direction for dkvmn; transformer peaks mid-ladder). Do not write
the monotone claim; a reviewer falsifies it from Table 1. The honest
statement: with ample data per parameter (the synthetic bed), richer
formats are BENIGN -- option-level data identifies its own parameters --
and prediction even improves; the price appears when the richer head
meets thin per-item data.

**The load axis (exhibit E3, computed tonight from stored records).**
Responses-per-parameter across the four real cells: kdd-2pl 533,
ednet-2pl 488, timss-gpcm 305, ednet-nrm 8 -- a sixty-fold starvation
drop at exactly the cell where the paper's fragility lives. Cross-design
agreement (SH vs SK, lstm, seed-mean item parameters, 25 folds):

| cell | resp/param | discrimination SH~SK | location SH~SK |
|---|---|---|---|
| kdd-2pl | 533 | .948 | .961 |
| ednet-2pl | 488 | .978 | .998 |
| timss-gpcm | 305 | .976 | .994 |
| ednet-nrm | 8 | .772 | .919 |

One table, four cells, both findings: at high load both parameter
families are design-invariant; under starvation, discrimination
agreement breaks (.77) while location degrades far less (.92) -- the
stratification and its mechanism together. This upgrades the paper's
structure from "a benchmark plus two case studies plus a control" to
"four points on one curve": TIMSS is benign not because it is designed
but because 305 responses/parameter is a safe regime; EdNet-NRM is
fragile because 8 is starvation; KDD stops being an orphan and becomes
the highest-load anchor; and the incidental-parameter theory (currently
orphaned, lens A2) finds its consumer -- thin-per-item regimes are
exactly Neyman-Scott territory. The load axis also absorbs the honest
boundary: under starvation a shared representation's smoothing can help
PREDICTION while corrupting the readout -- which is the axis sentence
(prediction is not measurement) made mechanistic.

(File: `docs/framing_review/E3_load_axis.md`. Refinements for the
author's rewrite if adopted: per-item -- rather than per-cell -- version
of the same curve inside EdNet-NRM, exposure-binned; the synthetic bed's
resp/param values annotated as calibration points; the NRM cell uses the
paper's own keyed contrast/keyed intercept objects.)

**Revised recommendation.** F2+ stands in content but changes status:
not a spine replacement -- an INTEGRATION LAYER over the intended
plan-v2 arc. What the rewrite adds to the intended paper: the taxonomy
paragraph (one family), the inferential-contract sentences (one
mechanism, two evidence regimes), the load axis with E3 (one curve; why
here), the archetype sentences as the venue-facing wrapper of the load
mechanism, the axis sentence (prediction is not measurement), and the
promoted buried results (+.08-.12; the two-resolution capstone). The
plan-v2 beats (motivation, mechanism, repair, invoice, guidance) keep
their order and their exhibits. Beat-map deltas vs section D.2: beat 2
drops the monotone-ladder phrasing for the ample-data-benign /
starvation-pays formulation; beat 6 leads with the load axis and E3,
demoting the archetype sentences to its wrapper; beat 7-8 present
TIMSS/EdNet as the safe and starved ends of the curve. Decision-menu
deltas: R-C strengthens (KDD's role is now quantitative -- keep it);
new R-I: adopt E3 (or its per-item refinement) as a main-text exhibit
vs a one-sentence citation of the four numbers [either works; the
per-item refinement is the strongest version and is computable from
stored arrays].

