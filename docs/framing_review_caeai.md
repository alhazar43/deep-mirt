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

