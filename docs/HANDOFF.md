# Project Handoff (START HERE)

Last updated 2026-08-17. Read `CLAUDE.md` and the memory index first.
This file is the state pointer. Plain language throughout, no internal
codenames.

Repo root `C:/Users/steph/documents/deep-mirt`, branch
`feat/prediction-loss`. Paper repo github.com/alhazar43/JEDM-paper
(name is historical). Framework submodule `kt-irt/`
(github.com/alhazar43/kt-irt). The working record with the second
reviewer is kt-irt issue #3, which is long but is the honest history.

## 1. The paper in plain words

A knowledge tracing model is trained only to predict whether a learner
answers the next question correctly. Some of these models also report
item response theory quantities as a side product, such as how hard an
item is and how sharply it separates strong learners from weak ones.
People read those numbers as measurements.

We asked whether those numbers can be trusted. The answer is that they
depend heavily on architecture choices that prediction accuracy cannot
see. Two models that predict almost identically can report very
different item parameters.

We can name two causes. The first is how wide the representation is
that the parameter readout reads. The second is whether the sequence
model is allowed to change that same representation during training. A
control that leaves the forward computation completely unchanged and
only removes the sequence model's ability to write into the item
representation improves the reported parameters in all nine model
families we tested.

The original claim of the earlier draft, that splitting the item
representation into two tables is what helps, is dead. It died because
the old comparison also widened the readout at the same time. When
widths are matched, splitting alone does almost nothing.

## 2. What is settled

All numbers below come from five data seeds by five folds per cell, on
synthetic data, across three sequence models and three response
formats, which is nine families.

- Widening the representation the parameter readout reads improves
  recovery in nine of nine families.
- Widening the item representation on the sequence side helps in zero
  of nine and clearly hurts in four.
- Removing the sequence model's write access, with the forward pass
  held identical, improves recovery in nine of nine. Accuracy moves by
  a median of 0.002.
- Splitting into two tables at matched width does almost nothing. Of
  45 matched comparisons, 42 fall below the agreed materiality bar.
  The three that pass are all at narrow widths and share neither
  family nor width.
- Prediction accuracy across all these designs moves within about half
  a percentage point while recovery ranges from about 0.30 to 0.96.
- There is a mathematical account, checked numerically. At a training
  optimum, designs that deny the sequence model write access satisfy
  exactly the calibration equations a psychometrician would solve. The
  shared design satisfies a version of those equations pushed off
  target. This also predicts, correctly, which parameter suffers most
  and why the effect shrinks as the readout widens.
- One architecture behaves differently in a useful way. The memory
  based model gains almost nothing from width but still gains from
  removing write access. Its own design keeps ability outside the item
  representation. Caution, our version of that model ties both of its
  internal roles to one item table, unlike the published design, and
  that must be disclosed.

## 3. What is not working, and it is the main thing to talk about

The figures do not communicate. That judgment is the author's and it
is correct. My diagnosis, offered so the next conversation does not
repeat the mistake.

Everything we have drawn so far is a picture of agreement statistics.
Panels of correlation coefficients, effect sizes with error bars,
points in a plane of two effects. A reader sees 0.57 against 0.90 and
has no way to know whether that difference would change anything they
do. We converted a stats table into pictures of the same stats.

Three specific problems.

First, no consequence is ever shown. Nothing in the figures tells a
teacher, a test developer, or a modeler what breaks. The natural fix
is to show a decision. Two models that predict equally well disagree
about which twenty items are the hardest, or about which items are too
weak to keep, or about where a particular learner stands. That is a
picture with a victim.

Second, we buried the most striking result we have. When the estimated
values are plotted against the generating values with the identity
line drawn, every design compresses the scale badly. The slopes are
about 0.15, 0.19, 0.25 and 0.56 across the four designs. Every model
we fit understates how sharply items separate learners, by between two
and seven times. The literature audit found that nobody in this field
reports scale at all, only correlations, so this may be the most
publishable single fact in the study, and we found it by accident an
hour before stopping.

Third, the real data section is weak and should be presented as weak.
On the assessment data the two designs give agreement of about 0.12
and 0.43 with a classical calibration, so the better one is still poor.
On the log data, when item estimates are averaged over all runs, the
two designs are indistinguishable at about 0.90 each, and only single
runs separate them, which points to run to run instability rather than
a systematic difference.

One idea worth trying next, not yet built. The most common display in
this field is a single learner's estimated ability over time. We have
full ability trajectories for four hundred held out learners in every
design. Nobody has ever drawn that display for two models that predict
identically. Showing the same learner's ability curve under two such
models, side by side with the responses underneath, would be
immediately readable by the knowledge tracing audience and would make
the point without a single correlation coefficient.

## 4. Open decisions

These belong to the author and the second reviewer, not to me.

1. Whether the paper's center is a warning and an audit procedure, or
   an architecture recommendation. The evidence supports the first
   more comfortably than the second.
2. Whether the scale compression result is promoted to the headline.
3. Which figure system to use, now that the four prototypes have been
   seen and judged not to work.
4. Whether the real data section stays in the main text at its current
   strength, moves to an appendix, or is presented explicitly as a
   limitation.
5. On the log data, whether to report averaged item estimates or single
   run estimates. They disagree and the choice must be stated.

## 5. Rules in force

- No prose goes into the manuscript file directly. Everything for the
  paper is drafted in `overleaf-sync/rewrite_kit/` and the author
  splices it.
- Plain language everywhere, including figures, captions, tables and
  posts on the issue thread. No internal codenames.
- If something cannot be done with the data or compute that exists,
  say so at once and stop. The decision to spend more is the author's.
- Do not run new fits without being asked.
- Commits carry no assistant attribution.

## 6. Where things live

- Paper planning, current: `overleaf-sync/rewrite_kit/v6/`. Start with
  `paper_blueprint.md` and `evidence_ledger.md`, then
  `provenance_and_layout.md` for the agreed figure hierarchy.
- The mathematical account: `rewrite_kit/two_role_formalization.md`.
- Claim by claim audit: `rewrite_kit/central_claim_audit.md`.
- What the two fields actually print when they claim a parameter is
  trustworthy: `rewrite_kit/v6/validity_displays_kt.md` and
  `validity_displays_psychometrics.md`. These are the most useful
  documents produced in the last session.
- What we can draw from stored results:
  `rewrite_kit/v6/measurement_object_inventory.md`.
- The memory model question: `rewrite_kit/v6/memory_encoder_check.md`.
- Figures, captions and every value behind them:
  `rewrite_kit/v6/exhibits/`.
- One generator produces all of it, from committed results only, with
  no refitting. `kt-irt/src/deep_irt/bench/_p2_v6_exhibits.py`. Run it
  with the research environment active. It cross checks itself against
  the frozen report before writing anything.
- Latest commits, kt-irt 53c8422 and paper repo 7ed99f0.

## 7. Data and compute state

Everything needed is on disk. Per item estimated and generating
parameters exist for all nine families in every design, at five seeds
by five folds, including the nominal response format. Ability
trajectories over sixty steps exist for every design. Two real
datasets have classical calibrations to compare against.

Known gaps. The reversed design, wide on the sequence side and narrow
on the parameter side, was never fit for the memory model. The nominal
format on real data has no defensible classical comparison. No trained
weights were kept, so anything needing the model's internals would
require refitting.

On 2026-08-17 the author authorized filling the missing widths, which
added 42 cells and 1050 fits in about three hours on the local card.
Seven of the nine families received their middle and top widths in
that fill. Those cells are follow up evidence and the paper must not
read as though a full width sweep was part of the original design.

## 8. Rest of the repo

- `kt-irt/` is the active framework and is portable. Install with
  `pip install -e kt-irt`. Tests with `python -m pytest` from inside
  it. Cluster notes in `kt-irt/slurm/README.md`.
- `kt-mirt/` is an active sideline on multi concept tracing. Start at
  `kt-mirt/_planning/PLAN.md`.
- `ma-irt/` is the frozen first chapter. `rl/` is parked. The thesis
  north star is `docs/Thesis_overview.md`.
