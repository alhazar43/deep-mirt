# Project Handoff (START HERE)

Last updated 2026-08-18, after the overnight campaign that closed the
mechanism programme: theory formalization, the canonical storage factorial,
the ordered-head route replication, the nominal routing audit, eight rendered
exhibits, the results spine, and the real-data echo tail.
Read `CLAUDE.md` and the memory index first. This file is the state pointer.
Plain language throughout, no internal codenames.

Repo root `C:/Users/steph/documents/deep-mirt`, branch
`feat/prediction-loss`. Paper repo github.com/alhazar43/JEDM-paper (name is
historical). Framework submodule `kt-irt/` (github.com/alhazar43/kt-irt).

The working thread with the second reviewer is **kt-irt issue #5**. Issue #3
is an archive of how the evidence was found and should not be treated as the
plan. The v6 four-figure system is dead and its prototypes are not to be
iterated on.

## 1. The paper in plain words

A knowledge tracing model is trained only to predict whether a learner answers
the next question correctly. Some of these models also report item response
theory quantities as a side product, such as how hard an item is and how sharply
it separates strong learners from weak ones. People read those numbers as
measurements.

The paper asks which gradients actually train those numbers. An item
representation can receive two very different kinds of credit. When the item is
answered, the response feeds back through the psychometric head, and that
channel has the same shape as calibrating the item. When the same item sits in
a learner's history, later responses feed back through the sequence model, and
that channel only asks the item representation to help predict the future. The
second channel is not a calibration signal and nothing in the objective asks it
to be one.

So the question is not whether item embeddings should be split in two, and not
which width is best. It is which prediction-credit paths are allowed to write
into the storage the psychometric parameters are read from. Forward separation
is not backward separation.

## 2. What is settled

### From the frozen synthetic grid, five data seeds by five folds

Three sequence models by three response formats, nine families.

- Removing the sequence model's ability to write into the measurement storage,
  with the forward computation held bitwise identical, improves the primary item
  parameter in nine of nine families. It clears the agreed materiality bar of
  0.05 in the six non-nominal families; the three nominal families improve by
  0.020 to 0.037 and fall short of it. Say it that way, not "nine of nine".
- Held-out likelihood under that intervention is equal or better in eight of
  nine families, largest gain 0.0085, worst loss 0.0003. Prediction is not
  merely preserved, it is very slightly better.
- Widening the representation the parameter readout reads improves recovery in
  nine of nine. Widening the item representation on the sequence side helps in
  zero of nine and clearly hurts in four.
- Splitting into two tables at matched width does almost nothing. Of 45 matched
  comparisons, 42 fall below the materiality bar.
- Prediction accuracy across all these designs moves within about half a
  percentage point while recovery ranges from about 0.30 to 0.96.

### From the Gate 0 autograd audit, `kt-irt/docs/v7_audit.md`

Verified numerically on a rerunnable instrument, not asserted from comments.

- The credit decomposition is exact. The two parts are computed independently
  and reconstruct the full gradient to 1e-7 or better, at initialisation and at
  fitted weights.
- The one-step shift does what it claims. Credit reaching an item's row from
  losses at or before its own occurrence is exactly zero in all three encoders,
  so the attention mask does not leak.
- The isolation intervention leaves response probabilities equal to the last bit.
- The published DKVMN uses a separate interaction table for memory content; ours
  reuses the question-side table there. But the published Deep-IRT reads
  difficulty from the same embedding that drives addressing and the summary, so
  its difficulty is already exposed to cross-time credit through those two
  routes. Our implementation adds a third. Verified against both papers.
- The three DKVMN routes can be cut independently and forward-identically, and
  are additive as vectors. Cuts must recompute a route's input from a detached
  embedding; detaching the shared per-step key also strips another parameter's
  gradient, which is what the old summary-key ablation did.
- The measurement readout subspace is one global two-dimensional plane for the
  static two-parameter head, the span of the two readout weight rows, the same
  for every item. Call it a shared measurement subspace. Do not say items
  compete for the same directions.

### From M1 and M2 on 54 fitted checkpoints, `kt-irt/results/p2_v7_m2/report.md`

- M1 passes at fitted weights. Stop A of the freeze is cleared.
- The refits reproduce their historical parents bit for bit, 53 of 54 units at
  exactly zero difference, because the bed and protocol are inherited by import
  rather than copied.
- **Stop B is triggered, and the author has accepted it.** The in-plane component of cross-time credit does not
  track the item score or the recovery error better than total cross-time
  magnitude. Every item-level coefficient is under 0.11 and the in-plane
  quantity is the weaker of the two in eight of twelve comparisons.
- The isotropic reference was never applicable. Cross-time credit lives in
  between two and four effective dimensions at both widths, so the object is an
  angle between two low-dimensional subspaces, not a dimensional dilution.
  Against random planes holding the gradients fixed, the fitted readout plane
  captures less than random at width 8 and more than random at width 64 for the
  two non-memory encoders, with the memory model at chance.
- The checkpoints are far from stationary, so the estimating-equation
  displacement was never testable at them. That is a protocol limit, not a
  refutation.

### From M3 and M4, `kt-irt/results/p2_v7_m3m4/report.md`

- The width boundary is complete for the two-parameter family, 25 units per
  cell. Blocking cross-time write access improves discrimination recovery in 15
  of 15 cells and materially at widths 8, 16 and 32. Prediction never moves.
  Gains by width, 8 through 128: LSTM 0.145, 0.134, 0.069, 0.019, 0.005;
  Transformer 0.076, 0.079, 0.069, 0.021, 0.002; memory model 0.125, 0.146,
  0.107, 0.046, 0.026. The memory model keeps a detectable effect at width 128
  where the other two lose it.
- The damage has a single route. Blocking only the path where the question
  embedding enters the per-step summary state recovers 0.88 of what full
  isolation buys. That is the path published Deep-IRT reads its difficulty from.
  The memory-content route, which this implementation adds and the published one
  does not have, is exactly null at 0.0001 with p = 0.89. Addressing is 0.0075.
- Two cautions. The route split at initialisation, 0.01, 0.97 and 0.13, is not
  the split at fitted weights, which is 0.28, 0.82 and 0.61; never quote the
  first as the mechanism. And gradient magnitude does not predict damage at the
  route level either, since the memory-content route carries 0.61 of the
  cross-time credit and none of the damage.

### From the overnight campaign, `kt-irt` and the paper repo, 2026-08-18

- The theory document exists: `kt-irt/docs/v7_theory_formalization.md`,
  fourteen parts, every claim statused with its falsifier in a dependency
  table. The v7 GPCM cells train on the cumulative-link ordinal cross-entropy,
  not the legacy weighted loss; the methods caveat is written there.
- The harmful summary route has exactly one-step temporal reach, verified.
- The published key/content storage split does not protect the measurement
  table: write-access harm +0.161 under a faithful canonical variant against
  +0.125 tied, and canonical writable is worse than tied writable. D5 crossed
  out; the phenomenon belongs to the lineage.
- The route story crosses to the ordered head with a refinement: content null
  again, the question-side pair carries 0.82 of the full effect, no single
  route material alone.
- Nominal routing is a repair, not a treatment: ownership direction is
  head-invariant, materiality is head-dependent, and the adopted head absorbs
  part of the harm. Zero fits; 79 cells rescored with zero failures.
- Eight figures are rendered to the binding plotting guide from frozen
  packets, resolved titles included; the storyline lives in
  `overleaf-sync/rewrite_kit/v7/results_spine.md` with the open decisions in
  `open_questions_register.md` beside it.
- The real-data echo is complete: 225 isolated units, zero failures, report
  at `kt-irt/results/p2_v7_m9/report.md` and Figure 9 rendered. On TIMSS
  ordered and EdNet binary the synthetic pattern echoes (agreement and
  stability rise, prediction flat); on EdNet nominal, which has no external
  reference, isolation lowers cross-fit stability for all three encoders.
  Both directions are in spine claim H at equal prominence.
- The assistant's independent second reading is at
  `overleaf-sync/rewrite_kit/v7/v7_second_reading.md`, committed with its
  probe pre-registration before the probe ran; its section 6 carries the
  probe and conditioning results.
- The Overleaf git remote (`origin` in `overleaf-sync/`) is a stale stub
  abandoned 2026-07-06; the live remote is `jedm` (github JEDM-paper).
  Reconciling or recreating the Overleaf project is the author's call.

## 3. Open decisions

These belong to the author and the second reviewer.

Two that were open on 2026-08-17 are now closed by the author. Stop B is
accepted, so the projection and width-geometry mechanism is out of v7 as an
explanatory claim and the low-rank subspace observation survives only as an
exploratory note. Fits continued to near stationarity are refused, so the
estimating-equation displacement stays a conditional theorem about stationary
points and never an empirical claim about the fitted models.

1. Whether the paper's centre is a warning and an audit procedure or an
   architecture recommendation. The evidence still supports the first more
   comfortably.
2. Whether the real data section stays in the main text at its current strength,
   moves to an appendix, or is presented as a limitation. On the assessment data
   agreement with a classical calibration is about 0.12 and 0.43, so even the
   better design is poor. On the log data, averaged item estimates make the two
   designs indistinguishable at about 0.90 each and only single runs separate
   them, which points to run to run instability.
3. On the log data, whether to report averaged or single-run item estimates.
   They disagree and the choice must be stated.

## 4. Rules in force

- No prose goes into the manuscript file directly. Everything for the paper is
  drafted in `overleaf-sync/rewrite_kit/` and the author splices it.
- Plain language everywhere, including figures, captions, tables and issue
  posts. No internal codenames.
- If something cannot be done with the data or compute that exists, say so at
  once and stop. The decision to spend more is the author's.
- Do not run new fits without being asked. When a fit is authorised, inherit the
  bed and protocol by import and check the result against its historical parent.
- No publication figures until the mechanism stores are frozen.
- Commits carry no assistant attribution.

## 5. Where things live

- The autograd audit and the graph maps: `kt-irt/docs/v7_audit.md`.
- Its instrument, which trains nothing and reruns in twenty seconds:
  `kt-irt/src/deep_irt/bench/_p2_v7_gate0.py`.
- The M1 and M2 runners and report:
  `kt-irt/src/deep_irt/bench/_p2_v7_m1_fits.py`,
  `_p2_v7_m2_gradients.py`, `_p2_v7_m2_report.py`.
- Their stores: `kt-irt/results/p2_v7_m1`, `p2_v7_m2`, and the report at
  `p2_v7_m2/report.md`. Checkpoints at `kt-irt/weights/v7_m1`, untracked.
- Paper-level framing and the experiment freeze:
  `overleaf-sync/rewrite_kit/v7/global_picture.md` and `experiment_freeze.md`.
- The mathematical account: `overleaf-sync/rewrite_kit/two_role_formalization.md`.
- What the two fields print when they claim a parameter is trustworthy:
  `rewrite_kit/v6/validity_displays_kt.md` and
  `validity_displays_psychometrics.md`. Still the most useful documents in the
  kit.
- The v6 exhibit generator, now historical:
  `kt-irt/src/deep_irt/bench/_p2_v6_exhibits.py`.

## 6. Data and compute state

Per item estimated and generating parameters exist for all nine families in
every design at five seeds by five folds. Ability trajectories over sixty steps
exist for every design. Two real datasets have classical calibrations to compare
against. Per-item gradients, scores, Jacobians and projectors now exist for the
54 shared checkpoints.

Known gaps.

- The nominal format on real data has no defensible classical comparison.
- The toggle-family stores hold no held-out likelihood for the two-parameter and
  ordinal formats, so the prediction companion the freeze names primary does not
  exist for the width and read-only cells.
- Trained weights exist for the misspecification grid, the memory-model probe
  cells, six nominal cells, the 54 shared checkpoints of the gradient audit and
  the 100 route-block fits. Nothing else.
- The width ladder and the five lost width-64 units are no longer gaps. Both were
  closed on 2026-08-17.

On 2026-08-17 the author authorised filling missing widths, which added 42 cells
and 1050 fits, and separately the 54 shared checkpoints above. Both are follow-up
evidence. The paper must not read as though a full width sweep was part of the
original design.

## 7. Rest of the repo

- `kt-irt/` is the active framework and is portable. Install with
  `pip install -e kt-irt`. Tests with `python -m pytest` from inside it. Cluster
  notes in `kt-irt/slurm/README.md`.
- `kt-mirt/` is an active sideline on multi concept tracing. Start at
  `kt-mirt/_planning/PLAN.md`.
- `ma-irt/` is the frozen first chapter. `rl/` is parked. The thesis north star
  is `docs/Thesis_overview.md`.
