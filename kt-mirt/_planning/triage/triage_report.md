# STAGE-0 bed triage

Computed directly from the raw local files for all four candidate beds. Every
number below comes from a full pass (KDD, TIMSS) or a disclosed, seeded
sample (EdNet), never from literature estimates. Machine-readable detail
(including the full top-30 KC-pair tables) lives in the per-bed JSON files
next to this report; this file is the narrative read plus headline tables.

Scripts: `kt-mirt/scripts/triage/{triage_common,triage_kdd,triage_ednet,
triage_timss,triage_slam}.py`. Outputs: `kt-mirt/_planning/triage/
{kdd_algebra_2008_2009,ednet_kt1,timss_g8_usa,duolingo_slam_en_es}_stats.json`.

## Decoupling metric

Adapted from the qmirt-archaeology "positivity condition"
(`kt-mirt/_planning/research/qmirt-archaeology.md` section 2, internal
threshold >=0.75 clean / 0 unidentifiable). For a KC pair (A, B) that
co-occurs at least once, a "slot" is one interaction (one KDD step, one
EdNet question attempt) tagged with A and/or B. With n_A, n_B the slot
counts for each and n_both the slots tagging both at once,

```
decoupling(A,B) = (n_A_only + n_B_only) / (n_A_only + n_B_only + n_both)
```

is the fraction of practice touching the pair that touches exactly one of
the two. Reading A as "source," n_A_only / n_A is exactly the qmirt
quantity for that direction; the symmetric form generalizes over both
possible directions since real KC-pair data has no privileged one. Both
one-directional readings are also stored per pair in the JSON. Full
derivation in the `triage_common.py` module docstring. Pairs are ranked by
n_both (most-practiced-together first), restricted to pairs that actually
co-occur, and the reported "fraction clearing 0.75" is over the top 30 such
pairs per bed/KC-model.

## Sampling and runtime

| Bed | Coverage | Rows/interactions | Runtime |
|---|---|---|---|
| KDD Algebra 2008-2009 | full file | 8,918,054 steps | 110 s |
| EdNet KT1 | 4000 of 784,309 users, `random.Random(seed=42).sample()` over the sorted filename list | 467,806 attempts | 25 s |
| TIMSS 2019 G8 USA | full local extract (itself a minresp>=5 filtered subset of the raw .sav) | 5,135 students x 31 items | <1 s |
| Duolingo SLAM en_es | n/a, data absent | 0 | <1 s |

All four are far under the 15-minute per-bed budget.

---

## KDD Cup 2010 Algebra 2008-2009

Streamed the full 8.9M-row transaction log once, computing all three
DataShop KC models (SubSkills, KTracedSkills, Rules) in the same pass. Unit
of analysis is a step, identified by Problem Hierarchy + Problem Name + Step
Name; response is Correct First Attempt (0 invalid/dropped rows).

| KC model | distinct KCs | correct rate | per-KC rate q1/median/q3 | frac KC > 0.85 | opp q1/median/q3 | frac opp >= 5 / 10 / 20 | growth opp1 -> opp10 | frac slope > 0 | decoupling: pairs w/ cross-occ, frac clearing 0.75 | anchors: frac KC with >= 3 pure |
|---|---|---|---|---|---|---|---|---|---|---|
| SubSkills | 541 | 0.845 | 0.746 / 0.850 / 0.915 | 0.495 | 5 / 11 / 27 | 0.794 / 0.547 / 0.328 | 0.713 -> 0.832 | 0.774 | 697, **0.267** | 0.725 |
| KTracedSkills | 515 | 0.824 | 0.754 / 0.855 / 0.925 | 0.517 | 4 / 8 / 16 | 0.737 / 0.423 / 0.202 | 0.734 -> 0.839 | 0.731 | 246, **0.800** | 0.852 |
| Rules | 1069 | 0.860 | 0.470 / 0.786 / 0.903 | 0.372 | 3 / 9 / 21 | 0.683 / 0.494 / 0.274 | 0.671 -> 0.862 | 0.750 | 1601, **1.000** | 0.905 |

**Verdict.** Saturation risk is real and model-dependent: overall correct
rate sits at 82-86% across all three taggings, and 37-52% of individual
KCs already clear the 0.85 ceiling flag, worst for SubSkills (its coarser
tag set pools easy and hard sub-steps together less than the finer models
do, so its per-KC ceiling looks slightly less bad, but its median rate of
0.85 means half its KCs are already at or past the flag). Growth headroom
is genuinely present, not just plausible: the pooled early-opportunity
curve rises 12-19 points from opportunity 1 to opportunity 10 in every
model, and 73-77% of individual KCs have a positive OLS slope, so there is
real passive learning signal in the raw data before any modeling. The
magnitude is modest (median per-KC slope 0.006-0.014 correct-rate per
opportunity), consistent with a supplementary rather than primary growth
readout given how quickly the curve approaches its ceiling. Decoupling
feasibility is where the three tag models diverge sharply and this is the
single most decision-relevant number in this section: SubSkills fails the
gate badly (only 27% of its top-30 practiced pairs clear 0.75; its
single most-practiced pair, "Isolate variable term in linear equation" x
"Remove constant," sits at a decoupling of 0.010, i.e. almost perfectly
co-scheduled, the qmirt "completely unidentifiable" regime almost exactly),
while KTracedSkills clears 80% and Rules clears 100%. Rules' 100% needs a
caveat before it is used as a selling point: 24 of its top 30 pairs (and 24
of the 30 that clear 0.75) involve "unknown bug element," a diagnostic
catch-all tag for unrecognized error productions, not a real skill, so
Rules' apparent decoupling strength is partly an artifact of a low-content
tag dominating the ranking rather than genuine pedagogical-skill
separation; KTracedSkills' 80% is the more trustworthy number of the two
finer-grained models. Anchor feasibility is comfortable everywhere (73-90%
of KCs have at least 3 pure single-KC items, mean item-KC arity 1.0-1.8),
though the Rules item-to-KC map is a last-write-wins approximation (its tag
can vary attempt-to-attempt since it reflects which specific production
rule/hint fired, not a fixed content classification, unlike SubSkills/
KTracedSkills which are static per step) so its f/g numbers are somewhat
less trustworthy than the other two models' numbers. **Net: KTracedSkills
is the recommended KC model for this bed** if a single one must be picked
for downstream transfer-identification work; it clears decoupling
comfortably without Rules' catch-all-tag artifact and without SubSkills'
near-total co-scheduling on its heaviest pairs.

---

## EdNet KT1

Sampled 4000 of 784,309 per-user files (seed 42), joined to the full
13,169-question item bank. Unit of analysis is one question attempt;
response is the chosen option against `questions.csv`'s `correct_answer`.
Item-KC arity and pure-anchor stats (f/g) use the full item bank, not the
sample, since that file is small, static, and exact.

| | value |
|---|---|
| distinct tags (sample / full bank) | 189 / 189 (verified independently against `questions.csv`) |
| overall correct rate | 0.652 |
| chosen-option distribution | a 27.1%, b 28.4%, c 26.4%, d 18.1% (+0.03% blank/skipped) |
| per-KC correct-rate q1 / median / q3 | 0.613 / 0.679 / 0.716 |
| frac KC > 0.85 | **0.021** |
| opportunity q1 / median / q3 | 1 / 2 / 5 |
| frac opportunity >= 5 / 10 / 20 | 0.279 / 0.141 / 0.072 |
| pooled growth curve, opportunity 1 -> 10 | 0.569 -> 0.697 |
| frac KC with positive slope | 0.818 |
| decoupling: pairs w/ cross-occurrence, frac clearing 0.75 | 948, **0.867** |
| anchors: frac KC with >= 3 pure items | 0.402 |
| item-KC arity mean / max (full bank, 13,169 items) | 2.18 / 6 |

**Verdict.** Saturation risk is low, the opposite profile from KDD: overall
correct rate is 65.2%, only just above the 25% multiple-choice floor's
double, and only 2.1% of KCs clear the 0.85 flag (median per-KC rate 0.68,
the single highest KC tops out at 0.87). This 65.2% is itself a finding
worth flagging on its own: the dataset survey (`kt-mirt/_planning/research/
datasets.md`) could only locate downstream DKT/BKT *model* accuracies in
the 65-73% range for EdNet and explicitly warned not to reuse those as a
raw correct-rate proxy; the real number lands at the low end of that band,
confirming the raw task is comfortably unsaturated rather than
near-ceiling. Growth headroom is present and clean in pooled form
(opportunity 1 at 56.9% rising to 69.7% by opportunity 9-10, 82% of KCs
with a positive slope, the cleanest growth curve of the three growth-
bearing beds) but individual learner-KC opportunity density is thin. Median
opportunity count is only 2 and just 7.2% of learner-KC pairs reach 20
opportunities, so a pooled/aggregate growth read is trustworthy here but
any per-learner trajectory fit will be data-starved for the bulk of
learner-KC pairs (the long tail, up to 1825 opportunities on the most-
drilled pairs, is the exception, not the norm). Decoupling feasibility is
good, 87% of the top 30 practiced tag pairs clear 0.75, better than two of
KDD's three KC models. Anchor feasibility is moderate, 40% of tags have at
least 3 pure items against a mean item-KC arity of 2.18 (EdNet's questions
are genuinely multi-tag by design, unlike KDD's mostly-single-KC steps),
so anchor-dependent designs have real but not abundant headroom here.
Tag identities are numeric IDs only (no semantic labels shipped locally in
`questions.csv`), which does not affect any statistic above but means a
separate tag-to-name crosswalk would be needed before any pair or anchor
result could be discussed by content.

---

## TIMSS 2019 Grade 8 USA

Read from the already-extracted local files
(`data/timss/timss_g8_usa_poly_{matrix,triplets}.csv`, produced earlier by
`data/timss/_build_timss_gpcm.R` from the raw `.sav`; this triage script
only reads them, it does not touch the `.sav` or R). 5,135 students, 31
genuinely polytomous (0/1/2 partial-credit) constructed-response items,
each student answering only a 5-6 item rotated-booklet subset.

| | value |
|---|---|
| overall mean score proportion (mean score / 2) | 0.418 |
| category distribution (0 / 1 / 2) | 44.3% / 27.7% / 28.0% |
| per-item mean-score-proportion q1 / median / q3 | 0.295 / 0.441 / 0.485 (range 0.142-0.729) |
| frac items > 0.85 | **0.0** |
| responses per item q1 / median / q3 | 590 / 1092 / 1128 |
| responses per student | 5-6 (rotated booklet design) |
| GPCM discrimination (a) median | 0.946 (range 0.58-1.68) |
| domain split | 10 math items, 21 science items (coarse id-prefix only) |

**Verdict.** Saturation risk is essentially absent by design: mean score
proportion sits at 0.418 with zero items above the 0.85 flag, exactly what
a psychometrically targeted assessment aims for, in sharp contrast to both
KT beds. Growth headroom is structurally not applicable, not merely
unmeasured. TIMSS is a single-occasion assessment: there is no repeated
practice, no opportunity index, and no learning-curve to compute, so this
was recorded as N/A rather than estimated or approximated. Decoupling
feasibility and anchor feasibility are equally structurally N/A: no
item-to-KC crosswalk exists locally for these items, only a coarse
math/science prefix on the item id, which is a subject split, not a
skill-level Q-matrix, so KC-pair and pure-anchor analysis cannot be run
without acquiring an external item-content mapping. **TIMSS's role in the
program is necessarily different from the other three beds**: a
well-behaved, unsaturated, professionally calibrated static assessment,
useful for cross-sectional or readout-stability checks against items with
already-fit classical GPCM parameters on hand, but not a candidate for any
growth-, decoupling-, or anchor-dependent design without additional data
acquisition.

---

## Duolingo SLAM (en_es track)

`data/slam_raw/` does not contain the actual response data. It contains
only a Harvard Dataverse metadata blob (`ds.json`), a failed-download log
(`dl.err`, `curl: (22) ... error: 400`), and two 122-byte files whose
actual content is the Dataverse error body: `"You may not download this
file without the required Guestbook response for guestbookID 205."` This
is a mandatory access-gate rejection, not a slow, malformed, or oversized
transfer, so no local parsing effort recovers real data from what is on
disk today.

The triage script (`triage_slam.py`) is not a dead stub for this: it
actively searches `data/slam_raw/` for real split files by the official
naming convention and, if found, parses them for real (token-level and
exercise-level K=3-ordinal response stats, the aggregation scheme
documented in `rl/src/ordrec/data/slam.py`). That code path was smoke-
tested against that module's own synthetic unit-test fixture (not real
SLAM data, and not used for any number in this report) and works;
re-running this script after the data is downloaded will pick it up
automatically with no code changes.

**Verdict.** Bed status is unavailable, not merely deprioritized. To
unblock, complete the guestbook form at
`https://doi.org/10.7910/DVN/8SWHNO`, download `data_en_es.tar.gz`, and
extract under `data/slam_raw/`. Independent of availability, SLAM is
structurally different from the other three beds in a way worth flagging
now: it ships no skill/KC tag at all. It is a linguistic per-token
error-annotation log (each token in an exercise carries a binary mistake
label, morphological features, and a dependency parse), not a KC-tagged
item bank, so even once downloaded, (b)-(g) all require a prior design
decision about what should serve as a "KC" (candidates include POS tag,
grammatical feature, or lemma) that KDD and EdNet get for free from their
native tagging. A prior, unrelated project on this repo (OrdRec,
`rl/src/ordrec/data/slam.py`) recorded category-balance figures in a code
docstring from a run it apparently had access to at the time (~62%
all-correct / ~36% partial / ~2% all-wrong exercises, en_es train fold);
that is carried in the JSON output as an explicitly unverified prior
record, not a fresh computation, and should not be treated as satisfying
this triage pass.

---

## Cross-bed synthesis

- **Saturation.** KDD is the most saturated bed (82-86% correct, 37-52% of
  KCs flagged) and TIMSS the least (42% mean, 0% flagged); EdNet sits
  between but far closer to TIMSS (65% correct, 2% flagged). Any design
  that needs headroom above the 0.85 per-KC ceiling should lean on EdNet or
  TIMSS-style beds over KDD, or restrict to KDD's below-ceiling KC subset.
- **Growth headroom.** Both KT beds show a genuine, clean, positive
  passive early-opportunity signal (not an artifact: 73-82% of KCs
  positive-sloped in both, curves rising 12-19 points over the first ten
  opportunities), so a growth-detection claim on either is starting from
  real signal, not noise. KDD's opportunity density is deeper (median 8-11
  vs EdNet's median 2), so KDD is the better bed for anything needing
  individual learner-KC trajectories rather than pooled curves; EdNet's
  pooled curve is cleaner but its per-pair density is thin outside a long
  tail of heavily drilled tags.
- **Decoupling.** This is the sharpest and most actionable split in this
  triage. KC-model choice within a single bed can flip the decoupling
  verdict entirely (KDD SubSkills 27% vs KTracedSkills 80% on the same
  underlying log), so "is bed X decoupled enough" is not well-posed without
  also naming the tag scheme. KTracedSkills (KDD) and tags (EdNet) both
  clear comfortably; KDD Rules' apparent 100% is inflated by a
  non-content diagnostic tag and should not be quoted without that caveat.
- **Anchors.** All three KC/tag-bearing beds clear the Gate-C-style >=3
  pure-item bar for a majority of their KCs (KDD 73-90% depending on
  model, EdNet 40%), so pure-anchor-dependent designs (e.g. frozen
  cross-loading discrimination recovery, per the qmirt Gate C precedent)
  have real headroom on either bed.
- **TIMSS and SLAM play different roles entirely.** TIMSS is a clean,
  unsaturated, non-KC static calibration bed; SLAM is currently
  unavailable and, even once downloaded, ships no KC tag at all. Neither
  competes with KDD/EdNet on the growth/decoupling/anchor axes this triage
  was built to measure; they answer different downstream questions.

---

# STAGE-0 bed triage -- extension (XES3G5M, Junyi 2020, Eedi NeurIPS 2020)

The three sections below extend the triage above to three newly landed
beds, computed with the same aggregator and the same metric definitions
(`kt-mirt/scripts/triage/triage_common.py`, unchanged), so the numbers
are directly comparable to the four beds above. Scripts:
`kt-mirt/scripts/triage/{triage_xes3g5m,triage_junyi2020,triage_eedi2020}.py`.
Outputs: `kt-mirt/_planning/triage/{xes3g5m,junyi2020,eedi2020}_stats.json`.

## Sampling and runtime (new beds)

| Bed | Coverage | Rows/interactions | Runtime |
|---|---|---|---|
| XES3G5M | full local extract (both kc_level and question_level train_valid files) | 4,446,374 true interactions (question_level grain); 5,139,044 KC-slots (kc_level grain) | 23 s |
| Junyi 2020 | full file, no sampling needed | 16,217,311 problem attempts | 38 s |
| Eedi NeurIPS 2020 (task 1/2) | full file, no sampling needed | 15,867,850 question attempts | 82 s |

All three are far under the 20-minute per-bed budget; none needed the
fixed-seed user-sampling fallback the task spec allowed for.

---

## XES3G5M

Two local extracts cover the same log at two granularities, both pyKT
convention: `kc_level/train_valid_sequences.csv` pre-explodes multi-KC
questions into one fixed-length-200-window position per (question, KC)
pair (same question id/response/timestamp repeated once per tagged KC);
`question_level/train_valid_sequences_quelevel.csv` keeps the true
one-position-per-real-interaction grain, with multi-KC questions'
concepts joined by `_`. Padding in both files is `selectmasks == -1`,
masked out strictly on `selectmasks == 1` throughout. Following the task
spec's own division: (a) overall correct rate/response distribution and
(e)/(f)/(g) KC-pair co-occurrence and item-KC arity/anchors are computed
from question_level (true-event grain, no KC-driven row duplication,
matching how KDD/EdNet computed their (a) block at the un-exploded slot
grain); (b)/(c)/(d) per-KC and per-(learner,KC)-opportunity stats are
computed from kc_level (the pre-exploded per-KC grain). A programmatic QC
check (not merely assumed) confirmed every uid's chunk-rows are
contiguous in file order and every uid carries a single fold value across
all its rows (both fractions 1.0), which is what licenses treating
on-disk row order, then on-disk position order within a row, as each
student's true chronological interaction order for the opportunity/growth
computation.

| | value |
|---|---|
| distinct leaf KC | 865 (matches the ~865 expectation exactly) |
| overall correct rate (question_level, true-event grain) | 0.795 |
| per-KC correct-rate q1 / median / q3 | 0.729 / 0.809 / 0.887 |
| frac KC > 0.85 | **0.369** |
| opportunity q1 / median / q3 | 1 / 2 / 3 |
| frac opportunity >= 5 / 10 / 20 | 0.139 / 0.026 / 0.0007 |
| pooled growth curve, opportunity 1 -> 10 | 0.778 -> 0.816 |
| frac KC with positive slope | 0.676 |
| decoupling: pairs w/ cross-occurrence, frac clearing 0.75 | 759, **0.633** |
| anchors: frac KC with >= 3 pure items | 0.449 |
| item-KC arity mean / max (full question bank, 7,618 items) | 1.164 / 6 |

**Scale check.** train_valid_sequences.csv alone (used here, per the task
spec) covers 14,453 students; the held-out `test.csv` (not read by this
script) adds 3,613 more, 14,453+3,613=18,066, matching the ~18k
expectation almost exactly. Distinct questions (7,618) and leaf KCs (865)
match the ~7.6k/865 expectation closely to exactly. The true-interaction
count here (4.45M) undercounts the ~5.5M expectation only because
`test.csv` is excluded, as the task spec directed; the kc_level (KC-slot,
i.e. exploded-by-tag) grain count of 5.14M is much closer to 5.5M, so the
original ~5.5M estimate this task was checked against most likely counted
KC-slots rather than deduplicated interactions -- flagged here rather
than silently reconciled.

**Verdict.** Saturation sits between the two original KT beds: 79.5%
overall correct with 37% of KCs already past the 0.85 ceiling, worse than
EdNet (65%, 2%) but better than KDD (82-86%, 37-52%), so XES3G5M's
headroom is real but already partly spent, closer to KDD's profile than
EdNet's. Growth is present but modest and mostly front-loaded: the pooled
curve rises 0.778 to 0.816 (about 4 points) over ten opportunities with a
small dip at opportunity 8, and 68% of KCs have a positive slope, weaker
in both magnitude and cleanliness than KDD's 12-19-point curves or
EdNet's cleanest-of-the-original-four curve. Opportunity density is
thin and close to EdNet's, not KDD's or Junyi's: median opportunity count
is 2 and only 2.6% of learner-KC pairs reach 10 opportunities, so pooled
growth reads are usable but individual-trajectory work would be
data-starved for most learner-KC pairs here. Decoupling is moderate,
63.3% of the top-30 practiced pairs clear 0.75, below both KDD
KTracedSkills (80%) and EdNet (87%) from the original four, though well
above KDD SubSkills' 27%. Anchor feasibility is comparable to EdNet's:
mean item-KC arity 1.16 (genuinely multi-tag, unlike KDD's mostly-single-
KC steps), 45% of KCs have at least 3 pure items. **Net: XES3G5M is a
usable but not standout bed on any single axis** -- its profile sits
between KDD and EdNet on saturation and decoupling, and it does not lead
any of the four target axes (G1 decoupling, G2 population growth, G2
individual-rate depth, A2 option tracing -- it has no option-level
response field at all) against the full seven-bed field.

---

## Junyi 2020

Info_Content.csv ships a strict 4-level topic hierarchy per exercise
(level1_id has 1 distinct value, all "math"; level2_id has 10, median 108
exercises each; level3_id has 42, median 23 each; level4_id has 171,
median 8, max 18). This release has no separate prerequisite/skill field
(verified in an earlier pass over this dataset), so level4_id, the finest
level offered, is used as the KC layer here: it lands at a plausible
single-skill grain, comparable in exercise-count-per-KC to KDD's
SubSkills granularity, where level2/level3 are closer to a coarse
subject-area split. Log_Problem.csv's own column literally named `level`
is an unrelated per-exercise adaptive difficulty/mastery-ladder stage
(0-based), not a KC candidate, and is not used anywhere in this triage.
Because level1..4 is a strict tree, every exercise (and so every
practice slot) carries exactly one KC by construction: item-KC arity is
trivially 1.0/1 everywhere, and two distinct KCs can never co-occur on
the same slot, so the KC-pair decoupling metric is structurally vacuous
here (0 pairs with cross-occurrence, not an unlucky sample). A full
streamed pass over all 16,217,311 rows (chronological per-user order
recovered by sorting the ISO-formatted `timestamp_TW` string directly,
no expensive datetime parse needed) ran in 38 seconds, so the task
spec's 50k-user sampling fallback was not needed.

| | value |
|---|---|
| distinct leaf KC (level4_id) | 171 |
| overall correct rate | 0.704 |
| per-KC correct-rate q1 / median / q3 | 0.546 / 0.650 / 0.725 |
| frac KC > 0.85 | **0.018** |
| opportunity q1 / median / q3 | 7 / 19 / 44 |
| frac opportunity >= 5 / 10 / 20 | 0.879 / 0.704 / 0.498 |
| pooled growth curve, opportunity 1 -> 10 | 0.676 -> 0.726 |
| frac KC with positive slope | 0.708 |
| decoupling | structurally N/A, 0 pairs (see above) |
| anchors: frac KC with >= 3 exercises | 0.942 (trivial: arity is always 1) |
| item-KC arity mean / max (full exercise bank, 1,330 items) | 1.0 / 1 |

**Verdict.** Saturation risk is low, similar in spirit to EdNet: 70.4%
overall correct, only 1.8% of KCs (3 of 171) past the 0.85 flag, median
per-KC rate 0.65. Growth headroom is present (0.676 to 0.726, about 5
points) but noticeably noisier than any of the four original beds: the
curve peaks at opportunity 5 (0.746), dips at opportunity 6 (0.723), and
plateaus around 0.72-0.73 for the rest, and the median per-KC slope
(0.0023) is roughly a quarter of KDD's or EdNet's (0.006-0.014), so the
pooled positive trend is real but shallow. Opportunity density is by far
the deepest of any bed in this triage, including the original four:
median opportunity 19, third quartile 44, with 70% of learner-KC pairs
reaching 10 opportunities and half reaching 20, well past KDD
KTracedSkills' median of 8 and XES3G5M's median of 2. **This makes Junyi
the standout bed for individual-learner-trajectory work** (the G2
individual-rate axis), the one place in this whole triage where a
per-learner-per-KC growth curve, not just a pooled one, has real data
behind it. Decoupling and pure-anchor-in-the-Gate-C-sense are both
structurally not applicable, for the same underlying reason: level1..4
is a single-parent tree, not a Q-matrix, so no two KCs ever share an
item and "pure anchor" collapses to "has >=3 exercises," a much weaker
property than the multi-KC-Q-matrix anchor tests the other beds support.
Junyi therefore cannot serve G1 (influence/decoupling) work at all under
this KC scheme without acquiring or constructing a genuine multi-tag
Q-matrix for these exercises, a real limitation worth flagging rather
than working around silently.

---

## Eedi NeurIPS 2020 (task 1/2)

subject_metadata.csv is a strict tree (Level 0-3, 388 subjects);
question_metadata_task_1_2.csv tags each of 27,613 questions with a
SubjectId list verified programmatically to always be a single
root-to-leaf ancestor chain (every non-root subject's parent is also in
the same list, holds for 100% of questions), not an independent
multi-skill tag set. The KC layer used here is the leaf: for each
question, the subject id(s) at the maximum level actually reached by its
own tag list (branches differ in depth, so this is not a fixed global
depth cut). 91.3% of questions have exactly one such leaf; 8.7% have two
or more tied-depth leaves, a question spanning two equally-specific
sub-topics, which is what gives this bed a genuine, if modest, multi-KC
Q-matrix (arity mean 1.10, max 6) unlike Junyi's strictly single-parent
tree. DateAnswered (needed for opportunity ordering; not present in
train_task_1_2.csv itself) was joined in from
`answer_metadata_task_1_2.csv` via AnswerId, 100% coverage, and sorted as
a plain ISO-formatted string, the same no-parse-needed trick used for
Junyi. A full pass over 15,867,850 rows ran in 82 seconds, again with no
sampling needed. Separately, 48 of 27,613 questions (0.17%) have an
inconsistent `CorrectAnswer` value across rows, a known minor wrinkle of
this export; `IsCorrect` is used as-shipped everywhere and this does not
affect (a)-(g), noted for completeness.

| | value |
|---|---|
| distinct leaf KC | 314 |
| overall correct rate | 0.643 |
| per-KC correct-rate q1 / median / q3 | 0.580 / 0.638 / 0.699 |
| frac KC > 0.85 | **0.041** |
| opportunity q1 / median / q3 | 1 / 2 / 5 |
| frac opportunity >= 5 / 10 / 20 | 0.289 / 0.107 / 0.018 |
| pooled growth curve, opportunity 1 -> 10 | 0.639 -> 0.647 |
| frac KC with positive slope | 0.557 |
| decoupling: pairs w/ cross-occurrence, frac clearing 0.75 | 993, **0.967** |
| anchors: frac KC with >= 3 pure items | 0.857 |
| item-KC arity mean / max (full question bank, 27,613 items) | 1.104 / 6 |

**Bed-specific addition: option tracing (task's "A2" ask).** Chosen-
option (`AnswerValue`) and correct-answer-position distributions are both
close to flat across the four positions (24.1/26.2/24.4/23.3% chosen;
24.3/25.2/26.2/24.2% correct), so there is no baked-in position bias to
control for. Among questions with at least 10 wrong answers (25,862 of
27,613), the modal-wrong-share (the fraction of a question's wrong
answers landing on its single most-common wrong option) has median 0.486
and a volume-weighted pooled mean of 0.498, both well above the 0.33
flat/no-concentration baseline for a 3-distractor question, and 47.1% of
these questions have a majority (>=50%) of their wrong answers piled on
one distractor. This is a real, substantial misconception-clustering
signal, not noise around a flat baseline.

**Verdict.** Saturation risk is low and close to Junyi's: 64.3% overall
correct, only 4.1% of KCs past the 0.85 flag. Growth headroom is the
weakest of any bed in this triage: the pooled curve is nearly flat
(0.639 to 0.647, about 1 point, with a dip to 0.632 mid-curve), the
median per-KC slope (0.0016) is the smallest of all seven beds, and only
56% of KCs even have a positive slope, barely above chance. Opportunity
density is thin, in the same range as XES3G5M (median 2, third quartile
5), not close to Junyi's depth. **Decoupling is the standout result of
this whole extension and of the seven-bed program to date**: 96.7% of
the top-30 practiced KC pairs clear the 0.75 threshold, ahead of every
other bed and KC model triaged so far, including EdNet's 87% and KDD
KTracedSkills' 80%, and this is not inflated by a diagnostic catch-all
tag the way KDD Rules' 100% was. Anchor feasibility is strong (86% of
KCs have >=3 pure items, arity mean 1.10), comparable to the better end
of KDD's range. Combined with the misconception-clustering result above,
**Eedi is now the clear lead bed for both G1 avenues at once**: A1's
general signed-transfer identifiability (via its best-in-program
decoupling) and A2's misconception-channel negative-transfer flagship
(via its real per-question wrong-option concentration), which is a
favorable, non-accidental alignment since Eedi was already earmarked for
A2 on independent grounds before this triage ran.

---

## Revised cross-bed verdict (2026-07-18, seven beds)

- **G2 individual rates (per-learner-KC trajectory depth).** Junyi 2020
  leads by a wide margin: median opportunity count 19, third quartile 44,
  70% of learner-KC pairs reaching 10 opportunities and half reaching 20.
  No other bed in the program, old or new, comes close (KDD KTracedSkills'
  median of 8 was the previous best; XES3G5M and Eedi both sit at a
  median of 2, no deeper than EdNet). If an avenue needs a genuine
  per-learner-per-KC curve rather than a pooled one, Junyi is now the
  bed to use, with the caveat that it offers no usable Q-matrix (see
  below), so it cannot simultaneously support G1 work.
- **G2 population growth (pooled early-opportunity signal).** The
  original KDD/EdNet pair still leads on magnitude: pooled curves rising
  12-19 points over ten opportunities, versus 4-5 points for XES3G5M and
  Junyi and about 1 point for Eedi. Among the three new beds, XES3G5M and
  Junyi are roughly tied and both modest; Eedi's growth signal is the
  weakest of all seven beds triaged to date, essentially flat. Nothing in
  this extension displaces KDD/EdNet as the growth-magnitude leads; Junyi
  adds depth (see above) rather than magnitude.
- **G1 decoupling.** Eedi is the new program-wide leader: 96.7% of its
  top-30 practiced KC pairs clear 0.75, ahead of EdNet (87%) and KDD
  KTracedSkills (80%) from the original triage, and not inflated by a
  non-content catch-all tag the way KDD Rules' 100% was. XES3G5M is
  moderate (63%), between KDD SubSkills and the better beds. Junyi cannot
  be scored on this axis at all under its current KC scheme: its topic
  hierarchy is a strict single-parent tree, so no two KCs ever co-occur
  on an item and the metric is structurally undefined, not merely small.
- **A2 option tracing / misconception clustering.** Eedi is the only one
  of the three new beds with an option-level response field, and its
  per-question wrong-option concentration (median 49% piling onto one
  distractor, well above the 33% flat baseline) is a real signal, not an
  artifact. EdNet (from the original four) also ships option-level
  responses and was already known to have a plausible overall option
  distribution, but its per-question wrong-option concentration was
  never computed in the original pass, so a direct, apples-to-apples
  comparison between EdNet and Eedi on this specific axis is an open gap,
  not a settled result; on the evidence computed so far, Eedi leads by
  default and is independently the bed already earmarked for the A2
  avenue.
- **Net read across all seven beds.** No single bed wins on every axis,
  and that split is itself the finding: Junyi for growth depth, KDD/EdNet
  for growth magnitude, Eedi for decoupling and option tracing, XES3G5M
  as a usable middle-of-the-road bed that does not lead any axis,
  TIMSS/SLAM playing structurally different roles outside this triage's
  four axes entirely. Program-level bed choice should be made per-avenue
  (A1/A2/A4/A5), not by a single overall ranking; this triage's job was
  to make each avenue's bet-cost visible, not to declare one bed a
  universal winner.

---

# STAGE-0 bed triage -- FINAL extension (Junyi Academy 2015, KDD Bridge to Algebra 2008-2009)

The two sections below cover the last two beds of the nine-bed program,
computed with the same aggregator and metric definitions
(`kt-mirt/scripts/triage/triage_common.py`, unchanged) as every bed above.
Scripts: `kt-mirt/scripts/triage/{triage_junyi15,triage_kdd_bridge}.py`
(the latter a thin wrapper that imports its chunk-parsing logic directly
from `triage_kdd.py` rather than forking it). Outputs:
`kt-mirt/_planning/triage/{junyi15,kdd_bridge_2008_2009}_stats.json`. NOTE
ON NAMING: "Junyi 2015" here is a different release from "Junyi 2020"
above (different files, different schema, same institution) -- referred
to as **Junyi15** throughout this section to keep the two apart.

## Sampling and runtime (final two beds)

| Bed | Coverage | Rows/interactions | Runtime |
|---|---|---|---|
| Junyi15 | full file, streamed in chunks, no sampling | 25,925,992 problem attempts | 85 s |
| KDD Bridge to Algebra 2008-2009 | full file streamed | 20,012,498 steps | 172 s |

Both are far under the 25-minute per-bed budget.

---

## Junyi Academy 2015 (EDM 2015 exercise-relationship release)

Distinct dataset from Junyi 2020 above: this is the original EDM 2015
paper's release (Chang, Hsu & Chen), shipping one problem-attempt log
(`junyi_ProblemLog_original.csv`, 25.9M rows, 247,606 users, 722 of 835
distinct exercises actually attempted) and one exercise table
(`junyi_Exercise_table.csv`, 837 rows, 2 exact-duplicate rows dropped to
835 distinct exercises) that carries a `topic`/`area` tag pair and a
free-text, comma-separated `prerequisites` cell. `relationship_annotation_
{training,testing}.csv` (human similarity/difficulty/prerequisite
opinion scores for a different modeling task) are not KC or log data and
are not consumed here.

**KC layer.** `area` (8 values) is a coarse subject split; `topic` (40
values, median 16 exercises/topic, range 4-57) is the finest tag this
release ships and is used as the KC here, per the task spec -- stated
explicitly since, as with Junyi2020, it is a repurposed content tag, not
a purpose-built skill tag. There is no finer skill field. Because
prerequisites relate EXERCISES to each other, not topics, the
exercise-level grain is not run as a second, degenerate KC-as-item
battery (item universe would equal KC universe); instead it is covered
by the three extras below.

| | value |
|---|---|
| distinct topics attempted in log (of 40) | 39 |
| overall correct rate | 0.828 (true 82.8% / false 17.2%) |
| per-topic correct-rate q1 / median / q3 | 0.663 / 0.758 / 0.811 |
| frac topics > 0.85 | **0.128** (5 of 39) |
| opportunity (per learner-topic) q1 / median / q3 | 3 / 8 / 32 |
| frac opportunity >= 5 / 10 / 20 | 0.696 / 0.459 / 0.332 |
| pooled growth curve, opportunity 1 -> 10 | 0.626 -> 0.815 |
| frac topics with positive slope | 0.974 (38 of 39); median slope 0.0174 |
| decoupling (topic-pair grain) | structurally vacuous, 0 pairs (see below) |
| decoupling (exercise-pair grain, top-30 by volume) | 2 of 30 clear 0.75, **0.067** (see below) |
| anchors: frac topics with >= 3 pure exercises | 1.0 (all 40, full 816-exercise tagged bank) |
| item-KC arity mean / max | 1.0 / 1 (trivial: single-tag by construction) |

**(i) Prerequisite graph.** 835 nodes (all deduped exercises), 981
distinct directed edges (prerequisite -> exercise-that-requires-it), 979
excluding two genuine self-loops (`number_sense_length_l1`,
`proportions_1`, each listing itself as its own prerequisite -- a real
data quirk, not a parsing artifact). **Not a DAG**, with or without the
self-loops: Kahn's-algorithm topological sort only orders 617 of 835
nodes including the self-loops, 623 of 835 excluding them, so 212-218
nodes sit in at least one genuine multi-exercise cycle even after the
two trivial self-loops are set aside. This is worth flagging on its own:
a knowledge-map curated well enough to publish nonetheless contains
prerequisite cycles, so "prerequisite" here should be read as a curated
pedagogical relation, not a strict partial order, and any downstream use
assuming acyclicity (e.g. a topological curriculum ordering) needs to
handle or break these cycles explicitly rather than assume none exist.
Out-degree (this exercise is a prerequisite for N others) and in-degree
(this exercise requires N others first) both have median 1, q3 2 and 1
respectively, max 14 and 7; most exercises sit on the shallow end of the
graph, with a handful of hub exercises (`order_of_operations`,
`congruent_triangles_2`, both out-degree 14; `solid_geometry`, in-degree
7) doing disproportionate structural work.

**(ii) Decoupling at both grains.** At the topic grain, the metric is
structurally vacuous for exactly the reason it was in Junyi2020: one row
is one exercise, one exercise carries exactly one topic, so no two topics
ever co-occur on the same slot and `n_pairs_with_cross_occurrence` is 0
by construction. The exercise-pair grain is where this bed's real signal
for A1's external-validation direction lives, and it required adapting
the metric's notion of a "slot": since a row still only ever touches one
exercise, the per-row co-occurrence reading of an exercise pair would
trivially always be 1.0 (perfectly decoupled by construction, an equally
vacuous degeneracy in the other direction), so a "slot" is redefined
here as one learner's ENTIRE logged history -- n_A/n_B = number of
distinct learners who ever attempted exercise A/B, n_both = number who
attempted both at any point in their history, formula and 0.75 threshold
otherwise unchanged. Applied to the 979 non-self-loop prerequisite edges
(898 with attempt data on both sides in this log), the **top 30 by
co-occurrence volume clear the 0.75 bar only 2 times out of 30 (6.7%)**,
the worst decoupling number of any bed or KC model in the nine-bed
program (previous worst: KDD Algebra SubSkills at 27%). This is a real
result, not a bug, but it comes with a selection-effect caveat that
changes its interpretation: ranking by n_both here surfaces the most
foundational, near-universally-encountered exercise pairs in the
curriculum (`addition_1`x`subtraction_1`, `telling_time`x
`telling_time_0.5`, `multiplication_0.5`x`multiplication_1`, and similar),
and those are mechanically also the pairs nearly every learner eventually
attempts on both sides of, so "most practiced together" and "worst
decoupled" are strongly correlated in a way they are not for the KC-tag
pairs elsewhere in this program. Over the FULL population of 898
prerequisite edges with log data (not just the top-30-by-volume slice),
decoupling is comfortable: median 0.851, mean 0.814, **72.2% clear
0.75**. So the honest read is bidirectional: this bed's *most heavily
co-practiced* prerequisite pairs, which is what the standard top-30-by-
volume convention (used identically for every other bed's KC pairs in
this program) selects for, are badly collinear; its prerequisite pairs
*in general* are not. Any A1 design drawing on junyi15 prerequisite pairs
should pick specific edges by looking at their own decoupling number
rather than assuming volume-ranked edges are representative, and should
note that this per-learner-ever-attempted operationalization says
nothing about practice *timing* (recency/ordering) -- a session- or
order-based decoupling read was out of scope here and could tell a
different, and possibly more directly A1-relevant, story.

**(iii) Opportunity distributions at both grains.**

| | per-learner-exercise | per-learner-topic |
|---|---|---|
| n pairs | 2,289,789 | 721,581 |
| q1 / median / q3 | 2 / 8 / 12 | 3 / 8 / 32 |
| max | 5,174 | 15,112 |
| frac >= 5 / 10 / 20 | 0.655 / 0.317 / 0.132 | 0.696 / 0.459 / 0.332 |
| source | `problem_number` column, used as-shipped (independently verified against `time_done` chronological order to be an exact running attempt count) | derived: cumcount per (user, topic) after a stable sort by (user_id, time_done) |

Median opportunity count is coincidentally 8 at both grains, but the
topic grain reaches much greater depth in the tail (q3 32 vs 12,
frac >= 20 a third of pairs vs an eighth) because a topic pools roughly
16-20 exercises' worth of practice into one running count, exactly the
expected effect of coarsening the KC grain.

**Verdict.** Saturation is moderate: 82.8% overall correct sits closer to
KDD's saturated profile than to EdNet/Junyi2020's low-saturation profile,
but only 12.8% of individual topics clear the 0.85 flag (vs KDD's
37-52%), so the coarse topic grain pools easy and hard exercises together
without fully erasing headroom. Growth is this bed's standout result:
the pooled curve rises 0.626 to 0.815 (about 19 points) over ten
opportunities, with 97.4% of topics positive-sloped (38 of 39, the
highest fraction in the whole nine-bed program) and a median per-KC slope
of 0.0174, larger than every other bed's median slope computed in this
triage (KDD's best was 0.006-0.014). Junyi15 at the topic grain is
therefore the strongest growth-MAGNITUDE signal in the program, not just
Junyi2020's previously-noted growth-DEPTH signal -- worth a caveat before
either number is leaned on for a growth claim: topic pools several
exercises of likely differing intrinsic difficulty, so part of this rise
could reflect within-topic exercise sequencing (easier exercises first)
rather than pure learning, a confound this triage did not separate out
and that any downstream growth-detection design on this bed should check
directly (e.g. by conditioning on exercise identity within a topic).
Opportunity depth at the topic grain (median 8, q3 32) is deep, behind
only Junyi2020's exercise-level depth (median 19, q3 44) in the whole
program. Decoupling is genuinely two-sided, as detailed above: vacuous at
the topic grain (structurally, like Junyi2020), and badly failing (6.7%)
at the exercise-pair grain specifically for the highest-volume
prerequisite edges the standard top-30 convention selects, while the
broader population of prerequisite edges clears comfortably (72.2%) --
a program-first finding that the same ranking convention applied
elsewhere without checking the underlying distribution can silently pick
out an unrepresentative, adversarial slice. The prerequisite graph itself
is a genuine, if imperfect, asset: 981 human-curated exercise-to-exercise
edges, but not a DAG (212+ nodes sit in cycles even discounting two
trivial self-loops), so any use treating it as a strict prerequisite
order needs to handle those cycles explicitly. **Net: Junyi15 earns a
place alongside Junyi2020 as a growth-bearing bed in its own right (now
the magnitude leader, where Junyi2020 was the depth leader), while its
A1 external-validation promise is real but conditional** -- it depends on
picking specific, individually-checked prerequisite edges rather than
trusting a volume-based ranking, which here is actively misleading.

---

## KDD Cup Bridge to Algebra 2008-2009

Same DataShop transaction-log format as the Algebra 2008-2009 set above
(step-grain unit of analysis, Correct First Attempt response, `~~`-
separated multi-KC cells); `triage_kdd_bridge.py` imports its chunk
parser directly from `triage_kdd.py` rather than duplicating it. This
file's header carries only two KC(<model>) columns, KC(SubSkills) and
KC(KTracedSkills) -- there is no KC(Rules) column at all in Bridge to
Algebra 2008-2009, unlike Algebra 2008-2009 which ships all three -- so
both available models are run, matching the task's KTracedSkills-primary/
SubSkills-secondary framing without needing to drop anything. 20,012,498
steps streamed in full, 0 invalid Correct-First-Attempt rows, 0 KC/
Opportunity length mismatches for either model.

| KC model | distinct KCs | correct rate | per-KC rate q1/median/q3 | frac KC > 0.85 | opp q1/median/q3 | frac opp >= 5/10/20 | growth opp1 -> opp10 | frac slope > 0 | decoupling: pairs w/ cross-occ, frac clearing 0.75 | anchors: frac KC with >= 3 pure | arity mean/max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| KTracedSkills | 807 | 0.815 | 0.685 / 0.803 / 0.896 | 0.379 | 4 / 7 / 15 | 0.669 / 0.399 / 0.190 | 0.707 -> 0.815 | 0.737 | 489, **0.800** | 0.815 | 1.081 / 4 |
| SubSkills | 933 | 0.820 | 0.667 / 0.801 / 0.893 | 0.361 | 4 / 8 / 17 | 0.680 / 0.425 / 0.224 | 0.705 -> 0.817 | 0.738 | 934, **0.500** | 0.690 | 1.546 / 8 |

**Key question this bed answers.** The avenue map's UNVERIFIED figure for
KDD Cup Bridge-to-Algebra **2006-07** (a different, earlier vintage not
present locally) was "about 1.01 KCs/step, near one-to-one," cited as
part of that development set's appeal as the program's G1 favorite. That
exact file is not available to check directly, but this triage answers
the adjacent, checkable question: does the newer, larger **2008-09**
Bridge vintage keep that same profile? **Mostly, for KTracedSkills, and
no, for SubSkills.** KTracedSkills' arity mean here is 1.081 (max 4) --
close to 1-to-1 but measurably higher than the 1.01 claimed for the
2006-07 set, so "near-one-to-one" holds in spirit but not to the same
decimal, and its top-30-pair decoupling clears 0.75 in exactly 80% of
pairs, an EXACT match to Algebra 2008-2009's KTracedSkills figure in the
first extension of this report. SubSkills, by contrast, has arity mean
1.546 (max 8), clearly not near-one-to-one, and its top-30 decoupling
clears only 50%, half of KTracedSkills' rate. Inspecting SubSkills' top
30 pairs directly shows why: 8 of the 15 pairs that FAIL to clear 0.75
involve either a generic "DON'T TRACK ME" bookkeeping tag (e.g. "Enter
answer digit -- DON'T TRACK ME", a non-content placeholder, not a skill)
or a generic difficulty modifier ("Using simple numbers" / "Using small
numbers"), the same kind of non-content-tag artifact that inflated KDD
Rules' apparent decoupling in the original Algebra 2008-2009 section --
here the artifact instead drags SubSkills' number down, since these
catch-all tags co-occur with almost everything. KTracedSkills' top-30
list has no such artifact; its 6 failing pairs are genuine
near-duplicate skill pairs within the same operation family (e.g.
"Calculate difference digit -- no borrow" x "...from 2 digits", both
real sub-skills of digit subtraction that are taught and practiced
together by design).

**Saturation, growth, and anchors, compared to Algebra 2008-2009.**
Bridge's profile is close to Algebra's on every axis but consistently a
touch less saturated and a touch shallower: correct rate 81.5-82.0% (vs
Algebra's 82.4-86.0%), frac KC > 0.85 36-38% (vs Algebra's 37-52%, at the
low end of that range), opportunity depth median 7-8 (vs Algebra's 8-11),
growth curve rising about 11 points over ten opportunities for both
models (nearly identical to Algebra's 10.5-11.9-point rises), frac
positive-sloped KCs 73.7-73.8% (vs Algebra's 73.1-77.4%, essentially the
same). Anchor feasibility is comparable, if slightly weaker: 69-82% of
KCs have >= 3 pure items (vs Algebra's 72-91%). None of these differences
are large enough to call Bridge a meaningfully different bed from Algebra
on the growth or saturation axes; the decoupling/arity read above is
where the two vintages diverge.

**Verdict.** Bridge to Algebra 2008-2009 is best read as Algebra
2008-2009's close sibling rather than a fresh profile: same saturation
band, same modest-but-real growth signal, same anchor feasibility, and --
this triage's actual key finding -- KTracedSkills keeps a genuinely
strong, near-1-to-1-ish arity (1.08) and the identical 80% top-30
decoupling rate Algebra's KTracedSkills already showed, so the profile
that made the unavailable 2006-07 development set attractive for G1 does
carry forward into this larger, later companion set, at least
approximately (1.08 is close to but not the claimed 1.01). SubSkills does
NOT carry that profile forward; it is a materially weaker choice on this
bed exactly as it was on Algebra, and for the same reason (non-content
catch-all tags corrupting the top-30 ranking), reinforcing rather than
complicating the original triage's KTracedSkills-over-SubSkills
recommendation. **Net: if a G1 bed is needed beyond Algebra 2008-2009 or
the still-unavailable 2006-07 set, Bridge to Algebra 2008-2009 with
KTracedSkills is a legitimate, only-slightly-weaker substitute; Bridge
SubSkills is not.**

---

# Final all-nine-beds closing note (2026-07-18)

Nine beds triaged across three passes (four original, three mid-program,
two final): KDD Algebra 2008-2009, EdNet KT1, TIMSS 2019 G8 USA, Duolingo
SLAM (unavailable), XES3G5M, Junyi 2020, Eedi NeurIPS 2020, Junyi15, KDD
Bridge to Algebra 2008-2009. Every number in every section above came
from a full pass over the raw local file(s) or a disclosed, seeded
sample, computed with one shared metric library
(`kt-mirt/scripts/triage/triage_common.py`) so bed-to-bed and KC-model-to-
KC-model comparisons are apples-to-apples throughout.

The final two beds sharpen, rather than overturn, the seven-bed
synthesis above. **Growth**: Junyi15's topic grain is now the program's
strongest growth-MAGNITUDE bed (19-point pooled rise, 97.4% of KCs
positive-sloped, largest median slope of any bed), a genuinely new
program-level leader that sits alongside, not beneath, KDD/EdNet's
previous magnitude lead and Junyi2020's depth lead -- three different
Junyi/KDD-family beds now each hold a distinct growth-axis record, with
the caveat that Junyi15's topic-pooling could partly reflect within-topic
exercise sequencing rather than pure learning, unchecked in this triage.
**Decoupling**: KDD Bridge to Algebra 2008-2009 confirms, on a second,
larger sample, that KTracedSkills is the trustworthy KC model for the
KDD Cup family (80% top-30 clearance, matching Algebra 2008-2009 exactly)
while SubSkills is not, for a repeatable and now twice-observed reason
(non-content bookkeeping/difficulty tags corrupting the volume-ranked
top of the pair list); this de-risks KTracedSkills as a program default
for any KDD-family bed, not just the one originally checked. Junyi15
also delivers this program's single most important methodological
caution: the standard "rank KC pairs by co-occurrence volume, take the
top 30" convention, applied uniformly across all nine beds specifically
because it usually behaves well, can -- at the exercise-pair grain, under
a per-learner-ever-attempted slot definition -- systematically surface
the WORST-decoupled pairs in the population rather than a representative
sample, because volume and universality are mechanically correlated for
foundational curriculum exercises. This was only caught because this
bed's extra deliverables required checking the full pair population
alongside the required top-30 slice; any future bed using an ever-
attempted (rather than same-interaction) notion of co-occurrence should
check for the same effect before trusting a volume-ranked top-K read.
**Prerequisite structure**: Junyi15 additionally hands the program its
first genuine, human-curated exercise-to-exercise relationship graph
(981 edges), demonstrably not a DAG (212+ of 835 nodes in cycles beyond
two trivial self-loops), a concrete asset for A1 external validation with
a concrete caveat attached (pick edges individually; do not trust
volume-ranking; do not assume acyclicity).

No bed in the nine-bed program wins every axis, and the two closing
extensions add two more single-axis leaders (Junyi15 for growth
magnitude, confirmed KTracedSkills-over-SubSkills for the KDD family)
rather than displacing any of the seven-bed synthesis's existing calls.
STAGE-0 triage is complete for all nine candidate beds; downstream
avenue-level bed selection (A1/A2/A4/A5) should draw on the per-axis
leaders identified across both this report and its two extensions,
each with the specific caveats recorded alongside them, rather than on
any single overall ranking.
