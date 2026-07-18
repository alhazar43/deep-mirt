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
