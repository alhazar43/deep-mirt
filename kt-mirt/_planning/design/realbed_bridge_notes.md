# Real-bed bridge notes: KDD Cup 2010 Algebra 2008-2009 (KTracedSkills)

Build stage: the A4 real-bed bridge (`_planning/design/a4_design.md` v1.1,
sections 2.5 and 6). Scope: `kt-mirt/src/kt_mirt/growth/{kc_data,qmatrix}.py`
(new), additive wiring in `kt-mirt/src/kt_mirt/growth/run.py`, tests under
`kt-mirt/tests/`. No edits to the frozen design doc, LEDGER, THINKING, or
the campaign store. CPU only.

## 1. Modules and tests

New:
- `kt-mirt/src/kt_mirt/growth/qmatrix.py` -- item-to-KC expansion policy
  (`ExpansionPolicy`, `ALL_TAGS`), the ragged `QMatrix` buffer
  (`build_qmatrix`), pure-anchor identification
  (`pure_anchor_item_mask`, `pure_anchor_kc_ids`, `pure_anchor_stats`),
  and the circularity guard (`check_no_circularity`,
  `CircularQMatrixError`).
- `kt-mirt/src/kt_mirt/growth/kc_data.py` -- the chunked KDD Algebra
  2008-2009 loader (`load_kdd_kc_traced`), producing per-learner
  `kt_mirt.growth.synth.LearnerLog`s (no adaptation needed by
  `bank.build_calibration_rows`, verified in
  `test_learners_feed_build_calibration_rows`), the 3-level item
  hierarchy (`bank.build_kdd_hierarchy`), and the `QMatrix`.
- `kt-mirt/src/kt_mirt/growth/run.py` -- additive only (see section 4):
  `KddRealBedLoader` (concretes the existing `RealBedLoader` Protocol),
  `run_kdd_slice_cell` (the same bank-calibration + saturation + slices +
  PAS-G measurement layer `run_slice_cell` runs on synthetic twins,
  sourced from a real-bed loader), and a `--profile kdd_real` CLI branch.

Tests:
- `kt-mirt/tests/test_growth_qmatrix.py` -- 16 test functions (all
  unparametrized), unit tests for the expansion policy, padding, arity,
  the circularity guard (fires on total item/KC-vocabulary identity,
  tolerates incidental overlap), pure-anchor identification, and
  `pure_anchor_stats`'s field-for-field reproduction of
  `triage_common.pure_anchor_and_arity_stats` on a hand-traced example.
- `kt-mirt/tests/test_growth_kc_data.py` -- 17 test functions (one,
  `test_load_stats_row_accounting`, parametrized over 4 chunk sizes -> 20
  collected items), all on an in-test synthetic KDD-FORMAT fixture (never
  committed real data): multi-KC `~~`-split tags, an untagged row, a
  KC/Opportunity length mismatch, an invalid Correct-First-Attempt value,
  out-of-file-order chronology, and a same-timestamp `Row`-tiebreak case.
  Three of the 17 were added on the fix pass (section 6): a missing
  `Anon Student Id` (row excluded, counted), an unparseable `Row` value
  (row kept, sentinel-ordered, counted), and a missing `Step Start Time`
  (already-empty-string fallback, now counted). One test
  (`test_real_kdd_file_loads_a_small_prefix`) is gated
  `skipif(not real file present)` and only smoke-tests a 20k-row prefix of
  the real file (the full acceptance run is section 2 below, not part of
  the test suite).
- `kt-mirt/tests/test_growth_run.py` -- 5 new test functions appended
  (`test_kdd_real_bed_loader_satisfies_protocol`,
  `test_run_kdd_slice_cell_uses_same_measurement_layer`,
  `test_run_kdd_slice_cell_is_idempotent`,
  `test_main_kdd_real_profile_cli_branch`,
  `test_main_kdd_real_profile_requires_kdd_path`), all on a second,
  slightly larger synthetic KDD-format fixture, none touching real data.

Suite count: `qmatrix`+`kc_data` collect 36 items
(16 + 20 parametrized, after the fix pass's 3 new `kc_data` tests);
`test_growth_run.py` collects 33 items total (28 pre-existing + 5 new).
Full `kt-mirt/tests` collects **454** items (confirmed via
`pytest --collect-only`; was 451 before the fix pass's 3 new tests).

## 2. Acceptance check: real file vs. triage JSON

Run: `kt_mirt.growth.kc_data.load_kdd_kc_traced("data/kdd/algebra_2008_2009_train.txt",
chunksize=500_000)`, full 8,918,054-row file, no sampling. Wall clock
**121.0 s** (CPU only, single pass, well inside the "few minutes" budget).
Compared against `_planning/triage/kdd_algebra_2008_2009_stats.json`'s
`kc_models.KTracedSkills` block (the number `triage_kdd.py` itself
produced from the same raw file). Re-run after the fix pass (section 6):
**141.2 s** (same 8,918,054 rows; the +20s is CPU contention from the
concurrently-running synthetic certification campaign, not the fix's own
cost -- the fix adds a handful of vectorized boolean-array operations per
chunk, no new Python-level loop). Every number below was re-verified
after the fix and is unchanged except where marked otherwise.

**What "EXACT" does and does not prove (added on the fix pass, section
6).** Most rows below are a self-consistency check between two
closely-related implementations, not independent validation against
ground truth: `kc_data.py`'s docstring states outright that it reuses
"`triage_kdd.py`'s proven logic exactly (same file, same columns, same
CFA validity rule, same `~~`-split convention, same length-mismatch
handling)". Any misreading of DataShop's format that both scripts share
(e.g. an assumption baked into the shared `~~`-split convention) would
reproduce identically in both and never surface as a discrepancy here.
Two checks in the table ARE independent: the distinct-learner count
(cross-checked via a from-scratch `pandas` `nunique()` pass, not this
loader's own vocabulary code) and `qmatrix.pure_anchor_stats` (a
differently-structured vectorized algorithm than `triage_common`'s
dict/`Counter`-based one, unit-tested against hand-computed values in
`test_growth_qmatrix.py`). The rest of the "EXACT" rows should be read as
"this loader and `triage_kdd.py` agree with each other," which is
necessary but not sufficient evidence of correctness against DataShop's
actual semantics.

| Row | Triage | This loader | Verdict |
|---|---|---|---|
| Rows read | 8,918,054 | 8,918,054 | **EXACT** |
| Invalid-CFA rows dropped | 0 | 0 | **EXACT** |
| KC/Opportunity length mismatches | 0 | 0 | **EXACT** |
| Rows with a usable KTracedSkills tag (`a_overall.n`) | 4,419,705 | 4,419,705 | **EXACT** |
| Distinct KCs | 515 | 515 | **EXACT** |
| Overall correct rate (tagged rows) | 0.8238855308216272 (n_correct=3,641,331) | 0.8238855308216272 (n_correct=3,641,331) | **EXACT** |
| Per-KC correct rate: min/q1/median/q3/max/mean | 0.0 / 0.753714232149977 / 0.8545285935085007 / 0.9248994403637636 / 1.0 / 0.812248054041237 | identical to all 15 significant digits shown | **EXACT** |
| Per-KC correct rate: frac KC > 0.85 | 0.516504854368932 | 0.516504854368932 | **EXACT** |
| Learner-KC pairs (n) | 335,430 | 335,430 | **EXACT** |
| Opportunity q1 / median / q3 | 4 / 8 / 16 | 4 / 8 / 16 | **EXACT** |
| Opportunity max | 646 | 510 | EXPLAINED DELTA (below) |
| Opportunity mean | 17.612530185135498 | 15.892457442685508 | EXPLAINED DELTA |
| Opportunity frac >= 5 / 10 / 20 | 0.7365918 / 0.4231613 / 0.2021316 | 0.7337805 / 0.4167785 / 0.1940882 | EXPLAINED DELTA (tail-only, ~1.5-4% relative) |
| Item bank size (n_items, nonempty tag set) | 291,084 | 291,084 | **EXACT** |
| Item arity max | 3 | 3 | **EXACT** |
| Pure items total | 285,707 | 285,707 | **EXACT** |
| Frac KC with >= 3 pure items | 0.8524271844660194 | 0.8524271844660194 | **EXACT** |
| Item arity mean | 1.0218768465460142 | 1.0218802819804593 | EXPLAINED DELTA (below, Delta = 1 unit of summed arity over 291,084 items) |
| Distinct learners | not reported by triage | 3,310 | NEW, independently cross-checked (below) |
| Rows missing an Anon Student Id (new counter, section 6) | n/a | 0 | -- |
| Rows with an unparseable Row value (new counter, section 6) | n/a | 0 | -- |
| Rows missing Step Start Time (new counter, section 6) | n/a | 265,516 (2.98%) | NEW FINDING (below) |

**Explained delta 1: opportunity max/mean/tail fractions.** Quartiles
(q1/median/q3) and the pair COUNT match exactly, so the SET of
(learner, KC) slices is identical; only the tail of the length
distribution differs, and only downward (this loader's counts are
never longer than triage's). Triage's number is the raw
`Opportunity(KTracedSkills)` column value as DataShop shipped it; this
loader's number is a running count of the rows actually PRESENT in this
file for that (learner, KC) pair (`bank.build_calibration_rows`'s own
opportunity index, replayed via `slices.build_slices`, per the design's
own definition, section 1: "the opportunity index n = 1, 2, ... counts a
learner's interactions within the slice"). The two agree exactly
whenever every occurrence of a (learner, KC) pair is present in this
file, which is the common case (hence q1/median/q3 match). KDD Cup 2010's
Algebra 2008-2009 "train" file is a known train/test split of one
underlying transaction log, with later per-student interactions held out
into a companion (unloaded) test file; DataShop's own `Opportunity`
counter was computed against the full pre-split log, so a (learner, KC)
pair whose practice continues past the train/test boundary keeps
climbing in DataShop's numbering even though the intervening test-set
rows are invisible to any loader that only reads the train file. This
predicts EXACTLY the observed signature (bulk unaffected, tail thinned,
never inflated) and is not otherwise contradicted by anything in this
loader's own row accounting (which matches triage exactly everywhere the
mechanism does not predict a difference). This hypothesis was not
verified against the actual (locally absent) KDD Cup test file; it is the
best-supported explanation available, not a confirmed one, and is flagged
as such.

**Explained delta 2: item arity mean.** The delta is `(1.0218802819804593
- 1.0218768465460142) * 291,084 = 1.0000000000298899` -- i.e. the two
loaders' item -> KC-tag maps disagree on the tag SET of exactly one item
(one step, somewhere in 291,084) by exactly one tag. This is the
consequence of a documented judgment call (`kc_data.py` module docstring
note 4): this loader records a step's KC tag set at its FIRST tagged
occurrence in file order, while `triage_kdd.py` records it via a
per-chunk `dict.update` (last CHUNK wins, first occurrence WITHIN a
chunk). A step's KC tagging should be a static content property and is,
for all but one step out of 291,084; this single case is a real,
tiny data-quality artifact in the raw log (the step's tag set was not
recorded identically on every occurrence), not a bug in either script.

**New number: distinct learners (3,310).** Neither
`triage_common.KCModelAggregator` nor its JSON output tracks a distinct-
student count (only `(student, kc)` pairs), so there is no triage
baseline to agree or disagree with. Independently cross-checked via a
from-scratch `pandas` `nunique()` pass over the raw file's
`Anon Student Id` column (not reusing this loader's own vocabulary code):
**3,310** distinct students across every valid-CFA row (matches this
loader's `n_learners` exactly), of which 3,287 have at least one
KTracedSkills-tagged interaction (23 students practice only
KTracedSkills-untagged steps).

**New finding: 265,516 rows (2.98%) have no ``Step Start Time`` at all**
(added on the fix pass, section 6 -- this counter did not exist before
and the fact was invisible). Verified against the raw file directly, not
just this loader's own count: e.g. `Row` 958-962, student
`stu_de2777346f`, problem `BH1T31C`, steps R1C1/R1C2/R2C1/R2C2/R3C1, all
five consecutive rows with a real `Correct First Attempt` value and a
blank `Step Start Time`. This is a genuine DataShop data-quality property
of the raw log, not a parsing artifact of this loader or of the fix (the
value is genuinely absent in the source file). Impact on judgment call 4
below (chronological order): every such row's timestamp falls back to
`""`, which sorts BEFORE any real timestamp for that student (module
docstring note 8) -- so a student's untimed rows are placed first in
their sequence regardless of when they actually occurred, tie-broken
only by `Row` among themselves and against each other. Since untimed
rows tend to arrive in contiguous `Row`-order blocks (as in the example
above), their INTERNAL relative order is very likely correct; what is
NOT verified is their order RELATIVE TO the student's timestamped rows.
This is a real, now-quantified limitation of judgment call 4 that
previously had zero visibility (the counter this fix pass added is what
surfaced it) -- flagged here rather than silently left as an unstated
assumption. Changing the fallback ordering policy (e.g. interpolating an
untimed row's position from neighboring `Row` values) is a design
decision outside this fix pass's scope (the review asked that failures
be counted and reported, not that the sort semantics be redesigned); it
is future work if temporal precision for this ~3% of rows becomes
load-bearing for a downstream claim.

## 3. Judgment calls

1. **Row inclusion is every valid-CFA interaction, not only
   KTracedSkills-tagged ones** (kc_data.py note 1). ~50.4% of rows in the
   real file carry no KTracedSkills tag at all; excluding them from the
   per-learner stream (as `triage_kdd.py` does for its own per-KC-only
   statistics) would starve the bank calibrator of item-difficulty signal
   on every untagged step, since item difficulty is a step property, not
   a KC property. Untagged rows carry an all-`False` `tag_mask`, which
   every existing consumer (`slices.build_slices`'s `kc_mask` filter,
   `bank._BankModel.growth_term`'s masking) already treats as a no-op for
   slice membership and opportunity counting.
2. **A KC/Opportunity length mismatch is treated as untagged, not
   dropped from the interaction stream** (note 2) -- same rationale as
   (1): the row is a real step-practice event; only its KC tag is
   unusable. (0 mismatches occur on the real file, so this call is
   currently inert there, but it is exercised by the in-test fixture.)
3. **A step's KC tag set is recorded at its first tagged occurrence in
   file order**, not `triage_kdd.py`'s per-chunk-last-wins rule (note 4).
   Verified empirically inert except for exactly one step of 291,084
   (section 2's explained delta 2).
4. **Chronological order** = `Step Start Time` (lexicographically
   sortable ISO-like string, no datetime parse), tie-broken by the file's
   own `Row` counter, via a three-stage stable-argsort chain rather than
   `np.lexsort` (object/string dtype support). Not checked against any
   external ground truth (no independent timestamp source exists locally)
   beyond internal consistency (opportunity quartiles matching triage
   exactly wherever the train/test-split mechanism does not predict a
   difference is indirect evidence this ordering is correct). **Updated
   on the fix pass:** 2.98% of real-file rows have no `Step Start Time`
   at all (section 2's new finding) and sort first, by construction,
   within their student -- their order relative to that student's
   TIMED rows is unverified, though their order relative to EACH OTHER
   (tie-broken by `Row`) is very likely correct given they arrive in
   contiguous `Row`-blocks. A pre-existing fallback (`fillna("")`) whose
   real-file incidence was not previously counted or surfaced.
5. **`LearnerLog` gains no timestamp field.** Timestamps are carried on
   `KddLoadResult.learner_timestamps`, a parallel array per learner,
   rather than widening the shared, already-tested 4-field
   `LearnerLog` dataclass every other module already consumes.
6. **`pandas` is a real, currently-undeclared runtime dependency** of
   `kc_data.py` (matching `triage_kdd.py`'s own dependency). It is
   available in the `research` conda env (verified, pandas 2.3.0) and is
   already a declared dependency of the sibling triage scripts, but
   `kt-mirt/pyproject.toml` was outside this stage's permitted edit set
   (only `src/kt_mirt/growth/`, `tests/`, and this notes file), so the
   gap is flagged here rather than silently worked around. Follow-up:
   add `pandas` to `kt-mirt/pyproject.toml`'s `[project.dependencies]`
   the next time that file is in scope.
7. **Circularity-guard overlap threshold (50%)** (`qmatrix.py`
   `_CIRCULARITY_OVERLAP_THRESHOLD`) is a judgment number: the ASSISTments
   failure mode is total item/KC-namespace identity, not any incidental
   string collision, and 50% is chosen as comfortably above "coincidence"
   and comfortably below "total identity" with no data point in this
   design pinning it more precisely. **Residual risk, not a closed
   question** (raised on the review pass): a coarser circularity --
   e.g. KC tags partially DERIVED from problem/step names, or overlap
   concentrated in a minority of items -- would sit below the 50%
   threshold and go undetected. This is the sole automated guard against
   the exact failure mode this module exists to prevent
   (`avenue_map.md`'s ASSISTments lesson), so its coarseness stays an
   open risk rather than something this fix pass resolved; tightening it
   (e.g. a secondary check on overlap concentrated within a specific
   hierarchy branch) is future work, not attempted here since it is a
   design-level heuristic choice, not a code defect.
8. **Two-tier vocabulary interning** (`_intern_new_only`/`_intern_items`/
   `_intern_kc_matrix` in `kc_data.py`, note 6): a vectorized
   `pandas.Series.map` handles every already-seen key; only genuinely new
   keys (bounded by final vocabulary size -- 291,084 items / 515 KCs /
   3,310 students, not by the 8.9M row count) fall through to a Python
   loop. This is what keeps the full-file pass at 121 s.
9. **`run_kdd_slice_cell` runs the slice-only subset of design section
   8's R3 step** (bank calibration with `KDD_HIERARCHY_SPEC`, saturation
   stats, `slices.build_slices`, `gate.compute_gate_result`) and
   deliberately omits the RB0 tri-spec refit and the full permutation
   battery: those are heavier, later-stage machinery
   (`scripts/a4/prep_kdd.py`, R3-R8) outside this build stage's permitted
   file set. The result dict has no `bank_recovery`/`true_rise_per_kc`/
   `silent_kc_mask` fields (no generator ground truth exists for a real
   bed); it instead reports `qmatrix.pure_anchor_stats`.
10. **`KddRealBedLoader.load(seed)` ignores `seed`** beyond Protocol
    conformance: the current loader always reads the full file
    deterministically. A future seeded user-subsample (as the design's
    EdNet bed-table row anticipates for that bed) would consume it.
11. **`n_rows_has_kc`/`n_kc_opp_mismatch` are scoped to `valid_cfa &
    has_kc` (`triage_kdd.py`'s own `mask`), not to `has_kc` alone or to
    the raw row count** -- fixed on the review pass (section 6 below);
    was previously a genuine definitional mismatch with `triage_kdd.py`
    that the real file's 0/0 numbers happened to paper over.
12. **`Row`, `Anon Student Id`, and `Step Start Time` now get the same
    validation treatment `Correct First Attempt` always had** -- a
    missing student id excludes the row (it cannot be attributed to any
    `LearnerLog`); an unparseable `Row` is counted and given a
    deterministic last-in-tie-group sentinel instead of an
    implementation-defined cast; a missing timestamp (already
    empty-string-falling-back) gets a counter. Added on the review pass
    (section 6).

## 4. Additive run.py wiring and the byte-equivalence proof

Everything above the new "Real-bed KDD wiring" section of `run.py`
(`RealBedLoader`, `RunConfig`, `run_slice_cell`, `run_neural_cell`,
`certify_twin`, `run_campaign`, `render_markdown`) is untouched; the only
edit to an existing line is `build_arg_parser`'s `--profile` choices list
widening from `["tiny", "kdd", "ednet"]` to add `"kdd_real"`, plus one new
`--kdd-path` argument and one new `if args.profile == "kdd_real":` branch
at the top of `main()` (every other `--profile` value falls through to
the pre-existing, unmodified code path below it unchanged).

Byte-equivalence check: `run_slice_cell(cfg, "syn_ng", 0)` on a fixed tiny
synthetic config (`n_kcs=2, n_learners=12`, `bank_epochs=4`,
`tracker_epochs=3`, `act_epochs=3`, `n_perm_bed=4`, `n_perm_kc=3`,
`n_reshuffles=1`, `drill_repeats=5`, `run_act_p1=False`, seed 0), hashed
(SHA-256 of `json.dumps(result, sort_keys=True)`):

```
before edits: a3f1ef433700859d1492adcd45888858b547c1c2e196dbe789cdcb513b70caee
after edits:  a3f1ef433700859d1492adcd45888858b547c1c2e196dbe789cdcb513b70caee
```

Identical. Additionally, all 33 tests in `test_growth_run.py` pass (28
pre-existing + 5 new), confirming the synthetic wiring is unaffected by
functional test as well as by hash.

## 5. Suite status

Full `kt-mirt/tests` collects **451** items (`pytest --collect-only`,
confirmed after every edit in this stage was in place). Every one of the
451 is confirmed passing, established via two overlapping full/near-full
runs rather than one single invocation (both completed under heavy CPU
contention from the concurrently-running synthetic certification
campaign -- this build stage's hard rule is not to touch or slow that
campaign's own files, so the contention itself was left alone rather than
worked around, and simply made both runs slow, ~33-38 min wall clock
apiece):

1. A full `kt-mirt/tests` run launched partway through this stage (after
   `qmatrix.py`/`kc_data.py` and their 33 tests existed, but before the 5
   new `test_growth_run.py` tests and the `run.py` wiring were added):
   **446 passed, 0 failed, 0 skipped** (1971.5 s).
2. `test_growth_run.py` alone, run to completion AFTER the `run.py`
   wiring and all 5 new tests were added: **33 passed** (28 pre-existing +
   5 new), 0 failed, 0 skipped (1379.5 s) -- this run's 5 new tests are
   exactly the ones absent from run (1)'s snapshot.

446 (run 1, everything except the 5 new `run.py` tests) + 5 (run 2's new
tests, independently confirmed passing both alone and inside the full
`test_growth_run.py` file) = **451**, matching the final collection count
exactly, with zero failures or skips across either run. `kc_data.py`/
`qmatrix.py` are new files imported by nothing pre-existing (verified by
grep before this stage began), and the only other file touched is
`run.py`, additively (section 4) -- no pre-existing module (`bank.py`,
`slices.py`, `gate.py`, `rate.py`, `tracker.py`, `active.py`, `synth.py`,
`battery.py`, `report.py`, `newton.py`) was edited, so no mechanism exists
for this stage's changes to regress anything outside the three files
those two runs directly cover.

The two warnings both runs surface (`RuntimeWarning: Mean of empty
slice` in `test_run_campaign_writes_verdict_and_report`, `run.py` lines
889-890) are pre-existing, in a test this stage did not touch, and
unrelated to the real-bed bridge.

## 6. Review-fix disposition

A code review of this bridge raised one blocking finding and four
important ones. Disposition of each, in the review's own order:

**Blocking -- `LoadStats.n_rows_has_kc`/`n_kc_opp_mismatch` computed
over a different row universe than `triage_kdd.py`'s own basis, so the
"EXACT" 0/0 real-file match proved nothing about a shared definition.**
CONFIRMED and FIXED. `_process_chunk` previously computed
`n_has_kc = (valid_cfa & has_kc).sum()` (no `length_ok` filter) and
`n_mismatch = (has_kc & ~length_ok).sum()` (no `valid_cfa` filter) --
two different, wrong scopes, neither matching `triage_kdd.py`'s own
`mask = valid_cfa & has_kc` basis for both its `a_overall.n` and its
`n_kc_opp_mismatch`. Both are now computed as
`scoped_has_kc = valid_cfa_np & has_kc`, then `n_has_kc = (scoped_has_kc
& length_ok).sum()` and `n_mismatch = (scoped_has_kc & ~length_ok).sum()`
-- identical scope to `triage_kdd.py`, field for field. The fixture in
`test_growth_kc_data.py` gained a new row (R8: invalid CFA AND a
KC/Opportunity length mismatch) specifically to exercise the case the
review identified as unverified by the old fixture (a row that is BOTH
invalid-CFA and length-mismatched must be excluded from BOTH scripts'
mismatch count, not just one) -- `test_load_stats_row_accounting` now
asserts `n_rows_has_kc == 4` (was incorrectly 5) and
`n_kc_opp_mismatch == 1` (unchanged, but now for the right reason: R8 is
correctly excluded rather than accidentally not double-counted). Re-run
against the real file confirms the two counters are still 0 and
4,419,705/0 respectively (section 2) -- now because the definitions
genuinely agree, not because the real file happens to have zero
invalid-CFA/mismatch overlap.

**Important 1 -- no validation/counting for `Row`, `Anon Student Id`, or
`Step Start Time`; an unparseable `Row` was silently cast to an
implementation-defined int64.** CONFIRMED and FIXED (module docstring
note 8, `LoadStats.n_rows_missing_student_id` /
`n_rows_bad_row_num` / `n_rows_missing_timestamp`). A missing
`Anon Student Id` now excludes the row (previously it would have flowed
through unguarded into `_intern_new_only`, where a `NaN` key hashes
inconsistently against itself and could have silently minted a new bogus
learner per occurrence -- never observed on the real file, since it has
zero such rows, but a real latent bug). An unparseable `Row` no longer
reaches an implementation-defined cast: the fix's first attempt (sentinel
value assigned to the float array pre-cast) itself turned out to be
wrong -- `int64`'s max value is not exactly representable as `float64`,
so `float(sentinel).astype(int64)` silently overflowed
(`RuntimeWarning: invalid value encountered in cast`), caught by the new
`test_unparseable_row_number_is_counted_not_silently_cast` test failing
on first run. Corrected to zero the bad slots pre-cast, cast the whole
array to int64, THEN overwrite with the int64 sentinel post-cast --
entirely in integer arithmetic, no float intermediate. All three counters
are 0 (student id, row num) or a genuinely new finding (timestamp,
below) on the real file; three new fixture-based tests exercise each
counter's non-zero path in isolation.

**Important 2 -- the real-data "EXACT" agreement table is mostly
self-consistency between two closely-related implementations, not
independent validation.** CONFIRMED, no code change (a framing issue,
not a defect). Section 2 above now carries an explicit caveat paragraph
naming which two checks in the table are actually independent
(distinct-learner `nunique()`, `qmatrix.pure_anchor_stats`) and which are
not (the rest, which would reproduce a shared misreading of DataShop's
format identically in both scripts).

**Important 3 -- `pandas` is a real, undeclared runtime dependency in
`kt-mirt/pyproject.toml`.** CONFIRMED, NOT fixed -- `pyproject.toml`
remains outside this fix pass's permitted edit set (`src/kt_mirt/growth/`,
`tests/`, and this notes file only, per the harness's hard rules, same
restriction as the original build stage). Still flagged as a follow-up
for the next stage that has that file in scope; no workaround attempted.

**Important 4 -- the circularity guard's 50% overlap threshold is a
heuristic that would miss a subtler, partial circularity.** CONFIRMED,
not a code defect (a documented design-level judgment call, not
something this fix pass's scope covers). Section 3 item 7 above now
states explicitly that this is a residual, open risk rather than a
closed question, per the review's framing; no threshold change or
secondary check was implemented, since doing so is a design decision
(what would the secondary check even measure -- partial-token overlap?
overlap within a hierarchy branch?) rather than a bug fix, and is future
work.

**Incidental finding surfaced by fixing important 1:** re-running the
loader after adding the `Step Start Time` counter revealed 265,516 rows
(2.98%) with no timestamp at all on the real file -- a genuine,
previously invisible DataShop data-quality property (verified directly
against the raw file, not just this loader's count; section 2). This
narrows judgment call 4's (chronological ordering) confidence for that
~3% of rows specifically, as documented there; it does not change any
row-inclusion or KC-attribution number, since `Step Start Time` was
already falling back to `""` before this fix pass -- only the counter
that would have surfaced this fact was missing.

**Suite after the fix:** `kt-mirt/tests` collects **454** items (451 +
3 new `kc_data` tests), confirmed via `pytest --collect-only` (no import
or collection errors). `test_growth_kc_data.py` alone (with the fixed
scoping and the 3 new tests): **20 passed**, 0 failed, 0 skipped, fresh
run. `test_growth_qmatrix.py` (untouched by this fix pass, re-run as a
regression check since it shares `kc_data.py`'s `build_qmatrix` call
path): **16 passed**, 0 failed, 0 skipped, fresh run. This fix pass
edited exactly two files with runtime effect --
`src/kt_mirt/growth/kc_data.py` and `tests/test_growth_kc_data.py` --
touching neither `qmatrix.py`, `run.py`, nor any pre-existing module
(`bank.py`, `slices.py`, `gate.py`, etc.; confirmed by `git status`
showing no other file modified during this pass). `run.py`'s 33 tests
and the rest of the ~400 pre-existing tests do not reference any of the
changed `LoadStats` fields (confirmed by grep) and exercise no code path
this pass touched, so section 5's already-established 451-passed result
for those tests still holds without needing to be re-run from scratch.
A `test_growth_qmatrix.py + test_growth_run.py`-only confirmatory run
(49 items, covering `run.py`'s 33 tests directly) was launched and was
still executing, competing for CPU with the concurrently-running
synthetic certification campaign (same contention section 5 documented,
~23-38 min precedent), at the time this fix pass's work was handed off.
Its result was NOT available before hand-off; this is disclosed rather
than assumed. The disposition above rests on the modular argument (only
two files with runtime effect changed, both independently 100% green,
confirmed by fresh direct runs) rather than on that still-pending run's
number -- if it later surfaces a failure, it would have to be in
`run.py`'s own pre-existing 28 tests or the untouched `qmatrix.py`, since
nothing in this fix pass's diff reaches either file.
