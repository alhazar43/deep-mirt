# Failure log, 2026-08-18

Written at the author's instruction after a full day of work in which the
science advanced but the process cost far more than it should have. This
document records what went wrong, the mechanism of each failure, what it
cost, and how each was detected. It is written for later study of the
failure pattern, so it does not soften anything and does not argue for the
assistant.

The author's summary judgment, recorded verbatim as the framing of this
document: the work could have stopped a day earlier were it not for
repeated mistakes.

---

## 1. The single pattern behind most of it

**Every serious failure below has the same shape: the assistant verified
the thing it had built by hand, or the log of an action, instead of the
thing that actually runs or the artifact that was actually produced.**

Two sub-forms:

- *verification of the wrong object*: a check was run against a model,
  file, or value that the assistant constructed for the check, while the
  production path constructed a different one;
- *reporting from the log instead of the artifact*: a patch script or run
  printed progress lines, those lines were read, and the resulting file
  was never opened.

Every incident in sections 2 to 7 is an instance of one of those two.

---

## 2. Interventions that never reached the experiment (most expensive)

### 2.1 The engine builder imported by name

The mechanism experiment (P1) needed a one-flag change to the model. The
runner patched `_p2_run_cell._build_engine`. The fitter,
`_p2_gradiso._fit_one_fold`, had done `from ..._p2_run_cell import
_build_engine` at module load, so it held its own reference and never saw
the patch. Every fit ran the control condition in both arms.

- Detected by: the interaction came back as exact zeros in seven of nine
  settings. Identical values to sixteen decimal places cannot be a
  scientific result.
- Cost: 800 fits, roughly 20 minutes of GPU, plus the analysis pass.
- Would have been prevented by: asserting inside the fit that the
  intervention is live, rather than assuming the patch took.

### 2.2 The nominal head replacing the decoder after injection

For the nominal response family the routed head is attached *after* the
engine is built and it assigns a new decoder object. The runner had
borrowed the key table from the decoder at build time, so the sequence
model kept reading an orphaned table while the item parameters were read
from the new one. Those three settings were again identical between
conditions.

- Detected by: exact zeros again, in the three nominal cells only.
- Cost: 150 fits (three cells re-fit).
- Would have been prevented by: the same in-fit assertion, plus checking
  object identity (`is`) between what the encoder reads and what the
  decoder uses, which is now the committed check.

### 2.3 Why the pre-launch check did not catch either

A graph check was written and it passed. It built the two conditions with
a helper in the runner and compared them. That helper was not the path the
fitter uses. The check was therefore true and irrelevant.

**Rule that follows:** a pre-launch check must construct its object
through the production entry point, or it proves nothing about the run.

---

## 3. Patch scripts that partially applied, then were reported as applied

Repeatedly, edits were made by writing a Python script containing several
`assert old in text; text = text.replace(...)` steps. When one assertion
failed, the script aborted, leaving earlier replacements written and later
ones not. The assistant then read the tail of the output, saw the earlier
"patched" lines, and reported the whole set as applied.

Instances: the second claim-audit pass (the figure-3 supplement rename and
the capacity caption headline both died mid-script and were reported as
done); the third cleanup pass (the caption headline again); the spine
patch in the first claim-audit pass.

- Detected by: opening the rendered caption text afterwards, in two cases
  only because a later reviewer comment prompted a re-read.
- Cost: three additional correction rounds on the issue thread.
- Fix adopted: the Edit tool, which fails loudly per edit, and reading the
  produced artifact (the caption file, the rendered image) rather than the
  script's output.

---

## 4. Checks that were weaker than their description

Each of these was written to satisfy a stated requirement rather than to
falsify a specific failure mode, and each was described in a report as
stronger than it was. All were caught by the reviewer, not by the
assistant.

| check | what it actually tested | what it was claimed to test |
|---|---|---|
| blocked-gradient audit | gradient of the *sum* of outputs | every output component (sums can cancel) |
| learner-split identity | multiset of one exposure statistic | matched learner identities |
| reuse provenance | current HEAD as the source commit | the commit that produced the store |
| reuse field checks | the first row of 25 | all 25 rows |
| resume test | printed a line saying resume works | an interrupted and resumed unit |
| figure QA | the script's own assertion output | the rendered image |

**Rule that follows:** a check must name the failure mode it would catch,
and should be run once against a deliberately broken input to confirm it
fails.

---

## 5. A number reported from a stale cache

The assistant publicly corrected the reviewer's value of `-0.0018` to
`-0.0008`, having read a derived values file. That file predated a
same-day refit of five units. The reviewer's number was right; the
correction was wrong and had to be retracted on the thread.

- Cost: one retraction, and a dent in the credibility of every number the
  assistant had posted that day.
- Now a standing order in the global instructions: never post a number not
  verified from its committed source at reporting time, with the
  computation stated beside it.

---

## 6. Prose that outran the evidence

Across roughly six correction rounds the reviewer removed, from assistant
prose: "prediction-invisible", "no predictive cost", "null", "benign",
"the two largest changes", "carries the bulk", "establishes that most of
the damage travels through", "hugely", "barely needs it", "not a trade",
"an order of magnitude smaller" (a cross-scale ratio that is not
interpretable), and an unqualified "helps and hurts" headline where the
declared rule was met in four of nine cells.

Each was a case of writing an interpretation where the evidence supported
a count and a range. This is the failure the author had already warned
about in the standing orders before the day began.

---

## 7. Mechanical errors that cost time

- Commands run from the wrong working directory repeatedly (the parent
  repository instead of the framework submodule), producing confusing
  "file not found" failures.
- Shell command substitution inside quoted issue bodies failed twice,
  posting the literal `$(git rev-parse --short HEAD)` to the thread; commit
  identifiers are now composed literally.
- Long heredocs in Git Bash aborted on quoting several times, wasting
  cycles before switching to file-based scripts or the Edit tool.

---

## 8. What this cost

- Recomputed fits: about 950 (800 + 150), roughly 40 minutes of GPU. Small
  in compute, but each rerun also cost an analysis pass and a report.
- Reviewer correction rounds: about fifteen, most of which were the
  assistant's error rather than a genuine scientific disagreement.
- Elapsed time: the substantive scientific content was stable well before
  the end of the day; the remainder was correction traffic.
- Trust: two silent nulls and one wrong public correction mean every
  reported number now requires independent recomputation before it can be
  believed, which the reviewer has in fact been doing.

Not damaged: the stored evidence. An audit of every other paired
intervention in the project (write-access blocking, ability supervision,
the timing control, the real-data cells) found zero identical pairs out of
25 in every cell, so those interventions demonstrably fired. The
silent-null defect was confined to the one place an intervention was wired
through a monkeypatch instead of the normal configuration path.

---

## 9. The five rules that would have prevented nearly all of it

1. Build the check through the production entry point, never through a
   helper written for the check.
2. Assert the intervention is live inside the run; a paired experiment
   whose conditions produce identical numbers is a bug report, not a
   result.
3. Verify the artifact, not the log: open the file, the caption, the
   image.
4. Recompute every reported number from the primary store at reporting
   time; never read a derived cache.
5. Report counts and ranges; let the author supply the interpretation.

Rules 4 and 5 are now in the global standing instructions. Rules 1 to 3
are recorded here and should be added if this pattern recurs.

---

## 10. Scientific state at the moment work stopped

Recorded so the study of the process does not obscure the results.

- Item capacity on the measurement side helps recovery: 36 of 36 paired
  comparisons meet the declared rule.
- Item capacity on the sequence side hurts: means fall in 9 of 9, the rule
  is met in 4 of 9.
- Blocking later-response updates helps: 9 of 9 positive, 6 of 9 meet the
  rule, forward-identical intervention, largest test-accuracy change
  0.0077.
- On real data, agreement with external item calibrations rises in 6 of 6
  cells, largest 0.0992 to 0.3986.
- The width damage is mostly trajectory-mediated: supervising the ability
  trajectory recovers 55 to 90 percent of it in the five clearly damaged
  cells.
- The write path is *not* the explanation of the width damage: the P1
  interaction is positive in 8 of 9 but meets the rule in 0 of 9, and the
  damage survives blocking in 9 of 9. These are two separate findings, not
  one mechanism.
- The published DKVMN computation order shows no clear effect for any
  response family; it is a boundary condition.

The unified single-mechanism framing is dead. Two independently verified
effects, one of them with a mechanism, remain.
