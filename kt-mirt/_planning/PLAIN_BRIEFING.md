# Plain-language briefing: the growth-detection run

Prepared 2026-07-21. Covers the multi-day autonomous run launched on
2026-07-17. Written to be read cold, no prior context assumed. Everything
here traces to the program's own records in this folder (the plan, the
decision journal, the run log, the verdict files, and the research and
dataset reports). Nothing in this run is a real-data result; that is by
design, explained below.

---

## 1. The goal

We are building a model that watches a learner work through practice
questions and tracks, one question at a time, how their mastery of each
individual skill changes. The question for this phase is narrow and
concrete. Can that model, read through a standard Item Response Theory lens
(that is, as a learner ability plus each item's difficulty and how sharply
it separates strong from weak learners), reliably tell that a learner
actually got better at a specific skill, and tell that apart from noise. If
it can, we have a trustworthy learning-detector, which is the foundation
for the larger aim of measuring real learning inside real courseware.

The program has a second, still-future goal, detecting whether practicing
one skill helps or hurts another. This run did not touch that. It was
entirely about growth detection.

## 2. What was built and run

The run followed a deliberate arc, cheapest and safest first.

- **Set up the code.** A self-contained package was assembled so this line
  of work can move without disturbing the frozen thesis code.
- **Surveyed the field.** A literature sweep confirmed the two target
  claims are genuinely open (nobody reliably recovers signed skill-to-skill
  influence, and nobody audits these ability readouts for trustworthiness),
  and it flagged the known traps.
- **Measured nine candidate datasets.** Rather than guess, every candidate
  was measured directly from its raw files for the properties that matter
  (how often learners are already near-perfect, how many practice attempts
  there are per skill, whether skills can be told apart). No such numbers
  existed anywhere in the literature, so this step alone de-risked every
  later dataset choice.
- **Designed and built a growth-detection test harness.** This is the heart
  of the run. A generator produces synthetic practice logs where the true
  growth is known, in four deliberately different flavors, and a battery of
  pre-registered checks decides whether each detector passes.
- **Ran a large synthetic certification on a compute cluster.** The full
  battery ran across two opposite data-density settings, many random
  repeats each, on the university GPU cluster.

One rule held throughout. Certify every detector on synthetic data where
the truth is known, before making any claim on real data.

## 3. The main result, plainly

Read the two-setting certification as a partial success with well-understood
limits. Not a triumph, not a failure.

**What works (certified).** The coarse growth detector is solid. Fed data
with no growth, it correctly stays silent. Fed data with real growth, it
correctly fires. It does this in every random repeat, and, crucially, it
does it on both data-density settings we tested, a "few big skills,
practiced deeply" setting and an opposite "many skills, practiced barely
twice each" setting. This is the one clean win, and it is the exact thing
an earlier project failed to do when it worked at the whole-test level
instead of skill by skill.

"Coarse" means population-level. We can say with confidence "this group of
learners grew on this set of skills." We cannot reliably say which specific
skill grew, or for which learner. That distinction is the whole story of
the three limits below.

**The three limits, each real and each honestly recorded.**

1. **Per-skill resolution does not work, and it looks fundamental.** We can
   detect that the group grew, but we cannot reliably rank which individual
   skills grew, or by how much. The important and slightly hard part is
   this. The failure was identical at both density extremes (in fact
   marginally worse where the data was denser), so it is not a "we just
   need more data" problem. It reads as a built-in identifiability limit of
   the method, not something a bigger dataset relieves.
2. **Near-perfect skills cause false alarms.** When learners are already
   answering a skill almost perfectly (a ceiling effect), the detector
   fires "growth present" even where it should stay silent. This is a
   genuine limitation, present in both settings, and it has to be patched
   before the detector can be trusted on easy or already-mastered skills.
3. **The "how much" is unreliable, only the "whether."** The version of the
   model that carries a built-in growth mechanism can tell that growth
   happened and roughly which direction, but it badly under-estimates the
   amount, by five to ten times, and its discipline of staying quiet when
   nothing happened breaks down on thin data.

Two smaller caveats, one line each. The standard neural tracker fails a set
of trustworthiness audits, but mostly by design; that failure is itself
part of the intended story, showing why the naive readout cannot be
trusted. And one reliability check was never actually wired into the code,
so that particular per-skill claim simply cannot be made from this run, a
code gap rather than a finding.

## 4. What the hard road was

You already know this run dragged, and the reason is honest. The engine
that runs the statistical battery was too slow, and it was rebuilt several
times, each fix real and kept (batching the resampling, hand-coded
derivatives, a specialized solve for oversized skills, and finally the true
culprit, a single-threaded data-assembly step that had been hiding
underneath all the others). The deeper miss was strategic, and the records
own it plainly. For about a day and a half, progress was mistakenly
measured as "fraction of the grid finished" while a complete, reportable
result sat finished but un-summarized the whole time. Your stopping the
firefighting to ask for the bigger picture is what corrected the course.
The compute path now works, and the verdict was delivered.

## 5. The datasets

The synthetic certification deliberately used only two "profiles," meaning
synthetic data shaped to match two real datasets at opposite ends of the
density spectrum. One (matched to a math-tutor log) has a few big skills
practiced deeply. The other (matched to a test-prep log) has many skills
practiced barely twice each. Testing both extremes is what makes the
verdict robust. When a result holds at both ends, it is unlikely to be a
quirk of one data shape, and two extremes are enough to bracket the range,
so no third synthetic profile was needed.

The real datasets are measured but not yet used. Nine datasets were triaged
in all. Two sit outside the growth question (a professionally calibrated
one-sitting test with no repeated practice, and a language dataset that
turned out to be download-gated). The seven that can carry a growth claim,
including two versions of the Taiwanese practice platform Junyi, a large
Chinese K-12 set, the British diagnostic-quiz set Eedi, a widely used tutor
log called ASSISTments, and the math-tutor and test-prep logs the two
synthetic profiles were matched to, are all queued for the real-data phase.
None has been used for a growth claim yet, and by design that phase only
begins once synthetic certification licenses it.

One dataset is worth singling out. Junyi has both the strongest raw growth
signal of the entire set and a human-curated map of which skills are
prerequisites for which. That prerequisite map is directly useful for the
second, future goal (detecting skill-to-skill influence), which makes Junyi
the most valuable single dataset on the horizon.

## 6. What is planned, and the two open decisions

The verdict lays out two live paths and, correctly, does not choose for you.

**Decision A. Chase the fixable gaps toward per-skill certification.** Some
gaps are wiring or method fixes, not dead ends (the unwired reliability
check, and a named remedy for the near-perfect-skill false alarm). But be
warned. The core per-skill resolution failure is the one thing that did not
budge across a full density flip, so it reads as fundamental, not a quick
fix. This path risks turning into open-ended research.

**Decision B. Take the working coarse detector to a real-data pilot now.**
The coarse detector is the one certified, most robust thing we have, and
the primary real dataset is exactly suited to the population-level claim it
supports. The precondition is that the near-perfect-skill false alarm must
be fixed first, because on real data there is no ground truth to catch a
spurious fire, and any real-data reporting has to stay scoped to the coarse
claim only.

The two paths are compatible. The false-alarm fix is both a deliverable of
A and a precondition for B.

**Separately, the larger roadmap is still ahead.** This entire run was the
growth goal. The influence goal (does practicing skill A help or hurt skill
B, with a recovered sign) and its three planned approaches have not started.
Those are the next major block of work after this decision.

## 7. Bottom line

The program is healthy. It produced a real, honestly characterized result,
a certified coarse growth detector that survives a full inversion of data
density, and the discipline held throughout. Nothing is oversold, every
limit is understood, and both data extremes agree. This is a partial
success, not a headline, and it is exactly the kind of result the
certify-before-claiming approach is designed to produce. The single
decision that matters now is A versus B, spend more effort trying to
certify growth skill by skill (which the evidence suggests may be
fundamentally out of reach with this method), or fix the one false-alarm
bug and take the working population-level detector to real data. The
evidence leans toward B, but that is your call.
