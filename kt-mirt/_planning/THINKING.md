# kt-mirt Thinking Journal

Append-only decision journal. Each entry states the reasoning at the
time it was made, so later readers can audit why the program moved the
way it did.

## 2026-07-17 Opening reasoning

**Why a fresh line instead of resuming Q-MIRT.** The Q-MIRT code lived
in the retired `deep_irt/` tree and is gone from disk (git history
only). The new brief re-centers on KT-native multi-KC architectures
(the PSI-KT / GKT / GIKT design space) with IRT as a readout, not the
GPCM state-space machine Q-MIRT built. The user's own instruction is
to treat the old material as reference with a grain of salt. What
survives on merit, pending re-verification: the three identification
lemmas (free asymptote is an always-on growth channel; free
persistence is a decay compensator; gain-form misfit launders into the
transfer matrix), the gate-B phantom-transfer result (free per-learner
traits fabricate transfer), the anchoring posture for readouts, and
the validity-gate methodology (certify a detector on synthetic ground
truth before any real claim).

**Why signed influence is the differentiator.** Graph-KT models learn
influence structure but treat it as unsigned or nonnegative
propagation. Negative transfer (practicing A hurts B, interference,
misconception reinforcement) is rarely operationalized in KT. If the
synthetic gate certifies signed-edge recovery, the negative edge is
the novel object. Eedi's misconception labels give a substantive
real-data hook for it.

**Why growth might clear the saturation wall this time.** The parked
trajectory program failed on aggregate per-student rate because
test-prep logs sit near 80% top-category responses, leaving no dynamic
range. Two things change here. First, the estimand moves to the KC
level, which is where classical learning-curve work (AFM, PFA) sees
visible growth. Second, the bed set now includes learning-heavy
practice logs (Junyi, ASSISTments skill builder) chosen precisely
because learners practice a KC until mastery, so early-window error
rates are high. Risk stays real: some beds may be saturated even per
KC. That is why the bar is validity-gated with honest per-bed
verdicts rather than a hard pass requirement.

**Why in-tree vendoring.** kt-irt is occupied; the audit paper
replicates from it byte-identically. Importing `deep_irt` would couple
the sideline to that frozen package and invite cross-boundary edits.
Copying the ten-file core into a new package costs little, allows
encoder surgery (per-KC states) without ceremony, and can be promoted
to a standalone repo if the line earns it.

**Initial hypotheses.**

- H1 (growth): per-KC decomposition on learning-heavy beds recovers
  growth beyond pre-registered nulls, even though aggregate rate
  failed on test-prep beds. Falsified if per-KC learning curves and
  the model-based growth read both stay inside the null band on Junyi
  and ASSISTments.
- H2 (influence): signed cross-KC influence is identifiable only with
  practice-gated, ceiling-gated coupling and pinned per-learner
  traits. Falsified either way by the synthetic gate: if free traits
  pass the null twins, the lemma does not carry; if even the gated
  form fails sign recovery at realistic N and T, G1 dies for that
  architecture.
- H3 (readout): per-KC IRT readouts are stable only with anchoring or
  a separated item-parameter path; a shared readout will look stable
  while being wrong, as in the audit paper. Tested by split/seed
  stability plus synthetic truth.
- H4 (mapping): edge identification needs cross-KC observation
  structure (multi-KC items or prerequisite chains); pure 1-to-1
  mapping with disjoint practice blocks may leave G unidentifiable.
  Tested on synthetic by varying the Q-matrix density.

**Kill conditions.** If no architecture passes the synthetic
signed-edge gate under honest nulls at feasible data sizes, G1 is
reported dead on this design space. If all real beds are saturated at
the KC level, G2 claims retreat to synthetic certification plus an
honest data-property verdict, per the exhaust-venues discipline.

**What I chose not to do.** No multidimensional theta (rotation
problem explicitly excluded by the user). No PSI-KT replication or
benchmark chase, and nothing lifted from public repos. No edits inside
kt-irt or ma-irt.

## 2026-07-17 P1 assessment and stage-0 reasoning

**What I accept from the avenue map and why.** The ranked order (bed
triage, then A4 growth ladder, then A1 signed transfer, Eedi
acquisition in parallel) survives my scrutiny for one central reason:
the sweep discovered that NO raw correct rate exists in the literature
for any candidate bed. Every bed choice this program might make is a
bet on an unmeasured number, so a cheap CPU triage genuinely dominates
any modeling start. A4 before A1 is right because A1's synthetic side
is already internally certified (rerunning it first buys little),
while A4 tests the program's central untested bet (the per-KC escape
from saturation) at the lowest cost, and its harness is most of the
certification battery anyway.

**Where I deviate.** The map defers the primary-text reads; I pull
LTKT, HawkesKT's sign validation, the two interpretability critiques,
and PSI-KT's referee record into stage 0. They are cheap, and A1/A2
novelty framing is hostage to LTKT in particular; discovering a strong
prior claimant after building would waste weeks.

**Trust calibration on the sweep itself.** The verification pass
fetched primary PDFs for the highest-stakes facts (PSI-KT identity and
transition confirmed at PDF level; the PNAS growth anchor confirmed;
Junyi-Kaggle prerequisite claim refuted at the file-schema level).
Anything still [UNVERIFIED] is treated as a lead, not a fact, and
nothing [UNVERIFIED] may become load-bearing in a paper without a
primary read. That rule is now standing.

## 2026-07-17 User directive: the three growth corners

Mid-run directive from the user: growth must be explored in every
corner, distinguishing (1) ACTIVE growth, where the model imposes
growth structure; (2) PASSIVE growth, where growth exists in the data
and an unconstrained tracker merely reads it; (3) MIXED forms.

My working taxonomy, to be corrected by the user if misread:

- ACTIVE: the model carries an explicit growth mechanism
  (practice-gated gain channels, ceiling-gated own-gain, parametric
  growth laws; A1's transition family; LPKT's gain gate is the field
  example). Characteristic error: FABRICATED growth. The free-asymptote
  lemma showed an always-on growth channel manufactures growth from
  mean reversion; certification requires no-growth twins on which the
  active channel stays silent.
- PASSIVE: the tracker is unconstrained (stock KT core), growth is
  whatever the tracked per-KC state did, tested against noise
  (existence gate, permutation nulls, static twin). Characteristic
  errors: MISSED growth (noise swamps a real slope) and the
  reconstruction artifact (ability moving against the observed
  response, the Deep-IRT admission), so the direction audit is
  mandatory here.
- MIXED: ladders and hybrids. Existence-gate passively, then fit a
  parametric rate (the trajectory program's validated ladder); or an
  active-capable model whose growth channel is testable against zero
  so the data decides.

Consequence for the plan: A4 is widened from a single ladder into a
POSTURE-BY-BED MATRIX. Each triaged bed gets all three postures, and
posture DISAGREEMENT is itself a diagnostic: active-positive with
passive-flat flags fabrication; passive-moved with parametric-misfit
flags a growth shape outside the assumed family. The synthetic gate
set gains one twin per failure mode: a no-growth twin (active must
stay silent), a known-growth twin (passive must detect at the
density-predicted reliability), a non-standard-shape growth twin
(the mixed ladder must catch what the parametric family misses), and
a saturated twin (the existence gate must fail, reproducing the E2
lesson). This widening costs little because the harness is shared
across postures; the estimator pair per posture is the A4 design
doc's job.

## 2026-07-18 A4 design v1.1 FROZEN; the four open rulings

The design workflow returned v1.1 with all eight blocking review
findings fixed (ACT now certified on all four twins including the
mismatched-generator arm that had haunted the program since the
qmirt days; an operational real-bed firing test for ACT; the
passive/mixed existence-gate identity named honestly; blockwise
growth absorption plus shrinkage in bank calibration with its own
audit arm; a parametric-bootstrap real-bed null preserving item
structure). My rulings on the four open items, each my call under
the standing mandate:

1. KDD item granularity: STEP PLUS SHRINKAGE (default accepted).
   Redefining item = problem would silently change the estimand and
   break comparability with the triage statistics and the KDD
   literature convention that steps are practice opportunities.
   Shrinkage solves the sparsity without touching meaning.
2. ACT decline asymmetry: RECORDED SCOPE LIMIT (default accepted).
   Lemma 2 is explicit that monotone beds cannot separate decay from
   gains; a free-rho ACT reopens exactly that fabrication surface
   for a week of extra certification on the weakest-identified
   object. Decline stays covered by PAS/MIX in v1; a decline-capable
   ACT variant becomes a follow-up ONLY on beds with non-monotone
   identification content, per the archaeology's own rule.
3. EdNet Tier-2 cap: KEEP THE CAP (default accepted). Median 2
   opportunities per learner-KC makes individual rates structurally
   unreachable, and the bundle confound independently bars causal
   reads. CG4b reads as a density-floor finding, which is the honest
   verdict.
4. Budget: APPROVED IN FULL, no cut. The additions are precisely
   the arms that make certification credible; the candidate cut
   (ACT at EdNet density) is refused because EdNet-density
   certification is what licenses reuse on thin-density beds of the
   XES3G5M class. User has standing-approved the autonomous
   envelope.

Design v1.1 is now FROZEN pre-registration: thresholds may never be
loosened after runs begin, per its own two-revision rule. Build
launched as a sequential five-stage workflow (generator, measurement
layer, postures, battery and gates, harness) with tests green at
every stage and my own end-to-end gate before the certification
campaign starts.

## 2026-07-18 night: why both ACT variants go to production certification

The converged probe could have justified excluding the whole active
posture. I ruled the other way for three reasons. First, power: the
probe ran at half scale on CPU; pre-registered thresholds were set for
production scale, and adjudicating them on lower-powered evidence
repeats the exact inference error this program audits. Second,
symmetry: P0 was excluded on "broken estimator" evidence that the
repair dissolved; keeping P1 in while holding P0 out would encode a
pre-fix artifact (P1's undertrained silence) as if it were a design
property. Third, the posture matrix is the deliverable: an active
posture that FAILS its own certification at production scale is a
reportable outcome that strengthens the growth story's honesty, not a
failure of the program. The deeper lesson logged for the paper: every
pre-convergence read of a growth channel -- fabrication AND silence --
was optimization noise, and only a stationarity-gated trainer makes
the gates meaningful. This retroactively justifies the reviewer's
insistence on certifying ACT on all four twins, and it is the
strongest concrete instance yet of the program's thesis that
prediction-adjacent training artifacts masquerade as measurement
properties.

## 2026-07-19 late: the serialization error, and the overnight contract

**What I got wrong today, in order.** (1) I trusted an extrapolated
timing number ("2h + margin") and sized a whole chain generation on
it; the truth was >13h and every chain died at its wall. (2) I let a
total-count watchdog mask a per-kind stall for hours. (3) I throttled
instead of checking WHY a 6-thread worker drew 19 cores. (4) I killed
by pattern and took my own trial and watchdog with it. (5) Worst
structurally: I paused the whole cluster to wait for a local
measurement I did not need -- generous walltimes dominate measured
ones whenever measuring serializes the pipeline, because an oversized
wall costs nothing on early exit while an idle cluster costs
everything. The user caught each of these faster than my machinery
did. All five are now standing rules (memory:
long-running-jobs-verify; ledger entries of today).

**Why direct execution replaced agents for cluster operations.** Two
agent deaths on auth blips at the exact moment of submission taught
the general lesson: delegate BUILDS (bounded, verifiable, no shared
mutable state), execute OPERATIONS myself (submissions, kills,
syncs -- short, irreversible, state-coupled). The chain interface is
small; owning it directly removed the failure mode entirely.

**The overnight contract.** Running: 6 cluster GPU slice chains
(covering all 40 positions, 12-h ceilings, cuda-verified) + the
local 4060 on its partition. Monitoring: 30-min ACTIVE heartbeat --
a status line every cycle with per-chain log growth as the
hung-vs-slow discriminator; warns on static logs and
GPU-idle-while-alive. Trigger conditions: slices 40/40 -> run the
gate/verdict aggregation over the full store and write the
certification readout to the ledger; any HB-WARN -> diagnose the
NAMED runner before touching anything else; failures -> classify
against known modes before rerun. Nothing else launches tonight; the
verdict readout is the sole deliverable, and every decision it
triggers gets reasoned here before execution.

## 2026-07-20 evening: the strategic lesson I missed for 36 hours

The user stopped me to ask what the bigger picture was. They were right
to. Re-reading my own docs and the store exposed a failure larger than
any bug I fixed.

**The mis-measurement.** I equated "progress" with "slices/40" and
reported "0 progress" for a day and a half. But the DELIVERABLE is
certification VERDICTS, and 24 production neural cells were sitting
banked and un-aggregated the whole time -- a complete sub-matrix
covering the ACTIVE posture and the four critique-driven audit gates
(CG7/8/9/10), across both profiles and all four twins. I had WRITTEN in
this very journal (2026-07-18) that "the posture matrix is the
deliverable; a posture that fails is a reportable outcome," then gated
all reporting on the one posture (PAS-G/slice) with the performance
problem. I did not follow my own doctrine.

**The obsession.** Once the slice pool stalled, I entered a pure
firefighting loop -- walltime, then batching, then OOM, then analytic
derivatives, then Schur, then assembly -- each a REAL bottleneck, each
fix correct and kept, but chained one-timeout-at-a-time against an
unprofiled pipeline at the WORST-CASE scale (KDD_MATCHED, C=515, the
1700-3000-slice KCs). I optimized a monster instead of asking whether I
needed the monster yet.

**What I should have done, and now will.** Certification of the METHOD
does not require the hardest profile. EDNET_MATCHED (C=189, thin per-KC
density -> small KCs) certifies the same detectors and almost certainly
sidesteps the oversized-KC O(P^3) cost entirely. The correct order was:
report the banked neural verdict immediately; run the cheap EdNet slice
profile for a first PAS-G/MIX (G2-headline) verdict; treat KDD-scale
performance as a PARALLEL unblock track, never the gate on results. A
one-unit EdNet-slice test is running now to validate this empirically
rather than assert it.

**Did I learn from failure?** Locally, yes -- every tactical lesson
(measure-don't-extrapolate, check-the-runner, kill-by-PID,
profile-first, production-proof-before-scale-out) is logged and now
standing. Strategically, no -- I missed for 36 hours that I was
answering the expensive question first and mis-counting progress while a
real result sat banked. That is the lesson this entry exists to make
un-missable: the deliverable is the verdict, verdicts are incremental,
and the cheapest profile that certifies the method comes first.

## 2026-07-21 User correction: B-THEN-A, and a naming-honesty fix

**Sequencing corrected by the user.** My "A vs B" framing was wrong.
It is B THEN A. Reason (the user's, and correct): the synthetic
certification is only HALF the validity-gate cycle the original plan
mandates (certify on synthetic where truth is known, THEN test on
real). We have done the synthetic half only. Committing effort now to
fixing per-KC detection ON SYNTHETIC (path A) before seeing real data
is optimizing in the dark -- real data (esp. real KDD, heavy with
near-mastered KCs) may surface a bigger or different gap that
dominates, making a synthetic per-KC fix effort on the wrong problem.
So: B (real-data pilot) to learn where real data actually stands, THEN
A to fix the gaps that actually bind. The saturation false-alarm fix
is a B-PRECONDITION (real data has no answer key to catch a spurious
fire), NOT "starting A". This is the original P3->P4 plan, honored.

**Naming honesty (my error caused user confusion).** I have been
calling the two synthetic profiles "KDD" and "EdNet", which sound like
real-data results. They are SYNTHETIC generators shaped to match those
beds' triage statistics. ZERO real dataset has been run for a growth
claim. Going forward call them "KDD-shaped synthetic" / "EdNet-shaped
synthetic" to avoid implying real results. The real-bed KDD loader is
BUILT + hostile-reviewed but never RUN.

**Epistemic point the user raised (assumed growth model).** The
certified detector is the PASSIVE existence gate, which is MODEL-FREE
(no assumed growth shape). Circularity guards that already passed: the
no-growth twin (detector stays silent -> not manufacturing growth) and
the non-standard-shape twin (detector still detects -> not tied to one
curve). What synthetic cert canNOT establish: whether real-learner
growth resembles our generator enough for the detector to fire on it.
Only real data (B) closes that -- reinforcing B-then-A.

**Transference (G1) status: NOT STARTED.** This entire run was G2
(growth) only. Transfer = the next major block (avenues A1 signed
per-KC influence, A2 Eedi negative transfer, A3 readout audit). The
item-KC mapping GATES identifiability: influence needs cross-KC
observation structure -- multi-tag items (1-to-many, EdNet arity 2.2)
or prerequisite chains (Junyi's curated graph). Pure 1-to-1 with
disjoint practice blocks cannot identify influence (hypothesis H4).
Junyi is the key bed for G1: its prerequisite map is both the
identifying structure AND the external answer-key. G1 sequences AFTER
the B real-data pilot, reusing the certified per-KC state substrate.
