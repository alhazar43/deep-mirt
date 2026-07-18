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
