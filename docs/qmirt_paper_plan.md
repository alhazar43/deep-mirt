# Q-MIRT paper plan v1.1 (plan for the plan; DRAFT FOR APPROVAL)

SUPERSEDED IN PART (2026-07-05 evening): the author redirected the design
(KT encoder + IRT decoder; growth modeled from event content; one-to-many
mapping reconsidered; generative training considered). The MODEL sections
of this document are superseded by docs/qmirt_blueprint.md v1.1 (sweep-
grounded, twice adversarially reviewed). Everything else here remains in
force and is inherited by the blueprint: certification instruments and
results, bank recipe, lemmas, rulings, cast, operational rules, venue
posture. Start at the blueprint; return here for the inherited machinery.

The plan of record for the second paper, the successor to the CAEAI manuscript
("On the Prediction-Recovery Trade-off in Interpretable Knowledge Tracing",
overleaf-sync/main_caeai.tex, submitted). Q-MIRT is the internal thread name
only; the paper never uses it. This document is the contract an ultracode
session executes phase by phase; each phase opens by writing its own detailed
spec (the exposure_rerun_plan.md pattern) and closes with a gate ruled by the
lead and the editor. Evidence trail: docs/overnight_transfer_active_campaign.md
(venues 0-4), docs/overnight_findings.md Part A, memory
qmirt-learning-transfer-paper.md.

v1.1 (2026-07-04): v1 was reviewed pre-approval by two adversarial opponents
(an editor pass and a hostile psychometric-referee pass, both fresh-context).
Their verdicts and demands are folded in below and listed in "Adversarial
review". The two reviews converged on one point: certify the frozen bank
against misspecification, not against itself, and re-derive every spine number
under the readout the paper actually builds. v1.1 restructures the plan around
that.

Status: APPROVED IN SUBSTANCE 2026-07-04 ("go play with it"; checklist
defaults in force) and EXECUTING. Overnight session 0 (2026-07-04/05), raw
record docs/qmirt_experiment_results.md: G0 bridge PASS (recovered venue-1
machinery reproduces digit-exact on g-independent arms); G0.5 adversarial
code review PASS-WITH-FIXES (6 findings applied, incl. a sign-inverted NLL
gap call and per-concept gauge centering); P1b spine bridge PASS 9/9 on the
canonical sparse A->B under FB-OFF simple structure (score gap +0.041
+-0.009, null ~0, interference certified, reverse/cross G entries ~0 in
every signed cell; matched-bed scope caveat noted -- the mismatched-form
arm is queued; the campaign's sign reversal was the FB pathway, vindicating
R9); C1 bank certification FAILED the joint-calibration paths (alpha rank
~0.05 dynamic vs high oracle; refit-discrepancy detects it truth-free;
position-bias detector validated on infosched; MECHANISM RESOLVED
2026-07-05 by the fixed-budget MML/JML/oracle race: budget dominated the
collapse, marginalization genuinely helps, cohort spread sets the ceiling;
wide-cohort MML reaches oracle parity (0.775 vs 0.802), so D3's bank recipe
is DEMONSTRATED: marginally calibrated measurement-regime bank, frozen).
Converged-budget reruns closed the rendering debt (growth agreement 0.051
pos / 0.007 neg) and eliminated the null-twin G entirely (-0.0004);
provisional kill thresholds registered in the record. D2 is AMENDED
by three identification lemmas (fixed floor mu; rho frozen on monotone beds;
gap-scaled gains, with interference gated by (theta-floor)+, both routes
matched-form with the generator) -- see the model docstring and the record.
Measurement is state-inert by design (reference items fire no practice);
measurement-as-practice is a named robustness twin. Certification is read on
the SCORE scale as matched-null paired contrasts; NLL secondary.

## The door paper 1 opened

Paper 1 fixed and priced the ITEM side. Its scope fence is explicit: the
claims hold "without any claim about learning dynamics"; ability is validated
at one time point, never as a trajectory; the online regime is named "the
natural next experiment"; state-conditioned item heads were set aside. Paper 2
inverts the frame. The item side is calibrated, frozen, and certified, and the
person side moves against it. Fixed measurement, moving person, which is
standard growth-measurement posture. The certification burden this creates
(items must not silently absorb person motion) is the plan's first experiment,
not a footnote.

## The one sentence

The paper proves the two things a deployed tracker is trusted to report and
currently cannot certify, that the learner actually changed (growth scores in
partial-credit units on a fixed, misspecification-audited item bank) and that
practicing one concept moves another (signed transfer, certified by forecast
ablation plus a confound battery, identified exactly where practice schedules
give it support); the contribution is the certification recipe, not a new
tracker, and the model is deliberately the plainest state-space instrument
that can carry it.

## What sells, who hurts

- THE GAP: growth models measure change between occasions and trackers predict
  the next response, but nobody certifies that the change a tracker reports is
  real change. Paper 1 proved structure read off a prediction-trained model
  can be reproducible and wrong; the overnight campaign proved fabricated
  transfer looks exactly like real transfer until you ablate it in a masked
  forecast. Every "the student improved" and every "skill A helps skill B"
  read off a deployed tracker is currently uncertified. PSI-KT itself reports
  a recovered concept graph with no such certification; that is the novelty
  edge, stated plainly: we do not propose a tracker, we show how to certify
  one.
- THE GIFTS: (i) the certification recipe, forecast ablation plus the confound
  battery (correlation, co-scheduling, shuffled order, reverse direction, null
  twins), the battery being the product, the bare ablation alone was
  underpowered; (ii) the rendering recipe, growth and transfer reported as
  observed and predicted growth scores on fixed calibrated items, per learner;
  (iii) the identification boundary itself, transfer is identified where
  practice gives it support (the positivity condition) and provably not where
  curricula fully co-schedule, which tells practitioners when a transfer
  readout can be trusted at all.
- THE RECEIPTS, banked only (campaign, compensatory readout, D=3): forecast
  active gap +0.066 to +0.36 across regimes, the LOW end is the realistic
  non-monotone regime (9/9 seeds), the high end clean monotone; battery 5/5;
  isolation 1e-8; sign recovered under verified configurations. These numbers
  MOTIVATE; none of them is paper evidence until re-established under the
  simple-structure readout (P1b). Magnitude is never claimed (gauge-bound,
  CARRIED); ranges above describe evidence strength, not effect size.
- WHO HURTS: anyone reading skill-to-skill structure or growth curves off a
  prediction-trained tracker without certification; adaptive-practice systems
  scheduling remediation on fabricated transfer edges.

## Fresh read of the overnight campaign (what we keep, change, drop)

KEEP (validated, CARRIED):
- The state family: per-concept state z, diagonal decay + practice-gated
  own-gain + zero-diagonal fitted transfer matrix G as the sole cross-concept
  route (isolation verified to 1e-8); OU mean-reversion for non-monotone
  trajectories (kills the fabrication, null -0.0001); two-stage training
  (items first, freeze, then release G under L1; sweep l1_G 0.001-0.01).
- The certification instrument: masked-forecast ablation plus the venue-2
  confound battery.
- The identifiability boundary as a design rule and headline concept: transfer
  is cleanly identified only with decoupled practice (fraction of A-only
  practice >= 0.75 clean, 0.25-0.50 weak, 0 unidentifiable). Framed as the
  POSITIVITY condition (overlap): practice support where A occurs without B.
  External-validity cost conceded in the paper: transfer is identified in
  isolated drills and NOT under fully integrated curricula, where it matters
  most. Central limitation, not footnote.
- The invariance posture: uniform item drift is a location gauge, inseparable
  from uniform growth; stated as the equating assumption, STRENGTHENED (see
  R10, item re-exposure). Differential drift detectable; companion check on
  every real-data claim. Direction and existence only, never magnitude.
- Operational lessons: no detached fire-and-forget runs; sweeps checkpoint per
  cell, write JSON, return short summaries; C or KC-dim for concepts, K only
  for answer categories.

KEEP WITH A DEMOTION:
- The response-feedback (FB) pathway. It rescued within-learner tracking
  (0.10-0.42 -> 0.26-0.72) but it corrupts sparse-edge sign: R3/R4 found
  G[B,A] systematically sign-reversed for the sparse A->B case because the
  feedback projection absorbs the transfer signal and the optimizer
  compensates negative. RULING: identification and certification runs are
  FB-OFF by default; FB-ON is a prediction-quality variant only; a sign claim
  is never read from an FB-ON fit (R9).

CHANGE:
- Readout structure: flip to between-item multidimensionality (D1). All
  campaign evidence was gathered under the compensatory readout, so NOTHING
  transfers automatically; P1b re-derives every spine number and kill
  threshold under the new readout before any gate can use them.
- The presentation: score-scale, per-learner, observed vs predicted growth
  scores; latent trajectories demoted to internal diagnostics.
- The passive baseline: retracted frozen-LSTM comparison stays retracted; the
  load-bearing control is the within-condition G-zeroed arm.

DROP:
- The compensatory cross-loading readout as primary (robustness appendix
  only). Raw theta trajectories as evidence. "Beats a passive LSTM" claims.

OPEN DEBTS the campaign left, now scheduled: the forecast metric has NEVER run
under simple structure on the canonical sparse A->B case (P1b); D-scaling to
C=5,8 never ran (P3 sub-gate); the real-data leg never ran (P5); M3 rate
recovery needs rate variance in the generator (P3).

## Adversarial review (2026-07-04): verdicts and accepted demands

Two fresh-context opponents reviewed v1. Combined verdict table (worst of the
two where they differ):

- C1 measurement certified: PLAUSIBLE. Gate was self-fulfilling
  (estimator=generator); refit-discrepancy is internal-consistency only and
  blind to motion-into-items bias; threshold tau=0.152 is paper-1-calibrated
  and does not transfer.
- C2 growth real/observable: PLAUSIBLE, near FANTASY as previously written.
  The explicit state model's on-disk observed-vs-predicted growth (+1.18 obs
  vs +0.39 pred, venue 0) fails the old 0.05 kill; the 0.03 agreement on
  record came from the flexible tracker, a different model; the FB variant's
  score agreement was never measured.
- C3 transfer signed/certified: PLAUSIBLE. Strong under the retired readout;
  unproven under D1; sparse A->B sign reversal under FB is the one datum at
  the new configuration.
- C4 CAT comparison: FANTASY as pillar, PLAUSIBLE as appendix validation.
  Static-CAT-on-moving-truth is a new simulation, not a port; checkpoint
  blindness to within-window learning is true by construction (a manufactured
  win); shared-bank agreement partly tautological.
- C5 real data: FANTASY as claim, PLAUSIBLE as audit. Verdict space must
  split positive / informative-null / untestable; the decoupled-practice
  audit is itself unrun.
- D1 PLAUSIBLE (principled, PSI-KT-grounded; unproven with the core metric).
  D2 GROUNDED as family, PLAUSIBLE in the exact combo (see R9). D3 PLAUSIBLE
  (leans on paper 1 out-of-regime). D4 GROUNDED. D5 PLAUSIBLE (C=3 only,
  old readout). D6 demoted (above).

Accepted demands, folded in: bridging gate P1b re-deriving all spine numbers
and kills under D1 (editor 1); R9 sign-confound rule + sparse-A->B sub-gate
(editor 2, referee F3); calibration certified against generator MISMATCH with
anchor-first default and dual-calibration sensitivity, plus local-independence
audit (editor 3, referee F2, the convergent ONE THING); C2 re-registration
with K-normalized units and estimand named (editor 4, referee F6b); C4
demotion and redesign with disjoint pools and independently calibrated CAT
bank, manufactured win dropped, drift invoice sized honestly (editor 5,
referee F5); D>3 harness sub-gate with D=3-only fallback (editor 6); honest
receipts range with regime labels, magnitude stripped from kills (editor 7);
delta threshold re-derivation with continuous-alarm fallback (editor 8); C5
three-way verdict space (editor 9); certification-forward framing (editor 10);
measurement doctrine rewritten to item-side invariance with established naming
(referee F1); single-exposure scoring rule and re-exposure-targeted drift
check (referee F4); per-item threshold-ordering and category-usage screens on
every exhibit, deceleration decomposition (referee F6a/F6c). Referee's
license table and naming rulings adopted into the vocabulary contract (P0).

Rejected or modified: none rejected outright. Two modifications: "explainability
layer" is deleted from paper vocabulary (referee F1) but retained as internal
shorthand in scratch code comments; E6 (drift invoice) is retained as an
optional exhibit rather than dropped, on the condition it is sized honestly in
realistic regimes where it is expected to be SMALL (referee F5's own framing).

## Design freezes proposed (ratify at approval)

D1. Readout: between-item multidimensional GPCM (simple structure, Adams,
    Wilson, and Wang 1997). Every item is tagged to exactly one concept and
    read from that concept's occasion-specific ability alone through a GPCM
    head with per-item discrimination and step thresholds. Rotation dies by
    construction; each concept's scale is pinned by its own frozen item set.
    PSI-KT's readout ground generalized from Bernoulli-sigmoid to ordinal
    GPCM. The compensatory within-item arm is a robustness appendix and the
    handler for multi-KC real items.
D2. Person layer: state z_{t,c} (the state-space modeling label; on first use
    in the paper it is paired with the established term, the occasion-specific
    ability theta_c(t), occasion extended to practice step per MRMLC) updates
    by decay + Q-gated own practice gain + signed G; OU mean-reversion on by
    default. Identification and certification runs are FB-OFF (R9); FB-ON
    exists only as a prediction-quality variant. The decoder never moves after
    stage 1.
D3. Measurement layer: separate item heads (paper 1's winning design).
    DEFAULT calibration is anchor-first / baseline where the design permits
    (fixed-parameter calibration posture, FIPC), and stage-1 dynamic
    calibration otherwise; EVERY headline claim carries a dual-calibration
    sensitivity (bank frozen from dynamic stage-1 vs bank frozen from a
    static MML fit; downstream claims must be robust to which bank is
    frozen). The bank is certified by the P1 misspecification study, not by
    matched-generator recovery. The refit-discrepancy test runs with a
    re-derived threshold on paper 2's own synthetic bank, or as a continuous
    alarm if the binary flag does not transfer (paper 1's own fallback).
D4. Transfer term: G zero-diagonal, signed, L1-regularized, released only in
    stage 2, practice-gated; responses never enter the cross-concept route.
    Sign claims are made only under FB-OFF fits and verified edge
    configurations (R9), and only where the positivity condition holds.
D5. Certification: masked-forecast ablation + full confound battery + null
    twins + dose-response + sign recovery, at C=3 first, then C=5,8 behind
    the harness sub-gate. The decoupled-practice positivity condition is a
    stated identification condition of the method, reported with every claim.
D6. Comparator (appendix-level validation, demoted from pillar): simulated
    CAT checkpoints using paper 1's selection/stopping machinery REBUILT for
    a moving-truth generator (a new simulation, stated as such). Circularity
    controls: the CAT administers from a held-out calibrated item pool
    disjoint from the tracker's training items, and an arm with an
    INDEPENDENTLY calibrated CAT bank breaks the shared-instrument loop; an
    oracle-bank arm is the reference. Claims limited to level agreement at
    checkpoints and the item bill (efficiency). No "CAT misses within-window
    learning" claim (true by construction). Optional E6 prices the
    within-session drift effect honestly, expected SMALL in realistic
    regimes.

## The measurement doctrine (banked rebuttal, rewritten after review)

What the paper asserts is ITEM-SIDE INVARIANCE: the measurement model (item
parameters, link function) is fixed across occasions, calibrated once, frozen,
and certified against misspecification. The person side is an occasion-
specific ability, a latent trajectory theta_c(t), with occasion extended from
test administration to practice step; this is the growth-measurement lineage
owned explicitly (Andersen's longitudinal Rasch, Embretson's MRMLC, Fischer's
LLRA, dynamic IRT of Wang, Berger, and Burdick), and the model is a state-
space instance of it. "Trait" is reserved for time-invariant person
characteristics and never names the moving quantity. The anticipated "a
moving theta means your model is not calibrated" objection is answered in two
parts: (i) a STATIC model fit to a learning process is the miscalibrated one,
trait drift there is a misfit symptom, whereas here change is modeled
explicitly against an invariant measurement layer; (ii) the sharp version of
the objection, that stage-1 calibration can launder person motion into item
parameters, is conceded as real and answered by DESIGN (D3 anchor-first
default, dual-calibration sensitivity, the P1 misspecification study,
single-exposure scoring), not by rhetoric. All growth claims additionally
carry their observable dual, the growth score (expected-score change on fixed
items), so nothing rides on the latent's gauge. Prose bans (vocabulary
contract): never "theta grows/evolves" (the occasion-specific ability
changes; the measurement does not); never "explainability layer" in the
paper; never "trait" for the moving quantity.

## Claims and kill criteria

The spine is TWO claims, no more: dynamic learning is real (C2) and transfer
is real (C3). C1 is the measurement precondition, C4 is appendix validation,
C5 extends the claims to real data conditionally. ALL numeric kill thresholds
below are provisional until P1b re-registers them under the simple-structure
readout; inheriting compensatory-readout thresholds is banned.

C1 (measurement certified against misspecification). Three legs, all gated:
   (a) generator-MISMATCH item recovery: calibrate under the model's
   transition, generate under non-monotone, heterogeneous-rate, forgetting,
   and informatively-scheduled twins; quantify alpha and beta bias (the
   Debeer-Janssen position-effect artifact is the named threat). Pass =
   item-parameter bias bounded and downstream growth claims insensitive under
   the dual-calibration check. (b) local-independence audit: serial
   dependence from any feedback pathway must not inflate discrimination
   (FB-ON vs FB-OFF alpha comparison). (c) refit-discrepancy with re-derived
   threshold (or continuous alarm). KILL: bank cannot be certified ->
   anchor-first redesign before anything else; no growth claim over an
   uncertified bank.
C2 (growth real and observable). Within-learner trajectory recovery on
   directed synthetic; observed vs predicted growth scores on fixed reference
   items agree within a threshold set at P1b in PROPORTION-OF-MAXIMUM-SCORE
   units (K-normalized, referee F6b); held-out prediction beats a
   constant-ability null with a seed-clustered CI; estimand is per-learner
   (population bands secondary). Exhibits carry the deceleration
   decomposition (link compression and ceiling vs actual slowing, F6c) and
   the per-item threshold-ordering/category-usage screen (F6a). KILL:
   observed and predicted growth scores disagree under the certified bank ->
   the rendering claim dies; the paper collapses to a transfer-only note.
   Known risk going in: the explicit state model's venue-0 score agreement
   FAILED at 33% of observed; the fix (FB) is barred from identification
   runs, so P1b must find the agreement FB-OFF or the constant-gain family
   needs a richer own-gain (mastery-ceiling M2 is the sanctioned variant).
C3 (transfer real, signed, certified). Active forecast gap > 0 with >= 8/9
   seed configurations positive and null twins at zero; battery 5/5;
   negative-G twin recovers sign; dose-response monotone in |g|; ALL under
   FB-OFF simple structure, sign read only where positivity holds. KILL: any
   battery member fails after diagnose-fix-retry -> report the boundary; the
   paper becomes "certification catches what recovery cannot", still
   publishable, weaker.
C4 (validation, appendix). Level agreement between tracker and CAT checkpoint
   estimates on the disjoint-pool design; the independently-calibrated-bank
   arm within tolerance of the shared-bank arm (else the shared-bank number
   is discarded as circular); the item bill. KILL: agreement fails ->
   diagnose; C4 is already appendix, so failure bounds the validation
   section, not the paper.
C5 (real data, conditional, three-way verdict space). On KDD Cup 2010
   (fallback EdNet): first the decoupled-practice audit over KC pairs (the
   positivity screen; itself a reportable result about real curricula); if
   pairs pass, stage-1 + certified bank + forecast gaps + differential-
   invariance companion keyed to re-exposure counts. Verdicts: POSITIVE (CI
   excludes zero), INFORMATIVE NULL (audit passed, gap null), UNTESTABLE
   (no pairs pass positivity; reported as the identification boundary in the
   wild). Real ordinal codings stay coerced (paper 1 demoted them); the real
   leg is a bounded extension, existence only; synthetic + design carry the
   paper.

CARRIED rulings binding all claims: magnitude gauge-bound (and stripped from
every kill criterion); uniform anchor drift stated as assumption with the R10
re-exposure strengthening; seed-clustered intervals on every headline number;
sign consistency across seeds reported next to means.

## Rendering contract (the presentation fix, first-class deliverable)

Every learning claim has an observable dual. Exhibits are specified and mocked
in P2, user signs off on the figure system before proliferation (paper 1
rule). Every exhibit item passes the threshold-ordering and category-usage
screen; growth scores are reported in proportion-of-maximum units when items
of different K are compared.

E1 Growth panel: per-learner observed vs predicted growth scores on the fixed
   per-concept reference items, early/mid/late windows, with the deceleration
   decomposition (link/ceiling vs slowing) shown, plus the seed-clustered
   population band as secondary.
E2 Transfer event plot: B-concept growth score around a pure-A practice
   block, transfer twin vs null twin vs interference twin; the "learn in
   test" exhibit; category-upgrade probabilities as the ordinal rendering.
E3 Ablation ledger: forecast NLL and growth-score error, with-G vs G-zeroed
   vs null twin, per battery condition; the certification table.
E4 CAT overlay (appendix): tracker trace with CAT checkpoint estimates from
   the disjoint pool, item bill annotated; the independent-bank arm shown
   beside the shared-bank arm.
E5 Sign panel: facilitation vs interference dose-response, direction
   recovered, magnitude axis unlabeled beyond sign.
E6 (optional, honest) Static-person invoice: CAT bias/length when ability
   drifts within a session, sized in realistic regimes where it is expected
   small; dropped without regret if it stays negligible.

Language: growth score, expected score, partial credit, reference items,
occasion-specific ability, positivity condition. Naming rulings from the
referee adopted wholesale (P0 vocabulary contract).

## Prior art and collisions (recon 2026-07-04; verify every cite in P0)

Nearest neighbors and why they are not this paper:
- HawkesKT (WSDM 2021): cross-skill temporal excitation, binary, validated by
  fit, no certification, no ordinal.
- Option Tracing (Ghosh, Raspat, Lan, AIED 2021) and DP-MTL (AAAI 2022):
  ordinal/option-level, no transfer term.
- PSI-KT (ICLR 2024): the readout ground and transition operator class; ELBO
  generative, binary (readout is Bern(sigmoid) of the single tagged concept's
  state), claims a recovered concept graph WITHOUT certification, which is
  the paper's explicit novelty edge. AGPL-3.0 verified 2026-07-04: design
  reference only, never vendor code (CARRIED).
- GKT (Nakagawa et al. 2019) and GIKT (Yang et al. 2020): GKT holds
  per-concept states moved by graph-neighbor propagation (dynamic transfer,
  BCE-trained, exactly the class whose reported structure needs
  certification); GIKT has no per-concept state, its question-skill graph
  enriches embeddings statically. The old "middle ground" role is retired.
- Growth IRT lineage: Embretson MRMLC, Andersen longitudinal Rasch, Fischer
  LLRA, dynamic IRT (Wang, Berger, Burdick 2013), item-position learning
  (Debeer and Janssen 2013): occasion-level change, no per-concept transfer
  operator, no continuous tracker, no certification instrument. The paper
  OWNS this lineage (doctrine) and extends occasion to practice step; the
  license table from the referee review is banked for related work.
- M-ERS (Park et al. 2019): multidimensional Elo in practice environments;
  no measurement layer, no certification, binary.
- CAT-for-growth practice (NWEA MAP Growth; Qian 2018): within-session-static
  short tests measuring between-occasion growth; the honest foil framing
  (not "person static forever").
- TransKT (IJCAI 2025), skill-to-skill supervision (2023),
  cross-disciplinary transfer KT (2025): graph or supervised transfer,
  binary, no certification.
Confirmed white space: ordinal partial-credit x signed practice-driven
transfer x forecast-ablation certification (+ CAT-anchored validation). The
certification battery is the product; the model is deliberately minimal.
(Haiku recon citations unverified; P0 re-verifies every entry against the
actual paper. boost_refs.bib conventions. Add Adams-Wilson-Wang 1997 and the
FIPC line, Kim 2006, to the verification list.)

## Experiment program (phases and gates)

P0 SETUP AND CONTRACTS (Sonnet + Haiku, half a day).
   Reference stock: the 15 recovered _qmirt_*.py scripts (restored 2026-07-04
   from session transcripts) are REFERENCE ONLY, dead piles to mine, not a
   foundation; they were removed for a reason. The _qm2_ build is a fresh,
   smaller rewrite taking only what the review certified (state family,
   forecast-harness design, battery protocols). Codex boundary unchanged:
   never touch core/, run_*.py, datagen.py, engines.py.
   Vocabulary contract (referee naming rulings in); venue fact-check;
   citation verification; PSI-KT license note. On approval: flip memory to
   ACTIVE, add HANDOFF pointer (coordinate with the live paper-1 session).
   GATE G0: contracts frozen; compensatory sanity bridge runs green (rebuilt
   generators reproduce venue-1's active gap within seed noise, proving the
   rebuild is faithful BEFORE the readout flips).
P1 MODEL AND MEASUREMENT CERTIFICATION (Opus think-to-code; Sonnet harness).
   Simple-structure dynamic GPCM per D1-D4, FB-OFF default; generator family
   including the MISMATCH twins (non-monotone, heterogeneous rates,
   forgetting, informative schedules) and decoupled-practice schedules;
   anchor-first and dynamic stage-1 calibration paths; the C1 battery
   (mismatch recovery, LI audit, re-derived refit-discrepancy). The
   identification note is a PRE-FREEZE deliverable of this phase (Opus +
   psychometric adversary): rotation, gauges, G sign/existence under
   positivity, the FB confound, calibration-under-motion conditions.
   GATE G1 = C1. No downstream phase starts on an uncertified bank.
P1b SPINE BRIDGE (the review's ONE THING; Sonnet runs, lead rules).
   Re-derive under FB-OFF simple structure at C=3: the forecast active gap,
   the sparse A->B case specifically (sign via the forecast metric, R9's
   named confound test), observed-vs-predicted growth scores, null twins.
   RE-REGISTER every C2/C3 kill threshold from these runs (K-normalized
   units). GATE G1b: spine numbers exist and the kills are frozen. If the
   sparse A->B forecast gap is not clean here, STOP and redesign before any
   scaling.
P2 GROWTH AND RENDERING (Sonnet sweeps; lead rules the figures).
   C2 battery under the frozen kills; E1/E2 prototypes with the deceleration
   decomposition and threshold screens; user visual sign-off. GATE G2 = C2.
P3 TRANSFER CERTIFICATION (Opus think-to-code on the harness, Sonnet grid).
   Full battery under the new readout; sign and dose-response; mechanism
   robustness (linear, mastery-ceiling, rate-plus-forgetting; rate generator
   fixed to carry real rate variance). SUB-GATE: the D>3 masked-forecast
   harness runs green at C=5 (silent-runner protocol: foreground, JSON to
   disk, summaries under 600 words) BEFORE the C=5,8 sweep; declared
   fallback = D=3-only with scope stated. GATE G3 = C3.
P4 VALIDATION COMPARATOR (appendix-level; Opus think-to-code on the design).
   The moving-truth CAT simulation built and validated as NEW machinery;
   disjoint-pool protocol; independent-bank arm; E4; optional E6 sized
   honestly. GATE G4 = C4. Venue decision ratified here at the latest.
P5 REAL DATA (lead present; judgment-heavy).
   KDD Cup 2010: positivity audit FIRST (reportable either way); ordinal
   coding declared; certified bank (dual calibration on real responses,
   first-attempt scoring per R10); forecast gaps on passing pairs;
   differential-invariance keyed to re-exposure. GATE G5 = C5 verdict in the
   three-way space, recorded either way.
P6 WRITING AND ADVERSARIAL REVIEW (skeleton after G2; Haiku logs, Sonnet
   drafts under contracts, lead rewrites, Opus editor rules).
   Style contract rebuilt for the chosen venue from 3 published exemplars;
   editor passes at every gate plus full-paper GROUNDED/PLAUSIBLE/FANTASY;
   the psychometric opponent re-runs at G1 (identification note), G1b, and
   the full draft; cold-read guardrail before submission; figures through
   _paperfig_style; personal visual sign-off on every figure.

Throughout: negative-results ledger from day 0; every dead end named,
verdict recorded, never re-litigated. Raw numbers land in
docs/qmirt_experiment_results.md (new file; never write into paper 1's
records).

## Venue (decide by G4; verify facts in P0)

Lean: JEDM primary (no page cap, methods depth welcome, free OA, journal
track presents at EDM). EDM 2027 full paper as the conference variant. CAEAI
sibling only if paper 1 lands there and the editor wants the pair. A serious
measurement venue (JEM, APM) becomes thinkable ONLY if C1's misspecification
story turns out strong enough to lead with; not the default.

## Cast and constitution (for the ultracode session)

- Lead (Fable, this seat): design freezes, gate rulings, claims adjudication,
  figure sign-off, final prose control, this document's upkeep.
- Editor (Opus, separate context, adversarial): GROUNDED/PLAUSIBLE/FANTASY at
  every gate; cold read at the end; no drafting.
- Psychometric opponent (Opus, separate context): re-attacks at G1, G1b, and
  the full draft; owns the license table and naming compliance.
- Think-to-code (Opus): model core, identification note, moving-truth CAT
  design, anything where the code is the argument.
- Scaffold (Sonnet): generators, sweep harnesses, adapters, figures, venue
  fact-checks, doc upkeep from specs.
- Dirty work (Haiku): searches, citation verification, logging, records,
  vocabulary sweeps, first-pass summaries.
- Domain agents on call: psychometric-researcher, ml-math-researcher.
Rules: model economy as above; no detached runs; harness-tracked background
only; explicit staged paths, never git add -A; no Co-Authored-By; PSI-KT
AGPL design-reference only; C/KC-dim vs K categories; seed-clustered stats
everywhere; retraction ledger public inside the results doc.

## Risks and open decisions (defaults chosen, ratify or override)

R1 Simple-structure flip loses within-item compensation; real items are
   multi-KC. Default: single-KC steps for the real leg (report coverage),
   composite-KC concepts where sensible, compensatory arm in the appendix.
R2 CAT comparator circularity. Resolved by demotion + disjoint pools +
   independent-bank arm (review fold-in); residual risk is that the
   independent-bank arm disagrees, which itself becomes the reported number.
R3 Real transfer may be unidentifiable in the wild (co-scheduled curricula).
   Default: the positivity audit IS a result; three-way verdict space.
   Highest-risk claim; C5 stays conditional and bounded.
R4 Real ordinal is coerced (thresholds disordered on 45-73% of items in
   paper 1). Default: inherit the demotion; synthetic + design carry the
   paper; per-item screens on every real exhibit.
R5 Scale of the build: fresh _qm2_ rewrite with recovered scripts as
   reference. Two to three GPU-days synthetic, one real. P1b adds a day.
R6 Two-paper dependency. Paper 2 cites paper 1 as companion under review,
   defines everything inline, re-verifies every imported instrument on its
   own beds (delta re-validation is C1(c), not an import).
R7 Working title (slate, NOT frozen; workshop at P6): 1. "Learning in the
   Test: Certified Cross-Concept Transfer in Ordinal Knowledge Tracing";
   2. "Fixed Measurement, Moving State: Certifying Growth and Transfer in
   Prediction-Trained Knowledge Tracing"; 3. "Does Practice on One Skill
   Move Another? Forecast-Ablation Evidence in Partial-Credit Knowledge
   Tracing".
R8 Calibration under a moving person (the review's convergent core).
   Resolved from escalation to DEFAULT design: anchor-first where possible,
   dual-calibration sensitivity always, generator-mismatch certification as
   the C1 gate, single-exposure scoring. The identification note treats it
   pre-freeze.
R9 The FB/sparse-G sign confound (named by both reviews). RULE: sign is
   never read from an FB-ON fit; identification runs are FB-OFF; the sparse
   A->B forecast-metric case is a P1b hard gate. If FB-OFF cannot meet C2's
   within-learner recovery, the sanctioned richer own-gain (M2 mastery
   ceiling) is tried before FB is reconsidered, and any FB reintroduction
   requires a new identification argument.
R10 Item re-exposure / retest effects (referee F4). Within-person practice
   makes item-side change MORE plausible than the equating baseline the
   invariance assumption borrows. Default: single-exposure items in
   synthetic reference sets; first-attempt-only scoring on real reference
   items; differential-drift companion keyed to re-exposure count; the
   strengthened assumption stated in limitations.

## Approval checklist (what the user ratifies to unfreeze execution)

1. D1 readout flip (simple structure primary, compensatory demoted), now
   with the P1b spine bridge as its safety net.
2. The measurement doctrine rewrite: item-side invariance + occasion-specific
   ability, "explainability layer" retired from paper vocabulary (it was
   your coinage; the referee's case for retiring it is folded in above).
3. FB-OFF as the identification default (R9), FB-ON demoted to a prediction
   variant.
4. C4/CAT demoted to appendix validation with the redesigned protocol; E6
   optional and honestly sized.
5. C5 three-way verdict space (positive / informative-null / untestable).
6. Venue lean (JEDM primary).
7. The rendering contract (E1-E6 as revised).
8. Cast and model assignments, including the psychometric opponent as a
   standing reviewer at G1, G1b, and the draft.
