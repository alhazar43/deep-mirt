# Q-MIRT archaeology for kt-mirt

Internal excavation of the retired/parked "Q-MIRT" transfer-paper thread (deep-mirt repo,
docs/qmirt_blueprint.md, docs/qmirt_paper_plan.md, docs/qmirt_experiment_results.md,
docs/qmirt_plain_state.md, docs/overnight_transfer_active_campaign.md,
docs/overnight_findings.md). This thread is PARKED, superseded by the current
measurement-audit paper (see MEMORY.md: "Q-MIRT transfer paper (PARKED)"). Nothing here is
re-verified; every claim below is internal record, not gospel, and is marked
`load_bearing=false` in the structured output per program instructions — kt-mirt should treat
these as hypotheses to re-test on its own harness, not results to inherit.

No web sources used (internal archaeology only). All citations below are to the repo's own
docs, which is where the "source" field points throughout.

## 1. Model classes tried for cross-concept transfer

All variants share one skeleton: per-concept state `z_{t,c}`, a frozen/calibrated IRT-style
item decoder, and a signed cross-concept route restricted to a fitted zero-diagonal matrix
`G` (source concept `A` moves target concept `B` only through `G[B,A]`, never through shared
hidden state). The variants differ in the transition mechanism and in how person parameters
are estimated.

- **ExplicitStateModel (base, "M1" linear own-gain).** `z_{t+1,c} = decay_c * z_{t,c} +
  own_gain_c * Q[item_t,c] + (prac_t @ G.T)[c]`. Structural isolation verified to ~1e-8 (a
  `G=0` control run reduces non-practiced concepts to pure decay). Individual-learner
  tracking was weak without response feedback (within-learner correlation ~0.10-0.42
  (docs/qmirt_experiment_results.md, "venue 1", docs/overnight_findings.md Part A)).

- **FB variant (response feedback, borrowed from PSI-KT).** Adds a Q-gated own-concept
  innovation term driven by the actual response. Rescued within-learner tracking (0.10-0.42
  -> 0.26-0.72, and up to 0.80 in later runs) but was found to sign-reverse the fitted
  cross-concept coefficient for the sparse-edge case (`G[B,A]` systematically negative for a
  single true A->B edge) because the response-feedback projection absorbs the transfer signal
  and the optimizer compensates with a negative `G`. RULING (R9): identification/certification
  runs are FB-OFF by default; FB-ON is a prediction-quality variant only, never used to read a
  sign (docs/qmirt_paper_plan.md "KEEP WITH A DEMOTION"; docs/overnight_transfer_active_campaign.md
  venue 3 "R3").

- **OU mean-reverting transition (borrowed from PSI-KT).** Adds mean-reversion toward a
  target `mu_c` to handle non-monotone/forgetting trajectories. With a FREELY FITTED `mu`,
  this became an unidentified always-on growth channel that explained away real transfer
  (Lemma 1, below) — the fix was to fix `mu=0` as a hard buffer and gate all growth through
  practice indicators. With `mu` fixed, OU cleared the non-monotone fabrication (null gap
  ~-0.0001) while linear-decay models fabricated (null gap = 96% of the active gap) on the
  same non-monotone bed (docs/overnight_transfer_active_campaign.md venue 3 "R1").

- **Mastery-ceiling gain (M2, diminishing returns) and rate+forgetting (M3, learnable decay
  x active forgetting).** Alternative own-gain functional forms tested to see whether the
  active-change / transfer result depended on the linear-gain functional form. All three
  mechanisms (M1/M2/M3) preserved isolation exactly and produced matched-size active gaps
  (+0.355, +0.364, +0.357 respectively, 6/6, 5/6, 6/6 seeds positive)
  (docs/overnight_transfer_active_campaign.md venue 4). M3's own per-concept rate did not
  visibly recover, but only because the generator gave near-identical true rates across
  concepts — inconclusive, not a negative (see open items below).

- **qm3 "blueprint" model (redesign, later thread): compensatory multidimensional GPCM
  emission + explicit transition + encoder for person parameters.** Per-item discrimination
  is now a VECTOR `a_j` with support restricted to the item's Q-row (cross-loading items have
  several nonzero entries; pure items have one); thresholds stay scalar. This nests the
  frozen Chapter-0 (ma-irt) unidimensional GPCM exactly at C=1
  (docs/qmirt_blueprint.md "The emission"). Rotation is killed by construction: each
  concept's scale is pinned by its own frozen pure-anchor item set (never estimated via a
  free rotation matrix) — this is the concrete mechanism by which the qmirt thread avoided
  the MIRT rotation-freedom problem the kt-mirt program context explicitly rules out.
  Person parameters (`z0`, per-learner learning-rate multiplier `lambda`, per-learner
  transfer multiplier `gamma`) are handled either as free per-learner parameters or via an
  LSTM recognition network (amortized posterior), gated behind two new certification gates
  (Gate B "encoder honesty", Gate C "bank gate") described in section 2.

## 2. Identification lemmas and gate results (design constraints)

These were derived empirically over one overnight campaign plus a follow-on session, each
diagnosed to a specific mechanism before being patched (source:
docs/qmirt_experiment_results.md "Pilot findings" and later gate sections).

**Lemma 1 — free asymptote is an always-on growth channel.** If the mean-reversion target
`mu` in an OU-style transition is fitted rather than fixed, stage-2 training explains a
target concept's entire conditioning-window rise as mean reversion toward a high fitted
`mu`; `G` becomes unidentified (near zero or negative), the no-G forecast arm still rises,
and the null twin (true `G=0`) fabricates a nonzero `G`. FIX: `mu` is a fixed buffer at 0;
every growth route must be gated by practice indicators, never allowed to run "for free."

**Lemma 2 — free persistence is a decay compensator.** If the transition's persistence
(`rho < 1`, i.e. decay-with-memory) is fitted freely on MONOTONE data, decay-down-per-step
and gain-up-per-step become non-identifiable and cancel on the conditioning window: the null
twin fabricates `G` (+0.03) as a decay compensator, and the no-G arm spuriously decays in the
wrong direction. FIX: `rho` is frozen at 1 (no decay) on monotone beds; a decay/OU variant is
only fit on beds with genuine non-monotone (dip-containing) identification content.

**Lemma 3 — gain-form misfit launders into the cross-concept term.** If own-gain is modeled
as constant-per-practice against a TRUE generator with decelerating (ceiling-approaching)
gains, stage-2 training sculpts the residual deceleration into a fabricated, non-monotone
`G` pattern (a "zigzag": `G[B,A]=+0.041` alongside `G[B,U]=-0.102` for a concept `U` that
should carry zero signal). The schedule's positivity variation (needed for identification)
also supplies the sculpting basis when the gain family can't express deceleration. FIX:
own-gain must be gap-scaled / mastery-ceiling shaped (`(ceiling - z)+`-type saturation); the
same ceiling-gated form is used for both own-gain and cross-concept transfer, and the
fabricated residual dropped to near-null (+0.02, no predictive content) after the fix
(docs/qmirt_experiment_results.md "LEMMA 3").

**Bounded interference (engineering patch, not an independent finding, per the editor
audit).** Symmetric ceiling-gating saturated the interference (negative-transfer) twin at
the response floor. Fix: gate positive transfer by `(ceiling - theta)+` and negative
transfer by `(theta - floor)+` — "you can only lose what you have built" — in both generator
and model.

**Metric ruling — certification is read on the score scale, as a matched-null paired
contrast.** Fitted `G` carries a per-seed additive offset and cannot be compared to zero
directly; the load-bearing quantity is `(no-transfer forecast error) - (with-transfer
forecast error)` on the target concept, minus the SAME quantity computed on a same-seed
twin with `G_true=0` (the "matched-null paired contrast"). Primary metric = forecast
expected-score error gap in proportion-of-max units (bounded, sharpness-insensitive); NLL is
reported secondary and, once a sign-inversion bug was fixed (see below), agrees in direction
with the score metric (docs/qmirt_experiment_results.md "METRIC RULING", "G0.5").

**Phantom transfer from free/under-informed per-learner traits (Gate B / B2, the sharpest
result for kt-mirt's constraint (a)).** A dedicated fabrication test compared several
postures for per-learner parameters against a null twin (`G_true=0`), reading BOTH the
population-mean transfer effect AND the 95th-percentile PER-LEARNER transfer effect (the
population mean alone was found to hide fabrication — this is a load-bearing methodological
point, not just a result). Findings, in order of increasing certification:
  - Free per-learner transfer-ability trait `gamma` (analogous to PSI-KT's per-learner
    transfer multiplier): FAILS at every estimation posture tested — free parameter (p95
    per-learner effect 0.0306, ceiling 0.01), early-window-only amortized encoder (0.0234),
    and full-conditioning-window amortized encoder (0.0167-0.0232). The recognition network
    was NOT, in the end, uniquely blamed (an earlier attribution to "the encoder fabricates"
    was revised): the fabrication source is the per-learner TRAIT MULTIPLIER itself, under
    any estimation posture, not the encoder mechanism per se.
  - Per-learner ability level `z0` and learning-rate multiplier `lambda`: PASS, but ONLY
    under full-conditioning-window amortized (encoder) inference (p95 = 0.0068, clean); the
    SAME quantities as free parameters or under early-window (information-starved) encoding
    FAIL (p95 0.0306 for free; encoder+gamma-cut+early-window 0.0234).
  - Net gate ruling: canonical certified config = per-learner `z0` and `lambda` via
    full-window amortized encoder, `gamma` PINNED at 1 (population-level transfer only, no
    per-learner transfer multiplier). This annotates PSI-KT's own per-learner transfer trait
    as "exactly the object that fails certification under every posture tested"
    (docs/qmirt_experiment_results.md "GATE B ATTRIBUTION FLIP", "GATE B2 COMPLETE").

**Practice-gating / response-exclusion from the cross-concept route.** In every surviving
model class, the cross-concept transition is driven ONLY by practice indicators (which
concept was practiced, i.e. the Q-row), never by the observed response; responses may only
feed an OWN-concept innovation term (the FB pathway), and even that is barred from
identification-grade runs (R9). This is stated as a structural design rule across the whole
record, not a single lemma.

**Positivity condition (schedule identifiability boundary).** Cross-concept transfer is only
identified where the practice schedule "decouples" the source and target concepts (episodes
where the source is practiced without the target). Quantified as the fraction of source-only
practice slots: `>=0.75` clean, `0.25-0.50` weak/suggestive (~1 sigma), `0` completely
unidentifiable (co-scheduled practice makes own-gain and cross-concept gain collinear,
G collapses to exactly 0). This reappears throughout as "the positivity condition," treated
as a stated design/reporting requirement, not a bug (docs/overnight_transfer_active_campaign.md
venue 2/3 "R2"; docs/qmirt_paper_plan.md).

**Bank / cross-loading recoverability (Gate C, one-to-many mapping).** For a compensatory
multidimensional GPCM with cross-loading items (an item can load on 2-3 concepts), frozen
discrimination VECTORS are recoverable under marginal ML (Gauss-Hermite quadrature over the
per-concept ability prior) if and only if there are `>=3` PURE anchor items per concept; one
pure anchor per concept is insufficient at every tested between-concept correlation (0, 0.3,
0.6). Moderate correlation (r=0.3) is fine; high correlation (r=0.6) degrades cross-loading
RATIO recovery (the quantity every one-to-many attribution claim actually rides on) to 0.833.
The gate is CONDITIONAL on knowing the ability correlation `R` at first, but a follow-up
found that an independence-prior calibration + one-iteration EAP estimate of `R` +
recalibration recovers nearly as well as knowing `R` exactly, dissolving that caveat
(docs/qmirt_experiment_results.md "GATE C RESULT", "Gate C misspecification arms").

**Discrimination recovery under JOINT dynamic calibration (the paper-1 "stable-and-wrong"
disease, reproduced without a shared head).** Item discrimination (alpha) systematically
fails to survive joint calibration with free per-learner ability, even on a matched-generator
twin (alpha rank ~0.05 vs oracle 0.8-0.9), while item location recovers fine. This was
INITIALLY attributed to the classical incidental-parameters problem (Neyman-Scott), but a
controlled race (equalized optimization budget: marginal ML vs converged joint ML vs oracle)
resolved the mechanism as THREE STACKED CAUSES, in order of size: (1) optimization budget
dominated the original collapse (0.05 -> 0.47 from budget alone), (2) marginalization over
person parameters genuinely helps beyond budget (rank +0.10-0.12 over converged joint ML,
slope de-attenuation, seed-variance reduction — a real but SECONDARY incidental-parameters
effect), (3) calibration-cohort ability SPREAD sets the ceiling (narrow-spread cohorts stay
range-restricted; wide-spread cohorts reach oracle parity, 0.775 vs 0.802). The demonstrated
recipe: calibrate on a measurement-regime cohort with adequate ability spread, using marginal
ML, then freeze (docs/qmirt_experiment_results.md "C1 bank certification", "G1 MECHANISM
RESOLVED"). This is close in spirit to kt-mirt's constraint (c) ("anchoring / separated
item-parameter paths stabilize" stable-and-wrong readouts) and is the qmirt thread's own
evidentiary basis for a similar constraint.

## 3. Synthetic generator designs and null twins that worked

- **Matched-null paired contrast as the primary read.** Every headline number is (metric on
  the transfer-bearing generator) minus (SAME metric, same seed, on a `G_true=0` twin), never
  a bare value or a comparison against zero. Necessary because fitted `G` and forecast-error
  gaps carry per-seed additive offsets.
- **Confound battery (4-part), reported as "5/5"/"battery" passing:** (i) correlated-no-transfer
  (concepts share a prerequisite/correlated ability, `G_true=0`) returns near-zero active gap;
  (ii) co-scheduling (A and B always practiced together, `G_true=0`) returns near-zero in
  aggregate but shows an identifiability boundary rather than fabrication at zero decoupling;
  (iii) shuffle-order control (destroy the causal lag by shuffling practice order) collapses
  the gap and fitted `G` to exactly zero on every seed — transfer needs the causal lag, not
  mere temporal co-occurrence; (iv) reverse-direction probe (fit `G[A,B]` when only the true
  edge is `G[B,A]`) returns zero on every seed — direction is identified, not just magnitude
  (docs/overnight_transfer_active_campaign.md "venue 2").
- **State-inert reference/measurement items.** Items used purely for measurement (not
  practice) must not move the state at all (`ref_inert=True`); otherwise "the target concept's
  only route to change is transfer" is false as stated, and a fitted ceiling can silently act
  as a measurement-cadence asymptote. After the fix, the null-twin target concept is flat to
  the generator's exact zero (docs/qmirt_experiment_results.md "G0.5" finding 2).
- **Permuted-credit / permuted-init twin (mapping-boundary / multimodality diagnostic).**
  Refit the routing matrix from an adversarial (wrong-route) initialization; the size of the
  resulting fit-gap (does the adversarial init walk back to the true routing, or find an
  equally-good wrong solution?) is used as a truth-free diagnostic for whether the
  attribution problem has a flat, ambiguous likelihood surface. In the one completed arm this
  behaved well (adversarial init walked back to the true routing at indistinguishable
  conditioning loss) — no multimodality trap found, but the check itself is the reusable
  design (docs/qmirt_blueprint.md "V1 MAPPING BOUNDARY"; docs/qmirt_experiment_results.md "V1
  partial reading").
- **Per-learner TAIL statistic on the null twin, not just the population mean.** The single
  most consequential methodological choice in the whole record: reading the 95th-percentile
  per-learner effect (not just the population-mean effect) on a null twin is what caught the
  per-learner-trait fabrication that population-level checks completely missed (Gate B).
- **Isolation check as a positive-control leak test, not just a decay check.** An early
  isolation check that only verified "off-target concept decays to near-zero under `G=0`"
  could never actually detect a leak (it was checking a degenerate case). The corrected
  version requires `G` to be genuinely nonzero and then asserts the zero-ROW (untouched)
  concept stays flat — a real positive control that the cross-concept route can move its
  target and does NOT move others (docs/qmirt_experiment_results.md "G0.5" finding 6).
- **Growth-score reliability via classical Spearman-Brown decomposition, no model fitting
  needed.** When observed-growth reliability came in below a 0.80 bar, a no-fitting classical
  decomposition (predicted reliability from response-sampling noise via Spearman-Brown
  arithmetic) matched the observed reliability almost exactly (0.52/0.68 predicted vs
  0.53/0.66-0.69 observed unpooled/pooled) and explained the shortfall as measurement density,
  not a model deficiency — yielding a quantified deployment rule ("~4x the reference-item
  density needed to reach 0.80"). Reusable as a design-time sanity check for any growth-score
  claim (docs/qmirt_experiment_results.md "GROWTH-SCORE RELIABILITY", "CLOSURE").

## 4. What was certified, what was killed, what was left open

**CERTIFIED (survived the record's own gates, on synthetic beds under matched generator
form, C=3, seed counts 3-9 per cell):**
- Signed cross-concept transfer direction+existence, under FB-OFF simple-structure/GPCM
  readout, matched-null paired contrast, positivity-satisfying schedules: paired contrast
  positive 9/9 seeds, sign correct 9/9 at dose `|g|=0.025` (P1b spine bridge, converged
  budget).
- Observed-vs-predicted growth-score agreement within ~0.05 proportion-of-max units at
  converged optimization budget (this healed almost entirely once optimization budget was
  fixed — the earlier "33% recovered" number was a budget artifact, not a modeling limit).
- Frozen cross-loading discrimination VECTOR recovery for a one-to-many (multi-KC item)
  mapping, via marginal ML with `>=3` pure anchors per concept and moderate concept
  correlation (Gate C).
- Encoder-amortized per-learner ability level (`z0`) and learning-rate multiplier (`lambda`),
  under full-conditioning-window inference, with the per-learner transfer trait `gamma`
  pinned at 1 (Gate B2).
- Mechanism-robustness of the active-transfer result across three own-gain functional forms
  (linear, mastery-ceiling, rate+forgetting) — not an artifact of one gain family.
- Non-monotone/noisy trajectories: OU transition with FIXED (not fitted) mean-reversion
  target, weak L1, clears fabrication and recovers signal at power.

**KILLED (named failure modes, with a stated mechanism, not just "it didn't work"):**
- Per-learner transfer-ability trait (`gamma`, PSI-KT-style) as an ESTIMATED quantity, under
  every posture tested (free parameter, amortized encoder, windowed encoder) — this is the
  qmirt thread's strongest, most directly reusable negative result for kt-mirt.
- Response-feedback into the transition, for identification/sign-reading purposes (sign
  reversal on sparse edges) — kept only as a prediction-quality variant, never for reading
  transfer sign (R9).
- Direct comparison of fitted `G` against zero (additive per-seed offset makes it
  uninterpretable) — replaced by the matched-null paired contrast.
- Direct-`G` recovery as a metric under the response-feedback model specifically for the
  sparse single-edge A->B case (resp_proj confound reverses sign; B->A and dense-graph cases
  recover fine) — the masked-forecast active gap was adopted as the metric that is NOT
  confounded this way.
- Frozen passive-LSTM comparator as evidence for "active, not passive, change" — a
  free-decoder-vs-frozen-decoder confound (the LSTM wins on absolute forecast NLL regardless
  of tracking quality) — replaced by a within-condition `G`-zeroed control using the SAME
  frozen decoder.
- Joint dynamic item-parameter calibration with free per-learner ability, at low optimization
  budget — badly biased discrimination recovery, root-caused to budget + marginalization +
  cohort spread, not motion contamination per se.
- Anchor-first / static-early-window calibration as a naive fix for the above — also failed
  (insufficient spread/motion in the early window).
- "LLTM discipline" as the warrant for modeled gain (item-difficulty explanatory-model
  framing) — renamed "recovery-certified stated gain" because the warrant is recovery
  certification plus held-out event-level forecast, not a likelihood-ratio argument.
- NRM (nominal response model) `a_k` vs `c_k` leverage split, hypothesized as analogous to
  GPCM's alpha-vs-beta low-discrimination-leverage story — does NOT replicate; Fisher
  information is near-symmetric between `a_k` and `c_k` in NRM (ratio ~0.90 vs 5-10x for
  GPCM alpha/beta). A subsequent "allocation inversion" claim was itself retracted as a
  theta-scale gauge artifact after a gauge audit.

**LEFT OPEN (queued repeatedly, never closed in the provided record):**
- D-scaling of the transfer-certification harness beyond D=3 concepts to D=5, D=8 (built,
  crashed twice on execution/tooling issues — an output-token limit killed background
  agents — not a scientific failure, an infra gap).
- KDD Cup 2010 (and EdNet) real-data leg for the Q-MIRT transfer claim specifically — never
  completed; would need the D-scaled harness plus a KC-to-concept mapping and was explicitly
  deferred as "judgment-heavy, better done with the user awake."
- Mismatched-gain-form / mismatched-transfer-form robustness arm — repeatedly flagged as
  "the live C3 threat" (the matched-bed results are best-case identifiability, not a
  robustness demonstration) and never run to completion in this record.
- The V2 "modeled gain" program (gain as a function of event content: outcome magnitude,
  spacing, item features, graph position) — fully pre-registered (functional form, recovery
  rule, per-component fabrication rule, held-out event-level forecast warrant) but not
  reported as executed/closed in the material read.
- Per-concept learning-RATE recovery (M3) — inconclusive (not negative): the synthetic
  generator used near-identical true rates across concepts, so there was no rank signal to
  recover; a generator with genuinely distinct per-concept rates was named as the needed fix
  and left to a separately-parked "rate program" (consistent with kt-mirt program context
  (b): the parked rate program elsewhere found rate unrecoverable on real EdNet/ASSISTments/
  KDD near response-saturation ceilings).
- Item-drift / re-exposure detector refinement for real data (circularity control passed,
  but statistical power against ORACLE states was judged poor; an anchor-linked window-refit
  refinement was queued before real-data use).
- Projected-mastery early-stopping study — scoped only as a controlled appendix
  proof-of-concept with a harm-twin control, not executed in the material read.

## 5. Relation to kt-mirt program context (no direct contradictions found)

This archaeology reads as largely CONSISTENT with, and in places the direct evidentiary
source for, the constraints stated in the kt-mirt program context:

- Constraint (a) ("free per-learner trait multipliers fabricate phantom transfer, and shared
  encoders mimic transfer passively") is well supported by Gate B/B2, with one nuance worth
  carrying forward rather than a contradiction: the qmirt record's own final attribution is
  narrower than "shared encoders mimic transfer" — it pins the failure specifically on the
  per-learner TRANSFER-ability trait (`gamma`), not on amortized/shared encoding per se. In
  the same experiment, encoder-amortized `z0` and `lambda` (also per-learner, also passing
  through the same shared LSTM) were certified as honest under full-window inference. If
  kt-mirt's design space includes an encoder producing several per-learner quantities, this
  suggests the risk may be concentrated in whichever quantity plays the role of a
  cross-concept transfer multiplier, rather than being uniform across everything the encoder
  outputs — worth treating as a hypothesis to re-test, not an established general law.
- Constraint (b) (aggregate learning-rate unrecoverable near response saturation on real
  datasets) is consistent with, though not directly tested by, this thread; the qmirt M3
  rate-recovery inconclusion has a different proximate cause (generator symmetry) and points
  to the same "need real rate variance" gap that the separately-parked rate program
  addresses.
- Constraint (c) (stable-and-wrong IRT readouts under prediction training; anchoring/
  separated-item-parameter paths stabilize them) is closely mirrored by the Gate C /
  discrimination-under-joint-calibration findings: item discrimination collapses under joint
  dynamic calibration with free person parameters, and the fix (marginal ML on a
  measurement-regime cohort, then freeze) is exactly an anchoring/separated-parameter-path
  recipe.
- The kt-mirt program context rules multidimensional-theta rotation freedom out of scope;
  the qmirt thread's own multidimensional emission (vector discriminations, Q-row support,
  per-concept pure-anchor pinning) is one concrete way rotation was avoided rather than
  solved generally — relevant if kt-mirt ever needs a multi-concept IRT-style readout without
  reopening the rotation question.

No claim in the archived material directly contradicts the program context; the strongest
caveat is the nuance on constraint (a) above.
