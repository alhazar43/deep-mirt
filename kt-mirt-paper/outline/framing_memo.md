# Framing memo (consolidated judgment, 2026-08-10)

Consolidating judge's output over five persona attacks (AUC-culture
reviewer, measurement theorist, causal-inference reviewer, novelty
editor, split editor; all verdicts "framing-survives-with-fixes") and
two full-text fact-checks (HawkesKT, LTKT; both verdicts SIGNED).

Status. This memo tightens `outline/claim_language_constraints.md`
(tightening is permitted by that file's preamble; no rule is relaxed).
Where this memo is stricter, this memo governs. Claim strength still
comes only from `evidence/claim_evidence_map.md`; no tag is upgraded
here, and the conservative reading of the S11/S12/S13 tag conflicts is
adopted for all prose. F1's substance stands (certify-then-claim, map
as centerpiece, refusals first-class); its wording is revised as listed
in section 6.

---

## 1. The frozen framing paragraph

Deep knowledge-tracing models are routinely read as measurement
instruments. Their latent states are published as growth curves,
per-skill mastery, and cross-skill influence graphs, and the nearest
published neighbors (Deep-IRT, PSI-KT, HawkesKT, LTKT) validate those
readouts by prediction accuracy alone. Prediction cannot certify a
readout, and neither can raw learning curves, which are attrition-biased
and blind to the false-fire mechanisms our harness isolates. This paper
builds a certification protocol that asks, before any readout is read,
whether the estimator recovers known truth. Its instruments are matched
synthetic twins calibrated to named real datasets, pre-registered bars
with frozen kill conditions, permutation nulls, designed confound arms,
and seed discipline, with refusals published as first-class
deliverables. The protocol runs end to end once and transfers once. The
completed cycle is growth existence. A passive detector is certified at
cohort grain on twin profiles at both density extremes, silent on every
null twin and at the permutation floor on every growth twin, surviving a
full density inversion. Its one diagnosed failure, a false fire on the
saturated twin, is traced to reference-model misspecification and
repaired with a saturation-aware null. The repaired detector fires on
real KDD Algebra (one seed, one cohort split, on the unsaturated 257 of
515 skills) and stays correctly silent on thin-practice Junyi, mapping a
data-density boundary of the method rather than an absence of learning.
The second instantiation, signed cross-skill influence, is caught
mid-cycle and reported that way. On the synthetic harness at a named
operating point (D=3, N=500, single-tag density, calibrated bank) the
readout earns exactly one licensed sentence, per-edge signed
dose-association with a measured detectable-dose floor of 0.04, twice
the field-anchored reference dose, and the paper itself shows that floor
move under density and bank error. An order-shuffle negative control
refutes the temporal reading. The causal reading is left unearned
pending a pre-registered endogenous-scheduling control, and the
remaining confound battery is pre-registered and unexecuted. Around
these two arms the protocol mostly refuses, with the mechanism attached
to every refusal. Per-skill growth resolution fails at both density
extremes with the same signature. Growth magnitude and per-skill ranking
fail their corridors. The field-representative shared-state neural
tracker fails the entire faithfulness battery at production scale. The
association sign collapses outright at multi-tag density. The
deliverable is the resulting licensing map, every cell labeled certified
on the synthetic harness, confirmed on real data, refused with
mechanism, or pending pre-registered, plus the harness to re-check it.
The paper moves no AUC and never claims to. It states, with ground truth
rather than opinion, which interpretive sentences these readouts license
and exactly where they stop.

---

## 2. Contribution list

One contribution sentence in the introduction, four numbered clauses,
all conjunction-scoped ("first certified", never "first").

1. **A certification protocol for the latent-state readouts of
   multi-skill KT models.** Matched synthetic twins calibrated to named
   real beds (the ownable device), pre-registered bars with frozen kill
   conditions, permutation nulls against fitted no-growth references,
   designed confound arms, end-to-end seed discipline, refusals as
   deliverables. Lineage owned preemptively (simulation-based
   calibration, randomization sanity checks, permutation nulls and
   negative controls, minimum-detectable-effect floors); the claimed
   novelty is the assembly for this object and its outputs, not the
   epistemic move. Backed by M1-M8, M12, M15, G22; lineage T7-T10.

2. **A certified growth-existence readout with one closed
   certify-then-confirm loop.** Certified at cohort grain on both
   density-extreme twin profiles (G1, G2), strength capped at the
   permutation floor (G4), the saturation false-fire diagnosed as
   reference-model misspecification and repaired (G5, G6, G7, G8), a
   real-data fire on KDD at exact stated breadth (R1) structurally
   paired with a correct real-data silence on Junyi that maps a density
   boundary (R2; deep-Junyi cell pending, P1), all framed as validation
   of the readout against the model-free baseline (R5), rolled up as
   partially certified (R3, G23).

3. **The first certified signed cross-skill readout, honestly demoted
   to dose-association at a named harness operating point.** Sign
   recovered from the fitted model near truth (S2), the claim grammar
   fixed by its own controls (S16), a harness-measured negative-edge
   detectable-dose floor of 0.04 at D=3, N=500, single-tag density,
   calibrated bank, untuned default trainer (S11, conservative tag),
   with measured floor-movers (S7 density collapse, S8 bank
   sensitivity), a false-edge background that mandates multiplicity
   control (S12), a passing phantom-transfer control (S13), the
   temporal reading refuted by the order-shuffle arm (S14, S15), the
   causal reading unearned pending scheduling controls (P5), and the
   full battery pre-registered and unexecuted (P3). Positioning per S21
   plus the HawkesKT and LTKT full-text verdicts (section 3).

4. **The licensing map, with refusals as first-class, mechanism-attached
   results.** Per-KC resolution refused at both density extremes (G9,
   G10, G11), the bank-fidelity floor named (G12), active-posture
   magnitude and rank refused (G13, G15), the misfit clause unusable
   (G16), the field-representative neural tracker refused on the
   faithfulness battery (G17), the L1 repair refuted (S9, S10), decline
   out of scope by construction (G21); four cell states with pending
   marked (P1, P2, P3, P4, P9); prescriptive licensing rules (G8/R1
   saturation-aware null before any growth claim, M9 full-pair-population
   check, S12/M10 multiplicity control, R4 real-bed licensing tiers).

---

## 3. Claim-language card

Updates to the binding constraints (all tightenings): rule 3 is
RESOLVED, rule 9 is SPLIT, rule 1 gains the model-recovered clause.
Everything else in `claim_language_constraints.md` stands unchanged.

### Fact-check verdicts (stated for the record)

**HawkesKT verdict: SIGNED.** Its cross-skill alphas are unconstrained
reals (raw embedding dot products, no clamp; only betas are clamped),
their signs are explicitly interpreted for prerequisite discovery
(Sec. 4.5, softmax-ratio mining), and validation is ranking agreement
with three expert annotators only (NDCG .8267). No ground-truth sign
recovery, no identifiability analysis, no per-entry certification; the
negative direction is response-conditioned (wrong answer on a
prerequisite depresses the target logit), not a skill-to-skill
interference estimate. Consequence: any signedness-priority wording is
dead; rule 1's conjunction carries the entire association claim;
HawkesKT is cited as the nearest model-learned signed cross-effect
precedent, "signed but uncertified and response-conditioned".

**LTKT verdict: SIGNED but STIPULATED.** Its signed transfer graph is a
statistical pre-computation, not a model readout: signs come from a
hand-designed heuristic on adjacent-event co-occurrence counts with
mean-value thresholds, frozen before training, and built on train,
validation, and test data jointly (test leakage the paper states).
Validation is predictive only (AUC/F1, ablations, one post-hoc narrated
heatmap); the authors themselves flag unreliability for sparse concepts.
Consequence: we may never claim prior KT lacks signed cross-concept
modeling (LTKT claims that priority); our claim scopes to CERTIFIED,
MODEL-RECOVERED signs. LTKT is credited for the positive-and-negative
framing and differentiated as stipulated-input, prediction-validated.

### Updated rules

- **Rule 1 (tightened).** The association novelty is the three-way
  conjunction: recovered from the fitted model (unlike LTKT's
  pre-computed inputs) AND certified against ground truth with confound
  arms (unlike HawkesKT's expert-ranking check) AND accompanied by a
  harness-measured detectable-dose floor. "First certified", never
  "first"; "signed" is a described property, never the novelty.
- **Rule 3 (resolved).** HawkesKT is signed. The conjunction framing
  carries the claim. Related work must say so explicitly; priority
  sentences that would fail if HawkesKT were signed are permanently
  banned.
- **Rule 9 (split into 9a/9b).**
  - 9a. The order-shuffle negative control refutes the TEMPORAL-transfer
    reading (both signed magnitudes retain ~3/4 against the 0.10
    collapse bar; S14, S15). Reported as a designed falsification that
    worked.
  - 9b. The CAUSAL cumulative-dose reading is UNEARNED, not refuted: it
    coincides with the causal effect only under exogenous scheduling,
    which holds in the harness by construction (M15) and is untested on
    real beds; the endogenous-scheduler control (CT3-v, K-T6) is
    pre-registered and unexecuted (P5). Never write that the shuffle
    killed the causal reading.

### Headline claim patterns

**Growth existence (R1/R2).**
- Allowed pattern: "The synthetic-certified cohort-grain existence gate
  transfers to real data: it fires on KDD Algebra under the
  saturation-aware null, on the unsaturated KC subset (257 of 515), at
  breadth one seed and one cohort split (B=99, p=0.01), and stays
  correctly silent on thin-practice Junyi, mapping a density boundary
  rather than an absence of learning." The exclusion count and breadth
  clause travel in the same sentence as any fire claim. Growth sentences
  validate the READOUT; the raw-curve baseline (R5) appears in the same
  section, with the statement that the fire confirms transfer of the
  certified instrument, not that growth was unknown.
- Forbidden: "first to detect learning/growth"; growth-as-discovery
  phrasing; plural "datasets confirm"; any per-KC growth claim on any
  bed; any magnitude claim; any fire sentence without the subset and
  breadth clauses; "detects growth" without the cohort-grain scope.
- Conditionality: the growth certificate is stated conditional on bank
  fidelity at the measured 0.70-0.80 level (G12), with the
  difficulty-opportunity ordering confound of a mastery-managed ITS
  named in the R1 scoping; bank-error propagation is measured for the
  association sign (S8), not for the growth gate, and the text says so.
- Estimand scope: certified object is existence of monotone ability
  gain at cohort grain; decline is out of scope by construction (G21),
  and the map carries that row.
- Pending: any density-boundary interval sentence carries the
  deep-Junyi cell (P1) with both outcomes drafted neutrally (rule 10).

**Signed association (S16 et al.).**
- Allowed pattern: "the first certified signed cross-skill readout,
  scoped to per-edge signed dose-association, certified on the
  synthetic harness at D=3, N=500, single-tag density, calibrated bank,
  untuned default trainer, with a harness-measured negative-edge
  detectable-dose floor of 0.04 (twice the field-anchored reference
  dose) at that operating point." The operating point is inline at
  every floor mention; floor-measurement is the certified capability,
  floor-transport is an explicitly refused claim (the floor moves: S7,
  S8).
- Forbidden: "first signed cross-skill influence/transfer/model" (dead
  against both LTKT and HawkesKT); "the detectable-dose floor of the
  readout" unqualified; "influence" without the dose-association
  demotion; ANY real-data association claim (real-bed names appear in
  association context only as triage or planned-work language); any
  curriculum or intervention implication; "certified" on S11/S12/S13
  material without "on the synthetic harness"; any abstract sentence
  implying a real-data influence readout exists.
- Scale arithmetic stated, not hidden: at K=515 the ordered-pair count
  is ~265,000, so a 5-15% dose-independent false-edge background implies
  tens of thousands of false signed edges before correction; per-edge
  existence at deployment K is refused pending CT8/CT9 plus
  matrix-level multiplicity certification; M10's BH certification was
  earned on per-KC growth statistics, never cited for the G matrix
  without saying so.
- Use argument (two legs only): the certified quantity is what the
  field's signed readouts estimate at best, so certifying it and
  refusing everything above it is an audit result about current
  interpretive practice; and it is a screening quantity whose upgrade
  path (scheduler control, external graph corroboration) is
  pre-registered with named licensing conditions. Kane's
  interpretation/use argument is the one-paragraph anchor.

**Temporal and causal grammar (rules 9a/9b).**
- Allowed: the 9a and 9b sentences above, verbatim or tighter.
- Forbidden: "the shuffle killed the causal reading"; "practicing A
  raises/lowers B" anywhere outside the pre-registered-future-work
  framing; temporal-mechanism language for the surviving association.

**Refusals.**
- Allowed pattern: certified, harness-and-dataset-specific verdict plus
  ancestor citation plus the sentence that the deliverable is the
  verdict, not the fact. G9 exemplar: "per-KC growth resolution fails
  certification at both tested density extremes by comparable margins
  with the same failure signature, invariant under a full density
  inversion; we infer, and label as inference, that more data at these
  densities will not license it." Ancestors cited: Beck & Chang and
  Khajah et al. for per-skill non-identifiability, classical ceilings
  and wheel-spinning for saturation, textbook negative controls for the
  shuffle.
- Forbidden: refusal contents presented as discoveries; "fundamental
  floor" as a certified fact without the inference label; "will not
  come from more data" as an unlabeled theorem.

**The word "certified" and the protocol noun.**
- First use defines it operationally: "passed pre-registered bars on
  the synthetic harness under seed discipline, thresholds frozen before
  any run"; one sentence states certification is necessary, not
  sufficient (it licenses claims against known failure modes, it does
  not guarantee truth under unknown ones); M2's two-profile bracket
  supports robustness to density, not robustness to process, stated
  exactly. Scope suffix thereafter. Conservative reading for
  S11/S12/S13 in all prose; the stronger-tag adjudication may appear
  once, as a footnote, so the paper is internally consistent.
- The contribution noun is "protocol", never "framework". Generality is
  claimed as "demonstrated end to end once and transferred once"; the
  protocol transfers, the verdicts do not. The certificates attach to
  the paper's own instruments (the passive gate, the pinned response-
  blind transfer readout); the field-representative tracker was tested
  and refused (G17 is a headline result); PAS-N2 is a construction
  guarantee with no measured verdict (P2 wording mandatory).
- Banned phrase: "two certified demonstrations" (unqualified, or at
  all). The architecture sentence is "one completed certify-then-confirm
  cycle, plus the protocol's second instantiation caught mid-cycle."

---

## 4. Threat-response table

Fatal and major threats, with the neutralizing framing/outline change.
Persona key: P1 AUC-culture reviewer, P2 measurement theorist, P3
causal-inference reviewer, P4 novelty editor, P5 split editor.

| # | Persona / severity | Threat (compressed) | Neutralizing change |
|---|---|---|---|
| 1 | P2 fatal | Growth centerpiece trivialized by model-free baseline (R5 shows the rise; Rogosa objection) | Every growth sentence validates the READOUT, never discovers growth; R5 co-located with R1; map rows defined as readout capabilities, not facts about learners (card, growth pattern) |
| 2 | P3 fatal | Endogenous scheduling can generate signed dose-association with zero transfer; no real-data association cell exists | Hard scope wall: association arm carries zero real-data claims and zero intervention implications; real-bed names in association context are triage/planned-work only; abstract clean (card, association pattern) |
| 3 | P4 fatal | Boundary map sold as complete while P1/P2/P3/P9 are pending and RB4 is dead code | Centerpiece rescoped to "the licensing map as measured by the executed campaign"; every cell status-labeled; pending visually distinct; P3 named plainly in text; incompleteness framed as the thesis (claims stop where certification stops) |
| 4 | P5 fatal | False two-pillar symmetry (complete growth arc vs mid-cycle association pilot) | Architecture restructured: one completed cycle plus one mid-cycle instantiation; asymmetry stated in the same sentence wherever the pair is introduced; "two certified demonstrations" banned |
| 5 | P1 major | Flagship redundancy ("a learning curve shows this for free") | Early load-bearing paragraph: raw curves attrition-biased (Nixon via T4), raw separation is a density artifact (G3), naive gate false-fires on saturation that raw curves cannot diagnose (G5/G6); R1 framed as instrument transfer |
| 6 | P1 major | Association arm synthetic-only at toy K, presented co-equal | Same as #4 plus inline D=3/N=500/single-tag scope on every association sentence (rule 7; card) |
| 7 | P1 major | Harness circularity (twins near own model family; bank below bar) | Saturation inversion presented as the harness probing misspecification (G5/G6, G22); S8 reported as measured propagation; certification stated necessary-not-sufficient; M2 scoped to density-robustness only |
| 8 | P1 major | No priced stakes | Motivational architecture: field's uncertified consumption (LTKT/HawkesKT per fact-checks, prediction-only validation); Paper 1 decision-cost receipts imported as item-side precedent (continuity brief section 4); limitation sentence that growth/influence misread cost is argued from precedent, not measured here |
| 9 | P1 major | Map reads as a pile of negatives | Rendered as prescriptive licensing table: rows = claim tiers (pooled existence / per-KC / magnitude / signed association / causal), columns = data regimes, cells = licensed-with-conditions or refused-with-mechanism; actionable rules named (G8/R1, M9, S12, G9) |
| 10 | P2 major | "Two certified demonstrations" self-refuting with P3 unexecuted | Same as #4; the honest architecture sentence replaces the phrase everywhere |
| 11 | P2 major | Permutation null is model-conditional; R1 rests on 257/515 | Text states what the permutation exchanges and for which null family the test is size-valid; saturation inversion presented as certification-relative-to-reference; exclusion count promoted to headline condition of R1 |
| 12 | P2 major | Bank below its own bar; growth-gate propagation unmeasured; difficulty-opportunity confound | Growth certificate stated conditional on bank fidelity at 0.70-0.80 (G12); ordering confound named in R1 scoping and map; S8 scoped to association, gap stated; gauge point (M11) reported alongside |
| 13 | P2 major | Change-measurement lineage absent | One-paragraph strand added to related work: Andersen, Embretson, Fischer, latent change scores, Cronbach & Furby, Rogosa & Willett; test form conceded classical, protocol and map claimed |
| 14 | P2 major | Dose floor is a single-cell MDE presented as intrinsic | Operating-point conditioning at every mention; Hertzog and Card cited as the two MDE lineages; floor presented as the first measured point of a power surface whose density axis is known to diverge (S7) |
| 15 | P2 major | G9 "will not come from more data" overreaches two density points | Rescoped wording with inference label (card, refusals pattern); killed-refused register kept, theorem register banned |
| 16 | P3 major | Conflated refusals: shuffle kills temporal, not causal dose | Rule 9 split into 9a/9b; two separately sourced verdicts everywhere (S14/S15 vs P5) |
| 17 | P3 major | Floor does not transport (K, density, bank, generator family) | "Harness-measured floor at the certified operating point"; floor-measurement sold as the capability, floor-transport refused; S7/S8 folded into the map as floor-modifier cells |
| 18 | P3 major | Multiplicity at deployment K (~265k pairs) never demonstrated | Scale arithmetic stated in the map legend/limitations; per-edge existence at scale refused pending CT8/CT9 plus matrix-level multiplicity; M10 never cited for the G matrix without its scope |
| 19 | P3 major | Certified quantity's usefulness undefended | Two-leg use argument (audit of the field's actual estimand; screening quantity with pre-registered upgrade path); Kane anchor; intervention utility never implied |
| 20 | P4 major | "Framework" is an assembly of textbook practices | Contribution noun "protocol"; lineage cited in the first related-work paragraph; matched twins named as the ownable device; framework language deleted |
| 21 | P4 major | Thin positives inflated by framing | Exact-breadth language structurally enforced (card patterns); no plural datasets; R1 always with subset and breadth |
| 22 | P4 major | Refusal contents are known results | Verdict-not-fact writing with ancestor citations (card); G9 led with as the one refusal carrying genuinely new content |
| 23 | P4 major | Certificates attach to bespoke instruments, not field models | Explicit scope statement in the introduction: the gate certifies readout instruments; certified instruments here are the passive gate and the pinned transfer readout; G17 headlined as the field-default architecture failing the battery; N2 stated per P2; "protocol transfers, verdicts do not" |
| 24 | P4 major | No single crisp contribution | One contribution sentence, four conjunction-scoped clauses (section 2) |
| 25 | P5 major | S11/S12/S13 tag conflicts inside the paper's own map | Conservative reading in all prose ("on the synthetic harness" attached); adjudication stated once in a footnote |
| 26 | P5 major | Map is a half-filled ledger | Same as #3: four cell states, honest caption, P1 drafted both ways |
| 27 | P5 major | All real-data weight on one cell (R1) | R1 structurally paired with R2 as a designed positive/negative real-data pair; the protocol claim rests on the protocol and the map, never on R1 alone |
| 28 | P5 major | Framework overreach; lineage owned elsewhere | Same as #20; generality claimed as "demonstrated end to end once and transferred once" |

Minor threats (fixes adopted, one line each):
- One-seed/one-split asymmetry vs M7 (P1): pre-empted in text; breadth
  clause mandatory; certification claims carry the seed discipline, the
  real fire is a transfer confirmation.
- Framework-lineage optics (P1): covered by #20.
- "Certified" self-issued (P2): operational definition at first use;
  footnoted adjudication (card).
- No-decline scope (P2): estimand sentence in the framework section;
  map row "decline: out of scope by construction" (G21).
- Single-replication reliability (P2): R1 slotted as "closing the loop
  once"; limitations name replication breadth as the map's thinnest
  point.
- Map mixes epistemic realms (P3): two-layer legend,
  harness-certified vs real-data-confirmed visually distinct;
  association column entirely in the first layer.
- Lab-notebook padding (P4): only M9 and E3 enter the main text as
  methodological findings; seed discipline gets one paragraph; the rest
  goes to a reproducibility appendix or nowhere.
- Paper 1 inheritance (P4, P5): one paragraph, exactly as the
  continuity brief scopes it (item-side static audit there, person-side
  dynamics and multi-KC structure here); never re-litigate, never imply
  SK fixes person-side readouts.
- S7/R2 parallelism (P5): named in one sentence as the same kind of
  boundary finding, the union's best defense (section 5).
- HawkesKT read before priority language (P5): DONE; verdict SIGNED;
  conjunction wording already survives it (card).

---

## 5. Union defense

One paper. The deliverable is the licensing map, and the map's
strongest single finding exists only in the union. The growth arm's
real-data density boundary (deep KDD fires, thin Junyi is correctly
silent) and the association arm's synthetic density collapse (sign
recovery dies at multi-tag density) are the same kind of finding, an
identifiability boundary of a readout mapped by the same protocol, and
they are visible side by side only in one paper. The protocol claim
also needs both arms. The growth arm proves the gate can license and
confirm; the association arm proves it can refuse, demote, and stop,
which is the evidence that the gate is not rigged toward either
outcome. Split, the parts are worth less. The growth arm alone is a
solid narrow paper that loses its generality evidence and its map. The
association arm alone is an unpublishable synthetic-only pilot at toy K
with its battery unexecuted. The split editor's own strongest version
concedes the union on one condition, that the asymmetry sit on the
cover rather than in the limitations, and the architecture adopted here
(one completed cycle, one instantiation caught mid-cycle, stated
wherever the pair appears) satisfies exactly that condition.
Recommendation: union.

---

## 6. Deltas vs F1 as approved

1. Contribution noun "framework" replaced by "protocol"; lineage (SBC,
   sanity checks, negative controls, MDE floors) cited preemptively;
   matched twins named as the ownable device.
2. "Growth and signed association are the two certified demonstrations"
   replaced by "one completed certify-then-confirm cycle (growth) plus
   the protocol's second instantiation caught mid-cycle (association)";
   the old phrase is banned.
3. Association arm demoted behind a hard real-data scope wall: no
   real-bed association claims, no intervention language, operating
   point inline everywhere, battery-pending stated plainly.
4. Boundary map re-specified as a licensing map with four cell states
   (certified on synthetic harness / real-data confirmed / refused with
   mechanism / pending pre-registered), two-layer epistemic legend, and
   prescriptive rows; incompleteness framed as thesis-consistent.
5. Rule 9 split (temporal refuted vs causal unearned); rule 3 resolved
   (HawkesKT signed); rule 1 tightened with the model-recovered clause
   (LTKT's signs are stipulated inputs).
6. Growth framing fixed as readout validation, never growth discovery;
   raw-curve baseline co-located with R1; exact-breadth and
   subset clauses mandatory; bank-fidelity conditionality and the
   no-decline estimand scope stated.
7. "Certified" operationally defined at first use, necessary-not-
   sufficient stated, conservative tag reading adopted for S11/S12/S13.
8. Added related-work obligations: change-measurement strand (Andersen,
   Embretson, Fischer, latent change scores, Rogosa), Kane use-argument
   paragraph, HawkesKT and LTKT differentiation sentences per the
   full-text verdicts, active SLC differentiation retained.
