# Paper outline (kt-mirt paper, plan of record)

Reconciled 2026-08-10 from `outline/outline_evidence_first.md` (EF) and
`outline/outline_narrative_first.md` (NF). The framing memo
(`outline/framing_memo.md`) governs wherever anything conflicts; claim
strength comes only from `evidence/claim_evidence_map.md`; the binding
rules in `outline/claim_language_constraints.md` (as tightened by the
memo) apply to every sentence. Conservative reading of S11/S12/S13 ("on
the synthetic harness") in all prose, adjudication footnoted once. The
banned phrase "two certified demonstrations" appears nowhere; the
architecture sentence everywhere is "one completed certify-then-confirm
cycle, plus the protocol's second instantiation caught mid-cycle." The
association operating point (D=3, N=500, single-tag density, calibrated
bank, untuned default trainer) is inline at every floor or certification
mention. Register sources for all prose: memory `writing-style.md` and
`overleaf-sync/old/main_magpcm_ijaied.tex`.

Structural model: the Paper 1 skeleton (motivation, strand-based related
work, framework before experiments, a design section that names its
comparisons, synthetic truth-based core, real-data scope-limited checks,
honest discussion), per `evidence/paper1_continuity.md` section 3.

Body budget: 9,600 words across sections 1-9 (envelope 8,000-10,000).
Appendices and abstract are outside the body budget. Divergences between
EF and NF are resolved in the reconciliation decisions appendix at the
bottom; decision IDs (D1..D19) are cited where they bind.

---

## Section 1. Introduction

Job: establish that deep KT latent-state readouts are consumed as
measurement while validated only by prediction, price the stakes by
precedent, and state the contribution sentence with four
conjunction-scoped clauses and the completed-cycle/mid-cycle asymmetry
on the cover.

Target: 850 words.

Ordered claims and evidence:

1. The field publishes growth curves, mastery states, and cross-skill
   influence graphs from KT latents and validates them by prediction
   alone. Evidence: S21 (LTKT DOI, HawkesKT NDCG .8267, sweep found both
   target claims open); framing memo section 3 fact-check verdicts
   (HawkesKT SIGNED but uncertified and response-conditioned; LTKT
   SIGNED but stipulated pre-computed inputs, stated test leakage).
   Named consumers: Deep-IRT, PSI-KT, HawkesKT, LTKT.
2. Prediction cannot certify a readout; neither can raw learning curves.
   Load-bearing early paragraph (threat row 5): raw curves are
   attrition-biased (Nixon via originality threat T4), the clean raw
   separation on the deep profile is itself a density artifact (G3,
   preview), the naive gate false-fires on saturation in a way raw
   curves cannot diagnose (G5, preview).
3. Paper 1 precedent, one paragraph exactly as the continuity brief
   scopes it: item-side static readouts can be stable and wrong with a
   priced decision cost in simulation; person-side dynamics and multi-KC
   structure are this paper's object. Never re-litigated, never implying
   SK fixes person-side readouts. Evidence: continuity brief sections 1
   and 4 (threat row 8).
4. The paper's answer, the certification protocol (M1, M2 preview), the
   frozen framing paragraph (framing memo section 1) compressed to
   introduction length.
5. One contribution sentence, four clauses verbatim from framing memo
   section 2:
   - C1 protocol (M1-M8, M12, M15, G22; lineage T7-T10 owned
     preemptively);
   - C2 certified growth-existence readout with one closed
     certify-then-confirm loop (G1, G2, G4, G5-G8, R1, R2, R5, R3);
   - C3 first certified signed cross-skill readout, honestly demoted to
     dose-association at a named operating point (S2, S16, S11
     conservative, S7, S8, S12, S13, S14, P5, P3);
   - C4 the licensing map with refusals as first-class,
     mechanism-attached results (G9-G17, G21, S9, S10; pending P1-P5,
     P9).
6. Architecture sentence (mandatory wording), scope statement (threat
   row 23: certificates attach to the paper's own instruments, the
   passive gate and the pinned response-blind transfer readout; the
   field-representative tracker was tested and refused, G17 preview;
   the protocol transfers, the verdicts do not), and a roadmap paragraph
   carrying the asymmetry sentence.

Killed/refused presence: previews only (G3, G5, G17), each flagged as a
result delivered in sections 5-6.

## Section 2. Related work

Job: convert every originality threat into cited lineage before a
reviewer can, in four strands (D3).

Target: 1,000 words.

Ordered strands and evidence:

1. Uncertified readout consumption in KT. Deep-IRT, PSI-KT, GKT, GIKT,
   HawkesKT, LTKT, and the 2025-26 newcomers (KeenKT, UKT, PAKT, KTCF);
   audit-flavored newcomers stop at prediction protocols or
   teacher-facing use; SLC actively differentiated (post-hoc repair
   without certification vs certify-then-claim). HawkesKT and LTKT get
   the full-text differentiation sentences from framing memo section 3.
   Evidence: S21; named_analogs 2-7; newcomers gap assessment; rules 1,
   3 (resolved), 5.
2. Growth detection and its known limits. P(J) (T3), AFM/iAFM slopes
   (T4), Khajah et al. (T6), Beck & Chang identifiability (T5),
   wheel-spinning and classical ceilings (T16). Detectors and slope
   inference are prior art; certification is not. Evidence: rules 2, 6.
3. Methodological lineage, owned preemptively (threat row 20). SBC
   (T7), Adebayo sanity checks (T8), permutation nulls and negative
   controls (T9), MDE floors with Hertzog and Card as the two lineages
   (T10), synthetic-recovery routine in KT (T11). The assembly for KT
   readouts and the matched-twin device are the claimed residue
   (contribution clause C1 wording).
4. Change measurement and the validity argument. Andersen, Embretson,
   Fischer, latent change scores, Cronbach & Furby, Rogosa & Willett
   (threat row 13); one Kane interpretation/use paragraph converting
   "this is just validity theory" into lineage (threat row 19; T15).

Killed/refused presence: none delivered; strand 2 pre-cites the
ancestors so later refusals read as verdicts, not discoveries (rule 6).

## Section 3. The certification protocol

Job: define the protocol operationally before any result, so every later
verdict reads as a mechanical application of stated rules.

Target: 1,050 words.

Ordered claims and evidence:

1. Certify-before-claim discipline (M1). "Certified" operationally
   defined at first use (passed pre-registered bars on the synthetic
   harness under seed discipline, thresholds frozen before any run);
   necessary-not-sufficient stated in the same paragraph; conservative
   tag reading stated, adjudication footnoted once (framing card).
2. Matched twins as the ownable device: synthetic replicas calibrated to
   named real beds, growth or transfer switched on and off; two profiles
   at opposite density extremes bracket the range, supporting
   density-robustness only, never process-robustness (M2; threat row 7).
   Naming discipline "KDD-shaped" / "EdNet-shaped" introduced here (R3
   clause).
3. Pre-registration mechanics: A4 v1.1 and A1 v1.1 frozen before any
   run, 10 arms, 13 gates, K1-K7 and K-T1..K-T6 kill conditions,
   thresholds tighten but never loosen, two-revision cap, every kill
   condition carrying a named honest verdict (M3, M4, M8). Table 2
   (compact gate/kill registry) attaches here; full registry in
   Appendix A.
4. Statistical machinery: permutation null against fitted no-growth
   references with the model-conditional caveat stated (what the
   permutation exchanges, for which null family the test is size-valid;
   threat row 11); matched-null band methodology, never bare zero (M6);
   seed discipline end to end, tuning seeds 100-102 separated from
   certification seeds 0-4, seed-clustering rule for confirmatory
   claims (M5, M7); multiplicity control itself certified under matched
   dependence, scope limited to per-KC growth statistics (M10).
5. Anchoring and estimands: magnitudes anchored to the PNAS population
   learning rate, 20-fold rate spread, pre-registered dose sweep (M12);
   rate estimand affine-invariant, displacements gauge-dependent on the
   frozen anchored scale (M11); exactly three independent existence
   inputs, active-only firing never claimed as growth (M14); reference
   transfer edges placed only between pairs sharing no co-tagged item,
   same-slot co-observation a separate confound arm (M15).
6. Estimand scope sentence: certified object is existence of monotone
   ability gain at cohort grain; decline out of scope by construction,
   carried as a map row (G21, the first refused row delivered).
7. Refusal-as-deliverable defined as a protocol output class; the four
   verdict states named (certified on the synthetic harness, real-data
   confirmed, refused with mechanism, pending pre-registered).

Killed/refused presence: G21 delivered as the protocol's first refused
scope row.

## Section 4. Instruments, twins, and beds

Job: name the objects under audit, the synthetic twins, and the real
beds, with the comparisons the experiments will run, so sections 5-6
read as executions rather than exploration (D1, D4, D5, D6).

Target: 900 words.

Ordered claims and evidence:

1. Instruments: the passive existence gate; the active posture (ACT)
   with the stationarity-gated trainer and the rho=1 no-decay pin (G21
   context); the pinned response-blind transfer readout with gamma=1
   and structural isolation of the cross-KC route (S17 qmirt ancestry
   of the pin, S18 response-blindness by construction, S19 isolation
   unit-tested); the field-representative shared-state neural tracker
   PAS-N1 as the field-default architecture under test; PAS-N2 stated
   as a construction guarantee with no measured verdict, mandatory P2
   wording.
2. Trainer stationarity as a precondition (D4): both pre-convergence ACT
   reads were optimization artifacts in opposite directions; the
   no-growth twin caught a defect prediction metrics never would (G22).
   The protocol auditing its own optimizer, placed before results to
   defuse the circularity threat (row 7); the same pathology reappears
   in the transfer setting (S6 forward reference).
3. Twin construction and profiles: KDD-shaped few-deep, EdNet-shaped
   many-thin; four twins (ng, kg, ns, sat) x five slice seeds x two
   profiles; three model seeds on neural arms as a stated pre-registered
   compute concession (M2, M7).
4. Bank: hierarchical MAP difficulty bank forced by extreme step
   sparsity (E13), penalized slice fits shared identically by null,
   alternative, and permutation replicates (E14); recovery stuck at
   0.70-0.80 against the 0.9 bar, and the announcement that this
   bank-fidelity floor rides into every downstream readout (G12).
5. Real beds in one triage table (Table 1): nine beds, no bed wins every
   axis and the split is itself a finding (R15); KDD primary growth bed
   (R14, R7; R5 context), Junyi thin-practice (R10), EdNet Tier-1 cap
   with the bundle confound, ACT never on real EdNet (R8, R9), Eedi
   decoupling lead and the only real negative-transfer hook, the
   misconception-label premise dead (S22, R11), junyi15 magnitude with
   its sequencing confound (R6), exclusions (R12, R13, S26 structural
   vacuity). KC-model choice measured, not assumed (S24); same-slot
   positivity bars, Junyi's unmeasured (S25); decoupling metric one
   line (M13, detail Appendix B).
6. The top-K volume-selection hazard as the program's single most
   important methodological caution, placed here because it shapes every
   later pair selection (M9, 6.7% vs 72.2%) (D6).

Killed/refused presence: G12, G21, G22 as standing conditions; R9, R13,
S26 as bed-level exclusions with mechanisms.

## Section 5. The growth-existence readout: a completed cycle

Job: walk the full certify, diagnose, repair, confirm loop for the one
readout that earned it, with the refusals at finer grain as results of
equal rank inside the same section, and the real-data pair closing the
cycle (D1, D2).

Target: 2,200 words (5.1: 400, 5.2: 400, 5.3: 650, 5.4: 300, 5.5: 450).

### 5.1 Certification at cohort grain

1. The passive gate certifies as a twin-level detector on both density
   profiles: silent on every null twin, at the permutation floor on
   every growth twin, surviving a full density inversion (G1: NG p
   .128-.973 and .222-.876, KG/NS p=.001 all seeds both beds). Null
   cleanliness (G2: split-half gaps .042-.098, BH 0/189 x5). Figure 2.
2. The raw bed_stat separation is partly a density artifact and is not
   sold as general; only the calibrated permutation p-value recovers
   separation at thin density (G3, killed-refused, delivered as a
   result that disciplines the headline; Fig 2 inset, D13).
3. Strength cap: the gate says growth is present, never how much (G4,
   p=.001 floor arithmetic).

### 5.2 The saturation false fire, diagnosed and repaired

1. CG6 inverts on both profiles: the gate fires hardest exactly where it
   must stay silent (G5, killed-refused; fires p=.001 x5). Framed per
   threat row 7 as the harness probing reference-model misspecification.
2. Mechanism: the no-growth reference approximates a saturating curve
   worse, corroborated by independent numerical degeneracy (G6); density
   modulates severity, not existence (G7, 3.7x vs 1.32x).
3. The saturation-aware bed null eliminates the false fire without
   harming detection or the null; per-KC statistics untouched (G8,
   syn_sat p 0.01 to 1.00). Certification thereafter stated relative to
   the repaired reference. Figure 3.

### 5.3 Refusals at finer grain (results, not caveats)

1. Per-KC growth resolution refused at both density extremes, comparable
   margins, same signature, invariant under density inversion; G9 leads
   as the one refusal carrying genuinely new content (threat row 22),
   with "more data at these densities will not license it" written as
   labeled inference, never theorem (framing card). Supporting zero-power
   detail (G10) and the pre-registered K7 rate-recovery floor (G11, rank
   ~0.08 vs 0.6 bar). Ancestors cited in place: Beck & Chang, Khajah.
   Figure 4.
2. The bank-fidelity floor restated as a rider on every downstream
   readout (G12, 0.70-0.80 vs 0.9 bar).
3. ACT refused as magnitude estimator and per-KC ranker (G13, undershoot
   5-10x, corridor and rank failures), with the honest remainder:
   presence and direction recovered, thin-density rank markedly better
   (G14), no variant clean on both nulls and both profiles
   simultaneously (G15); the misfit clause unusable by opposite numeric
   routes (G16). Table 3 (D11).
4. Split-half reliability passes on non-saturated twins and is
   diagnostically blown out on the saturated twin (G19); Tier-2 KC rate
   reliability not certifiable from any artifact, a named harness wiring
   gap, not a statistical null (G20, engineering).

### 5.4 The field-default tracker fails the battery

1. PAS-N1 fails the faithfulness battery consistently on both beds at
   production scale, all four audits (G17, killed-refused, headlined as
   the field-default architecture failing, threat row 23); the single
   borderline cell honestly reported (G18). Table 4.
2. PAS-N2's superiority is a construction guarantee with no measured
   CG7-CG10 verdict in any cell; sanity probe only (P2, mandatory
   wording).

### 5.5 Real data: the fire, the silence, the boundary

1. Synthetic roll-up first, licensing the real-data step (G23: partially
   certified, the exact earned/not-earned list) (D19).
2. The fire, allowed pattern verbatim from the framing card: the
   synthetic-certified cohort-grain gate fires on real KDD Algebra under
   the saturation-aware null, on the unsaturated 257 of 515 KCs, one
   seed, one cohort split, B=99, p=0.01 (R1). Subset and breadth clauses
   in the same sentence as the fire, always. Conditionality in the same
   passage: bank fidelity at 0.70-0.80 (G12), the difficulty-opportunity
   ordering confound of a mastery-managed ITS named, bank-error
   propagation measured only for the association sign (S8 cross-
   reference), per-KC remains 0/515 on real data.
3. R5 co-located (threat row 1): model-free first-attempt curves rise
   12-19 points, so the fire validates the instrument against a
   known-growth bed, never discovers growth.
4. The designed silence: thin-practice Junyi shows no model-detectable
   growth, the biased-sample twin rules out a sampling artifact, and the
   verdict is a data-density boundary, not absence of learning (R2,
   rows-per-student bracket 105/557/2688). R1 and R2 presented as one
   designed positive/negative real-data pair (threat row 27). Figure 5.
5. The pending deep-Junyi cell, both outcomes drafted neutrally (P1,
   rule 10).
6. Overall verdict: partially certified, exact R3 wording; real-bed
   reporting licensed only at Tier 1 (R4); tiers RB4/RB5 never
   exercised, honest-outcome pre-registration stated (P9).

Killed/refused presence: this section is the refusal engine of the
growth arm; G3, G5, G9, G10, G11, G12, G13, G15, G16, G17 delivered
here as numbered results with mechanisms and ancestors.

## Section 6. Signed cross-skill association: caught mid-cycle

Job: report the second instantiation exactly as far as it got, one
licensed sentence at a named operating point, floor and floor-movers
measured, kills reported as designed falsifications, everything
real-data behind a hard scope wall. Zero real-data association claims
anywhere in this section; real-bed names appear only as planned-work
pointers back to section 4 (threat row 2, D5). Operating point inline
at every floor or certification mention.

Target: 1,800 words (6.1: 350, 6.2: 300, 6.3: 450, 6.4: 700).

### 6.1 Feasibility and sign recovery

1. CT0 inconclusive-but-alive: no cell clears the full CT1 bar at D=3,
   the fail-fast sign-unidentifiability kill does not fire (S1).
2. Per-edge signed coefficients recovered near truth at single-tag
   density at every tested N, refuting sign-unidentifiability outright
   (S2).
3. The failure is discrimination, not recovery: true-zero leakage
   overlaps the negative reference dose (S3); the positive half
   sign-separable with a measured effective-sample ladder (S4); the
   negative half without a stable threshold at 1x dose (S5); the
   matched-null band grows with epochs in a flat-NLL basin, reproducing
   the ACT-P0 pathology (S6, tied back to G22).

### 6.2 The repair that failed, and what that taught

1. The L1-and-early-stopping repair hypothesis refuted: no configuration
   pins true zeros while sparing true edges; winner rule frozen in code
   before results, tuned only on held-out seeds (S9, killed-refused).
2. CT0's trainer-artifact explanation overturned: the leak is a
   small-signal identifiability property of the objective at this grain
   (S10, killed-refused); the stationarity re-derivation closed with
   default numbers standing (S20, engineering). Figure 7a.

### 6.3 The certified floor and its movers

1. The negative-edge detectable-dose floor is |g|=0.04, twice the
   field-anchored reference dose, measured on clean certification seeds
   under pre-registered per-dose bars, certified on the synthetic
   harness at D=3, N=500, single-tag density, calibrated bank, untuned
   default trainer (S11, conservative tag; adjudication footnote here).
   Floor-measurement is the capability; floor-transport is an explicitly
   refused claim (threat row 17). Figure 6.
2. The floor moves, measured: sign recovery collapses outright at
   multi-tag density, a genuine per-density identifiability limit (S7,
   killed-refused, signF1 max 0.444); bank error at the measured
   recovery floor degrades without collapsing the positive half, the
   sign claim is bank-sensitive (S8). Both fold into the map as
   floor-modifier cells; framed with Hertzog and Card as the first
   measured point of a power surface (threat rows 14, 17). Figure 7c.
3. The binding constraint: a dose-independent 5-15% true-zero false-edge
   background mandating multiplicity control on any real-data use (S12,
   conservative tag); scale arithmetic stated here (threat row 18, D18):
   at K=515 roughly 265,000 ordered pairs, tens of thousands of false
   signed edges before correction; per-edge existence at deployment K
   refused pending CT8/CT9; M10 never cited for the G matrix without
   its scope.

### 6.4 Controls, the kill, and the licensed sentence

1. The phantom-transfer control passes as pre-registered: freeing gamma
   fabricates transfer on the null twin 5/5; the pin retained on
   structural grounds (S13, conservative tag; construction facts S17,
   S18, S19 by back-reference to section 4).
2. The order-shuffle negative control refutes the temporal-transfer
   reading: about three quarters of both signed magnitudes survive
   against the 0.10 collapse bar, a designed falsification that worked;
   rule 9a wording only (S14, ratios 0.741/0.815; S15 lag component
   bounded and uncertified). Figure 7b.
3. The causal cumulative-dose reading is unearned, not refuted: exogenous
   scheduling holds in the harness by construction (M15) and is untested
   on real beds; the endogenous-scheduler control is pre-registered and
   unexecuted (P5; rule 9b verbatim). Never "the shuffle killed the
   causal reading."
4. The licensed sentence in full (S16): per-edge signed dose-association
   at the named operating point, negative-half floor 0.04 inheriting the
   S11 caveat (conflict 3 adopted conservatively), positive half robust
   from +0.05.
5. Positioning: the three-way conjunction (model-recovered AND certified
   against ground truth AND floor-measured) against LTKT's stipulated
   inputs and HawkesKT's uncertified response-conditioned signs (S21;
   rule 1). The two-leg use argument with the Kane anchor (D7): the
   certified quantity is what the field's signed readouts estimate at
   best, and it is a screening quantity with a pre-registered upgrade
   path (threat row 19).
6. What remains unexecuted, stated plainly: the re-baselined full
   battery (P3, Table A1), external graph corroboration conditional on
   the unmeasured Junyi positivity bar (P4; S25 and Appendix D by
   pointer), the decoupling-bar re-certification (P7), and the
   pre-registered clean-negative verdict that would report the claim
   unsupported (P6).

Killed/refused presence: S7, S9, S10, S14, S15 delivered here as
results; the hard scope wall itself presented as an output of the
protocol's refusal discipline.

## Section 7. The licensing map

Job: assemble every verdict into the centerpiece deliverable, a
prescriptive licensing table whose incompleteness is thesis-consistent
(claims stop where certification stops).

Target: 700 words.

Ordered content:

1. Map specification: rows are claim tiers (pooled existence, per-KC
   resolution, magnitude and rank, per-learner rates and reliability,
   signed association, temporal reading, causal reading, decline);
   columns are data regimes (KDD-shaped synthetic, EdNet-shaped
   synthetic, real deep bed, real thin bed, multi-tag density) (D9);
   cells carry one of four states (certified on the synthetic harness,
   real-data confirmed, refused with mechanism, pending pre-registered).
   Two-layer epistemic legend, harness-certified visually distinct from
   real-data-confirmed; the association column entirely in the harness
   layer. Table 5.
2. Cell population, each with its map ID: G1, G2 certified; S11, S12,
   S13, S16 certified on the synthetic harness at the named operating
   point; R1, R2 confirmed; G3, G5 (pre-fix), G9-G13, G15, G16, G17,
   G21, S7, S9, S10, S14 refused with mechanism; P1, P2, P3, P4, P5, P9
   pending (D10).
3. Prescriptive licensing rules named as rules, not narrative (threat
   row 9): the saturation-aware null before any growth claim (G8/R1);
   the full-pair-population check before any top-K read (M9);
   multiplicity control before any per-edge existence claim (S12/M10
   with scope stated); the real-bed licensing tiers (R4/P9).
4. Legend carries the scale arithmetic in compact form (threat row 18,
   D18) and the replication-breadth honesty (conflict 7: R1 is the sole
   real-data positive, the map's thinnest point under a certified
   headline).
5. The union's best sentence (D8): the growth arm's real-data density
   boundary (R2) and the association arm's synthetic density collapse
   (S7) are the same kind of finding, an identifiability boundary of a
   readout mapped by the same protocol, visible side by side only here
   (framing memo section 5).

Killed/refused presence: every refusal reappears as a labeled cell; the
map is the structural answer to "a pile of negatives" (threat row 9).

## Section 8. Discussion

Job: defend the union, state what transfers and what it costs, own the
corrections, and bound the claims, in the register of Paper 1's honest
discussion.

Target: 900 words.

Ordered content:

1. What transfers: the protocol, demonstrated end to end once and
   transferred once; the verdicts do not transfer (contribution noun
   rules).
2. Union defense, full paragraph (framing memo section 5, D8): the map's
   strongest finding exists only in the union; the growth arm proves the
   gate can license and confirm, the association arm proves it can
   refuse, demote, and stop; split, the parts are worth less (the growth
   arm loses its generality evidence and its map, the association arm
   alone is an unpublishable synthetic-only pilot); the asymmetry sits
   on the cover, which is the split editor's own condition.
3. Certification is necessary, not sufficient; the two-profile bracket
   supports density-robustness only; the harness shares model-family
   assumptions and the saturation inversion is the internal probe of
   that limit (M2, G5, G6, G22; threat row 7). One-line pointer to the
   use argument delivered in 6.4 (D7).
4. Methodological findings promoted to results: the top-K selection
   hazard (M9) and the memory-bounded permutation deployment lesson
   (E3); the rest of the engineering ledger goes to Appendix C (framing
   memo minor threats).
5. Corrections the protocol caught, the honest-error subsection in
   Paper 1's genre: the two ACT optimization artifacts and the
   stationarity repair (G22, E4); the bridge LoadStats defect found by
   review and fixed, with the self-consistency caveat (E7, E6); the RB4
   wiring gap (G20); the grid-fraction mismeasurement, one sentence
   (E19).
6. Paper 1 boundary paragraph, exactly as the continuity brief scopes
   it: item-side static audit there, person-side dynamics and multi-KC
   structure here; misread-cost limitation argued from precedent, not
   measured here (threat row 8).
7. Limitations, each already delivered as a result and here only
   collected: replication breadth (conflict 7; R1 one seed, one split);
   bank floor and unmeasured growth-gate propagation (G12; S8 scoped to
   association, gap stated); junyi15 sequencing confound (R6, conflict
   8); Q-matrix circularity residual risk (E12); decline scope (G21);
   pending cells enumerated with both-ways P1 language (P1-P7, P9
   subset).

Killed/refused presence: collected by pointer only; nothing delivered
here for the first time.

## Section 9. Conclusion

Job: restate the licensing map as the deliverable and the claim grammar
as the contribution. The paper moves no AUC and never claims to; it
states, with ground truth rather than opinion, which interpretive
sentences these readouts license and exactly where they stop. The
protocol transfers, the verdicts do not; one closing sentence on the map
plus the harness to re-check it. No new claims, no forward-looking
association promises beyond the pre-registered battery (P3).

Target: 200 words.

## Appendices (outside body budget)

- A. Gate and kill-condition registry, full, with verdicts: every
  pre-registered gate and kill with its frozen bar and executed verdict
  (M3, M4, M8; a4_design.md sections 4-5, a1_design.md sections 4-5),
  absorbing the unexecuted A1 battery rows with re-baselined 0.04 bars
  and status "pre-registered, unexecuted" (P3), the three clean-negative
  triggers (P6), and the RB tier definitions (P9) (D15). Table A1.
- B. Bed triage detail and metric library: per-bed statistics behind
  Table 1 (R5-R14), data-quality facts (E10), runtime and availability
  notes (E17, E18, E21), the decoupling metric definition (M13), the
  EdNet-vs-Eedi wrong-option gap (P8).
- C. Reproducibility and engineering ledger: vendoring and test-suite
  growth (E1), permutation-battery surgeries and cost (E2),
  stationarity-gated trainer (E4), loader exactness and the
  independent-check caveat (E5, E6), byte-identity guarantee for the
  synthetic path (E8), opportunity-tail hypothesis (E9), loader policy
  findings (E11), sparse-bank hierarchy (E13), penalized slice fits
  (E14), complexity and budget notes (E15, E16, E20); seed discipline
  one paragraph (M5, M7).
- D. Junyi15 prerequisite-graph pilot, appendix-scoped per PLAN.md
  decision 2 (S23 cycles, P4 conditionality) (D16).

---

## Figure and table plan (deduped; 7 figures, 6 tables)

Every source is a file under `kt-mirt/outputs/` or a named table in a
doc under `kt-mirt/_planning/`. Real-bed result JSONs live on the
cluster; their numbers are sourced from the ledger and verdict docs as
noted.

| # | Name | Content | Data source | Section |
|---|------|---------|-------------|---------|
| Fig 1 | Protocol flow and cell states | Certify-then-claim pipeline: twins, frozen bars, permutation null, confound arms, seed discipline, four output states; the asymmetric two-instantiation architecture drawn in | Schematic; gate lists in `kt-mirt/_planning/design/a4_design.md` sections 4-5 and `a1_design.md` sections 4-5; framing memo section 1 | 3 |
| Fig 2 | Growth certification matrix | Per-twin, per-seed, per-profile permutation p-values (4 twins x 5 seeds x 2 profiles), null silence vs floor detection, density inversion visible; G3 inset showing raw-stat overlap at thin density (D13) | `kt-mirt/outputs/a4/campaign/{kdd_matched,ednet_matched}/syn_{ng,kg,ns,sat}/slice_seed{0..4}.json`; roll-up in `kt-mirt/_planning/verdict_synthetic_complete.md` section 1 | 5.1 |
| Fig 3 | Saturation false fire and repair | Pre-repair sat p=.001 x5 both beds with the 3.7x vs 1.32x inflation contrast; post-repair p 0.01 to 1.00 with kg/ns detection held | Campaign syn_sat JSONs (pre-repair); verify_sat_fix table in `kt-mirt/_planning/LEDGER.md` (2026-07-21) and `verdict_synthetic_complete.md` section 2 | 5.2 |
| Fig 4 | The per-KC refusal | BH discovery counts (0/515, 0/189 across all seeds, both profiles); bank rank-corr 0.70-0.80 vs 0.9 bar; K7 rate-recovery rank ~0.08 vs 0.6 bar; same failure signature both densities (G9-G12) | Campaign slice JSONs; CHECK C tables in `kt-mirt/_planning/verdict_synthetic_complete.md` section 3; `verdict_kdd_g2.md` section 1 | 5.3 |
| Fig 5 | The real-data pair and the density boundary | KDD fire (bed_stat 6113.8, p=0.01, B=99, 257/515 subset flagged on the panel) vs Junyi silence (-11763.8 random, -11038.9 biased) on a rows-per-student density axis (105/557/2688), deep-Junyi cell drawn pending; breadth clause printed on the panel, not only the caption (D14) | Real-data verdict tables in `kt-mirt/_planning/LEDGER.md` (2026-07-23 entries) | 5.5 |
| Fig 6 | Detectable-dose floor | Negative-edge metrics (Gneg, separation, negF1, FER) across doses 0.01/0.02/0.04/0.08 with per-dose bars, the certified 0.04 point marked, the false-edge background band shown dose-independent; operating point in the caption | `kt-mirt/outputs/a1/r0a1/r0a1_floor_cert.json`; table in `kt-mirt/_planning/r0a1_interference_verdict.md` section 2 | 6.3 |
| Fig 7 | Kills and floor-movers panel (D12) | (a) L1 ladder, real edges shrinking faster than the null band (S9, S10); (b) order-shuffle ratios 0.741/0.815 against the 0.10 collapse bar (S14); (c) density collapse single-tag vs multi-tag signF1 plus bank sensitivity 0.889 to 0.756 (S7, S8) | `kt-mirt/outputs/a1/r0a1/r0a1_kill_arms.json` and `r0a1_study.json`; `kt-mirt/_planning/r0a1_interference_verdict.md` sections 1-3; mechanism 4-5 tables in `kt-mirt/_planning/ct0_power_result.md` | 6.2-6.4 |
| Tab 1 | Bed triage | Nine beds x growth/saturation/depth/decoupling axes, tier caps and exclusions (R9, R13) marked as refusals with reasons | `kt-mirt/_planning/triage/triage_report.md` synthesis table; per-bed JSONs `kt-mirt/_planning/triage/*_stats.json` | 4 |
| Tab 2 | Gate and kill-condition registry (main-text compact; full as Table A1) | Every pre-registered gate/kill with its frozen bar and executed verdict | `kt-mirt/_planning/design/a4_design.md` sections 4, 5.6; `a1_design.md` sections 4, 5.4; verdicts from `verdict_synthetic_complete.md` sections 6-7 and `r0a1_interference_verdict.md` | 3 |
| Tab 3 | ACT verdicts (D11) | Magnitude corridor, rank bars, null silence across variants and profiles, the thin-density exception row (G13-G15) | Campaign JSONs (act arms); `kt-mirt/_planning/verdict_synthetic_complete.md` sections 4-5; `verdict_kdd_g2.md` section 3 | 5.3 |
| Tab 4 | Neural tracker faithfulness battery | PAS-N1 CG7-CG10 cell-by-cell fails on both beds, the single borderline cell marked; PAS-N2 row rendered "no measured verdict" per P2 | `kt-mirt/outputs/a4/campaign/*/syn_*/neural_modelseed{0,1,2}.json`; `verdict_synthetic_complete.md` section 4; `verdict_kdd_g2.md` section 4 | 5.4 |
| Tab 5 | THE LICENSING MAP (centerpiece) | Claim tiers x data regimes, four cell states, two-layer legend, map IDs in cells, scale arithmetic and prescriptive rules in the legend | Assembled from `evidence/claim_evidence_map.md` IDs per cell; roll-ups in `verdict_synthetic_complete.md` sections 6-7, `verdict_kdd_g2.md` section 6, `r0a1_interference_verdict.md` sections 4-5, `LEDGER.md` 2026-07-23 | 7 |
| Tab A1 | Full registry incl. unexecuted battery (D15) | Full version of Tab 2 plus the pre-registered A1 arms (CT1 anti-gaming, CT3-i..v, CT4, CT5, CT7, CT8/CT9) with re-baselined 0.04 bars and status "pre-registered, unexecuted", the three clean-negative triggers, RB tier definitions | `kt-mirt/_planning/design/a1_design.md` sections 4-5; `a4_design.md` sections 4-5, 5.2; re-baselining note in `r0a1_interference_verdict.md` section 5 | App. A (referenced from 6.4) |

---

## Abstract skeleton (5 sentences, one clause per contribution) (D17)

1. [Disease] Deep knowledge-tracing models are read as measurement
   instruments, their latent states published as growth curves and
   cross-skill influence graphs, yet the field validates these readouts
   by prediction accuracy, which cannot certify them.
2. [C1, protocol] We present a certification protocol that decides,
   before any readout is read, whether the estimator recovers known
   truth: matched synthetic twins calibrated to named real datasets,
   pre-registered bars with frozen kill conditions, permutation nulls,
   designed confound arms, and end-to-end seed discipline, with refusals
   published as first-class results.
3. [C2, completed cycle] One cycle completes end to end: a passive
   growth-existence detector is certified at cohort grain on twin
   profiles at both density extremes, its one diagnosed false fire
   repaired with a saturation-aware null, and the certified instrument
   then fires on real KDD Algebra (unsaturated 257 of 515 skills, one
   seed, one cohort split, p=0.01) while staying correctly silent on
   thin-practice Junyi, mapping a data-density boundary of the method.
4. [C3, mid-cycle] A second instantiation, caught mid-cycle and reported
   that way, yields the first certified signed cross-skill readout,
   demoted by its own controls to per-edge signed dose-association on
   the synthetic harness at a named operating point, with a measured
   negative-edge detectable-dose floor of 0.04, an order-shuffle control
   refuting the temporal reading, and the causal reading left unearned
   pending pre-registered scheduling controls.
5. [C4, map] The deliverable is a licensing map labeling every readout
   cell certified, confirmed, refused with mechanism, or pending, with
   the refusals (per-skill resolution at both density extremes,
   magnitude corridors, the field-default neural tracker's faithfulness
   battery) carrying their mechanisms, so the paper states with ground
   truth which interpretive sentences these readouts license and exactly
   where they stop.

---

## Reconciliation decisions

Each divergence between the evidence-first (EF) and narrative-first (NF)
outlines, the pick, and one line of reasoning. Neither outline was
averaged; every pick is the stronger choice under the framing memo.

- D1 Macro-architecture: NF. The design-before-results order plus real
  data inside the growth section encodes the mandated completed-cycle /
  mid-cycle asymmetry structurally; EF's standalone real-data section
  cuts the certify-then-confirm cycle in two.
- D2 Section titles: NF. "A completed cycle" and "Caught mid-cycle" put
  the architecture sentence on the section heads, exactly where the
  split editor demanded the asymmetry live.
- D3 Related work: NF's four strands. The dedicated change-measurement
  plus Kane strand gives threats 13 and 19 named space instead of
  folding them into other strands (EF's three-strands-plus-paragraphs).
- D4 G22 placement: NF (section 4, instruments), not EF (section 3,
  protocol). The stationarity lesson is a property of the instrument
  under audit and must precede results to defuse the circularity threat
  where the instrument is introduced.
- D5 Bed and association-bed facts (triage table, S22, S24, S25, S26):
  NF (section 4), not EF (sections 5-6). Moving all real-bed matter
  ahead of section 6 thins real-bed names inside the association
  section, hardening the threat-2 scope wall.
- D6 M9 top-K hazard: NF (section 4), re-promoted in discussion. It
  shapes every later pair selection, so it must precede any top-K read
  rather than arrive with real-data results (EF section 5).
- D7 Use argument: NF (delivered in 6.4). The usefulness defense sits
  beside the demoted claim it defends; EF's discussion delivery arrives
  two sections after the challenge. Kane lineage stays in related work,
  discussion keeps a one-line pointer.
- D8 Union material split: the R2+S7 side-by-side sentence goes to the
  map section (NF), the full union-versus-split defense to the
  discussion (EF). The sentence needs the cells next to it; the split
  counterfactual is discussion material. Each element has exactly one
  home.
- D9 Map columns: EF (fifth column = multi-tag density regime), not
  NF's "real association beds". A real-association-bed column would
  create cells the hard scope wall forbids; multi-tag is the harness
  regime where S7 actually lives.
- D10 Map pending cells: EF's list, which includes P5. The
  causal-reading row needs its pending cell or rule 9b loses its map
  home. S10 added to the refused list (both outlines under-listed it).
- D11 ACT display: EF (dedicated table), not NF (ACT folded into
  Fig 4). The corridor/rank/null grid across variants and profiles is
  inherently tabular; folding it in overloads the paper's sharpest
  refusal figure.
- D12 Fig 7: EF's three-panel version, with S8 added to the density
  panel. NF drops the L1-ladder panel and leaves the S9/S10 kill with
  no visual; EF alone omits bank sensitivity, the second measured
  floor-mover, so it joins panel (c).
- D13 Fig 2: EF's G3 inset kept. The raw-stat density artifact needs a
  visual anchor or G3 stays text-only against a figure that would
  otherwise look cleaner than the claim allows.
- D14 Fig 5: NF's risk note kept (breadth clause printed on the panel).
  The mandated exact-breadth language must travel with the image
  wherever it is reused, not only in the caption.
- D15 Registry vs unexecuted-battery table: NF's registry (Tab 2,
  compact main text; full in Appendix A) adopted, EF's Table 5 absorbed
  into the Appendix A full registry. One registry with a verdict column
  covers executed and unexecuted arms; a standalone unexecuted table
  duplicates rows. The 0.04 re-baselining note and P6 triggers survive
  as registry rows.
- D16 Appendices: union of both sets, keeping NF's Appendix D (junyi15
  graph pilot). It is appendix-scoped by a recorded PLAN decision; EF
  silently dropped a documented decision.
- D17 Abstract skeleton: NF base, with EF's end-to-end seed-discipline
  clause restored in sentence 2. NF carries the mandated architecture
  wording more exactly; seed discipline is load-bearing protocol
  content the abstract should name.
- D18 Scale arithmetic: delivered once where S12 is a result (6.3) and
  repeated compactly in the map legend. Threat 18 mandates the legend
  copy; stating the full arithmetic twice would pad.
- D19 G23 roll-up position: NF (opens 5.5). The synthetic roll-up
  licenses the real-data step in the same breath, which is the
  certify-then-confirm hinge.
