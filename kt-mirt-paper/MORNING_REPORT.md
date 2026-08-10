# Morning report (2026-08-11)

Overnight pipeline output for the Paper 2 drafting campaign. Everything
below lives under `C:/Users/steph/documents/deep-mirt/kt-mirt-paper/`.
This report supersedes the earlier assembly that ran before the review
outputs existed. The review stage did run. Its findings (33, from three
reviewers), the revised draft, and the disposition are all in the tree.

## What was produced tonight

Planning and ledger.
- `PLAN.md`, the frozen plan of record with the author's four decisions.
- `PLAN_OF_PLAN.md`, the approved proposal.
- `LOG.md`, the stage ledger. Note that it ends at the Stage 2 launch
  and is stale relative to everything below.

Evidence (Stage 1).
- `evidence/claim_evidence_map.md`, 109 consolidated claims with
  strength tags, the sole source of claim strength.
- `evidence/paper1_continuity.md`, the companion-paper continuity brief.

Related work (Stage 1).
- `related-work/named_analogs.md`
- `related-work/newcomers_2025_2026.md`
- `related-work/originality_threats.md`
- `related-work/citation_fixes.md`, eleven citations verified against
  publisher pages, none fabricated, three usage corrections.

Framing and outline (Stage 2).
- `outline/claim_language_constraints.md`, the binding claim-language
  rules from the lit sweep.
- `outline/framing_memo.md`, the consolidated judgment over five
  persona attacks and two full-text fact-checks. It carries the frozen
  framing paragraph, the contribution list, the claim-language card,
  the 28-row threat-response table, and the union defense.
- `outline/outline_narrative_first.md` and
  `outline/outline_evidence_first.md`, the dual outlines.
- `outline/paper_outline.md`, the reconciled outline.

Drafts (Stage 3).
- `drafts/01_introduction.md` through `drafts/09_conclusion_abstract.md`,
  the nine section drafts.
- `drafts/draft_v0.md`, the assembled full draft.
- `drafts/fix_change_log.md`, the ground-truth fix pass, 57 changes,
  every one sourced to a verdict file or the claim map.
- `drafts/draft_v1.md`, the post-fix draft that went to review.

Review and revision (Stage 4).
- The review ran three reviewers (claims-audit, hostile-reviewer,
  register-audit) and returned 33 findings across all severities.
- `review/revision_disposition.md`, the finding-by-finding disposition.
  25 applied, 7 applied in part, 1 skipped, with the skip reasoned.
- `drafts/draft_v2.md`, the current draft of record. All must-fix
  findings applied or applied as far as prose can carry them, the
  abstract rewritten as twelve plain sentences, an availability
  statement added, and the register pass done.

This package.
- `MORNING_REPORT.md`, this file.
- `PLAIN_SUMMARY.md`, the no-jargon version for a cold read.

## The frozen framing paragraph

Quoted verbatim from `outline/framing_memo.md`, section 1.

> Deep knowledge-tracing models are routinely read as measurement
> instruments. Their latent states are published as growth curves,
> per-skill mastery, and cross-skill influence graphs, and the nearest
> published neighbors (Deep-IRT, PSI-KT, HawkesKT, LTKT) validate those
> readouts by prediction accuracy alone. Prediction cannot certify a
> readout, and neither can raw learning curves, which are attrition-biased
> and blind to the false-fire mechanisms our harness isolates. This paper
> builds a certification protocol that asks, before any readout is read,
> whether the estimator recovers known truth. Its instruments are matched
> synthetic twins calibrated to named real datasets, pre-registered bars
> with frozen kill conditions, permutation nulls, designed confound arms,
> and seed discipline, with refusals published as first-class
> deliverables. The protocol runs end to end once and transfers once. The
> completed cycle is growth existence. A passive detector is certified at
> cohort grain on twin profiles at both density extremes, silent on every
> null twin and at the permutation floor on every growth twin, surviving a
> full density inversion. Its one diagnosed failure, a false fire on the
> saturated twin, is traced to reference-model misspecification and
> repaired with a saturation-aware null. The repaired detector fires on
> real KDD Algebra (one seed, one cohort split, on the unsaturated 257 of
> 515 skills) and stays correctly silent on thin-practice Junyi, mapping a
> data-density boundary of the method rather than an absence of learning.
> The second instantiation, signed cross-skill influence, is caught
> mid-cycle and reported that way. On the synthetic harness at a named
> operating point (D=3, N=500, single-tag density, calibrated bank) the
> readout earns exactly one licensed sentence, per-edge signed
> dose-association with a measured detectable-dose floor of 0.04, twice
> the field-anchored reference dose, and the paper itself shows that floor
> move under density and bank error. An order-shuffle negative control
> refutes the temporal reading. The causal reading is left unearned
> pending a pre-registered endogenous-scheduling control, and the
> remaining confound battery is pre-registered and unexecuted. Around
> these two arms the protocol mostly refuses, with the mechanism attached
> to every refusal. Per-skill growth resolution fails at both density
> extremes with the same signature. Growth magnitude and per-skill ranking
> fail their corridors. The field-representative shared-state neural
> tracker fails the entire faithfulness battery at production scale. The
> association sign collapses outright at multi-tag density. The
> deliverable is the resulting licensing map, every cell labeled certified
> on the synthetic harness, confirmed on real data, refused with
> mechanism, or pending pre-registered, plus the harness to re-check it.
> The paper moves no AUC and never claims to. It states, with ground truth
> rather than opinion, which interpretive sentences these readouts license
> and exactly where they stop.

Note one delta the review forced after the freeze. The draft no longer
asserts the density boundary as a finding. Every occurrence now reads
as a cross-bed contrast consistent with a density account, with the
deciding cell pending (decision 4 below).

## The five most consequential pipeline decisions

1. **Framing. Protocol, not framework, and novelty as a three-way
   conjunction.** Every ingredient's lineage (simulation-based
   calibration, randomization sanity checks, permutation nulls and
   negative controls, minimum-detectable-effect floors) is cited
   preemptively; the claimed novelty is the assembly plus the
   matched-twin device. The HawkesKT full-text read came back SIGNED
   and the LTKT read SIGNED but STIPULATED, so all signedness-priority
   wording is permanently dead. The association claim now rests
   entirely on the conjunction. Recovered from the fitted model,
   certified against ground truth, and carrying a measured
   detectable-dose floor. Always "to our knowledge the first
   certified", never "first".

2. **Framing. The two-pillar symmetry was banned.** The phrase "two
   certified demonstrations" is dead everywhere. The architecture
   sentence is now one completed certify-then-confirm cycle (growth)
   plus the protocol's second instantiation caught mid-cycle
   (association), stated wherever the pair appears. This was the split
   editor's fatal objection and also the condition under which that
   persona conceded the union.

3. **Adjudication. The S11/S12/S13 strength-tag conflict.** The
   campaign's records disagreed on whether the dose floor, the
   false-edge background, and the phantom control are certified or
   synthetic-only. The stage gate adjudicated the certified tag upheld
   on procedural grounds (rigor and data realm are separate axes). All
   prose adopts the conservative composite, certified on the synthetic
   harness, and the review then forced Section 3's weaker-reading rule
   to name this as its one disclosed exception rather than claim a rule
   it breaks.

4. **Demotion. The density boundary and the causal reading.** The
   hostile review flagged the boundary as causally unsupported at n=2
   beds with the deciding cell unexecuted. Every occurrence (abstract
   through conclusion) was rewritten to a cross-bed contrast consistent
   with a data-density account, deep-Junyi pending. Separately, rule 9
   was split. The order-shuffle control refutes only the temporal
   reading; the causal reading is unearned, not refuted, pending the
   endogenous-scheduler control. And the association arm sits behind a
   hard real-data scope wall with the operating point inline at every
   floor mention.

5. **Demotion. The calibrated-bank condition declared idealized.** The
   certified 0.04 floor assumes a calibrated bank the program's own
   bank does not reach (0.70 to 0.80 against the 0.9 bar). The abstract
   and every headline claim now say so, with the bank-error probe named
   as the measured face of the gap. The one skipped review item is
   adjacent. Relabeling to "certified within-family" was judged a
   claim-strength change outside the revision's mandate and is left to
   you (author-decision list, and open item 7).

## AUTHOR-DECISION items from review, verbatim

Twelve findings carried severity author-decision. The revision pass
applied none of them by policy. Each is quoted verbatim (location,
issue, suggestion), with a one-line status in italics.

**1. claims-audit.** Location: "drafts/draft_v1.md line 55 (Section 3) vs footnote at line 181"
> Issue: "Section 3 promises 'Where the campaign's own records disagreed on the strength of a verdict, the weaker reading governs every sentence here', but for S11/S12/S13 the draft adopts 'certified on the synthetic harness', while the map's own never-upgrade rule (conflicts 1-2) kept the weaker synthetic-only tag. The footnote discloses the conflict and leans on the 2026-08-11 adjudication, but calling the composite 'conservative' is a judgment, and the blanket sentence in Section 3 is now not literally true."
> Suggestion: "Either carve out the adjudicated cells in the Section 3 sentence ('the weaker reading governs, except where a stage-gate adjudication resolved the conflict, footnoted where it does'), or soften the three prose uses to 'measured under pre-registered bars on the synthetic harness'."

*Status. The carve-out branch was in effect applied via a should-fix
(disposition 16); the deeper choice, whether the certified tag stands
at all, remains yours.*

**2. claims-audit.** Location: "drafts/draft_v1.md line 9 (Section 1)"
> Issue: "The sentence asserts that Deep-IRT and PSI-KT 'validate those readouts by prediction accuracy alone, never by testing whether the estimator recovers known structure' under [S21], but the S21 row documents only LTKT and HawkesKT validation practice plus the open-claims sweep verdict. The universal 'never' for the other two named works exceeds the map row."
> Suggestion: "Confirm the Deep-IRT/PSI-KT characterization against the P1 sweep notes and add it to the S21 row, or scope the sentence to what the sweep documents."

*Status. Open. The sentence survives in draft_v2 Section 1.*

**3. claims-audit.** Location: "drafts/draft_v1.md line 37 (Section 2.1)"
> Issue: "Two related-work specifics exceed the S21 row: the LTKT leakage claim ('computed over training, validation, and test data jointly, a leakage the paper itself states') and the HawkesKT mechanism claim ('the negative direction arises when a wrong answer on one skill depresses the prediction on another'). Both are plausible from ltkt_read.md / hawkeskt_read.md (listed as S21 sources) but neither detail is in the map, so the audit chain cannot verify them. Line 175 repeats the HawkesKT detail."
> Suggestion: "Verify both details against the read notes and fold them into the S21 row (or a new row) so the claims are auditable; the leakage accusation in particular should be checkable against the LTKT paper before print."

*Status. Open. Both details survive in draft_v2 and both are stated in
the framing memo's fact-check verdicts, but the map row was never
extended, so the audit chain still cannot verify them.*

**4. hostile-reviewer.** Location: "Sec. 6 opening ... and Sec. 6.4"
> Issue: "Half the paper is an interrupted experiment. The battery is frozen, priced, and re-baselined, which means the obstacle to running it is time, not design. Section 8's argument that the two arms must be published together (\"Split, the parts are worth less\") is a rhetorical case for publishing preliminary work, not an empirical one. A serious venue can reasonably ask why the paper was not submitted one battery later."
> Suggestion: "Execute the re-baselined battery (and ideally the endogenous-scheduler control) before resubmission; the paper's own framing says every arm of it is ready to run."

*Status. Open. An experiment and a submission-timing call.*

**5. hostile-reviewer.** Location: "Sec. 3 [M10] and Sec. 6.3 [S12]"
> Issue: "The paper mandates multiplicity control for any per-edge use while acknowledging its certified BH machinery does not cover the dependence structure of the edge matrix, leaving the mandated control uncertified for the object it is mandated on. This is stated honestly but leaves rule 3 of Sec. 7 without certified machinery behind it."
> Suggestion: "Certify BH/BY on the cross-KC matrix null in the re-baselined battery, and say in Sec. 7 that the rule currently names a requirement, not a certified procedure."

*Status. Open. Draft_v2's rule 3 states the scope; the certification
itself is an experiment.*

**6. hostile-reviewer.** Location: "Sec. 4: 'Its per-KC-factorized variant PAS-N2 makes the audited confusions impossible by construction, but that is a guarantee, not a measured verdict'"
> Issue: "Introducing PAS-N2 with an unfalsified construction guarantee and a single predictive probe adds an object the paper then repeatedly declines to make claims about; in a paper about refusing unmeasured claims, its presence mostly generates disclaimers."
> Suggestion: "Either run the faithfulness battery on PAS-N2 (the natural constructive counterpart to the PAS-N1 refusal) or cut it to a one-line pointer in future work."

*Status. Open. PAS-N2 keeps its current treatment in draft_v2.*

**7. register-audit.** Location: "L63, L73, L75 and elsewhere: 'posture'"
> Issue: "'Posture' as a name for a model configuration is invented vocabulary; the use-established-names rule prefers standard terms. It also collides with its ordinary-English sense in 'sanity-check posture' (L39)."
> Suggestion: "Consider 'variant' or 'configuration' ('the active variant (ACT)'). If 'posture' stays, define it once at first use and stop using the word in its ordinary sense elsewhere."

*Status. Open. Draft_v2 still uses posture.*

**8. register-audit.** Location: "L85, L125, L139 (section headings)"
> Issue: "Three headings carry colons ('The growth-existence readout: a completed cycle'; 'Real data: the fire, the silence, the boundary'; 'Signed cross-KC association: caught mid-cycle'). The colon ban targets running prose, so headings are arguably exempt, but the guide's spirit and the noun-phrase-title rule cut against them, and L125's triad subtitle is also a flourish."
> Suggestion: "'The growth-existence readout, a completed cycle'; 'Signed cross-KC association, caught mid-cycle'; L125 could become 'Real-data confirmation and the density boundary.'"

*Status. Partly mooted. 5.5 was renamed; the Section 5 and Section 6
headings still carry colons.*

**9. register-audit.** Location: "L5, L21, L129 (plus variants L133, L187, L222): 'mapping a (data-)density boundary of the method rather than an absence of learning'"
> Issue: "This full clause repeats verbatim four-plus times. The consistency rule licenses reusing a phrase for a concept, but sentence-length verbatim repetition across abstract, intro, results, and conclusion reads templated."
> Suggestion: "Keep the full formulation at L129 where the result lands and in the conclusion; compress elsewhere to 'a density boundary, not absent learning.'"

*Status. Mooted. The must-fix density-boundary rewrite eliminated the
repeated clause.*

**10. register-audit.** Location: "L5, L15, L25, L87: 'first-class results' / 'results of the same rank'"
> Issue: "'First-class' is a programming-language idiom used as a buzzword for 'published with equal standing.' Repeated four times."
> Suggestion: "'refusals published as results in their own right' or 'with the same standing as positive results'; vary across uses."

*Status. Largely handled. Kept once as the Contribution 4 title, varied
elsewhere; whether it survives there is yours.*

**11. register-audit.** Location: "L206 (Discussion): 'the gate can license, repair, and confirm; the association arm shows it can refuse, demote, and stop'"
> Issue: "Paired triads (also 'the fire, the silence, the boundary', L125; 'certified, confirmed, refused, or pending' is fine as the actual taxonomy). The mirrored three-verb constructions are a generated-reading rhythm."
> Suggestion: "Loosen one side: 'the growth arm shows the protocol can license and confirm a claim; the association arm shows it can stop one.'"

*Status. Largely handled in the register pass; the Section 8 sentence
still runs licensing, repairing, confirming against refusing, demoting,
stopping.*

**12. register-audit.** Location: "Abstract and throughout: bracketed evidence tags [G1][S21][M2] etc."
> Issue: "Internal evidence-ledger IDs sit in the abstract and prose. Presumably scaffolding for drafting, but if any survive to submission they are unexplained notation; and an abstract should carry none regardless."
> Suggestion: "Strip from the abstract now; decide the mapping convention (footnote, appendix key, or removal) for the body before the next draft."

*Status. Abstract stripped. The body convention (keep with a published
registry key, footnote, or remove) is yours before LaTeX.*

## Remaining open items

1. Unverified citations. The eleven classics in
   `related-work/citation_fixes.md` all check out against publisher
   pages, none fabricated, three usage corrections applied. The
   2025-26 entries (KeenKT, UKT, PAKT, KTCF, Kim and Kim 2026,
   Yamkovenko et al. 2025, Lee et al. 2025, Khalid, Deriyeva, and
   Paassen 2025, Yan, Tang, and Shimada 2026, Kuskova, Zaytsev, and
   Coppedge 2026) remain unverified; draft_v2's References placeholder
   names them. The verification pass covered the related-work and
   growth sections only, so citations elsewhere rest on the Stage 1
   dossier without a publisher-page pass.
2. The deep-Junyi growth cell (P1) is pending on the cluster, job id
   560630 (B=39, chunk=1, 72 hour limit). The draft carries both
   outcomes neutrally. On completion, resolve the pending sentences in
   the abstract, Sections 5.5 and 7, and the conclusion.
3. The external-alignment pilot (junyi15 prerequisite graph, full-K
   association read, P4) is deferred by author instruction. No compute
   now. It feeds an appendix or a scoped subsection at best and is
   gated on the unmeasured exercise-grain positivity bar and on
   re-certifying the decoupling admission bar (P7).
4. Paper 1's plan of record moved to docs/paper_revision_plan_v3.md
   mid-drafting; the continuity brief was built from plan v2 and the
   tex; a spot-check found no contradiction in the draft's Paper 1
   passages, but recheck continuity against v3 before LaTeX conversion.
5. The S23 flag from the fix change log. The junyi15 prerequisite
   graph is not a DAG, 212-218 nodes sit in genuine cycles, and both
   the draft's external-corroboration sentences and the deferred pilot
   must handle cycles. Draft_v2's P4 sentences still describe the graph
   check without the cycle caveat; inserting it is an author call under
   the no-added-content rule.
6. Production items the revision could not generate. Final title (a
   working title is in place), Figures 1-7, Tables 1-5 (Table 5 is the
   deliverable), Appendices A-D, and the compiled reference list.
   Also the Figure 7 panel swap flagged in the fix change log (movers
   panel becomes b, order-shuffle panel becomes c; the text already
   reads the new order).
7. Author flags from `review/revision_disposition.md`. The exact
   near-mastered masking criterion and whether to run the
   threshold-sensitivity and fresh-twin checks; the v1.0 to v1.1
   registration changelogs; whether to adopt "certified within-family"
   (the one skipped finding); operational definitions of the four
   faithfulness audits and "production scale"; the seed-pooling test
   specification; and a check of the verbal bed-statistic definition
   against the battery code.
8. The twelve author-decision review items quoted above, none applied
   by policy.
9. `LOG.md` is stale. It ends at the Stage 2 launch and records none
   of the outlines, drafts, review, or revision.

## Suggested reading order, 30 minutes

1. `outline/framing_memo.md`, section 1 only, the frozen paragraph.
   The paper in one page. 3 minutes.
2. `drafts/draft_v2.md`, abstract and Section 1. Check the twelve-
   sentence abstract and the four contribution clauses against what
   you approved. 7 minutes.
3. `drafts/draft_v2.md`, Section 7 and Section 5.5. The licensing map
   and the real-data pair, the two places the review hit hardest.
   6 minutes.
4. `review/revision_disposition.md`, the summary table and the
   flagged-for-the-author list at the end. What was changed on your
   behalf and what was deliberately not. 5 minutes.
5. The author-decision list in this report. Twelve calls, most one
   sentence each. The substantive ones are 1 through 6. 6 minutes.
6. The open items above. Three decisions are ripe now. The S23 cycle
   caveat, insert or hold. The "certified within-family" wording,
   adopt or decline. And whether the paper waits for the re-baselined
   battery or goes as an interrupted second cycle. 3 minutes.

`PLAIN_SUMMARY.md` beside this file gives the no-jargon version for a
cold read.
