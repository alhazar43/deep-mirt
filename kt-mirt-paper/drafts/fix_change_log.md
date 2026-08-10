# Fix change log: draft_v0.md -> draft_v1.md

Ground-truth fix pass, 2026-08-11. Sources consulted: `evidence/claim_evidence_map.md` (the map), `related-work/citation_fixes.md`, `kt-mirt/_planning/verdict_kdd_g2.md`, `kt-mirt/_planning/verdict_synthetic_complete.md`, `kt-mirt/_planning/ct0_power_result.md`, `kt-mirt/_planning/triage/triage_report.md`, `kt-mirt/_planning/LEDGER.md` (entries 2026-07-21 and 2026-07-23), `kt-mirt/_planning/design/a4_design.md` (twin definitions). Rule set: minimum necessary change, no claim-strength changes, no added content, claim IDs kept, style rules kept.

## Contradiction resolutions

| # | Location | Before | After | Ground truth |
|---|---|---|---|---|
| 1 | Sec 4, beds paragraph | "with growth verdicts running on the unsaturated 48.3 percent of KCs [R14]" | "with growth verdicts running on the unsaturated 257 of 515 KCs [R14][R1]" | LEDGER 2026-07-23: "bed_stat 6114 on 257/515 unsaturated real KTracedSkills KCs"; map R1 ("258/515 KCs excluded as near-mastered"). 48.3% was the design-time 0.85-flag estimate (a4_design 6-7, map R14); the executed verdict number is 257 of 515, which Abstract/1/5.5/7/9 already carried. [R1] added because the executed count is R1's fact. |
| 2 | Sec 5.2, repair sentence | "It eliminates the false fire, moving the saturated twin from p = 0.01 to 1.00, while detection on both growth twins holds at p = 0.01 and the no-growth null stays clean [G8]." | "It eliminates the false fire in a verification rerun at B=99 replicates, whose smallest attainable p-value is 0.01, moving the saturated twin from p = 0.01 to 1.00, while detection on both growth twins holds at the B=99 floor of p = 0.01 and the no-growth null stays clean [G8]." | Certification battery ran at B=999 (floor 0.001; verdict_synthetic_complete Sec 1, "p=0.001 = 1/(999+1)"), stated in 5.1. The saturation-fix verification (verify_sat_fix.py, LEDGER 2026-07-21) reports p 0.01 -> 1.00 with kg/ns held at 0.01, the B=99 floor; LEDGER states the B=99 <-> p-floor-0.01 equivalence explicitly (2026-07-23 entry). Each p now sits next to its B. |
| 3 | Sec 7, refused cells | "The field-representative shared-state tracker fails the faithfulness battery on both beds at production scale [G17]." | "...on both twin profiles at production scale [G17]." | Map G17 scope note ("Scope: synthetic twins"); verdict_synthetic_complete Sec 6 ("nothing here is a real-data result"). Sec 5.4 already said "profiles"; "beds" is reserved for real datasets. |
| 4 | Sec 5.5, silence paragraph | "the bed sits at a median of about 8 opportunities per student against KDD's roughly 2,688 interactions per student [R2]" | "on mean rows per student, the ladder quantity of Figure 5, the silent bed sits at 105 against firing KDD's roughly 2,688, while its median practice, a different statistic kept off that axis, is about 8 opportunities per student [R2]" | Map P1: rows/student bracket 105 (silent Junyi) -> 557 (deep cohort) -> 2,688 (KDD). The v0 sentence compared Junyi's median-opportunities statistic with KDD's mean rows per student, mixing units; the ladder quantity is mean rows per student, and median opportunities (R2, ~8) is kept but labeled as a separate statistic off the axis. |
| 5 | Sec 5.5, same paragraph | "Figure 5 presents the pair on a rows-per-student axis" | "Figure 5 presents the pair on the mean-rows-per-student axis" | Same as #4; the figure axis is the mean-rows-per-student ladder. |
| 6 | Sec 5.5, pending cell | "on the rows-per-student axis (105 for silent Junyi, 557 for the deep cohort, 2,688 for firing KDD)" | "on the mean-rows-per-student axis (105 for silent Junyi, 557 for the deep cohort, 2,688 for firing KDD)" | Same as #4 (map P1 bracket unchanged). |
| 7 | Sec 6.3, bank mover | "perturbed to the bank's measured recovery floor (rank correlation about 0.759, Section 4)" | "perturbed to the bank's measured recovery floor (rank correlation 0.759, the probe point within Section 4's measured 0.70 to 0.80 band)" | ct0_power_result bank-perturbed table header ("difficulty rank_corr ~0.759, the A4 floor"); map S8. The 0.70-0.80 range stays where the range is the claim (Sec 4 and 5.3, map G12); the point value stays where the perturbation experiment is described, now tied to the band so the two no longer read as conflicting numbers. |
| 8 | Sec 4, beds paragraph opening | "Nine real beds were triaged through one shared metric library (Table 1)." | "Nine real beds, KDD Algebra, KDD Bridge to Algebra, EdNet KT1, Junyi 2020, junyi15, Eedi, XES3G5M, TIMSS, and Duolingo SLAM, were triaged through one shared metric library (Table 1)." | triage_report.md closing note (the nine-bed roster). v0 named seven plus Bridge in passing and omitted XES3G5M. |
| 9 | Abstract | "p=0.01" (KDD fire) | "p=0.01 at 99 replicates" | Same B-statement rule as #2; LEDGER 2026-07-23 (B=99). Sec 1 and Sec 7 already stated B=99. |
| 10 | Sec 9 | "at one seed and one cohort split (p=0.01)" | "at one seed and one cohort split (B=99, p=0.01)" | Same as #9. |

## Term fixes: skills vs KCs (rule: KCs for the growth beds and for the D=3 association object, since the map and ct0_power_result use cross-KC there; the field's own vocabulary is retained when describing the field's readouts)

| # | Location | Before | After |
|---|---|---|---|
| 11 | Abstract | "unsaturated 257 of 515 skills" | "unsaturated 257 of 515 knowledge components (KCs)" (first-use expansion; abstract self-contained) |
| 12 | Abstract | "the first certified signed cross-skill readout" | "the first certified signed cross-KC readout" |
| 13 | Abstract | "per-skill resolution at both density extremes" | "per-KC resolution at both density extremes" (map G9) |
| 14 | Sec 1, companion-paper paragraph | "person-side dynamics and multi-skill structure" | "person-side dynamics and structure across multiple knowledge components (KCs)" (first body expansion; matches Sec 8's "multi-KC structure") |
| 15 | Sec 1, contribution 2 | "on the unsaturated skill subset (257 of 515)" | "on the unsaturated KC subset (257 of 515)" |
| 16 | Sec 1, contribution 3 lead | "The first certified signed cross-skill readout" | "The first certified signed cross-KC readout" |
| 17 | Sec 1, contribution 3 | "at D=3 skills, N=500 learners" | "at D=3 KCs, N=500 learners" (ct0_power_result uses KC throughout at D=3) |
| 18 | Sec 1, contribution 4 | "Per-skill growth resolution fails" | "Per-KC growth resolution fails" (map G9) |
| 19 | Sec 1, contribution 4 | "growth magnitude and per-skill ranking fail their corridors" | "growth magnitude and per-KC ranking fail their corridors" (map G13) |
| 20 | Sec 2.2 | "refuses per-skill resolution" | "refuses per-KC resolution" |
| 21 | Sec 3, anchoring paragraph | "per-skill rates are spread twenty-fold" | "per-KC rates are spread twenty-fold" (map M12) |
| 22 | Sec 3, transfer-edges sentence | "placed only between skill pairs sharing no co-tagged item" | "placed only between KC pairs sharing no co-tagged item" (map M15) |
| 23 | Sec 4, Junyi 2020 sentence | "structurally barring cross-skill work under its shipped KC scheme" | "structurally barring cross-KC work under its shipped KC scheme" |
| 24 | Sec 5.5, Junyi silence | "0 of 40 per-skill discoveries" | "0 of 40 per-KC discoveries" (map R2 count unchanged) |
| 25 | Sec 6 title | "Signed cross-skill association: caught mid-cycle" | "Signed cross-KC association: caught mid-cycle" |
| 26 | Sec 6 opening | "The signed cross-skill readout is the protocol's second instantiation" | "The signed cross-KC readout is..." |
| 27 | Sec 6.1 | "plants known cross-skill edges among three skills (D=3)" and "whether the fitted cross-skill matrix recovers" | "plants known cross-KC edges among three KCs (D=3)" and "whether the fitted cross-KC matrix recovers" |
| 28 | Sec 6.1 | "(one skill per item, the KDD-shaped density)" | "(one KC per item, the KDD-shaped density)" |
| 29 | Sec 6.3 | "items tag 2.2 skills on average" | "items tag 2.2 KCs on average" (map M2: tag arity 2.2 over KCs) |
| 30 | Sec 6.3 | "cross-skill transfer collinear with a skill's own gain" | "cross-KC transfer collinear with a KC's own gain" (map S7) |
| 31 | Sec 6.3 | "At K=515 skills there are roughly 265,000 ordered pairs" | "At K=515 KCs there are roughly 265,000 ordered pairs" |
| 32 | Sec 6.3 | "never on the cross-skill matrix" | "never on the cross-KC matrix" |
| 33 | Sec 6.4 | "Re-drawing each learner's cross-skill interleaving" | "Re-drawing each learner's cross-KC interleaving" (map S14) |
| 34 | Sec 6.4 | "practicing a source skill raises or lowers a target skill" | "practicing a source KC raises or lowers a target KC" (map P5) |
| 35 | Sec 6.4 | "reference edges connecting only skill pairs that share no co-tagged item" | "...only KC pairs that share no co-tagged item" (map M15) |
| 36 | Sec 6.4, licensed sentence | "the fitted cross-skill matrix licenses per-edge signed dose-association" | "the fitted cross-KC matrix licenses..." (map S16) |
| 37 | Sec 6.4 | "that conjunction, the first certified signed cross-skill readout, is the claim" | "...the first certified signed cross-KC readout, is the claim" (map S21) |
| 38 | Sec 7 | "to signed cross-skill association, its temporal and causal readings" | "to signed cross-KC association, ..." |
| 39 | Sec 8 | "a synthetic-only pilot at three skills" | "a synthetic-only pilot at three KCs" |
| 40 | Sec 9 | "unsaturated 257 of 515 skills" | "unsaturated 257 of 515 KCs" |
| 41 | Sec 9 | "per-skill resolution at both density extremes" | "per-KC resolution at both density extremes" |

Deliberately NOT changed (field vocabulary describing the field's own objects): "per-skill mastery" and "cross-skill influence graphs" (Sec 1 and Abstract, DKT-literature readouts), "multi-skill KT models" (contribution 1), "single-skill KT models" (companion audit), Sec 2.1/2.2/6.4 descriptions of HawkesKT/LTKT/LFA/P(J)/Khajah ("between skills", "skill-to-skill", "per-skill learning-rate slopes", "per-skill tracking", "learned a skill"), and Sec 5.3's "Per-skill non-identifiability has been on record since Beck & Chang (2007)" (the ancestors' claim in their terms).

## Term fixes: other

| # | Location | Before | After | Ground truth |
|---|---|---|---|---|
| 42 | Sec 4, twins paragraph | "two growth twins of different shape" | "two growth twins of different shape (kg, known growth of the generator's standard shape; ns, non-standard-shape growth that the fitted family misfits)" | a4_design.md: SYN-KG "known-growth twin ... the standard"; SYN-NS "non-standard-shape twin ... growth exists but the bounded-exponential family is wrong". Codes kg/ns previously first appeared unexpanded in 5.1. |
| 43 | Sec 4 | Decoupling defined in the KC-model paragraph, after "Eedi leads decoupling at 0.967" and "0.800 against 0.267" had already appeared | Definition sentence ("Decoupling, the triage's pair-separation metric, is the fraction of co-occurring KC-pair slots in which exactly one member is observed, with a 0.75 admission bar (detail in Appendix B) [M13]") moved up to directly after the bed-roster sentence, before any decoupling number; removed from its old position | Term rule: define before any number. Map M13 wording preserved; [M13] travels with the sentence. |
| 44 | Sec 4, KC-model paragraph | "Against that bar EdNet at 0.87..." | "Against the 0.75 bar EdNet at 0.87..." | Follow-on from #43 (the referent sentence moved); map S25 bar value. |
| 45 | Sec 5.1 | "The passive existence gate asks one question of a bed" | "...asks one question of a dataset" | "Beds" reserved for real datasets; in 5.1 the gate is being run on synthetic twins. Harness proper names ("bed statistic", "bed-level decision", "bed null") are kept as machinery names. |
| 46 | Sec 1, contribution 3 | "a measured negative-edge detectable-dose floor of 0.04, twice the field-anchored reference dose" | "...twice the field-anchored negative reference dose of 0.02" | Reference-dose disambiguation (0.05 positive, 0.02 negative); map S11 (0.04 = 2x the 0.02 negative dose). |
| 47 | Sec 3, anchoring paragraph | "Growth magnitudes and the reference transfer dose are set from..., and a dose-response sweep at 0.5, 1, 2, and 4 times the reference dose is pre-registered" | "Growth magnitudes and the two reference transfer doses, +0.05 facilitation and -0.02 interference, are set from..., and a dose-response sweep at 0.5, 1, 2, and 4 times the reference doses is pre-registered" | ct0_power_result (g_pos=+0.05, g_neg=-0.02); map M12. Names which dose every time the term appears. |
| 48 | Sec 6.1 | "The negative half has no stable threshold at the reference dose under the default trainer" | "...no stable threshold at the 0.02 negative reference dose under the default trainer" | Map S5; disambiguation rule. |
| 49 | Sec 6.3 | "|g| = 0.04, twice the field-anchored reference dose" | "|g| = 0.04, twice the field-anchored negative reference dose of 0.02" | Map S11; disambiguation rule. |

## Figure 7 panel order

| # | Location | Before | After | Note |
|---|---|---|---|---|
| 50 | Sec 6.3, movers paragraph | "(Figure 7c)" | "(Figure 7b)" | First references now run 7a (6.2, L1 ladder), 7b (6.3, movers), 7c (6.4, shuffle). |
| 51 | Sec 6.4, shuffle paragraph | "(Figure 7b)" | "(Figure 7c)" | Pair of #50. ACTION for the figure asset: panels b and c must swap letters in the artwork (movers panel becomes b, order-shuffle panel becomes c). |

## Citation fixes (per related-work/citation_fixes.md)

| # | Location | Before | After | Basis |
|---|---|---|---|---|
| 52 | Sec 2.2 | "and ceiling effects and wheel-spinning bound what near-mastered data can reveal (Beck and Gong 2013)" | "and wheel-spinning, extended practice that never reaches mastery, bounds what near-mastered data can reveal (Beck and Gong 2013)" | Fixes item 9: Beck & Gong 2013 supports wheel-spinning only, not ceiling effects; sentence reworked so the citation supports exactly what it says. |
| 53 | Sec 5.2 | "Ceiling effects are classical measurement territory (Beck & Gong, 2013); what the harness adds..." | "The limits of near-mastered data are classical territory, with wheel-spinning its sharpest prior name (Beck & Gong, 2013); what the harness adds..." | Fixes item 9: the ceiling-effects attribution was unsupported as written; reworked to the wheel-spinning/non-mastery claim the paper does support. |
| 54 | Sec 2.4, opening sentence | "a formal tradition of structured models for growth and a long-running debate over the reliability of difference scores" (no citations) | "a formal tradition of structured models for growth (Andersen, 1985; Embretson, 1991; Fischer, 1995; McArdle, 2009) and a long-running debate over the reliability of difference scores, opened by the classic pessimism of Cronbach and Furby (1970) and answered by Rogosa and Willett (1983), who showed difference scores are often highly reliable" | Fixes items 1-6. Fischer (1995) chosen over Fischer (1973) per item 3, since the clause is about change/growth models specifically. Items 5-6 note the debate is accurately cited only with both sides' actual positions: Cronbach & Furby as the pessimistic classic, Rogosa & Willett as the rejoinder showing difference scores are often highly reliable (never as evidence they are unreliable). |
| 55 | Sec 2.4 | "Kane's interpretation and use arguments treat validation as an explicit chain of inferences" | "Kane's interpretation and use arguments treat validation as an explicit chain of inferences (Kane, 2013)" | Fixes item 7 (corrected in-text form). |
| 56 | Sec 2.4 | "in the register of Mayo's severe testing" | "in the register of Mayo's severe testing (Mayo, 2018)" | Fixes item 8. |
| 57 | Sec 6.4 | "in the spirit of Kane's argument-based validation, in which..." | "in the spirit of Kane's argument-based validation (Kane, 2013), in which..." | Fixes item 7 (second named Kane sentence). |

Not changed: Khajah et al. 2016 phrasing (item 11 verdict "ok"; the softer fallback wording is noted for a rebuttal but not required); Beck & Chang 2007 (item 10, ok); all other citations were verified as described and left untouched.

## S23 check

S23 exists in the claim map (junyi15 prerequisite graph: real and denser than advertised, 835 nodes / 981 edges, but NOT a DAG, with 212-218 nodes in genuine cycles plus 2 self-loops, so downstream use must handle cycles explicitly; source triage_report.md and LEDGER 2026-07-18). Draft v0 never cites S23 and no record marks it deliberately dropped, so it is an OMISSION, not a curation decision. Its natural home is the planned junyi15 external-corroboration sentences (Sec 6.4 [P4] and Sec 8), which currently describe the graph check without the cycle caveat; map conflict note 9 makes the same point. Per the no-added-content rule, S23 was NOT inserted into draft_v1; flagged here for the author.

## Contradiction-source notes

- Contradiction 1 (257 of 515): verdict_kdd_g2.md itself carries no real-data count (it is the synthetic KDD-profile verdict); the recorded verdict number lives in LEDGER 2026-07-23 ("257/515 unsaturated real KTracedSkills KCs") and map R1 ("258/515 excluded"), which agree. 48.3% (map R14, a4_design) is the design-time 0.85-flag estimate, a different operationalization; it no longer appears in the draft.
- Contradiction 2 (B per battery): verdict_synthetic_complete.md fixes the certification battery at B=999 (floor 0.001). The repair figures (p 0.01 -> 1.00; kg/ns held at 0.01) are the verify_sat_fix.py rerun (LEDGER 2026-07-21), whose 0.01 values are the B=99 floor; the B=99 <-> 0.01-floor equivalence is explicit in LEDGER 2026-07-23. Both Bs are now stated where their p-values appear.

Total changes: 57.
