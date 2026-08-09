# Results critique and venue fit (v2 program, 2026-08-09)

The redo of the overnight framing program under the corrected premise: the
results in the paper are the up-to-date object; the front matter is the
intended paper; the question is what the evidence actually supports, where
it should be sent, and what each option costs.

**Method.** Five expert boards ran as independent subagents, each ordered
to verify every claim against code and artifacts before writing:
B1 confounds-and-controls, B2 DL-novelty (ICLR standard), B3 psychometric
validity (measurement-journal standard), B4 evidence-chain, B5 statistical
rigor. Reports in `docs/framing_review/B1..B5_*.md`. My own exhibit chain
E1-E5 (`docs/framing_review/E*.md`) feeds in. I then spot-verified the
load-bearing claims myself; verification status is marked below. The v1
program (`docs/framing_review_caeai.md`, lenses A1-A7, F2+ framing) stands
as narrative diagnosis; this dossier is the science-level audit the author
asked for.

**Bottom line.** The core synthetic result is real and gets *stronger*
under correct statistics (paired t between 8 and 47). But the paper as
printed has three fatal defects, and the striking fact of this audit is
that almost everything needed to fix them is already banked on disk from
the frozen campaign. The missing storyline the criticism complained about
is also on disk: the paper builds a truth-free audit, then never runs it
on its own flagship real dataset, where it fires. One new experiment (the
already-built misspecification probe) is demanded independently by all
five boards.

---

## Part I. Consolidated defect ledger

Deduplicated across boards. [V] = I verified the claim myself this
session; [B] = verified by the board against named files; board IDs in
parentheses.

### FATAL

**F1. The TIMSS ordered-thresholds finding is a sort artifact, and the
classical calibration contradicts it.** [V] The GPCM head trains
unconstrained thresholds; every export path sorts them
(`kt-irt/src/deep_irt/core/decoders.py:419,430`; `core/model.py:541,633,
638,658,731`). The appendix table's all-yes Ordered column and the claims
at tex 1162/1168 are therefore true by construction. The classical MML
calibration of the same bank (`kt-irt/data/timss/timss_g8_usa_gpcm_coef.csv`)
has **12/31 items with b1 > b2** (I recounted). The campaign record already
called this an export artifact (R8, `docs/exposure_rerun_results.md`); the
demotion never reached the paper. No weights were saved for real-study
cells, so the honest number (fraction ordered on raw head output) needs a
retrain of the TIMSS cells. (B3 D1, B4 F1)

**F2. TIMSS is the paper's own disease, presented as reassurance.** [B]
The flagship real section reads as "the stable side of the story" while
the paper's own pre-registered diagnostic flags it: truth-free refit
discrepancy .68-.87 against the certified flag threshold tau = .152, MML
concordance .32-.34, while SH and SK agree with each other at .976, and
the neural slope range [.23,.64] vs classical [.58,1.68] shows the shared
compression. High mutual agreement, low external concordance, discrepancy
five times threshold: the exact stable-and-wrong signature the synthetic
phase defines, occurring in the flagship, framed as support. All numbers
are already on disk (`kt-irt/results/p2_realstudy/realstudy_table_*.md`,
`p2_v3_analysis/timss_disc_check.md`). This defect is also the storyline
opportunity; see Part III. (B3 D2)

**F3. The headline table is not yet attributable to the claimed
mechanism, and the controls that would fix it are banked but unprinted.**
[B: B1 counted from code] SK differs from SH in separation AND readout
capacity: 21,125 vs 8,101 parameters at the GPCM-LSTM reference cell,
2.6x. The paper's hyperparameter table names a key-16 control and a
width-96 control (tex 1708-1709) but no numbers appear anywhere. The
banked results (`outputs/p2_width`, `outputs/p2_narrowkey`;
`docs/exposure_rerun_results.md` phases 8b) largely settle it: widened
shared plateaus about .056 below SK at 2.3x the parameters, a width-16
key already beats SH, and B1's capacity-matched estimate has the routing
effect surviving at a third to a fifth of the printed gap. As printed,
tab:mass conflates mechanism with width. (B1 D1/D2, B2 F1, B4 M3)

### MAJOR

**M1. Real SH is not the SH the paper defines.** [V] The real-study
driver gives the shared arm a width-64 item embedding for 2PL/GPCM
(`kt-irt/src/deep_irt/bench/_p2_realstudy.py:165-169`), where the
synthetic bed and the methods define SH as width-8. NRM is width-matched.
So "SH" means three different objects across the paper, the
synthetic-to-real bridge is broken by design, and the person-side story
(SK the stronger anchor) is confounded with encoder capacity in exactly
the cells that carry it. (B3 D4, B4 M2)

**M2. The reported intervals are invalid, and the pairing was thrown
away.** [B: B5 recomputed from raw folds] Five seed clusters cannot
support a percentile bootstrap (the honest df=4 t interval is about 40%
wider than the printed one), and the design is fully paired (shared data
seed, fold split, init seed) yet every printed interval treats arms as
independent. The paired seed-level differences are the correct estimand
and are *stronger*: +.344 (t about 8.7), +.463 (t about 47), +.223
(t about 8.2) at the reference cells, 5/5 seeds positive. This fix costs
nothing and strengthens the headline. (B5 D1/D3)

**M3. Outcome switching on the pre-registered primary.** [B] Every
real-study table header declares concordance-with-MML the frozen primary
metric. The paper reports it only for EdNet-2PL (.776/.787), where it is
favorable. Unreported: KDD .44-.51, TIMSS .32-.56, EdNet-NRM .13-.43, all
per-fold discrepancy values, and the transformer reversal (SK worse than
SH on EdNet-2PL on every metric) which is visible in the accuracy table
but never named, while the appendix agreement table silently shows the
LSTM. All numbers on disk. (B3 D3)

**M4. The NRM gains-over-direct claim is a strawman, and nothing beats
the majority baseline.** [B, 4 boards] The item-wise popular-option
baseline reaches .653; every NRM head sits at .636-.648 and every direct
head at .526-.565. A direct head 9-13 points below a frequency table is
an under-provisioned baseline (single shared MLP, no per-item option
intercepts), not evidence for IRT structure. The defensible claim is
"IRT heads reach the item-popularity floor; these direct heads do not."
The paper concedes this at tex 1277-1283 and still asserts large gains at
1093-1096. (B1 D6, B2 M1, B3 D8, B4 M1)

**M5. DKVMN real cells are a single seed.** [B] Every DKVMN real cell on
disk is data-seed 0 only (3 of 4 SH columns a single fold), printed to
three decimals beside 25-fold cells, and the table note misstates the
coverage. Under the paper's own inference unit these are n=1. (B5 D2,
B4 M6)

**M6. The person-side conclusion flips under each arm's natural head.**
[V] The paper's routed comparison gives SK .334 vs SH .28 on the
theta-vs-raw anchor; the *unrouted* SH head anchors at .595
(`kt-irt/results/p2_v3_arm1r/VERDICT_arm1r.md` refs line). Routing was
built to fix SK's collapse; imposing it on SH halves SH's anchor and
manufactures the ordering. Uniform routing is defensible; hiding the
unrouted number is not. (B3 D9)

**M7. The MML anchor is compromised where it matters.** [V for (a)] (a)
The EdNet-2PL mirt reference records `"converged": "FALSE"`
(`kt-irt/results/p2_realstudy/mirt/ednet_2pl/reference.json:1264`),
undisclosed, while anchoring the concordance and EAP rows. (b) MML item
parameters were calibrated on all 2000 learners including scored tails,
while neural rows train on 4/5 folds: item-side leakage in MML's favor in
the accuracy table. (c) MML EdNet-NRM .609 is scored on the 34% covered
subset; the all-positions value is .279, in a column where every other
entry is all-positions. (B3 D7, B4 m4)

**M8. Theta recovery is defined as a headline metric and never
reported, and the training protocol in the appendix is wrong.** [B] No
theta column exists in any results table, while theta is the most
fragile quantity (peaks then collapses; retention as low as 48%). The
real study trains 120 epochs with patience=None
(`_p2_realstudy.py:113,380`); the appendix claims 150 with patience 10.
My exhibit E1 already computed synthetic theta recovery from stored
records: SK better in all 9 cells (.61-.96 to .86-.97). (B3 D10; E1)

**M9. Rank correlation is the only currency; the promised downstream
evidence is absent; the banked CAT results are the strongest unpublished
asset.** [B] No linked-scale RMSE anywhere despite tex 860 promising
linear-scale inspection; the raveled-beta Spearman inflates threshold
recovery (between-step spread is guaranteed by construction); the design
table promises an adaptive-testing comparison whose paragraph is
commented out. Meanwhile the campaign banked the full CAT invoice
(`outputs/p2_cat`, `p2_cluster/cat_clustered.json`: shared 196.8% test
length, +2.3pp misclassification) and this session's real-bank
replication (`kt-irt/results/p2_cat_realbank/`: SH stops at 8 items
certifying SE .29 vs true RMSE .69, confidently wrong; order transports,
form is bank-dependent). (B3 D5, B4 m5, B1 D7; E2)

**M10. Zero misspecification evidence, with the harness already
built.** [V: no outputs on disk] Every synthetic claim is conditioned on
a correctly specified, static-ability, uniform-exposure generator.
`run_misspecification_probe.py` (7 violations, tested, preserved at
parent commit 440e454 and portable) has never produced a result. All
five boards flagged it; B2, B3, and B4 each made it their single demand.
The real-data record already holds three reversal signatures (transformer
EdNet, the wide-key NRM reliability crater .695 to .065, TIMSS
discrepancy .68-.87), so the direction is genuinely uncertain, which is
what makes the run informative either way. (B1 D5, B2, B3, B4 M5, B5 D7)

**M11. The disattenuated person agreements misapply the paper's own
metric ruling.** [B] .59/.63 in the table body rest on dividing .178 by
reliabilities of .245 and .373 estimated from five seeds on the same
response matrix (shared sampling error inflates the numerator and
appears in neither denominator; seed disagreement in a symmetric
likelihood is partly structure). The paper's own ruling
(`theta_metric_ruling.md`) mandates observed-rho headline with a labeled
ceiling. My E5 shows the honest version of the same move on the item
side: disattenuation lifts nrm-ednet only .77 to .81, so the design
divergence is real, not attenuation. (B3 D6, B4 m2; E5)

**M12. Every interpretive real exhibit is one encoder, and the
person-side exhibits one fold.** [B] All five real-data figures are
LSTM; reversal and trajectory quantities are fold 0. The three-encoder
table replicates prediction only. Transformer contradicts the LSTM story
where it is reported (M3). (B4 M4)

### MINOR (compressed)

Sub-1pp NRM prediction edges elevated against the paper's own tie rule,
with learner-clustered rather than seed-clustered intervals (B5 D5).
"Prediction changes remain small" is false for GPCM/NRM (+1.5 to +4pp,
above the tie threshold); the honest reading there is "SK dominates both
axes," a different claim (B4 m1). "18/18 positive" carries no
multiplicity statement, and 5 seeds put the sign-test floor at p=.0625;
the effect sizes carry it, the sign count does not (B5 D6). KDD is
decoration, reported only where it ties, while its unflattering
concordance (.41-.51) and discrepancy (.53-.57) sit on disk (B4 m3). The
synthetic NRM tab:mass rows come from the routed head via a frozen input
with no surviving generator, undisclosed mixed provenance that also
contradicts the data-availability statement (B3 D12d). TIMSS trajectory
agreement quoted as median .84 while the mean is .72 and the EdNet
counterpart .21 goes unreported (B3 D12b). Threshold SDs are cross-fold,
not seed-clustered (B5 D8). The p-value agreement rows are
near-tautologies riding a shared anchor and should be labeled
manipulation checks (B3 D11). Decoder line says K=4 where TIMSS is K=3
(B3 D12e).

---

## Part II. What survives

Cross-board consensus, stated at the standard of the strictest board
that accepted it.

**S1. The prediction-recovery dissociation is real, large, and
strengthens under correct statistics.** At accuracy differences within
about 1pp (binary), the item-to-parameter path moves discrimination rank
recovery by +.15 to +.46; paired within-seed t between 8 and 47; sign
consistent in all 72 cells; not closed by 10x data, not closed by 500
epochs (banked trajectory study), not closed by width alone (plateau
.056 below SK at 2.3x parameters). Scope: synthetic, correctly
specified, static theta, uniform exposure. Conditional on printing the
capacity controls (F3) and paired intervals (M2).

**S2. The general law is encoder-conditional: whichever item-parameter
group stays pooled is the laggard.** For LSTM/transformer that is
discrimination; for DKVMN it is difficulty (the paper's own appendix
rows show it). The universal "discrimination is fragile" framing is
refuted by the paper's own table and must become "the pooled parameter
is fragile," which is a cleaner and more novel claim. (B2 M2)

**S3. Location is the robust readout everywhere.** SH~SK .998, versus
empirical p-value -.975, versus MML .73 on real data; degrades far less
than discrimination in every bed. Genuine, defensible, and the paper's
most transportable psychometric fact. (E3-E5 refine: the load/exposure
mechanism I proposed for the divergence is dead, killed by my own E4
within-cell test; the nrm-ednet divergence is real and format-specific,
E5.)

**S4. Real reliability gains for SK slopes stand as reliability only.**
Split-half .755 to .847 (EdNet-LSTM), .748 to .802 (KDD), .795 to .839
(DKVMN, n=1 seed caveat), with the transformer reversal named alongside.
Consistency, not validity; the paper's own stability artifact says so.

**S5. The honest ceiling. MML dominates both neural arms at every
tested cell.** Keep, and keep prominent.

**S6. Prediction parity on real binary/ordinal cells** within about
1-2pp, mixed sign (LSTM and transformer; not DKVMN).

**S7. The TIMSS null as exposure-regime confirmation.** At about 5,000
learners on 31 items the exposure law predicts SH~SK, and that is what
occurs; claimable as "at operational exposure the path choice is
immaterial to the fitted structure," never as validity of that
structure (F2).

**S8. The person-side divergence is the sharpest real symptom the paper
owns** (final-theta cross-reading agreement .18-.33; SH~SK trajectory
agreement .21 on EdNet-2PL) and is currently dressed as an agreement
result. Feeds the Part III arc directly.

**S9. Banked-but-unpublished assets, verified on disk:** oracle ladder
(SH .719, two-stage .934, oracle-clamp .979, SK .941, mirt .982), width
and narrow-key capacity controls, 500-epoch trajectory study, synthetic
theta recovery (E1), full CAT invoice plus the real-bank CAT replication
(E2), per-fold discrepancy values for every real cell.

---

## Part III. The storyline that closes the glue criticism

The criticism was "math and datasets glued to prove a point that was
never there." The audit finds the point IS there; the paper stops one
step short of making it.

The intended arc: accuracy hides parameter corruption (synthetic
disease); a truth-free discrepancy audit detects it (built, certified,
tau = .152); a bounded repair (SK) fixes what is fixable; the invoice
(CAT) prices what remains. The printed paper demonstrates the disease,
defines the audit, then on its own real data never runs the audit, and
instead offers SH~SK agreement as reassurance, which the synthetic phase
itself proved is exactly what stability looks like when both arms share
the pathology. Meanwhile the audit numbers sit computed in every real
cell JSON: the instrument fires on TIMSS (.68-.87) and flags KDD
(.53-.57), and the CAT invoice (banked, plus E2's real-bank replication)
prices the consequence: an operator trusting the shared readout stops
testing at 8 items certifying SE .29 when the truth is .69.

Run the paper's own audit on the paper's own case studies and the two
"glued" datasets become the two ends of one demonstration: EdNet the
cell where the repair works and is visible against MML; TIMSS the cell
where truth is absent and the audit, not agreement, is what a
practitioner has. The aha the author wanted is that the audit is the
product; the SK arm is evidence the audit's alarms are real, not the
product itself. This reframe costs wording plus tables already on disk,
and it converts F2 from the paper's worst embarrassment into its
centerpiece. It also aligns with the DKT-home framing rule: the
contribution is what a prediction-grade model must add before its
readouts are measurement-grade, and the audit is that addition.

---

## Part IV. Venue by evidence fit

Verdicts assume the defect ledger is addressed at the stated tier;
"delta" is what that venue requires beyond the current draft.

| Venue | Story it supports | Verdict as-is | Required delta | Cost | Fit |
|---|---|---|---|---|---|
| **CAEAI** (current target) | The audit arc (Part III): prediction-grade KT readouts, truth-free audit, bounded repair, CAT invoice | Reject (F1/F2 discoverable by one careful reviewer; storyline criticism already received) | Tier 1 + Tier 2 in full; TIMSS unsorted re-export (cheap retrain); E9 strongly advised, not strictly required | ~2-3 author-weeks + hours of cluster time (+1 GPU-week if E9) | **Best.** Applied audience values the audit-as-tool; evidence sufficient once banked assets are printed |
| **IEEE TLT** | Same arc, system-flavored (audit + invoice as the engineering deliverable) | Same | Same as CAEAI | Same | Strong fallback; identical prep, so the CAEAI package covers it |
| **JEDM** | Methodological: the dissociation + encoder-conditional law + capacity-matched attribution | Reject (F3, M4 are exactly what EDM reviewers check) | CAEAI delta + capacity-matched headline promoted to main text + tuned direct baseline or full retraction of the gains claim; E9 near-required | +1-2 weeks over CAEAI | Good second target; slower, more demanded |
| **Computers & Education** | Operational: what institutions risk when reading KT parameters; CAT invoice front and center | Reject (method sections would be cut, but significance bar is different) | CAEAI delta + heavier practical framing; less method detail tolerated | Similar | Possible, but the audience may not care about SH/SK mechanics; lower fit |
| **AIED/EDM conference** | One crisp claim: the pooled parameter is the laggard, and capacity does not close it | n/a (different artifact) | Extract 8-10 pages from S1+S2+F3 controls + E9 subset | ~1 week from the CAEAI package | Good companion paper, not a home for the whole program |
| **APM / JEM / measurement journals** | Amortized calibration as an estimator, audited to psychometric standard | Reject (B3's ledger IS their review) | Everything above + width-matched real rerun (M1) + linked-scale and within-step recovery (raw-beta re-exports) + per-fold MML refits + E9 mandatory | +3-4 weeks and the most GPU | **Not recommended.** Conflicts with the standing framing rule (DKT-home, IRT-flavor; never a psychometrics-theory contribution) and costs the most |
| **TMLR-class DL** | Amortization failure of weakly identified parameter families under shared representations | Reject, score ~3 (B2) | Capacity-matched headline, E9, tuned baselines, and a generality story beyond IRT | Highest relative delta | Not recommended; the contribution reads thin against a DL bar and the toy-template memory already covers the generalization idea separately |

---

## Part V. Costed action menu

### Tier 1: wording and demotion (author-days, zero compute)

1. Withdraw or demote the ordered-thresholds claim (F1) pending the raw
   re-export; one paragraph.
2. Reframe TIMSS per Part III (F2): agreement is the symptom, the audit
   is the reading. The discrepancy numbers to cite are on disk.
3. Retract "gains are large over direct" (M4); state the popularity
   floor .653 first; reframe as "reaches the floor; direct heads do
   not."
4. Rename the mechanism per S2: pooled-parameter fragility, not
   discrimination fragility; the DKVMN rows become confirming evidence
   instead of an anomaly.
5. Honor the 1pp tie rule for the NRM edges; restate GPCM/NRM as "SK
   dominates" (B4 m1); fix the appendix protocol table (M8), the K=3
   typo, the median/mean trajectory quote, and label the p-value rows
   manipulation checks.
6. Move the disattenuated values out of the table body into a labeled
   ceiling with both caveats (M11), per the paper's own ruling.
7. Disclose: mirt non-convergence (M7a), DKVMN per-cell n (M5 as
   annotation), routed-vs-natural head choice with the .595 number
   (M6), NRM tab:mass provenance (minor d).

### Tier 2: analyses from banked artifacts (author-days + CPU-hours, no GPU)

8. **Paired seed-level statistics throughout** (M2, B5's demand): df=4
   t or wild-cluster-t on SK-SH differences, one joint statement, all
   from stored fold JSONs. Strengthens the headline.
9. **Print the capacity controls** (F3): key-16 and width-96 rows from
   `outputs/p2_width`, `outputs/p2_narrowkey` into the main table plus
   one attribution sentence (mechanism survives matched at a third to a
   fifth of the gap).
10. **Print the pre-registered primary everywhere** (M3): concordance
    and discrepancy columns for all real cells, transformer reversal
    named; all on disk.
11. **Promote the audit table** (F2/Part III): per-cell truth-free
    discrepancy vs tau in the real-data section; this is the arc's
    hinge and costs a table.
12. **Restore the CAT invoice by fulfillment** (M9): banked clustered
    results + the E2 real-bank replication; un-comment the promised
    paragraph and deliver it.
13. **Add the theta column** (M8): synthetic theta recovery from stored
    records (E1's computation, promoted); plus the trajectory-study
    sentence (banked) closing the under-training objection.
14. **Oracle ladder into the discussion** (S9/B1 D1): SH .719, SK
    .941, two-stage .934, oracle .979, mirt .982; positions SK as
    closing most of the closable gap.
15. Per-fold MML refits for the accuracy table + common-set NRM
    scoring (M7b/c): mirt on CPU, hours.
16. EdNet cross-encoder interpretive replication where stored JSONs
    permit (M12): compute the agreement/anchor quantities for
    transformer/DKVMN from existing fold files; where a quantity needs
    missing folds, scope the claim to LSTM explicitly.

### Tier 3: new compute (cluster; the only new experiments worth buying)

17. **E9 misspecification battery** (M10; every board, three boards'
    single demand). Design consolidated from B3/B4: reference cohort
    (N=2000, Q=200), LSTM + transformer, 2PL + GPCM, SH + SK, 5 seeds
    x 5 folds, violations at control plus two doses. Priority order if
    compute binds: drifting theta, local dependence, threshold
    disorder, exposure imbalance (full); response style, DIF, noisy
    thresholds (mild only). Report per condition: paired
    delta-accuracy, paired delta-recovery, and the truth-free
    discrepancy for both arms, so the run simultaneously tests whether
    the audit instrument still fires under violation. Pre-registered
    reading: the SK advantage may shrink with dose; the claim to
    defend is that it does not reverse, and that the audit flags what
    misspecification corrupts. Either outcome is publishable: robust
    repair upgrades the paper; scoped repair promotes the audit to the
    contribution, which Part III already does. Cost: about one
    GPU-week on the UT cluster via autopilot; harness exists
    (preserved at parent 440e454), needs porting into the current
    package and wiring to the SH/SK toggle. THE recommended buy.
18. **TIMSS raw-threshold re-export** (F1): retrain lstm_gpcm_timss
    x {SH,SK} x 25 folds exporting unsorted beta; hours on the
    cluster. Cheap, closes a fatal. Do with 17.
19. **DKVMN seed completion** (M5): 4 more seeds for the real DKVMN
    cells, overnight autopilot; or drop DKVMN rows (free).
20. **Width-matched real rerun** (M1, B4's second demand): EdNet-2PL +
    TIMSS with SH at emb 8, three encoders, 25 folds; a few cluster
    days. Buy only if the person-side story stays in the paper;
    otherwise Tier-1 scoping (drop the person-side directional claim)
    is honest and free.
21. **Tuned direct-NRM baseline** (M4): direct head with per-item
    option intercepts; one cell, cheap. Buy only if the gains claim is
    kept; retraction is free.
22. **Not recommended:** a second options-bearing dataset (Eedi-class;
    1-2 weeks acquisition + pipeline for the weakest thread; scope NRM
    as exploratory instead), any MIRT/multidimensional extension, new
    encoders, and any measurement-journal-specific package (venue
    ruled out above).

---

## Part VI. Recommendation

**Target CAEAI with the Part III arc. Execute Tier 1 and Tier 2 in
full, buy items 17-19 from Tier 3, decide 20/21 by whether the
person-side and gains claims stay.** Total: roughly two to three
author-weeks of rewriting (the author writes; the tables and numbers
are ready or one script away), hours of CPU, about one GPU-week of
cluster time, most of it the misspecification battery. IEEE TLT is the
no-extra-cost fallback; an AIED/EDM companion extraction is available
later from the same package. The measurement and DL venues are ruled
out on cost and on the standing framing rule.

What makes this cost/gain favorable: the audit found that the paper's
liabilities and its missing assets are the same items. Printing the
banked evidence simultaneously closes F2, F3, M3, M8, M9, M11 and
supplies the storyline; the statistics fix (M2) strengthens the
headline while making it honest; and the one genuinely new experiment
(E9) is the one whose harness is already written and whose every
outcome has a publishable reading.

---

## Part VII. Post-dossier executions (2026-08-10): the Tier-3 buys, landed

All three purchased runs are complete, committed, and change verdicts:

**E9 misspecification battery (item 17) — the strong outcome.**
2500/2500 units, zero failures. The SK-SH recovery delta stays positive
in 49/50 cells (5/5 seeds, t 2.3-18.4) and never reverses; it GROWS
under local dependence and threshold disorder (the feared reversal
cases); |d(acc)| <= .011 everywhere (the dissociation survives every
violation); the truth-free discrepancy tracks true corruption at
Spearman .93 (SH) / .72 (SK) and rises with dose within every family.
Boundary: extreme exposure imbalance floors both arms (the exposure
law). M10 closed. Exhibit E7; tables
`kt-irt/results/p2_misspec/battery_report.md`.

**TIMSS raw-threshold rerun (item 18) — F1 closed with an upgrade.**
Honest ordered fraction .43 in both arms (sorted export said 1.00;
classical .613), 18/31 stably disordered, containing all 12 classical
non-modal items; the raw order gap tracks the classical gap at
Spearman .98. The sort had erased a true, classically-agreeing signal.
Exhibit E6; store `kt-irt/results/p2_realstudy_rawbeta/`.

**DKVMN seed completion (item 19) — M5 closed.** Both arm1r DKVMN
cells at 5 seeds x 5 folds (was seed 0 only).

Also filed: `docs/framing_review/format_unification.md` — the
author-requested framing that strings 2PL/GPCM/NRM into one storyline
(one divide-by-total family; one law, location-preserving /
slope-corrupting readout path, tested at three doses of slope
structure; the battery's cross-format table is its closure).

---

*Program artifacts: boards `docs/framing_review/B1..B5_*.md`; exhibits
`docs/framing_review/E1..E7_*.md`; v1 narrative dossier
`docs/framing_review_caeai.md` (sections A-E); CAT real-bank harness
`kt-irt/src/deep_irt/bench/_p2_cat_realbank.py` with results in
`kt-irt/results/p2_cat_realbank/`. Verified-this-session facts: sort
artifact call chain, 12/31 classical disorder count, width-64 real SH,
mirt converged FALSE, unrouted SH .595, no banked weights for real
cells, no misspecification outputs anywhere under results/ (now
superseded by the E9 store).*
