# v3 results record — run plan + gathered results

Single ledger for the v3 campaign (plan: docs/paper_v3_plan.md). Every run
lands here: what ran, where outputs live, key numbers. Paper untouched until
results are in and reviewed.

## Conventions (all new figures/tables)

- **Naming:** `<ENC>-SH` (shared head) / `<ENC>-SK` (separated key):
  LSTM-SH, LSTM-SK, TF-SH, TF-SK, DKVMN-SH, DKVMN-SK. One sweep brings v2's
  existing figures to the same scheme when they are regenerated.
- **Palette:** the paper's class colors — SH orange #E69F00 (dark
  #D55E00), SK blue #0072B2 (sky #56B4E9); markers LSTM diamond, transformer
  plus, DKVMN star. No new colors.
- **Metrics:** item parameters = Spearman ρ_s; ability = Pearson r at the
  last observed step (paper's metrics section). The workshop pareto figure
  used Spearman for ability; the redo uses Pearson for consistency with
  fig_surface — both are stored per fold, so switching is one line.
- **Pareto redo drops the "decoupled dynamic" point** (state-conditioned
  heads are out of the paper by decision; the workshop figure predates that).
  SH width curve + SK star only.

## 0. Run plan (GPU queue, one job at a time on the 8 GB card)

| # | Job | What | Status |
|---|-----|------|--------|
| G3 | Export pass (small, first — unblocks case studies) | LSTM SK real cells: TIMSS-GPCM full step thresholds across 25 folds + θ trajectories/final-θ (5 seeds); EdNet-NRM θ trajectories (slopes+intercepts already on disk 4220x4x25); EdNet-2PL θ for completeness | DONE |
| G4 | Extended prediction metrics (small) | tab:real metric columns: AUC+NLL (binary), NLL (ordinal, QWK exists), option-acc+macro-F1+NLL (nominal); neural rows first, MML rows if cheap via the EAP predictor | DONE |
| G1 | Transformer width sweep | TF-SH GPCM+2PL at emb w∈{16,32,64,96}, N=2000 Q=200, 25 folds/cell (W=8 anchor + TF-SK point already in outputs/p2_toggle N2000 cells) | DONE |
| G2 | DKVMN width sweep (reduced) | DKVMN-SH GPCM+2PL at w∈{16,32,64,96}, N=2000, 5 seeds x 1 fold (DKVMN ~10 min/fold; anchors from toggle N2000 cells) | DONE |
| G5 | Optional, if the night allows | Extend DKVMN realstudy cells from 1 to 5 folds for tab:real credibility | DONE (n=4-6 per cell; DKVMN concordance now favors SK on all four cells with intervals clear of zero; NRM delta still favors SH) |

Verification duties inside the queue: confirm p2_width provenance (encoder +
N; rows store no encoder field) before reusing it as the LSTM panel; confirm
toggle N2000 cells store theta_pearson_lastvalid.

Parallel (no GPU): metrics-precedent literature research (see §6);
pareto/width redraw + the naming/palette figure pass happen after G1/G2.

## 1. Pareto / width-vs-separation (redo of workshop pareto_escape)

Figure spec: per encoder (3 panels), x = discrimination recovery (ρ_s),
y = ability recovery (Pearson r); SH width curve W=8..96 traced in class
orange with W labels; SK star in class blue. GPCM headline; 2PL companion;
NRM stays LSTM-only (width data exists only there) or appendix.
- LSTM panel: plot-only from outputs/p2_width (verify N/encoder).
- TF/DKVMN panels: after G1/G2.
### RESULTS
The separated design escapes the shared width curve on two of three encoders.
LSTM shows the classic trade-off (widening the shared design raises
discrimination recovery but lowers ability recovery) and LSTM-SK sits above
and to the right of the whole curve. DKVMN-SH is flat and high; DKVMN-SK is
weakly better everywhere. TF-SK does NOT escape: it lands on the shared width
trend at matched capacity (GPCM TF-SK 0.913/0.966 vs TF-SH-W64 0.917/0.968).
The paper's "width is not separation" claim must carry a transformer
exception. p2_width provenance CONFIRMED (encoder lstm, N=2000 Q=200,
realistic bed, theta read at last valid step). DKVMN panels are n=5 folds.
Figures: outputs/p2_v3_analysis/figs/fig_pareto_gpcm.*, fig_pareto_2pl.*
(+ fig_pareto_caption.txt).

## 2. tab:real fills (paper TODOs, lines ~1201-1210)

From existing disk data (no training): TF-SH/TF-SK rows (realstudy
transformer cells, 25 folds); DKVMN-direct TIMSS .577 / NRM .565 (direct
chain, reduced); DKVMN-SH/DKVMN-SK rows (n=1 folds — either caveat or wait
for G5). Extended metric columns after G4.
### RESULTS (G4; every row n=25, refits reproduce the original runs exactly)
| model | EdNet-2PL | KDD-2PL | TIMSS-GPCM | EdNet-NRM |
|---|---|---|---|---|
| Direct (DKT) | AUC .640 / NLL .774 | AUC .810 / NLL .402 | QWK .397 / NLL .903 | acc .526 / mF1 .524 / NLL 2.262 |
| LSTM-SH | AUC .638 / NLL .767 | AUC .821 / NLL .382 | QWK .403 / NLL .895 | acc .648 / mF1 .646 / NLL .917 |
| LSTM-SK | AUC .676 / NLL .654 | AUC .832 / NLL .366 | QWK .401 / NLL .898 | acc .586 / mF1 .583 / NLL 1.438 |
| MML (EAP) | AUC .705 / NLL .628 | AUC .808 / NLL .393 | QWK .415 / NLL .892 | acc .609 / mF1 .604 (on its covered 34%) |

Reading: LSTM-SK is the best neural model on both binary datasets by AUC and
NLL; the ordered TIMSS is a tie for everyone; on options every structured
head clears the Direct predictor by a wide margin and macro-F1 nearly equals
accuracy (balanced, not majority-class). MML-NRM convention: scored
on the positions its calibration covers (26.1% of positions once failed
calibrations are excluded; an earlier 34% counted them), not chance-filled.
Option Tracing (AIED 2021) floors for context: random .25, majority .18-.21,
their models .31-.33 macro-F1 (different data slice; compare to floors, not
head-to-head). Table: outputs/p2_v3_export/tab_real_metrics.{md,json}.

### Architecture-wide accuracy (all encoders, assembled from disk)

The extended-metric table above is LSTM-scoped; the accuracy comparison
below covers all three encoders (transformer n=25; DKVMN n=3-6, the
reduced/extended runs). The pattern replicates across architectures: the
IRT heads are free on binary/ordinal, every structured head clears the
direct predictor on options by a wide margin, and SH outpredicts SK on
the nominal cell under ALL three encoders (the SK option-prediction cost
is architectural, not an LSTM quirk). DKVMN-SK is the strongest neural
cell on EdNet binary (.645, matching MML).

| model | EdNet-2PL | KDD-2PL | TIMSS-GPCM | EdNet-NRM |
|---|---|---|---|---|
| LSTM-direct (DKT) | 0.599 | 0.838 | 0.577 | 0.526 |
| TF-direct | 0.602 | 0.823 | 0.579 | 0.554 |
| DKVMN-direct | 0.621 | 0.847 | 0.577 | 0.565 |
| LSTM-SH | 0.600 | 0.840 | 0.580 | 0.648 |
| LSTM-SK | 0.626 | 0.844 | 0.579 | 0.586 |
| TF-SH | 0.611 | 0.823 | 0.581 | 0.644 |
| TF-SK | 0.593 | 0.812 | 0.580 | 0.583 |
| DKVMN-SH | 0.639 | 0.845 | 0.578 | 0.648 |
| DKVMN-SK | 0.645 | 0.844 | 0.584 | 0.614 |
| MML (classical) | 0.645 | 0.830 | 0.584 | 0.609* |

*MML EdNet-NRM on its covered 26.1% of positions.

### n folds
| model | EdNet-2PL | KDD-2PL | TIMSS-GPCM | EdNet-NRM |
|---|---|---|---|---|
| LSTM-direct (DKT) | n=25 | n=25 | n=25 | n=25 |
| TF-direct | n=25 | n=25 | n=25 | n=25 |
| DKVMN-direct | n=3 | n=3 | n=3 | n=3 |
| LSTM-SH | n=25 | n=25 | n=25 | n=25 |
| LSTM-SK | n=25 | n=25 | n=25 | n=25 |
| TF-SH | n=25 | n=25 | n=25 | n=25 |
| TF-SK | n=25 | n=25 | n=25 | n=25 |
| DKVMN-SH | n=6 | n=5 | n=5 | n=5 |
| DKVMN-SK | n=5 | n=5 | n=5 | n=5 |

Source: outputs/p2_v3_analysis/tab_real_allenc.md.

### Extended metrics, all encoders — RUNNING (placeholders)

| model | EdNet-2PL | KDD-2PL | TIMSS-GPCM | EdNet-NRM |
|---|---|---|---|---|
| TF-SH | AUC .658 / NLL .686 | AUC .791 / NLL .434 | QWK .403 / NLL .893 | acc .645 / mF1 .642 / NLL .957 |
| TF-SK | AUC .631 / NLL .766 | AUC .774 / NLL .482 | QWK .400 / NLL .894 | acc .585 / mF1 .582 / NLL 1.436 |
| DKVMN-SH | AUC -- / NLL -- | AUC -- / NLL -- | QWK -- / NLL -- | acc -- / mF1 -- / NLL -- |
| DKVMN-SK | AUC -- / NLL -- | AUC -- / NLL -- | QWK -- / NLL -- | acc -- / mF1 -- / NLL -- |

Transformer rows complete (n=25/cell; reproduction bit-exact on TIMSS, within
0.015 accuracy elsewhere). Claim checks: the SH-over-SK option gap REPLICATES
on the transformer (acc .645/.585, NLL .957/1.436, mirroring LSTM); TIMSS QWK
parity replicates; SK's binary AUC edge does NOT transfer -- TF-SH wins both
binary cells, the transformer exception again. DKVMN rows still running on its
reduced folds. Source: outputs/p2_v3_export/tab_real_metrics_allenc.md.

## 3. Export-pass artifacts (G3)

Target files: per-fold thresholds (TIMSS), per-learner θ trajectories +
final θ (TIMSS + EdNet). Feeds case studies (category/expected-score curves,
learner trajectories, ability bands) and threshold-stability rows.
### RESULTS
Delivered: TIMSS full step thresholds (31 x 2, all 25 folds)
-> outputs/p2_v3_export/timss_gpcm_sk/; theta trajectories + final theta for
5 seeds of TIMSS-GPCM, EdNet-NRM, EdNet-2PL -> outputs/p2_v3_export/traj/;
NRM alpha/beta confirmed as option slopes / option intercepts. Not exported
(noted for a later small run): KDD theta, shared-head theta rows.

## 4. Real-data parameter stability — DEPRECATED as an evaluation metric

**DECISION (user, 2026-07-09): stability is deprecated as a quality measure.**
Being stable does not mean being right: consistency is blind to bias, and the
SH-vs-SK comparison demonstrated it on real data (SH option slopes are the
MOST stable group, .93, while agreeing LESS with the empirical distractor
statistic, .587 vs .705). The pooled design is wrong the same way in every
resample, so it agrees with itself. Stability numbers below are retained for
the audit trail and at most as an internal sanity check (a wildly unstable
estimate is uninterpretable), but they are NOT reported as evidence of
parameter quality. The agreement suite (difficulty vs proportion correct,
discrimination vs the classical index, option slopes vs the distractor
statistic, DOA, item fit) is the real-data evidence; synthetic recovery is
the only truth-based check.

Split(fold)/seed/exposure-stratified Spearman per parameter group, SK design
(gate framing per plan T1; no SH-vs-SK stability contest). Groups: TIMSS
discrimination + step thresholds; EdNet option slopes + intercepts; EdNet-2PL
difficulty + discrimination.
### RESULTS (SK design; Spearman rank stability)
| group | n items | split | seed | exposure>=50 |
|---|---|---|---|---|
| EdNet-2PL difficulty | 250 | .982 | .997 | .982 |
| TIMSS step thresholds | 31 | .971 | .994 | .971 |
| EdNet-NRM option intercepts | 4220 | .862 | .976 | .888 |
| EdNet-2PL discrimination | 250 | .847 | .974 | .847 |
| KDD-2PL difficulty | 250 | .839 | .972 | .840 |
| KDD-2PL discrimination | 250 | .802 | .957 | .800 |
| TIMSS discrimination | 31 | .644 | .908 | .644 |
| EdNet-NRM option slopes | 4220 | .423 | .781 (sign-aligned; .226 raw) | .460 |

Reading: difficulty-type parameters are stable enough to inspect;
discrimination is moderate; option slopes are the flagged group. The
mirror-image finding: 9/25 LSTM (8/25 TF) runs converged to the globally
reversed sign solution; sign-aligned, seed stability rises .226 -> .781 and
encoder agreement .314 -> .782, while split stability stays .423-.460 =
genuine item-level noise of sparse options. Rule for the paper: interpret
slopes in aggregate and for well-exposed items; intercepts are fully stable.
Stability rises with exposure for every group (one small-bin dip on KDD, n=11)
(figs/fig_stability_exposure.*). Table: stability_table.{md,json}.

## 5. Case-study analyses (post-G3)

TIMSS: category-probability curves, expected-score curves, threshold
distribution, 2-3 learner trajectories. EdNet: option-probability curves,
correct-option slope-orientation distribution, distractor attractiveness by
low/mid/high θ band, intercept-vs-option-frequency, binary-vs-nominal
comparison.
### RESULTS
TIMSS (fig_timss_case.*): category probabilities move mass 0 -> 1 -> 2 as
ability rises; expected-score curves are monotone; both step thresholds are
correctly ordered on 100% of items (seed-mean); discrimination positive in
all 25 folds. Threshold locations: timss_thresholds.{md,json}.
EdNet (fig_ednet_case.*): among well-exposed items (>=50 obs, n=1712) the
correct option carries the largest learned slope on 92.9% and a positive
slope on 99.2%; learned option intercepts track empirical option frequency
(Pearson .80, Spearman .93); distractors split into fades-with-ability
(n=3448), persistent (268), weak (1420). Checks:
ednet_option_checks.{md,json}. The mirror-image correction is applied before
any averaging (naive averaging would report 71% instead of 92.9%).

## 6. Metrics with precedent (literature)

Question: which evaluation metrics for learned-parameter quality have
precedent in KT/ML papers with sequential encoders + parameterized
(interpretable) heads — so nothing is invented — and which of them show a
structural gap over standard binary KT (candidates: option-level NLL /
macro-F1 / distractor prediction, which binary KT cannot produce at all;
agreement with classical item statistics such as proportion-correct and
point-biserial, which have CTT precedent; stability/consistency precedents)?
Fallback if no big-gap metric: the item-analysis / interpretability route
with named precedents for how learned-parameter quality is evaluated.
### FINDINGS (2026-07-09, web-verified with citations)

Adoptable metrics, all precedented — nothing invented:
1. **Learned difficulty vs empirical p-value + vs classical 1PL/MML fit
   (Pearson).** Deep-IRT (Yeung, EDM 2019: r~0.56 vs proportion-incorrect,
   r~0.64 vs 1PL); VIBO (Wu et al., EDM 2020). We already hold the classical
   fits for EdNet/KDD/TIMSS; p-values are one groupby away. -> analysis A1.
2. **Learned discrimination vs point-biserial** (Crocker & Algina 2008;
   Lord & Novick 1968; VIBO companion). -> A2.
3. **Degree of Agreement (DOA)** on theta — NeuralCD (Wang et al., AAAI
   2020): higher-proficiency learners should out-answer lower ones per item;
   ground-truth-free, ML-venue precedent. Needs final theta (G3 export). -> A3.
4. **Option-level accuracy + macro-F1** — Option Tracing (Ghosh, Raspat,
   Lan, AIED 2021; AAAI 2022) on EdNet+Eedi: THE task precedent for our NRM
   0.65 vs direct 0.53. Metric columns already queued (G4). -> A4.
5. **Option/distractor characteristic curves** — Thissen, Steinberg &
   Fitzpatrick 1989 ("the distractors are also part of the item"); Bock 1972.
   Our EdNet case-study curves reproduce the canonical device. -> A5.
6. **Split-sample/seed parameter stability** — item-parameter invariance
   lineage (Rasch 1960; Lord 1980; documented r~.97 cross-sample difficulty
   invariance). NAMING RESOLUTION: tables keep the revision plan's word
   "stability"; prose cites the invariance literature as the precedent
   (avoids the loaded "measurement invariance"). -> feeds section 4.

Structural gap over binary KT (lead with these): option prediction (4),
distractor trace lines (5), partial-credit NLL + QWK on TIMSS (Cohen 1968;
Taghipour & Ng, EMNLP 2016) — binary KT cannot produce any of them. AKT and
QIKT evaluate interpretability only qualitatively/via AUC, so metrics 1-3
differentiate us.

Traps recorded: never anchor to publisher/expert difficulty labels
(Deep-IRT got r=0.08 there); no multiclass AUC for options (use acc+macro-F1
per Option Tracing); DOA reported as concordance with its monotonicity
assumption stated + random floor; ECE optional, no KT precedent claim;
p-value stays the PRIMARY anchor vs classical-fit correlation (independence
note against circularity).

New bib entries needed at write-time: Wang et al. AAAI 2020 (NeuralCD);
Ghosh/Raspat/Lan AIED 2021 (+AAAI 2022); Thissen-Steinberg-Fitzpatrick 1989;
Bock 1972; Crocker & Algina 2008; Cohen 1968; (optional Taghipour & Ng 2016,
Guo et al. 2017).

### Verified nearest papers (primary sources, full texts checked)
QIKT (AAAI 2023): prediction AUC/ACC + two qualitative case studies; no
quantitative parameter check (verified by full-text search). IEKT (SIGIR
2021): prediction + a visual distribution comparison; authors concede the
learned distribution is degenerate. Option Tracing (AIED 2021): option
accuracy + macro-F1 (no AUC) + expert-label clustering (ARI .372) — the task
and metric precedent for our EdNet study. Partial-credit KT line (Wang &
Heffernan 2011/2013; Ostrow et al. 2015): prediction only; no partial-credit
deep-KT paper validates any learned parameter — the TIMSS case study is the
first to do so. pyBKT (2021) is the one synthetic-recovery precedent (binary
BKT). Positioning sentence this supports: the nearest interpretable-KT
models validate nothing quantitatively; this paper's suite exceeds the
field's standard rather than matching it.

## 7. Expanded evaluation suite (breadth pass)

| check (precedent) | headline | consistent with the picture? |
|---|---|---|
| Ability vs raw total score (VIBO) | EdNet .54, TIMSS .84 | yes |
| Ability vs classical ability (VIBO) | EdNet .65, TIMSS .83 | yes |
| Ability concordance DOA (NeuralCD) | EdNet .584, TIMSS .784 vs .500 floor | yes |
| Item fit, infit/outfit (Wright & Masters) | medians ~1.0 both datasets; EdNet outfit flags 32%, high side, static-theta caveat stated | yes, with caveat |
| Distractor index vs learned slopes (Thissen/Haladyna) | Pearson .705; correct options positive 93%, distractors negative 80% | yes - strongest convergent number |
| Test information vs classical (Lord/Muraki) | TIMSS peaks coincide; shape r=.982; SK peak ~half MML height (shrunken discrimination spread) | shape yes, scale honestly not |
| Calibration ECE (Guo 2017) | deferred - needs a ~10-min GPU probability re-score (calibration_status.md) | pending |

NRM person-agreement is weak (r=.14 even sign-aligned) - consistent with the
known NRM ability pathology, reported not hidden.

Figures for this suite: fig_agreement (a)-(d) (difficulty vs p-value, discrimination vs point-biserial, ability vs total score, option slopes vs option point-biserial) and fig_item_fit (a)-(d) (infit/outfit, EdNet + TIMSS, conventional band shaded, flag rates annotated); all annotations verified against the stored JSONs. Case-study and exposure figures regenerated as labeled (a)-(d) panels with legends.

## 8. Alignment with the revision plan

| plan item | status | where |
|---|---|---|
| T1 evaluation-design table | writing-phase | - |
| F1 SH/SK architecture figure | keep + recaption (writing-phase) | existing fig_arch |
| T2 synthetic summary (Acc/Recovery deltas) | data ready (dd_points) | outputs/p2_v3_analysis |
| F2 recovery scatter | keep, simplify caption (writing-phase) | existing fig_scatter |
| F3 delta-recovery vs delta-prediction | DONE, 18/18 points (NRM re-sourced from the adopted oriented-head tree; the 13-point v1 used stale pre-orientation NRM cells) | figs/fig_dd.* |
| F4 width-vs-separation (pareto, 3 encoders) | DONE, with transformer exception | figs/fig_pareto_*.* |
| T3 real prediction compatibility (+NLL/AUC/mF1) | DONE (DKVMN rows filling, G5) | tab_real_metrics |
| T4 real parameter stability | DEPRECATED by user decision (consistency is blind to bias; SH/SK proof case) | stability_table kept for audit |
| F5 stability vs exposure | DEPRECATED with T4 (figure kept for audit) | figs/fig_stability_exposure.* |
| sec 9 split/seed/exposure experiments | DONE | stability_table |
| sec 9.5 agreement with offline reference | DONE | precedent_metrics |
| sec 10 TIMSS ordinal case study | figures DONE, prose pending | figs/fig_timss_case.* |
| sec 11 EdNet nominal case study | figures DONE, prose pending | figs/fig_ednet_case.* |
| sec 12 MML one-paragraph | writing-phase | - |
| sec 13 refit -> appendix | writing-phase | - |
| sec 14 CAT -> appendix | writing-phase | - |
| sec 15/16 abstract + title | user-owned | - |

## 9. Open items

1. Calibration ECE: one ~10-min GPU re-score once G5 finishes.
2. KDD theta export + shared-head theta rows (small, optional).
3. G5 completion - DONE (n=4-6 per cell).
4. Transformer exception wording for the width/pareto claim.
5. NRM slope rule for the case study (aggregate + well-exposed only).
6. Writing-phase items in section 8 (all analysis dependencies now met).

## 10. SH vs SK on real data (the comparison pass)

All real-data checks recomputed for the shared design beside the separated
one. Scalar parameters are near-parity (SK marginally more consistent on
discrimination, ties on difficulty; ability-side DOA and score agreement at
parity). The headline is the nominal decoder, and it is the paper's thesis in
miniature, visible on real data with no ground truth:

| check | SK | SH |
|---|---|---|
| option-slope consistency (split) | .423 | .929 |
| option-slope agreement with the distractor statistic | .705 | .587 |
| correct option carries the largest slope | 92.9% | 91.8% |
| mirror-flipped runs | 9/25 | 8/25 |

The shared head's option slopes are far MORE self-consistent and LESS
faithful to the empirical distractor structure: smoothing inflates
consistency, not correctness. A reader who checked only stability would
prefer the design whose parameters agree less with the data. This is the
stable-and-wrong pattern demonstrated in the field, and it is why stability
is a gate and never a verdict.

Figures: figs/fig_agreement_shsk.* (paired dots, SH orange vs SK blue, one
row per metric x dataset - the option-slope row is the lone visible gap);
figs/fig_case_shsk.* (same-item expected-score and option curves under both
designs). Tables: stability_shsk, orientation_shsk, ability_shsk. Queued and
running: EdNet SH ability export (person-side SH rows auto-append when it
lands).

### Case studies under both designs (fig_timss_case_shsk, fig_ednet_case_shsk)

TIMSS: SH and SK tell the same story in every panel - category curves,
expected scores, matched-learner trajectories, and thresholds nearly
coincide; ordered-threshold fraction is 1.00 under BOTH designs. EdNet:
item-side conclusions agree (correct-option orientation 92.9% vs 91.8%,
archetype distractors behave the same, intercepts track option frequency for
both, SH r=.83 / SK r=.80), with SH visibly smoothing: its option-slope
distribution is compressed into a narrow band and its correct-option curves
rise more gently.

One finding stands out beyond the figures. For the SAME learners, the two
designs' ability trajectories correlate at median .84 on TIMSS (six-step
sequences) but only .21 on EdNet (two-hundred-step sequences). Between-learner
ordering agrees under both designs (the DOA and score-agreement checks pass
for both), but the fine-grained within-sequence path a practitioner would
read as "the learning curve" is largely design-specific on long sequences.
Prediction-equivalent designs need not agree on the latent trajectory - the
paper's thesis extended to the person side, and the number behind the
revision plan's own caution that trajectory panels are model-based traces,
not measurements. Numbers: case_shsk_numbers.{md,json}.

## 11. Psychometric review + verification pass (final analysis gate)

An independent psychometric read of every table and figure returned:
publishable as the real-data half; calls endorsed (stability deprecation,
sign convention, run-averaged estimator) or qualified (MML coverage); one
wording rejected. The five verifications it ordered, resolved:

1. **MML prediction is NOT leaky.** Both scoring paths compute the ability
   for position t from strictly earlier responses (prefix EAP, verified in
   code). MML's EdNet-2PL AUC edge (.705 vs SK .676) stands, with the honest
   frame: its item parameters come from one global calibration that saw the
   evaluation learners' responses, so MML carries an in-sample
   item-parameter advantage the neural rows do not have.
2. **Equal-footing comparison added** (tab_real_common). On the 26.1% of
   option positions MML can score, MML edges SH on rank metrics only; SH
   ties its accuracy and additionally covers the 74% MML cannot score.
   True coverage is 26.1%; the record's earlier 34% counted failed
   calibrations and is corrected above.
3. **TIMSS discrimination is scoped out.** Learned vs classical
   discrimination correlates only ~.4 - yet SH and SK agree with each other
   at .97, so this is under-identification from six-step sequences, not a
   design artifact. The TIMSS parameter claim covers step thresholds and
   difficulty (.97/.95 vs classical); discrimination is excluded.
4. **Trajectory wording settled.** Smoothing raises the between-design
   agreement from .22 to .50 (window 20) and net trends agree at .56 -
   nothing reaches .6. Final sentence: part of the gap is step-level
   design-specific noise, but even smoothed paths agree only moderately, so
   fine-grained ability dynamics remain underdetermined by fit. The
   "prediction-equivalent" phrasing is retired (the EdNet pair differs in
   AUC); the finding stands as underdetermination.
5. **Figure corrections shipped:** error bars on the delta-delta figure
   (plus a bug fix in its NRM accuracy spread), labeled overflow bins on
   item fit, dataset label + shrinkage caveat on the information figure,
   shared y-axis on the GPCM trade-off row (2PL keeps per-panel axes with
   an explicit caption warning).

Standing instruction for the writing phase, from the review: tell the
SK-NRM prediction cost as a finding (SK loses ~6 accuracy points and pays
NLL vs SH on EdNet options while winning the agreement side); scope the
free-interpretability claim to synthetic + binary; use the attenuation
account (prediction-only training shrinks discrimination spread: SH slope
compression, the information scale gap, and TIMSS's narrow range are one
phenomenon at three strengths).

## 12. EdNet as a two-in-one case study (binary + nominal), the flip explained, and the completed design trio

**Cross-decoder coherence.** The 2PL and NRM readings of the same 250 items
order the bank the same way on difficulty: an item that is hard in binary
terms is one where the correct option overtakes the distractor mass only at
high ability (Spearman .87 SK / .91 SH, robust to the mapping choice).
Discrimination bridges only moderately (~.35), the campaign's
hard-to-identify parameter. One calibration, two resolutions, on the
difficulty axis. (ednet_coherence.{md,json})

**The NRM accuracy flip, tested.** Sparsity is real but explains only ~40%
of it: the SK deficit narrows with exposure (-.078 thin -> -.047 rich) yet
never closes, even on items with every option well sampled. The decisive
control: predicting each item's most popular option (no ability at all)
scores .653 -- SH sits AT that floor (.652), SK sits .061 BELOW it. Option
choice at the argmax level is nearly ability-independent; the shared head's
"win" is the popularity table wearing a model, and the separated key's
per-item freedom actively costs accuracy under a 4-way softmax at every
exposure. The calibration side is where data helps: the NLL gap shrinks 2.5x
with exposure. The 2PL control shows SK ahead in every exposure bin (the
exposure-narrowing is a general variance effect; the FLIP is the sign
difference plus its persistent asymptote). (flip_forensics.{md,json},
fig_flip_forensics)

**The design trio is complete** (TIMSS, EdNet-NRM, now EdNet-2PL:
fig_ednet_2pl_shsk). On the binary side the item map is design-robust
(between-design difficulty rho .998, discrimination .978), and the NRM-style
slope compression is ABSENT -- the shared head's discrimination distribution
is in fact wider (sd .72 vs .35); a 2PL item has one slope, nothing
within-item to squeeze. What separates the designs on binary is the person
side: final ability tracks raw score at r=.54 (SK) vs .36 (SH), and matched
learners' raw paths agree between designs only at r=.22 (.49 smoothed).
(ednet_2pl_shsk_numbers.{md,json})

**Agreement figure upgraded.** fig_agreement_both overlays both designs in
the same four scatters (replacing the dumbbell as the main display): item map
at parity (difficulty -.97 both; discrimination rho .67 both), person side
and option orientation favor SK (.54 vs .36; .705 vs .587). The EdNet 2-in-1
centerpiece (fig_ednet_2in1) carries the bridge panel: one item's binary
curve decomposed into which-distractor trace lines -- the resolution binary
KT discards. (fig_agreement_both, fig_ednet_2in1)
