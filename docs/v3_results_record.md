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
| G3 | Export pass (small, first — unblocks case studies) | LSTM SK real cells: TIMSS-GPCM full step thresholds across 25 folds + θ trajectories/final-θ (5 seeds); EdNet-NRM θ trajectories (slopes+intercepts already on disk 4220x4x25); EdNet-2PL θ for completeness | pending |
| G4 | Extended prediction metrics (small) | tab:real metric columns: AUC+NLL (binary), NLL (ordinal, QWK exists), option-acc+macro-F1+NLL (nominal); neural rows first, MML rows if cheap via the EAP predictor | pending |
| G1 | Transformer width sweep | TF-SH GPCM+2PL at emb w∈{16,32,64,96}, N=2000 Q=200, 25 folds/cell (W=8 anchor + TF-SK point already in outputs/p2_toggle N2000 cells) | pending |
| G2 | DKVMN width sweep (reduced) | DKVMN-SH GPCM+2PL at w∈{16,32,64,96}, N=2000, 5 seeds x 1 fold (DKVMN ~10 min/fold; anchors from toggle N2000 cells) | pending |
| G5 | Optional, if the night allows | Extend DKVMN realstudy cells from 1 to 5 folds for tab:real credibility | pending |

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
- RESULTS: (pending)

## 2. tab:real fills (paper TODOs, lines ~1201-1210)

From existing disk data (no training): TF-SH/TF-SK rows (realstudy
transformer cells, 25 folds); DKVMN-direct TIMSS .577 / NRM .565 (direct
chain, reduced); DKVMN-SH/DKVMN-SK rows (n=1 folds — either caveat or wait
for G5). Extended metric columns after G4.
- RESULTS: (pending)

## 3. Export-pass artifacts (G3)

Target files: per-fold thresholds (TIMSS), per-learner θ trajectories +
final θ (TIMSS + EdNet). Feeds case studies (category/expected-score curves,
learner trajectories, ability bands) and threshold-stability rows.
- RESULTS: (pending)

## 4. Real-data parameter stability (post-hoc, no GPU)

Split(fold)/seed/exposure-stratified Spearman per parameter group, SK design
(gate framing per plan T1; no SH-vs-SK stability contest). Groups: TIMSS
discrimination + step thresholds; EdNet option slopes + intercepts; EdNet-2PL
difficulty + discrimination.
- RESULTS: (pending)

## 5. Case-study analyses (post-G3)

TIMSS: category-probability curves, expected-score curves, threshold
distribution, 2-3 learner trajectories. EdNet: option-probability curves,
correct-option slope-orientation distribution, distractor attractiveness by
low/mid/high θ band, intercept-vs-option-frequency, binary-vs-nominal
comparison.
- RESULTS: (pending)

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
