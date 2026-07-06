# Real-data study: pre-registration (Gate A)

Committed BEFORE running any real-data cell. The defense of this study is that
the metric suite and panel are fixed by principle up front and every cell is
reported, whatever it shows. No metric or dataset is chosen after seeing
results. The word "recovery" is never used on real data (no ground truth
exists); we measure agreement, detection, and cost.

## Instruments (fixed, chosen by principle -- not by which flatters the fix)

1. **Concordance with classical MML (PRIMARY).** Spearman between each head's
   item parameters and a classical GPCM/2PL MML calibration of the same
   responses. Rationale: classical MML is the field's reference calibration and
   the closest truth-adjacent standard. Pre-declared hypothesis: the separate
   head concords with classical more than the shared head does.
2. **delta transfer (deployable).** The truth-free refit discrepancy delta on
   each deployed model, ranked against classical disagreement. Rationale: delta
   is the only diagnostic that runs in the field. Hypothesis: delta flags the
   shared heads and clears the separate ones.
3. **Prediction accuracy parity.** Shared vs separate held-out accuracy.
   Rationale: verifies the no-cost claim survives on real data.
4. **Cross-class divergence.** Top-item overlap between shared and separate.
   Rationale: shows the design choice changes the item conclusions.
5. **Split-half reliability (CAUTIONARY ONLY, not primary).** Reported with the
   explicit caveat that reliability rewards smoothing and is blind to systematic
   bias, so a distorted-but-consistent parameter passes it. It is included to
   demonstrate that failure mode, not to rank the designs.

## Panel (fixed)

- **Binary (2PL), native, no coercion:** EdNet, KDD Cup 2010 algebra,
  ASSISTments 2009, ASSISTments 2017 (all native binary correctness). (Add others only if declared here first.)
- **Ordinal (GPCM), genuine partial credit, no coercion:** TIMSS 2019 Grade 8
  (IEA public database), human-rater constructed-response rubric scores
  (0 incorrect / 1 partial / 2 full). Pilot confirmed usable: USA sample =
  5,135 learners x 31 K=3 items, classical GPCM (mirt/MML) converged, CR
  sequence ~35 items/learner. Scale by adding countries (same URL pattern).
  Ship a download+parse script, not the raw .sav (IEA redistribution
  restriction); cite DOI 10.58150/IEA_TIMSS_2019_G8. R 4.5.0 required for mirt
  (conda env R 3.6.1 too old).
- **Nominal (NRM), native option-level, no coercion:** EdNet, the learner's
  chosen 4-option distractor (EdNet records the selected option, not just
  correct/incorrect). Confirmed available: 3,744 four-option items, 3,000
  learners, K=4. Classical reference = Bock nominal-model MML.
- **Robustness anchor (genuine rubric-ordinal, off-KT):** ASAP essay set
  (already in repo), reported as a separate check that the shared-vs-separate
  gap is not an artifact of coerced ordinal coding.

## Protocol

- Encoders: LSTM (reference) and DKVMN (the shipped architecture).
- Seeds/folds: fixed at the campaign default (5 data seeds x 5 folds),
  seed-clustered 95% intervals.
- Run each cell once. Report EVERY cell. Heterogeneity across datasets is a
  finding, not grounds for omission.
- State-conditioned (dynamic) heads are barred from this study; static readouts
  only. (Rationale held privately; a dynamic add-on is a simple rerun if ever
  needed.)

## Gate

Gate A (this metric suite + framing) is cleared before the PISA/TIMSS data
preparation and the mass run. No real-data cell is run until this file is
committed.
