# E7: E9 misspecification battery — verdict (2026-08-10)

The dossier's one bought experiment (Part V item 17), complete:
2500/2500 units (7 violation families, control + doses, lstm +
transformer, 2PL + GPCM, SH + SK, 5 seeds x 5 folds), zero failures.
Local GPU 2299 units + cluster 201, converged through the shared store.
Full tables: `kt-irt/results/p2_misspec/battery_report.{md,json}`;
per-unit JSONs with raw thresholds and tier-1 weights in the same store.

## Pre-registered question

Does the seed-clustered SK-SH discrimination-recovery delta remain
positive under violations? **Yes. Positive in 49 of 50 cells with 5/5
seeds and paired t 2.3-18.4; it never reverses.** The single exception
is extreme exposure imbalance (dose 1.5, lstm-2PL): +.029, t=1.3, 3/5 —
a tie where BOTH arms crater (.56 vs .59), the exposure-starvation
floor the campaign's own exposure law predicts, not a reversal.

## The stronger findings inside the answer

1. **The dissociation survives every violation.** |d(acc)| <= .011 in
   all 50 cells while recovery deltas run +.02 to +.34. Prediction
   accuracy is uninformative about parameter quality under EVERY
   misspecification tested — the paper's core claim, now with
   robustness evidence.
2. **The repair's value GROWS under the violations most feared.**
   Local dependence (B3's predicted reversal case): the delta RISES
   with dose (lstm-2PL +.059 -> +.124 -> +.246). Threshold disorder:
   +.083 -> +.123 -> +.228 (lstm), +.216 -> +.267 (transformer,
   t=18.4). The wide key does not absorb nuisance structure into the
   slopes; the shared path degrades faster.
3. **The audit instrument works under misspecification.** Across all
   50 cells the truth-free refit discrepancy tracks TRUE slope
   corruption at Spearman .93 (SH) / .72 (SK), and rises with dose
   within every family. The truth-free audit is a certified detector
   under the full violation battery — the evidence that makes it the
   paper's transferable contribution.
4. **Drifting theta (the KT-central worry): robust.** No reversal at
   any dose; narrows to +.02 (still t=3.5, 5/5) only for lstm-2PL at
   sigma .3; transformer deltas stay +.24 to +.31.
5. **Internal consistency:** dose-0 cells reproduce the campaign's
   known encoder pattern on an independent exposure draw (transformer
   SH .44-.66 vs SK .75-.89; lstm SH .77-.86 vs SK .83-.93).

## Caveats to carry into the paper

- Extreme exposure imbalance is the boundary: under starvation both
  arms fail and the key stops paying (consistent with the exposure
  law; state it, do not hide it).
- SH shows slightly better last-step theta than SK in most lstm cells
  (~.03-.08); the smoothing-helps-ability observation, one sentence
  alongside the prediction-is-not-measurement axis.
- Mild-only families (DIF, response style, noisy thresholds) were run
  at control + mild on lstm only; claims there are scoped accordingly.

## Paper consequence

Dossier M10 (zero misspecification evidence) is closed with the strong
outcome: "the repair is robust across the violation battery AND the
audit flags what violations corrupt." The scope sentence of the
synthetic claims can now cite dose-response tables instead of a
correct-specification caveat.
