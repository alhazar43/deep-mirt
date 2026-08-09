# E6: TIMSS raw (unsorted) thresholds — the F1 rerun verdict (2026-08-09)

Fresh 25-fold reruns of both TIMSS arms with the raw head output
exported alongside the sorted read (`beta_raw` in
`kt-irt/results/p2_realstudy_rawbeta/`; driver `--out-tag rawbeta`,
cluster job 559796; code kt-irt fc0e234).

## Numbers

| quantity | shared (SH) | separate (SK) | classical MML | paper (sorted export) |
|---|---|---|---|---|
| ordered fraction (mean over 25 folds) | .432 (sd .033) | .428 (sd .037) | .613 | 1.000 by construction |
| items disordered in >half the folds | 18/31 | 18/31 | 12/31 | 0/31 |
| overlap with the classical 12 | 12/12 | 12/12 | — | — |
| item-level flag agreement | 25/31 | 25/31 | — | — |
| Spearman(neural disorder rate, classical flag) | .813 | .813 | — | — |
| Spearman(mean raw order gap, classical gap) | **.979** | **.978** | — | — |

## Reading

1. The paper's §5.4 claim ("both designs retain ordered thresholds for
   every item") was enforced by the export sort (F1) and is falsified by
   the raw readout: the honest ordered fraction is ~.43, BELOW the
   classical calibration's .613.
2. The upgrade, not just the demotion: the raw thresholds are not noise.
   Their order structure tracks the classical calibration almost
   perfectly (gap correlation .98), and every one of the 12 classically
   non-modal items is recovered as stably disordered by the neural fits.
   The sort did not merely fabricate the ordered-thresholds table; it
   destroyed a true, classically-agreeing signal about category
   structure that the unconstrained head had learned.
3. Design-invariant to three decimals (SH = SK), consistent with the
   location-family robustness finding (dossier S3): threshold ORDER
   behaves like a location-family quantity, recoverable through pooling.
4. The neural fits over-flag six items beyond the classical twelve
   (18 vs 12). With per-fold sd .03 and stable-across-folds flags, this
   is systematic, not sampling noise; candidate causes are shrinkage
   differences and the epsilon-free unconstrained head. The honest
   sentence for the paper caps the claim at the gap correlation and the
   12/12 containment.

Author-facing rewrite consequence (dossier Part V, item 1): replace the
withdrawn ordered-thresholds paragraph with the raw-readout finding —
stronger, true, and it converts fatal F1 into an exhibit of exactly the
audit arc the paper argues (an eval-time convenience silently replacing
evidence; the evidence recoverable once the readout is honest).
