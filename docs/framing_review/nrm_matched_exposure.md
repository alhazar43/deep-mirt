# EdNet-NRM diagnosed and closed out (2026-08-10)

The author flagged the real nominal cell as "very weird" and asked
where the problem lives: dataset, sampling, or loading. Answer: a
dataset-construction asymmetry plus a broken anchor; no bug. Then the
follow-up cell that settles what the format actually does.

## Diagnosis (three probes, all committed)

1. **Geometry (the root cause).** The canonical NRM cell loads the FULL
   4,220-item option bank (the 2PL cell restricts to the top 250):
   285k observations, 8 parameters per item, median **5.1 responses per
   parameter** (other real cells: ~300-530). 30% of items have <30
   responses. The cell was the starvation boundary by construction.
2. **Key and loading: clean.** Correct-option rate .632 (matches
   accuracy); 1.9% of well-observed items below chance (plausibly
   source-miskeyed); no alignment or loader fault.
3. **The MML anchor is nearly empty.** `converged: TRUE`, but estimates
   exist for **395 of 4,220 items (9.4%)**. Every concordance printed
   for this cell (.13-.43, both arms) is against 395 noisy small-n NRM
   fits; it is not a usable external standard.

Under starvation the observed pathology follows: SH's shared channel
shrinks 33,760 slopes toward a popularity pattern (stable, weakly
informative); SK's per-item key is unshrunken at ~5 obs/param
(fold-level variance explodes; the audit's per-item re-estimate is
itself noise, delta .78). Both arms display the exposure floor the
battery found, in opposite styles. (Per E4, part of SK's fold
instability is the nominal likelihood's near-symmetric optima, not
sampling noise alone.)

## The matched-exposure cell (new; `kt-irt/results/p2_nrm250`)

Same routed head, same fold protocol, same seeds, same metrics
(`_p2_nrm_matched.py`); only the bank changes: **top-250 option-rich
items x 8,493 learners = 191 responses per parameter** (the regime of
the other cells). 50 fits, zero failures, frozen stores untouched.

| readout | starved bank (5.1 r/p) | matched bank (191 r/p) |
|---|---|---|
| audit delta SH / SK | .408 / .454 | **.239 / .325** |
| cross-fit stability SH / SK | .400 / .492 | **.762 / .780** |
| SH~SK agreement (seed-mean) | .772 | .600 |
| empirical anchor agreement SH / SK | — | **.437 / .705** |

(Cross-fit stability = mean pairwise Spearman of the keyed contrast
across all 25 fits; this differs from the frozen table's WITHIN-fit
split-half, which is why SH could show .93 there and .40 here. The
empirical anchor = per-item keyed contrast of option point-biserials
against a leave-one-out correctness proficiency proxy, computed from
raw data with no model; 250/250 items covered.)

## Verdict

1. **The canonical cell's chaos was regime, not format and not a
   bug.** At matched exposure both arms stabilize (~.77) and the audit
   calms into the healthy cells' range.
2. **The design divergence is real and now interpretable.** Stable
   arms that agree only at .60 are reading genuinely different slope
   structures. The starved bank's higher agreement (.77) was two
   noisy/shrunken readouts sharing popularity structure.
3. **The external anchor sides with the separated key: .705 vs .437.**
   At the first exposure level where anyone can read anything, SK
   tracks the observable option-discrimination structure far better
   than SH -- the real-data nominal result finally aligns with the
   synthetic story, and with the paper's own earlier distractor
   point-biserial finding (.75 vs .60), now confirmed clean of the
   starvation confound.

## Caveats

Single encoder (lstm) so far; the anchor is a proxy (linear
point-biserial against a correctness score), not truth; both deltas
remain above the synthetic alarm threshold .152 (no real readout is
fully clean -- consistent with the study-wide finding); accuracies are
not comparable across banks (different item mix); the canonical cell
stays in the paper as the honestly-labeled starvation boundary, with
this cell as the regime where the format question is answerable.

Paper consequence: the study's weakest exhibit ("everything nominal is
suggestive") is replaced by a directional real-data result for the
slope family, and the two NRM cells together become a real-data
demonstration of the exposure law the battery established.
