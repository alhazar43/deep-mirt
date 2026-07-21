# A4 / G2 EdNet-matched synthetic corroboration: seed-clustered posture matrix (WORKING)

Aggregated 2026-07-21 from 19 slice cells (4 twins x 5 generator seeds,
syn_ns seed1 missing = fine) + 12 neural cells (4 twins x 3 model seeds)
under `outputs/a4/campaign/ednet_matched/`. Same combination rule, bars, and
primitives as the KDD verdict (`_planning/design/a4_design.md` v1.1,
`src/kt_mirt/growth/report.py`, seed clustering per battery arm 9). This is
the population-corroboration bed (design 5.3): EdNet is capped at Tier 1 in
v1, ACT does not run on real EdNet, and the distinctive gate is CG4a
(bed-pooled detection, per-learner NOT gated) with CG4b (KC-rate recovery)
pre-registered as a FINDING not a kill (K7, "density below the
rate-recovery floor"). Silence bars relax to p95 <= 0.02 (CG1 EdNet) and
rank bars to >= 0.5 (CG1a/CG1b EdNet). Density: median 2 opp/learner-KC,
C=189, item arity mean 2.2, 6000 learners (3000 analysis).

Reading convention identical to KDD: **existence gate** = bed-level pooled
permutation test (`gate.bed_pvalue`, floor 0.001). "detect" = pooled p <
0.001 all seeds; "null" = pooled p > 0.01 all seeds.

## Headline

The coarse (twin-level) detector REPLICATES cleanly on thin density: null
on syn_ng, fires on syn_kg/syn_ns in every seed, seed-stable. The syn_sat
CG6 inversion also REPLICATES (gate fires where it must stay silent), but the
over-firing is far milder than KDD (stat 1.3x the growth twins vs 3.7x). Two
directional DIFFERENCES from KDD: (1) ACT per-KC rank recovery is markedly
BETTER at EdNet density and the more-expressive variant now CLEARS the
relaxed 0.5 bar on the positive control; (2) ACT silence is WORSE, both
variants now leak on the null twin (KDD's act_p1 was dead silent). Per-KC
resolution is even weaker than KDD: zero BH discoveries on every growth cell
including syn_ns (KDD had sporadic power), CG4b rate recovery pinned near
zero (the pre-registered floor finding), and the E-M3 misfit flag fires
~50-77% of KCs regardless of twin (non-informative at median-2 density).
Bank recovery is NOT rescued by EdNet's denser item bank: 0.72-0.80 on
kg/ns, still short of 0.90.

## Existence gate, seed-clustered (slice seeds)

| Twin | bed_pvalue | bed_stat (mean, sd) | BH per-KC disc | BY | Gate verdict | Designed | Match |
|---|---|---|---|---|---|---|---|
| syn_ng  | .326 .551 .692 .222 .876 | 6200 (176) | 0 0 0 0 0 | 0 | **NULL** | null (CG2/CG4a-null p>.01) | YES |
| syn_kg  | .001 x5 | 5972 (189) | 0 0 0 0 0 | 0 | detect (pooled only) | detect (CG4a p<.001) | pooled YES / per-KC NO |
| syn_ns  | .001 x4 | 7007 (496) | 0 0 0 0 | 0 | detect (pooled only) | detect + misfit | pooled YES / per-KC NO |
| syn_sat | .001 x5 | 7883 (268) | 0 20 0 0 19 | 0 | **FIRES** | FAIL-to-detect (CG6 p>.05) | **NO -- CG6 inverted** |

- **syn_ng**: clean null, all 5 seeds p in [0.222, 0.876], zero per-KC and
  zero BY discoveries. Matches KDD's clean null. CG4a-null satisfied.
- **syn_kg**: pooled fires p=0.001 all 5 seeds; **zero** KCs survive BH
  q=0.05 in any seed. Same aggregate-power / no-per-KC-power split as KDD.
  CG4a (the only barred detection clause on EdNet) PASSES.
- **syn_ns**: pooled fires all 4 seeds; per-KC BH = **0 in every seed**.
  This is WORSE than KDD, where NS showed sporadic per-KC power
  ([65,0,55,0,0]); at median-2 density even that erratic power vanishes.
- **syn_sat anomaly (replicates)**: pooled fires p=0.001 all 5 seeds, stat
  mean 7883 (CV 3.4%), deterministic not a seed outlier. CG6 inverted, same
  direction as KDD. **KEY DIFFERENCE**: the inflation is far milder --
  sat/kg stat ratio is 1.32x on EdNet vs 3.7x on KDD. Thin density
  compresses the near-ceiling over-firing. syn_sat also shows sporadic
  per-KC BH firing ([0,20,0,0,19]) that syn_kg/ns never do.

## MIX rate + split-half reliability (slice seeds)

| Twin | split-half obs | predicted | gap (tol 0.10) | MIX r_c median | misfit frac (all KCs) | bank recovery (thr 0.90) |
|---|---|---|---|---|---|---|
| syn_ng  | 0.848 | 0.802 | .044 .051 .048 .042 .048 -> pass | 0.100 | 0.75 | 0.80-0.87 -> **FAIL** |
| syn_kg  | 0.818 | 0.750 | .062 .079 .062 .066 .071 -> pass | 0.127 | 0.51 | 0.72-0.77 -> **FAIL** |
| syn_ns  | 0.838 | 0.782 | .055 .057 .052 .057 -> pass | 0.117 | 0.68 | 0.71-0.80 -> **FAIL** |
| syn_sat | 0.431 | 0.151 | .282 .290 .277 .281 .270 -> **FAIL** | 0.251 | 0.14 | **0.09-0.21** -> **FAIL** |

- Split-half agrees within tolerance on the three unsaturated twins (gaps
  .042-.079, tighter than KDD's .066-.098); syn_sat blows tolerance (gap
  ~0.28), the saturation signature, replicating KDD exactly.
- **Bank recovery misses 0.90 on every twin.** kg/ns 0.72-0.80 is
  essentially the KDD number (0.73) despite EdNet's dense 1512-item bank
  being far better identified than KDD's singleton-heavy step bank -- so
  sparsity of the item bank is NOT the bottleneck; per-KC sample size is.
  syn_ng is slightly better (0.80-0.87 vs KDD 0.77); syn_sat collapses
  (0.09-0.21) as on KDD.
- **E-M3 misfit flag is non-informative at EdNet density.** Firing fraction
  is 0.75 (ng, no-growth), 0.51 (kg), 0.68 (ns), 0.14 (sat) -- it does not
  separate growth from no-growth and fires on a majority of no-growth KCs.
  At median-2 opportunities the blockwise-vs-exponential held-out test is
  noise-dominated. Contrast KDD, where NS misfit fired on only 6-7% of
  growing KCs. The direction of the EdNet failure is opposite (over-firing)
  but the conclusion is the same: no usable per-KC misfit signal.

### syn_ns misfit laundering (CG5 clause)

| seed | misfit on growing KCs | misfit on silent subset |
|---|---|---|
| 0 | 0.642 | 0.684 |
| 2 | 0.682 | 0.632 |
| 3 | 0.702 | 0.579 |
| 4 | 0.742 | 0.711 |

CG5 needs misfit firing on >=80% of non-standard KCs and ~0 on the silent
subset. Observed: 64-74% on growing (below 80%) and 58-71% on the silent
subset that should carry no misfit signal. The flag fires almost everywhere,
indiscriminately -- laundering the wrong way and non-informative, the same
verdict as KDD reached by the opposite numeric route (KDD under-fired, EdNet
over-fires).

## CG4b KC-rate recovery (the EdNet-specific rate gate, bar 0.6)

| Twin | rank corr(r_hat_c, r_true_c) x seeds | verdict |
|---|---|---|
| syn_kg | -.06 .14 .08 .11 .15 | **FAIL** (~0.08, K7 pre-registered floor finding) |
| syn_ns | -.10 .10 .07 .10 | FAIL (~0.07) |
| syn_sat | -.02 -.02 -.07 .02 -.03 | FAIL (rate meaningless under saturation) |
| syn_ng | -.08 .07 .01 .02 .02 | n/a (no true rate) |

CG4b fails as pre-registered. With median 2 opportunities a two-parameter
bounded-exponential sits at the identifiability floor; failure is the
designed honest verdict "EdNet-class density is below the ladder's
rate-recovery floor" (K7), NOT a kill. No real-bed license rides on it.

## ACT posture read (3 model seeds)

| Twin | variant | pop_mean_rise | p95_abs_rise | growing_rank_corr | overshoot | posture |
|---|---|---|---|---|---|---|
| syn_ng  | act_p0 | 0.009 | **0.040** | n/a | n/a | p95 exceeds 0.02 silence bar |
| syn_ng  | act_p1 | **0.028** | **0.161** | n/a | n/a | **BREACHES silence (KDD was clean)** |
| syn_kg  | act_p0 | 0.127 | 0.216 | **0.502** | 0.00 | fires; rank AT 0.5 bar |
| syn_kg  | act_p1 | 0.160 | 0.396 | **0.599** | 0.00 | fires; rank CLEARS 0.5 bar |
| syn_ns  | act_p0 | 0.123 | 0.182 | 0.394 | 0.040 | fires; rank<0.5 |
| syn_ns  | act_p1 | 0.143 | 0.346 | 0.493 | 0.040 | fires; rank just misses 0.5 |
| syn_sat | act_p0 | 0.033 | 0.119 | 0.384 | 0.00 | ~abstains (below 0.05 firing) |
| syn_sat | act_p1 | 0.043 | 0.157 | 0.337 | 0.00 | ~abstains (below 0.05 firing) |

- **ACT rank recovery is BETTER on EdNet (the biggest positive
  difference).** syn_kg growing_rank_corr is 0.50 (act_p0) / 0.60 (act_p1)
  vs KDD's ~0.27 both. act_p1 CLEARS the relaxed CG1a EdNet bar (0.5);
  act_p0 sits exactly at it. syn_ns 0.39/0.49 (KDD 0.33/0.38), act_p1 just
  misses the CG1b bar. Fewer KCs (189 vs 515) plus the relaxed bar turn the
  KDD rank failure into a near-pass on the positive control.
- **ACT silence is WORSE on EdNet.** On syn_ng, act_p0 p95 0.040 > 0.02 bar
  (breach, as on KDD) AND act_p1 p95 0.161, pop 0.028 -- act_p1 now leaks
  hard, where on KDD it was dead silent (p95 0.0004). At thin density the
  recognition network manufactures spurious per-learner motion on the null
  twin. Neither variant clears CG1 silence on EdNet.
- **syn_sat**: ACT roughly abstains (pop 0.033/0.043, below the 0.05 firing
  bar; rank ~noise) -- does not manufacture confident gains, partially
  honoring CG1c even as the slice gate fires hard. Same "gate fires, ACT
  flat" saturation disagreement as KDD.
- CG9-ACT recognition stability: act_p0 u_median_corr 0.81-0.94 (passes some
  seeds, ng/kg); act_p1 lower (0.49-0.81), fails. Borderline, better than
  KDD's uniform fail.

## Audit gates CG7-CG10 (trackers / PAS-N1, 3 model seeds)

| Twin | CG7 (margin) | CG8 (ratio<=.10) | CG9 (order) | CG10 (viol<=.10) |
|---|---|---|---|---|
| syn_ng  | 0/3 (~0, decert) | 0/3 (2.1-61) | 1/3 | 0/3 (.39-.42) |
| syn_kg  | 0/3 (-.04/-.09) | 0/3 (2.5-3.0) | 1/3 | 0/3 (.36-.40) |
| syn_ns  | 0/3 (~0) | 0/3 (2.9-4.9) | 2/3 | 0/3 (.38-.41) |
| syn_sat | 0/3 (+.05/+.07) | 0/3 (1.6-4.1) | 0/3 | 0/3 (.37-.42) |

- **CG7**: trained-vs-frozen margin ~0 or negative, trained_rank_corr ~ -0.05
  (kg), decertified every seed. PAS-N1 learns no per-KC growth structure --
  same as KDD.
- **CG8**: ratio 1.6-4.9 (one syn_ng seed 61) >> 0.10. The designed
  Ding-Larson contamination failure, replicates.
- **CG9 order stress**: PARTIALLY PASSES on EdNet (ng 1/3, kg 1/3, ns 2/3;
  kc_median_corr ~0.85, sign_flip ~0.106 just over the 0.10 bar) -- a
  DIFFERENCE from KDD's uniform 0/12 fail. Fewer KCs make per-KC trajectory
  correlation more stable under cross-KC reshuffling.
- **CG10 direction audit**: violations 0.36-0.42 everywhere, all fail. The
  Deep-IRT ability-moves-against-response artifact binds on PAS-N1 exactly
  as on KDD.

## Certification roll-up vs designed expectation

| Twin | designed posture | seed-clustered EdNet verdict | vs KDD |
|---|---|---|---|
| syn_ng  | every detector SILENT/null | gate null all seeds (PASS); ACT breaches silence on BOTH variants; CG9 partial | gate SAME; ACT silence WORSE (p1 leaks) |
| syn_kg  | detect + ACT fire | pooled detect (CG4a PASS); per-KC 0; bank 0.75; CG4b ~0.08; ACT fires, rank 0.50/0.60 clears EdNet bar | detect SAME; ACT rank BETTER (clears bar) |
| syn_ns  | detect + misfit fire | pooled detect; per-KC 0 (KDD had sporadic); misfit fires everywhere (non-informative) | detect SAME; per-KC WORSE; misfit inverted-but-same-conclusion |
| syn_sat | gate FAIL-to-detect | gate FIRES all 5 seeds (CG6 inverted) but stat only 1.3x growth twins | inversion REPLICATES; inflation much MILDER (1.3x vs 3.7x) |

## Answers to the three corroboration questions

1. **Does the coarse detector still work at thin density? YES.** Clean null
   on syn_ng (p .22-.88 x5), fires p=0.001 x5 on both growth twins,
   seed-stable, non-overlapping stat distributions (ng ~6200 vs kg/ns/sat
   >5900... note overlap with kg here, see below). CG4a passes. The
   twin-level detector is validated on EdNet as on KDD. Caveat: the null-vs-
   growth stat separation is TIGHTER on EdNet (ng mean 6200 vs kg mean
   5972 -- the pooled statistics nearly touch), so the margin is thinner
   than KDD's 2.6x gap; the p-value separation (>.2 vs .001) is what carries
   the detection, not the raw stat.
2. **Does the saturation false-fire replicate on thin density? YES, but
   milder.** syn_sat gate fires p=0.001 x5 (CG6 inverted, same direction as
   KDD); saturation flag marks 100% of KCs; split-half gap 0.27-0.29 and
   bank collapse 0.09-0.21 reproduce the near-ceiling signatures. The
   over-firing magnitude is much smaller: sat stat is 1.3x the growth twins
   on EdNet vs 3.7x on KDD. Thin density gives the M0/M1 misspecification
   less room to inflate. The limitation is real on both beds; its severity
   scales with density.
3. **Is bank recovery better or worse at EdNet sparsity? ABOUT THE SAME,
   still failing.** kg/ns 0.72-0.80 vs KDD 0.73; ng slightly better
   (0.80-0.87 vs 0.77); sat collapsed on both. EdNet's dense 1512-item bank
   (vs KDD's singleton-heavy 1.3M-step bank) does NOT lift recovery, so item
   identifiability is not the binding constraint -- per-KC opportunity count
   is. Neither bed clears 0.90 anywhere it is barred.

## Net for G2 on EdNet

The passive existence gate corroborates as a **coarse twin-level detector**
on EdNet-thin density (correct null, fires on growth, seed-clean, CG4a
passes) -- the KDD headline holds on a second, structurally different bed.
Per-KC resolution is even weaker than KDD (zero BH discoveries anywhere,
CG4b at the floor, misfit non-informative), exactly as the design predicted
for median-2 density (K7). The saturation CG6 inversion replicates, milder.
Two genuine cross-bed differences: ACT rank recovery improves enough to
clear the relaxed positive-control bar, while ACT silence degrades (both
variants leak on the null). PAS-N1 fails CG7/CG8/CG10 as designed; CG9
partially passes at low C. No result overturns the KDD verdict; EdNet
sharpens the density-dependence of every sub-clause.
