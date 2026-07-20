# A4 / G2 KDD-matched synthetic certification: seed-clustered posture matrix (WORKING)

Aggregated 2026-07-20 from 20 slice cells (4 twins x 5 generator seeds) +
12 neural cells (4 twins x 3 model seeds) under
`outputs/a4/campaign/kdd_matched/`. Combination rule and bars from
`_planning/design/a4_design.md` v1.1 (CG1-CG10, section 5); primitives from
`src/kt_mirt/growth/report.py`. Seed clustering per battery arm 9: a claim
must be sign-consistent in every seed AND significant seed-pooled; verdicts
tightened, never loosened.

Reading convention: the **existence gate** is the bed-level pooled
permutation test (`gate.bed_pvalue`, floor 0.001). "detect" = pooled
p < 0.001 all seeds; "null" = pooled p > 0.01 all seeds. Per-KC BH
discoveries (`bh_reject`) are the CG3/CG5 detection-power clause. Bank
recovery, split-half, misfit, and the ACT/tracker gates are the remaining
certification sub-clauses.

## Headline

The gate DIRECTION matches design on three of four twins (NG null, KG
detect, NS detect) but INVERTS on SYN-SAT (fires hardest exactly where it
must stay silent). Beyond direction, NO growing/saturated twin clears its
full certification: per-KC BH power, bank recovery, misfit firing, and ACT
per-KC rank recovery all miss their bars in every seed. Only SYN-NG passes
its designed posture outright. The audit gates CG7-CG10 fail on all four
twins (CG8/CG9 failures on PAS-N1 are the *designed* Ding-Larson finding;
CG7's near-zero trained recovery and CG10's reconstruction violations are
not).

## Existence gate, seed-clustered (5 slice seeds)

| Twin | bed_pvalue x5 | bed_stat (mean) | BH per-KC disc x5 | Gate verdict | Designed | Match |
|---|---|---|---|---|---|---|
| syn_ng  | .268 .640 .128 .973 .743 | 758 | 0 0 0 0 0 | **NULL** | null (CG2 p>.01) | YES |
| syn_kg  | .001 x5 | 2960 | **0 0 0 0 0** | detect (pooled only) | detect (CG3 p<.001 AND >=60% KCs) | pooled YES / per-KC NO |
| syn_ns  | .001 x5 | 2882 | 65 0 55 0 0 | detect (pooled only) | detect (CG5 p<.001) | pooled YES / per-KC erratic |
| syn_sat | .001 x5 | **10955** | 0 0 0 0 0 | **FIRES** | FAIL-to-detect (CG6 p>.05) | **NO -- CG6 inverted** |

- **SYN-NG**: clean null, all 5 seeds p in [0.128, 0.973], zero per-KC
  discoveries, zero BY discoveries. CG2 satisfied. Single-seed read
  (p=0.268) confirmed.
- **SYN-KG anomaly (new, not in single-seed read)**: pooled fires p<0.001
  all seeds, but **zero** KCs survive BH q=0.05 in any seed (min per-KC
  kc_pvalue = 0.005; BH threshold for rank 1 of 515 is 9.7e-5). CG3's
  ">=60% of unsaturated KCs discovered" clause fails at 0%. The gate has
  aggregate power but no per-KC power at KDD density.
- **SYN-NS**: pooled fires all seeds; per-KC BH erratic (55-65 in seeds
  0,2; zero in seeds 1,3,4). Detection direction right, per-KC unstable.
- **SYN-SAT anomaly (consistent)**: pooled fires p=0.001 all 5 seeds,
  bed_stat mean 10955, sd 207, **CV 1.9%**, range [10626, 11173] --
  ~3.7x the KG/NS statistic. This is deterministic, not a seed outlier.
  Design line 697: gate firing robustly on SYN-SAT = "certification
  failure requiring diagnosis, not a power bonus". Single-seed read
  (p=0.001, stat 11039) confirmed and shown seed-stable.

## MIX rate + split-half reliability (5 slice seeds)

| Twin | split-half obs (mean) | predicted (mean) | gap x5 (tol 0.10) | MIX r_c median | misfit frac (all KCs) | bank recovery (thr 0.90) |
|---|---|---|---|---|---|---|
| syn_ng  | 0.806 | 0.722 | .073 .078 .087 .081 .097 -> pass | 0.129 | 0.11 | 0.77 x5 -> **FAIL** |
| syn_kg  | 0.753 | 0.663 | .076 .094 .098 .094 .086 -> pass | 0.154 | 0.11 | 0.73 x5 -> **FAIL** |
| syn_ns  | 0.774 | 0.700 | .066 .077 .074 .068 .085 -> pass | 0.145 | 0.08 | 0.73 x5 -> **FAIL** |
| syn_sat | 0.437 | 0.159 | .259 .274 .292 .288 .276 -> **FAIL** | 0.375 | 0.21 | **0.08** x5 -> **FAIL** |

- Split-half agrees within tol on the three unsaturated twins (observed
  systematically ~0.08 above predicted, over-reliable but inside 0.10);
  SYN-SAT blows the tolerance (gap ~0.28), the expected saturation
  signature.
- **Bank recovery misses the 0.90 CG3/CG5 bar on every twin** (0.72-0.77
  unsaturated; 0.08 saturated). The single shared frozen artifact every
  posture reads through does not recover rank-order at KDD sparsity.
  This alone blocks CG3 and CG5.
- **SYN-NS misfit clause fails**: CG5 needs misfit firing on >=80% of
  non-standard KCs; observed 6-7% on growing KCs (25-29/412) and *higher*
  (14-22/103) on the silent subset that should carry no misfit signal --
  laundering in the wrong direction, and far below 80%.

## ACT posture read (3 model seeds)

| Twin | variant | pop_mean_rise | p95_abs_rise | growing_rank_corr | overshoot | posture |
|---|---|---|---|---|---|---|
| syn_ng  | act_p0 | 0.0064 | **0.044** | n/a | n/a | p95 exceeds 0.01 silence bar |
| syn_ng  | act_p1 | 0.0001 | 0.0004 | n/a | n/a | silent (clean) |
| syn_kg  | act_p0 | 0.0435 | 0.115 | 0.28 | 0.00 | below 0.05 firing bar |
| syn_kg  | act_p1 | 0.0574 | 0.159 | 0.27 | 0.00 | fires (>=0.05) |
| syn_ns  | act_p0 | 0.060 | 0.146 | 0.33 | 0.022 | fires; rank<0.5 |
| syn_ns  | act_p1 | 0.074 | 0.173 | 0.38 | 0.027 | fires; rank<0.5 |
| syn_sat | act_p0 | 0.0105 | 0.069 | 0.07 | 0.00 | ~abstains (below firing) |
| syn_sat | act_p1 | 0.023 | 0.094 | ~0.17 | 0.00 | ~abstains |

- **act_p1 separates cleanly** (silent 0.0001 on NG, fires 0.057 on KG);
  **act_p0 sits in a muddy middle** (p95 0.044 > 0.01 silence bar on NG,
  yet pop 0.0435 < 0.05 firing bar on KG) -- the *primary* pinned-lambda
  variant neither fully silences nor fires, the extension does. Contrary
  to the usual "extension is riskier" intuition.
- **CG1a positive-control rank clause fails both variants**: growing_rank
  _corr ~0.27 (KG) / ~0.33-0.38 (NS) vs bars 0.6 (CG1a) / 0.5 (CG1b). ACT
  fires in aggregate but cannot rank per-KC rises anywhere.
- On SYN-SAT ACT roughly abstains (pop 0.01-0.028, below firing, rank
  ~noise) -- it does NOT manufacture confident gains, partially honoring
  CG1c intent, even as the slice gate fires hard. That is the "gate fires,
  ACT flat" disagreement row, here a saturation artifact.
- CG9-ACT recognition-stability: `passed=false` every seed (u_median_corr
  ~0.5-0.66 vs 0.9 bar), though rise_profile_min_corr clears 0.95.

## Audit gates CG7-CG10 (trackers / PAS-N1, 3 model seeds)

| Twin | CG7 (margin) | CG8 (ratio<=.10) | CG9 (order) | CG10 (viol<=.10) |
|---|---|---|---|---|
| syn_ng  | 0/3 (~0) | 0/3 (3.3-4.0) | 0/3 | 0/3 (.42-.47) |
| syn_kg  | 0/3 (~0/-.02) | 0/3 (3.3-4.2) | 0/3 | 0/3 (.21-.42) |
| syn_ns  | 0/3 (~0) | 0/3 (2.6-4.2) | 0/3 | 0/3 (.42-.45) |
| syn_sat | 0/3 (-.01/-.08) | 0/3 (1.4-2.1) | 0/3 | 1/3 (.03/.20/.43) |

- **CG7**: trained-vs-frozen margin ~0 and trained_rank_corr ~ -0.03 (KG)
  on every seed -- the shared-state tracker PAS-N1 learns **no** per-KC
  growth structure; the fit is decertified (`detail.decertified=true`).
- **CG8 / CG9 failures are the DESIGNED finding**: PAS-N1 is the config
  Ding-Larson predict fails contamination and order-invariance (design
  2.2, battery arms 4-5); its failure is a result, not a bug. CG8 ratio
  1.4-4.2 >> 0.10; CG9 kc_median_corr ~0.11, sign-flip ~0.30.
- **CG10 reconstruction violations** 0.20-0.47 (one lone SYN-SAT seed at
  0.028) -- the Deep-IRT ability-moves-against-response artifact is present
  and NOT designed away; it binds on PAS-N1.

## Certification roll-up vs designed expectation

| Twin | designed posture | seed-clustered verdict | certified? |
|---|---|---|---|
| syn_ng  | every detector SILENT/null | gate null all seeds; act_p1 silent; act_p0 p95 marginal; CG2 pass | **PASS** (act_p0 p95 flag) |
| syn_kg  | PASSIVE detect + ACT fire | pooled detect; per-KC 0%; bank 0.73; ACT rank 0.27; act_p1 fires | **CG3 FAIL** (per-KC, bank, rank) |
| syn_ns  | detect + misfit fire + r withheld | pooled detect; misfit 6%; bank 0.73; per-KC erratic | **CG5 FAIL** (misfit, bank) |
| syn_sat | gate FAIL-to-detect | gate FIRES hard all 5 seeds (stat ~11000) | **CG6 INVERTED** (anomaly) |

## Disagreements vs the 2026-07-20 single-seed read

The single-seed read reported only bed_pvalue and got all four DIRECTIONS
right; the seed cluster reverses none of them. New information lives in the
sub-clauses invisible to a single bed_p:
- KG: per-KC BH = 0 in ALL 5 seeds and bank recovery 0.73 -- CG3 not met
  despite pooled detect.
- NS: per-KC BH erratic ([65,0,55,0,0]); misfit only 6% of growing KCs.
- SAT: the firing is CONFIRMED CONSISTENT (CV 1.9%, all 5 seeds), so the
  anomaly is a deterministic gate malfunction under saturation, not a
  seed artifact. Diagnosis required before any real-bed interpretation
  (positive-control-first ordering, design line 671).
