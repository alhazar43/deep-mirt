# Exposure rerun results, 2D N x Q grid

Raw record. Config in, numbers out, minimal interpretation. One doc.

Bed: N in {500,1000,2000,5000} x Q in {200,500,1000,2000}, L=60 fixed, spiraled admin
(exact E = N*60/Q), SHARED embedding, full 3x3 {lstm,dkvmn,transformer} x {2pl,gpcm,nrm},
5 data seeds x 5 folds, corrected last-valid theta, rho primary. Single RTX 4060.
Workflow budget-grid-2d-phase1 (w7cacdtgs).

STATUS: Phase 1 COMPLETE (144 cells, 3600 slope folds). Gate = PERSIST (3/3).
Phase 2 (toggle study) LAUNCHED (see bottom). All numbers below recomputed from grid_ folds.

## 1. Slope / discrimination rho surface (median/25 folds; rows N, cols Q=200/500/1000/2000)
```
lstm/2pl      lstm/gpcm     lstm/nrm      | dkvmn/2pl     dkvmn/gpcm    dkvmn/nrm
N500 .384 .253 .201 .083   .412 .283 .184 .156   .280 .321 .385 .233 | .378 .236 .163 .097   .632 .406 .248 .158   .682 .618 .598 .535
N1k  .458 .319 .235 .230   .531 .459 .249 .302   .554 .368 .464 .416 | .578 .343 .236 .185   .645 .542 .485 .325   .711 .689 .651 .596
N2k  .555 .391 .262 .294   .703 .689 .462 .429   .545 .364 .552 .584 | .780 .436 .365 .256   .861 .869 .797 .569   .841 .810 .792 .710
N5k  .615 .577 .465 .441   .790 .828 .752 .743   .535 .659 .731 .705 | .856 .779 .673 .644   .920 .909 .878 .869   .932 .924 .895 .877

transformer/2pl              transformer/gpcm             transformer/nrm
N500 .096 .107 .063 .064     .266 .123 .091 .085     .291 .331 .155 .108
N1k  .257 .203 .142 .122     .310 .244 .172 .153     .439 .444 .413 .271
N2k  .360 .299 .202 .179     .433 .317 .227 .230     .504 .543 .555 .552
N5k  .432 .319 .258 .299     .637 .588 .612 .563     .689 .697 .701 .651
```

## 2. Additive (difficulty b / intercept c_k) rho surface (same layout)
```
lstm/2pl      lstm/gpcm     lstm/nrm      | dkvmn/2pl     dkvmn/gpcm    dkvmn/nrm
N500 .723 .698 .658 .396   .811 .763 .757 .679   .382 .348 .289 .130 | .581 .562 .617 .381   .798 .778 .721 .680   .584 .537 .477 .287
N1k  .627 .747 .738 .687   .813 .800 .806 .768   .481 .408 .401 .348 | .591 .659 .679 .602   .830 .800 .779 .750   .647 .592 .581 .549
N2k  .767 .778 .782 .788   .834 .832 .854 .825   .538 .554 .555 .506 | .666 .667 .717 .729   .824 .845 .824 .820   .643 .778 .768 .739
N5k  .658 .760 .777 .843   .863 .861 .857 .869   .548 .598 .676 .636 | .617 .729 .714 .738   .862 .855 .859 .848   .765 .811 .795 .834

transformer/2pl              transformer/gpcm             transformer/nrm
N500 .442 .413 .405 .308     .709 .674 .649 .634     .319 .292 .195 .163
N1k  .535 .595 .606 .476     .745 .727 .712 .715     .479 .414 .375 .373
N2k  .615 .667 .787 .721     .766 .785 .779 .778     .541 .575 .600 .554
N5k  .638 .749 .802 .804     .815 .827 .855 .842     .640 .693 .756 .777
```

## 3. Exposure axis (gate-verified; adjudicators recomputed all 144 cells, 0 discrepancy)
Deficit = additive rho - multiplicative rho (positive = multiplicative recovers worse).
- Global-richest cell (E=1500, Q=200, N=5000), pooled over encoders: 2pl 0.035, gpcm 0.053, nrm 0.000.
- PER ENCODER at that cell: transformer +0.207 (2pl) / +0.155 (gpcm) PERSISTS; lstm +0.068 / +0.035 small; dkvmn -0.170 / -0.030 REVERSES (slope beats thresholds).
- Thinner-column tops (still N=5000): 2pl 0.19/0.30/0.36 (Q500/1000/2000); gpcm 0.10/0.12/0.14.
- Gap declines MONOTONICALLY with exposure (no true plateau). Multiplicative rho keeps climbing with N (gpcm Q=1000: 0.18 -> 0.31 -> 0.49 -> 0.73 as N 500->5000). N (data quantity) governs.

## 4. Linking axis (gate-verified) -- NO penalty
Along constant-E anti-diagonals (same E via different N,Q), multiplicative recovery IMPROVES
toward the larger sparse bank (deltas, pooled): E=150 2pl +0.15 / gpcm +0.27 / nrm +0.30;
E=300 +0.02 / +0.23 / +0.28; E=600 -0.01 / +0.07 / +0.08. No linking penalty for any decoder.
Coverage L/Q does not govern recovery; N does.

## 5. Diagnostics
E < 300 K=4 cells (gpcm/nrm at high Q, low N) are outer-category-starved: read the slope, distrust
the thresholds there. This understates additive rho, so it understates the gap -- it does not rescue vanish.

## 6. Timing (median fit_time_s; n_params)
```
enc          dec   nparams   N500   N1000  N2000  N5000
lstm         2pl   10467      0.5    0.7    0.9    1.5
lstm         gpcm  10501      0.8    1.2    1.8    3.1
lstm         nrm   10537      0.4    0.5    0.8    1.9
dkvmn        2pl    8907      8.8   10.4   16.1   30.7
dkvmn        gpcm   8941     11.3   14.2   25.3   42.5
dkvmn        nrm    8977     10.5   13.5   24.5   45.7
transformer  2pl   46403      0.6    1.3    1.8    3.4
transformer  gpcm  46437      1.0    2.0    3.0    6.2
transformer  nrm   46473      0.6    1.4    2.3    4.9
```
dkvmn is the SMALLEST in params (~9k) yet 10-30x slower than lstm (~10k); transformer is the
biggest (~46k) yet fast. The cost is architectural (dkvmn memory read/write), not size.

## 7. Gate decision: PERSIST (3/3) -- NARROW, exposure-axis-driven, architecture-scoped
- The multiplicative < additive deficit is an lstm+transformer, LOCATION-SCALE (2pl/gpcm) effect.
- VANISHES for nrm (symmetric per-option a_k/c_k co-recover, |gap| <= 0.04 everywhere).
- REVERSES for dkvmn (slope recovered BETTER than thresholds). THREAT to the headline; must be
  explained before publication.
- Reads as data-limitation / identifiability (N-governed, monotonic in exposure, no plateau),
  NOT a linking effect. The toggle study targets the exposure/identifiability axis and the
  location-scale-vs-symmetric parameterization asymmetry.

## Phase 2 (toggle / decoupling study) -- COMPLETE
Bed: spiraled, Q=200, N-axis; decoupled = discrimination on its own wide item key; dataset-level mean +/- SE.

### 2a. Decoupling closes the deficit (STATIC; a = discrimination rho, gap = difficulty b - a)
```
enc/dec/N        a_shared  a_decoupled  delta_a   gap_shared  gap_decoupled
lstm 2pl  500      0.368      0.694      +0.327      0.313        0.209
lstm 2pl 1000      0.465      0.817      +0.353      0.213        0.126
lstm 2pl 2000      0.553      0.898      +0.344      0.170        0.060
lstm 2pl 5000      0.613      0.946      +0.333      0.068        0.011
lstm gpcm 500      0.433      0.853      +0.420      0.347        0.078
lstm gpcm 1000     0.554      0.905      +0.351      0.239        0.047
lstm gpcm 2000     0.719      0.941      +0.223      0.108        0.024
lstm gpcm 5000     0.810      0.964      +0.155      0.035        0.011
transf 2pl 500     0.135      0.539      +0.404      0.350        0.347
transf 2pl 1000    0.264      0.698      +0.433      0.303        0.232
transf 2pl 2000    0.373      0.806      +0.433      0.231        0.149
transf 2pl 5000    0.416      0.911      +0.495      0.207        0.059
transf gpcm 500    0.271      0.658      +0.386      0.433        0.227
transf gpcm 1000   0.337      0.798      +0.461      0.401        0.121
transf gpcm 2000   0.438      0.900      +0.463      0.330        0.047
transf gpcm 5000   0.671      0.953      +0.282      0.155        0.018
```
- delta_a = +0.16 to +0.50 EVERYWHERE. Decoupling lifts discrimination to ~0.91-0.96 by N=5000, reaching parity with difficulty (gap 0.01-0.06).
- 2PL (low info on a): the fix is ARCHITECTURAL. Shared plateaus far below ceiling even at N=5000 (lstm ~0.61, transformer ~0.42); decoupled races to 0.91-0.95. DATA DOES NOT CURE the shared deficit; only decoupling does (delta flat for lstm, GROWS for transformer).
- GPCM (more info on a): data partly catches up (shared lstm 0.43 -> 0.81), delta shrinks (+0.42 -> +0.16), but a residual architectural gap persists at N=5000 (0.81 vs 0.96).

### 2b. Dynamic head (static -> dynamic, delta on a)
- SHARED arm: +0.05 to +0.23 (largest transformer). Minor second-order lever, far below decoupling.
- DECOUPLED arm: ~null for lstm (|delta| <= 0.013), fading for transformer. Once decoupled, static vs dynamic barely matters.
- CAVEAT: the DGP discrimination is STATIC, so the dynamic head recovering a static truth reads as extra head capacity, not temporal signal.

### 2c. NRM 10-config (lstm, pooled over N): slope a_k / intercept c_k
```
coupling        static a_k / c_k    dynamic a_k / c_k
shared            0.478 / 0.508       0.104 / 0.717
a_only_dec        0.265 / 0.279       0.133 / 0.366   <- WORST (craters both) = pathology confirmed
c_only_dec        0.886 / 0.895       0.255 / 0.887   <- best static
decoupled         0.654 / 0.806       0.497 / 0.877
all_decoupled     0.700 / 0.759       0.284 / 0.782
```
- a_only_dec craters both (decouple the slope alone -> pathology). c_only_dec wins (decouple the intercept). Dynamic HURTS the slope in every coupling.

### 2d. dkvmn reversal probe: shared -> decoupled (static): slope a / difficulty b
```
dkvmn 2pl  N500:  a 0.378 -> 0.724   b 0.581 -> 0.888
dkvmn 2pl  N5000: a 0.856 -> 0.958   b 0.617 -> 0.967
dkvmn gpcm N500:  a 0.632 -> 0.878   b 0.798 -> 0.935
dkvmn gpcm N5000: a 0.920 -> 0.972   b 0.862 -> 0.982
```
- Decoupling helps dkvmn too, raising BOTH a and b, and closes the reversal by lifting the LAGGARD (difficulty b). So decoupling fixes whichever parameter shares the constrained pool: discrimination for lstm/transformer, difficulty for dkvmn (whose memory already gives items their own slots = partial decoupling). The reversal is consistent with the mechanism, not a counterexample.

### 2e. Bottom line
Decoupling is the dominant, robust lever: it fixes the shared-pool laggard everywhere. For the information-poor decoder (2PL) it is the ONLY fix (data cannot cure the shared deficit); for GPCM data partly catches up but a residual architectural gap remains. Dynamic head is minor and null once decoupled. NRM a_only_dec pathology and dkvmn "reversal" both reduce to the same mechanism.

## Phase 3 (recovery-vs-epoch trajectory) -- COMPLETE
Bed: lstm x {2pl,gpcm} x {shared,decoupled} x N{500,2000} x Q=200, trained to 500 epochs (NO early
stop), recovery checkpointed every 25 epochs, mean over 25 runs.

VERDICT: the "decoupling delays the degradation" hypothesis is REFUTED. Both arms peak early then
decay. Decoupling shifts the whole curve UP and the peak EARLIER (better, faster recovery) but decays
from that peak AT LEAST AS STEEPLY, usually MORE. In all 12 (dec x N x param) cells the SHARED arm
retains a HIGHER fraction of its peak by ep500 (53-98%) than DECOUPLED (48-95%). Shared looks
"flatter" only because it never climbs as high, not because it is protected.

### peak_epoch / peak_rho / final@500 / retention%  (mean over 25 runs)
```
SHARED                              DECOUPLED
dec  N    param peakEp peak  fin  ret%     peakEp peak  fin  ret%
2pl  500  a      150  .477 .399  84%        75  .774 .532  69%
2pl  500  b      100  .675 .530  78%        50  .911 .503  55%
2pl  500  theta   25  .838 .445  53%        50  .896 .432  48%
2pl  2000 a      125  .656 .604  92%        75  .913 .756  83%
2pl  2000 b       50  .725 .603  83%        50  .960 .765  80%
2pl  2000 theta   25  .813 .587  72%        50  .926 .621  67%
gpcm 500  a      225  .621 .566  91%       100  .884 .544  62%
gpcm 500  b      175  .806 .791  98%       100  .938 .787  84%
gpcm 500  theta   50  .924 .587  63%        50  .955 .562  59%
gpcm 2000 a      175  .776 .743  96%        75  .949 .836  88%
gpcm 2000 b      150  .833 .821  98%        75  .969 .922  95%
gpcm 2000 theta  125  .931 .820  88%        50  .971 .799  82%
```

### Reads
- THETA is the main over-training casualty in BOTH arms: peaks earliest (25-125) and highest, then collapses (drop 0.11-0.46). b overfits under 2pl (drop 0.12-0.20), ~flat under gpcm (0.01-0.05). a (discrimination) is slowest-rising and decays mildest (drop 0.03-0.08 shared).
- DECOUPLED peaks higher in ALL 12 cells but retains a SMALLER fraction in ALL 12 -- it does not hold flat.
- final@500 delta (decoupled - shared), order 2pl_N500/2pl_N2000/gpcm_N500/gpcm_N2000:
  a +0.13/+0.15/-0.02/+0.09 ; b -0.03/+0.16/-0.00/+0.10 ; theta -0.01/+0.03/-0.03/-0.02.

### Bottom line
Decoupling buys a better, EARLIER peak, not stability. Its item-param (a,b) advantage SURVIVES to
ep500 at N=2000 (a +0.09-0.15, b +0.10-0.16) but is ERASED for theta (both arms overfit theta
equally). Over-training corrodes the readout regardless (theta worst); early stopping is required in
BOTH arms, and realizing decoupling's benefit means stopping near its (early) peak.

## Phase 4a (decoupling crossover at starved exposure) -- COMPLETE
Bed: spiraled, Q=200, L=60, lstm, N in {50,100,200,333} -> E in {15,30,60,100},
{shared_static, decoupled_static}, 25 folds/cell, paired on identical data draws.
Outputs outputs/p2_crossover/. delta_a = decoupled - shared discrimination rho.

```
dec  E(N)      a_shared a_decoup delta_a  wins/25   b_sh  b_dec   th_sh th_dec
2pl  15(50)     0.160    0.103   -0.057     7       .387  .508    .642  .696
2pl  30(100)    0.203    0.240   +0.036    16       .492  .695    .721  .787
2pl  60(200)    0.278    0.515   +0.236    23       .622  .841    .696  .839
2pl  100(333)   0.329    0.620   +0.291    25       .664  .884    .765  .887
gpcm 15(50)     0.135    0.148   +0.013    13       .656  .700    .833  .813
gpcm 30(100)    0.250    0.382   +0.132    23       .699  .802    .828  .863
gpcm 60(200)    0.295    0.643   +0.348    25       .748  .870    .869  .930
gpcm 100(333)   0.370    0.765   +0.395    25       .764  .906    .870  .939
```
- CROSSOVER CONFIRMED for 2PL: E* in (15, 30]. Shared wins at E=15 (18/25 folds); decoupled
  leads by E=30, dominates from E=60. GPCM: no crossover down to E=15 (coin flip there), so
  its E* sits at or just below 15.
- The crossover is DISCRIMINATION-SPECIFIC: difficulty and theta never reverse (decoupled wins
  both at every E). Consistent with discrimination being the observation-hungry parameter.
- HONEST SCALE CHECK: exposure starvation at E=15 produces a modest deficit (both arms ~0.10-0.16),
  NOT a collapse. The real EdNet NRM reversal (0.695 -> 0.207 at E~12) is far larger, so exposure
  starvation is a contributing mechanism in the right neighborhood but NOT the dominant cause of
  the EdNet collapse; the option-level nominal structure or real-data misspecification likely
  dominates there. Do not overclaim the crossover as the EdNet explanation.
- Deployable rule: decouple when items see roughly >= 30 responses (2PL bound; GPCM even lower).

## Phase 4b (classical MML control, mirt 1.44.0, GPCM K=4) -- COMPLETE (one cell at 2/5 seeds, finishing)
Same regenerated spiraled beds (seeds 0-4), full response matrix (no CV split, conservative in
mirt's FAVOR). Scored with the same metrics_bench.item_recovery. Outputs outputs/p2_mml/.

### Discrimination rho at fixed per-item exposure E, small (Q,N) vs large (Q,N)
```
E    small(Q,N)   mirt        large(Q,N)    mirt        neural-shared small -> large
150  (200,500)    0.938+-.009 (2000,5000)   0.939+-.002 (n=2, finishing)   0.433 -> 0.759
300  (200,1000)   0.967+-.006 (1000,5000)   0.969+-.002                    0.554 -> 0.733
600  (200,2000)   0.982+-.003 (500,5000)    0.983+-.002                    0.719 -> 0.844
```
Step thresholds b mirror it exactly (0.949->0.950, 0.976->0.974, 0.987->0.987).

### Two findings, both load-bearing
1. DISSOCIATION CONFIRMED. Classical MML is FLAT in N at fixed E (+0.001/+0.002/+0.001, inside
   seed noise). The neural shared readout RISES at fixed E (+0.33/+0.18/+0.13). An item's
   classical information depends on its own takers; the shared encoder pools the whole cohort.
   The amortized scaling law is real.
2. CLASSICAL DOMINATES ABSOLUTE RECOVERY at every tested cell. mirt sits at 0.94-0.98 while the
   neural SHARED arm sits at 0.43-0.84. Even the DECOUPLED arm at its best (0.941 at N=2000,
   0.964 at N=5000, Q=200) lands ~0.02-0.04 BELOW the mirt ceiling (0.982 at the matched E=600
   cell). On clean well-specified data the true-model MML is the ceiling; the prediction-trained
   readout pays a calibration tax everywhere, large when shared, small when decoupled.

### Caveats
- mirt saw the full matrix (no CV) -- deliberately conservative in its favor; the neural fits see
  train folds only. The Q>=1000 mirt fits hit the 800-EM-cycle cap (converged=FALSE) but rho is
  stable within 0.002 across seeds -- slow tail, not a bad fit.
- CORRECTION to an earlier quote: the E=150 anti-diagonal neural endpoints are 0.433 -> 0.759
  (mean over folds), NOT "0.156 -> 0.743" (0.156 was the (Q2000,N500) E=15 cell, a different
  point). The anti-diagonal rise is +0.33, still large.

## Phase 5a (oracle ladder, CPU) -- COMPLETE [outputs/p2_oracle]
Per-item GPCM MLE with theta fixed, on the toggle bed (lstm, Q=200), vs the readouts and mirt.
```
                           N=2000            N=500
shared readout             0.719 +/- .062    0.433 +/- .127
per-item MLE | theta_hat   0.934 +/- .028    0.870 +/- .019
decoupled readout          0.941 +/- .007    0.853 +/- .017
per-item MLE | theta_true  0.979 +/- .003    0.929 +/- .011
mirt full-MML              0.982 +/- .004    0.939 +/- .010
```
Decomposition (N=2000): CHANNEL 0.22, THETA-NOISE 0.05, INFORMATION RESIDUAL 0.003
(mirt 0.9820 - clamp 0.9794; corrected from an earlier 0.02 misquote -- with theta fixed to
truth, per-item MLE saturates the classical ceiling). The shared-channel effect dominates;
decoupling ~= a per-item estimator inside end-to-end training. (Old-bed "two-stage NO_FIX"
is consistent: there the shared arm was already wide/strong.)

### Phase 5a extension: the 2PL ladder (CPU) -- COMPLETE [outputs/p2_oracle, p2_mml/MML2PL_*]
```
                           N=2000            N=500
shared readout             0.553 +/- .097    0.368 +/- .056
per-item MLE | theta_hat   0.874 +/- .068    0.725 +/- .072
decoupled readout          0.898 +/- .018    0.694 +/- .081
per-item MLE | theta_true  0.959 +/- .005    0.865 +/- .014
mirt full-MML (2PL)        0.962 +/- .005    0.877 +/- .015
```
Decomposition: CHANNEL +0.321/+0.357 (dominant at both N); THETA-NOISE +0.084/+0.140 (real,
nearly doubles at low N -- errors-in-variables); mirt - clamp +0.003/+0.011 (~zero). The 2PL
"architectural, data cannot cure" claim is supported FOR THE CHANNEL TERM, which stays
dominant; the smaller theta-noise term is N-sensitive and honest phrasing keeps them apart.
GPCM report reproduced byte-identically as a regression check when the driver was
parametrized.

## Phase 5b (two-law scaling figure, CPU) -- COMPLETE [outputs/p2_scaling]
- mirt: rho = 0.784 + 0.072*log10(E), R2 0.96; exactly flat in N at fixed E.
- Neural decoupled channel: rho = -0.160 + 0.404*log10(E), R2 0.84 (5.6x steeper per decade).
- Shared-arm collapse on TOTAL responses is DECODER-DEPENDENT: nrm clean (R2 0.90 pooled),
  gpcm partial (R2 0.80, Q offset +0.166), 2pl FAILS (Q-sensitivity 0.240/decade > N 0.075).
  State honestly; do not claim a uniform amortized collapse.

## Phase 5c (interpretation-flip demo, CPU) -- COMPLETE [outputs/p2_flip]
N=5000: |d acc| = 0.006; between-arm top-20 'most discriminating' Jaccard 0.44 (=56%
disagreement); shared arm vs TRUE top-20 Jaccard 0.39 (=61% wrong); within-arm stability 0.80
(STABLE AND WRONG); tercile truth-agreement shared 0.67 vs decoupled 0.86. Exhibit: items
ranked 9/11/18/20 by the shared arm sit at true ranks 111/105/106/129 of 200.

## Phase 6a (real-data calibration tax, EdNet+KDD) -- PARTIAL [outputs/p2_realtax + /mirt]
Neural arms re-fit with full-matrix + per-half item params persisted; mirt refit on the
IDENTICAL matrices (first-attempt-only filter applied: ~17% of learner-item pairs repeat and
naive pivoting corrupts mirt; the neural model consumes repeats natively -- state as a
comparability caveat AND a point: classical calibration must discard repeats).
FINAL per-method split-half reliability table (Spearman-Brown; mirt b = robust/Spearman):
```
dataset dec   neural-shared a/b   neural-decoupled a/b   mirt a    mirt b(robust)
ednet   2pl     0.754 / 0.960       0.821 / 0.946        0.878     0.848
ednet   gpcm    0.678 / 0.959       0.745 / 0.959        0.883     0.901
kdd     2pl     0.730 / 0.812       0.786 / 0.826        0.699     0.810
kdd     gpcm    0.714 / 0.884       0.786 / 0.886        0.711     0.879
```
- CRATER VERDICT: ARTIFACT. The initially reported mirt-b collapse (0.13-0.24) was Pearson on
  mirt's IRTpars b = -d/a, which diverges when an item's |a|<~0.15 flips sign between halves
  (10-30/250 items; a few |b| in the thousands dominate the covariance). Sign-aligned
  Spearman on the same quantity gives 0.81-0.90, reconciling with the full-fit b concordance
  0.83-0.87. Persisted as b_reliability_robust; the Pearson number kept diagnostic-only.
  META-NOTE for the paper's pitfalls section: our own first-pass result was exactly the
  rank-vs-linear evaluation artifact the certification protocol exists to catch.
- FINAL FLAGSHIP READING: on real data the calibration tax SHRINKS TO NEAR-PARITY with
  parameter- and dataset-specific edges: mirt better on EdNet discrimination (0.88 vs
  0.75-0.82); neural decoupled slightly better on KDD discrimination (0.786 vs 0.699-0.711)
  and on EdNet difficulty (0.946-0.959 vs 0.848-0.901); KDD difficulty ~parity. The
  pre-registered prediction (shrinks/reverses under misspecification) is CONFIRMED in the
  "shrinks to near-parity, small parameter-specific reversals" form.
- Concordance (full fits, Spearman): EdNet a 0.64-0.83 / b 0.83-0.87 (decoupled concords
  better with mirt than shared does); KDD weaker (a ~0.39-0.40).
- EdNet NRM, one-key arm (the synthetic winner) RUN: slope reliability 0.065 (!) --- WORSE
  than the pathological a_only_dec (0.685 slope / 0.224 intercept) and far below shared
  (0.695 / 0.707). TRIPLE-CONFIRMED: the real-data NRM reversal is coupling-independent;
  ANY wide-key decoupling breaks the real option-level slope channel; only shared survives.
  Synthetic guidance does not transfer for NRM; the reliability screen is what catches it.

## Phase 6b (cold items, skewed-exposure bed) -- COMPLETE, AUDIT-CORRECTED [outputs/p2_coldstart]
40 items pinned at exact E {5,10,20,40} (10 each), 160 warm (~700); lstm gpcm shared vs
decoupled vs mirt (per-seed refits). FINAL numbers from mirt_fit/starved_report.json (the
authoritative, later artifact; the first-pass agent's tier counts disagreed and are
superseded):
- COVERAGE CLAIM RETRACTED: mirt fit 49/50 items at E=5 and 50/50 at E=10 (n_dropped 1/0);
  the earlier "9/50 estimable" was not supported by the final artifact. Do NOT use the
  coverage argument in the paper.
- Difficulty: the neural advantage is LARGER than first quoted: E=5 0.756/0.798 (shared/
  decoupled) vs mirt 0.378; E=10 0.809/0.815 vs 0.465; E=20 0.866/0.879 vs 0.773 (neural
  clearly ahead, not a tie); E=40 neural ahead.
- Discrimination: mirt wins every tier; neural a weak cold (unchanged).
- LAW REFINEMENT (quotable, intact and STRONGER): LOCATION IS RECOVERABLE THROUGH POOLING,
  SCALE IS NOT. The amortized channel earns its keep on cold-item DIFFICULTY (2x mirt at
  E=5), not coverage and not discrimination. Small-n caveat: 10 items/tier x 5 seeds;
  bootstrap CIs required in the paper.

## Phase 6d (real-data flip exhibit) -- COMPLETE [outputs/p2_flip/flip_real.json]
On real EdNet, accuracy-near-tied arms (2pl delta 0.023, gpcm 0.030) DISAGREE on 67% of the
top-20 most-discriminating items (both decoders); mirt on the identical matrix agrees MORE
with the decoupled arm (Spearman 0.84/0.79) than the shared arm (0.81/0.64). NRM one-key
corroboration: its top-20 shares 0% with shared and does not reproduce across halves (mean
Jaccard 0.006) -- the 0.065 reliability is visible as pure ranking noise.
LABELING NOTE (audit): on disk, ednet_nrm_decoupled_shared_key = the one-key WINNER coupling
(the 0.065 arm); ednet_nrm_decoupled_one_key = nrm_channel a_only_dec (the pathological
control, 0.685/0.224). Directory names are misleading; values verified correct.

## Phase 6c (trajectory with prediction, N=2000 gpcm) -- COMPLETE [outputs/p2_trajpred]
- SHARED: validation prediction plateaus ~ep89 (theta co-moves, ~93) but a_rho peaks ~167,
  b_rho ~171: prediction-selected checkpoints LOSE 0.099 discrimination recovery vs the
  a-optimal stop.
- DECOUPLED: all peaks collapse together (pred ~72, a ~76, b ~82): selection loss 0.001.
- NEW decoupling property: it ALIGNS the prediction-selection schedule with recovery,
  making prediction-based early stopping measurement-safe.

## Phase 7 (CAT harm simulation, the venue gate) -- COMPLETE, MATERIAL [outputs/p2_cat]
25 folds x 6 arms x M=2000 examinees, GPCM K=4 quadrature-EAP CAT; responses always from
TRUE params; arms differ only in selection+scoring params. Simulator adversarially verified
pre-run (4 core checks passed; RMSE-at-stop metric + deterministic seeding added). Fold-level
bootstrap 95% CIs (NOTE: must be re-stated seed-clustered per R1; margins are wide enough to
survive n=5).
```
arm             RMSE@20  items-to-SD.30  length-infl%        excess miscl @cut0   @cut+1
oracle           0.159      4.51           100 (ref)            0 (ref)            0 (ref)
mirt             0.162      4.35            96.6 [95.6,97.5]   +0.001 ns          +0.002
neural shared    0.236      8.84           196.8 [190,204]     +0.023 [.020,.027] +0.017
neural decoupled 0.204      7.07           157.2 [151,163]     +0.006 [.004,.009] +0.008
ablation a-only  0.187      8.19           182.1 [176,188]     +0.008             +0.005
ablation b-only  0.220      5.00           111.1 [109,114]     +0.013             +0.015
RMSE-AT-STOP excess vs oracle: mirt +0.010; shared +0.036 [.029,.044]; decoupled -0.008;
a-only -0.012; b-only +0.096 [.083,.109] (the WORST arm at its own stop).
```
- SHARED ARM HARM IS MATERIAL: ~2x test length to the same stopping rule AND worse theta at
  its own stop (+0.036); +2.3pp misclassification at cut 0 at fixed length 20. Decoupled
  halves the harm (157%, +0.6-0.8pp) -- a partial fix in decision terms, tracking its
  intermediate recovery. mirt ~ oracle.
- TWO-CHANNEL ATTRIBUTION (richer than designed): DISCRIMINATION error drives length
  inflation (a-only reproduces 85% of shared's excess length; b-only 11%). DIFFICULTY error
  drives decision corruption: b-only alone is the worst stop-point arm (+0.096) -- accurate
  a + biased b makes the posterior FALSELY CONFIDENT, stopping near oracle length on the
  wrong theta. Consequence claim is two-channel: a-error inflates the test, b-error corrupts
  the decision the shorter test is trusted to make.
- VENUE GATE RESOLVED: MATERIAL -> education-Q1 route opens (IEEE TLT first per sec 8 of the
  plan). R4 linking check on the misclassification metric still lands via the rigor track.

## Phase 8b (anchor: the width/capacity control) [outputs/p2_width, run pre-Phase-1]
For the record (editor anchoring fix): widening the SHARED embedding 8->96 plateaus
discrimination ~0.06 BELOW the decoupled point in all three decoders (gap +0.056 rho,
identical to the third decimal), at 49k params vs decoupled's 21k (2.3x). With Phase 8's
key-16 control (decoupled key-16 beats shared-w96), capacity is excluded as the mechanism
in both directions. Full tables outputs/p2_width/width_table.md.

## Phase 8 (rigor execution R1-R10) -- COMPLETE [outputs/p2_cluster, p2_oracle, p2_realtax_fa, p2_narrowkey, p2_scaling]
All statistics re-stated with SEED-CLUSTERED bootstrap (5 independent banks; folds/items/
persons nested). Full survival table in outputs/p2_cluster/.

SURVIVES (clustered CI excludes zero):
- Channel > theta-noise, all 4 (decoder x N) ladder cells.
- Decoupling delta_a: 32/32 cells, 5/5 seeds positive, every clustered CI excludes zero.
- NRM crater delta [0.594,0.667]; flip disagreement (N2k, N5k); CAT harm (196.8% [190,204],
  +2.3pp miscl, RMSE-at-stop +0.036); crossover from E>=60.
- LADDER GENERALITY (R2): dkvmn CONFIRMS the slots prediction (a-channel gap only +0.075,
  deficit moves to difficulty +0.132); transformer LARGEST a-gap +0.476 (extreme pooling);
  lstm +0.216; second geometry (Q=500) holds. The spine's scope: all three encoders, two
  geometries; the laggard is whichever parameter stays pooled.
- R10 (LABEL CORRECTED at Stage-2.5 integrity, 2026-07-03): narrow KEY-16 decoupled beats
  the STANDARD SHARED ARM at the operating bed (2pl a 0.780 vs 0.553; gpcm 0.824 vs 0.719;
  the earlier "shared-w96" label was wrong -- no w96 cell exists on the spiraled bed; the
  w96 comparison lives in Phase 8b's width sweep, plateau +0.056 at 2.3x params). Combined,
  the two facts still kill the capacity rebuttal: width plateaus below decoupled, and a
  16-wide key beats the shared arm.
- BANKED (Stage-2.5), CORRECTED + SUPERSEDED at Stage-4 hardening (outputs/p2_stat_hardening/):
  NO formal TOST artifact ever existed; manuscript equivalence language replaced by the
  clustered-delta criterion (tied = clustered accuracy delta within +/-1pp). 32-pair audit:
  STATIC max |dacc| = 0.0417 (transformer gpcm N1000; N500 0.0415 [0.031,0.053]); every
  static delta >= 0 (decoupled never worse on accuracy, anywhere); exhibit budgets (lstm
  N5000): gpcm 0.0059, 2pl 0.0006 (tied). DYNAMIC pairs (refuted arm) reach 0.0663; the old
  "all other pairs smaller" phrasing was wrong for dynamic. Wild-cluster floor at G=5:
  p_min = 1/16; all headline contrasts at floor with 5/5 sign agreement. Slack tau LOCO:
  tau reproduces (10/11 folds = 0.1519), sens 0.923, held-out spec 0.913 (FPs concentrated
  in dkvmn gpcm), per-cell AUC 0.63-0.96. mirt EM cycle-cap: 5/25 seed-cells Q=200, 5/5
  Q=1000, 4/4 Q=2000, 0/5 Q=500; capped vs converged recovery ranges fully overlap.
- STAGE-4 CAT AT LARGEST BUDGET (outputs/p2_cat_n5000/, clustered in cat_clustered.json,
  no mirt arm exists at Q200/N5000): shared infl 183.7 [155.6,214.8] (unresolved vs 196.8),
  stop excess +0.043 [0.028,0.063], miscl cut0 +0.029 [0.017,0.043]; decoupled 153.2
  [145.0,160.1]. Shared a-recovery 0.719 [0.669,0.770] N2000 -> 0.810 [0.774,0.862] N5000:
  more data buys rank, NOT decision costs (b-channel + scale dominate). "Does not resolve."
- STAGE-4 SLACK ROBUSTNESS (outputs/p2_slack_robustness/, report.md): (A) theta-quality
  bins: r 0.986 overall, 0.956 worst tercile, 0.843 worst decile (theta_rho floor 0.683);
  (B) deliberate degradation (12 fits, noise to 1.0sd + shuffle to 25%): failures
  one-directional (slack INFLATES; over-audit, never mask), bad readouts flagged 8/8 in
  every condition, breaking point ~1.0sd/25% and only boundary fits; (C) REAL EdNet silver
  validation: slack gpcm_shared 0.277 > 2pl_shared 0.198 > tau > 2pl_dec 0.124 > gpcm_dec
  0.113; rank IDENTICAL to mirt-disagreement (Spearman 1.000); first-attempt concordances
  0.856-0.920. DA attack 1 neutralized-in-envelope, bounded beyond.
- FLAGSHIP under apples-to-apples (R3, first-attempt-only neural): same pattern as
  repeats-native -- near-parity, parameter/dataset-specific edges (EdNet-a to mirt, EdNet-b
  to neural, KDD-a to neural-decoupled). The fairness fix confirms, not dents.
- R4: CAT scoring uses a fixed true-scale prior; misclassification is NOT a gauge artifact;
  corrected numbers ~unchanged.

WEAKENS / RETRACTS:
- E* CROSSOVER REWRITTEN: the naive E=15 "shared wins" was a FALSE REVERSAL (clustered CI
  [-0.128,0.037] includes zero); 2PL is unresolved below E=60; GPCM decoupled already wins
  from E=30. Honest statement: decoupled wins from E~30-60 upward; below that the contrast
  is UNRESOLVED (no confirmed reversal on this bed). The EdNet option-level reversal remains
  a separate, real-data fact.
- "5.6x STEEPER" DIES: Fisher-z slope ratio 0.76-0.92; the raw-rho ratio was a compression
  artifact. Two-law figure stays QUALITATIVE (mirt flat-in-N; per-item channels
  exposure-scaled).
- COLD-ITEM LEG DIES: all four tiers' item-resampled clustered CIs span zero (10 items/tier
  too thin; the cold20 naive positive was a false positive). The "location is recoverable
  through pooling" refinement RETRACTS with it (descriptive note at most). Attack-2
  re-anchors on online/sequential + no-calibration-catastrophe.
- R8 COERCION CHECK lands badly for K=3 real GPCM: mirt's real step thresholds mostly
  DISORDERED (54.8% EdNet / 27% KDD ordered), and the neural fits' 100%-ordered is a
  torch.sort EXPORT ARTIFACT, not evidence. Polytomous real claims demote hard per R9
  (coerced-ordinal robustness checks, prominent caveat).

## Phase 9 (slack test + retrofit validation) -- COMPLETE [outputs/p2_slack, p2_cat_retrofit]
- SLACK TEST (truth-free diagnostic: slack = 1 - Spearman(readout a, per-item refit a on the
  model's OWN theta_hat)): VALIDATED. Pooled 275 fold-points, 11 cells (both decoders, 3
  encoders, 2 geometries, shared+decoupled): slack-vs-wrongness r = 0.986 (per group >=
  0.896); flag at tau = 0.152 -> sensitivity 0.923, specificity 0.988, AUC 0.987; false
  alarms on truly-healthy folds 1.2% (0% on healthy decoupled); genuinely-bad decoupled
  folds correctly flagged. Truth-free confirmed in code (truth enters only eval/oracle
  arms). Caveat: inherits theta_hat quality (refit tracks truth at mean rho 0.892 here).
  Consistent with the old dense-bed NO_FIX (healthy readout -> low slack -> no alarm).
- RETROFIT (per-item refit as a no-retraining repair): RANK-REPAIR ONLY, honestly bounded.
  Recovers rank (0.72 -> 0.93 gpcm) and ~52% of CAT length inflation (146.5% [126,168] vs
  shared 196.8%), but ~4% of the ACCURACY gap (RMSE@20 +0.003 [-0.037,+0.040] vs shared,
  ~zero; RMSE-at-stop shared-level +0.035; RMSE@40 worse than shared). MECHANISM: the refit
  bank lives in the theta_hat GAUGE; per-fold alpha-scale ratio correlates r = -0.98 with
  harm (the scale gauge reappears as the retrofit's failure mode). Product segmentation:
  refit repairs INTERPRETATION (rank uses); decisions need the rebuild (decoupling) or a
  classical refit. Retrofit CAT row in outputs/p2_cat_retrofit/summary.json.

## Phase 10 (encoder-coverage closure) -- LANDED, two finishing runs in flight
- TRANSFORMER FLIP [outputs/p2_flip/flip_transformer.json]: the disease is WORSE under
  extreme pooling: N=5000 between-arm top-20 disagreement 68% (lstm 56%), shared flags 71%
  wrong vs truth (lstm 61%), |d acc| 0.016; N=2000 disagreement 83%. Decoupled still tracks
  truth better. Encoder-generality of the symptom banked.
- ARCHITECTURAL CROSS-TEST [outputs/p2_cat/transformer_*, dkvmn_*]: verdict MIXED, reported
  plainly. GENERAL mechanism held on all three encoders (a-channel drives length inflation;
  b-channel drives quality/misclassification cost). Transformer pre-registration HELD in
  full: 343.9% inflation, length-dominated, a-only ablation reproduces most (321.2%).
  dkvmn K=4 HELD (161.5%, smaller/mixed as pre-registered). dkvmn K=2 pre-registration
  FAILED: length-dominated like lstm (351.1% vs 359.6%), excess misclassification
  smaller/negative (-0.004 vs lstm +0.018), NOT decision-dominated. Post-hoc HYPOTHESIS
  (labeled as such): the falsely-confident signature requires accurate-a-with-biased-b (the
  ablation's construction); dkvmn-shared has BOTH degraded, so it pays the length channel.
  The failed pre-registration is reported in the paper next to the held ones.
- COVERAGE STATUS: dkvmn N=2000 toggles retrained clean by a CPU recovery pass after the
  GPU background job DIED SILENTLY with its parent session (its claimed 23/200 progress was
  never persisted -- lesson: foreground slices only for training). dkvmn N=1000 partial and
  the dkvmn EdNet real arms not yet run; BOTH being finished now (foreground slices).

### Phase 10 completion: dkvmn ladder + shipped-design real audit [outputs/p2_toggle, p2_realtax]
- DKVMN 4-POINT LADDER (N 500/1000/2000/5000, 25 folds/cell, disk-audited 400/400 done):
  decoupling repairs dkvmn's DIFFICULTY laggard across the whole ladder. Under sharing, b is
  nearly N-FLAT (2pl b 0.618/0.570/0.652/0.667; gpcm b 0.808->0.874) while decoupled b sits
  0.88-0.98 everywhere and grows. THE SYMMETRY IS NOW COMPLETE: whichever parameter is
  pooled, MORE DATA DOES NOT CURE IT (lstm/transformer: discrimination; dkvmn: difficulty);
  an own channel does. Monotone gap, largest for 2pl.
- DKVMN EDNET REAL (the shipped Deep-IRT architecture, 36/36 units done): the repair holds
  on real data: a-reliability shared->decoupled 0.795->0.839 (2pl), 0.746->0.809 (gpcm),
  mirroring lstm (+0.067); b ~0.95-0.96 unchanged. dkvmn slightly beats lstm on
  a-reliability in every matched arm. CAREFUL FRAMING (do not overclaim): dkvmn's real
  split-half laggard is discrimination, not difficulty; the synthetic b-deficit does NOT
  surface in real reliability -- and it COULD NOT, because reliability measures cross-half
  consistency, not correctness: a systematically distorted b would still be reliable. This
  is the paper's own ritual-table point recurring: on real data, a stable b-deficit is
  invisible to the screen. State as: synthetic predicts a b-deficit for this architecture;
  real reliability cannot adjudicate it; truth-free detection is exactly what the slack
  test is for.

## Phase 11 (seed-clustered CAT CIs, editor fix 5) -- COMPLETE [outputs/p2_cluster/cat_clustered.json]
All headline CAT-harm claims SURVIVE seed-clustering (CIs ~2x wider, all clear of the null):
shared inflation 196.8 -> [180.4,210.8]; decoupled [144.1,168.0]; b-only RMSE-at-stop
[0.069,0.117]; retrofit [111.8,199.8]; transformer 343.9 -> [271.9,410.3]; dkvmn gpcm
[134.5,200.0]; dkvmn 2pl [281.8,420.6]; lstm 2pl [299.1,417.7]; all excess-misclassification
CIs clear of zero. ONE REVERSAL: decoupled's "lower RMSE-at-stop than oracle" (-0.008) loses
significance clustered ([-0.019,+0.002]) -- NEVER state decoupled as significantly beating
the oracle at its own stop. Prose quotes the CLUSTERED intervals.

---
CAMPAIGN COMPLETE, phases 1-11, all evidence banked, seed-clustered, disk-audited. Plan of
record docs/paper_plan_v2.md is FROZEN (GO-for-prose); venue CAEAI-first; title = user's
pick from the ranked six. Nothing running. The paper enters writing with: three contributions (channel
mechanism + oracle ladder across 3 encoders/2 geometries; decision-cost audit = flip + CAT
two-channel harm; benchmark + certification protocol), fully seed-clustered inference, the
flagship fair-comparison-confirmed, and on-record retractions (cold-start leg, 5.6x, E=15
reversal, dynamic head, decoupling-delays-degradation, linking penalty, mirt-b crater).
