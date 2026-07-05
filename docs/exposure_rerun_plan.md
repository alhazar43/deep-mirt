# Budget rerun plan (2D N x Q recovery grid)

Active rerun of the "Not All Parameters Learn Alike" synthetic study on a 2D grid of
learner cohort N by item bank Q, with corrected theta scoring (last-valid step) and
rho-primary metrics. Supersedes the old bed (Q=200, N=2000, admin U(40,80)), which
carried the theta padded-column bug, an over-wide admin spread, and uncontrolled budget.

## Administration (psychometric consult 1, LOCKED)
- FIX L = 60 items/learner, deterministically. DROP Uniform(40,80). (Wide spread injects
  heteroscedastic theta SE + unequal exposure; low-L learners miscalibrate every item they
  touch. L=60 holds 2PL theta reliability rho ~= 0.94, polytomous higher, so theta is a
  non-binding control.)
- SPIRALED single-connected-cycle administration so each item is seen ~EXACTLY E = N*L/Q
  times (exact exposure) and the item co-occurrence / linking graph stays connected.

## Bed
- theta ~ N(0,1); alpha ~ LogNormal(0,0.5). Corrected theta at last-valid step. rho primary
  + Pearson secondary. Report discrimination/slope by ALPHA-DECILE (low-alpha <0.5 caps rank
  recovery intrinsically). Per-category K=4 counts + sparse-tail flag. NRM: sum-to-zero
  identification matched between truth and readout; per-fold theta sign alignment.
- Calibration floors (consult 1): 2PL stable E ~= 500 (breakdown < 150-200); GPCM K=4
  ~= 500-750; NRM K=4 ~= 750-1000. E < 300 STARVES K=4 outer categories -> at those cells
  interpret the SLOPE (multiplicative, robust) not the thresholds/intercepts.

## Phase 1: the 2D N x Q grid (the finite-budget surface)
ONE grid subsumes the exposure sweep, the bank sweep, and the matched-E control.
- N in {500, 1000, 2000, 5000}  x  Q in {200, 500, 1000, 2000} = 16 (N,Q) cells.
  Per-item exposure E = N*60/Q is a DERIVED readout of each cell, not set:

  ```
              Q=200   Q=500   Q=1000  Q=2000      (coverage L/Q)
     N=500     150      60      30      15
     N=1000    300     120      60      30
     N=2000    600     240     120      60
     N=5000   1500     600     300     150
            L/Q 0.30   0.12    0.06    0.03
  ```
- FULL 3x3 at EVERY cell: {lstm, dkvmn, transformer} x {2pl-K2, gpcm-K4, nrm-K4}. 16*9 = 144
  cells + 1 doubled-embedding-dim control (lstm x gpcm, emb_dim=16 vs 8) at (N=5000, Q=2000)
  to split embedding-table crowding from the linking graph. 145 cells x 5 seeds x 5 folds =
  3625 fold-units. SHARED embedding baseline (no decouple / no dynamic).
- HOW TO READ IT: down a column = fix the bank, grow the cohort (exposure rises). Across a
  row = fix the cohort, grow the bank (exposure thins, the realistic large-bank case).
  Anti-diagonals = CONSTANT E reached by different (N,Q), so a recovery change there is a
  bank-size / LINKING effect, not exposure -- this is the matched-E isolation, built in.
- Decision gate (plateau-below-good, over BOTH axes): PERSIST (-> Phase 2) if the
  multiplicative parameter's rho saturates below the additive one at high exposure OR degrades
  at the large sparse bank beyond what exposure explains (along an anti-diagonal). VANISH
  (-> pause and report to the user) if it closes to within ~0.03 at high exposure AND is flat
  along the constant-E anti-diagonals.

## Phase 2 (CONDITIONAL on PERSIST): toggle / decoupling study, DOUBLE SWEEP, LSTM
- {shared, decouple, dynamic} across the 2D grid (or a chosen subset of it), so the decoupling
  advantage is tested across BOTH thin exposure (calibration) AND large sparse banks (linking).
- NRM: the FULL 10 configs = 5 couplings {shared, a_only_dec, c_only_dec, decoupled-one-key,
  all_decoupled-separate-keys} x {static, dynamic}. REQUIRED (user directive): re-confirm on
  the new bed that decoupling a_k ALONE craters c_k (a_only_dec pathology; old bed slope 0.499
  / intercept 0.308, bimodal) and that c_only_dec / decoupled-one-key win. Do NOT shortcut NRM
  to shared-vs-decoupled.
- Later extension (user-flagged, not this pass): make dkvmn / transformer togglable too.

## Phase 3 (QUEUED behind Phase 2): recovery-vs-epoch trajectory -- "decoupling delays the degradation"
Motivation: on the 2D grid more N lifts all three params together in 32/36 columns; the
overtraining trade-off (discrimination keeps improving while difficulty/theta DECAY) is a
TRAINING-TIME (epoch) effect the early-stopped grid cannot see. This phase tests it directly.
- Design: train PAST early-stop to ~500 epochs, checkpoint recovery (slope / difficulty /
  theta rho) every ~25 epochs. Cells: lstm x {2pl, gpcm} x {shared, decoupled} x N {500, 2000}
  x Q=200, ~5 seeds. Compare SHARED (expect alpha peak-then-decay + beta/theta overfit) vs
  DECOUPLED (expect monotone hold). The win: if decoupling holds the readout stable while
  shared corrodes, that is a real contribution even though the multiplicative deficit is NOT
  universal (narrow, location-scale, dkvmn-reversed).
- Prior evidence (learning-dynamics scratch): shared-wide alpha 0.906@ep50 -> 0.787@ep500;
  theta 0.97 -> 0.68@ep500; decoupled rose monotonically and held.
- Needs a trajectory driver that scores recovery at epoch checkpoints (not just the final
  early-stopped point). Launch when Phase 2 (wc83f42t0) frees the GPU.

## Amortization claim: novelty verdict (web scan, 2026-07-03)
Claim: in prediction-trained KT-IRT models, ITEM-parameter recovery is governed by TOTAL sample
size, not per-item exposure (cross-item pooling through the shared encoder/embedding); classical
MML tracks per-item exposure. Verdict: PARTIALLY ANTICIPATED, THE DISSOCIATION IS OPEN.
RESULT (4b): dissociation CONFIRMED (mirt flat +-0.002 at fixed E; neural rises +0.13..+0.33),
BUT mirt DOMINATES absolute recovery at every tested cell (0.94-0.98 vs neural 0.43-0.84 shared,
0.94-0.96 decoupled). FRAME ACCORDINGLY: mirt is the attainable CEILING; the story is the
SCALING LAW + the CALIBRATION TAX the readout pays (large shared, small decoupled), NOT
"amortization beats classical." If you have the matrix and want a calibrated bank, refit
classically; the readout's value is sequential/online use, and the paper is the discipline for
reading it. (Also corrected: the E=150 anti-diagonal is 0.433 -> 0.759, not 0.156 -> 0.743.)
- Closest prior: AutoIRT (Sharpnack et al., arXiv:2409.08823 / PMLR 2025): item calibration
  quality tracks total bank size even at zero exposure, BUT via AutoML on item TEXT features
  (content-based), framed as cold start; no shared-encoder-from-responses-only mechanism, no
  classical-MML floor, no factorial N-vs-E ablation. MUST cite and distinguish.
- VIBO (Wu et al., EDM 2020): proved the PERSON-side mirror (amortization pools across sparse
  people); item-side open. MUST cite (vocabulary + the mirror claim).
- Koenig, Spoden & Frey (2020): Bayesian hierarchical pooling for small-sample item calibration
  = the pre-neural partial-pooling folklore. MUST cite (pooling per se is not new).
- Our novelty wedge: response-data-ONLY amortization (no item content), the explicit factorial
  E-vs-N dissociation on one bed, the classical mirt control line, in the sequential KT setting.
LOCKING RUNS IN FLIGHT: classical-MML control (mirt, CPU, anti-diagonals E=150/300/600) and the
decoupling crossover at starved exposure (E in {15,30,60,100}, GPU). Outputs outputs/p2_mml/,
outputs/p2_crossover/.

## Superseded / boundaries
- The earlier exposure sweep (Q=200, weird N 333..3333), its matched-E control, and the
  abandoned fix-E-scale-N bank grid (N up to 20000) are ALL superseded by this 2D grid. Old
  main_* / bank_ on-disk data is not reused (different N). New grid_ cells only.
- Old synthetic benchmark/toggle numbers (docs/experiment_results.md) superseded once Phase 1
  lands. Real-data reliability is a separate later thread. New `_`-prefixed scratch only; do
  not edit Codex-owned files. Downstream CAT / adaptive-testing stakes experiment (E8) and the
  prediction-harms-recovery trajectory are the "heat" layer, planned for after the mechanism.

## Timing / efficiency (speed IS being recorded now; PANELS deferred)
- RECORDED per fold NOW: fit_time_s (training wall-clock) + n_params, for every
  enc x dec x N x Q cell, durable under outputs/p2_exposure/. Reference from the
  exposure sweep: lstm ~2 s/fit vs dkvmn ~26 s/fit at N=2000 (10-25x), transformer ~3 s.
  The cost is ARCHITECTURAL (DKVMN memory read/write ops), not parameter count (dkvmn is
  only ~2x lstm's params). This grounds an efficiency-adjusted "choose LSTM" argument
  (comparable recovery at an order of magnitude less compute).
- DEFERRED (after the grid lands / for Phase 2, do NOT block Phase 1):
  (a) VERIFY fit_time_s brackets the TRAINING LOOP only (exclude datagen + scoring) so the
      encoder ratio is not diluted;
  (b) ADD an epochs-to-early-stop field (decompose faster-per-epoch vs fewer-epochs);
  (c) BUILD the panels: recovery-per-GPU-second and an encoder training-cost comparison,
      reported RELATIVE (a ratio) + n_params, same hardware / batch / early-stop protocol,
      single-GPU (no contention).
