# Experiment blueprint (Phase 2): "Not All Parameters Learn Alike"

Status: reconciled from the Phase 2 rigorous protocol (research-scientist) and the pipeline architecture
(ml-system-architect), 2026-07-01. Phase 2 DESIGN done. AWAITING USER CONFIRMATION and ultracode before the
build and runs. Everything is rebuilt from scratch at JEDM rigor; no workshop result is reused as evidence.

## Pipeline (a config layer over existing engines, not a reimplementation)
The train -> eval -> recover harness already exists (DeepIRTEngine / MaIrtEngine in deep_irt/bench/engines.py,
plus datagen.py, metrics_bench.py, nrm_metrics.py). Phase 2 adds a thin, config-driven orchestration layer.
All new code is underscore-prefixed scratch in deep_irt/bench/; no Codex-owned file is edited.
Flow: configs_p2/<cell>.yaml -> _p2_config.py (schema + sha256 hash) -> _p2_run_cell.py (deterministic: seeds,
cuDNN-deterministic, build engine, fit -> predict_heldout -> recover -> score) -> outputs/p2/<cell>/results.json
(full arrays) + a compact summary. _p2_sweep.py loops a config dir sequentially, foreground, resumable (a cell
with a matching config-hash is skipped, so a killed sweep resumes).

## One shared protocol (every experiment is one point in it)
- Data: simulate from the true decoder (ground truth known). STATIC ability = primary clean item-recovery bed;
  DYNAMIC (slow random walk) = the KT-realism arm (score theta as within-learner net drift, never pooled level).
  Priors: theta ~ N(0,1); alpha ~ LogNormal(mean ~1, sigma 0.4); GPCM ordered thresholds; NRM a_k, c_k sum-to-zero.
  Well-specified except where an experiment deliberately misspecifies.
- Encoders: LSTM (default), Transformer (causal), DKVMN; match parameter budget within 10% for cross-encoder claims.
- Decoders: GPCM (K=4), NRM (K=4), binary (K=2). Prediction loss only; recover item params post-hoc from frozen weights.
- Two optimizer regimes, kept separate: PRACTICAL (Adam, lr 1e-3, early-stop) for E1/E2/E-levers/real/E8; RATE
  (plain GD, single global step size) ONLY for E-budget's optimization axis (Adam preconditioning compresses kappa).
- Seeds/CIs: separate DATA seed from INIT seed. Main recovery panels 5 data x 3 init = 15 fits/cell, bootstrap 95% CI,
  PAIRED comparisons (Wilcoxon signed-rank, rank-biserial). E-levers/NRM >= 8 data seeds. E-budget 10 seeds/grid point.
  Real data: 8 split-half resamples + item-bootstrap CIs.
- Metrics: recovery = sign-aligned Spearman of recovered vs true per parameter (primary), disattenuated Pearson
  (secondary); reliability = odd/even split-half, Spearman-Brown corrected, per parameter (the real-data echo of recovery).
- Gauge-fixing: rank metrics need only sign alignment; level metrics fix the 2-parameter affine gauge by ONE GLOBAL
  linear map (never per-item, which would absorb the identifiability failure E2 probes). Under coupling, a global affine
  cannot rescue the per-item flat directions, so degraded beta rank recovery IS the identifiability signature.

## Experiments mapped to config levers (the two decouplings stay distinct by construction)
- E1 swap matrix: encoder{dkvmn,lstm,transformer} x decoder{gpcm,nrm,binary}, decoupled+static, 15 seeds; per-parameter
  Spearman + Kendall's W ordering-concordance. Expect GPCM alpha worst in all encoders; NRM a_k not worst.
- E2(a) coupling -> IDENTIFIABILITY (structural): ability_coupling coupled vs decoupled, GPCM, LSTM+DKVMN; beta rank is
  the readout. E2(b) width -> RATE (finite): item_key_dim (decoupled) vs shared narrow, coupling held fixed; alpha rank
  is the readout. 2x2 ANOVA per readout. NRM is never wired to (a).
- E-budget: (i) training-T on plain GD, fit gap(T) = A exp(-T/kappa) + c (plus an Adam arm to show compression);
  (ii) sample-N, reliability gap ~ O(kappa/N) + EIV attenuation ~ O(1/N); (iii) the I(theta)-at-fixed-K knob: sweep
  difficulty spread sigma_b at fixed K to move kappa, crossed with a K-sweep to show and then break the K-kappa collinearity.
- E-levers: {static, dynamic head} x {gpcm, nrm}, 8+ seeds + real reliability; NRM confound controls (hold K,N,Q,seeds;
  score the slope VECTOR; param-count-matched control; report c_k). Expect dynamic rescues GPCM alpha, hurts NRM a_k;
  static wins real reliability.
- Oracle-clamp: teacher-force theta = theta*, recover items only; causal Fisher-rate attribution + MLE-oracle ceiling.
- a_star: sweep mean alpha {0.4..2.5}, locate the recovery crossover, compare to analytic a_star ~ 1.
- Real data: ASSISTments, EdNet, KDD; split-half reliability + coverage moderation (the EdNet reversal near 12 obs/item
  stated as a finding); E9 appendix = classical MML-EM cross-calibration (fragile long pole, off the critical path).
- E8 CAT: Fisher-max item selection under a JOINT (alpha, beta) rule, recovered params vs oracle ceiling + random floor,
  ability RMSE at fixed test length.

## Build (exists vs new; order)
Reuse: engines.py, datagen.py, metrics_bench.py, nrm_metrics.py, nrm_datagen.py. Lift/consolidate existing scratch:
_kappa_probe (I(theta) knob), _dynamic_alpha_test/_decouple_stable (oracle arm), _ednet/_kdd_reliability (split-half),
_paper2_consistency (bootstrap CI). Build new (_p2_ scratch): _p2_config, _p2_run_cell, _p2_sweep (the spine),
_p2_oracle, _p2_datagen_budget, _p2_coupled_theta (E2a, only if we want deep_irt-native coupling), _p2_reliability,
_p2_cat (E8, fully new), _p2_aggregate; and a configs_p2/ dir.
Order: (0) config + schema + manifest; (1) _p2_run_cell on one static cell; (2) _p2_sweep resumable driver;
(3) _p2_oracle; (4) _p2_datagen_budget; (5) E2(a) vehicle; (6) _p2_reliability; (7) _p2_cat; (8) _p2_aggregate + figure
hooks. Smoke-test each with a --quick config before the full grid.

## Reproducibility (JEDM-grade)
Per cell: schema_version, full resolved config + sha256, env manifest (git SHA, torch/numpy/scipy/sklearn, device,
CUDA/cuDNN, determinism flags), seed log, per-seed rows, agg with bootstrap 95% CI. Top-level manifest.json lists every
cell hash + status. The config hash gates resume and proves provenance.

## Compute (honest, cell-size dependent)
Wall-clock is driven by cell size (N, Q, T) and the seed multiplier, and the single 8 GB GPU runs sequentially.
- Lighter cells (N~800, Q~60, T~60): ~1-3 GPU-min/cell, ~25-30 GPU-h total over a few days.
- Full-power cells (N~4000, Q~300, T~100): several GPU-days to ~2-3 weeks.
Cell size trades statistical power against wall-clock. Ultracode parallelizes the CODING and orchestration, not the GPU
runs (those stay sequential on one card), so it shortens the build, not the fit throughput.

## Honest hard/delicate points
- The exp(-T/kappa) coefficient is delicate (full-batch GD may never reach the local-quadratic regime); it stays a
  form-and-ordering claim leaning on the sigma_b knob, not a tight coefficient.
- Coupled-theta (E2a) and the theta-clamp (oracle) are new, small, load-bearing code.
- Classical MML-EM on real sparse banks is unstable/slow; keep E9 in the appendix, off the critical path.
- a_star inversion is noisy at low alpha (needs large N to localize the crossover).
- Well-specification is within-model; on real data "vanishes asymptotically" is stated as a within-model property, and
  real evidence is reliability plus calibration, not recovery-vs-truth.

## Open decisions (before build+run)
1. E2(a) coupling vehicle: ma_irt engine only (cheapest, already has the item-blind theta toggle, covers DKVMN+LSTM) vs
   also a deep_irt-native coupled-theta head (_p2_coupled_theta, more code, fuller cross-encoder claim).
2. Cell size vs compute: full-power N~4000 (weeks, strongest CIs) vs a lighter N (~800-2000, days). A middle N~2000 with
   subsampled real data is a reasonable balance.
3. The exp(-T/kappa) result stays a form-and-ordering claim (acknowledged honest limit).
4. Ultracode for the build (speeds the coding; the GPU runs remain sequential).

## Next
On confirmation and ultracode, build in order (0-8), smoke-test each with --quick, then run the sweep foreground and
record results honestly into outputs/p2/ + the tracker.
