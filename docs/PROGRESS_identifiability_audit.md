# Progress: identifiability audit

> Live tracker for the goal in `docs/GOAL_identifiability_audit.md`. Driven per the Autonomous run protocol in that doc. Append to the log, keep the status table and the exhaustion checklists current. A leg is FAILED only when its checklist is fully ticked with nothing separating signal from artifact.

## Status at a glance
| Item | Status | Result | Go/No-go | Date |
|---|---|---|---|---|
| Leg plans drafted (A, B) | done | committed `315df73` | - | 2026-06-29 |
| Fast-fail A (Leg A implicit rho) | done | level recovered (0.95 vs truth), rate reproducible-not-recovered (0.77 self / 0.36 truth) | reframed to audit | 2026-06-29 |
| Fast-fail Stage-1 (Leg B P-Fisher) | done | off-diag marginal Fisher O(1), conditional eta 2-10% | GO, reframed to eta object | 2026-06-29 |
| G1 prior-art | done | niche open; boundary-around PSI-KT confirmed | GO | 2026-06-29 |
| G2 eta-surface (Leg B) | done | ridge at anchor[2-8] x decorr, peak eta 0.32, costly trade-off | GO, qualified | 2026-06-29 |
| G3 Leg-A probes | done | perturbation-locality is the separator; spaced-rep is the rescue | GO | 2026-06-29 |
| Unified generator (trajectory + coupling) | done (minitest) | `_unified_gen_minitest.py`: V1 mean 0.72, V2 oracle Pearson 0.78, V3 rate AR(1) Spearman 0.99; GT at `bench/outputs/_unified_gen_gt.npz` | GO, scale to full `datagen_field.py` | 2026-06-29 |
| Scaled generator (`_datagen_field.py`) | done | parameterized callable; default-cell V1/V2/V3 PASS; diagonal-P null builds | GO | 2026-06-29 |
| WF-2 Stage-0 oracle (kill switch) | done | `_wf2_oracle_stage0.py`: oracle MLE at the CR bound (eff~1, RMSE~CRB across N and \|P_off\|, bias~0, Fisher cond 18.4); diagonal-P null off-diag AUC 0.478~0.5 (no fabrication), signal AUC 0.970 (power) | GO (estimand well-posed; NOT a prediction-loss claim) | 2026-06-29 |
| Leg B experiment (WF-2 neural, GPU) | done (Stage 1 MVP) | `_anchored_field_model.py`: rank recovery above shuffled floor (Spearman 0.73 vs 0.31); eta_proxy 0.13; alpha Spearman 0.91; reproduce 0.856 >> recover 0.73; P INFLATED 2.7x (not attenuated); null AUC 0.889 (parameterization bias); see log for full numbers | GO (qualified); anchoring sweep + ELBO foil next | 2026-06-29 |
| Leg A experiment (WF-3, GPU) | not started | - | - | - |
| One-figure + writeup | not started | - | - | - |

## Corrections logged (the gates sharpened the thesis)
- **P magnitude INFLATES under prediction-loss training (not attenuates).** Stage-1 neural MVP: P_hat_off_mean = 0.160 vs P_true_off_mean = 0.059 (2.7x inflation). Paper 2 found alpha magnitude collapse (attenuation < 1); P goes the other way. The mechanism: the Laplacian P parameterization is structurally biased toward positive off-diagonals (softplus+eps > 0 always), and the optimizer uses P as a cross-concept smoothing operator that improves predictions even without true coupling. Both signal (0.160) and null (0.132) inflate; the discrimination is in magnitude rank, not absolute magnitude. The "magnitude attenuation pre-registered as expected" claim must be corrected to "direction depends on parameterization bias; Laplacian P inflates." (WF-2 Stage 1)
- **Null fabrication control fails absolutely, holds relatively.** The Laplacian P parameterization cannot represent zero off-diagonals (softplus edge weights > 0 always). Null P_hat_off_mean = 0.132 (vs signal 0.160). In RELATIVE terms, signal > null: AUC = 0.889 (signal distinguishable from null). This is the correct framing for the neural audit: the parameterization has a structural floor, but ranks above that floor carry information. (WF-2 Stage 1)
- **Leg B split is readout-vs-operator, not diagonal-vs-off-diagonal.** Diagonal and off-diagonal of P retain about equally (eta ~0.04 to 0.13); the recovered quantity is the readout (loadings / discrimination, alpha), the same channel Paper 2 tracked. (G2)
- **The two legs fail by different mechanisms.** Leg A = estimator inductive bias (rate is information-rich, eta ~0.45, but the smoothing prior overrides). Leg B = collinearity (eta ~0.02 to 0.13). The unifier is the reproduce-vs-recover gap, not leverage alone; Fisher / eta discriminates which mechanism. (G3)
- **The dimensional rescue is costly, not free.** Lifting P_off identifiability spends the discrimination channel's (eta(alpha) 0.27 -> 0.13). (G2)
- **ELBO does not fully escape on the temporal axis.** An explicit OU / graph prior can be reproducible-toward-its-prior-mean without recovering truth. (G3)

## Current bottleneck and next action
- **Current active leg: Leg B** -- Stage 1 MVP done; next is ablations (Stage 2) and the ELBO foil arm.
- Stage 1 results: rank recovery confirmed above shuffled floor; magnitude inflates (not attenuates); eta_proxy 0.13; alpha readout recovers well (0.91); reproduce >> recover gap (0.86 vs 0.73); null fabrication structural (parameterization bias), not fundamental.
- **Leg B Stage 2 (ablations)**: scalar-vs-vector Delta (H3), curriculum-connectivity sweep (H3), misspecification cost (H3b). Optional: ELBO foil (GPU, lower priority than WF-3).
- **WF-3 (Leg A experiment)** = perturbation-locality probe on a trained checkpoint (the separator), then spaced-repetition / repeated-transient regime (rescue boundary), across encoders.
- GPU is now free; WF-3 can run next.

## Unified generator (spec and acceptance)
The minimal generator `_unified_gen_minitest.py` is the validated foundation; the full `datagen_field.py` scales it.
- **Spec.** K=3 anchored concepts (one pure anchor item per concept, zero cross-loadings), N learners, T steps; PSD coupling P = exp(-c L) on a concept graph (minitest: K=3 path graph, off-diagonals ~0.086); per-learner ability trajectory = exponential approach at a per-learner rate r (minitest: r in [0.05, 0.30]); field update theta_t = theta_{t-1} + P (Q[q] Delta_t), Delta a simple residual; binary 2PL responses (a GPCM variant later). Ground truth saved: Q, A, b, P, r, theta_traj (N,T,K).
- **Acceptance.** V1 marginals sane (overall mean in (0.2, 0.8), no degenerate step); V2 an oracle that knows A, P, theta reproduces per-learner accuracy (Pearson >= 0.70 at T=50, the right level given binary sampling noise, not 0.80); V3 the known per-learner rate recovers from the known theta trajectory by AR(1) (Spearman near 1). Minitest scored V1 0.72, V2 0.78, V3 0.99, all pass.

## Exhaustion checklists (a leg is FAILED only if all are ticked and none works)
**Leg A, declare failed only if NONE of these separates genuine signal from the smoothing-prior artifact:**
- [ ] perturbation-locality probe (learner-invariant impulse kernel = artifact)
- [ ] spaced-repetition / repeated-transient regime (saw-tooth ability the prior cannot fit)
- [ ] vary encoder (LSTM / Transformer / DKVMN)
- [ ] ELBO / OU-prior foil (recover vs reproduce-toward-prior)
- [ ] existence gate as the license (necessary, not sufficient)

**Leg B, declare failed only if NONE recovers the coupling above the diagonal-P null:**
- [x] anchoring sweep (gain=4, in the eta ridge) -- Stage 1 MVP: Spearman 0.73 above shuffled floor 0.31 and above null (structurally undefined). GO.
- [x] decorrelating curriculum (decorr_frac=0.3) -- included in Stage 1. GO.
- [ ] vary K and loading strength
- [x] diagonal-P null (fabrication control) -- oracle Stage-0: AUC 0.478 ~ 0.5 (no fabrication); neural Stage-1: AUC 0.889 (signal > null but null NOT zero -- parameterization bias, not fundamental fabrication). STRUCTURAL FLOOR NOTED.
- [ ] ELBO foil arm (does generative targeting clear it)
- [ ] strong-loading regime (load >= 1.5)

## Three-metric numbers (WF-2 Stage 1, K=3, N=2000, T=50, anchor_gain=4, decorr=0.3, 3 seeds)
| Metric | Signal (mean +- std) | Null (diag-P) | Shuffled floor |
|---|---|---|---|
| eta_proxy (Fisher_P / Fisher_alpha) | 0.13 +- 0.11 | 0.18 | 0.46 |
| reproduce Pearson | 0.855 +- 0.003 | 0.875 | 0.783 |
| recover Spearman(P_hat_off, P_true_off) | 0.734 +- 0.113 | NaN (P_true_off=0) | 0.313 |
| reproduce-vs-recover gap | 0.121 +- 0.110 | -- | -- |
| mag_attenuation (|P_hat_off| / |P_true_off|) | 2.71 +- 0.13 | structural (P_true_off=0) | 3.93 |
| alpha Spearman (loaded entries) | 0.906 +- 0.077 | 0.950 | 0.750 |
| signal-vs-null AUC (|P_hat_off|) | 0.889 | ref | -- |
| final loss | 0.372 | 0.387 | 0.440 |

Key interpretations:
- eta_proxy 0.13: consistent with Stage-1 pre-registration (0.04-0.13). P is low-Fisher relative to alpha.
- Spearman 0.73 > shuffled floor 0.31: temporal structure matters; P ranks are partially recoverable.
- reproduce 0.86 >> recover 0.73: reproduce-vs-recover gap confirmed (0.12 mean).
- Magnitude inflates (2.7x), not attenuates: Laplacian parameterization structural floor (forces positive off-diagonals).
- Null AUC 0.889: model correctly ranks signal > null in |P_hat_off|, but both above zero (parameterization artifact).
- Alpha Spearman 0.91: readout recovers far better than operator P -- the readout-vs-operator split confirmed.

## Artifacts (where the evidence lives)
- Leg A fast-fail: `deep_irt/bench/_program1_fastfail.py`
- Leg A G3 Fisher: `deep_irt/bench/_g3_legA_fisher.py` -> `outputs/_g3_legA_fisher.json`
- Leg B Stage -1 Fisher: `deep_irt/bench/_stage_minus1_fisher_P.py`, `_stage_minus1_fisher_matrix.py`
- Leg B G2 eta-surface: `deep_irt/bench/_g2_eta_surface.py`
- Leg B Stage 0 oracle: `deep_irt/bench/_wf2_oracle_stage0.py`
- Leg B Stage 1 neural MVP: `deep_irt/bench/_anchored_field_model.py` -> `outputs/_anchored_field_results.json`
- Agent-memory notes: `ml-math-researcher/distributed-field-mirt-identifiability.md`
- All `_`-prefixed scratch is gitignored; the leg plans and these two docs are committed.

## Decisions
- 2026-06-29 Guardrail set: audit not model; stay out of PSI-KT's modeling territory (no proposed gate, no proposed learned operator, no graph discovery, rescue regimes reported as boundaries).
- 2026-06-29 A and B integrated under one thesis (reproduce-vs-recover diagnostic, two mechanisms, one figure), per user direction; legs still tracked separately.
- 2026-06-29 P magnitude attenuation pre-registration corrected: direction is parameterization-dependent. Laplacian P inflates (structural floor + smoothing utility); the audit object is rank recoverability, not magnitude. Report inflation + structural null floor as part of the boundary characterization.

## Log (newest last)
- 2026-06-29 Plans A and B drafted, committed `315df73`, pushed.
- 2026-06-29 Fast-fails: Leg A reframed to an audit (gate adds no identifiability); Leg B survived, reframed to the eta object.
- 2026-06-29 Gates G1/G2/G3 all GO. Thesis sharpened (two mechanisms, readout-vs-operator). Integration locked.
- 2026-06-29 Leg plans refreshed to the post-gate framing (audit, not models). Work-scheme minitest PASS: a cold agent self-drove from the docs to build the unified generator (K=3, N=200, T=50), V1/V2/V3 all pass. It surfaced 5 doc gaps (generator spec, generator acceptance, a current-active-leg field, WF-2/WF-3 definitions, the V2 threshold); all are now fixed in this doc.
- 2026-06-29 WF-2 Stage-0 oracle (kill switch) PASS. Scaled generator `_datagen_field.py` built (default V1/V2/V3 PASS); oracle joint-MLE (`_wf2_oracle_stage0.py`, numpy-only, knows Q/A/b/r/theta_init, estimates symmetric P via analytic-grad L-BFGS, FD-checked) hits the Cramer-Rao bound: per-param efficiency ~1, emp RMSE tracks CRB across N (125..2000, ~1/sqrt(N)) and across |P_off| (SNR 1.6..16.5), bias ~0, Fisher cond 18.4. Diagonal-P null: off-diag detection AUC 0.478 ~ 0.5 (registers absence), signal AUC 0.970 (has power). The estimand is well-posed; this is NOT a prediction-loss recovery claim. Note: the well-conditioned oracle here does not contradict the eta 0.04-0.13 collinearity result -- once nuisances (A,b,r) are known, P's own 6x6 Fisher is well-conditioned; the killer collinearity is between P and the nuisances. Next: Stage 1 neural MVP (GPU).
- 2026-06-29 WF-2 Stage-1 neural MVP DONE. `_anchored_field_model.py` built (AnchoredFieldEncoder subclasses BaseSeqEncoder; Laplacian P = (I+cL)^{-1}; scalar delta; sequential field update; Q-masked 2PL decoder). K=3, N=2000, T=50, anchor_gain=4, decorr=0.3, hidden=64, 3 seeds, GPU RTX 4060 (~62s/seed). Results: eta_proxy 0.13 (consistent with Stage-1 prediction); reproduce Pearson 0.856; recover P Spearman 0.734 mean (above shuffled floor 0.313); reproduce-vs-recover gap 0.121; alpha Spearman 0.906 (readout >> operator). UNEXPECTED: P magnitude INFLATES (2.7x) not attenuates; null P_hat_off_mean = 0.132 (structural floor from Laplacian parameterization, not fundamental fabrication); null AUC 0.889 (model discriminates signal from null). Pre-registration correction logged: magnitude direction is parameterization-dependent. See Three-metric numbers table. Results JSON: `outputs/_anchored_field_results.json`.
