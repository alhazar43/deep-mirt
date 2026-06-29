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
| Leg B experiment (WF-2, GPU) | not started | - | - | - |
| Leg A experiment (WF-3, GPU) | not started | - | - | - |
| One-figure + writeup | not started | - | - | - |

## Corrections logged (the gates sharpened the thesis)
- **Leg B split is readout-vs-operator, not diagonal-vs-off-diagonal.** Diagonal and off-diagonal of P retain about equally (eta ~0.04 to 0.13); the recovered quantity is the readout (loadings / discrimination, alpha), the same channel Paper 2 tracked. (G2)
- **The two legs fail by different mechanisms.** Leg A = estimator inductive bias (rate is information-rich, eta ~0.45, but the smoothing prior overrides). Leg B = collinearity (eta ~0.02 to 0.13). The unifier is the reproduce-vs-recover gap, not leverage alone; Fisher / eta discriminates which mechanism. (G3)
- **The dimensional rescue is costly, not free.** Lifting P_off identifiability spends the discrimination channel's (eta(alpha) 0.27 -> 0.13). (G2)
- **ELBO does not fully escape on the temporal axis.** An explicit OU / graph prior can be reproducible-toward-its-prior-mean without recovering truth. (G3)

## Current bottleneck and next action
- **Current active leg: Leg B** (the fast-fail recommended committing to Leg B first; Leg A is shelved-as-posed and runs its audit probes second).
- No bottleneck. The unified generator passed its minitest (spec and acceptance below). Next: scale it into a full `datagen_field.py`, then run WF-2 (Leg B), then WF-3 (Leg A), GPU-serialized, never both at once.
- **WF-2 (Leg B experiment)** = build full `datagen_field.py`, the oracle Stage-0 well-posedness check, then prediction-trained recovery across the anchoring-by-decorrelation grid with the diagonal-P null and the ELBO foil arm.
- **WF-3 (Leg A experiment)** = the perturbation-locality probe on a trained checkpoint (the separator), then the spaced-repetition / repeated-transient regime (the rescue boundary), across encoders.
- First GPU steps are small: the perturbation re-encode on a trained checkpoint; a spaced-repetition generator plus one small train.

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
- [ ] anchoring sweep (gain 2 to 8, the eta ridge)
- [ ] decorrelating curriculum (source-isolation blocking)
- [ ] vary K and loading strength
- [ ] diagonal-P null (fabrication control, must register absence)
- [ ] ELBO foil arm (does generative targeting clear it)
- [ ] strong-loading regime (load >= 1.5)

## Artifacts (where the evidence lives)
- Leg A fast-fail: `deep_irt/bench/_program1_fastfail.py`
- Leg A G3 Fisher: `deep_irt/bench/_g3_legA_fisher.py` -> `outputs/_g3_legA_fisher.json`
- Leg B Stage-1 Fisher: `deep_irt/bench/_stage_minus1_fisher_P.py`, `_stage_minus1_fisher_matrix.py`
- Leg B G2 eta-surface: `deep_irt/bench/_g2_eta_surface.py`
- Agent-memory notes: `ml-math-researcher/distributed-field-mirt-identifiability.md`
- All `_`-prefixed scratch is gitignored; the leg plans and these two docs are committed.

## Decisions
- 2026-06-29 Guardrail set: audit not model; stay out of PSI-KT's modeling territory (no proposed gate, no proposed learned operator, no graph discovery, rescue regimes reported as boundaries).
- 2026-06-29 A and B integrated under one thesis (reproduce-vs-recover diagnostic, two mechanisms, one figure), per user direction; legs still tracked separately.

## Log (newest last)
- 2026-06-29 Plans A and B drafted, committed `315df73`, pushed.
- 2026-06-29 Fast-fails: Leg A reframed to an audit (gate adds no identifiability); Leg B survived, reframed to the eta object.
- 2026-06-29 Gates G1/G2/G3 all GO. Thesis sharpened (two mechanisms, readout-vs-operator). Integration locked.
- 2026-06-29 Leg plans refreshed to the post-gate framing (audit, not models). Work-scheme minitest PASS: a cold agent self-drove from the docs to build the unified generator (K=3, N=200, T=50), V1/V2/V3 all pass. It surfaced 5 doc gaps (generator spec, generator acceptance, a current-active-leg field, WF-2/WF-3 definitions, the V2 threshold); all are now fixed in this doc.
