# DRL-MAIRT Build Progress Log

> **Historical.** This document tracks the jobrec build, archived
> 2026-06-03. The active OrdRec build progress lives at
> [`docs/ordrec_progress.md`](ordrec_progress.md). This file is kept
> for the historical record only and is no longer updated.

*Continuously updated. Source of truth for build state.*

Canonical spec lives at [`drl_mairt_plan_v1.md`](drl_mairt_plan_v1.md). This
file tracks execution state across milestones. The next workflow run reads
this file first and appends new entries.

---

## 1. Status snapshot

- **Date created.** 2026-06-04
- **Date last updated.** 2026-06-04 09:30
- **Current state.** M0, M2, M3, M4-RL complete on `main`. v2
  simulator (continuous delta_j, K=5 GPCM, lambda_u heterogeneity)
  is merged and validated with a preliminary results report at
  `rl/results/v2/RESULTS.md`. M1 implemented on
  `feat/online-step-api`, awaiting user review before merge.
- **Active branch.** `main` carries M0, M2, M3, v1 prelim, the
  headline append, the M4-RL merge (`f2fa629`), and the v2 prelim
  results (`fed5db2`). `feat/online-step-api` carries the M1 work,
  six commits, tip `dd1d8bf`.
- **Next milestone.** M1 merge after user review, then M5-RL
  (StudentEnv + UserTower training against the v2 simulator) on
  `main`.
- **Eight locked decisions.** D1 subdir `deep-mirt/rl/`, D2 1D theta, D3 O*NET
  2024, D4 textless items, D5 binary ratings, D6 heuristic DecisionController,
  D7 replay simulator, D8 Option A preference model. See plan Section 2.

---

## 2. Milestones

| Milestone | Status | Branch | Tasks | DoD status | Notes |
|---|---|---|---|---|---|
| **M0**, spec lock + O*NET data prep | complete | `main` (`8a0cb4c`) | spec.md, `build_onet_pool.py`, `onet_v1.parquet` | met | `rl/` scaffold landed in `1c6386a`; planning docs and progress log landed in `8a0cb4c`. 9/9 rl tests pass. |
| **M1**, ma-irt online step API | in_review | `feat/online-step-api` (`dd1d8bf`) | `EncoderDecoderModel.step`, `StepState`, `forward_with_state` per encoder, `compute_logits_from_state` per decoder, `freeze_irt`, `test_step_api.py`, `test_step_microbenchmark.py`, `step_api.md` | met | 13/13 parity tests pass to atol=1e-5 on DKVMN, LSTM, Transformer. 3/3 CPU latency budgets met (plan section 6.1). Full ma-irt suite: 150 passed, 12 skipped (pre-existing slow/artifact skips), 0 failed. Awaiting user review then merge to `main`. |
| **M2**, rl/ skeleton + ItemTower + RetrievalIndex | complete | `main` (`98c07c9`) | `item_tower.py`, `index.py`, `pool.py`, `register_pool.py`, `onet_v1_embed.npy`, `test_retrieval.py` | met | Frozen BGE-small-en-v1.5 + 8-dim structured branch over (work_zone, education_zscore, weighted RIASEC), L2-norm at the head output. 9/9 retrieval tests pass, pool-swap smoke green. UMAP plus nearest neighbor spot checks live at `rl/results/v1/`. |
| **M3**, synthetic data generator (Option A) | complete | `main` (`2bff5ae`) | `synth_users.py`, `synth_likes.py`, `onet_pool_attach.py`, two YAML configs, `build_synthetic_dataset.py`, `test_synth_generator.py` | met | Two presets, dev N=500 and recovery N=5000. EAP theta recovery on true items hits r=0.978 RMSE=0.207 at recovery and r=0.975 RMSE=0.224 at dev. Like rate lands at 0.202 (target 0.20). 16/16 synth generator tests pass. |
| **M4-RL**, v2 simulator (continuous delta_j + K=5 GPCM + lambda_u) | complete | `main` (`f2fa629` merge, `fed5db2` prelim) | continuous delta_j composite, K=5 GPCM in `synth_likes.py`, engagement mixture removed, `lambda_u` per user, `JobTower` rename with shim, `generate_v2.py` entrypoint, `test_delta_j_continuity.py`, prelim plots and `RESULTS.md` at `rl/results/v2/` | met | 923/923 unique delta_j values, std 1.0. 1D Bayes-ceiling Hit@10 lifts from v1's 0.158 to 0.261. Popularity drops from 0.263 to 0.236, restoring the expected oracle > popularity ordering. theta_hat r=0.974 RMSE=0.222 on v2 GPCM responses. |
| **M5-RL**, StudentEnv + UserTower trained on v2 | next | `main` | `student_env.py`, `tracker.py`, `user_tower.py`, `train_user_tower.py`, `user_tower_v2.pt` | not met | Hit@10 floor 0.261 (v2 1D oracle), popularity 0.236. Unblocked by M4-RL prelim. Headroom above the scalar delta_j must come from multi-dimensional matching the JobTower embedding can support. |
| **M6**, evaluation harness + headline plots | blocked | `main` | three `eval_*.py` scripts, `RESULTS.md`, plots | not met | Blocked by M5-RL. Three buckets per plan Section 9, sensitivity sweep on five DecisionController thresholds. |

M7 (LLM simulator) and M8 (bandit controller) are deferred to v2. See plan
Section 10.

### 2.1 Dependency graph

```
M0 ────┬──── M1 (feat/online-step-api PR to ma-irt)
       │
       ├──── M2 (rl/ skeleton + ItemTower + RetrievalIndex)
       │
       └──── M3 (synthetic data generator, Option A)
                          │
M1 + M2 + M3 ───────────► M4 (UserTower + BeliefTracker + training)
                                          │
                                          ▼
                                         M5 (policy + service + E2E smoke)
                                          │
                                          ▼
                                         M6 (evaluation + plots + RESULTS.md)
```

M0 has no blockers. M1, M2, M3 fan out in parallel after M0. M4 needs all
three. M5 follows M4. M6 closes the v1 cycle.

### 2.2 Branch policy

- M0 lands on `main`. Scaffolding work is low risk and unblocks parallel
  work in M2, M3.
- M1 lands on a feature branch `feat/online-step-api` so the user can
  review the ma-irt surface change before merging.
- M2, M3, M4, M5, M6 default to `main` but a feature branch is acceptable
  whenever a milestone touches shared ma-irt code.

---

## 3. Change log

Reverse chronological. Most recent entry first.

- **2026-06-04 09:30.** M4-RL lands on `main` with a preliminary
  results report at `rl/results/v2/`. Three commits since the v1
  prelim. The merge commit `f2fa629` brings `feat/v2-simulator-delta-j`
  (tip `bd23b4f`) onto main. The v2 simulator carries four functional
  changes plus one rename. First, `onet_pool_attach.py` now produces a
  continuous `delta_j` composite, a z-scored sum of work_zone (0.45),
  education_zscore (0.35), and a complexity composite from work
  activity categories (0.20), plus N(0, 0.30) seeded noise,
  re-standardised to unit variance. Second, `synth_likes.py` switched
  from the binary sigmoid to a K=5 GPCM with step thresholds beta
  = (-1.5, -0.5, 0.5, 1.5), with `IsLiked = 1[y >= 3]` kept for
  backward compatibility. Third, the engagement mixture is gone, all
  users are engaged and heterogeneous via `lambda_u` ~ LogNormal(log
  1.5, 0.4). Fourth, `ItemTower` is renamed to `JobTower` with a
  shim. Fifth, `generate_v2.py` is the new entrypoint, dev preset
  N=2000. Commit `fed5db2` is the prelim, nine files, five plots,
  RESULTS.md, `eval_v2_baselines.py`, `build_v2_plots.py`, and
  `test_delta_j_continuity.py`. Headline v2 numbers, n_unique = 923
  of 923 jobs (v1 had 4), delta_j mean 0, std 1, range [-2.16, 2.58].
  Bayes-ceiling Hit@10 = 0.287 on the held-out test partition, 1D
  oracle Hit@10 = 0.261 under the v1-matched 80/20 protocol on the
  v2 dev N=2000 cohort, popularity Hit@10 = 0.236, random 0.157,
  theta-hat 1D = 0.261. Popularity now sits below the oracle, the
  expected ordering that v1's 4-bucket delta_j had inverted. EAP
  theta recovery on v2 GPCM responses lands at r=0.974 RMSE=0.222.
  The continuity test passes its three checks (unique-near-pool,
  zscored-and-finite, Bayes-ceiling > 0.20 floor). Implication for
  M5-RL, the trained UserTower must clear Hit@10 = 0.261 (v2 1D
  oracle) on held-out users and headroom above that must come from
  multi-dimensional matching the JobTower embedding can support but
  the scalar delta_j cannot.
- **2026-06-04 07:30.** M2 and M3 land on `main` together with a
  preliminary results report. Four commits total. `98c07c9` is the
  M2 retrieval pillar, `rl/src/irtrec/retrieval/{pool,item_tower,
  index}.py` plus `register_pool.py` and `test_retrieval.py`, frozen
  BGE-small-en-v1.5 with an 8-dim structured branch over work_zone,
  education_zscore, and a weighted RIASEC code, output L2-normalised,
  precomputed v_j for the 923 occupation O*NET pool persisted to
  `rl/artifacts/onet_v1_embed.npy` (gitignored). `2bff5ae` is the M3
  synthetic data generator, mixed 2PL plus GPCM bank (25 K=2, 15 K=3,
  8 K=5, 2 K=6), Option A preference model with lambda and bias
  calibrated by bisection to a target overall like rate of 0.20, two
  presets at `rl/configs/sim_v1_{dev,recovery}.yaml` for N=500 and
  N=5000. `fb973eb` is the prelim results commit. UMAP of the 923
  O*NET embeddings, theta recovery diagnostics, like rate by
  engagement class, K distribution, delta_j distribution, plus a
  four baseline recommender comparison (random, popularity,
  theta-true 1D, theta-hat 1D) under an 80/20 user split with 500
  bootstrap CIs. M2 retrieval tests 9/9 pass, M3 synth tests 16/16
  pass. Headline preliminary numbers, EAP theta r=0.978 RMSE=0.207
  at recovery preset, like rate 0.202 (target 0.20), Hit@10 random
  0.070, popularity 0.263, theta-true 1D oracle 0.158, theta-hat 1D
  0.158. Popularity beats both 1D matchers, an artefact of the v1
  simulator's 4-valued delta_j (work_zone driven) on a 923 item pool
  rather than a flaw in theta recovery. Implication for M4, the
  trained UserTower must clear Hit@10 = 0.263 (popularity) to claim
  value and must leave the single ability axis to do so.
- **2026-06-04 05:35.** M1 work-in-progress on `feat/online-step-api`,
  branch tip `dd1d8bf`. Six commits land the online step API end to
  end. The encoder ABC gains `forward_with_state`, implemented for
  DKVMN, LSTM, and Transformer. The decoder ABC gains
  `compute_logits_from_state` (default delegates to `forward`) and
  `irt_parameters()` overrides for GPCM and Rasch.
  `EncoderDecoderModel.step(item_id, response, state, sigma_prior)`
  drives the encoder plus decoder for one timestep and returns a
  fresh `StepState` carrying theta_t, sigma_t, the encoder carry,
  the running alpha_log and beta_log, and the item_log audit trail.
  Sigma_t is computed via observed Fisher
  (`gpcm_observed_fisher` in `components/irt.py`) so each step costs
  O(1) rather than O(t). `freeze_irt(flag)` flips requires_grad on
  decoder IRT sub-networks only. Spec doc at
  `ma-irt/docs/step_api.md`. Tests: 13 parity assertions in
  `tests/test_step_api.py` (theta, alpha, beta, logits, probs match
  batched forward to atol=1e-5 across all three encoders; sigma at
  init is prior std; freeze_irt toggles grads; item_log accumulates;
  DKVMN value memory mutates). All pass. CPU latency at t=200 in
  `tests/test_step_microbenchmark.py` is well inside budgets
  (DKVMN <20ms, LSTM <10ms, Transformer <40ms). All pass.
- **2026-06-04 05:15.** M0 committed at `8a0cb4c`. Planning docs
  (plan v1, synthesis, evidence, track assessment, track
  recommendation) and the progress log are now under version
  control. The `rl/` scaffold and O*NET pool landed earlier in
  commit `1c6386a`. Working tree is clean of DRL-MAIRT changes;
  ready to branch `feat/online-step-api` for M1.
- **2026-06-04.** Progress log initialized. See
  [`drl_mairt_plan_v1.md`](drl_mairt_plan_v1.md) for the full spec.
- **2026-06-04.** Plan v1 locked. Eight decisions committed. See
  [`drl_mairt_plan_v1.md`](drl_mairt_plan_v1.md).

---

## 4. Test inventory

Tests are listed by file with milestone and current implementation status.
"Not yet implemented" means the file does not exist on disk. "Stub" means
the file exists with a skeleton but no real assertions. "Passing" means
the test runs and asserts the documented behavior.

| Test file | Milestone | Coverage | Status |
|---|---|---|---|
| `ma-irt/tests/test_step_api.py` | M1 | Iterated `step()` parity vs batched `forward()` to atol=1e-5 on logits/probs/theta/alpha/beta, initial-state sigma equals prior, freeze_irt grad toggling, item_log accumulation, DKVMN value-memory mutation. 13 parametrized cases across DKVMN, LSTM, Transformer. | passing on `feat/online-step-api` (13/13) |
| `ma-irt/tests/test_step_microbenchmark.py` | M1 | CPU per-step latency at t=200 under budgets DKVMN <20ms, LSTM <10ms, Transformer <40ms. Marked `@pytest.mark.benchmark`. | passing on `feat/online-step-api` (3/3) |
| `rl/tests/test_retrieval.py` | M2 | Pool schema load, attach_text format, education mask invariant, ItemTower output L2-normalised, text encoder is frozen, top-K determinism over a fixed pool, top-K respects mask, input validation, pool-swap round trip on a 50 occupation fake pool. 9 cases. | passing on `main` (9/9) |
| `rl/tests/test_synth_generator.py` | M3 | All Section 5.5 sanity checks. Schema files exist, sequences/jobs/likes/true_irt_parameters/true_preference_parameters/metadata schemas, byte-identical reruns at fixed seed, like rate within +/-0.02 of target, engagement class shares within +/-0.02, K distribution matches config, per-user response count >= 30, finite delta_j, rejecter users emit only zero likes, per-user candidate set sizes in the configured clip. 16 cases. | passing on `main` (16/16) |
| `rl/tests/test_belief_tracker.py` | M4 | ma-irt `state_dict` byte-equal before and after 100 `on_rate` calls (ratings never reach ma-irt, Section 3.3 rule); debounce policy fires correctly on contrived theta and h_t trajectories. | not yet implemented |
| `rl/tests/test_reflection_cap.py` | M5 | Adversarial dislike-every-recommendation trajectory does not collapse q_t to last v_j; cosine-shift cap at 0.2 fires; per-session reset clears prior likes. | not yet implemented |
| `rl/tests/test_fisher_selector.py` | M5 | Toy IRT bank with known optimal items, MFI picks them at t>=5; KL-info cold-start fallback fires for t<5; randomesque exposure mask picks from top n=5. | not yet implemented |
| `rl/tests/test_controller.py` | M5 | Confidence flag logic (rho_high, jaccard floor, cold_start_min); terminate rules (user stop, max_items, rho_terminate); `offer_more_questions` flips after `contradict_threshold` rating contradictions. | not yet implemented |
| `rl/tests/test_e2e.py` | M5 | End-to-end smoke. Drive one held-out synthetic student through respond, rate, stop; assert top-K materialized and confidence flag matches expected branch. | not yet implemented |

GPU-hour budgets per milestone are listed inside the plan (Section 11).

Smoke and infrastructure tests (e.g., import-only sanity, config schema
parsing, pyproject install check) will be added under `rl/tests/` as they
become useful and tracked here as a single "rl smoke" row.

---

## 5. Known issues and open decisions

- **Open decision, M1.** Transformer `forward_with_state` uses
  prefix-recompute over the accumulated `v_history` and
  `q_signal_history`. Parity to the batched forward is by construction
  but per-step cost is O(t), per-session cost O(T^2). The plan section
  6.1 latency budget at t=200 is met empirically on CPU. A true
  KV-cache implementation drops the constant factor but requires a
  custom attention layer that exposes the cache. Deferred to v2 once
  the latency budget is load-bearing at higher T or on real users.
- **Open decision, M1.** Sigma_t assumes D=1. Multidimensional Fisher
  information at theta is a v2 concern (plan section 10.5), aligned
  with locked decision D2 (1D theta in v1).
- **Limitation, M1.** Encoders without `forward_with_state`
  (currently `dkt_gru`) raise `NotImplementedError` from the Encoder
  ABC default. Add the method per encoder as the recommender takes on
  new backbones. The three encoders supported by M1 (DKVMN, LSTM,
  Transformer) cover all paper-equivalent configurations.
- **Limitation, M1.** Decoders without IRT parameters (binary,
  softmax) return `theta_t=0`, `sigma_t=inf`, empty `alpha_log` and
  `beta_log`. Logits and probs are still populated; IRT fields are
  placeholders. The DRL-MAIRT recommender uses GPCM, so this does not
  affect the v1 path.

Expected categories.

- **Issue.** Unexpected behavior or failing test, with the test or commit
  that surfaced it and the workaround if any.
- **Open decision.** A choice the plan deferred or did not resolve, with
  the constraints we have learned since.
- **Deviation from plan.** A documented divergence from the spec, with
  the reason and the agreement record.

---

## 6. How to use this document

This file is the source of truth for the cross-milestone build state. The
next autonomous workflow run should:

1. Read this file in full before doing anything else.
2. Read the canonical spec [`drl_mairt_plan_v1.md`](drl_mairt_plan_v1.md)
   to confirm scope.
3. Update the **status snapshot** date, current state, active branch.
4. Append a new entry to the **change log** for each significant event
   (milestone start, milestone DoD met, test added, decision changed,
   issue discovered).
5. Update the **milestones table** when status changes.
6. Update the **test inventory** when test files are added, change
   status (stub to passing, passing to failing), or are removed.
7. Append to **known issues and open decisions** as they arise.

Keep this file under ~250 lines. If a section grows past its share, split
the detail into a separate doc under `docs/` and link to it from here.

### 6.1 Style

- American English.
- No em-dashes or en-dashes.
- No colons in flowing prose. Tables, code blocks, and label-style
  bullets are fine.
- Cite literature with author-year + venue where it grounds a design
  choice.

### 6.2 Cross-references

- Canonical spec, [`drl_mairt_plan_v1.md`](drl_mairt_plan_v1.md).
- Research synthesis, [`drl_mairt_synthesis.md`](drl_mairt_synthesis.md).
- Evidence review, [`drl_mairt_evidence.md`](drl_mairt_evidence.md).
- Background, [`drl_mairt_background.md`](drl_mairt_background.md).
- Recommendation track assessment,
  [`drl_mairt_track_recommendation.md`](drl_mairt_track_recommendation.md).
- Assessment track,
  [`drl_mairt_track_assessment.md`](drl_mairt_track_assessment.md).
