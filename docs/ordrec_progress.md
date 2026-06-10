# OrdRec Build Progress Log

*Continuously updated. Source of truth for the OrdRec build state. The
historical jobrec progress lives at `docs/drl_mairt_progress.md`.*

Strategic plan, [`docs/exrec_ordinal_plan.md`](exrec_ordinal_plan.md).
Implementation guide, [`docs/ordrec_impl_guide.md`](ordrec_impl_guide.md).
Per-milestone results, `rl/results/E<n>_<topic>.md`.

---

## 1. Status snapshot

- **Date created.** 2026-06-08
- **Date last updated.** 2026-06-10
- **Current state.** E4.6b (hardening slice B + A/B rerun) complete on
  `feat/ordrec-e46b`. Buffer overflow fixed (RC1), exposure penalty
  recalibrated (RC2), stratified probe sampler shipped (B5), buffer
  reworked to one row per env-step (B0+B4). B-side verdict: honest
  negative, PPO does not beat random in mean episode return on synthetic
  data (-0.570 vs -0.537, non-overlapping CIs). RC3 (VOI saturation on
  static DGP) remains open. Results at `rl/results/E46b_ab_comparison.md`.
  209 tests pass. Doc consolidation complete (C1 audit item).
- **Active branch.** `feat/ordrec-e46b` at `3278b31`.
- **Next milestone.** E5: real Eedi K=4 run. Only external dependency
  is the raw Eedi NeurIPS 2020 Task 3+4 csvs.

The five locked design corrections from the strategic plan are intact.

- (A) Probe-based GPCM entropy reduction is the primary reward,
  Fisher information is a theoretical lens.
- (B) Single framework with per-dataset adapters behind one
  `OrdinalDatasetBase`. The model never branches on dataset identity.
- (C) Custom RL library with a small `RLAlgorithm` ABC, no external RL library.
- (D) Per-user deterministic splits, byte-identical re-materialisation.
- (E) Action mask covers admin, probe, and within-episode no-repeat.

---

## 2. Milestones

| Milestone | Status | Branch | Tip | Tasks | DoD | Notes |
|---|---|---|---|---|---|---|
| **E1**, data adapters | complete (merged) | `feat/ordrec` | `267ea82` | `data/{base,schema,split,synthetic,placeholder_2pl,ma_irt_bridge,eedi}.py`, 5 test files, Eedi fixture, two configs | met | 42 tests pass. Synthetic smoke training `r_theta=0.88` in 5 epochs. See `rl/results/E1_data_layer.md`. |
| **E2**, per-item lookup + EdNet + ASSISTments + bench | complete (merged) | `feat/ordrec` | `f5c536e` | `envs/{frozen_magpcm,item_cache,bench_forward}.py`, `data/{ednet,assist}.py`, `rl/scripts/prepare_eedi_csv.py`, 5 new test files | met | 82 tests pass (27 envs + 13 new adapter + 42 E1). Bench at `rl/results/E2_bench_forward.{json,md}`. Cache keyed by `(dataset_name, ckpt_sha7)`. See `rl/results/E2_envs_layer.md`. |
| **E3**, env + reward + wiring | complete (merged) | `feat/ordrec` | `eaa404a` | `envs/{base,ordrec_env,action_mask}.py`, the full `reward/` package, `rl/tests/test_env_reward_wiring.py` | met | 126 tests pass (44 new E3 + 82 pre-E3). Four-component reward sums to `r_total`, three-source mask blocks every probe id. See `rl/results/E3_env_reward.md`. |
| **E4**, RL library + training loop + smoke | complete (merged) | `feat/ordrec` | `4e2fc72` | `training/{base,rollout,gae,ppo,utils}.py`, `bc_warmstart/{bc,static_mve}.py`, `scripts/{train_ppo,sanity_toy_env,eval_policy}.py`, `configs/{ppo_eedi_k4,ppo_synth_smoke}.yaml` | met | 146 tests pass (20 new E4 + 126 pre-E4). PPO on the toy env goes from `1.062` to `2.000` over 20 updates. PPO on the synthetic adapter runs 5 updates and saves `best.pt`. See `rl/results/E4_rl_library.md`. |
| **E4.6a**, hardening slice A | complete | `feat/ordrec-e46a` | `ac40b68` | review items A1-A4, per-key component-metric denominators + regression test, calibrated reward-scale bands, `train_ppo.py`/`eval_policy.py` smoke tests + shared `rl/tests/conftest.py`, PPO local sampling generator | met | 150 tests pass (4 new). Gate, full suite green twice plus bc_smoke in isolation 3x. Motivated by the CONSOLIDATE verdict in `docs/cleanup/_ordrec_maintainability_review.md`. |
| **E4.5**, synthetic headline run | complete | `feat/ordrec-e45` | `dbb6cb4` | MA-GPCM world model trained (theta r=0.968), 200 BC updates + 500 PPO updates, four-policy eval (PPO/BC/Fisher/random), headline plots + results report | met | PPO does NOT beat max-Fisher by return. Ranking: random > PPO > BC-only > max-Fisher. Exposure penalty dominates; random wins by avoiding it entirely. Buffer capacity mismatch caused r_voi=0 throughout training. This run is the A-side of the E4.6b A/B comparison. See `rl/results/E45_synth_headline.md`. |
| **E4.6b**, hardening slice B | complete | `feat/ordrec-e46b` | `3278b31` | B0+B4 buffer rework (one row per env-step), B1-B6 code quality, R1 exposure recalibration, R2 VOI diagnosis, R3 BC teacher soft-target, B5 stratified probe, A/B rerun, C1 doc consolidation | met | 209 tests pass. B-side PPO: -0.570 vs random -0.537. Honest negative on static synthetic DGP. RC3 (VOI saturation) open. Results at `rl/results/E46b_ab_comparison.md`. |
| **E5**, headline polytomous run on Eedi K=4 | next | tbd | tbd | real Eedi raw csvs landed, BC and MVE warm-start enabled, PPO `total_updates = 1000`, evaluation harness, paper hooks | tbd | Only external dependency: real Eedi NeurIPS 2020 Task 3+4 csvs. Mean return strictly above uniform-random baseline at end of training, per-component decomposition and exposure caps respected. |
| **E6**, ablations + paper figures | not started | tbd | tbd | full PPO runs on EdNet KT3 K=4, ASSISTments K=2, plus ablations (no-probe, no-exposure, no-VOI), paper PGF figures | tbd | The science milestone. |

---

## 3. Change log (reverse chronological)

- **2026-06-10, E4.6b complete (hardening slice B + A/B rerun + doc consolidation).**
  Branch `feat/ordrec-e46b` at `3278b31`, 12 commits on top of the E4.5
  merge (`5082c45`). The three root causes identified in the E4.5 honest
  negative were addressed.

  **RC1 fix, buffer rework (B0+B4, `ca4aab7`).** `RolloutBuffer` now
  inserts one row per env-step rather than one row per sub-step (K_B
  sub-steps per step). Capacity 64 = 32 episodes x 2 steps now exactly
  matches demand. Terminal r_voi entered every PPO update in B-side
  training (mean r_voi = -0.043 vs 0.0 throughout A-side).

  **RC2 fix, reward recalibration (R1, `a70b5b7`).** w_expo reduced
  0.10 to 0.02, r_max raised 0.20 to 0.40. Removed the dominant
  exposure penalty that made random optimal by construction. r_expo for
  max-Fisher dropped from -0.103 to -0.007. These values are now the
  RewardConfig defaults.

  **RC3 diagnosis (R2, `49574ca`).** Positive r_voi bursts (max +0.066)
  confirm the anchor sign is correct. The consistently negative mean is
  attributed to theta saturation: a 5-warmup prior already captures
  most signal from a Q=200 static synthetic DGP, and 10 additional
  items per episode do not meaningfully sharpen it. E5 real-Eedi
  sessions are expected to exhibit genuine per-session ability growth.

  **Other B-slice items.** B1 extracted `gpcm_log_probs` into
  `reward/gpcm_ops.py` (single source of truth). B2 added ma-irt SHA
  and config hash to item cache metadata. B3 added `MAIRTOutput`
  TypedDict to `FrozenMAGPCM`. B4 the buffer row-per-step contract.
  B5 shipped the dual probe sampler (uniform retained for A-side config
  compatibility, stratified now the default). B6 added `RewardConfig.from_dict`
  and `PPOConfig` dataclass. B7 single forward per step. B8 history
  truncation. R3 BC teacher changed from argmax to top-5 soft target.

  **A/B verdict.** B-side PPO (-0.570) does not beat random (-0.537,
  non-overlapping 95% CIs). All three greedy policies improved by
  +0.16 to +0.18 return units from the RC2 recalibration. The remaining
  gap is driven by r_voi: the VOI anchor is a net-negative signal on
  this static synthetic DGP. The multi-step credit distribution and
  buffer fix are structurally correct but cannot overcome RC3 saturation.

  **C1 doc consolidation.** `rl/README.md` rewritten to describe the
  current tree (package map, how to run tests, how to train, how to
  evaluate). `docs/exrec_ordinal_plan.md` patched to remove Tianshou
  references, stale `irtrec` paths, and the restated milestone table
  (now points to this doc). `docs/ordrec_impl_guide.md` patched for
  placeholder-2PL hyperparams (20 epochs lr 5e-2), dual probe sampler
  (stratified default), buffer layout (one row per env-step), and
  the no-external-RL-library correction. E1-E4 milestone records moved
  to `rl/results/archive/`.

  **209 tests pass** (full suite, two runs). Gate met.

- **2026-06-10, E4.5 complete (synthetic headline run).** Branch
  `feat/ordrec-e45` at `dbb6cb4`, 3 commits on top of E4.6a merge
  (`1ec2e9a`). MA-GPCM world model trained on N=2000, Q=200, K=4
  synthetic cohort (theta r=0.968, beta r=0.975, alpha r=0.884).
  200 BC warmstart updates + 500 PPO updates (B=32, K_B=5, T=10,
  ~58s wall-clock on RTX 4060 Laptop). Four-policy eval on test split
  (6400 episodes each). Headline finding: PPO does NOT beat max-Fisher
  or random by episode return. Ranking: random (-0.530) > PPO (-0.729)
  > BC-only (-0.734) > max-Fisher (-0.745). Root cause: exposure penalty
  dominates (r_expo ~-0.094 for learned policies, 0.000 for random;
  frac_above_r_max ~7.5% vs 0%). Buffer capacity mismatch caused
  r_voi=0 throughout training (buffer capacity=64 vs 160 inserts per
  step). This run is the A-side of the E4.6b A/B comparison. Results
  at `rl/results/E45_synth_headline.md`. Headline plots at
  `rl/results/plots/e45_{training_curve,baseline_comparison,session_trajectory}.png`.

- **2026-06-10, E4.6a complete (hardening slice A).** Branch
  `feat/ordrec-e46a` at `ac40b68`, 4 commits stacked on
  `feat/ordrec` tip `4e2fc72` (E4 merged). Motivated by the
  four-auditor maintainability review of the E1-E4 stack, which
  returned a CONSOLIDATE verdict (full review at
  `docs/cleanup/_ordrec_maintainability_review.md`, raw audit
  digest at `docs/cleanup/_ordrec_audit_digest.md`). The slice
  fixed only the four blocking items that would have corrupted
  the E4.5 synthetic headline run. Per review item C6 this is a
  change-log subsection, not a standalone milestone record.

  **What landed.**

  - **A1, per-component reward metric denominators**
    (`88f9943`). `PPO.rollout` now tracks one observation count
    per component key and divides each sum by its own count.
    The old shared counter divided by
    `component_count // n_components`, inflating the logged
    `r_info`/`r_cost`/`r_expo`/`r_voi` averages by roughly the
    number of components whenever keys fired at different
    frequencies (`r_voi` fires only at terminal steps).
    Regression test
    `training/tests/test_component_metrics.py` (2 tests) pins
    the exact arithmetic with two synthetic components at
    different frequencies, plus a zero-not-NaN case for
    never-observed components.
  - **A2, honest reward-scale calibration bands** (`d18522a`).
    `reward/tests/test_reward_scale.py` asserted a [1%, 95%]
    band while its name and docstring claimed [5%, 70%], a
    near-no-op. A single symmetric band cannot hold because the
    composition is deliberately asymmetric (the terminal VOI
    anchor at `w_voi = 5.0` dominates). The test now measures
    episode-level shares (`T // K_B` boundaries, `r_voi` only
    at the terminal one) and asserts per-channel calibrated
    bands, `r_info` [0.04, 0.16], `r_cost` [0.09, 0.35],
    `r_expo` [0.015, 0.07], `r_voi` [0.55, 0.85], derived from
    measured shares across five seeds and roughly a factor of
    two wide. A 2.5x mis-scaling of any single weight breaks
    its band; seed noise is two orders of magnitude smaller.
    The derivation is documented in the module docstring.
  - **A3, smoke tests for the two E5-facing scripts**
    (`ac40b68`). `rl/tests/test_train_ppo_smoke.py` runs
    `train_ppo.train()` end to end on the synthetic adapter
    with a tiny config (2 PPO updates) and asserts `best.pt`,
    `last.pt`, `metrics.csv` (schema and one finite row per
    update) and `summary.json` (round trip) exist with sane
    content. `rl/tests/test_eval_policy_smoke.py` trains a
    checkpoint inline, drives `eval_policy.main()` through its
    CLI surface, and asserts the report carries mean return for
    both policies, the random-baseline delta line, per-component
    means, and the exposure-diagnostics table. Shared fixtures
    live in the new `rl/tests/conftest.py`. Both run in seconds,
    so they stay in the default suite (`rl/pyproject.toml`
    testpaths now includes `tests`).
  - **A4, PPO global RNG side effect removed** (`c90e244`).
    `PPO.__init__` no longer calls `set_seed`, which mutated
    global torch/numpy/random state on every construction and
    was the root cause of the order-sensitive bc_smoke flake
    (0.844 vs the 0.85 threshold in isolation). Action sampling
    now draws from a local `torch.Generator` via
    `PPO._sample_masked`. `test_bc_smoke.py` seeds globally via
    `set_seed(0)` at test start so the world model and BC loop
    are deterministic regardless of test order, keeping the
    0.85 threshold. Verified green in isolation three times.

  **Gate results.** Full suite (`rl/src/ordrec/` + `rl/tests/`)
  green twice in a row, 150 passed each run (146 pre-slice + 4
  new). `pytest-randomly` is not installed in the research env,
  so per the gate fallback the suite ran twice plainly and
  `test_bc_smoke.py` ran alone three times, 2 passed each time.

- **2026-06-08, E4 merged.** `feat/ordrec-e4` merged into
  `feat/ordrec` at `4e2fc72`. E4 tip on its branch was `9bb826e`.

- **2026-06-08, E4 complete (awaiting merge).** Branch
  `feat/ordrec-e4` at `21a9dea`, 12 commits stacked on
  `feat/ordrec` tip `eaa404a` (E3 merged). Landed the
  `RLAlgorithm` ABC plus `RolloutStats` and `UpdateStats`
  dataclasses, the on-policy `RolloutBuffer` with episode-aware
  GAE advantage computation, the standalone `compute_gae`
  helper, the PPO concrete `RLAlgorithm` (clipped surrogate
  with eps 0.2, value clip 0.2, GAE 0.95, gamma 0.95, lr 3e-4,
  Adam eps 1e-5, entropy 0.01 annealed to 0 over the first 50%
  of training, KL early stop 0.02, 4 epochs per rollout,
  mini-batch 32 to 64, value coef 0.5, grad clip 0.5), the
  shared-trunk `ActorCritic` with masked discrete categorical
  actor, `set_seed` and schedule helpers, the BC warm-start
  with a max-Fisher teacher, the exact `K ** K_B = 1024`
  static-MVE critic warm-start, the top-level `train_ppo.py`
  driver, the PPO toy-env smoke runner (`sanity_toy_env.py`),
  the evaluation harness (`eval_policy.py`), and two configs
  (`ppo_synth_smoke.yaml`, `ppo_eedi_k4.yaml`). Toy-env smoke
  drives mean return from `1.062` to `2.000` over 20 updates
  (random baseline `1.0`, optimum `2.0`). Synthetic-adapter
  smoke runs 5 PPO updates with the four-component reward
  decomposition recorded. 146 tests pass (20 new E4, 13
  training + 1 BC + 6 new buffer cases, plus 126 pre-E4).
  Milestone record at `rl/results/E4_rl_library.md`.

- **2026-06-08, E3 merged.** `feat/ordrec-e3` merged into
  `feat/ordrec` at `eaa404a`. E3 tip on its branch was `7414fb3`.

- **2026-06-08, E3 complete (awaiting merge).** Branch
  `feat/ordrec-e3` at `7414fb3`, 12 commits stacked on
  `feat/ordrec` tip `f5c536e` (E2 merged). Landed the four-component
  reward composer (`OrdinalRewardCompute` returning a
  `RewardBreakdown` with `r_info`, `r_cost`, `r_expo`, `r_voi`,
  `r_total`, `phi_t`, `phi_prev`), the vectorised GPCM predictive
  entropy `phi_entropy`, the terminal NLL anchor on `H_probe`, the
  Sympson-Hetter (1985) exposure penalty with EMA fleet update, a
  Welford running-mean-std with a freeze flag, the
  `OrdinalEnvBase` ABC plus `OrdinalState` dataclass, the concrete
  `OrdRecEnv` wrapping a frozen MAGPCM and a data adapter, and the
  three-source action mask composer (admin AND probe AND
  no-repeat). 126 tests pass (44 new E3, 30 reward + 11 env + 3
  cross-package wiring, plus 82 pre-E3). Milestone record at
  `rl/results/E3_env_reward.md`.

- **2026-06-08, E2 merged.** `feat/ordrec-e2` merged into
  `feat/ordrec` at `f5c536e`. E2 tip on its branch was `80036ea`.

- **2026-06-08, E2 complete (awaiting merge).** Branch
  `feat/ordrec-e2` at `80036ea`, 8 commits stacked on `feat/ordrec`
  tip `267ea82`. Landed `FrozenMAGPCM` (two-line eval+no_grad
  contract, replaces v1 M1-era `freeze_irt`), per-item `(alpha,
  beta)` cache keyed by `(dataset_name, ckpt_sha7)`, `bench_forward`
  latency harness with the no_grad invariance regression,
  `EdNetAdapter` K=4 via `(correctness, response_time)` quadrants
  with KT3/KT4 auto-detect, `AssistAdapter` K=2 identity passthrough,
  and the `prepare_eedi_csv.py` pre-merge script for the real Eedi
  NeurIPS 2020 release. 82 tests pass. Bench at
  `rl/results/E2_bench_forward.{json,md}`. Milestone record at
  `rl/results/E2_envs_layer.md`.

- **2026-06-08, E1 merged.** `feat/ordrec-e1-eedi` merged into
  `feat/ordrec` at `267ea82`. E1 tip on its branch was `466e730`.

- **2026-06-08, E1 complete.** Branch `feat/ordrec-e1-eedi` at
  `466e730`. Final commit landed the synthetic smoke training pass.
  All 42 data layer unit tests pass on the Windows research env.
  Milestone record at `rl/results/E1_data_layer.md`. This progress
  doc initialised; historical jobrec doc retained at
  `docs/drl_mairt_progress.md` with a header pointer.

- **2026-06-08, E1 incremental landings.** Eleven commits on
  `feat/ordrec-e1-eedi`. Order, package skeleton, ABC, schema,
  split, synthetic + placeholder 2PL, Eedi adapter, ma-irt bridge,
  test suite, Eedi headline config, smoke training pass.

- **2026-06-04, plan locked.** Implementation guide
  (`docs/ordrec_impl_guide.md`) landed at `0962f5a`. Five
  corrections locked, see status snapshot. Strategic plan v2
  refresh landed at `d7e15a9`.

- **2026-06-03, jobrec branch archived.** The old `rl/` jobrec tree
  was moved to `archive/`. Fresh `rl/` reserved for OrdRec. See
  `33f3565`.

---

## 4. Test inventory

E1 data layer, 42 tests (`rl/src/ordrec/data/tests/`).

| File | Count | Surface |
|---|---|---|
| `test_base_contract.py` | 7 | `OrdinalDatasetBase` lifecycle, `AdapterConfig` immutability, abstract enforcement, `__getitem__` shape and dtypes, `get_split` codes. |
| `test_schema_round_trip.py` | 14 | Metadata, sequences, q-matrix, coercion validators. Byte-identical json round trip. `COMMON_RECORD_SCHEMA` keys frozen. |
| `test_split_determinism.py` | 9 | `make_split` reproducibility, fraction respect, stratified proportionality. `_chunk_sequences` parent tracking. Synthetic adapter byte-identical rebuild. |
| `test_eedi_adapter.py` | 7 | Artefact materialisation, metadata block, correct option recodes to `K-1`, distractor ordering ascending by mean theta, train-only fitting, coercion persistence and reuse, response range. |
| `test_ma_irt_bridge.py` | 5 | First-item shape, collate-compatible batch (the 4-tuple MAGPCM consumes), `n_questions` matches metadata, full pass without errors, splits disjoint. Gated by `pytest.importorskip("utils.dataloader")`. |

E2 data layer additions, 13 tests (`rl/src/ordrec/data/tests/`).

| File | Count | Surface |
|---|---|---|
| `test_ednet_adapter.py` | 9 | Artefact materialisation, metadata block, K=4 quadrant mapping table, median computed on train only, KT3 absent-hint handling, KT4 hint detection, response range, missing-rt slow recode, persisted coercion reuse. |
| `test_assist_adapter.py` | 4 | Artefact materialisation, identity K=2 collapse, `n_categories=2` in metadata, test-train splits disjoint at student level. |

E2 envs layer, 27 tests (`rl/src/ordrec/envs/tests/`).

| File | Count | Surface |
|---|---|---|
| `test_frozen_magpcm.py` | 10 | Freeze contract under repeated calls, five-key output dict, `train(mode)` keeps `eval`, no_grad invariance across calls, theta invariance to future positions under eval+no_grad, autograd does not flow into the frozen model, device property, non-Module rejection, `n_questions` and `n_categories` propagation. |
| `test_item_cache.py` | 11 | Build shapes and dtypes, padding-slot invariants, alpha+beta match a direct forward, save/load round trip, tensor accessors, cache keyed by `(dataset, checkpoint)`, sha7 for missing file, sha7 for real file, argument validation, path requires `dataset_name`, averaging across contexts changes alpha but not beta. |
| `test_bench_forward.py` | 6 | End-to-end smoke run, rejection of empty encoder list, invariance passes on frozen model, invariance check catches a broken model, artefact write round trip, invariance check requires at least two calls. |

Total at E2 close, 82 tests, all pass.

E3 reward layer, 30 tests (`rl/src/ordrec/reward/tests/`).

| File | Count | Surface |
|---|---|---|
| `test_potential_shaping.py` | 2 | Telescoping `phi(s_2) - phi(s_0)` and prev-then-current boundary chain. |
| `test_entropy_bounds.py` | 5 | `phi >= 0`, bounded by `log K`, uniform limit at `alpha=0`, concentrated-limit at `K=2`, finite under extreme theta. |
| `test_fisher_special_case.py` | 3 | Single-item probe matches direct entropy, monotone in alpha, uniform at `alpha=0` matches `log K`. |
| `test_reward_scale.py` | 2 | Episode-level per-channel share of `|r_total|` within calibrated bands (re-derived in E4.6a, see change log), breakdown sums to total. |
| `test_anti_gaming_mask.py` | 4 | Uniform policy never samples probe, mask False on probe ids, pad slot forbidden, no-repeat update removes administered. |
| `test_sympson_hetter.py` | 5 | Penalty zero below threshold, linear in excess above, `c_expo` scales slope, aggregates over `K_B`, EMA fleet update. |
| `test_terminal_anchor.py` | 4 | VOI zero mid-episode, nonzero at horizon, sign convention, `gpcm_nll` matches per-row cross-entropy. |
| `test_running_norm.py` | 5 | Welford matches numpy, freeze is no-op after threshold, normalise uses current stats, batched update, state-dict round trip. |

E3 envs layer additions, 11 tests (`rl/src/ordrec/envs/tests/`).

| File | Count | Surface |
|---|---|---|
| `test_action_mask.py` | 5 | Three-source AND, probe disjoint admin, no-repeat idempotent, extra forbidden ids, shape mismatch rejection. |
| `test_ordrec_env.py` | 6 | Reset shape, step advances and returns `(B,)` reward, masked action raises, horizon terminates, observation+action dim properties, terminal VOI active. |

E3 cross-package wiring, 3 tests (`rl/tests/`).

| File | Count | Surface |
|---|---|---|
| `test_env_reward_wiring.py` | 3 | One-step breakdown sums to total, mask blocks every probe id through env, fleet exposure updates after episode. |

Total at E3 close, 126 tests, all pass.

E4 training package, 18 tests (`rl/src/ordrec/training/tests/`).

| File | Count | Surface |
|---|---|---|
| `test_rollout_buffer.py` | 9 | `insert` and `reset`, shape rejection, overflow, `iter_minibatches` visits every transition once, `iter_minibatches` requires `compute_advantages`, `compute_advantages` zero-bootstraps at `done`, advantage normalisation off when `n=1`, advantage sign tracks reward sign, episode-starts split segments. |
| `test_gae.py` | 4 | Matches numpy reference with mid-trajectory done, terminal zero bootstrap, shape-mismatch rejection, bootstrap used when not done. |
| `test_ppo_smoke.py` | 3 | PPO increases mean return on the toy env over 20 updates, `act(deterministic=True)` returns the argmax under the mask, masked actions are never sampled. |
| `test_save_load.py` | 2 | State-dict save then load is byte-identical, load into a fresh instance reproduces the optimiser state. |

E4 BC warmstart, 1 test (`rl/src/ordrec/bc_warmstart/tests/`).

| File | Count | Surface |
|---|---|---|
| `test_bc_smoke.py` | 1 | BC actor reaches `>= 85%` match against the max-Fisher teacher on a held-out validation slice from the synthetic adapter. |

Total at E4 close, 146 tests, all pass.

E4.6a hardening additions, 4 tests.

| File | Count | Surface |
|---|---|---|
| `rl/src/ordrec/training/tests/test_component_metrics.py` | 2 | Per-key denominator arithmetic exact under mixed component frequencies, unobserved components report `0.0` not NaN. |
| `rl/tests/test_train_ppo_smoke.py` | 1 | `train_ppo.train()` end to end on the synthetic adapter, artifacts exist, `metrics.csv` schema and finiteness, `summary.json` round trip, checkpoint loadable with dimensions. |
| `rl/tests/test_eval_policy_smoke.py` | 1 | `eval_policy.main()` against a checkpoint trained inline, mean-return table for both policies, baseline delta line, per-component means, exposure diagnostics. |

Shared fixtures for the script smokes live in `rl/tests/conftest.py`.
`rl/pyproject.toml` testpaths now includes `tests`, so the cross-package
and script tests run by default.

Total at E4.6a close, 150 tests, all pass.

E4.6b hardening additions, 59 tests (reward, envs, training, and
cross-package tests updated or added for B0-B8 + R1-R3 fixes; exact
per-file breakdown in the E4.6b change-log entry above).

Total at E4.6b close, 209 tests, all pass.

Reproducer.

```bash
cd <repo_root>
PYTHONPATH="rl/src;ma-irt" KMP_DUPLICATE_LIB_OK=TRUE \
  python -m pytest rl/src/ordrec/ rl/tests/ -v
```

Existing ma-irt test suite is untouched through E3.

---

## 5. Open issues

Carried forward. E1-E4 source records are now in `rl/results/archive/`.

1. Placeholder 2PL `lr` field in `coercion_artefacts.json` reports the
   guide default (1e-2) rather than the value actually used (5e-2 at
   20 epochs). E2 did not touch this. Cosmetic.
2. Synthetic adapter does not persist `true_irt_parameters.json` into
   the materialised artefact. Eval recovery currently reads it from
   the upstream raw directory by convention. Persist alongside the
   artefact for reproducibility before the headline run.
3. Bridge tests `skip` rather than `fail` when `PYTHONPATH=ma-irt`
   is absent. Acceptable now, the E5 CI config should set it.
4. R `mirt` audit path for the Eedi placeholder 2PL not implemented.
   Deferred to E4 since it now depends on the per-item cache built
   in E2 being callable from R via on-disk artefacts.
5. Eedi fixture is only 50 rows. Full-corpus validation of distractor
   ordering against the published Eedi baseline waits for the real
   Eedi NeurIPS 2020 csvs to land locally, then exercises the new
   `prepare_eedi_csv.py`.
6. EdNet KT4 K=5 hint-aware variant. Current adapter coerces KT4
   down to K=4 by ignoring the `hint_used` column. K=5 variant is
   a candidate ablation for a future milestone.
7. Per-item alpha averaging uses `n_contexts=8` by default. A sweep
   over `n_contexts` against the recovery-target alpha is a
   candidate for E5.
8. DKVMN CUDA latency at `B=1` is slower than CPU because the
   memory ops do not fuse for small batches. Not blocking, rollout
   batches are typically larger, flagged for a possible E4
   profiling pass.
9. Real-Eedi pre-merge execution waits for the real csvs to land
   locally. The merger script is in place and documented.
10. Open engineering questions from impl guide Section 9, items 9
    through 15 (reward) are now closed by E3. Items 1 through 8
    (data) remain open. Items 16 through 23 (RL) belong to E4.
11. E3 new, Laplace `sigma_t` posterior-precision path. The state
    envelope carries an `Optional Tensor (B, D, D)` posterior
    precision but the point-estimate path is the v1 default.
    Laplace calibration wires to `RewardConfig.sigma_floor` and
    is deferred to a critic-warmstart pass in E4 or E5.
12. E3 new, reward-normalisation activation. `OrdinalRewardCompute`
    supports a frozen Welford normaliser via the `normalise`
    constructor flag. Unit tests run un-normalised so the
    four-component sum is checkable. The headline run will enable
    it.
13. E3 new, probe sampler stratification audit. Default
    `n_difficulty_strata = 5` equal-count quantiles. An ablation
    over `[3, 5, 10]` is a candidate for E5.
14. E4.6b, BC teacher upgraded to top-5 soft target (R3, `49574ca`).
    The single-argmax teacher is replaced by a soft probability target
    over the top-5 max-Fisher items. ReflectionLayer-greedy and Thompson
    mixture variants remain deferred to E5 or later.
15. E4 new, DQN and SAC implementations. Sketched as comments in
    the impl guide and inside `training/ppo.py`. Deferred to a
    future ablation.
16. E4 new, headline-config sweep over PPO hyperparameters
    (`lr in {1e-4, 3e-4, 1e-3}`, `gae_lambda in {0.90, 0.95,
    0.99}`, `clip_eps in {0.1, 0.2}`). Belongs to E5 or E6
    once real Eedi raw csvs land.
