# OrdRec Build Progress Log

*Continuously updated. Source of truth for the OrdRec build state. The
historical jobrec progress lives at `docs/drl_mairt_progress.md`.*

Strategic plan, [`docs/exrec_ordinal_plan.md`](exrec_ordinal_plan.md).
Implementation guide, [`docs/ordrec_impl_guide.md`](ordrec_impl_guide.md).
Per-milestone results, `rl/results/E<n>_<topic>.md`.

---

## 1. Status snapshot

- **Date created.** 2026-06-08
- **Date last updated.** 2026-06-08
- **Current state.** E4 (RL library + training loop) complete on
  `feat/ordrec-e4`, awaiting merge. The `RLAlgorithm` ABC, the
  on-policy `RolloutBuffer`, the GAE advantage estimator and the
  PPO concrete implementation (clipped surrogate, value clip, KL
  early stop, entropy anneal, grad clip) all land. BC warm-start
  with a max-Fisher teacher and the exact `K ** K_B = 1024`
  static-MVE critic warm-start are in place. Top-level
  `train_ppo.py` reads a YAML config and runs the full pipeline
  (adapter, frozen MAGPCM, env, reward, optional warm-start,
  PPO). Toy-env smoke drives mean episode return from `1.062`
  to `2.000` over 20 updates (random baseline `1.0`, optimum
  `2.0`). Synthetic-adapter smoke completes 5 PPO updates with
  the four-component reward decomposition recorded. 146 unit
  tests pass (126 pre-E4 + 13 training + 1 BC, plus 6 new buffer
  cases).
- **Active branch.** `feat/ordrec-e4` at `21a9dea`, 12 commits
  stacked on `feat/ordrec` tip `eaa404a` (E3 merged).
- **Next milestone.** E5, the real polytomous training run on
  Eedi K=4 plus CI, repro and paper hooks. The DoD is the
  headline numbers regenerable from a clean checkout, with PGF
  figures wired into the paper.

The five locked design corrections from the strategic plan are intact.

- (A) Probe-based GPCM entropy reduction is the primary reward,
  Fisher information is a theoretical lens.
- (B) Single framework with per-dataset adapters behind one
  `OrdinalDatasetBase`. The model never branches on dataset identity.
- (C) Custom RL library with a small `RLAlgorithm` ABC, no Tianshou.
- (D) Per-user deterministic splits, byte-identical re-materialisation.
- (E) Action mask covers admin, probe, and within-episode no-repeat.

---

## 2. Milestones

| Milestone | Status | Branch | Tip | Tasks | DoD | Notes |
|---|---|---|---|---|---|---|
| **E1**, data adapters | complete (merged) | `feat/ordrec` | `267ea82` | `data/{base,schema,split,synthetic,placeholder_2pl,ma_irt_bridge,eedi}.py`, 5 test files, Eedi fixture, two configs | met | 42 tests pass. Synthetic smoke training `r_theta=0.88` in 5 epochs. See `rl/results/E1_data_layer.md`. |
| **E2**, per-item lookup + EdNet + ASSISTments + bench | complete (merged) | `feat/ordrec` | `f5c536e` | `envs/{frozen_magpcm,item_cache,bench_forward}.py`, `data/{ednet,assist}.py`, `rl/scripts/prepare_eedi_csv.py`, 5 new test files | met | 82 tests pass (27 envs + 13 new adapter + 42 E1). Bench at `rl/results/E2_bench_forward.{json,md}`. Cache keyed by `(dataset_name, ckpt_sha7)`. See `rl/results/E2_envs_layer.md`. |
| **E3**, env + reward + wiring | complete (merged) | `feat/ordrec` | `eaa404a` | `envs/{base,ordrec_env,action_mask}.py`, the full `reward/` package, `rl/tests/test_env_reward_wiring.py` | met | 126 tests pass (44 new E3 + 82 pre-E3). Four-component reward sums to `r_total`, three-source mask blocks every probe id. See `rl/results/E3_env_reward.md`. |
| **E4**, RL library + training loop + smoke | complete (awaiting merge) | `feat/ordrec-e4` | `21a9dea` | `training/{base,rollout,gae,ppo,utils}.py`, `bc_warmstart/{bc,static_mve}.py`, `scripts/{train_ppo,sanity_toy_env,eval_policy}.py`, `configs/{ppo_eedi_k4,ppo_synth_smoke}.yaml` | met | 146 tests pass (20 new E4 + 126 pre-E4). PPO on the toy env goes from `1.062` to `2.000` over 20 updates. PPO on the synthetic adapter runs 5 updates and saves `best.pt`. See `rl/results/E4_rl_library.md`. |
| **E5**, headline polytomous run on Eedi K=4 | next | tbd | tbd | real Eedi raw csvs landed, BC and MVE warm-start enabled, PPO `total_updates = 1000`, evaluation harness, paper hooks | tbd | Mean return strictly above the uniform-random baseline at the end of training, with per-component decomposition and exposure caps respected. |
| **E6**, ablations + paper figures | not started | tbd | tbd | full PPO runs on EdNet KT3 K=4, ASSISTments K=2, plus ablations (no-probe, no-exposure, no-VOI), paper PGF figures | tbd | The science milestone. |

---

## 3. Change log (reverse chronological)

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
| `test_reward_scale.py` | 2 | Per-component contribution within `[5%, 70%]` of `|r_total|`, breakdown sums to total. |
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

Reproducer.

```bash
cd <repo_root>
PYTHONPATH="rl/src;ma-irt" KMP_DUPLICATE_LIB_OK=TRUE \
  python -m pytest rl/src/ordrec/ rl/tests/ -v
```

Existing ma-irt test suite is untouched through E3.

---

## 5. Open issues

Carried forward from `rl/results/E1_data_layer.md`,
`rl/results/E2_envs_layer.md`, `rl/results/E3_env_reward.md`,
and `rl/results/E4_rl_library.md`.

1. Placeholder 2PL `lr` field in `coercion_artefacts.json` reports the
   guide default (1e-2) rather than the value actually used (5e-2 at
   20 epochs). E2 did not touch this. Cosmetic, fix during the E3
   wiring.
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
14. E4 new, BC teacher mixture, max-Fisher only at v1. The
    ReflectionLayer-greedy (30%) and Thompson (20%) teachers
    described in the impl guide need infrastructure not built
    in E4. v2 enhancement before the headline run.
15. E4 new, DQN and SAC implementations. Sketched as comments in
    the impl guide and inside `training/ppo.py`. Deferred to a
    future ablation.
16. E4 new, headline-config sweep over PPO hyperparameters
    (`lr in {1e-4, 3e-4, 1e-3}`, `gae_lambda in {0.90, 0.95,
    0.99}`, `clip_eps in {0.1, 0.2}`). Belongs to E5 or E6
    once real Eedi raw csvs land.
