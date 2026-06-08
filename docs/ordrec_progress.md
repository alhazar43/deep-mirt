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
- **Current state.** E3 (env + reward layer) complete on
  `feat/ordrec-e3`, awaiting merge. The Gym-style `OrdRecEnv`
  wraps a `FrozenMAGPCM` and a data adapter, the four-component
  reward (`r_info + r_cost + r_expo + r_voi`) composes Lindley
  probe-entropy shaping, an ask cost, the Sympson-Hetter (1985)
  exposure penalty and a terminal NLL anchor on the held-out
  probe `H_probe`, and the three-source action mask (admin,
  probe, no-repeat) is enforced through the public env surface.
  126 unit tests pass (the 82 pre-E3 tests plus 30 reward + 11
  env + 3 cross-package wiring).
- **Active branch.** `feat/ordrec-e3` at `7414fb3`, 12 commits
  stacked on `feat/ordrec` tip `f5c536e` (E2 merged).
- **Next milestone.** E4, the RL library and the training loop.
  `RLAlgorithm` ABC, PPO, GAE, rollout collector, BC warmstart,
  sanity toy env, headline-config sweep on Eedi K=4. The DoD is
  PPO on a toy env shows strictly increasing return over 20
  updates, and PPO on the real env runs 5 updates and saves
  `best.pt`.

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
| **E3**, env + reward + wiring | complete (awaiting merge) | `feat/ordrec-e3` | `7414fb3` | `envs/{base,ordrec_env,action_mask}.py`, the full `reward/` package, `rl/tests/test_env_reward_wiring.py` | met | 126 tests pass (44 new E3 + 82 pre-E3). Four-component reward sums to `r_total`, three-source mask blocks every probe id. See `rl/results/E3_env_reward.md`. |
| **E4**, RL library + training loop + smoke | not started | tbd | tbd | `training/{base,rollout,gae,ppo,utils}.py`, `bc_warmstart/{bc,static_mve}.py`, `scripts/{train_ppo,sanity_toy_env,eval_policy}.py`, `configs/ppo_eedi_k4.yaml` | tbd | PPO on a toy env shows strictly increasing return over 20 updates. PPO on the real env runs 5 updates and saves `best.pt`. |
| **E5**, CI + repro + paper hooks | not started | tbd | tbd | github actions yaml, repro script, eval table generator, paper PGF figures | tbd | Headline numbers regenerable from a clean checkout. |
| **E6**, headline runs + ablations | not started | tbd | tbd | full PPO runs on Eedi, EdNet, ASSISTments, plus ablations (no-probe, no-exposure, no-VOI) | tbd | The science milestone. |

---

## 3. Change log (reverse chronological)

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
`rl/results/E2_envs_layer.md`, and
`rl/results/E3_env_reward.md`.

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
