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
- **Current state.** E1 (data layer) complete on
  `feat/ordrec-e1-eedi`. The `rl/ordrec/data/` package, schema,
  Eedi K=4 distractor-difficulty adapter, synthetic adapter,
  placeholder 2PL, ma-irt bridge, and the 50-row Eedi fixture are
  landed. 42 unit tests pass. Synthetic smoke training pass at
  `r_theta = 0.880`, `rho_theta = 0.904` after 5 epochs.
- **Active branch.** `feat/ordrec-e1-eedi` at `466e730`, 11 commits
  stacked on `feat/ordrec` tip `0962f5a` (the implementation guide).
- **Next milestone.** E2, per-item `(alpha, beta)` lookup +
  `FrozenMAGPCM` wrapper + `bench_forward.py` timing harness, plus
  the EdNet and ASSISTments adapters.

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
| **E1**, data adapters | complete | `feat/ordrec-e1-eedi` | `466e730` | `data/{base,schema,split,synthetic,placeholder_2pl,ma_irt_bridge,eedi}.py`, 5 test files, Eedi fixture, two configs | met | 42 tests pass. Synthetic smoke training `r_theta=0.88` in 5 epochs. See `rl/results/E1_data_layer.md`. |
| **E2**, per-item lookup + EdNet + ASSISTments + bench | not started | tbd | tbd | `envs/item_cache.py`, `envs/bench_forward.py`, `data/ednet.py`, `data/assist.py`, plus a thin `FrozenMAGPCM` wrapper exposing `forward_no_grad` | tbd | Out-of-band cache keyed by `(raw_csv_md5, fit_seed)`. Per-item sweep produces `(Q+1, D)` alpha and `(Q+1, K-1)` beta tables. |
| **E3**, env + reward + wiring | not started | tbd | tbd | `envs/{base,ordrec_env,action_mask}.py`, the full `reward/` package, `tests/test_env_reward_wiring.py` | tbd | Reward returns `(B,) reward` plus a four-component breakdown. Ng-Harada-Russell invariance enforced via fixed probe sets. |
| **E4**, RL library + training loop + smoke | not started | tbd | tbd | `training/{base,rollout,gae,ppo,utils}.py`, `bc_warmstart/{bc,static_mve}.py`, `scripts/{train_ppo,sanity_toy_env,eval_policy}.py`, `configs/ppo_eedi_k4.yaml` | tbd | PPO on a toy env shows strictly increasing return over 20 updates. PPO on the real env runs 5 updates and saves `best.pt`. |
| **E5**, CI + repro + paper hooks | not started | tbd | tbd | github actions yaml, repro script, eval table generator, paper PGF figures | tbd | Headline numbers regenerable from a clean checkout. |
| **E6**, headline runs + ablations | not started | tbd | tbd | full PPO runs on Eedi, EdNet, ASSISTments, plus ablations (no-probe, no-exposure, no-VOI) | tbd | The science milestone. |

---

## 3. Change log (reverse chronological)

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

Reproducer.

```bash
cd <repo_root>
PYTHONPATH="rl/src;ma-irt" KMP_DUPLICATE_LIB_OK=TRUE \
  python -m pytest rl/src/ordrec/data/tests/ -v
```

Existing ma-irt test suite is untouched in E1.

---

## 5. Open issues

Carried forward from `rl/results/E1_data_layer.md`.

1. Placeholder 2PL `lr` field in `coercion_artefacts.json` reports the
   guide default (1e-2) rather than the value actually used (5e-2 at
   20 epochs). Cosmetic, fix during the E2 cache wiring.
2. Synthetic adapter does not persist `true_irt_parameters.json` into
   the materialised artefact. Eval recovery currently reads it from
   the upstream raw directory by convention. Persist alongside the
   artefact for reproducibility before the headline run.
3. Bridge tests `skip` rather than `fail` when `PYTHONPATH=ma-irt`
   is absent. Acceptable now, the E5 CI config should set it.
4. R `mirt` audit path for the Eedi placeholder 2PL not implemented.
   Deferred to E2 since it depends on the same caching machinery as
   the per-item alpha/beta lookup.
5. Eedi fixture is only 50 rows. Full-corpus validation of distractor
   ordering against the published Eedi baseline is an E2 task.
6. Open engineering questions from impl guide Section 9, items 1
   through 8 (data), all remain open at E1 close. Items 9 through
   15 (reward) and 16 through 23 (RL) belong to E3 and E4.
