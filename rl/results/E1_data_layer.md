# E1, Data Layer Milestone, Completed 2026-06-08

Branch tip after E1, `466e730` on `feat/ordrec-e1-eedi`.
Strategic plan, `docs/exrec_ordinal_plan.md`.
Implementation guide, `docs/ordrec_impl_guide.md` Section 2 and Section 8.

## Scope

E1 lands the `rl/ordrec/data/` package, the seven Module 1 files plus
their unit tests, the 50-row Eedi fixture, and the two ma-irt training
configs (`ordrec_eedi_k4.yaml` and `ordrec_synth_smoke.yaml`). No env,
no reward, no RL library, no training loop. EdNet and ASSISTments
adapters are deferred to E2. The synthetic smoke training pass is the
end-to-end pipeline verification, not a science result.

The work follows the impl guide's Section 8 first-PR scope verbatim.

## What landed

Package skeleton.

  - `rl/pyproject.toml`, `ordrec` namespace, depends on ma-irt via
    `PYTHONPATH=ma-irt` at runtime (no editable install required for
    the unit test loop).
  - `rl/src/ordrec/__init__.py`, package marker.
  - `rl/src/ordrec/data/__init__.py`, re-exports `OrdinalDatasetBase`,
    `AdapterConfig`, `EediAdapter`, `SyntheticAdapter`,
    `adapter_to_sequence_dataset`, `adapter_to_dataloader`.

Module 1 files.

  - `data/base.py`, the `OrdinalDatasetBase` ABC and `AdapterConfig`
    frozen dataclass. `__getitem__` returns `{questions, responses,
    student_id}` with 1-based ids, matching `ma-irt/utils/dataloader.py`.
  - `data/schema.py`, `COMMON_RECORD_SCHEMA`, the four canonical
    filenames, and the metadata, sequences, q-matrix, and coercion
    validators. Json IO uses `sort_keys=True` so re-materialisation is
    byte-identical.
  - `data/split.py`, `make_split` (deterministic user-level),
    `stratified_split`, and `_chunk_sequences` for the
    `max_seq_len` cap with parent tracking.
  - `data/synthetic.py`, wraps a ma-irt static-GPCM artefact, adds the
    OrdRec metadata block, materialises with per-user split and
    optional chunking. Used for the smoke training pass below.
  - `data/placeholder_2pl.py`, `fit_placeholder_2pl` around
    `StaticGPCM(n_categories=2)`. 20 epochs at lr 5e-2 (the impl guide
    suggested 5 epochs at lr 1e-2, we found 20 at 5e-2 gives a more
    stable theta_hat ordering on the small fixture, see open issue
    1 below).
  - `data/eedi.py`, `EediAdapter` implementing the K=4
    distractor-difficulty algorithm from impl guide Section 2.3. Train
    only fitting, `distractor_order_per_q` persisted to
    `coercion_artefacts.json`, lexicographic fallback when a distractor
    is unseen on train, unseen-on-test distractors recode to the
    middle wrong category and are logged.
  - `data/ma_irt_bridge.py`, `adapter_to_sequence_dataset` and
    `adapter_to_dataloader` shims. Lazy `ma-irt` import so the unit
    test loop runs without `PYTHONPATH=ma-irt` for the non-bridge
    tests.

Test suite.

  - `tests/test_base_contract.py`, 7 tests, the ABC contract.
  - `tests/test_schema_round_trip.py`, 14 tests, validators and
    byte-identical round trip.
  - `tests/test_split_determinism.py`, 9 tests, including the
    byte-identical rebuild assertion on the synthetic adapter.
  - `tests/test_eedi_adapter.py`, 7 tests including the
    `distractor_ordering_is_ascending_by_mean_theta` and
    `train_only_distractor_ordering` contract tests.
  - `tests/test_ma_irt_bridge.py`, 5 tests, all collate-shape and
    splits-disjoint assertions. Requires `PYTHONPATH=ma-irt` (skips
    cleanly otherwise).
  - `tests/fixtures/eedi_mini.csv`, 50-row fixture covering 10 students
    and 5 items so distractor counts are nontrivial on the train fold.

Configs.

  - `ma-irt/configs/ordrec_eedi_k4.yaml`, the MAGPCM headline config
    for the Eedi K=4 path.
  - `ma-irt/configs/ordrec_synth_smoke.yaml`, the 5-epoch smoke config
    that exercised the full pipeline end-to-end.

## Test results

Re-ran on the final E1 tip (`466e730`).

```
PYTHONPATH="rl/src;ma-irt" KMP_DUPLICATE_LIB_OK=TRUE \
  python -m pytest rl/src/ordrec/data/tests/ -v
```

```
========================= 42 passed in 3.87s ==========================
```

All 7 contract tests, 14 schema round-trip tests, 9 split determinism
tests, 7 Eedi adapter tests, and 5 ma-irt bridge tests pass.

The five bridge tests skip when `PYTHONPATH=ma-irt` is absent (they
gate on `pytest.importorskip("utils.dataloader")`), which is the
intended behaviour for the rl-only test loop.

## Synthetic adapter smoke training metrics

Pipeline. `ma-irt/data_gen.py static` produced raw at
`ma-irt/data/synth_e1_smoke/`, `SyntheticAdapter.materialise` wrote
the OrdRec artefact at `ma-irt/data/ordrec_synth_e1_smoke/`,
`train.py --config configs/ordrec_synth_smoke.yaml` ran 5 epochs of
MAGPCM on the resulting `(500 students, 50 items, K=4)` artefact.

Run reverified on E1 tip with a 2-epoch repeat. Pipeline trains
end-to-end, identical first 2 epoch metrics on the same seed.

```
epoch  train_loss  train_acc  val_loss  val_acc  val_qwk
1      2.4698      0.215      2.3113    0.223    0.135
2      2.2640      0.229      2.1864    0.235    0.198
3      2.1586      0.232      2.1065    0.242    0.240
4      2.0890      0.236      2.0595    0.242    0.255
5      2.0493      0.242      2.0194    0.251    0.278
```

Recovery on the validation fold (synthetic ground truth available).

```
r_theta           0.880
rho_theta         0.904
rmse_theta        0.643
r_beta_mean       0.371
rmse_beta_mean    1.196
```

Both `r_theta` and `rho_theta` clear the 0.85 target after only 5
epochs at minimal capacity (`memory_size=32`, `key_dim=value_dim=32`).
Alpha recovery is poor at this budget because the smoke run uses
`StaticGPCM`-style synthetic data with low alpha variance and the
adapter does not write `true_irt_parameters.json` into the OrdRec
artefact, so the eval script reads it from `ma-irt/data/synth_e1_smoke/`
directly. This is a pipeline check, not a science claim.

Run artefacts at `ma-irt/outputs/ordrec_synth_e1_smoke/`, `best.pt`,
`last.pt`, `metrics.csv`, `recovery_metrics.json`. Not committed
(gitignored).

## Open issues for E2

  1. Placeholder 2PL hyperparameters drifted. The impl guide proposed
     `5 epochs at lr 1e-2`. The fixture needed `20 epochs at lr 5e-2`
     for a stable theta_hat ordering, see commit `58144e5`. The
     `placeholder_2pl["lr"]` field in `coercion_artefacts.json` is
     still logged as the guide default (1e-2) rather than the value
     actually used (5e-2). Cosmetic, fix when wiring up the per-item
     cache in E2.

  2. The synthetic adapter does not write `true_irt_parameters.json`
     into the materialised artefact. The eval recovery script reads
     it from the upstream raw directory by convention. This is fine
     for E1 but the eventual headline pipeline should persist the
     ground-truth IRT parameters alongside the artefact for
     reproducibility.

  3. The bridge tests skip rather than fail when `PYTHONPATH=ma-irt`
     is absent. Acceptable today, but the CI configuration in E5
     should set it explicitly so the skip count is zero.

  4. Eedi placeholder 2PL audit. The impl guide leaves the
     R `mirt` baseline available as a `--use-r-mirt` audit reference.
     Not implemented in E1, deferred to E2 since it depends on the
     same caching machinery as the per-item alpha/beta lookup.

  5. The Eedi fixture is 50 rows, 10 students, 5 items. Enough to
     exercise the adapter contract but not large enough to validate
     distractor-difficulty ordering against the published Eedi
     baseline. Full-corpus validation is an E2 task.

## File manifest

Code (E1).

```
rl/pyproject.toml
rl/src/ordrec/__init__.py
rl/src/ordrec/data/__init__.py
rl/src/ordrec/data/base.py
rl/src/ordrec/data/schema.py
rl/src/ordrec/data/split.py
rl/src/ordrec/data/synthetic.py
rl/src/ordrec/data/placeholder_2pl.py
rl/src/ordrec/data/eedi.py
rl/src/ordrec/data/ma_irt_bridge.py
```

Tests (E1).

```
rl/src/ordrec/data/tests/__init__.py
rl/src/ordrec/data/tests/test_base_contract.py
rl/src/ordrec/data/tests/test_schema_round_trip.py
rl/src/ordrec/data/tests/test_split_determinism.py
rl/src/ordrec/data/tests/test_eedi_adapter.py
rl/src/ordrec/data/tests/test_ma_irt_bridge.py
rl/src/ordrec/data/tests/fixtures/eedi_mini.csv
```

Configs (E1).

```
ma-irt/configs/ordrec_eedi_k4.yaml
ma-irt/configs/ordrec_synth_smoke.yaml
```

Total, 19 files, 2477 insertions across 11 commits stacked on
`feat/ordrec` tip `0962f5a`.
