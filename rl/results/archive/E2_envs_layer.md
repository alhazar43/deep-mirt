# E2, Envs Layer Milestone, Completed 2026-06-08

Branch tip after E2, `80036ea` on `feat/ordrec-e2`.
Strategic plan, `docs/exrec_ordinal_plan.md`.
Implementation guide, `docs/ordrec_impl_guide.md` Section 5 and Section 7.

## Scope

E2 lands the world-model freeze wrapper, the per-item `(alpha, beta)`
cache, the forward-pass latency harness, the EdNet K=4 and ASSISTments
K=2 adapters, and the real-Eedi pre-merge script. No Gym env, no
reward, no RL library. The new `rl/src/ordrec/envs/` subpackage holds
the freeze wrapper, the item cache and the bench harness, the
adapters extend the existing `rl/src/ordrec/data/` package, and the
pre-merge script lands at `rl/scripts/prepare_eedi_csv.py`. The E3
env and reward are scoped to a separate branch.

The work follows the impl guide's Section 7 E2 deliverable list
verbatim.

## What landed

Frozen world model.

  - `envs/frozen_magpcm.py`, the two-line freeze contract. The
    helper `freeze_magpcm` applies `model.eval()` and
    `model.requires_grad_(False)` in place. The `FrozenMAGPCM`
    wrapper owns the underlying model as a submodule, enforces the
    contract at construction, overrides `train(mode)` so an upstream
    `ppo.train()` cannot silently re-enable dropout in the world
    model, and exposes the stable callable `forward_no_grad(questions,
    responses) -> dict` that returns the same five-key dict the native
    MA-IRT forward returns (`logits`, `probs`, `theta`, `alpha`,
    `beta`). Replaces the v1 M1-era `freeze_irt` helper.

Per-item cache.

  - `envs/item_cache.py`, the `(alpha, beta)` cache builder, loader
    and on-disk format. The cache sweeps each item id `q` in
    `[1..Q]` through one frozen forward pass to read off the per-item
    `alpha_q` and `beta_q` tensors, averaging alpha across a small
    number of synthetic student contexts per the impl guide's
    paper-style averaging recipe so the alpha estimate is stable.
    Beta depends only on the item embedding and is invariant in the
    student context. The cache is keyed by
    `(dataset_name, ckpt_sha7)` where `ckpt_sha7` is the first
    seven hex characters of the SHA-256 digest of the MAGPCM
    checkpoint, persisted to
    `rl/artifacts/item_cache/<dataset>/<ckpt_sha7>/item_cache.npz`,
    so two distinct trained models produce two distinct caches.
    `alpha_table` has shape `(Q + 1, D)`, `beta_table` has shape
    `(Q + 1, K - 1)`, both `float32`. Row 0 is the padding slot.

Bench harness.

  - `envs/bench_forward.py`, the encoder-agnostic forward-pass
    latency harness. Measures CPU and CUDA (when available) latency
    at `(B=1, T=50)` and `(B=128, T=50)` for each of DKVMN, LSTM,
    and Transformer encoders, ten repeats with three warmup iters
    each, median plus mean plus min plus max in milliseconds.
    Writes the numbers to `rl/results/E2_bench_forward.json` plus
    a small markdown summary at `rl/results/E2_bench_forward.md`.
    Also exposes `no_grad_invariance_check`, the freeze regression
    confirming identical `(questions, responses)` yields identical
    logits across calls under `no_grad` plus `eval`.

Adapters.

  - `data/ednet.py`, `EdNetAdapter` for EdNet KT3 or KT4. Coerces
    K=4 from `(correctness, response_time)` using a per-question
    response-time median computed on the train fold only. Quadrant
    table is `ord = 0` for incorrect slow, `ord = 1` for incorrect
    fast, `ord = 2` for correct slow, `ord = 3` for correct fast.
    KT3 lacks a hint field, the adapter detects the column at load
    time and records `coercion_artefacts.json["ednet_level"]`
    accordingly. KT4's hint column is acknowledged but does not
    change the K=4 recode (a K=5 hint-aware variant is left as a
    follow-up). Per-question median is persisted to
    `coercion_artefacts.json["rt_median_per_q"]`, cold items on test
    fall back to the train-fold global median with the count logged.

  - `data/assist.py`, `AssistAdapter` for ASSISTments 2009. Identity
    passthrough at K=2, `y_ord = correct` clipped to `{0, 1}`,
    `metadata["n_categories"] = 2`,
    `ordinal_coercion_method = "binary"`. Provides the binary
    ablation control against which the K=4 datasets must outperform.

Pre-merge script.

  - `rl/scripts/prepare_eedi_csv.py`, the Eedi NeurIPS 2020 Task 3
    plus Task 4 csv merger. Joins per-attempt responses with
    per-question metadata into the single-csv format the
    `EediAdapter` expects. Documents the raw-to-merged column
    mapping in the module docstring. The deliverable is the script,
    not its execution, the real csvs are not in this repo.

Tests.

  - `envs/tests/test_frozen_magpcm.py`, 10 tests covering the freeze
    contract under repeated forward calls, the five-key output dict,
    `train(mode)` keeping `eval` mode, no_grad invariance across
    repeated calls, theta invariance to future positions under
    eval+no_grad, autograd does not flow into the frozen model,
    device property propagation, non-Module rejection, and the
    `n_questions` plus `n_categories` propagation. Replaces the v1
    M1-era parity tests.
  - `envs/tests/test_item_cache.py`, 11 tests covering build shapes
    and dtypes, padding-slot invariants, alpha and beta match a
    direct forward pass, save/load round trip, tensor accessor
    parity, cache keyed by `(dataset, checkpoint)`, sha7 handling
    of a missing file, sha7 for a real file, argument validation,
    path requires `dataset_name`, and averaging across contexts
    changes alpha but not beta.
  - `envs/tests/test_bench_forward.py`, 6 tests, a smoke run
    end-to-end, rejection of an empty encoder list, the no_grad
    invariance passes on a frozen model, the invariance check
    catches a broken model, the artefact write round trips,
    and the invariance check requires at least two calls.
  - `data/tests/test_ednet_adapter.py`, 9 tests covering artefact
    materialisation, the metadata block, the quadrant mapping
    table, median computed on train only, KT3 absent-hint
    handling, response range, KT4 hint detection, missing
    response time labeled as slow, and persisted coercion reuse.
  - `data/tests/test_assist_adapter.py`, 4 tests covering
    materialisation, identity passthrough K=2 collapse,
    `n_categories=2` in metadata, and no-test-in-train splits.

Re-exports.

  - `rl/src/ordrec/__init__.py` now imports both `data` and `envs`
    subpackages. The package version bumped to `0.0.2`.
  - `rl/src/ordrec/data/__init__.py` re-exports `EdNetAdapter` and
    `AssistAdapter` and adds them to the `build_adapter` registry
    under the `"ednet"` and `"assist"` keys.
  - `rl/src/ordrec/envs/__init__.py` re-exports `FrozenMAGPCM`,
    `freeze_magpcm`, `ItemCache`, `build_item_cache`,
    `save_item_cache`, `load_item_cache`, `checkpoint_sha7`,
    `item_cache_path`, `ITEM_CACHE_DTYPE`, `BenchConfig`,
    `BenchResult`, `bench_forward_pass`,
    `no_grad_invariance_check`, and `write_bench_artifacts`.

## Test results

Re-ran on the final E2 tip (`80036ea`).

```
PYTHONPATH="rl/src;ma-irt" KMP_DUPLICATE_LIB_OK=TRUE \
  pytest rl/src/ordrec/ -v
```

```
============================= 82 passed in 4.00s ==============================
```

All 42 E1 data layer tests pass. The 13 new data layer tests pass
(`test_ednet_adapter.py` 9 + `test_assist_adapter.py` 4). The 27 new
envs layer tests pass
(`test_frozen_magpcm.py` 10 + `test_item_cache.py` 11 +
`test_bench_forward.py` 6).

## Bench results headline

Numbers measured on Windows 11, RTX 4060 Laptop GPU, torch 2.7.1+cu126,
ten repeats with three warmup iters, MAGPCM at `Q=200`, `K=4`.
Single-user latency is `B=1`, batched is `B=128`, sequence length `T=50`
in both. Median milliseconds.

| Encoder | Device | B=1 median (ms) | B=128 median (ms) |
| --- | --- | ---: | ---: |
| dkvmn | cpu | 5.07 | 38.00 |
| lstm | cpu | 4.03 | 9.03 |
| transformer | cpu | 3.03 | 13.52 |
| dkvmn | cuda | 19.74 | 20.95 |
| lstm | cuda | 1.10 | 1.50 |
| transformer | cuda | 2.63 | 3.03 |

Read at face value. On CPU at `B=1`, all three encoders sit in the
3 to 5 ms range, well inside any conceivable RL rollout budget.
DKVMN's `B=128` CPU latency (38 ms) is the highest of the three, and
its CUDA latency does not improve over CPU at `B=1` because the
DKVMN memory ops are not yet kernel-fused for small batches. LSTM
and Transformer on CUDA are an order of magnitude faster than CPU
at either batch size. DKVMN remains the headline encoder per the
strategic plan, the bench numbers are recorded so a future E4
profiling pass can swap encoders without re-architecting.

The no_grad invariance regression passes for all three encoders.

Full numbers at `rl/results/E2_bench_forward.json` and
`rl/results/E2_bench_forward.md`.

## EdNet K=4 ordinal coercion validation

Smoke validation on the 50-row `ednet_mini.csv` fixture. The fixture
covers 10 students and 5 items with controlled per-question response
times around a known median, and asserts in
`test_ednet_adapter.py::test_quadrant_mapping_table` that
`(correctness, fast_relative_to_median)` maps to the expected ordinal
code on every row.

Coercion artefacts. The adapter persists `rt_median_per_q` (per
question, train fold only), `ednet_level` (`"kt3"` or `"kt4"` based
on detected `hint_used` column), `cold_item_count` (test items not
seen on train), `cold_item_fallback` (the train-fold global median
used for cold items), and `ordinal_coercion_method = "rt_quadrant"`.

KT3 vs KT4. The KT3 branch is covered by
`test_kt3_no_hint_field`. The KT4 branch is covered by
`test_kt4_detected_when_hint_field_present`. Both pass at
checkpoint-equivalent ordinal codes, because the K=4 recode does not
read the hint column.

Full-corpus EdNet materialisation is deferred to E3 when a downloaded
copy lands locally.

## ASSISTments K=2 collapse validation

Smoke validation on the 50-row `assist_mini.csv` fixture. The fixture
covers 10 students and 5 items with binary correct outcomes only.
`test_assist_adapter.py` asserts that `metadata["n_categories"] = 2`,
that `metadata["ordinal_coercion_method"] = "binary"`, that the test
fold and train fold are disjoint at student level, and that all
materialised responses lie in `{0, 1}`. The identity passthrough is
deliberately small in surface, the value lives in the binary ablation
control it provides for the headline run.

## Open issues for E3

Carried forward where still open from E1, plus new items from E2.

  1. EdNet KT4 hint-aware K=5 variant. The current adapter coerces
     KT4 down to K=4 by ignoring the `hint_used` column. The impl
     guide's Section 2.4 leaves an opening for a K=5 variant that
     places `(hint, incorrect)` between the `incorrect` codes and
     `(correct, no-hint)` above `(correct, hint)`. Deferred to a
     future ablation milestone.

  2. Per-item alpha averaging. `build_item_cache` averages alpha
     across `n_contexts` synthetic student contexts (default 8).
     The number is conservative. A larger ablation sweep over
     `n_contexts` against the recovery-target alpha is a candidate
     for E5 when the headline pipeline lands.

  3. DKVMN CUDA latency at small batch. The bench shows DKVMN at
     `B=1` on CUDA is slower than CPU, because the memory ops do
     not fuse for small batches. Not blocking, the rollout
     batches are typically `B>=32`, but flagged for a possible E4
     profiling pass.

  4. Real-Eedi pre-merge execution. The merger script is in place,
     execution waits for the real Eedi NeurIPS 2020 csvs to land
     locally. Not blocking E3, the synthetic + Eedi-mini fixture
     are sufficient for the env + reward smoke loop.

  5. Carry-over from E1, item 1, the placeholder 2PL `lr` field in
     `coercion_artefacts.json` still reads the guide default
     (1e-2) rather than the value actually used (5e-2). E2 did not
     touch this. Cosmetic, fix during the E3 wiring.

  6. Carry-over from E1, item 2, the synthetic adapter still does
     not persist `true_irt_parameters.json` into the materialised
     artefact. Defer to the headline pipeline.

  7. Open engineering questions from impl guide Section 9, items 1
     through 8 (data) and items 9 through 15 (reward) remain open
     at E2 close. Items 16 through 23 (RL) belong to E4.

## File manifest

Code (E2).

```
rl/src/ordrec/envs/__init__.py
rl/src/ordrec/envs/frozen_magpcm.py
rl/src/ordrec/envs/item_cache.py
rl/src/ordrec/envs/bench_forward.py
rl/src/ordrec/data/ednet.py
rl/src/ordrec/data/assist.py
rl/scripts/__init__.py
rl/scripts/prepare_eedi_csv.py
rl/scripts/run_bench_forward.py
```

Tests (E2).

```
rl/src/ordrec/envs/tests/__init__.py
rl/src/ordrec/envs/tests/test_frozen_magpcm.py
rl/src/ordrec/envs/tests/test_item_cache.py
rl/src/ordrec/envs/tests/test_bench_forward.py
rl/src/ordrec/data/tests/test_ednet_adapter.py
rl/src/ordrec/data/tests/test_assist_adapter.py
rl/src/ordrec/data/tests/fixtures/ednet_mini.csv
rl/src/ordrec/data/tests/fixtures/assist_mini.csv
```

Re-exports updated (E2).

```
rl/src/ordrec/__init__.py
rl/src/ordrec/data/__init__.py
```

Results (E2).

```
rl/results/E2_bench_forward.json
rl/results/E2_bench_forward.md
rl/results/E2_envs_layer.md
```

Total, 21 files, 3070 insertions across 8 commits stacked on
`feat/ordrec` tip `267ea82`.
