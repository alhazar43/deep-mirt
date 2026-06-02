# pyKT-Style 5-Fold CV Refactor Plan

## Goal

Make the **binary KT benchmark** (`tab:combined_perf`) and the **proxy-ordinal
ASSIST prediction tables** (`tab:assistments_pred`) directly comparable to
pyKT-published numbers, by switching from "single 80/20 split + 5 model-init
seeds" to "fixed 20% test + 5-fold CV on the 80% pool, mean ± sd over folds".
Synthetic-5 follows DKVMN's per-version protocol (5 versions × 5 folds, mean
across both).

The synthetic recovery experiments (Static, Discrete, Continuous, Imbalanced,
scalability) keep their current single-split + 5-seed protocol. They're
about parameter recovery, not benchmark predictive comparability.

## Decisions locked

| # | Decision |
|---|---|
| 1 | `split_seed = 42` (not pyKT's 1024). Independent eval, deterministic across our experiments. |
| 2 | **Plain AUC**, not window AUC. Cheaper to implement; ~1-3 point lower numerical comparability cost we accept. |
| 3 | Early-stop metric: `auc` for binary K=2, `qwk` for proxy-ordinal K=4. |
| 4 | Patience = 10, improvement margin = 1e-3 (pyKT canonical). |
| 5 | Max epochs = 200 (pyKT canonical). Early stop ends most runs much earlier. |
| 6 | **5 folds × 1 seed each**, no seed multiplication. |
| 7 | ASSIST2017 `max_seq_len = 500` (matches pyKT for that one dataset). All others stay at 200. |
| 8 | Synthetic-5 = 5 versions (v0..v4), 5 folds each = 25 runs per model. Reported as mean ± sd over all 25. |
| 9 | Aggregator: 5-fold mean ± sd per (model, dataset). For Synthetic-5: mean across 5 versions of (per-version 5-fold mean), with sd over the 5 versions. |
| 10 | Recovery experiments unchanged. |

## Scope of code changes

| File | Change | Diff size | Type |
|---|---|---|---|
| `ma-irt/config/types.py` | Add `CVConfig`; add `early_stop_metric`, `patience`, `early_stop_margin` to TrainingConfig | ~15 LOC | additive |
| `ma-irt/utils/metrics.py` | Add `auc` key to `compute_metrics` output when K=2 | ~10 LOC | additive |
| `ma-irt/dataloading/loaders.py` | Add `DataModule.build_cv()` returning (train, valid, test); existing `build()` unchanged | ~40 LOC | additive |
| `ma-irt/training/trainer.py` | No change. `evaluate_epoch` already returns metrics dict. | 0 | — |
| `ma-irt/scripts/train.py` | Branch on `cv.enabled`: legacy loop OR new (train, valid, test) loop with patience-based early stop, pick best by `early_stop_metric`, eval on test once at end | ~50 LOC | invasive but gated |
| `ma-irt/scripts/evaluate.py` | Already computes AUC. Verify it picks up the `valid_auc` correctly. | 0 expected | verify only |
| `ma-irt/scripts/_gen_pykt_configs.py` | New: emit per-(model, dataset, fold) configs | ~100 LOC | new file |
| `ma-irt/scripts/_run_pykt_sweep.sh` | New: sweep launcher | ~30 LOC | new file |
| `ma-irt/scripts/_aggregate_pykt_results.py` | New: aggregate across folds (and across versions for synthetic-5) | ~60 LOC | new file |
| `ma-irt/scripts/_build_pykt_synthetic5.py` | New: convert each `dkvmn-ori/data/synthetic/naive_c5_q50_s4000_v{V}_train.csv + _test.csv` → `data/synthetic5_v{V}/sequences.json`; merge train+test into one file (loader will re-split) | ~50 LOC | new file |

**Total**: ~115 LOC modified across existing files, ~240 LOC in new files.
All changes to existing files are backward-compatible (gated by `cv.enabled`
default false).

## Implementation steps (in order)

1. **Config** — add `CVConfig` and early-stop fields to `config/types.py`.
   Defaults preserve current behavior. ✅ Already done in this session.

2. **Metrics** — add `auc` to `compute_metrics()` return dict when K=2.
   Falls back to `nan` if K≠2 or only one class present.

3. **Loader** — add `DataModule.build_cv()`. Steps:
   - Load full sequences and metadata as today.
   - Use `np.random.RandomState(cv.split_seed)` to permute student indices.
   - First `test_frac` of permuted indices → test; rest → 80% pool.
   - Split the pool into `n_folds` disjoint folds (`np.array_split`).
   - Valid = fold[fold_id]; train = concat(other folds).
   - Build three `SequenceDataset`s + three `DataLoader`s, return triple.
   - Existing `build()` unchanged.

4. **Trainer integration** — modify `scripts/train.py`:
   - Branch on `cfg.data.cv.enabled`.
   - If true: call `build_cv()` for train/valid/test loaders. Loop max_epochs;
     each epoch: train + eval on valid (NOT test). Track best metric on valid;
     save best.pt on improvement; increment patience counter on stagnation;
     break when patience exhausted.
   - After loop: load best.pt, run final eval on test, write
     `recovery_metrics.json` with test AUC/ACC.
   - If false: existing loop unchanged (literally no diff under default config).

5. **Synthetic-5 data prep** — `_build_pykt_synthetic5.py`:
   - For v ∈ {0..4}, read `dkvmn-ori/data/synthetic/naive_c5_q50_s4000_v{v}_train.csv`
     and `..._test.csv`, parse 3-line records.
   - Concatenate (train + test = 4000 students per version).
   - Write `data/synthetic5_v{v}/sequences.json` + metadata.
   - The loader's CV split (seed=42) will produce a fresh train/valid/test
     split per version. We do NOT preserve DKVMN's original 50/50 split since
     pyKT-style CV imposes its own.

6. **Config generator** — `_gen_pykt_configs.py`:
   - For each (model, dataset, fold) ∈ M × D × {0..4}:
     - Emit YAML with `cv.enabled=true, cv.fold_id=f, cv.split_seed=42`.
     - Set `early_stop_metric` per dataset (auc binary, qwk ordinal).
     - Set `epochs=200, patience=10, early_stop_margin=1e-3`.
     - Set `max_seq_len=500` for `assist2017_bin`, else 200.
     - For synthetic-5: emit per-version configs (`synthetic5_v0`..`synthetic5_v4`).

7. **Sweep launcher** — `_run_pykt_sweep.sh` runs all configs sequentially,
   logging completion + per-fold AUC to `outputs/_pykt_sweep.log`.

8. **Aggregator** — `_aggregate_pykt_results.py`:
   - For each (model, dataset): mean ± sd over the 5 fold runs.
   - For synthetic-5: mean across 5 versions of per-version 5-fold means,
     with sd computed over per-version means (so sd reflects version
     variance, not fold variance).
   - Output formatted LaTeX rows for direct paste into `tab:combined_perf`.

9. **Table update** — replace tab:combined_perf and tab:assistments_pred
   numbers, update captions to mention 5-fold CV protocol.

10. **Backward-compat smoke test** — re-run one existing
    `bench_dkt_static_q200_k2_s42` config (cv disabled) and verify val_qwk
    output matches the pre-refactor metrics.csv to within 1e-6.

## Sweep specification

**Binary KT** (5 models: DKT, DKVMN, Deep-IRT, DKVMN+GPCM, MA-GPCM):

| Block | Dataset | Note | Runs | Est |
|---|---|---|---|---|
| Synthetic-5 (5 versions) | `synthetic5_v0..v4` | DKVMN-orig data | 5×5×5 = 125 | ~2h |
| Static | `static_q200_k2` | unchanged | 5×5 = 25 | ~1h |
| ASSIST2009 | `assist2009_bin` | **keep current Piech 4217/123 file** (per user) | 5×5 = 25 | ~3h |
| ASSIST2017 | `assist2017_bin` | `max_seq_len=500` (pyKT default) | 5×5 = 25 | ~5h |

**Proxy-ordinal K=4** (4 models: Dynamic GPCM, DKVMN+Softmax, DKVMN+GPCM, MA-GPCM):

| Block | Dataset | Note | Runs | Est |
|---|---|---|---|---|
| ASSIST2009 K=4 | `assist2009_ord_k4` | 2596 students, 3910 questions | 4×5 = 20 | ~3h |
| ASSIST2017 K=4 | `assist2017_ord_k4` | 1699 students, 772 questions | 4×5 = 20 | ~3h |

**Totals**: 240 runs, ~17 GPU-hours.

**Critical**: DKT, DKVMN, Deep-IRT only support K=2. Static GPCM is excluded
from proxy-ordinal (per existing paper). Config generator must enforce this:
- Binary block: 5 models, K=2
- Proxy-ordinal block: 4 models, K=4 (NO DKT/DKVMN/Deep-IRT, NO Static GPCM)

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Loader split correctness (leakage between fold pools) | Unit-test that train/valid/test indices are disjoint and union = full population |
| Per-fold model state leak | Re-instantiate model + optimizer per run (already standard since each run = separate process) |
| Backward-compat regression | Smoke test: existing static config produces identical val_qwk to pre-refactor |
| Early-stop oscillation | margin=1e-3 prevents tiny AUC noise from resetting patience counter |
| AUC undefined when only one class in valid set | Return nan; trainer treats nan as no-improvement |
| Aggregator off-by-one on synthetic-5 (5 vs 25 cells) | Explicit assertion: 25 result files per model, group by version_id then aggregate |

## Rollback plan

All changes to existing files are gated by `cv.enabled` (default false). If
the refactor breaks, set `cv.enabled=false` everywhere (or delete the new
configs) and the codebase reverts to current behavior. New scripts (`_gen_*`,
`_run_*`, `_aggregate_*`) are additive and can be deleted without affecting
the trained models or recovery experiments.

## What I won't do

- No window AUC. User explicit decision; cost not worth marginal protocol
  fidelity.
- No checkpoint-format changes. `best.pt` keeps current schema.
- No modifications to `evaluate.py single`. Recovery eval path unchanged.
- No dataset re-preprocessing of ASSIST2009 binary unless we explicitly
  decide to switch to DKVMN's 4151/110 file (TBD; default is to keep our
  current 4217/123).

## Open questions — RESOLVED

1. **ASSIST2009 binary**: ✅ keep current Piech 4217/123 file.
2. **ASSIST2017 maxlen=500**: ✅ confirmed.
3. **Proxy-ordinal ASSIST2017**: ✅ exists at `data/assist2017_ord_k4/`.
4. **Proxy-ordinal model count**: ✅ 4 models (Dynamic GPCM, DKVMN+Softmax,
   DKVMN+GPCM, MA-GPCM). DKT/DKVMN/Deep-IRT/Static GPCM excluded.
