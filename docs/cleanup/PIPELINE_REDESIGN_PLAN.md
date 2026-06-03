# Pipeline Redesign Plan

Inventory and plan only. No files moved or deleted in this phase. Authority documents are `CLEANUP_VERIFICATION_2026.md` (the hard-stop list) and `benchmarks.md` (the three canonical paper tables). The redesign target is a `ma-irt/scripts/` directory under 30 files plus three new sub-packages under `ma-irt/`.

Inventory snapshot, 2026-06-03. `ma-irt/scripts/` contains 94 entries, 52 `.py`, 38 `.sh`, plus `.tex` and `.R` files and `__pycache__/`.

## (a) Canonical orchestrator trace

Authority for "canonical" is `CLEANUP_VERIFICATION_2026.md` Section 1 and `benchmarks.md` "How to reproduce". Four orchestrators produce the three headline tables.

| Orchestrator | Canonical? | Tables | Python invoked | Config dirs iterated |
|---|---|---|---|---|
| `_run_pykt_sweep.sh` | YES | Table 2 (binary K=2 plus ASSIST K=4 proxy-ordinal) | `train.py` | `configs/bulk/bench_*_pykt_fold{0..4}.yaml` |
| `_run_k4_cv_recovery.sh` | YES | Tables 1 and 3 at K=4 across static, discrete, continuous DGPs | `train.py`, `evaluate.py single` | `configs/bulk/bench_*_q200_k4_pykt_fold{0..4}.yaml` |
| `_run_k356_cv_recovery.sh` | YES | Tables 1 and 3 at K=3, 5, 6 across static, discrete, continuous DGPs | `train.py`, `evaluate.py single` | `configs/bulk/bench_*_q200_k{3,5,6}_pykt_fold{0..4}.yaml` |
| `run_bulk_retrain.sh` | YES (cited in benchmarks.md) | Tables 1 and 3 in the older five-seed protocol; ASSIST 2017 ordinal pass | `train.py`, `evaluate.py single`, `_extract_row.py` | `configs/bulk/<dgp>_<model>_q200_k<K>_s{0,1,7,42,123}.yaml`, `configs/bulk/assistments_*.yaml` |

Aggregators on the canonical path,

- `_aggregate_pykt_results.py` aggregates Table 2 from the pyKT fold protocol.
- `aggregate_recovery_v5.py` aggregates Tables 1 and 3 from both the fold and seed layouts.
- `mirt_baseline_all_k.R` and `mirt_predict.R` produce the GPCM (EM) ceiling row in Table 3.

Every other `run_*.sh`, `_run_*.sh`, and `eval_*.sh` is unreachable from the four canonical orchestrators above. Detailed list,

- Bench-phase chain runners superseded by `_run_pykt_sweep.sh`. `run_bench_sweep.sh`, `run_bench_phase1.sh`, `run_bench_phase2_queue.sh`, `run_bench_phases_chain.sh`, `run_bench_postphase2.sh`, `run_bench_extra_2015_2017.sh`, `_run_assist2017_binary.sh`.
- Chunked30 variants not on the headline table path. `_run_chunked_bench_seeds.sh`, `_run_chunked_bench_sweep.sh`.
- Synthetic-5 variants not on the headline table path. `_run_synthetic5_shuf_sweep.sh`, `_run_synthetic5_yeung_sweep.sh`.
- Imbalance and scaling extensions. `_run_imb_scale_cv.sh`, `run_imbalance_extension.sh`, `run_after_imbalance.sh`.
- ASSIST 2009 superseded by `_run_pykt_sweep.sh`. `run_assist2009.sh`, `run_assist2009_ord.sh`.
- One-off chain runners and re-evals. `run_after_chain.sh`, `run_remaining.sh`, `eval_remaining.sh`, `eval_all_collect.sh`, `eval_and_compare_learned.sh`, `rerun_discrete_evals.sh`, `rerun_static_evals.sh`, `resweep_eval_all.sh`, `retrain_baselines.sh`, `_reeval_discrete.sh`, `_k4_digest.sh`.
- Softmax CV one-off. `_run_softmax_cv.sh`.
- Learned-embedding sweep. `run_learned_sweep.sh`, `train_learned_repr.sh`.
- Alpha=1 ablation. `train_alpha1.sh`.
- Ablations. `train_ablations.sh`, `gen_ablation_data.sh`.
- Monitoring. `monitor_retrain.sh`.

## (b) Per-script reachability table

Status legend, CANONICAL (invoked by canonical orchestrator), DIRECT_CLI (called by user per CLAUDE.md or paper figure), DIAGNOSTIC (one-off exploration), DEPRECATED_BY_REFACTOR (superseded), STALE (references missing or renamed paths).

| File | Status | Target archive path | Note |
|---|---|---|---|
| `train.py` | CANONICAL | keep | Hard-stop list. |
| `evaluate.py` | CANONICAL | keep | Hard-stop list. Inlines linking via `utils/linking`. |
| `data_gen.py` | DIRECT_CLI | keep | Smoke and static DGP data. |
| `_build_pykt_synthetic5.py` | CANONICAL | keep | Builds `synthetic5_v{0..4}` from external CSVs. |
| `_aggregate_pykt_results.py` | CANONICAL | keep | Table 2 aggregator. |
| `aggregate_recovery_v5.py` | CANONICAL | keep | Tables 1 and 3 aggregator. |
| `mirt_baseline_all_k.R` | CANONICAL | keep | GPCM (EM) ceiling. |
| `mirt_predict.R` | CANONICAL | keep | R `mirt` helper. |
| `convert_assistments.py` | DIRECT_CLI | keep | Required for ASSIST 2017 rows. Move to `dataloading/converters/`. |
| `convert_assistments_2009.py` | DIRECT_CLI | keep | Required for ASSIST 2009 rows. Move to `dataloading/converters/`. |
| `_run_pykt_sweep.sh` | CANONICAL | keep | Table 2. |
| `_run_k4_cv_recovery.sh` | CANONICAL | keep | Tables 1 and 3 at K=4. |
| `_run_k356_cv_recovery.sh` | CANONICAL | keep | Tables 1 and 3 at K=3, 5, 6. |
| `run_bulk_retrain.sh` | CANONICAL | keep | Older seed protocol; cited in `benchmarks.md`. |
| `_extract_row.py` | CANONICAL | keep | Invoked by `run_bulk_retrain.sh`, line 107. |
| `data_gen_block.py` | DIRECT_CLI | move to `dataloading/dgps/block.py` | Used for dynamic-DGP rows in Tables 1 and 3 via `_run_k*_cv_recovery.sh`. |
| `data_gen_randomwalk.py` | DIRECT_CLI | move to `dataloading/dgps/randomwalk.py` | Used for continuous DGP rows. |
| `data_gen_staircase.py` | DIRECT_CLI | move to `dataloading/dgps/staircase.py` | Used for staircase figures (still cited). |
| `data_gen_imbalanced.py` | DIRECT_CLI | move to `dataloading/dgps/imbalanced.py` | Imbalance section. |
| `plot_metrics.py` | DIRECT_CLI | move to `plotting/metrics.py` | Cited in `CLAUDE.md` Commands. |
| `plot_recovery_split.py` | DIRECT_CLI | move to `plotting/recovery.py` | Paper Figure. |
| `plot_trajectory_comparison.py` | DIRECT_CLI | move to `plotting/trajectory.py` | Paper Figure. |
| `plot_block_and_rw.py` | DIRECT_CLI | move to `plotting/dgp_panels.py` | Paper Figure. |
| `plot_theta_temporal.py` | DIRECT_CLI | move to `plotting/theta_temporal.py` | Paper Figure. |
| `plot_learner_trajectories.py` | DIRECT_CLI | move to `plotting/learner_trajectories.py` | Paper Figure. |
| `plot_assistments_item_params.py` | DIRECT_CLI | move to `plotting/assistments_items.py` | Paper Figure. |
| `plot_assistments_item_params_learned.py` | DIRECT_CLI | move to `plotting/assistments_items.py` (merge with above as `--variant learned`) | Near-duplicate of the above. |
| `plot_assistments_theta.py` | DIRECT_CLI | move to `plotting/assistments_theta.py` | Paper Figure. |
| `compute_linking.py` | DEPRECATED_BY_REFACTOR | `ma-irt/archive/scripts/` | Now redundant; the math lives in `utils/linking.mean_sigma_coefs`. The CLI loops over a model and emits per-seed (A, B). The aggregator hook is described in `NOTES_linking_appendix.md` but is not on the canonical path. |
| `aggregate_recovery_v4.py` | DEPRECATED_BY_REFACTOR | `ma-irt/archive/scripts/` | Superseded by v5 (handles both seed and fold layouts). |
| `generate_tables.py` | DEPRECATED_BY_REFACTOR | `ma-irt/archive/scripts/` | Superseded by `_aggregate_pykt_results.py`. |
| `_aggregate_bench.py` | DEPRECATED_BY_REFACTOR | `ma-irt/archive/scripts/` | Older seed-based binary aggregator. Listed in CLEANUP_VERIFICATION Section 3.2 but no canonical orchestrator calls it and Table 2 is now produced by `_aggregate_pykt_results.py`. The seed-based rows it consumes (`bench_*_s{0,1,7,42,123}`) were superseded by `pykt_fold` rows. Verify with one regression cell before archiving. |
| `convert_dkvmn_format.py` | DEPRECATED_BY_REFACTOR | `ma-irt/archive/scripts/` | Historical converter; raw inputs no longer used. |
| `_convert_yeung_synthetic.py` | DEPRECATED_BY_REFACTOR | `ma-irt/archive/scripts/` | Synthetic-5 Yeung variant; not in `benchmarks.md`. |
| `_emit_k4_tables.py` | DEPRECATED_BY_REFACTOR | `ma-irt/archive/scripts/` | Superseded by aggregators. |
| `gen_all_configs.py` | DEPRECATED_BY_REFACTOR | `ma-irt/archive/scripts/` | Bench configs are frozen in repo. |
| `gen_bench_configs.py` | DEPRECATED_BY_REFACTOR | `ma-irt/archive/scripts/` | Older seed-based config generator; superseded by `_gen_pykt_configs.py`. |
| `_gen_pykt_configs.py` | DEPRECATED_BY_REFACTOR | `ma-irt/archive/scripts/` | Configs already frozen; the generator is documentation, not on path. |
| `_gen_chunked_bench_configs.py` | DEPRECATED_BY_REFACTOR | `ma-irt/archive/scripts/` | Chunked30 not in `benchmarks.md`. |
| `_gen_imb_scale_pykt_configs.py` | DEPRECATED_BY_REFACTOR | `ma-irt/archive/scripts/` | Imbalance section, not in `benchmarks.md`. |
| `gen_ablation_configs.py` | DEPRECATED_BY_REFACTOR | `ma-irt/archive/scripts/` | Ablation, separate. |
| `gen_alpha1_configs.py` | DEPRECATED_BY_REFACTOR | `ma-irt/archive/scripts/` | Alpha=1 ablation. |
| `gen_assist_learned_configs.py` | DEPRECATED_BY_REFACTOR | `ma-irt/archive/scripts/` | Learned-emb ASSIST variants. |
| `gen_raw_alpha_configs.py` | DEPRECATED_BY_REFACTOR | `ma-irt/archive/scripts/` | Variant configs. |
| `gen_table_updates.py` | DEPRECATED_BY_REFACTOR | `ma-irt/archive/scripts/` | Draft table emitter. |
| `_gen_table_rows.py` | DEPRECATED_BY_REFACTOR | `ma-irt/archive/scripts/` | Draft row emitter. |
| `_bench_table_draft.tex` | DIAGNOSTIC | `ma-irt/archive/scripts/` | Draft, not active LaTeX. |
| `analyze_threshold_ordering.py` | DIAGNOSTIC | `ma-irt/archive/scripts/` | One-off. |
| `compare_alpha1.py` | DIAGNOSTIC | `ma-irt/archive/scripts/` | Alpha=1 ablation. |
| `diag_alpha_collapse.py` | DIAGNOSTIC | `ma-irt/archive/scripts/` | Diagnostic. |
| `investigate_wol_threshold.py` | DIAGNOSTIC | `ma-irt/archive/scripts/` | Diagnostic. |
| `eval_retrained.py` | DIAGNOSTIC | `ma-irt/archive/scripts/` | One-off. |
| `monitor.py` | DIAGNOSTIC | `ma-irt/archive/scripts/` | Monitoring utility. |
| `_linking_learned.py` | DIAGNOSTIC | `ma-irt/archive/scripts/` | Learned-emb linking experiment. |
| `_orbit_align_static_experiment.py` | DIAGNOSTIC | `ma-irt/archive/scripts/` | One-off. |
| `_profile_dkvmn.py` | DIAGNOSTIC | `ma-irt/archive/scripts/` | Profiling. |
| `_verify_bench_configs.py` | DIAGNOSTIC | `ma-irt/archive/scripts/` | Config audit; nice to have but not on path. |
| `_verify_datasets.py` | DIAGNOSTIC | `ma-irt/archive/scripts/` | Dataset audit. |
| `_verify_fix_behavior.py` | DIAGNOSTIC | `ma-irt/archive/scripts/` | One-off. |
| `_extract_row.py` | CANONICAL | keep | Used by `run_bulk_retrain.sh`. |
| `monitor_retrain.sh` | DIAGNOSTIC | `ma-irt/archive/scripts/` | Monitoring. |
| Bench-phase chain `.sh` listed in (a) | DEPRECATED_BY_REFACTOR | `ma-irt/archive/scripts/` | Superseded. |
| One-off `.sh` rerun and eval scripts listed in (a) | DIAGNOSTIC | `ma-irt/archive/scripts/` | One-off. |
| `_k4_digest.sh`, `_reeval_discrete.sh` | DIAGNOSTIC | `ma-irt/archive/scripts/` | One-off. |
| Imbalance and scaling `.sh` | DEPRECATED_BY_REFACTOR | `ma-irt/archive/scripts/` | Not on `benchmarks.md` path. |
| Learned-sweep `.sh` | DEPRECATED_BY_REFACTOR | `ma-irt/archive/scripts/` | Not on path. |

Counts after applying the table,

- Keep in `ma-irt/scripts/`, 14 files. `train.py`, `evaluate.py`, `data_gen.py`, `_build_pykt_synthetic5.py`, `_aggregate_pykt_results.py`, `aggregate_recovery_v5.py`, `_extract_row.py`, `mirt_baseline_all_k.R`, `mirt_predict.R`, `_run_pykt_sweep.sh`, `_run_k4_cv_recovery.sh`, `_run_k356_cv_recovery.sh`, `run_bulk_retrain.sh`, `__pycache__/`. Plus `__init__.py` if pytest needs it.
- Move into `dataloading/converters/`, 2 files (ASSIST 2017 and 2009 converters).
- Move into `dataloading/dgps/`, 4 files (block, randomwalk, staircase, imbalanced data generators).
- Move into `plotting/`, 9 files (8 plot scripts plus a merged `assistments_items` helper).
- Archive into `ma-irt/archive/scripts/`, 65 files.

`ma-irt/scripts/` ends at 14 active files, well under the 30 target.

Important sanity check before any move,

- `_aggregate_bench.py` appears in `CLEANUP_VERIFICATION_2026.md` Section 3.2 as a hard-stop file. Section 4.1 simultaneously says it is superseded. Resolution, the older seed-based aggregator is not on the canonical reproduction path for the three current `benchmarks.md` tables, but the verification doc still names it. Either (i) keep `_aggregate_bench.py` in `scripts/` for back-compat and accept 15 files, or (ii) archive it and amend `CLEANUP_VERIFICATION_2026.md` Section 3.2 in the same commit. Recommendation, archive, with the doc amended in the same change. The risk is one regression cell on the older seeded protocol; nothing in `benchmarks.md` consumes it today.

## (c) Cross-directory consolidation

Five consolidation candidates, scored by file count reduction and risk.

### C1. `plotting/` package replaces eight `plot_*.py` scripts

Current, eight scripts at the top of `ma-irt/scripts/`, totaling about 3260 lines. Each loads matplotlib with the same pgf preamble, parses argparse, loads model checkpoints via `MAGPCM`/`StaticGPCM`/`DynamicGPCM`, computes a figure.

Proposed,

```
ma-irt/plotting/
  __init__.py
  _common.py            # pgf preamble, color palette, checkpoint loader
  metrics.py            # was plot_metrics.py
  recovery.py           # was plot_recovery_split.py
  trajectory.py         # was plot_trajectory_comparison.py
  dgp_panels.py         # was plot_block_and_rw.py
  theta_temporal.py     # was plot_theta_temporal.py
  learner_trajectories.py
  assistments_items.py  # merge plot_assistments_item_params.py + _learned variant
  assistments_theta.py
ma-irt/scripts/plot.py  # thin argparse dispatch
```

Net effect, eight files in `scripts/` collapse to one entry-point CLI plus a new sub-package. Shared pgf preamble lives once in `_common.py`. The two `plot_assistments_item_params*` variants differ only in embedding-type handling and merge cleanly under a `--variant {static_item,learned}` flag. Saves about 100 lines of duplicated rcParams.

Net file delta, `scripts/` loses 8 files and gains 1 entry-point. Net minus 7.

### C2. `dataloading/converters/` package replaces three converters

Current, `convert_assistments.py`, `convert_assistments_2009.py`, `convert_dkvmn_format.py`. The first two share an `apply_ordinal_mapping` helper and the same proxy-ordinal logic; only column names differ.

Proposed,

```
ma-irt/dataloading/converters/
  __init__.py
  _ordinal.py            # shared proxy-ordinal K=4 mapping
  assistments_2017.py    # was convert_assistments.py
  assistments_2009.py    # was convert_assistments_2009.py
ma-irt/scripts/convert.py # thin argparse dispatch
```

`convert_dkvmn_format.py` is DEPRECATED_BY_REFACTOR and is archived rather than merged. The hard-stop list keeps `convert_assistments.py` and `convert_assistments_2009.py`; both retain their public CLI through the `scripts/convert.py` dispatcher, so the recipes in `CLEANUP_VERIFICATION_2026.md` Section 1.2 still resolve.

Net file delta, `scripts/` loses 3 files and gains 1. Net minus 2.

### C3. `dataloading/dgps/` package replaces four data generators

Current, `data_gen.py` (static GPCM), `data_gen_block.py`, `data_gen_randomwalk.py`, `data_gen_staircase.py`, `data_gen_imbalanced.py`. Five files, each with its own argparse and a `*GPCMGenerator` class that shares the same alpha and beta generating distribution from `data_gen.py`.

Proposed,

```
ma-irt/dataloading/dgps/
  __init__.py
  _gpcm_core.py    # shared alpha, beta sampling and sequence emission
  static.py        # was data_gen.py
  block.py         # was data_gen_block.py
  randomwalk.py    # was data_gen_randomwalk.py
  staircase.py     # was data_gen_staircase.py
  imbalanced.py    # was data_gen_imbalanced.py
ma-irt/scripts/data_gen.py    # thin argparse dispatch with --dgp {static,block,rw,staircase,imbalanced}
```

`scripts/data_gen.py` remains as the user entry point and forwards to the right generator. The hard-stop reference in `CLEANUP_VERIFICATION_2026.md` Section 1.1 (`python scripts/data_gen.py --name static_q200_k3 ...`) still works because the dispatcher preserves the existing CLI flags for the default static DGP.

Net file delta, `scripts/` loses 4 files (block, randomwalk, staircase, imbalanced) and `data_gen.py` stays as the dispatcher. Net minus 4.

### C4. Aggregator collapse

Current, `aggregate_recovery_v4.py`, `aggregate_recovery_v5.py`, `_aggregate_bench.py`, `_aggregate_pykt_results.py`, `generate_tables.py`, `_emit_k4_tables.py`, `_extract_row.py`, `_gen_table_rows.py`.

Canonical, `aggregate_recovery_v5.py` and `_aggregate_pykt_results.py` (the active fold protocol). `_extract_row.py` stays because `run_bulk_retrain.sh` calls it inline. The remaining five archive.

Proposed,

```
ma-irt/scripts/
  aggregate_recovery_v5.py   # keep
  _aggregate_pykt_results.py # keep
  _extract_row.py            # keep, called by run_bulk_retrain.sh
```

Optional follow-up, rename `aggregate_recovery_v5.py` to `aggregate_recovery.py` once v4 is archived and no doc still references the `_v5` name. Deferred to a second tier so this redesign stays additive.

Net file delta, scripts loses 5 files (`_aggregate_bench.py`, `aggregate_recovery_v4.py`, `generate_tables.py`, `_emit_k4_tables.py`, `_gen_table_rows.py`). Net minus 5.

### C5. Config generator collapse

Current, `gen_all_configs.py`, `gen_bench_configs.py`, `_gen_pykt_configs.py`, `_gen_chunked_bench_configs.py`, `_gen_imb_scale_pykt_configs.py`, `gen_ablation_configs.py`, `gen_alpha1_configs.py`, `gen_assist_learned_configs.py`, `gen_raw_alpha_configs.py`, `gen_table_updates.py`. Ten files.

Reality, every bench config is frozen in `configs/bulk/`. No canonical orchestrator regenerates configs. The generators are documentation of provenance, not active code.

Proposed, archive all ten to `ma-irt/archive/scripts/`. If config provenance must remain inspectable, the single most recent generator (`_gen_pykt_configs.py`) can move into a new `config/generators/` sub-package, but the recommendation is archive-only since the bench configs are frozen and regenerating them would change the test hash.

Net file delta, scripts loses 10 files. Net minus 10.

### Consolidation summary

| ID | Topic | Files removed from `scripts/` | Files added | New sub-package |
|---|---|---|---|---|
| C1 | plotting | 8 | 1 dispatcher | `ma-irt/plotting/` |
| C2 | converters | 3 | 1 dispatcher | `ma-irt/dataloading/converters/` |
| C3 | DGP generators | 4 | 0 (dispatcher reuses `data_gen.py` name) | `ma-irt/dataloading/dgps/` |
| C4 | aggregator collapse | 5 | 0 | none |
| C5 | config generator collapse | 10 | 0 | none |

Net loss from `scripts/`, 30 files. Combined with the diagnostic and superseded shell scripts (around 35 more), `scripts/` ends at roughly 14 files.

## (d) Stale config trees

Confirmed on disk and not referenced by any canonical orchestrator.

| Path | Files | Reachable from canonical? | Recommendation |
|---|---|---|---|
| `ma-irt/configs/_archive_s0p5/` | 125 | No (already archive-named) | Move to `ma-irt/archive/configs/_archive_s0p5/` to fold into a single archive tree. |
| `ma-irt/configs/experiments/{ablation,rq1,rq4,rq5}/` | 4 subdirs | No (legacy RQ folders) | Move to `ma-irt/archive/configs/experiments/`. |
| `ma-irt/configs/tmp_alpha1/` | 125 | No (alpha=1 ablation) | Move to `ma-irt/archive/configs/tmp_alpha1/`. |
| `ma-irt/configs/dynamic_seeds/` | 160 | No (legacy dynamic seed configs) | Move to `ma-irt/archive/configs/dynamic_seeds/`. |

Root-level dynamic-DGP configs (`configs/block_*.yaml`, `configs/rw_*.yaml`, `configs/staircase_*.yaml`) stay. They are not on the `benchmarks.md` path but they are referenced by the dynamic-DGP plot scripts (`plot_block_and_rw.py`, `plot_trajectory_comparison.py`) and by `_run_k4_cv_recovery.sh` via the `discrete_q200_k4` and `continuous_q200_k4` data directories. Verify before any future archive.

The 1652 files in `configs/bulk/` are not part of this redesign. They are the actual sweep configs and the hard-stop list pins the `bench_*_pykt_fold*` subset. The older seeded `*_s{0,1,7,42,123}.yaml` family is superseded by the pyKT fold configs but `run_bulk_retrain.sh` still consumes them, so they stay.

## (e) Target directory structure

```
ma-irt/
  config/
    __init__.py
    loader.py
    types.py
  dataloading/
    __init__.py
    loaders.py
    converters/                # NEW (C2)
      __init__.py
      _ordinal.py
      assistments_2017.py
      assistments_2009.py
    dgps/                      # NEW (C3)
      __init__.py
      _gpcm_core.py
      static.py
      block.py
      randomwalk.py
      staircase.py
      imbalanced.py
  models/
    __init__.py
    magpcm.py
    static_gpcm.py
    dynamic_gpcm.py
    dkvmn_softmax.py
    dkt.py
    dkvmn.py
    deep_irt.py
    components/
      __init__.py
      memory.py
      irt.py
      embeddings.py
    heads/
      __init__.py
      gpcm.py
  training/
    __init__.py
    trainer.py
    losses.py
  utils/
    __init__.py
    metrics.py
    linking.py
  plotting/                    # NEW (C1)
    __init__.py
    _common.py
    metrics.py
    recovery.py
    trajectory.py
    dgp_panels.py
    theta_temporal.py
    learner_trajectories.py
    assistments_items.py
    assistments_theta.py
  scripts/                     # 14 files
    train.py                       # entry point
    evaluate.py                    # entry point
    data_gen.py                    # dispatcher into dataloading/dgps/
    convert.py                     # dispatcher into dataloading/converters/
    plot.py                        # dispatcher into plotting/
    _build_pykt_synthetic5.py      # builds synthetic5 from external CSVs
    aggregate_recovery_v5.py       # active Tables 1, 3 aggregator
    _aggregate_pykt_results.py     # active Table 2 aggregator
    _extract_row.py                # used by run_bulk_retrain.sh
    mirt_baseline_all_k.R          # GPCM (EM) ceiling
    mirt_predict.R                 # R mirt helper
    _run_pykt_sweep.sh             # Table 2 sweep
    _run_k4_cv_recovery.sh         # Tables 1, 3 at K=4
    _run_k356_cv_recovery.sh       # Tables 1, 3 at K=3, 5, 6
    run_bulk_retrain.sh            # seed-protocol sweep
  configs/
    base.yaml
    smoke*.yaml                   # 8 files
    block_*.yaml, rw_*.yaml, staircase_*.yaml  # dynamic-DGP root configs
    bulk/                          # 1652 sweep configs, hard-stop pinned
  data/                            # 12+ dataset dirs, hard-stop pinned
  tests/                           # hard-stop pinned
  outputs/                         # not in scope
  archive/
    scripts/                       # 65 archived scripts
    configs/
      _archive_s0p5/
      experiments/
      tmp_alpha1/
      dynamic_seeds/
```

Directory count delta. `ma-irt/` gains three packages (`plotting/`, `dataloading/converters/`, `dataloading/dgps/`) and one archive root (`ma-irt/archive/`). Net root directory count is unchanged because the four stale config trees collapse under `archive/configs/`. File count in `scripts/` drops from 94 to 14.

## (f) Execution order for the Archive phase

Each tier is verified by the smoke layer in `CLEANUP_VERIFICATION_2026.md` Section 2.1 (pytest plus 8 smoke trainings) before the next tier starts. The regression layer in Section 2.2 runs once after Tier 5.

1. **Tier 0, generated artifacts.** Remove `ma-irt/scripts/__pycache__/`. Zero risk. Verifies the move script and the smoke gate.
2. **Tier 1, draft and one-off scripts.** Archive all DIAGNOSTIC `.py` and `.sh` files listed in (b). Includes `_bench_table_draft.tex`, `_extract_row.py` stays (canonical). Smoke pytest must pass.
3. **Tier 2, deprecated-by-refactor scripts.** Archive `compute_linking.py`, `aggregate_recovery_v4.py`, `_aggregate_bench.py`, `generate_tables.py`, `convert_dkvmn_format.py`, `_convert_yeung_synthetic.py`, `_emit_k4_tables.py`, all `gen_*.py` and `_gen_*.py` generators, all superseded `run_bench_*.sh`, `_run_*` non-canonical sweeps, and the imbalance/scaling/learned `.sh` runners. The `_aggregate_bench.py` move amends `CLEANUP_VERIFICATION_2026.md` Section 3.2 in the same commit.
4. **Tier 3, stale config trees.** Move `configs/_archive_s0p5/`, `configs/experiments/`, `configs/tmp_alpha1/`, `configs/dynamic_seeds/` into `ma-irt/archive/configs/`. Update any relative references (none expected from the canonical orchestrators). Smoke configs must continue to resolve from `ma-irt/configs/`.
5. **Tier 4, converters consolidation (C2).** Create `dataloading/converters/`, move ASSIST 2017 and 2009 scripts, factor `_ordinal.py`. Replace `scripts/convert_assistments*.py` with `scripts/convert.py` dispatcher that exposes both CLIs under the same flags. Verify by rerunning the data conversion smoke (regenerate `assist2009_bin/sequences.json` with the same seed and diff against the cached one).
6. **Tier 5, DGP consolidation (C3).** Create `dataloading/dgps/`, move all five generators, factor `_gpcm_core.py`. Replace top-level `scripts/data_gen.py` with the dispatcher. Verify by running the canonical `data_gen.py` recipe for `static_q200_k4` and diffing the output bytes against the cached dataset.
7. **Tier 6, plotting consolidation (C1).** Create `ma-irt/plotting/`, move all eight plot scripts, factor `_common.py`. Add `scripts/plot.py` dispatcher. Verify by regenerating one paper figure (`plot_recovery_split` output) and pixel-diffing against the cached PGF.
8. **Tier 7, regression layer.** Run the 15-config regression sweep from `CLEANUP_VERIFICATION_2026.md` Section 2.2. Append a row to `cleanup_log.md` with the deltas.

## (g) Risk and rollback per consolidation

| Consolidation | Risk | Mitigation | Rollback |
|---|---|---|---|
| C1 plotting | Paper figures regress in PGF rendering (matplotlib path changes, font fallback) | Pixel-diff one PGF per script class before and after. Keep the `_common.py` rcParams byte-identical to the existing scripts. | Per-figure rollback. Restore the original `plot_*.py` from `ma-irt/archive/scripts/` and re-run. Each script is independent. |
| C2 converters | ASSIST sequences hash changes, downstream pyKT folds drift, Table 2 ASSIST rows shift | Diff `sequences.json` bytes for `assist2009_bin` and `assist2017_bin` before and after the consolidation. If the diff is non-empty, the shared `_ordinal.py` does not match the per-script logic, revert. | Restore both `convert_assistments*.py` from archive. The pyKT folds are deterministic given the input sequences, so reverting the converter restores the bench numbers. |
| C3 DGP generators | Static dataset hash changes for `static_q200_k{2..6}`, breaks the recovery baseline | Diff `sequences.json` for `static_q200_k4` (the regression layer dataset) before and after. The shared `_gpcm_core.py` must reproduce the static generator bit-exact under seed 42. | Restore `data_gen.py` from archive, leave the sub-package in place but stop the dispatcher from calling it. |
| C4 aggregator collapse | Older tables regenerated from seed protocol stop working | The seed protocol is documented in `_aggregate_bench.py` and used by no current orchestrator. Verify by checking that no `*.sh` or `*.py` outside `ma-irt/archive/` imports the archived modules. | Restore the specific aggregator from archive. Independent restore per file. |
| C5 config generators | Future config regeneration requires fishing the generator out of archive | All bench configs are frozen and tracked in git, so regeneration would only be needed for a new experimental cell. Document the archive location in `ma-irt/README.md` under "Provenance of frozen configs". | Restore from archive. Each generator is independent. |
| Tier 3 config trees | Plot scripts hardcode paths into archived config trees | Grep the eight plot scripts plus the four canonical orchestrators for `configs/experiments`, `configs/tmp_alpha1`, `configs/dynamic_seeds`, `configs/_archive_s0p5`. Confirmed no canonical orchestrator references them. | Move the tree back to its original location under `ma-irt/configs/`. |

Rollback affordances common to all tiers. Every tier corresponds to one commit on a `cleanup_pipeline_redesign` branch with a single revert SHA. The `outputs/` cache is untouched, so the cached benchmark numbers do not need to be regenerated even on full rollback.
