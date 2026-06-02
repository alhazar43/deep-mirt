# Script Taxonomy

This manifest classifies the current `ma-irt/scripts/` tree before any
archival moves. It is a cleanup planning document, not a deletion request.

Inventory date: 2026-06-02.

Current count: 93 script/report files plus one generated `__pycache__/`
directory.

## Classification Rules

Status values:

- **KEEP**: public or paper-critical path.
- **KEEP-DOC**: useful supporting script, but not a primary entry point.
- **REVIEW**: may be needed for a paper section, figure, or legacy rerun.
- **ARCHIVE-CANDIDATE**: one-off, superseded, diagnostic, or draft.
- **GENERATED-CLEANUP**: generated runtime artifact, safe remove-only cleanup.

No file should be moved until its status is reviewed against
`CLEANUP_VERIFICATION_2026.md`, `benchmarks.md`, `overleaf-sync/main.tex`,
and the generated/cached outputs needed for paper figures.

## Public Entry Points

| File | Status | Rationale |
|---|---|---|
| `train.py` | KEEP | Main training CLI. |
| `evaluate.py` | KEEP | Main prediction/recovery evaluation CLI. |
| `data_gen.py` | KEEP | Static synthetic GPCM generator and smoke data path. |
| `data_gen_block.py` | KEEP | Dynamic block-change DGP used by dynamic experiments. |
| `data_gen_randomwalk.py` | KEEP | Continuous random-walk DGP used by dynamic experiments. |
| `data_gen_staircase.py` | KEEP | Staircase DGP used by dynamic experiments. |
| `data_gen_imbalanced.py` | KEEP | Imbalance/scaling extension data generator. |
| `plot_metrics.py` | KEEP | Public training-curve plotting utility. |
| `plot_recovery_split.py` | KEEP | Main IRT recovery figure utility. |

## Paper-Reproduction Scripts

| File | Status | Rationale |
|---|---|---|
| `_run_pykt_sweep.sh` | KEEP | Paper binary and ASSISTments benchmark sweep. |
| `_run_k4_cv_recovery.sh` | KEEP | K=4 synthetic recovery sweep. |
| `_run_k356_cv_recovery.sh` | KEEP | K=3,5,6 synthetic recovery sweep. |
| `_aggregate_pykt_results.py` | KEEP | Aggregates pyKT-style benchmark tables. |
| `aggregate_recovery_v5.py` | KEEP | Current synthetic recovery aggregator. |
| `mirt_baseline_all_k.R` | KEEP | Offline GPCM (EM) recovery baseline. |
| `mirt_predict.R` | KEEP | R `mirt` prediction helper. |
| `_build_pykt_synthetic5.py` | KEEP | Builds Synthetic-5 data from external DKVMN raw CSVs. |
| `convert_assistments.py` | KEEP | ASSISTments 2017 conversion. |
| `convert_assistments_2009.py` | KEEP | ASSISTments 2009 conversion. |

## Figure and Diagnostic Plot Scripts

| File | Status | Rationale |
|---|---|---|
| `plot_assistments_item_params.py` | REVIEW | Paper/supporting learned item-parameter figure path. |
| `plot_assistments_item_params_learned.py` | REVIEW | Learned-representation ASSISTments figure variant. |
| `plot_assistments_theta.py` | REVIEW | ASSISTments theta figure path. |
| `plot_block_and_rw.py` | REVIEW | Dynamic DGP figure path. |
| `plot_learner_trajectories.py` | REVIEW | Learner trajectory figure path. |
| `plot_theta_temporal.py` | REVIEW | Temporal theta diagnostics. |
| `plot_trajectory_comparison.py` | REVIEW | Model trajectory comparison figure path. |

Keep these until every active paper figure in `overleaf-sync/figures/` has a
documented regeneration path or has been declared final.

## Config and Table Generation

| File | Status | Rationale |
|---|---|---|
| `gen_bench_configs.py` | KEEP-DOC | Generated frozen benchmark configs; useful for regeneration provenance. |
| `gen_all_configs.py` | REVIEW | Older bulk config generator; likely superseded by frozen configs. |
| `generate_tables.py` | REVIEW | Table generation utility; may be superseded by aggregators. |
| `gen_table_updates.py` | ARCHIVE-CANDIDATE | Draft/update helper, not a canonical paper path. |
| `_gen_pykt_configs.py` | REVIEW | Generated pyKT-style configs; keep until config provenance is documented. |
| `_gen_chunked_bench_configs.py` | ARCHIVE-CANDIDATE | Chunked30 variant generator, not on active paper path. |
| `_gen_imb_scale_pykt_configs.py` | REVIEW | Imbalance/scaling configs may support paper extensions. |
| `_gen_table_rows.py` | ARCHIVE-CANDIDATE | Draft row emitter. |

## Monitoring and Rerun Orchestrators

| File | Status | Rationale |
|---|---|---|
| `run_bulk_retrain.sh` | REVIEW | Older/current bulk retrain runner cited in benchmark docs; verify before archive. |
| `run_bench_sweep.sh` | ARCHIVE-CANDIDATE | Superseded by `_run_pykt_sweep.sh` and recovery runners. |
| `run_bench_phase1.sh` | ARCHIVE-CANDIDATE | One-off phase runner. |
| `run_bench_phase2_queue.sh` | ARCHIVE-CANDIDATE | One-off phase runner. |
| `run_bench_phases_chain.sh` | ARCHIVE-CANDIDATE | One-off chain runner. |
| `run_bench_postphase2.sh` | ARCHIVE-CANDIDATE | One-off phase runner. |
| `run_after_chain.sh` | ARCHIVE-CANDIDATE | One-off chain runner. |
| `run_after_imbalance.sh` | REVIEW | May support imbalance/scaling section. |
| `run_imbalance_extension.sh` | REVIEW | May support imbalance/scaling section. |
| `_run_imb_scale_cv.sh` | REVIEW | Imbalance/scaling CV runner; verify paper section/appendix role. |
| `run_learned_sweep.sh` | REVIEW | Learned-representation sweep; verify paper role. |
| `run_bench_extra_2015_2017.sh` | REVIEW | Extra ASSISTments benchmark variants. |
| `run_assist2009.sh` | ARCHIVE-CANDIDATE | Superseded by pyKT/ASSIST orchestrators. |
| `run_assist2009_ord.sh` | REVIEW | Ordinal ASSISTments runner; verify paper figure/table role. |
| `run_remaining.sh` | ARCHIVE-CANDIDATE | One-off completion runner. |
| `eval_all_collect.sh` | ARCHIVE-CANDIDATE | One-off collector. |
| `eval_remaining.sh` | ARCHIVE-CANDIDATE | One-off evaluator. |
| `eval_and_compare_learned.sh` | REVIEW | Learned-representation comparison. |
| `rerun_discrete_evals.sh` | ARCHIVE-CANDIDATE | One-off re-eval. |
| `rerun_static_evals.sh` | ARCHIVE-CANDIDATE | One-off re-eval. |
| `resweep_eval_all.sh` | ARCHIVE-CANDIDATE | One-off resweep evaluator. |
| `retrain_baselines.sh` | ARCHIVE-CANDIDATE | Older baseline retrain helper. |
| `monitor.py` | KEEP-DOC | Runtime monitoring utility; useful but not paper-critical. |
| `monitor_retrain.sh` | KEEP-DOC | Runtime monitoring utility; useful but not paper-critical. |

## Ablations, Variants, and Diagnostics

| File | Status | Rationale |
|---|---|---|
| `gen_ablation_configs.py` | REVIEW | Ablation configs may support paper limitations/appendix. |
| `gen_ablation_data.sh` | REVIEW | Ablation data helper. |
| `train_ablations.sh` | REVIEW | Ablation runner. |
| `gen_alpha1_configs.py` | REVIEW | Alpha=1 ablation config generator. |
| `train_alpha1.sh` | REVIEW | Alpha=1 ablation runner. |
| `compare_alpha1.py` | REVIEW | Alpha=1 ablation analysis. |
| `gen_assist_learned_configs.py` | REVIEW | Learned ASSISTments config generator. |
| `train_learned_repr.sh` | REVIEW | Learned representation runner. |
| `gen_raw_alpha_configs.py` | REVIEW | Raw-alpha variant generator. |
| `_linking_learned.py` | REVIEW | Learned-linking diagnostic. |
| `compute_linking.py` | REVIEW | Linking helper; may become source for future utility extraction. |
| `eval_retrained.py` | ARCHIVE-CANDIDATE | One-off evaluation helper. |
| `analyze_threshold_ordering.py` | ARCHIVE-CANDIDATE | Diagnostic analysis. |
| `diag_alpha_collapse.py` | ARCHIVE-CANDIDATE | Diagnostic analysis. |
| `investigate_wol_threshold.py` | ARCHIVE-CANDIDATE | Diagnostic analysis. |
| `_orbit_align_static_experiment.py` | ARCHIVE-CANDIDATE | One-off experiment. |
| `_profile_dkvmn.py` | ARCHIVE-CANDIDATE | Profiling script. |
| `_verify_fix_behavior.py` | ARCHIVE-CANDIDATE | One-off fix verifier. |
| `_verify_bench_configs.py` | KEEP-DOC | Useful config audit helper. |
| `_verify_datasets.py` | KEEP-DOC | Useful dataset audit helper. |

## Drafts and Superseded Artifacts

| File | Status | Rationale |
|---|---|---|
| `_bench_table_draft.tex` | ARCHIVE-CANDIDATE | Draft table artifact, not active LaTeX. |
| `_emit_k4_tables.py` | ARCHIVE-CANDIDATE | Superseded by current aggregators. |
| `_extract_row.py` | ARCHIVE-CANDIDATE | One-off digest helper. |
| `_k4_digest.sh` | ARCHIVE-CANDIDATE | One-off digest helper. |
| `_aggregate_bench.py` | REVIEW | Older aggregator; verify whether any cached/legacy table depends on it. |
| `aggregate_recovery_v4.py` | ARCHIVE-CANDIDATE | Superseded by `aggregate_recovery_v5.py`. |
| `_convert_yeung_synthetic.py` | ARCHIVE-CANDIDATE | Yeung synthetic variant, not current benchmark path. |
| `convert_dkvmn_format.py` | ARCHIVE-CANDIDATE | Historical converter; current conversions use ASSISTments scripts. |
| `_reeval_discrete.sh` | ARCHIVE-CANDIDATE | One-off re-eval. |
| `_run_assist2017_binary.sh` | ARCHIVE-CANDIDATE | Superseded by `_run_pykt_sweep.sh`. |
| `_run_chunked_bench_seeds.sh` | ARCHIVE-CANDIDATE | Chunked30 variant, not active paper path. |
| `_run_chunked_bench_sweep.sh` | ARCHIVE-CANDIDATE | Chunked30 variant, not active paper path. |
| `_run_softmax_cv.sh` | ARCHIVE-CANDIDATE | One-off softmax CV helper. |
| `_run_synthetic5_shuf_sweep.sh` | ARCHIVE-CANDIDATE | Shuffled Synthetic-5 variant. |
| `_run_synthetic5_yeung_sweep.sh` | ARCHIVE-CANDIDATE | Yeung Synthetic-5 variant. |

## Generated Runtime Artifacts

| Path | Status | Rationale |
|---|---|---|
| `__pycache__/` | GENERATED-CLEANUP | Python bytecode cache. Safe remove-only cleanup. |

## Next Action

Before moving scripts, create a small archive batch from only
`GENERATED-CLEANUP` and obvious `ARCHIVE-CANDIDATE` draft artifacts, then run
the smoke/test gates. Keep all `REVIEW` files until paper figure and appendix
dependencies are resolved.
