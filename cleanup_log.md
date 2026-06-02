# Cleanup Log

Per `CLEANUP_VERIFICATION_2026.md` Section 5. One row per tier applied.

| Date | Tier | Before (git ref) | After (git ref) | Layer 1 smoke | Layer 2 regression | Layer 3 full | Pass/Fail |
|---|---|---|---|---|---|---|---|
| 2026-06-02 | T0 (build artifacts) | 40f26f1 | e9beb44 | 65/65 pytest in 7.53s | N/A (no source/config touched) | N/A | **PASS** |
| 2026-06-02 | T1 (stale markdowns) | 40f26f1 | e9beb44 | 65/65 pytest in 7.53s | N/A (move-only, no source/config touched) | N/A | **PASS** |
| 2026-06-02 | T1.5 (markdown rescan) | 5b8b290 | 3ffa3e9 | 65/65 pytest in 6.47s | N/A (md and gitignored content only) | N/A | **PASS** |
| 2026-06-02 | T1.6 (archive inventory reports) | 3ffa3e9 | 9d4f1ff | N/A (move-only) | N/A | N/A | **PASS** |
| 2026-06-02 | T1.7 (archive phd_research_proposal) | 9d4f1ff | TBD | N/A (move-only) | N/A | N/A | **PASS** |

T0 and T1 share commit `e9beb44`. T1.5 (the comprehensive markdown rescan triggered after T1) gets its own commit.

## T1.5 actions

Driven by `MARKDOWN_INVENTORY_2026.md` (ml-system-architect) cross-referenced with `MARKDOWN_CRITICALITY_2026.md` (research-scientist).

**Archived** (moved to `docs/archive/2026-06-cleanup/ma-irt/scripts/`):
- `ma-irt/scripts/_bench_writeup_draft.md` (prose shipped to `main.tex:468-471`, kept for history)

**Deleted** (truly redundant or in gitignored parents):
- `ma-irt/scripts/_profile_dkvmn_report.md` (engineering scratch, optimization landed in commit `0fba5e8`)
- `ma-irt/.pytest_cache/README.md`, `kt-mirt/.pytest_cache/README.md` (auto-generated stubs)
- `archive_sigma03_20260422_0534/outputs/recovery_summary_v4.md` (sigma-0.3 result dump superseded by sigma-0.5)
- `deep-gpcm/TODO.md` (4-month-old TODO from legacy project not in paper)
- `deep-gpcm/results/hyperopt/*_hyperopt_report.md` (4 files, document `attn_gpcm` model variant not in paper)

**Updated**:
- `ma-irt/README.md` rewritten in place to fix stale refs (`run_all_experiments.sh`, `plot_recovery.py`, `plot_recovery_figure.py`), switch to 5-fold CV protocol throughout, list the real orchestrators (`_run_pykt_sweep.sh`, `_run_k4_cv_recovery.sh`, `_run_k356_cv_recovery.sh`, `run_bulk_retrain.sh`), real aggregators (`_aggregate_pykt_results.py`, `_aggregate_bench.py`), and correct `embedding_type` values
- `.gitignore` adds `.pytest_cache/`

**Kept** (research-scientist confirmed paper/rebuttal load-bearing):
- `ma-irt/NOTES_linking_appendix.md` (source for planned linking-constants appendix, backed by `compute_linking.py`)
- `ma-irt/REVIEW_converged.md`, `REVIEW_psychometric.md`, `REVIEW_research_scientist.md` (unresolved rebuttal items F7, P2, R5, R8 against IJAIED review round)

## Notes

- T1.5 is named "1.5" because it executes a follow-up rescan that should have been part of T1 but was scoped too narrowly the first time. It is not a separate tier in the original `CLEANUP_PLAN_2026.md` but a tightening of T1.
- The two reports `MARKDOWN_INVENTORY_2026.md` and `MARKDOWN_CRITICALITY_2026.md` are themselves KEEP, they document the rescan's reasoning.

## T1.6 actions

Tightening pass after the user asked for "extra clean". Moved the two T1.5 inventory artifacts to archive since their purpose is done.

- `MARKDOWN_INVENTORY_2026.md` -> `docs/archive/2026-06-cleanup/`
- `MARKDOWN_CRITICALITY_2026.md` -> `docs/archive/2026-06-cleanup/`

## T1.7 actions

User flagged `phd_research_proposal.md` as redundant at root. Distilled the key points into a new "Research roadmap" section in `README.md` (one paragraph plus four-direction bullet list, ~12 lines) and moved the full proposal to archive.

- `phd_research_proposal.md` -> `docs/archive/2026-06-cleanup/`
- `README.md` updated with the distilled roadmap; the "See also" link to the proposal removed (the roadmap points at the archived full version)

Root markdowns after T1.7: `README.md`, `CLAUDE.md`, `benchmarks.md`, `CLEANUP_PLAN_2026.md`, `CLEANUP_VERIFICATION_2026.md`, `cleanup_log.md`. Six files, all active.

## Pending

## T2.0 actions

Stabilization pass before moving or deleting more code. The current working tree had many paper-critical files present on disk but untracked by git, including the DKT/DKVMN/Deep-IRT baselines, smoke configs, benchmark CV configs, and the sweep/aggregation scripts required by `CLEANUP_VERIFICATION_2026.md`.

Committed as `410efed` (`Track paper-critical benchmark assets`):

- Tracked the 300 benchmark configs required by `CLEANUP_VERIFICATION_2026.md` Section 3.3.
- Tracked `ma-irt/models/{dkt.py,dkvmn.py,deep_irt.py}`.
- Tracked smoke configs for DKT, DKVMN, and Deep-IRT.
- Tracked the required sweep/aggregation scripts: `_build_pykt_synthetic5.py`, `_aggregate_pykt_results.py`, `_aggregate_bench.py`, `aggregate_recovery_v5.py`, `_run_pykt_sweep.sh`, `_run_k4_cv_recovery.sh`, `_run_k356_cv_recovery.sh`.
- Updated `.gitignore` so large legacy/vendor/raw-data trees and standalone manuscript/reference artifacts stop appearing as untracked source.

Verification:

- `python -m py_compile` on the newly tracked Python scripts and baseline model files: pass.
- `cd ma-irt; PYTHONPATH=. pytest tests -q`: `65 passed`.
- Regression layer from `CLEANUP_VERIFICATION_2026.md` Section 2.2 was not run; no source behavior was changed in this pass.

## Pending

T2 legacy moves for root reference repos, T3 dead-code/script archive in `ma-irt/`, T4 config consolidation, T5 deeper refactor. The next safe step is T3 reachability archiving of tracked one-off scripts/configs, but only after separating it from the pre-existing uncommitted source/config edits.
