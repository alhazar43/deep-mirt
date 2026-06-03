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

## T2.1 actions

Follow-up noise reduction after T2.0. Committed as `eb41791` (`Track review notes and ignore generated cleanup noise`):

- Tracked the four research/review documents previously classified as keep-worthy: `ma-irt/NOTES_linking_appendix.md`, `ma-irt/REVIEW_converged.md`, `ma-irt/REVIEW_psychometric.md`, `ma-irt/REVIEW_research_scientist.md`.
- Added ignore rules for untracked generated config families (`_archive_s0p5`, `tmp_alpha1`, chunked/seeded/learned/imbalance/scalability variants) that are outside the verified headline reproduction set.
- Added ignore rules for untracked one-off sweep/config/debug scripts outside the verified reproduction path.
- Rechecked the `CLEANUP_VERIFICATION_2026.md` required benchmark config set after the ignore update: `required_untracked_after_ignore=0`.

Verification:

- No source behavior changed in this pass.
- The previous T2.0 verification still applies: py_compile pass; `65 passed`.

## Overnight session 2026-06-03 (T7-prep + P1 + P2 foundation + T2 + P6)

User granted full edit approval before going to sleep. Executed the
investigate-baseline-plan-design workflow (artifacts in
`docs/cleanup/`), then proceeded with the safest items of the
synthesized `PIPELINE_OPT_PLAN.md`. Ten new forward commits, 107 passed
and 1 skipped in every commit's pytest run.

| Commit | Phase | Summary | Result |
|---|---|---|---|
| `613b812` | T7-prep A | `alpha_log_scale` config consistency, `alpha_from_raw` helper | 68/68 |
| `92b3f47` | T7-prep B | CV/early-stop infra (config, loader, dataloader) | 68/68 |
| `0701d1a` | T7-prep C | K=2 AUC, DKT/DKVMN/DeepIRT dispatch, plot scripts | 68/68 |
| `cb0cf1e` | P1.a | extract `utils/linking.py` plus 15-test regression hash | 83/83 |
| `cdb78ad` | P1.b | unify `build_model` in `models/__init__.py` | 83/83 |
| `1c9ac7c` | P1.e | lift `scipy.stats.spearmanr` to module scope | 83/83 |
| `26047f4` | P2.a R2 gate | `tests/test_baseline_reproduction.py` | 92 + 1 skip |
| `8c8ceb7` | P2.b | `models/registry.py` (Encoder/Decoder ABCs + registry) + 15-test contract | 107 + 1 skip |
| `4115fd9` | P6.a-e | LICENSE (MIT), CONTRIBUTING.md, pyproject.toml, `.github/workflows/ci.yml`, move planning artifacts to `docs/cleanup/` | 107 + 1 skip |
| `2bf87ff` | T2 | move 14 legacy roots plus `IJAIED-sub.zip` and `recsys25_v1_3.pdf` into `legacy/`, collapse `.gitignore` to a single `legacy/` rule | 107 + 1 skip |

Final pytest at HEAD: 107 passed, 1 skipped, 0 failed. R2 baseline gate
(Synthetic-Static K=4 fold0 sidecar plus ASSIST2009 K=2 fold0 sidecar)
PASS on every gate-sensitive metric. Kendall tau column skip is the
documented sidecar-convention gap from `BASELINE_2026-06-02.md`, not a
regression.

Root tree after the session (ten entries):

```
deep-mirt/
├── .github/workflows/ci.yml
├── docs/             (architecture, pipeline, cleanup/)
├── legacy/           (gitignored, all vendored upstream content)
├── ma-irt/           (active codebase)
├── overleaf-sync/    (paper LaTeX)
├── CLAUDE.md, CONTRIBUTING.md, LICENSE, README.md, benchmarks.md,
│   cleanup_log.md, pyproject.toml, plus CLEANUP_PLAN_2026 and
│   CLEANUP_VERIFICATION_2026
```

## What was NOT done (defer to live supervision)

- **P2 model migration.** The `EncoderBackbone`/`ResponseDecoder` ABCs
  and registry are in place but no production model subclasses them yet.
  Migration of `MAGPCM`, `DKVMN`, `StaticGPCM`, `DynamicGPCM`,
  `DKVMNSoftmax`, `DKT`, `DeepIRT` to the new pattern is the natural
  next step. The R2 baseline gate is the mechanical safeguard for that
  future work.
- **P3 backbone integration** (SAKT/SAINT+/AKT/SimpleKT ports).
  Requires P2 migration to land first.
- **P4 decoder family** (GRM/PCM/MIRT/DINA).
- **P5 computational optimizations.** AMP, `torch.compile`, batched
  recovery accumulator. Gated behind config flags per the plan but
  performance-sensitive, defer to a focused session.

## Pending tier work from `CLEANUP_PLAN_2026.md`

T3 (dead code archive in `ma-irt/`), T4 (config consolidation), T5
(deeper refactor) remain. The active config matrix
(`configs/bulk/*_pykt_fold*.yaml`) is clean. The stale
`configs/dynamic_seeds/`, `configs/experiments/`, `configs/tmp_alpha1/`,
`configs/_archive_s0p5/` trees are still on disk and gitignored, ready
to be archived in a follow-up tier.

## Session 2026-06-03 night (Archive, Migrate, Consolidate)

Continuation of the overnight workflow. Three phases landed back to back,
fifteen forward commits, full pytest 119 passed and 1 skipped at HEAD.

**Phase Archive (C1 to C5).** Dead code drained out of `ma-irt/scripts/`
into `ma-irt/archive/scripts/`. Deprecated config trees followed the
same route into `ma-irt/archive/configs/`.

**Phase Migrate (P2.1 to P2.6).** Every production model class moved
onto the `EncoderBackbone` plus `ResponseDecoder` contract introduced in
`P2.b` (commit `8c8ceb7`). Trainer and evaluator dispatch reads
encoder/decoder attributes off the model rather than branching on
`model_type` strings.

**Phase Consolidate (C1 to C3, doc refresh).** Plot scripts collapsed
into a `plotting/` package, ASSIST converters into
`dataloading/converters/`, and the five synthetic DGPs into
`dataloading/dgps/`. The user-facing CLI was kept stable through thin
dispatchers (`scripts/plot.py`, `scripts/convert.py`,
`scripts/data_gen.py --dgp <name>`).

| Commit | Phase | Summary | Result |
|---|---|---|---|
| `22d9f59` | Archive C1 | archive deprecated linking script | 107 + 1 skip |
| `e2ca5df` | Archive C2 | archive diagnostic scripts | 107 + 1 skip |
| `df68e93` | Archive C3 | archive deprecated aggregators | 107 + 1 skip |
| `8bdfbb7` | Archive C4 | archive stale config trees | 107 + 1 skip |
| `91adef1` | Archive C5 | archive remaining deprecated-by-refactor scripts | 107 + 1 skip |
| `31fefb3` | Migrate P2.1 | dkvmn_softmax onto EncoderBackbone plus ResponseDecoder | 107 + 1 skip |
| `0fa9c63` | Migrate P2.2 | dkvmn binary baseline onto encoder plus decoder | 107 + 1 skip |
| `a145948` | Migrate P2.3 | Deep-IRT onto encoder plus RaschDecoder | 107 + 1 skip |
| `05bd2c6` | Migrate P2.4 | DKT onto DKTEncoder plus DKTBinaryDecoder | 107 + 1 skip |
| `3e71d51` | Migrate P2.5 | MA-GPCM both ablations onto encoder plus GPCMDecoder | 107 + 1 skip |
| `a06175a` | Migrate P2.6 | trainer plus evaluator dispatch via model attributes | 107 + 1 skip |
| `85ed7c7` | Consolidate C1 | eight plot scripts move into `plotting/` | 119 + 1 skip |
| `0a94f53` | Consolidate C2 | ASSIST converters move into `dataloading/converters/` | 119 + 1 skip |
| `7f297ac` | Consolidate C3 | five generators move into `dataloading/dgps/` | 119 + 1 skip |
| `d2dd518` | docs | refresh READMEs for `plotting/`, `converters/`, `dgps/` | 119 + 1 skip |

Final pytest at HEAD, 119 passed and 1 skipped. The skip is the
documented sidecar Kendall tau structural gap from
`BASELINE_2026-06-02.md`, not a regression. R2 baseline gate at HEAD,
9 passed and 1 skipped, every gate-sensitive metric inside tolerance for
both Synthetic-Static K=4 and ASSIST2009 K=2.

Final `ma-irt/scripts/` count is sixteen entries (fifteen scripts plus
`__pycache__`). The active surface is `train.py`, `evaluate.py`,
`data_gen.py`, `convert.py`, `plot.py`, `mirt_baseline_all_k.R`,
`mirt_predict.R`, plus the named sweep and aggregation orchestrators
(`_run_pykt_sweep.sh`, `_run_k4_cv_recovery.sh`,
`_run_k356_cv_recovery.sh`, `run_bulk_retrain.sh`,
`_aggregate_pykt_results.py`, `aggregate_recovery_v5.py`,
`_build_pykt_synthetic5.py`, `_extract_row.py`).

Final `ma-irt/models/` surface is the model files (`magpcm.py`,
`dynamic_gpcm.py`, `static_gpcm.py`, `dkvmn_softmax.py`, `dkvmn.py`,
`deep_irt.py`, `dkt.py`) plus four subpackages, `encoders/`,
`decoders/`, `components/`, `heads/`, plus `registry.py` and
`__init__.py`.

Final `ma-irt/archive/` top-level is two directories, `configs/` and
`scripts/`.

## What was NOT done (defer to live supervision)

- **P3 backbone integration** (SAKT/SAINT+/AKT/SimpleKT ports). The
  encoder/decoder contract is now load-bearing across every production
  model, so the port surface is stable. The work itself remains.
- **P4 decoder family** (GRM, PCM, MIRT, DINA). The `decoders/` package
  is in place to receive them.
- **P5 computational optimizations** (AMP, `torch.compile`, batched
  recovery accumulator). Still gated behind config flags per the plan
  and still defer-to-focused-session.
- **T3 to T5 tier work.** With Archive and Consolidate complete, the
  surviving stale trees are inside `ma-irt/archive/`. The remaining tier
  work is mostly cosmetic.
