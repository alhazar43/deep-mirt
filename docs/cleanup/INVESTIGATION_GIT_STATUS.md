# Investigation 3, git history and paper-critical experiment status

Date, 2026-06-03. Repo, `C:\Users\steph\documents\deep-mirt`. Branch, `main`. HEAD, `7793d2f`.

## (a) Recent commits categorized

Last 30 commits classified.

### Cleanup tier work, Plan v2 (2026-06-02)

The current cleanup is run under `CLEANUP_PLAN_2026.md` Plan v2, which redesigned the tier order around the MA-GPCM pipeline contract. The commits land in pipeline order, not the legacy Tx numbering from earlier sweeps.

| Commit | Tier | Subject |
|---|---|---|
| `7793d2f` | T7 readiness | Guard cleanup refactor behind readiness check |
| `37c37c2` | T6 | Add public pipeline reproducibility tests |
| `6bbbda1` | T5 | Add cleanup taxonomy manifests |
| `eaeb7a3` | T4 | Add ordinal baseline smoke configs |
| `24a3cfc` | T3 | Document MA-GPCM architecture contract |
| `f4eb5b7` | T2 | Document MA-GPCM pipeline contract |
| `435e2df` | T1 | Refresh public README entry surface |
| `4d3edbe` | T0 | Record T0 cleanup baseline evidence |
| `c0e5427` | Plan reset | Redesign cleanup plan around MA-GPCM pipeline |

### Cleanup tier work, legacy Tx (2026-06-02, before Plan v2 reset)

| Commit | Tier | Subject |
|---|---|---|
| `6f5c60f` | T2.1 follow-up | Record generated cleanup noise pass |
| `eb41791` | T2.1 | Track review notes and ignore generated cleanup noise |
| `fc4c3ca` | T2.0 follow-up | Record benchmark asset tracking cleanup |
| `410efed` | T2.0 | Track paper-critical benchmark assets |
| `9dd20ae` | T1.7 | Archive phd_research_proposal, distill roadmap into README |
| `9d4f1ff` | T1.6 | Archive MARKDOWN_INVENTORY/CRITICALITY reports |
| `3ffa3e9` | T1.5 | Comprehensive markdown rescan and ma-irt/README rewrite |
| `5b8b290` | T0+T1 record | Add cleanup_log.md, record verification (65/65 pytest, PASS) |
| `e9beb44` | T0+T1 | Build artifacts and stale planning markdowns |

### Codex-related cleanup

No commit message contains the literal token "codex". The only on-disk mentions of "codex" are smoke-fixture dataset names (`cleanup_t0_smoke_codex`, `cleanup_t4_bin_codex`) inside `docs/cleanup/T0_BASELINE_2026-06-02.md` and `docs/cleanup/T4_SMOKE_CONFIGS_2026-06-02.md`. Those are smoke artifact names, not Codex attribution.

The Codex-recommended Category A dead-code pruning landed earlier under Claude co-authorship in April 2026. Two commits match the Codex-style "Category A" cleanup pattern.

| Commit | Date | Subject | Co-author |
|---|---|---|---|
| `614f8c7` | 2026-04-26 | Prune dead code (Category A) | Claude Opus 4.7 (1M context) |
| `436889f` | 2026-04-26 | Prune dead alpha path in IRTParameterExtractor | (no co-author tag) |

Both commits remove unreachable methods in `models/components/{memory,irt}.py` and `dkvmn_softmax.py` after grep-verified call-graph analysis. They predate the 2026-06-02 cleanup tiers and are already on `main`.

### Pre-cleanup paper work

| Commit | Subject |
|---|---|
| `40f26f1` | Rewrite PhD proposal, narrative, multidim MA-IRT prereq, four directions |
| `0fba5e8` | Add PhD research proposal |
| `73cb344` | Fix CV student-ID remap on Static GPCM MLE theta path |
| `09c7085` | DKVMN bench speedups, hoist attention in DKVMN+Softmax, DataLoader knobs |
| `9653246` | Trim model docstrings, remove false "ordered thresholds" claims |
| `e6b6b9d` | ASSISTments retrained under StaticItem, remove learned configs/outputs |
| `ab852b2` | Restructure ability-tracking narrative with two-signature thread |
| `27e34a7` | Imbalance QWK bold, explicit minimal-gap framing for scalability |
| `c2f150f` | Scalability/imbalance tables refreshed from ablation batch, prose reframed |
| `dab1083` | Item-params suptitle drops embedding label |

### Other

None.

## (b) Cleanup tier status

Two parallel tier streams exist. The legacy Tx tiers in `cleanup_log.md` and the Plan v2 tiers in `CLEANUP_PLAN_2026.md`.

### Plan v2 status (current, from `CLEANUP_PLAN_2026.md` and `docs/cleanup/`)

| Tier | Scope | Status |
|---|---|---|
| T0 | Baseline guardrails, capture smoke command sequence | PASS (`4d3edbe`, evidence `T0_BASELINE_2026-06-02.md`, 65/65 pytest) |
| T1 | Public entry surface, README rewrite | PASS (`435e2df`, evidence `T1_README_ENTRY_SURFACE_2026-06-02.md`) |
| T2 | Scientific pipeline documentation, `docs/pipeline.md` | PASS (`f4eb5b7`, evidence `T2_PIPELINE_DOC_2026-06-02.md`) |
| T3 | Architecture documentation, `docs/architecture.md` | PASS (`24a3cfc`, evidence `T3_ARCHITECTURE_DOC_2026-06-02.md`) |
| T4 | Smoke configs across all model types | PASS (`eaeb7a3`, evidence `T4_SMOKE_CONFIGS_2026-06-02.md`) |
| T5 | Script/config taxonomy manifests | PASS (`6bbbda1`, evidence `T5_TAXONOMY_MANIFESTS_2026-06-02.md`) |
| T6 | Reproducibility tests, `test_public_pipeline.py` | PASS for public pipeline path (`37c37c2`, evidence `T6_REPRODUCIBILITY_TESTS_2026-06-02.md`). Model output contract tests and root-level CLI ergonomics remain pending |
| T7 | Architecture refactor readiness | DEFERRED (`7793d2f`, evidence `T7_REFACTOR_READINESS_2026-06-02.md`). Guardrail flags uncommitted edits in `train.py`, `config/loader.py`, `config/types.py`, `models/__init__.py`, `dataloading/loaders.py`, `models/components/irt.py`, `utils/metrics.py` that must be reconciled before cleanup-owned refactor commits |
| T8 | Artifact hygiene and archival cleanup | PENDING |
| T9 | Large legacy and dead-code moves | PENDING |

### Legacy Tx status (from `cleanup_log.md`, superseded by Plan v2 but evidence rows remain)

| Tier | Scope | Status |
|---|---|---|
| T0 | Build artifacts (Appendix A inventory) | PASS, 65/65 pytest |
| T1 | Stale planning markdowns (Appendix B) | PASS |
| T1.5 | Comprehensive markdown rescan, README rewrite | PASS |
| T1.6 | Archive inventory reports | PASS |
| T1.7 | Archive phd_research_proposal | PASS |
| T2.0 | Track paper-critical benchmark assets and `.gitignore` updates | PASS |
| T2.1 | Track review notes, ignore generated cleanup noise | PASS |
| T2 (root legacy repos to `legacy/`) | Move `mirt-dkvmn/`, `dkt-ori/`, `akt/`, `pykt/`, `_overleaf_old/`, `figures/`, `archive_sigma03_*` | PENDING |
| T3 (dead-code archive in `ma-irt/`) | PENDING (gated by T7 readiness) |
| T4 (config consolidation) | PENDING |
| T5 (deeper refactor) | PENDING |

T2 root legacy repos remain at the repo root. `git status` confirms `akt`, `deep-1pl`, `deep-gpcm`, `dkt-ori`, `dkvmn-ori`, `mirt-dkvmn`, `pykt`, `archive_sigma03_20260422_0534`, `_overleaf_old`, top-level `figures/` are still in place.

## (c) Surviving checkpoints in `ma-irt/outputs/`

1509 `best.pt` files survive. The paper-critical subset for the verification gate.

### Table 3 (and Table 1), Synthetic-Static K=4, MA-GPCM recovery (`bench_<model>_static_q200_k4_pykt_fold{0..4}`)

All five folds present with full artifact set. Each has `best.pt`, `last.pt`, `metrics.csv`, `test_metrics.json`, `recovery_metrics.json`.

| Output dir | Config | Paper row | Artifacts |
|---|---|---|---|
| `bench_magpcm_static_q200_k4_pykt_fold0` | `configs/bulk/bench_magpcm_static_q200_k4_pykt_fold0.yaml` | Table 1 + 3, MA-GPCM K=4 | best.pt, last.pt, test_metrics.json, recovery_metrics.json |
| `bench_magpcm_static_q200_k4_pykt_fold1` | `bench_magpcm_static_q200_k4_pykt_fold1.yaml` | same | full |
| `bench_magpcm_static_q200_k4_pykt_fold2` | `bench_magpcm_static_q200_k4_pykt_fold2.yaml` | same | full |
| `bench_magpcm_static_q200_k4_pykt_fold3` | `bench_magpcm_static_q200_k4_pykt_fold3.yaml` | same | full |
| `bench_magpcm_static_q200_k4_pykt_fold4` | `bench_magpcm_static_q200_k4_pykt_fold4.yaml` | same | full |

Sample fold0 numbers, QWK 0.6804, r_alpha 0.887, r_beta_mean 0.966, r_theta 0.956. Compare to `benchmarks.md` cells QWK 0.681 +/- 0.001, r_alpha 0.894 +/- 0.009, r_beta 0.967 +/- 0.002, r_theta 0.957 +/- 0.001. All within the Section 2.2 acceptance intervals.

Sibling K=4 static folds for the four other ordinal models (Static GPCM, Dynamic GPCM, DKVMN+Softmax, DKVMN+GPCM) also have full fold0..fold4 directories under `bench_<model>_static_q200_k4_pykt_fold*`. Table 3 has K=4 coverage across all five models.

### Table 3 / Table 1 other K values

| K | MA-GPCM folds present |
|---|---|
| K=2 | `bench_magpcm_static_q200_k2_pykt_fold0..4` (Table 2 row) |
| K=3 | `bench_magpcm_static_q200_k3_pykt_fold0..4` |
| K=4 | `bench_magpcm_static_q200_k4_pykt_fold0..4` |
| K=5 | `bench_magpcm_static_q200_k5_pykt_fold0..4` |
| K=6 | `bench_magpcm_static_q200_k6_pykt_fold0..4` |

Full K = {2, 3, 4, 5, 6} x fold = {0..4} coverage for MA-GPCM.

### Table 2, ASSIST2009 binary K=2, MA-GPCM (`bench_magpcm_assist2009_bin_pykt_fold{0..4}`)

All five folds present. Each has `best.pt`, `last.pt`, `test_metrics.json`. No `recovery_metrics.json`, expected since `assist2009_bin` has no `true_irt_parameters.json`.

Sample fold0, AUC 0.8369. `benchmarks.md` cell, 83.50 +/- 0.23. Within the Section 2.2 widened interval [83.03, 83.97].

### Baseline phase readiness verdict

- MA-GPCM K=4 Synthetic-Static recovery, Table 3, gate "at least 3 surviving checkpoints", SATISFIED (5 of 5 folds present with `recovery_metrics.json`).
- MA-GPCM K=2 ASSIST2009 binary, Table 2, gate "at least 1 surviving checkpoint", SATISFIED (5 of 5 folds present with `test_metrics.json`).

The <0.5% tolerance verification can run immediately on the cached `recovery_metrics.json` and `test_metrics.json` without retraining.

### Caveat

Checkpoint loadability after Plan v2 T7 lands is uncertain. The T7 readiness note flags pre-existing uncommitted edits in `models/components/irt.py`, `models/__init__.py`, and `config/types.py`. If those edits change `IRTParameterExtractor` state-dict keys (the `monotonic_betas` branch in `CLEANUP_VERIFICATION_2026.md` Section 4.3 is the specific risk), existing `best.pt` files trained before the change may fail to load. Cached `test_metrics.json` and `recovery_metrics.json` files remain valid as published evidence regardless, since they were written at training time.

## (d) Pending active work

The user's task list references task #88 (K=4 5-fold CV recovery batch, in_progress) and task #82 (Aggregate results, pending). I do not have direct read access to that task list from inside this repo, but the on-disk evidence is consistent.

- The K=4 5-fold CV recovery output set is complete on disk for all five models in Table 3 (`bench_{static_gpcm, dynamic_gpcm, dkvmn_softmax, dkvmn_gpcm, magpcm}_static_q200_k4_pykt_fold{0..4}`). If task #88 is the run that produced these, it is artifact-complete. If task #88 is a re-run under refactored code, it has not yet started and would be gated by T7 readiness.
- Task #82 aggregation feeds Table 3 from `recovery_metrics.json`. The aggregator is `scripts/aggregate_recovery_v5.py` (kept under `CLEANUP_VERIFICATION_2026.md` Section 3.2). All input files exist.

Refactor risk to active work.

- T7 (architecture refactor readiness) is held back behind a guardrail. No code-touching cleanup commits have landed since `7793d2f`. The pre-existing uncommitted edits in `train.py`, `config/loader.py`, `config/types.py`, `models/__init__.py`, `dataloading/loaders.py`, `models/components/irt.py`, `utils/metrics.py` are user-owned and not yet committed. Any retraining launched right now will pick up those uncommitted edits and may not match the cached `outputs/bench_*` numbers exactly.
- T8 (artifact hygiene) explicitly avoids deleting `ma-irt/outputs/` wholesale. Cached `best.pt` and metrics JSONs are protected.
- T2 root legacy repo moves are unrelated to task #88 and #82, since the K=4 CV pipeline does not touch `akt/`, `pykt/`, etc.

Net, the baseline phase can verify the <0.5% gate immediately from cached metrics without depending on the deferred T7 refactor.
