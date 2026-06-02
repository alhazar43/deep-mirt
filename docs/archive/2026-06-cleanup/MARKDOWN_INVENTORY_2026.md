# Markdown Inventory 2026

Round-2 audit after T0+T1 cleanup (commit `5b8b290`). Goal is a complete pass over every `.md` file under `C:\Users\steph\documents\deep-mirt\`, with one disposition per file. Counts via `wc -l`, modification dates via `stat -c %y` (date only). Sorted by directory group.

Disposition codes,

- **KEEP** is in active use by the user, the paper, or the live code.
- **MOVE** is reference reading that should follow the legacy/ archive rule of T2 once approved (typically a legacy README that nothing in `ma-irt/` imports against).
- **DELETE** is duplicated, superseded, or refers to work that is already in git history.
- **RESEARCH-CRITICAL-CHECK** is uncertain. The research-scientist agent (running in parallel) should confirm or override before T2 execution.

Cross-references, [`CLEANUP_PLAN_2026.md`](CLEANUP_PLAN_2026.md) for the tiered plan and [`CLEANUP_VERIFICATION_2026.md`](CLEANUP_VERIFICATION_2026.md) for the verification protocol. T1 archive landed at `docs/archive/2026-06-cleanup/`.

---

## Group A. Repository root

| Path | Lines | Modified | Disposition | Reason |
|---|---:|---|---|---|
| `CLAUDE.md` | 171 | 2026-05-01 | **KEEP** | Active Claude Code guidance, sole reference for env and commands |
| `README.md` | 87 | 2026-06-02 | **KEEP** | Repo entrypoint, refreshed during T1 |
| `CLEANUP_PLAN_2026.md` | 389 | 2026-06-02 | **KEEP** | T2+ planning document, still authoritative |
| `CLEANUP_VERIFICATION_2026.md` | 611 | 2026-06-02 | **KEEP** | Verification protocol for T2+ |
| `cleanup_log.md` | 21 | 2026-06-02 | **KEEP** | Append-only log per the verification protocol |
| `benchmarks.md` | 117 | 2026-06-02 | **KEEP** | Paper benchmark tables, referenced from `README.md` |
| `phd_research_proposal.md` | 273 | 2026-05-22 | **KEEP** | Active user document, referenced from `README.md` |
| `MARKDOWN_INVENTORY_2026.md` | this file | 2026-06-02 | **KEEP** | This inventory; supersedes nothing |

---

## Group B. ma-irt/

| Path | Lines | Modified | Disposition | Reason |
|---|---:|---|---|---|
| `ma-irt/README.md` | ~500 | 2026-06-02 | **KEEP** | Code-usage guide, just updated in this round |
| `ma-irt/NOTES_linking_appendix.md` | 100 | 2026-04-23 | **KEEP** | Per T1 disposition, source material for an IJAIED appendix |
| `ma-irt/REVIEW_psychometric.md` | 184 | 2026-04-24 | **RESEARCH-CRITICAL-CHECK** | Mid-project reviewer notes, kept in T1 but research-scientist should confirm whether they still feed paper revisions or are stale |
| `ma-irt/REVIEW_research_scientist.md` | 117 | 2026-04-24 | **RESEARCH-CRITICAL-CHECK** | DL/KT reviewer notes, same status as above |
| `ma-irt/REVIEW_converged.md` | 176 | 2026-04-24 | **RESEARCH-CRITICAL-CHECK** | Synthesis of the two REVIEW_ files; if both upstream files are still relevant, keep; if not, DELETE the trio together |
| `ma-irt/scripts/_bench_writeup_draft.md` | 36 | 2026-04-27 | **MOVE** | Replacement-paragraph draft for the binary KT benchmark section; should have been folded into the paper by now. Archive to `docs/archive/2026-06-cleanup/` if no longer load-bearing |
| `ma-irt/scripts/_profile_dkvmn_report.md` | 45 | 2026-04-27 | **DELETE** | One-off CPU profile from 2026-04-27; optimization landed in commit `0fba5e8`, profile is historical |
| `ma-irt/.pytest_cache/README.md` | 8 | 2026-02-18 | **DELETE** | pytest's auto-generated cache stub, regenerated on first test run, should be `.gitignore`-d (gitignore note below) |

Note, `.pytest_cache/` is conventionally git-ignored. Confirm whether the repo's `.gitignore` already covers it before T2 deletion; if it does, this file is already not tracked and the inventory line is informational only.

---

## Group C. docs/archive/2026-06-cleanup/

All items here were archived by T1. They are kept as historical context per the T1 disposition. Listed for completeness so the user sees what was rehomed.

| Path | Lines | Modified | Disposition | Reason |
|---|---:|---|---|---|
| `docs/archive/2026-06-cleanup/CHANGELOG.md` | 69 | 2026-03-29 | **KEEP** | T1 archive, leave in place |
| `docs/archive/2026-06-cleanup/CODE_CHANGES_2026-03-29.md` | 224 | 2026-03-29 | **KEEP** | T1 archive |
| `docs/archive/2026-06-cleanup/PAPER_CHANGES_2026-03-29.md` | 276 | 2026-03-29 | **KEEP** | T1 archive |
| `docs/archive/2026-06-cleanup/PAPER_NOTES.md` | 84 | 2026-04-06 | **KEEP** | T1 archive |
| `docs/archive/2026-06-cleanup/RETRAIN_PLAN.md` | 324 | 2026-04-06 | **KEEP** | T1 archive |
| `docs/archive/2026-06-cleanup/TODO.md` | 124 | 2026-04-06 | **KEEP** | T1 archive |
| `docs/archive/2026-06-cleanup/proxy-ord-mapping.md` | 73 | 2026-04-06 | **KEEP** | T1 archive, referenced by docstrings in `convert_assistments*.py` |
| `docs/archive/2026-06-cleanup/BENCH_OPT_PLAN.md` | 55 | 2026-04-27 | **KEEP** | T1 archive |
| `docs/archive/2026-06-cleanup/BINARY_BENCH_TODO.md` | 65 | 2026-04-27 | **KEEP** | T1 archive |
| `docs/archive/2026-06-cleanup/PYKT_REFACTOR_PLAN.md` | 171 | 2026-04-30 | **KEEP** | T1 archive |
| `docs/archive/2026-06-cleanup/phd_blueprint_d1.md` | 321 | 2026-05-22 | **KEEP** | T1 archive |
| `docs/archive/2026-06-cleanup/phd_blueprint_d1_v2.md` | 223 | 2026-05-22 | **KEEP** | T1 archive |
| `docs/archive/2026-06-cleanup/phd_blueprint_d2.md` | 429 | 2026-05-22 | **KEEP** | T1 archive |
| `docs/archive/2026-06-cleanup/phd_blueprint_d2_v2.md` | 271 | 2026-05-22 | **KEEP** | T1 archive |
| `docs/archive/2026-06-cleanup/phd_blueprint_d3.md` | 377 | 2026-05-22 | **KEEP** | T1 archive |
| `docs/archive/2026-06-cleanup/phd_blueprint_d3_v2.md` | 241 | 2026-05-22 | **KEEP** | T1 archive |
| `docs/archive/2026-06-cleanup/phd_research_blueprint.md` | 585 | 2026-05-22 | **KEEP** | T1 archive |
| `docs/archive/2026-06-cleanup/phd_research_blueprint_v4.md` | 187 | 2026-05-22 | **KEEP** | T1 archive |
| `docs/archive/2026-06-cleanup/ma-irt/CLEANUP.md` | 164 | 2026-03-29 | **KEEP** | T1 archive |
| `docs/archive/2026-06-cleanup/ma-irt/CLEANUP_PLAN.md` | 866 | 2026-03-29 | **KEEP** | T1 archive |
| `docs/archive/2026-06-cleanup/ma-irt/PLAN_sigma05_bulk_retrain.md` | 653 | 2026-04-22 | **KEEP** | T1 archive |
| `docs/archive/2026-06-cleanup/ma-irt/RETRAIN_PLAN.md` | 69 | 2026-03-23 | **KEEP** | T1 archive |
| `docs/archive/2026-06-cleanup/ma-irt/TODO_alternating_optim.md` | 26 | 2026-03-22 | **KEEP** | T1 archive |

---

## Group D. Legacy reference repos at root

These are inactive repos kept for archival per the top-level `README.md`. Their READMEs are the only documentation. The CLEANUP_PLAN_2026 T2 calls for moving the entire repo trees under `legacy/`. Until that move, the READMEs themselves are individually kept in place (the move is a directory-level operation, not a file-level deletion).

| Path | Lines | Modified | Disposition | Reason |
|---|---:|---|---|---|
| `mirt-dkvmn/` | (no `.md` files found) | n/a | n/a | No markdown to inventory in this legacy repo |
| `kt-mirt/.pytest_cache/README.md` | 8 | 2026-03-11 | **DELETE** | pytest auto-generated stub, same as ma-irt's |
| `akt/README.md` | 52 | 2026-04-30 | **MOVE** | Legacy reference repo, candidate for `legacy/akt/` move in T2 |
| `dkvmn-ori/README.md` | 82 | 2026-04-30 | **MOVE** | Legacy reference repo, used by `_build_pykt_synthetic5.py` for data; README is documentation only |
| `deep-1pl/README.md` | 85 | 2026-04-30 | **MOVE** | Legacy reference repo, used by `_convert_yeung_synthetic.py` for data |
| `dkt-ori/README.md` | 19 | 2026-04-30 | **MOVE** | Legacy reference repo |
| `pykt/README.md` | 97 | 2026-04-30 | **MOVE** | Legacy reference repo |
| `pykt/docs/source/contribute.md` | 188 | 2026-04-30 | **MOVE** | pyKT upstream docs, follow `pykt/` parent move |
| `pykt/docs/source/faqs.md` | 4 | 2026-04-30 | **MOVE** | pyKT upstream docs |
| `pykt/docs/source/history.md` | 2 | 2026-04-30 | **MOVE** | pyKT upstream docs |
| `pykt/docs/source/installation.md` | 14 | 2026-04-30 | **MOVE** | pyKT upstream docs |
| `pykt/docs/source/quick_start.md` | 240 | 2026-04-30 | **MOVE** | pyKT upstream docs |
| `pykt/docs/source/quick_start_cn.md` | 346 | 2026-04-30 | **MOVE** | pyKT upstream docs |
| `pykt/examples/competitions/aaai2023_competition/README.md` | 146 | 2026-04-30 | **MOVE** | pyKT upstream example |

`pykt/docs/source/` are upstream pyKT documentation. The user is not maintaining pyKT, so these have zero project-specific value. They ride along with the directory move in T2.

---

## Group E. deep-gpcm/ (legacy repo with its own doc tree)

`deep-gpcm/` is one of the largest legacy reference repos. It has a substantial markdown corpus that was deep-gpcm's own doc tree, not migrated from this project. None of these are imported or referenced by `ma-irt/`.

| Path | Lines | Modified | Disposition | Reason |
|---|---:|---|---|---|
| `deep-gpcm/README.md` | 328 | 2026-02-18 | **MOVE** | Legacy README, follow parent move to `legacy/deep-gpcm/` |
| `deep-gpcm/ARCHITECTURE.md` | 366 | 2026-02-18 | **MOVE** | Legacy architecture doc |
| `deep-gpcm/CHANGELOG.md` | 138 | 2026-02-18 | **MOVE** | Legacy changelog |
| `deep-gpcm/COMPARISON.md` | 185 | 2026-02-18 | **MOVE** | Legacy comparison doc |
| `deep-gpcm/GPCM_VISUALIZATION_GUIDE.md` | 215 | 2026-02-18 | **MOVE** | Legacy guide |
| `deep-gpcm/MATH.md` | 844 | 2026-02-18 | **MOVE** | Legacy math doc, kept as scholarly reference for derivations |
| `deep-gpcm/MODEL_CONFIG.md` | 228 | 2026-02-18 | **MOVE** | Legacy config doc |
| `deep-gpcm/SUMMARY.md` | 219 | 2026-02-18 | **MOVE** | Legacy summary |
| `deep-gpcm/TODO.md` | 335 | 2026-02-18 | **DELETE** | Stale TODO from a legacy project, no value in migration |
| `deep-gpcm/USAGE.md` | 95 | 2026-02-18 | **MOVE** | Legacy usage doc |
| `deep-gpcm/docs/QWK_Agreement_Guide.md` | 189 | 2026-02-18 | **MOVE** | Legacy QWK guide |
| `deep-gpcm/docs/ordinal_suppression_solutions.md` | 196 | 2026-02-18 | **MOVE** | Legacy ordinal-suppression notes |
| `deep-gpcm/results/hyperopt/attn_gpcm_learn_synthetic_500_200_4_hyperopt_report.md` | 129 | 2026-02-18 | **DELETE** | Hyperopt run report from 4 months ago, on a legacy `attn_gpcm` model not in this paper; zero downstream consumers |
| `deep-gpcm/results/hyperopt/attn_gpcm_linear_synthetic_500_200_4_hyperopt_report.md` | 128 | 2026-02-18 | **DELETE** | Same as above |
| `deep-gpcm/results/hyperopt/deep_gpcm_synthetic_500_200_4_enhanced_report.md` | 94 | 2026-02-18 | **DELETE** | Same, on legacy `deep_gpcm` enhanced variant |
| `deep-gpcm/results/hyperopt/deep_gpcm_synthetic_500_200_4_hyperopt_report.md` | 114 | 2026-02-18 | **DELETE** | Same |

The four hyperopt reports could equally **MOVE** with the parent. They are flagged DELETE because they document results for a model variant (`attn_gpcm`) that is not in the paper, are not referenced from anywhere, and are larger-than-it-looks intermediate dumps. If the user wants symmetry with the rest of `deep-gpcm/`, override to MOVE.

---

## Group F. Stray locations

| Path | Lines | Modified | Disposition | Reason |
|---|---:|---|---|---|
| `archive_sigma03_20260422_0534/outputs/recovery_summary_v4.md` | 210 | 2026-04-17 | **DELETE** | Intermediate result dump inside an already-archived sigma=0.3 outputs snapshot; supplanted by current sigma=0.5 results in `ma-irt/outputs/agg/` |
| `overleaf-sync/` | (no `.md` files found) | n/a | n/a | Paper source tree is `.tex` only |
| `_overleaf_old/` | (no `.md` files found) | n/a | n/a | No markdown |
| `elsarticle/` | (no `.md` files found) | n/a | n/a | No markdown |
| `figures/` | (no `.md` files found) | n/a | n/a | No markdown |
| `submission_2026-05-09/` | (no `.md` files found) | n/a | n/a | No markdown |

---

## Summary counts

| Disposition | Count | Notes |
|---|---:|---|
| KEEP | 31 | 8 at repo root, 2 in `ma-irt/` proper, 21 in `docs/archive/2026-06-cleanup/` (T1 archive frozen as-is) |
| MOVE | 17 | All in legacy repos, ride along with parent-directory move in T2 |
| DELETE | 9 | 4 deep-gpcm hyperopt reports + deep-gpcm TODO + 2 pytest cache stubs + 1 sigma-03 result dump + 1 dkvmn profile report |
| RESEARCH-CRITICAL-CHECK | 3 | The three `ma-irt/REVIEW_*.md` files |

---

## Recommended T2 cuts (aggressive)

If the user wants to be more aggressive than T2's directory-move-only posture, the following file-level DELETEs are safe to fold in,

1. `ma-irt/.pytest_cache/README.md` and `kt-mirt/.pytest_cache/README.md`. Auto-generated by pytest. Add `**/.pytest_cache/` to `.gitignore` if not already present.
2. `ma-irt/scripts/_profile_dkvmn_report.md`. The optimization it informed has been merged. The profile data is reproducible from `_profile_dkvmn.py`.
3. `archive_sigma03_20260422_0534/outputs/recovery_summary_v4.md`. The whole `archive_sigma03_*` tree is itself a candidate for removal in a later tier.
4. `deep-gpcm/results/hyperopt/*.md` (4 files). Document a model variant not in the paper. If the user is keeping the deep-gpcm tree for code reference, the hyperopt subdirectory has nothing to offer.
5. `deep-gpcm/TODO.md`. Stale by 4 months and authored against a legacy project.

The three `ma-irt/REVIEW_*.md` files are deliberately not in this list. They are recent (2026-04-24) and were retained in T1 with explicit user authorization. The research-scientist agent should confirm whether the paper revisions they feed are still pending; if all three reviewer-action items have landed in `overleaf-sync/main.tex`, they can be archived to `docs/archive/2026-06-cleanup/`.

---

## What is not in this inventory

By design, the following are not enumerated,

- Lockfiles, requirements files, anything not ending in `.md`. The cleanup plan covers non-markdown classes separately.
- Files inside `outputs/` other than the one explicit `recovery_summary_v4.md` callout. Per-run training output trees do not contain markdown.
- Files inside `data/` and `figures/`. Verified empty of markdown.
- `submission_2026-05-09/` markdown content. Verified empty of markdown.

Anything not listed above and not found by the Glob `**/*.md` scan does not exist in the repo at commit `5b8b290`.
