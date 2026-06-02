# Cleanup Log

Per `CLEANUP_VERIFICATION_2026.md` Section 5. One row per tier applied.

| Date | Tier | Before (git ref) | After (git ref) | Layer 1 smoke | Layer 2 regression | Layer 3 full | Pass/Fail |
|---|---|---|---|---|---|---|---|
| 2026-06-02 | T0 (build artifacts) | 40f26f1 | e9beb44 | 65/65 pytest in 7.53s | N/A (no source/config touched) | N/A | **PASS** |
| 2026-06-02 | T1 (stale markdowns) | 40f26f1 | e9beb44 | 65/65 pytest in 7.53s | N/A (move-only, no source/config touched) | N/A | **PASS** |

T0 and T1 share commit `e9beb44` because the changes are independent of source/config and were combined in one push. Each subsequent tier (T2 onward) gets its own row and its own commit.

## Notes

- T0 deletions were all gitignored or untracked, so no `git rm` was required. `__pycache__/` and `*.pyc` are regenerated on first import; LaTeX droppings have no source; sweep stdout dumps are not referenced by any reporting script.
- T1 archives went to `docs/archive/2026-06-cleanup/`. KEEP list (in place): `ma-irt/NOTES_linking_appendix.md`, `ma-irt/REVIEW_*.md`, `phd_research_proposal.md`, `README.md`, `CLAUDE.md`, `benchmarks.md`.
- Two docstring references in `ma-irt/scripts/convert_assistments{,_2009}.py` were updated to point at the new location of `proxy-ord-mapping.md`.
- `README.md` and `benchmarks.md` were updated to drop the `run_all_experiments.sh` reference (does not exist) and point at the real orchestrators (`run_bulk_retrain.sh`, `_run_pykt_sweep.sh`, `_run_k4_cv_recovery.sh`, `_run_k356_cv_recovery.sh`).

## Pending

T2 (legacy reference repos at root), T3 (dead code in `ma-irt/`), T4 (config consolidation), T5 (deeper refactor). Each requires user authorization.
