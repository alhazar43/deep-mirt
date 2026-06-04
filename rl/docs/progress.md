# DRL-MAIRT progress log

Running notes from the autonomous build sessions. Newest entry on
top. One entry per milestone deliverable, dated, with the commit hash
when the work is checked in.

## 2026-06-04, M0 step 1, O\*NET 2024 pool acquired

Owner. Codex autonomous agent.
Branch. `main`.

What landed.
- `rl/scripts/build_onet_pool.py`. Downloads or accepts a local copy of
  the O\*NET text release and emits the per-occupation parquet matching
  plan Section 5.2. Each column has an explicit derivation rule
  documented in the file.
- `rl/artifacts/onet_v1.parquet`. 923 occupations from O\*NET 30.2 (the
  latest 2026-02-24 release on the public CDN). All rows have
  non-empty `occupation_code`, `title`, `description`, `tasks_concat`,
  `riasec_code`, and a valid `work_zone in {2, 3, 4, 5}`. The 1 zone
  is absent from the upstream release.
- `rl/artifacts/onet_v1.SOURCE.md`. Records the exact upstream URL,
  release version, license, the per-column source files, and the
  manual reproduction recipe.
- `rl/tests/test_onet_pool.py`. Nine modular tests covering schema,
  null guarantees, work zone range, occupation_code uniqueness, RIASEC
  well-formedness, education z-score centring, distribution
  non-degeneracy, plus a synthetic-tree round trip of the loader.

Work zone distribution in the parquet.

| zone | count |
|---|---|
| 2 | 331 |
| 3 | 213 |
| 4 | 225 |
| 5 | 154 |
| total | 923 |

Notes for downstream.
- `work_activities_summary` is empty for 29 occupations that have no
  IM rows in `Work Activities.txt`. Treat as a missing optional field.
- `education_zscore` is NaN for 45 occupations with no
  `Required Level of Education` element. The synthetic preference
  generator should impute zero or drop those rows when used as the
  `delta_j` source under sensitivity analysis.
- The CDN URL pattern `db_{maj}_{min}_text.zip` worked for every
  version in [28.3, 30.2]. v30.2 was selected as the latest.

Next deliverables (still on M0).
- None. M0 is complete.

Open risks logged.
- R13 from plan Section 14, the sensitivity of headline numbers to the
  `delta_j` source. The pool now carries three candidates implicitly
  (`work_zone`, `education_zscore`, and a future
  `complexity_composite`). Build-time only writes the first two; the
  composite gets computed in M6.
