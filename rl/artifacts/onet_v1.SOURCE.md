# onet_v1.parquet source provenance

## Release used

`onet_v1.parquet` was built from the O\*NET Database 30.2 text release.
This is the public-domain TSV bundle published on 2026-02-24 at the
O\*NET Resource Center. It is the latest 2024+ cycle release available
as of 2026-06-04.

- Download URL: <https://www.onetcenter.org/dl_files/database/db_30_2_text.zip>
- Discovery page: <https://www.onetcenter.org/database.html>
- License: O\*NET Database is licensed under Creative Commons
  Attribution 4.0 International (CC BY 4.0). Attribution is "Source:
  O\*NET 30.2 Database by U.S. Department of Labor, Employment and
  Training Administration."
- Local archive (transient): unpacked into a temporary directory at
  build time and discarded once the parquet is written.

## How to reproduce

```bash
# 1. Download the release (no manual click-through required).
python rl/scripts/build_onet_pool.py --download \
    --onet-version db_30_2_text \
    --output rl/artifacts/onet_v1.parquet

# 2. Or, if the zip is already cached, point at the extracted folder.
python rl/scripts/build_onet_pool.py \
    --onet-dir /path/to/db_30_2_text \
    --output rl/artifacts/onet_v1.parquet
```

## Source files consumed

| O\*NET file | Used for |
|---|---|
| `Occupation Data.txt` | `occupation_code`, `title`, `description` |
| `Task Statements.txt` | `tasks_concat` (Core then Supplemental, " \| " separated) |
| `Work Activities.txt` | `work_activities_summary` (top 5 by IM score, "; " separated) |
| `Interests.txt` | `riasec_code` (High-Point scale `IH`, falls back to ranking the six `OI` scores) |
| `Job Zones.txt` | `work_zone` (integer 1..5) |
| `Education, Training, and Experience.txt` | `education_zscore` (expected RL category, z-scored across the pool) |

## Build outcome (deterministic)

- Rows in parquet: 923 (occupations with a Job Zone assignment).
- All rows have non-empty `occupation_code`, `title`, `description`,
  `tasks_concat`, `riasec_code`.
- `work_activities_summary` is empty for 29 occupations that lack any
  IM rows in `Work Activities.txt`.
- `education_zscore` is NaN for 45 occupations missing the
  `Required Level of Education` element. Downstream consumers should
  fall back to 0 or impute.

## Manual fallback (only needed if the download is blocked)

If the CDN URL above is unreachable from the build environment,
download `db_30_2_text.zip` manually from
<https://www.onetcenter.org/database.html> (no login required, just
click "Download the O\*NET Database, 30.2"), unzip it, and run the
second `build_onet_pool.py` command with `--onet-dir` pointing at the
extracted folder.
