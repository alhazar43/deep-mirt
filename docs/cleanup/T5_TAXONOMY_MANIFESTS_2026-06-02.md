# T5 Taxonomy Manifests - 2026-06-02

This note records verification for Plan v2 T5, Script and config taxonomy.

## Files Changed

- `docs/script_taxonomy.md`
- `docs/config_taxonomy.md`
- `CLEANUP_VERIFICATION_2026.md`

The changes are documentation/manifest-only. No core model, data-loading,
training, evaluation, plotting, or config YAML files were intentionally
edited in this tier.

## What Changed

- Added a script taxonomy for every current entry under `ma-irt/scripts/`.
- Added a config taxonomy with exact current counts by directory and major
  filename pattern.
- Updated `CLEANUP_VERIFICATION_2026.md` to reflect the real smoke config
  set added in T4:
  - `configs/smoke.yaml`
  - `configs/smoke_dkvmn_gpcm.yaml`
  - `configs/smoke_static_gpcm.yaml`
  - `configs/smoke_dynamic_gpcm.yaml`
  - `configs/smoke_dkvmn_softmax.yaml`
  - `configs/smoke_dkt.yaml`
  - `configs/smoke_dkvmn.yaml`
  - `configs/smoke_deep_irt.yaml`

## Inventory Evidence

Current script inventory:

```text
93 script/report files plus one generated __pycache__/ directory
```

Current config inventory:

```text
total_yaml 2294
. 47
_archive_s0p5 125
bulk 1652
dynamic_seeds 160
experiments/ablation 5
experiments/rq1 100
experiments/rq4 60
experiments/rq5 20
tmp_alpha1 125
```

## Coverage Checks

Script taxonomy coverage command:

```powershell
python - << equivalent check:
  for every file under ma-irt/scripts except __pycache__,
  assert the filename appears in docs/script_taxonomy.md
```

Result:

```text
missing_scripts 0
```

Config count check:

```text
total_ok True 2294
. 47 ok
bulk 1652 ok
dynamic_seeds 160 ok
experiments/rq1 100 ok
experiments/rq4 60 ok
experiments/rq5 20 ok
experiments/ablation 5 ok
_archive_s0p5 125 ok
tmp_alpha1 125 ok
```

Smoke-count consistency scan:

```powershell
rg -n "A copy of `smoke|seven smoke|7 configs above|Configs 5..7|K/7" CLEANUP_VERIFICATION_2026.md docs\config_taxonomy.md docs\script_taxonomy.md
```

Result: no matches.

## Unit Tests

Command:

```powershell
cd C:\Users\steph\Documents\deep-mirt\ma-irt
$env:PYTHONPATH='.'
pytest tests -q
```

Result:

```text
65 passed in 2.11s
```
