# T3 Architecture Documentation - 2026-06-02

This note records verification for Plan v2 T3, Architecture documentation.

## Files Changed

- `docs/architecture.md`

The change is documentation-only. No core model, training, data-loading,
evaluation, plotting, or config files were intentionally edited in this tier.

## What Changed

`docs/architecture.md` now documents:

- MA-GPCM input and output tensor contracts.
- DKVMN encoder flow: item query embedding, value embedding, attention,
  memory read, and memory write.
- The causal read-before-write convention.
- The separated ability pathway used when `separate_theta: true`.
- The DKVMN+GPCM shared-path ablation used when `separate_theta: false`.
- GPCM parameter extraction and logits/probabilities.
- Which model families have meaningful IRT recovery semantics.
- Cleanup constraints for performance-sensitive core files.

## Reference Verification

Command:

```powershell
$paths = @(
  'ma-irt/models/magpcm.py',
  'ma-irt/models/components/memory.py',
  'ma-irt/models/components/irt.py',
  'ma-irt/models/heads/gpcm.py',
  'ma-irt/scripts/train.py',
  'ma-irt/scripts/evaluate.py'
)
foreach ($p in $paths) {
  if (-not (Test-Path $p)) { Write-Output "MISSING $p" } else { Write-Output "OK $p" }
}
```

Result:

```text
OK ma-irt/models/magpcm.py
OK ma-irt/models/components/memory.py
OK ma-irt/models/components/irt.py
OK ma-irt/models/heads/gpcm.py
OK ma-irt/scripts/train.py
OK ma-irt/scripts/evaluate.py
```

Stale/mojibake reference scan:

```powershell
rg -n "鈥|鈹|胃|伪|尾|卤|脳|run_all_experiments|plot_recovery.py|plot_recovery_figure.py" docs\architecture.md docs\pipeline.md README.md ma-irt\README.md
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
65 passed in 2.23s
```
