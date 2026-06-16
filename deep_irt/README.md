# substrate/  --  Anchoring Drift v0

A self-contained PyTorch experiment testing **structural anchoring**: can a
deep ability scale absorb new items without recalibrating, holding existing
estimates fixed at a cost proportional only to the new items?

## What it tests

Phase 1 trains a small LSTM-based ability model on a base item bank (B=100 items,
N=2000 learners, K=4 ordered categories).  Phase 2 freezes the entire model and
fits only E=30 new item embeddings, calibrated against the frozen ability estimates
of the learners who also answered the extension items.  A full-retrain baseline
trains the whole model from scratch on B+E responses.

Four metric groups are reported: new-item recovery, theta consistency, scale
stability, and computational cost.

## How to run

From the repository root, with the `research` conda environment active:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate research
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=substrate python substrate/run_v0.py
```

Outputs are written to `substrate/outputs/`:
- `v0_results.json`  -- all numeric results
- `v0_summary.md`    -- short readable summary table

## Files

| File | Purpose |
|------|---------|
| `data.py` | Synthetic GPCM data generation |
| `model.py` | LSTM encoder + GPCM decoder |
| `train.py` | Phase 1: base calibration |
| `anchor.py` | Phase 2: anchored extension |
| `baseline.py` | Full recalibration baseline |
| `metrics.py` | All four metric groups |
| `run_v0.py` | Orchestrator: data -> P1 -> P2 -> baseline -> metrics |
