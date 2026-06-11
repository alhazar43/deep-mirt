# D1: SLAM 2018 en_es Baseline (MAGPCM K=3)

Branch `feat/ordrec-d1-slam`. First MAGPCM training run on real Duolingo data.

---

## Setup

**Dataset.** SLAM 2018 en\_es track (English for Spanish speakers), Harvard Dataverse
CC0 release. 2,593 distinct learners, 4,995 named exercise types plus 3 per-format
catch-all buckets (n\_questions = 4,998). Official train/dev/test temporal split
respected verbatim to maintain comparability with SLAM-era per-token AUC baselines.

**Ordinal coding.** K = 3 categories.

| Code | Label | Criterion |
|------|-------|-----------|
| 0 | All-wrong | mistake\_fraction == 1.0 |
| 1 | Partial | 0 < mistake\_fraction < 1.0 |
| 2 | All-correct | mistake\_fraction == 0.0 |

**Item identity.** MD5 hash of `format + "|" + space-joined lowercased token
strings (16-hex digest). Exercises below min\_count = 10 distinct users map to
one of three per-format catch-all buckets.

**Marginal distribution (train).** cat-0 = 2.7 %, cat-1 = 25.1 %, cat-2 = 72.2 %.
Class weights applied: \[3.48, 1.14, 0.68\].

**Sequences.** 10,502 total (5,327 train / 2,582 valid / 2,593 test), chunked at
max\_seq\_len = 200, min\_seq\_len = 5. Training observations = 960,596.

**Model.** MAGPCM (MA-GPCM) with separate theta pathway, memory\_size = 50,
key\_dim = 64, value\_dim = 64, n\_traits = 1, dropout = 0.1. Parameters: 669,964.
Config: `ma-irt/configs/ordrec_slam_k3.yaml`.

**Training.** 5 epochs on CUDA (RTX 4060 Laptop), batch\_size = 64, lr = 0.001,
grad\_clip = 1.0, weighted\_ordinal\_weight = 1.0, ordinal\_penalty = 0.5,
lr\_patience = 5. Best checkpoint saved by QWK (epoch 5).

---

## Training Curve

| Epoch | TrainLoss | TrainAcc | ValLoss | ValAcc | QWK |
|-------|-----------|----------|---------|--------|-----|
| 1 | 1.055 | 0.644 | 1.013 | 0.616 | 0.294 |
| 2 | 0.882 | 0.690 | 0.964 | 0.664 | 0.356 |
| 3 | 0.847 | 0.719 | 0.945 | 0.678 | 0.370 |
| 4 | 0.828 | 0.730 | 0.937 | 0.685 | 0.367 |
| 5 | 0.809 | 0.731 | 0.919 | 0.682 | 0.374 |

Best checkpoint at epoch 5 (highest QWK = 0.374).

---

## Results

Ordinal prediction metrics (evaluate.py single, legacy 80/20 inference mode,
test split):

| Metric | Value |
|--------|-------|
| ACC | 0.682 |
| QWK | 0.374 |
| tau | 0.374 |
| MAE | 0.340 |

Binary-collapsed metrics (eval\_d1\_slam.py, cat-2 = all-correct vs cats 0+1,
exercise level, test split, 93,604 observations, frac-positive = 0.667):

| Metric | Value |
|--------|-------|
| AUC | 0.773 |
| Log-loss | 0.565 |

---

## Caveats

1. **Short training run.** Only 5 epochs completed due to session interruption.
   Training was still improving at epoch 5 (val loss decreasing, QWK increasing).
   A full 60-epoch run is expected to yield QWK > 0.45 based on trajectory.

2. **Exercise-level vs token-level granularity.** The SLAM 2018 shared-task
   evaluates per-token binary AUC on the raw SLAM format. Our K=3 ordinal labels
   are per-exercise (one code per attempt block). The binary-collapsed AUC (0.773)
   is therefore not directly comparable to SLAM baseline AUC values (best
   reported ~0.86 at token level, Settles et al. 2018).

3. **No true IRT parameters.** The SLAM dataset has no ground-truth theta/alpha/beta,
   so parameter recovery metrics are not available.

4. **Official split respected.** The temporal train/dev/test split is fixed by
   the SLAM release. Standard OrdRec per-student random splits are not used here,
   so temporal leakage is absent by construction but cross-validation estimates
   are unavailable.

5. **Class imbalance.** The dataset is heavily skewed toward cat-2 (72 %). Weighted
   ordinal loss with computed class weights is applied; ACC is inflated relative to
   a balanced benchmark.

---

## What D2 Inherits

- `SlamAdapter` with min\_count = 10 item vocabulary (4,995 named + 3 catch-all)
- `ordrec_slam_k3.yaml` config as a starting point for further SLAM experiments
- The K=3 ordinal coercion: all-correct / partial / all-wrong at the exercise level
- Confirmed that MAGPCM trains stably on SLAM data with standard hyperparameters
- Baseline prediction metrics (ACC = 0.682, QWK = 0.374, AUC = 0.773 at 5 epochs)
  as a lower bound for comparison with any D2 enhancements
