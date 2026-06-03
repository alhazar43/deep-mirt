# Benchmarks

Tables reproducing the headline results from the MA-GPCM paper (manuscript under review at IJAIED 2026). Each row reports mean $\pm$ standard deviation across seeds. **Bold** marks the best result in a column (ties bolded). _Italics_ mark the GPCM (EM) offline ceiling.

ACC is reported as a percentage. AUC, QWK, Kendall's $\tau$, MAE, Pearson $r$, RMSE, bias in their natural units.

## 1. Ordinal prediction on Synthetic-Static

Five-seed runs at $K \in \{3, 4, 5, 6\}$, $Q = 200$ items, 5000 students. GPCM (EM) is omitted because it scores items using the complete response pattern and is therefore reconstruction, not sequential prediction.

### $K = 3$

| Model | ACC | QWK | $\tau$ | MAE |
|---|---|---|---|---|
| Static GPCM | 47.80 ± 0.20 | 0.297 ± .005 | 0.269 ± .005 | 0.708 ± .005 |
| Dynamic GPCM | 58.20 ± 0.30 | 0.549 ± .001 | 0.498 ± .001 | 0.497 ± .001 |
| DKVMN+Softmax | **60.20** ± 0.10 | 0.564 ± .002 | 0.517 ± .002 | 0.490 ± .002 |
| DKVMN+GPCM | 59.70 ± 0.10 | **0.590** ± .001 | 0.536 ± .001 | 0.465 ± .001 |
| **MA-GPCM** | 59.60 ± 0.20 | **0.590** ± .001 | **0.537** ± .001 | **0.464** ± .001 |

### $K = 4$

| Model | ACC | QWK | $\tau$ | MAE |
|---|---|---|---|---|
| Static GPCM | 39.20 ± 0.10 | 0.305 ± .000 | 0.263 ± .001 | 1.050 ± .001 |
| Dynamic GPCM | 50.40 ± 0.20 | 0.628 ± .001 | 0.543 ± .001 | 0.667 ± .001 |
| DKVMN+Softmax | **53.60** ± 0.10 | 0.647 ± .001 | 0.571 ± .001 | 0.657 ± .002 |
| DKVMN+GPCM | 52.70 ± 0.20 | 0.680 ± .000 | **0.592** ± .001 | 0.609 ± .002 |
| **MA-GPCM** | 52.80 ± 0.10 | **0.681** ± .001 | **0.592** ± .000 | **0.607** ± .001 |

### $K = 5$

| Model | ACC | QWK | $\tau$ | MAE |
|---|---|---|---|---|
| Static GPCM | 37.40 ± 0.10 | 0.372 ± .001 | 0.316 ± .001 | 1.300 ± .003 |
| Dynamic GPCM | 47.90 ± 0.60 | 0.702 ± .001 | 0.597 ± .002 | 0.789 ± .001 |
| DKVMN+Softmax | **51.20** ± 0.10 | 0.716 ± .002 | 0.624 ± .001 | 0.779 ± .004 |
| DKVMN+GPCM | 50.00 ± 0.10 | 0.751 ± .001 | 0.644 ± .001 | 0.711 ± .002 |
| **MA-GPCM** | 50.00 ± 0.20 | **0.754** ± .000 | **0.646** ± .001 | **0.706** ± .001 |

### $K = 6$

| Model | ACC | QWK | $\tau$ | MAE |
|---|---|---|---|---|
| Static GPCM | 33.60 ± 0.00 | 0.366 ± .002 | 0.309 ± .001 | 1.636 ± .005 |
| Dynamic GPCM | 45.20 ± 0.50 | 0.736 ± .001 | 0.622 ± .002 | 0.917 ± .001 |
| DKVMN+Softmax | **48.40** ± 0.10 | 0.753 ± .002 | 0.648 ± .002 | 0.903 ± .004 |
| DKVMN+GPCM | 47.30 ± 0.30 | 0.790 ± .001 | 0.672 ± .001 | 0.810 ± .002 |
| **MA-GPCM** | 47.20 ± 0.10 | **0.793** ± .001 | **0.674** ± .001 | **0.804** ± .002 |

## 2. Binary prediction ($K = 2$), five-fold CV

Synthetic-Static, Synthetic-5 (five dataset versions), ASSIST2009, ASSIST2017.

| Model | Synthetic-Static ACC | Synthetic-Static AUC | Synthetic-5 ACC | Synthetic-5 AUC | ASSIST2009 ACC | ASSIST2009 AUC | ASSIST2017 ACC | ASSIST2017 AUC |
|---|---|---|---|---|---|---|---|---|
| DKT | 69.91 ± 0.10 | 77.35 ± 0.09 | 74.50 ± 0.22 | 82.00 ± 0.33 | **78.21** ± 0.04 | **83.70** ± 0.14 | **68.53** ± 0.06 | **71.79** ± 0.12 |
| DKVMN | **70.61** ± 0.07 | **78.31** ± 0.07 | **75.22** ± 0.34 | **82.99** ± 0.39 | 78.02 ± 0.03 | 83.19 ± 0.07 | 67.64 ± 0.04 | 69.92 ± 0.17 |
| Deep-IRT | 70.52 ± 0.04 | 78.28 ± 0.03 | 75.19 ± 0.36 | 82.97 ± 0.41 | 78.03 ± 0.04 | 83.09 ± 0.29 | 67.72 ± 0.08 | 69.91 ± 0.17 |
| DKVMN+GPCM | 70.60 ± 0.06 | 78.18 ± 0.05 | 74.88 ± 0.41 | 82.93 ± 0.41 | 77.13 ± 0.22 | 83.47 ± 0.32 | 66.15 ± 0.27 | 69.91 ± 0.14 |
| **MA-GPCM** | 70.55 ± 0.06 | 78.11 ± 0.03 | 75.01 ± 0.33 | 82.90 ± 0.35 | 77.20 ± 0.21 | 83.50 ± 0.23 | 65.51 ± 0.41 | 69.19 ± 0.32 |

DKT leads on ASSIST2009 and ASSIST2017 because repeated attempts at the same item violate attempt-invariant item parameters and favor a purely sequential view. On the synthetic datasets where each item is administered at most once per student, the DKVMN-based models lead.

## 3. IRT parameter recovery on Synthetic-Static

Recovery of $\alpha$, $\beta$, $\theta$ against ground truth at $K \in \{3, 4, 5, 6\}$. Linking applied (log-space z-score with target std 0.3 for $\alpha$, z-score for $\beta$).

### $K = 3$

| Model | $r_\alpha$ | $\text{bias}_\alpha$ | $r_\beta$ | $\text{bias}_\beta$ | $r_\theta$ | $\text{RMSE}_\theta$ | $\text{bias}_\theta$ |
|---|---|---|---|---|---|---|---|
| _GPCM (EM)_ | _0.992_ | _-0.00_ | _0.991_ | _0.01_ | _0.971_ | _0.24_ | _0.01_ |
| Static GPCM | 0.540 ± .085 | -0.68 ± .01 | 0.959 ± .001 | -0.03 ± .01 | **0.960** ± .001 | 0.90 ± .03 | -0.13 ± .01 |
| Dynamic GPCM | **0.866** ± .006 | **-0.26** ± .00 | 0.965 ± .005 | -0.06 ± .00 | 0.918 ± .002 | 0.62 ± .01 | -0.02 ± .01 |
| DKVMN+GPCM | 0.822 ± .026 | -0.66 ± .03 | 0.697 ± .023 | -0.06 ± .00 | 0.917 ± .001 | 1.48 ± .04 | **-0.00** ± .03 |
| **MA-GPCM** | 0.841 ± .016 | -0.55 ± .03 | **0.966** ± .001 | **0.00** ± .01 | 0.940 ± .001 | **0.55** ± .03 | 0.06 ± .05 |

### $K = 4$

| Model | $r_\alpha$ | $\text{bias}_\alpha$ | $r_\beta$ | $\text{bias}_\beta$ | $r_\theta$ | $\text{RMSE}_\theta$ | $\text{bias}_\theta$ |
|---|---|---|---|---|---|---|---|
| _GPCM (EM)_ | _0.991_ | _-0.00_ | _0.990_ | _0.01_ | _0.980_ | _0.20_ | _0.00_ |
| Static GPCM | 0.447 ± .028 | -0.75 ± .00 | 0.947 ± .002 | -0.08 ± .02 | **0.968** ± .000 | 0.98 ± .01 | -0.18 ± .02 |
| Dynamic GPCM | 0.842 ± .007 | **-0.25** ± .00 | 0.964 ± .001 | **-0.08** ± .00 | 0.936 ± .001 | 0.64 ± .01 | **0.00** ± .01 |
| DKVMN+GPCM | 0.880 ± .013 | -0.53 ± .04 | 0.631 ± .015 | -0.10 ± .00 | 0.938 ± .001 | 1.03 ± .02 | -0.08 ± .02 |
| **MA-GPCM** | **0.894** ± .009 | -0.39 ± .03 | **0.967** ± .002 | -0.10 ± .00 | 0.957 ± .001 | **0.47** ± .01 | -0.05 ± .02 |

### $K = 5$

| Model | $r_\alpha$ | $\text{bias}_\alpha$ | $r_\beta$ | $\text{bias}_\beta$ | $r_\theta$ | $\text{RMSE}_\theta$ | $\text{bias}_\theta$ |
|---|---|---|---|---|---|---|---|
| _GPCM (EM)_ | _0.986_ | _0.01_ | _0.986_ | _0.01_ | _0.986_ | _0.17_ | _0.00_ |
| Static GPCM | 0.499 ± .013 | -0.76 ± .00 | 0.950 ± .001 | **-0.06** ± .01 | **0.973** ± .000 | 0.85 ± .01 | -0.14 ± .01 |
| Dynamic GPCM | 0.785 ± .011 | **-0.26** ± .01 | **0.973** ± .003 | -0.07 ± .00 | 0.946 ± .003 | 0.62 ± .01 | **-0.03** ± .01 |
| DKVMN+GPCM | 0.894 ± .008 | -0.43 ± .04 | 0.421 ± .018 | -0.08 ± .00 | 0.944 ± .001 | 1.02 ± .02 | -0.04 ± .03 |
| **MA-GPCM** | **0.906** ± .009 | -0.31 ± .03 | 0.972 ± .002 | -0.07 ± .00 | 0.965 ± .001 | **0.46** ± .01 | -0.03 ± .01 |

### $K = 6$

| Model | $r_\alpha$ | $\text{bias}_\alpha$ | $r_\beta$ | $\text{bias}_\beta$ | $r_\theta$ | $\text{RMSE}_\theta$ | $\text{bias}_\theta$ |
|---|---|---|---|---|---|---|---|
| _GPCM (EM)_ | _0.988_ | _0.00_ | _0.985_ | _0.01_ | _0.987_ | _0.16_ | _-0.00_ |
| Static GPCM | 0.380 ± .021 | -0.79 ± .00 | 0.946 ± .002 | -0.08 ± .01 | **0.978** ± .000 | 0.79 ± .01 | -0.19 ± .01 |
| Dynamic GPCM | 0.789 ± .008 | -0.27 ± .01 | 0.973 ± .001 | **-0.05** ± .00 | 0.949 ± .002 | 0.64 ± .01 | **-0.02** ± .00 |
| DKVMN+GPCM | **0.906** ± .012 | -0.35 ± .05 | 0.559 ± .021 | -0.10 ± .00 | 0.955 ± .003 | 0.94 ± .03 | -0.08 ± .03 |
| **MA-GPCM** | 0.884 ± .010 | **-0.21** ± .04 | **0.975** ± .001 | -0.09 ± .00 | 0.972 ± .000 | **0.50** ± .01 | -0.06 ± .02 |

## How to reproduce

See [`ma-irt/README.md`](ma-irt/README.md) and the per-model configs in `ma-irt/configs/`. The active sweep orchestrators in `ma-irt/scripts/` are:

- `run_bulk_retrain.sh`, iterates `configs/bulk/` for the ordinal Synthetic-Static and dynamic-DGP experiments (Tables 1 and 3 here)
- `_run_pykt_sweep.sh`, the 240-run binary plus ordinal-ASSIST 5-fold CV sweep (Table 2 here)
- `_run_k4_cv_recovery.sh` and `_run_k356_cv_recovery.sh`, the ordinal Synthetic-Static and dynamic-DGP CV sweeps

Each table is aggregated by `_aggregate_pykt_results.py` (Table 2, fold-based protocol). The older seed-based aggregator `_aggregate_bench.py` has been moved to `ma-irt/archive/scripts/` and is no longer on the headline reproduction path.
