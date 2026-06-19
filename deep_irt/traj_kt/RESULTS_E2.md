# E2 Results: Learning-Curve Rate Recovery on EdNet

## Cohort
- Learners (before T≥200 filter): 15000
- Learners (after  T≥200 filter): 15000
- Median sequence length: 1719
- 90th pct sequence length: 4059
- Max sequence length: 33655
- Item vocabulary size: 12276
- Bundle-collapsed rows: 25.40%

## Training
- N learners: 15000
- r_hat fit success: 77.8%
- r_hat median: 0.0569
- r_hat mean:   0.2101

## Validation

### (a) Predictive validity (primary)
- Spearman rho = -0.017  95% CI [-0.035, 0.001]  n = 11547
- Negative control rho = -0.014  [-0.031, 0.004]

### (b) AFM concurrent validity (primary)
- Spearman rho = -0.013  95% CI [-0.031, 0.006]  n = 11670

#### Part-stratified AFM breakdown
- part_1: rho=-0.011  [-0.029, 0.008]  n=10907
- part_2: rho=-0.007  [-0.024, 0.011]  n=11582
- part_3: rho=-0.011  [-0.029, 0.009]  n=10392
- part_4: rho=0.005  [-0.015, 0.024]  n=9976
- part_5: rho=0.006  [-0.014, 0.025]  n=11667
- part_6: rho=-0.018  [-0.038, 0.002]  n=10158
- part_7: rho=-0.014  [-0.035, 0.007]  n=8189

### (c) Split-half reliability
- Spearman rho = 0.253  95% CI [0.233, 0.273]  n = 10228

### (d) Convergent (aligned vs responsive theta)
- Spearman rho = 0.355  95% CI [0.336, 0.376]  n = 9701

## Runtime
- Wall time: 261.5 min
- Peak VRAM: 0.86 GB

## Verdict

NEGATIVE: recovered rate does not predict future accuracy gains (rho=-0.017); noise floor likely dominates at T=200 cutoff