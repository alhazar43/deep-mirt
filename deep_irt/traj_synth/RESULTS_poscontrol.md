# Positive Control: E2/E2b Predictive-Validity Metric

**Purpose.** On real data (E2 EdNet, E2b ASSISTments) the test
Spearman(r_hat_firsthalf, delta_acc = late_acc - early_acc) returned
null to negative (-0.04, -0.08). This positive control checks whether
the metric itself is valid when a true learning rate is present, using
synthetic data with known ground-truth rates.

**Suspected confound.** A fast learner saturates in the first half of
the sequence. Their late-half accuracy change (delta_acc) is small or
negative because the curve has already flattened. A slow learner keeps
improving into the second half, yielding large delta_acc. This makes
rate NEGATIVELY correlated with delta_acc by construction, regardless
of recovery quality.

---

## Setup

| Parameter | Value |
|---|---|
| n_items | 120 |
| n_respondents | 600 |
| seq_len | 80 |
| K (GPCM categories) | 4 |
| split midpoint | 40 |
| n_epochs | 300 |
| model | DeepIRTModel(gpcm, lstm, decouple=True) |
| seed | 0 |

---

## Results

All correlations are Spearman rho with bootstrap 95% CI (1000 resamples).

### A. Recovery (clean target)

`corr(r_hat, r_true)` -- does the model recover the true rate over the
full sequence?

**rho = +0.464  [+0.394, +0.529]  (n=600)**

Expected ~0.4 from prior E0 runs. A positive result here confirms that
the recovery machinery works on this synthetic dataset.

### B. The E2/E2b Predictive-Validity Metric

`corr(r_hat_firsthalf, delta_acc)` -- does the first-half recovered rate
predict late-half accuracy gain, exactly as E2/E2b tested?

**rho = -0.260  [-0.335, -0.180]  (n=600)**

Permuted-order control (shuffled r_hat): -0.033  [-0.113, +0.050]

If this is near zero or negative despite recovery working (A positive),
the metric is confounded.

### C. Metric Validity Check (the crux)

`corr(r_true, delta_acc)` -- does the GROUND-TRUTH rate predict late-half
accuracy gain?

**rho = -0.377  [-0.439, -0.301]  (n=600)**

This is the decisive test. If even the true rate does not predict
delta_acc, the predictive-validity metric is ill-posed regardless of
recovery quality.

### D. Tercile Diagnostic (saturation effect)

Learners binned by r_true tercile. Fast learners should show smaller
delta_acc (saturation) and larger total_gain.

| group              |    n | r_true mean | delta_acc mean | total_gain mean |
|:-------------------|-----:|------------:|---------------:|----------------:|
| slow (low r)       |  200 |       0.063 |         0.1045 |          0.2325 |
| mid                |  200 |       0.154 |         0.0622 |          0.1938 |
| fast (high r)      |  200 |       0.386 |         0.0298 |          0.1077 |

### E. Alternative Criteria

E1 tests recovery in the identifiable regime (slow to mid learners where
E0 showed the best recovery). E2 and E3 use total_acc_gain (last-10-step
mean minus first-10-step mean) as a whole-sequence alternative, but see
the note below.

| Criterion | rho | 95% CI | n |
|---|---|---|---|
| E1: corr(r_hat, r_true) [low/mid rate learners] | +0.254 | [+0.141, +0.365] | 300 |
| E2: corr(r_hat, total_acc_gain) | -0.479 | [-0.536, -0.413] | 600 |
| E3: corr(r_true, total_acc_gain) [sanity] | -0.334 | [-0.400, -0.260] | 600 |

**Note on E2/E3.** total_acc_gain is also negative because within a
fixed 80-step window a fast learner exhausts most of their gain in the
first 10 steps, leaving less to accumulate by the last 10. The tercile
table confirms this: total_gain decreases monotonically from 0.23 (slow)
to 0.11 (fast). Any gain-based criterion over a fixed window is
confounded when the rate range is wide enough that fast learners
plateau before the window ends. The ONLY unconfounded criterion is
direct corr(r_hat, r_true), which requires ground-truth rates.

---

## Verdict

METRIC ARTIFACT. Recovery works (A positive) but the predictive-validity metric does not fire (B null/negative) because the true rate does not predict late-half gain (C null/negative). The saturation confound is confirmed: fast learners plateau in the first half and show little late gain, inverting the expected correlation. The null/negative predictive validity in E2/E2b is a metric artifact, NOT evidence that recovery fails.

### Implications

1. **Does recovery work on synthetic data (A)?**
   A rho of +0.464 confirms
   that the encoder recovers the true rate on synthetic sequences of
   length 80.

2. **Does the predictive-validity metric fire (B)?**
   B = -0.260. It does not fire, matching the E2/E2b null.

3. **Is the metric itself valid -- does the TRUE rate predict late gain (C)?**
   C = -0.377. No. The true rate does not predict delta_acc, confirming the saturation confound is structural.

**If A is positive but B and C are null/negative,** the correct
interpretation is that the predictive-validity test (Spearman of
r_hat_firsthalf vs delta_acc) is confounded by saturation and is the
wrong validation criterion. The human-front negative predictive
validity in E2/E2b is largely a metric artifact, not a failure of
recovery.

The E2/E3 results show that total_acc_gain is also negatively
correlated with r_true, so replacing late-minus-early with
tail-minus-head does not escape the confound. Any gain-based criterion
over a fixed window is structurally flawed when the rate distribution
spans a range wide enough for fast learners to plateau within the
window. The correct validation criterion is direct
corr(r_hat, r_true), which requires ground-truth rates. For datasets
without ground truth, a design-level fix is needed: either use an
early-slope criterion (rate of improvement in the first few steps)
or compare rate ranks across sub-populations known to differ in
learning speed by construction.

---

*Generated by `deep_irt/traj_synth/run_poscontrol.py`.*
*Plots: `outputs/poscontrol_plots.png`. JSON: `outputs/poscontrol_results.json`.*
