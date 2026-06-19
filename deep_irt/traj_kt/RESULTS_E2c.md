# E2c Results: KDD Cup 2010 -- Decisive Human-Front Test

## Why KDD Cup 2010

EdNet (E2) was single-pass; no repeated practice, no learning curve.
ASSISTments (E2b) used skill_id as the item key, which made the AFM
concurrent check circular (the encoder's item key is the same KC the AFM
slope is fit on).  KDD Cup 2010 has step-level items (Problem Name +
Step Name) that are DISTINCT from the KC labels (KTracedSkills), so:

1. The existence gate is non-circular (item key != KC).
2. The AFM concurrent test is non-circular (AFM opportunity count is over
   KCs; the encoder never saw KC labels).
3. The dataset has genuine repeated practice across sessions.

## Dataset and Cohort

- Source: algebra_2008_2009_train.txt (from kddcup_challenge.tar.gz)
- Format: tab-separated, ~8M rows; Correct First Attempt is the binary response.
- Item key: Problem Name + '|' + Step Name (step-level granularity).
- KC label: KC(KTracedSkills) (first KC per step, for AFM only).
- Opportunity count: Opportunity(KTracedSkills) (provided in the data).

| Stat | Value |
|---|---|
| Students (raw) | 3,310 |
| Students (>= 200 steps) | 2,717 |
| Seq len median / p90 / max | 2280 / 7349 / 17,547 |
| Item vocab size (step-level) | 50,000 |
| Distinct KCs | 515 |
| Mean steps per KC per student | 30.04 (repeated-practice property) |

## Training

- Model: DeepIRTModel(n_cats=2, decoder='binary', encoder='lstm', decouple=True)
- Seq cap: 500 steps, batch=64, epochs=30, lr=0.001
- 80/20 student split (train / held-out for existence gate and oracle)
- Final train loss: 0.4026
- Wall time (total): 258.0s
- Peak VRAM: 3414 MB

## Validation

### (a0) Model-free trend diagnostic (does a trajectory exist at all?)

Before any model, a direct check: does within-student accuracy rise over the
sequence?  First-quartile vs last-quartile correct rate, capped at 500.

| Metric | Value |
|---|---|
| First-quartile acc | 0.774 |
| Last-quartile acc | 0.835 |
| Mean gain (last - first) | +0.0608 [0.0566, 0.0646] |
| Fraction of students improving | 0.745 |
| Overall correct rate | 0.804 |

A learning trajectory clearly exists in the data, model-free and unambiguous:
accuracy rises 6.1 points over the sequence and
74% of students improve, with a bootstrap CI
well above zero.  But note the overall correct rate (0.80)
is high, so the binary signal is near-saturated; this caps how much a dynamic
theta can add over a static one in held-out prediction.

### (a) Existence Gate (PRIMARY, the validated test)

Holds out the last 20% of each student's sequence.
Dynamic predictor: the model's aligned theta at the last fit step.
Static null: MLE constant theta on the full fit window (the validated
full-window-static comparator from _validity_criterion_exp.py).
delta_NLL = NLL_static - NLL_dynamic per student (positive = dynamic wins).

| Metric | Value |
|---|---|
| N students | 2717 |
| mean delta_NLL | 0.0008 [-0.0006, 0.0023] |
| frac delta_NLL > 0 | 0.537 |
| Wilcoxon p (one-sided, > 0) | 2.707e-04 |

**Existence verdict**: WEAK PASS: dynamic > static is statistically significant (Wilcoxon p < 0.05) but the effect is at the measurement floor (mean delta_NLL CI crosses zero).

The Wilcoxon test (a signed-rank test on the median) is significant, but the
mean effect is at the measurement floor and its bootstrap CI crosses zero. Read
together with (a0): a trajectory exists, and the dynamic model has a real but
tiny held-out edge, exactly as expected when correctness is near-saturated.

### (b) Oracle Magnitude (binary 2PL)

Recovers per-item a, b from the model then fits the full learning curve
theta_0, theta_inf, r per student by MLE (oracle_rate_mle from
deep_irt.traj_synth.metrics).  This is the validated magnitude estimator.

| Metric | Value |
|---|---|
| N finite | 2717 |
| mean r_oracle | 0.6382 |
| median r_oracle | 0.1956 |
| p90 r_oracle | 3.0000 |

### (c) Non-Circular AFM Concurrent (PRIMARY)

Per student: logistic regression correct ~ opportunity_count_within_KC,
weighted mean slope across KCs with >= 3 observations.
Correlated with oracle r_hat (not delta_NLL, which is not valid for magnitude).
NON-CIRCULAR: encoder item key = Problem|Step; AFM KC = KTracedSkills.

rho=-0.005 [-0.041, 0.032] (n=2716)

This is null. It must be read alongside (d): the per-student oracle rate is barely
self-consistent, so this null is a measurement-floor artifact, not evidence that
the recovered rate and the AFM slope disagree about real learning.

### (d) Split-Half Reliability of r_oracle

rho=0.167 [0.130, 0.206] (n=2717)

This is the load-bearing diagnostic for magnitude. A reliability of
0.17 means the per-student rate is mostly noise on this data.
You cannot validate a measurement against an external criterion (the AFM slope)
when the measurement is not reliable, so the null in (c) is uninterpretable as a
true absence of signal. The cause is the near-saturated binary response: with an
0.80 overall correct rate, a 2PL learning
curve has little dynamic range per student.

### (e) Convergent: Aligned vs Responsive Theta

rho=0.793 [0.767, 0.819] (n=2717)

The two encoder theta streams agree strongly, so the trajectory the model reads
is internally stable; the unreliability in (d) is in the parametric RATE fit on a
saturated signal, not in the theta trajectory itself.

## Contrast with E2 and E2b

| Dataset | Existence gate | AFM concurrent | Non-circular? |
|---|---|---|---|
| EdNet-KT1 (E2) | Not applicable (single-pass) | Not applicable | N/A |
| ASSISTments 2009 (E2b) | Not run (old method) | rho from prior run | NO (skill_id == KC) |
| KDD Cup 2010 (E2c) | mean=0.0008 p=2.71e-04 | rho=-0.005 | YES |

## Reading the result honestly

Three facts, in order of how well they are established.

1. A learning trajectory EXISTS in KDD, model-free and unambiguous. Accuracy
   rises 6.1 points within students and
   74% improve (CI above zero). This is the
   property EdNet lacked by construction.

2. The model's dynamic theta has a REAL but TINY held-out predictive edge over a
   static-ability null (Wilcoxon p=2.7e-04, mean
   delta_NLL=0.0008 with a CI that crosses zero).
   The edge is small because correctness is near-saturated, which leaves little
   for a moving theta to add at held-out prediction.

3. The non-circular AFM concurrent test is NULL, but uninterpretable. The
   per-student oracle rate is only 0.17 reliable (split-half),
   so there is no stable per-student quantity to correlate with the AFM slope.
   This is a measurement-floor failure on a saturated binary signal, not a
   demonstration that the recovered rate is wrong.

What E2c does NOT deliver: the clean, decisive non-circular AFM confirmation it
was designed to produce. The decisive test is gated on a reliable per-student
rate, and KDD's near-saturated binary response does not supply one. A polytomous
or partial-credit response (more dynamic range per item) or a lower-accuracy
cohort would be the natural next venue for the magnitude-concurrent claim.

## Verdict

**QUALIFIED POSITIVE on EXISTENCE, NULL on MAGNITUDE-CONCURRENT. A within-student learning trend clearly exists (model-free accuracy gain +0.061 [0.057, 0.065], 74% of students improve). The existence gate is WEAK PASS (floor-level): dynamic > static at held-out prediction, delta_NLL mean=0.0008 [-0.0006, 0.0023], Wilcoxon p=2.71e-04, but the effect sits near the measurement floor because correctness is near-saturated (overall 0.80). The non-circular AFM concurrent is null (Spearman=-0.005 [-0.041, 0.032]); however the per-student oracle rate is NOT reliable (split-half rho=0.17), so the AFM null is uninterpretable as a true absence of signal . Honest read: KDD shows a real but small dynamic-tracking edge and a clear model-free learning trend, but does NOT deliver a clean non-circular AFM confirmation; the limiting factor is rate-estimate reliability on a near-saturated binary signal, not an absence of learning.**

Wall time: 258.0s  |  Peak VRAM: 3414 MB
