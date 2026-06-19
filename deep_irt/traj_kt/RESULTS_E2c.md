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
- Wall time (total): 204.3s
- Peak VRAM: 3414 MB

## Validation

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

**Existence verdict**: PASS

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

### (d) Split-Half Reliability of r_oracle

rho=0.167 [0.130, 0.206] (n=2717)

### (e) Convergent: Aligned vs Responsive Theta

rho=0.793 [0.767, 0.819] (n=2717)

## Contrast with E2 and E2b

| Dataset | Existence gate | AFM concurrent | Non-circular? |
|---|---|---|---|
| EdNet-KT1 (E2) | Not applicable (single-pass) | Not applicable | N/A |
| ASSISTments 2009 (E2b) | Not run (old method) | rho from prior run | NO (skill_id == KC) |
| KDD Cup 2010 (E2c) | mean=0.0008 p=2.71e-04 | rho=-0.005 | YES |

## Verdict

**Candid read, the human-front claim is NOT supported even here.** Two
things temper the existence-gate pass. The Wilcoxon test is significant
(p=2.7e-04), but the effect is practically negligible, the mean delta_NLL
is 0.0008 with a bootstrap CI that SPANS ZERO ([-0.0006, 0.0023]) and only
53.7% of students beat the static null, a hair above chance. So the
trajectory model out-predicts a constant ability by a vanishing margin.
More decisively, the non-circular AFM concurrent validity is a flat null
(Spearman -0.005, CI [-0.041, 0.032]), the recovered rate carries no
classical learning-rate signal, and split-half reliability is low (0.167).

So on the best available real-data test, repeated practice, problem-level
items, a non-circular design, and the validated existence-then-magnitude
pipeline, the recovered human learning rate has NO demonstrated external
validity. Combined with the EdNet and ASSISTments results, the human-front
claim is thoroughly tested and unsupported on three real datasets. The
clean positives in the program remain synthetic (E0) and machine (E1b).
The convergent readout agreement (0.793) only confirms the rate is a
stable feature of the encoder, not that it tracks real learning.

Wall time: 204.3s  |  Peak VRAM: 3414 MB
