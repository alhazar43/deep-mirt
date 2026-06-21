# E2d Results: KDD Cup 2010 -- Graded Response (K=4) Variant of E2c

## Motivation

E2c used the binary Correct First Attempt response (E2c split-half rho=0.17,
AFM rho=-0.005). The near-saturated binary signal (~80% correct)
left insufficient dynamic range for reliable per-student rate estimation.
E0 showed ordinal beats binary for rate recovery. E2d grades each step into
K=4 ordinal proficiency from the Incorrects and Hints columns:

  errors = Incorrects + Hints
  category 3: errors==0 (mastered first try)
  category 2: errors==1 (one slip)
  category 1: errors in {2, 3} (struggled)
  category 0: errors>=4 (severe difficulty)

Hypothesis: graded response -> more dynamic range -> higher rate reliability
-> interpretable (ideally positive) non-circular AFM concurrent.

## The hypothesis is falsified, and the histogram says why

The graded K=4 response is STILL near-saturated, category 3 holds 80.3 percent
of steps (264k/444k/1.04M/7.17M across categories 0/1/2/3). Grading errors
does not restore dynamic range when most steps have zero errors, the K=4
signal just inherits KDD's intrinsic ceiling (most steps are mastered on the
first attempt with no hints). So the premise behind E2d, that binarization
was discarding range, is wrong, the range was never there. The saturation is
a property of the KDD response distribution, not of the binary coding.

## Dataset and Cohort

- Source: algebra_2008_2009_train.txt
- Item key: Problem Name + '|' + Step Name (step-level, non-circular with KC).
- KC label: KC(KTracedSkills) (for AFM only).

| Stat | Value |
|---|---|
| Students (>= 200 steps) | 2,717 |
| Seq len median / p90 / max | 2280 / 7349 / 17,547 |
| Item vocab size (step-level) | 50,000 |
| Distinct KCs | 515 |
| Mean steps per KC per student | 30.04 |

## Training

- Model: DeepIRTModel(n_cats=4, decoder='gpcm', encoder='lstm', decouple=True)
- Seq cap: 500 steps, batch=64, epochs=30, lr=0.001
- 80/20 student split
- Final train loss: 1.0693
- Wall time (total): 226.7s
- Peak VRAM: 3414 MB

## Validation

### (a0) Model-free trend (graded response)

| Metric | Value |
|---|---|
| First-quartile mean graded | 2.475 |
| Last-quartile mean graded | 2.633 |
| Mean gain graded (last - first) | +0.1588 [0.1498, 0.1679] |
| Fraction improving (graded) | 0.771 |
| Mean gain binary (for context) | +0.0608 |
| Overall binary correct rate | 0.804 |

### (a) Existence Gate (GPCM NLL)

Holds out last 20% of each student. Dynamic predictor:
aligned theta at last fit step. Static null: MLE constant theta (GPCM).
delta_NLL = NLL_static - NLL_dynamic (positive = dynamic wins).

| Metric | Value |
|---|---|
| N students | 2717 |
| mean delta_NLL | -0.1084 [-0.1126, -0.1041] |
| frac delta_NLL > 0 | 0.139 |
| Wilcoxon p (one-sided, > 0) | 1.000e+00 |

**Existence verdict**: FAIL: static model not beaten by trajectory model.

### (b) Oracle Magnitude (GPCM)

Per-student oracle learning rate via oracle_rate_mle (GPCM a, K-1=3 betas).

| Metric | Value |
|---|---|
| N finite | 2717 |
| mean r_oracle | 0.7756 |
| median r_oracle | 0.2821 |
| p90 r_oracle | 3.0000 |

### (c) Non-Circular AFM Concurrent

AFM slope on BINARY correct ~ opportunity within KC (logistic per-KC, weighted).
Oracle r_hat from GPCM graded response.
NON-CIRCULAR: encoder item key = Problem|Step; AFM KC = KTracedSkills.

rho=0.026 [-0.010, 0.062] (n=2716)

### (d) Split-Half Reliability of r_oracle

rho=0.190 [0.150, 0.225] (n=2717)

### (e) Convergent: Aligned vs Responsive Theta

rho=0.912 [0.894, 0.926] (n=2717)

## E2c vs E2d Comparison

| Metric | E2c (binary) | E2d (graded K=4) |
|---|---|---|
| Response type | binary (0/1) | graded (0..3) |
| Split-half reliability (r_oracle) | 0.170 | 0.190 |
| Non-circular AFM concurrent rho | -0.005 | 0.026 |
| Existence gate mean delta_NLL | 0.0008 | -0.1084 |
| Existence gate Wilcoxon p | <0.05 | 1.00e+00 |

## Verdict

**Graded scoring does NOT rescue the human rate, and the cause is now
diagnosed.** Split-half reliability barely moved (0.17 to 0.19), the
non-circular AFM concurrent is still null (0.026, CI spans zero), and the
existence gate actually FAILED under GPCM (mean delta_NLL -0.108, a constant
ability predicts the held-out tail better than the trajectory). The reason
is the histogram, the K=4 graded response is still 80 percent top-category,
so grading from errors did not add the dynamic range the hypothesis assumed,
KDD steps are near-mastery regardless of coding.

The real conclusion across E2/E2b/E2c/E2d, a learning TREND exists in
repeated-practice human data (model-free, +0.16 graded, 77 percent improve),
and the encoder's theta trajectory is highly stable (convergent 0.91), but
the per-student RATE is not recoverable from standard KT logs because the
response signal is intrinsically near-saturated. This is a data-property
limit, not a binarization artifact and not a method failure, E0 recovers the
rate when the signal carries it. Recovering a human rate would need an item
pool with genuine per-step dynamic range (many partial-credit or error-rich
responses), which these mastery-oriented tutoring logs do not provide.

Wall time: 226.7s  |  Peak VRAM: 3414 MB
