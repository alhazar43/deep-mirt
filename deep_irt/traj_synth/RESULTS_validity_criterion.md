# Held-Out Prediction as a Validity Criterion

## Setup

600 synthetic learners, 120 items, K=4 GPCM, seq_len=80 (rate log-uniform in
[0.04, 0.6]). The last 20% (T_tail=16) of each learner's sequence is the
held-out tail. The first 80% (T_fit=64) is the fit window. DeepIRTModel
(lstm, gpcm, decouple=True), 500 epochs. The no-rate control fixes the gain at 0.05
so theta is flat.

All comparisons use item parameters recovered from the dynamic model.

The dynamic prediction fits an exponential curve (A, B, r) by MLE on the
fit-window responses and extrapolates it to the tail time points. Two static
baselines predict the tail with one constant theta.

  Full-window static. A single theta fitted by MLE to all 64 fit-window
  responses. This pools the whole trajectory and ignores time.

  Recent-window static. A single theta fitted by MLE to the last 16
  fit steps only, the window immediately before the tail.

delta_NLL = NLL_static - NLL_dynamic, per learner, in nats per held-out step.
Positive means the trajectory model predicts the future better.

## The two questions and the baseline each one needs

The held-out criterion answers two separable questions. They have different
answers, and each is served by a different static baseline.

  EXISTENCE. Does the dynamic model beat the static model when, and only when, a
  rate exists? This is the cross-condition contrast, with-rate vs no-rate. The
  right comparator is the full-window static theta, the genuine constant-ability
  null that ignores time. The recent-window static is itself a partial trajectory
  read (it tracks where the learner ended up), so it is not a clean stand-in for
  the no-trajectory hypothesis.

  MAGNITUDE. Does the per-learner margin delta_NLL rank-correlate with the true
  rate r? This is Spearman(delta_NLL, r_true) within the with-rate condition. The
  right comparator is the recent-window static theta. The full-window static is an
  unfair comparator here, because for fast early-plateau learners it pools the
  plateau and predicts the tail almost as well as the trajectory does, which drives
  the correlation negative.

## Results

### With-rate condition

| Metric | Full-window static | Recent-window static |
|---|---|---|
| Mean delta_NLL | 0.0172 | 0.0190 |
| Fraction delta_NLL > 0 | 0.582 | 0.565 |
| Spearman(delta, r_true) | -0.177 [-0.259, -0.096] | 0.131 [0.049, 0.210] |

Rate ceiling. oracle_rate_mle fitted on the model's recovered item parameters
(n=300) gives Spearman(r_oracle, r_true) =
0.410
[0.308, 0.512].

For contrast, the prior run's weak rate anchor (curve fit to smoothed encoder theta
on the first half of the fit window) scored Spearman =
0.180, and the broken gain-over-window metric
scored Spearman = -0.042.

### No-rate control (gain fixed at 0.05)

| Metric | Full-window static | Recent-window static |
|---|---|---|
| Mean delta_NLL | -0.0357 | 0.0379 |
| Fraction delta_NLL > 0 | 0.440 | 0.543 |
| Spearman(delta, r_true) | 0.038 | 0.014 |

### Existence test (with-rate vs no-rate delta_NLL)

Full-window static (the correct existence comparator).
Mann-Whitney U = 218892, p = 4.610e-11
(one-sided, with-rate greater).
Mean gap (with minus no) = 0.0530
[0.0348, 0.0711].

Recent-window static (for completeness, not the right comparator here).
Mann-Whitney U = 177216, p = 6.786e-01.
Mean gap (with minus no) = -0.0189
[-0.0398, -0.0007].

## Diagnosis

The existence question and the magnitude question come apart, and each is read off
a different baseline.

Existence. Against the full-window static null, the dynamic model's held-out
advantage is far larger when a rate exists than when ability is flat. With a rate
the mean delta_NLL is 0.017 and
58% of learners are predicted better by the
trajectory. With flat ability the mean drops to -0.036 and only
44% favor the trajectory. The cross-condition gap
is 0.053 with a CI above zero and Mann-Whitney
p = 4.6e-11. The criterion detects the presence of a
trajectory and does not reward trajectory modeling when there is nothing to track.
This is the saturation-robust property the criterion was meant to have.

The recent-window static baseline collapses this separation
(p = 0.68) for a clear reason. That baseline is already
a one-step trajectory estimate. It tracks the learner's recent ability, so against
it the curve model has little left to add even when a rate is present, and the
with-rate and no-rate margins look alike. That is why the existence test must use
the full-window null, not the recent-window read.

Magnitude. The per-learner margin does not rank learners by rate. Against the
full-window static the correlation with r_true is negative
(-0.177), because that baseline pools the plateau and
for fast learners predicts the tail nearly as well as the trajectory. Switching to
the recent-window static removes that head start and flips the correlation positive
(0.131, CI above zero), but the signal is weak. The
reason is structural. By the tail, fast and slow learners are both near plateau, so
the held-out window carries little rate information no matter which static theta it
is compared against.

The rate is nonetheless recoverable from this data. oracle_rate_mle, which fits the
full parametric curve to the responses rather than comparing two windowed theta
estimates, recovers r with Spearman 0.410 from the
model's own recovered item parameters, well above the held-out margin and the broken
metric. The ceiling is 0.410 rather than the ~0.9
seen with true item parameters because the estimated items add noise, but it is the
strongest available estimator and is the correct one for magnitude.

## Side-by-side with the broken metric

The original gain-over-window metric scores Spearman =
-0.042 against the true rate, near zero, and it is
structurally ill-posed because fast learners plateau inside the window. The held-out
existence test, by contrast, separates with-rate from no-rate at
p = 4.6e-11. The held-out criterion is a real,
saturation-robust signal for the existence question where the broken metric is not.
For magnitude, neither the broken metric nor the held-out margin is usable;
oracle_rate_mle is.

## Verdict

EXISTENCE. Valid. Held-out predictive improvement, measured against a full-window
constant-ability null, is a saturation-robust detector of whether a learning
trajectory exists. The dynamic model beats the static null significantly more when a
rate is present than when ability is flat (p = 4.6e-11, mean
gap 0.053 with CI above zero), and the no-rate control confirms
the criterion does not reward trajectory modeling in the absence of a trajectory. It
needs no ground-truth rate, so it transfers to real data. Use it as the validity
gate that licenses a dynamic ability claim. Use the full-window static theta as the
null, not a recent-window read, which is itself a partial trajectory and washes the
test out.

MAGNITUDE. Not valid. The per-learner delta_NLL margin does not track the rate. It
is negative against the full-window null and only weakly positive against the
recent-window comparator, because the held-out tail sits near plateau for fast and
slow learners alike. For the magnitude of the rate, fit the parametric learning
curve (theta_0, theta_inf, r) by MLE to each learner's full response sequence given
the model's estimated item parameters (oracle_rate_mle). That estimator uses the
full curve structure and is not confused by saturation in any subwindow.

Recommendation for real data (EdNet, ASSISTments, KDD). Report the two together and
keep their roles distinct. Held-out prediction against a full-window null is the
existence gate that licenses calling a learner dynamic. oracle_rate_mle on the
model's estimated item parameters is the rate estimator once that gate is passed.
The held-out margin itself is not a magnitude readout and should not be reported as
one.
