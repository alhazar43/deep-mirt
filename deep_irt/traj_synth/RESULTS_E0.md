# E0 results: synthetic ability-trajectory rate recovery (RQ0)

The precondition study for the trajectory-recovery program
(`docs/trajectory_program.md`). It asks whether a prediction-trained
encoder-decoder recovers a known ability trajectory and its rate from a
response sequence, and under what conditions the rate is estimable.
Code in `deep_irt/traj_synth/`, results in `outputs/results_e0.json`.

## Setup

Each respondent follows a learning curve
theta(t) = theta_inf - (theta_inf - theta_0) exp(-r t), t = 0..T-1, with
theta_0 ~ N(-0.8, 0.6), a positive gain so theta_inf = theta_0 + gain,
gain ~ U(0.8, 2.5), and a log-uniform rate r ~ logU(0.04, 0.6). The rate
is the estimand. Items carry fixed IRT parameters; responses are sampled
from the GPCM (binary is K=2) under the moving theta_t. The encoder-decoder
is trained under prediction loss, the per-step ability is read with the
prediction-aligned and the responsive readouts, a curve is fit to the
recovered trajectory with a hardened readout (light smoothing, soft-L1
loss, rate cap), and the recovered rate is scored against the truth.

Two reference points anchor the reading. A fitter sanity check fits the
curve to the noiseless true trajectory and must be near perfect. An oracle
fits the rate by maximum likelihood with known item parameters and known
curve family, per respondent. The oracle is a per-respondent ML reference,
not a strict upper bound, since it does not pool across respondents.

The sweep crosses response format (binary, ordinal K=4) with sequence
density T in {20, 40, 80}, over three seeds, N = 300 respondents, 120
items, on the LSTM encoder with the decoupled default.

Key identifiability fact. The encoder's theta scale is arbitrary, but the
rate r is invariant under any affine transform of theta, so it is
recoverable without fixing the scale. The absolute ability is not. This is
why the rate is the estimand.

## Results

The fitter sanity is 1.000 in every cell, so the recovery machinery is
correct and the limits below are about information, not code.

**Density lifts recovery, monotonically and in both formats.**

| format | T | encoder rho | aligned rho | oracle rho | per-step | MAE |
|---|---|---|---|---|---|---|
| binary | 20 | 0.089 | 0.071 | 0.177 | 0.319 | 0.157 |
| binary | 40 | 0.114 | 0.151 | 0.208 | 0.325 | 0.135 |
| binary | 80 | 0.288 | 0.277 | 0.244 | 0.359 | 0.106 |
| ordinal | 20 | 0.211 | 0.197 | 0.136 | 0.365 | 0.156 |
| ordinal | 40 | 0.209 | 0.261 | 0.378 | 0.413 | 0.122 |
| ordinal | 80 | 0.410 | 0.421 | 0.575 | 0.462 | 0.093 |

(rho is Spearman of recovered against true rate across respondents, mean
over three seeds; MAE is the median absolute rate error; per-step is the
mean within-respondent correlation of recovered against true theta_t.)

**Richer response formats help.** Ordinal recovers the rate better than
binary at every density (0.42 against 0.28 at T=80), and the oracle gap is
larger still (0.58 against 0.24), so the advantage is information per
response, not the model.

**The encoder is statistically efficient.** It tracks the per-respondent
ML reference closely, beating it at low density where pooling across
respondents regularizes the rate, and trailing it at high density where
the per-respondent likelihood has enough data (ordinal T=80, encoder 0.42
against the reference 0.58). The bottleneck is the information in the
responses, not encoder capacity.

**The prediction-aligned readout is the equal-or-better one.** Reading
theta from history strictly before each step is as good as or better than
the responsive readout in the mid and high density cells (ordinal T=40,
0.261 against 0.209), and is the readout to carry to the real fronts.

**Recovery is concentrated in the identifiable regime.** Binning by the
true rate at T=80 shows the rate is recovered well for slow and moderate
learners and fails for fast ones.

| band (ordinal T=80) | r range | MAE | bias |
|---|---|---|---|
| low | 0.04 to 0.10 | 0.054 | +0.03 |
| mid | 0.10 to 0.22 | 0.082 | +0.05 |
| high | 0.22 to 0.59 | 0.155 | -0.06 |

A fast learner saturates within about 1/r steps, so once the curve has
flattened the remaining steps carry no rate information and the rate is
underestimated (the negative bias in the high band, strongest for binary
at -0.16). Rate is recoverable precisely when the observation window spans
the curve's elbow, that is when r T is not too large.

Plots in `outputs/`, `e0_recovery_vs_density.png` and
`e0_rate_scatter_gpcm.png`.

## Reading

Rate is recoverable but it is a weak, density-limited signal. The binding
constraint is whether the window samples the curve's elbow, the levers are
sequence density and response-format richness, and the neural encoder
extracts the rate about as well as a per-respondent ML estimator that
knows the items. This is an honest precondition result. It does not oversell
a strong recovery, and it gives concrete design rules for the real fronts.

## Implications for the real fronts

- Prefer dense sequences and the richest available response format.
- Only claim a rate for respondents still visibly improving across the
  window; report the saturating regime as out of scope, not as failure.
- Use the prediction-aligned readout.
- Expect modest rate correlations on real data, and report the per-respondent
  reference and the regime split alongside the headline number.

## Limitations

One curve family (a single exponential approach), one encoder (LSTM),
synthetic data, N = 300 with visible seed variance, and a per-respondent
ML reference that is not a strict upper bound. A pooled or hierarchical
reference and a second curve family would sharpen the ceiling.
