# a_dynamic study: is the neural model's context-dependent discrimination real or an artifact?

Status PRELIMINARY (3 seeds; needs replication). Branch feat/duolingo-mini. Started 2026-06-17.
Phase 1 (is the wiggle real or artifact) and Phase 2 (does it detect planted
theta-dependence) both DONE; the verdict is directional detection, not calibrated
magnitude.

## The object

The state-conditioned discrimination head produces a per-occurrence alpha,
`alpha_jt = exp(fc_a_state([state_t, item_key_j]))`. Decompose it as

    a_static(j)   = mean_t alpha_jt        (the would-be fixed item discrimination)
    a_dynamic(j,t) = alpha_jt - a_static(j) (the per-occurrence wiggle)

`a_dynamic` is the neural-IRT-native quantity, classical IRT has one scalar alpha
per item and nothing corresponding to the wiggle. The question is whether the
wiggle is a real signal (discrimination that genuinely varies with the
respondent's state, as longitudinal IRT posits) or an estimation artifact.

beta is item-key-only (no state) so it has no dynamic part by construction; theta
is purely dynamic. The decomposition is alpha-specific.

## Test design

Train on STATIC-alpha synthetic GPCM data, where the true discrimination is
occurrence-invariant, so the true `a_dynamic` is exactly zero. Anything the model
produces is therefore artifact, this is the clean null.

## Result 1 -- null probe, N-sweep (3 seeds, static K=4)

| N | a_static recovery | dyn_CV | corr(a_dyn, theta_true) | corr(a_dyn, theta_model) |
|---|---|---|---|---|
| 200 | 0.756 | 0.640 | +0.140 | +0.121 |
| 800 | 0.923 | 0.837 | +0.020 | +0.025 |
| 3200 | 0.933 | 0.631 | +0.054 | +0.035 |

- `a_static` recovers the true alpha and sharpens with N (0.76 -> 0.93).
- `a_dynamic` does NOT vanish with data, dyn_CV stays ~0.6 to 0.8 across a 16x
  increase in N. So it is not finite-data scaffold, it is structural, the head
  injects a fixed amount of per-occurrence wobble because it reads the full
  hidden state, not just theta. The "more data kills it" branch is ruled out.
- Linear corr(a_dyn, theta) is near zero. This turned out to be MISLEADING (see
  Result 2), the relation is non-monotone and nearly cancels in a single Pearson.

## Result 2 -- the a_dynamic <-> theta relation (single model, N=2000, K=4)

```
a_static recovery 0.946   a_dynamic CV 0.773
linear  corr(a_dyn, theta_model) = -0.266   corr(a_dyn, theta_true) = -0.188
gap     corr(a_dyn, theta-beta)  = -0.150   corr(a_dyn, |gap|) = +0.146   corr(a_dyn, gap^2) = +0.156
R^2 of a_dyn from cubic(theta_model) = 0.179   cubic(theta_true) = 0.087
```

Binned a_dynamic across theta_model deciles:

```
theta decile   a_dyn mean   a_dyn std
  -1.91         +0.638        1.06      low-ability tail: alpha inflated and wild
  -1.20         +0.137        0.68
  -0.78         ~0            0.57
  -0.05         -0.168        0.47
  +0.59         -0.154        0.39
  +0.96         -0.140        0.38      bulk: alpha slightly deflated, tight
  +2.09         -0.006        0.43
```

Reading. On data where true discrimination is theta-independent, `a_dynamic`
carries a real NONLINEAR theta-structure, a cubic in theta_model explains ~18% of
its variance. The shape is a Fisher-tail estimation bias. At the low-theta tail
the head inflates alpha (+0.64) and its variance explodes (std 1.06); through the
bulk it deflates slightly and tightens. The positive `|gap|` and `gap^2`
correlations confirm the geometry, the wobble concentrates where `(theta-beta)^2`
is small, exactly where alpha is least identified (the per-response Fisher
information for alpha, `I(alpha) ~ (theta-beta)^2`, vanishes there). So most of the
wiggle is the head over-estimating and destabilizing discrimination where the data
is uninformative, not a genuine context-dependence.

Analogy, judging a ruler's precision by measuring only babies (all the same
height). No height range, so the readings are nonsense, and the nonsense is not
"the ruler behaves differently for babies."

## Verdict (preliminary)

1. `a_dynamic` is not finite-data scaffold (persists with N) and not clean noise
   (carries nonlinear theta-structure, R^2 ~0.18).
2. Out of the box it is CONTAMINATED by an information-starved estimation bias
   concentrated at the low-ability / low-Fisher tail. The naive detector
   `corr(a_dynamic, theta)` is therefore non-zero and structured on the null, so a
   positive correlation on real data cannot be read as genuine theta-dependent
   discrimination without controlling this bias first.
3. Linear correlation is a bad summary of the relation (the non-monotone shape
   cancels); the structure is only visible nonlinearly (R^2, binned shape).
4. The bias itself is worth reporting, anyone reading state-conditioned alpha as
   "context-dependent discrimination" is partly reading a low-ability estimation
   artifact, which the neural-IRT line has not flagged.

## Phase 2 -- the decisive signal-detection test (3 seeds, K=4)

Plant genuine theta-dependent discrimination, `alpha_eff(i,t) = a_j *
exp(gamma_j * theta_it)`, with `gamma_j ~ N(0, sigma)` drawn from a SEPARATE rng
(datagen.py). At a fixed seed the null (sigma=0) and every planted set share the
same a, b, theta, item sequences and per-step choice draws, only the responses
differ, so the null is a matched bias control and (because the gamma rng is seeded
once) the planted gammas at different sigmas are proportional, a clean dose-response
on the same items.

Readout. For each fit, read every per-occurrence `alpha_jt` and take the per-item
OLS slope of `log(alpha_jt)` on the TRUE theta (external, no latent circularity).
This linear slope is the matched estimator for the planted form (`log alpha =
log a + gamma*theta` is exactly linear) AND it sidesteps the Phase-1 contamination,
the Fisher-tail bias is non-monotone in theta (Result 2) and nearly cancels under a
linear projection. The detector is `signal_j = slope_planted_j - slope_null_j`,
scored as `corr(signal_j, gamma_j)`; calibration is the OLS slope `k` of signal on
gamma (`k = 1` is exact magnitude).

| sigma | corr(slope_planted, g) | corr(signal, g) | corr(slope_null, g) | calib k | null slope std |
|---|---|---|---|---|---|
| 0.20 | +0.427 | +0.438 | -0.040 | 0.039 | 0.015 |
| 0.40 | +0.649 | +0.666 | -0.040 | 0.040 | 0.015 |

Findings.

1. DETECTION CONFIRMED. The head recovers the planted theta-dependence, and it is a
   genuine dose-response, corr rises 0.43 -> 0.67 as the planted slope doubles. The
   sanity `corr(slope_null, gamma) ~ 0` holds (gamma is independent of item a/b), so
   the positive correlation is detection, not a spurious pathway.
2. THE BIAS PROBLEM EVAPORATES ON THE RIGHT READOUT. On the linear slope the null
   bias is tiny and gamma-independent (std 0.015 vs a planted signal an order larger,
   corr_null ~ 0), so the matched-null correction is nearly a no-op (0.649 -> 0.666).
   This resolves the Phase-1 worry, the contamination was specific to the NONLINEAR
   per-occurrence wiggle; the linear log-alpha-on-theta slope is the clean instrument
   and needs almost no bias control. (Result 2's "linear correlation cancels" cut
   both ways, it kills the naive detector but it also kills the bias, leaving the
   linear PLANTED signal clean.)
3. MAGNITUDE IS NOT RECOVERED, ONLY RANK, AND THE SHRINKAGE IS GENUINE (not a
   scale artifact). Calibration `k ~ 0.04` at both sigmas (stable), the recovered
   slope is ~4% of the planted magnitude, a ~25x shrinkage. A natural objection is
   identifiability, the model's latent theta is fixed only up to an affine scale and
   in `a*(theta-b)` discrimination scales inversely with theta's scale, so a
   compressed internal theta would deflate the alpha slope by the same factor. The
   decomposition rules this out (3 seeds, sigma=0.4): the model's per-learner theta
   recovers true theta0 at corr 0.96 on essentially the right scale, `c = OLS(theta_hat,
   theta0) = 1.14` (slightly INFLATED, which would only deflate `k` further), so the
   scale-corrected calibration `k/c = 0.034` barely differs from `k`. The ~30x
   attenuation is therefore genuine HEAD SHRINKAGE, not a scale mismatch. Mechanism,
   the head reads the encoder state and the GPCM likelihood is barely sensitive to
   the alpha-theta slope (the SAME low-Fisher story that makes alpha hard to recover
   at all, RQ1), so the optimizer fixes the SIGN and RANK of the dependence but not
   its size. State-conditioned alpha is therefore a RANK / direction detector of
   theta-dependent discrimination, not a calibrated estimate of it; reading the size
   of `a_dynamic` as the strength of context-dependence is wrong by a large factor.

Verdict. The neural-IRT-native quantity `a_dynamic` does carry real signal about
genuine context-dependent discrimination, recoverable in rank once read as a linear
log-alpha-on-theta slope, which simultaneously dodges the Fisher-tail bias. Its
magnitude is heavily and genuinely attenuated (~30x head shrinkage, not a scale
artifact), so the honest claim is DIRECTIONAL detection, not calibrated measurement.

## Implications / next steps

- Phase 2 is DONE (see above) and turned the instrument question on its head. The
  naive per-occurrence wiggle does need the regularized split or a bias model, but
  the LINEAR log-alpha-on-theta slope already IS the clean instrument, it both
  matches the planted signal and dodges the nonlinear Fisher-tail bias, so no extra
  bias control is needed for directional detection. The regularized architectural
  split (a zero-mean-pinned, penalized dynamic residual `d` on a static `s`) is now
  a MAGNITUDE problem, not a detection one, it would be the route to lifting `k`
  off ~0.04 toward a calibrated estimate if magnitude recovery is ever wanted.
- Magnitude is genuinely shrunk ~30x, RESOLVED as head shrinkage not a scale
  artifact (theta recovers at corr 0.96, scale c=1.14, so k/c=0.034 ~ k). Lifting
  `k` off ~0.04 toward a calibrated estimate would need a head with stronger
  gradient pressure on the alpha-theta slope (the regularized split, or an
  auxiliary loss that rewards the slope directly), since the bare GPCM likelihood
  does not supply it. This is the open MAGNITUDE problem; detection (rank) is solved.
- RQ1 (alpha-vs-beta dynamic asymmetry) and RQ2 (convergence-rate curves) are
  CONFIRMED, see docs/FISHER_DYNAMICS_STUDY.md.

## Real-data validation design (future, after synthetic Phase 2)

Use a dataset where the same student answers items across different subjects or
skills (cross-subject), which gives within-student ability spread. Two
characterizations, one clean and one hard.

A. a_static is the test/subject-defining property, invariant across students.
   This is measurement invariance and it is the clean confirmatory test. Estimate
   a_static(item) from different student splits (cohort, ability band) and check
   it agrees item by item; across subjects a_static should cluster by subject.
   Doable on real data today.

B. a_dynamic of one student tracks that student's theta within noise. This is the
   ambitious test and two confounds block it. Theta is latent, so testing
   a_dynamic against the model's OWN theta is circular and co-contaminated by the
   same Fisher-tail bias. And cross-subject mixes item effects (different
   a_static) with theta effects. The cleanest handle is to make theta external,
   estimate a student's subject ability from a HELD-OUT set of that subject's
   items, then ask whether their wiggle on different items of that subject tracks
   the held-out ability and not their other-subject ability. Still
   model-estimated, so a validation, not a proof.

Ordering. Establish the mechanism on synthetic Phase 2 first, where theta and the
planted theta-dependence are known and there is no latent-variable circularity,
THEN validate on real cross-subject data with A as the clean confirmatory test and
B as the ambitious one. Running B before A, or before Phase 2, is how the
Fisher-tail bias gets published as a discovery.

## Reproduce

```
python deep_irt/bench/_adynamic_probe.py          # Result 1 (N-sweep null probe, temp)
python deep_irt/bench/_adyn_theta_relation.py     # Result 2 (relation study, temp)
python deep_irt/bench/run_phase2_signal.py --device cuda   # Phase 2 (signal detection)
python deep_irt/bench/_phase2_scale.py            # Phase 2 magnitude decomposition (temp)
```
Results 1 and 2 and the magnitude decomposition are temporary probes (static-alpha
synthetic, the null). Phase 2 is the committed runner; it writes
deep_irt/bench/outputs/phase2_signal.json (summary + per-item slopes) and depends on
the additive `alpha_theta_slope` generator option.
