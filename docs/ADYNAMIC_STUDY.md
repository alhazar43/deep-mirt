# a_dynamic study: is the neural model's context-dependent discrimination real or an artifact?

Status PRELIMINARY (1 to 3 seeds; needs replication). Branch feat/prediction-loss. Started 2026-06-17.

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

## Implications / next steps

- The clean instrument requires either the regularized architectural split (a
  zero-mean-pinned, penalized dynamic residual `d` on top of a static `s`, so the
  Fisher-tail wobble is suppressed at the source) OR an explicit bias model
  (characterize this null theta-shape and subtract it), before any signal test.
- Phase 2 (the decisive test) extends the generator to plant genuine
  theta-dependent discrimination, then asks whether `corr(a_dynamic, theta)` lifts
  ABOVE the null shape, not above zero. Build the bias control in first.
- Parallel RQs in flight, RQ1 (alpha-vs-beta dynamic asymmetry, needs the
  state_beta head) and RQ2 (convergence-rate curves, needs per-epoch recovery
  logging).

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
python deep_irt/bench/_adynamic_probe.py          # Result 1 (N-sweep null probe)
python deep_irt/bench/_adyn_theta_relation.py     # Result 2 (relation study)
```
Both are temporary probes under deep_irt/bench/ (static-alpha synthetic, the null).
