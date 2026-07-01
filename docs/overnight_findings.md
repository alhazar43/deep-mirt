# Overnight Findings

Two threads ran in parallel. Part A covers the Q-MIRT transfer and active-change campaign
(venues 0-4). Part B covers the NRM parameter-representation study (objective A and thread B,
including the EdNet real-data leg).

---

## Part A: Transfer and Active Change (Q-MIRT)

### Model

An explicit per-concept state `z` (D=3 concepts) evolves by a causal transition:

```
z_{t+1,c} = decay_c * z_{t,c}
           + own_gain_c * Q[item_t, c]
           + resp_feedback_c          (FB variant only: Q-gated own-concept innovation)
           + (prac_t @ G.T)[c]        (G = zero-diagonal fitted matrix)
```

`G` is the sole cross-concept route. Item discriminations and thresholds are fit in Stage 1
then frozen; Stage 2 releases `G` with an L1 penalty. The readout is a compensatory GPCM
logit summed over concepts. Active-change isolation is structural: the update of `z_B` has no
term in `z_A`; setting `G=0` in a control run reduces non-practiced concepts to pure decay,
verified at 1e-8 error.

### Metric

The primary metric is the masked-forecast active gap. After conditioning on the first 60
steps with real responses, forecast NLL is evaluated over the next 30 steps with responses
masked. The target concept is measured but not practiced in the forecast window; the source
concept is practiced. Active gap = (no-transfer forecast NLL) minus (with-transfer forecast
NLL) on the target, within condition (item params and the initial state cancel). The metric is
read as a matched-null paired contrast: transfer run minus a same-seed null twin in which
the generator has `G_true=0`. Fitted `G` values are never compared against zero directly
because they carry a per-seed additive offset.

### Findings by venue

| Venue | Topic | Result |
|-------|-------|--------|
| 0 | Structural isolation + recovery | Active change structural (G-zero control decays to 1e-8). Transfer direction and existence confirmed; inhibitory G sign-correct 3/3; model-free B-U contrast 0.236 (transfer) vs -0.031 (null). Within-learner r 0.37. |
| 1 | Masked-forecast test | Active gap on B: +0.263 +/- 0.211 (3/3 positive). Near-zero on U (specificity -0.0002). Null active gap +0.002. With feedback (PSI-KT innovation): within-learner r rises to 0.715, passive gap +0.629. |
| 2 | Confound discrimination | Correlated-no-transfer ~0 (2/9 positive, near-zero). Co-scheduling ~0 in aggregate (5/9, 9x below baseline). Shuffle-order collapses to -0.0025, G_hat exactly 0.0 (9/9). Reverse-direction G[A,B]=0.0 (9/9). |
| 3/3b | Noisy and non-monotone theta | Exponential-approach model fabricates on non-monotone data (null gap 96% of active gap). OU mean-reverting transition clears it (null -> -0.0001). With L1 loosened to 0.001: active gap +0.066 +/- 0.064, 9/9 positive, null +0.003. |
| 4 | Alternative active mechanisms | Linear own-gain (M1), mastery-ceiling gain (M2), and rate+forgetting (M3) all give active gap +0.355, +0.364, +0.357 on 6/6, 5/6, 6/6 seeds. Null near zero for all. Isolation verified to 1e-8. |

**Individual learning recovery** rises from 0.37 (no feedback) to 0.80 (with PSI-KT
response feedback), driven by the own-concept innovation term.

### Corrections made during the run

The earlier "beats a passive LSTM by +0.63" number from venue 1 (model_seed=0 only) did not
survive the full seed sweep in venue 4: the frozen passive LSTM has a free decoder while the
explicit state model has a frozen decoder, so the LSTM wins absolute NLL by construction and
the comparison is confounded. That number is dropped. The load-bearing active-versus-passive
contrast is the within-condition no-G arm (same frozen decoder, same split-state initial
condition), which all three mechanisms beat.

Excitatory transfer must always be read as a matched-null paired contrast. The per-seed
additive offset in the fitted `G` values makes any comparison of `G_hat` against zero
uninterpretable.

Magnitude is gauge-bound throughout; only direction and existence are claimed.

### Limits and open items

The entire campaign is synthetic under correct model specification (estimator matches
generator family), D=3 concepts, seed counts of 3-9 per venue. Co-scheduling without
decoupling episodes is non-identified (collinear practice counts): this is a stated
requirement, not a bug, and has the same structure as the measurement-invariance gauge
(uniform item drift is not separable from person-level location).

Open items that require a rebuilt harness:

- **D-scaling to D=5, 8.** The current masked-forecast runner bakes concept roles, the single
  directed edge, and the practice schedule into a fixed `ITEM_SEQ_TOTAL` constant. Direct G
  recovery is confounded by the PSI-KT response feedback (for the practiced direction only),
  so the forecast metric is the right one to generalize. Build a silent runner (JSON output
  only) to avoid the 32k output-token crash that killed two workflow agents.

- **KDD Cup 2010 real data.** Needs the generalized harness plus a knowledge-component-to-concept
  mapping. No ground truth, so the claim steps down to predictive improvement (transfer beats
  no-transfer on held-out NLL) plus differential-invariance checks and seed stability. Best
  done with the user present.

---

## Part B: NRM Parameter Representation

### Setup

Three runs of the analytical and recovery study (objective A) then a full synthetic
architectural sweep (thread B), followed by an EdNet option-tracing real-data leg and a
coverage fix. N=800, Q=50-60, K=4-5, 150-200 epochs, 8 seeds for the sweep. Codex core
untouched throughout; heads built additively in scratch files.

### Fisher information and the dissociation

Analytical Fisher at the true parameters is near-symmetric: I(a_k) = 0.117, I(c_k) = 0.133,
ratio 0.90 +/- 0.015. This is expected: both `a_k` and `c_k` ride the same `P_k(1-P_k)`
factor; the theta-squared multiplier on `a_k` averages to roughly 1 under theta ~ N(0,1).
The 5-10x asymmetry seen for GPCM alpha versus beta does not appear here.

What does appear is an early-versus-late dissociation. At epoch 5, c_k Spearman = 0.146,
a_k = 0.087. By epoch 80-120, a_k catches c_k and exceeds it at convergence. The MLE oracle
(per-item estimation on known theta, no encoder) removes the early lag entirely: sp_a = 0.985
+/- 0.002, sp_c = 0.976 +/- 0.003, a_k leading from epoch 5. The early lag is therefore an
encoder initialization effect: the encoder forces theta near zero at init, so the gradient
of the loss with respect to `a_k` (which carries a theta factor) is suppressed by roughly
18x (mean |grad| for fc_a = 1.5e-3 vs fc_c = 2.67e-2) rather than anything attributable to
Fisher information.

Two claims retracted after the gauge audit. The late c_k decay (Spearman dropping 0.835 to
0.640) and the allocation inversion (std ratio 2.58) do not survive pinning the theta-scale
gauge: with a theta-scale penalty the allocation ratio falls to 0.47 and c_k > a_k at
convergence. The sigma sweep confirms the crossover: the a_k advantage disappears at sigma
= 0.5 (halved Fisher for the slope), isolating the Fisher axis from gauge drift. The
"allocation inverted / opposite-of-2PL" headline is dropped.

### Architectural sweep (thread B)

Six coupling configurations ({shared, decoupled, all-decoupled} x {static, dynamic}) plus a
shared-width frontier sweep (embedding dimension 8 to 64) and asymmetric per-parameter
decoupling cells.

**Three trade-offs identified.**

- Theta versus a_k is real and opposing. Over the width sweep, a_k rises +0.093 (0.846 to
  0.939, w=8 to 32) while theta falls monotonically -0.177 (0.797 to 0.620). Same sign as
  the GPCM alpha-versus-theta trade-off.

- Theta versus c_k is present but softer. c_k peaks at w=16 then declines with further
  widening; both item params collapse together at w=64 (over-parameterized).

- a_k versus c_k is absent. Decoupled/static gives a_k = 0.980 +/- 0.003, c_k = 0.982 +/-
  0.004; all-decoupled/static gives 0.978 / 0.973. Within confidence intervals, so the two
  params do not compete for the shared wide key.

**Decoupling escapes the trade-off.** The best shared point (w=8: a_k = 0.846, c_k = 0.887,
theta = 0.797) is beaten on all three by decoupled/static (0.980 / 0.982 / 0.853 +/-
0.024). A thin theta value plus a wide readout key escapes the shared Pareto surface for
both item params simultaneously.

**The dynamic head hurts a_k.** State-conditioning drops a_k recovery by 0.23 to 0.26 in
every coupling configuration (shared: 0.846 to 0.615; decoupled: 0.980 to 0.735;
all-decoupled: 0.978 to 0.715), while barely moving c_k (decoupled c_k: 0.982 to 0.969).
Dynamic a_k has high split-half reliability (0.995 to 0.999) but low recovery: a stable
readout of the wrong quantity. This matches the prediction from the Fisher analysis: NRM has
no low-Fisher channel for the dynamic head to rescue, so state noise degrades recovery
without providing identifiability help.

**Only c_k needs the wide key.** The asymmetric-coupling cells show: giving a_k its own wide
key does not free it (a_only_dec a_k = 0.873, barely above shared 0.846) and breaks c_k
(collapses to 0.392). Giving c_k its own wide key frees both (c_only_dec: c_k = 0.975, a_k
= 0.964, theta = 0.870, within confidence intervals of fully decoupled). Once c_k stops
competing for the thin encoder value, that value carries enough residual discrimination
structure to serve a_k at near-decoupled quality. The minimum sufficient intervention is
c_only_dec; full decoupling provides a marginal edge on a_k (0.980 vs 0.964).

**What this means for the workshop claim.** The theta-versus-slope trade-off replicates even
with near-symmetric Fisher (I_a/I_c = 0.90), so it is a representation effect, not a
Fisher-information effect. The dynamic-head rescue that helped GPCM discrimination is absent
and harmful in NRM, so it is a Fisher-information effect, not a representation effect. GPCM
discrimination suffers both (low Fisher plus slope-on-ability representation), NRM a_k
suffers only the second.

### Real data (EdNet KT1 option tracing)

N=1200 learners, T=100 steps, Q=10189 items (~12 observations per item), K=4, 1 seed.

**The escape reverses under item-level sparsity.** Shared/static has the best theta
reliability (split-half 0.681) and best option accuracy (0.650); decoupled/static drops to
0.292 and all-decoupled collapses to -0.021. All configs sit at or below the always-correct
baseline (0.661). Root cause: at 12 observations per item the 64-dimensional decoupled key
table has far more capacity than the data supports, leaving too little gradient signal to train the encoder.

**The dynamic-hurts asymmetry replicates.** decoupled/dynamic collapses theta reliability
to -0.009 while giving moderate item-param reliability (0.720 / 0.716). Independent of
coverage.

**Coverage fix confirms sparsity as the sole cause.** KC-level pooling (142 knowledge
components, ~845 observations per KC, matching the synthetic regime) collapses the
theta-reliability gap between shared and decoupled from 0.389 to 0.036: decoupled/static
revives from 0.292 to 0.664, statistically within noise of shared/static (0.700). A
high-frequency item filter (top-100 items, ~270 observations per item) reverses the rank
as predicted (decoupled/static 0.341 beats shared/static -0.669). The representation escape
is coverage-contingent: it holds at synthetic and KC-level density and collapses under item
sparsity, where shared/static is the robust default.

The real-data leg is a paragraph, not a slide. Static item parameters are 1.000 by
construction (no reliability signal for the item-param half of the escape), one dataset,
one seed, floor at or below baseline. The slide-worthy content from thread B remains the
synthetic sweep (8 seeds, known parameters). The coverage-contingency caveat refines the
existing note ("decoupling is synthetic-only / flips sign on real data") to: "flips sign
under extreme item sparsity, revives at KC-level coverage."

This thread is feeding into docs/paper2.tex.
