# Paper 2 draft plan (v2 — preliminary-pruned, evidenced)

Supersedes the earlier draft. A learning-dynamics / representation paper about
**amortized neural IRT**; the scope is neural IRT, not a universal claim. The spine
is the question: *how and why do ability (theta), difficulty (beta), and
discrimination (alpha) recover differently when an encoder infers ability and item
parameters are read off shared learned representations?*

This version is honest about what three preliminary probes pruned. They did their
job: they removed the over-reaches and left a smaller, evidenced core.

## 0. What the prelims settled (the pruning, recorded)

- **Not a universal representation law.** A non-IRT static-code toy showed the
  decoupling benefit is exactly zero (a per-item MLE oracle matches both arms); the
  effect needs the amortized-encoder structure. Scope = neural IRT. (Pillar 1)
- **Not kappa, as predictor or mechanism.** K and kappa are perfectly collinear in
  the K-sweep (Spearman 1.0). Breaking that — varying kappa 45x at fixed K — the gap
  is FLAT in kappa (Spearman -0.3 to -0.5), over a kappa range wider than the whole
  K-sweep. Both arms improve as alpha gets more informative, so their difference does
  not track kappa. The lever is K (ordinal information / shared-channel capacity), not
  the Fisher conditioning number; conditioning is dropped as a named mechanism.
  (Pillar 2 + the kappa-identification probe.)
- **Not a confound.** Recovered alpha is *not* contaminated by ability or difficulty
  (leakage correlations null in both arms). The distortion is **attenuation +
  geometric under-allocation**, not cross-parameter leakage. (Pillar 3)

## 1. The claim (the middle ground, in its evidenced form)

> In amortized neural IRT, recovery of an item parameter is gated by its
> **representational allocation**, not by the data's information about it. The
> low-information parameter (discrimination) is geometrically under-allocated in a
> shared item code and recovered with attenuated, seed-unstable rank. This is a
> finite-budget property of the representational *arrangement* — the parameter is
> recoverable at convergence (the data identify it) — and it is relieved by giving
> discrimination its own representation (decoupling) or access to the inferred state
> (dynamic). A parameter-specific, representation-steerable recovery distortion that
> classical estimation does not have.

It is **not** trivial-ML ("low-curvature directions train slowly" is about a
parameter in isolation; this is the recovery-vs-information dissociation in an
amortized estimator, with a measured geometric signature). It is **not** a challenge
to psychometrics (the model is identified, the estimator is consistent, Fisher is
right; we characterize a new property of the amortized estimator and give a caution).
It lives at the seam: it needs both the amortized encoder and interpretable recovery.

## 2. Claims (numbered, status-tagged)

- **C1. Differential rate.** theta and beta recover fast, alpha slow. *Proved-local
  (gradient P1 + Fisher P2: I(alpha)=(theta-beta)^2 w vanishes where responses
  concentrate) + Empirical (recovery-speed curves).*
- **C2. Recovery is gated by representation, not information.** Holding data, true
  parameters, and information fixed, changing the representation arrangement moves
  alpha recovery; the same data recover alpha at the optimum (free-table invariant).
  *Proved (free-table invariant, the "not data" half) + Empirical (the gate, the
  manipulations).*
- **C3. The under-allocation is geometric and visible.** In the shared code, alpha
  rides ~11% of the variance; difficulty dominates the code (probe R^2 0.96 vs 0.64);
  the item code carries no respondent-ability axis. *Empirical (Pillar 3b).*
- **C4. The distortion is attenuation, not a confound.** Recovered-alpha error does
  not correlate with theta or beta in either arm. *Empirical (Pillar 3a) — an honest
  negative that sharpens the claim.*
- **C5. Two steering levers.** Decoupling (own code) and dynamic conditioning
  (reading the state) each relieve alpha; difficulty is indifferent. *Empirical
  (gate; RQ1 at 12 seeds: state-conditioning helps alpha for K>=4, paired Wilcoxon
  p<=0.007, CIs exclude zero, grows with K, delta_beta null).*
- **C6. Rate-limited, grows with K.** At a fixed budget the gap does not shrink with
  data; it grows with the number of categories (more ordinal information, exploitable
  only by a decoupled code). *Empirical (N-sweep, K-sweep) — stated as the
  phenomenon, not as a conditioning law.*

### The two mechanisms, kept distinct
- **alpha:** a transient RATE / allocation effect on an invariant endpoint (P9/P4b).
- **theta:** a finite-data ENDPOINT / capacity effect, outside the free-table
  invariant (P10), Argued for the encoder. Do not let the title fold theta into the
  allocation narrative.
- **beta:** the indifferent control (P11).

## 3. Theory and its honest status

- The differential rate is the solid theoretical core: per-parameter gradients (P1)
  and Fisher (P2), with alpha the low-information mode. State as Proved-local.
- The shared-code allocation is the mechanism: the thin code's capacity is spent on
  the high-information consumers (difficulty, and the encoder's item-identity need),
  starving alpha; **the geometric probe (C3) is the direct evidence**, which makes
  "allocation" literal rather than metaphorical.
- The free-table invariant (P4b) carries the "not a data limit" half.
- **Dropped:** kappa entirely. The identification probe found the gap flat in kappa
  at fixed K, so conditioning is refuted as the lever, not merely unidentifiable. The
  K-growth is **channel capacity**: more categories strain the single narrow shared
  item channel, while the decoupled arm's own wide channel exploits the extra ordinal
  information. The smooth-map / block-Hessian results survive only as minor lemmas.
- No conditioning claim. No universality claim. No confound claim.

## 4. Figure / table inventory (post-prelim, bound to artifacts)

| Slot | Shows | Headline | Source | Claim |
|---|---|---|---|---|
| F1 gate | allocation not capacity; decoupled above the frontier | theta 0.97->0.88, alpha 0.66->0.91; decoupled both-high | gate_table.md | C2 |
| F2 trajectory | reached then left (rate) | shared-wide alpha 0.91@50 -> 0.79@500; decoupled holds | trajectory_table.md | C1 |
| F3 geometric under-allocation | alpha rides a thin slice; beta dominates | alpha ~11% code variance; probe R^2 0.64 vs beta 0.96; no theta axis | shared_alpha_leakage_results.json | C3 (the literal-allocation figure) |
| F4 K-growth | the gap grows with categories | delta_alpha rises with K (described directly, not via kappa) | ksweep_table.md | C6 |
| F5 N-sweep | rate-limited not data-limited | gap flat-to-widening with N at fixed budget | ndata_sweep_plot.png | C6 |
| T1 dynamic asymmetry | dynamic helps the low-info readout; beta null | delta_alpha sig for K>=4 (12 seeds, paired p<=0.007); delta_beta ~0 | alpha_beta_asymmetry_stats | C5 |
| T2 attenuation-not-confound | leakage null in both arms | err~theta, err~beta within 1 SD of 0 | shared_alpha_leakage_results.json | C4 |

## 5. Positioning (carry, trimmed to what survived)

- **Build on** Saxe 2014 (rate from curvature) and the amortization-gap idea (Cremer
  et al. 2018: amortized inference is worse than per-instance) — ours is the
  *parameter-recovery* version, non-uniform across interpretable parameters.
- **Affirm** classical IRT estimation (identifiability, consistency, Fisher) — we add
  a property of the amortized estimator, not a correction to the theory.
- **Distinguish from** multi-task gradient conflict (PCGrad/GradNorm): the gradients
  are orthogonal (Phase-2 cos~0); the cost is geometric under-allocation, not
  interference. And from the implicit-bias-of-parameterization line: our maps are
  smooth-monotone and the problem is determined, so the map is not the lever.
- **Closest analog** Salimans-Kingma weight-norm (decouple a gain from a shared
  direction) — they found the symptom and a fix; we give the geometric mechanism and
  tie it to a specific low-information parameter.

## 6. Scope, ceiling, and the running probes

- **Scope:** neural IRT; rate not magnitude; attenuation not confound; K-growth not a
  kappa law. All dynamics evidence synthetic; real-data is a stability proxy (SLAM),
  framed as ordinal corroboration only.
- **Honest ceiling:** a careful characterization of differential recovery in amortized
  IRT plus a measurement-validity caution, with a geometric signature. Workshop /
  measurement-methods scale, not a discovery.
- **Two ceiling-probes running** (could enlarge or further prune, expected to prune):
  - *kappa-identification (done, negative):* with kappa varied 45x at fixed K the gap
    is flat (Spearman -0.3 to -0.5). Conditioning does not survive; the lever is K /
    channel capacity. Dropped.
  - *amortized-encoder generality toy:* a non-IRT model whose shared code is inferred
    per-instance by an encoder — decides whether the phenomenon is general to
    amortized-encoder settings or specific to IRT.

## 7. Experimental rigor standard (unchanged contract)

>=10 seeds for any claim-bearing run; paired across-seed tests and bootstrap CIs;
the geometric probe and the leakage null reported with seed spread; a regularization
(LayerNorm) control to separate genuine width-driven theta variance from
unregularized-substrate decay; every number bound to an experiment and seed count.

## 8. Logistics

- **Venue:** an ML learning-dynamics or measurement-methods workshop (HiLD, M3L, or an
  AIED methods track). Not a main-conference discovery claim on synthetic evidence.
- **Writing order:** positioning (mature) -> theory C1 + the free-table invariant ->
  the geometric-allocation figure (C3, the literal evidence) -> the manipulations
  (C2/C5) -> rate-limited/K-growth (C6) -> the attenuation-not-confound honesty (C4)
  -> scope.
- **Risks/contingencies:** if the kappa-probe is null, drop "conditioning" as a named
  mechanism and keep the geometric-allocation account (it does not need kappa). If the
  encoder toy is null, the claim is explicitly IRT-scoped. Either way the C1-C6 spine
  stands.

## 9. The retired claims (honest record, appendix)

exp-is-special (refuted, smooth-map equivalence); the population-limit law
(downgraded to finite-budget rate); kappa as predictor AND mechanism (the gap is flat
in kappa at fixed K over a 45x range; the K-growth is channel capacity, not Fisher
conditioning); the ability-into-discrimination confound (leakage null); universality
beyond IRT (static-code toy null). Kept as the honest record of what we tested and ruled out.
