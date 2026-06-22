# Paper 2 draft plan (rigorous, panel-integrated)

Supersedes `paper2_outline.md`. A representation-learning / learning-dynamics
paper; neural IRT is the *falsifiable instantiation*, not the subject. Integrates
a four-lens adversarial review (theory rigor, experimental design, novelty,
draft-plan completeness). Every claim carries a status tag; every number is bound
to an experiment and a seed count; the Proved / Argued / Empirical boundary from
`learning_dynamics_theory_support.md` is preserved, not laundered.

Register resolution: the **general** claim is primary (a shared-representation
encoder-decoder with a structurally low-information readout). IRT is the instance
where "low information" has a closed-form Fisher characterization, so we can
*predict in advance* which readout is the slow one. IRT is what makes the general
claim falsifiable, not decoration and not the topic.

---

## 1. The single defensible novel claim

> In a prediction-trained shared-representation encoder-decoder, the slow mode is
> the readout whose Fisher leverage on the prediction is structurally suppressed
> (here discrimination, `I(alpha) ~ (theta-beta)^2 w`, vanishing where responses
> concentrate). Its convergence-rate deficit is a **task-indexed** conditioning
> number `kappa(K) = I(theta)/I(alpha)` that grows provably with the answer-category
> count K. Block-decoupling that readout makes its block `O(1)`-conditioned, buying
> an `O(kappa(K))` rate advantage at no endpoint cost.

The novelty is the **conjunction**, not any single mechanism. We state up front
what each cited lineage already owns, and defend only the composition:

- **It is not Saxe (2014).** Saxe gives one curvature (singular value) per mode and
  no task knob; we add the structural identification of *which* readout is low-Fisher
  and a task index K along which the conditioning provably grows.
- **It is not mirror descent / reparameterization (Li-Wang-Lee-Arora 2022;
  Chou-Maly-Stoger 2023).** That framework gives the map-invariance for free (our
  Claim 4 is its instance). We use it to *rule out* the parameterization as the lever,
  not as the contribution.
- **It is not K-FAC (Martens-Grosse 2015).** K-FAC preconditions the *optimizer*
  over a fixed architecture; our lever is *architectural* (which readouts share a
  code block). The two are provably not the same lever: a diagonal optimizer (Adam)
  *compresses* the effect 2.2x->1.6x (rung-7) but does not remove it; decoupling does.
- **It is not weight-norm (Salimans-Kingma 2016).** That is the closest structural
  analog (decouple a gain from a shared direction), but it reports the symptom and an
  engineering fix; we give the structural reason (representation conditioning) and tie
  it to a *specific* low-information parameter with a closed-form Fisher.

---

## 2. The five claims (numbered, status-tagged)

Status vocabulary from the theory doc: **Proved-local** (under stated assumptions,
near a gauge-fixed optimum), **Argued**, **Empirical**.

- **C1. The low-Fisher readout is the slow mode.** The per-response Fisher is
  rank one with score `s = (alpha, theta-beta, -alpha)`; the (theta,alpha) block's
  small eigenvalue is alpha-aligned because `I(alpha) = E[w (theta-beta)^2]` is
  suppressed where `w` peaks (theta ~ beta). *Status: Proved-local (P2-full, P4a).
  Empirical (recovery-speed curves).*
- **C2. Sharing throttles the discrimination rate by `kappa`; it is a rate effect,
  not a bias, for discrimination.** *Status: Proved-local under A1-A6 (P9b);
  endpoint-invariance Proved for alpha only (P4b free-table invariant). Empirical
  (gate, N-sweep).* The endpoint guarantee does NOT extend to theta (see the
  two-mechanism box).
- **C3. The deficit is task-indexed: it scales with K via `kappa(K)`.** The GPCM
  Fisher gives `kappa(K) = alpha^2 Var_p(k) / Var_p(k.theta - B_k)`, provably
  monotone in K (P6a). *Status: Proved for the Fisher forms (P6a). Empirical for the
  tracking (Spearman 0.891, 10 seeds, K=2..11). The theta-side driver of the
  trade-off frontier is Argued (P10).* This is the headline novel piece (no cited
  lineage is K-indexed).
- **C4. The positive map is not the lever; smooth strictly-monotone maps are
  recovery-equivalent.** *Status: endpoint and rank invariance Proved (P7 i-ii) under
  matched effective-alpha init and per-map LR; speed-equivalence Argued (P7 iii, the
  empirical +-0.002 clustering, not byte-identity). The two exceptions (ReLU dead
  zone `m_g=0`, square non-injective) Proved.*
- **C5. Decoupling the readout removes the bottleneck and replicates across
  encoders.** The decoupled block is `O(1)`-conditioned. *Status: Proved-local that
  the decoupled block is O(1) under A4-A6 (P9b). Architecture replication is
  Empirical: holds on the recurrent and memory encoders; the attention encoder is
  optimization-limited at high K (see section 7).*

### Two-mechanism box (do not collapse; the title must not generalize one to both)

- **Discrimination (alpha): a transient RATE effect** on an invariant endpoint
  (P9, inside the free-table invariant P4b). Vanishes only at the joint
  infinite-data-and-training limit.
- **Ability (theta): a finite-data ENDPOINT / variance effect** on an amortized
  readout (P10, `Var(theta_hat) ~ sigma^2 W/n`), explicitly OUTSIDE the free-table
  invariant. Argued for the encoder (Proved only for a linear amortizer), and
  partly confounded with an unregularized-substrate decay (no LayerNorm) that needs
  a control to separate (see section 6).
- **Difficulty (beta): the indifferent control** (high Fisher, non-pooled; `delta_beta
  ~ +0.003` vs `delta_alpha ~ +0.042`).
- **The Pareto escape (P12) is the synthesis**, and inherits the Argued status of the
  theta arm.

---

## 3. The central theorem and its assumptions

A draft must state one theorem before the experiments, with the assumptions and a
single definition of `kappa`.

**Definition (the one kappa).** `kappa := kappa^sh_j`, the condition number of the
Gauss-Newton Hessian block of the shared item code `e_j`,
`kappa^sh_j = [I(theta)/I(alpha)] . [|v|^2 / (g'_j)^2]`. Two surrogates appear and
must be labeled as such: the **diagonal** `I(theta)/I(alpha)` (what the empirical
K-tracking 0.891 uses; equals `kappa^sh_j` only under A6), and the **coupled 2x2**
`kappa_2 = I_tt/[I_aa(1-rho^2)] >= I(theta)/I(alpha)` (the exact (theta,alpha) block
value, tighter because the off-diagonal coupling worsens it).

**Assumptions** (tag each in the paper):
- A1 local regime near a gauge-fixed identifiable optimum. *(scope)*
- A2 Gauss-Newton = Fisher, exact at a zero-residual optimum, a near-optimum
  approximation otherwise. *(so the rate law is local; the early transient is
  Empirical.)*
- A3 single shared step size (plain GD). *Adam compresses the effect (2.2x->1.6x,
  rung-7); the empirical numbers are the compressed version.*
- A4 free-table expressivity (rank >= K), so the zero-residual fit is reachable
  and P4b applies.
- A5 block-diagonal reduction (cross-item/person off-diagonal Hessian couplings
  subdominant). *Argued. The theorem is conditional on it.*
- A6 matched Jacobian regularity (the code-block top eigenvalue is a constant
  fraction of the trace). *Argued. Reduces `kappa^sh_j` to the diagonal.*

**Theorem (boxed, promoted from P9).** Under A1-A6, for a shared item code feeding
a high-information readout (ability) and a low-information readout (discrimination):
the discrimination direction is the slow eigen-mode of the code block with condition
number `kappa`; plain gradient descent with one step size resolves discrimination in
`t = kappa . log(1/eps)` iterations, versus `O(log(1/eps))` when the readout has its
own block (decoupled); both reach the same fixed point (P4b). Speedup `= O(kappa)`,
tolerance-independent.

**Lemmas consumed:** P2-full (rank-one Fisher + exact eigenvalues), P4b (free-table
invariant), P6a (`kappa(K)` monotone), P7 (map-invariance). State each as a lemma.

**Scope-of-the-rate-law paragraph (required).** Proved for linearized GD near the
optimum under one step size; the transient and the Adam-compressed regime are
Empirical; the theta side is Argued. **Falsifier:** if A6 fails the prefactor
reweights `kappa` but cannot flip its sign or the K-monotonicity.

---

## 4. Figure and table inventory (mapped to existing artifacts)

| Slot | Shows | Headline number | Source | Claim |
|---|---|---|---|---|
| F1 gate | allocation not capacity; decoupled above the frontier at matched size | theta 0.97->0.88, alpha 0.66->0.91; decoupled both-high at 6x budget | `gate_table.md` | C2, C5 |
| F2 trajectory | the good solution reached then left (dynamics) | shared-wide alpha 0.906@ep50 -> 0.787@ep500; decoupled -> 0.912 holds | `trajectory_table.md` | C1 (dynamics) |
| F3 gradient split | mechanism diagnostic: NOT a tug-of-war | theta-pathway grows ~28x, cos(g_theta,g_alpha)~0 | `gradient_conflict_table.md` | C2 (motivates conditioning) |
| F4 K-sweep | the task-indexed rate law | delta_K vs stiffness Spearman 0.891, K=2..11, 10 seeds | `ksweep_table.md` | C3 |
| F5 N-sweep | rate-limited, not data-limited | gap flat-to-widening at fixed budget | `ndata_sweep_plot.png` | C2 (rate) |
| F6 map convergence | smooth maps tie, non-smooth lag | smooth cluster +-0.002; ReLU/square lag | `map_convergence_K4.json` | C4 |
| T1 RQ1 asymmetry | beta is the indifferent control | delta_alpha +0.042 vs delta_beta +0.003; Pearson(delta_alpha,K)=0.877 | `alpha_beta_asymmetry` | C3, control |
| T2 architecture lift (K=4) | decoupling lift on three backbones | alpha shared->decoupled ~0.65->0.92 (LSTM/Transformer/DKVMN) | `swap_table.md` | C5 |
| T3 architecture K-sweep | the K-scaling across encoders | DKVMN ρ=0.90 (replicates); Transformer optimization-limited | `arch_ksweep_dkvmn.json`, `arch_ksweep_transformer.json` | C5, scope |

F3 is a **diagnostic that rules out** the tug-of-war story (cos~0), not a positive
proof of conditioning; the conditioning evidence is the rung-7 GD-vs-Adam control.
Label it that way.

---

## 5. Baseline / ablation matrix

Rows = configs; columns = metrics. Held-fixed vs swept axes explicit.

| Config | theta (static) | theta (drift) | alpha Spearman | beta Spearman | seed std |
|---|---|---|---|---|---|
| shared-narrow | high | - | low (~0.73) | high | report |
| shared-wide | falls (overfit) | - | peak-then-decay | high | report |
| decoupled | high | - | high, holds | high | low |
| decoupled @ matched total capacity | high | - | high | high | low |

Controls and robustness rows: **beta = negative control** (P11, delta ~ +0.003,
recovery ~0.98 everywhere); **matched-total-capacity = the allocation-not-budget
control** (gate); **GD vs Adam** (rung-7, 2.2x->1.6x); **NLL vs WOL** loss-invariance
(rho_WOL 0.903 vs rho_NLL 0.89).

---

## 6. Experimental rigor standard (the contract)

- **Seed floor >= 10** for any claim-bearing run. Currently under-powered and must be
  re-run before submission: the trajectory (C1, 3 seeds), the gradient split (C2,
  effectively 1 SHARED-WIDE run for the 28x), RQ1/RQ3 (3 seeds). The gate (>=5) and
  the K-sweep (10) are adequate.
- **Inferential statistics on every headline number.** Bootstrap or seed-level CIs;
  for decoupled-vs-shared deltas use a *paired* test across seeds (same seed, both
  arms), report effect size + CI, not "9/10 seeds positive." For the stiffness
  correlation, give a permutation/bootstrap CI and acknowledge the small n (~10
  K-points); reconcile the two reported values (5-seed 0.70 vs 10-seed 0.89) with
  their sample sizes rather than quoting the larger.
- **A regularization control** (LayerNorm / q-residual on the encoder) on the
  trajectory and the ability-overfit experiments, to separate the genuine
  width-driven theta variance (the P10 claim) from the known unregularized-substrate
  theta decay (theta 0.97->0.68). Without it the alpha peak-then-decay can be read as
  the same substrate pathology.
- **Per-claim numeric falsifier**, stated as a number not a direction.

### The architecture lift, handled honestly (section 7 of the paper)

- The mechanism (decoupling advantage) is Empirical and replicates on the recurrent
  and memory encoders. The per-encoder K-sweep now exists (T3): **DKVMN replicates
  the K-scaling cleanly (Spearman 0.90, vs LSTM 0.89)**; the **Transformer shows the
  advantage but its K-trend is optimization-limited at high K** (large seed variance,
  several decoupled high-K runs did not converge in budget).
- The decay *magnitude* is encoder-modulated (LSTM ~0.12, Transformer ~0.02, DKVMN
  ~0.01) and this is *expected*: it is the theta-overfit (capacity) term, which the
  two-mechanism split says is an encoder property. State it as a confirmation of the
  split, not a hedge.
- **The attention rerun protocol and falsifier (before any attention claim):**
  (1) confirm the attention encoder reaches LSTM-comparable train/val prediction loss
  at each K (an underfit diagnostic); (2) sweep attention optimization (LR, warmup,
  longer epochs, grad clip) until prediction loss plateaus; (3) only then read
  decoupling delta_K. **Falsifier:** if attention matches LSTM prediction loss but
  delta_K still does not track stiffness, architecture-independence is *refuted for
  attention* and must be reported as such. Until then, the contribution says
  "replicates on recurrent and memory encoders; attention optimization-limited,
  rerun pending," not "architecture-independent."

---

## 7. Positioning (rows to add to the map)

Existing rows (build-on Saxe/Amari/Martens-Grosse; instantiate
Li-Arora/Chou-Maly-Stoger/Amid-Warmuth; contrast Woodworth/Gunasekar/Vaskevicius;
analog Salimans-Kingma/van Laarhoven). Add:

- **Amortized-IRT parameter recovery (the instantiation is non-trivial).**
  Tsutsumi EDM2021 (hardcodes alpha), VIBO EDM2020 (ability-only recovery),
  Urban-Bauer 2021 (loadings, no coupling/decoupling analysis), JE-IRT 2025
  (maximally-shared geometry, no scalar alpha). This is why "discrimination is the
  structurally low-Fisher readout" is a non-obvious choice, not arbitrary.
- **Multi-task / auxiliary-head gradient dynamics (the boundary).** GradNorm,
  Standley et al., PCGrad. Draw the line: this is NOT gradient conflict (cos~0,
  Phase 2 refutes the tug-of-war) and NOT pure magnitude imbalance; the rate penalty
  is the *eigenvalue spread of the shared-code Hessian block* (P9b), which survives
  orthogonal pathway gradients. That distinction is the delta over PCGrad/GradNorm.
- **The "rate is just slow convergence" rebuttal (pre-empt it).** The rate effect is
  (i) on the metric the deployed tracking model is judged on (rank at a finite
  budget), (ii) does not narrow with data at fixed budget (N-sweep, flat-to-widening),
  (iii) worsens with the task knob K. "A rate effect that does not vanish with data
  and worsens with task complexity at any practical budget" is the framing.

### External corroboration (section 7), stated honestly

- **Misspecification taxonomy:** the robust message is *better prediction does not
  imply valid recovery* (prediction-recovery dissociation). Report it by violation
  class, do NOT reduce to "discrimination collapses first": beta collapses under
  threshold disorder, alpha under exposure/response-style, both under DIF, recovery
  survives under drifting-theta. The clean cross-cut is the dissociation, not a single
  parameter.
- **Real-data SLAM stability** (difficulty 0.89x vs discrimination 0.83x of a 16-seed
  reliability ceiling) is a **stability proxy, not ground-truth recovery**. Frame it
  strictly as ordinal corroboration of one prediction (difficulty more stable than
  discrimination). Do not let it stand in rhetorically for recovery.
- **What would constitute real recovery (named future experiment):** items
  pre-calibrated by classical MML/MCMC IRT on a held-out large sample, then check the
  neural decoupled-vs-shared discrimination recovers the classical alpha ordering on
  an error-rich graded corpus (the data proposal targets exactly this).

### The K-vs-classical reconciliation (pre-empt the psychometrics reviewer)

We claim K worsens the theta-vs-alpha *training conditioning* (a dynamics quantity),
NOT that K improves static per-parameter estimability (it does not; per-category
dilution stands, threshold info falls 0.187->0.111 with K). The two point opposite
ways; we claim only the former.

---

## 8. Scope: minimal-viable vs full

- **MVP (the core, ~workshop length).** Claims C1, C2, C5 on the recurrent encoder:
  the gate (F1), the trajectory (F2), the gradient split (F3), the decoupling fix,
  with the P4a/P9 theory and the boxed theorem. This is the self-contained
  "conditioning sets the discrimination rate; decoupling fixes it" result.
- **Full / appendix / extension.** C3 (the K-scaling, F4, T3 + the GPCM Fisher), C4
  (map-invariance, F6 + P7), the architecture lift (T2/T3), the RQ3 detector (scope
  decision below), the misspecification + SLAM corroboration, and the honest record
  of the two retired claims.

**RQ3 scope decision (must be explicit).** State-conditioned discrimination is a
directional detector with ~30x magnitude under-identification. Decision: fold it
into section 7 as "the same low-Fisher leverage under-identifies the magnitude,
recoverable only in rank," tied to the corroboration. Do not leave it silently out.

---

## 9. Logistics

- **Target venue.** The result is workshop-shaped (the study's own words). First
  home: an ML learning-dynamics workshop (HiLD, M3L, or a Science-of-DL workshop),
  4-8 pages. A main-conference version waits on the real-data recovery experiment.
  The AIED/psychometric framing is a *separate* paper (the thesis anchoring chapter),
  not this one.
- **Writing order (dependencies).** (1) Positioning map (mature, write first).
  (2) Theory section 3 + the boxed theorem (must precede the experiment captions so
  "theory before experiments" is real, not retrofitted). (3) The figure/table
  inventory and ablation matrix (binds claims to artifacts). (4) Experiments, written
  against the inventory. (5) Section 7 corroboration, gated by the RQ3 scope decision.
- **Risk / contingency register.**
  - Attention rerun fails or stays unreadable -> fall back to "recurrent + memory
    encoders; attention deferred," and keep the contribution as the two-encoder
    replication.
  - Reviewer demands real-data ground-truth recovery -> point to the named
    classical-IRT-anchored future experiment and the stability proxy as the available
    evidence; do not overclaim.
  - Reviewer attacks single-architecture RQ1-3 -> cite the swap bench (T2) for the
    spine's architecture-independence.
  - A6 challenged -> the falsifier line (reweights but cannot flip sign or K-trend).

---

## 10. The "intentionally omitted details" checklist (fill before drafting)

These were left out of the outline and must be pinned in the paper or its appendix:

- The exact assumptions A1-A6 of the theorem (section 3 above).
- The gauge-fixing convention (theory doc section 8) and the definition of "recovery"
  (sign-aligned Spearman, gauge-invariant).
- The loss form: cross-entropy / NLL and the WOL variant, and the CE-family scope
  boundary (the invariance is claimed only for the softmax-CE family; EMD / margin /
  aggressive ordinal penalty are untested, flagged).
- The GPCM step-threshold (B_jk) parameterization.
- The data-generating priors: N=800, Q=60, T=60, alpha ~ LogNormal(0,0.3), static
  ability; and that all dynamics evidence is synthetic.
- Seed counts per result, with the under-powered ones (RQ1/RQ3 at 3 seeds) labeled
  preliminary and the robust-signal-first ordering adopted (endpoints, beta null,
  K-correlation, the 10-seed stiffness Spearman).

---

## 11. Honest scope (carried verbatim, not softened)

- All dynamics evidence is synthetic; the real-data leg is a stability proxy, not
  ground-truth recovery.
- No population-limit law; the effect is finite-budget (the strong law was refuted
  and downgraded).
- The exp-is-special hypothesis is refuted (smooth-map equivalence) and kept only as
  the honest record.
- The theta side (P10) is the softest formalization: Proved for a linear amortizer,
  Argued for the encoder, no closed-form over-training trajectory; the Pareto frontier
  inherits this Argued status on the theta arm.
- The attention-encoder K-scaling is optimization-limited and pending the rerun
  protocol in section 6.

---

## Discipline carried into the writing

- Theory and predictions written as if they preceded the experiments; theory curves
  not fitted to data.
- Every threshold/peak claim checked against a continuous progress metric (Schaeffer
  2023).
- General claim primary, IRT the falsifiable instantiation; gloss every term; no
  jargon pile-up in either direction.
- Every claim tagged Proved-local / Argued / Empirical; every number bound to an
  experiment and a seed count.
