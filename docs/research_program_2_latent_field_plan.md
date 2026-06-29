# Distributed Latent Field over Anchored Concepts (Program II)

> Cross-references: `docs/GOAL_identifiability_audit.md` (charter), `docs/PROGRESS_identifiability_audit.md` (live tracker).

Final research plan. Revised to address the adversarial review by down-scoping
where the critique landed, adding the no-training and easiest-synthetic
fast-fails, fixing the identifiability and control story, and narrowing the
novelty claim to what survives PSI-KT. Updated 2026-06-29 per the post-gate section below.

---

## Post-gate update (2026-06-29)

Gates G1/G2/G3 and the Leg B fast-fail (Stage -1 Fisher) are all done. Three points update the plan.

**Leg B is an audit, not a measurement-geometry model.** The goal is to measure the conditional Fisher retention eta of the propagation operator P under prediction-loss training. The architecture in sections 4 and 5 is the audit instrument, not a thesis contribution in its own right. No learned-operator model is proposed or delivered.

**Correction: the recovered-vs-dominated split is readout vs operator, not diagonal vs off-diagonal.** The Stage -1 Fisher calculation (`deep_irt/bench/_stage_minus1_fisher_P.py`) found eta(P_diag) and eta(P_off) are indistinguishable, both approximately 0.04 to 0.13 in the realistic regime. The whole operator P is the dominated party. The recovered quantity is the readout alpha (discrimination), with eta approximately 0.27, the same Paper 2 channel. The mechanism is collinearity: the Q-masked ability signal in any one response is nearly collinear with the information needed to pin individual P entries, whether diagonal or off-diagonal.

**Rescue regime as a boundary.** Strong anchoring (gain approximately 2 to 8) crossed with a decorrelating (source-isolation) curriculum lifts eta to a ridge at approximately 0.32. This is a boundary condition, not a recipe. Past anchor strength approximately 10, P identifiability collapses. Buying coupling identifiability costs the discrimination channel (eta(alpha) drops from approximately 0.27 to 0.13). Rescue is reported as a boundary, not a recommended operating point.

**Diagonal-P null is the fabrication control.** A diagonal true P must yield off-diagonal detection AUC near 0.5. This is H1's decisive control and is not a real-data null.

**ELBO foil arm.** PSI-KT's operator is reimplemented as a comparison arm in the generator (AGPL-safe, no vendored code from `mlcolab/psi-kt`). Real-data PSI-KT runs are isolated and cited only. The foil tests whether generative targeting (ELBO) clears the coupling recovery boundary where prediction loss does not.

**Integration.** Leg B is the dimensional axis of the identifiability audit. It shares the three-metric diagnostic (marginal Fisher, conditional retention eta, reproduce-vs-recover gap) and the unified generator with Leg A (temporal axis). Synthesis in `docs/GOAL_identifiability_audit.md`.

**Guardrail.** No proposed learned-operator model, no concept-graph or prerequisite-graph deliverable, no learner-trait discovery. Rescue regimes are reported as boundaries only. PSI-KT owns the generative modeling; this work owns the audit and the boundary.

---

## 0. Framing and the one constraint that defines the contribution

The **audit object** is the propagation operator P among anchored concepts.
The plan does not propose a learned-operator model; it measures whether P
survives prediction-loss training and under what conditions. Recovery of a
**known** P against a **diagonal-P null** is the primary experimental control.
Anchoring (a fixed Q-matrix) removes rotational and sign indeterminacy so P
becomes a named object, and the IRT field on anchored slots makes per-concept
mastery readable.

**The binding prior-art constraint, do not claim architecture novelty.** PSI-KT
(Zhou et al., ICLR 2024, `mlcolab/psi-kt`) already unifies anchored per-concept
states, a learned directed propagation operator, a gated OU transition
`m_n = r_n z_{n-1} + (1-r_n) mu_tilde_n`, and background propagation
`mu_tilde_n = mu_n + (gamma/K) sum_{i!=k} a^{ik} z^i`, fit by variational Bayes.
That is, in substance, this design. The critique is correct that
`P = (I + cL)^{-1}` with a learned Laplacian L **is a learned propagation
adjacency trained end to end**, the same operator class as GKT-learned and
PSI-KT's background term. **The "P is a different kind of object than a
knowledge graph" rhetoric is dropped.** The only defensible daylight, which
must be earned and not asserted, is exactly three things no prior work does:

1. **Recoverability under prediction loss, not ELBO.** PSI-KT fits the
   generative model directly and never asks whether P survives a
   prediction-trained encoder. Our Paper 2 lens predicts it will survive in
   rank but not in magnitude, P being the lowest-Fisher object in the model.
   That prediction, made precise and tested, is the result.
2. **Identifiability as the headline**, with a general signed P (PSI-KT
   constrains to a skew-symmetric prerequisite DAG), a **known-P synthetic**
   ground truth, and a **diagonal-P null** that PSI-KT never runs.
3. The **gauge-free, Fisher-leverage analysis** of the dynamic matrix flow,
   carried from `docs/LEARNING_DYNAMICS_STUDY.md`, the lab's actual
   differentiator.

**The honest sizing the critique forces.** Stage 1 neural recovery succeeding
(rank up, magnitude attenuated) is Paper 2's discrimination result transposed to
a multidimensional setting. It is **confirmatory, not a discovery**. The
genuinely new knowledge is concentrated in two cheap early checks (the Fisher
calculation and the easiest-synthetic falsifier), and the program is sized
around them, not around a real-data headline. Everything below is engineered so
that even if the real-data leg yields only existence detections (likely, see
risks), the synthetic stages deliver a clean identifiability and recoverability
methods contribution that is novel against PSI-KT and graph-KT.

**The structural limit stated up front, not as a tunable risk.** The estimand is
P under a **scalar error-driven gain**. A scalar gain means every concept an
item touches moves by the same scalar, so any world with **per-concept learning
gains** (the real one) is misspecified by the identifiable model. You can have
an identifiable P or a per-concept-realistic update, not both. The plan treats
this as a fixed property of the estimand and measures its cost (H3b), rather
than pretending it can be tuned away.

---

## 1. Premise and the precise scientific question

A freely learned multidimensional ability inflicts rotational and sign
indeterminacy, so the cross-concept coupling in a free MIRT is not a
recoverable object. Anchoring the axes to known concepts through a fixed
Q-matrix removes that indeterminacy and turns the coupling into a named,
potentially recoverable operator.

**Q.** Under prediction-loss training of a neural IRT model whose latent ability
field is anchored to known concepts (fixed Q-matrix, no rotation), is the
cross-concept **propagation operator P**, the coupling by which practice on a
loaded concept perturbs the mastery estimate on non-loaded concepts, a
**recoverable, identifiable** estimand, and what is recoverable, in rank or in
magnitude, under what identifiability conditions, and at what cost when the
identifying scalar-gain assumption is violated?

---

## 2. Hypotheses

**H0 (Fisher precondition, a calculation, no training) -- DONE.** The conditional
retention eta for P entries is approximately 0.04 to 0.13 in the realistic
regime, with eta(P_diag) and eta(P_off) indistinguishable; eta(alpha) is
approximately 0.27. Both diagonal and off-diagonal P fall well below alpha in
retention, confirming the magnitude claim is dead and the program is
**rank-only** with magnitude attenuation pre-registered as expected. This result
**precedes Stage 0** and was the fast-fail gate.

**H1 (well-posedness and the SNR threshold).** An oracle joint-MLE that knows Q
and the state-space form recovers known P entries (diagonal and off-diagonal) on
the observable bilinear forms `a_q'^T P a_q`, and a known **diagonal** P_true
yields near-zero recovered off-diagonals. The identifying observable for `P_{c'c}` is
second-order small, `a_q'^T P a_q ~ P_{c'c} a_{q'c'} a_{qc}`, so the threshold is
**the oracle's Cramer-Rao bound, not an arbitrary 0.9**. *Falsifiable: oracle
recovery reaches the CR bound on recovery-vs-N and recovery-vs-|P_offdiag|
curves, and the diagonal-P off-diagonal detection AUC ~ 0.5.* A failure here is
ill-posedness or coverage, almost always a curriculum-support problem, and is
the cheapest kill switch after H0.

**H2 (neural recovery, rank-not-magnitude, CONFIRMATORY).** A prediction-trained
`AnchoredMIRTModel` recovers P entries **in rank/sign** above both the
diagonal-P null and a shuffled-sequence floor, with **attenuated magnitude**,
the multidimensional analog of Paper 2's discrimination-magnitude collapse.
*Falsifiable: P-entry Spearman vs truth > null floor at Wilcoxon p < 0.01
across 5 seeds, magnitude attenuation pre-registered as expected, not a
failure.* This confirms a known mechanism in a new setting and is sized as a
methods result, not a discovery.

**H3 (identifiability conditions are load-bearing).** Removing any one
constraint breaks recovery in the predicted direction. A **vector** update gain
destroys P-recovery by absorbing its routing. A **free** (non-PSD) P diverges or
oscillates on data directions where `a_q^T P a_q < 0`. Degrading curriculum
connectivity concentrates recovery error on the un-traversed concept-pair
entries. *Falsifiable: each ablation degrades the H2 metric in the named
direction, and the coverage error concentrates on off-support P entries.*

**H3b (the scalar-gain misspecification cost, a measured structural limit).**
Generate data with **per-concept** learning gains (the realistic world), fit the
identifiable **scalar-gain** model, and measure how far P-recovery degrades
relative to the scalar-gain-matched generator. *Pre-registered: this quantifies
the structural limit named in section 0. It cannot be tuned away, only
reported.* The recovered P is interpreted as "coupling under a scalar-gain
approximation," never as the unconstrained per-concept transfer.

**H4 (separability vs dynamic-flow tension).** A state-conditioned P (the natural
dynamic-matrix-flow design) degrades per-concept separability relative to an
item-only deterministic P, replicating the `ednet_sep` finding that
separability needs an item-only readout. The recovered-vs-propagated confound (P
rescues a starved low-Fisher dim and inflates apparent recovery) is caught only
by the diagonal-P null. *Falsifiable: separability index drops under
state-conditioning, and the diagonal-P null catches any fabricated propagation.*

**H5 (real data, exploratory, positive-detection-only, pre-registered).** On
KDD Cup 2010 (first-attempt correctness, a chosen KC model as Q-matrix), the
real leg may claim only that **off-diagonal coupling EXISTS on a co-practiced
concept pair** above a size-matched placebo, using the rq8 range-restriction-
aware protocol restricted to traversed pairs. *Pre-registered scope: the real
leg makes positive detection claims only. It makes NO null claim and does NOT
recover the named-concept propagation graph (see Risk 1 and 2). A clean detector
on a well-traversed pair is the ceiling of what real data can deliver.*

---

## 3. Integration with prior work and prior-art positioning

### 3.1 Reuse, do not rebuild

| Asset | Where | Role here |
|---|---|---|
| `BaseSeqEncoder` swap contract, shape-generic `_shift` | `core/encoder.py:86-197` | Base for the field encoder; `_shift` is documented to handle `(B,T)` and `(B,T,H)`, so a `(B,T,K)` field rides the causal alignment unchanged |
| `_ENCODER_CHOICES`/`_DECODER_CHOICES` + staticmethod factories | `core/model.py:130-131,238-257` | Verified injection point, no Codex-owned file is edited |
| `DeepIRTModel.fit` loop | `core/model.py` | Reused, only the scalar-theta methods overridden |
| GPCM cumulative-logit math | `core/decoders.py:435-463` | Math reference for the reimplemented decoder, not imported and mutated |
| ma-irt multidim GPCM (`n_traits`) | `ma-irt/models/components/irt.py` | Frozen math reference only, not importable |
| GPCM cumsum sampler, lognormal a, sorted-normal b, theta_traj container | `bench/datagen.py` | Read-only template for the new generator |
| sign-aligned recovery metrics | `bench/metrics_bench.py:55-147` | Per-concept recovery once anchoring fixes axes (sign-only alignment) |
| existence-gate methodology | `traj_kt/run_e2c.py:418-515` | Gate for "does coupling exist" before any magnitude claim |
| rq8 range-restriction-aware DIF protocol | `deep_irt/` (rq8) | Real-data invariance test (matched placebo, link-free Spearman, per-item delta-b) |
| `ednet_sep` separability index | `deep_irt/ednet_sep/` | H4 separability metric and the item-only-readout design principle |

### 3.2 Avoid re-deriving, especially the dynamic-theta and Paper 2 results

- **Paper 2 (`docs/LEARNING_DYNAMICS_STUDY.md`) low-Fisher recovery is the whole
  H2 risk, transposed.** P has near-zero leverage on the current prediction in
  both its faces: off-diagonal (the current item does not load the perturbed dim)
  and diagonal (theta absorbs the within-concept variance). Both arise from
  collinearity. The recovered quantity remains alpha at eta approximately 0.27; P
  sits at approximately 0.04 to 0.13. **Do not present rank-not-magnitude
  P-recovery as a finding.** It is the predicted shape. The new content is the
  diagonal-P null (does the estimator fabricate) and the oracle-vs-neural gap at
  the CR bound.
- **The dynamic-theta / trajectory program (`docs/trajectory_findings.md`) is the
  saturation precedent.** The per-student rate had no external validity on
  near-saturated logs (split-half 0.17). The off-diagonal P effect is
  second-order on top of that same saturation, strictly harder than the rate.
  **Do not re-attempt a magnitude claim on real data.** The real leg is
  existence-detection only (H5). This is the explicit overlap the task flags
  under "esp. dynamic-theta": Program II shares the trajectory program's root
  blocker, so it inherits the trajectory program's verdict on real data and does
  not relitigate it.
- **`ednet_sep` already tells us the dynamic head re-breaks separability.** Do
  not assume Paper 2's dynamic-discrimination fix transfers. Default to item-only
  P and re-check separability when state-conditioning (H4).

### 3.3 Prior-art positioning, named methods and the novelty gap

- **vs DKVMN / Deep-IRT (`jennyzhang0215/DKVMN`).** They never anchor (latent
  exchangeable slots, rotation not removed) and read a scalar ability. Anchoring
  plus a per-concept IRT field is a gain over them, but that gain is the
  NCDM/PSI-KT move, not ours.
- **vs NeuralCD / NCDM / KaNCD (`bigdata-ustc/EduCDM`).** Canonical Q-matrix
  anchoring, but static, no sequential dynamics, off-concept terms zeroed rather
  than propagated. Our sequential field with off-diagonal propagation beats them
  on dynamics, but the dynamics-plus-propagation part is PSI-KT/GKT territory.
- **vs GKT / GIKT / SKT (graph-KT).** **Same operator class.** A learned
  Laplacian propagation adjacency is exactly what GKT-learned does. We do **not**
  claim a different kind of object. The daylight is the recovery question, no
  graph-KT paper recovers a known P against a diagonal-P null under prediction
  loss with gauge-free metrics.
- **vs PSI-KT (`mlcolab/psi-kt`, ICLR 2024), the binding competitor.** Identical
  gated form and background propagation, but fit by ELBO with a skew-symmetric
  prerequisite DAG. **The residual, and the entire contribution, is:**
  recoverability under prediction loss, identifiability as the headline with a
  general signed P and a known-P synthetic and a diagonal-P null, and the
  gauge-free Fisher-leverage analysis of the matrix flow. **Read PSI-KT in full
  before locking framing.**

**One-line novelty statement.** Same operator class as graph-KT and PSI-KT, a
novel **recovery and identifiability** question, is P recoverable from a
prediction-trained encoder, in rank or magnitude, under what gauge and coverage
conditions, and at what cost under scalar-gain misspecification.

---

## 4. Architecture on deep_irt and the baseline to beat

All new files live in a parallel package `deep_irt/anchored_field/`. **No
Codex-owned file is edited** (`core/model.py`, `core/decoders.py`,
`bench/datagen.py`, `engines.py` are read-only). Extension is by subclassing,
confirmed against the code.

```
deep_irt/anchored_field/
  qmatrix.py        # Q-matrix loader; anchor-item selector (one pure item/concept); masked-loading builder
  field_encoder.py  # AnchoredFieldEncoder(BaseSeqEncoder): vector Theta state + scalar Delta + Laplacian-P update
  decoder.py        # AnchoredMIRTDecoder: compensatory M-GPCM logit z = a_j^T Theta - b
  model.py          # AnchoredMIRTModel(DeepIRTModel): overrides factories + the 4 scalar-theta methods; owns P + P-reg
  datagen_field.py  # VAR-with-practice-input generator: known A, known sparse anchored loadings, known PSD P
  metrics.py        # gauge-fix via anchors; P-recovery on bilinear forms; off-diag AUC; per-concept recovery; Fisher calc
  run_anchored.py   # driver: Fisher calc -> oracle -> neural recovery -> ablations -> exploratory real
```

**Injection mechanism (verified).** `AnchoredMIRTModel(DeepIRTModel)` sets
`_ENCODER_CHOICES = DeepIRTModel._ENCODER_CHOICES + ("anchored_field",)` and
`_DECODER_CHOICES += ("anchored_mirt",)`, and overrides the staticmethod
factories `_make_encoder`/`_make_decoder`. `__init__` (model.py:238,252) picks
them up unchanged.

**Backbone for v1, plain vector-theta LSTM (DKVMN-anchoring deferred).** Subclass
`BaseSeqEncoder`, keep `_direct_hidden` for the black-box hidden, replace the
scalar readout with a **vector field** `Theta_t in R^K` via `theta_proj:
hidden -> K`. Override `aligned_theta_and_state` to return `(Theta_in: (B,T,K),
state: (B,T,hidden))`. Per step, item j_t with Q-row mask `m_{j_t} in {0,1}^K`:
- **scalar error-driven gain** `Delta_t in R` (a scalar head of the state, never
  a vector, the identifiability fix and the structural limit of section 0);
- field update `Theta_t = A Theta_{t-1} + P (m_{j_t} * Delta_t)`, `A = diag(rho)`
  persistence;
- **P = (I + cL)^{-1}** with L a learned graph Laplacian (PSD by construction,
  satisfies the quadratic-form stability bound `0 < a^T P a < 2/(eta sigma')`,
  off-diagonals of L are the interpretable propagation graph). Default readout is
  **item-only / deterministic** (H4), a `state_conditioned_P=True` ablation routes
  c (or L) through the state.

The DKVMN-with-fixed-Q-addressing realization (anonymous content-softmax slots
replaced by a fixed Q indicator, P as cross-slot mixing in `_write`) is real
surgery and its own contribution. It is **cut from v1** and noted as future
work. The plain vector-theta LSTM is sufficient for the recovery question.

**Decoder (`decoder.py`).** A new module (cannot edit `decoders.py`).
`item_params` returns loadings `a_j` **pinned by the Q-mask**
(`a_jk = m_jk * exp(0.3 raw_jk)` on loaded dims, exactly 0 off, the ma-irt trick)
and step thresholds `b`. `logits` computes the compensatory M-GPCM projection
`z = a_j^T Theta - b` then the existing GPCM cumulative-logit cumulation,
reimplemented from the math at `decoders.py:435-463`. Exposes
`item_params` / `logits_from_emb` / `log_probs` / `nll` to satisfy existing call
sites. Compensatory only in v1, the non-compensatory (smirt-style product)
variant is a robustness note, not a parallel matrix.

**Model (`model.py`).** Reuse `DeepIRTModel.fit`. Override the four scalar-theta
methods, `_compute_loss` (no `reshape(B*T)` collapse, carry `(B,T,K)`),
`_predict_loss`, `track` -> `(N,T,K)`, `recover_item_params` -> loadings + P (+ L).
Owns P and the P-regularizer (an L1 on off-diagonals for **parsimony only**,
explicitly not claimed as identifying).

**Explicit baseline to beat.** **Anchored independent concepts, diagonal P
(P = I, c = 0).** The classical compensatory-MIRT-over-time null. The
contribution holds only if (i) the full model recovers the known off-diagonal P
on synthetic above this null, and (ii) the null does **not** fabricate
off-diagonal structure on diagonal-P-true data. Secondary baselines: a frozen
**GKT-style given-graph** P (co-occurrence adjacency, not learned) to show the
gain is the learned identifiable operator, and the **oracle joint-MLE** (the
positive ceiling and the Stage-0 well-posedness check).

---

## 5. Data: synthetic ground truth first, then real, saturation-aware

Per the synthetic-MIRT literature, no package ships a temporal MIRT generator,
the standard recipe is "simulate Theta yourself, sample responses per slice"
(`mirt::simdata` / `girth` own only the per-step draw). The generator owns the
dynamics.

**Synthetic positive control (`datagen_field.py`), built first.** A
VAR-with-exogenous-practice-input state space.
- Known **sparse anchored loadings** A (Q-pattern plus positive lognormal
  magnitudes), **with one pure anchor item per concept** (loading e_k) so the
  MIRT gauge M is pinned.
- Known **PSD P_true**, a normalized heat kernel `exp(-cL)` on a small concept
  graph (K = 3 to start, up to 8). A diagonal-P_true variant for the null.
- **Scalar** error-driven Delta in the generator (all cross-concept routing lives
  in P), plus a **per-concept-gain variant** for the H3b misspecification cost.
- A **connected curriculum** that interleaves concepts so learners who practice
  source concept c are later probed on items loading target c'. P is identified
  only on the transition-graph support, a never-co-practiced pair carries zero
  information about its P entry.
- Compensatory M-GPCM responses, matching the scalar projection deep_irt reads.
  A non-compensatory variant is a robustness note only.
- A **saturation-matched variant**, high baseline-correct, low dynamic range (the
  EdNet/KDD regime), to locate the operating point and test what survives.
- Outputs ground truth `Q, A (K-loadings), P, A_persistence, theta_field (N,T,K)`.
- Sizes for the 8GB GPU, N = 2000 to 5000 learners, T = 50 to 200 steps, K <= 8.
- **R cross-check**, write each time slice's Theta to `mirt::simdata(a, d,
  Theta=..., itemtype='gpcm')` and confirm response marginals match the GPCM
  sampler. Cheap external validation of the sampler.

**Real data, exploratory and positive-detection-only.** On disk: EdNet-KT1
(binary, single-pass, about 80% top category, no Q-matrix) and **KDD Cup 2010
Algebra** (step-level, KC tags as a real Q-matrix). EdNet is unusable here, no
concept map and saturated. **KDD is the only viable substrate**, a
moderate-granularity KC model (e.g. KC(SubSkills)) as the Q-matrix and
first-attempt correctness as the error-rich channel. **KDD has no pure anchor
items**, so M is not pinned and `P_hat` carries an unknown congruence `M P M^T`.
The bilinear forms `a_q'^T P a_q` are gauge-invariant, so the real leg can detect
**whether** coupling exists on co-practiced pairs but **cannot** recover the
named-concept graph. The named graph is a **synthetic-only deliverable**. The
real leg carries both blockers at once (near-saturation plus a second-order
effect), so H5 is exploratory by construction and restricted to co-practiced
pairs, positive detections only.

---

## 6. Evaluation per hypothesis, with null and identifiability controls

All recovery metrics are **gauge-free** (Paper 2 discipline), per-concept
sign-aligned Spearman/Pearson after the anchor gauge-fix, never raw entries
before M is pinned.

| H | Primary metric | Controls / null |
|---|---|---|
| H0 | `I(P_ij)` under the planned curriculum vs `I(alpha)` from Paper 2 | None, a closed-form calculation; pre-registers rank-only if `I(P_ij)/I(alpha) << 1` |
| H1 | Oracle recovery of P_hat vs P_true on observable bilinear forms `a_q'^T P a_q`, scored against the **Cramer-Rao bound**; recovery-vs-N and recovery-vs-|P_offdiag| curves; eigenstructure of L_hat | Diagonal-P_true must give off-diag detection **AUC ~ 0.5** (the decisive "registers absence" control) |
| H2 | P-entry Spearman vs truth (rank), 5 seeds, mean +/- 95% bootstrap CI | Diagonal-P null floor; shuffled-sequence negative floor; oracle joint-MLE positive ceiling; report the magnitude attenuation ratio (expected per Paper 2) |
| H3 | H2 metric under each ablation, scalar-vs-vector Delta, PSD-vs-free P, curriculum-connectivity sweep | Each must degrade in the predicted direction; coverage-sweep error must concentrate on off-support P entries |
| H3b | P-recovery under the scalar-gain model on per-concept-gain data vs scalar-gain-matched data | The matched generator is the ceiling; the gap is the reported misspecification cost |
| H4 | Separability index (`ednet_sep`) for item-only vs state-conditioned P; per-concept reliability | Diagonal-P null separates recovered-from-propagated; split-half reliability of the field |
| H5 | Off-diag P **existence** on co-practiced KDD KC pairs vs a size-matched placebo (rq8 link-free Spearman + per-item delta-b), traversed pairs only | rq8 range-restriction-aware protocol; **no null claim, no named-graph claim** |

**Dynamic matrix flow, defined rigorously** (kept verbatim from the draft, the
critique endorsed it). Report the well-posed objects, the discrete
state-transition (fundamental) matrix `Phi(t,0) = prod_{s<=t} J_s` with
`J_s = I - eta sigma'_s P a_{q_s} a_{q_s}^T`, its top Lyapunov exponent (the
stability invariant), and the cross-concept cumulative leakage
`L_ij = sum_t P_ij a_{j,q_t} Delta_t`. **Two-clocks discipline**, the theta-field
flow is a response-step object, item-parameter recovery is a training-epoch
object (the Paper 2 axis) unless items are state-conditioned. Do not conflate
the two clocks.

**Seed and CI protocol.** Synthetic, 5 seeds (data + init), mean +/- 95%
bootstrap CI (1000 resamples over items and learners), paired Wilcoxon
signed-rank for model-vs-null. Real, 3 seeds, same CI and test.

**Cross-encoder invariance is cut from the main result.** P-recovery across
lstm/transformer/dkvmn is an **appendix or a later paper**, not load-bearing for
an identifiability contribution. v1 uses one backbone (the plain vector-theta
LSTM).

---

## 7. Milestone ladder, MVP to full, each with go/no-go and fast-fail

**Stage -1, the Fisher calculation -- DONE.** eta(P_diag) and eta(P_off) both
approximately 0.04 to 0.13; eta(alpha) approximately 0.27. Magnitude claim is
dead; program is pre-registered **rank-only**. Artifacts:
`deep_irt/bench/_stage_minus1_fisher_P.py`, `_stage_minus1_fisher_matrix.py`.

**Stage 0, generator + oracle identifiability (days, the kill switch).** Build
`datagen_field.py` (known A, known P, K=3) and the oracle joint-MLE that knows Q
and the state-space form. *Go:* oracle recovery reaches the CR bound on the
observable bilinear forms AND the diagonal-P null off-diag AUC ~ 0.5. *No-go:* if
the oracle cannot recover P, the estimand is ill-posed (almost always a coverage
problem), fix the curriculum or redesign before any neural training. **Stage 0
proves the estimand is well-posed, it does NOT prove prediction loss recovers
it.**

**Stage 1, the MVP and the true neural falsifier (days to 1 wk).** Train
`AnchoredMIRTModel` (plain vector-theta LSTM) on the **easiest possible**
synthetic, K=3, strong dynamic range, connected curriculum, scalar-gain-matched,
one backbone. *Go:* off-diag P recovered in rank above the diagonal-P null and
the shuffled floor (Wilcoxon p < 0.01, 5 seeds), magnitude attenuation
pre-registered as expected. *No-go:* if rank fails on the easiest synthetic, the
prediction-loss claim is dead, stop and write the negative as a Fisher-leverage
limit result. Run the saturation-matched variant immediately after to locate the
operating point. **This is the MVP, and the earliest experiment that can falsify
the core claim with training.**

**Stage 2, identifiability ablations, structural limit, separability (1-2 wks,
the publishable methods core).** scalar-vs-vector Delta (H3), PSD-vs-free P (H3),
curriculum-connectivity sweep (H3), the per-concept-gain misspecification cost
(H3b), item-only-vs-state-conditioned P (H4). *Go:* scalar-Delta + Laplacian-P is
the stable identifiable config, each ablation breaks recovery in the predicted
direction, and the misspecification cost is quantified. **This is the
contribution even if Stage 3 yields only detections.**

**Stage 3, real data, exploratory (1-2 wks).** KDD Cup 2010, KC-model Q-matrix,
first-attempt correctness. *Soft go:* a clean positive detection of coupling on a
co-practiced pair above the placebo, reported as existence, with the named-graph
and null framings withheld. *No-go-as-result:* no detection survives, reported
honestly as consistent with the saturation blocker, **not** as evidence the
world is diagonal. The methods contribution stands entirely on Stages -1 to 2.

---

## 8. Risks and de-risking

1. **PSI-KT priority (architecture not novel).** De-risk, reframe the
   contribution as recoverability-under-prediction-loss plus identifiability plus
   the gauge-free analysis, cite PSI-KT as the nearest analog, drop the
   "different object" rhetoric. Read PSI-KT in full before locking framing.
2. **P is the lowest-Fisher object (magnitude will not recover).** Mapped by
   Paper 2 and now made precise by H0. De-risk, pre-register rank-not-magnitude,
   report P as a directional detector, bracket with the oracle ceiling and the
   diagonal-P null.
3. **Anchoring does not pin the basis on real data (fatal to the named-graph leg
   on real data).** KDD has no pure anchor items, so M is unpinned and `P_hat`
   carries `M P M^T`. De-risk, **cut the named-graph deliverable to synthetic
   only**, restrict the real leg to gauge-invariant existence detection on
   bilinear forms.
4. **A real-data null is not interpretable.** It confounds genuine diagonality,
   low-Fisher attenuation, unpinned gauge, and missing coverage. De-risk, **the
   real leg makes positive detection claims only, never a null claim** (H5).
5. **Scalar Delta guarantees misspecification (a structural limit, not a risk).**
   An identifiable P forces a scalar gain, which misspecifies any per-concept-gain
   world. De-risk by measuring not hiding, the H3b misspecification-cost
   experiment, and by scoping every recovered P as "under a scalar-gain
   approximation."
6. **Stability is a quadratic-form bound, not spectral-radius(P) < 1.** A free P
   with a negative eigenvalue diverges on some data direction. De-risk,
   `P = (I + cL)^{-1}` or `exp(-cL)`, PSD by construction.
7. **Coverage / circularity.** P is identified only on the curriculum's
   transition-graph support. De-risk, connected interleaved synthetic curriculum,
   real-data reporting on co-practiced pairs only, a coverage-sweep ablation.
8. **State-conditioning re-breaks separability (`ednet_sep`).** De-risk, default
   item-only deterministic P, state-conditioned only as an H4 ablation with the
   separability index reported, do not assume the Paper 2 dynamic-head fix
   transfers without re-checking.
9. **Repeating a known negative.** Stage 1 success is Paper 2's alpha result
   transposed. De-risk by sizing Stages 1-2 as a methods/identifiability
   contribution and concentrating the new knowledge in H0, the diagonal-P null,
   and the oracle-vs-neural CR-bound gap.

---

## 9. Honest feasibility verdict and what was cut

**Verdict.** The synthetic identifiability and recoverability program (Stages -1
to 2) is well-posed, buildable additively on `deep_irt/` with a vector-theta
subclass and no Codex-owned edit, and novel against PSI-KT and graph-KT on the
recovery question. It is the deliverable. The most likely outcome is a
rank-not-magnitude recovery of P with a clean diagonal-P null and a quantified
scalar-gain misspecification cost, which is a clean methods contribution and a
small, honest one, not a discovery. The real-data leg is exploratory and capped
at existence detection, because KDD carries the unpinned-gauge blocker and the
saturation blocker simultaneously. **Effort: L**, concentrated in the field
encoder and the generator.

**What was cut from the draft, and why.**
- **Parallel build of Program I and Program II.** Sequenced. Both share the
  Fisher root, so **both Stage 0s (and Stage -1 calculations) run first**, the
  cheapest and highest-information checks, and the program commits to whichever
  clears. Program I is planned in its own document, not here.
- **Cross-encoder invariance from v1.** Moved to appendix or a later paper. One
  backbone for the main result.
- **DKVMN-with-fixed-Q-addressing as a v1 realization.** Cut. Real surgery, its
  own contribution. v1 uses the plain vector-theta LSTM.
- **The non-compensatory (smirt) synthetic variant.** Cut to a robustness note.
  Compensatory only, matching the decoder.
- **The real-data NULL-as-result framing (old H5).** Cut. Positive detections
  only.
- **The named-propagation-graph deliverable on real data.** Cut to synthetic
  only. Real data cannot pin the gauge.
- **The "P is a different kind of object than a knowledge graph" rhetoric.** Cut.
  Same operator class, novel recovery question.

**What was kept verbatim because the critique endorsed it.** Gauge-before-
geometry discipline, the scalar-Delta directionality fix (now also stated as a
structural limit), PSD/Laplacian P, the connected-curriculum coverage
requirement, the diagonal-P null as the fabrication control (not as a real-data
null license), the oracle-MLE as the Stage-0 kill switch, the rigorous
dynamic-matrix-flow objects (fundamental matrix, top Lyapunov exponent, leakage
matrix), and the two-clocks discipline.

---

## Cross-program coordination (brief)

Program I (structured learning dynamics) is planned separately and shares this
program's root obstacle, prediction loss under-identifies low-Fisher quantities
and real corpora lack dynamic range. P (here) and the per-student rate (there)
are the same class of low-leverage object. Both programs run their
no-training Fisher check and their synthetic-known Stage 0 **first**, and the
effort commits to whichever clears. Shared infrastructure, the
existence-gate-then-parametric-readout pipeline, gauge-free metric discipline,
the rq8 range-restriction-aware DIF test, and matched-null/positive-control
discipline. PSI-KT (`mlcolab/psi-kt`) is the must-read, must-cite competitor for
both.

**Grounding files (verified).** `deep_irt/core/encoder.py:86-197`
(swap contract, shape-generic `_shift`), `deep_irt/core/model.py:130-131,238-257`
(factory override injection point), `deep_irt/core/decoders.py:435-463` (GPCM
math reference), `deep_irt/bench/datagen.py` (generator template),
`deep_irt/ednet_sep/` (separability principle), `deep_irt/bench/run_swap_bench.py`
(cross-encoder apparatus, appendix-only here),
`docs/LEARNING_DYNAMICS_STUDY.md` (the Fisher-leverage result this transposes),
`docs/trajectory_findings.md` (the saturation negative the real leg inherits),
`docs/Thesis_overview.md` (Chapter C, the thesis arc this realizes).
