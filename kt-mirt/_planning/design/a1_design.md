# A1 pre-registered design: explicit-route signed cross-KC influence (G1)

Status: PRE-REGISTERED, v1.1, 2026-07-23. v1 was registered earlier the
same day and revised before any run after two independent pre-run
reviews; NO RUN HAS STARTED, so thresholds may be set or tightened now
and never loosened after runs begin (the A4 two-revision rule is
inherited, section 5.4). This document freezes the A1 estimand, model,
generator, battery, thresholds, kill conditions, bed plan, and compute
plan before any run. v1.1 adds the per-edge resolution precondition
(CT0, section 2.1.1), the endogenous-schedule and dynamic-confound twins
(SYN-T-ENDO / SYN-T-CNT-DYN, section 3.2), a saturation-matched null and
a floor-side range restriction (section 4.7), the mismatched-generator
arm at both densities over a misfit family (section 4.5), an itemized
compute budget with a measured per-step probe (section 7), a re-derived
G-augmented stop criterion (section 2.3), and a clean-negative
"unsupported" verdict (section 5.4). Review dispositions that push back
rather than fold are in section 9 (Rebuttals); decisions escalated to
the program lead are in "Open rulings for the orchestrator" at the end.

Goal G1: detect whether practicing knowledge concept A raises or lowers
performance on concept B, a SIGNED cross-KC influence (facilitation OR
interference / negative transfer), certified beyond pre-registered nulls.
This is avenue A1 of `_planning/research/avenue_map.md`. It reuses the A4
harness certified in `_planning/verdict_synthetic_complete.md`: the
frozen bank, the per-KC state substrate, the recognition network, the
saturation-aware machinery, and the whole certification battery.

Inputs: `_planning/research/{avenue_map, qmirt-archaeology, ltkt_read,
hawkeskt_read, graph-kt}.md`, `_planning/triage/triage_report.md`,
`_planning/vendor_report.md`, `_planning/design/a4_design.md`,
`_planning/verdict_synthetic_complete.md`, `_planning/THINKING.md`
(2026-07-21 G1-status entry). Built modules read for interface fidelity:
`src/kt_mirt/growth/{active,recognition,synth,battery,bank,kc_data,
qmatrix,slices,tracker,junyi_data}.py`.

---

## 0. Scope and claim boundary

- A1 estimates and certifies SIGNED cross-KC influence only. No growth
  magnitude claim beyond what G2/A4 already covers; own-gain is present
  only as the confound A1 must not launder transfer into (Lemma 3).
- The novelty is CERTIFIED sign, a learned continuous signed coefficient
  validated by synthetic sign recovery, honest nulls, an external
  reference, and stability. It is NOT a bare predictive-lift claim. Prior
  art (section 1) already owns predictive lift and even signed edges as a
  fixed heuristic label; A1 owns none of the certification.
- No multidimensional theta, no rotation freedom (program constraint).
  Per-KC attribution rides on the pure-anchor discipline (Gate C) exactly
  where A4 already uses it; the only new learnable object is the K x K
  matrix G, pooled across learners (not per-learner).
- Measurement-validity framing only. No pedagogical decision-utility
  claim. Present DKT/prediction as home, IRT/G as the readout flavor
  (memory: framing-dkt-home-irt-flavor).
- v1 pins rho = 1 (own-gain no-decay) and mu = 0 on the monotone real
  beds, exactly as A4. Negative transfer does NOT reopen the free-decay
  fabrication surface (Lemma 2), because it is PRACTICE-GATED and
  floor-bounded (section 2.2), not an always-on persistence term. A
  free-mu OU variant stays a pre-registered extension for beds with
  genuine non-monotone identification content only.

---

## 1. Prior art and the novelty this design must earn

Read in full: `ltkt_read.md`, `hawkeskt_read.md`, `graph-kt.md`.

- LTKT (Tsinghua Sci&Tech 2026, DOI 10.26599/TST.2024.9010201) is the
  only KT paper claiming signed positive-and-negative transfer. Its sign
  is a FIXED co-occurrence-heuristic label assigned by preprocessing
  (right-then-right builds a positive edge, wrong-then-right a negative
  one), routed through two nonnegative ReLU channels. The network never
  discovers sign; it learns magnitude within a pre-assigned polarity.
  Validation is predictive ablation only. No synthetic ground truth, no
  external label, no null, no seed variance.
- HawkesKT (WSDM 2021) has a genuinely unconstrained excitation
  alpha_{(source,resp),target} that can go negative, but it is fit by
  plain cross-entropy with no sign validation. Its one external check
  (NDCG 0.83 vs expert "helpfulness") runs on a softmax-collapsed,
  positive-only prerequisite score that structurally cannot express
  interference. The low-rank factorization it uses is never checked for
  sign fidelity.
- No graph-KT model (GKT, GIKT, SKT, PEBG, PKT, PSI-KT) uses signed
  edges; PSI-KT's a_ik is a direction probability in [0,1], not a
  valence. No dataset carries expert-labeled NEGATIVE transfer between
  named KCs (open risk 6).

A1 is distinguishable from all of the above iff it delivers what none of
them has: a LEARNED continuous signed coefficient whose sign is
certified by (a) synthetic recovery of injected signs, (b) honest nulls
and the confound battery, (c) external alignment of the positive half
against Junyi15's curated prerequisite graph beyond a null-graph
permutation (the PSI-KT template, extended to sign), and (d) stability.
If A1 stops at "we fit a signed matrix and prediction improves," it
restates LTKT/HawkesKT and adds nothing. Every certification arm below
exists to clear that bar.

---

## 2. The estimand and the model

### 2.1 The estimand: the signed influence matrix G

G is a K x K real matrix, zero-diagonal, pooled across all learners
(learner- and time-independent, the PSI-KT posture for structure).
G[c, a] is the signed influence of practicing SOURCE KC a on TARGET KC
c's latent ability state, per source-practice opportunity, in state
(logit) units before ceiling/floor gating.

Operational meaning of a signed edge, stated on the score scale so it
neutralizes the theta-scale gauge (proportion-of-max units, the qmirt
metric ruling). Comparability across beds is CONDITIONAL ON BANK
FIDELITY, not unconditional: the implied-score change is read through the
frozen 1PL difficulties, and A4 found bank recovery rank_corr stuck at
0.70-0.80 (difficulties mis-ordered by roughly 20-30%,
`verdict_synthetic_complete.md` section 3). The score transform removes
the theta-scale gauge but NOT this difficulty-calibration error, and
"median-difficulty c item" is defined against the same imperfect bank.
Every score-scale contrast is therefore reported with a bank-error
sensitivity band (section 4.2), and the "comparable across beds" claim is
scoped to hold conditional on the measured bank fidelity of each bed:

- G[c, a] > 0 (facilitation / positive transfer): holding c's own
  practice fixed, one additional opportunity of practicing a raises the
  model-implied success probability on a median-difficulty c item.
- G[c, a] < 0 (interference / negative transfer): the same source
  practice LOWERS c's implied success probability.
- G[c, a] ~ 0: a is inert for c.

Both a state-units read (G directly) and a score-scale read (the implied
change in P(correct on c) over a fixed number of a-opportunities on a
median-difficulty c item, computed by the same closed-form recurrence
`active.implied_score_rise` uses) are reported. Score scale is primary
for comparability and for the matched-null contrast; state units are
secondary and gauge-dependent, reported only on the frozen anchored
scale.

CERTIFIED SIGN, the deliverable, requires ALL of the following to hold
under the frozen thresholds of section 5, and no subset substitutes for
the whole:

1. Synthetic signed-edge recovery: sign-F1 against injected +/-/0 ground
   truth at the reference dose, sign accuracy on true edges, false-edge
   rate on zero cells, seed-consistent, at D = 3, 5, 8 (section 3, 5.1).
2. Matched-null paired contrast: the target-KC forecast-score-error gap
   (no-transfer minus with-transfer) minus the same on a G_true = 0 twin
   is sign-correct and seed-pooled significant (section 4.2).
3. The confound battery all passes: correlated-no-transfer (static and
   dynamic), co-scheduling, shuffle-order, reverse-direction, and the
   endogenous-scheduler arm that supplies the causal warrant for the real
   leg (section 4.3).
4. Per-learner p95 tail on the null twin within band, not just the
   population mean (the Gate B lesson; section 4.4).
5. The mismatched-generator arm: G stays clean on true-zero cells under
   own-gain-family misfit (Lemma 3 laundering measured directly; the
   never-run C3 threat; section 4.5).
6. The phantom-transfer sensitivity control: a free/amortized
   per-learner-multiplier variant is run on the null twin as a NEGATIVE
   control that the p95 tail metric is sensitive to per-learner
   fabrication. Its EXPECTED behavior is to fabricate; a fabrication
   confirms the metric bites, but its outcome does not by itself re-earn
   or overturn the gamma pin, which rests on the qmirt Gate B evidence
   and the architectural argument (section 4.6, reframed in v1.1 per
   review).
7. External reference: recovered positive edges align with Junyi15's
   curated prerequisite graph beyond a null-graph permutation, at exercise
   grain and conditional on Junyi clearing order-based positivity (section
   4.8). The negative half has no external answer key anywhere and is
   certified by 1-6 plus, if A2 lands, the Eedi misconception channel.
8. Stability: seed-clustered sign consistency and split-half sign
   reproducibility (section 4.9).

### 2.1.1 The per-KC resolution floor and the CT0 power precondition

A1's sibling result A4 found that per-KC RESOLUTION is a fundamental
identifiability floor, not a density artifact: per-KC BH power was flat
0/515 (KDD-shaped) and 0/189 (EdNet-shaped), bank recovery rank_corr
stuck at 0.70-0.80, INVARIANT across a full density inversion, and
explicitly labeled a property of the estimator and test construction
(`verdict_synthetic_complete.md` section 3, CHECK C). The A4 detector
that WORKED was the twin-level POOLED existence gate ("this group grew");
the per-KC read failed. A1's objects are finer than the A4 per-KC read:
the matched-null contrast (section 4.2) is scored per TARGET KC c, and
sign-F1 (section 4.1) per off-diagonal CELL. A reviewer who reads both
verdicts will kill the sign claim unless this is confronted before any
run. Two things must be said, one structural, one pre-registered.

STRUCTURAL, the pooling axis differs. A4's failing per-KC read estimated
one KC's growth from that KC's OWN slices (per-KC N). A1's per-edge
G[c,a] is estimated by POOLING every learner who practiced a in a slot
decoupled from c and was then measured on c (the PSI-KT population
posture, section 2.1). The effective sample per edge is the number of
such learner-observations, not the per-KC slice count, and it grows with
N. So A1's per-edge object is not literally the per-KC object A4 could
not resolve; it sits on a THIRD pooling axis (across learners for one
fixed edge), between A4's failing per-KC read and its working
across-KC twin pool. This is a reason to test, NOT a reason to assume it
lands on the working side. It does not dissolve two facts the reviewer is
right about: the per-edge read is finer than a pooled read, and A1 reuses
the SAME frozen bank whose 0.70-0.80 recovery IS the A4 floor, so the
floor rides along unchanged unless measured.

PRE-REGISTERED, CT0, a resolution power precondition that runs BEFORE
CT1/CT2 are interpreted. On SYN-T-KG at the reference dose, sweep the
per-edge effective sample (via N and decoupling) and report per-edge
sign-F1 as a function of effective observations per edge and of
decoupling, at both densities and at D = 3, 5, 8 and full K. CT0 states a
pre-registered MINIMUM effective sample per edge below which a per-edge
sign verdict is declared unidentified for that edge, chosen as the
smallest effective sample at which sign-F1 first clears the CT1 bar on
the power curve (a data-driven, pre-registered cut, frozen from the curve
before any confirmatory read). Real edges whose effective sample falls
below the CT0 minimum are reported as UNIDENTIFIED, never as zero and
never sign-verdicted. CT0 also carries the bank-floor propagation check:
the same power curve is re-run with difficulties perturbed by the A4
measured recovery error (section 4.2's sensitivity band), and if per-edge
sign-F1 collapses under that perturbation the sign claim is bank-limited,
a reported scope, not a clean pass. If NO feasible (N, decoupling)
reaches the CT1 bar at D = 3, per-edge sign is unidentifiable at feasible
data sizes and G1 dies on this design space exactly as K-T1 states,
sharpened here so the kill is decided on the power curve, not on a single
failed fit.

### 2.2 The model on top of the vendored core

A1's model is the A4 ACTIVE model (`growth/active.py`) plus exactly one
new route: a fitted signed G driven by practice indicators. Nothing else
about the certified A4 substrate changes. Per-learner per-KC scalar state
z_{i,c,t} on the frozen bank's theta scale, initialized z0_{i,c} = u_i +
v_c. The transition, the ONLY state-moving law, adds the cross-KC term to
ACT's own-gain term:

```
z_{i,c,t+1} = z_{i,c,t}
            + lambda_i * g_c * (M - z_{i,c,t})+ * 1[c practiced at t]        (own-gain, A4 ACT, unchanged)
            + SUM_{a != c} 1[a practiced at t] * T(G[c,a], z_{i,c,t})         (cross-KC transfer, NEW)
```

with the sign-asymmetric ceiling/floor gate (the qmirt bounded-
interference patch, "you can only lose what you have built"):

```
T(g, z) =  g * (M - z)+        if g >= 0   (facilitation, ceiling-gated)
           g * (z - floor)+    if g <  0   (interference, floor-gated)
```

Load-bearing properties, each tracing to a named constraint:

- PRACTICE-GATED, RESPONSE-BLIND. The cross-KC term reads only the
  practice indicator `1[a practiced at t]` (which source KCs were
  practiced at t, i.e. `kc_ids[:, t, :]`), never y. This is the qmirt R9
  rule and constraint (1): responses feeding the transition sign-reverse
  the coefficient on sparse edges and fabricate on nulls. The transition
  is a pure function of the practice schedule and the amortized seed.
- G IS THE ONLY CROSS-KC ROUTE. Structural isolation: with G = 0, a
  non-practiced KC's z never leaves z0 (verified to ~1e-8 in qmirt; a
  unit test asserts it here, section 2.4). The shared encoder is NOT the
  knowledge carrier (that would be passive transfer mimicry, constraint
  1); it is DEMOTED to a recognition network (below).
- PER-LEARNER TRANSFER MULTIPLIER PINNED. gamma = 1 on the transfer term
  by construction. There is no free or amortized per-learner transfer
  trait anywhere. This is the single sharpest qmirt negative result
  (Gate B/B2: gamma fabricates under every estimation posture) and
  exactly PSI-KT's per-learner transfer trait. The pin rests on that Gate
  B evidence plus the structural argument (a per-learner transfer trait
  is the object that failed certification under every posture); the
  phantom-transfer arm (section 4.6) is a SENSITIVITY control that the
  fabrication metric bites on this harness, not a test whose outcome
  re-earns or overturns the pin (v1.1 reframing, per review).
- OWN-GAIN CEILING-GATED. `(M - z)+` diminishing returns, so gain-form
  misfit cannot be sculpted into G by the schedule's positivity
  variation (Lemma 3). g_c constrained positive via softplus (a negative
  own-gain would be an unpracticed decline channel, reopening Lemma 2).
- PINS. mu = 0 (no mean-reversion target), rho = 1 (no always-on decay).
  Negative transfer is the only downward channel and it is practice-gated
  and floor-bounded, so it is not the Lemma-2 free-persistence
  compensator. A free-mu OU own-transition is a pre-registered extension
  for non-monotone beds only.
- PER-LEARNER z0 AND lambda AMORTIZED, ENCODER DEMOTED. z0_{i,c} = u_i +
  v_c with u_i from `recognition.RecognitionNetwork` over the learner's
  FULL conditioning window (the only posture Gate B certified for z0 and
  lambda) and v_c a free population per-KC offset. Variant A1-P0 pins
  lambda_i = 1; A1-P1 amortizes a positive scalar lambda_i, full-window
  only. Real-bed claims use the richest variant that stayed clean on the
  null twin. The transfer multiplier is NOT among the amortized outputs;
  the recognition net emits only (u_i, lambda_i), identical to ACT.
- READOUT. Frozen 1PL bank (`growth/bank.py`), binary Bernoulli forecast
  NLL through `Binary2PLDecoder` with difficulties frozen and only the
  ability path free, exactly A4. Multi-tag readout: the predicted logit
  at t is mean over the item's tagged KCs of the PRE-update z minus b_j,
  the `active.run_transition` rule verbatim.
- G REGULARIZATION, PRE-REGISTERED TUNING. Zero-diagonal is a hard
  constraint (the diagonal is never a parameter). Off-diagonal G carries
  an L1 penalty for sparsity (real influence graphs are sparse) and, on
  real beds at K in the hundreds, an optional HawkesKT-style low-rank
  factorization G = P Q^T, D_rank << K. Both the L1 weight and D_rank are
  hyperparameters and therefore researcher degrees of freedom; they are
  fixed on a DEDICATED held-out generator seed (distinct from the five
  test seeds and every test config) by maximizing per-edge sign-F1 at the
  reference dose subject to the CT1 false-edge-rate bar, then frozen
  before any test-config or real-bed fit. Two sign-biasing failure modes
  are pre-registered as reported metrics, not hidden: L1 can game the
  false-edge-rate by shrinking every cell toward zero, which
  disproportionately kills the weaker negative half (bounded interference
  is floor-limited), so sign-F1 is reported SEPARATELY for positive and
  negative edges and the negative-half recall is a gate clause (section
  5.1); low-rank G = P Q^T ties correlated rows, so one strong true edge
  can induce same-sign phantom edges in correlated rows, so the low-rank
  fit is scored against GROUND-TRUTH sign-F1 on the synthetic full-K
  config, not only against the full fit (agreement is not correctness,
  section 4.10). This closes the exact gap HawkesKT never checked.

### 2.3 Two-stage training

Stage 1: calibrate and freeze the item bank (`bank.calibrate_bank` /
`freeze_bank`, reused unchanged, including the KDD hierarchical MAP bank
and the tri-spec RB0 robustness refits). Stage 2: fit {g_c, v_c, M, G,
recognition weights} jointly against the frozen bank by forecast NLL plus
the L1(G) penalty, under a convergence-gated, windowed-mean trainer whose
STRUCTURE is `active.train_active`'s (windowed-mean relative-loss leg,
parameter-drift leg, epoch-ceiling floor) but whose NUMBERS are
re-derived, not ported verbatim. `active.train_active`'s `rel_tol =
1e-5`, `drift_tol = 5e-2`, and `ACT_MIN_EPOCHS_CEILING = 3000` were fixed
by a dedicated stationarity study on the A4 g_c/v_c/M landscape (3 seeds
x 3000 epochs on both twins, `active.py` note 8), precisely because a
verbatim-inherited budget once reproduced the ACT-P0 fabrication
pathology (Adam stopped one to two orders of magnitude before g_c
converged). A1's stage-2 objective adds G (up to roughly 265k
off-diagonal cells at full K) under Adam plus an L1 subgradient, whose
convergence dynamics (slow per-cell shrinkage, no exact zeros, a noisier
windowed-mean trace) are not the landscape those numbers were tuned
against; porting them verbatim risks silently reproducing ACT-P0 in the
larger space, where a premature stop leaves G's true-zero cells above the
CT1 false-edge-rate bar and fails as a certification MISS, not an obvious
training bug. Therefore a G-augmented stationarity study (R0-A1, section
7) re-derives `rel_tol`, `drift_tol`, and the epoch ceiling for this
objective before the certification matrix runs, the drift snapshot is
extended to track G (specifically the max change in |G| on true-zero
cells, the quantity CT1/CT5 read), and a convergence positive control is
pre-registered: on SYN-T-NG the true-zero cells must have shrunk below
the CT1 false-edge threshold AND be quiet (not still descending) at the
stop, else the stop criterion is retuned before any verdict. The
permutation null, matched-null twin, and every confound arm refit stage 2
only; the bank is frozen once. Neural evaluation is the A4 80/20
learner-split forecast regime.

### 2.4 Attachment-point mapping and new modules

Mapping onto the five per-KC attachment points of `vendor_report.md`,
mirroring the A4 table:

| Attachment point | Use in A1 |
|---|---|
| #1 ability readout (`theta_proj`, `state_for_prediction`) | Unchanged from A4. The recognition net reads the encoder's full-window final hidden state to emit (u_i, lambda_i). The per-KC state z is a STRUCTURED external (B, C) tensor advanced by the transition, NOT the encoder hidden state. This demotion is the whole point of constraint (1): the encoder recognizes the learner, G carries the influence. |
| #2 item-to-KC mapping (`kc_data` ragged kc_ids) | Reused as-is, and it doubles as the PRACTICE INDICATOR. `kc_ids[:, t, :]` names the source KCs practiced at t; the transfer term sums G[c, a] over those a. No new collate field: practice indicators are the tags already carried, read structurally (never gated by y). `qmatrix.pure_anchor_stats` gates per-KC attribution as in A4. |
| #3 item parameters (frozen bank, `Binary2PLDecoder`) | Frozen 1PL bank feeds the decoder unchanged; difficulties frozen, ability path free. No per-KC alpha in v1. Polytomous beds (Eedi, A2 territory) would swap in GPCM/NRM; A1 v1 is binary. |
| #4 loss (`CombinedLoss` home) | Bernoulli forecast NLL, binary BCE via the decoder `nll`; NEW additive L1(G) sparsity term. No per-KC class weights (binary). |
| #5 anchoring (item-axis only) | Bank item-axis anchoring reused. The K x K G is a NEW learnable population table with its own zero-diagonal hard constraint, L1, sign-asymmetric gate, and optional low-rank factor. It is pooled (not per-learner), so no KC-axis anchoring primitive is needed; the pooling IS the identification discipline (PSI-KT posture). This is the one learnable object beyond A4's set, flagged here. |

New modules, a new subpackage on top of `growth/` (imports downward from
`growth/` and `core/`, never sideways; no runtime import from
`deep_irt`):

- `kt-mirt/src/kt_mirt/transfer/__init__.py`
- `kt-mirt/src/kt_mirt/transfer/model.py` -- `TransferModel` (ACT
  transition + signed G term + asymmetric gate + zero-diagonal + optional
  low-rank + phantom-gamma flag), `run_transfer_transition` (extends
  `active.run_transition`), the two-stage trainer, the closed-form
  score-scale influence read. The phantom-gamma head (section 4.6) is a
  NEW, PARALLEL, opt-in output built inside this module, NOT an edit to
  `growth.recognition.RecognitionNetwork`: that module is A4-certified for
  (u_i, lambda_i) under CG9/CG9-ACT order stress and stays byte-identical,
  so the phantom variant either adds a free per-learner gamma_i parameter
  or a separate amortized head here, behind a flag off by default. The
  default `TransferModel` calls the unmodified `RecognitionNetwork` for
  (u_i, lambda_i) exactly as ACT does.
- `kt-mirt/src/kt_mirt/transfer/synth.py` -- signed-edge generator
  (injects G_true into `growth.synth`'s substrate trajectory sim;
  positive + negative edges; D = 3/5/8 and full-K profiles; the
  positivity/decoupling knob; the confound twins).
- `kt-mirt/src/kt_mirt/transfer/battery.py` -- the A1-specific arms
  (signed-edge F1, matched-null paired contrast, confound battery,
  reverse-direction, mismatched-generator, phantom-transfer, low-rank
  fidelity). REUSE IS LIMITED to the GENERIC `growth.battery` primitives:
  the shuffle primitive `permute_cross_kc_interleaving` (it takes and
  returns `LearnerLog` sequences, genuinely model-agnostic) and the stats
  helpers `seed_pooled_pvalue`, `seed_cluster_verdict`, `spearman_brown`,
  `direction_violation_fraction`. `growth.battery.run_permutation_battery`
  is NOT reusable: reading it directly, it is hardwired to PAS-G's
  slice-based logistic gate (`bank.build_calibration_rows`,
  `build_slices`, `gate.compute_gate_result`, `gate.permutation_null`) and
  has no path to refit a `TransferModel`. The shuffle-order arm (CT3-iii)
  and the permutation null therefore need a NEW orchestration in this
  module that parallels `run_permutation_battery`'s STRUCTURE (shuffle
  order, refit, compare recovered |G| and the paired contrast to the
  null band) but calls the stage-2 transfer trainer, not the gate. This
  is new code, not inherited harness.
- `kt-mirt/src/kt_mirt/transfer/external.py` -- Junyi prerequisite-graph
  alignment (recovered-G vs curated graph, null-graph permutation,
  MRR/AUC/Jaccard/direction agreement), reusing
  `junyi_data.load_prerequisite_graph` and `load_junyi_kc_traced`.
- `kt-mirt/src/kt_mirt/transfer/report.py` -- verdict assembly, signed-
  edge tables, the confound matrix, the external-alignment readout.
- Scripts: `kt-mirt/scripts/a1/{run_synth,run_bed,run_battery,
  run_external,make_report}.py`.
- Tests: `kt-mirt/tests/test_transfer_{model,synth,battery,external}.py`,
  including the G = 0 isolation leak test (a nonzero G moves its target
  and leaves zero-row KCs flat, the qmirt corrected positive-control
  form) and the response-blindness assertion SCOPED to
  `run_transfer_transition` (its output is invariant to permuting y GIVEN
  fixed amortized (z0, lambda)). The assertion is NOT applied end to end:
  the recognition network legitimately reads y to produce (z0, lambda)
  upstream (`active.py` note 6), so a whole-model invariance test would
  fail by design; the boundary under test is the transition given its
  amortized seed, exactly as ACT scopes it.

---

## 3. Synthetic generator: known signed edges

Extends `growth/synth.py`. The certified A4 substrate (KC popularity,
learner-KC incidence, per-learner heterogeneity, item bank, calibrated
difficulties, matched-twin discipline) is reused verbatim; the ONE
addition is a known signed sparse G_true injected into the per-slice
trajectory so that a target KC's state responds to source-KC practice.

### 3.1 Generator family

Per learner, the standard A4 own-growth trajectory z*_{i,c}(n) is
advanced, then at each practiced source opportunity the target states are
pushed by the same practice-gated, sign-asymmetric-gated law the model
uses (generator and model share the gate, the qmirt discipline). Because
the transfer is practice-gated, the realized target trajectory depends on
the interleaving of source and target practice, which is exactly what the
positivity condition governs.

- SIGNED EDGES. G_true is sparse with BOTH signs. Reference edge set per
  D: a fixed number of positive edges and negative edges (e.g. at D = 3,
  one +, one -, one zero-row control KC; scaled with D). Edge dose
  |g_ref| is pre-registered on the 0.1-log-odds/opportunity anchor scale
  (Koedinger PNAS 2023): the reference dose gives a per-opportunity
  score-scale push near the population growth anchor, with a dose-
  response sweep {0.5, 1, 2, 4} x |g_ref| reported so recovery is
  characterized as a function of dose and N*T, not asserted at one point.
- CO-TAGGING PLACEMENT, an explicit generator decision. The multi-tag
  item bank inherited from `growth.synth._build_item_bank_multi_tag`
  bakes cross-KC co-tagging into individual items (a Poisson number of
  bystander tags per item, fixed at bank construction, independent of any
  schedule or interleaving knob), so a learner attempting a co-tagged item
  practices two KCs at the SAME timestep, a structural co-observation the
  schedule-level decoupling knob cannot sweep to zero. To keep the
  positive control clean, `transfer/synth.py` places the reference G_true
  edges ONLY between KC pairs that share NO co-tagged item, so SYN-T-KG
  recovery measures PURE practice-gated transfer with temporal lag, not
  same-slot co-observation. Co-tagged pairs are instead the subject of a
  distinct confound arm (SYN-T-CO, section 3.2), where a same-slot
  co-observed pair carries G_true = 0 and must not be read as transfer;
  this is the synthetic analog of EdNet's bundle-exposure confound
  (section 5.3). The same-slot decoupling metric of the real-bed
  positivity precondition (section 5.2) is exactly the quantity that
  measures this baked co-tagging on real beds.
- D-SCALING. D (KC count) in {3, 5, 8}. D = 3 reproduces the qmirt
  internal certification on the kt-mirt harness (the port); D = 5 and 8
  are the scale-up qmirt built but never ran. Each D crossed with both
  density profiles below.
- DENSITIES MATCHED TO TRIAGE. Two profiles reused from `growth.synth`,
  `KDD_MATCHED` (near-1-to-1, median 8 opp/slice, saturated mass) and
  `EDNET_MATCHED` (multi-tag arity 2.2, median 2 opp/slice). The two
  carry the two identifying structures H4 names: KDD-shaped identifies
  via PREREQUISITE CHAINS (interleaved 1-to-1 practice with temporal
  lag), EdNet-shaped via MULTI-TAG items (cross-KC co-observation). Both
  are tested because A1's real beds split the same way.
- POSITIVITY KNOB. The schedule has a controllable decoupling fraction
  (fraction of source-only practice slots, the qmirt positivity
  quantity). Certification runs at decoupling ~0.8 (clean) AND at the
  real-bed admission threshold 0.75, which must independently hold the
  null band and clear sign recovery, because a real bed admitted at, say,
  0.76 must not sit in a fabricating regime no gate ruled out (the 0.75
  bar is inherited from qmirt and is re-certified here on the kt-mirt
  harness, not assumed). Degradation is then characterized at 0.5 and
  0.25; the co-scheduling twin sits near 0 and must show the
  identifiability BOUNDARY (G collapses toward 0), not fabrication
  (section 4.3).
- FULL-K STRESS. Beyond D = 8, one run each at K = 189 (EdNet-shaped) and
  K = 515 (KDD-shaped), and one at K ~ 835 (Junyi exercise grain, section
  5.3), with a sparse G_true, to test whether recovery and the low-rank
  factorization survive real KC counts (open risk 5; low-rank G at real K
  is unvalidated). Because laundering and per-learner fabrication may
  scale with cell count (more L1 and low-rank freedom to sculpt spurious
  signs), the full-K stress is NOT sign-recovery only: it also runs the
  mismatched-generator laundering check (CT5) and the per-learner p95 tail
  (CT4) at full K, so the K in the hundreds real leg does not inherit an
  UNCERTIFIED laundering surface. If CT5 or CT4 cannot be afforded at full
  K (section 7 probe), the real-bed sign claim is scoped to the K-range
  where they ran (K-T3). This is the scaling gate, still gated by CT0's
  full-K power point.

Acceptance checks (reused pattern): the A4 four checks (rate quartiles,
pooled rise, density quantiles, implied slope) plus two new ones. The
realized target-KC displacement attributable to source practice must
match the injected dose within tolerance on the KG twin, and must be
exactly zero (to generator precision) on the NG twin. A generator run
failing acceptance is regenerated, never analyzed.

### 3.2 The certification twins

Twins share seeds, schedules, and the A4 substrate wherever the dynamics
allow (matched-twin discipline). Naming keeps the A4 `syn_*` convention
with a `_t` transfer marker.

- SYN-T-KG (known signed edges; both densities, all D): the standard
  generator with the reference signed G_true. The positive control for
  sign recovery (section 5.1).
- SYN-T-NG (no transfer; both densities, all D): G_true = 0, everything
  else identical. Own-growth still present (this is not the A4 no-growth
  twin; own-gain is on, only transfer is off), so it is the honest null
  for the TRANSFER claim: any nonzero recovered G here is fabrication.
  The primary matched-null reference and the per-learner-tail twin.
- SYN-T-CNT (correlated-no-transfer, STATIC; KDD density): source and
  target share a correlated ability component (a common prerequisite
  raises both z0) but G_true = 0. Confound arm (i): a passive static
  correlation the model must not read as transfer.
- SYN-T-CNT-DYN (correlated-no-transfer, DYNAMIC; KDD density): a shared
  latent effort/engagement TRAJECTORY co-moves over time, driving BOTH
  the source practice intensity AND the target's growth, with G_true = 0.
  This is the classic observational-transfer confound a static z0
  correlation cannot simulate (a common time-varying cause reads as
  a-practice -> c-growth). Confound arm (i-dyn): the dynamic correlation
  must not read as transfer.
- SYN-T-ENDO (endogenous scheduler; both densities): the practice
  schedule is generated by a MASTERY-GATED controller that practices a KC
  when the learner's latent state on it is low (schedule depends on
  state), G_true = 0. Every real bed has such a controller (KDD is a
  mastery-managed ITS, Junyi is mastery-gated), so 1[a practiced at t] is
  chosen by a process reacting to prior performance and is correlated with
  the learner's latent state, including state on c. Response-blindness
  removes y from the TRANSITION but not the response information the
  SCHEDULE ITSELF carries, so this is the one confound that tests whether
  G absorbs schedule-selection rather than causal transfer. It is
  orthogonal to positivity (K-T4) and has NO exogenous-schedule analog
  among the other arms. Confound arm (v): the fitted G and paired contrast
  must stay in the null band under an endogenous scheduler.
- SYN-T-CO (co-scheduling; KDD density): source and target almost always
  practiced together (decoupling ~0), including via CO-TAGGED items (the
  baked same-slot co-observation of section 3.1), G_true = 0. Confound arm
  (ii): the identifiability boundary; own-gain and transfer are collinear,
  so G must collapse toward 0, NOT fabricate above the null band. This is
  the synthetic analog of EdNet's bundle-exposure confound.
- SYN-T-NS (mismatched generator; BOTH densities, over a misfit FAMILY):
  the OWN-GAIN family is misspecified relative to the model, while a known
  sparse signed G_true is present AND a pre-registered subset of target
  cells carries true-zero transfer. The misfit is not one shape but a
  pre-registered family, because passing on one alternative does not
  certify against the continuum of unknown real gain laws: {step and
  dip-recover, reusing A4's `_syn_ns_partition`; a decelerating
  log-gain whose curvature differs from the model's ceiling-gated form;
  and a faster/slower true ceiling constant}. The adversarial worst case
  is pre-registered as the shape whose own-gain residual most resembles a
  practice-gated cross-KC signal (the decelerating-versus-ceiling
  mismatch Lemma 3 named, where the schedule's positivity variation
  supplies the sculpting basis), and CT5 must beat it. Because that
  sculpting basis is a property of the schedule's decoupling, which
  differs by density, SYN-T-NS runs at BOTH densities, not KDD only, so
  the thin-density laundering regime (where A4's own misfit flag fired
  everywhere) is covered. The never-run C3 threat, now run: certification
  requires G to stay clean on the true-zero cells (no gain-misfit
  laundering into the transfer matrix) while still recovering the true
  edges' signs. Lemma 3 measured directly for G (section 4.5).
- SYN-T-SAT (dynamic-range targets; KDD density): a CEILING subset of
  target KCs starts near ceiling (reusing A4's `_sat_shift`) and a FLOOR
  subset sits near floor with low downward range, both with true signed
  edges present. The asymmetric gate is correct here: the ceiling gate
  `(M - z)+` bounds positive transfer to ~0 near mastery, and the floor
  gate `(z - floor)+` bounds negative transfer to ~0 near floor (you
  cannot lose what you have not built). Certification therefore requires
  no confident signed edge manufactured from range-starved observations
  in EITHER direction, and primary verdicts are restricted to targets
  with adequate dynamic range on the relevant side: positive verdicts off
  saturated targets, negative verdicts off floored targets (the symmetric
  A4 saturation lesson applied to both signs of transfer, section 4.7).

The PHANTOM variant is a MODEL change, not a data twin: A1-P0/P1 refit on
SYN-T-NG with a free/amortized per-learner transfer multiplier gamma_i
added to the transfer term. It must fabricate (section 4.6).

Seeds: 5 generator seeds per config for all slice-based / closed-form
statistics; neural arms (recognition, low-rank fits) on generator seed 0
with 3 model seeds (the A4 compute concession, pre-registered).

---

## 4. The certification battery

One module (`transfer/battery.py`) reusing `growth/battery.py`
primitives, applied uniformly to synthetic and real beds. Ten arms,
mapped to the estimand's eight requirements plus scaling and saturation.

### 4.1 Signed-edge recovery F1 (SYN-T-KG)

Threshold recovered G against zero at the matched-null / permutation band
(section 4.2's offset-robust band, never a bare compare-to-zero, the
qmirt metric ruling). Score each off-diagonal cell as +, -, or 0; compute
sign-F1 against G_true's +/-/0 pattern (the PKT synthetic-recovery
template, extended to three classes). Report sign accuracy on true
nonzero edges, false-edge rate on true-zero cells, and the dose-response
curve. Seed-clustered.

DEGENERATE BASELINES, pre-registered and reported alongside every F1,
because a sparse G_true makes some scores gameable. An all-zero predictor
scores false-edge-rate 0 (passing the CT1 zero-cell clause outright) and
leaves sign accuracy on true edges vacuous, and a random-sign predictor
scores 0.5 on binary +/- true edges; the recovered sign-F1 must beat BOTH
by a pre-registered margin (section 5.1), which is why the headline metric
is sign-F1 (recall-bearing, so all-zero scores 0), not false-edge-rate or
accuracy alone. Sign-F1 is additionally reported SEPARATELY for the
positive and negative half, and negative-half recall is its own gate
clause, so L1 shrinkage cannot pass by silently zeroing the weaker
negative edges (section 2.2).

### 4.2 Matched-null paired contrast (offset-robust primary metric)

Fitted G carries a per-seed additive offset and cannot be compared to
zero (qmirt). The load-bearing quantity is, per target KC c, the
forecast expected-score error on c with the source's transfer route
zeroed at forecast time minus the error with it active, MINUS the same
difference computed on the same-seed SYN-T-NG twin. Score scale,
proportion-of-max, bounded and sharpness-insensitive (the qmirt primary
metric). Positive edges must yield a positive seed-consistent contrast,
negative edges a negative one; seed-pooled significance via
`battery.seed_pooled_pvalue`. This is the metric all downstream gates
read; F1 in 4.1 thresholds against ITS band.

Two refinements the A4 record forces. First, the null band is NOT one
global band: on saturated or floored targets the matched null is drawn
from a RANGE-MATCHED twin (the SYN-T-SAT null subset), not the
unsaturated SYN-T-NG band, because the forecast-NLL comparison inflates
exactly where the reference model approximates a range-limited curve
worse (the CG6 inversion, section 4.7). Second, because the contrast is
read through the frozen difficulties and the bank recovers only at
rank_corr 0.70-0.80 (`verdict_synthetic_complete.md` section 3), every
contrast is accompanied by a BANK-ERROR SENSITIVITY band: re-read the
contrast with difficulties perturbed by the measured recovery error
(resampled from the bank-recovery residual distribution), and report the
resulting spread. A sign verdict that flips within the bank-error band is
downgraded to bank-limited, not confirmed.

### 4.3 The confound battery

The qmirt confound quartet plus two arms the A4 record and the endogenous
real-bed schedule force, each a pre-registered arm.

- (i) Correlated-no-transfer, STATIC (SYN-T-CNT): paired contrast within
  the null band; shared static ability must not read as transfer.
- (i-dyn) Correlated-no-transfer, DYNAMIC (SYN-T-CNT-DYN): a shared
  time-varying effort trajectory drives both source practice and target
  growth with G_true = 0; the paired contrast must stay within the null
  band. The static z0 correlation of arm (i) cannot simulate this
  common-time-varying-cause confound, so it is a separate arm.
- (ii) Co-scheduling (SYN-T-CO): recovered |G| collapses toward 0
  (identifiability boundary), the paired contrast within the null band.
  The failure signature to reject is fabrication ABOVE the band; the
  correct behavior is a near-zero, honestly unidentified G. Reported as a
  boundary, not a pass/fail on magnitude alone.
- (iii) Shuffle-order: destroy the causal lag by shuffling each learner's
  practice order, refit, and require recovered |G| and the paired
  contrast to collapse to the null band (transfer needs the causal lag,
  not co-occurrence). This is also the permutation null for empirical p.
  Reuse is the GENERIC shuffle primitive `permute_cross_kc_interleaving`
  only; the shuffle-refit-compare orchestration is NEW code in
  `transfer/battery.py` calling the stage-2 transfer trainer, since
  `growth.battery.run_permutation_battery` is hardwired to the slice-based
  gate and cannot refit a `TransferModel` (section 2.4). Stated limit: a
  WITHIN-learner shuffle destroys the temporal lag but PRESERVES the
  marginal schedule-state correlation, so this arm does not by itself
  neutralize endogenous scheduling; arm (v) does.
- (iv) Reverse-direction: fit G[a, b] when only G[b, a] is a true edge;
  the reverse cell must sit in the null band (direction identified, not
  just magnitude). The qmirt reverse-direction probe.
- (v) Endogenous scheduler (SYN-T-ENDO): under a mastery-gated controller
  that schedules practice as a function of latent state (G_true = 0), the
  fitted G and paired contrast must stay in the null band. This is the
  only arm that tests whether G absorbs schedule-selection rather than
  causal transfer, the direct causal-identification threat on every real
  bed (section 5.3). It runs at both densities and is the synthetic
  warrant for the real "A raises/lowers B" reading.

### 4.4 Per-learner p95 tail on the null twin (SYN-T-NG)

The single most consequential qmirt methodological choice: read the 95th-
percentile per-learner transfer effect on the null twin, not just the
population mean, because the mean hides per-learner fabrication (Gate B).
Reuse the A4 CG1-style p95 machinery. Pre-registered band matches A4's
density-specific silence bars (0.01 KDD-density / 0.02 EdNet-density
proportion-of-max).

### 4.5 The mismatched-generator arm (SYN-T-NS), MANDATORY

The live C3 threat the qmirt archaeology flagged repeatedly and never
ran, now run at BOTH densities against a misfit FAMILY (section 3.2). Fit
the model (matched own-gain family, so it is WRONG here by construction)
against SYN-T-NS. Certification requires, on the true-zero transfer cells,
false-edge rate and paired contrast within the null band (no Lemma-3
laundering of the own-gain shape residual into G), AND, on the true edges,
sign accuracy above the degraded-but-correct bar with bounded overshoot.

The metric here is the G false-edge-rate on true-zero cells plus the
paired contrast, NOT A4's CG5 misfit-FIRE fraction (the non-informative
flag that fired everywhere on both profiles, `verdict_synthetic_complete.md`
section 5). This arm reuses the SYN-NS generator machinery and the
ground-truth-bearing F1 and contrast primitives, both of which carry
injected signs and are informative; it does not inherit the CG5 flag's
non-informativeness, because it does not use that flag (section 9,
Rebuttals). Running at both densities is load-bearing: Lemma-3 laundering
is driven by the schedule's positivity variation, which differs by
density, so the thin-density regime where A4's misfit flag saturated must
be tested, not assumed. This is the arm that separates matched-form
best-case identifiability (all prior internal certifications) from a
robustness claim. Its failure is a kill-adjacent verdict (section 5.4,
K-T2).

### 4.6 The phantom-transfer sensitivity control (metric-sensitivity, not pin re-earning)

Refit A1-P0/P1 on SYN-T-NG with a free/amortized per-learner transfer
multiplier gamma_i restored, implemented as a NEW parallel head inside
`transfer/model.py`, never an edit to the A4-certified
`recognition.RecognitionNetwork` (section 2.4). Pre-registered
expectation: this variant FABRICATES, i.e. its per-learner p95 on the
null twin exceeds both the pinned-gamma variant's p95 AND the null band.
This is a SENSITIVITY control that the p95 tail metric bites on
per-learner fabrication on the kt-mirt harness. What it is NOT: it does
not re-earn or overturn the gamma pin, because (a) the design retains the
pin on the qmirt Gate B evidence and the structural argument regardless
of this arm's outcome, so no outcome changes the design, and (b) a free
gamma_i confounds the transfer-multiplier SEMANTICS with the extra output
CAPACITY it adds, so a fabrication does not isolate the transfer-trait
mechanism. Reading: fabrication confirms the metric is sensitive (kept
under that name); NON-fabrication is an informative reported finding
(section 5.4, informative non-kill), not evidence that the pin is
unnecessary, and the pin is retained either way. Reframed in v1.1 per
review; the earlier "re-earns the pin" claim is withdrawn.

### 4.7 Saturation and floor refusal, the range-matched null (CT7)

The A4 record is explicit that the passive existence gate INVERTED under
saturation on both profiles (CG6, `verdict_synthetic_complete.md` section
2): the gate fired HARDEST on saturated twins, because the no-growth
reference approximates a saturating curve worse than the growth model and
is handed a near-universal held-out NLL edge regardless of true dynamics.
A1's matched-null contrast (section 4.2) is the SAME held-out-NLL
comparison, so on a saturated target the with-transfer model can win the
same spurious edge over the no-transfer model and MANUFACTURE a signed
edge. The generator's ceiling gate bounds the GENERATOR's transfer, not
the FITTED model's NLL-edge inflation, so it does not by itself neutralize
the inversion. One nuance separates A1 from A4: A1's no-transfer reference
STILL carries the own-gain ceiling-gated term, so it can fit the
saturating OWN trajectory that A4's M0 could not, and the transfer route
adds only the cross-KC degree of freedom. The inversion risk is therefore
real but narrower, and it is met by a range-matched null, not by hoping.

Pre-registered mechanism. On SYN-T-SAT the fitted contrast on saturated
true-zero cells must stay within a SATURATION-MATCHED null band (drawn
from the SYN-T-SAT G_true = 0 subset, ceiling and floor subsets scored
against their OWN range-matched twin), not the unsaturated SYN-T-NG band.
This makes CT7 falsifiable in the intended direction: the gate must REFUSE
(return the null band) on range-starved targets, and a signed edge above
the range-matched band is a FAIL. The exclusion of range-starved targets
from primary verdicts is decided from the calibration-cohort dynamic-range
flag (`slices.saturation_stats`, the A4 primitive), never from the fit, so
the decision cannot itself be a fitting artifact. The refusal is symmetric:
positive verdicts are restricted to targets with adequate ceiling range,
negative verdicts to targets with adequate FLOOR range (bounded
interference gates negative transfer by (z - floor)+, so a floored target
cannot express interference and its contrast is ~0). This mirrors the
ceiling restriction on the floor side, so CT2's all-edges-all-seeds bar is
not set to fail on any negative edge whose target sits at floor. If the
saturation-matched null still inverts, that is a reported limitation and
range-starved targets are excluded, the same fix A4 flagged as still owed.

### 4.8 External reference: Junyi15 prerequisite graph

The positive half's external answer key, the PSI-KT validation template
extended to sign. The curated graph in `junyi_Exercise_table.csv` is an
EXERCISE-to-exercise prerequisite graph (`junyi_data.
load_prerequisite_graph`; the loader deduplicates to 835 exercises under
40 topics), so the external check runs at EXERCISE grain: the Junyi leg
fits G with item = KC = exercise (K ~ 835, full-K territory, section 3.1),
and a curated edge a -> c predicts G[c, a] > 0 (facilitation). The exact
usable edge count is a stage-0 MEASUREMENT, not an asserted number:
annotation may cover only ~370-742 of the exercises (open risk 10,
avenue-map caution), so the count of non-self-loop edges with attempt data
is computed in stage-0 and the null-graph permutation is restricted to the
annotated subset, with coverage reported (an unrestricted permutation
would be biased by the unannotated exercises). Metrics: AUC / MRR /
Jaccard of the recovered positive-edge ranking against the curated edge
set, and direction agreement (G[c, a] vs G[a, c]), each compared to a
NULL-GRAPH permutation over the annotated subset (random graphs of matched
density and degree), significant at p < 0.01. Two scope limits stated, not
hidden: the external check is CONDITIONAL on Junyi clearing the
order-based positivity precondition at exercise grain (section 5.2, an
UNMEASURED quantity that stage-0 must return before the external leg is
promised); and the negative half has NO external key anywhere (Junyi is
positive-only, open risk 6), so it rests on 4.1-4.7 plus, if A2 lands, the
Eedi misconception channel. The differentiator from LTKT/HawkesKT,
certified NEGATIVE sign, is externally uncorroborated by construction.

### 4.9 Stability

Seed-clustered sign consistency (every confirmatory edge sign-consistent
across all seeds, `battery.seed_cluster_verdict`) and split-half sign
reproducibility (split learners, refit G on each half, correlate signed
edges and count sign flips), reusing the A4 split-half primitives. This
is the "stable, not stable-and-wrong" audit turned on G itself, the gap
LTKT/HawkesKT leave open (single-run point estimates, no seed variance).

### 4.10 Low-rank sign fidelity (scaling arm)

On the full-K stress configs and real beds, fit G both full (or L1-
sparse) and low-rank (G = P Q^T). Certification requires the low-rank fit
to preserve synthetic GROUND-TRUTH sign-F1 on the full-K SYN-T-KG (the
load-bearing bar, because agreement is not correctness: low-rank ties
correlated rows, so the full and low-rank fits can agree while both are
wrong), and, secondarily, to preserve the full fit's signs above the
agreement bar. This closes the exact fidelity gap HawkesKT's CF ablation
never checked (it measured only AUC).

---

## 5. Pre-registered thresholds and kill conditions

All numeric bars are pre-registered bets, sized where possible against
the qmirt record and the A4 bars, flagged as judgment calls in section 8
otherwise. They may tighten, never loosen, and never after runs begin.

### 5.1 Synthetic certification gates (ALL must pass before any real-bed interpretation)

Baselines: every sign-F1 gate below is reported against, and must beat by
a pre-registered margin (>= 0.15 absolute), BOTH an all-zero predictor
(sign-F1 0 by construction, false-edge-rate 0) and a random-sign
predictor (~0.5 on binary +/- true edges), so a sparse G_true cannot be
gamed by predicting all-zero or by chance sign (section 4.1).

| Gate | Twin / target | Pass condition |
|---|---|---|
| CT0 resolution precondition | SYN-T-KG power sweep, both densities, D=3/5/8 + full K | per-edge sign-F1 reported vs effective sample/edge and decoupling; a pre-registered MINIMUM effective sample/edge (the smallest at which the curve first clears CT1) is frozen from the curve, edges below it are UNIDENTIFIED not zero; the CT1 bar must be reachable at some feasible (N, decoupling) at D=3 both densities, else K-T1; the bank-perturbed power curve (section 4.2 band) must not collapse the reachable region, else the claim is bank-limited |
| CT1 sign recovery | SYN-T-KG, D=3/5/8, both densities, decoupling {0.8, 0.75} | sign-F1 >= 0.80 at |g_ref| AND beating both baselines by >= 0.15; sign accuracy on true edges >= 0.85; false-edge rate on zero cells <= 0.05; positive-half and negative-half sign-F1 reported separately, negative-half sign-F1 >= 0.75 (judgment, floor-limited); seed-consistent signs (5/5); dose-response monotone in |g| and N*T; bars hold at decoupling 0.8 AND 0.75 |
| CT2 matched-null contrast | SYN-T-KG vs SYN-T-NG | paired contrast sign-correct for every true edge WHOSE TARGET HAS ADEQUATE RANGE on the relevant side (positive off-ceiling, negative off-floor, section 4.7) in every seed; seed-pooled p < 0.01; negative edges give a negative contrast (the signed half, not magnitude only) |
| CT3-i correlated-no-transfer, static | SYN-T-CNT | paired contrast and recovered |G| within the null band; false-edge rate <= 0.05 |
| CT3-i-dyn correlated-no-transfer, dynamic | SYN-T-CNT-DYN | shared time-varying trait: paired contrast and |G| within the null band; false-edge rate <= 0.05 |
| CT3-ii co-scheduling boundary | SYN-T-CO | recovered |G| collapses toward 0; contrast within band; NO fabrication above band (a magnitude above band here is a FAIL, reported as boundary breach) |
| CT3-iii shuffle-order | SYN-T-KG shuffled | recovered |G| and contrast collapse to <= 10% of the matched-form magnitude |
| CT3-iv reverse-direction | SYN-T-KG | reverse cell within the null band; reverse magnitude <= 20% of the true-direction magnitude |
| CT3-v endogenous scheduler | SYN-T-ENDO, both densities | mastery-gated schedule, G_true=0: fitted G and paired contrast within the null band; false-edge rate <= 0.05 (the causal warrant for the real leg) |
| CT4 per-learner tail | SYN-T-NG (and at full K, CT9) | p95 per-learner transfer effect <= 0.01 (KDD density) / <= 0.02 (EdNet density) proportion-of-max |
| CT5 mismatched generator | SYN-T-NS, BOTH densities, misfit family | on the worst-case misfit shape: true-zero cells false-edge rate <= 0.05 and contrast within band (no laundering); true edges sign accuracy >= 0.70 with implied-push overshoot <= 1.5x true on <= 10% of edges; also run at full K (CT9) |
| CT6 phantom sensitivity control | SYN-T-NG, free-gamma variant | EXPECTED: free-gamma p95 per-learner > pinned p95 AND > the CT4 band (metric is sensitive). Outcome does not change the pin (section 4.6); non-fabrication is an informative reported finding, not a pass/fail on the design |
| CT7 saturation + floor refusal | SYN-T-SAT (ceiling and floor subsets) | on range-starved true-zero cells the fitted contrast stays within the SATURATION-MATCHED null band (drawn from the SYN-T-SAT G_true=0 subset), not the SYN-T-NG band; a signed edge above the range-matched band is a FAIL; verdicts restricted to targets with adequate range on the relevant side (positive off-ceiling, negative off-floor), the restriction decided from the calibration-cohort range flag not the fit; pipeline flags "insufficient dynamic range" |
| CT8 D-scaling | SYN-T-KG, D=5 and D=8 | CT1/CT2 hold at D=5 and D=8, both densities (recovery survives realistic KC counts up to 8) |
| CT9 full-K + low-rank + laundering | K=189/515/~835 SYN-T-KG | sign-F1 >= 0.70 at full K (relaxed one band for scale) beating both baselines; low-rank preserves GROUND-TRUTH sign-F1 (not only >= 0.90 agreement with the full fit); CT5 laundering and CT4 tail also hold at full K (else scope to the K-range where they ran, K-T3) |

### 5.2 Real-bed positivity precondition (gates the real leg)

Before any real-bed transfer fit, the identifying structure must be
measured, not assumed. Per bed:

- Multi-tag beds (EdNet, Eedi): same-slot decoupling fraction over the
  candidate KC pairs must clear ~0.75 (triage: EdNet 0.87, Eedi 0.967).
  The 0.75 bar is not merely inherited from qmirt: CT1 certifies clean
  recovery and null-hold AT decoupling 0.75 on the kt-mirt harness
  (section 5.1), so a bed admitted just above 0.75 is not in an
  uncertified fabricating regime.
- Chain beds (KDD-KTracedSkills, Junyi15 at exercise grain): an ORDER-
  BASED decoupling (source practiced BEFORE target, in windows where the
  target is not co-practiced) must clear ~0.75 on the candidate pairs.
  The triage's same-slot / ever-attempted decoupling is NOT this
  quantity (triage flagged exactly this gap for Junyi); `prep`-time
  computation of order-based decoupling is a pre-registered stage-0 step,
  run before the fit. Junyi's exercise-grain order-decoupling is
  currently UNMEASURED (open risk 3; the topic-KC layer is a near-tree,
  so the order-decoupling that governs identifiability at exercise grain
  has to be computed, not assumed), and the EXTERNAL leg is promised only
  if this stage-0 measurement clears 0.75; if it does not, the external
  reference is unidentifiable (K-T5) and G1 keeps the KDD self-certified
  leg only. Only pairs clearing the bar are eligible edges; ineligible
  pairs are reported as unidentified, never as zero.

If no bed clears positivity on any candidate pair set, the real leg is
unidentifiable and G1 stays synthetic-only (K-T4).

### 5.3 Real-bed licensing conditions

- KDD-KTracedSkills (PRIMARY, chains): reuse the A4 frozen bank (RB0 tri-
  spec stability), the M0-bootstrap in-situ null (RB1: the whole
  transfer pipeline must return the null band on the bootstrap twin,
  p95 clause included), the saturation and floor restriction (transfer
  verdicts on the adequate-range target subset), and the shuffle-order
  permutation null on the real schedule. ENDOGENOUS-SCHEDULE CAVEAT: KDD
  is a mastery-managed ITS, so the practice schedule is chosen by a
  controller reacting to prior performance and is correlated with latent
  state, including state on the target c; the shuffle-order null does NOT
  neutralize this (it preserves the marginal schedule-state correlation,
  section 4.3). The causal "A raises/lowers B" reading is licensed only by
  CT3-v passing on the endogenous-scheduler twin (SYN-T-ENDO), and on the
  real bed the schedule-selection covariate available in the log (the
  learner's running success rate on the source before each target
  opportunity) is added as a control and the contrast re-read; if the
  edge does not survive the control it is reported as schedule-associated,
  not causal. A confirmed real edge requires: paired contrast beyond the
  permutation null at BH q = 0.05 across candidate pairs (BY-sensitivity
  reported, the A4 dependence discipline), sign-consistent across seeds,
  surviving RB0/RB1, and surviving the schedule-selection control. No
  external answer key on KDD; edges are self-certified and, where the
  problem-hierarchy ordering is informative, informally sanity-checked.
- Junyi15 exercise grain (EXTERNAL REFERENCE): identify via prerequisite
  chains on the order-decoupled edge subset; validate recovered positive
  edges against the curated graph (section 4.8). This is the leg that
  makes the positive-half sign claim externally corroborated rather than
  self-certified. The prerequisite graph is EXERCISE-to-exercise, so the
  fit is at exercise grain, item = KC = exercise, K ~ 835 (the loader's
  deduped catalog; full-K territory, section 3.1's K~835 stress point),
  the hierarchical MAP path handling difficulty. The exact usable edge
  count and annotation coverage (possibly only ~370-742 exercises, open
  risk 10) are stage-0 measurements, and the null-graph permutation is
  restricted to the annotated subset (section 4.8), not asserted here. The
  cycle-containing graph is used as an edge SET, never assumed a DAG. The
  same endogenous-schedule caveat applies (Junyi is mastery-gated); the
  external check corroborates the POSITIVE half only, and only if Junyi
  clears order-based positivity (K-T5).
- EdNet (multi-tag, SECONDARY, caveated): passes same-slot positivity
  (0.87) but the BUNDLE-EXPOSURE confound co-schedules KCs and is, in
  the A4 and triage rulings, fatal to clean causal reads. EdNet therefore
  runs only with a bundle covariate and its transfer edges are reported
  as bundle-confounded robustness evidence, never as a primary causal
  claim. Honest tension flagged: the task names EdNet a positivity-gated
  real bed; positivity it passes, causal cleanliness it does not.
- Eedi (multi-tag, decoupling leader 0.967): the strongest same-slot
  positivity in the program and the one bed with a real negative-transfer
  hook (misconception-per-distractor labels). Reserved for A2 as the
  negative-half flagship; noted here as the natural home for the negative
  edge's only external key, to be used if A2 lands.

### 5.4 Kill conditions and revision budget

Revision budget: at most TWO revision rounds on the synthetic
certification matrix (the A4 rule verbatim); each may fix mechanisms and
bugs, never loosen a threshold; every revision is a LEDGER entry. Pre-
registered fallbacks (e.g. the fixed-M refit, low-rank in place of full
G at K in the hundreds) are once-only, logged, non-consuming.

- K-T1 (sign recovery): CT0's power curve shows the CT1 bar is
  unreachable at any feasible (N, decoupling) at D = 3 (both densities),
  or CT1/CT2 still fail at D = 3 after the budget => G1 is DEAD on this
  design space. This is the H2 falsification decided on the power curve,
  not a single failed fit: even the gated, pinned, practice-driven form
  cannot recover signed edges at feasible N, T. Reported as such (exhaust-
  venues discipline), not buried.
- K-T2 (robustness): CT5 fails (SYN-T-NS launders gain-misfit into true-
  zero cells) after the budget => the sign claim is matched-form best-
  case only, not robust; retreat to a matched-form-only synthetic
  statement with the C3 threat named as unresolved. Not a full kill, a
  scoped-down claim.
- K-T3 (scaling): CT8/CT9 fail (recovery holds at D = 3 but collapses at
  D = 8 or under low-rank G at real K) => the sign claim caps at small K;
  the scaling limit is the reported result, real beds at K in the
  hundreds are out of reach.
- K-T4 (positivity): section 5.2 fails on every real bed => the real leg
  is unidentifiable; G1 is synthetic-only plus an honest positivity data-
  property verdict.
- K-T5 (external): Junyi alignment does not beat the null-graph
  permutation => the positive-half sign claim is self-certified only, no
  external corroboration; reported as a weaker but still novel result
  (LTKT/HawkesKT have neither synthetic certification nor an external
  check, so self-certified-plus-nulls still clears their bar).
- K-T6 (endogenous schedule): CT3-v fails (SYN-T-ENDO fabricates a signed
  edge under a mastery-gated scheduler) after the budget => the causal "A
  raises/lowers B" reading is unlicensed; the synthetic claim retreats to
  "signed influence under EXOGENOUS scheduling," and every real edge is
  reported as schedule-associated, not causal, unless it survives the
  real-bed schedule-selection control (section 5.3). A scoped-down claim,
  not a full kill, but it is the one that most directly threatens the
  paper's causal reading, so it is named separately from K-T4.
- Informative non-kill: CT6's phantom variant failing to fabricate => the
  p95 metric's sensitivity is not demonstrated on this harness (a reported
  finding about the CONTROL, not about the pin); the gamma pin is retained
  regardless, on the qmirt Gate B evidence and the structural argument
  (section 4.6). The pin's status does not depend on this arm's outcome.

THE CLEAN-NEGATIVE "SIGNED CLAIM UNSUPPORTED" VERDICT. The kills above are
mostly scope-downs, each with a publishable residual, which risks
reproducing A4's soft "partially certified" landing. Stated up front, the
pattern that makes A1 report the LEARNED SIGNED CLAIM ITSELF as
UNSUPPORTED (not merely scoped) is: K-T1 fires (per-edge sign
unrecoverable at feasible N, T), OR the negative half fails CT1/CT2
everywhere it has adequate floor range (so only facilitation, already
owned by prior art, survives), OR the laundering and causal warrants both
fail together (CT5 AND CT3-v both fail after the budget, so recovered
signs cannot be separated from gain-misfit and schedule-selection). Any of
these three is a clean negative on the novelty (certified LEARNED sign),
reported as such, distinct from the scoped residuals K-T2/T3/T4/T5/T6
which each keep a narrower but genuine claim.

---

## 6. Bed plan

Order: synthetic first (nothing real is interpreted before the full CT
matrix passes, the positive-control-first discipline), then the real
legs gated on positivity.

| Bed | Role | Identifying structure | Data / reuse |
|---|---|---|---|
| Synthetic | Certification | Both (chains + multi-tag) | `transfer/synth.py`; twins SYN-T-KG/NG/CNT/CNT-DYN/ENDO/CO/NS/SAT; D=3/5/8 x {KDD-, EdNet-shaped}; full-K stress at K=189/515/~835 with CT4/CT5 |
| KDD Algebra 2008-2009, KTracedSkills | PRIMARY real, self-certified | Prerequisite chains (arity ~1.08, order-decoupling ~0.80) | Reuse A4 `growth` KDD loader + frozen bank; 515 KCs; adequate-range target subset; step-grain items; schedule-selection control |
| Junyi15 (exercise grain) | EXTERNAL REFERENCE (positive half only, conditional) | Prerequisite chains (curated exercise graph as answer key) | `junyi_data.load_junyi_kc_traced` + `load_prerequisite_graph`; exercise grain K~835; usable edge count + annotation coverage measured in stage-0; order-based positivity gate first |
| EdNet KT1 | SECONDARY, bundle-caveated | Multi-tag (arity 2.2, same-slot decoupling 0.87) | Reuse A4 EdNet prep; bundle covariate mandatory; robustness only |
| Eedi (task 1/2) | NEGATIVE-half hook (deferred to A2) | Multi-tag (decoupling 0.967) + misconception labels | Reserved; the one external key for the negative edge |

Junyi is BOTH the identifying bed for its own external check and the only
external answer key; KDD is the primary self-certified identifying bed
with denser per-learner practice; EdNet/Eedi are the multi-tag
counterparts, EdNet caveated by its bundle confound and Eedi held for
A2. This split honors H4 (chains vs multi-tag) and the positivity triage
per bed.

---

## 7. Compute plan

Principles inherited from the A4 real-data lessons (memory: long-running-
jobs-verify; THINKING 2026-07-19/20): profile before scaling, prove on
the cheapest certifying profile first, measure never extrapolate, check
the named runner on any stall, kill by PID never by pattern, and treat
the verdict (not a slice count) as the deliverable.

- WHAT IS CHEAP AND WHAT IS NOT. Unlike A4's slice-based logistic gate,
  A1's confound arms each consume a FITTED G, so the matched-null
  contrast, the confound arms, the shuffle-order null, and the phantom
  variant are dominated by their stage-2 NEURAL refits, not by closed-form
  slice math. Only the CT0 power-curve summarization and the sign-F1
  scoring against injected ground truth are cheap CPU post-processing over
  cached fitted-G tensors. The A4 exclusive-node, parse-once discipline
  still applies to the permutation replicates (a naive re-parse per
  replicate blows the budget, the measured 37-CPU-h KDD lesson), but the
  binding cost here is GPU stage-2 fits, budgeted in the table below.
- SMALL-B / SMALL-D FIRST. Certify D = 3 (KDD- then EdNet-shaped) end to
  end before D = 5, then D = 8, then the full-K stress. The cheapest
  profile that certifies the method comes first; K = 515 is a parallel
  stress track, never the gate on the core sign result (the A4 EdNet-
  before-KDD-scale lesson).
- THE G TERM'S PER-STEP COST, the one real architectural cost increase
  A4's numbers do not cover. A4's `run_transition` does O(T*B*A_max) work
  (a gather/scatter over the practiced item's <= A_max = 6 tagged KCs).
  A1's transfer term, at each step, pushes every target c from each of
  the <= A_max practiced SOURCES a by a scatter-add of G's C-length column
  G[:, a], i.e. O(T*B*C*A_max) forward, with the same-shape backward
  through the dense column, and G is a continuous L1-penalized parameter
  (not sparsity-masked during training, so zero cells are not skipped in
  the forward pass). At K = 515, A_max ~ 1 (KDD-shaped), this is ~C = 515x
  A4's per-step work; the reviewer's O(T*B*C^2) is the bound only when
  every source is practiced every step (a dense schedule), which
  practice-gating does not produce. Two things follow. First, the FLOP
  ratio is not the wall-clock ratio: the update is a single fused dense
  vector add, GPU-parallel with a tiny constant, so wall-clock scales far
  below 515x, but by how much is a MEASURED question, not an extrapolation
  (the standing measure-never-extrapolate rule). Second, CT9 requires a
  DENSE full-K "full fit" as the low-rank reference, so the expensive case
  cannot simply be skipped.
- R-A1p, THE PER-STEP PROBE (runs BEFORE the D=3/5/8 + full-K plan is
  committed). Measure transfer forward+backward wall-clock per epoch and
  peak memory at K in {50, 189, 515, 835} on the 4060, and set every
  K-dependent GPU-h line below from the measurement. FALLBACK, pre-
  registered and non-consuming: if the dense fit at K = 515 or ~835
  exceeds a wall budget of 12 GPU-h/fit or OOMs the 4060, CT9's "full fit"
  reference is fit either with an L1-sparse-masked forward (cells L1 has
  zeroed are skipped) or on a learner-subsample, and the dense-at-full-K
  requirement is dropped, logged, and CT9's bar reads low-rank vs
  L1-sparse-full instead of vs dense-full. This is escalated to the
  program lead (open rulings), since it trades a weaker CT9 for
  feasibility.
- R0-A1, THE G-AUGMENTED STATIONARITY RE-STUDY (runs BEFORE the
  certification matrix, section 2.3). 3 seeds x ceiling epochs on SYN-T-KG
  and SYN-T-NG at D = 5, both densities, re-deriving `rel_tol`,
  `drift_tol`, and the epoch ceiling for the L1-penalized G objective
  (whose per-cell subgradient shrinkage is not the g_c/v_c/M landscape
  A4's numbers were tuned against), and validating the convergence
  positive control (SYN-T-NG true-zero cells shrunk below the CT1
  false-edge threshold and quiet at the stop). ~1-2 GPU-days on the 4060.
- Preprocessing writes compact npz (slice tensors + practice-indicator
  tensors + per-pair order-decoupling stats), rsynced to the cluster;
  raw beds never leave local disk; no credentials in the repo.

Itemized budget (K-dependent lines carry the R-A1p caveat; mirrors the A4
section-7 table format):

| Workload | Where | Estimate |
|---|---|---|
| Build + tests (transfer/ subpackage, generator twins, battery orchestration) | local CPU | 1.5-2.5 weeks agent time (the G term, signed-edge generator with 8 twins, confound/external arms, and the new shuffle-refit orchestration are net-new; the vendored ACT transition and stats primitives are reused) |
| R0-A1 stationarity re-study: 2 twins x 3 seeds x 2 densities | local 4060 | 1-2 GPU-days |
| R-A1p per-step probe: K in {50,189,515,835}, forward+backward timing/memory | local 4060 | hours |
| Bank calibration (reused A4): KDD tri-spec, EdNet | local 4060 | 3-9 GPU-h KDD, minutes EdNet |
| Closed-form / slice battery (CT0 power sweep, F1, matched-null contrast, confound arms i/i-dyn/ii/iv) | local CPU / SLURM CPU array | exclusive-node, cached-tensor replicates; ~1 day per density |
| Synthetic G fits, core: SYN-T-KG/NG x {KDD-,EdNet-shaped} x D=3/5/8 = 12 configs x 3 seeds | local 4060 + SLURM | 36 fits; small-K, ~20-40 min each at D<=8; ~1-2 nights |
| Synthetic G fits, confound/robustness: CNT, CNT-DYN, CO, SAT (KDD, ref D) + ENDO, NS (both densities, NS over the misfit family) + phantom-gamma on NG (both densities) | local 4060 + SLURM | ~30-45 fits x 3 seeds; ~2-3 nights |
| Full-K stress: K=189/515/~835, SYN-T-KG + NG(CT4) + NS(CT5), dense AND low-rank | SLURM 2-GPU cap + local 4060 | SET BY R-A1p; dense-at-K=835 is the cost driver and the CT9 fallback target |
| Real KDD: bank freeze (reuse) + G fit P0/P1 x 3 seeds + bootstrap twin + schedule-selection control refit | local 4060 + SLURM | SET BY R-A1p (K=515 dense); expect the largest single line |
| Real Junyi (exercise grain, K~835): G fit + external alignment + null-graph permutation | SLURM 2-GPU cap | SET BY R-A1p; low-rank likely mandatory at K~835 |
| Real EdNet (multi-tag, secondary): G fit with bundle covariate | local 4060 + SLURM | bundle-caveated robustness only |
| Report assembly, external-alignment readout | local CPU | hours |

Envelope: because the full-K and real-bed GPU-h are probe-set, the honest
calendar is stated as build 1.5-2.5 weeks, then the R0-A1 / R-A1p
prerequisites (a few days), then the synthetic matrix (small-K first,
~1 week), then the real legs whose length the probe determines. Verdicts
are incremental; each run gets a LEDGER entry with expectation-before /
reality-after.

---

## 8. Judgment calls flagged for review

1. The gamma pin rests on qmirt Gate B and the structural argument. CT6
   is a SENSITIVITY control that the p95 metric bites, NOT a test that
   re-earns the pin (v1.1 reframing); its outcome does not change the pin,
   and the free-gamma variant confounds transfer semantics with extra
   output capacity, so it cannot isolate the trait mechanism (section
   4.6).
2. Negative transfer is practice-gated and floor-bounded (not free
   decay), so rho = 1 is kept on monotone beds; a free-mu OU variant is
   an extension for non-monotone beds only. This is the design's bet that
   signed transfer and the Lemma-2 discipline are compatible; the
   confound and mismatched-generator arms are the test.
3. The reference edge dose |g_ref| is anchored to the 0.1-log-odds/opp
   population number; the dose-response sweep exists precisely because a
   single-dose claim is uninterpretable.
4. External validation certifies the POSITIVE half only (Junyi has no
   negative edges; nobody does). The negative half is synthetic-plus-
   nulls certified, with Eedi/A2 as the only possible external hook. This
   asymmetry is a stated scope limit.
5. EdNet passes positivity but fails causal cleanliness (bundle confound);
   it is secondary and caveated, not primary. Flagged because the task
   named it a real bed.
6. Low-rank G fidelity (CT9) is scored primarily against GROUND-TRUTH
   sign-F1 on the synthetic full-K config (agreement is not correctness);
   the secondary full-vs-low-rank sign-agreement bar (0.90) is a judgment
   number. The L1 weight and rank D are fixed on a held-out generator
   seed, never on a test config (section 2.2).
7. F1/accuracy/false-edge bars (0.80/0.85/0.05), the reverse-direction
   and shuffle-order collapse fractions (20%/10%), the CT5 degraded
   sign-accuracy bar (0.70), and the null bands (0.01/0.02, from A4 CG1)
   are pre-registered bets, tightenable only.
8. Item = step on KDD (reused A4 estimand) and item = exercise on Junyi
   (so the curated exercise graph applies directly); both are estimand
   choices, not tuning.
9. G is pooled across learners (PSI-KT posture), the identification
   discipline that makes the per-learner transfer trait unnecessary; per-
   learner influence is explicitly out of scope (it is the phantom the
   arm rejects).
10. The CT0 minimum effective sample per edge is read from the power curve
    (the smallest sample at which sign-F1 first clears the CT1 bar), frozen
    before any confirmatory read. Data-driven, but pre-registered as a
    procedure, so it is not a post-hoc cut.
11. The causal "A raises/lowers B" reading is licensed by SYN-T-ENDO
    (CT3-v) plus the real-bed schedule-selection control, not by
    response-blindness alone; without them the claim is signed
    ASSOCIATION under exogenous scheduling (K-T6). The realistic worst
    case is that real endogeneity is stronger than the twin, so the real
    control is mandatory, not optional.
12. Full-K and real-bed GPU-h are set by the R-A1p per-step probe, not
    extrapolated from A4's ACT numbers (the G term is ~C x A4's per-step
    cost). The dense-at-full-K CT9 reference has a pre-registered low-rank
    or subsample fallback escalated to the program lead.
13. The negative half's floor-side range restriction (CT7 mirror) means
    CT2's all-edges-all-seeds bar applies only to edges whose target has
    adequate downward range; a floored target that cannot express
    interference is reported unidentified, not failed.

---

## 9. Rebuttals

Three review points are addressed by partial rebuttal, conceding the
substance and pushing back on one load-bearing detail each; the folded
changes are named.

- On CT5 inheriting A4 CG5's non-informativeness (blocking): CONCEDED that
  SYN-T-NS reuses the `_syn_ns_partition` generator and the battery
  primitives, and FOLDED the two solid parts (run at both densities, beat
  a misfit FAMILY with a named worst case, section 4.5). PUSHED BACK on
  the core inference: A4's CG5 clause that fired everywhere is the misfit-
  FIRE FLAG (a detector of shape misfit), whereas CT5's metric is the G
  false-edge-rate on true-zero cells plus the paired contrast, computed
  against injected ground-truth signs. CT5 does not use the CG5 flag, so
  it does not inherit that flag's non-informativeness; the shared
  machinery (generator, F1/contrast) is ground-truth-bearing and
  informative. The non-informativeness claim is a category match on the
  generator, not on the statistic.

- On the transfer term being O(T*B*C^2) and A4's numbers being an
  unexamined analogy (blocking): CONCEDED fully that section 7's "small
  models fit on the 4060" was a parameter-count claim, not a runtime one,
  and FOLDED the demand (itemized table, explicit per-step analysis, a
  measured probe, a dense-reference fallback, section 7). PUSHED BACK on
  the exponent: with practice-gating only <= A_max sources are active per
  step, so the transfer term is O(T*B*C*A_max), not O(T*B*C^2); the C^2
  bound needs a dense schedule practice-gating does not produce. The
  magnitude concern (~C x A4's per-step cost at K = 515) stands and is why
  the probe, not an estimate, sets the full-K lines.

- On A1's per-edge object being exactly the per-KC resolution A4 could not
  resolve (blocking): CONCEDED the two facts that matter (the per-edge
  read is finer than a pooled read, and A1 reuses the same frozen bank
  whose 0.70-0.80 recovery is the A4 floor, so the floor rides along), and
  FOLDED the precondition (CT0 power curve, a stated minimum effective
  sample per edge, the bank-perturbed power check, a K-T1 decided on the
  curve, section 2.1.1). PUSHED BACK on the identity: A4's FAILING per-KC
  read estimated one KC's growth from that KC's own slices; A1's per-edge
  G[c, a] pools every learner with a decoupled a-before-c observation, a
  DIFFERENT pooling axis (across learners for one edge) whose effective
  sample grows with N. So the per-edge object is not literally the object
  A4 could not resolve; it sits between A4's failing per-KC read and its
  working across-KC pool, which is a reason to MEASURE the power (CT0), not
  to assume either outcome.

---

## Open rulings for the orchestrator

Decisions that exceed the design's authority and need the program lead;
each is pre-registered as a default the lead can override, so no ruling
blocks the build.

1. COMPUTE ENVELOPE. A1's G term costs ~C x A4's per-step ACT cost, and
   the enlarged twin matrix (8 twins, SYN-T-NS and SYN-T-ENDO at both
   densities, phantom refits, full-K CT4/CT5) plus the Junyi exercise-grain
   fit at K ~ 835 materially exceed A4's "tiny models" envelope. The
   probe R-A1p sets the actual GPU-h. Ruling needed: approve the enlarged
   envelope and the SLURM budget the probe will quantify, or cap the
   synthetic matrix (e.g. confounds at one density only) before the build.
   Default: approve, gate on the probe.

2. DENSE FULL-K CT9 REFERENCE. If the probe shows the dense G fit at
   K = 515 or ~835 exceeds 12 GPU-h/fit or OOMs the 4060, CT9's "full fit"
   reference drops to L1-sparse-full or a learner-subsample, weakening the
   low-rank fidelity guarantee. Ruling needed: accept the weaker CT9
   fallback, or authorize a SLURM allocation for the dense fit. Default:
   accept the fallback, logged and non-consuming.

3. JUNYI AS THE SOLE EXTERNAL KEY. The external corroboration rides
   entirely on Junyi clearing an UNMEASURED exercise-grain order-decoupling
   bar (0.75), covers the POSITIVE half only, and its annotation may reach
   only ~370-742 exercises. Ruling needed: promise the external leg
   contingent on the stage-0 measurement (and accept self-certified-only,
   K-T5, if Junyi fails), or seek an alternative external reference before
   committing. Default: contingent promise, K-T5 as the honest fallback.

4. THE CLEAN-NEGATIVE VERDICT. Section 5.4 pre-registers a pattern under
   which A1 reports the LEARNED SIGNED CLAIM itself as unsupported, rather
   than always landing on a scoped residual. Ruling needed: confirm the
   program wants A1 able to return a clean negative on the novelty (vs
   always reporting a defensible remnant, the A4 "partially certified"
   pattern). Default: keep the clean-negative option.

5. CAUSAL FRAMING SCOPE. The endogenous-schedule threat (K-T6) means the
   real "A raises/lowers B" reading is licensed only if SYN-T-ENDO passes
   and real edges survive the schedule-selection control. Ruling needed:
   confirm the paper may fall back to a signed-ASSOCIATION framing (still
   novel vs LTKT/HawkesKT's uncertified signs) if the causal warrant does
   not hold, or whether a failed causal warrant should scope A1 out of the
   real leg entirely. Default: association framing as the fallback.
