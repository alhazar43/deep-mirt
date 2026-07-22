# kt-mirt Program Plan (living)

Maintained by the agent under the 2026-07-17 brief. Three documents:
PLAN.md (current plan, always up to date), LEDGER.md (run-by-run
changelog, expectation versus reality), THINKING.md (decision journal,
the reasoning behind every call). The paper framing rule applies
throughout: knowledge tracing and prediction are home, IRT is the
flavor and the explainability layer.

## Mission

Track actual learning inside tests and assignments with a multi-KC
(knowledge concept) knowledge-tracing model, using IRT readouts as the
interpretability layer. Showing either target claim is success:

- G1 (influence): practicing KC A increases or decreases performance on
  KC B, with recovered sign, beyond pre-registered nulls.
- G2 (growth): the per-learner digital twin shows ability growth or
  decline beyond noise, read on a trustworthy scale.

Explicitly out of scope: multidimensional theta in the MIRT sense. No
rotational indeterminacy is taken on. States are per-KC; IRT enters as
a readout, not as the latent geometry.

## Ruled decisions (user, 2026-07-17)

- Evidence bar: validity-gated pilots. Every detector is certified on
  synthetic ground truth before any real-data claim. Real-data
  demonstration is best-effort on the most favorable bed. Honest
  per-avenue verdicts; negatives recorded, not re-litigated.
- Beds: EdNet KT1, KDD Cup 2010, TIMSS (on hand in kt-irt caches);
  Junyi, ASSISTments, XES3G5M, Eedi (to acquire). Synthetic always in.
- Repo form: in-tree `kt-mirt/` package, vendored copy of the kt-irt
  core, no runtime import from `deep_irt`. kt-irt itself is untouched;
  it is occupied by the measurement-audit paper.
- Autonomy: full envelope including UT HPC (slurm layer documented in
  kt-irt/slurm/; credentials provided in session, never stored here).
- PSI-KT, GKT, GIKT are reference designs only. Nothing is copied from
  their repositories and PSI-KT is not benchmark-chased.

## Phases

- P0 scaffold: these documents plus the vendored package with passing
  tests. DONE when `python -m pytest kt-mirt/tests` is green.
- P1 research sweep: external literature, dataset facts, internal
  archaeology; load-bearing claims adversarially verified; output is
  the avenue map (`_planning/research/avenue_map.md`).
- P2 design: architecture, loss family, Q-matrix policy (1-to-1 vs
  1-to-many), and the pre-registered synthetic gates and null battery.
  Decisions recorded in THINKING.md before building.
- P3 synthetic gates: generator with known signed transfer and known
  growth curves; certify sign recovery, growth-beyond-noise detection,
  and clean null twins. A failed gate kills the avenue, not the rules.
- P4 real-data pilots: beds in order of expected signal; stability
  checks precede substantive readouts.
- P5 scale-out: HPC sweeps where the local GPU is the bottleneck;
  consolidation and verdicts.

## Carry-over constraints (from prior threads; re-verify, never assume)

- Phantom transfer. Free per-learner trait multipliers fabricate
  transfer and shared encoders mimic it passively (Q-MIRT gate B,
  lemmas 1-3). Transfer claims need explicit practice-gated coupling
  and interventional or ablation reads with null twins.
- Saturation wall. Aggregate human rate was unrecoverable on EdNet,
  ASSISTments, KDD (~80% top category). The per-KC route must prove it
  escapes this, not assume it.
- Readout trust. Shared readouts can be stable and wrong; anchoring or
  a separated item-parameter path is the default posture for the IRT
  layer (measurement-audit lesson).

## Adopted build order (P1 outcome, 2026-07-17)

Full avenue detail in `_planning/research/avenue_map.md` (six avenues
A1-A6 with gates, costs, and flagged uncertainty).

- Stage 0, bed triage and de-risking (RUNNING): raw correct rates,
  per-learner-KC opportunity distributions, KC-pair decoupling
  fractions, and pure-anchor counts computed from raw files (none of
  these numbers exist anywhere, so every bed choice is currently a
  bet); acquisition probes for ASSISTments, Junyi (both releases),
  XES3G5M, Eedi; primary-text reads of LTKT (the sole prior
  signed-transfer claimant), HawkesKT's sign validation, the two
  interpretability critiques, and PSI-KT's referee record.
- 1. A4 per-KC growth study (G2 primary), widened by user directive
  into a POSTURE-BY-BED MATRIX over three growth postures: ACTIVE
  (model imposes growth structure; certified against fabrication on
  no-growth twins), PASSIVE (unconstrained tracker read against
  noise: existence gate, permutation null, static twin, direction
  audit), MIXED (existence-gate-then-parametric-rate ladder;
  growth channels testable against zero). Posture disagreement is
  itself diagnostic. Then split-half reliability and truncation
  stress. Existing core suffices; the harness doubles as the A6
  certification battery.
- 2. A1 explicit-route signed transfer (G1 primary). Port the
  internally certified per-KC skeleton (zero-diagonal signed G,
  practice-gated, ceiling-gated gains, pinned gamma, amortized
  z0/lambda); synthetic D-scaling and the mismatched-generator arm
  first; the real leg is gated on stage-0 decoupling numbers.
- 3. A2 Eedi misconception-channel negative transfer (G1 flagship for
  the negative half; model-free leg first). Acquisition now.
- 4. A5 frozen-anchor digital twin (G2 measurement leg), conditional
  on A4 passing on at least one bed.
- 5. A3 signed readout audit on the stock core, last, full battery
  only; its likely failure is measurement-audit material, not G1's
  failure.
- 6. A6 portable certification battery write-up once battle-tested.

## ALIGNED BLUEPRINT (2026-07-21) -- B-then-A on the original phases

Reconciliation of the user's B-then-A correction with the original
P0-P5 / A1-A6 plan. NOT a new plan; it makes the original
synthetic-then-real discipline explicit.

- P0-P3 DONE. A4 growth (G2) SYNTHETIC certification complete: the
  coarse population-level growth detector is certified and
  profile-robust; per-KC resolution, saturation robustness, and
  magnitude are characterized limits (verdict_synthetic_complete.md).
- **B = P4 real-data pilot for A4/G2** (NEXT, now). Run the certified
  COARSE detector on real KDD (bridge built + reviewed) and real
  Junyi (loader to build). Scoped to the population-level claim only.
  PRECONDITION: a saturation-aware null fix -- real KCs sit near
  ceiling, and the synthetic verdict proved the gate false-fires
  there, so an unfixed real fire is untrustworthy. B closes the
  validity-gate cycle (synthetic -> real) the plan always mandated.
- **A = per-KC method refinement** (a revisit of A4/P3), DEFERRED
  until B shows which gaps actually bind. The synthetic evidence says
  per-KC resolution is a FUNDAMENTAL (gauge-floor) limit, not a data
  problem, so A is open-ended research to enter with real-data eyes,
  not blind. This is why it is B THEN A, not A-or-B.
- **Then G1 / transference** = A1 (signed per-KC influence) -> A2
  (Eedi negative transfer) -> A3 (readout audit), original avenue
  order. NOT started. Reuses the certified per-KC substrate. The
  item-KC mapping gates identifiability: influence needs cross-KC
  structure (1-to-many multi-tag items OR prerequisite chains); pure
  1-to-1 disjoint practice cannot identify it (H4). Junyi is the key
  bed -- its prerequisite graph is both the identifying structure and
  the external answer-key.
- A5 (trustworthy-scale measurement leg) folds into B where the real
  pilot needs an anchored scale for any magnitude read.

Compute (user-authorized): cluster = heavy duty (real-data
batteries); laptop = periodical light work (fix dev, small verifies).
Both GPUs in play.

## Open design questions (P2; each closes in its avenue design doc)

- A4 estimator pair for the existence gate at per-KC granularity
  (dynamic vs constant-ability model on learner-KC slices).
- A1 transition-module API on top of the vendored core (encoder
  demoted to recognition network for z0/lambda).
- Q-matrix policy: adopted per bed from avenue_map.md section 4
  (Q-row loadings with >=3 pure anchors per KC where attribution is
  claimed; exact expansion policy stated per run).
- Loss: prediction NLL home; auxiliary terms must earn their place.

## RE-SORT (2026-07-20, after user reset): verdict is the deliverable

The deliverable is CERTIFICATION VERDICTS, reportable incrementally per
posture and profile -- NOT "campaign 100% complete". Corrected task
order:

1. DONE -- neural sub-matrix verdict. The 24 banked production neural
   cells aggregated to `_planning/partial_verdict_neural.md`: the
   ACTIVE posture (ACT) certifies as a DIRECTION detector (silent on
   no-growth, fires on growth) but a poor MAGNITUDE estimator (~5-10x
   undershoot); the field-standard neural tracker (PAS-N1) FAILS all
   four faithfulness audits (CG7 does not beat a random encoder, CG8
   cross-KC contamination, CG9 order-sensitive, CG10 moves against the
   data ~40%). This is the program's thesis shown on its own code.
2. NEXT -- EdNet-matched slice pool (PAS-G / MIX, the G2 HEADLINE:
   does per-KC growth clear noise on a model-free read). Cheap profile
   (C=189, small KCs) that certifies the method and sidesteps the
   oversized-KC cost. One-unit validation RUNNING
   (outputs/a4/ednet_slice_validation.log); if it completes fast, run
   the 12 EdNet slice cells and report the EdNet G2 verdict.
3. PARALLEL TRACK (not a gate) -- the battery assembly fix (single-core
   permutation-assembly bottleneck) that unblocks KDD-matched slices,
   the density-matched-to-real-KDD certification. Ships behind a
   production-proof unit; does not block steps 1-2.
4. THEN -- real-bed KDD pilot (bridge already built + hostile-reviewed)
   once the method is certified synthetically.

Engine improvements banked en route (all kept, all with exact-decision
equivalence): replicate batching (6b579ac), analytic Newton derivatives
(06ff3ed, 10.7x), Schur arrow solve (685bed6). The slowness saga's
tactical lessons are standing rules; the strategic lesson is in
THINKING (2026-07-20).

Status: P0-P2 DONE; P3 harness BUILT + EXECUTED (504-test suite);
NEURAL certification verdict DELIVERED (partial_verdict_neural.md);
slice-pool G2 headline PENDING (EdNet-first). Progress in LEDGER.md;
reasoning in THINKING.md.
