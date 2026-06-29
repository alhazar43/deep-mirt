# Research Program 1. Dynamic Latent Ability as Structured Learning Dynamics

## Premise and the precise scientific question

The learner ability trajectory is already a first-class object in `deep_irt`. The
encoder emits a per-step scalar `theta_t`, causally aligned and read by the
decoder. Today its evolution is a black box inside the LSTM hidden state. This
program makes the evolution an explicit low-dimensional dynamical system, a
learned scalar gate `theta_t = rho_t * theta_{t-1} + (1 - rho_t) * theta_tilde_t`,
and asks one question.

**Question.** Treating the full trajectory `Theta_i = {theta_{i,1..T}}` as the
estimand, which of its gauge-invariant dynamical descriptors (gating inertia
`rho`, relaxation timescale `tau`, a coarse stability bin) are recoverable from a
**prediction-trained** encoder, and which of those survive on real
near-saturated logs where the per-student *rate* had no external validity?

This is deliberately narrow. It is one application, to the scalar trajectory, of
the lab's single identifiability thread, recovery of a structured latent object
under prediction loss, scored gauge-free through the Fisher-leverage lens
(`docs/LEARNING_DYNAMICS_STUDY.md`). Paper 2 ran that thread on item parameters.
A later program (anchored multidimensional field) would run it on a coupling
operator. **This program does not claim a novel architecture.** The gated form is
PSI-KT's transition. The contribution is the recoverability question and the
controls, not the recurrence.

Two prior results are settled and must not be re-derived.

- The human per-student **rate** is unrecoverable from standard KT logs. Cause
  fully diagnosed, response-distribution near-saturation (around 80 percent top
  category, no per-step dynamic range), not the coding (graded K=4 failed in E2d)
  and not the method (E0 recovers a rate when the signal carries it). Split-half
  reliability of the per-student rate on KDD is 0.17. Do not propose another
  standard-log rate recovery (`docs/trajectory_findings.md`).
- The trajectory **provably exists** model-free. On KDD within-student accuracy
  rises +6.1 points, 74 percent of students improve, the encoder theta track is
  stable across seeds (convergent readout 0.79, range 0.79 to 0.91). So existence
  and population-level stability are established facts to build on, not findings
  to reclaim.

## Hypotheses

The two fast-fail probes (H0a, H0b) gate the program before any encoder is built.
See the milestone ladder for their go/no-go wiring.

- **H0a (LSTM leaves identifiability on the table).** The black-box LSTM's
  *implicit* inertia, estimated post hoc by fitting AR(1) to its trained
  `theta_t` tracks, is **seed-unstable**. Only then does an explicit gate have
  room to add identifiability. *Falsifiable.* If the LSTM's implicit `rho` is
  already seed-stable, the gate buys nothing and the program is "a constrained
  LSTM." Runs on existing checkpoints, zero new code.

- **H0b (the structure-vs-rate split has a home).** Within one data-generating
  process, a dimensionless inertia/retention parameter recovers in a regime where
  a single-curve **magnitude rate** is ill-posed, and dies on a plateau regime.
  The only regime where this is well-posed is **repeated transients**
  (spaced-repetition / forgetting), where many decay episodes pin `rho` even when
  one curve's rate is unidentified. *Falsifiable and the headline falsifier.* If
  `rho` dies whenever the rate dies on the synthetic saturation variant, the
  "survives where rate did not" claim is dropped, not reported as a sub-result.

- **H-I1 (positive control).** On synthetic data with a *known* dynamical law and
  a **starved** `theta_tilde` (rank-1 or linear readout), the gated encoder
  recovers `rho` (rank-correlated with truth across 5 seeds) and the ordinal
  timescale `tau`, while a no-dynamics null recovers a flat `rho`. *Falsifiable.*
  `rho` not recoverable on clean data means the form or its identification is
  broken, stop at I-M1.

- **H-I2 (explicit earns its place on identifiability, not accuracy).** At matched
  prediction budget the gate achieves held-out NLL parity with the LSTM **and**
  recovers the known synthetic `rho` with lower seed-variance than the LSTM's
  post-hoc AR(1) `rho`. Validity is anchored to synthetic ground truth, not to
  seed-stability alone. *Falsifiable.* No identifiability gain at parity demotes
  the gate to an ablation, not a headline.

- **H-I3 (structure survives where rate did not, conditional).** On
  repeated-transient real data (Duolingo SLAM, adapter already present at
  `rl/src/ordrec/data/slam.py`) the existence gate fires and the dimensionless
  inertia recovers and clears the 0.17 split-half floor, even though a single
  learning-curve rate is ill-posed. On plateau-dominated single-pass data
  (EdNet-KT1) the transient is absent and the descriptors do not recover. The
  split is the conditional headline, pre-registered to H0b and to SLAM carrying a
  resolvable transient.

- **H-I4 (gauge discipline, a control on our own metrics).** Only G-invariant
  (`rho` exactly, stability bin) and W-ordinal (`tau`) quantities replicate across
  seeds. Quantitative curvature and smoothness, being neither G- nor W-invariant,
  do not. Verified by a numerical affine-gauge transform and a monotone-readout
  warp applied to recovered trajectories.

- **H-I5 (cross-encoder invariance), DEFERRED.** The recovered `rho`-ordering of
  learners is stable across lstm / transformer / dkvmn backbones above a
  matched-null floor. Deferred to backlog until I-M1 and I-M2 pass, to avoid
  spending a 3-backbone x 5-seed sweep on a form that has not earned its place.

## Integration with prior work and prior-art positioning

### Reuse, do not rebuild

| Asset | Path | Use |
|---|---|---|
| Existence gate (dynamic-beats-constant-theta held-out NLL, Wilcoxon, bootstrap; MLE constant-theta comparator) | `deep_irt/traj_kt/run_e2c.py:399,418` | License every descriptor claim before any magnitude read |
| Known-curve synthetic generator + oracle ceiling | `deep_irt/traj_synth/data_gen.py`, `metrics.py:93` (`oracle_rate_mle`) | Base for the known-dynamical-law generator and the upper bound |
| Validity battery (split-half reliability, convergent aligned-vs-responsive, predictive + order-shuffle null) | `deep_irt/traj_kt/validate.py:55,237,285` | The trajectory-invariant probes and the 0.17 floor comparison |
| Cross-encoder warm-checkpoint harness | `deep_irt/bench/run_arch_trajectory.py` | The DEFERRED H-I5 leg only |
| Gauge-free recovery metrics (sign-align, Spearman/Pearson) | `deep_irt/bench/metrics_bench.py` | Score rank-vs-magnitude so attenuation is visible |
| `model.track()`, `BaseSeqEncoder._shift` | `core/model.py:460`, `core/encoder.py:190` | Trajectory extractor and causal alignment, unchanged for scalar theta |
| SLAM data adapter | `rl/src/ordrec/data/slam.py` | The repeated-transient real-data home for H-I3 |

### Avoid repeating (the dynamic-theta negatives)

Do not re-run a per-student rate recovery on EdNet / ASSISTments / KDD. Do not
re-establish that the trajectory exists or that theta is seed-stable. Both are
settled. The open object is *per-respondent dynamical structure under prediction
loss*, gated on a synthetic precondition and a real corpus with genuine
transients.

### Prior-art positioning and the novelty gap

- **PSI-KT** (Zhou et al., ICLR 2024, `mlcolab/psi-kt`, arXiv 2403.13179) is the
  binding constraint. Its transition `m_n = r_n z_{n-1} + (1 - r_n) mu_tilde_n` is
  literally the gated form, with interpretable traits, fit by variational Bayes
  (ELBO). The architecture is taken. The defensible residual is (i) **prediction-loss
  training**, asking whether the structured trajectory is recoverable from a
  prediction-trained encoder at all, which PSI-KT never tests because it fits the
  generative model directly, (ii) the **gauge-free invariant and
  seed/encoder-stability analysis** as the object of study, (iii) the **saturation
  confrontation**. State the honest bind in the paper. If recovery is "no" (likely
  under saturation and Fisher), we have shown the weaker objective recovers less,
  which arguably argues *for* ELBO. If "yes," it is a narrow methods equivalence.
  The expected and still-publishable answer is "partially, in rank."
- **VTIRT** (arXiv 2311.08594), **DynAEsti**, dynamic state-space IRT (Wang,
  Berger, Burdick 2013), **D-BIRD** (arXiv 2506.21723) index ability by *calendar
  time*, fit by EM/VI/MCMC, validate by prediction. We re-pose the trajectory as
  the estimand indexed by accumulated *evidence* (response steps) and ask
  recoverability under prediction loss. That re-posing is uncontested.
- **DKT** (Piech 2015) is the black-box baseline. **Vie and Kashima** (arXiv
  2309.12334) prove the implicit dynamic-MIRT object exists inside DKT. We test
  whether making it explicit buys recoverability, never accuracy.
- **Latent ODE** (`YuliaRubanova/latent_ode`) and **Neural CDE**
  (`patrick-kidger/torchcde`) are deferred. The continuous-time edge is unused
  under the evidence clock (response steps, not calendar time). The gate, not the
  ODE, is the competitor.

**The novelty gap in one sentence.** No prior work treats the learner ability
trajectory as a recoverable structured object and asks which of its dynamical
invariants survive prediction-loss training on near-saturated data, with
gauge-free, Fisher-leverage controls and an explicit no-dynamics null.

## Architecture on deep_irt and the baseline to beat

**Baseline to beat.** The black-box LSTM, `encoder="lstm"`
(`deep_irt/core/encoder.py:200,244`), which already emits a per-step scalar theta.
The gate must earn its place against it (H0a, H-I2).

`theta` stays **scalar**, so the decoder, loss, `track`, and recovery are all
inherited unchanged. New files only, no edits to Codex-owned core.

```
deep_irt/struct_dyn/structured_encoder.py  # GatedThetaEncoder(BaseSeqEncoder)
deep_irt/struct_dyn/model.py               # StructuredIRTModel(DeepIRTModel) shim
deep_irt/struct_dyn/datagen_traj.py        # known-dynamical-law generator + variants
deep_irt/struct_dyn/invariants.py          # rho / tau / stability / gauge-warp tests
deep_irt/struct_dyn/run_structure.py       # driver: fast-fails -> synth -> adjudication -> real
```

**Encoder (the only build of substance).**

```
theta_t       = rho_t * theta_{t-1} + (1 - rho_t) * theta_tilde_t
rho_t         = sigmoid(g([h_t, theta_{t-1}]))   # low-complexity head
theta_tilde_t = w_tilde(h_t)                     # STARVED: rank-1 or linear in h_t
```

`GatedThetaEncoder(BaseSeqEncoder)` keeps `_direct_hidden` so the decoder's
state-conditioned alpha head is unchanged, and **overrides
`aligned_theta_and_state`** (`encoder.py:137`) so the prediction path runs through
the gate, not `theta_proj(h)`. This override is the single subtlety. Without it the
gate is decorative.

The critical identifiability fix, operationalized. `theta_tilde` must be
**starved** (rank-1 or linear). A high-capacity `theta_tilde` reproduces any
target and frees `rho_t` to be anything, including a stable degenerate. Capacity is
a knob, not a hope. I-M1 verifies by swapping `theta_tilde` capacity and checking
`rho` stability.

`StructuredIRTModel(DeepIRTModel)` extends `_ENCODER_CHOICES += ("gated",)`
(`model.py:131`) and overrides the `_make_encoder` staticmethod (`:885`, called as
`self._make_encoder` at `:238`). No other override is needed because theta stays
scalar.

**Cut from the architecture.** The LinearSSM second formulation is a footnote, not
a build, the gate nests it at constant `rho`. Hamiltonian is rejected, conservative
bias is wrong for dissipative relaxation. SDE is deferred, diffusion `sigma` is not
identifiable under point-estimate prediction. Latent-ODE / Neural-CDE deferred per
the evidence clock.

## Data, synthetic-ground-truth-first, then real, saturation-aware

Per the MIRT-generation literature, no package ships a temporal generator. We own
the dynamics and call a static GPCM draw per slice.

**Synthetic positive control, first and decisive.** Extend
`traj_synth/data_gen.py` in `datagen_traj.py`, planting a known law:

- Gated inertia, the user's form, known `rho` to recover persistence.
- OU / mean-reverting, `theta_t = theta* + rho(theta_{t-1} - theta*) + sigma eps`,
  known `theta*`, `rho`, `sigma`.
- A known practice-impulse gain so controllability has ground truth.
- **Saturation variant**, high-baseline-correct, low-dynamic-range (the
  EdNet/ASSISTments profile), theta pinned on a plateau. Drives H0b directly,
  does `rho` recover where the rate does not within the same DGP.
- **Repeated-transient / forgetting variant** (spaced-repetition), many decay
  episodes per learner. The only DGP where a dimensionless retention is well-posed
  while a single-curve rate is not. This is the regime H-I3 needs.

**Real, chosen around the blocker.**

1. **Duolingo SLAM 2018** (human, spaced repetition). The repeated forgetting /
   relearning transients pin `rho` even when a single curve's rate is ill-posed.
   This is the **load-bearing real home for the headline**, not an optional
   stretch. Ingestion already exists (`rl/src/ordrec/data/slam.py`), so cost is a
   metadata bridge to `SequenceDataset`, not a fresh acquisition. Pre-registered
   gate, SLAM must carry a resolvable transient (existence gate fires).
2. **LLM in-context adaptation curves** (machine front, E1b precedent). Transient
   guaranteed by construction. Note honestly, the *rate* also recovers here, so
   this is **not** the structure-vs-rate split. It is a positive recoverability
   leg aligned with the north star (humans or LLMs), nothing more.
3. **KDD Cup 2010** (human, on disk). Existence-gated, used for the **settled
   existence and population/shape invariants only**, not transient-stratified for
   `rho` (selection on dynamic range would make the split-half circular).
4. **EdNet-KT1** (saturated negative contrast). Plateau-dominated single-pass
   binary. The cell where the descriptors die. The contrast against SLAM is the
   H-I3 evidence.

Keep it small for the 8 GB GPU, hidden <= 64, T <= 200, synthetic sweeps overnight.

## Evaluation per hypothesis, with controls

| Hypothesis | Measurement | Controls |
|---|---|---|
| H0a | Seed-variance of post-hoc AR(1) `rho` fit to existing trained LSTM tracks | Compare against the synthetic-known seed-variance an identified estimator would show |
| H0b | `rho`-recoverability under the existence gate on the saturation variant vs the forgetting variant, same DGP | Single-curve rate recovery as the contrast that *should* fail in saturation |
| H-I1 | Spearman(`rho_hat`, `rho_true`) and (`tau_hat`, `tau_true`) over 5 seeds; oracle-MLE ceiling | No-dynamics null must recover flat `rho`; `theta_tilde` capacity-swap must not move `rho`; permuted-order floor on a well-posed metric |
| H-I2 | Paired held-out NLL parity (Wilcoxon signed-rank, reuse `run_e2c.py`) + seed-variance of gate `rho` vs LSTM post-hoc `rho` + both scored against synthetic truth | Win requires parity AND lower seed-variance AND closer-to-truth on synthetic, not seed-stability alone |
| H-I3 | Existence gate then `rho`/stability recovery per corpus; split-half reliability of `rho_hat` must clear the 0.17 rate floor | Constant-theta comparator (`run_e2c.py:399`); EdNet plateau as the negative; SLAM existence-gate as the precondition |
| H-I4 | Replication rate of each descriptor across seeds; explicit pass/fail that curvature does not replicate while monotonicity / turning-points do | Numerical affine-gauge transform + monotone-readout warp; confirm `rho` exactly invariant, `tau` ordinal, curvature variant |
| H-I5 (deferred) | Sign-aligned Spearman of learner `rho`-ordering across backbones | Matched-null floor from `run_arch_trajectory.py` |

**Identifiability guards baked into every read.** Items held fixed during the
per-respondent read (kills the rate/difficulty confound). `rho` read only in the
transient (it is unidentified at fixed points). `rho_t` and `theta_tilde_t` both
low-complexity. **Stability is not validity**, a degenerate `rho -> 1` is perfectly
seed-stable, so seed-stability alone never licenses a recovered-inertia claim. Only
the synthetic positive control licenses magnitude. On real data, with no ground
truth, the claim is restricted to existence-gated rank and the cross-floor
comparison.

## Milestone ladder, each with go / no-go and the fast-fails first

| Stage | Scope | Go / No-go |
|---|---|---|
| **Fast-fail A (H0a)** zero new code, ~1 afternoon | Fit AR(1) to existing trained LSTM theta tracks across seeds; measure `rho` seed-stability | GO: LSTM implicit `rho` seed-unstable, the gate has identifiability room. **NO-GO: LSTM `rho` already seed-stable, the gate is a constrained LSTM, reconsider the whole program before writing one file** |
| **Fast-fail B (H0b)** generator only + post-hoc estimate, ~1 day | On the saturation and forgetting variants, test whether `rho` recovers where a single-curve rate does not | GO: `rho` recovers in the repeated-transient regime where the rate is ill-posed. **NO-GO: `rho` dies whenever the rate dies, DROP the "survives where rate did not" headline, revert to existence + shape invariants** |
| **I-M0 MVP (~1 day)** | `GatedThetaEncoder` forward; `StructuredIRTModel` shim; assert scalar-theta path unchanged; tiny-synthetic train | GO: trains, held-out NLL within noise of LSTM. NO-GO: cannot match LSTM, form over-constrained, revisit head |
| **I-M1 positive control** | Recover known `rho`/`tau`; no-dynamics null flat; `theta_tilde` capacity-swap stability | GO: Spearman(`rho_hat`,`rho`) >= ~0.7 over 5 seeds, null flat, `rho` invariant to `theta_tilde` capacity. **NO-GO: `rho` unrecoverable on clean data, STOP, identifiability broken** |
| **I-M2 LSTM adjudication** | Gate vs LSTM, matched budget, parity + recovery-of-truth + seed-variance | GO: NLL parity AND gate `rho` closer to synthetic truth with lower seed-variance than LSTM post-hoc `rho`. NO-GO: demote to "constrained LSTM" ablation, do not headline |
| **I-M3 saturation / forgetting confrontation (conditional headline)** | Existence gate + descriptors on SLAM, KDD (shape only), EdNet contrast; + synthetic forgetting variant | GO: descriptors recover where repeated transients exist, die on plateau, the structure-vs-rate split. NO-GO (uniform death): report the clean negative, restrict to synthetic + machine, headline already dropped by H0b |
| **I-M4 cross-encoder invariance (DEFERRED)** | Gate added as a fourth arm to `run_arch_trajectory.py` | Backlog until I-M1 and I-M2 pass |

The two fast-fails are the make-or-break. Either can kill or re-scope the program
in under two days before the encoder is built.

## Risks and de-risking

- **Inertia is a rate-class quantity and may inherit the rate's
  non-identifiability.** `rho -> 1` on saturated data is exactly what the data
  supports, re-deriving "no dynamics" as "high persistence." *De-risk.* Fast-fail B
  pre-registers this death; report `rho` only where the existence gate fires; SLAM
  / the forgetting variant carries the positive claim.
- **Stability is not validity.** A degenerate `rho -> 1` is seed-stable. *De-risk.*
  Magnitude is licensed only by the synthetic control; real-data claims are
  existence-gated rank plus the cross-floor comparison, never seed-stability alone.
- **`theta_tilde` capacity frees `rho`.** *De-risk.* Rank-1 / linear `theta_tilde`,
  verified by the I-M1 capacity swap.
- **Over-smoothing erases the object.** A smoothness prior on saturated data
  flattens the curve and manufactures high persistence. *De-risk.* No smoothness
  regularization on the gate; never report quantitative curvature (H-I4).
- **Justify-over-LSTM bar unmet.** *De-risk.* Fast-fail A and I-M2 are hard gates;
  if they fail the honest contribution is "a constrained LSTM" and we say so.
- **PSI-KT subsumes the architecture.** *De-risk.* The contribution is
  recoverability-under-prediction-loss plus gauge-free invariant analysis plus the
  saturation result, stated up front; the build claims no architectural novelty,
  and the paper states the honest ELBO-vs-prediction bind.
- **W-warp contaminates non-ordinal invariants.** *De-risk.* The numerical
  gauge/warp test (H-I4) runs before any cross-dataset claim.

## Honest feasibility verdict and what was cut

**Verdict: GO on the spine, conditional GO on the headline.** The spine,
synthetic-known recovery (I-M1) plus the LSTM adjudication (I-M2) plus
existence-gated shape invariants on real data, is architecture-cheap (scalar theta,
no Codex edits, M effort) and well supported by reused infrastructure. The
**headline**, "trajectory structure survives where the rate did not," is **not
asserted, it is gated**, on fast-fail B and on SLAM carrying a resolvable
transient. SLAM's adapter already exists, which lowers the cost materially, but the
data-property gate stands. If both fail, the program still ships a real result, the
synthetic recovery plus the LSTM adjudication plus the clean negative, which is
publishable and pre-registered.

The deeper honest point. This is **one identifiability result on a third class of
object**, not a new architecture and not two programs. Paper 2 ran the thread on
item parameters. Whether this scalar-trajectory slice carries a chapter on its own
merit, or belongs as one section of a single identifiability chapter alongside the
anchored-field slice, is a scoping decision for the thesis author, surfaced here
rather than papered over.

**Cut or deferred from the draft.**

- Program II (anchored multidimensional field) removed entirely. It is a separate
  program and a separate file. This document is Program I only.
- LinearSSM second formulation, demoted to a footnote (the gate nests it).
- SDE rejected as unidentifiable under point-estimate prediction; Hamiltonian
  rejected as the wrong (conservative) bias; latent-ODE / Neural-CDE deferred under
  the evidence clock.
- Cross-encoder invariance (H-I5 / I-M4) deferred to backlog until the gate earns
  its place at I-M1 and I-M2.
- The "structure survives where rate did not" headline is no longer assumed. It is
  pre-registered as conditional on the repeated-transient regime, the only place it
  is well-posed, and dropped if fast-fail B says `rho` dies with the rate.
- The two fast-fails moved to experiment zero, ahead of the MVP, per the critique.
