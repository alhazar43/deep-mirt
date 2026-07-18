# Trajectory-dynamics archaeology for kt-mirt

Internal archaeology, no web search. Distilled from the ability-trajectory
recovery program (`docs/trajectory_program.md`, `docs/trajectory_findings.md`)
and the parameter-recovery learning-dynamics study
(`docs/learning_dynamics_research_plan.md`,
`docs/learning_dynamics_progress.md`,
`docs/learning_dynamics_real_data_stability.md`,
`docs/LEARNING_DYNAMICS_STUDY.md`). All claims below are drawn from these
internal docs only; none independently re-verified for this note, so every
claim is marked non-load-bearing for the avenue map until re-checked against
the underlying result files (`deep_irt/traj_*/RESULTS_*.md`,
`deep_irt/bench/outputs/*`).

These two document families are different studies that happen to share a
codebase era (`feat/prediction-loss` branch, retired `deep_irt/` module).
The trajectory program (theta(e) and its rate dtheta/de) is the one directly
relevant to kt-mirt's G2 (per-learner ability growth). The learning-dynamics
study (alpha/beta/theta recovery speed under prediction loss) is a different
question, about how fast IRT parameters converge during training, not about
whether a learner's ability trajectory exists in real data. It is included
here only because the task asked for it and because its SLAM real-data
protocol has reusable ceiling-normalization machinery.

## 1. The saturation evidence that killed per-student rate recovery

The trajectory program ran the human front (E2 through E2d) on three real KT
datasets and one synthetic positive control, in this order:

- **EdNet-KT1** (E2). Wrong data-generating process outright, not a
  saturation problem: EdNet-KT1 is described as "a single-pass diagnostic
  stream" with no learning curve in the data, so the AFM (Additive Factors
  Model) concurrent-validity slope came back "near zero" on what the docs
  call "a wrong (single-pass) DGP." (`docs/trajectory_program.md` table in
  section 6b; `docs/trajectory_findings.md` results table.)
- **ASSISTments** (E2b), repeated practice. AFM concurrent signal "small
  positive (0.14)" but the item key used for the encoder was "the same skill
  id the AFM slope is fit on," so the result is circular (same identifier
  serves as both the model's item key and the classical model's grouping
  variable). The document explicitly discards this as evidence.
  (`docs/trajectory_program.md` §6b; result file referenced as
  `deep_irt/traj_kt/RESULTS_E2b.md`, not independently read here.)
- **KDD Cup 2010** problem-level, binary (E2c), the "decisive non-circular
  test." Model-free evidence a trajectory exists: within-student accuracy
  rises **6.1 points** and **74% of students improve**. The encoder's theta
  trajectory is stable (convergent readout **0.79**). But per-student RATE
  magnitude is unreliable: **split-half reliability of the per-student rate
  is only 0.17**. The stated reason is that the binary response is
  "near-saturated (80 percent correct)," leaving little per-step dynamic
  range. The non-circular AFM concurrent null on this data is read as "a
  measurement-floor artifact, not a refutation" — i.e., you cannot validate
  an unreliable measurement against an external criterion.
  (`docs/trajectory_program.md` §6b, §6c synthesis paragraph.)
- **KDD Cup 2010 graded K=4** (E2d), built from the KDD error/hint counts, to
  test whether adding graded (partial-credit-like) response categories
  rescues the rate. It does not: "the graded signal is STILL 80 percent
  top-category so no range was added," split-half reliability **0.19**
  (still low), AFM **0.026** (a null), and the existence gate (see §3 below)
  fails. Conclusion stated explicitly: "the saturation is a DATA-property
  limit, not a binarization artifact." (`docs/trajectory_program.md` §6b
  table row "E2d".)
- **Positive control**, synthetic data with a known rate (`RESULTS_poscontrol.md`,
  not independently read). Recovery corr(r_hat, r_true) = **0.46**, i.e. the
  method itself recovers rate on data engineered to contain one. But on this
  same synthetic data, the "gain over a fixed window" predictive-validity
  metric scores **-0.26**, and even the TRUE rate (not the estimated one)
  scores **-0.38** under that metric. This is the control that proves the
  predictive-validity metric, not the recovery method, was the source of the
  earlier null readings on EdNet/ASSISTments (E2/E2b).

Net summary from the docs: three levels of failure were disentangled.
(a) EdNet is the wrong DGP (no curve exists) — not a saturation problem.
(b) ASSISTments has a real repeated-practice DGP but a circular test.
(c) KDD Cup 2010, run cleanly, shows the trajectory genuinely EXISTS but its
RATE MAGNITUDE is not reliably measurable because the response variable
itself (near-saturated binary correctness, ~80% correct) carries too little
per-step dynamic range, and adding a coarse K=4 graded score does not change
that 80% top-category floor. The stated general lesson: "rate recovery needs
dynamic range (ordinal beats binary)," and the fix direction proposed is "a
genuinely error-rich or partial-credit human corpus," not a coarse re-coding
of the same near-ceiling behavior. (`docs/trajectory_program.md` §6b, §6c;
`docs/trajectory_findings.md` "Honest scope and limitations".)

## 2. Does anything speak to PER-KC growth rather than aggregate rate?

No. Every human-front experiment (E2, E2b, E2c, E2d) targets a single scalar
per-STUDENT rate, not a per-knowledge-concept (per-KC) decomposition. The
item granularity choices in these runs were about avoiding circularity
between the model's item key and the classical model's grouping variable
(moving from skill-id items in ASSISTments, judged circular, to
problem-level items in KDD Cup 2010, judged non-circular), not about
decomposing growth by concept. The kt-mirt program-context framing that
"per-KC decomposition on learning-heavy datasets is the untested escape
route" is consistent with what these documents show: the trajectory program
never attempted a per-KC rate, only a per-student aggregate rate, and its
one clean positive (KDD Cup 2010, problem-level) still hit a data-saturation
wall on the aggregate rate before any concept-level split was tried. This
archaeology does not contain evidence either for or against per-KC growth
being recoverable; it only establishes that the aggregate-rate approach on
KDD-like binary logs is dynamic-range-limited, which is a reason a per-KC
split (concept-specific streams could have different saturation profiles,
or a KC with denser easy/hard mixed items could avoid the 80%-correct
ceiling) remains an open, untested angle rather than a foreclosed one.
(Inferred from absence in `docs/trajectory_program.md`; not a direct
finding.)

## 3. The validity-gate methodology

**Why gain-over-a-fixed-window is ill-posed.** The originally planned test
was "predictive validity": fit a rate on a learner's early window, and check
whether it rank-predicts that learner's late-window accuracy GAIN
(`docs/trajectory_program.md` §6, E2 design paragraph, "the rate fit on a
learner's early window rank-predicts the late-window accuracy gain, with a
shuffled-order negative control"). The positive control (§1 above) showed
this metric is structurally broken: with a KNOWN true rate on synthetic
data, the metric scores -0.38, worse than chance, because "fast learners
plateau within the window and so show less late-window gain" — the
learners whose true rate is highest have already approached their asymptote
by the time the late window arrives, so they show the LEAST additional gain,
inverting the intended relationship. Any gain-over-a-fixed-window criterion
has this problem when rates are heterogeneous across respondents. The
permuted-order (shuffled) control that was meant to validate this metric
gave a false sense of security: it "sat at the same near-zero value as the
real correlation not because there was no signal but because the
gain-over-window metric is ill-posed for both," so a control can only
validate a metric that is itself well-posed. (`docs/trajectory_program.md`
§6b, paragraph beginning "The controls are what make a result
interpretable".)

**The replacement: existence gate, then parametric rate.** The methodology
that survived splits validation into two stages, run in order.

1. **Existence gate** (ground-truth-free, valid on real data). Compare
   held-out predictive improvement of a dynamic (with-rate) model against a
   constant-ability null. This is a model-comparison test, not a metric on
   the rate itself, so it sidesteps the plateau confound. On KDD Cup 2010
   this separation was significant at **p ≈ 5e-11** (with-rate vs no-rate).
   On the KDD graded K=4 data (E2d) this gate FAILS, which the docs read as
   confirming the saturation is a real data property (there genuinely isn't
   enough signal to license even the existence claim on that recoding).
   (`docs/trajectory_program.md` §6b, "Existence gate first, then the
   parametric rate"; `docs/trajectory_findings.md` "Three findings" item 3.)
2. **Parametric rate** (only after the gate passes). Fit the bounded-approach
   curve theta(t) = theta_inf − (theta_inf − theta_0)·exp(−r·t) to the
   model's own estimated item parameters (not fixed to ground truth, since
   real data has none) and read off r. Recovery ceiling under this
   estimated-item regime is reported around **0.41** (vs 0.46 with the
   items held at their true/oracle values in the synthetic positive
   control). The gate does NOT itself rank learners by rate — "the
   per-learner margin does not rank learners by rate" — so passing the
   gate licenses the claim "a trajectory exists," while the magnitude claim
   still needs the parametric fit and, ideally, external concurrent
   validity (agreement with a classical AFM/iAFM slope, "the Koedinger 2023
   reference," on non-circular item granularity).
   (`docs/trajectory_program.md` §6, E2 design; §6b/§6c.)

**Secondary validity checks named in the design** (not all independently
confirmed as run): split-half reliability of the per-student rate (used
as a diagnostic that flagged the KDD saturation problem, 0.17 / 0.19 above),
and "aligned-vs-responsive convergent validity." (`docs/trajectory_program.md`
§6, E2 paragraph.)

**Identifiability conditions stated for rate recovery in general** (from the
synthetic E0 study, which underlies the whole ladder): enough sequence
density, and a window that spans the curve's elbow (not too early, not
entirely past the asymptote); and the rate is affine-invariant in theta, so
an encoder's arbitrary internal scale does not bias the rate estimate — this
is given as the reason rate (not raw ability level) is the estimand of
choice. (`docs/trajectory_program.md` §4, §6 E0 entry.)

## 4. The SLAM anchored-extension real-data stability protocol

This is from the DIFFERENT learning-dynamics study (parameter-recovery
speed under prediction loss), not the trajectory/rate program, but it is the
one piece of machinery in scope with an explicit "ceiling normalization"
recipe, per the task.

**Dataset.** SLAM 2018 en_es (a language-learning response log), ordinal
K=3, catch-all items excluded. (`docs/learning_dynamics_real_data_stability.md`
"Design" section.)

**Protocol.**
1. Split named items into 70% base / 30% new.
2. Fit the base model on base-item histories only.
3. "Anchored extension": fit the new 30% of items' parameters with the base
   item scale held fixed (frozen).
4. "Full recalibration": fit ALL named items from scratch, independently,
   across many random seeds (16 in the reported run).
5. Compare the anchored new-item parameters against full recalibration.
6. **Normalize** the anchored-vs-full agreement by the seed-to-seed
   full-recalibration reliability ceiling — i.e., divide the anchored
   agreement by how much two independent full recalibrations agree with
   each other, which itself is bounded below 1.0 by ordinary training
   noise. This is the "ceiling normalization": it prevents overstating
   anchored-extension quality by comparing it only to a single
   (potentially noisy) recalibration run, and instead compares it to the
   noise-limited best a fresh full refit could do.
   (`docs/learning_dynamics_real_data_stability.md` "Protocol"; explicit
   statement "raw agreement alone would overstate the claim.")

**Gold-standard run parameters:** base_epochs=80, ext_epochs=300,
recal_epochs=80, recal_seeds=16, bootstrap_samples=10000, new_items=1499,
shared_learners=2593.

**Results (16-seed sweep, the "safer" denominator):**

| Parameter | Full-recal ceiling | Anchored recovery | Fraction of ceiling | 95% CI |
|---|---:|---:|---:|---:|
| Difficulty | 0.811 | 0.718 | 0.885 | [0.876, 0.894] |
| Discrimination | 0.717 | 0.592 | 0.826 | [0.817, 0.835] |

An older two-seed run gave higher (less reliable) fractions (0.910 /
0.843), and the docs flag the 16-seed sweep as the trustworthy one because
"the denominator is much less noisy." (`docs/learning_dynamics_real_data_stability.md`
"Results".)

**Explicit scope limits stated in the doc**: this supports real-data
item-scale stability under extension, NOT ground-truth item-parameter
recovery, NOT calibrated real theta recovery, and NOT external validity
against an independent assessment. A fresh SLAM rerun is currently blocked
because `rl/data/slam_raw` and `rl/data/slam_artefact_mc10` are reported
missing in the checkout as of the doc's writing.
(`docs/learning_dynamics_real_data_stability.md` "Interpretation",
"Current rerun status".)

**Relevance to kt-mirt / G2.** The ceiling-normalization idea — judge a
recovered quantity's stability against the seed-to-seed noise floor of an
independent full refit, not against an arbitrary absolute threshold — is
directly reusable for judging whether a recovered per-learner (or per-KC)
growth trajectory is "real" versus noise, distinct from the existence-gate
and split-half tools from §3/§5. It was applied here to item difficulty and
discrimination stability under an anchoring protocol, not to ability
trajectories or rates; porting it to trajectory work is unproven.

## 5. Reusable machinery and numbers for a growth-beyond-noise null battery

Machinery already implemented and exercised in the archaeology, potentially
reusable for kt-mirt's G2 null battery:

1. **Existence gate via model comparison** (dynamic-vs-constant-ability
   held-out prediction, §3 above). Concretely: fit a constant-theta null and
   a with-rate/dynamic-theta model on the same held-out split, compare
   predictive likelihood. Reported significant separation on KDD Cup 2010
   at p ≈ 5e-11; reported FAILURE (non-significant) on the graded K=4
   recoding of the same source data, which the docs treat as informative
   (confirms saturation rather than a coding bug). This is a ground-truth-
   free test usable directly on any real KT log, including per-KC slices.
2. **Split-half reliability of the recovered rate** (or growth parameter),
   as a diagnostic for whether the response signal carries enough dynamic
   range to support a magnitude claim at all — independent of whether the
   existence gate passes. Numbers on file: 0.17 (KDD binary), 0.19 (KDD
   graded K=4), both read as "too low to trust magnitude" even though the
   existence gate passed for the binary case.
3. **Positive control on synthetic data with a KNOWN growth parameter**,
   run BEFORE trusting any null result on real data. This is presented as
   load-bearing methodology, not just good practice: the ASSISTments/EdNet
   nulls were initially misread as "no learning signal" until the positive
   control (recovery corr 0.46 on synthetic, known-rate data) proved the
   estimation method works and the earlier null-reading metric (gain over
   fixed window) was the actual defect.
4. **Shuffled/permuted-order controls**, with the caveat from §3: a shuffle
   control only validates a metric that is already well-posed; it cannot
   rescue an ill-posed metric (both real and shuffled scored near zero for
   the wrong reason). Use only after the metric itself is validated on a
   positive control.
5. **Label-shuffled and content-free demonstration controls** (Min et al.
   2022, cited as load-bearing for the separate LLM in-context-learning
   front, E1/E1b/MT) — a true-vs-shuffled gap distinguishes genuine
   adaptation from priming/order artifacts. This is from the machine-front
   line of the SAME trajectory program, not the human front, but the
   general pattern (a shuffled-label or shuffled-order gap as the
   discriminating control) recurs across both fronts and is the template
   the docs converge on for "is this a real signal or an artifact of
   exposure/order."
6. **Bounded-exponential curve family** theta(t) = theta_inf − (theta_inf −
   theta_0)·exp(−r·t), used throughout as the parametric rate model, with
   the explicit design property that r is affine-invariant to the encoder's
   internal theta scale (so an arbitrary learned scale does not bias the
   rate estimate). Reusable as the default curve family for any growth
   estimator.
7. **Ceiling normalization against seed-to-seed full-refit noise** (§4
   above), reusable for scoring any recovered per-learner or per-KC
   parameter's real-data stability, separate from the existence-gate /
   split-half tools (§1, §2 above), which test existence and reliability
   respectively rather than absolute agreement with a reference fit.

None of this machinery has been applied at the per-KC level, and the
saturation finding (§1) is a warning specific to near-ceiling binary
correctness logs like KDD Cup 2010's ~80%-correct regime; whether a
learning-heavy or lower-accuracy dataset (the kind flagged as the
"untested escape route" in the kt-mirt program context) would hit the same
wall is untested by this archaeology.

## Caveats on this archaeology itself

This report is a second-hand read of the docs, not a re-run of any
experiment or a read of the underlying result files
(`deep_irt/traj_synth/RESULTS_E0.md`, `deep_irt/traj_kt/RESULTS_E2b.md`,
`deep_irt/bench/outputs/*`, etc.), which were referenced but not opened here.
The `deep_irt/` module referenced throughout has since been retired in favor
of the portable `kt-irt/` package (per `CLAUDE.md`), so none of this code is
runnable as-is; only the documented findings and numbers are portable. The
learning-dynamics study (alpha/beta/theta recovery speed, §4) is a distinct
research question from the trajectory/rate program (§1-3, §5) and should not
be conflated with it when building the avenue map; they share only a
codebase era and, per this task, the ceiling-normalization technique.
