# The Ability-Trajectory Recovery Program

Internal research plan, not for external sharing. This is the science
behind the outward-facing `data_proposal.tex`. It states the problem, the
estimand, the questions, the three fronts, the methodology, and the
concrete experiment ladder. It is a living document, updated as prior
search and results come in.

## 1. Problem and estimand

The three anchoring legs (SLAM re-estimation, EdNet cross-instrument,
synthetic cross-format) established that a learned ability scale is stable
under extension. They never tested the framework's distinctive estimand,
the ability trajectory itself. Item response theory estimates a fixed
trait. Knowledge tracing predicts the next response. We estimate a third
quantity, the latent ability written as a function of accumulated
evidence, and the rate at which it changes.

Estimand. For a respondent, the trajectory theta(e) of ability over
accumulated evidence e, and its rate dtheta/de. The respondent is
interchangeable. For a human learner the trajectory is a learning curve.
For a language model it is an in-context adaptation curve, the speed at
which performance rises as context accumulates.

## 2. Research questions

- RQ0, recovery. Does a prediction-trained encoder-decoder recover a
  known ability trajectory and its rate from a response sequence, and
  under what density and identifiability conditions does the rate become
  estimable. This is the precondition for everything else.
- RQ1, machine. Does a language model under growing in-context evidence
  show a recoverable ability trajectory on a shared item scale, and is the
  rise genuine in-context learning rather than format priming.
- RQ2, human. Does the recovered rate track real human learning on dense
  knowledge-tracing sequences.
- RQ3, transfer. Do item difficulties derived from machine respondents
  predict human item difficulty once grading is rubric-based rather than
  exact-match.

## 3. The three fronts, run in order

These are three avenues on one estimand, from the most controllable
respondent to the least. Operating rule, exhaust the approaches within a
front before conceding it, and record dead-ends honestly rather than
quietly dropping them.

1. Machine learning curves. Recover an LLM in-context adaptation curve
   theta(k) and rate as a function of shot count k, on a shared scale,
   with controls that separate learning from priming.
2. Human learning-curve rate recovery. Dense real knowledge-tracing
   sequences where ability genuinely moves, after the synthetic
   precondition study below.
3. Respondent transfer beyond exact match. Break the Spearman 0.34
   ceiling using graded items where models genuinely attempt tasks.

## 4. Methodology

Recovery procedure. Train the encoder-decoder under prediction loss on
response sequences. The encoder exposes a per-step ability estimate
theta_hat_t. Fit a parametric curve to the per-step trajectory and read
off the rate. Score against ground truth on synthetic data, and against
held-out responses on real data.

Curve family. A bounded approach to an asymptote,
theta(t) = theta_inf - (theta_inf - theta_0) exp(-r t), with per-respondent
rate r. This is the standard learning-curve form and gives a single
interpretable rate parameter.

Identifiability. A rate is a derivative and amplifies noise. Two levers
keep it estimable, holding item parameters fixed during the per-respondent
read, and restricting the analysis to respondents with enough sequence
density. Both are reported, not assumed.

Controls for the machine front. Label-shuffled demonstrations,
content-free demonstrations, and held-out query items, following Min et
al. 2022. A rate that survives label shuffling is priming, not learning.

Metrics. Per-step ability recovery (correlation of theta_hat_t with the
truth), rate recovery (correlation and error of r_hat against r across
respondents), curve-shape error, and calibration of the predicted
responses. Never a single accuracy number.

## 5. Success criteria and failure modes

Success for RQ0. Recovered rates rank-correlate with true rates above a
stated floor, the per-step ability tracks the true curve, and the rate
becomes estimable as density rises, with a clean degradation as density
falls. A defensible negative is also a result, for example a density
threshold below which the rate is not identifiable.

Failure modes to watch. Rate confounded with item-difficulty drift, the
encoder collapsing to a static ability, over-smoothing that erases the
curve, and on the machine front a rise that is priming rather than
learning. Each has a control or an ablation attached.

## 6. Experiment ladder

- E0, synthetic known-rate recovery (immediate, no external data,
  implemented in `deep_irt/traj_synth/`). Generate N respondents with
  parametric learning curves of varied rate, generate responses through
  the GPCM and binary decoders, train the encoder-decoder under prediction
  loss, recover the per-step ability with `model.track()`, fit the curve,
  and score the recovered rate against the truth. Swept over response
  format and sequence density. Two reference points anchor the reading, a
  fitter sanity check (curve fit to the noiseless true trajectory, which
  must be near perfect) and an oracle ceiling (rate MLE with known item
  parameters and known curve family), which separates a model failure from
  a fundamental identifiability limit. The rate is affine-invariant in
  theta, so the encoder's arbitrary scale does not bias it, which is why
  the rate, not the absolute ability, is the estimand. This is the RQ0
  study and the precondition for the real fronts. DONE, see
  `deep_irt/traj_synth/RESULTS_E0.md`. Headline, rate is recoverable but a
  weak, density-limited signal, lifted by sequence density and ordinal
  format, with the encoder tracking a per-respondent ML reference, and
  recovery concentrated where the window spans the curve's elbow.
- E1, machine front (in progress, `deep_irt/traj_icl/`). A ladder of
  open-weight Qwen2.5-Instruct models (0.5B, 1.5B, 3B) answers a shared
  400-item ARC bank at shot counts k in {0,1,2,4,8,16} under true-label and
  shuffled-label demonstrations. Each (model, k, condition) is a static
  examinee; a joint 2PL places them all on one shared ARC scale; theta(k)
  per model is the in-context adaptation curve and its slope the rate; the
  true-minus-shuffled gap separates genuine learning from priming (Min et
  al. 2022). DONE, an honest null with a validity check, see
  `deep_irt/traj_icl/RESULTS_E1.md`. The shared 2PL orders the ladder
  correctly (0.5B about -1, 1.5B about 0, 3B about +1), but theta(k) is
  flat to declining (0.5B is hurt by demos, 1.5B and 3B flat) and the
  true-minus-shuffled gap is about zero for all. On a benchmark the models
  already know, demonstrations only prime, and the measure correctly
  registers the absence of adaptation rather than inventing a curve.
- E1b, synthetic in-context adaptation (DONE, the machine analog of E0,
  `deep_irt/traj_icl/synth_remap.py`, RESULTS_E1b.md). Four
  known categories are relabeled with ARBITRARY single letters by a fixed
  bijection revealed only through the demos, so the mapping must be inferred
  in context. Accuracy is forced to chance at k=0 (the model cannot know an
  arbitrary mapping) and rises with k under true labels, while shuffled
  (random per-demo labels) stays at chance, the clean positive-adaptation
  and priming controls ARC structurally could not give. Same 2PL and rate
  readout as E1. Uses abstract LETTERS, not nonce tokens, so the
  literature's small-model semantic-override failure does not apply (the
  model learns a novel association, it does not override known semantics).
  RESULT, theta(k) rises sharply under true and stays at chance under
  shuffled, the priming-corrected gain GROWS with model size (0.67, 2.65,
  5.42 for 0.5B, 1.5B, 3B). The shape is threshold-like (a phase transition
  at k about 10, robust across three arbitrary mappings with the gap
  positive and 3B over 1.5B in every seed), not the smooth exponential of a
  human curve, so the machine front's comparable estimand is the adaptation
  MAGNITUDE and threshold, not a smooth rate. Cross-respondent finding, the trajectory is
  one object but its parameterization is respondent-specific (smooth rate
  for humans, threshold gain for LLMs).
- E2, human front (design locked, `deep_irt/traj_synth/RESULTS_E0.md`
  density rules carry over). EdNet-KT1 dense learners (T_min 200 after
  bundle dedup; tens of thousands qualify). The hard problem is that real
  data has no ground-truth rate, so validation is by criteria, not by error.
  Two load-bearing tests, predictive validity (the rate fit on a learner's
  early window rank-predicts the late-window accuracy gain, with a
  shuffled-order negative control) and concurrent validity (the recovered
  rate agrees with the classical AFM/iAFM learning slope, the Koedinger
  2023 reference). Secondary, split-half reliability and aligned-vs-
  responsive convergent validity. Needs a one-time merge of the 784k
  per-user files into a flat CSV (streaming) and a GPU training pass.
- E3, transfer front (design locked, pre-registered). SciEx, expert
  partial-credit scores for about 8 LLM respondents over 154 graded CS
  exam items. Derive per-item machine difficulty two ways, GRADED (mean
  proportional credit) and BINARY (pass-fail at 0.5, the exact-match
  analog), and correlate each with the lecturer Easy/Medium/Hard
  difficulty. Pre-registered success, graded Spearman exceeds binary
  Spearman on the same items, ideally above 0.44 (the binary ceiling). An
  honest caveat, the respondent pool is small (about 8) and one domain, and
  a null is a reportable negative finding. The weakest front on novelty
  (binary transfer exists), so the contribution is narrowly the graded
  improvement. STATUS, pipeline built (`deep_irt/traj_transfer/run_e3.py`,
  graded and binary difficulty, bootstrap CIs, dispersion split, a GRM
  secondary), but the real run is BLOCKED, SciEx is a gated Hugging Face
  dataset and no HF_TOKEN is set. User action, agree to the dataset terms
  and `export HF_TOKEN=...`, then `python -m deep_irt.traj_transfer.run_e3`.
  A simulation against the paper's published statistics validates the
  pipeline and shows the mechanism is plausible (graded 0.48 vs binary 0.35,
  larger on high-dispersion items), but this is NOT real data and is not a
  result. The GRM is underpowered at about 8 respondents, so the graded
  mean is the primary difficulty proxy. Alternative if no token, generate
  graded responses with the local model ladder on a graded bank.

## 6b. Cross-front synthesis (what we now know)

The unified estimand was tested across respondents and across
data-generating processes. Six experiments, read together.

| exp | respondent, data | does the DGP contain a trajectory | result |
|---|---|---|---|
| E0 | synthetic, human-like | yes, by construction | rate recovered, weak and density-limited, encoder near the ML reference |
| E1 | LLM, ARC (known task) | no, the model already knows it | null, theta(k) flat, true equals shuffled |
| E1b | LLM, synthetic remap | yes, mapping must be inferred | strong positive, magnitude scales with model size, threshold shape |
| E2 | human, EdNet KT | no, single-pass item stream | AFM near zero on a wrong (single-pass) DGP; the predictive metric used here later shown ill-posed |
| E2b | human, ASSISTments (repeated practice) | yes, recurring skills | AFM small positive (0.14) but skill-id circular; predictive metric shown ill-posed, discarded |
| E2c | human, KDD Cup 2010 (problem-level, non-circular) | yes, repeated practice | existence gate significant but NEGLIGIBLE (mean dNLL 0.0008, CI spans 0, 53.7% win); non-circular AFM concurrent NULL (-0.005); no external validity |
| pos-ctrl | synthetic, known rate | yes | recovery corr(r_hat,r_true)=0.46, but the predictive metric is -0.26 and even the TRUE rate scores -0.38, so that metric is structurally ill-posed |
| E3 | transfer, SciEx graded | n/a | REAL run NULL, graded 0.07 vs binary 0.10 (graded does NOT beat binary), neither tracks the coarse human label; the sim positive was an artifact |

Three findings carry across the fronts.

**The measure recovers a trajectory cleanly on controllable data, and the
real human front is inconclusive, not negative.** Clean positives appear
where the generating process is controlled, synthetic human-like curves
(E0, recovery 0.46) and the synthetic machine remapping task (E1b). On real
data the machine front behaves correctly (ARC null E1, remap positive
E1b). The real HUMAN front was first read as a failure because the
predictive-validity test was null to negative on EdNet and ASSISTments, but
a positive control overturned that reading. On synthetic data with KNOWN
rates, recovery works (0.46) yet the same predictive metric returns -0.26
and the TRUE rate itself scores -0.38, because fast learners plateau within
the window and so show less late-window gain. The metric is structurally
ill-posed, any gain-over-a-fixed-window criterion is, so the E2 and E2b
predictive numbers measure that confound, not a recovery failure. What
survives as valid real-human evidence is the AFM concurrent test, near zero
on EdNet (a single-pass DGP where no learning curve exists) and small and
positive on ASSISTments (0.14) but confounded because the encoder's item
key is the same skill id the AFM slope is fit on. The decisive test then
ran on KDD Cup 2010 with problem-level items (non-circular) and the
validated pipeline (E2c), and it is null, the existence gate is significant
but negligible (mean delta_NLL 0.0008, CI spanning zero, 53.7 percent win)
and the non-circular AFM concurrent validity is flat (-0.005). So after
ruling out the metric artifact, using the validated existence-then-rate
pipeline, and using a non-circular repeated-practice dataset, the recovered
human learning rate still shows no external validity. The honest state is
synthetic-and-machine positive, and the real human learning-rate claim is
thoroughly tested and UNSUPPORTED on three real datasets.

**The controls are what make a result interpretable, and a wrong control
misleads.** The shuffled-label baseline on the machine front works, E1b's
gap is real because shuffled stays at chance. But the human front's
permuted-order control was misleading, it sat at the same near-zero value
as the real correlation not because there was no signal but because the
gain-over-window metric is ill-posed for both (the positive control proved
this). A control only validates a metric that is itself well-posed.

**Recoverability is gated by identifiability conditions, and we can state
them.** A rate needs enough density and a window that spans the curve's
elbow (E0), and it needs the data to actually contain a curve. But the
real human results add a caution, even with a curve present (E2b) the
recovered rate can fail the cleanest external test, so density and a
present curve are necessary, not sufficient, and a clean item granularity
(problem level, not skill level) is needed to test concurrent validity
without circularity.

**The trajectory shape is respondent-specific.** Human and human-like
learning approaches an asymptote smoothly, so a single rate captures it
(E0). LLM in-context adaptation is threshold-like, a phase transition once
enough demonstrations accumulate, so the comparable quantity is the
adaptation magnitude and its threshold, not a smooth rate (E1b). The
latent trajectory is one object, but its right parameterization differs by
respondent type.

**Honest scope.** The strongest positives are on controllable data (E0,
E1b). The real-data human front is unestablished, EdNet-KT1 is the wrong
generating process (a single-pass diagnostic stream, E2), and ASSISTments,
which does have repeated practice (E2b), gives only a small AFM concurrent
signal confounded by skill-id-as-item. The predictive-validity evidence is
set aside, a positive control proved that metric is structurally ill-posed
(even a perfect rate fails it), so the human front is unestablished rather
than negative. The transfer front
ran on real SciEx and is NULL, graded difficulty does not beat binary
(0.07 vs 0.10) and neither tracks the coarse 3-level human label, the
simulation's earlier positive was an artifact of a built-in correlation
and is discarded. So both real-data educational fronts came back
null-or-unestablished while the clean positives are synthetic and machine.
The demonstrated
contributions are the estimand and its identifiability conditions, the
machine in-context adaptation curve with a clean learning-versus-priming
separation and a size-scaled magnitude, and the cross-respondent shape
difference. A methodological contribution falls out and was then validated.
Gain over a fixed window is invalid for a learning rate when rates are
heterogeneous, fast learners plateau inside the window. The replacement
splits, held-out predictive improvement of a dynamic model over a
constant-ability null is a valid, ground-truth-free GATE for whether a
trajectory exists (with-rate versus no-rate at p about 5e-11), but it does
NOT rank learners by rate, so the magnitude is read from a parametric curve
fit on the model's estimated item parameters (recovery about 0.41 with
estimated items). Existence gate first, then the parametric rate. That
decisive test ran on problem-level KDD Cup 2010 (E2c, non-circular,
validated pipeline) and returned null, a negligible existence effect and a
flat AFM concurrent, so the real human learning-rate claim is now
thoroughly tested and unsupported on three real datasets, not merely open.
The strongest real next step for it would be a denser within-skill design
or a constructed speed-contrast cohort, but on the evidence the claim does
not hold as posed. Nothing is buried.

## 6c. Open follow-ups (gated)

- E2b is DONE (ASSISTments, `deep_irt/traj_kt/RESULTS_E2b.md`), weak and
  confounded. A positive control (`deep_irt/traj_synth/RESULTS_poscontrol.md`)
  showed the predictive-validity metric is ill-posed, so the decisive human
  test needs both PROBLEM-LEVEL items (KDD Cup 2010 Algebra via PSLC
  DataShop, gated) for a non-circular AFM check AND a well-posed rate
  criterion (early-slope or constructed speed sub-populations), not gain
  over a fixed window.
- E3, set HF_TOKEN after agreeing to the SciEx terms, then run the built
  pipeline on real data.
- A real-task LLM ICL front (low-resource translation) to show the
  adaptation curve outside a synthetic task.
- Read D-BIRD (2506.21723) and the ZPD paper (2502.06990) in full before
  any submission, per the novelty check.

## 7. Positioning and venues

The program can feed more than one venue, for example a measurement and
learning-analytics venue for the human front and a machine-learning or
evaluation venue for the machine front. The downstream targets noted
elsewhere are IJAIED, BEA, and EDM. Final placement is deferred until
results decide which front is strongest. Prior-work positioning is filled
in from the prior search.

## 8. Prior work

The literature splits into worlds that do not meet. Each touches a piece
of the estimand and none names it.

**Classical learning-rate models.** Bayesian Knowledge Tracing (Corbett
and Anderson 1995) carries a learn probability, the Additive Factors
Model (Cen, Koedinger, Junker 2006) and Performance Factors Analysis
(Pavlik, Cen, Koedinger 2009) carry a learning slope over practice count,
and Koedinger et al. 2023 (PNAS) make the human learning rate the target,
finding it strikingly uniform across learners while initial knowledge
varies. These treat the rate as a scalar regression coefficient, target
next-response prediction or population comparison, and use no neural model
and no trajectory function.

**Dynamic and neural ability.** Deep Knowledge Tracing (Piech et al.
2015), DKVMN (Zhang et al. 2017), Deep-IRT (Yeung 2019, our architectural
ancestor), and the DKT-as-dynamic-MIRT result (Vie and Kashima 2023) all
produce a time-varying ability sequence, but universally as a means to
next-step prediction, never as a recovered curve with a rate. The closest
trajectory work is psychometric, not neural. DynAEsti (Ghosh et al. 2019)
recovers a continuous ability curve under dynamic IRT by EM, dynamic
state-space IRT (Wang et al. 2013) carries a per-person growth trend, and
VTIRT (Piech et al. 2023) gives fast temporal theta estimates. None
defines the rate dtheta/de as an estimand, none trains under prediction
loss, and all index by calendar time rather than accumulated evidence.

**IRT for language models.** tinyBenchmarks (Polo et al. 2024) and Growing
Pains (Polo/Stanovsky et al. 2026, the nearest analog and a must-cite)
place models as respondents on a shared IRT scale, but ability is a static
vector at a fixed prompting configuration. Item Response Scaling Laws
(Polo et al. 2026) is the only work treating IRT ability as a trajectory,
but over training compute, not inference-time shot count. The one paper
joining IRT and in-context learning, the ZPD study (Guo et al. 2025),
classifies whether an item is learnable in context rather than tracing
theta(k) and its rate. The many-shot ICL curves (Agarwal et al. 2024)
show performance versus shot count but on raw, non-comparable accuracy
with no measurement model and no rate. Min et al. 2022 is the
load-bearing methodological constraint, demonstrations are partly a
priming signal, so a shot-count curve must be read against a
priming-controlled baseline.

**LLM-to-human difficulty transfer (front 3).** Item difficulty derived
from LLM respondents predicts human difficulty (Liu et al. 2025; Cardoso
et al. 2025), but all of this work is binary-scored, and Merhav et al.
2025 warns that models may converge to a machine consensus rather than
human difficulty. Graded and polytomous transfer is untested, which is
our opening.

**The gap.** No prior work names the trajectory theta(e) over accumulated
evidence and its rate dtheta/de as the primary estimands, asks whether
prediction-loss training recovers them and under what identifiability
conditions, and extends the question jointly to human learners and
language models on one shared scale. That is the program.

**To verify before submission.** Two recent items flagged as possible
near-duplicates need a full read, D-BIRD (arXiv 2506.21723), which models
an individual growth slope in Bayesian IRT, and the ZPD study (arXiv
2502.06990). The IRT-for-LLMs area is moving fast, so a late-breaking
overlap is possible.

**Must-cite anchors.** Corbett and Anderson 1995; Cen et al. 2006; Pavlik
et al. 2009; Koedinger et al. 2023 (PNAS); Piech et al. 2015; Zhang et al.
2017; Yeung 2019; Vie and Kashima 2023; Ghosh et al. 2019 (DynAEsti);
Piech et al. 2023 (VTIRT); Min et al. 2022; Agarwal et al. 2024; Polo et
al. 2024 (tinyBenchmarks); Polo/Stanovsky et al. 2026 (Growing Pains);
Polo et al. 2026 (Item Response Scaling Laws); Liu et al. 2025; Cardoso
et al. 2025; Merhav et al. 2025; Guo et al. 2025 (ZPD).

## 8b. Novelty verdict and positioning (adversarial check)

An adversarial verification (tasked to refute, not to flatter) read the
closest work and graded the threat to each front.

- Front 0 (rate recovery) is essentially uncontested. DynAEsti (Ghosh et
  al. 2019) recovers a continuous ability curve but by EM, on calendar
  time, with no rate estimand. A new June 2026 Bayesian dynamic IRT paper
  (arXiv 2606.15525) tracks human trajectories across course chapters and
  finds ability mostly stable, which supports rather than undercuts the
  motivation. Low risk.
- Front 1 (LLM in-context curve) carries the one real overlap, Item
  Response Scaling Laws (arXiv 2606.07616, same group as Growing Pains),
  which treats IRT ability as a trajectory over PRE-TRAINING COMPUTE and
  fits a slope, a learning rate in log-FLOP space. The defense is that
  inference-time shot count is a categorically different axis from training
  compute, the rate is per-respondent not a population summary, and the
  priming control and shared cross-size scale are absent there. Must cite
  it prominently and make the inference-versus-training distinction crisp.
  Medium risk, managed.
- Front 3 (graded transfer) is the weakest. Binary LLM-to-human difficulty
  transfer is established (Liu 2025, Cardoso 2025, BRIDGE 2026), so the
  contribution is narrowly that GRADED scoring beats binary, an empirical
  bet, pre-registered, with a null reportable as a negative finding.

Positioning moves, separate the axes (time-indexed vs evidence-indexed,
calendar vs shot-count vs compute) in one stroke; name prediction-loss
training as the mechanism that distinguishes us from all the EM and MCMC
dynamic-IRT work; lead the abstract with the conjunction, no prior work
names theta(e) and dtheta/de as primary estimands, audits their
identifiability under prediction-loss training, and extends the question
jointly to human learners and language models on one shared scale.

## 9. Engineering and conventions

New code lives under `deep_irt/` in a dedicated module, additive and
config-driven, following the existing pattern of `jointfmt/`, `dynjoint/`,
and `slam_extend/`. No edits to the frozen Chapter 0 kernel. Deterministic
seeds, tests before merge, explicit file staging, results written to a
results file alongside the code. New datasets enter through the adapter
interface in `rl/src/ordrec/data/`.
