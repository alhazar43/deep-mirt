# Paper 1 continuity brief (for Paper 2)

Source of record. `overleaf-sync/main_caeai.tex` (draft, title TBD in tex; frozen
title per plan v2.1 is "Detecting Item-Parameter Error in Knowledge-Tracing Models
Without Ground Truth") and `docs/paper_plan_v2.md` (plan of record). Venue CAEAI
first, TLT fallback. Paper 2 must neither repeat nor contradict what follows.

## 1. What Paper 1 claims, and what it does not

Claims (all single-skill, static-ability, item-parameter territory):

- Prediction-only training of IRT-parameterized KT models leaves prediction accuracy
  and item-parameter fidelity SEPARABLE. Shared-head (SH) models predict as well as
  separated-key (SK) models while recovering item parameters far worse (2PL
  discrimination .553 vs .898 LSTM, .373 vs .806 transformer, reference cohort).
- The failure is truth-free-invisible ("stable and wrong"): accuracy TOST, rerun
  stability, split-half reliability all PASS on the shared arm while 61% of its
  top-flagged items are wrong against truth and accuracy-tied models disagree on
  two thirds of flags.
- The gap is a per-item AMORTIZATION error in the shared readout channel, not a data
  or capacity limit. Estimator ladder: shared 0.719 -> refit on own theta_hat 0.934
  ~= decoupled (SK) 0.941 -> clamp 0.979 ~= mirt MML 0.982. The information is in the
  model; the shared head does not read it out. Not cured by more data (dkvmn pooled
  difficulty N-flat under sharing) or wider shared embeddings.
- Vulnerable parameter groups follow response geometry. Ordered heads lose
  discrimination-like directions (Fisher information suppressed near the operating
  point); the NRM additionally has a distractor-plane pathway into ability, removed
  by the gradient-routed head (stop-gradient on the orthogonal slope component).
- The unrepaired channel has a decision cost in CAT simulation: ~197% test length,
  worse theta at its own stop, +2.3pp cut-score misclassification; a-error inflates
  length, b-error makes stops falsely confident. Repairs are priced: slack test
  (truth-free detector, slack = 1 - Spearman(readout, per-item refit on own
  theta_hat); r=.986 with wrongness, AUC .987), per-item refit (rank repair
  0.72 -> 0.93, interpretation only, gauge-bound, never decision-grade), SK rebuild
  (halves the CAT invoice, residual cost remains).
- Real data are prediction and external-agreement checks only. Structured heads
  impose no systematic prediction cost; the NRM head improves selected-option
  prediction on EdNet over direct predictors; SH edges SK on EdNet-NRM likelihood
  while SK wins the person-side anchor and the distractor-statistic agreement.

Does NOT claim:

- Anything about LEARNING or GROWTH. Ability is generated static (theta ~ N(0,1));
  trajectories shown (TIMSS, EdNet) are explicitly "model-based traces, not causal
  estimates of learning". Limitations state plainly "not learning dynamics".
- Anything about MULTIPLE SKILLS or cross-skill structure. Every model is
  single-theta, single-construct; no Q-matrix, no KC layer, no transfer.
- No global SH-vs-SK ranking. SK is a repair for item-parameter readout, not a
  better model everywhere (SH wins EdNet-NRM held-out likelihood; near-parity rules
  bind). Paper 2 must not cite Paper 1 as "SK is superior".
- No claim that neural heads replace classical calibration. MML near-parity is
  stated up front; when the matrix exists, refit classically.
- No real-data parameter-recovery claim. Recovery is synthetic-only; real data
  cannot validate parameters, only prediction and external agreement.
- No claim about interpretability in general, only the specific IRT-parameterized
  head family under next-response training.

Where Paper 2 slots in. Paper 1 audits STATIC, SINGLE-SKILL item-side readouts.
Paper 2's territory (growth detection, signed cross-skill association) is exactly
what Paper 1 disclaims. The shared inheritance is the validity-gated posture:
prediction accuracy never certifies a readout; every interpretable quantity needs
its own recovery or truth-free check before it is read. Paper 2 extends the gate to
person-side dynamics and multi-KC structure; it must not re-litigate the item-side
audit and must not assume SK fixes person-side or dynamic readouts (Paper 1 shows
person-side summaries remain path-sensitive even when item maps agree).

## 2. Terminology to reuse verbatim

- Model names: knowledge tracing (KT), Deep-IRT, DKT, DKVMN, SAKT, AKT; encoders
  "LSTM, Transformer (transformer), DKVMN"; decoders/response heads "2PL, GPCM,
  NRM"; "routed NRM" / "gradient-routed nominal head".
- The two designs: "shared-head (SH)" and "separated-key (SK)"; "item-to-parameter
  path"; "the shared readout channel"; "dedicated item key". Never invent synonyms.
- Parameters: ability theta_i (learned estimate \hat{theta}_{i,t}^-, read at the
  last valid step), discrimination alpha_q, difficulty beta_q, "step thresholds"
  (GPCM; never "category boundaries"), option slopes a_k and option intercepts c_k,
  item-side parameters psi_q, item embedding e_q, item key k_q.
- Estimators: "per-item refit on frozen learner states" / "refit on own theta_hat",
  "marginal maximum-likelihood (MML) reference" (mirt), "the estimator ladder",
  "clamp" (refit at true theta).
- Audit vocabulary: "parameter recovery" (Spearman rank recovery as the primary item
  metric), "amortization gap" / "item-wise amortization error", "stable and wrong",
  "truth-free", "the slack test", "the ritual table", "the invoice" / decision cost,
  "the rebuild" (decoupling), "validity" framed as recovery-based checking,
  "dataset-clustered bootstrap intervals" / seed-clustered, "reference cohort"
  (N=2000, Q=200, E=600), "spiraled administration", exposure E = 60N/Q.
- Datasets: EdNet, KDD Cup 2010 Algebra ("KDD"), TIMSS 2019 grade-8
  constructed-response ("TIMSS"). Synthetic bed named as "the synthetic benchmark".
- Loss: "next-response prediction" / "next-response cross-entropy"; the objective is
  always prediction-only, no parameter supervision.

## 3. Section architecture of the tex (the structural model)

1. Introduction. Dilemma (predictive vs interpretable), the validity question,
   exactly three contributions, roadmap paragraph.
2. Related Work. Three subsections: KT as sequence prediction (incl. the pyKT/
   simpleKT audit precedent), parameterized response heads in KT, parameter
   recovery / amortization / interpretability checks (probing, control tasks).
3. Methodological Framework ("Instruments"). Problem formulation (encoder +
   response head), the interpretable head family (2PL/GPCM/NRM, SH vs SK, one
   architecture figure), gradient paths and the recovery gap (the mechanism math,
   ending in three empirical implications).
4. Experiments. Experimental design (four comparisons, beds table), synthetic
   recovery (mass table + delta-delta figure + scatter), real-data prediction
   beyond accuracy, TIMSS ordinal case study, EdNet two-resolution case study.
   (The CAT/downstream section is referenced as sec:downstream in the Discussion
   but not yet present in the draft; plan S6 carries it.)
5. Discussion. Offline calibration vs sequential use (decision table by setting),
   corrections-the-protocol-caught (four honest error case studies), limitations
   and generalizability.
6. Conclusion. Restated decision guide, evaluation matched to intended use.
   Appendices: full benchmark grid, TIMSS item table, EdNet detail, hyperparameters.

The reusable skeleton for Paper 2: motivation -> related work in three strands ->
formal framework with the mechanism argument BEFORE experiments -> design section
that names its comparisons explicitly -> synthetic truth-based core -> real-data
scope-limited checks -> discussion that includes an honest corrections section ->
limitations that list retractions plainly.

## 4. Paper 1 results Paper 2 can cite as motivation

- The stable-and-wrong disease. Truth-free rituals (accuracy, rerun stability,
  split-half reliability) all pass while item parameters are badly wrong; stability
  is why the failure is undetectable. Motivates Paper 2's validity gate: a growth
  or cross-skill readout that merely predicts well and replicates is not yet
  evidence of growth or influence.
- Separability of prediction and parameter fidelity (18/18 cells improve recovery
  under SK with accuracy ties). License for Paper 2 to demand recovery-style
  evidence for its detectors rather than accuracy comparisons.
- The amortization localization (ladder: information present, readout at fault; not
  cured by data or capacity). Motivates designing Paper 2's readouts with dedicated
  channels and testing them against refit-style references.
- The repair story with priced boundaries. Slack test as the truth-free detector
  template (a self-consistency check against the model's own refit); refit as
  rank-repair only; rebuild as the training-time fix that halves but does not erase
  the decision cost. Paper 2 can present its own gates as the same genus of
  instrument, citing Paper 1 as the item-side precedent.
- The decision-cost receipts (CAT ~197% length, +2.3pp misclassification, two-
  channel attribution), always with "in simulation" attached. Motivates that
  readout error is not academic; downstream decisions pay for it.
- The person-side sensitivity finding (EdNet 2PL: near-identical item maps, ability
  vs raw score r=.36 SH vs .54 SK). Direct motivation for Paper 2: person-side
  summaries are the fragile quantity, and Paper 2's growth detector lives there.
- The honesty conventions worth inheriting: pre-registration with reported misses,
  clustered intervals with the generated dataset as the unit, thin-exposure regions
  declared unresolved, retractions listed once and plainly.

Binding rules that also bind Paper 2 when citing Paper 1: "wrong" is always scoped
to the shared arm; near-parity with MML is never a win; the refit is never
decision-grade; the slack test always carries the theta-quality caveat; every
cut-score claim carries "in simulation".
