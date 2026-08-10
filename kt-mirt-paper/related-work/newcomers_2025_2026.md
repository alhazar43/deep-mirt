# Newcomers 2025-2026: literature adjacent to validity-gated KT readouts

Survey date 2026-08-10. Scope: work published 2025-2026 that certifies
latent-state readouts against ground truth, measures detection floors, or
audits KT ability estimates. All entries verified to resolve on the date
above. Verdict up front: no one found who certifies KT readouts with
synthetic twins plus confound arms; the audit-flavored newcomers stop at
prediction reliability, deployment behavior, or teacher-facing usefulness.

## Audit and reliability flavored (closest in spirit)

- **Ensuring Reliability in Programming Knowledge Tracing: A Re-evaluation
  of Attention-augmented Models and Experimental Protocols** (Kim & Kim,
  ITS 2026, arXiv:2605.04727). Audits experimental protocols, shows
  attention-model gains shrink under controlled conditions. Audits
  prediction pipelines, not latent readouts; no ground truth for states.
- **Practical Evaluation of Deep Knowledge Tracing Models for use in
  Learning Platforms** (Yamkovenko et al., EDM 2025 industry track).
  500k-student deployment audit of LSTM/SAKT beyond AUC: poor specificity
  on wrong answers, cold start, noise sensitivity, order dependence (0.28
  probability swings from reordering alone). Behavioral audit of
  predictions; never checks the latent state against any truth.
- **Investigating the Robustness of Knowledge Tracing Models in the
  Presence of Student Concept Drift** (Lee et al., Heffernan lab, 2025,
  arXiv:2511.00704). Multi-year real data; all four KT families degrade
  under drift, BKT most stable. Predictive robustness, no state audit.
- **Does Interpretability of Knowledge Tracing Models Support Teacher
  Decision Making?** (Khalid, Deriyeva & Paassen, AIED 2025 workshop,
  arXiv:2511.02718). Interpretable KT barely changes teacher task
  selection despite higher trust ratings. Validity of use, not validity
  of the estimate itself.

## Post-hoc correction of readouts

- **Recovering Stranded Discrimination in Knowledge Tracing: Per-Item
  Bias Correction via Empirical-Bayes Shrinkage** (Yan, Tang & Shimada,
  2026, arXiv:2606.14123). Corrects per-item logit bias in frozen KT
  models via Kalman-smoothed empirical-Bayes shrinkage; notes global
  calibrators leave item discrimination stranded. Repairs prediction
  logits on real data only; no synthetic ground truth, no readout
  certification. Closest neighbor to our repair ladder; cite and
  differentiate (they fix probabilities, we certify and gate readouts).

## Synthetic students and simulation

- **SMART: Simulated Students Aligned with Item Response Theory**
  (Scarlatos et al., EMNLP 2025, arXiv:2507.05129). DPO-aligns LLM
  simulated students to a ground-truth IRT model to predict item
  difficulty. Uses IRT as alignment target for content calibration, not
  to certify a tracer's latent states.
- **Language Bottleneck Models for Qualitative Knowledge State Modeling**
  (Berthon & van der Schaar, 2025 rev. 2026, arXiv:2506.16982). LLM
  textual bottleneck as interpretable knowledge state, validated on
  synthetic and real datasets. Synthetic use is proof-of-concept for
  summaries, not a twin-based certification with confound arms.
- **Simulating Students with Large Language Models: A Review** (2025,
  arXiv:2511.06078) and the ACL 2026 line on simulated-student realism
  (arXiv:2601.04025) evaluate simulator faithfulness to personas, the
  inverse of our problem (we test the tracer, they test the simulator).

## Modeling improvements that gesture at validity

- **KeenKT: Knowledge Mastery-State Disambiguation** (Li et al., 2025,
  arXiv:2512.18709). Distributional (NIG) mastery states to separate true
  ability from carelessness/guessing. Motivated by readout ambiguity but
  validated only by prediction benchmarks.
- **Uncertainty-aware Knowledge Tracing (UKT)** (Cheng et al., AAAI 2025,
  arXiv:2501.05415). Stochastic state embeddings with Wasserstein
  attention; uncertainty is modeled, never calibrated against truth.
- **PAKT: Disentangling Knowledge States with Ability and Proficiency
  Modeling** (2026, arXiv:2607.13103). Phase-aware decomposition into
  ability and proficiency; a richer readout, not an audited one.
- **LTKT: Knowledge Tracing Based on Positive and Negative Learning
  Transfers** (Xu et al., Tsinghua Science and Technology 2025). First KT
  model using signed (positive and negative) concept-transfer relations
  in a transfer graph. Overlaps our signed cross-KC influence goal at the
  modeling level; no validity gate, no ground-truth certification of the
  recovered signs. Must-cite for the kt-mirt influence claim.
- **KTCF: Actionable Recourse in Knowledge Tracing via Counterfactual
  Explanations** (Kim, Lee & Kim, AAAI-26 AISI oral, arXiv:2601.09156).
  Counterfactual recourse from KT models; explanation generation, not
  estimate validation.

## Adjacent methodology outside KT

- **When Are Neural Interaction Discoveries Real? Identifiability,
  Recoverability, and a Pre-Fit Diagnostic** (Kuskova, Zaytsev &
  Coppedge, 2026, arXiv:2606.08390). Pre-fit effective-rank diagnostic
  for whether neural interaction discoveries are recoverable at all.
  Same epistemic move as our detection-floor question, different domain;
  useful precedent that "is the readout recoverable" is a publishable
  question.
- **Growing Pains: Extensible and Efficient LLM Benchmarking Via Fixed
  Parameter Calibration** (Habba et al., 2026, arXiv:2604.12843).
  Fixed-anchor MIRT calibration across evolving benchmarks; the known
  nearest analog for our anchor-guardrail machinery, on LLM evaluation
  rather than student KT.
- **Exploratory DeepCDMs** (Psychometrika, Cambridge Core 2025-2026) and
  the Q-matrix identifiability line (Kim, BJMSP 2025) bring
  identifiability guarantees to deep cognitive diagnosis, but for static
  DCMs with theory-first proofs, not simulation-based certification of
  sequential KT readouts.

## Gap assessment

Nothing found in 2025-2026 combines (a) synthetic twins with known latent
trajectories, (b) confound/corruption arms, and (c) a pass/fail gate on
which readouts may be reported. The audit energy in the field is spent on
prediction protocols (2605.04727), deployment behavior (EDM 2025
industry), and teacher-facing usefulness (2511.02718). Two entries need
active differentiation in the paper: SLC (2606.14123) as post-hoc readout
repair without certification, and LTKT as signed transfer modeling
without a validity gate.
