# Paper plan: "Not All Parameters Learn Alike"

Provisional title, from the workshop deck. Working venue: JEDM full article (the EDM
Journal Track carries an accepted JEDM paper to the conference stage). Backup: EDM 2027
full (10 pages, EDM's own template). Next JEDM/EDM cycle opens around fall 2026, so there
is runway for the added experiments.

**Home and flavor.** This is a KT paper: a KT encoder plus an IRT decoder (some read the
pair as neural-IRT, the name does not matter). IRT is the readable flavor, not the subject.
The goal is not architecture novelty. The goal is to show the prior KT-plus-IRT literature
is more flexible, and more limited, than it is believed to be. Keep the measurement and
deployment register that made MA-GPCM IRT-centric out of the spine.

---

## 1. Thesis (one sentence)
A KT encoder with an interpretable IRT decoder, trained only on response prediction,
recovers its readout parameters unevenly, and the unevenness is a finite-data, finite-budget
property that Fisher geometry explains and that one robust lever, decoupling, addresses.

## 2. The message (five points, this is the contribution)
1. **The encoder and decoder are swappable and modular**, not the rigid pairing prior work
   assumed. We demonstrate it (encoder x decoder matrix). This is the instrument, not the claim.
2. **Plain prediction loss suffices.** Prior neural-IRT recovers parameters through
   variational machinery (VIBO, VTIRT, ELBO). We show standard prediction-loss KT training
   recovers them too, no variational posterior. This both simplifies and sets up the catch.
3. **But some parameters recover badly, and it is a finite-data problem.** Math says why
   (low Fisher curvature makes discrimination the slow direction; coupling the ability
   readout to the item breaks the location gauge), experiments say how (a stable recovery
   ordering). Said loudly: most of this penalty vanishes asymptotically (the rate gap closes
   with infinite data and training); real datasets are finite, so it bites in practice. The
   contribution lives in the realistic finite regime.
4. **Two levers, and which wins is genuinely mixed.** On synthetic recovery a state-conditioned
   (dynamic) head can rescue the starved channel; on real-data reliability the winner is
   decoupling with a static head. We report the mix straight.
5. **NRM explains the mix.** The nominal decoder, whose slope loads on ability but is not
   low-information, separates the two levers: decoupling is a representation effect (robust,
   fires regardless of Fisher), the dynamic head is a Fisher effect (conditional, only helps a
   low-information parameter, can hurt otherwise). So decoupling+static winning on finite real
   data is a prediction of the theory, not an embarrassment.

## 3. The single storyline (organic, KT home)
KT nets bolt on IRT readouts for interpretability and declare it done -> we ask whether the
readouts are actually recovered -> to ask cleanly we need a framework whose encoder and decoder
swap (DKVMN, LSTM, transformer x GPCM, NRM, binary; MA-GPCM is the precursor) -> on synthetic
ground truth it recovers, but unevenly and stably across swaps, with plain prediction loss and no
variational machinery -> a local gradient-flow analysis of the readout explains the unevenness,
and it is a finite-budget effect that vanishes asymptotically -> embedding and readout
configuration is the lever, decoupling (robust, representation) versus a dynamic head (conditional,
Fisher), with the NRM decoder as the control that separates them -> it holds on real data, where
the honest verdict is decoupling+static -> it changes a downstream decision (adaptive item
selection). One thread: change the configuration, watch recovery move.

## 4. The finite-data spine
The headline is not an abstract two-axis theorem; it is "what a prediction-trained tracer can and
cannot recover under a real budget, and the one robust lever." Fisher geometry is the *why*. Two
components, stated with their asymptotic behavior:
- **Rate (finite-budget, vanishes asymptotically).** Near the optimum each readout direction
  relaxes at a rate set by its Fisher curvature. Discrimination is slow because
  `I(alpha) = E[w (theta - beta)^2]` is suppressed where responses concentrate (`theta` near
  `beta`). The gap closes with enough data and training, so the penalty is real only in the
  finite regime. This needs its own evidence (the budget sweep, see experiments).
- **Identifiability / coupling (structural).** Whether the ability readout is coupled to the item
  representation decides which parameters are identifiable at all. A coupled ability readout lets
  ability absorb the current item's location, breaking the ability-difficulty split; decoupling
  restores it. This does not vanish with more data; it is a gauge choice.

## 5. The two levers and the NRM adjudication

> Phase 1 refinement (see `docs/theory_memo.md`): keep TWO decouplings distinct. Ability-item coupling ->
> identifiability (structural, persists at all budgets); width sharing -> representation/rate (finite,
> vanishes asymptotically). NRM adjudicates only the representation one; E2 is the vehicle for
> identifiability. Do not let one borrow the other's credit.
- **Decoupling = representation lever (robust).** A narrow ability embedding plus a separate wide
  item key. Escapes the shared-capacity trade-off (rate) and, applied to the ability readout,
  restores the gauge (identifiability). Fires whenever a readout shares a representation, regardless
  of Fisher. This subsumes MA-GPCM's separated pathway.
- **Dynamic head = Fisher lever (conditional).** Feeds the learner state into the discrimination
  readout. Helps a low-information parameter, can hurt one that is not.
- **NRM separates them.** Its slope `a_k` loads on ability but has near-symmetric information
  (`I(a_k)/I(c_k) ~ 0.90`). On NRM the decoupling escape replicates (so it is representation), while
  the dynamic rescue reverses and hurts `a_k` (so it is Fisher). Present NRM strictly as the control
  that separates the levers, never as a standalone third result.
- **Honest real-data verdict.** On EdNet the decoupling escape is coverage-contingent (it reverses
  at ~12 observations per item, revives at knowledge-component density), and the dynamic head does
  not rescue. Between decoupling+static and decoupling+dynamic, static is the reliability winner. We
  say so.

## 6. MA-GPCM's place (demoted honestly)
Its novelty is oversold and we drop that framing. The separated pathway was an early, ad-hoc
instance of decoupling; the systematic decoupling study is what answers "why separate", separating
the ability read from the item representation is decoupling applied to one readout, and it works for
the representation reason NRM isolates. MA-GPCM is the precursor that motivated the question, not a
co-contribution.

## 7. Theory (formalized at the decoder/readout level, encoder-flexible)
- Core decoder math is the rigorous backbone: GPCM and NRM likelihoods, the residual `r = p - y`,
  per-parameter gradients (`d_theta L = r alpha`, `d_beta L = -r alpha`, `d_alpha L = r(theta-beta)`;
  `d_{a_k} L = r_k theta`, `d_{c_k} L = r_k`), and per-parameter Fisher information.
- The rate statement (curvature sets recovery rate under gradient flow) is derived for a generic
  readout parameter, plus the asymptotic-vanishing characterization.
- The identifiability statement is formalized at the readout level: a coupled readout
  `theta_t = phi(h_t, e_i)` versus a decoupled `theta_t = phi(h_t)`. Encoder-agnostic. The DKVMN
  separated pathway and the LSTM narrow-value-plus-wide-key are two *instances*, exhibited not
  claimed as the only realization. We never assert a specific architecture's trick is encoder-generic;
  the math is the general statement.

## 8. Methodology approach
Formalize the problem and the method with math, heavy on motivation, but keep the technical/encoder
side flexible. The encoder is "any sequence model emitting a state `h_t`". The levers are defined at
the readout/embedding level. Architectures are swappable realizations.

## 9. Structure
1. Introduction, KT premise flip, the five-point message, contributions.
2. The swappable framework (instrument), abstract encoder slot plus IRT-readout slot; MA-GPCM precursor.
3. Related work, lead with KT and interpretable KT (home), then IRT-as-tool, then training-dynamics /
   Fisher (method); the anti-variational positioning; pre-empt VTIRT.
4. Method and theory, core decoder math, the rate result with asymptotics, the coupling/identifiability
   result, the levers, all encoder-flexible.
5. Swappability on synthetic, recovery across encoder x decoder swaps, surfaces the stable ordering.
6. The finite-data problem, 6.1 recovery follows Fisher (rate) and the budget sweep (asymptotic
   vanishing); 6.2 configuration is the lever (shared/decoupled and the readout coupling, = why separate);
   6.3 decoupling+static vs decoupling+dynamic (the mixed comparison); 6.4 the NRM control separates the levers.
7. Real data, ASSISTments, EdNet, KDD, recovery and reliability, the honest verdict.
8. Downstream, adaptive item selection.
9. Discussion, boundaries, threats.
10. Conclusion; appendices (GPCM and NRM gradients and Fisher, gauge handling, reproducibility).

## 10. Experiments (REDO FROM SCRATCH at journal rigor)

> Directive (2026-07-01): the workshop-grade results are NOT conference/journal-grade and are NOT reused
> as evidence. Rebuild the ENTIRE program from scratch under ONE clean protocol, with proper seeds and CIs
> and the full control suite in docs/theory_memo.md (oracle-clamp, eigenmode/a_star, an I(theta)-at-fixed-K
> knob, the NRM confound controls, and the two decouplings kept distinct). The rigorous protocol is a
> Phase 2 design deliverable.

Rebuilt from scratch (not consolidated), under one protocol: three-speeds recovery; shared-vs-decoupled trade-off and
decoupling escape; dynamic-head rescue (synthetic + EdNet/KDD); the NRM dissociation (8 seeds + EdNet);
MA-GPCM DKVMN+GPCM recovery and ASSISTments; EdNet/KDD reliability.
Need:
- **E1, swappability matrix.** {DKVMN, LSTM, transformer} x {GPCM, NRM, binary} recovery on synthetic.
  Shows the framework swaps and the ordering is stable across it. Kills the "DKVMN artifact" objection.
- **E2, the coupling lever across encoders (= why separate).** Implement coupled-vs-decoupled ability
  readout in at least DKVMN and LSTM; show decoupling restores threshold identifiability in both, and
  settle whether it lifts discrimination (rate) or only thresholds (identifiability).
- **E-budget, finite-vs-asymptotic sweep.** Recovery gap vs data size and training budget; show it closes
  in the limit but is large at realistic N. Earns the finite-data framing; pre-empts "just train longer."
- **E-levers, the mixed comparison.** decoupling+static vs decoupling+dynamic on synthetic recovery and
  real reliability, with NRM as the adjudicating control.
- **E8, adaptive item selection.** Select by maximum Fisher information at the running ability estimate,
  using each configuration's recovered parameters against the per-item oracle, scored by ability error
  at fixed test length.
- **Consolidation pass**, re-run the key panels in one training protocol for internal consistency.
- **Optional E9**, classical MML-EM cross-calibration on real data, held as an appendix strengthener.

## 11. Related work and positioning
Three rings, lead with the nearest scientific neighbors even though they are not the target venue:
- Interpretable KT / neural-IRT (Converse AIED 2021; Yeung Deep-IRT; VTIRT EDM 2023; Wide & Deep IRT
  JEDM 2024). They assert or face-validate interpretability; we audit recovery.
- Discrimination-recovery failures (Converse's shuffled-response section; beta4-IRT 2023). Same symptom
  (discrimination worst), diagnosed as data-ordering nuisance or sign symmetry; we supply the Fisher
  mechanism. beta4-IRT Figure 2's unexplained magnitude shrinkage is our low-information signature.
- Option / choice modeling (Ghosh Option Tracing AIED 2021). The NRM-style readout exists as a
  diagnostic tool; we repurpose it as a control.
Anti-variational positioning (point 2): VIBO / VTIRT recover via variational inference; VTIRT even
reports discrimination recovered *best*, but that is well-specified generative inference, not
prediction loss. Pre-empt this explicitly.
So-what refs for the downstream: BanditCAT / AutoIRT (arXiv 2410.21033), GMOCAT (arXiv 2310.07477).
Corrections: do NOT cite "Growing Pains" (arXiv 2604.12843) as KT prior art, it is LLM-benchmarking
MIRT. Read Converse (AIED 2021, LNCS 12749) full text and cite-and-distinguish Ma et al. 2024
(arXiv 2310.12010, same deficit under a variational objective, blamed on the variational gap).

## 12. Reuse and trim from MA-GPCM (overleaf-sync/main.tex)
Reuse: the framework and architecture diagram (generalized to both slots); the separated-pathway
formulation (recast as one decoupling instance); the synthetic protocols (static, discrete, continuous),
metrics, baselines; the discrimination-worst ordering (now the headline the theory explains); ASSISTments.
Trim hard: the K=3..6 sweep to an appendix note; the weighted-ordinal loss to a one-line training detail;
cut class-imbalance robustness, binary compatibility, and the formative-assessment / deployment discussion.

## 13. Venue, destination, format
JEDM full (two-column, no hard page cap, APA, ~3 month review, open access, EDM Journal Track). The paper
evolves overleaf-sync/main.tex into the fused draft. EDM 2027 full (10pp) is a trimmed backup only.

## 14. Honest caveats we state, not hide
The rate penalty is finite-budget and vanishes asymptotically; the coupling penalty is structural.
"Swappable" is shown by E1, not sold; no single architecture's trick is claimed encoder-generic.
The Fisher theory is classical in parts (natural gradient, spectral learning dynamics), sold as
translation to KT, not as new psychometrics. Real-data evidence is reliability plus, with E9, calibration.
Decoupling's real-data benefit is itself coverage-contingent (EdNet), state that.

## 15. Decisions
Locked: provisional title = "Not All Parameters Learn Alike"; venue = JEDM full; home = KT, IRT flavor;
destination = overleaf-sync; MA-GPCM is a free draft, foldable, demoted to precursor.
Defaults set by me (user may override): encoders for E1/E2 = DKVMN + LSTM (transformer optional stretch);
the finite-vs-asymptotic budget sweep is IN scope (load-bearing for point 3); E9 cross-calibration held as
an appendix strengthener, not a launch blocker.
