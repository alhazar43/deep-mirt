# Research Gap -- representational capacity coupling in neural IRT parameter recovery

Opened 2026-06-16. Status: candidate contribution, NOT yet earned. The empirical
result is in hand; the math-level "why" is the open core and decides everything.

**UPDATE 2026-06-16 (Phases 0-3b run; the strong claim is REFUTED).** The gate
confirmed a real theta-vs-alpha trade-off (capacity killed), and the empirical
decoupling fix is robust and architecture-independent. BUT the toy ladder refuted
the population-limit learning-DYNAMICS-LAW framing below: in the tractable toys
the alpha bias is FINITE-DATA errors-in-variables (vanishes as reps -> inf) and is
NOT decoupling-fixable; "low Fisher sets the RATE, not the ENDPOINT." The
empirical decoupling-fix needs a learned encoder the toys omit, and a clean
dynamics law does not exist in tractable form. Honest framing: a finite-data +
learned-representation recovery effect, not a fundamental dynamics law. The claim
sentence below is the ORIGINAL hypothesis, retained for the record but downgraded;
see docs/LEARNING_DYNAMICS_STUDY.md Phase 3b RESULT for the current status. The
decoupled deep-irt ENGINE decision is unaffected (it stands on its empirical
merits).

---

## The claim in one sentence

Amortized neural IRT couples its psychometric parameters through the shared
learned representation, and this coupling silently degrades discrimination
(alpha) recovery. The degradation is a learning-dynamics phenomenon, not a
statistical one, and it is a measurement-validity condition the field has not
examined.

---

## The phenomenon (precise)

Neural / amortized IRT reads ability (theta), discrimination (alpha), and
difficulty (beta) off a SHARED learned representation (a shared item embedding
and a shared encoder state). Theta and alpha then have OPPOSING capacity needs.

- Theta wants the item channel NARROW. It is a low-rank latent. Widen the item
  representation and the encoder routes item identity into the ability state and
  overfits theta, and it overfits MORE with more training.
- Alpha wants the item channel WIDE. It is the lowest-information, hardest-to-
  recover IRT parameter. A narrow shared key STARVES alpha recovery, which comes
  out biased low AND high-variance across seeds.

One shared representation cannot be both narrow and wide, so sharing forces a
compromise that degrades alpha. Beta is content narrow (it recovers well from a
thin item code), so it sits with theta.

## The fix

Decouple. Give alpha its own separate, wider item-embedding table that feeds
ONLY the discrimination head; keep the ability encoder narrow. This lifts alpha
recovery to a strong baseline's level and collapses its across-seed variance,
with no cost to theta.

---

## Evidence so far (synthetic only)

- 4-way recovery (GPCM, 3 seeds, 150ep): the decoupled model ties the
  carefully-engineered ma-irt baseline on alpha (0.929 vs 0.935), keeps theta
  (static 0.967, drift 0.729), at fewer params and one encoder pass.
- ARCHITECTURE-INDEPENDENT (swap bench, LSTM / Transformer / DKVMN, 3 seeds):
  decoupling keeps theta and lifts alpha to ma-irt's level on every backbone.
  alpha cheap -> decoupled: LSTM 0.654 -> 0.929, Transformer 0.650 -> 0.925,
  DKVMN 0.708 -> 0.916.
- VARIANCE COLLAPSE: cheap-alpha is high-variance on every backbone (+-0.11 to
  +-0.19 across seeds); decoupling collapses it to +-0.02 to +-0.035 as well as
  lifting the mean. The variance signal is cleaner than the mean lift.
- One dynamics signature already in hand: the bare encoder's responsive theta
  overfits the static level WITH MORE TRAINING (0.97 at 150ep -> 0.68 at 500ep),
  while the regularised baseline is stable. Overfitting that grows with training
  is a trajectory property, not a capacity ceiling.

Sources: deep_irt/bench/outputs/alpha_fix_table.md, deep_irt/bench/outputs/swap_table.md.

Caveats: all synthetic, dense, K=4, Q=60, LogNormal(0,0.3) alpha. No real data.
The decoupled theta still softens under long training (it lacks the baseline's
LayerNorm / q-residual regularisation).

---

## Prior-art verdict (deep-research, 2026-06-16; 99 agents, 17 primary sources, 22/25 claims adversarially confirmed)

All three layers came back GENUINELY OPEN, and no single work or combination
covers the conjunction.

- MECHANISM (per-IRT-parameter embeddings): open. Nearest is Tsutsumi, Kinoshita
  & Ueno (EDM 2021, "Deep-IRT with independent student and item networks"), but
  it splits STUDENT vs ITEM, not theta vs alpha, and is Rasch-style with
  discrimination HARDCODED at 3.0, so it recovers no alpha at all. The recsys
  "multi-embedding" work was explicitly refuted as a true analog (it assigns no
  embedding to an interpretable parameter).
- PHENOMENON (alpha-theta capacity coupling degrading recovery, architecture-
  independent): open. No neural-IRT paper reports or fixes a discrimination-
  recovery degradation from representation sharing. VIBO (EDM 2020) measures
  recovery ONLY as ability correlation and reports no discrimination metric.
  Architecture-independence is addressed by nobody.
- FRAMING (learning-dynamics / measurement-validity gap between classical IRT,
  DKT, neural IRT): open. No work contrasts the three settings and locates a
  coupling or validity problem unique to the third.

Nearest neighbors to cite and distinguish:
- Tsutsumi et al. EDM 2021 -- student/item split, Rasch, no alpha.
- VIBO, EDM 2020 (arXiv 2002.00276) -- amortized variational IRT, recovery is
  ability-only.
- Urban & Bauer 2021 (arXiv 2109.09500), amortized IFA -- the sharpest neighbor;
  it DOES recover discrimination (loadings) via an amortized autoencoder, but no
  coupling analysis, no per-parameter decoupling.
- JE-IRT (arXiv 2509.22888, Sept 2025) -- the structural antithesis; everything
  read off one maximally-shared geometric space, no alpha scalar.
- PCGrad / Standley / embedding-collapse -- the MTL-interference machinery, but
  prediction-only; PCGrad's fix is gradient surgery that leaves the shared
  representation intact, the opposite of decoupling.

White space: the CONJUNCTION of the three layers.

---

## The honest caution: capacity or dynamics?

The empirical result alone is an ablation ("alpha needs more parameters"). It
becomes a contribution only if the cause is the COUPLING (a learning-dynamics
property of the shared parameterisation) and not mere CAPACITY (alpha simply
wants more degrees of freedom, which is trivial). These are two different claims
and must be separated:

- CAPACITY (static, trivial): a narrow shared embedding lacks the expressivity
  to encode both theta-relevant and alpha-relevant item variation. Independent
  of optimisation. A bigger embedding fixes it.
- DYNAMICS (optimisation, interesting): even with enough total capacity, the
  gradient flow drives the shared embedding toward theta's optimum (theta's
  gradient dominates) and alpha is left under-fit. The fix is SEPARATION, not
  size.

THE CRUX EXPERIMENT. Compare a shared embedding of width (w_theta + w_alpha)
against a decoupled model with the SAME TOTAL width split across the two tables.
If decoupling wins at matched total capacity, the effect is the SEPARATION (a
dynamics / coupling result, the interesting claim). If the matched-total shared
model catches up, it is just capacity (trivial, and we say so). The "64x64
shared trap" hints at dynamics (wide shared still underfit alpha at 0.77 while
breaking theta), but the theta-break confounds it, so the matched-total clean
experiment is required.

This single comparison decides whether the gap is worth a paper.

---

## The open core: how to study the learning dynamics (math-level "why")

Ranked by rigour-per-effort. The minimal credible package is A + B + C; D
sharpens the framing; E is aspirational.

A. MINIMAL ANALYTICALLY-SOLVABLE SURROGATE (the rigorous core). Strip to the
   simplest model that still shows the phenomenon: binary 2PL (not GPCM), a
   LINEAR encoder (a per-item embedding e_q fed directly to linear alpha/beta
   heads and a linear theta readout, no recurrence), static ability,
   continuous-time gradient flow. Write the population gradient-flow ODEs for
   (e_q, w_theta, w_alpha, w_beta) with e_q SHARED. Show analytically that the
   shared fixed point trades off alpha-fidelity against theta-fidelity (the
   embedding direction best for theta is not the one best for alpha), and that a
   separate alpha embedding decouples the ODEs so both reach their own optima. A
   toy where the dynamics are exact is the gold standard for a learning-dynamics
   claim.

B. GRADIENT-CONFLICT MEASUREMENT (empirical mechanism, runnable now). On the
   real model during training, split the gradient on the shared item embedding
   into its alpha-pathway and theta-pathway components (stop-grad the other
   head). Track over training: the magnitude ratio ||g_theta|| / ||g_alpha|| on
   e_q (hypothesis: theta dominates, it enters the loss at every occurrence while
   alpha enters multiplicatively and is low-information), and cos(g_theta,
   g_alpha) (hypothesis: conflict, negative or near-orthogonal). Confirm
   decoupling removes the competition. This tests "sharing couples them and theta
   wins" directly.

C. RECOVERY TRAJECTORY (the dynamics, not the endpoint). Learning dynamics is
   the trajectory. Track alpha-recovery and theta-recovery vs epoch, per
   embedding width, for shared vs decoupled. Hypotheses: alpha is learned LATE
   (after theta saturates) and is then crowded out as theta overfits; decoupling
   lets alpha keep improving. Extend the theta-overfit-vs-epoch curve we already
   have to alpha trajectories. Cheap.

D. FISHER-INFORMATION x REPRESENTATION-RANK BRIDGE (connect statistics to
   representation). Compute the 2PL/GPCM Fisher information w.r.t. (alpha, theta,
   beta). alpha's low Fisher information is the statistical fact. Then argue and
   measure: a d-dim shared embedding must encode per-item alpha variation in a
   subspace; if d is small and the dominant variation serves theta, the
   alpha-relevant direction is under-resolved (a rank/capacity argument).
   Measurable via the effective rank of the embedding and the projection of the
   alpha-gradient onto the top-k embedding directions. This bridge makes the
   framing precise: statistical low-information (Fisher) times representational
   squeezing (rank) equals the recovery failure.

E. NTK / SPECTRAL COUPLING (aspirational). In the lazy regime the recovered-
   parameter dynamics are linear ODEs governed by the NTK; the off-diagonal
   alpha-theta NTK block is the coupling and its eigenstructure gives convergence
   rates (alpha as the slow / under-determined mode). Likely intractable exactly
   for the full model; feasible on the toy (A) if it is set up in the lazy
   regime.

HONEST BOUND. Studying learning dynamics rigorously is a deep-learning-theory
subfield and full theory is hard. The realistic, defensible target is the toy
(A) plus gradient-conflict (B) plus trajectories (C): a solvable model that
exhibits the coupling, evidence the same mechanism operates in the real model,
and the dynamics over training. If even the toy does not cleanly show the
coupling, that is a signal the effect is capacity not dynamics, and we reframe
honestly.

---

## Risks to clear before claiming (ranked)

1. THE VARIATIONAL-BIAS BOUNDARY (sharpest). GVEM / IW-GVEM variational-IRT
   works document discrimination-estimate bias under variational inference. If
   they attribute it to the STATISTICAL variational approximation, our
   "learning-dynamics not statistics" line must draw that boundary carefully; a
   reviewer could say "known variational bias." Resolve first.
2. FRESHNESS. The LLM-as-examinee IRT line moves fast (JE-IRT is Sept 2025).
   Refresh the prior-art sweep near submission.
3. DIRECT PRE-EMPTION. Check whether Deep-IRT's Synthetic-5 or any DKVMN-IRT
   follow-up contains a simulation recovery experiment that reports
   discrimination.
4. REAL DATA + external corroboration of the architecture-independence claim
   (currently our own).

---

## Why this matters to the thesis (not a side-quest)

If recovered parameters are representation-coupled artifacts, then "a stable
measurement scale, invariant under extension" is meaningless -- the scale would
be an optimisation byproduct. Decoupling is a VALIDITY CONDITION for the learned
scale being a real measurement instrument. The gap is load-bearing for the
thesis's central measurement claim, not a detour.
