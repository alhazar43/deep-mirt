# Interpretability critiques, primary-text read

Two papers read in full (PDF text extracted via PyMuPDF, not abstracts).
Local copies: `data/bed_interp_critiques/p1_2101.11335.{pdf,txt}`,
`data/bed_interp_critiques/p2_2511.02718.{pdf,txt}` (gitignored `data/`
tree, not committed).

## Paper 1: Ding & Larson, "On the Interpretability of Deep Learning
Based Models for Knowledge Tracing" (arXiv:2101.11335, AAAI 2021
workshop; extends their 2019 EDM paper from ASSISTments to EdNet KT1,
millions of interactions, 1495 skills)

**Methods.** Four experiments on a standard DKT (LSTM, output layer
`y_t` = per-skill correctness probability):
1. t-SNE of the concatenated LSTM gate/state vector
   `a_t = [f_t, i_t, C̃_t, C_t, o_t, h_t]` at the first timestep across
   many students, colored by skill ID and correctness.
2. "Oracle Student" simulation: feed the model 100 consecutive correct
   answers restricted to one skill, repeated for 3 different skills
   (#7, #8, #24); track the activation vector's 2-D t-SNE trajectory
   and the Euclidean distance between successive/cross-skill activation
   vectors. Repeated for an "anti-oracle" (always incorrect).
3. Order-sensitivity test: two synthetic sequences with identical
   per-skill counts and identical within-skill correct/incorrect
   ordering, differing only in how skills are interleaved; and a
   dataset-wide "spread out" shuffle preserving within-skill order.
   Evaluate AUC and r² before/after, for PFA, BKT, DKT, DKT-spread.
4. Untrained-recurrent-network ablation: randomly initialize the LSTM,
   train only the final linear readout (`W_o`, `b_o`); compare AUC/r²
   to the fully trained DKT (replicates Wieting & Kiela 2019's random-
   encoder result for sentence embeddings, applied to KT).
5. DKVMN value-memory read/write trace for single-tag vs multi-tag
   (composite-skill) questions.

**Findings (exact).**
- t-SNE at t=0 clusters by correct/incorrect, not by skill ID — no
  skill sub-clustering visible.
- Oracle-student convergence: activation vectors seeded from 3
  different skills, drilled independently, converge to *the same*
  region of activation space after ~20 correct responses (Euclidean
  distance between them and between successive steps shrinks toward
  zero). After 20 correct answers on one skill, predicted probabilities
  for essentially all skills rise above 0.5, "regardless of the
  specific practice skill." Anti-oracle drilling shows the mirror-image
  collapse to a shared low-mastery state.
- Order sensitivity: reordering (holding within-skill order fixed)
  produces a large, consistent AUC drop for DKT — e.g. ASSISTments
  09-10(b) AUC .82 (original order) -> .72 (spread) — while BKT/PFA are
  provably invariant to this reordering by construction. Five of six
  datasets show a comparable AUC drop under DKT (09-10(a) .81->.72,
  09-10(c) .75->.71, 14-15 .70->.67, KDD .79->.76).
- Untrained-recurrent (only `W_o`,`b_o` trained) AUC/r² is close to,
  sometimes only marginally below, the fully trained DKT across all six
  benchmarks (e.g. 09-10(b): AUC .82 trained vs .79 untrained, r² .31
  vs .26; KDD: AUC .79 vs .76). It also beats BKT and roughly matches
  or beats PFA on most datasets despite zero learned recurrence.
- DKVMN: per-question memory-slot activation is consistent
  (correctness-independent) for single-tag questions, but multi-tag
  (composite) questions do not activate a superset of their component
  tags' slots — no evidence of compositional per-skill memory.

**Direct implication for per-KC ability readouts.** This is the
sharpest available primary-source evidence for exactly the failure mode
kt-mirt calls "stable and wrong": a standard recurrent KT encoder,
under no explicit per-skill constraint, (a) collapses multi-skill state
toward one scalar "oracle" attractor under repeated single-skill
practice — i.e. drilling KC A visibly inflates the readout for KC B,
which would masquerade as false positive cross-KC transfer under A1,
and as spurious growth on undrilled KCs under A4 — and (b) gets most of
its accuracy advantage from high-dimensional random projection rather
than learned recurrent dynamics, since an untrained core with only the
readout head trained is nearly as accurate. Both effects are about the
DKT *encoder itself*, independent of any IRT decoder layered on top, so
they attack A3/A4 at the representation level, before the IRT readout
is even applied.

## Paper 2: Khalid, Deriyeva & Paaßen, "Does Interpretability of
Knowledge Tracing Models Support Teacher Decision Making?"
(arXiv:2511.02718, preprint, Bielefeld University, Nov 2025)

**Methods.** Not a hidden-state analysis; a decision-utility study.
Defines "interpretable" narrowly as "provides an explicit ability
estimate θ_t,k per skill per timestep" (true for BKT, PFA; false for
DKT, which is the deliberate non-interpretable control — the paper
explicitly declines to use existing DKT-interpretability add-ons
because the research question is whether interpretability is needed at
all). Two studies, both built on an Elo-simulated ground-truth student
(2 skills, 4 tasks, task 3 easier/dual-skill, task 4 hard/dual-skill):
1. **Simulation study.** Task selection is driven automatically by each
   model's own expected-learning-gain formula, computed directly from
   that model's outputs (BKT/PFA use their own θ; DKT has no θ, so its
   learning-gain proxy is the predicted-probability delta from a
   simulated next success/failure on each candidate task). 500 students
   generate training data; 1000 held-out students per model evaluate
   it.
2. **Teacher study, N=12** (university teaching staff, >=6 months
   experience, no KT background, blinded to which model is running).
   Each teacher runs 9 sessions (3 per model) picking tasks and stop
   points from a dashboard that shows past outcomes and predicted
   success probabilities for all three models, plus an explicit
   ability-estimate graph for BKT/PFA only (DKT has none to show).
   SUS usability and TOAST trust questionnaires after each model.

**Findings (exact).**
- Simulation: BKT/PFA reach true (Elo-defined) mastery in ~6 steps,
  matching the theoretical optimum; DKT needs a median of 14 steps,
  fails to detect mastery at all in most runs (hits the 30-task cap),
  and stops early (before true mastery) in 24% of runs. Difference
  significant at p<1e-10 (Wilcoxon). So when the *model itself* drives
  decisions, the non-interpretable model is a measurably worse teacher.
- Teacher study: BKT/PFA rated higher on usability (SUS ~60 vs DKT
  ~50, p<.05) and trust (TOAST understanding subscale, BKT vs DKT
  p<.01 d=0.84; both TOAST subscales, PFA vs DKT p<.05). But **task
  quality did not follow**: median steps-to-mastery under teacher
  control was 8 (BKT) vs 7 (PFA and DKT, not significantly different),
  and DKT-condition students actually reached mastery *faster* than
  BKT-condition (p≈.047, one-sided Welch t) because teachers picked the
  objectively optimal dual-skill task 4 more often under DKT — despite
  the DKT interface offering no explicit ability number to justify
  that choice. Teachers also stopped prematurely more often under BKT
  (16% of runs) than PFA (5%) or DKT (3%), i.e. the explicit ability
  graph correlated with more, not fewer, premature-stop errors.

**Direct implication.** The presence of an explicit, legible per-skill
ability estimate reliably buys subjective trust and usability, but in
this controlled test it did **not** reliably buy better downstream
human decisions, and in one arm (premature stopping) correlated with
worse decisions. This does not bear on whether a readout is measurably
correct (kt-mirt's A3/A4 question); it bears on a different, further
claim — that a validated readout is thereby useful for pedagogical
decisions — which the kt-mirt mission does not currently make (G1/G2
are measurement claims, not decision-support claims) but which the
"explainability layer" framing risks implying by association.

## Verdict for the kt-mirt battery

**Required new arms, both targeting A3 (signed readout audit) and A4
(existence gate / posture-by-bed growth study), motivated by paper 1:**

1. **Untrained/frozen-encoder null.** Add a random-encoder control
   (freeze the transition core at initialization, or shuffle its
   trained weights, train only the IRT/ability readout head) as a
   mandatory arm alongside the already-planned permutation null and
   static twin. Paper 1's headline result is that this null nearly
   matches a fully trained DKT on real benchmarks (AUC gap 2-3 points
   on 5/6 datasets); if kt-mirt's per-KC readout survives A3/A4 but an
   untrained core would pass the same gates, the certification is not
   diagnostic of learned per-KC state and the "stable and wrong"
   verdict applies.
2. **Single-KC-drill contamination probe.** An oracle/anti-oracle-style
   synthetic arm: drill one KC to saturation (or to zero) in isolation
   and check whether *other* KCs' readouts move. This is a sharper,
   mechanistic version of the existing "stable and wrong" concern and
   is directly actionable for A1 (a contaminated readout would register
   as false-positive signed transfer) and A4 (contamination would
   inflate apparent growth on KCs that were never practiced). Paper 1
   shows this failure mode is not hypothetical — it is what a vanilla
   LSTM-based DKT does by default.
3. **Order/recency-invariance stress test**, added to A4's truncation-
   stress arm: hold within-KC response order fixed, permute the
   cross-KC interleaving, and check per-KC readout stability. BKT/PFA
   are invariant by construction; paper 1 shows plain DKT is not
   (AUC drops of 3-10 points under reordering on 5/6 real benchmarks).
   A large per-KC readout shift under reordering is evidence the model
   is tracking aggregate recency, not per-KC state.

**Framing requirement, motivated by paper 2 (not a new experimental
gate):** A3/A4 passing establishes measurement validity of the per-KC
readout only. Do not state or imply, in the paper or in the
"explainability layer" framing, that a validated readout is thereby
useful for pedagogical task-selection or stopping decisions — that is
an separate, unvalidated decision-utility claim, and paper 2's own
controlled test found interpretable ability estimates did not reliably
improve (and in one arm, worsened) human decisions relative to a
non-interpretable model. If the program ever wants to make a decision-
support claim, it needs a decision-simulation study of paper 2's kind,
not just a recovered-signal audit.
