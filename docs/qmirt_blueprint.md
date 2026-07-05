# Q-MIRT blueprint v1.1 (plan for the plan; REVIEWED, awaiting author sign-off)

The redesign contract following the author's 2026-07-05 redirect (KT encoder
+ IRT decoder; learning modeled from event content, not merely detected;
one-to-many mapping considered; generative training considered). v1 was
attacked by two fresh-context adversarial reviews (editor; hostile
psychometric referee); v1.1 folds every accepted demand. Verdicts appended
at the end. This document supersedes the MODEL sections of
docs/qmirt_paper_plan.md v1.1 and inherits everything else from it: the
certification instruments and their results, the bank-calibration recipe,
the identification lemmas, the vocabulary/reporting rulings, the cast, the
operational rules. Evidence: the five-lane sweep (2026-07-05), this week's
record (docs/qmirt_experiment_results.md), and the two reviews.

HEADLINE CHANGES FROM v1 (both reviews, accepted):
1. PREDICTION-PRIMARY, not ELBO-primary. Our own record shows prediction
   loss plus the structural lemmas already delivered certified transfer;
   the KL buys the latent gauge, not structural consistency (Wang, Blei,
   Cunningham 2021, correctly scoped). Person parameters are handled by
   MARGINALIZATION (quadrature; amortized posteriors as the fast
   approximation with a mandatory amortization-gap diagnostic). The ELBO
   head-to-head demotes to a pre-registered appendix ablation.
2. TWO NEW HARD GATES ahead of everything: the encoder-honesty gate (the
   recognition network is a response-to-transfer channel until proven
   otherwise) and the bank gate (frozen cross-loading discriminations must
   be certified recoverable at C>1 before any one-to-many claim).
3. SCOPE SURGERY. The paper's spine: harness port + mapping boundary +
   modeled-gain certification + real data + heavy test. The locked exhibit
   moves to the introduction; the objective comparison and the
   projected-mastery stopping study move to appendices (the latter with
   its harm controls, or it is cut).
4. THE EMISSION IS NOW WRITTEN DOWN (below), as the multidimensional
   extension Chapter 0 itself names as future work, nesting its published
   unidimensional GPCM at C=1. Continuity by NESTING, not by a
   multidimensional spec that does not exist.

## The emission (exact; review-supplied specification, adopted)

Per-concept ability z_t in R^C, occasion = practice event. Item j carries a
DISCRIMINATION VECTOR a_j in R^C whose support equals its Q-row (a_jc = 0
wherever q_jc = 0; a pure anchor has one nonzero entry, a cross-loading
item several) and SCALAR step thresholds b_{j,0..K-2} (K-1 of them;
thresholds do not become per-concept). Compensatory composite
eta_jt = sum_c q_jc a_jc z_tc. Category-k logit = k * eta_jt - sum_{i<k}
b_{j,i}; softmax over categories. At C=1 this is exactly the Chapter 0
GPCM (main_magpcm_ijaied.tex eq. 297).

Calibration-time identification constraints (then FREEZE): per-concept
unit variance of z in the calibration cohort; one positive-loading anchor
per concept for sign; per-concept pure anchors with the confirmatory-MIRT
anchor/rank condition satisfied (and the three-items-per-concept floor
honored where the stronger guarantee is wanted); threshold centering per
item. Marginal ML on a measurement-regime cohort with adequate PER-CONCEPT
spread and joint coverage of cross-loading items; collinearity guarded
when concepts correlate. NOTE (citation-class fix): Xu-Zhang 2016 / Gu-Xu
2019 are discrete diagnostic-model completeness laws; the operative
condition for this continuous compensatory model is the confirmatory-MIRT
anchor + rank condition, stated in the paper as such.

## The five questions, adjudicated (v1.1)

Q1 (mapping). One-to-many remains the ambition, now behind TWO gates: the
bank gate (frozen cross-loading a_j recovered to pre-registered precision
under the marginal recipe at C>1; if this fails, the between-item readout
is the paper's design, immediately and without spending V1) and the
mapping boundary study (V1, demoted from gate to exhibit): the
permuted-credit twin confirms the finite-sample boundary around the STATED
anchor/rank condition, with a multimodality check (refit from
permuted-credit initialization; the fit gap is a truth-free routing
diagnostic). One-to-many becomes the paper's primary design ONLY if it
beats the certified between-item design on the full battery at a
pre-registered margin; otherwise the boundary is reported as a finding.

Q2/Q3 (modeled growth). Gain per practice event = stated softplus-linear
function of event content: partial-credit outcome magnitude (PFA
generalized; unclaimed in the literature), spacing (half-life form),
item features, graph position. RENAMED per review: this is
RECOVERY-CERTIFIED STATED GAIN, not "LLTM discipline" (LLTM explains a
first-identified difficulty; our gain explains increments of a latent, so
the warrant is recovery certification plus held-out EVENT-LEVEL forecast,
not the LLTM likelihood-ratio route). The spacing component is gated
behind beds with non-monotone identification content (the Lemma 2 rule
generalized: decay-like components are only fit where dips identify
them), and spacing regressors must vary independently of sequence
position. Growth's definition is unchanged (score gain on the frozen
reference items, with reliability and SEM per the consult rulings).

Q4 (locked). Motivating exhibit in the introduction: prediction-equivalent
fits with conflicting growth/transfer stories (person-learning vs
item-drift vs selection twins), anchored to Beck-Chang / van de Sande /
Doroudi-Brunskill (constraint-dependent identifiability), Rupp-Zumbo
(gauge), Jacovi-Goldberg (faithfulness). Not a gated program item.

Q5 (generative scheme). The transfer assumption lives in the TRANSITION
regardless of objective; the objective is not what buys identification
(the constraints are). The generative machinery earns its place as
uncertainty quantification and as the principled treatment of person
parameters (marginalization); whether ELBO training adds anything for
dynamics recovery is the pre-registered appendix ablation (TOST-style
equivalence margin, registered before the run; if prediction matches
ELBO within the margin on every certified quantity, we say so plainly).
Novelty is staked on certification + fitted-ablatable transfer + modeled
gain, never on the generative wrapper (the VTIRT collision otherwise).

## The model (v1.1)

1. EMISSION: as specified above; bank marginally calibrated on a
   measurement-regime cohort, frozen.
2. STATE: per-concept ability z_t; occasion = practice event.
3. TRANSITION (mechanism; all change practice-event-gated): own growth =
   (ceil_c - z)+ times lambda_n times g_own(event); transfer = signed
   per-pair G scaled by gamma_n, fired by source-concept practice, target
   gap/headroom scaling (bounded interference); optional relation-type
   factorization of G (SKT as foil). Lemmas 1-3 carried as structural
   constraints.
4. PERSON PARAMETERS AND INFERENCE: z0, lambda_n (learning-rate
   multiplier), gamma_n (transfer multiplier) are person parameters with
   IDENTIFICATION CONSTRAINTS E[lambda]=E[gamma]=1 and unit z0 scale
   (constraints, not hyperparameters; the multiplicative gauge is
   resolved by construction). Estimation posture: MARGINALIZATION
   (quadrature where dimensionality allows; the LSTM recognition network
   is the amortized approximation). Mandatory diagnostics: the
   amortization-gap check (amortized vs per-learner-optimized posterior
   on a held-out subsample; structural estimates must not move) and an
   N-scaling consistency probe on gain and G. The R9-compatible variant
   (traits inferred from the conditioning window only) is tested first
   at the encoder-honesty gate.

OBJECTIVE: prediction-primary (next-category prediction objective with
person parameters marginalized; never called a likelihood, per the
standing ruling). ELBO variant of the same architecture exists only for
the appendix ablation and for posterior uncertainty.

## Program (v1.1; gates before exhibits, spine before appendices)

GATE A, HARNESS PORT (was V0): battery re-targeted; encoder-OFF degenerate
  case must reproduce this week's certified results. Kill thresholds are
  RE-REGISTERED here for the new model class (nothing inherited
  automatically; the P1b principle applied to ourselves).
GATE B, ENCODER HONESTY (new; the editor's one thing): encoder ON,
  per-learner transfer multiplier free, on the null and permuted twins.
  If the recognition network fabricates transfer where the response-free
  transition could not, the redesign is unsound; the v1.1 plan's model
  stands and this blueprint dies honestly. Variants tested:
  conditioning-window-only inference vs full-sequence inference.
GATE C, BANK GATE (new; the psychometrician's one thing): C>1
  cross-loading discrimination recovery under the marginal recipe,
  precision pre-registered, correlated-concept cohorts included. Fail ->
  between-item readout for the whole paper, one-to-many spend stops.
V1 MAPPING BOUNDARY (demoted to exhibit): the anchor/rank condition
  stated; permuted-credit twin maps the finite-sample boundary;
  multimodality/permuted-init diagnostic; one-to-many promoted only if it
  beats between-item on the battery at a pre-registered margin.
V2 MODELED-GAIN CERTIFICATION (the paper's novelty core): per-component
  recovery twins AND per-component G-FABRICATION checks (Lemma 3 recurs
  once per component: misspecify component k, confirm null G) AND
  held-out event-level forecast (does the model know which events teach
  more). Spacing only in non-monotone beds. Coverage reported (below).
V3 OBJECTIVE ABLATION (appendix): ELBO vs prediction, TOST margin
  pre-registered, posterior-collapse diagnostics, coverage.
V4 LOCKED EXHIBIT (introduction).
V5 PROJECTED-MASTERY STOPPING STUDY (appendix proof-of-concept at most;
  no invented construct name): treated as a CLASSIFICATION problem:
  decision consistency and accuracy, false-mastery rate vs synthetic
  truth, stopping on the LOWER CREDIBLE BOUND (never the point
  projection), the growth-without-transfer control arm (else the gain
  term wins and transfer is decoration), and the HARM TWIN (false
  transfer believed -> stops too early), which is the dangerous failure
  of any transfer-aware stopping and is the study's real question.
V6 REAL DATA: KDD Cup 2010 + EdNet; positivity audit first; first-attempt
  scoring; three-way verdict space carried.
HEAVY TEST on the surviving configuration: full battery, scale,
  mismatched-gain-form AND mismatched-transfer-form twins (the R6 double
  debt), state-noise-inflated twins, converged budgets, seed clustering.

COVERAGE DISCIPLINE (deployed, not promised): empirical-vs-nominal
coverage WITH width at V1 (routing posteriors), V2 (component
coefficients, monotone vs non-monotone), V3, V5 (projection vs realized),
each against matched + mismatched-form + noise-inflated twins. Coverage
that holds matched and collapses mismatched is the expected honest result
and is reported as such.

MINIMAL PUBLISHABLE CORE (if everything after V2 fails): Gate A + Gate B
+ Gate C outcome + V1 boundary + V2 + V6 = "certified modeled learning on
a frozen ordinal bank, with a stated transfer-attribution boundary."

## Collisions (v1.1)

Nearest neighbors and the stake: PSI-KT (always-on empowered asymptote,
no measurement layer, no certification; AGPL, design reference only);
SKT (given graph in the transition, prediction-only, uncertified; our
structural foil); VTIRT (ELBO + amortized inference + IRT emission; the
dangerous collision if the generative wrapper were the identity, which is
why it is not); Dynamic LENS (inference/mechanism split, no gating, no
transfer); AFM/PFA/HLR/DAS3H (gain-function lineages, no state-space, no
certification); AKT/GIKT/NeuralCD (readout enrichment; AKT names our
credit-assignment problem and leaves it open). The paper's identity:
the certification recipe applied to a model that MODELS learning, built
as the multidimensional extension Chapter 0 named as future work.

## Risks (v1.1 additions to the carried set)

Encoder fabrication (Gate B exists for it); frozen cross-loading
discrimination quality (Gate C exists for it; C1 precedent says this is
THE fragile parameter); amortization gap that does not vanish with N
(diagnostic mandated); multiplicative trait gauge (constraints stated);
spacing/decay confounding (component gated to non-monotone beds);
correlated-concept collinearity at calibration (cohort design +
guard); scope creep (spine frozen; appendices are appendices).

## Feasibility

About two months solo at campaign discipline, with V1 and V2 on
independent synthetic beds run in parallel; Gates A-C are days, not
weeks, because they reuse this week's beds and recipes. The ELBO variant
is built once, small, and only for V3/UQ.

## Review verdicts (2026-07-05, condensed; full texts in the session log)

EDITOR: blocking finding on the encoder as a response-to-transfer channel
(Gate B created); kill re-registration demanded (folded into Gate A);
prediction-primary inversion demanded (adopted); V5 cut to controlled
appendix (adopted); per-component fabrication checks (adopted); scope =
one paper via the spine above (adopted); novelty staked off the
generative wrapper (adopted); feasibility ~2 months after cuts.
PSYCHOMETRICIAN: blocking findings on the unwritten emission (now
written, nesting Chapter 0 at C=1) and on frozen cross-loading
discrimination (Gate C created); citation-class fix (adopted); "LLTM
discipline" renamed recovery-certified stated gain (adopted); spacing
gated to non-monotone beds (adopted); marginalization posture with
identification constraints and amortization-gap diagnostic (adopted);
V5 reframed as classification with harm twin and lower-bound stopping
(adopted); coverage deployment grid (adopted).

## Author sign-off checklist

1. Prediction-primary inversion (ELBO demoted to appendix + UQ).
2. The two new hard gates (encoder honesty; bank gate) as go/no-go.
3. One-to-many as gated ambition with between-item as the certified
   fallback (your Q1 preference survives exactly as far as the gates
   allow, and the boundary is a publishable finding either way).
4. V5 as controlled appendix without an invented name (your finish-early
   payoff survives as the stopping study with its harm controls).
5. The emission specification (nesting Chapter 0; this seeds the paper's
   setup section).
6. Scope spine + two-month posture.
