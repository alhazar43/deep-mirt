# Survey: interpretable / IRT-flavored / learning-gain knowledge tracing (2015-2026)

Scope: for each named model, what interpretable parameters it exposes, how (if at
all) interpretability was validated, whether readouts were audited for
stability or faithfulness, and whether per-KC ability trajectories are
produced. Then critique papers on KT interpretability, state degeneracy, and
benchmark practice. Closing section addresses the program's working
hypothesis: that nobody audits or anchors the readout layer.

## Deep-IRT (Yeung, 2019)

arXiv:1904.11738, EDM 2019. Wraps a DKVMN (dynamic key-value memory network)
encoder with an IRT-style read: student ability `θ_tj = tanh(W_θ·f_t + b_θ)`
per knowledge component (KC) `j` at time `t`, and item difficulty
`β_j = tanh(W_β·q_t + b_β)`, one value per KC, not per individual item.
Predicted correctness is the standard 1PL logistic `σ(3.0·θ_tj − β_j)`.

Validation: the paper does compare its learned difficulty against three
external references (classical item analysis, a fitted traditional IRT
model, and PFA) on the FSAI-F1toF3 dataset, reporting Pearson correlations
of roughly r=0.56, r=0.60, r=0.45 respectively, i.e. positive but moderate
agreement, not a strong recovery claim. It also runs t-tests showing
Deep-IRT's predictive accuracy is statistically indistinguishable from
plain DKVMN on most datasets, i.e. the interpretability layer is "free" in
predictive terms. (arXiv:1904.11738, https://arxiv.org/pdf/1904.11738)

Stability: the paper's own text acknowledges the model "still suffers from
the reconstruction issue... discovered in the DKT model" — i.e. estimated
ability can move the wrong direction after an observed response (increase
after a wrong answer, decrease after a correct one). This is the same
failure mode later formalized as the "reconstruction" and "waviness"
problems in Yeung & Yeung's DKT-regularization paper (arXiv:1806.02180,
"Addressing Two Problems in Deep Knowledge Tracing via
Prediction-Consistent Regularization", CSEDU 2018), which adds explicit
regularizers penalizing non-monotonic, oscillatory predicted-mastery
curves. So the closest thing to a stability audit of an IRT-flavored KT
readout in this literature is self-reported by the original authors as an
open, only partially patched problem — not something later papers went
back and re-audited under the Deep-IRT name specifically.

Per-KC ability trajectories: yes, this is the model's central deliverable —
θ_tj is emitted at every timestep for every KC the student has touched,
and the paper's qualitative figures plot it over time. But there is no
learning-gain parameter separated out; the trajectory is just the raw
per-step ability estimate, and its trustworthiness as a growth signal is
exactly what the reconstruction-problem admission undercuts.

## Deep-IRT follow-up: post-test correlation validation

Scruggs, Baker, McLaren, "Extending Deep Knowledge Tracing: Inferring
Interpretable Knowledge and Predicting Post-System Performance"
(arXiv:1910.12597, ICCE 2020,
https://www.cs.cmu.edu/~bmclaren/pubs/ScruggsBakerMcLaren-ExtendingDeepKnowledgeTracing-ICCE2020.pdf).
This is the one paper in the survey that validates a KT-derived knowledge
estimate against an external ground truth rather than an internal
consistency check: it extends DKT/DKVMN to also output a knowledge
estimate and shows this estimate correlates better with an external
**post-test** than knowledge estimates from Bayesian Knowledge Tracing
(BKT) or Performance Factors Analysis (PFA). This is the strongest
external-validity result found in the survey and is a genuine
counter-data-point to "nobody validates KT interpretability" — but note it
validates a knowledge/mastery estimate against a posttest, not an IRT
ability/difficulty parameter's stability or faithfulness to the model's
internal computation, and it is a single paper, not a norm the field
adopted.

## AKT (Ghosh, Heffernan, Lan, 2020)

"Context-Aware Attentive Knowledge Tracing," arXiv:2007.12324, KDD 2020
(https://arxiv.org/abs/2007.12324; code
https://github.com/arghosh/AKT). Four modules: Rasch-model-based
embeddings, an exercise encoder, a knowledge encoder, and a knowledge
retriever, tied together by a monotonic attention mechanism (exponential
decay plus a context-aware relative-distance term) over past responses.
The Rasch component regularizes concept and question embeddings so that
individual questions under the same KC get their own discrimination-like
offset without needing a full per-question embedding table — this is the
one named interpretable parameter (a question-specific deviation term
in Rasch style), used to keep the embedding table small.

Validation: the paper's own framing is purely predictive (AUC gains up to
~6% over prior KT methods on several benchmarks); it does not run a
separate study validating the Rasch-derived difficulty offsets against an
external difficulty measure, nor does it audit the attention weights for
faithfulness to the underlying decision. The Rasch embedding is motivated
and used as an inductive bias / regularizer for prediction accuracy, not
put forward as an interpretability contribution that is itself validated.
No per-KC ability trajectory is exposed; AKT does not produce a scalar
ability estimate per KC per timestep the way Deep-IRT does — its interpretable
surface is the attention weights and the per-question Rasch offset, not a
trait trajectory.

## qDKT (Sonkar, Waters, Lan, Grimaldi, Baraniuk, 2020)

"qDKT: Question-centric Deep Knowledge Tracing," arXiv:2005.12442, EDM 2020
(https://arxiv.org/abs/2005.12442). Extends DKT to track per-question
success probability instead of collapsing all questions under a KC into
one observation, using graph-Laplacian regularization to smooth
predictions within a KC and a fastText-style embedding initialization.
This is a prediction-accuracy paper, not an interpretability paper: no
named ability/difficulty parameter is exposed as an interpretable
construct, no validation of interpretability is attempted, and no
per-KC ability trajectory is produced (its unit of output is a
per-question correctness probability, not a trait estimate). Relevant to
this survey mainly as the item-granularity precedent that Deep-IRT-style
per-KC-only difficulty erases (it treats "all questions under one KC are
interchangeable" as the flaw to fix), which several IRT-flavored KT models
inherit uncritically.

## KTM (Vie & Kashima, 2019)

"Knowledge Tracing Machines: Factorization Machines for Knowledge
Tracing," arXiv:1811.03388, AAAI 2019
(https://arxiv.org/abs/1811.03388; code
https://github.com/jilljenn/ktm). Shows factorization machines subsume
additive factor models, performance factor analysis, and multidimensional
IRT as special cases; each factor (student, item, skill, attempt count,
etc.) gets both a scalar bias weight and a low-dimensional embedding, and
the log-bilinear form means the weights are directly interpretable as
per-factor biases (analogous to difficulty/ability offsets) and the
embeddings capture pairwise interactions. Not a deep sequence model
(no LSTM/transformer/DKVMN encoder) — it is a shallow generalized linear
/ bilinear model, so "interpretability" here means linear-model
interpretability (inspect a weight), not extraction from a trained neural
hidden state, and the interpretability question this survey cares about
(does the readout track a genuinely time-varying encoder state faithfully)
does not really arise for KTM. No validation study of the interpretability
of the weights beyond the standard claim that log-bilinear models are
"interpretable by construction." No per-KC ability trajectory over time —
KTM is not inherently sequential/recurrent in the way DKT-family models
are, though attempt-count features let it capture a coarse practice-effect
curve.

## LPKT (Shen et al., 2021)

"Learning Process-consistent Knowledge Tracing," KDD 2021
(http://staff.ustc.edu.cn/~huangzhy/files/papers/ShuanghongShen-KDD2021.pdf;
DOI 10.1145/3447548.3467237; pyKT docs https://pykt.org/lpkt). The one
model in this set with an explicit, named **learning gain** parameter: it
models a "learning cell" combining question, answer-correctness, and
response-time/interval-time embeddings, computes learning gain as the
difference between successive learning-cell states gated by a learning
gate, and separately models forgetting via interval time in a forgetting
gate. This is architecturally the closest analog in the published
literature to the program's G2 target (a per-learner growth signal
distinguished from a static state).

Validation: the paper's interpretability check is a case study, not a
quantitative audit. Students are split into three score-based groups;
students whose group improves are labeled "High QLG" (quality learning
gain) and the paper shows the model's learning-gain magnitude is larger
for that group than for group-decliners, i.e. a qualitative,
directionally-consistent case study on one split of one dataset. No
correlation against an external growth measure, no stability check
across seeds/inits, no separated item-parameter path, no audit of whether
the "learning gain" readout could be produced by a model that has not
actually learned KC-specific dynamics (the stable-and-wrong failure mode
this program is built around). Learning gain is emitted per interaction
(tied to the KC of the current item) rather than as a clean
per-KC-per-student trajectory decomposed the way Deep-IRT emits θ_tj;
LPKT's state is a single fused vector, not a per-KC ability vector, so it
does not straightforwardly give "per-KC ability trajectories" in the
Deep-IRT sense — it gives a scalar learning-gain-at-this-step tied to
whichever KC was just practiced.

## IKT (Minn, Vie, Takeuchi, Kashima, Zhu, 2022)

"Interpretable Knowledge Tracing: Simple and Efficient Student Modeling
with Causal Relations," arXiv:2112.11209, AAAI 2022
(https://arxiv.org/abs/2112.11209;
https://cdn.aaai.org/ojs/21560/21560-13-25573-1-2-20220628.pdf). Not a
deep sequence model. Builds three explicit, human-legible features per
student-skill-time observation — individual skill mastery, an "ability
profile" meant to capture learning transfer across skills, and problem
difficulty — via data mining, then feeds these into a Tree-Augmented
Naive Bayes (TAN) classifier rather than a neural net. Interpretability
here is definitional (the model literally is the three named features
plus a shallow Bayes-net structure showing which feature depends on which),
so there is no separate "was the readout faithful to the network"
question — there is no opaque network to be unfaithful to. This is useful
as a contrast case: IKT achieves interpretability by giving up the deep
encoder entirely, which is the opposite move from what this program wants
(a genuine sequence encoder with an audited readout layer). No reported
stability audit is needed/attempted since the model is shallow; no
per-KC ability trajectory beyond the single mastery/ability-profile
feature per skill per step.

## QIKT (Chen, Liu, Huang, Luo, 2023)

"Improving Interpretability of Deep Sequential Knowledge Tracing Models
with Question-centric Cognitive Representations," arXiv:2302.06885, AAAI
2023 (https://arxiv.org/abs/2302.06885; pyKT docs
https://pykt.org/qikt). Learns question-sensitive cognitive
representations from a question-centric knowledge-acquisition module and
a question-centric problem-solving module, then finishes with an
IRT-based prediction layer explicitly built to "generate interpretable
prediction results." This is architecturally the closest published
analog to the program's own DeepIRTModel design (swappable encoder,
IRT-style decoder head) among the surveyed models. Validation described in
secondary sources is prediction-accuracy benchmarking (QIKT beats prior
KT baselines on AUC across pyKT's standard datasets) plus qualitative case
studies of the interpretable representations, not a quantitative
faithfulness or stability audit of the IRT layer against ground truth or
against perturbation. No evidence found of per-KC ability trajectories
being extracted and validated as growth signals; the "interpretable
prediction results" claim centers on question-level cognitive state, not
a longitudinal per-KC ability curve.

## DIMKT (Shen et al., 2022)

"Assessing Student's Dynamic Knowledge State by Exploring the Question
Difficulty Effect," SIGIR 2022, DOI 10.1145/3477495.3531939
(http://staff.ustc.edu.cn/~cheneh/paper_pdf/2022/Shuanghong-Shen-SIGIR.pdf).
Explicitly incorporates a difficulty level into the question
representation and models knowledge-state updates as a function of that
difficulty in three stages (subjective difficulty perception, personalized
knowledge acquisition, knowledge-state update). Difficulty here is a
discretized level (a bucketed representation of empirical question
difficulty), not a continuous IRT-style β. Validation is again predictive
benchmarking plus a qualitative claim of "superior interpretability" via
difficulty-conditioned behavior, not a quantitative audit against an
external difficulty scale or a stability check. No per-KC ability
trajectory reported; the knowledge state is per-KC but the paper's own
interpretability claims center on the difficulty axis, not a longitudinal
ability readout.

## Per-skill-ability Deep-IRT variants

Two adjacent lines were found but not independently verified at source
depth in this pass:
- Time-and-Concept Enhanced Deep Multidimensional IRT for Interpretable
  Knowledge Tracing (ScienceDirect,
  https://www.sciencedirect.com/science/article/abs/pii/S0950705121000824,
  Knowledge-Based Systems 2021) — extends Deep-IRT toward a
  multidimensional-IRT-style readout with time and concept enhancements;
  abstract-level only, full text not fetched, flag as unverified detail.
- "Incorporating Item Response Theory into Knowledge Tracing" (Springer,
  AIED 2021 workshop/companion,
  https://link.springer.com/chapter/10.1007/978-3-030-78270-2_20) — another
  IRT-into-KT integration; not fetched at full-text depth, flag as
  unverified detail.
Both are lower-confidence entries: found via search snippets only, not
read in full, so treat any claim about their validation methodology as
unconfirmed until read directly.

## Critique / meta-level papers

**pyKT** (Liu et al., "pyKT: A Python Library to Benchmark Deep Learning
based Knowledge Tracing Models," arXiv:2206.11460, NeurIPS 2022 Datasets
and Benchmarks track,
https://proceedings.neurips.cc/paper_files/paper/2022/hash/75ca2b23d9794f02a92449af65a57556-Abstract-Datasets_and_Benchmarks.html).
The reform argument is specifically about **prediction** benchmark
practice: prior KT papers used private, inconsistent data-preprocessing
and evaluation protocols (e.g. inconsistent train/test splits, leakage
across a student's own future interactions, non-standardized KC-question
mappings), which pyKT standardizes. This is a benchmark-hygiene critique,
not an interpretability critique — it does not audit IRT-style readouts
at all. Its relevance to this program is indirect: it establishes that
even the *prediction* numbers underlying these models were not reliably
comparable pre-2022, which is a weaker but adjacent form of "nobody
checked whether the reported numbers meant what they claimed to mean."

**A Survey of Explainable Knowledge Tracing** (Bai, Zhao, Wei et al.,
arXiv:2403.07279, Applied Intelligence 2024,
https://link.springer.com/article/10.1007/s10489-024-05509-8). Splits
explainable-KT (xKT) methods into transparent models (Markov-process and
logistic-regression-based, i.e. BKT/IRT/PFA-family) versus black-box
models needing post-hoc explanation, and further into ante-hoc and
post-hoc interpretable methods. Directly states that "current evaluation
methods for xKT are lacking" and, to make the point concrete, runs its
own contrast-and-deletion experiments to explain a plain DKT model on
ASSISTments2009 with three off-the-shelf explainability (xAI) techniques
rather than relying on any published KT paper's self-reported
interpretability claim. This is the strongest direct third-party
confirmation found that the field lacks a validation standard for KT
interpretability claims — a 2024 survey explicitly says evaluation is
lacking and has to build its own probe rather than cite an existing
audit methodology.

**Reconstruction/waviness critique** (Yeung & Yeung, "Addressing Two
Problems in Deep Knowledge Tracing via Prediction-Consistent
Regularization," arXiv:1806.02180, CSEDU 2018,
https://arxiv.org/pdf/1806.02180). Predates most of the IRT-flavored
models above but is the origin of the stability failure mode
(non-monotonic, "wavy" mastery estimates that move the wrong direction
relative to the observed response) that Deep-IRT's own authors later
admit their model inherits. Proposes loss-side regularizers as a partial
fix, not a readout-layer audit or anchoring scheme.

**LRP interpretation of DKT on EdNet** (arXiv:2111.00419, "Interpreting
Deep Knowledge Tracing Model on EdNet Dataset,"
https://ar5iv.labs.arxiv.org/html/2111.00419). Applies Layer-wise
Relevance Propagation to attribute a DKT prediction to specific past
questions and reports moderate self-consistency (roughly 70% of
positive-prediction sequences and 56% of negative-prediction sequences
reach ≥90% "consistent rate" under their own metric) plus deletion
experiments showing relevance-ranked deletions degrade predictions faster
than random deletions. Notable because it is a genuine faithfulness-style
audit — but of an attention/relevance attribution over history, not of an
IRT-style ability/difficulty parameter, and its own consistency numbers
(56-70%) read as evidence *against* strong faithfulness, not for it. The
paper explicitly flags "skill-level interpretability" as unaddressed
future work, i.e. even this audit stops short of the per-KC readout layer
this program targets.

## Verdict on the program's working gap claim

The claim to confirm or refute: **nobody audits or anchors the readout
layer** of prediction-trained KT models before treating its outputs
(per-KC ability, difficulty, learning gain) as scientific claims.

**Largely confirmed, with one partial and one adjacent counter-example.**
Across the eleven models/lines surveyed:
- Only Deep-IRT itself reports any quantitative check of a readout
  against an external reference (moderate correlations, r≈0.45-0.60, with
  classical item analysis / traditional IRT / PFA difficulty) — and even
  its own authors flag that the same model exhibits the DKT
  reconstruction problem (ability moving the wrong direction after an
  observed response), i.e. the one paper that checks correlation does not
  also check monotonicity/faithfulness, and the one paper (Yeung & Yeung
  2018) that addresses monotonicity does not check correlation to an
  external scale. No single paper does both.
- The Scruggs/Baker/McLaren extension (arXiv:1910.12597) is the one clear
  counter-example of genuine external validation — a knowledge estimate
  checked against an actual post-test, beating BKT and PFA — but it
  validates a knowledge-mastery estimate, not an IRT ability/difficulty
  parameter's internal faithfulness, and it stands alone; it did not
  become standard practice for later IRT-flavored KT papers (AKT, qDKT,
  QIKT, DIMKT all skip this kind of check).
- LPKT's learning-gain validation is a qualitative case study on one
  dataset split (High-QLG vs Low-QLG groups), not a quantitative or
  adversarial audit, and is the closest any surveyed paper comes to
  validating a growth/gain readout specifically — still short of anything
  resembling a pre-registered null test.
- AKT, qDKT, QIKT, DIMKT treat their interpretable components (Rasch
  offsets, question-centric representations, difficulty levels) as
  regularizers or architectural choices justified by predictive-accuracy
  gains; none of them independently audits whether the resulting
  parameter actually tracks the construct it is named after.
- The one direct third-party meta-critique found (Bai et al. 2024 survey)
  states outright that "current evaluation methods for xKT are lacking"
  and has to construct its own probe rather than point to a field
  standard, which is a fairly direct third-party confirmation of the gap
  at the survey level, not just an absence-of-evidence inference from
  this report.
- No stability audit under seed variation, hyperparameter
  perturbation, or adversarial/null-model construction (i.e. "does a
  model that provably has not learned per-KC dynamics still produce a
  plausible-looking per-KC ability trajectory") was found for any of the
  eleven models. That specific test — a pre-registered null designed to
  catch a stable-and-wrong readout — appears to be genuinely absent from
  this literature, which is the sharpest form of the program's bet and
  the part most safely still claimed as open white space.

Caveat on completeness: this pass is a targeted survey of the eleven named
models plus meta-critiques, built from search snippets and five full-text
fetches (Deep-IRT arXiv HTML, the EdNet LRP paper, plus search-level
summaries for the rest). It is not exhaustive of the 2015-2026 KT
literature — in particular the two "per-skill Deep-IRT variant" entries
above were not read at full-text depth and should be verified before any
paper-facing claim leans on them. The core verdict (no paper combines
external-reference validation with a stability/monotonicity audit of an
IRT-style readout, and no paper runs a pre-registered null against a
provably-uninformed encoder) is corroborated across every model with
enough retrieved detail to check, plus one explicit third-party survey
statement that evaluation practice is lacking, so confidence in the
overall verdict is moderate-to-high even though individual model entries
vary in retrieval depth.
