# Thesis Overview

Scientific and theoretical scope. This document states what the thesis
claims and why. When a later decision is unclear, this document and the
two beside it (`Engineering_overview.md`, `Thesis_chapter.md`) take
precedence. The principles here are meant to hold. The specifics are
meant to be revised as results come in.

## The framework in one paragraph

We build a measurement framework that estimates a respondent's latent
ability from a history of responses and tracks how that ability changes
as evidence accumulates. The framework is modular. An encoder maps a
response history to a latent ability state, and a decoder maps that
state to observed responses under whatever response format applies,
including right-wrong, partial credit, and graded scores. The respondent
is interchangeable, so the same machinery applies to a human learner and
to a language model. The item bank is open, so new items, formats, and
domains attach to an existing scale rather than forcing a full
recalibration.

## What is being measured

Item response theory estimates a fixed trait. Knowledge tracing predicts
the next response. We estimate a third quantity, the trajectory of
improvement and its rate. For a human learner this is a learning curve,
and progression on one domain can be read from practice on a related
domain through shared latent structure. For a language model this is an
in-context adaptation curve, the speed at which performance rises as
context accumulates. Both are one object, a latent ability written as a
function of accumulated evidence, where the shape and slope carry the
information rather than a single correctness score.

## What is claimed

Two claims carry the thesis.

**A versatile measurement system.** The encoder and decoder are modular
and chosen by use. A stronger encoder replaces the current one through a
fixed interface. A new response format is handled by changing the
decoder. The latent ability state is the interface between the two, and
it remains stable across these substitutions and across growth of the
item bank.

**A model of adaptation dynamics.** The estimand is the rate and shape
of improvement, recovered for both human learners and language models,
and transferable across related content.

## What is not claimed

We do not claim a stronger predictor than current knowledge tracing
models, nor a better static calibration than standard item response
theory. The unification of deep sequence models with item response
theory as one encoder-decoder family is already established (Vie and
Kashima 2023). Architectural interchangeability is therefore a design
property, not the contribution.

## Avoiding the measurement trap

Judged by prediction accuracy, the framework competes inside two mature
fields at once and wins neither. Success is therefore defined on a
different axis, the stability of the latent scale under extension, the
transfer of estimates across respondents and content, and the faithful
recovery of an interpretable improvement trajectory. A single accuracy
number does not capture these properties, and they are where the
framework is distinct.

## Chapter 0, the groundwork

The framework begins from an earlier model (`ma-irt`, under review),
treated here as Chapter 0. On its own terms it falls short of a
standalone contribution, since as an item response model or a knowledge
tracing model it is incremental against established work. Its value is as
groundwork. It established that a deep encoder produces a latent ability
well behaved enough to treat as a psychometric scale, recovering ability,
discrimination, and step thresholds for ordinal responses and allowing
ability to vary over time. It also exposed the limitations that motivate
the rest of the thesis, a single fixed item bank, a single domain, and no
mechanism to extend the scale or to track the rate of change. A scale
that is not well behaved cannot be anchored, interpreted, or extended, so
this groundwork is the precondition for everything that follows. The
model is frozen, and the thesis builds beyond it rather than developing
it further.

## Positioning relative to prior work

Deep recovery of item response parameters from response sequences is
established (Yeung 2019; Wu et al. 2020; Tsutsumi et al. 2021; Vie and
Kashima 2023). These models are largely binary, single domain, and
provide no mechanism for extension.

Fixed-parameter linking that holds a scale stable as content grows is
classical (Kim and Cohen 1996; Kolen and Brennan 2004) and was recently
shown at scale for language-model benchmarks (Habba et al. 2026). That
demonstration is static, binary, and has no sequence model or selection
policy. It is the static limit of our framework, the case in which the
respondent does not change.

Policy-driven adaptive item selection exists (Li et al. 2025; BanditCAT
2024), but the policy operates on pre-calibrated fixed parameters, with
no joint adaptation of the measurement model and the policy.

Cross-domain knowledge tracing transfers between academically similar
subjects with alignable concept structure (AEGOT-CDKT 2024), not across
heterogeneous formats and respondents.

Improvement-rate estimation has precedent in learning-curve and growth
models (Corbett and Anderson 1995; Cen et al. 2006; Pavlik et al. 2009).
The open problem is the deep, modular, continual, respondent-agnostic
form, and the application to in-context adaptation in language models.

## Thesis arc

The thesis is a sequence of chapters, each able to stand as a paper.

**Chapter 0, frozen, the groundwork.** A deep ordinal and time-varying
recovery model that established the precondition, a usable and
interpretable scale, and exposed the limitations the framework addresses.

**Chapter A.** Anchored extension of a neural scale. New items and
formats attach to the existing scale without recalibration. A binary and
static setting is an acceptable controlled start, with the graded
setting as the closing target.

**Chapter B.** Respondent-agnostic transfer. Item parameters calibrated
from language-model responses are tested for prediction of human item
behavior, at K equal to three on the existing partial-credit corpus.

**Chapter C.** Interpretable multidimensional growth. New trait
dimensions are added with a loading mask, and existing dimensions keep
their meaning through anchoring. Progression on one domain is tracked
from practice on another.

**Chapter D.** Co-adaptive selection. A selection policy and the
measurement model adapt jointly on the continual scale.

The first paper realizes the entry point of this arc. Its plan is in
`Thesis_chapter.md`.

## Risks and limits

**Cold start.** With no item information, a new item is not identifiable
from a single response, since one observation confounds item difficulty
and respondent ability. Item-agnostic means no hand-engineered item
features, not zero information. Anchoring or a small number of responses
resolves this.

**Dimension growth.** Adding a trait dimension can rotate existing
dimensions in multidimensional models. A loading mask reduces this but
does not by itself preserve meaning, so existing loadings must be
anchored.

**Rate estimation.** A rate is a derivative and amplifies noise, so
trajectory estimates need sufficient response density per respondent,
which is in tension with the cold-start and agnostic goals.

**Interpretation for language models.** A rise in performance as context
grows may reflect in-context learning or prompt sensitivity.
Distinguishing genuine adaptation requires explicit controls.

**Data availability.** Logged response data is predominantly binary.
Graded data can be generated from language-model judging, and the
existing corpus carries partial-credit labels.

**Scope discipline.** A modular system on its own is infrastructure. The
contribution is the finding the trajectories reveal, not the
interchangeability.

## References

Citation years and venues are to be confirmed against primary sources
before submission.

- Cen, H., Koedinger, K., Junker, B. 2006. Learning Factors Analysis.
  Intelligent Tutoring Systems.
- Corbett, A., Anderson, J. 1995. Knowledge Tracing, Modeling the
  Acquisition of Procedural Knowledge. User Modeling and User-Adapted
  Interaction.
- Habba, E., Itzhak, I., Yehudai, A., Perlitz, Y., Bandel, E.,
  Shmueli-Scheuer, M., Choshen, L., Stanovsky, G. 2026. Growing Pains,
  Extensible and Efficient LLM Benchmarking via Fixed Parameter
  Calibration. arXiv 2604.12843.
- Kim, S.-H., Cohen, A. 1996. On the link function for fixed-parameter
  item calibration. (IRT linking and equating.)
- Kolen, M., Brennan, R. 2004. Test Equating, Scaling, and Linking.
  Springer.
- Li, J., Gibbons, R., Rockova, V. 2025. Deep CAT, reinforcement
  learning for computerized adaptive testing. arXiv 2502.19275.
- Muraki, E. 1992. A Generalized Partial Credit Model. Applied
  Psychological Measurement.
- Pavlik, P., Cen, H., Koedinger, K. 2009. Performance Factors Analysis.
  Artificial Intelligence in Education.
- Piech, C., et al. 2015. Deep Knowledge Tracing. NeurIPS.
- Samejima, F. 1969. Estimation of Latent Ability Using a Response
  Pattern of Graded Scores. Psychometrika Monograph.
- Tsutsumi, E., et al. 2021. Deep Item Response Theory with independent
  student and item networks. Educational Data Mining.
- Vie, J.-J., Kashima, H. 2023. Deep Knowledge Tracing as an implicit
  dynamic multidimensional item response theory model. arXiv 2309.12334.
- Wu, M., et al. 2020. Variational Item Response Theory for fast and
  accurate ability estimation. Educational Data Mining.
- Yeung, C.-K. 2019. Deep-IRT, making deep learning based knowledge
  tracing explainable using item response theory. Educational Data
  Mining. arXiv 1904.11738.
- Zhang, J., et al. 2017. Dynamic Key-Value Memory Networks for
  Knowledge Tracing. The Web Conference.
- AEGOT-CDKT. 2024. Adversarial and graph optimal transport for
  cross-domain knowledge tracing. World Wide Web Journal.
- AutoIRT. 2024. Calibrating item response theory models with automated
  machine learning. arXiv 2409.08823.
- BanditCAT. 2024. Bandit approach to adaptive testing for language
  assessment. arXiv 2410.21033.
