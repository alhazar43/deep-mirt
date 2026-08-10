# Named nearest analogs (verified)

Verification sweep of 2026-08-10. Every entry below was resolved to a live
arXiv id or venue page and its abstract read. Our position against all of
them is the same and none of them takes it. We certify each detector on
synthetic twins with pre-registered bars and confound arms before any
real-data claim, and we publish the measured boundaries and refusals.
None of these works certifies its readouts before use or reports where
its claims stop holding.

## 1. Growing Pains (arXiv 2604.12843) -- VERIFIED, characterization corrected

> Habba, E., Itzhak, I., Yehudai, A., Perlitz, Y., Bandel, E.,
> Shmueli-Scheuer, M., Choshen, L., Stanovsky, G. (2026). Growing Pains:
> Extensible and Efficient LLM Benchmarking Via Fixed Parameter
> Calibration. arXiv:2604.12843.

Claims that a multidimensional IRT framework with fixed-parameter anchor
calibration lets new benchmarks join an existing suite while keeping
scores comparable over time, predicting full evaluation performance
within 2-3 points from about 100 anchor items and preserving model
rankings (Spearman >= 0.9) across 400+ models.

Note. This paper is about LLM benchmark extensibility, not learner-model
growth versus noise. It is still the nearest analog for our growth arm
because anchored fixed-parameter calibration is exactly the mechanism
that makes ability change measurable against a stable scale, which is
what growth-beyond-noise requires. It does not certify the anchoring
procedure on synthetic ground truth or state where the comparability
claim breaks; we do both before claiming growth.

## 2. PSI-KT -- VERIFIED

> Zhou, H., Bamler, R., Wu, C. M., Tejero-Cantero, A. (2024).
> Predictive, scalable and interpretable knowledge tracing on structured
> domains. ICLR 2024. arXiv:2403.13179.

Claims a hierarchical Bayesian generative KT model whose latent learner
traits and inferred prerequisite graph are interpretable by design while
matching or beating deep KT on multi-step prediction and scaling via
amortized continual inference.

We differ in that PSI-KT asserts interpretability by construction; we
treat every structural readout (including cross-KC influence) as a claim
that must first pass a certified detector on synthetic twins with
confound arms, and we report the regimes where the readout fails.

## 3. GKT -- VERIFIED

> Nakagawa, H., Iwasawa, Y., Matsuo, Y. (2019). Graph-based knowledge
> tracing: modeling student proficiency using graph neural network.
> IEEE/WIC/ACM Web Intelligence 2019, pp. 156-163.

Claims that casting KT as node-level classification on a concept graph
with a GNN improves AUC over prior deep KT and yields more interpretable
concept-state traces.

We differ in that GKT's interpretability claim rests on face validity of
the learned graph; we require the influence readout to recover known
structure on synthetic twins before any real-data interpretation, and we
publish the recovery boundary.

## 4. GIKT -- VERIFIED

> Yang, Y., Shen, J., Qu, Y., Liu, Y., Wang, K., Zhu, Y., Zhang, W.,
> Yu, Y. (2020). GIKT: A Graph-based Interaction Model for Knowledge
> Tracing. ECML-PKDD 2020, LNCS 12457. Springer.

Claims that propagating question-skill relations through a GCN and
modeling question-level interactions lifts prediction AUC by 2-6 points
absolute over prior KT models on three datasets.

We differ in that GIKT is a pure prediction contribution with no
measurement claim; we target the validity of the structural readouts
themselves, which prediction gains alone cannot establish.

## 5. LTKT -- VERIFIED

> Xu, J., Tang, R., Lv, P., Yu, M., Yu, G., Chen, E. (2025). LTKT:
> Knowledge Tracing Based on Positive and Negative Learning Transfers.
> Tsinghua Science and Technology 31(3). DOI 10.26599/TST.2024.9010201.

Claims the first KT model to use both positive and negative transfer
relations among concepts, via a learning-transfer graph with direct and
transfer effect components, improving prediction over positive-only
methods.

We differ in that LTKT builds signed cross-concept influence into the
architecture and validates it by prediction accuracy; we ask whether
signed influence is recoverable at all, certify the detector against
confound arms that mimic transfer without it, and refuse the claim where
the bar is not met.

## 6. HawkesKT -- VERIFIED

> Wang, C., Ma, W., Zhang, M., Lv, C., Wan, F., Lin, H., Tang, T.,
> Liu, Y., Ma, S. (2021). Temporal Cross-Effects in Knowledge Tracing.
> WSDM 2021. ACM.

Claims the first use of a Hawkes process in KT, modeling fine-grained
temporal cross-skill effects through mutual excitation and adaptive
decay kernels, improving prediction over time-agnostic KT.

We differ in that HawkesKT reads its excitation parameters as
cross-skill effects without testing whether the estimator can recover
known effects; our cross-KC influence claims are gated by exactly that
recovery test, with the failure regimes reported.

## 7. Deep-IRT (Yeung) -- VERIFIED

> Yeung, C.-K. (2019). Deep-IRT: Make Deep Learning Based Knowledge
> Tracing Explainable Using Item Response Theory. EDM 2019.
> arXiv:1904.11738.

Claims that attaching an IRT output layer to a DKVMN memory network
yields per-skill ability and item difficulty estimates, making deep KT
explainable at a small cost in predictive power.

We differ in that Deep-IRT treats the IRT-flavored readouts as
explainable by construction; our audit line shows such readouts can be
stable and wrong, so we certify them on synthetic ground truth before
trusting them and we report where they cannot be trusted.

## 8. VIBO -- VERIFIED

> Wu, M., Davis, R. L., Domingue, B. W., Piech, C., Goodman, N. (2020).
> Variational Item Response Theory: Fast, Accurate, and Expressive.
> EDM 2020. arXiv:2002.00276.

Claims a variational lower bound with amortized inference networks for
person and item parameters that makes Bayesian IRT fast and scalable,
including with neural response functions.

We differ in that VIBO is an inference-efficiency contribution that
assumes the IRT readout is meaningful once fit; we test that assumption
directly with certified detectors and confound arms before real-data
interpretation.

## 9. beta4-IRT -- VERIFIED

> Ferreira-Junior, M., Reinaldo, J. T. S., Silva Filho, T. M., Lima
> Neto, E. A., Prudencio, R. B. C. (2023). beta4-IRT: A New beta3-IRT
> with Enhanced Discrimination Estimation. arXiv:2303.17731.

Claims a fix to beta3-IRT's symmetry-driven non-identifiability, using
gradient descent with link functions to recover discrimination
parameters correctly, in the line of IRT models applied to machine
learning classifiers as respondents.

We differ in that beta4-IRT repairs one named identifiability failure
found after the fact; our protocol screens for such failures up front,
on synthetic twins with pre-registered bars, before any substantive
claim rests on the parameters.
