# HawkesKT (Wang et al., WSDM 2021): full read

Source: Chenyang Wang, Weizhi Ma, Min Zhang, Chuancheng Lv, Fengyuan Wan, Huijie Lin,
Taoran Tang, Yiqun Liu, Shaoping Ma. "Temporal Cross-Effects in Knowledge Tracing." WSDM
'21. DOI 10.1145/3437963.3441802. Author PDF obtained via Wayback Machine snapshot
(2024-04-30) of http://www.thuir.cn/group/~YQLiu/publications/WSDM2021Wang.pdf (the live
thuir.cn link now 404s). Saved locally at
`data/hawkeskt/WSDM2021Wang.pdf` (1,498,429 bytes, 9 pages, verified against the
archive's reported Content-Length). Code: github.com/THUwangcy/HawkesKT (not cloned in
this session; referenced only from the paper's own footnote, not independently
inspected). This note supersedes the prior [UNVERIFIED] characterization in
`avenue_map.md` ("sign validation is a qualitative visualization") with a close reading
of the actual text.

## 1. Task setup and notation

An interaction is a tuple x_i = (q_i, t_i, a_i): question, timestamp, correctness in
{0,1}. A skill map s(q_i) gives the KC of question q_i. KT predicts a_{n+1} given the
past sequence and (q_{n+1}, t_{n+1}) (Def 3.1, Sec 3.1).

## 2. Empirical motivation: conditional mutual information (CMI)

Before proposing the model, Section 3.3 defines CMI between the responses of a
pre-interaction and a post-interaction pair (x_i, x_j) drawn from the same student's
sequence, conditioned on a restriction c (e.g., "pre-interaction skill = s1, post
skill = s2"):

CMI(a_i; a_j) = sum_{a_i,a_j in {0,1}} P(a_i,a_j) log [P(a_i,a_j) / (P(a_i)P(a_j))]  (Eq. 1)

This is the standard mutual-information formula (Kullback-Leibler form), computed by
frequency counting over interaction pairs satisfying c. **Structural point: mutual
information as defined here is non-negative by construction** (it is a KL divergence
between the joint and product-of-marginals). CMI cannot express sign. It answers "how
much does knowing a_i shift the distribution of a_j" but has no notion of "in which
direction" (facilitation vs. interference).

Findings on ASSISTments 12-13 top-10 skills (Fig. 2, Fig. 3):
- CMI is largest on the diagonal (same-skill pairs), consistent with self-effects
  dominating.
- Off-diagonal CMI is nonzero and structured: skill groups {0,1,2} and {7,8,9} (see
  Table 1 skill names) show elevated CMI, "which makes sense because these skills are
  generally perceived to be related" (Sec 3.3, p.3-4).
- Conditioning further on log time-gap (Fig. 3), the *decay rate* of CMI differs by
  skill pair: pair (8,7) has large short-term CMI that decays fast; pair (8,2) has
  smaller CMI that decays slowly. This pairwise-distinct decay is the paper's
  namesake "temporal cross-effect" and is the sole empirical justification for giving
  the kernel function per-pair parameters rather than a single global decay rate.

This section validates *existence and heterogeneity* of cross-skill temporal
dependency. It does not and structurally cannot validate *sign*.

## 3. The HawkesKT model

### 3.1 Intensity function (Eq. 4)

lambda(x_i) = lambda_0^{x_i} + sum_{x_j in S_{t_i}} alpha_{x_j,x_i} * kappa_{x_j,x_i}(t_i - t_j)

Two components carry the cross-effect claim:
- **Mutual excitation alpha_{x_j,x_i}**: degree of cross-effect from history event x_j
  onto target skill of x_i.
- **Kernel kappa_{x_j,x_i}(.)**: adaptive per-pair temporal decay.

**Critical implementation detail, easy to miss on a skim**: lambda(x_i) is *not* used
as a point-process conditional intensity in the classical sense (which would require
lambda >= 0 and be fit by point-process MLE on inter-arrival times). Instead, Eq. 7
feeds lambda(x_i) straight into a sigmoid, y_hat_i = 1/(1+exp(-lambda(x_i))), and the
whole model is fit by ordinary cross-entropy on correctness labels (Eq. 11, plain
Adam, weight decay on the factor matrices only). So HawkesKT borrows the *functional
form* of a Hawkes intensity (base rate + sum of pairwise excitation-times-kernel
terms) as a parametric logit, not the Hawkes likelihood. This matters for sign: **there
is no non-negativity constraint anywhere in the architecture.** alpha and beta are
free real parameters (init N(0, 0.01^2), Sec 5.1.4), unconstrained by activation
function, clipping, or prior. A negative alpha_{x_j,x_i} is a first-class, freely
reachable model state, not an edge case the architecture fights.

### 3.2 Base intensity (Eq. 5)

lambda_0^{x_i} = lambda_0^{q_i} + lambda_0^{s(q_i)}: one scalar per question plus one
per skill, additive. This is the item-difficulty term; not part of the cross-effect
claim but relevant to A3's anchored-item-path constraint since it shows HawkesKT
already separates a per-question term from a per-skill term, similar in spirit to
anchoring but not identical (no held-fixed/frozen anchor items, both terms are
jointly learned).

### 3.3 Mutual excitation: is it per-KC-pair, and can it be negative?

Sec 4.2.2, first paragraph: alpha_{x_j,x_i} is indexed by the **(skill, response) pair
of the history event** (s(q_j), a_j) and the **target skill** s(q_i) — explicitly not
question-specific ("this will be too fine-grained to learn meaningful mutual
parameters," footnote 2, p.5). So the raw parameter is a matrix A in
R^{2|S| x |S|}: rows = (source skill, source correctness) joint index (size 2|S|),
columns = target skill (size |S|). This is per-directed-KC-pair, **and separately
parameterized by whether the source interaction was answered correctly or
incorrectly** — i.e., four distinct excitation values relate any two skills s1, s2:
alpha_{(s1,1),s2}, alpha_{(s1,0),s2}, alpha_{(s2,1),s1}, alpha_{(s2,0),s1}. This is a
richer signed object than a single scalar per KC pair; sign and correctness-condition
are both present in the raw parameterization. Confirmed answer to the assignment's
question: **yes, per-KC-pair, and yes, unconstrained (so negative values are
representable)** — but see Section 5 below on whether any negative value is ever
externally validated as a real inhibitory relationship.

### 3.4 Kernel function (Eq. 6)

kappa_{x_j,x_i}(t_i - t_j) = exp(-(1+beta_{x_j,x_i}) log(t_i - t_j)) = 1/(t_i-t_j)^{1+beta}

A power-law forgetting curve (log-time transform chosen because time gaps are
long-tailed, Sec 4.2.2). beta is per-pair, same (source skill+response) x (target
skill) indexing as alpha, also unconstrained in the base formulation.

### 3.5 Reparameterization: matrix factorization (Sec 4.3, Eqs 8-10)

Both A and B (the alpha and beta matrices, each in R^{2|S| x |S|}) are replaced by a
low-rank inner-product factorization: P_A in R^{2|S| x D}, Q_A in R^{|S| x D} (same
for B), with alpha_{x_j,x_i} = sum_d P_A[row,d] * Q_A[col,d]. Motivation given is
twofold: (1) raw pair counts are sparse relative to 2|S|^2 combinations, so most
entries of A/B would never be updated; (2) independent per-pair parameters cannot
generalize to unseen pairs. Parameter count drops from O(4|S|^2) to O(6|S|D) with
D << |S|. This is explicitly framed as collaborative filtering (cites Koren & Bell
2015). **Consequence for sign interpretation**: every alpha_{x_j,x_i} the model
actually reports is a rank-D inner product of two learned embeddings, not an
independently-fit scalar — so any individual "negative cross-effect" is a projection
through a shared low-dimensional space, exactly the kind of object the kt-mirt program
already worries about for shared-encoder passive mimicry (per the avenue map's
constraint (a) discussion). HawkesKT never tests whether this factorization preserves
or distorts sign fidelity relative to the unconstrained (unfactorized) matrix — the
\CF ablation (Sec 5.4) tests only whether removing the factorization changes
*predictive AUC*, not whether it changes which pairs are called positive vs.
negative.

## 4. Parameter learning

Plain cross-entropy on next-interaction correctness (Eq. 11), Adam, weight decay only
on the factor matrices, embedding/hidden size 64 across models for fair comparison
(Sec 5.1.4). No auxiliary loss term, regularizer, or constraint targets the alpha or
beta matrices specifically toward interpretability or sign correctness; interpretability
is asserted as a byproduct of the architecture, not optimized for or checked during
training.

## 5. How the signed/cross effects were actually validated

This is the crux of the read. There are four distinct pieces of evidence in the paper
that could be mistaken for "validating signed cross-effects." None of them validates
the raw signed alpha matrix against ground truth for interference (negative transfer).

**(a) CMI (Sec 3.3, pre-model).** As shown in Section 2 above: motivational, run
*before* the model exists, on raw data, and structurally non-negative. It shows
cross-effects exist and are pairwise heterogeneous in magnitude and decay rate. It
never touches alpha and cannot represent sign. This is the paper's only *quantitative*
real-data cross-effect analysis that isn't downstream of the fitted model, and it is
unsigned.

**(b) Ablation study (Sec 5.4, Fig. 4).** Three variants: \Temporal (drop the kernel
entirely, Eq. reduces to lambda = lambda_0 + sum alpha), \Cross (keep a kernel but
force beta global instead of per-pair), \CF (skip the matrix factorization,
optimize A, B directly). All three are evaluated purely on held-out AUC. Findings:
\Temporal causes the largest AUC drop (temporal information matters most); \Cross
is consistently worse than full HawkesKT on all three datasets (per-pair decay adds
predictive value beyond a global decay rate); \CF causes a moderate AUC loss (the
low-rank factorization itself helps, not just parameter-count reduction). **This
validates that the cross-effect machinery improves prediction. It says nothing about
whether the specific fitted alpha signs are correct or meaningful** — a model can gain
AUC from a signed per-pair mechanism whose individual signs are partly or wholly
fitting artifacts, exactly the "stable-and-wrong" failure mode the kt-mirt program is
built to catch. HawkesKT has no null-model or permutation control on the alpha matrix
itself (no shuffled-skill-identity control, no comparison to an uninformed/randomly
initialized-and-frozen alpha, no seed-clustering report on which pairs are
sign-stable across reruns).

**(c) Prerequisite score and its NDCG evaluation (Sec 4.5, 5.5) — the closest thing to
a quantitative sign-adjacent test, but not a signed-transfer test.** The paper
defines (Eq. 12):

r(s_i) = softmax(alpha_{(s_i,1),s}) [dot] softmax(alpha_{(s,0),s_i})

i.e., a score for "how likely is each other skill s a prerequisite of s_i," built from
two slices of the raw alpha matrix (excitation from *correctly* answering s_i onto
other skills; excitation from *incorrectly* answering other skills onto s_i), each
passed through a softmax over the skill axis. **Two things collapse the sign
information here.** First, softmax outputs are strictly positive and sum to 1 by
construction — any negative raw alpha entries are remapped into the positive simplex,
so the derived prerequisite score cannot itself be negative regardless of the sign of
the underlying alpha. Second, the score is explicitly interpreted only as a magnitude
("how likely to be a prerequisite"), never as a bipolar facilitation/interference axis.
Validation of r(s_i):
  - *Qualitative*: Fig. 5, a case study on ASSISTments 12-13 top-10 skills, hand
    interpreted by the authors as matching intuitive prerequisite structure (e.g.,
    Order of Operations depending on addition/subtraction and multiplication/division).
  - *Quantitative*: an annotation experiment (Sec 5.5, second paragraph) — top-20
    frequency skills, 3 experts label pairwise binary "helpfulness," inter-annotator
    kappa = 0.52 (moderate agreement by conventional Landis-Koch bands, not itself
    characterized that way in the paper), averaged annotations used as ground truth,
    ranked lists from r(s_i) evaluated against them by NDCG = 0.8267.

This NDCG=0.8267 figure is real and is the paper's only externally-grounded
quantitative number for the interpretability claim. But note precisely what it
certifies: a *ranking of positive-only, binary "does this skill help" judgments*,
derived from a softmax-collapsed, response-conditioned slice of the raw matrix. It
does not certify: the sign of any individual raw alpha_{x_j,x_i} entry; the existence
of a real negative/interfering relationship; or any property of the beta (decay)
matrix. There is no annotation task in the paper asking experts to label negative
transfer or interference between skill pairs, and no evaluation of the model against
such labels.

**(d) Relation-graph visualization (Fig. 5).** Purely qualitative, drawn from the same
already-positive r(s_i) scores as (c), arrows scaled by thickness for score magnitude.
No negative/inhibitory edges are shown or discussed anywhere in the figure or its
caption.

**Net verdict on validation**: the avenue map's prior flag ("sign validation is a
qualitative visualization") undersells what HawkesKT actually did (there is a real
quantitative NDCG/kappa annotation experiment) but overstates what it validates for
*this program's* purposes (it is a positive-only "helpfulness"/prerequisite ranking
test, run on a softmax-collapsed derived score, not a test of the raw signed
excitation parameter, and never touches negative/interference cases at all).

## 6. Experimental results (context, not the paper's novelty claim)

Three datasets after filtering to users with >=5 interactions, first 50 interactions
kept per user (Table 2): ASSISTments 09-10 (3.7k students, 111 skills, 110.2k
interactions), ASSISTments 12-13 (25.3k students, 245 skills, 879.5k interactions),
slepemapy.cz (81.7k students, 1473 skills, 2877.5k interactions, geography place
recall). 5-fold CV by user, AUC metric, early stopping on a 10%-of-train validation
split (Sec 5.1.2). Baselines: IRT, DKT, SAKT (no temporal term); DKT-Forgetting, KTM,
AKT-R (temporal-aware). HawkesKT AUC: 0.7629 / 0.7676 / 0.7500 on the three datasets
respectively (Table 3), beating all baselines, most improvements significant at
p<0.01 against each baseline; the smallest margin is on slepemapy.cz, which the
authors attribute to geography's cross-skill structure being weaker than mathematics'
(Sec 5.2). HawkesKT is also markedly cheaper: 0.5s/epoch and 74.8k params on
ASSISTments 09-10 vs. DKT's 0.8s/57.4k, KTM's 7.5s/1760.7k, AKT-R's 2.3s/160.7k
(Table 4).

## 7. What a certified signed-transfer claim would add beyond HawkesKT

1. **A ground-truth or matched-null test on the raw signed parameter itself.**
   HawkesKT never asks "is this specific alpha_{x_j,x_i} < 0 real or noise." A3-style
   certification (injected-edge synthetic recovery with known signs, uninformed-encoder
   null, KC-identity permutation, seed clustering) is a strictly higher validation bar
   than anything in this paper, applied to an object (the raw signed matrix) HawkesKT
   only ever inspects after softmax-collapsing it to a positive ranking.
2. **An actual negative-transfer/interference test case.** Every validation datum in
   the paper (CMI, ablation AUC, NDCG-against-expert-"helpfulness") is either sign-blind
   or positive-only by construction. A program that surfaces and externally confirms
   even one clearly negative (interfering) skill-pair coefficient would be doing
   something this paper structurally cannot: its own prerequisite-score construction
   (softmax over both slices) makes it impossible for r(s_i) to express interference,
   and no other quantitative check in the paper touches sign.
3. **Robustness of sign under the collaborative-filtering factorization.** The \CF
   ablation checks AUC, not sign stability. A program checking whether the low-rank
   embedding factorization preserves, distorts, or fabricates specific pairwise signs
   relative to an unfactorized fit (or across seeds) would close a real gap.
4. **Decoder-side, anchored-item integration into an already prediction-trained
   core**, rather than a from-scratch model whose only per-question term is a
   jointly-learned scalar difficulty. HawkesKT's base intensity separates
   question and skill difficulty but does not freeze/anchor items the way constraint
   (c) requires before trusting a magnitude claim.

## 8. Does the mechanism suit A3?

Yes, architecturally, with caveats. The (alpha, beta, exponential/power-law kernel)
construction is exactly the "HawkesKT-shaped exponential-decay kernel" A3 already
names, and three properties make it a good fit for a decoder-side add-on:
- It is cheap (low-rank factorized, O(6|S|D) parameters) and already demonstrated to
  train fast (Table 4) and scale to |S| in the hundreds to low thousands (slepemapy.cz,
  1473 skills).
- It is architecturally unconstrained toward sign already — no code-level positivity
  clamp needs to be removed or added; alpha is free to fit negative the moment the
  loss wants it to, which is the necessary (not sufficient) precondition for a
  meaningful signed-excitation audit.
- The (source skill, source correctness) x (target skill) indexing already gives a
  natural per-KC-pair, per-direction-of-evidence signed object, richer than a single
  scalar per pair, which maps cleanly onto "does observing correct/incorrect evidence
  on KC A shift the readout on KC B, and in which direction."

What HawkesKT does *not* supply, and A3 must build from scratch: any certification
tooling at all. There is no null model, no permutation control, no synthetic
ground-truth recovery test, no negative-transfer-specific validation, and the one
external check that exists (NDCG against expert annotation) evaluates a
sign-collapsed derived quantity on a different dataset scale (top-20 skills) than
would be needed for a real per-KC-pair audit. The paper's implicit message for A3 is
reassuring on architecture and cautionary on validation: the mechanism is a
reasonable, cheap, off-the-shelf building block; the burden of proof for "the signs
are real" is entirely unaddressed by prior art and would need to be built by this
program from the ground up, which is consistent with A3 being scoped as a cheap
probe whose main deliverable is the audit itself, not the mechanism.
