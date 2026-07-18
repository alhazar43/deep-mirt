# PSI-KT: Predictive, Scalable and Interpretable Knowledge Tracing on Structured Domains

Zhou, Bamler, Wu & Tejero-Cantero. ICLR 2024 (published, not just submitted — the
PDF header reads "Published as a conference paper at ICLR 2024").
arXiv: [2403.13179](https://arxiv.org/abs/2403.13179) (v1, 19 Mar 2024).
OpenReview forum: https://openreview.net/forum?id=NgaLU2fP5D.
Code: https://github.com/mlcolab/psi-kt (license: **GNU AGPL-3.0**, confirmed
by fetching the repo's `LICENSE` file directly — this is a copyleft license
with network-use provisions, the strictest common open-source license; it
would require any KT tool built on it, including one served over a network,
to release its own source).

All institutional affiliation is University of Tübingen / Cluster of
Excellence Machine Learning / Tübingen AI Center / IMPRS-IS. First two
authors (Zhou, Bamler) are marked equal contribution.

**Access note.** OpenReview's forum page and its API (`api2.openreview.net`)
both refused automated access during this research pass, returning an
explicit `ChallengeRequiredError` (HTTP 403, Cloudflare-style bot
verification) rather than review content. I could not retrieve the verbatim
reviewer comments, scores, or the meta-review. The "reviewer criticisms"
section below is therefore reconstructed only from the paper's own stated
limitations (Discussion and Ethics Statement sections), not from actual
referee text — flagged explicitly, not glossed over.

---

## 1. The generative model

PSI-KT is a **three-level hierarchical state-space model (SSM)**, per-learner,
with one global (shared) structural component. Notation: learner $n$, KC
(knowledge component) $k$, time $t$.

**Level 2 — cognitive traits** $\boldsymbol{s}_n(t)$: a Gaussian-transition
Markov process, learner-specific, evolves over time (this is new relative to
most KT models, which give learners either no latent trait or a *static*
one).

**Level 1 — knowledge states** $\boldsymbol{z}_n(t)$: one continuous scalar
state per (learner, KC) pair, conditioned on the traits.

**Level 0 — observed performance** $y_n(t) \in \{0,1\}$: Bernoulli emission
from the state of the KC being practiced at time $t$.

### Transition operator (how per-KC states evolve)

Each isolated KC's knowledge state follows a **mean-reverting Ornstein-
Uhlenbeck (OU) process** (continuous-time stochastic differential equation,
Eq. 4 in the paper):

$$dz_k = \gamma\,(\mu_k - z_k)\,dt + \sigma\,dW_t$$

i.e. the state decays/reverts toward a KC-specific long-term mean $\mu_k$ at
rate $\gamma$ (a learner trait), with volatility $\sigma$ (another learner
trait) driving Wiener-process noise. This nests the standard exponential
forgetting curve (Ebbinghaus 1885) as the noiseless, $\mu_k \to$ baseline
case, and the authors are explicit that they chose OU specifically because
it is **analytically marginalizable over irregular time gaps**: integrating
the SDE over an elapsed interval $\Delta t$ gives a closed-form Gaussian
transition kernel (mean and variance both closed-form, citing Särkkä &
Solin 2019), so no numerical ODE/SDE solver is needed at inference time —
this is a big part of the "scalable" claim.

**How the prerequisite graph enters the transition**: it does not act on the
per-step dynamics directly, but on the long-term mean $\mu_k$ that the OU
process reverts to (Eq. 5):

$$\mu_{n,k}(t) = \mu_k^{(0)} + a_n \sum_{k'} w_{k'\to k}\, z_{n,k'}(t)$$

i.e. the reversion target for KC $k$ is a KC-specific baseline plus a
weighted sum of the learner's *current* mastery of prerequisite KCs $k'$,
scaled by a learner-specific **transfer-ability trait** $a_n$. This is the
mechanism that produces cross-KC transfer in the model: doing well on a
prerequisite raises the ceiling/target the dependent KC's state drifts
toward. Structural coupling is thus injected only through the mean-reversion
target, which the authors say "justifies the conditional independence
assumed in Eq. 2" (the states remain first-order Markov per KC; the graph's
influence is folded entirely into the time-varying target).

### Shared prerequisite graph parameterization

The graph $\mathcal{A}$ (edge weights $w_{k\to k'}$) is **learner-independent
and shared/pooled across all learners** ("in the spirit of collaborative
filtering ... we can pool evidence from all learners to estimate them").
To avoid $O(K^2)$ scaling in the number of KCs $K$, edges are not modeled
directly but **factored through low-dimensional KC embedding vectors**
$e_k \in \mathbb{R}^{d}$ (paper uses this to avoid quadratic blow-up for
Junyi15's 722 KCs). The weight matrix uses a factorization borrowed from
Lippe et al. 2021 (NRI-style / DAG-learning literature) into a probability of
edge existence and a definite-directionality term, combined skew-
symmetrically so that mutual prerequisites ($k\to k'$ **and** $k'\to k$
both strong) are structurally discouraged — "no mutual prerequisites" is
enforced by construction, not learned. The graph is a point estimate (not a
distribution) obtained by direct optimization of the marginal likelihood,
not sampled per learner.

### What is per-learner vs. shared

| Component | Scope |
|---|---|
| Knowledge states $z_{n,k}(t)$ | per learner × per KC, latent, inferred |
| Cognitive traits $s_n(t) = (\gamma_n,\mu_n,\sigma_n,a_n)$ | per learner, 4-dimensional, **time-varying** |
| Prerequisite graph $\mathcal{A}$ / KC embeddings $e_k$ | shared across all learners, global parameters |
| Transition/generative parameters $\theta$ (OU coefficients, priors) | shared/global, point-estimated by (variational) EM |
| Inference-network weights $\phi$ | shared/global, amortized (see §2) |

The four traits have explicit interpretive labels the paper assigns by
construction, not just post hoc: $\gamma$ = forgetting rate, $\mu$ (via its
role in Eq. 5) = long-term memory consolidation / expected performance on
novel KCs, $\sigma$ = knowledge volatility, $a$ = transfer ability from
prerequisites.

---

## 2. Inference: what is amortized vs. directly fitted

Uses **variational inference / variational EM** (Dempster et al. 1977; Beal
& Ghahramani 2003), mean-field factorized posterior over the two latent
levels ($q(\boldsymbol z)q(\boldsymbol s)$).

- **Point-estimated / directly fitted**: generative parameters $\theta$ —
  the KC-graph parameters, the OU transition coefficients, and the initial-
  prior parameters. These are estimated from *all* learners' data jointly
  (M-step of variational EM), i.e., not amortized, not per-learner.
- **Amortized via a neural inference network** $\phi$: the variational
  parameters of $q(\boldsymbol z_n, \boldsymbol s_n)$ for each learner,
  produced by an LSTM-based encoder (see §3) that maps a learner's
  interaction history to the parameters of the approximate posterior,
  reparameterized (Kingma & Welling 2014) for the continuous states and a
  Gumbel-Softmax trick (Jang et al. 2016) for a **mixture-of-Gaussians**
  posterior over traits (needed because a single learner's traits are
  believed to be genuinely multimodal across a diverse population — a plain
  unimodal Gaussian VAE posterior was judged too restrictive).
- **Continual-learning inference**: rather than re-running full VI on the
  growing history, the current posterior at time $t$ is turned into the
  *prior* for time $t+1$ (variational continual learning / VCL, Nguyen et
  al. 2017), and a single new interaction updates the posterior directly by
  maximizing a modified ELBO (Eq. 9-10) — this is what gives the
  scalability/low-retraining-cost claim in §4.2, not the amortized network
  itself (the network is reused as-is; only the per-learner variational
  parameters update).

Prediction (§3.3) analytically convolves the current variational
distributions over $z$ and $s$ forward through the OU/trait transition
kernels (no sampling needed for the forward step because both are Gaussian),
then draws $z$ and applies the Bernoulli emission; multi-step prediction
repeats this without conditioning on previously *predicted* outcomes (i.e.
it's a genuine open-loop forecast, not teacher-forced).

---

## 3. Interpretability claims and how they were validated

The paper is explicit that "diverse approaches to interpretability exist ...
a comprehensive evaluation framework is still lacking" and positions itself
as also contributing that framework, not just the model. Two objects are
claimed interpretable: (A) learner traits, (B) the prerequisite graph.

### (A) Learner traits — four validation axes, only the 4th is "meaning"

1. **Specificity** — traits should identify *which* learner they came from.
   Quantified as mutual information $I(\text{trait representation}; \text{learner
   ID})$, estimated via fitted covariance matrices (Eq. 24) across 1,000
   learners × 50 interactions each, with a held-out 20% learner validation
   split (Table 3, Appendix A.6.1). PSI-KT is *competitive but not superior*
   here (e.g., Junyi15: baseline 13.5 vs PSI-KT 14.4, higher is better — a
   narrow win; Assist12: baseline 8.8 vs PSI-KT 8.4, a narrow **loss**) —
   despite using only 4 dimensions vs. baselines' 16, which the authors
   frame as evidence of efficiency, not dominance.
2. **Consistency** — traits inferred from *different, non-overlapping
   subsets* of a learner's interaction history (split by matched average
   presentation time, not sequential order, to reduce time-of-training
   confounds) should agree with traits inferred from the full history.
   Quantified as a KL-style divergence metric (Eq. 25) between full-learner
   and sub-learner posteriors; PSI-KT wins clearly on all three datasets
   (Table 3; e.g. Assist12: baseline 12.2 vs PSI-KT 7.4, lower is better).
3. **Disentanglement** — each of the 4 trait dimensions should be
   individually informative about learner identity, not just jointly.
   Measured (Eq. 26, Appendix A.6.3) via the discrepancy between full-
   covariance and diagonal-covariance conditional entropy, following
   Kim & Mnih 2018 but relaxing their independence assumption. PSI-KT wins
   by a wide margin on all three datasets (e.g. Assist17: baseline 0.6 vs
   PSI-KT 8.4).
4. **Operational interpretability** — this is the closest thing to a
   semantic-grounding check: does each trait dimension *predict a specific,
   independently-observable behavioral signature*? Concretely: (i) the
   forgetting-rate trait $\gamma_n$, when used to rescale the observed
   time-since-last-interaction as $\gamma_n \Delta t$, produces a clean
   exponential decay of one-step performance drop vs. this rescaled
   interval — a relationship invisible in the raw (unscaled) data because
   different learners forget at different rates (Fig. 4, top row); (ii) the
   prerequisite-mastery aggregate $\mu_{n,k}$ (Eq. 5) predicts a novel-KC's
   initial performance better than any single dimension of any baseline's
   learned embedding (Fig. 4, bottom row). Both relationships are tested
   with mixed-effects linear regression against **held-out test-set**
   behavioral data (not training data — Appendix A.6.4 is explicit that this
   step "goes beyond a simple sanity check ... so we use the testing
   data"), with regression coefficients and p-values reported (Table 4):
   e.g. performance-difference regression on Assist17, best baseline
   coefficient −0.03 (p=.30, not significant) vs PSI-KT 0.56 (p<.001).
   This significance-testing-against-unseen-behavior step is the paper's
   strongest interpretability evidence; the earlier three (specificity/
   consistency/disentanglement) are information-theoretic self-consistency
   checks, not checks against an external ground truth.

### (B) Prerequisite graph — validated two ways, one against human labels

1. **Alignment with human-annotated ground truth**, on Junyi15 only (the
   only dataset with such annotations: 553 expert-identified prerequisite
   edges from 3 teachers, plus 1,954 crowd-sourced pairwise ratings from 51
   graduates on a 0-9 scale). Metrics: mean reciprocal rank (MRR) of expert
   edges within the model's globally-sorted inferred-probability list,
   negative log-likelihood of inferred edge weight against a Gaussian
   fit to rescaled crowd ratings, and Jaccard similarity of edge sets
   against expert and crowd-thresholded (rating > 5) graphs. PSI-KT wins on
   all four sub-metrics vs. the best baseline (Table 5 left), though the
   absolute MRR gap is small (.0086 vs .0082) — a very sparse-graph,
   large-candidate-set regime (722 KCs), so both numbers are near the floor.
2. **Causal support from behavioral data**, on all three datasets — this is
   the more independent check. For each candidate KC pair $(k, k')$, the
   authors compute a Bayesian "causal support" score (Griffiths & Tenenbaum
   2009 formalism, Eq. 11, detailed in Appendix A.7.3) from *consecutive*
   interaction pairs in the held-out behavioral data, comparing likelihood
   under a "causal edge exists" graph vs. a null graph with only a shared
   background-ability cause. They then regress this causal-support score
   against the model's inferred edge probability. PSI-KT's regression
   coefficients are significant and positive on all three datasets (e.g.
   Junyi15: 0.97, p<.001); the paper states "all baseline models either
   lack significance or negatively predict causal support" (Table 5 right,
   Appendix Fig. 12) — a claim of directional failure for competitors, not
   just lower magnitude.

**Ablation study** (Appendix A.8, Table 16) isolates which architectural
piece drives the gains: removing the graph, removing individualized traits,
and removing time-varying (dynamic) traits each *reduces* accuracy, but by
dataset-dependent amounts (e.g. Assist17 loses more from removing dynamic
traits [−.09] than from removing the graph [−.04]; Junyi15 loses more from
the graph [−.07] than from dynamic traits [−.03]) — the paper reads this as
evidence the components are complementary and their relative importance is
domain-dependent, not evidence any one component is dispensable.

---

## 4. Datasets and preprocessing

Three datasets, all pre-college mathematics, chosen from a larger candidate
pool by two explicit inclusion criteria (Appendix A.3.2):

1. Interactions must carry **identifiable KC labels** (not just an opaque
   assignment/task ID covering several KCs) — this excluded Statics2011.
2. Interactions must be **timestamped at seconds-or-finer resolution** —
   this excluded Assistments2009/2015 (no timestamps) and Junyi2020 (only
   15-minute resolution, "too coarse").

| | Assist12 | Assist17 | Junyi15 |
|---|---|---|---|
| Source | ASSISTments (Worcester Polytechnic Institute), US grades 4-12, mostly MA middle school | same platform, 2017 release | Junyi Academy, non-profit Chinese platform |
| Learners (all / ≥50 interactions) | 46,674 / 12,443 | 1,709 / 1,697 | 247,606 / 77,655 |
| KCs | 265 (263 in ≥50 subset) | 102 | 722 (721) |
| Interactions (all) | 6,123,270 | 942,816 | 25,925,992 |
| Ground-truth KC structure | none | none | 553 expert edges + 1,954 crowd-rated pairs (unique among the three) |

Preprocessing/splitting for the main prediction experiments: **first 10
interactions per learner for training, next 10 for open-loop test**
(explicitly not the field-standard "predict one step given all history";
the authors argue 10-step open-loop forecasting from minimal data better
matches an ITS cold-start/equipoise setting), 20% of learners held out for
validation, Adam optimizer, batch size 32, learning-rate decay every 200
epochs.

**Self-stated dataset limitation** (Appendix A.3.2, "Limitations", and
echoed in the Ethics Statement): the domain restriction to pre-college
mathematics is acknowledged as a scope limit, and the authors flag that
their KC-identifiability requirement excludes the common real-world case
where one assignment tests multiple KCs simultaneously — they explicitly
frame extending to that setting as future work, citing Wang et al. 2020 for
the multi-KC-per-item complication.

---

## 5. Metrics vs. baselines

**Metrics**: prediction accuracy (and F1, in the appendix tables) on the
open-loop 10-step-ahead forecast, evaluated within-learner (train and test
splits from the same learner's history) and between-learner (fine-tuned or
zero-shot on unseen learners, Table 2). Not AUC-based; the main text does
not report AUC at all despite this being near-universal in the KT literature
(the paper's own related-work section cites `dkt`/`akt`/etc. papers that
mostly use AUC) — worth noting as a metric-choice divergence from the field
norm if kt-mirt wants apples-to-apples comparison later.

**Baselines** (8 total, Table 6 / Appendix A.3.1), spanning psychological
regression models through deep sequence and graph models:
- `hlr` (Half-Life Regression, Settles & Meeder 2016, the Duolingo model) —
  count-based logistic regression on correct/incorrect/total interaction
  counts, no KC structure, no per-learner state.
- `ppe` (Predictive Performance Equation, Walsh et al. 2018) — elapsed-time-
  weighted activation function with separate learning/forgetting rate
  terms.
- `dkt` (Piech et al. 2015) — the original Deep Knowledge Tracing LSTM.
- `dktf` (Nagatani et al. 2019) — `dkt` plus explicit interval/repetition-
  count input features.
- `hkt` (Wang et al. 2021) — models cross-KC structural influence via a
  multivariate Hawkes process; the paper calls this "the most similar model
  to our psi-kt" but notes it lacks any learner-specific representation.
- `akt` (Ghosh et al. 2020) — transformer/self-attention KT, structure
  captured implicitly through attention weights, not an explicit graph.
- `gkt` (Nakagawa et al. 2019) — explicit graph neural network over KCs, but
  the graph is *undirected* and *learner-independent-only* (no prerequisite
  directionality).
- `qikt` (Chen et al. 2023) — an interpretability-oriented deep model with
  three separately-supervised 1-D-interpretable components (acquisition,
  mastery, problem-solving ability).

**Headline accuracy numbers** (within-learner, Table 2; best-in-row bolded
in the original): PSI-KT is at or near the top on Assist12 (.68 vs next-best
`akt` .67) and Assist17 (.63, tied with `qikt`), but on Junyi15 — the
largest-cohort dataset (≥60k learners) — deep baselines close the gap or
edge ahead (`qikt` .81 with fine-tuning vs. PSI-KT .80). The Discussion
section states this directly as a limitation: "psi-kt has remarkable
predictive performance when trained on small cohorts whereas baselines
require training data from at least 60k learners to reach similar
performance. An open question ... is how to combine psi-kt's unique
continual learning and interpretability properties with performance that
grows beyond this extreme regime" — i.e., the authors themselves flag that
their advantage is a small-data/cold-start regime advantage, not a
uniform one.

---

## 6. Continual-learning / scalability angle

Separate from standard batch retraining, §4.2 (Appendix A.5.3) simulates a
realistic ITS deployment: train on 10 interactions × 100 learners, then feed
one new interaction per learner at a time repeatedly, comparing **cumulative
training time** and **retained prediction accuracy** after each increment
across models (Figure 3). The claim is PSI-KT achieves the best time-vs-
accuracy trade-off — cheapest to update, while degrading least. Mechanism:
the VCL scheme in §3.2.2 (posterior-becomes-prior) avoids recomputing a full
ELBO over the growing history; only a per-learner variational-parameter
update is needed per new datum, using the *shared* amortized inference
network (unchanged) to process only the new interaction rather than the
full sequence. This is the paper's "scalable" pillar and is architecturally
distinct from the interpretability pillar — the two didn't have to travel
together, but the paper's headline claim is that this particular design
(hierarchical SSM + amortization + VCL) buys both at once.

---

## 7. Stated limitations (paper's own Discussion + Ethics Statement)

Direct from §5 (Discussion) and the Ethics Statement, not inferred:

- Performance advantage concentrated in the small-cohort regime; unclear how
  to retain interpretability/continual-learning benefits while scaling
  accuracy competitiveness into the large-cohort regime (Junyi15, ≥60k
  learners) where deep baselines catch up or win.
- The OU process gives an **exponential** forgetting law by construction;
  the authors note ongoing debate in cognitive science over power-law vs.
  exponential forgetting (citing Wixted & Ebbesen 1997) and flag this as a
  modeling choice, not settled fact.
- The prerequisite graph already excludes mutual/reciprocal prerequisites
  by construction but does **not** enforce global acyclicity or other
  regional structural constraints; the authors "anticipate" (their word)
  this could help but did not implement or test it.
- All three evaluation datasets are pre-college **mathematics**; the authors
  explicitly ask for more diverse-domain (biology, chemistry, linguistics)
  and more diverse-stage (primary, college) datasets to test ecological
  validity, and flag this in both the Discussion and (separately) the
  Ethics Statement as a fairness/generalization concern, not just a
  technical one.
- Dataset selection criteria (KC-identifiability, fine timestamp
  resolution) structurally excluded several standard KT benchmarks
  (Statics2011, Assistments2009/2015, Junyi2020); the paper is transparent
  about this filtering and its consequence (their comparison set is
  narrower than the field's usual benchmark suite).
- The KC-identifiability requirement in particular means the model as
  evaluated cannot yet handle the common case of one assessment item
  covering multiple KCs simultaneously — acknowledged as future work.
- Ethics Statement separately notes traits are inferred from behavioral
  data only, not demographic fields present in the raw data (age, gender,
  school name in Assist17), specifically to avoid encoding those into the
  interpretable trait space.

**Reviewer criticisms**: not recovered — see the access note at the top.
No independent reviewer text is represented anywhere in this report; only
the authors' own self-assessment is included above.

---

## 8. Relevance to the kt-mirt G1/G2 avenue map

- **Direct precedent for G1 (signed cross-KC transfer)**: PSI-KT's causal-
  support validation (§3, item B2 above) is essentially a KT-native version
  of the "beyond pre-registered nulls" bar this program wants for G1 — it
  compares inferred edge probability against an independently-computed
  Bayesian causal-support statistic derived from *held-out* consecutive-
  interaction data, with an explicit null-graph baseline. That machinery
  (Griffiths & Tenenbaum causal-support formalism, Appendix A.7.3) is a
  candidate template for how to certify signed transfer against a null,
  worth reading in full if G1 work proceeds — it was not deep-read line by
  line here (time-boxed), only located and summarized at the level of what
  it compares against what.
- **Direct relevance to the lab's "stable-and-wrong" concern**: PSI-KT's
  four-axis trait-interpretability battery (specificity/consistency/
  disentanglement/operational) is a much more rigorous validation stack
  than typical KT interpretability claims, and notably the first three axes
  (self-consistency metrics) are *not* enough by the paper's own logic —
  only the 4th axis (regression against held-out behavioral signatures) is
  treated as establishing actual semantic grounding. This maps onto this
  program's "truth-free slack test" concern directly: PSI-KT's own
  structure implicitly concedes that internal-consistency metrics alone
  (their axes 1-3) don't prove a readout means what it claims — they had to
  add an external behavioral-prediction check (axis 4) to make that case.
- **Caution on architecture reuse**: PSI-KT's free per-learner trait
  vectors are exactly the "free per-learner trait multipliers fabricate
  phantom transfer" failure mode this program has already burned effort
  ruling out — except PSI-KT's traits are *time-varying* (not a fixed
  multiplier) and constrained to a specific interpretable parametric role
  (forgetting rate, volatility, transfer-ability, long-term-mean) rather
  than a free black-box embedding, which may or may not be enough
  structure to avoid the phantom-transfer failure mode this lab has
  already diagnosed elsewhere; this is not evaluated in the PSI-KT paper
  itself; it should be treated as an open question for the avenue map, not
  a resolved one.
- **License caution**: AGPL-3.0 on the reference implementation means any
  code adapted from it (even indirectly, if kt-mirt's own code were derived
  from reading psi-kt source) would carry copyleft obligations. Per the
  task instructions this report did not read their code beyond the README
  attempt (which 404'd — only the LICENSE file and repo metadata were
  fetched), so no code-level exposure occurred.
