# PSI-KT OpenReview referee record

Forum `NgaLU2fP5D` (arXiv:2403.13179, ICLR 2024, Submission 8234).
**Provenance**: the OpenReview API is Turnstile-gated and blocked every
scripted retrieval route tried in this environment (see prior attempt,
now superseded); the user cleared the Turnstile in an ordinary browser
and saved the rendered forum page as a PDF
(`data/psikt/review.pdf`, 14 pages). That PDF has no extractable text
layer (glyphs are vector-drawn, not encoded characters — 0 chars via
PyMuPDF `get_text()` on every page), so the record below was recovered
by rendering each page to a 3x-zoom PNG and OCR'ing with Tesseract
(one page re-rendered at 5x and cross-checked visually where OCR
garbled an overlapping footer). `data/psikt_reviews_raw.json` and
`data/psikt/psikt_reviews_v2.json` are both `ChallengeRequiredError`
403 payloads from the blocked API route, not review content — ignored.
All 15/15 forum replies are accounted for below (decision, meta-review,
authors' overview comment, 4 official reviews, 4 response threads
including a 3-part reply, and one reviewer's post-rebuttal follow-up).

## Scores at a glance

| Reviewer | Rating | Confidence | Soundness | Presentation | Contribution |
|---|---|---|---|---|---|
| rAUs | 8 (accept, good paper) | 3 (fairly confident) | 3 good | 3 good | 3 good |
| J4qa | 8 (accept, good paper) | 3 (fairly confident) | 3 good | 3 good | 3 good |
| k3Pn | 5 (marginally below threshold) | 4 (confident) | 3 good | 2 fair | 2 fair |
| da66 | 6 (marginally above threshold) | 4 (confident) | 2 fair | 2 fair | 3 good |

**Final decision: Accept (spotlight)**, program chairs, 16 Jan 2024.
da66's rating (6) is the *post-rebuttal* number — the review record
shows `modified: 21 Nov 2023, 23:16`, matching a same-day follow-up
comment in which da66 states the review was updated after the authors'
reply. k3Pn's rating (5, the most critical score) has no follow-up
comment anywhere in the 15 replies — k3Pn never explicitly re-scored
after the authors' 3-part rebuttal, yet the paper was accepted as a
spotlight regardless. The meta-review explicitly names this: "several
doubts raised by the reviewers, such as lack of explanations and
experiments, have been adequately resolved by the author responses" —
an AC judgment call made on the authors' rebuttal content, not on a
reviewer's own re-score.

## Meta-review (Area Chair A3Bu, 08 Dec 2023)

> This paper proposes a novel generative modeling approach to the
> knowledge tracing problem. The combination of a sophisticated model
> based on Bayesian deep learning with a graphical representation of
> knowledge relations, which achieves both predictive accuracy and
> interpretability, is a significant technical contribution. Several
> doubts raised by the reviewers, such as lack of explanations and
> experiments, have been adequately resolved by the author responses.

Justification for why not higher:

> It is a rather complex modeling that delves into the specific nature
> of the target domain and is somewhat less transferable to other
> fields. It may not be of much benefit to anyone other than those
> specifically interested in the target problem.

Justification for why not lower:

> It is technically well-designed and will be welcomed with technical
> interest not only in the learning analytics field, but also in the
> ICLR community.

## Reviewer rAUs — 8/10, confidence 3

**Summary**: PSI-KT combines individual learning dynamics with
structural prerequisite influences via Bayesian inference; achieves
"superior predictive accuracy and scalability while also providing
interpretable representations."

**Strengths**: "technically strong in its probabilistic modeling and
inference methodology"; "well-written and provides intuitive
explanations of the model components"; integration of cognitive
science and AI "significant for developing systems that leverage
psychological insights."

**Weaknesses** (verbatim):

> The evaluations focus on three specific educational datasets. Testing
> on a more diverse range of datasets could better reveal the model's
> capabilities and limitations.

> Long-term retention modeling could be enhanced. The current
> exponential decay may be simplistic. Exploring more complex
> forgetting functions based on memory research literature could
> improve long-term predictions.

> While superior overall, some accuracy metrics are comparable to
> certain baselines. Further ablation studies could provide insight
> into which model components contribute most to accuracy gains.

**Questions**: dataset-limitation discussion; richer forgetting
functions; ablation studies to attribute gains to components.

**Author response** (21 Nov 2023): reframed dataset scope as a
first-class limitation, adding to the Discussion — "Although we
designed PSI-KT with general structured domains in mind, our empirical
evaluations were limited to mathematics learning by dataset
availability" — and specified the exact inclusion/exclusion logic in
Appendix A.3.2: datasets need (1) KC labels and (2) high temporal
resolution, which rules out Statics2011 (fails 1), Assist09/Assist15
(fail 2), Junyi20 (fails 2). On forgetting, conceded the point
("exponential forgetting alone... may be simplistic") while defending
the OU-process choice for analytic marginalizability, and added to the
Discussion: "Future work should support ongoing debates in cognition by
offering alternative modeling choices for memory decay e.g.,
power-law... thus facilitating empirical studies at scale." Added a new
ablation study (Appendix A.8, Table 16, Fig. 13) ablating graph
structure, individual traits, and learner dynamics separately: "significant
drops in accuracy across all three datasets" for each, "no single
ablation reliably corresponds to the largest drop" — used as evidence
all three components are jointly necessary.

## Reviewer J4qa — 8/10, confidence 3

**Summary**: a "novel hierarchical probabilistic state-space model,"
distinguished from cross-entropy-trained discriminative KT models by
using approximate Bayesian inference and variational continual
learning; validates specificity, consistency, disentanglement, and
operational interpretability of traits plus reliability of the
inferred prerequisite graph.

**Strengths**: "Good textual expression, mathematical notation, and
formula derivations"; motivation "both novel and reasonable"; method
"intriguing" (three-level hierarchy, ELBO instead of cross-entropy);
"extensive confirmatory experiments with detailed and favorable
results."

**Weaknesses** (verbatim):

> The cognitive traits in the paper lack somewhat interpretability...
> it is advisable to explicitly state in the text which specific
> cognitive psychology traits the four dimensions of cognitive traits
> represent.

> Experiments are somewhat insufficient... there is a notable absence
> of ablation study to demonstrate the effectiveness of the two proposed
> motivations in the paper, namely cognitive traits and the prerequisite
> relationship graph... it seems somewhat inadequate not to include some
> explicit baseline models that utilize knowledge concept graphs for
> comparison.

**Questions**: what do the four trait dimensions mean cognitively
(only two of four had been related to behavior); add GKT/SKT as
graph-aware baselines.

**Author response**: named each trait explicitly in a revised Sec. 3.1
— "$\alpha$ represents the forgetting rate, $\mu_\infty$ (via $f$)
captures long-term memory consolidation for practiced KCs and expected
performance for novel KCs, $\sigma$ is knowledge volatility, and
$\tau$ indicates transfer ability from performance on prerequisite
KCs" — and added a behavioral-regression check for the two previously
unvalidated dimensions (transfer ability, knowledge volatility) against
held-out behavioral measures (Fig. 10, Appendix A.6.4), stating "in all
cases, our model achieves the highest match between parameters and
behavior." Added GKT as a baseline (plus QIKT, requested by another
reviewer). Declined to add SKT with a specific, substantive reason:
"the graph is calculated from performance statistics (i.e.,
correct/incorrect counts), which cannot be directly used to predict the
same performance data (to avoid circularity)... a fair comparison seems
challenging" — i.e., SKT's own graph-construction pipeline would leak
the prediction target into the predictor.

## Reviewer k3Pn — 5/10 (marginally below threshold), confidence 4

The most critical review; Presentation and Contribution both scored
"fair," and no post-rebuttal re-score appears anywhere in the forum.

**Summary**: "generative knowledge tracing method that places emphasis
on predictive accuracy, scalable inference, and interpretability...
extensive experimental results clearly showcase the method's
superiority over various baselines from multiple angles."

**Strengths**: "carefully designed and comprehensive"; interpretability
is "the pain point of the knowledge tracing field" and this paper
targets it; "well-structured."

**Weaknesses** (verbatim, all five):

> 1. The method's description is not sufficiently clear. As indicated
> in the appendix, PSI-KT also employs neural networks to generate
> cognitive parameters. However, the main body of the paper only
> briefly touches upon this aspect, potentially leading to the
> misconception that PSI-KT is not a deep learning approach.

> 2. The experimental setup lacks persuasiveness. As demonstrated in
> Table 1, two datasets contain over 10,000 learners, yet the authors
> chose to use only 100-1,000 learners as training data. Conducting
> experiments with a small dataset may unfairly disadvantage deep
> learning baselines, which can effectively leverage the abundance of
> available data. The reasoning provided, "to simulate real-world data
> constraints in education," may not hold in the context of the vast
> amount of student learning data generated today.

> 3. The introduction of interpretable KT methods is not comprehensive.
> For instance, recent approaches like IKT, ICKT, and QIKT incorporate
> interpretable psychological and cognitive modules into their methods.
> These relevant methods are not referenced in this paper, let alone
> included as baselines in the experiments.

> 4. The assessment of the model's interpretability is not entirely
> convincing. The limited dimensionality of hidden learner
> representations in deep learning methods (e.g., DKT, AKT) at just 16
> may constrain the neural networks' capabilities. Furthermore, there
> is no supporting evidence indicating that the learner representations
> of PSI-KT and these deep learning baselines capture the same
> underlying student features, making direct comparisons less rational.

> 5. Perhaps conducting case studies of PSI-KT could offer a more
> intuitive understanding of its interpretability, such as visualizing
> trends in students' knowledge mastery, as shown in Figure 1(a).

**Author response** (3-part, 21 Nov 2023): on setup persuasiveness,
reframed the small-cohort regime as the deliberate object of study
rather than an ad hoc constraint — reworded Sec. 4.1's opening
("Good KT performance with little data is key in practical ITS to
minimize the number of learners on an experimental treatment
[principle of equipoise, similar to medical research]... to mitigate
the cold-start problem, and extend the usefulness of the model to
classroom-size groups") and *also* ran the missing large-cohort
experiments: "PSI-KT's within-learner prediction performance is
robustly above baselines for all but the largest cohort size (>60k
learners, Junyi15), where all deep learning models perform similarly,"
adding to the Discussion: "An open question for future KT research is
how to combine PSI-KT's unique continual learning and interpretability
properties with performance that grows beyond this extreme regime." On
missing baselines, added QIKT and referenced IKT/IEKT in related work,
but explicitly could not add IKT (official code "lacks a key
component") or verify ICKT existed as distinct from IEKT. On the
interpretability challenge, the authors' central defense is that the
mutual-information-based metrics (specificity, consistency,
disentanglement) are constructed to be theory-agnostic: "mutual
information only quantifies the amount of information that... a
learner representation contains about the learner's identity... It is
agnostic to the form in which this information is encoded... If a
baseline model finds different underlying learner features that do not
match our psychologically-motivated cognitive traits then the model
can still have a high specificity score" — i.e., the metric does not
presuppose the authors' own trait taxonomy is the right one, so a
same-metric comparison across differently-parameterized models is
claimed to be fair despite the dimensionality mismatch (4-dim traits vs
16-dim baseline hidden states). Added a case-study figure (Fig. 11,
A.6.5) as requested and retitled Sec. 3.2 to foreground the inference
network, addressing weakness 1 directly. The rebuttal closes by asking
k3Pn to reconsider the score; no reply from k3Pn is recorded.

## Reviewer da66 — 6/10 (marginally above threshold, post-rebuttal), confidence 4

**Summary**: "scientifically sound model... takes into account past
performance, prerequisite knowledge graphs, and individual learner
traits... exceed the baseline."

**Strengths**: "Predictive Accuracy was reasonable and well evaluated."
Per the four ICLR review dimensions: Originality — "combining knowledge
tracing and knowledge mapping into one method is a nice combination of
ideas"; Quality — "good... well presented"; Clarity — "left a lot to be
desired... but the graphs and tables were [helpful]"; Significance —
"primary significance... is in the interpretability of the results."

**Weaknesses** (verbatim, the sharpest language in the record):

> Most of the focus of this paper was on the accuracy. Interpretability
> and scalability were not well evaluated and much of that was in the
> form of "correct by construction".

> The prerequisite graph was interesting, although the correctness of
> the graph was not well quantified.

> And although I thought the accuracy beat the provided baseline and
> had sufficient data to support that, I do not think the results are
> good, only that they are better than the baseline. For a binary
> problem, getting accuracy of 55-80 is not a strong result.

**Questions/concerns**:

> I would also like to more details on the datasets, particularly from
> the perspective of diversity. Claims about educational effectiveness
> and knowledge graphs that do no reflect a sufficient cross section are
> suspect at best and can be actively harmful.

**Author response** (18 Nov, follow-up 21 Nov): on raw accuracy,
clarified the 55% figure is specifically between-learner generalization
on Assist17 ("deliberately designed to be difficult since the lack of
historical data prevents learner personalization... note that baselines
perform even worse"), with within-learner accuracy considerably higher
(63-83%, beating all baselines), and explained the ceiling structurally:
"data are collected from learning systems designed to engage learners
with KCs that are at the edge of their abilities... the data collection
process inadvertently focuses on learner/KC-pairs where performance is
particularly hard to predict" (selection bias, not a modeling
shortfall). On dataset diversity, softened the introduction's framing
and added the same inclusion/exclusion criteria given to rAUs. On
"correct by construction," pushed back directly: "far from assuming the
interpretability is 'correct by construction,' we spend the entire Sec.
4.3 to critically test this hypothesis using multiple metrics
(specificity, consistency, disentanglement, and operational
interpretability). Here, the first three are information theoretical
metrics, which are agnostic to any preconceived notion of
interpretability precisely to avoid the risk of assuming 'correct by
construction'." On graph correctness, added expert- and crowd-sourced
alignment metrics (3 metrics, across 7 models) plus a causal-analysis
check of graph edges against future learning outcomes (Table 5, Fig.
12), and the new ablation (Table 16/Fig. 13) showing the graph is
necessary for predictive accuracy.

**da66's follow-up comment** (21 Nov 2023, 23:23, verbatim):

> Thank you, authors, for all the clarifications, updates, and changes.
> I still think that the strongest parts of this paper are in the
> interpretability of the work and the knowledge graph, but after
> clarification. I have updated my review based on your revisions and I
> hope to see this work and any follow on papers in publication soon.

## Authors' own framing of the rebuttal's significance

From the authors' overview comment (23 Nov 2023), on why the
interpretability contribution matters beyond this one paper: "we agree
with reviewer k3pn that [interpretability is] 'the pain point of the
knowledge tracing field'... providing a comprehensive evaluation
framework for interpretability. Our framework covers i) specificity,
consistency, and disentanglement of learner representations, ii) graph
alignment, and iii) operational interpretability by relating inferred
representations to future behavioral outcomes."

## Objections kt-mirt must pre-empt

1. **"Self-consistency metrics are not proof of interpretability"
   (k3Pn weakness 4, da66 "correct by construction," the sharpest and
   most repeated objection in the record).** PSI-KT's own defense was
   that specificity/consistency/disentanglement are information-
   theoretic and don't presuppose the authors' trait taxonomy — but
   three reviewers were unmoved until a *fourth*, behavioral-regression
   axis (operational interpretability against held-out outcomes) was
   added. kt-mirt's own "truth-free slack test" concern (project
   memory) already anticipates exactly this gap for signed cross-KC
   influence (G1) and per-KC growth (G2): an internally-consistent
   readout is not evidence the number means what it's claimed to mean.
   The design already answers this in principle — validity-gated
   certification against synthetic ground truth is the program's
   default bar, stronger than PSI-KT's post hoc behavioral check — but
   the real-data legs (A4, A1) must carry an external, held-out check
   analogous to PSI-KT's Fig. 10 from the first real-data pilot, not
   as a later patch, or this objection lands unchanged.

2. **"The experimental setup lacks persuasiveness" — small-cohort
   framing invites a fabricated-necessity read (k3Pn weakness 2).**
   PSI-KT was pushed to admit deep baselines "reach our predictive
   performance with just 300 learners" once trained on the full
   >60k-learner cohort, i.e., its advantage is regime-bound, not
   universal, and it had to add the large-cohort experiment reactively.
   kt-mirt's growth and influence claims (G1, G2) are validity-gated
   per bed already (PLAN.md), which is the right shape of answer, but
   the paper draft must state the regime honestly up front (which beds
   show the effect, which don't) rather than lead with the best bed and
   let a reviewer discover the boundary, exactly PSI-KT's failure mode.

3. **"Dataset scope/diversity" (rAUs, da66, both — flagged twice by
   PSI-KT itself, in Discussion and a separate Ethics Statement).**
   PSI-KT's fix was to state explicit inclusion/exclusion criteria
   (KC labels + high temporal resolution) rather than leave the
   restriction to mathematics-only datasets implicit. kt-mirt already
   does this in PLAN.md (beds named, Q-matrix policy stated per bed
   in avenue_map.md section 4) — this objection is already substantially
   pre-empted by existing documentation discipline; the remaining work
   is making sure the eventual paper states the same criteria as
   plainly as PSI-KT's rebuttal did, not just in internal planning docs.

4. **"No evidence that different models' representations capture the
   same underlying construct" — the baseline-comparability objection
   (k3Pn weakness 4, dimensionality mismatch 4-dim vs 16-dim).** This is
   a sharper version of objection 1: even if a metric is
   theory-agnostic, comparing it across architecturally different latent
   spaces invites doubt that like is being compared to like. For kt-mirt,
   this maps directly onto the signed-influence matrix G (G1): a
   recovered sign or magnitude is only informative if the estimator
   is shown, on synthetic data with a *known* G, to recover that
   specific known structure — not merely to produce *some* internally
   coherent structure. The program's synthetic-gate-before-real-claim
   rule (PLAN.md, P3) already is the correct answer to this; no
   amendment needed, but the write-up must make the analogy to this
   exact PSI-KT weakness explicit so a reviewer doesn't independently
   re-invent it as a fresh objection.

5. **"Graph correctness not well quantified" / no global structural
   constraint (da66; PSI-KT's own Discussion also concedes no
   acyclicity is enforced).** PSI-KT answered with multi-metric graph
   alignment against human annotation plus a causal, outcome-predictive
   check. kt-mirt's G1 avenue (A1, avenue_map.md) is architecturally
   different — it targets a *signed*, zero-diagonal, practice-gated
   influence matrix rather than a DAG-like prerequisite graph, and
   explicitly is not chasing PSI-KT's graph-recovery framing (PLAN.md:
   "PSI-KT... reference design only... not benchmark-chased"). This
   objection therefore transfers only partially: kt-mirt does not need
   PSI-KT's human-annotation alignment metric, but does need the same
   underlying discipline — an external check on G beyond the model's
   own internal consistency, which is what the synthetic-certification
   gate already provides for the signed-edge estimator. No design
   change indicated, but the eventual paper should preempt a reviewer
   asking "how do you know G is right" with the synthetic-recovery
   result front and center, the same way PSI-KT eventually did.

None of the five items above requires a design change beyond what
PLAN.md/THINKING.md already commit to (validity gates before real
claims, honest per-bed verdicts, anchoring against "stable and wrong"
readouts). The main actionable lesson is procedural, not architectural:
PSI-KT's rebuttal repeatedly *added missing controls reactively*
(behavioral-regression axis, large-cohort run, explicit dataset
criteria) that a stronger initial submission would have included from
the start, and kt-mirt's stated methodology already plans to include
the analogous controls up front — the risk is failing to say so as
plainly as PSI-KT was eventually forced to.
