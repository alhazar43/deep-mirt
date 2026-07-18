# Transfer, interference, and forgetting: how "practicing KC A affects KC B" has been operationalized

Scope note: this report covers G1 (signed cross-KC influence). It does not cover per-learner
ability-growth validation (G2), except where the same literature bears on both.

## 1. The classical toolkit: LFA/AFM, PFA, and their transfer extension

**Learning Factors Analysis (LFA) / Additive Factors Model (AFM).** Cen, Koedinger, and Junker
(ITS 2006, published as "Learning Factors Analysis: A General Method for Cognitive Model
Evaluation and Improvement," Springer LNCS 4053) introduced a semi-automated method combining a
statistical model, human expertise, and combinatorial search to evaluate and refine a cognitive
model (a mapping of items to knowledge components, i.e. a Q-matrix). The statistical core that
emerged is now called the Additive Factors Model: a logistic-regression item-response model with
one intercept (difficulty) and one slope (learning rate) per knowledge component (KC), fit on
student x KC x opportunity-count data.
(https://link.springer.com/chapter/10.1007/11774303_17)

AFM's KC slopes are *within-KC* learning-rate estimates: each KC accrues its own practice effect,
independent of other KCs, by construction. AFM does not have a cross-KC term. In practice a KC
slope estimated as negative (performance getting worse with more attempts tagged to that KC) is
used diagnostically as a red flag that the KC label is wrong (the cognitive model
mis-decomposes the skill), not as a measurement of true negative transfer between two distinct
KCs (https://link.springer.com/chapter/10.1007/11774303_17;
https://dl.acm.org/doi/10.1145/3375462.3375491 discusses robustness/generalizability issues with
AFM fits). This is a structural point for the avenue map: the field's default cognitive-model-
fitting tool cannot represent "A hurts B" at all; it can only flag "A's own KC label looks
wrong."

**Performance Factors Analysis (PFA).** Pavlik, Cen, and Koedinger ("Performance Factors
Analysis — A New Alternative to Knowledge Tracing," AIED 2009,
https://files.eric.ed.gov/fulltext/ED506305.pdf) reformulated AFM as a logistic regression over
per-KC counts of prior successes and failures (rather than a single opportunity count), enabling
adaptive item selection and multi-KC (conjunctive) items. Like AFM, PFA's terms are indexed by KC
and item, not by KC-pairs; cross-KC effects are not part of the base model.

**Learning Factors Transfer Analysis (LFTA) — the closest classical antecedent to G1.** A
companion/extension paper, Pavlik, Cen, and Koedinger, "Learning Factors Transfer Analysis: Using
Learning Curve Analysis to Automatically Generate Domain Models" (EDM 2009,
https://pact.cs.cmu.edu/koedinger/pubs/Pavlik,%20Cen%20Kodeinger%2009.pdf), is the one place in
this classical line that explicitly targets a transfer *relationship between item types*. It uses
learning-curve analysis and a pairwise statistical test across item types to search for transfer
relationships, encoding the result as a Q-matrix domain model; Q-matrices produced this way give
better cross-validated learning-curve fits than expert-authored baselines. Critically, the
paper frames its two competing hypotheses using classical transfer theory: an "Identical Transfer"
model grounded in Thorndike's identical-elements theory (transfer exists only through literally
shared task elements, and is additive/non-negative) versus a "Faculty Transfer" model (a general,
also non-negative, practice effect). Both hypothesis families are structurally positive-or-zero
transfer; neither entertains a signed (positive-or-negative) cross-item effect. Validation of the
resulting Q-matrix is internal (cross-validated fit to held-out learning curves), not against an
independent, external ground truth of "which skills really transfer to which."
(https://www.semanticscholar.org/paper/Learning-Factors-Transfer-Analysis:-Using-Learning-Pavlik-Cen/0b27e45dd0e2b23f28a6478554043f594bf62437)

A related methodological thread: Yudelson, "Towards Better Understanding of Transfer in Cognitive
Models of Practice" (EDM 2011 poster,
https://pact.cs.cmu.edu/pubs/edm2011_poster28_Yudelson%20(1).pdf) and Gong et al., "How to
Construct More Accurate Student Models: Comparing..." (2010,
https://web.cs.wpi.edu/~nth/pubs_and_grants/papers/2010/Journals/Gong%20How%20to%20Construct%20More.pdf)
compare alternative transfer-model formulations by predictive fit — again, internal
cross-validation against held-out response accuracy, not an external transfer ground truth.

## 2. KLI: a theoretical, not empirical, organizing framework

The Knowledge-Learning-Instruction (KLI) framework (Koedinger, Corbett, and Perfetti, "The
Knowledge-Learning-Instruction Framework: Bridging the Science-Practice Chasm to Enhance Robust
Student Learning," Cognitive Science 2012,
https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1551-6709.2012.01245.x;
open PDF http://pact.cs.cmu.edu/pubs/Koedinger,%20Corbett,%20Perfetti%202012-KLI.pdf) organizes
KCs, learning processes (memory-and-fluency, induction-and-refinement, understanding-and-sense-
making), and instructional principles into a taxonomy meant to predict which instructional designs
should transfer and at what grain size. It is a theoretical scaffold used to interpret AFM/PFA/LFA
results and to reason about "robust learning" (learning that transfers, prepares for future
learning, and is retained). It does not itself supply a statistical test for cross-KC influence
and is not, on the evidence found here, validated against a held-out transfer ground truth; it is
best read as vocabulary and hypothesis-generation for the field, load-bearing for how a KT paper
frames its claims but not as an empirical method to reuse directly.

## 3. Forgetting-aware and cross-skill neural KT

Two deep-KT lines operationalize "does practicing A change performance on B" through explicit
temporal/cross-skill terms, both built to improve next-response prediction rather than to test a
scientific transfer hypothesis:

**DKT-Forget** (Nagatani, Zhang, Sato, Chen, Chen, Ohkuma, "Augmenting Knowledge Tracing by
Considering Forgetting Behavior," WWW 2019, pp. 3101-3107,
https://www.researchgate.net/publication/333067403_Augmenting_Knowledge_Tracing_by_Considering_Forgetting_Behavior)
adds three engineered forgetting features to DKT's input: the repeated time gap (interval since
the last interaction on the *same* KC), the sequence time gap (interval since the last interaction
of *any* kind), and past trial counts. This is single-KC forgetting (how a KC's own trace decays),
not a cross-KC influence term, though the sequence-time-gap feature is sensitive to what a student
did between two attempts on the same KC (i.e., practice on other KCs shows up only as elapsed
time, not as identified cross-KC content).

**HawkesKT** (Wang, Ma, Zhang, Lv, Wan, Lin, Tang, Liu, Ma, "Temporal Cross-Effects in Knowledge
Tracing," WSDM 2021, https://dl.acm.org/doi/10.1145/3437963.3441802; PDF
http://www.thuir.cn/group/~YQLiu/publications/WSDM2021Wang.pdf; code
https://github.com/THUwangcy/HawkesKT) is the model in this space that most directly matches the
shape of G1. It uses a Hawkes point process: every past interaction on some KC exerts a
mutual-excitation term on the "intensity" governing future correctness on a target KC, with a
learned, KC-pair-specific decay kernel (different KCs are assumed to be forgotten at different
rates). The excitation terms are signed, so the model can in principle represent both positive and
negative cross-KC effects. The paper reports a qualitative "Cross-effects Matrix Interpretation"
(CMI) visualization of the learned skill-pair effects as an interpretability case study. I could
not confirm from available sources (arxiv.org and thuir.cn were both unreachable through the
fetch tool in this session — flagging as an evidence gap rather than a finding) whether the CMI
signs were checked against any external ground truth (curriculum order, expert judgment, a held-
out transfer experiment) or presented purely as a face-validity visualization on the model's own
fitted matrix. Given that HawkesKT's downstream metric is next-step AUC, and given this program's
own prior finding that KT readouts can be "stable and wrong" under prediction-only training, the
CMI signs should be treated as *unverified* until the paper's validation method is confirmed
first-hand.

Both models are widely benchmarked on ASSISTments-family and similar datasets, per standard KT
survey coverage (Liu et al., "A Survey of Knowledge Tracing: Models, Variants, and Applications,"
ACM Computing Surveys 2023, https://dl.acm.org/doi/10.1145/3569576;
arXiv survey https://arxiv.org/pdf/2201.06953 and https://arxiv.org/pdf/2105.15106).

## 4. Interpretability critiques directly relevant to trusting cross-KC claims

**"On the Interpretability of Deep Learning Based Models for Knowledge Tracing"** (arXiv
2101.11335, https://arxiv.org/abs/2101.11335) reports that DKT tends to learn something closer to
an aggregate "ability" signal than genuine per-skill tracking; the recurrent architecture can
reinforce information not clearly tied to the target skill; and an *untrained* recurrent network
can match a trained DKT's predictive performance on some diagnostics, which the authors use to
argue the recurrence is not doing the skill-specific work the interpretation would assume. This is
a direct warning against reading DKT/DKVMN-family hidden states or their derived readouts as
literal per-KC (let alone cross-KC) knowledge signals without an external check.

**"Does Interpretability of Knowledge Tracing Models Support Teacher Decision Making?"** (arXiv
2511.02718, https://arxiv.org/html/2511.02718v1 / https://arxiv.org/pdf/2511.02718) is a directly
on-topic, recent title that questions whether KT models' interpretable outputs are actually useful
or trustworthy for real pedagogical decisions. I was unable to fetch the full text in this session
(arxiv.org fetch blocked) and can only report the title and search-snippet framing; treat the
existence and framing of this paper as confirmed, its detailed findings as unverified until read
directly. Flagging this explicitly as a paper the avenue map should have someone actually open.

## 5. Ground truth and external validation for cross-KC / prerequisite claims

This is the crux of the RQ. Across everything surveyed, external, independent ground truth for
transfer relationships is scarce, and where it exists it is almost always for *prerequisite
structure* (a partial order: "B requires A"), not for *signed magnitude of influence* (how much
practicing A moves performance on B), and essentially never for *negative* transfer.

**Junyi Academy / "Junyi15."** The Junyi Academy dataset (a Khan-Academy-derived Chinese e-
learning platform, ~16M interaction logs across roughly 1,300+ math exercises,
https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-dataset-by-junyi-academy;
EduData docs https://edudata.readthedocs.io/en/latest/build/blitz/junyi/junyi.html) is unusual in
carrying an *expert-authored* exercise hierarchy with explicit prerequisite links, plus, in the
variant used by later papers ("Junyi15"), human-annotated prerequisite **and similarity**
relations between knowledge concepts. This is the dataset used by PSI-KT (Zhou, Bamler, Wu,
Tejero-Cantero, "Predictive, scalable and interpretable knowledge tracing on structured domains,"
ICLR 2024 spotlight, https://arxiv.org/abs/2403.13179, code
https://github.com/mlcolab/psi-kt) as external ground truth: they measure alignment between their
model's inferred KC graph and the Junyi15 human-annotated graph, and separately correlate their
inferred prerequisite probability with a Bayesian causal-support measure computed from held-out
behavioral data. This is, on current evidence, the strongest example found of a KT model's
inferred cross-KC structure being checked against an *independent* human-annotated reference
rather than only against next-step prediction accuracy. It validates *prerequisite existence*
(positive, directional dependency), not signed magnitude and not negative transfer.

**MOOC prerequisite-relation datasets.** Pan, Li, Li, Ding, Yang, and Chua, "Prerequisite Relation
Learning for Concepts in MOOCs" (ACL 2017, https://aclanthology.org/P17-1133/) built a course/
video-level prerequisite dataset from the NPTEL MOOC platform; search snippets reported figures on
the order of several hundred course-dependency pairs and roughly a thousand KC-prerequisite pairs,
but I could not independently confirm the exact counts from the primary source in this session —
treat the specific numbers as approximate. Prerequisite-relation-learning is now enough of a
subfield to have a survey (Prerequisite Relation Learning: A Survey and Outlook, ACM Computing
Surveys 2025, https://dl.acm.org/doi/10.1145/3733593), which is worth consulting directly for a
fuller list of ground-truth resources. A parallel, non-behavioral resource is ESCO-PrereqSkill (Le
and Abel, "How Well Do LLMs Predict Prerequisite Skills? Zero-Shot Comparison to Expert-Defined
Concepts," arXiv 2507.18479, https://arxiv.org/abs/2507.18479): 3,196 occupational skills with
expert-defined prerequisite links from the EU's ESCO taxonomy, used to benchmark LLMs, not tied to
student response logs at all, but illustrative of the kind of formal, expert-curated prerequisite
ground truth that education-adjacent fields are building.

**Eedi / NeurIPS 2020 Education Challenge.** Eedi diagnostic multiple-choice items map each
incorrect answer (distractor) to a specific, named misconception, and items carry a
construct/concept hierarchy (construct-text, construct-ID) (NeurIPS 2020 Education Challenge
results paper: https://www.researchgate.net/publication/350808255; challenge proceedings
http://proceedings.mlr.press/v133/wang21a/wang21a.pdf). This is the closest thing found to
ground truth for the *misconception-reinforcement* mechanism of negative transfer (practicing a
KC could reinforce a specific wrong rule that then hurts a related KC), because the
misconception label is independent of any model's fitted parameters. I found no paper in this
search that actually uses Eedi's misconception labels to test a cross-KC negative-transfer
hypothesis; this looks like an open, usable resource rather than a validated method.

**Explicit admission of the ground-truth gap.** Annabi and Nguyen, "Prerequisite Structure
Discovery in Intelligent Tutoring Systems" (ICDL 2023; arXiv 2402.01672,
https://arxiv.org/abs/2402.01672) state plainly that evaluating a discovered knowledge structure
requires comparison with a ground-truth structure that is "absent from publicly available
datasets" for their setting, and fall back to evaluating via simulated students and a downstream
recommendation task rather than real ground truth. This corroborates, from a second independent
source, that the field-wide default when no ground truth is available is to substitute predictive
or simulation-based proxies — exactly the substitution this program's prior work has already
flagged as capable of being "stable and wrong."

**Q-matrices more broadly.** Expert-authored Q-matrices (e.g., Cognitive Tutor / PSLC DataShop
domain models) are themselves a form of ground truth for *item-to-KC* mapping, and there is a
literature on validating and refining them (e.g., "Evaluation of Expert-Based Q-Matrices
Predictive Quality in Matrix Factorization Models," https://dl.acm.org/doi/10.1007/978-3-319-24258-3_5;
regularized Q-matrix validation, https://journals.sagepub.com/doi/10.3102/10769986241240084). This
is adjacent ground truth (which KC an item belongs to) rather than direct ground truth for
cross-KC influence, but it is the mechanism by which any cross-KC claim gets its KC labels in the
first place, so a G1 pipeline inherits whatever Q-matrix noise is present upstream.

## 6. Negative transfer: how rare, and what would make a finding credible

**How rare.** Negative transfer / interference is a long-established construct in general
learning psychology — the classical definition (prior learning actively interfering with new
learning, as opposed to simply failing to help) is summarized at
https://en.wikipedia.org/wiki/Negative_transfer_(memory), and it has motivated substantial general
cognitive-training literature (e.g., "Does Far Transfer Exist? Negative Evidence From Chess,
Music, and Working Memory Training," https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5724589/, and
"There is No Supporting Evidence for a Far Transfer of General Perceptual or Cognitive Training to
Sports Performance," https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11560981/). But that literature
is almost entirely about the *absence of positive far transfer* (training X does not help Y),
which is a different and much weaker claim than *negative* transfer (training X actively hurts
Y). Positive near transfer is the norm reported in ITS/EDM contexts; far transfer is reported as
rare and hard to demonstrate even when actively sought (general summary consistent across
multiple sources retrieved here). None of the classical KT-adjacent statistical toolkit (AFM, PFA,
LFTA) is structured to represent negative cross-KC transfer at all (see Sections 1 and 2); the one
model found here with a signed cross-KC term, HawkesKT, has an unconfirmed validation method for
its signs (Section 3). On the evidence gathered in this session, I found **no paper reporting a
specific, externally-corroborated instance of negative transfer between two named KCs in real
student log data.** This should be treated as a gap statement bounded by this session's search
depth, not a definitive claim that no such paper exists — a dedicated citation-chase from the
prerequisite-relation-learning survey (ACM CSUR 2025, https://dl.acm.org/doi/10.1145/3733593) and
from the KLI literature's discussion of misconception-driven interference would be the next step
before relying on this absence.

**What would make a negative-transfer finding credible**, synthesized from the above and from this
program's own prior "stable-and-wrong" and "truth-free slack test" lessons:

1. **A priori mechanism.** The KC pair should have a stated, independently motivated reason to
   interfere before the data are seen — e.g. shared surface features with contradictory
   procedures (the canonical psych example is procedural interference between similar-looking but
   incompatible rules), or a known common misconception (Eedi-style distractor-to-misconception
   mapping would let this be checked directly rather than asserted).
2. **A pre-registered null / control condition**, per this program's own established practice: a
   time-matched or exposure-matched baseline (e.g., practicing an unrelated KC, or elapsed time
   alone) to rule out fatigue, boredom, or simple forgetting-with-time as confounds that could
   masquerade as KC-specific interference.
3. **Robustness under resampling.** In any model that fits a full KC x KC cross-effect matrix
   (HawkesKT-style), a lone negative cell is exactly the kind of thing that can be a fitting
   artifact in a large, weakly identified parameter matrix; the sign and magnitude should survive
   seed variation, cross-validation folds, and ideally permutation testing against a null where
   the KC-pair identity is shuffled.
4. **External corroboration.** Either expert/curriculum judgment (the Junyi15-style annotation
   precedent, though existing annotations are for prerequisite/similarity, not interference, so
   new annotation would likely be needed) or a controlled behavioral check (e.g., a held-out
   pretest/posttest cohort, or an interleaving-vs-blocking manipulation drawing on the spacing/
   interleaving literature) rather than only within-model fit.
5. **Effect size and direction stated on a defensible scale**, given this program's own emphasis
   on trustworthy scales for ability/effect claims (relevant to G2 but the same discipline should
   carry over to G1 magnitudes) — a signed cross-effect reported only in log-odds units of an
   opaque neural readout is not yet an interpretable claim about "learning."

## 7. Bearing on the avenue map

- The classical AFM/PFA/LFTA family gives principled machinery for per-KC learning curves and one
  genuine transfer-relationship test (LFTA's pairwise learning-curve comparison), but it is
  structurally non-negative and validates internally (cross-validated fit), not externally.
- HawkesKT is the nearest existing signed cross-KC mechanism compatible with a neural KT backbone,
  but its interpretability claims need first-hand verification before being trusted or reused;
  this is a concrete, cheap next step (read the WSDM 2021 paper's CMI section directly).
- Junyi15 is, on current evidence, the best available dataset carrying independent human-
  annotated prerequisite (and similarity) relations that could serve as an external check for a
  G1 pipeline's *prerequisite-direction* claims, though not for signed magnitude or negative
  transfer specifically. It is a concrete, reusable external-validation resource worth acquiring.
- Eedi's misconception/distractor labels are an unexploited resource for testing the
  misconception-reinforcement mechanism of negative transfer, independent of any KT model's own
  parameters.
- No external ground truth for negative transfer specifically was found; any G1 negative-transfer
  claim this program makes will likely have to build its own credibility case (pre-registered
  null, a priori mechanism, resampling robustness) rather than lean on an existing benchmark.
- Two papers directly bear on whether KT-derived cross-KC readouts can be trusted at all (arXiv
  2101.11335 and arXiv 2511.02718) and should be read in full before this program finalizes its
  own validation story for G1; both were only accessible as search snippets in this session
  (arxiv.org and thuir.cn fetches were blocked by network policy here), so their detailed claims
  below are marked accordingly.

## Evidence-access caveat

WebFetch to arxiv.org and thuir.cn was blocked in this session ("unable to verify domain safety").
All claims above sourced from those domains rest on WebSearch result snippets and secondary
mentions (ResearchGate, Semantic Scholar, ACM DL, publisher pages, GitHub READMEs), not on reading
the primary PDFs directly. Where this materially affects confidence, it is flagged inline. A
follow-up pass with working arXiv access should re-verify Sections 3 (HawkesKT CMI validation
method) and 4 (both interpretability critique papers) before those specific claims are treated as
settled.
