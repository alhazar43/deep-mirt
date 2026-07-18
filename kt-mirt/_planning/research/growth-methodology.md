# Growth methodology: detecting real learner growth beyond noise

Survey for the kt-mirt avenue map (G1: signed cross-KC transfer; G2: per-learner
digital-twin growth). Covers learning-curve standards in educational data mining
(EDM), growth-curve modeling reliability, null models for trend detection,
permutation testing on longitudinal educational data, and digital-twin/simulated-
learner validation. WebFetch was unavailable in this session (network policy
blocked arxiv.org, pnas.org, and other domains), so claims below are sourced from
search-engine result snippets rather than full-text extraction; each claim notes
this where it matters.

## 1. Learning-curve analysis: the EDM standard

The dominant EDM convention for "is a skill being learned" is the **opportunity-
indexed error-rate curve**: for each knowledge component (KC), plot mean error
rate against the count of prior opportunities a student has had on that KC (first
attempt only, to avoid confounding with within-item feedback or guessing), then
fit a parametric learning curve. CMU's DataShop platform operationalizes this
directly, with an "Opportunity Cutoff" filter and a documented learning-curve
algorithm (PSLC DataShop, Learning Curve help pages,
https://pslcdatashop.web.cmu.edu/help?page=learningCurve,
https://pslcdatashop.web.cmu.edu/help?page=learningCurveAlgorithm).

The standard parametric model is the **Additive Factors Model (AFM)**, a mixed-
effects logistic regression: P(correct) is a function of a per-student intercept
(prior knowledge), a per-skill difficulty (β_k), and a per-skill learning-rate
slope (γ_k) multiplied by opportunity count. AFM traces to Cen, Koedinger &
Junker's Learning Factors Analysis (ITS 2006,
https://link.springer.com/chapter/10.1007/11774303_17), which combined this
statistical model with AIC/BIC-driven search over cognitive-model refinements.
**Performance Factors Analysis (PFA)**, Pavlik, Cen & Koedinger 2009
(https://files.eric.ed.gov/fulltext/ED506305.pdf), is the standard alternative:
it splits the opportunity count into successes and failures per skill (γ_k^S,
γ_k^F), letting correct and incorrect practice have different learning effects,
but treats all past opportunities as equally weighted regardless of recency —
a documented limitation motivating recency-weighted extensions such as Deep PFA
(https://www.researchgate.net/publication/352298606_Deep_Performance_Factors_Analysis_for_Knowledge_Tracing).

Ad hoc classification schemes exist for individual curves: DataShop-adjacent
work bins curves into categories like "Low and Flat" (all points under ~20%
error), "No Learning" (no significant improvement), "Still High" (final error
over ~40%), and "Good" (significant positive slope). This is a heuristic
visual-QA layer, not a formal test, and it is not a citable universal standard
— treat it as descriptive practice, not a validated methodology.

**Relevance to G1/G2.** AFM/PFA is the direct ancestor of the "per-KC learning
rate" quantity kt-mirt would need for G2 (per-learner ability growth on a
trustworthy scale) and is the natural null-model baseline: a per-skill γ_k not
significantly different from zero is EDM's operational definition of "no
detectable learning on this skill." It does not by itself give cross-KC
transfer (G1) — AFM is single-skill per row; cross-KC effects require an
explicit interaction or transfer term, which is not part of the vanilla model.

## 2. The null-model / significance-testing layer

The standard test for "is the learning-rate slope real" is a **likelihood-ratio
test (LRT) comparing nested mixed-effects models** — one with the opportunity
(or opportunity-by-skill) term, one without — using a chi-square reference
distribution on the deviance difference (general mixed-model methodology;
https://link.springer.com/article/10.3758/s13428-016-0809-y,
https://cambiotraining.github.io/stats-mixed-effects-models/materials/06-significance-and-model-comparison.html).
I could not confirm from search snippets alone whether Koedinger et al.'s PNAS
"astonishing regularity" paper (below) reports this exact LRT for each skill's
γ_k, or relies on inspecting the estimated random-effects distribution directly
(WebFetch of the PNAS page failed under this session's network policy, so this
is an unconfirmed gap — flagged as **not fully verified**).

**Permutation / surrogate-data tests** are the model-agnostic alternative and
are well established outside education (time-series surrogate testing,
https://en.wikipedia.org/wiki/Permutation_test; shuffle-based surrogate methods
reviewed at https://www.sciencedirect.com/science/article/pii/S0370157318301340).
The logic — shuffle the outcome sequence (or student/skill labels) to build a
null distribution for "no true ordering effect," then compare the observed
trend statistic against it — transfers cleanly to opportunity-ordered KT data:
shuffling a student's attempt order within a skill destroys any true learning
trend while preserving the marginal error rate, giving a null distribution for
slope-under-no-learning. I did not find an EDM-specific paper using exactly this
recipe on knowledge-tracing sequences; the permutation-test literature located
was general-purpose or from adjacent domains (peer-assessment learning gains,
https://arxiv.org/pdf/1410.3853). **This is a methodological gap and an
implementable idea, not a citable established practice** — worth flagging
explicitly for the avenue map rather than presenting as settled.

## 3. Growth-curve modeling: timepoint requirements

Classical growth-curve modeling (latent growth curve models, multilevel models
of change) requires **a minimum of three timepoints** to estimate a trajectory
shape at all; two timepoints only give a pre/post gain score, with no
information on trajectory shape, timing, or individual-level slope variance
(PMC review, https://pmc.ncbi.nlm.nih.gov/articles/PMC8941055/; Newsom's growth
model notes, https://web.pdx.edu/~newsomj/mlrclass/ho_growth.pdf). Critically,
**three timepoints is a floor for identifiability, not for reliability**: recent
work explicitly shows two-timepoint (and by extension sparse) designs are
"poorly suited to model individual differences in linear slopes" even though
the *group-level* mean slope can be recovered reasonably well from just two
points (https://www.sciencedirect.com/science/article/pii/S1878929324000148,
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11470183/). This is directly load-
bearing for G2: a per-learner digital twin needs individual-level slope
estimates, which is the harder, more data-hungry target — group-average growth
being recoverable at low density does not imply individual growth is.

Statistical power for detecting slope variance in growth models scales with
sample size, number of measurement occasions, and the magnitude of true
inter-individual slope variance (Frontiers,
https://www.frontiersin.org/articles/10.3389/fpsyg.2018.00294/full). Simulation-
based power analysis (Monte Carlo over the target design) is the standard
recommended tool when no closed-form power function fits the design
(https://link.springer.com/article/10.3758/BRM.41.4.1083). One general caution
from health-professions-education research is directly relevant by analogy:
a systematic review found a median sample size of 25 among no-intervention-
comparison studies, with only 0.3% reaching 80% power to detect small effects
(https://link.springer.com/article/10.1007/s10459-014-9509-5) — underpowered
designs are the norm in adjacent education-research literatures, a caution
against assuming any given KT dataset's practice density is "enough" without
checking.

## 4. Empirically observed per-skill learning-rate magnitudes

The single most load-bearing empirical result for the avenue map is Koedinger,
Carvalho, Liu & McLaughlin, "An astonishing regularity in student learning
rate," PNAS 120(13):e2221311120, 2023
(https://www.pnas.org/doi/10.1073/pnas.2221311120,
https://pubmed.ncbi.nlm.nih.gov/36940328/). Fitting AFM-style mixed-effects
models to 1.3 million observations across 27 datasets of online-practice-system
interactions (elementary through college, math/science/language), they report:
students vary widely in **initial performance** but are strikingly similar in
**learning rate**, on the order of **~0.1 log-odds (~2.5 percentage points of
accuracy) gained per practice opportunity**, largely independent of
skill/domain/dataset. This is the headline number to calibrate expectations for
G2: if per-KC growth in kt-mirt data is of a similar order, opportunity counts
in the tens are needed to move accuracy by double-digit percentage points, and
a handful of practice events per KC per learner is well below the detectable
regime.

This result has been both **replicated and contested**:
- **Replication at scale**: Simpson, Norberg & Fancsali, "Replicating an
  'Astonishing Regularity in Student Learning Rate,'" EDM 2024 short paper
  (https://educationaldatamining.org/edm2024/proceedings/2024.EDM-short-papers.40/,
  full text https://files.eric.ed.gov/fulltext/ED675636.pdf), refit on MATHia
  ITS data — over 15,000 students, 821,890 observations, 6 math topics — and
  report the regularity holds in a "business as usual," diverse population.
- **Replication at scale (industry)**: Beauchesne et al. (Campus AI), "Personalized
  AI Practice Replicates Learning Rate Regularity at Scale," 2026
  (https://arxiv.org/abs/2604.03246), 1.8M interactions (366k post-filtering),
  automatically generated KCs. They report the *initial-knowledge* IQR (2.78 to
  12.18 opportunities to reach 80% mastery) is wide while the *learning-rate*
  IQR (7.01 to 8.25 opportunities to reach mastery, read from their reported
  interquartile range on time-to-mastery) is narrow — consistent with the
  original PNAS framing that variance concentrates in starting point, not
  slope. Note: I am inferring the precise interpretation of "IQR = [7.01, 8.25]
  opportunities" for learning rate from a search snippet, not the full text;
  treat the exact framing as approximate, the qualitative direction (rate
  narrower than intercept) as solid given two independent replications state it.
- **Methodological contestation**: Lee, Lichand, Barnard, Klotz, Thille, Kim &
  Domingue, "The 'Astonishing Regularity' Revisited: Sensitivity of
  Learning-Rate Estimates to Practice-Sequence Length," L@S 2026
  (https://arxiv.org/abs/2605.01690), refit individual-AFM on 26 of the
  original 27 datasets and show the "regularity" is an artifact of how much
  practice-sequence data is included: **truncating to the first 10 opportunities
  per student-skill pair inflates the estimated interquartile range of
  student-level learning rates by 75%** relative to using full sequences. This
  is the single most important methodological caveat for the avenue map: a
  per-KC learning-rate estimate is not a fixed property of the data, it is a
  function of how much practice history per KC you condition the estimate on,
  and short practice histories (which is what most non-tutoring-system KT
  datasets like ASSISTments/Junyi realistically offer per KC per student)
  systematically inflate apparent between-student variance, i.e. can manufacture
  spurious individual differences in growth rate. A related critique (Justin
  Skycak's blog critique of the PNAS paper) exists but its specific arguments
  could not be verified in this session (WebFetch blocked); do not cite its
  content without direct verification.
- **Threshold folklore**: one source states "students needed about 7 additional
  opportunities per KC ... to master each KC" in DataShop-adjacent work (search
  snippet only, exact paper unidentified — **not independently verified, low
  confidence**), and the EDM 2024 "many small problems" programming-education
  paper (Demirtas, Fowler & Cunningham,
  https://educationaldatamining.org/edm2024/proceedings/2024.EDM-long-papers.5/)
  found that KCs tested by fewer problems more often failed to show a
  decreasing error-rate curve at all — i.e., **too few opportunities produces
  a "no learning" curve as an artifact of low power, not necessarily as a true
  null**, in a domain with ~90 small problems per course.

**Junyi/ASSISTments-specific per-skill rates.** I found no published numeric
per-skill learning-rate estimates specific to Junyi Academy or ASSISTments in
this search pass — the PNAS/replication/critique triad above uses ITS platforms
(largely MATHia/Cognitive Tutor-family and Campus AI), not Junyi or ASSISTments.
This is a genuine evidence gap: **kt-mirt cannot currently cite an established
per-skill learning-rate magnitude on its own target dataset family**, only the
general ~0.1 log-odds/opportunity order-of-magnitude from ITS platforms with
denser, more structured practice sequences than typical MOOC-style logs. Given
this lab's own prior finding that aggregate per-student learning rate was
unrecoverable on EdNet/ASSISTments/KDD due to ~80% accuracy saturation (per
program context), the ITS-platform numbers above are optimistic upper bounds on
what similar analysis would find on saturating benchmark datasets — ITS
platforms like MATHia are specifically engineered to keep students in a
non-saturating practice regime (mastery-based progression), which ASSISTments-
class logs, collected from unmanaged classroom usage, are not.

## 5. Digital-twin / simulated-learner fidelity validation

The relevant EDM-community literature calls these "simulated learners," not
"digital twins" (the digital-twin search returned mostly irrelevant industrial/
aviation hits — that term is not the field's vocabulary here). The
foundational validation methodology:

- Koedinger, Matsuda, MacLellan & McLaughlin, "Methods for Evaluating Simulated
  Learners: Examples from SimStudent"
  (https://www.semanticscholar.org/paper/Methods-for-Evaluating-Simulated-Learners:-Examples-Koedinger-Matsuda/a1f4546abff2e8ab3404e62a6726da35cc68726f)
  lays out evaluation goals for simulated learners (as a theory-of-learning
  test, as a tutor-authoring aid, as a generator of testable hypotheses) and
  validates SimStudent, in part, by **comparing the simulated student's
  opportunity-indexed error-rate learning curve against the human learning
  curve on the same task** — the same AFM-style curve machinery from Section 1,
  repurposed as a fidelity metric rather than a growth-detection metric.
- Käser & Alexandron, "Simulated Learners in Educational Technology: A
  Systematic Literature Review and a Turing-like Test," IJAIED 2024
  (https://link.springer.com/article/10.1007/s40593-023-00337-2), is the
  first systematic review of the field (decade 2010-2019 coverage per the
  search snippet), and validates using a **blinded Turing-like test**: human
  judges (or classifiers) try to distinguish real student data from simulated
  student data; a well-calibrated simulator is one where this discrimination
  fails at chance. This is a genuinely distinct validation criterion from
  curve-matching — it targets full behavioral distribution match, not just
  first-moment (mean error rate) match.
- More recent LLM-based student-simulator work continues this dual-criterion
  tradition: e.g. "Towards Valid Student Simulation with Large Language
  Models" (https://arxiv.org/html/2601.05473v1) explicitly separates
  black-box behavioral validation from white-box pedagogical-process
  validation, and "Simulating Students or Sycophantic Problem Solving? On
  Misconception Faithfulness of LLM Simulators"
  (https://arxiv.org/pdf/2605.12748) raises a fidelity failure mode specific
  to LLM simulators (sycophancy toward the tutor rather than faithful
  reproduction of a misconception) that is a useful cautionary analogy for any
  kt-mirt digital twin built on a neural sequence model: a model optimized on
  prediction loss can look behaviorally plausible while not tracking the
  causal quantity (true ability/misconception state) it is being used to
  explain — directly the "stable-and-wrong" concern already in this lab's
  prior work.

**Relevance to G2.** No paper in this pass validates a per-learner digital twin
specifically for *longitudinal growth trajectory* fidelity (as opposed to
single-snapshot or aggregate curve fidelity). The nearest analog — comparing
simulated vs. human learning curves — validates the *population* learning-curve
shape, not whether an individual simulated trajectory tracks an individual
real trajectory's growth over time. This is an open methodological gap the
avenue map should note explicitly: **there is no established fidelity test in
the EDM literature for individual-level digital-twin growth curves**, only for
population-level curve shape and full-distribution behavioral match.

## 6. Longitudinal/growth IRT models (bearing on G2's "trustworthy scale")

Longitudinal IRT is an active but distinct sub-literature from KT growth curves,
concerned with tracking latent ability over repeated testing occasions on a
fixed measurement scale. Relevant model families reported: longitudinal
unidimensional IRT (L-UIRT), longitudinal multidimensional IRT (L-MIRT),
longitudinal higher-order IRT (L-HO-IRT), and multilevel higher-order IRT
(ML-HIRT) for growth in both first- and second-order latent traits
(search-snippet summary of Wang & Nydick's didactic overview,
https://journals.sagepub.com/doi/10.3102/1076998619882026,
https://files.eric.ed.gov/fulltext/ED599228.pdf). Practical concerns raised
in this literature — vertical scaling across occasions, floor/ceiling effects
requiring adaptive testing, and separating gain magnitude from *where on the
scale* the gain occurs — are exactly the concerns this lab's prior work
flagged around "trustworthy scale" for G2. This sub-literature is a candidate
source of established scale-stability diagnostics (e.g., checking whether item
parameters used to anchor the scale are themselves stable across occasions) but
I did not locate an EDM/KT paper that imports these longitudinal-IRT
diagnostics into a neural KT + IRT-readout pipeline — this looks like an open
avenue, not a solved problem elsewhere.

## Bottom line for the avenue map

1. AFM/PFA-style opportunity-indexed learning curves plus a nested-model LRT
   (or, more robustly but less precedented in EDM, a permutation/shuffle null)
   is the standard toolkit for "is this KC's growth real." No paper found does
   the shuffle-null version specifically for KT opportunity sequences — that
   would be a methodological contribution, not just an application.
2. Growth-curve reliability needs ≥3 timepoints structurally, and individual-
   level slope reliability needs materially more than that; group-level slope
   is recoverable from sparse data but individual-level slope (what G2 needs)
   is not, per two-timepoint-limitation literature.
3. The best available per-skill learning-rate magnitude, ~0.1 log-odds
   (~2.5 pp accuracy) per opportunity, comes from ITS platforms (MATHia,
   Campus AI), not Junyi/ASSISTments; no directly on-target number exists.
   Treat it as an order-of-magnitude anchor only.
4. The Lee et al. 2026 sensitivity result is the sharpest actionable warning:
   short per-KC practice histories (≤10 opportunities) inflate apparent
   learning-rate heterogeneity by ~75%. Any G1/G2 pipeline that estimates
   per-KC or per-learner slopes must report and stress-test sensitivity to
   opportunity-count truncation, or risks reporting artifactual "growth" or
   "transfer" that is really a low-opportunity-count estimation artifact.
5. Digital-twin validation in EDM ("simulated learners") has two established
   fidelity criteria — aggregate learning-curve match, and blinded Turing-like
   behavioral match — neither of which tests individual-trajectory growth
   fidelity. Building and validating that is open territory, consistent with
   this program's framing of G2 as unproven.
