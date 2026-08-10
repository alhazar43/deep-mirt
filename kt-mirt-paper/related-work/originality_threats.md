# Originality threats (adversarial sweep, 2026-08-10)

Adversarial prior-art check for the three novelty claims of the kt-mirt paper.
Verdict scale per threat: KILLS (claim false as worded), WEAKENS (claim must be
rescoped or the ingredient is prior art), NEIGHBORS (adjacent, cite and
distinguish). "Verified" means the source was confirmed to exist and say what
we attribute to it; PLAUSIBLE flags an attribution needing a full-text read.

The three claims under attack:

- **C1.** Certifying a growth-existence detector on matched synthetic twins
  with permutation nulls before any real-data claim.
- **C2.** A signed (facilitation and interference) cross-skill association
  readout with a measured detectable-dose floor.
- **C3.** Refusals (per-skill resolution refused, saturation refused, causal
  reading refused after an order-shuffle arm) as first-class results.

## Highest-severity threats

### T1. LTKT: positive and negative learning transfer in KT

Xu, Tang, Lv, Yu, Yu, Chen. *LTKT: Knowledge Tracing Based on Positive and
Negative Learning Transfers.* Tsinghua Science and Technology (SSRN 4630827;
sciopen 10.26599/TST.2024.9010201). Verified.

Claims to be "the first attempt to concurrently utilize the positive and
negative learning transfer relations among concepts" in KT. Builds a signed
transfer graph and propagates practice effects through it.

**Touches C2. KILLS** any wording of the form "first to model signed
cross-skill transfer in KT". **Does not kill** the actual claim if scoped
correctly: LTKT constructs the signed graph as an architectural input to boost
prediction and never validates the signs against ground truth, reports no
detectable-dose floor, and makes no measurement claim about the graph. Our
contribution is the certified *readout* with a measured floor, not the
existence of signed transfer modeling. Must cite; must not claim sign priority.

### T2. HawkesKT: temporal cross-effects between skills

Wang, Ma, Zhang, Lv, Wan, Lin, Tang, Liu, Ma. *Temporal Cross-Effects in
Knowledge Tracing.* WSDM 2021, pp. 517-525. Verified to exist; signedness of
its excitation parameters is PLAUSIBLE, not confirmed (PDF fetch 404'd).

Estimates per-pair cross-skill temporal impacts (mutual excitation plus decay
kernel) and presents them as interpretable. Hawkes excitations are classically
nonnegative, but the parameterization may admit negative entries for incorrect
responses.

**Touches C2. WEAKENS** the "readout of cross-skill influence" part; the
matrix-of-pairwise-influences readout exists. No ground-truth certification, no
dose floor, prediction-validated only. ACTION: read the full PDF (ACM DL
10.1145/3437963.3441802) before camera-ready; if its cross-effects are signed,
C2's novelty rests entirely on certification plus the dose floor.

### T3. Moment-by-moment learning detection, P(J)

Baker, Goldstein, Heffernan. *Detecting Learning Moment-by-Moment.* IJAIED
2011 (10.3233/JAI-2011-015). Verified.

A machine-learned detector of the probability that a student learned a skill
at a specific step; used to classify gradual versus eureka learning.

**Touches C1. WEAKENS** any wording of the form "first detector of learning
existence in log data". The detector exists; what does not exist there is
certification. P(J) is distilled from BKT labels, never validated against
synthetic ground truth, has no permutation null, and no gate before real-data
use. Cite as the nearest detector ancestor; claim the certification protocol,
not the detector concept.

### T4. AFM / LFA learning-curve slopes

Cen, Koedinger, Junker. *Learning Factors Analysis.* ITS 2006. Plus Koedinger
et al., *An Astonishing Regularity in Student Learning Rate*, PNAS 2023
(iAFM across 27 datasets). Verified.

Per-skill learning-rate parameters with standard significance tests: inference
about the existence of per-skill growth is two decades old, and the 2023 paper
does it at scale.

**Touches C1. WEAKENS** "growth detection is new". The residual novelty is
exactly the certification: AFM slopes are trusted on parametric assumptions,
never certified against matched twins or permutation nulls, and the
learning-curve attrition-bias literature (Nixon et al., AIED 2018) shows those
slopes can be artifacts, which argues *for* our protocol. Cite both sides.

### T5. Identifiability of student-model parameters

Beck, Chang. *Identifiability: A Fundamental Problem of Student Modeling.* UM
2007. van de Sande, *Properties of the BKT Model*, JEDM 2013. Doroudi,
Brunskill, EDM 2017 (identifiable under mild conditions; local optima).
Verified.

Observed performance is compatible with families of parameter estimates that
predict identically but make different knowledge claims.

**Touches C3 (and C2's floor). WEAKENS** "per-skill resolution refused" as a
novel *finding*: the field has known since 2007 that some KT readouts are
undetermined by data. Our refusal must be presented as a certified,
dataset-specific verdict from the twin protocol (this dataset, this model,
this resolution refused at this dose), not as the discovery that
non-identifiability exists.

### T6. How deep is knowledge tracing?

Khajah, Lindsey, Mozer. EDM 2016 (arXiv 1604.02416). Verified.

Shows DKT's advantage is explained by generic ability, recency, and
contextual factors rather than per-skill tracking; BKT variants match DKT.

**Touches C3. WEAKENS** the content of the per-skill-resolution refusal; a
version of "the model is not tracking skills" is published. The refusal
*framing* (pre-registered detector, certified negative verdict published as a
deliverable) remains ours. Must cite prominently.

## Methodology-ingredient threats (certification, nulls, power)

### T7. Simulation-based calibration and the Bayesian workflow

Talts, Betancourt, Simpson, Vehtari, Gelman. arXiv 1804.06788. Verified.

Validate the inference machinery on data simulated from the model before
trusting real-data inferences.

**Touches C1. WEAKENS** "certify before real-data claims" as a general
methodological invention; it is established statistics. Our novelty is the
transplant into KT growth detection with *matched* twins (real-data-calibrated
synthetic replicas differing only in the presence of growth), which SBC does
not do. Cite as methodological lineage.

### T8. Sanity checks for interpretability readouts

Adebayo et al. *Sanity Checks for Saliency Maps.* NeurIPS 2018 (and the
Revisiting follow-up, NeurIPS 2021). Verified.

Randomization-based null tests that a readout must pass before being trusted;
many popular readouts fail.

**Touches C1 and the paper's whole posture. NEIGHBORS/WEAKENS**: "certify the
readout, not the model" exists in XAI. No KT, no growth, no dose floor. This
is the right citation to anchor the audit posture in ML terms (consistent
with the DKT-home framing).

### T9. Permutation nulls and negative controls

Standard practice: sklearn permutation_test_score; permutation tests for
confounding (Chaibub Neto et al.); negative-control outcomes in epidemiology
(Lipsitch et al. 2010); placebo tests in causal inference. Verified as a
practice, no single canonical cite needed.

**Touches C1 and C3. WEAKENS** the ingredients: a permutation null is not
novel, and the order-shuffle arm is a textbook negative-control design. The
assembly (permutation null gating a growth claim in KT; shuffle arm gating a
causal reading of cross-skill associations) is what we can own. Say so
explicitly rather than letting a reviewer say it first.

### T10. Power to detect change, and detectable-effect floors

Hertzog, Lindenberger, Ghisletta, von Oertzen (2006, 2008): power of LGCMs to
detect correlated change and individual differences in change; Rast & Hofer
2014. Card et al., *With Little Power Comes Great Responsibility*, EMNLP 2020
(minimum detectable effect in NLP benchmarks); Bloom's MDES tradition in
program evaluation. Verified.

**Touches C2 (and C1). WEAKENS** "measured detectable floor" as a concept:
psychometrics has quantified floors for detecting growth for twenty years, and
NLP has imported MDE. Application to a *cross-skill association readout of a
deep KT model* appears unclaimed. Scope the claim to that object; cite Hertzog
and Card as the two lineages.

### T11. Synthetic ground truth in KT evaluation

SPARFA-Trace (Lan et al. 2014): recovery of knowledge states on synthetics.
arXiv 2401.16832 (2024): KT performance on synthesized students, verified to
be about train-time data substitution, not detector certification. Prerequisite
structure discovery sims (arXiv 2402.01672). Verified.

**Touches C1. NEIGHBORS**: parameter recovery on synthetic data is routine.
None certify a *detector* with a null distribution and none gate real-data
claims on the outcome. The "matched twin" construction (same interface
statistics as the target dataset, growth switched on/off) is the
distinguishing device; keep that term front and center.

## Cross-skill structure neighbors (C2)

### T12. Influence and prerequisite graphs extracted from KT models

Piech et al. 2015 (DKT influence graph), GKT (Nakagawa et al. 2019), SKT,
Prerequisite-Driven DKT (Chen et al. 2018), PSI-KT (Zhou et al., ICLR 2024,
arXiv 2403.13179). Verified.

**Touches C2. NEIGHBORS**: extracting inter-skill structure from KT models is
a crowded lane, but it is prerequisite/similarity-shaped (effectively
nonnegative) and prediction-validated. PSI-KT is the strongest of these
(interpretability by design, inferred prerequisite graph) and still has no
interference sign and no dose floor. Cite PSI-KT as the strongest baseline
reading.

### T13. Causal discovery aspirations in KT

*A Conceptual Model for End-to-End Causal Discovery in Knowledge Tracing*
(arXiv 2305.16165). Verified.

**Touches C3. NEIGHBORS**: the field wants causal readings of skill graphs.
Our order-shuffle refusal is the counterpoint: we test the causal reading and
decline it. Useful contrast citation, no overlap in method.

## Refusal-framing neighbors (C3)

### T14. Null-results and severe-testing culture

Mayo's severe testing; *Nothing's plenty: null results in physics education
research* (arXiv 1810.10071); negative-results workshops in CV/ML. No
EDM-specific refusal protocol found; EDM/LAK pre-registration culture is thin
(nothing substantive surfaced). Verified as a landscape.

**Touches C3. NEIGHBORS**: publishing negatives has a movement behind it, but
"refusal as a certified, pre-registered deliverable of an audit protocol" in
student modeling did not surface anywhere. This is the most defensible part of
C3. Cite Mayo for the testing philosophy.

### T15. Argument-based validity (Kane)

Kane's interpretation/use arguments: validation as an explicit chain of
licensed inferences. Verified.

**Touches C3. NEIGHBORS**: a refusal is an explicitly unlicensed inference,
so C3 is Kane's program operationalized for deep KT readouts. Engaging Kane
converts a possible "this is just validity theory" objection into lineage. Do
not frame the paper as a psychometrics-theory contribution (framing memory);
one paragraph of lineage suffices.

### T16. Ceiling and saturation effects

Measurement ceilings are classical psychometrics; wheel-spinning (Beck & Gong
2013) detects failure to learn; our own parked trajectory program found human
KT rate unrecoverable near saturation. Verified.

**Touches C3. WEAKENS** "saturation refused" as a surprising fact; ceilings
are old news. The certified, dataset-specific saturation verdict from the twin
protocol is the claimable part.

## Non-threats checked and cleared

- **Growing Pains (arXiv 2604.12843)** is *LLM benchmark calibration via
  fixed-parameter MIRT anchoring*, relevant to the thesis's ability-tracking
  arc, not to these three claims. The memory note calling it the nearest
  analog predates this paper's claim set; it does not touch C1-C3.
- **Label-leakage / order-sensitivity papers** (arXiv 2403.15304, 2508.17092)
  concern evaluation hygiene and input encodings, not certified refusals.
- **LGCM measurement invariance** literature concerns scale stability across
  time/groups in SEM; it motivates rather than preempts C1.

## Rescoping directives (what survives, what must change)

1. Never write "first to detect learning/growth" (T3, T4) or "first signed
   cross-skill model" (T1, T2). Both are dead as worded.
2. C1 survives as: first *certification gate* for a growth-existence readout
   in KT, using matched synthetic twins and permutation nulls, with real-data
   claims conditional on passing. No prior work assembles this.
3. C2 survives as: first cross-skill association readout that is signed *and*
   certified *and* accompanied by a measured detectable-dose floor. The triple
   conjunction is the claim; each conjunct alone is prior art (T1, T2, T10,
   T12). Confirm HawkesKT signedness before camera-ready (T2 action).
4. C3 survives as: first student-modeling paper whose pre-registered protocol
   makes refusals first-class deliverables. The *contents* of two refusals
   have published ancestors (T5, T6, T16) and must be cited as such; the
   order-shuffle arm must be named a negative control (T9).
5. Mandatory citations this sweep adds: LTKT, HawkesKT, P(J), AFM/iAFM,
   Beck & Chang, Khajah et al., Talts et al., Adebayo et al., Hertzog et al.,
   Card et al., PSI-KT, Lipsitch et al., Kane, Mayo.
