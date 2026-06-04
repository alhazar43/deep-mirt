# DRL-MAIRT Track Recommendation, Path B

This is the Path B working document. It covers recommendation targets
that go beyond the next assessment item (courses, assignments,
vocational categories, tutorials) under the standing data constraint
that the project has no logged recommendation outcomes. Path B is the
weaker of the two project paths in the
[evidence synthesis](drl_mairt_evidence.md) and the
[plan synthesis](drl_mairt_synthesis.md), and this document does not
attempt to disguise that.

The defensibility ceiling for Path B is bounded, and stated up front.
Without outcome logs, no claim of the form "our recommender improves
career outcomes" or "our recommender improves learning gain" is
supportable. The strongest defensible claims are about the inputs (the
measurement quality of the deep-IRT belief), about the mapping (a
construct-validated rule that translates belief into recommendation),
and about the policy's behavior inside the simulator that the policy
was trained against (with cross-simulator robustness as the integrity
check). Anything beyond that requires data the project does not have.

Path B has two sub-options. Section 2 treats theory-driven mapping,
the stronger of the two, where validity attaches to an independently
calibrated mapping. Section 3 treats model-based RL with bounded
claims, the fallback when theory-driven mapping is not enough on its
own. Based on the literature this document leans toward theory-driven
mapping (Section 2 is the heavier section) because the published
mappings are richer than the model-based-RL methodology literature for
education.

## 1. Positioning

Path A (the headline of [drl_mairt_evidence.md](drl_mairt_evidence.md))
is computerized adaptive testing with a deep-IRT belief state. Its
reward is an intrinsic psychometric quantity (Fisher information,
posterior uncertainty reduction, expected score change), so it does
not need recommendation-outcome logs. Path B asks what is left when
the recommendation target is something other than the next assessment
item, an assignment, a tutorial, a course module, an occupation list.

Three honest observations frame this document.

First, when the target is content remediation tied to a skill graph,
Path B can be done as a measurement-and-mapping problem rather than as
an RL problem. The skill mastery is observed (CDM or IRT readout), the
prerequisite graph is curriculum-expert validated, and the
recommendation rule (outer fringe, weakest-skill remediation) is
defined ex ante. This is Section 2.5 to 2.7. Validity attaches to the
mapping, not to logged behavior.

Second, when the target is vocational, the same logic applies through
Holland congruence applied to O\*NET occupational codes. Validity
attaches to a decades-old meta-analytic relationship between interest
congruence and job satisfaction. The relationship is real but its
effect size is modest (recent estimates closer to r equals 0.19, see
Hoff, Song, Wee, Phan, & Rounds, 2020, Journal of Vocational Behavior
123, 103447), which itself becomes a quantitative ceiling on any
claim. This is Section 2.2 to 2.4.

Third, when the target is none of the above and the project still
wants to learn a policy, the only remaining option is a simulator that
ma-irt itself defines, with the bounded claim that the contribution is
methodological rather than a deployment result. This is Section 3.
Section 3 grows if the user wants RL specifically; Section 2 is the
preferred answer if the user is willing to settle for a measurement
plus mapping contribution.

The plan-synthesis hybrid leaves Path B as a follow-on to Path A. This
document keeps that ordering. If the team is choosing between Path A
and Path B in isolation, Path A is the correct choice on every
defensibility axis. Path B exists to (a) extend the published
contribution if Path A succeeds, and (b) cover the user's stated
recommendation use cases (course, career) within the data the project
actually has.

## 2. Theory-driven mapping

This is the stronger of the two Path B options. The principle, the
specific vocational and content mappings, the architecture extension
to ma-irt, and the validation strategy follow.

### 2.1 The principle

A theory-driven mapping is a function from a measured latent state to
a recommendation that is justified by an independently validated
substantive theory rather than by logged behavior. The validity
argument follows Cronbach and Meehl (1955, Psychological Bulletin
52(4), 281, 302), construct validity by convergent and discriminant
evidence and nomological-network alignment. Messick (1989, in
Educational Measurement, 3rd ed.) extends this to consequential
validity, which Path B must address explicitly because educational and
vocational recommendations have material consequences for the
recipient.

Two operational forms apply to this project.

- Skill-mastery to content. The latent state is a per-skill mastery
  vector. The mapping is a prerequisite graph plus a coverage rule
  (Q-matrix). Validity attaches to (a) the Q-matrix, validated against
  curriculum and against data (Section 6.3), and (b) the prerequisite
  graph, validated by curriculum experts.
- Interest profile to occupation. The latent state is a RIASEC
  six-vector. The mapping is Holland congruence to occupational codes
  in O\*NET. Validity attaches to the congruence-satisfaction
  relationship in the vocational psychology literature.

Both forms produce a recommendation without observing what happened
after the recommendation. That is the entire point.

### 2.2 RIASEC Holland congruence

Holland (1997, Making Vocational Choices, 3rd ed., Psychological
Assessment Resources) defines six interest types (Realistic,
Investigative, Artistic, Social, Enterprising, Conventional) arranged
on a hexagon. Person-environment congruence is the substantive
construct, scored as the agreement between the person's three-letter
Holland code and an occupation's three-letter Holland code (Rounds and
Tracey, 1993, Journal of Vocational Behavior 43, 207, 230).

Three congruence indices dominate the literature.

| Index | Reference | Range | Notes |
|---|---|---|---|
| Iachan | Iachan, 1984, Applied Psychological Measurement 8(2), 133-141 | 0 to 28 | Sums weighted matches by ordered letter position. Reciprocal of a distance |
| Brown-Gore C | Brown and Gore, 1994, Measurement and Evaluation in Counseling and Development 26, 178-186 | 0 to 18 | Sensitive to code ordering and to hexagonal adjacency. Often recommended as the default |
| Kwak-Pulvers | Kwak and Pulvers, 2003 (modification of C for unequal-length codes) | 0 to 18 | Handles 2-letter vs 3-letter codes cleanly |

The Brown-Gore C index is the current default in modern applications
because it reflects the hexagonal structure and is sensitive to the
order of letters in the code, while remaining cheap to compute. Iachan
remains widely cited. Robust practice is to compute at least two
indices in any study (the rdrr.io documentation of the holland R
package summarizes practitioner guidance).

The empirical validity ceiling for congruence as a predictor of job
satisfaction is the load-bearing number for Path B. The literature is
unambiguous.

- Tranberg, Slane, and Ekeberg (1993, Journal of Vocational Behavior
  42(3), 253, 264) reported a mean congruence-satisfaction correlation
  near r equals 0.17 with a 95 percent CI that included zero in the
  unconditional analysis. Significant subgroup means ranged 0.29 to
  0.42 in moderator analyses.
- Tsabari, Tziner, and Meir (2005, Journal of Career Assessment 13,
  216, 232) updated the meta-analysis and reported r near 0.21.
- Hoff, Song, Wee, Phan, and Rounds (2020, Journal of Vocational
  Behavior 123, 103447) is the most recent comprehensive
  meta-analysis. They reported an interest-fit to satisfaction
  meta-analytic correlation of approximately 0.19. The relationship
  was statistically significant, the confidence interval excluded
  zero, and the strongest moderators were the type of satisfaction
  (intrinsic > extrinsic) and the type of fit measure.

The honest number to cite is r approximately 0.19, with the qualifier
that it is statistically robust but practically small. Hoff et al.
characterize the literal headline as "interest is not a big predictor
of job satisfaction" (see the associated ScienceDaily summary). This
is the validity ceiling that any RIASEC-based recommender from Path B
inherits. The right framing for the contribution is "our recommender
recovers a known small but real signal more efficiently from fewer
items via deep-IRT measurement" rather than "our recommender helps
people find the right occupation".

### 2.3 O\*NET Interest Profiler

The O\*NET Interest Profiler is the public-domain instrument that
makes the RIASEC-to-occupation pipeline implementable without
proprietary licensing. The relevant references are Lewis and Rivkin
(1999, O\*NET Interest Profiler Manual), Rounds, Smith, Hubert, Lewis,
and Rivkin (1999, Development of Occupational Interest Profiles for
O\*NET), and Rounds, Su, Lewis, and Rivkin (2010) for the 60-item
short form. The long form has 180 items, the short form has 60, and a
mini-IP exists at 30 items. Cronbach alpha is reported in the 0.78 to
0.85 range across the six RIASEC scales on the short form, 0.83 to
0.93 on the long form. Items, scoring keys, manual, and psychometric
documentation are downloadable from onetcenter.org.

For this project the relevant fact is that the IP is fully public
domain in the United States. Items, scoring rules, and the
occupational coding tables (the O\*NET Occupational Interest Profile
for thousands of occupations) are downloadable as machine-readable
files. There is no licensing impediment.

Three engineering notes follow.

- An IRT calibration is not officially distributed by the O\*NET
  Resource Center, but the response data and items are public, so a
  GRM or 2PL calibration is straightforward to refit from any
  collected sample. The development methodology used iterative
  multidimensional scaling rather than IRT, so any IRT calibration is
  a project-level decision rather than something to inherit.
- The ideal-point variant (GGUM, Roberts, Donoghue, and Laughlin,
  2000, Applied Psychological Measurement 24(1), 3, 32) is a plausible
  alternative for interest items, which have a clearer ideal-point
  structure than achievement items. If the project produces an
  ma-irt-with-ideal-points variant, the IP is a natural testbed and
  the contribution is itself publishable as a measurement paper.
- Short-form selection via CAT is a known win for IP-style banks. The
  short-form manual reports comparable reliability to the long form
  with two-thirds fewer items, and an adaptive selection should
  improve further. This is the H-style "ordinal CAT for vocational
  interests" sub-paper.

A 2025 instrument extension worth citing as future-work direction is
the Comprehensive Assessment of Basic Interests, O\*NET edition
(CABIN-NET), Chu, Hoff, Liu, Heimpel, Greco, Oswald, and Rounds
(2025, Journal of Career Assessment), which adds 20 basic interest
scales nested inside the six RIASEC types. The CABIN-NET items are
available with documentation and offer richer coverage at the cost of
60 additional items.

### 2.4 Big Five and trait-occupation fit

The Big Five Factor Model (Costa and McCrae, 1992; Goldberg, 1992)
contributes a complementary axis to RIASEC. The classic meta-analytic
references are Barrick and Mount (1991, Personnel Psychology 44, 1,
26) showing Conscientiousness predicts job performance across all
occupational groups (rho about 0.22), and Hurtz and Donovan (2000,
Journal of Applied Psychology 85(6), 869, 879) replicating with
specific factor structure. Judge, Higgins, Thoresen, and Barrick
(1999, Personnel Psychology 52, 621, 652) extend this to career
success outcomes. For interest fit specifically, Barrick, Mount, and
Gupta (2003, Personnel Psychology 56(1), 45, 74) report Big Five to
RIASEC overlap in the moderate range.

For this project, Big Five via IPIP (the public-domain
International Personality Item Pool) is an adjunct, not a
replacement, for RIASEC. The incremental validity of Big Five over
RIASEC for interest-fit purposes is small. The right framing is to
include a Big Five readout to support a secondary "trait-occupation
fit" axis (Conscientiousness for clerical, Extraversion for sales,
Openness for artistic) alongside the primary RIASEC congruence
ranking. The Skills Confidence Inventory (Betz, Borgen, and Harmon,
1996, Journal of Career Assessment 4(4), 413, 424) and the Career
Decision Self-Efficacy Scale (Betz, Klein, and Taylor, 1996, Journal
of Career Assessment 4(1), 47, 57) sit in a similar adjunct role,
adding self-efficacy axes to interest measurement. The CDSE-SF has
been IRT-calibrated (e.g., Carbonero Martin and Merino Tejedor in a
Portuguese sample) and is freely available for non-commercial research
with registration.

A clean architectural separation is to keep ma-irt's primary head as
the interest measurement and to add a small secondary readout for Big
Five or self-efficacy when the bank includes those items. This means
the recommender can decompose its ranking explicitly into an interest
congruence term and a trait-fit term, both reportable separately.

### 2.5 Knowledge state to prerequisite graph

For content recommendation (assignments, tutorials, course modules),
Knowledge Space Theory (Doignon and Falmagne, 1985, International
Journal of Man-Machine Studies 23(2), 175, 196; Doignon and Falmagne,
1999, Knowledge Spaces, Springer) supplies the formal structure. A
knowledge state is a subset of items the learner has mastered. The
knowledge structure is the family of admissible states. The outer
fringe of a state is the set of items the learner can next master
without jumping over an unmastered prerequisite. The outer-fringe
policy minimizes expected steps to a target state under the KST
axioms, which makes it the closest classical analogue to a
recommendation policy.

The ALEKS system (Falmagne, Albert, Doble, Eppstein, and Hu, 2013,
Knowledge Spaces, Applications in Education, Springer) is the
operational precedent. ALEKS is proprietary but its design is fully
documented and its empirical track record (Doignon and Falmagne, 2015
practical perspective paper, Journal of Mathematical Psychology) makes
it the canonical reference for KST in production. ALEKS is not the
template for this project because the project is open, but ALEKS is
the existence proof that the mapping works at scale.

Open-source KST tooling is thinner. The Liu et al. CSEAL paper
(Exploiting Cognitive Structure for Adaptive Learning, KDD 2019)
implements the deep-learning analogue with an actor-critic on a
GRU-traced knowledge state and a prerequisite-graph navigator. The
CSEAL code is not publicly released. The Junyi prerequisite graph
(Chang, Hsu, and Chen, 2015 via PSLC DataShop) is the principal
publicly available curriculum-annotated graph. The MOOCCube dataset is
a recent public alternative with course-level prerequisite structure.
A 2025 line of work (Education-Oriented Graph Retrieval-Augmented
Generation for Learning Path Recommendation, arXiv 2506.22303;
Personalized Learning Path Recommendation on Knowledge Graphs survey,
MDPI Electronics 15(1), 238) shows that GraphRAG-style approaches are
the contemporary frame but they require curated graphs.

For Path B, the practical decision is to use Junyi as the
prerequisite graph and to formalize the recommendation rule as the
outer-fringe restriction of an MA-GPCM-derived skill mastery vector.
This needs no recommendation-outcome logs because the rule is
ex ante. The contribution then is methodological, "MA-GPCM provides a
calibrated skill-mastery readout that powers an outer-fringe content
recommender", and the comparison is against simpler readouts
(Section 5).

### 2.6 Cognitive diagnosis models

Cognitive diagnosis models (CDMs) produce a discrete or continuous
skill-mastery vector indexed by a Q-matrix. The relevant references
are DINA (Junker and Sijtsma, 2001, Applied Psychological Measurement
25(3), 258, 272; de la Torre, 2009, Journal of Educational and
Behavioral Statistics 34(1), 115, 130) for the non-compensatory
AND-gate, NIDA (Maris, 1999, Psychometrika 64(2), 187, 212) for the
noisy-input variant, G-DINA (de la Torre, 2011, Psychometrika 76(2),
179, 199) for the general parametric family that subsumes DINA, DINO,
ACDM, LLM, and RUM, FuzzyCDF (Liu, Wu, Chen, Wu, Chen, Hu, and Su,
2018, ACM TIST 9(4), 48) for continuous fuzzy mastery on objective
and polytomous items, and NeuralCDM and NeuralCD (Wang, Liu, Chen,
Huang, Wu, Wang, and Hu, 2019, AAAI; Wang, Liu, Chen, Huang, Wu, and
Wang, 2022, IEEE TKDE) for the monotonic-neural-network interaction
function with Q-matrix preservation.

For Path B the operational point is that the user's DKVMN backbone is
structurally close to NeuralCDM. Two architectural extensions follow.

- Add a Q-matrix-aware skill-mastery readout alongside the existing
  GPCM theta, alpha, beta readouts. This is the psychometric agent's
  strongest architectural recommendation (see Section 2.7).
- Treat FuzzyCDF as the published prior art for continuous-mastery
  CDMs on polytomous items. FuzzyCDF and ma-irt share the move from
  binary mastery to continuous proficiency on ordinal data, so
  citing FuzzyCDF as a baseline avoids reviewer pushback about
  novelty.

The EduCDM toolkit (bigdata-ustc) ships reference implementations of
IRT, MIRT, DINA, FuzzyCDF, NeuralCDM, and IRR, which is the most
direct reusable code base for the baseline comparison.

### 2.7 Architecture extension to ma-irt

The concrete extension recommended by the psychometric survey is to
add a Q-matrix-aware skill-mastery readout alongside the existing
GPCM theta, alpha, beta head. This is small, additive, and changes no
existing behavior when the new readout is unused.

Files and structure assuming the existing
`C:/Users/steph/documents/deep-mirt/ma-irt/` layout.

```
ma-irt/
  models/
    components/
      irt.py            # existing, contains IRTParameterExtractor, GPCMLogits
      cdm.py            # NEW, SkillMasteryReadout (NeuralCDM-style)
    magpcm.py           # existing, gains an optional skill_readout argument
  configs/
    bulk/
      *_cdm_*.yaml      # NEW, configs that enable the readout for the
                        # course-assignment evaluation
  scripts/
    evaluate.py         # existing, gains a Q-matrix-aware mastery
                        # evaluation block
```

Readout sketch.

```python
# models/components/cdm.py
class SkillMasteryReadout(nn.Module):
    """
    Maps the DKVMN/MA-GPCM summary vector to a per-skill continuous
    mastery alpha in [0, 1]^K under a NeuralCDM-style monotonic
    interaction.

    Q is the (n_questions, n_skills) Q-matrix as a buffer.
    """
    def __init__(self, hidden_dim, n_skills):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, n_skills)

    def forward(self, summary):
        # summary: (B, T, hidden_dim)
        return torch.sigmoid(self.proj(summary))  # (B, T, n_skills)
```

The interaction layer that turns mastery and Q-matrix into a response
probability uses the NeuralCDM monotonic positive-weight constraint
(Wang et al., 2019, AAAI). The full ma-irt forward step then yields
both the GPCM-headed probability and the CDM-headed probability per
item, both supervised jointly under a small auxiliary loss term. The
existing GPCM head is the headline output, the CDM head is the
mastery probe that powers the content recommender.

Reward and validation are kept identical to Section 2.5. The
recommender ranks items by combined "PWKL on the lowest-mastery skill"
and "outer-fringe membership in the prerequisite graph". No logged
outcome is required.

## 3. Model-based RL with bounded claims

This is the fallback if the project insists on training a policy
rather than fixing the policy as a rule. Section 3 is shorter than
Section 2 because the published literature on bounded-claim
methodology for education-recommendation RL is thinner than the
theory-driven mapping literature. The structure below is what is
defensible.

### 3.1 The principle

A policy trained inside a simulator is a methodological contribution
about the simulator, not a real-world deployment claim. This frame is
the only one that survives the absence of logged recommendation
outcomes. The contribution is "we show that a stable, sample-efficient
policy can be learned inside a learned IRT world model, with X
robustness properties verified by cross-simulator validation". The
contribution is not "this policy improves learning" because that
quantity cannot be measured.

The contrast point is Bassen, Howley, Fast, Mitchell, and Goel (2020,
CHI 2020), the only real-world online RL deployment in education to
date. Bassen et al. ran an active-learning RL scheduler in a large
online programming course (Stanford Code in Place) and reported real
learning-gain and time-on-task improvements. This project has none of
the conditions Bassen et al. had (live course, IRB, partner platform,
within-session randomization). The honest framing is that Path B's
model-based-RL sub-path is methodological in the sense that ExRec
(Ozyurt et al., 2025, arXiv 2507.11060) is methodological, ExRec
treats the KT model as the simulator and reports a battery of RL
algorithms run inside it, with no live deployment.

The wording template for the abstract follows. "We train a policy
inside a calibrated MA-GPCM world model and report its behavior under
X. We make no claim about deployed learning gain. We provide a
cross-simulator robustness check (Section 3.2) as the integrity
proxy that the policy is learning a property of the data rather than
exploiting a single simulator's idiosyncrasies."

### 3.2 Cross-simulator validation

The single integrity check that converts simulator-based RL into a
defensible methodological contribution is cross-simulator validation.
Train the policy inside ma-irt with the DKVMN encoder, evaluate the
policy inside ma-irt with the Transformer encoder trained on the same
data with a different seed (or with AKT or SAINT or a Transformer
ablation). A policy that exploits encoder-specific idiosyncrasies fails
this check.

The clearest current treatment of the gap that motivates this check is
AdvKT (Liu, Liu, Sun, Yao, Wu, Liu, Huang, and Wang, 2025, arXiv
2504.04706, ECML-PKDD 2025), which names the single-step-training to
multi-step-inference gap as the dominant source of cross-simulator
disagreement and proposes adversarial multi-step training as the
mitigation. AdvKT is the methodological reference to cite when
explaining the need for cross-simulator validation.

Operationally for ma-irt, the cross-simulator pair is

- Train sim. MA-GPCM with DKVMN encoder, fit on ASSISTments 2009.
- Eval sim. MA-GPCM with Transformer encoder, fit on the same split
  with a different seed (or AKT, also on the same split).

The policy is held fixed across the two sims. Report (a) the policy's
reward profile under the train sim, (b) its reward profile under the
eval sim, (c) the disagreement magnitude. Anything above a small
threshold disagreement disqualifies the policy as having learned a
data-property rather than a sim-property.

### 3.3 Candidate simulator-evaluable rewards

ExRec (Ozyurt et al., 2025) names four rewards that are evaluable
inside a KT simulator without any outcome log.

| Reward | Definition | Gameability | Suitability |
|---|---|---|---|
| Global KC improvement | Average mastery delta across all KCs | High, policy can spam already-mastered items | Weak |
| Practiced KC | Mastery delta only on KCs touched by the action | Moderate, biased by item coverage | Medium |
| Upcoming KC | Mastery delta on KCs scheduled by the curriculum | Low, requires a curriculum signal | Medium-strong, if curriculum is available |
| Weakest KC | Mastery delta on the lowest-mastery KC | Lowest, the policy is forced to target the bottleneck | Strongest |

The recommendation is "weakest KC" as the primary, with "global KC
improvement" as the gameability anti-baseline. Weakest-KC reward is
also the simplest to defend in writing because it forces the policy to
target the most informative remediation rather than the easiest reward
patch.

For ordinal data (MA-GPCM), the analogue is "weakest-skill expected
score delta" computed over the K-1 cumulative logits. This is novel,
existing weakest-KC formulations assume binary responses.

### 3.4 Bounded claim template

The bounded claim template is straightforward.

- May claim. The policy, trained inside simulator S1, achieves reward
  R when evaluated under simulator S2 (cross-simulator). The
  disagreement S1-versus-S2 is bounded by delta.
- May claim. The policy outperforms baselines B1 to Bn on R under both
  simulators.
- May not claim. The policy improves learning. This is a deployment
  outcome and the project has no deployment data.
- May not claim. The policy improves career outcomes. Same reason.
- May not claim. The policy is safe for live use without further
  validation, including IRB review, dosage controls, and a randomized
  exposure protocol.

This bounded form is consistent with the broader literature on
deployment-efficient RL (Matsushima, Furuta, Matsuo, Nachum, and Gu,
2021, ICLR 2021, Deployment-Efficient Reinforcement Learning via
Model-Based Offline Optimization, arXiv 2006.03647) and with recent
sim-to-real survey work (Da, Yuan, Pang, Liu, Zhao, Wei, Mei, Li,
Zheng, Han, Wang, and Li, 2025, A Survey of Sim-to-Real Methods in
RL, arXiv 2502.13187), which both name the same disclosure
requirements. ExRec's published treatment is the closest positive
template within education and should be cited as the precedent.

### 3.5 Validation

Without ground truth on recommendation outcomes, the validation suite
reduces to.

- Cross-simulator agreement (Section 3.2). The integrity check.
- Held-out KT prediction parity. The frozen KT model used as the
  simulator must remain calibrated, predict held-out responses at the
  same accuracy a non-policy-coupled training run achieves.
- Item exposure and overlap reports (Sympson-Hetter 1985,
  Stocking-Lewis 1995). Mandatory for any selection-style policy.
- No live deployment claim. State explicitly.
- Q-matrix dependence reporting if the reward depends on KCs. State
  which Q-matrix was used and its validation status (Section 6.3).

A practical paragraph for the limitations section is "We do not claim
deployment results. The policy is evaluated entirely inside a learned
world model. Cross-simulator agreement is reported as a proxy for
the policy having learned a property of the data rather than of the
simulator. Translation to deployed learning gain requires a
randomized live trial, which is outside the scope of this work."

## 4. Datasets

### 4.1 Theory-driven mapping route

The theory-driven route does not need recommendation logs but it does
need (a) calibration data for the latent measurement (RIASEC interest
profile or skill mastery), and (b) an independently validated mapping
target (occupational coding or prerequisite graph).

| Item | Resource | License |
|---|---|---|
| RIASEC items, public bank | O\*NET Interest Profiler, 60 or 180 items | US public domain |
| RIASEC items, public bank, secondary | IPIP RIASEC markers | Public domain via ipip.ori.org |
| Big Five items, public bank | IPIP-NEO-120 or BFI-2 | Public domain or free for research |
| Self-efficacy adjunct | CDSE-SF (Betz, Klein, and Taylor, 1996) | Free for research with registration |
| Occupational coding target | O\*NET Occupational Interest Profiles | US public domain |
| Prerequisite graph | Junyi prerequisite graph (Chang, Hsu, and Chen, 2015 via PSLC DataShop) | Academic |
| Skill-tagged response data, K-12 math | ASSISTments 2009-2010 skill builder | Academic |
| Skill-tagged response data, K-12 math, ordinal | Eedi NeurIPS 2020 Education Challenge | CC BY-NC-SA 4.0 |
| Health item bank for cross-domain validation | PROMIS (NIH) | Public domain |

Three instruments that are not available are worth naming. The Strong
Interest Inventory (CPP / The Myers-Briggs Company) is proprietary
and not usable here. The Self-Directed Search (PAR) is proprietary.
The MVPI (Hogan Assessment Systems) is proprietary. These are
canonical references in the vocational literature but cannot be used
for an open project.

For the K-12 content recommendation evaluation, the working pair is
ASSISTments 2009-2010 plus the Junyi prerequisite graph, because
ASSISTments has the skill tags ma-irt needs and Junyi has the
expert-validated prerequisite structure the outer-fringe rule needs.
Eedi NeurIPS 2020 is the natural ordinal companion.

### 4.2 Simulator-RL fallback route

The simulator-RL route uses ma-irt itself as the simulator, so the
data requirements are the same as Path A. The primary set is
ASSISTments 2009, EdNet KT1, and Eedi NeurIPS 2020, the same three
datasets named in [drl_mairt_evidence.md](drl_mairt_evidence.md) for
Path A.

## 5. Comparison and baselines

### 5.1 Theory-driven mapping route

The reviewer-facing comparison must establish that ma-irt's deep-IRT
state provides a measurement quality that improves the downstream
recommendation. The honest baselines are.

| Baseline | Mechanism | Why include |
|---|---|---|
| Random recommendation | Uniform over candidates | Sanity floor |
| Popularity recommendation | Most-frequent target | Naive content baseline |
| Raw RIASEC sum scores | Score IP responses by classical CTT, rank occupations by congruence | The historic vocational baseline (this is what most counsellors and websites use) |
| 2PL CAT on RIASEC | Standard 2PL IRT fit on IP, then MFI selection | The minimal IRT baseline |
| MLE CDM mastery | DINA or G-DINA fit, classical EM, outer-fringe recommender | The classical CDM baseline |
| EduCDM NeuralCDM | Reference NeuralCDM implementation | The deep-learning CDM baseline |
| FuzzyCDF | Continuous-mastery CDM | The polytomous-CDM baseline, prior art for continuous fuzzy mastery |
| ma-irt with CDM readout | The proposal (Section 2.7) | The headline |

Evaluation metrics for the vocational route are construct-validity
proxies (convergent correlation with O\*NET-coded reference samples,
short-form to long-form rank-order correlation, person-fit l_z) and
recommendation-distribution proxies (Iachan or C congruence
distribution, exposure rates over the occupational coding). For the
content route, metrics are mastery-recovery (against held-out
responses), prerequisite-respect (the fraction of recommendations
inside the outer fringe), and exposure.

### 5.2 Simulator-RL fallback route

The simulator-RL route adopts the same baseline matrix as Path A
(Section 7 of [drl_mairt_evidence.md](drl_mairt_evidence.md)) trimmed
to the relevant subset.

| Baseline | Why |
|---|---|
| ExRec best variant | The closest positive template (Ozyurt et al., 2025) |
| NCAT | The dominant RL CAT precedent (Zhuang et al., 2022, AAAI) |
| Theta-only DQN | The CaRReL-stripped negative control |
| MPC (model-predictive control) on the learned world model | The small model-based baseline. Plan a 3-step rollout, pick the action with the largest weakest-KC expected gain |
| Greedy weakest-KC | The zero-RL heuristic. Beats most "RL" baselines at small data scales |

For the content recommendation sub-task specifically, the model-based
MPC baseline is the strongest comparison because it captures the same
"learned world model + planning" intuition without the learnability
overhead of full RL.

## 6. Risks and mitigations specific to Path B

### 6.1 Defensibility ceiling

Without recommendation-outcome data, the highest-value claim
available is about (a) measurement quality of the latent state,
(b) the construct validity of the mapping, and (c) sample efficiency
of the deep-IRT-headed CAT relative to baselines. Career or
content-outcome claims require longitudinal outcome data the project
does not have.

Mitigation. State the bounded claim up front, in the abstract and in
the limitations. The recent literature on the small but real
RIASEC-satisfaction correlation (Hoff et al., 2020, r approximately
0.19) supplies the quantitative ceiling. Cite it.

### 6.2 Ethical caveats

Career and major recommendations that are grounded in observed
occupational distributions inherit the demographic biases of those
distributions. RIASEC norms have historically over-recommended
Realistic to men and Conventional to women. Holland code distributions
across O\*NET occupations are not race-balanced. Any vocational
recommender from this project must report.

- Differential item functioning (DIF) analyses by gender and by other
  available demographic axes. The Mantel-Haenszel and SIBTEST procedures
  (Holland and Wainer, 1993; Penfield and Camilli, 2007) are the
  defaults.
- Measurement invariance across the simulated cohorts used to validate
  the model (Strobl, Kopf, and Zeileis, 2015, for tree-based DIF).
- Group-conditional recommendation distributions. If the
  Holland-congruence-ranked occupation list differs in expected income
  or status across demographic groups, this must be disclosed.

The AERA, APA, NCME (2014) Standards for Educational and
Psychological Testing apply, particularly the chapters on fairness in
testing and on educational testing for high-stakes decisions. For
youth vocational guidance specifically, validated youth-form
instruments (O\*NET IP youth norms) and informed-consent procedures
are required. This is not a research-method footnote, it is a binding
requirement on any deployment claim.

### 6.3 Q-matrix dependence

A CDM-driven content recommender stands or falls on the Q-matrix. A
mis-specified Q-matrix produces wrong mastery, wrong fringe, wrong
recommendations. The de la Torre and Chiu (2016, Psychometrika 81(2),
253, 273) PVAF-based empirical Q-matrix validation procedure is the
default, with the GDINA R package as a ready implementation
(`Qval` function). More recent reviews (Ma and de la Torre, 2020;
Najera et al., 2023, Behavior Research Methods) and machine-learning
approaches to Q-matrix validation (2023 BRM paper on ML-based Q-matrix
validation) are the working literature.

Mitigation. Treat the Q-matrix as a first-class artifact. Report it.
Report its validation status. Report a sensitivity analysis where the
recommendation distribution is recomputed under a perturbed Q-matrix
(flip 5 percent of the entries) and report the recommendation overlap
between Q and Q-perturbed.

### 6.4 Simulator exploitation in the fallback

The model-based-RL sub-path is exposed to simulator exploitation. The
policy can learn to maximize a reward that the simulator overstates,
or to take actions that lie outside the support of the training data
but happen to look attractive under the simulator's extrapolation.

Mitigation. Three measures together.

- Cross-simulator validation (Section 3.2).
- Restrict the policy's action support to items observed in the
  training data, or to items inside the held-out 10 percent item bank
  used for held-out-item generalization (the workflow's pointer-scorer
  test set).
- Never report a deployment claim. The bounded-claim template
  (Section 3.4) is a hard constraint, not a stylistic preference.

## 7. Publishability assessment

Path B is smaller than Path A on every defensibility axis.

- Theory-driven mapping sub-path. Honest assessment is that this could
  be a follow-on short paper at AIED, EDM, or a psychometrics venue
  (Journal of Educational Measurement, Educational and Psychological
  Measurement, Journal of Vocational Behavior for the vocational
  variant). The vocational variant in particular is more naturally
  placed at a vocational psychology venue than at IJAIED. The
  contribution is "MA-GPCM as a unified online belief that supports
  both CAT measurement and CDM-style content recommendation, evaluated
  on K-12 math" or "MA-GPCM-based ordinal CAT for RIASEC interest
  measurement, evaluated on the O\*NET IP". Both are defensible
  bounded claims.
- Model-based RL with bounded claims sub-path. Honest assessment is
  that this is a methodology paper, probably at RLC, RLJ, or a
  workshop track at AIED. The headline is "stable cross-simulator
  policy learning inside a learned IRT world model", not "improved
  educational outcomes". ExRec is the comparable precedent.

A two-paper plan is the cleanest. Path A is the IJAIED main paper. A
follow-up Path B paper at AIED short, EDM, or a psychometrics venue is
the natural extension. The two-paper plan is what
[drl_mairt_evidence.md](drl_mairt_evidence.md) Option B2 already names
as the stretch case, and the present document supports it as the right
ordering rather than collapsing Path B into Path A.

The honest negative case for Path B as a standalone paper is that
without outcome data the contribution depends entirely on the
measurement quality of the latent state and the construct validity of
the mapping, both of which the user already needs to defend for Path
A. Without an additional empirical anchor (a small randomized trial,
even at N equal to 200 to 500), Path B is a methodological extension
of Path A rather than a contribution on its own.

## 8. Decision points for the user

### 8.1 Theory-driven mapping. Which use case?

If theory-driven mapping is the chosen direction, the user picks one
of three.

- Vocational career recommendation. RIASEC plus Holland congruence
  plus O\*NET. Data is public domain, instrument is public domain,
  validity ceiling is r approximately 0.19 against job satisfaction.
  Best fit for a vocational psychology venue.
- Course or assignment recommendation. CDM plus prerequisite graph.
  Data is ASSISTments plus Junyi. Maps onto the existing ma-irt
  pipeline. Best fit for an education venue.
- Both. A unified MA-GPCM online belief that supports CAT measurement,
  CDM-style remediation, and RIASEC-style vocational match through a
  shared encoder with multiple readouts. This is the most ambitious
  framing and is what the Section 2.7 architecture extension actually
  enables.

The recommendation here is the course or assignment variant first,
because it reuses ma-irt's existing data pipeline and skill structure.
The vocational variant is a separate paper.

### 8.2 Model-based RL with bounded claims. Which simulator pair?

If model-based RL is the chosen direction, the cross-simulator pair
must be chosen.

- Train inside DKVMN-based MA-GPCM, evaluate inside Transformer-based
  MA-GPCM. The default. Both are inside ma-irt, both can be trained
  on the same data with different encoders. Minimum reviewer
  friction.
- Train inside DKVMN-based MA-GPCM, evaluate inside AKT (not in
  ma-irt, requires a separate KT implementation). Stronger
  cross-simulator signal because the architectures are more different.
  Higher engineering cost.
- Train inside DKVMN-based MA-GPCM, evaluate inside an AdvKT-trained
  variant. The strongest signal because AdvKT explicitly addresses
  the single-step-to-multi-step gap, but requires implementing AdvKT.

The recommendation here is the default cross-simulator pair (DKVMN
versus Transformer inside ma-irt) for the first paper, with the AKT
or AdvKT extension flagged as future work.

### 8.3 Whether Path B is pursued at all

The simplest answer to the data constraint is to commit fully to
Path A and defer Path B until real recommendation outcomes (a
randomized trial, even at modest N) become available. This is not a
failure of nerve, it is the correct response to the data constraint
the project actually has.

Three signals that pursuing Path B is the right call.

- The user wants a second paper from the same model. Path B as the
  follow-on at AIED short or EDM is the cleanest second-paper path.
- The user has a partner who can supply even a small randomized trial
  (N around 200 to 500 with a 4 to 8 week follow-up). This converts
  Path B from bounded-claim to outcome-claim, which is a much
  stronger paper.
- The user wants to publish on the vocational side specifically, where
  the validity argument lives in a different literature than the
  educational side and where Path A's CAT framing is less natural.

Three signals that deferring Path B is the right call.

- The Path A paper is not yet written, in which case Path B is a
  distraction.
- No randomized-trial partner is available.
- The user has no specific commitment to the vocational use case.

The default recommendation, absent additional information, is to
commit to Path A first and treat Path B as a planned but optional
follow-on. This is also the stance of
[drl_mairt_synthesis.md](drl_mairt_synthesis.md) (Section "Hybrid
phasing", H11 is optional), and the present document does not
contradict it.

## File reference

- [drl_mairt_background.md](drl_mairt_background.md), Codex's
  feasibility dossier.
- [drl_mairt_recommender_plan.md](drl_mairt_recommender_plan.md),
  Codex's proposal.
- [drl_mairt_synthesis.md](drl_mairt_synthesis.md), the plan-level
  synthesis.
- [drl_mairt_evidence.md](drl_mairt_evidence.md), the evidence-level
  synthesis (Path A primary).
- [drl_mairt_track_recommendation.md](drl_mairt_track_recommendation.md),
  this document (Path B).
- `docs/cleanup/_drl_research_digest.md`, the raw research outputs.
