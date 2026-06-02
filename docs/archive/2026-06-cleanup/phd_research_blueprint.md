# PhD Research Blueprint

Working document. Records the multi-round exploration of PhD thesis directions
that build on MA-GPCM. Each round is appended; nothing is overwritten so the
reasoning trail stays auditable.

## Context

- **MA-GPCM (foundation, IJAIED submission)**: DKVMN encoder + GPCM IRT decoder
  with a separated ability pathway, for ordinal knowledge tracing. Validates
  single-pass parameter recovery against a generating GPCM under static,
  discrete-shift, and continuous-drift conditions.
- **Existing PhD project (original scope)**: DRL-based vocational recommender
  for a company partner. Data is thin (one-shot self-reported interest
  questionnaires). The candidate feels this lacks PhD-level novelty.
- **Candidate's openness**: keep MA-GPCM as foundation, willing to explore
  "AI to IRT" or "IRT to AI" directions, limited time horizon.

## Round 1 (initial brainstorm + triage)

Two agents (research-scientist, psychometric-researcher) independently
brainstormed, self-critiqued, and refined. Detailed agent outputs preserved
in transcripts. Top-level convergences:

- **Company recommender should NOT be the spine.** Both agents recommended
  demoting it to a deployment chapter.
- **MA-GPCM is foundation, not spine.** Both agents framed it as one piece
  of a larger arc.
- **IRT-for-AI-evaluation is the highest-novelty direction both surfaced.**
  Polytomous IRT calibration of LLM benchmarks, with MA-GPCM's dynamic
  decoder enabling agentic evaluation.
- **Spine + secondary pillar structure**, not monolithic.

Three candidate spines from synthesis:

- **S1 — Measurement-first**: validity argument for neural IRT estimates;
  applications to ed decisions and AI evaluation. Strongest for strict
  psychometric reviewers.
- **S2 — Application-first**: IRT-grounded sequential representations for
  ed decisions and AI evaluation. Strongest community reach.
- **S3 — Two-sided dynamic IRT**: measuring both learners and learning
  systems with one framework. Most ambitious; placement risk.

User concern after Round 1: lacks diversity, novelty, scientific rigor.
Round 2 should extend breadth and depth, with explicit SOTA-paradigm scan.

## Round 2 (SOTA-driven blueprint, in progress)

### Research-scientist round 2

Round 1 stayed inside a narrow band: extensions of KT+IRT, plus one IRT-for-AI
direction. The honest critique is that the brainstorm did not really leave the
gravity well of MA-GPCM. Round 2 widens the aperture, scans the actual SOTA
landscape, and only then proposes an umbrella that has its own intellectual
identity rather than being "MA-GPCM and friends."

#### Section A — SOTA scan with gap analysis

I take 12 paradigms seriously. For each I name the SOTA outside education, the
SOTA inside education (where it exists), the specific gap, and whether the gap
is PhD-tractable in 3-4 years or institution-bound.

##### A1. Foundation models for educational interactions

- **SOTA outside ed**: GPT-style decoder-only transformers pretrained on
  trillion-token web mixtures, with sequence-modeling capability that
  generalizes across tasks via in-context learning. In adjacent fields:
  Behavior Transformers and Decision Transformer for action sequences;
  ESM/AlphaFold for biological sequences; Time-LLM/Lag-Llama for time series.
- **SOTA inside ed**: Two camps. (i) BERT-style pretraining of KT
  (BEKT, AKT, SAINT+, CL4KT). These pretrain on a single platform's
  click-through data with a masked-prediction objective and finetune on a
  downstream KT loss. (ii) LLM-as-tutor work (Khanmigo, MathGPT-style)
  where pretraining is on text, not interactions.
- **Gap**: There is no cross-platform "GPT for educational interactions."
  The community has not seriously answered (a) what is the right "token"
  for an interaction (response, response+latency, response+context, item
  embedding, skill tag), (b) what scaling laws look like for KT (does
  test-set log-likelihood follow power laws in sequence count, item-bank
  size, model width?), (c) whether emergent in-context IRT calibration
  exists at scale (zero-shot recover theta/alpha/beta from a few-shot
  prompt).
- **PhD tractability**: Mostly tractable. The bottleneck is *cross-platform
  data*. Public KT datasets (ASSISTments, EdNet, Junyi, RAIEd2020, Eedi)
  are heterogeneous in schema; assembling a unified interaction corpus is
  itself a contribution. Scaling-law experiments are tractable at 100M-1B
  parameter scales on a single 8xA100 node. Emergent-IRT probing is
  tractable as a methodology contribution even if the answer is "no, not
  yet at this scale."

##### A2. World models of learners (model-based RL view)

- **SOTA outside ed**: Dreamer V3, MuZero, IRIS (transformer world model
  for Atari), DayDreamer for robotics. The pattern is: learn a latent
  dynamics model from observations, plan or do model-based RL in latent
  space.
- **SOTA inside ed**: Almost nothing serious. RL for ed has used model-free
  methods on small problems (DKT-DSC for next-question selection, EduRank
  for sequencing). KT itself is a *forward* model of a learner but is not
  used as a *world model* for planning interventions.
- **Gap**: This is one of the most under-served paradigms. A KT model is
  structurally a latent dynamics model p(s_{t+1} | s_t, a_t) where s is
  knowledge state, a is the item presented, and the observation is the
  ordinal response. Yet nobody uses it MuZero-style for *planning*
  curriculum, identifying maximally informative items, or simulating
  counterfactual learning trajectories. The missing pieces are
  (i) latent-state quality good enough to plan in (KT models are
  notoriously calibrated only at the observation level), (ii) reward
  specification (learning gain? mastery probability? a value function
  learned from outcomes?), (iii) off-policy evaluation, because we can
  never run real RL on real students at scale.
- **PhD tractability**: Strongly tractable. Synthetic DGPs let you test
  whether a learned world model recovers a known generating process and
  whether planning in latent space outperforms model-free baselines. The
  story extends naturally to MA-GPCM because the IRT decoder is exactly
  the kind of interpretable observation model a learner world model
  needs.

##### A3. LLMs as agents in the measurement loop

- **SOTA outside ed**: LLM-as-judge for instruction-following (MT-Bench,
  Arena), LLM agents (ReAct, Reflexion, AutoGPT-style), constitutional AI.
  The validity literature is starting to catch up (Zheng et al.'s
  position-bias studies, Liu et al. on LLM grader reliability).
- **SOTA inside ed**: LLM tutors deployed (Khanmigo, Synthesis), LLM
  automated essay scorers (e-rater, ETS GPT-4-based), LLM item generators
  (Edmentum, PrepAI). Mostly product, little science.
- **Gap**: When an LLM sits anywhere in the measurement loop, classical
  IRT assumptions break in interesting and unstudied ways. (a) An LLM
  scorer is a *fallible measurement device* whose error structure is not
  iid; classical generalizability theory does not handle correlated
  rater errors of the kind LLMs produce. (b) An LLM tutor changes the
  *administration conditions* between items, so local item independence
  is violated by construction. (c) LLM-generated items are not exchangeable
  with human-written ones; the item generator induces a covariate shift
  on the calibration sample. None of these are KT problems, they are
  measurement-theory problems with a clear deep-learning angle (model
  the LLM-as-rater jointly with the examinee).
- **PhD tractability**: Tractable, and a natural bridge to the
  psychometric-researcher's expertise. Concretely PhD-sized:
  joint IRT model for examinee theta and LLM-rater bias/variance, with
  identifiability analysis.

##### A4. Diffusion / generative models for controllable item generation

- **SOTA outside ed**: Latent diffusion (Stable Diffusion 3, DiT) with
  classifier-free guidance for controllable generation; diffusion language
  models (Diffusion-LM, SSD-LM) for controllable text.
- **SOTA inside ed**: A few papers on neural item generation (mostly
  GPT-style finetuning, e.g., AutoQG, KhanGen), with weak control over
  psychometric properties. Almost nothing diffusion-based.
- **Gap**: Item generation conditional on *target IRT parameters*
  (difficulty beta, discrimination alpha, target Q-matrix entries) is
  the right problem statement and has barely been attempted. This is the
  natural bridge between generative AI and computerized adaptive testing
  (CAT): instead of selecting the next item from a fixed bank, *generate*
  one with the desired information curve. Risks: post-hoc calibration
  may drift from target; cold-start measurement of generated items.
- **PhD tractability**: Tractable for text items; harder for math items
  needing verified answers. The deeper contribution is the
  *calibration-aware decoding* objective: differentiate through a
  predicted IRT calibration to backprop into the generator.

##### A5. Multimodal grounding of student state

- **SOTA outside ed**: CLIP/SigLIP, Flamingo, GPT-4V, audio-language
  models (AudioLM, AudioPaLM), video-language (VideoMAE, V-JEPA).
- **SOTA inside ed**: Very thin. Almost all KT is response-only. A few
  papers on engagement detection from webcam, gaze tracking, click streams.
  These are not integrated with knowledge state modelling.
- **Gap**: Engagement, confusion, effort are observable signals that
  should inform knowledge state but are mostly ignored. The deeper gap is
  conceptual: classical IRT treats response as the sole observation; a
  modern formulation should treat the *full multimodal trace* (response,
  time, hesitation, eye motion, written work) as an observation of a
  latent ability. This reformulates the measurement model.
- **PhD tractability**: Mixed. The methodology (multimodal observation
  model) is PhD-tractable. The data is institution-bound; you need IRB
  access to multimodal learner data, which most candidates don't have.
  This is a strong direction *only if* the candidate's institution has
  such data.

##### A6. In-context / meta-learning for cold-start measurement

- **SOTA outside ed**: MAML, Reptile, Prototypical Networks, and more
  recently in-context learning interpreted as Bayesian inference (Xie
  et al., the prior-data fitted network line of work — TabPFN, PFNs).
- **SOTA inside ed**: A handful of meta-KT papers (MetaKT, ML-KT) that
  apply MAML to fast-adapt to new students. Mostly incremental.
- **Gap**: The bigger idea is *amortized IRT estimation*. Train a
  transformer that, given a few-shot prompt of (item, response) pairs,
  outputs a posterior over theta/alpha/beta. This is the PFN philosophy
  applied to measurement. It replaces EM/MCMC with a forward pass,
  generalizes across populations, and gives a principled cold-start
  story. It also turns calibration into a learned, not estimated,
  procedure — which is a non-trivial epistemic claim.
- **PhD tractability**: Very tractable. Synthetic data is essentially
  free (you sample from the generative IRT model). Real-data validation
  is the harder half. This direction connects naturally to MA-GPCM
  (which is a single-shot estimator) by reframing both as instances of
  amortized inference.

##### A7. Self-supervised / contrastive student representations

- **SOTA outside ed**: SimCLR, MoCo, DINO v2, JEPA, masked autoencoders.
  The lesson is that representation quality matters more than the
  downstream head.
- **SOTA inside ed**: CL4KT, Bi-CLKT, a few contrastive KT papers. They
  apply contrastive learning to interaction sequences but treat the
  representation as a black box whose only test is downstream KT
  accuracy.
- **Gap**: Whether self-supervised pretraining produces representations
  that are *psychometrically meaningful* (linearly probe to theta?
  align with skill structure?) is essentially unstudied. The gap is
  methodological: there is no probe suite for ed representations
  analogous to BIG-bench or SuperGLUE.
- **PhD tractability**: Tractable but lower-novelty alone. Best as a
  supporting thrust under a larger umbrella.

##### A8. MoE / modular architectures

- **SOTA outside ed**: Switch Transformer, Mixtral, DBRX, GShard. The
  story is sparse activation by token routing.
- **SOTA inside ed**: Very little. One or two papers on per-skill
  experts in KT.
- **Gap**: A per-skill or per-population MoE is a natural fit for
  educational data, where heterogeneity across skills and learner groups
  is the norm. The gap is empirical: does MoE actually buy you
  fairness/calibration improvements on under-represented subgroups,
  beyond raw accuracy? This is a fairness+architecture question that
  the field has not asked rigorously.
- **PhD tractability**: Tractable, modest novelty alone, good as a
  fairness thrust.

##### A9. GNNs / structured priors over skill graphs

- **SOTA outside ed**: GAT, GraphSAGE, transformer-on-graph variants,
  geometric deep learning more broadly.
- **SOTA inside ed**: GKT, GIKT, HGKT — fairly active, but most assume
  a hand-built skill graph and bolt a GNN on top.
- **Gap**: Learned, *uncertain* skill graphs with Bayesian structure
  inference are barely explored. The deeper question is whether the
  Q-matrix (item-to-skill mapping) can be jointly learned with
  identifiability constraints. This connects directly to MIRT
  identifiability (our existing concern).
- **PhD tractability**: Tractable, moderate novelty. Strong only if
  paired with identifiability theory.

##### A10. Continual / lifelong learning under drift

- **SOTA outside ed**: EWC, Online EWC, replay-based methods, recent
  work on test-time adaptation (TENT, EATA).
- **SOTA inside ed**: Almost nothing systematic. KT models are typically
  trained once and evaluated on held-out students; nobody studies what
  happens after deployment when curriculum changes, new items enter the
  bank, populations shift.
- **Gap**: This is a *deployment* gap with academic potential. An IRT
  system that can extend its item bank without re-equating from scratch
  is a real engineering need with a real research problem under it
  (concept drift in latent traits; online IRT linking).
- **PhD tractability**: Tractable but easier to make into one strong
  paper than a whole thrust.

##### A11. Agentic AI / simulated tutor-student dyads

- **SOTA outside ed**: Multi-agent simulation environments (Voyager,
  MetaGPT, multi-agent debate). Self-play in language (Constitutional
  AI's red team / blue team).
- **SOTA inside ed**: Embryonic. A couple of EdNLP papers use LLM
  students to test LLM tutors. No serious science on whether simulated
  populations transfer to real ones.
- **Gap**: External validity of LLM-simulated learners. If we can build
  high-fidelity learner simulators (essentially MA-GPCM-as-environment),
  we can do safe policy learning for tutoring. The validity question is
  the hard part and the psychometric-researcher's territory.
- **PhD tractability**: Tractable, with the external-validity question
  potentially being the entire contribution.

##### A12. Neural-symbolic integration

- **SOTA outside ed**: Differentiable logic, scallop, Logic Tensor
  Networks, DeepProbLog.
- **SOTA inside ed**: MA-GPCM is itself an instance — neural encoder
  with symbolic IRT decoder. KT+CDM hybrids also fit.
- **Gap**: The principle is not articulated. MA-GPCM stumbled into
  neural-symbolic measurement; nobody has stated the general design
  pattern ("plug a calibrated psychometric model into a neural
  representation as a structured decoder") as a paradigm with rules
  about when it works and when it does not.
- **PhD tractability**: Strong contribution potential. Possibly the
  candidate's strongest narrative claim, because they have already
  *done* it once and can generalise.

##### Paradigms I'm deprioritising (and why)

- **RLHF/DPO for ed policies**: Interesting but the reward-signal
  problem in ed (we don't know what we're optimizing) eats most of the
  contribution. Better as one chapter than a thrust.
- **Tokenization paradigms (standalone)**: Important sub-question, but
  not a thrust on its own. Folds into A1.
- **Test-time adaptation (standalone)**: Strong technique, weak thrust.
  Folds into A10.

#### Section B — Umbrella framework

I propose:

**"Computational Psychometrics for Learning Systems"** —
A research program that treats both *learners* and *AI systems that
teach or test learners* as measurable entities, and develops the
deep-learning machinery to do that measurement with the rigour
classical psychometrics demands.

The intellectual identity is the combination of three claims:

1. **Measurement is a primary scientific object**, not a downstream
   utility. Most KT work treats predictive accuracy as the goal;
   psychometrics treats *what we are measuring and how well* as the
   goal. The umbrella keeps this as the load-bearing commitment.
2. **Modern DL must extend, not replace, measurement theory.** Foundation
   models, world models, generative models all promise capability;
   none come with validity arguments. Building those arguments is the
   PhD's contribution.
3. **Measurement applies symmetrically to humans and machines.**
   Learners are measured for placement and remediation; AI tutors and
   scorers are measured for validity and calibration; LLMs are measured
   for capability. Same statistical machinery, different objects.

This umbrella is broader than "KT+IRT" (Round 1's gravity well) and
narrower than "deep learning in education" (which is everyone's
research program). It is also placeable: ed-tech reviewers will recognise
the validity-first stance; ML reviewers will recognise the
foundation-model and world-model commitments.

##### Five thrusts under the umbrella

**T1 — Foundation models of educational interaction (compass).**
Pretrain a transformer on a unified cross-platform interaction corpus,
study scaling laws for KT, and probe for emergent in-context IRT
calibration. This is the new spine, replacing the original recommender.
- Methods: tokenization study (A1), self-supervised pretraining (A7),
  in-context evaluation (A6).
- Outputs: (a) unified corpus + tokenizer, (b) scaling-law paper,
  (c) in-context IRT probing paper.
- Validity question for PR: does pretraining-then-probing produce IRT
  estimates that satisfy invariance and equating?

**T2 — Learner world models (planning + intervention).**
Treat the KT model as a latent dynamics model and use it for planning
under model-based RL. MA-GPCM is the natural observation model. Test
on synthetic DGPs first (we already have block, staircase, random-walk,
plus static). Off-policy evaluation against real KT logs.
- Methods: world models (A2), MA-GPCM as observation model (A12),
  off-policy evaluation, simulated-population validity (A11).
- Outputs: (a) latent-dynamics paper with planning experiments,
  (b) intervention-policy paper with OPE,
  (c) simulated-learner validity paper (joint with PR).

**T3 — Measurement under generative and LLM-mediated AI (validity
in the measurement loop).**
When an LLM grades, generates, or tutors, IRT assumptions break in
specific ways. Develop joint models (examinee + LLM-rater) and
identifiability theory. Extend to LLM-generated items.
- Methods: LLM-as-judge IRT (A3), diffusion item generation
  conditioned on target IRT (A4), neural-symbolic decoders (A12).
- Outputs: (a) LLM-rater joint IRT paper, (b) calibration-aware item
  generation paper, (c) human-AI co-administered tests paper.
- Validity question for PR: what is the analogue of differential item
  functioning when items are generated rather than authored?

**T4 — Amortized and lifelong measurement (the deployment thrust).**
Replace EM/MCMC calibration with amortized inference (PFN-style
transformers conditioned on calibration samples). Handle item-bank
growth, population drift, and online linking.
- Methods: amortized IRT (A6), MoE for population heterogeneity (A8),
  continual learning (A10), GNN priors over skill graphs (A9).
- Outputs: (a) amortized-IRT paper that replaces classical calibration,
  (b) drift-aware linking paper, (c) skill-graph joint-learning paper.

**T5 — Neural-symbolic measurement as a design principle.**
Articulate the design pattern that MA-GPCM instantiated: a calibrated
psychometric model as the structured decoder of a neural representation.
State the rules (when does this help, when does it hurt) and apply to
two new settings beyond MA-GPCM (e.g., cognitive diagnostic decoder,
multidimensional decoder with learned Q-matrix).
- Methods: A12 generalisation, identifiability theory.
- Outputs: position paper + two empirical instantiations.

##### Dependencies and ordering

```
                          T5 (design principle)
                         /  \
        T1 (foundation) --- T4 (amortized + lifelong)
               |  \         /
               |   T3 (LLM-mediated measurement)
               |  /
        T2 (world models)
               |
        MA-GPCM (foundation: validates the building block)
```

- MA-GPCM (already done) validates that neural+IRT works on synthetic
  DGPs. Without it, T2 and T5 have no empirical anchor.
- T1 is the spine. Everything else gains data and representations from it.
- T2 and T3 are the two outward-facing thrusts (one to learners, one to
  AI systems).
- T4 makes T1+T2 deployable.
- T5 is the meta-contribution that frames the whole program.

##### What this is NOT

- Not "MA-GPCM applied to N datasets" — that's a tier-2 conference
  contribution, not a PhD program.
- Not "KT + transformer" — that's been done.
- Not "use LLMs as tutors" — product, not science.
- Not pure ML benchmark-chasing on EdNet — no validity story.

##### How this addresses Round 1's gaps

Round 1's directions D1-D8 mostly fold into T1 and T2. The IRT-for-AI
direction folds into T3. The diversity that was missing in Round 1
comes from T4 (amortization, deployment) and T5 (the meta-claim), and
from importing world-model and foundation-model paradigms that Round 1
never named.

#### Section C — Hand-off questions to psychometric-researcher

These are the questions where my answers are guesses and the
psychometric-researcher should react in their Round 2 pass.

**C1 — In-context IRT calibration: is "learned equating" a valid
psychometric procedure?**
If a foundation model in T1 produces theta/alpha/beta from an
in-context prompt of (item, response) pairs, the resulting scale is
defined by the model's pretraining distribution. Is that a legitimate
equating procedure under classical or modern (IRT-linking) standards,
or is it a category error? What would have to be shown for it to
count?

**C2 — LLM raters and local independence.**
T3 needs a joint IRT model for examinee ability and LLM-rater behaviour.
Generalizability theory handles random rater effects, but LLM raters
have *systematic* biases (length, prompt position, sycophancy). Is the
right framework many-facet Rasch with LLM as a fully-modelled facet,
or is there a better psychometric formalism for non-iid rater error?

**C3 — Calibration-aware item generation: where does validity come
from?**
T3 proposes generating items whose predicted IRT parameters match a
target. Even if calibration matches in expectation, content validity,
construct representativeness, and DIF are separate concerns. What is
the minimum validity argument an item-generation paper would need to
not be dismissed in an ed-measurement venue?

**C4 — Learner world models for intervention: what stops this from
being "RL on students"?**
T2 plans interventions in a learned latent space. Real RL on real
students is an ethics non-starter. Synthetic simulators are an
external-validity non-starter. Is there a middle path — perhaps
constraint-restricted policy classes, or measurement-theoretic OPE —
that makes this a respectable line of work rather than reckless
optimization?

**C5 — Symmetry of measurement across humans and machines.**
The umbrella claims that learners and AI systems can be measured with
the same machinery. From a strict measurement-theory standpoint, is
this defensible (capability as a latent trait of an LLM, with items
sampled from a domain) or does it conflate measurement (humans, with
construct-validity arguments) with benchmarking (machines, with no
such arguments)? Where exactly is the line?



### Psychometric-researcher round 2

Research-scientist did exactly the right thing by widening the aperture, but the umbrella has a measurement-theoretic seam running through it that needs to be exposed before it is sold to a strict ed-measurement reviewer. The umbrella will hold only if the candidate is honest about three facts. First, "measurement" is a contested word; what RS calls measurement of LLMs is closer to what classical psychometrics calls benchmarking, and the program needs to be explicit about which standard applies where. Second, validity is not a property a model has; it is an argument a researcher constructs (Kane, 2013; AERA/APA/NCME Standards, 2014), and several thrusts as written outsource that argument to scale and emergence. Third, the field's strongest reviewers (Embretson, von Davier, Sinharay, Mislevy) will read the umbrella as a deep-learning program decorated with psychometric vocabulary unless the validity arguments are baked in from the start.

What follows answers RS's five questions, reacts to T1 through T5 with measurement-side critique, and proposes additional thrusts the framework needs in order to read as a measurement program and not as ML-with-IRT-flavor.

#### Part 1 — Reactions to RS hand-off questions

**C1 (in-context IRT calibration as learned equating).** This is the question where the gap between "learned a function" and "performed a valid equating" is widest, and the honest answer is that in-context posterior emission is *not* equating in the Kolen and Brennan (2014) sense, and pretending otherwise will not survive review. Equating is a procedure that produces score interchangeability under explicit assumptions (common-population, common-item, or randomly-equivalent-groups designs) with diagnostics (Stocking-Lord, Haebara, mean-sigma, characteristic curve overlap). A foundation model that ingests (item, response) pairs and emits theta-hat is doing *amortized posterior inference* under the pretraining-induced prior, which is closer to plausible-value generation in NAEP or a prior-data-fitted network's predictive distribution (Muller et al., 2022) than to equating. For it to count as a valid measurement procedure, the candidate would need to demonstrate (i) scale invariance under affine reparameterizations of the latent trait, (ii) population invariance (estimates from cohort A and cohort B place the same examinee at the same theta, up to linking error), (iii) sensitivity to item-bank composition that matches IRT theory (adding more discriminating items reduces SE), and (iv) recoverability of known generating parameters across distributions different from pretraining. Treat T1's probing paper as making the *amortized inference* claim, not the equating claim, and reserve the equating claim for a separate paper that runs Stocking-Lord against the learned scale. Anything less reads as a category error to a Psychometrika reviewer.

**C2 (LLM raters and local independence).** Many-facet Rasch (Linacre, 1989) is the right starting point but is not sufficient because it assumes rater effects are exchangeable random or fixed offsets. LLM raters violate this in three structured ways. First, *content-correlated bias*; an LLM grader's leniency depends on response content (length, surface features, presence of formulaic phrases), which makes the rater effect a function of the response itself, breaking the additivity assumption of MFR. Second, *order and context dependence*; sycophancy and position bias create within-session dependencies across items scored by the same LLM, violating local independence over items. Third, *non-stationarity*; the same LLM rater changes behavior across versions, prompts, and temperature settings, so the rater is not a single facet but a population of facets indexed by configuration. The right formalism is closer to *hierarchical rater models with response-dependent bias* (Patz and Junker, 1999, extended) or, more ambitiously, an explicit *measurement-error model where the LLM rater is a learned but structured noise channel* conditional on the response. From a design standpoint, the candidate should include human double-scoring on a calibration subset and report rater-by-content interactions, not just rater main effects. The paper that does *only* MFR will be told it under-modeled the rater.

**C3 (validity for generated items).** The minimum bar in an ed-measurement venue is the Kane (2013) interpretation/use argument applied to a generated item bank, and the candidate needs to address four inferences in sequence. *Scoring*: does the item admit a defensible scoring rubric, and is that rubric reproducible across raters or automatic scorers? *Generalization*: does the sample of generated items represent the construct domain, or does the generator's prior collapse onto stylistic modes that under-sample parts of the construct? *Extrapolation*: do calibration estimates from one cohort generalize to operational use? *Implication*: are decisions made from these scores defensible? Calibration matching in expectation addresses none of these directly; it addresses item-level *statistical* properties only. The empirical minimum for a generated-item paper should include (a) human expert content review on a sample of N items per generator condition with explicit content-validity ratings (Lawshe, 1975, or modern variants), (b) differential item functioning analysis comparing generated and human-authored items on matched groups, ideally with Mantel-Haenszel and SIBTEST or their IRT analogues, and (c) a *construct representation* analysis showing the generator covers the intended skill space, not just the easy-to-generate slice of it. Without these, a positive calibration-match result will be read as Goodhart's law, not as a validity argument. There is also a deeper concern; if the generator is conditioned on target IRT parameters, the resulting items are *engineered to look like the calibration model*, which inflates apparent fit while saying nothing about whether the items measure the construct. This is the "teaching to the model" problem in item generation, and it needs to be named.

**C4 (world models versus reckless RL on students).** The middle path exists and has a name in adjacent fields; it combines *off-policy evaluation with measurement-grounded uncertainty quantification* and *bounded policy classes constrained by curriculum theory*. Three concrete moves make T2 defensible. First, *never run learned policies on students without expert-in-the-loop approval*; the policy proposes, a human curriculum designer disposes, and the policy is evaluated on the disposition rate and on downstream learning gains. This is the *recommend-then-vet* design used in clinical decision support and is the only ethically defensible deployment pattern. Second, *off-policy evaluation with doubly-robust estimators* (Dudik et al., 2011; Thomas and Brunskill, 2016) on real KT logs, with explicit *calibration of the learner model's predictive intervals* on held-out students; if the world model is miscalibrated, the OPE is invalid, and the candidate should report calibration diagnostics (interval coverage, expected calibration error on response distributions) alongside policy value estimates. Third, *measurement-theoretic policy constraints*; restrict the action space to policies that satisfy classical adaptive testing constraints (maximum item exposure, content balancing, ability-difficulty matching at the Fisher information argmax), which gives the policy class a foothold in established CAT theory rather than letting it optimize unconstrained. The framing that sells this to a measurement audience is "model-based adaptive instruction with off-policy validity guarantees," not "RL for tutoring."

**C5 (symmetry of measurement across humans and machines).** This is the question where I have to disagree with RS most strongly. The symmetry claim is *philosophically attractive and methodologically dangerous*. Human ability measurement rests on construct validity arguments grounded in cognitive theory, predictive validity against external criteria, and a substantive theory of what the latent trait means. LLM "capability" has none of this in the strict sense. An LLM benchmark score is a *summary statistic over a task distribution*, and treating it as a trait estimate inherits IRT's mathematical machinery while shedding IRT's construct-validity discipline (Mislevy, 2018, on the inferential chain; Messick, 1989). The cleanest position is to *use IRT as a benchmarking improvement* for LLMs, not as measurement in the construct-validity sense, and to be explicit about this in writing. IRT on LLM benchmarks gives you (i) item-level information curves for benchmark items, (ii) ability estimates that are invariant to item subset selection, (iii) standard errors, and (iv) DIF analysis across model families, all of which are *significant improvements over mean accuracy*. What it does not give you is a defensible claim that the latent dimension is "general capability" in the Spearman g sense. The right framing is "IRT-grounded benchmarking with validity arguments at the *benchmark-task-coverage* level, not the construct level." If RS wants to keep the symmetry claim as a unifying thread, the umbrella should explicitly demarcate where construct validity applies (humans) and where coverage validity applies (machines). Conflating the two will give an editorial board ammunition to reject the framing.

#### Part 2 — Reactions to T1 through T5

**T1 (foundation models of educational interaction).** Defensible as a *prediction* program, fragile as a *measurement* program. The hidden assumption-violation is that pretraining on cross-platform interaction data conflates populations, items, and constructs that classical psychometrics keeps separate; what does "the same theta" mean when the pretraining mix contains middle-school algebra from one platform, Korean SAT prep from another, and adult professional certification from a third? The construct is not constant across the mix, so a single learned representation cannot be claimed to measure a coherent latent trait. The strengthening move is to *separate the representation-learning claim from the measurement claim*. Pretraining yields useful representations; measurement is a *downstream* operation that requires a domain-restricted item bank, a defined target population, and an explicit construct definition. A strict Psychometrika reviewer will say T1's weakest link is the *scaling-law claim*; scaling laws have meaning only relative to a defined loss on a defined distribution, and KT log-likelihood across heterogeneous platforms is not a defined loss in the measurement sense. The fix is to report scaling laws on within-platform held-out data and treat cross-platform transfer as a separate, harder question with its own diagnostics (DIF, linking error, construct equivalence). What the ML lens misses here is that *tokenization is a construct-definition choice*, not just a representation choice; whether a "token" is response-only or response-plus-context determines what the model can in principle measure.

**T2 (learner world models).** This is the strongest thrust from a measurement standpoint *if* the observation model is honestly psychometric, and the weakest if it is not. The hidden assumption is that planning in latent space requires the latent space to be *causally* meaningful, not just predictively useful, and KT models are typically validated only at the predictive level. MA-GPCM is well-positioned because its IRT decoder makes the latent state *interpretable* in psychometric units, but interpretability is not the same as causal validity. The strengthening move is to *test interventional validity* on synthetic DGPs where the ground-truth causal structure is known; do interventions chosen by planning in the learned latent space produce the predicted learning gains in the true DGP? This is the deep-learning analogue of the validity-of-intervention argument from instructional psychology. A strict reviewer's complaint will be the *off-policy evaluation gap*; without strong assumptions (positivity, no unobserved confounding), OPE on real KT logs is biased, and the candidate must report sensitivity analyses or bound estimates rather than point estimates. What the ML lens misses is that the *reward function in education is multidimensional and contested* (short-term mastery, long-term retention, engagement, equity), so single-scalar reward optimization is methodologically suspect; reward modeling itself should be a contribution.

**T3 (measurement under LLM-mediated AI).** This is the thrust where the measurement community has the most to contribute and where the candidate's psychometric expertise becomes a decisive advantage rather than supporting decoration. Defensible as written, but the framing should be sharpened from "joint IRT for examinee and rater" to "*joint measurement model for examinee construct estimation under structured, content-correlated rater error*." The strengthening move is to commit to a specific identifiability analysis upfront; in a joint examinee-rater model, what configurations of items, raters, and double-scoring are *identifiable*, and what configurations are *empirically underdetermined* in the sense of Anderson and Rubin (1956)? This is exactly the analysis that pushes the work from "neural model with extra parameters" to "measurement model with provable properties." A strict reviewer's complaint will be that *content-correlated bias* (C2 above) requires a richer rater model than many-facet Rasch and that the paper must demonstrate this with simulation studies under known generating bias structures. What the ML lens misses is that *administration conditions* are part of the measurement model (Mislevy, 2018), and an LLM tutor changes administration conditions item-by-item; this is not noise to be averaged out, it is a structural feature of the test.

**T4 (amortized and lifelong measurement).** Defensible and timely, but the *amortized* and the *lifelong* are doing different work and should not be conflated. Amortized inference is a *computational* claim about replacing EM with a forward pass; lifelong calibration is a *substantive* claim about handling drift in item parameters and population characteristics over time. The amortized story has a clean analogue in TabPFN and prior-data-fitted networks and should produce a clear comparison against marginal MLE, MML, and Bayesian MCMC on parameter recovery, posterior coverage, and computation time. The lifelong story is harder because *drift in item parameters has been studied as parameter drift detection* (Bock et al., 1988; Donoghue and Isham, 1998) and the candidate must situate the proposed neural approach within that literature, not invent a parallel one. The strict reviewer's complaint will be on *coverage*; amortized posteriors learned by transformer regression often have *miscalibrated uncertainty* (overconfident at distribution edges), and the paper must include calibration diagnostics (posterior coverage at nominal levels, expected calibration error, sharpness-versus-coverage tradeoffs) rather than only point estimate quality. What the ML lens misses is that *online IRT linking* is a solved problem with known procedures (concurrent calibration, Stocking-Lord chain linking); the contribution should be in scaling these to large item banks or relaxing their assumptions, not in re-inventing them under a neural framing.

**T5 (neural-symbolic measurement as design principle).** This is the candidate's strongest narrative claim because they have already executed it once with MA-GPCM, but as written it is a methodology paper, not yet a paradigm. Defensible if elevated to a *theory of when structured decoders preserve measurement properties*. The strengthening move is to formalize the design pattern with explicit propositions; for instance, "a neural encoder with a calibrated GPCM decoder preserves parameter recoverability if and only if the encoder's ability summary is conditionally independent of item characteristics given the latent trait" (this is exactly the local independence condition restated in encoder terms). Stating such propositions, with synthetic-DGP empirical tests and counterexamples where they fail, turns T5 from "we did this and it worked" into "here is when this works and why." A strict reviewer's complaint will be on *generality*; one or two empirical instantiations are not a paradigm, they are a research program, and the candidate must either demonstrate the design pattern on a *substantially different* psychometric model (CDM, MIRT with learned Q, multistage adaptive testing as decoder) or scale back the claim. What the ML lens misses is that the neural-symbolic measurement pattern is implicitly an *invariance commitment*; the symbolic decoder enforces measurement invariances (monotonicity, ordinal coherence, parameter interpretability) that the neural encoder alone cannot guarantee. Naming this invariance commitment is the conceptual contribution.

#### Part 3 — New thrusts the framework needs

The RS umbrella scans ML paradigms and misses the assessment-side paradigms that have their own deep theoretical traditions and clear PhD scope. Three additions, in order of priority.

**T6 — Person fit, aberrance, and validity of individual measurements.** RS focuses on *parameter recovery* and *prediction*; the measurement community spends equal energy on *individual-level validity*, asking whether a specific examinee's response pattern is consistent with the model being applied to them. Person-fit statistics (lz, Drasgow's standardized residual, U3, ECI4z) detect aberrant responding, careless errors, test anxiety, and item pre-exposure, all of which invalidate individual score interpretations even when the model fits well on average. Neural KT models, including MA-GPCM, produce population-level metrics but say almost nothing about *for which students is the model wrong, and in what direction*. The PhD-scale contribution is a *neural person-fit framework* that flags individual sequences as model-discrepant with calibrated false-positive rates, and that distinguishes substantively different aberrance causes (item exposure leakage, cheating, disengagement, construct-irrelevant variance, model misspecification at the individual level). This connects naturally to T2 (a world model should know when its forward model fails for a particular learner) and T3 (LLM-mediated administration creates new aberrance signatures). The reason this is missing from the RS scan is that ML almost never asks for-which-instance-is-the-model-wrong as a primary question, while measurement does so routinely.

**T7 — Learning progressions and growth as measurement constructs.** RS's T2 treats temporal change as latent dynamics to be modeled; the educational measurement community treats *learning progressions* as a substantive theory of how learners traverse a defined sequence of conceptual milestones (Wilson, 2009; Briggs et al., 2006; Mislevy et al., 2010), with explicit measurement targets at each stage. Vertical scaling (Kolen and Brennan, 2014) and longitudinal growth modeling (Embretson, 1991, on multidimensional learning models; McArdle and Grimm latent change models) are mature traditions that the umbrella ignores. The PhD-scale contribution is a *neural longitudinal IRT framework* where the latent trajectory is constrained to be consistent with an explicit learning progression, with statistical tests for progression violations and identifiability conditions for separating growth from drift. This is closer to assessment-for-learning than the prediction-oriented framing of KT, and it produces *diagnostic* outputs (which milestone is the learner at, what is the expected next milestone) that are far more useful to teachers than mastery probabilities. The reason this is missing from the RS scan is that the ML literature treats time as a continuous dynamical-systems question, while educational measurement treats time as a *theoretically structured progression* with substantive content.

**T8 — Fairness, accommodations, and measurement invariance under AI mediation.** RS mentions fairness once under T4 as an MoE side benefit; this radically under-weights an area where measurement has decades of theory and the AI community has urgent unsolved problems. Differential item functioning (Holland and Wainer, 1993), measurement invariance testing (Meredith, 1993; Vandenberg and Lance, 2000), and Universal Design for Learning (Rose and Meyer, 2002) are mature frameworks for asking whether a test measures the same construct for different groups and whether the testing apparatus itself disadvantages certain learners. When LLMs grade, generate, or tutor, *every one of these invariances becomes a research question*; does an LLM grader exhibit DIF across demographic groups when content is held constant? Do generated items produce DIF that human-authored items did not? Does an LLM tutor's interaction style favor certain communication norms over others, biasing the apparent ability estimates of learners from different cultural and linguistic backgrounds? The PhD-scale contribution is a *systematic invariance audit framework for AI-mediated measurement*, with DIF analyses across LLM versions, prompts, languages, and accommodation conditions, plus theoretical work on what counts as a fairness violation when the measurement apparatus is itself a learned function. This is missing from the RS scan because ML fairness frameworks (demographic parity, equalized odds) are weaker than measurement invariance frameworks and the candidate's psychometric expertise is decisive here.

#### Part 4 — Synthesis recommendation

Keep the RS umbrella name and the five thrusts, but *modify the umbrella's intellectual commitments* in three ways and *add the three measurement-side thrusts* as core, not optional. First, demarcate construct validity (humans) from coverage validity (machines) explicitly in the umbrella's first paragraph; the symmetry claim survives only if this demarcation is enforced throughout. Second, elevate *validity arguments* to a load-bearing commitment alongside the three claims RS listed; without this, the program reads as ML with psychometric decoration. Third, reframe T1's "scaling laws" and "emergence" claims as *amortized inference under explicit distribution assumptions* rather than as discoveries about measurement, which is C1's resolution. With these modifications the umbrella becomes more defensible at Psychometrika or JEM and just as compelling at NeurIPS or ICML.

The assembled framework, then, has eight thrusts organized in three tiers. The *measurement-foundation tier* contains T5 (neural-symbolic as a design principle, elevated to a paradigm claim) and T6 (person fit and individual validity), which together state what counts as a valid neural measurement and how we know when it fails. The *learner-side tier* contains T2 (world models with interventional validity), T4 (amortized and lifelong measurement, with amortization and lifelong handled as separate sub-thrusts), T7 (learning progressions as measurement constructs), and T1 reframed as a *representation-learning* layer that feeds the measurement tiers rather than constituting measurement itself. The *systems-side tier* contains T3 (measurement under LLM-mediated AI, sharpened to joint examinee-rater models with identifiability theory) and T8 (fairness and invariance audits of AI-mediated measurement). MA-GPCM remains the empirical anchor for T5 and the natural observation model for T2 and T7. T8 cross-cuts T1, T2, and T3 because invariance failures arise in all three. This structure is what a strict ed-measurement reviewer would call a coherent program; the original five-thrust version reads as a deep-learning program that has been pointed at education, while the eight-thrust version reads as a measurement program that has appropriated deep learning to extend its reach.

## Synthesis (final)

### The umbrella

The PhD program is **Computational Psychometrics for Learning Systems**. It is a measurement program that has appropriated deep learning to extend its reach, not a deep-learning program decorated with psychometric vocabulary. The umbrella holds itself to three intellectual commitments that distinguish the program from adjacent ML-with-IRT-flavor work.

1. **Demarcate construct validity from coverage validity.** When the test taker is a human learner the latent trait has a substantive construct interpretation grounded in cognitive theory, predictive validity against external criteria, and a defined population. When the test taker is a machine the latent quantity is a summary statistic over a defined task distribution, and IRT plays the role of a benchmarking improvement (item information curves, subset-invariant ability estimates, standard errors, DIF across model families) rather than the role of construct measurement. The same mathematical machinery applies, but the inferences it supports do not. The umbrella enforces this demarcation throughout, so the symmetry between measuring humans and measuring machines is methodological rather than ontological.
2. **Treat validity as a load-bearing argument, not a model property.** Following Kane (2013) and the AERA/APA/NCME Standards (2014), every neural measurement procedure in the program comes with an interpretation and use argument across scoring, generalization, extrapolation, and implication. Calibration matching, parameter recovery, and predictive accuracy are necessary but not sufficient. Each thrust ships with an explicit validity argument and the empirical machinery that argument requires.
3. **Reframe scaling and emergence claims as amortized inference under explicit distribution assumptions.** Foundation-model-scale training over learning interaction data is a powerful representation-learning move and a defensible amortization of expensive psychometric inference (EM, MML, MCMC) into a forward pass. It is not, by itself, equating in the Kolen and Brennan (2014) sense and it does not licence cross-population, cross-construct measurement claims. Scaling laws are reported on within-population, within-construct held-out data, and cross-population transfer is a separate question with its own diagnostics (DIF, linking error, construct equivalence).

These three commitments are what turn the program from a deep-learning agenda pointed at education into a measurement agenda that has hired deep learning to do work the older toolkit could not do at scale.

### The three tiers

The program is organized in three tiers that mirror what a measurement system must do, with eight thrusts distributed across them.

#### Measurement-foundation tier

This tier states what counts as a valid neural measurement and how we know when it fails. It is the conceptual backbone of the program and the area where the candidate's psychometric expertise is decisive.

**T5, neural-symbolic measurement as a design paradigm.** Elevated from a methodology to a paradigm claim. The contribution is a theory of when structured decoders preserve the measurement properties of the underlying IRT or CDM model, stated as explicit propositions with synthetic-DGP empirical tests and counterexamples. The symbolic decoder enforces measurement invariances (monotonicity, ordinal coherence, parameter interpretability, conditional independence of ability summary from item characteristics given the latent trait) that a neural encoder alone cannot guarantee, and naming this invariance commitment is the conceptual contribution. Generalization beyond GPCM (to MIRT with learned Q, CDM, multistage adaptive testing as decoder) demonstrates the paradigm. MA-GPCM is the existing empirical anchor.

**T6, person fit, aberrance, and individual validity.** Population-level metrics say nothing about for which student the model is wrong and in what direction. The contribution is a neural person-fit framework that flags individual sequences as model-discrepant with calibrated false-positive rates, distinguishing substantively different aberrance causes (item-exposure leakage, cheating, disengagement, construct-irrelevant variance, individual-level misspecification). This tier-1 thrust restores the for-which-instance question that classical psychometrics asks routinely and ML almost never asks, and it connects to T2 (a world model should know when its forward model fails for a particular learner) and T3 (LLM-mediated administration creates new aberrance signatures).

#### Learner-side tier

This tier models the learner across items and across time and uses those models for measurement, prediction, and intervention. T1 lives here because, under commitment 3, it is a representation-learning layer that feeds measurement rather than constituting measurement itself.

**T1, foundation representations for educational interaction.** Reframed as a pretraining and amortized-inference layer. Tokenization is a construct-definition choice and is treated as such. Scaling laws are reported on within-platform, within-construct held-out data, and cross-platform transfer is reported separately with linking-error and construct-equivalence diagnostics. The output is reusable representations that downstream thrusts compose with explicit construct definitions, target populations, and item banks.

**T2, learner world models with interventional validity.** Latent-space planning requires the latent space to be causally meaningful, not merely predictively useful. The contribution combines a world model whose observation model is honestly psychometric (MA-GPCM or a generalization is the natural choice) with interventional validity tests on synthetic DGPs of known causal structure, off-policy evaluation with doubly-robust estimators and calibration diagnostics on real KT logs, and measurement-theoretic policy constraints (maximum exposure, content balancing, Fisher-information matching). Deployment uses a recommend-then-vet pattern with expert-in-the-loop approval. Reward modeling is itself a research contribution because educational reward is multidimensional and contested.

**T4, amortized and lifelong measurement.** Split into two sub-thrusts that are doing different work. Amortized inference is a computational claim and is benchmarked against EM, MML, and MCMC on parameter recovery, posterior coverage (calibration diagnostics, expected calibration error, sharpness-versus-coverage curves), and runtime. Lifelong calibration is a substantive claim about item-parameter and population drift and is situated inside the existing parameter-drift literature (Bock et al., 1988, Donoghue and Isham, 1998) and the online linking literature (concurrent calibration, Stocking-Lord chain linking) rather than re-invented in parallel.

**T7, learning progressions and growth as measurement constructs.** Time in education is not a continuous dynamical-systems question, it is a theoretically structured progression with substantive content (Wilson, 2009, Briggs et al., 2006, Mislevy et al., 2010). The contribution is a neural longitudinal IRT framework where the latent trajectory is constrained to be consistent with an explicit learning progression, with statistical tests for progression violations, identifiability conditions for separating growth from drift, and diagnostic outputs (which milestone is the learner at, what is the expected next milestone) that map to assessment-for-learning rather than to mastery-probability outputs that teachers cannot act on. Vertical scaling and latent change models (Embretson, 1991, McArdle and Grimm latent change) are the natural priors.

#### Systems-side tier

This tier models the apparatus that delivers and grades assessment when an AI is in the loop, and audits that apparatus for fairness and invariance.

**T3, joint measurement models for examinee construct under structured rater error.** Sharpened from many-facet Rasch with LLM raters to a joint examinee-rater model whose rater term is content-correlated, order-and-context dependent, and non-stationary across LLM versions. Identifiability analysis (Anderson and Rubin, 1956) is committed to upfront, so the program states what configurations of items, raters, and double-scoring are empirically identified and which are not. Simulations under known generating bias structures and human double-scoring on calibration subsets are minimum design elements. The framing is measurement under structured noise channels, not benchmarking of LLM graders.

**T8, fairness, accommodations, and measurement invariance under AI mediation.** A cross-cutting thrust that lives in the systems tier because it audits the apparatus, and that runs through T1, T2, and T3. When LLMs grade, generate, or tutor, every measurement invariance (Meredith, 1993, Vandenberg and Lance, 2000) becomes a research question. Does an LLM grader exhibit DIF across demographic groups when content is held constant. Do generated items show DIF that human-authored items did not. Does an LLM tutor's interaction style favor certain communication norms, biasing apparent ability for learners from different linguistic and cultural backgrounds. The contribution is a systematic invariance audit framework for AI-mediated measurement, plus theoretical work on what counts as a fairness violation when the measurement apparatus is itself a learned function. ML fairness criteria (demographic parity, equalized odds) are subsumed by measurement-invariance criteria.

### MA-GPCM in the program

MA-GPCM is the empirical anchor for T5 and the natural observation model for T2 and T7, and it is the prototype that licences the program. It already shows that a separated ability pathway plus a calibrated GPCM decoder recovers item parameters and dynamic ability under static and dynamic DGPs while remaining competitive on prediction. The thesis program extends MA-GPCM along three axes. *Decoder generality* extends the IRT decoder family to MIRT with learned Q, CDM, and progression-constrained decoders (T5, T7). *Encoder generality* extends the encoder family to attention-based and foundation-pretrained encoders that feed multiple decoders (T1, T4). *Apparatus generality* extends the model to handle structured rater error and AI-mediated administration (T3, T8). Person-fit (T6) and interventional planning (T2) extend what the model is used for. Each extension is a paper, each paper carries an explicit validity argument, and each validity argument cites back to the umbrella's three commitments.

### Cross-cutting threads

Three threads run through the program and tie the thrusts together.

The *validity-argument thread* runs through every thrust. Each thrust has an interpretation and use argument with scoring, generalization, extrapolation, and implication subarguments, and the candidate's job is to make those arguments explicit and empirically supported.

The *identifiability thread* runs through T3, T4, T5, and T7. Each thrust commits to upfront identifiability analysis covering which configurations of data and parameters are identified, which are not, what assumptions enable identification, and what diagnostics detect identification failure. This is the discipline that distinguishes a measurement program from a neural-modeling program.

The *invariance-audit thread* runs through T1, T2, T3, and T8. Every learned function used in measurement is audited for invariance properties (subset invariance, population invariance, content invariance, presentation-mode invariance) using DIF and measurement-invariance techniques rather than aggregate fairness metrics.

### The arc

The arc moves from foundations to applications. The measurement-foundation tier states the design principles (T5) and the failure-detection machinery (T6) that the rest of the program presumes. The learner-side tier builds the learner-modeling apparatus (T1, T2, T4, T7) on top of those foundations, with each thrust producing both a methodological contribution and a validity argument. The systems-side tier (T3, T8) addresses what happens when the measurement apparatus itself is an AI artifact, which is the most distinctive contemporary challenge to educational measurement and the area where the candidate's combined ML and psychometric expertise is decisive. MA-GPCM connects all three tiers as the working example.

For a strict ed-measurement reviewer (Embretson, von Davier, Sinharay, Mislevy) the program reads as a coherent measurement agenda extended into the deep-learning era with explicit validity discipline. For an ML reviewer (NeurIPS, ICML, ICLR) the program reads as a principled application of structured-output deep learning, amortized inference, world models, and invariance auditing to a domain where the structure is non-trivial and the validity questions are real. For an AIED reviewer (IJAIED, AIED, EDM, LAK) the program reads as a unifying framework for the interpretability, fairness, and measurement debates that the field has been having in pieces.

What remains under-specified is deliberate. The candidate has not yet committed to a fixed sequence of papers, a fixed timeline, or a fixed empirical dataset for each thrust. Those commitments belong in a thesis proposal, not in this blueprint. What this blueprint commits to is the umbrella, its three intellectual commitments, the three tiers, the eight thrusts, and MA-GPCM as the empirical anchor that licences the program and from which each thrust is one defensible step.
