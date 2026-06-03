# DRL-MAIRT Background Dossier

Date: 2026-06-04.

This note collects the background needed before implementing a realtime
MA-IRT-conditioned reinforcement-learning recommender. It is deliberately
separate from the proposal plan so that the proposal can be revised against
explicit evidence rather than intuition.

## Research Question

The intended contribution is not simply "use theta in a DQN." The research
question is:

> Can a deep IRT model with online sequence memory serve as both a calibrated
> learner-state estimator and a transition model for sequential recommendation,
> so that decisions are informed by psychometric parameters, deep memory state,
> uncertainty, and content constraints in realtime?

The target runtime is:

```text
response event -> MA-IRT online update -> candidate scoring -> decision policy
-> recommended item/resource/path -> next learner feedback
```

The hard part is the bidirectional coupling:

- MA-IRT must shape recommendation through theta, alpha, beta, response
  probabilities, memory summaries, and uncertainty.
- The recommender must shape future MA-IRT observations by changing which items
  or learning resources the learner sees.
- If MA-IRT is later updated from policy-generated data, distribution shift must
  be controlled so MA-GPCM performance and recovery do not degrade.

## Local MA-IRT Evidence

The current `ma-irt/` codebase is already modular enough to support this
without rewriting MA-GPCM.

Important files:

- `ma-irt/models/registry.py`: typed `EncoderOutput`, `DecoderOutput`,
  `Encoder`, and `ResponseDecoder` contracts.
- `ma-irt/models/base.py`: shared `EncoderDecoderModel.forward` returns
  `logits`, `probs`, `theta`, `alpha`, `beta`.
- `ma-irt/models/magpcm.py`: MAGPCM wrapper composes `DKVMNEncoder` and
  `GPCMDecoder`.
- `ma-irt/models/encoders/dkvmn.py`: DKVMN memory loop. It reads memory,
  emits summaries, then writes the observed `(question, response)`.
- `ma-irt/models/decoders/gpcm.py`: GPCM decoder; `separate_theta=true`
  reads theta from the separated ability path, while `false` is the
  DKVMN+GPCM ablation.
- `ma-irt/utils/dataloader.py`: offline sequence dataset and padded batch
  collation.

Design implications:

- The recommender should be outside `models/magpcm.py`.
- The first core addition should be an online state adapter around the DKVMN
  value memory, not a new model class.
- Whole-sequence `MAGPCM.forward` must remain the reference behavior.
- Online step-by-step scoring must be parity-tested against whole-sequence
  inference.

Current MA-IRT limitation:

- `DKVMNEncoder.forward` recomputes full sequences and does not expose
  `init_state`, `read_step`, `score_step`, or `write_step`.

Required online behavior:

```text
state_0 = init_value_memory(student)
score(q_t | state_t)       # read only, no write
observe(q_t, r_t, state_t) # write value memory after response
state_{t+1}
```

This preserves the causal convention of MA-GPCM.

## CaRReL Evidence

The local CaRReL clone lives at `C:\Users\steph\Documents\CaRReL`.

Observed files:

- `model/env.py`: `JobRecEnv` with candidate jobs, a current recommendation
  list `K`, and actions `{keep, add, remove}`.
- `model/agent.py`: basic DQN agent, replay buffer, target network.
- `model/agentv2.py`: enhanced DQN over `(theta, job_features)` state with
  variable-length job-feature lists.
- `model/networks.py`: plain DQN plus `EDQN`, where an LSTM encodes the current
  job slate and a learned theta projection scores slate relevance.
- `model/utils.py`: loads `est_theta_history` from pickle.
- `model/NRT.py`: traditional mixed-format IRT/adaptive testing simulator with
  LBFGS theta updates and item information functions.
- `train.py`: trains DQN over a fixed theta trajectory and saves best/worst
  checkpoints and per-step info logs.

Useful pieces:

- The environment/agent separation is a reasonable starting abstraction.
- Slate modification actions are relevant for recommendation lists and paths.
- Per-step `info` logging is useful for debugging policy behavior.
- A theta-only baseline can be derived from this structure and used as the
  direct comparison against richer MA-IRT state.

Limitations:

- It consumes an external theta trajectory. It does not update learner state
  from the action-response loop.
- It uses theta as the dominant learner representation and ignores response
  probabilities, item parameters, uncertainty, and hidden sequence memory.
- The reward is not grounded in causal learning gain; it is mainly a change in
  a learned recommendation score plus a terminal dot-product rating.
- The action space is rigid and tied to a fixed candidate pool/list size.
- The traditional MLE/LBFGS theta update is not native to MA-GPCM.

Conclusion:

CaRReL is a useful negative baseline and engineering sketch, not the framework
to port. DRL-MAIRT should explicitly include a `theta_only_dqn` baseline so the
paper can demonstrate what the richer deep IRT state adds.

## ExRec Evidence

ExRec is the closest positive template. It is public at
`https://github.com/oezyurty/ExRec`; a local inspection clone was placed at
`C:\tmp\ExRec`.

The ExRec README states that the framework:

- annotates questions with KCs and solution steps;
- learns semantic question/KC embeddings;
- trains and calibrates a KT model;
- plugs that calibrated KT model into RL algorithms for exercise
  recommendation;
- supports benchmark tasks for global, practiced, upcoming, and weakest-KC
  knowledge improvement.

Relevant files from the cloned repo:

- `exercise_recommender/envs/question_vector_env.py`: Gymnasium environment
  where actions are continuous question embeddings. The KT wrapper predicts a
  response probability, samples a response, updates hidden state, and rewards
  the change in prediction.
- `exercise_recommender/envs/kc_evolution_envs.py`: KC-targeted environments
  with cluster/KC mappings, question-bank enforcement, and batch/vectorized
  student simulation.
- `exercise_recommender/utils/question_bank.py`: FAISS-backed nearest-neighbor
  projection from continuous action embeddings to real corpus questions.
- `exercise_recommender/utils/history_generator.py`: circular dataloader for
  drawing initial histories.
- `exercise_recommender/wrappers/calibrationqdkt_wrapper.py`: wraps calibrated
  QDKT state; exposes `init_states`, `update_hidden_state`, and `predict_in_rl`.
- `exercise_recommender/agents/critic_dkt.py`: critic that reuses KT model
  components to estimate value under possible responses.
- `exercise_recommender/wrappers/*`: Tianshou wrappers for DQN, PPO, SAC, TD3,
  TRPO, C51, Rainbow, and related algorithms.

Transferable design ideas:

- Treat the trained KT/IRT model as an environment-facing wrapper with explicit
  `init`, `predict`, and `update` methods.
- Support both discrete action spaces and continuous embedding actions projected
  back to a real item bank.
- Use a question/content bank abstraction rather than hard-coding item IDs.
- Define multiple pedagogical tasks: global improvement, weakest skill,
  practiced skill, upcoming skill.
- Keep RL wrappers separate from the KT/IRT model.
- Consider vectorized environments for GPU throughput.
- Use model-based value estimation: evaluate possible response outcomes using
  the KT/IRT transition model instead of learning value purely from sparse
  rollouts.

What DRL-MAIRT should do differently:

- ExRec's KT state is KC mastery / hidden LSTM state. DRL-MAIRT should expose
  psychometric state: theta, alpha, beta, expected ordinal score, entropy,
  memory attention, and GPCM item information proxies.
- MA-GPCM must preserve IRT recovery, not only response prediction.
- MA-GPCM item parameters and response categories are ordinal/polytomous; reward
  and simulator design should not collapse everything to binary correctness.
- The first parity test must prove online MA-GPCM equals batched MA-GPCM.

## Related Literature

### ExRec

ExRec, "Personalized Exercise Recommendation with Semantically-Grounded
Knowledge Tracing" (arXiv 2507.11060), frames KT as an RL environment and
optimizes exercise recommendations for knowledge gain. The abstract emphasizes
semantic content, structured progression, KT training, RL methods, and
model-based value estimation using KT components.

Use for DRL-MAIRT:

- architecture template;
- semantic item/KC embeddings;
- KT/IRT-as-environment framing;
- multiple educational reward tasks;
- model-based value estimation.

### ALPN

ALPN, "Adaptive Learning Path Navigation Based on Knowledge Tracing and
Reinforcement Learning" (arXiv 2305.04475), combines AKT learner-state
estimation with entropy-enhanced PPO for learning-material recommendation. Its
abstract reports improved learning outcomes and path diversity.

Use for DRL-MAIRT:

- path diversity as an explicit metric;
- PPO-style policy for learning materials;
- separate evaluation of learning outcome and diversity.

### CSEAL

CSEAL, "Exploiting Cognitive Structure for Adaptive Learning" (arXiv
1905.12470), combines learner knowledge level and prerequisite/cognitive
structure for adaptive learning. It is important for constraining action spaces
with pedagogical structure.

Use for DRL-MAIRT:

- prerequisite graph constraints;
- action masking;
- cognitive-structure-aware reward and state features.

### Adaptive Learning Recommendation with Deep Q-Learning

This line directly connects IRT and DQN for adaptive learning recommendations.
It is closer to CaRReL in spirit: IRT estimates learner ability, while DQN
chooses learning materials.

Use for DRL-MAIRT:

- historical baseline for IRT + DQN;
- contrast with deep, online MA-IRT state instead of static MIRT theta only.

### Offline RL For Recommendation

"A General Offline Reinforcement Learning Framework for Interactive
Recommendation" highlights learning from logged feedback without online
exploration and proposes support, supervised, policy, dual, and reward
extrapolation constraints to reduce distribution mismatch.

Use for DRL-MAIRT:

- offline-first training;
- behavior-policy support constraints;
- conservative policy learning before live deployment.

"Offline Evaluation for Reinforcement Learning-based Recommendation" argues
that common next-item prediction protocols are unsuitable for evaluating RL
recommenders and can hide critical deficiencies.

Use for DRL-MAIRT:

- avoid claiming RL success from next-item prediction only;
- include simulator validation, counterfactual evaluation, and policy-safety
  diagnostics.

### RL4RS

RL4RS is a recommender-RL benchmark and codebase. Its README emphasizes
real-world sequential recommendation datasets, simulator environments, offline
RL algorithms such as BCQ/CQL, counterfactual policy evaluation, vectorized or
HTTP environments, and parametric-action DQN support.

Use for DRL-MAIRT:

- offline RL evaluation tooling;
- simulator/environment organization;
- slate and parametric-action policy ideas;
- separation between supervised simulator training and policy training.

### d3rlpy

d3rlpy provides offline RL algorithms with scikit-learn-style APIs, including
dataset abstractions, algorithms, off-policy evaluation, logging, online
finetuning, and policy selection.

Use for DRL-MAIRT:

- fast baseline implementation for offline RL once the state/action/reward
  dataset is formalized;
- avoid writing CQL/IQL/BCQ from scratch too early.

## Conceptual Model

DRL-MAIRT should be formulated as a partially observable Markov decision
process:

```text
latent learner state z_t
observed response event x_t = (q_t, r_t, context_t)
belief/state estimate b_t = MAIRT(history_t)
policy action a_t = pi(b_t, candidates_t, constraints_t)
feedback y_{t+1}
updated history history_{t+1}
```

MA-IRT is the belief updater:

```text
b_t = {
  value_memory_t,
  student_summary_t,
  joint_summary_t,
  theta_t,
  candidate probs,
  candidate alpha/beta,
  attention,
  uncertainty
}
```

The recommender can be:

- contextual bandit when actions do not affect future state;
- RL when recommendations affect future responses and mastery;
- model-based RL when MA-IRT is used to simulate response/state transitions;
- conservative offline RL when training from historical logs.

## Key Design Decisions

### 1. Frozen First, Joint Later

Start with frozen MA-GPCM. The first scientific claim should be that rich
MA-IRT state improves decisions over theta-only state. Joint training should be
attempted only after the frozen pipeline has reproducible value.

### 2. Online Parity Is The First Gate

Before any recommender work, implement:

```text
online_step_loop(history) == batched_MAGPCM_forward(history)
```

Acceptance should check logits, probs, theta, alpha, beta, and attention.

### 3. Candidate Scoring Before RL

The recommender needs a stable scoring API:

- expected ordinal score;
- probability of success or thresholded correctness;
- response entropy;
- GPCM item information proxy;
- alpha/beta difficulty/discrimination features;
- memory attention dispersion;
- state uncertainty.

### 4. Content Bank Is A First-Class Object

DRL-MAIRT should include:

```text
AssessmentItemBank
LearningResourceBank
CareerOrCourseBank
```

Each bank should expose IDs, embeddings, prerequisites, target skills, action
constraints, and nearest-neighbor lookup for continuous actions.

### 5. Multiple Action Modes

Support these action modes explicitly:

- discrete item ID;
- continuous item embedding projected to nearest valid item;
- slate/list edit;
- resource recommendation followed by assessment;
- hierarchical path: goal -> skill -> resource -> item.

### 6. Reward Is A Research Object

Rewards must be configurable and reported separately. Candidate components:

- predicted learning gain;
- uncertainty reduction;
- target-skill improvement;
- response success;
- retention;
- engagement/completion;
- diversity;
- prerequisite violation penalty;
- time/cognitive-cost penalty.

Do not collapse these into a single opaque scalar in analysis.

## Feasibility Risks

### Simulator Exploitation

If the policy is trained only against MA-IRT simulation, it may exploit model
artifacts. Mitigation:

- held-out human sequence validation;
- compare simulated response distributions against real logged responses;
- restrict actions to observed support early;
- use conservative offline RL.

### Reward Confounding

Correctness after recommendation may reflect prior ability rather than learning
caused by the recommendation. Mitigation:

- use randomized or semi-random exploration data when available;
- use delayed follow-up assessments;
- report causal limitations explicitly.

### MA-GPCM Degradation

Policy-driven exposure changes the training distribution. Mitigation:

- freeze MA-GPCM initially;
- when coupling, regularize to frozen outputs;
- maintain regression gates on prediction and recovery.

### Career Recommendation Horizon

Career outcomes are sparse, delayed, and socially constrained. Mitigation:

- treat career recommendation as a later domain instantiation;
- start with questionnaire/item-bank simulation and short-horizon proxies;
- require human-in-the-loop constraints.

## Concrete Research Hypotheses

H1. Full MA-IRT state improves recommendation value over theta-only state.

H2. Psychometric candidate features `alpha`, `beta`, expected score, and
uncertainty improve policy robustness compared with black-box hidden states.

H3. Continuous embedding actions with nearest-neighbor projection generalize
better to unseen items than fixed discrete actions.

H4. Prerequisite-constrained policies reduce unsafe recommendations without
substantially reducing learning gain.

H5. Conservative offline RL is more reliable than naive online-style DQN when
trained from historical education logs.

H6. Carefully constrained auxiliary coupling can improve recommendation without
degrading MA-GPCM prediction or IRT recovery, but unconstrained joint training
is likely unsafe.

## Minimal Evidence Needed Before Implementation

- Confirm MA-GPCM online parity can be implemented without changing
  whole-sequence outputs.
- Identify which local datasets contain enough sequential diversity for
  policy training or simulation.
- Decide whether item/content embeddings come from existing MA-IRT item
  embeddings, text/semantic metadata, or external encoders.
- Define a small item-bank schema.
- Define the first reward task. Recommended: synthetic assessment-item
  selection with expected theta gain and uncertainty reduction.
- Define MA-GPCM non-regression gates.

