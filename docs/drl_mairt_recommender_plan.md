# DRL-MAIRT Recommender Proposal

Date: 2026-06-04.

This document proposes a new research and development track for a realtime
decision system built on the current MA-IRT codebase. The working name is
**DRL-MAIRT**: a deep IRT state estimator coupled to a reinforcement-learning
recommender.

The target runtime loop is:

```text
learner response event -> online MA-IRT state update -> candidate scoring ->
RL recommender action -> next content / tutorial / course / career suggestion
```

The design goal is not to bolt a recommender onto a final test score. The
system should update after each response, expose psychometric and deep sequence
state immediately, and let the decision policy use that state before the next
action.

## Executive Position

The strongest design is **not** immediate end-to-end joint training. Start with
a frozen, calibrated MA-GPCM model as the online state estimator and learned
transition model. Then train a recommender policy against logged data and
MA-IRT-based simulation. Later, add constrained co-training where the policy
loss can shape selected representation layers without degrading MA-GPCM's
prediction and IRT recovery contracts.

Reason:

- MA-GPCM already has a scientifically meaningful output contract:
  response probabilities, theta, alpha, beta, and DKVMN memory dynamics.
- A deployed recommender changes item exposure distribution. Naive joint
  training can damage calibration and make the original MA-GPCM recovery
  results non-comparable.
- RL recommenders need sequential feedback. If the only observed feedback is a
  response to an assessment item, the policy can be trained offline or in a
  simulator, but claims about learning gains require careful causal evaluation.

So the first publishable framework should be **coupled but modular**:

1. MA-IRT estimates learner state and response distributions online.
2. The recommender consumes MA-IRT state, content metadata, and action history.
3. The recommender's chosen action determines the next observed event.
4. The resulting response/engagement event updates MA-IRT state.
5. Optional later stages allow policy gradients or auxiliary losses to influence
   selected representation layers under strict MA-GPCM regression gates.

## Current Codebase Fit

The current `ma-irt/` structure is a good base for this. It now has explicit
encoder/decoder contracts:

- `ma-irt/models/registry.py` defines `EncoderOutput`, `DecoderOutput`,
  `Encoder`, and `ResponseDecoder`.
- `ma-irt/models/base.py` standardizes the legacy output dict:
  `logits`, `probs`, `theta`, `alpha`, `beta`.
- `ma-irt/models/magpcm.py` builds `DKVMNEncoder + GPCMDecoder`.
- `ma-irt/models/encoders/dkvmn.py` contains the read-before-write memory loop.
- `ma-irt/models/decoders/gpcm.py` isolates the GPCM decoder and the
  `separate_theta` ablation.

The key missing piece is an online interface. Today, `DKVMNEncoder.forward`
recomputes a whole prefix sequence. Realtime serving needs per-learner memory
state:

```text
init_state(student_id)
score_candidates(state, candidate_items)      # read only, no write
observe_response(state, item_id, response)    # write/update
snapshot_state(state)                         # for logging/replay
```

This can be added without changing whole-sequence MAGPCM semantics.

## What To Preserve From MA-GPCM

These invariants are non-negotiable:

- **Read before write**. The model predicts/extracts theta, alpha, beta for
  item `q_t` from the memory state before observing `r_t`, then writes
  `(q_t, r_t)`.
- **Separated theta remains meaningful**. `separate_theta=true` is the
  MA-GPCM intervention; `false` is the DKVMN+GPCM ablation. The recommender
  must not collapse these paths.
- **Item-conditioned mode is performance-sensitive**. The code comments say
  `item_conditioned=true` matches the paper/real binary setting, while
  `false` improves synthetic ordinal recovery. The recommender framework must
  treat this as an experiment factor, not a cleanup detail.
- **Prediction and recovery gates stay intact**. Any core edit must pass shape,
  migration, smoke, and MA-GPCM regression tests before being accepted.
- **Serving state is per learner**. DKVMN value memory cannot leak across
  users.

## CaRReL Assessment

The local CaRReL clone at `C:\Users\steph\Documents\CaRReL` is useful mainly as
a prototype and warning.

Observed structure:

- `model/env.py` defines `JobRecEnv` with a candidate job pool, a current
  recommendation set `K`, and actions `{keep, add job, remove job}`.
- `model/agent.py` and `model/agentv2.py` implement DQN agents with replay
  buffers and target networks.
- `model/networks.py` defines a plain DQN plus an enhanced DQN that encodes a
  list of job features with an LSTM.
- `model/utils.py` loads an external `est_theta_history` sequence from pickle.
- `train.py` trains over fixed theta trajectories and saves best/worst/start/end
  policy checkpoints.
- `model/NRT.py` contains traditional simulation/calibration logic for mixed
  item types, with LBFGS theta updates.

What can be reused conceptually:

- Environment abstraction with explicit `reset`, `step`, `state`, `action`,
  `reward`, and `info`.
- Slate/list actions, not only single-item recommendation.
- Candidate item/course/job features and a learned encoder for the current
  slate.
- Replay buffer and target-network mechanics as a minimal DQN baseline.
- Rich per-step logging in `info` for later analysis.

What should not be ported directly:

- The learner state is an externally supplied theta trajectory, so the policy
  is not coupled to a live response model.
- The reward is mostly a change in recommendation probability plus a terminal
  dot-product rating; it is not grounded in learning gain or causal response
  outcomes.
- The model uses theta as the main learner feature and discards uncertainty,
  response distribution, item alpha/beta, memory attention, and content
  prerequisites.
- The action indexing is rigid for a fixed candidate pool and fixed maximum
  slate size.
- Traditional MLE/LBFGS theta updates are not the right native mechanism for
  MA-IRT.

CaRReL should be treated as a sandbox for action semantics and DQN mechanics,
not as the foundation.

## Related Work Direction

The closest positive template is **ExRec**: it turns a calibrated knowledge
tracing model into an RL environment and optimizes exercise recommendation
policies for knowledge gain. Its useful ideas are KT-as-environment,
semantically grounded item/KC embeddings, model-based value estimation, and
multiple pedagogical reward definitions.

Other useful directions:

- **ALPN**: combines knowledge tracing and PPO-style learning-path navigation.
- **CSEAL**: uses learner knowledge state plus prerequisite/cognitive structure
  for adaptive learning with actor-critic RL.
- **RL4RS**: not education-specific, but valuable for offline RL, slate
  recommendation, counterfactual policy evaluation, and simulator design.
- **Offline RL for learning paths**: important because education logs are
  policy-biased and live exploration with students is risky.

The research gap for this repo:

> Existing systems usually use KT/IRT as a state feature or simulator. DRL-MAIRT
> should use a deep IRT model as an online psychometric transition system whose
> interpretable outputs and hidden memory jointly condition recommendation.

## Formal Problem Definition

Let each learner generate a stream:

```text
e_t = (learner_id, item_id q_t, response r_t, optional context c_t, timestamp)
```

MA-IRT maintains hidden state:

```text
h_t = (DKVMN value memory, attention summary, theta_t, uncertainty_t, history)
```

At decision time, the recommender receives:

```text
s_t = phi(
    theta_t,
    MA-IRT student_summary_t,
    DKVMN memory/attention summaries,
    response distribution over candidate items,
    candidate alpha/beta/item embeddings,
    content graph/prerequisites,
    recent action history,
    optional demographics/context
)
```

It chooses an action:

```text
a_t in A(s_t)
```

Actions can be:

- next assessment item;
- tutorial or learning resource;
- course/module recommendation;
- slate of assignments;
- vocational interest prompt;
- career/course pathway suggestion.

The environment returns feedback:

```text
o_{t+1} = response, engagement, completion, dwell time, rating, follow-up quiz
```

The next MA-IRT update is:

```text
h_{t+1} = MAIRT.update(h_t, q_t, r_t)
```

For non-assessment actions such as tutorials, there are two choices:

- no direct MA-IRT write until a follow-up assessment response arrives;
- learn a tutorial transition model that predicts how the tutorial changes
  mastery before the next assessment.

The second is more ambitious and should be Stage 3+, not the first skeleton.

## State Representation

The recommender should not consume only `theta_t`. Use a layered state.

### Psychometric State

- `theta_t`: current ability estimate.
- `alpha(q, t)`: candidate discrimination under the current learner state.
- `beta(q)`: candidate step thresholds.
- Expected score and response entropy for each candidate item.
- Confidence/uncertainty proxy: entropy, margin, memory attention dispersion,
  disagreement across checkpoint ensemble or MC dropout.

### Deep Sequential State

- `student_summary_t` from the separated MA-GPCM ability path.
- `joint_summary_t` for item-conditioned interaction state.
- DKVMN attention over concepts/memory slots.
- Recent response-history features.

### Content And Action State

- Item/course/tutorial embeddings.
- Skill/KC tags or semantic embeddings.
- Prerequisite graph position.
- Previously recommended content.
- Current slate/list summary if recommending multiple resources.

### Domain-Specific Extensions

For course/tutorial recommendation:

- target skill gaps;
- course concept coverage;
- prerequisite satisfaction;
- time cost and difficulty.

For vocational/career recommendation:

- latent interest traits from questionnaire responses;
- career skill vectors;
- required qualifications;
- uncertainty-aware exploration prompts.

## Action Space

The framework should support three action levels.

### Level 1: Next Assessment Item

Action is a question/item. This is easiest because MA-IRT can immediately
simulate/predict a response distribution and update state after response.

Use cases:

- adaptive testing;
- diagnostic question selection;
- personalized practice item recommendation.

### Level 2: Learning Resource Or Tutorial

Action is a tutorial, lesson, hint, assignment, or course segment. A resource
maps to a skill/KC set and is followed by an assessment item.

Use cases:

- course-related question bank -> tutorial recommendation;
- weakness remediation;
- prerequisite preparation.

### Level 3: Path Or Slate

Action is a list or path:

- a slate of assignments;
- module sequence;
- career/course plan.

This needs slate policy support and delayed rewards.

## Reward Design

Do not use only immediate correctness. Recommended reward components:

```text
R_t =
  w_gain * expected_mastery_gain
  + w_diag * uncertainty_reduction
  + w_pred * calibrated_response_success
  + w_goal * target_skill_progress
  + w_engage * engagement/completion
  - w_cost * time_or_cognitive_load
  - w_repeat * repetition_penalty
  - w_unsafe * prerequisite_violation
```

Candidate operational rewards:

- **Assessment item**: expected theta improvement, entropy reduction, response
  likelihood, item information, QWK/accuracy-calibrated outcome.
- **Tutorial**: follow-up assessment improvement, reduced error on target KC,
  completion-adjusted gain.
- **Course/career**: progress toward declared goal, coverage of missing skills,
  human/teacher approval, delayed retention.

Important warning: correctness after recommendation is confounded by prior
ability. Evaluation should use counterfactual/off-policy methods and randomized
or teacher-approved trials where possible.

## Training Regimes

### Regime A: Frozen MA-IRT + Offline RL

Train MA-GPCM normally. Freeze it. Build logged transitions:

```text
(state_t, action_t, reward_t, state_{t+1}, done)
```

Train:

- behavior cloning;
- DQN or dueling DQN for small discrete item pools;
- CQL/IQL/BCQ for conservative offline RL;
- supervised learning-to-rank as a non-RL baseline.

This should be the first implemented research baseline.

### Regime B: Frozen MA-IRT + Model-Based Simulation

Use MA-IRT as a response simulator:

1. score candidate item response distributions;
2. sample or take expected response;
3. update memory state;
4. optimize policy against simulated learning objectives.

This is closer to ExRec. It must be validated against held-out human sequences
because a policy can exploit simulator artifacts.

### Regime C: Coupled Auxiliary Training

Keep MA-GPCM prediction/recovery losses primary. Add auxiliary recommender
losses on top of state features:

```text
L = L_MAIRT + lambda_rank * L_action_prediction + lambda_value * L_value
```

Gradient should initially update only recommender heads and optional projection
layers, not the core MA-GPCM encoder/decoder.

### Regime D: Constrained Joint Training

Allow selected MA-IRT representation layers to update under strict gates:

- no degradation on MA-GPCM smoke and regression metrics;
- no degradation in theta/alpha/beta recovery;
- KL regularization to the frozen MA-GPCM outputs;
- calibration checks on held-out response logs.

This is a later research contribution, not the skeleton.

## Proposed Architecture

Add modules in small, additive stages.

```text
ma-irt/
  online/
    state.py              # OnlineLearnerState dataclass
    magpcm_session.py     # read/score/write wrapper around MAGPCM
    candidate_scoring.py  # expected score, entropy, information proxies

  recommenders/
    envs/
      base.py             # RecEnv protocol
      mairt_env.py        # MAIRT-backed educational environment
      slate_env.py        # optional slate/list env
    policies/
      heuristic.py        # CAT, weakest-skill, prerequisite baselines
      dqn.py              # minimal DQN / dueling DQN
      conservative.py     # later CQL/IQL adapter
    rewards.py
    replay.py
    features.py

  scripts/
    train_recommender.py
    eval_recommender.py
    simulate_policy.py

  configs/recommenders/
    smoke_dqn_mairt.yaml
    smoke_heuristic_mairt.yaml
```

Do not place the RL policy inside `models/magpcm.py`. MA-GPCM should remain a
response model. The recommender is a decision layer that calls MA-GPCM.

## Online MA-IRT API

The minimum API should look like this:

```python
state = session.init_state(batch_size=1)
scores = session.score_candidates(state, candidate_item_ids)
action = policy.select_action(scores, state, candidate_features)
state = session.observe_response(state, item_id=action.item_id, response=r_t)
```

`score_candidates` should return:

- `item_id`;
- `probs`;
- `expected_score`;
- `entropy`;
- `theta`;
- `alpha`;
- `beta`;
- `student_summary`;
- `joint_summary`;
- `attention`;
- optional item/content metadata.

Implementation detail:

- Add `init_state`, `read_step`, and `write_step` helpers to `DKVMNEncoder`.
- Reuse `GPCMDecoder.forward` with an `EncoderOutput` of sequence length 1.
- Confirm that a loop of online `read_step/write_step` exactly matches
  whole-sequence `MAGPCM.forward` up to numerical tolerance.

This parity test is the first hard gate.

## Baselines

A credible paper needs strong non-RL baselines:

- random valid action;
- most popular / historical behavior policy;
- prerequisite-topological next item;
- weakest estimated theta/KC remediation;
- maximum item information / CAT-style policy;
- highest expected score;
- uncertainty sampling / diagnostic policy;
- supervised learning-to-rank;
- CaRReL-style DQN with theta-only state;
- DQN with full MA-IRT state;
- conservative offline RL policy.

The theta-only baseline is important: it directly tests whether MA-IRT's richer
deep IRT state improves over the earlier DRL-IRT attempt.

## Evaluation Plan

### MA-IRT Preservation

Every core edit must pass:

- existing shape and migration tests;
- online-vs-batched parity for MAGPCM;
- public smoke train/evaluate;
- representative MA-GPCM regression configs when feasible.

Acceptance: zero tolerated performance loss on MA-GPCM. If regression appears,
restore the frozen MA-GPCM behavior before proceeding.

### Recommender Offline Metrics

- logged-policy action prediction;
- off-policy evaluation when propensities are available or estimable;
- normalized discounted cumulative reward in simulator;
- safety/prerequisite violation rate;
- diversity and repetition rate;
- computational latency per decision.

### Learning Metrics

- expected score gain;
- theta gain and uncertainty reduction;
- target-skill mastery improvement;
- retention on delayed items;
- calibration of simulated vs real response outcomes.

### Ablations

- theta only vs theta + response probabilities;
- theta + alpha/beta vs full memory summary;
- no content graph vs prerequisite-constrained policy;
- frozen MA-IRT vs auxiliary coupled training;
- DQN vs conservative offline RL;
- single item vs slate.

## Development Roadmap

### Phase 0: Proposal And Audit

- Record this design.
- Audit current MA-IRT tests that protect `MAGPCM`.
- Confirm which datasets can support sequential recommendation evaluation.
- Inspect ExRec code more deeply and decide whether to vendor ideas only or
  create a small adapter.

Deliverable: proposal document and implementation checklist.

### Phase 1: Online MA-IRT State Adapter

- Add `OnlineLearnerState`.
- Add `DKVMNEncoder` step helpers without changing `forward`.
- Add `MAGPCMOnlineSession`.
- Add online/batched parity tests.

No recommender training yet.

### Phase 2: Candidate Scoring And Heuristic Policies

- Score candidate assessment items from a live state.
- Implement CAT-style item information, uncertainty sampling, weakest-skill,
  and expected-gain policies.
- Add a smoke simulation over synthetic data.

This phase should already demonstrate realtime response -> MA-IRT -> decision.

### Phase 3: RL Environment Skeleton

- Implement `MAIRTRecEnv`.
- Define transition records.
- Add replay buffer and DQN baseline.
- Add theta-only DQN baseline modelled after CaRReL.
- Add full-state DQN.

Phase goal: prove that richer MA-IRT state is usable by a policy.

### Phase 4: Offline RL And Counterfactual Evaluation

- Add logged-data training mode.
- Add conservative policy learning where possible.
- Add behavior-policy support checks.
- Add off-policy evaluation tooling.

Phase goal: avoid unsupported action recommendations.

### Phase 5: Resource/Course/Career Extension

- Add resource metadata schema:
  `resource_id`, `type`, `target_skills`, `prerequisites`, `duration`,
  `difficulty`, `embedding`.
- Map assessment items to skills and resources.
- Implement tutorial/course recommendation actions.
- Add follow-up assessment transitions.

Career recommendation should be treated as a separate domain instantiation with
longer-horizon rewards and weaker causal evidence.

### Phase 6: Coupled Training

- Add recommender auxiliary loss.
- Freeze MA-GPCM by default.
- Optionally unfreeze projection layers under regression gates.
- Add KL/calibration regularization to preserve MA-GPCM outputs.

Only attempt this after the frozen framework is stable.

## Immediate Skeleton Recommendation

The first code skeleton should be deliberately small:

1. `ma-irt/online/state.py`
2. `ma-irt/online/magpcm_session.py`
3. `ma-irt/recommenders/features.py`
4. `ma-irt/recommenders/policies/heuristic.py`
5. `ma-irt/tests/test_online_magpcm.py`

Do not implement PPO or CQL first. The first success criterion is stricter and
simpler:

> Online step-by-step MAGPCM outputs match whole-sequence MAGPCM outputs, and a
> heuristic recommender can select a next item from live MA-IRT candidate
> scores.

Once this is true, RL can be added without guessing the state interface.

## Open Research Questions

- What is the correct latent state for policy learning: theta, memory summary,
  item-conditioned GPCM parameters, or a learned projection of all of them?
- Can MA-IRT serve as a reliable simulator for counterfactual recommendations,
  or does it overfit to the historical item policy?
- How should tutorial/course actions update learner state when no immediate
  response is observed?
- Can the recommender improve downstream learning without degrading MA-GPCM's
  IRT recovery?
- How much semantic item/course metadata is required for generalization to new
  content?
- For vocational recommendation, what is the ethical and statistical definition
  of reward?

## Sources And Starting Points

- Local MA-IRT codebase:
  `ma-irt/models/registry.py`, `ma-irt/models/base.py`,
  `ma-irt/models/magpcm.py`, `ma-irt/models/encoders/dkvmn.py`,
  `ma-irt/models/decoders/gpcm.py`, `ma-irt/utils/dataloader.py`.
- Local CaRReL clone:
  `C:\Users\steph\Documents\CaRReL\model\env.py`,
  `model\agent.py`, `model\agentv2.py`, `model\networks.py`,
  `model\utils.py`, `model\NRT.py`.
- ExRec: https://github.com/oezyurty/ExRec and
  https://arxiv.org/abs/2507.11060.
- ALPN: https://arxiv.org/abs/2305.04475.
- CSEAL: https://arxiv.org/abs/1905.12470.
- RL4RS: https://github.com/fuxiAIlab/RL4RS and
  https://arxiv.org/abs/2110.11073.

