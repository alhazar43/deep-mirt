# DRL-MAIRT Plan Synthesis

Two parallel design passes have produced concrete plans for coupling a
DRL recommender to ma-irt's deep IRT state.

- Codex's two-document plan, **DRL-MAIRT**
  ([background](drl_mairt_background.md), [proposal](drl_mairt_recommender_plan.md)).
- A seven-agent Claude workflow plan, **deep-mirt-rl**, run on
  2026-06-03 (workflow id `wf_14078e6c-e20`). Raw outputs preserved at
  `docs/cleanup/_drl_workflow_digest.md`.

This file reconciles them, marks the points of agreement, names the
forks that need a decision, and recommends a hybrid path.

## What both plans agree on

These are not in question.

- Freeze ma-irt by default. Joint training is a research stretch goal
  with guardrails, not the v1 setting.
- A single integration boundary between ma-irt and the DRL code. Codex
  calls this `MAGPCMOnlineSession.score_candidates`; the workflow calls
  it `IRTBridge`. Same idea, same surface.
- The bidirectional inference contract. response -> ma-irt step ->
  belief b_t -> candidate scoring -> policy action -> next item -> next
  response.
- A layered state for the policy, not just `theta_t`. theta, item
  parameters of the candidate set, encoder hidden summaries (student
  and joint), and recent-history features.
- Multi-component reward. Information gain, learning gain, uncertainty
  reduction, and exposure/repetition penalties. Both plans warn that
  rewarding raw correctness is confounded by prior ability.
- CaRReL's failure mode is the post-mortem driving the design. Its
  diagnosis (theta-only state, MLE IRT, indexed Q-head, reward gradient
  leaking into the world model) is what each architectural decision in
  both plans is built to avoid.
- Pre-register the baseline set, including a CaRReL-stripped
  replication, before any RL training. Codex calls CaRReL a "negative
  baseline"; the workflow names the same ablation.
- Online step must equal whole-sequence forward up to numerical
  tolerance. Both treat this as gating, but they disagree about when
  the test gates (see Forks).

## Where they disagree

Five real forks.

### F1. Repo placement

- Codex puts everything as additive folders inside `ma-irt/`
  (`ma-irt/online/`, `ma-irt/recommenders/`, `ma-irt/scripts/`).
- The workflow proposes a sibling repo `deep-mirt-rl/` at
  `C:/Users/steph/documents/`, with `ma-irt/` pulled in as a git
  submodule under `vendor/ma-irt/`.

The workflow's argument is that recommender code, FastAPI service,
RL training infrastructure, and policy checkpoints would bloat what is
now a clean public-release ML repo. Codex's argument is implicit, the
recommender is a natural extension of the response model, so keep it
under one roof.

### F2. First gate

- Codex: online-vs-batched parity for `MAGPCMOnlineSession` is the
  first hard gate. No recommender code, no environment, no policy
  until iterated `model.step` matches `model.forward(full_seq)` on
  logits, probs, theta, alpha, beta, attention.
- Workflow: the first concrete step is "create the deep-mirt-rl
  skeleton". Parity tests are still mandatory and land as part of P1,
  but they happen alongside the new repo bootstrap rather than before
  it.

### F3. Algorithm commitment

- Codex: deliberately uncommitted. Implement order is heuristics, then
  behavior cloning, then DQN/dueling DQN, then CQL/IQL/BCQ. PPO and CQL
  are explicitly **not** the first cuts.
- Workflow: commits to PPO as the v1 algorithm, with concrete
  hyperparameters (clip 0.2, GAE 0.95, gamma 0.99, lr 1e-4,
  mini-batch 512, n_epochs 4 per rollout, entropy 0.01 -> 0.001 anneal,
  max-grad-norm 0.5).

### F4. Use case priority

- Codex: assessment-item recommendation is the first reward task,
  because the reward (expected theta gain, uncertainty reduction) is
  fully measurable with no real-data dependency. Course/tutorial/career
  are later domain instantiations.
- Workflow: course/assignment recommendation is the v1 (config
  `course_rec_dkvmn_default.yaml`); vocational assessment is P10
  (optional) and ~6 weeks of additional work.

### F5. Action space scope

- Codex: item, resource/tutorial, slate, and path actions all live in
  one framework from day one, with both discrete item-id actions and
  continuous parametric embedding actions plus nearest-neighbour
  projection.
- Workflow: item recommendation only in v1, pointer-network
  ItemEmbedScorer handles unseen items via the embedding head;
  vocational/career classification action is P10.

## Where one plan adds something the other lacks

Codex adds:

- An explicit four-regime taxonomy of training topologies
  (frozen+offline, frozen+sim, coupled-aux, constrained-joint) with
  named loss forms.
- A set of six research hypotheses (H1 to H6) that double as a
  publication claim ladder.
- Discipline around reward reporting. Components must be reported
  separately, not collapsed.
- POMDP framing rather than MDP, which is more honest about the latent
  z_t being partially observed.

The workflow adds:

- Concrete hyperparameters across the stack.
- A realtime latency budget. p95 <= 60ms on DKVMN, <= 100ms on
  Transformer. Microbenchmark CI test as a gate.
- An 80/10/10 item-bank split with a pointer-network scoring head as a
  named publishability hook. Held-out-item generalization is reported
  as a primary metric.
- 12 named baselines including MAAT, NCAT, BOBCAT, CSEAL,
  ExRec-best-variant, GMOCAT, and 12 pre-registered ablations.
- Per-phase cost estimates (agent rounds, GPU hours).
- A FastAPI service skeleton (`/session/start`, `/step`, `/end`) as
  P8.
- A hard auto-revert guardrail on the joint-fine-tune phase. Any IRT
  recovery correlation drop > 0.05 from the frozen baseline triggers a
  checkpoint restore.

## Recommended path (hybrid)

Pick a hybrid. Each fork resolved with the stronger of the two
arguments.

- **F1 repo placement**: sibling repo `deep-mirt-rl/`. The workflow is
  right that recommender training, FastAPI service, and policy
  artefacts do not belong inside a paper-anchor ML repo. ma-irt stays
  clean, the new repo vendors it via a submodule.
- **F2 first gate**: Codex's discipline wins. The first deliverable is
  the ma-irt online-step API plus parity tests, landed inside ma-irt as
  a tracked PR. Only after the parity gate is green does the sibling
  repo skeleton get created.
- **F3 algorithm**: Codex wins. Start with heuristics (MaxFisher, KLI,
  weakest-skill) and behavior cloning. Add DQN. PPO becomes a later
  experiment, not the first cut. Workflow's PPO hyperparameters are
  preserved as reference for when we get there.
- **F4 use case**: Codex wins. Assessment-item recommendation is v1
  because it has a clean measurable reward and no real-data
  dependency. Course rec is v1.5. Vocational career is later.
- **F5 action space**: workflow wins. Item-only first, with the
  pointer-network embedding head. Resource, slate, and path actions
  are explicit extension points but not built in v1. The embedding
  head means unseen items already work without redesign.

Keep both plans' guardrails, the joint-training auto-revert from the
workflow and the KL-to-frozen-MAGPCM regularizer from Codex are
complementary.

Keep Codex's H1 to H6 as the publication-claim ladder. Keep the
workflow's pointer-scorer + 80/10/10 item split as the main novelty
hook. Keep the workflow's named baseline set (MAAT, NCAT, BOBCAT,
CSEAL, ExRec, GMOCAT) as the reviewer-facing comparison matrix.

## Hybrid phasing

A phase ordering that respects all five resolved forks.

- **H0. Lock the spec.** Update Codex's proposal in place to reflect the
  five resolved forks. No code yet. ~1 session.
- **H1. ma-irt online step API.** Land inside ma-irt as a tracked PR.
  Add `EncoderDecoderModel.step`, per-encoder `forward_with_state`,
  per-decoder `compute_logits_from_state`, `StepState` dataclass,
  `freeze_irt` helper, equivalence tests, microbenchmark CI test, and
  `docs/step_api.md`. This is Codex's first hard gate and the
  workflow's P1 combined. ~8 to 10 agent rounds, 2 to 4 GPU hours.
- **H2. Sibling repo skeleton.** Create `deep-mirt-rl/` at
  `C:/Users/steph/documents/`, wire `vendor/ma-irt/` submodule, write
  the `IRTBridge`, `StepState`, `StepBundle`, `StudentSource` ABC,
  `SimStudent`, `ReplayStudent`. Equivalence tests at the bridge
  boundary. ~6 to 8 agent rounds, 0 GPU hours.
- **H3. Heuristic + offline baselines.** `MaxFisherInfoPolicy`,
  `KLIPolicy`, `BanditCAT`, `RandomPolicy`, `BehaviorClonePolicy`,
  `PopularityPolicy`. The "candidate scoring + heuristic policies"
  phase of Codex's plan, mapped onto the workflow's
  `@register_baseline` registry. ~4 to 6 agent rounds, 2 to 5 GPU
  hours.
- **H4. Env + reward + state bundle.** `StudentEnv`, `CandidateBuilder`
  with composable rules, `StateBundle`, `RewardConfig` with the
  workflow's named terms (InfoGain, LearningGain, RepeatPenalty,
  FrustPenalty) and Codex's reporting discipline (every component
  logged separately). ~6 to 8 agent rounds, < 1 GPU hour.
- **H5. DQN/dueling DQN first end-to-end RL.** Codex's recommended
  first algorithm. PolicyNet, ValueNet, ItemEmbedScorer, RolloutBuffer,
  trainer. Smoke training on SimStudent. ~8 to 10 agent rounds,
  10 to 20 GPU hours.
- **H6. Offline RL warm-start.** BC + FQE + doubly-robust OPE per the
  workflow's P7. ~8 to 10 agent rounds, 10 to 20 GPU hours.
- **H7. Pointer-scorer held-out-item evaluation.** The named
  publishability hook. 80/10/10 item-bank split, train policy on 80%,
  evaluate on held-out 10%. ~3 to 4 agent rounds, 5 to 10 GPU hours.
- **H8. PPO graduation.** Only after DQN and BC are solid. The
  workflow's full P3 simulator RL config with named hyperparameters.
  ~10 to 14 agent rounds, 30 to 60 GPU hours.
- **H9. Joint fine-tune (OPTIONAL).** Workflow's P9 with auto-revert
  gate, Codex's KL-to-frozen-MAGPCM regularizer added. ~10 to 12 agent
  rounds, 40 to 80 GPU hours.
- **H10. FastAPI realtime service (OPTIONAL).** Workflow's P8 with
  load tests at the named latency budget. ~6 to 8 agent rounds.
- **H11. Course rec + vocational extensions (OPTIONAL).** Workflow's
  P10. Resource/tutorial action space and terminal classification head.
  ~8 to 10 agent rounds.

## Decisions still needed from the user

These six are blockers for finalising the spec. Numbered to match the
workflow's open-question list where they overlap.

1. **Repo placement.** Adopt the hybrid recommendation (sibling repo
   `deep-mirt-rl/`)? Or follow Codex's additive-folders proposal and
   keep it all inside `ma-irt/`?
2. **First use case.** Adopt the hybrid recommendation
   (assessment-item recommendation as v1, course rec as v1.5)? Or
   follow the workflow's course-rec-first ordering?
3. **Algorithm starting point.** Adopt the hybrid recommendation
   (heuristics + DQN first, PPO later)? Or commit early to PPO per the
   workflow?
4. **Real-data availability.** Is real student data accessible for
   either use case (EdNet, ASSISTments09 already in ma-irt, course
   logs, vocational survey corpus)? If yes, which? This determines
   whether H6 has real targets or is synthetic-only.
5. **Publication target and deadline.** IJAIED long, AIED short, or
   ML-side (RLC / NeurIPS D&B)? The phasing budget and ablation depth
   should be set by the deadline.
6. **GPU budget.** Rough A100-hours per month available. H8 PPO at 64
   parallel sims, T=64, 5M steps, 5 seeds is about 200 to 300
   A100-hours. The full ablation matrix adds about 400 more.

## File reference

- [drl_mairt_background.md](drl_mairt_background.md) - Codex's
  feasibility dossier (465 lines).
- [drl_mairt_recommender_plan.md](drl_mairt_recommender_plan.md) -
  Codex's proposal (733 lines).
- `docs/cleanup/_drl_workflow_digest.md` - the seven-agent workflow
  raw outputs.
- `docs/cleanup/_drl_codex_digest.md` - the digest of Codex's two docs
  that this synthesis is built on. (Optional, can be deleted if the
  Codex docs are sufficient.)
