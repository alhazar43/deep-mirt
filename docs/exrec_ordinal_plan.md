# OrdRec, Ordinal Exercise Recommendation with MA-IRT

Status, draft v1, written 2026-06-04. Supersedes the archived `docs/archive/jobrec/` plan.
Author, research-scientist agent synthesis.
Project name (proposed), **OrdRec** (Ordinal-IRT Exercise Recommendation).
One-line summary, OrdRec is an ExRec-style item-recommendation system that uses a frozen MA-IRT ordinal GPCM model as the world model, with a model-based RL policy trained to maximise posterior-entropy reduction and per-trait mastery gain on real student response sequences from Eedi NeurIPS 2020.

---

## 1. Pivot statement

The original DRL-MAIRT project targeted job recommendation on a synthetic O*NET-derived simulator. That direction failed two structural tests on its own simulator. Under 1D theta with K=5 GPCM cumulative-tail preference, the job ranking is theta-invariant, so the recommendation-over-time curve is mathematically flat. The simulator-defined preference field has no external grounding source, so any win is internal to the simulator.

OrdRec reframes the same RL + MA-IRT machinery around the item recommendation problem on real KT data. The agent picks the next test item to administer, the reward is intrinsic knowledge gain measured by the frozen MA-IRT belief, and the data-generating process is real student responses, not a synthetic generator. The closest published template is ExRec (Ozyurt et al. 2025, arXiv 2507.11060). The differentiator is that MA-IRT is ordinal (K > 2 GPCM) where ExRec's KT model is binary, which delivers strictly more Fisher information per response and changes the reward geometry materially.

---

## 2. Background

**Why ExRec is the template.** ExRec is the most recent, cleanest published system that wires a knowledge-tracing world model into an RL policy for personalised exercise recommendation. It has four parts. A semantic question encoder (frozen text model plus contrastive projection), a calibrated KT world model with a KC-conditioned readout head, an RL agent trained with model-based value estimation (MVE) initialising the critic from the KT parameters, and a retrieval step at test time over the semantic embedding space. ExRec demonstrates four reward variants (global, practiced, upcoming, weakest KC improvement) on XES3G5M and shows that value-based continuous RL with MVE consistently beats policy-based methods and discrete methods. We copy the four-module structure, the offline-pretrain-then-RL recipe, the MVE critic, the 10-step episode horizon, and the percentage-of-maximum-achievable-improvement metric.

**Why ordinal KT is the differentiator.** ExRec's KT model is binary. For GPCM with K=5 equally spaced symmetric thresholds, Fisher information at the modal point is approximately 6 to 8 times larger than for 2PL at fixed alpha (Muraki 1993). The consequences for the RL system are concrete. Per-step entropy-reduction shaping rewards are 6 to 8 times denser at the modal point. The posterior reaches a target standard error of 0.3 in roughly one-sixth the items (4 to 6 ordinal items versus 30 binary items, van der Linden and Glas 2010). The information-greedy action is less concentrated on a narrow band of high-alpha modal-difficulty items, which softens the item-exposure pathology of binary maximum-Fisher-information policies (Sympson and Hetter 1985). MA-IRT carries psychometrically calibrated theta, alpha, beta with paper-verified recovery, so the belief is a measurement, not just a sufficient statistic, which makes the reward function interpretable and the per-step shaping invariant to the alpha-and-spread reward-hacking channel when defined as entropy reduction rather than raw Fisher information.

**What we bring that ExRec lacks.** A polytomous GPCM world model (K > 2), a Q-matrix-aware multi-trait MA-IRT (D > 1 with orthogonality penalty), an online step API with parity tests (feat/online-step-api branch), and a per-step CPU step budget under 15 ms verified across DKVMN, LSTM, and Transformer encoders. None of these exist in ExRec or its reference implementation.

---

## 3. Architecture

Pipeline from response to recommendation.

```
                     +-------------------------------------------+
                     |          frozen MA-IRT (D >= 1)           |
   history_t -----> | encoder E_theta (DKVMN | LSTM | Transformer)
   (q_i, y_i)       |    + IRTParameterExtractor (theta, alpha, beta)
                     |    + GPCMLogits (K-way categorical)        |
                     +-------------------------------------------+
                                       |
                                       v
                              StepState s_t = {h_t, theta_t,
                                              sigma_t, alpha_log,
                                              beta_log}
                                       |
                +----------------------+----------------------+
                |                                             |
                v                                             v
        +---------------+                            +-----------------+
        |  Actor pi(.|s) |  ---- chooses q_(t+1) --> | Retrieval index |
        +---------------+                            | (discrete bank  |
                |                                    |  or FAISS over  |
                |  z_q_(t+1) embedding               |  KC embeddings) |
                |                                    +-----------------+
                v
       +----------------+         simulate K outcomes
       |  Critic Q(s,a) | <----   via model.step(q, k=0..K-1)
       +----------------+         (model-based value estimation,
                |                  expectation under GPCM probs)
                v
        observe y_(t+1) ---> model.step(q_(t+1), y_(t+1), s_t) ---> s_(t+1)
                |
                v
        reward r_t = phi(s_t) - phi(s_(t-1))  (entropy reduction)
                  + 1[t = T] * w_voi * VOI_terminal
```

**Module 3.1, MA-IRT world model.** Frozen at deploy time. Three operational methods that mirror ExRec's init_states / predict_in_rl / update_hidden_state, already implemented as `model.step` on feat/online-step-api.

- `init_states(history)` runs the encoder forward over the first roughly 30 historical responses to build an opaque `StepState` containing `h_t` (encoder carry), `theta_t` (D-dim ability), `sigma_t` (posterior standard deviation, computed via Laplace approximation on the GPCM log-likelihood), and running tensors for alpha and beta needed for Fisher information.
- `predict_in_rl(s_t, q)` calls `model.decoder.predict_categorical(theta_t, q)` to return the K-way GPCM probability vector for a candidate item q without advancing the state.
- `update_hidden_state(s_t, q, y)` is exactly `model.step(q, y, s_t)`, advancing the encoder hidden in-place and refreshing theta and sigma.

For multi-trait MA-IRT (D > 1) the per-trait mastery readout is simply `theta_t[c]` for trait c. This replaces ExRec's KC calibration head, the orthogonality penalty during MA-IRT training ensures the traits are identifiable.

**Module 3.2, Actor and critic.** Both share an input encoder `phi(s_t) = MLP([theta_t, sigma_t, h_t_compressed])` where `h_t_compressed` is a small linear projection of the encoder hidden to keep input dimension bounded. The actor head produces a categorical distribution over the Q-item bank (v1, discrete actions) or a continuous embedding in R^d (v2, optional). The critic head is a scalar Q(s, a). For PPO the actor outputs categorical logits and the critic outputs a state value V(s); for SAC and TD3 the actor outputs an action embedding and the critic is twin-headed. Hidden width 256, two layers, GELU, layer norm. Following ExRec, the critic is initialised by warm-starting its phi(s) encoder from a copy of the MA-IRT encoder's parameters when the world model encoder is DKVMN or LSTM; for Transformer encoders we use a smaller learned phi(s) and rely on MVE for the value bootstrap.

**Module 3.3, Retrieval index.** v1 uses a plain discrete action over the Q-item bank (948 items on Eedi Task 3+4), so no retrieval is needed. v2 wires a FAISS IVF index over a KC-aware item embedding `z_q = [Q_matrix_row_q ; learned_residual_q]` so that an actor's continuous output can generalise to unseen items. The retrieval code already exists in `rl/src/irtrec/retrieval/index.py` and is reusable verbatim.

**Module 3.4, Reward computation.** Implemented as a thin wrapper that takes `(s_{t-1}, q_t, y_t, s_t)` and returns the per-step reward described in Section 4. The reward never touches the policy parameters directly, it reads only from the frozen MA-IRT through `predict_in_rl` and from the updated state through `model.step`.

**Tensor shapes (D=1, K=5, Q=948).** `theta_t` is `(D,) = (1,)`, `sigma_t` is scalar, `h_t` is `(d_h,)` with `d_h = 64` for DKVMN. `phi(s_t)` is `(256,)`. Actor categorical logits are `(Q,) = (948,)`. Critic Q-value is `(1,)`. MVE expansion per (s, q) requires K=5 forward `step` calls, batched.

---

## 4. Reward function

**Choice.** A potential-based per-step shaping (entropy reduction) plus an ordinal terminal anchor (value-of-information NLL). This is the recommendation from the psychometric audit in the Phase 1 bundle and is selected over the four ExRec variants because (a) it generalises to D=1 and D>1 MA-IRT uniformly, (b) it is potential-based so per-step shaping leaves the optimal policy invariant (Ng, Harada, Russell 1999), (c) it is invariant to the alpha-and-spread reward-hacking channel that direct Fisher-information rewards are exposed to, and (d) the terminal NLL anchor is non-circular against MA-IRT because the probe items used for terminal evaluation are held out from the policy's action set.

**Math.** Let `s_t` be the agent state, `theta_hat_t` and `sigma_t` be the Laplace-approximated posterior mean and standard deviation from the frozen MA-IRT belief, `y_t in {0, ..., K-1}` the observed ordinal response at step t, `q_t` the recommended item, and `(alpha_{q_t}, beta_{q_t})` the calibrated item parameters. Define the per-step reward,

    r_step_t = w_info * (phi(s_t) - phi(s_(t-1)))
             - w_cost
             - w_expo * 1[q_t in recent_window_t]

where the potential is the negative differential entropy of the 1D Laplace posterior,

    phi(s) = -0.5 * log(2 * pi * e * sigma^2)

so the entropy reduction equals `log(sigma_(t-1) / sigma_t)`. In closed form, with Laplace update,

    1 / sigma_t^2 = 1 / sigma_(t-1)^2 + I_GPCM(theta_hat_(t-1); alpha_(q_t), beta_(q_t))
    I_GPCM(theta) = alpha^2 * Var(X | theta, alpha, beta)

and therefore `r_step_t (w_info=1 only) approx 0.5 * log(1 + sigma_(t-1)^2 * I_GPCM)`, an explicit closed form in the polytomous Fisher information. For multi-trait MA-IRT (D > 1) the scalar `sigma_t` is replaced by `det(Sigma_t)^{1/(2D)}` and the shaping reward becomes the log-determinant ratio of the posterior covariance.

The terminal reward at session end T is an ordinal VOI term,

    r_term_T = w_voi * (NLL_prior - NLL_posterior_T)

where

    NLL_posterior_T = -(1/|H|) * sum_{(q_j, y_j) in H} log P(Y = y_j | theta_hat_T, alpha_(q_j), beta_(q_j))

`H` is a held-out probe of items not eligible for selection during the session, `y_j` is the actual ordinal response from the real student log, and NLL_prior is the same expression evaluated at the prior `theta_hat_0`. The composite is

    r_t = r_step_t + 1[t = T] * r_term_T

**Default weights.** `w_info = 1.0, w_cost = 0.05, w_expo = 0.1, w_voi = 5.0`. Exposure window length 5.

**Why this avoids reward-hacking.** A naive policy rewarded for `I_GPCM` directly can prefer items with widely spaced uninformative thresholds that inflate Var(X) without genuinely reducing posterior uncertainty (Muraki 1993 warns of this). Posterior entropy reduction in a Laplace approximation is invariant to that parameterisation, since it depends on the actual Bayesian update of sigma, not on the raw item-level information. The terminal NLL anchor is computed on a held-out probe that the policy could not have steered toward, so the policy cannot pick action sequences that artificially make MA-IRT look well-calibrated on the policy's own choices.

**Why this is non-circular against MA-IRT.** The per-step potential is computed from the frozen MA-IRT belief, but the optimal policy under a potential-based shaping reward is invariant to the choice of potential (Ng et al. 1999). The terminal anchor is computed on a probe set held out from policy action, so its expectation under any policy is the same up to the policy's effect on the final `theta_hat_T`. The reward therefore measures "how well the policy got MA-IRT to a posterior that predicts held-out real responses", which is grounded in real data, not in the simulator.

---

## 5. RL algorithm choice

**Choice.** **PPO** (Schulman et al. 2017), discrete-action over the Q-item bank, with the MVE-style critic warm-start adapted to ordinal GPCM as described below.

**Hyperparameters.**

| Parameter | Value | Rationale |
|---|---|---|
| Algorithm | PPO (Tianshou implementation) | Stable, on-policy, well-tested on Tianshou, matches ExRec's PPO variant. |
| Action space | Discrete over Q items (Eedi 948) | Matches MA-IRT's discrete item bank; no retrieval index needed in v1. |
| Clip eps | 0.2 | Standard PPO. |
| GAE lambda | 0.95 | Standard. |
| Discount gamma | 0.95 | Episode is 10 steps, so effective horizon is short and gamma < 1 keeps the terminal anchor's contribution well-conditioned. |
| Learning rate | 3e-4 (actor and critic) | Standard PPO. |
| Optimizer | Adam, eps 1e-5 | Standard. |
| Entropy bonus | 0.01 | Encourages item diversity, mitigates exposure collapse. |
| Rollout length | 16 episodes per update | Each episode is 10 steps, so 160 (s,a,r,s') per update. |
| Mini-batch size | 64 | Standard. |
| Epochs per update | 4 | Standard. |
| Value coef | 0.5 | Standard. |
| Max grad norm | 0.5 | Standard. |
| Warm-up | Behaviour cloning to a baseline policy (max-info on prior) for 200 updates | Stabilises early training, denser reward signal. |
| Critic encoder init | Linear projection from frozen MA-IRT encoder hidden | MVE-style warm-start, see below. |

**Why PPO over alternatives.**

- **vs DDPG, TD3, SAC.** ExRec found continuous value-based methods dominate continuous policy-based methods *on Task 4 with QDKT*, the hardest non-stationary-target reward variant. We use a discrete action space (item bank, not continuous embedding), so the continuous-value-based advantage does not directly apply. PPO with discrete action is the natural Tianshou choice. SAC-discrete is a viable alternative for v1 ablation.
- **vs DQN, C51, Rainbow.** Discrete value-based methods plateaued on ExRec's Task 4 because the target shifts dynamically. Our shaping reward is entropy reduction, not a moving KC target, so the non-stationarity argument is weaker. PPO is more stable than DQN at small batch sizes and is the Tianshou default for discrete policy gradient.
- **vs offline RL (CQL, BCQ).** Offline RL is attractive because we have a large logged corpus, but the world model is frozen and queryable, so we can generate on-policy rollouts cheaply by replaying the world model. On-policy PPO with simulated rollouts is therefore the right tool, exactly the ExRec recipe.

**MVE-style critic warm-start for ordinal GPCM.** ExRec's MVE expands the value as a binary-outcome expectation (`y_hat * y_c^1 + (1-y_hat) * y_c^0`). Our version is a K-way expectation,

    Q_MVE(s_t, q) = sum_{k=0..K-1} P(Y_(t+1) = k | theta_hat_t, alpha_q, beta_q) * R_k(s_t, q)

where `R_k(s_t, q)` is the per-step reward that *would* be received if the response were category k, computed by `model.step(q, k, s_t)`, reading the resulting `sigma_t` and applying the shaping formula. This is K=5 forward `step` calls per (state, action), which is cheap (under 75 ms per (s, q) on CPU given the 15 ms step budget). The critic is initialised by training it to match `Q_MVE` for the first 200 PPO updates on the BC warm-start rollouts, then handed over to the standard PPO TD-lambda update.

---

## 6. MDP specification

| Component | Specification |
|---|---|
| **State `s_t`** | `(h_t, theta_t, sigma_t)` from the frozen MA-IRT step API. Shapes for D=1, DKVMN encoder, `h_t in R^{64}`, `theta_t in R^{1}`, `sigma_t in R^{1}`. Concatenated and projected through a learned `phi(s_t) in R^{256}` for the policy. |
| **Action `a_t`** | Discrete, `a_t in {0, ..., Q-1}`, `Q = 948` for Eedi Task 3+4. v2 optional continuous `a_t in R^d` with d = 64, retrieved via FAISS over KC-aware item embeddings. |
| **Transition** | Deterministic in `s_t` given `(q_t, y_(t+1))`, where `y_(t+1) ~ P_GPCM(. | theta_(t+1), alpha_{q_t}, beta_{q_t})`. In training, `y_(t+1)` is sampled from the frozen MA-IRT predictive (model-based rollout); in evaluation, `y_(t+1)` is the held-out real response if available, otherwise sampled. |
| **Reward `r_t`** | Section 4 formula. Scalar. |
| **Horizon `T`** | 10 recommendation steps, matching ExRec. |
| **Discount `gamma`** | 0.95. |
| **Episode start** | Sample a student from the train cohort; replay the first `H_init = 30` historical responses through `model.step` to initialise `s_30`. (ExRec uses 100; 30 is more appropriate for Eedi Task 3+4 since median student has ~150 responses there.) |
| **Episode end** | After T=10 recommendation steps, emit terminal reward `r_term_T`. |

**Mask handling.** Items already seen in the initial history or in the current episode are masked to `-inf` in the actor logits to prevent recommending an already-administered item.

---

## 7. Dataset choice

**Primary, Eedi NeurIPS 2020 Education Challenge, Task 3+4 split.** 9,401 student sequences, 948 questions, 57 KCs, 1,399,470 interactions. K=4 multiple-choice items where each question has exactly one correct option and three nominal distractors. Native ordinal coercion via empirical-difficulty ordering of distractors. Train/val/test 80/10/10 by student id, using the official Task 4 split.

**Why Eedi.** It is the only widely used public KT benchmark that has native K=4 multiple-choice option labels per response, which is the structural prerequisite for exercising MA-IRT's ordinal GPCM advantage. The Task 3+4 split is sized for sequence modelling, large enough to train DKVMN-style memory but small enough to iterate fast. The 57 KCs support the KC-aware reward variants and the Q-matrix-aware multi-trait MA-IRT extension. Published baselines from the NeurIPS 2020 challenge and from pyKT/SimpleKT give a dense set of head-to-head comparisons. The challenge itself is an item-recommendation evaluation (Task 3 active learning, Task 4 prediction after 10 queries), so the ExRec framing translates directly.

**Secondary, XES3G5M.** 18,066 students, 7,652 questions, 865 KCs, 5,549,635 interactions, binary K=2. Used for direct head-to-head with ExRec on the same dataset. MA-IRT collapses to 2PL-IRT here, isolating the recommendation policy contribution from the ordinal response contribution. Train/val/test 5-fold 80/10/10 by student id (canonical XES3G5M protocol).

**Tertiary fallback, ASSISTments 2009.** Binary, sanity check only. Verify MA-IRT does not regress against published KT baselines on the most-benchmarked KT dataset.

**Preprocessing pipeline (Eedi).**

1. Load `train_task_3_4.csv` from the NeurIPS 2020 challenge release.
2. Sort each student's responses by timestamp.
3. Drop students with fewer than 20 responses.
4. Truncate sequences to `max_seq = 200`.
5. Map `QuestionId` to a contiguous `0 .. Q-1` index using train-fold appearance order.
6. Compute distractor ordering per question on train fold only. For each question q, rank its 3 distractors by the empirical mean ability of students who chose them (proxy ability is the student's overall accuracy on train items not equal to q). Define ordinal category `0 = least-able-distractor`, `1 = mid-distractor`, `2 = most-able-distractor`, `3 = correct`. Map `AnswerValue` (1..4) to ordinal category 0..3 using this ordering.
7. Persist the ordering for use at eval time.
8. Load `subject_metadata.csv` and build a `Q x 57` binary Q-matrix.

For ablation, run also with (b) K=2 binary collapse via `IsCorrect` to verify the ordinal advantage. Optionally run (c) random distractor ordering to verify the ordering is meaningful, not a free parameter.

**Preprocessing (XES3G5M and ASSISTments 2009).** Follow pyKT canonical preprocessing, no ordinal coercion (both are natively binary).

---

## 8. Baselines

The published baselines we are trying to beat on the primary dataset.

| # | Method | Citation | Reported metric | Reported number |
|---|---|---|---|---|
| 1 | Random policy | (Trivial baseline) | %-of-max-improvement, 10-step horizon | ~0% (definitional lower bound) |
| 2 | Historical replay (replay actual student responses in their original order) | ExRec, Ozyurt et al. 2025, arXiv 2507.11060 | %-of-max-improvement, 10-step horizon | 20--40% on XES3G5M (varies by reward variant) |
| 3 | Max-info on prior (greedy, picks `argmax_q I_GPCM(theta_hat_t; alpha_q, beta_q)`) | van der Linden and Glas 2010, *Elements of Adaptive Testing* | Posterior SE after T items | Strong CAT baseline; matches RL within a few percent on simple settings |
| 4 | DKT, DKVMN, SAKT, AKT (Track A, prediction only) | Liu et al. 2022, pyKT, NeurIPS 2022 D&B, arXiv 2206.11460 | AUC on Eedi Task 1 | DKT 0.7644, DKVMN 0.7673, SAKT 0.7546, AKT 0.7853 |
| 5 | Best Eedi NeurIPS 2020 Task 4 submission (Ghosh and Lan, meta-learning + active learning) | Wang et al. 2021, AISTATS 2021, arXiv 2104.04034 | Accuracy after 10 adaptive queries | 74.74% |
| 6 | Option Tracing | Ghosh and Lan 2021, AIED 2021, arXiv 2104.09043 | Option-level accuracy on Eedi | 55--60% on 4-way option prediction |
| 7 | ExRec, QDKT + PPO (binary KT, four reward variants, no MVE) | Ozyurt et al. 2025, arXiv 2507.11060 | %-of-max-improvement, 10-step horizon | Reported by reward variant, see paper Fig. 3 |
| 8 | ExRec, QDKT + DDPG/TD3/SAC + MVE | Ozyurt et al. 2025, arXiv 2507.11060 | %-of-max-improvement, 10-step horizon | 20--40% relative gain over PPO baseline |
| 9 | AKT-R (best published on AS2009) | Ghosh, Heffernan, Lan 2020, arXiv 2007.12324 | AUC on AS2009 | 0.8346 +/- 0.0036 |
| 10 | simpleKT | Liu et al. 2023, ICLR 2023, arXiv 2302.06881 | AUC on AS2009 | Competitive with AKT (~0.78) |

For the **Track B recommendation comparison**, baselines 1, 2, 3, 7, 8 are the direct comparators on the 10-step horizon. The proposed method must beat baselines 1, 2, 3 by a clear margin and at least match baselines 7, 8 *on XES3G5M* (the secondary dataset, head-to-head with ExRec). On the primary dataset Eedi, no published ExRec numbers exist, so we report against baselines 1, 2, 3 and against a re-implemented ExRec-binary on Eedi (ablation, MA-IRT collapsed to K=2 with same RL recipe).

For the **Track A prediction comparison**, baselines 4, 5, 6 are the direct comparators on Eedi, baselines 4, 9, 10 on AS2009. The proposed MA-IRT world model must at least match AKT on AUC.

---

## 9. Evaluation

**Track A, prediction.** Train MA-IRT on the train fold, evaluate on test fold.

- Metrics, next-response AUC (collapsing K=4 ordinal to correctness), per-category accuracy on the K=4 ordinal target, Brier score on the ordinal categorical, QWK between predicted argmax and true category.
- Statistical protocol, 5-fold cross-validation, paired bootstrap over test students with 10,000 resamples for confidence intervals.
- Comparators, baselines 4, 5, 6 on Eedi; baselines 4, 9, 10 on AS2009.

**Track B, recommendation.** Freeze MA-IRT, hold out 1,000 test students.

- Protocol, observe first 30 responses per test student to build a belief, then run RL policies for 10 recommendation steps, sampling responses from the frozen MA-IRT predictive distribution (model-based evaluation, matching ExRec's offline evaluation since true counterfactual responses to recommended items are unavailable in real data).
- Metrics,
  - **Primary,** percentage-of-maximum-achievable-improvement (matching ExRec's metric), where maximum is the oracle policy that has access to the true theta and picks the item-sequence maximising terminal `theta_hat_T - theta_hat_0`.
  - **Secondary,** terminal NLL on held-out probe items (the VOI reward at evaluation time, independent of training reward).
  - **Tertiary,** recommendation-quality-over-time curve, plotting per-step `sigma_t` and per-step `theta_hat_t` for the policy.
- Statistical protocol, paired bootstrap over 1,000 test students with 10,000 resamples for CIs.
- Comparators, baselines 1, 2, 3, 7, 8.

**Ablations to run.**

| Ablation | Question answered |
|---|---|
| A1, K=4 ordinal vs K=2 binary collapse | Does the ordinal advantage matter? |
| A2, Entropy-reduction reward vs raw I_GPCM reward | Does the alpha-and-spread hacking channel materialise? |
| A3, Random distractor ordering vs empirical-difficulty ordering | Is the ordinal coercion non-trivial? |
| A4, D=1 vs D=2 vs D=4 multi-trait MA-IRT | Does multi-trait help, or is 1D enough? |
| A5, MVE critic warm-start vs randomly initialised critic | Does the ExRec MVE trick port to ordinal? |
| A6, Frozen MA-IRT vs jointly trained MA-IRT | Validate the offline-pretrain-then-RL recipe. |
| A7, Replay buffer of real student responses vs simulated rollouts | Reward grounding sanity check. |
| A8, Different encoders (DKVMN, LSTM, Transformer) | Does the policy depend on the world-model architecture? |

---

## 10. Milestones

E-prefixed to break with the M-prefixed archived plan.

| Milestone | Scope | Deliverables | Estimated cost |
|---|---|---|---|
| **E1, Eedi pipeline and MA-IRT calibration** | Build the Eedi loader with K=4 ordinal coercion, fit MA-IRT on Eedi Task 3+4, achieve AUC at least equal to AKT (0.785) on Track A. | `dataloading/eedi.py`, `scripts/prepare_eedi.py`, MA-IRT checkpoint, Track A AUC table for baselines 4, 5, 6. | 2 weeks, 1 GPU-week. |
| **E2, RL environment, reward implementation, BC warm-start** | Wrap MA-IRT step API in a Tianshou-compatible env, implement the entropy-reduction + VOI reward, train a behaviour-cloning policy on the max-info baseline. | `rl/src/irtrec/envs/eedi_env.py`, `rl/src/irtrec/rewards/entropy_voi.py`, BC policy checkpoint, sanity-check rollouts. | 2 weeks. |
| **E3, PPO training and MVE critic warm-start** | Train PPO with the MVE-warm-started critic, get the first beat-the-random-and-historical-replay baselines on Eedi. | Trained PPO checkpoint, Track B %-of-max-improvement table for baselines 1, 2, 3. | 2--3 weeks, 1 GPU-week. |
| **E4, Ablations A1--A8** | Run the eight ablations, build the ablation table. | Ablation table, plots, statistical CIs. | 3 weeks, 2 GPU-weeks. |
| **E5, XES3G5M head-to-head with ExRec** | Re-fit MA-IRT on XES3G5M (K=2), train the same PPO policy, head-to-head against ExRec PPO and ExRec DDPG+MVE on the canonical XES3G5M protocol. | XES3G5M MA-IRT checkpoint, XES3G5M Track B table against baselines 7, 8. | 2 weeks, 1 GPU-week. |
| **E6, Paper writeup and IJAIED submission prep** | Write the paper, target IJAIED. Build figures, finalise tables, write the related work section against the 13-paper bibliography in the audit. | `overleaf-sync/main.tex`, all figures and tables, submission package. | 4 weeks. |

Total, approximately 15 weeks of calendar time, approximately 5 GPU-weeks of compute. (Estimates are loose, GPU usage assumes a single A100 per experiment cell.)

---

## 11. Risks and mitigations

| # | Risk | Mitigation |
|---|---|---|
| **R1** | Eedi K=4 distractor ordering is not meaningful (the empirical-difficulty ordering is too noisy to define a genuine ordinal scale), so MA-IRT trains on what is effectively a noisy multi-class signal and the ordinal advantage disappears. | Validate the ordering empirically before E1 by checking the monotonicity of category-response curves (CCC monotone in theta is the GPCM signature). If monotonicity fails on more than 20% of items, fall back to a K=3 collapse (correct, close-distractor, far-distractor) or to K=2 binary. Report ablation A3 (random ordering) as the lower bound. |
| **R2** | The frozen MA-IRT belief drifts between train and test populations (e.g., Eedi item bank covers different KCs at train and test time), so the reward signal becomes miscalibrated and the policy steers the belief into miscalibrated regions. | Monitor MA-IRT calibration on train vs test fold with Brier score and reliability diagrams. If Brier-score drift exceeds 0.02, re-freeze MA-IRT on a calibration set drawn from the test distribution before RL training. |
| **R3** | Posterior collapse from over-informative ordinal items causes per-step shaping rewards to be spiky (a single response can drop sigma by 3 to 5x), destabilising PPO. | Clip per-step shaping rewards at the 95th percentile during BC warm-start. The log-precision formulation `0.5 * log(1 + sigma^2 * I_GPCM)` is already log-smoothed and should be robust, but verify in E2 rollouts. |
| **R4** | The reward function inadvertently rewards items the policy can already predict perfectly (high theta, theta well above all betas), inflating I_GPCM without genuine learning. | The potential-based shaping is invariant to this (no reward for trivially-predictable items, since sigma will not drop much). The terminal VOI anchor on held-out probe items is the real teacher signal, monitor it during training. |
| **R5** | ExRec's MVE critic warm-start does not transfer to ordinal GPCM because the K-way expectation has higher variance than the binary expectation, so the warm-started critic's initial Q-estimates are noisier than its random-init counterpart. | Ablation A5 directly addresses this. If MVE warm-start hurts, drop it and use plain PPO with randomly initialised critic. The architectural commitment in this plan is "discrete PPO", not "PPO with MVE", so the fallback is clean. |
| **R6** | Eedi Task 3+4 is too small for stable RL training (9,401 students, 10-step episodes give roughly 94k transitions per epoch, which is small by RL standards). | Use model-based rollouts (simulate `y_t` from frozen MA-IRT) rather than on-policy real responses, which decouples sample efficiency from dataset size. The episode horizon is short (T=10) so even 1,000 episodes of real evaluation are statistically meaningful with bootstrap CIs. |
| **R7** | The empirical-difficulty distractor ordering leaks information across the train/val/test split (computing the ordering on train and applying to test is leakage-free, but if a question appears only in test, its ordering is undefined). | Restrict evaluation to questions appearing in the train fold. Eedi Task 3+4 has 948 questions total and the train fold covers nearly all of them, leakage risk is low but verify before E1 finalises. |
| **R8** | Item-exposure bias, the policy converges to a small set of high-information items and over-exposes them. | The exposure penalty `w_expo` in the reward and the PPO entropy bonus 0.01 mitigate this. Report exposure histograms in Section 9 ablations. |

---

## 12. First concrete step

Build `rl/src/irtrec/envs/eedi_env.py` after verifying Eedi distractor-ordering monotonicity.

Concretely, the first commit on a new branch `feat/ordrec-e1-eedi` should,

1. Download the Eedi NeurIPS 2020 Task 3+4 split into `data/eedi_nips2020_t34/raw/` (via the CodaLab link, registration required).
2. Write `scripts/prepare_eedi.py` that
   - parses the raw csv,
   - computes the empirical-difficulty distractor ordering on the train fold,
   - emits `data/eedi_nips2020_t34/sequences.json`, `metadata.json`, and a `q_matrix.npz` (Q x 57 binary).
3. Write `scripts/validate_eedi_ordering.py` that fits MA-IRT for one epoch on the K=4-ordinal coerced data and checks that the per-item GPCM category-response curves are monotone in theta on at least 80% of items. If they are not, branch to the K=3 collapse plan in R1.
4. Open a draft PR with the loader and the validation result. This is the gate for E1.

After E1's MA-IRT checkpoint exists, the second concrete step is `rl/src/irtrec/envs/eedi_env.py` exposing a Tianshou-compatible env that wraps the frozen `model.step` API behind a Gym-style `reset/step` interface.

---

## 13. Reusable components from the archived rl/ tree

**Keep (with minor or no changes).**

| Component | Location | Reason kept |
|---|---|---|
| `BeliefTracker` (Laplace posterior over theta) | `rl/src/irtrec/belief/` | The Laplace approximation logic for sigma is dataset-agnostic and works for any GPCM-headed model, including MA-IRT. Some refactoring needed to support D > 1 (track `Sigma_t` covariance, not scalar `sigma_t^2`). |
| `RetrievalIndex` (FAISS over item embeddings) | `rl/src/irtrec/retrieval/index.py` | Direct reuse for v2 continuous-action variant. v1 does not need this but keeping it costs nothing. |
| `model.step` API (StepState, online belief update) | `ma-irt/` on feat/online-step-api | Already parity-tested, sub-15ms CPU step budget. This is the bedrock the RL env wraps. Merge feat/online-step-api into main before E2. |
| Tianshou wiring boilerplate (PPO trainer config, episode collector) | `rl/scripts/run_m23_baselines.py` style | The PPO loop scaffolding (rollout collection, GAE, mini-batch update) is reusable. Strip out the JobTower-specific reward and replace with the Section 4 reward. |
| Bootstrap-CI evaluation script template | `rl/scripts/eval_v2_baselines.py`, `eval_v2_trajectory.py` | The paired-bootstrap-over-students evaluation utility is reusable across both Track A and Track B. |

**Remove or replace.**

| Component | Location | Reason removed |
|---|---|---|
| O*NET pool, JobTower text branch | `rl/src/irtrec/retrieval/job_tower.py`, `pool.py` | Job-recommendation specific. Replaced by the discrete Eedi item bank and a KC-aware item embedding for v2. |
| v1 and v2 synthetic data generators | `rl/src/irtrec/datagen/`, `rl/scripts/generate_v2.py`, `rl/scripts/build_synthetic_dataset.py` | Job-rec synthetic simulator. Replaced by Eedi real-data loader in `dataloading/eedi.py`. |
| `rl/configs/sim_v*.yaml` | `rl/configs/` | Job-rec configs. Replaced by `rl/configs/ordrec_eedi.yaml`, `ordrec_xes3g5m.yaml`. |
| `controller/` (if it hosts job-rec orchestration logic) | `rl/src/irtrec/controller/` | Verify on opening, replace any job-rec specifics with item-rec equivalents. |

**Archive.** Move `rl/src/irtrec/datagen/`, `rl/scripts/build_onet_pool.py`, `rl/scripts/build_synthetic_dataset.py`, `rl/scripts/generate_v2.py`, and all `sim_v*.yaml` configs to `docs/archive/jobrec/rl_legacy/` for historical reference.

---

## 14. References

- Bock, R. D. (1972). Estimating item parameters and latent ability when responses are scored in two or more nominal categories. *Psychometrika*, 37(1), 29-51.
- Converse, G., Curi, M., Oliveira, S. (2020). Autoencoders for educational assessment. *Lecture Notes in Computer Science* 12164.
- Curi, M., Converse, G. A., Hajewski, J., Oliveira, S. (2019). Interpretable variational autoencoders for cognitive models. *IJCNN 2019*.
- Ghosh, A., Heffernan, N. T., Lan, A. S. (2020). Context-aware attentive knowledge tracing. arXiv 2007.12324, KDD 2020.
- Ghosh, A., Lan, A. S. (2021). Option tracing, beyond binary knowledge tracing. arXiv 2104.09043, AIED 2021.
- Khajah, M., Lindsey, R. V., Mozer, M. C. (2016). How deep is knowledge tracing? *EDM 2016*.
- Liu, Z., Liu, Q., Chen, J., Huang, S., Tang, J., Luo, W. (2022). pyKT, a python library to benchmark deep learning based knowledge tracing models. arXiv 2206.11460, NeurIPS 2022 D&B.
- Liu, Z., Liu, Q., Chen, J., Huang, S., Luo, W. (2023). simpleKT, a simple but tough-to-beat baseline for knowledge tracing. arXiv 2302.06881, ICLR 2023.
- Liu, Q., Tong, S., Liu, C., Zhao, H., Chen, E., Ma, H., Wang, S. (2023). XES3G5M, A knowledge tracing benchmark dataset with auxiliary information. OpenReview Mn9oHNdYCE, NeurIPS 2023 D&B.
- Lord, F. M. (1980). *Applications of item response theory to practical testing problems*. Lawrence Erlbaum.
- Masters, G. N. (1982). A Rasch model for partial credit scoring. *Psychometrika*, 47(2), 149-174.
- Muraki, E. (1992). A generalized partial credit model, application of an EM algorithm. *Applied Psychological Measurement*, 16(2), 159-176.
- Muraki, E. (1993). Information functions of the generalized partial credit model. *Applied Psychological Measurement*, 17(4), 351-363.
- Ng, A. Y., Harada, D., Russell, S. (1999). Policy invariance under reward transformations, theory and application to reward shaping. *ICML 1999*.
- Ozyurt, Y. et al. (2025). Personalized exercise recommendation with semantically-grounded knowledge tracing. arXiv 2507.11060, NeurIPS 2025.
- Pavlik, P. I., Cen, H., Koedinger, K. R. (2009). Performance Factors Analysis, a new alternative to knowledge tracing. *AIED 2009*.
- Samejima, F. (1969). Estimation of latent ability using a response pattern of graded scores. *Psychometrika Monograph* 17.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. (2017). Proximal policy optimization algorithms. arXiv 1707.06347.
- Su, Y. et al. (2021). Time-aware multi-behavior knowledge tracing with attention. *Information Sciences*.
- Sympson, J. B., Hetter, R. D. (1985). Controlling item-exposure rates in computerized adaptive testing. *Proceedings of the 27th Annual Conference of the Military Testing Association*.
- Tsutsumi, E., Kinoshita, R., Ueno, M. (2021). Deep item response theory as a novel test theory based on deep learning. *Electronics*, 10(9), 1020.
- van der Linden, W. J., Glas, C. A. W. (2010). *Elements of adaptive testing*. Springer.
- Wang, Z. et al. (2021). Results and insights from diagnostic questions, the NeurIPS 2020 education challenge. arXiv 2104.04034, AISTATS 2021.
- Yeung, C. K. (2019). Deep-IRT, make deep learning based knowledge tracing explainable using item response theory. *EDM 2019*.
