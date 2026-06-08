# OrdRec v2, Deep Ordinal IRT as a World Model for Item Recommendation

Status, design refresh, supersedes v1 (commit a296496). Date locked, 2026-06-08.

## 1. Title and Abstract

**Deep Ordinal IRT as a World Model, Recommending Items with Calibrated Polytomous Knowledge Tracing**

The defining contribution of this work is not a recommender. It is a deep sequential knowledge tracing model whose hidden state is a measurement, a polytomous Generalized Partial Credit Model (GPCM) with calibrated (theta, alpha, beta) recovered to operational psychometric standards. Built on ma-irt, the model couples a DKVMN encoder with a separated ability pathway and a GPCM decoder, and on synthetic K=4 data recovers ground-truth person and item parameters at r_theta around 0.94, r_alpha around 0.80, r_beta above 0.90. The Fisher information per ordinal response is six to eight times that of binary KT at the modal point (Muraki 1993), turning the per-step posterior update into a dense, calibrated signal that black-box deep KT cannot expose. We then show that this measurement-grade world model enables a class of recommendation policies whose rewards are defined in trait space, posterior entropy reduction, value of information, threshold-targeted practice, that are infeasible without recoverable item parameters. The recommender is the demonstration. The deep ordinal IRT flavor is the contribution.

## 2. The Contribution Claim

Deep ordinal IRT, realised as ma-irt's DKVMN encoder with a separated ability pathway feeding a polytomous GPCM decoder, is the first knowledge-tracing model that simultaneously delivers operational-grade person parameter recovery, item parameter recovery sufficient for psychometric reuse, and a closed-form Fisher-information posterior update with a six-to-eight times per-response information density relative to binary KT at the modal point. This combination turns the KT hidden state from a black-box predictive sufficient statistic into a measurement in the Cronbach-Meehl, Embretson, and Mislevy sense, which is the precondition for recommendation policies whose rewards live in trait space.

## 3. Background

**ExRec as a template.** ExRec (Ozyurt et al. NeurIPS 2025) wires a frozen deep KT model into a PPO-style recommender via model-based value estimation (MVE). The critic is initialised by simulating both binary outcomes one step ahead through the KT encoder, predicting per-knowledge-component mastery on each branch, and forming Q(s,a) as the outcome-weighted mastery delta. The template is correct, freeze the world model, use it to bootstrap a Q function, then refine with policy gradient. The template is binary, the KT backbone is a semantically grounded SAKT variant, the reward is sum-of-correctness mastery. ExRec gives us the algorithmic skeleton and the empirical bar for learning-gain on real adaptive-learning tasks.

**Why ma-irt's IRT flavor changes the question.** Every published RL-over-deep-KT system the literature offers (CSEAL 2019, GEHRL 2023, DRAKT 2024, ExRec 2025, RL-DKT 2025) uses a binary, black-box KT backbone. The Deep-IRT line (Yeung 2019; Tsutsumi et al. 2021) is binary 1PL with no discrimination and no thresholds. The polytomous psychometric line (Muraki 1992; Wallmark et al. 2024) is classical static fitting, not deep. Vie and Kashima (2023) argued that DKT is implicitly dynamic MIRT, which sharpens our case rather than weakens it, an explicit, calibrated, polytomous deep MIRT is exactly the next step that closes the gap they identified. ma-irt provides that explicit version, and the closed-form Fisher information of GPCM at K=5 carries six to eight times the per-response signal of a 2PL response at the modal point, which is what makes the recommendation horizon actually move the posterior. The recommendation policy we build is the smallest sufficient demonstration that this measurement-grade state enables reward functions no black-box deep KT can support.

## 4. ma-irt Native Interface, the Integration Recipe under No-M1 Constraint

This section is load-bearing. v1 leaned on an M1 step API (per-encoder `forward_with_state`, per-decoder `compute_logits_from_state`, a `StepState` dataclass, a `freeze_irt` helper, parity tests at atol=1e-5). v2 ditches all of it. The native batched forward already returns per-step IRT parameters at every t, the only thing the M1 step API saved was redundant re-encoding cost on growing histories, which is recoverable by batching across users or by setting a re-encode cadence K_inflow > 1.

**Native forward signature.** From `ma-irt/models/base.py` L136 to L165,

    def forward(self, questions: Tensor, responses: Tensor) -> dict:

Inputs, `questions` of shape (B, S) long with item IDs in [1, Q] and 0 as padding, `responses` of shape (B, S) long with ordinal labels in [0, K-1]. Output dict, always the same five keys,

    return {
        "logits": dec_out.logits,   # (B, S, K), cumulative GPCM logits
        "probs":  dec_out.probs,    # (B, S, K), softmax categorical
        "theta":  theta,            # (B, S, D), per-step ability
        "alpha":  alpha,            # (B, S, D), per-step discrimination, > 0
        "beta":   beta,             # (B, S, K-1), per-step step thresholds
    }

For non-IRT decoders the base fills theta with zeros, alpha with ones, beta with zeros, so downstream consumers always see the same five keys.

**Stateless across forward calls.** The DKVMN encoder loops over S explicitly in Python (`ma-irt/models/encoders/dkvmn.py` L280), preallocates per-step outputs, and never retains hidden state between forward calls. LSTM omits the hidden argument and starts from zero h_0. Transformer is causally masked, so the per-position output at t depends only on positions <= t and is invariant to whether the sequence ends at t or extends past it. The native forward at length t and at length t+1 produces identical theta values at positions 0..t-1 under `model.eval()` and `torch.no_grad()`. This is the property that makes the M1 step API redundant.

**Batched inflow design.** Real-time arrives in batches at session boundaries, not one response at a time. Persist the user's history outside the model as two tensors. At each session boundary,

    q = torch.tensor([q_1..q_t], dtype=torch.long).unsqueeze(0)   # (1, t)
    r = torch.tensor([r_1..r_t], dtype=torch.long).unsqueeze(0)   # (1, t)
    with torch.no_grad():
        out = model(q, r)
    theta_t = out["theta"][:, -1, :]   # (1, D)
    alpha_t = out["alpha"][:, -1, :]
    beta_t  = out["beta"][:,  -1, :]
    probs_t = out["probs"][:, -1, :]

The cadence K_inflow is a deployment knob, not a model property. K_inflow = 5 means re-encode once every 5 responses. The episode horizon for OrdRec is T = 10, so two re-encodes per episode at K_B = 5 is the operating point.

**Action evaluation without advancing the model.** For an arbitrary candidate item q, evaluate P(Y = k | theta_t, alpha_q, beta_q) directly through `GPCMLogits` and the softmax head using a precomputed per-item table of (alpha, beta). The encoder does not have to be rolled forward for hypothetical (q, k) pairs.

**Per-item parameter table.** With the frozen model, for each item id q in [1..Q] run one encoder forward on a singleton sequence and read alpha and beta at that step. Cache as `item_alpha[Q, D]` and `item_beta[Q, K-1]`. For strict item-level alpha following the paper, average across many student contexts as `scripts/plot_recovery.py` does. The reward, the MVE expansion, and the entropy probe all read from this table, only theta requires a fresh encoder pass.

**Throughput.** For RL rollouts, stack many users' histories with right-padding and call `model(q_pad, r_pad)` once. Read `out["theta"]` at each row's true last index via `mask.sum(-1) - 1`. This is the path `utils/dataloader.py`'s `collate_sequences` already supports.

**Why batching is correct semantically, not just operationally.** The per-position output at t in a causal model is independent of any future-position content, so right-padding the suffix with zeros does not contaminate theta_t at the true tail. DKVMN writes only at the t-loop step, never reads from t' > t. LSTM is causal by construction. Transformer carries an upper-triangular mask that already zeros out future-position attention. The batched native forward and the single-user growing forward agree exactly at every position under `eval()` and `no_grad()`, so the v2 design does not need a parity test against a synthetic "online" code path. The native code path is its own reference.

**End-to-end episode rollout protocol.** For each student in the rollout batch, (a) replay the initial history H_init through `model(q, r)` and read theta at the last valid index, (b) for each batch boundary b in {1, 2, ..., T / K_B}, the policy samples K_B items conditioned on the current state s_b, (c) responses r are drawn from the GPCM predictive `P(Y | theta, alpha_q, beta_q)` for model-based rollouts or read from the real log for off-policy evaluation, (d) append the new K_B (q, r) pairs and re-call `model(q_hist_extended, r_hist_extended)` once, (e) update s_{b+1} from the new tail theta. The encoder is called exactly T / K_B times per episode, not T times. With T = 10 and K_B = 5 that is two encoder calls per episode, dominated by the decoder-only probe-entropy computations that run on top of the cached (alpha_q, beta_q) table.

**Reusable native components.**

| Component | Source | Role |
|---|---|---|
| `EncoderDecoderModel.forward` | `ma-irt/models/base.py` L136 | Single entry point, returns the five-key dict |
| `MAGPCM` | `ma-irt/models/magpcm.py` | Concrete wrapper, load checkpoint, `.eval()`, `.requires_grad_(False)` |
| `DKVMNEncoder`, `LSTMEncoder`, `TransformerEncoder` | `ma-irt/models/encoders/` | All three satisfy the same Encoder ABC, RL is encoder-agnostic |
| `GPCMDecoder` | `ma-irt/models/decoders/gpcm.py` | Holds IRT extractor plus GPCMLogits plus GPCMHead |
| `IRTParameterExtractor` | `ma-irt/models/components/irt.py` L32 | theta, alpha, beta heads, reusable for action scoring |
| `GPCMLogits`, `GPCMHead` | `ma-irt/models/components/irt.py` L115, L149 | Stateless, used directly in the reward function |
| `alpha_from_raw` | `ma-irt/models/components/irt.py` L190 | Single source of truth for positivity mapping |
| `SequenceDataset`, `collate_sequences` | `ma-irt/utils/dataloader.py` | Right-padded batched rollouts |

**What is gone.** The `StepState` dataclass, `forward_with_state` per encoder, `compute_logits_from_state` per decoder, the `freeze_irt` helper, the atol=1e-5 parity tests, and the `feat/online-step-api` branch infrastructure. Replaced by `model.eval(); model.requires_grad_(False)` and a single integration test that confirms theta[:, t, :] is invariant to T > t under no_grad.

**Latency budget.** No committed benchmark exists. First-principles estimate at T=50, B=1, default sizes,

| Encoder | CPU forward (B=1) | GPU forward (B=128) |
|---|---|---|
| DKVMN | 5 to 15 ms (Python loop overhead dominates) | 5 to 15 ms full batch |
| LSTM (cuDNN-fused) | 1 to 3 ms | 2 to 5 ms |
| Transformer | 2 to 6 ms | 1 to 3 ms |

If single-user DKVMN latency becomes the bottleneck, fall back to LSTM or Transformer, the IRT readout is encoder-agnostic. Before locking the v2 timeline, persist a one-off `bench_forward.py` measurement across (B in {1, 128}, T = 50) for the three encoders into `docs/`.

## 5. The MDP

**State.** s_t is derived only from the native forward output. Concretely,

    s_t = MLP_state( concat[
        theta_t,                                   # (D,)
        entropy_summary(probs over probe),         # (16,) reduced from probe
        exposure_mask_features,                    # (8,)
        batch_index_one_hot                        # (2,) for K_B=5, T=10
    ] )

theta_t is read at the last position of `out["theta"][:, -1, :]`. The entropy summary is computed from a single decoder pass on a fixed probe set C of M = 32 unseen items conditioned on theta_t, no encoder pass needed. No posterior sigma is tracked across calls. If a Laplace SEM is needed for diagnostics, recompute it at the boundary from the cached (q_<=t, r_<=t) and the per-item Fisher information lookup, do not carry it as state.

**Action.** a_t is a discrete choice over the Q-item bank, not a continuous embedding. ma-irt's (alpha, beta) are indexed per item id, so the predictive is only defined on the calibrated bank. On Eedi NeurIPS 2020 Task 3+4, Q = 948, well within Tianshou DiscretePPO scale. Already-administered items are masked to -inf and the mask is renewed at each batch boundary.

**Reward.** Defined in Section 6.

**Transition.** Stateless from the model's perspective. After action q_{t+1} is chosen, sample (or observe) r_{t+1}, append to the history, and the next state is built by re-calling `model(q_hist_extended, r_hist_extended)` and reading theta at the new last index. No external state machine, no per-step hidden plumbing.

**Horizon and discount.** T = 10 administrations per episode, batch boundary K_B = 5, so each episode produces two PPO transitions (at t = K_B and t = 2 K_B). Discount gamma = 0.95.

**Why the state is theta-derived, not encoder-hidden-derived.** A natural alternative is to feed the encoder's joint summary (the (read, item_embed) MLP output) into the policy. We reject this because the joint summary mixes the current item identity into the state, which couples action selection to state representation and violates the separation property that gives ma-irt its measurement claim. theta_t is the post-readout, item-blind ability estimate and is the only intermediate that supports trait-space reward functions. The probe-entropy summary is a function of theta_t and the precomputed item table, so the entire state is downstream of a measurement, not a hidden vector.

## 6. The Reward Function

**R1, batched posterior-entropy reduction over the GPCM decoder predictive, with a terminal NLL anchor.** Selected over three alternatives (categorical KL prior-to-current, decoder-logit entropy at the chosen item, per-trait mastery delta) for three reasons. R1 is invariant to the alpha-and-spread gaming channel (Muraki 1993) because categorical entropy is a function of the full distribution shape, not just alpha magnitude. R1 generalises trivially from D = 1 to D > 1 by reading theta from the relevant component. The probe set is fixed at episode start, so the policy cannot steer toward easier probes, preserving non-circularity against ma-irt itself.

**Setup.** At episode start, sample a fixed probe C = {q_1, ..., q_M} with M = 32 unseen items drawn uniformly without replacement from the bank, excluding the initial history. Also sample a held-out probe H_probe of 20 items with real responses, excluded from the action set.

**Potential.** Define the predictive-entropy potential,

    phi(theta) = - (1 / M) * sum_{q in C} sum_{k=0..K-1} P_qk(theta) * log P_qk(theta)

where P_qk(theta) = P(Y = k | theta, alpha_q, beta_q) is read directly from the GPCM decoder using the precomputed (alpha_q, beta_q) and one batched call to GPCMLogits plus softmax. No encoder pass.

**Per-batch shaping reward.** At each batch boundary t in {K_B, 2 K_B, ..., T},

    r_shape_b = w_info * (phi(theta_t) - phi(theta_{t - K_B}))
              - w_cost * K_B
              - w_expo * |overlap with last K_B actions|

This is potential-based in the Ng-Harada-Russell (1999) sense at batch boundaries, so the optimal policy is invariant to phi's specific form modulo the discount.

**Terminal NLL anchor.** At the final batch boundary,

    NLL_T = - (1 / |H_probe|) * sum_{(q_j, y_j) in H_probe}
                log P(Y = y_j | theta_T, alpha_{q_j}, beta_{q_j})
    r_term = w_voi * (NLL_prior - NLL_T)

with NLL_prior computed at theta_0. The terminal anchor is a value-of-information signal computed against real held-out responses, which prevents the shaping reward from being optimised in a way that does not improve the actual predictive on unseen items.

**Defaults.** w_info = 1.0, w_cost = 0.05, w_expo = 0.1, w_voi = 5.0.

**Why the multiplier matters.** With I_GPCM(K=5) approximately six to eight times I_2PL at the modal point, the per-batch phi differences are dense. To reach a posterior SEM of 0.3 (the operational adaptive testing target, van der Linden and Glas 2010), GPCM CAT needs roughly one-sixth to one-eighth the items of a binary CAT. The shaping reward inherits this density, making T = 10 actually move the posterior.

## 7. Algorithm and Hyperparameters

**PPO discrete-action (Schulman et al. 2017, arXiv 1707.06347).** Via Tianshou DiscretePPO. Justified over SAC-discrete, DQN/Rainbow, and continuous-action variants by three properties. On-policy PPO needs no off-policy correction, the batched MA-IRT forward provides clean per-batch transitions. PPO tolerates short rollouts (two transitions per episode) better than DQN-family methods that need replay-buffer warmup. Discrete categorical-over-Q is exactly the head Tianshou DiscretePPO expects. SAC-discrete is the ablation alternative.

**Hyperparameters.**

| Param | Value |
|---|---|
| Clip epsilon | 0.2 |
| GAE lambda | 0.95 |
| Discount gamma | 0.95 |
| Learning rate (actor and critic) | 3e-4, Adam, eps 1e-5 |
| Entropy bonus | 0.01 |
| Rollout length | 32 episodes per update, 64 transitions per update at K_B = 5 |
| Mini-batch size | 32 |
| Epochs per update | 4 |
| Value coefficient | 0.5 |
| Max grad norm | 0.5 |
| BC warm-start | 200 updates against max-Fisher-info-on-prior policy |

**Critic warm-start via static MVE.** At each batch boundary, the joint over K_B = 5 outcomes per item has K^{K_B} = 3125 terms (K = 5) which is tractable per state. Compute exactly,

    Q_MVE(s_b, a_b = (q_{t+1}..q_{t+K_B}))
      = sum_{k_vec in K^{K_B}} prod_i P(Y_{t+i} = k_i | theta_t, alpha_{q_{t+i}}, beta_{q_{t+i}})
                                * r_b(k_vec)
        - V_baseline

After 200 BC plus MVE warm-start updates, switch to standard PPO TD(lambda). This is the v2 reformulation of ExRec's MVE under batched inflow and ordinal expansion. The K-way ordinal expansion gives the critic K - 1 = 4 informative gradient directions per outcome dimension instead of one, which is the headline computational consequence of going polytomous.

**Action masking.** Already-administered items plus items chosen earlier in the current episode are masked to -inf in the policy head. Mask is renewed at each batch boundary.

**Action head, multi-pick within a batch.** v1 implementation, the actor selects K_B = 5 items per batch boundary by K_B independent categorical draws against the same theta_t with cumulative masking (no replacement within the batch). The state s_b is unchanged across the K_B picks, the theta value is the post-encoder tail and is held frozen during one batch. v2 alternative, a sequence-level action head emitting joint logits over K_B-tuples, reserved for the multi-trait extension where joint coverage across traits matters more than independent draws. For the single-trait headline we use the independent-draws head.

**Why not SAC-discrete or Rainbow.** SAC-discrete is on the shortlist as an ablation and would not change the headline measurement claim. Rainbow and other DQN-family methods need replay-buffer warmup that is awkward at two transitions per episode and 32 episodes per update, the per-update transition budget is 64 which is small for off-policy methods. PPO's on-policy regime maps cleanly onto the batched-inflow rollout protocol, the rollout buffer is filled, four optimisation epochs run, the buffer is discarded.

## 8. The Publishability Angle

**The novel claim.** First deep sequential KT with polytomous GPCM responses and validated parameter recovery for theta, alpha, beta_k on synthetic ground truth (r > 0.95 headline at K=4 already in hand), demonstrated as a measurement-grade world model that enables a class of RL recommendation rewards no black-box deep KT can produce. The Vie-Kashima (2023) result that DKT is implicitly dynamic MIRT becomes a stepping stone, explicit polytomous calibrated MIRT is the natural next step, and we are the first to take it with operational-grade recovery. The deep ordinal IRT in KT and the deep KT in RL are two separately mature literatures whose intersection is genuinely underexplored. The contribution is the bridge, not a breakthrough in either alone, and the bridge is defensible because the Fisher information multiplier and the recovery numbers are real.

**Venue and reviewer pushback.** Target IJAIED (already user-stated) with the measurement story as the headline and the recommender as the demonstration. Anticipate four pushbacks. First, "is this real or synthetic only," requires a real polytomous dataset, ASSISTments rubric items, Eedi distractor ordinal recoding, or PISA-style polytomous responses. Second, "how is this different from Deep-IRT (Yeung 2019)," counter with binary 1PL versus polytomous GPCM, no alpha versus alpha, no thresholds versus K-1 thresholds, no recovery validation versus headline recovery, fused encoder versus separated ability pathway. Third, "Vie and Kashima 2023 already showed DKT is dynamic MIRT," counter with implicit versus calibrated, they showed the equivalence in spirit, we deliver explicit and recoverable. Fourth, "does the recommender actually beat ExRec," requires head-to-head learning-gain comparisons against ExRec, CSEAL, GEHRL, DRAKT on the same task. Evidence we must gather, polytomous KT prediction metrics (AUC, accuracy, QWK) competitive with AKT, SAKT, SAINT, simpleKT on pyKT benchmarks at the binary-induced setting, plus recovery on a real polytomous dataset against a classical EM-fit GPCM baseline (already in ma-irt at `scripts/mirt_baseline_all_k.R`).

## 9. Milestones

| ID | Scope | Cost |
|---|---|---|
| **E1** | Eedi NeurIPS 2020 Task 3+4 loader plus ordinal recoding via empirical-difficulty distractor ordering on train only. MAGPCM pretraining to convergence with `separate_theta=True`, `n_traits=1`, embedding_type learned. Validate against Track A targets (AUC >= AKT 0.785, QWK >= 0.6 on K=4). Persist checkpoint at `outputs/ordrec_eedi/best.pt`. | 4 days |
| **E2** | Per-item parameter table (item_alpha, item_beta) extracted from frozen MAGPCM. Integration test that confirms theta invariance across growing-sequence forward calls. One-off `bench_forward.py` numbers committed. No M1 step API merge needed. | 1 day |
| **E3** | `rl/src/irtrec/envs/eedi_batched_env.py`, a Tianshou env whose `step` invokes the native batched forward on the growing history tensor at K_B = 5 boundaries. Episode horizon T = 10, two PPO transitions per episode. Probe sets C (size 32) and H_probe (size 20) sampled at reset. | 3 days |
| **E4** | PPO training loop with critic BC plus static-MVE warm-start (200 updates), then standard PPO TD(lambda) for 1000 updates. Ablations, A1 random-init critic, A2 binary-collapse reward (sum-of-correctness), A3 shared-pathway encoder (DKVMN+GPCM, `separate_theta=False`), A4 black-box softmax encoder (DKVMN+Softmax), A5 random policy. | 4 days |
| **E5** | Real polytomous dataset experiment, ASSISTments rubric items or Eedi distractor ordinal recoding. Recovery against classical EM-fit GPCM via R mirt. KT prediction metrics against AKT, SAKT, simpleKT on the same fold. | 4 days |
| **E6** | Paper draft, headline contribution as deep ordinal IRT measurement, recommender as demonstration. Figures, recovery scatter, posterior contraction over episode, learning-gain vs ExRec and binary-collapse baselines, ablation table. Target IJAIED submission window. | 5 days |

Total, approximately 21 working days. E1 through E3 are sequential, E4 and E5 can overlap once E3 is green.

## 10. What Changes vs the v1 Plan

| Aspect | v1 plan | v2 refresh |
|---|---|---|
| Model interface | M1 step API, `forward_with_state` per encoder, `compute_logits_from_state` per decoder | Native batched `forward(questions, responses)` only |
| State carried across calls | `StepState` (theta_t, sigma_t, h_t, item_log, t) | None, model is stateless, history is the only persistent object |
| Freezing the world model | `freeze_irt` helper with state-dict gymnastics | `model.eval(); model.requires_grad_(False)`, two lines |
| Posterior tracking | Per-step Laplace sigma_t recurrence in a `BeliefTracker` module | No sigma tracking, recompute at boundary from cached history if needed |
| Reward shaping signal | `phi(s_t) - phi(s_{t-1})` per step from Laplace posterior log-precision | `phi(theta_t) - phi(theta_{t-K_B})` per batch boundary from GPCM decoder predictive entropy over a fixed probe |
| Reward primary | sigma-based posterior contraction | GPCM categorical entropy reduction with terminal NLL anchor on real held-out responses |
| MVE expansion | Per-step K-way critic via `model.step` | Static MVE at batch boundary via K^{K_B} = 3125-term joint over K_B = 5 ordinal outcomes (tractable exact sum) |
| Transitions per episode | 10 (one per step) | 2 (one per batch boundary at K_B = 5, T = 10) |
| Integration tests | atol=1e-5 parity between step API and batched forward | One invariance test, theta[:, t, :] does not change with T > t under no_grad plus eval |
| Headline framing | ExRec-style RL with ordinal KT | Deep ordinal IRT as a measurement, RL as the demonstration |
| Algorithm | PPO with per-step state plumbing | PPO with batched updates and exact ordinal MVE warm-start |
| Branch prerequisite | Merge `feat/online-step-api` first | None, the branch is abandoned |
| Latency claim | "Per-step CPU under 15 ms across encoders" via M1 | First-principles batched throughput, validated by a one-off `bench_forward.py` before timeline lock |
| Recommendation-quality-over-time plot | Per-step | At batch boundaries |
| First concrete step | Wire model.step parity test | Eedi loader plus MAGPCM pretrain, then batched env |

## 11. Risks with Mitigations

| Risk | Mitigation |
|---|---|
| Synthetic recovery does not transfer to real polytomous data | E5 explicitly validates on a real ordinal dataset against classical EM-fit GPCM (R mirt), report per-item infit and outfit residuals as proxy validity |
| DKVMN single-user inference latency is too slow at K_inflow = 1 | Encoder-agnostic IRT readout means we can swap in LSTM (1 to 3 ms CPU) or Transformer (2 to 6 ms CPU) without re-architecting, fall through at E2 if `bench_forward.py` reveals a bottleneck |
| Per-step alpha varies across (student, step) for the same item | Average across student contexts when extracting `item_alpha`, follow `scripts/plot_recovery.py` recipe, document as a caveat in the paper |
| Multi-trait D > 1 identifiability via orthogonality penalty is soft, not hard | Keep v1 paper at D = 1 for the headline, add D > 1 as an ablation only after orthogonality recovery is verified on synthetic |
| K-1 GPCM thresholds non-monotone in step order on some items | GPCM definition allows it (Muraki 1992), document as a caveat, do not switch to PCM monotone constraint |
| Reward gaming, policy steers toward easier probe items | Probe C is fixed at episode start, sampled uniformly without replacement, and H_probe uses real held-out responses, so steering is structurally precluded |
| Reviewer "Deep-IRT already does this" objection | Section 8 enumerates four concrete differences (polytomous vs binary, alpha present vs absent, thresholds vs scalar b, separated pathway vs fused, validated recovery vs none), the ablations directly support the rebuttal |
| Reviewer "show me beating ExRec" demand | E4 ablation A2 (binary-collapse reward on same backbone) plus E6 comparison table against ExRec, CSEAL, GEHRL, DRAKT on the same task |
| MVE 3125-term exact joint becomes intractable if K_B grows | K_B = 5 is the operating point and 5^5 = 3125 is comfortably tractable per state, if K_B grows to 10, switch to a Monte Carlo MVE with 256 samples and document the bias |
| ExRec MVE expects binary, our ordinal expansion is non-standard | Section 6 plus Section 7 derive the ordinal MVE expansion explicitly and tie it to ExRec's formula in the binary limit (K=2 collapses to the original expression) |

## 12. First Concrete Step

E1 ships first. Concretely, in a single PR,

1. Add `ma-irt/utils/datasets/eedi_neurips2020.py` that loads Task 3+4, recodes responses to ordinal K = 4 via empirical-difficulty distractor ordering computed on the train fold only (no test leakage), and exposes the standard `SequenceDataset` interface.

2. Add `ma-irt/configs/ordrec_eedi_k4.yaml` matching the existing config schema with `model_type: magpcm`, `separate_theta: true`, `embedding_type: learned`, `n_traits: 1`, `n_categories: 4`.

3. Run `cd ma-irt && PYTHONPATH=. python scripts/train.py --config configs/ordrec_eedi_k4.yaml` to convergence, persist `outputs/ordrec_eedi/best.pt`.

4. Run `scripts/evaluate.py single` to confirm Track A targets (AUC >= 0.785, QWK >= 0.6).

No RL code in this PR. No M1 step API merge. After E1 is green, E2's two-line freeze plus integration test plus benchmark lands as a separate small PR, and E3 begins. The next code commit after E1 is the batched env, not anything in `feat/online-step-api`.

The `feat/online-step-api` branch is abandoned. Any prior latency claims tied to it (the "15 ms CPU per step" figure that motivated v1's framing) are replaced by the batched-throughput numbers `bench_forward.py` will produce in E2. No `StepState`, no per-encoder step methods, no per-decoder state-aware logits path, no atol=1e-5 parity tests carry forward.

## 13. References

Birnbaum, A. (1968). Some latent trait models and their use in inferring an examinee's ability. In Lord and Novick, *Statistical Theories of Mental Test Scores*, Addison-Wesley.

Bock, R.D. and Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters, *Psychometrika* 46, 443-459.

Cizek, G.J. and Bunch, M.B. (2007). *Standard Setting, A Guide to Establishing and Evaluating Performance Standards on Tests*, Sage.

Cronbach, L.J. and Meehl, P.E. (1955). Construct validity in psychological tests, *Psychological Bulletin* 52(4), 281-302.

Embretson, S.E. (1983). Construct validity, construct representation versus nomothetic span, *Psychological Bulletin* 93(1), 179-197.

Embretson, S.E. and Reise, S.P. (2000). *Item Response Theory for Psychologists*, Lawrence Erlbaum.

Ghosh, A., Heffernan, N. and Lan, A.S. (2020). Context-Aware Attentive Knowledge Tracing, KDD 2020.

Holland, P.W. and Wainer, H., eds. (1993). *Differential Item Functioning*, Lawrence Erlbaum.

Kolen, M.J. and Brennan, R.L. (2014). *Test Equating, Scaling, and Linking*, third edition, Springer.

Liu, Q., Huang, Z., et al. (2022). pyKT, A Python Library to Benchmark Deep Learning based Knowledge Tracing Models, NeurIPS 2022 Datasets and Benchmarks.

Lord, F.M. (1980). *Applications of Item Response Theory to Practical Testing Problems*, Lawrence Erlbaum.

Mislevy, R.J. (2018). *Sociocognitive Foundations of Educational Measurement*, Routledge.

Muraki, E. (1992). A generalized partial credit model, application of an EM algorithm, *Applied Psychological Measurement* 16(2), 159-176.

Muraki, E. (1993). Information functions of the generalized partial credit model, *Applied Psychological Measurement* 17(4), 351-363.

Ng, A.Y., Harada, D. and Russell, S. (1999). Policy invariance under reward transformations, theory and application to reward shaping, ICML 1999.

Ozyurt, Y., et al. (2025). ExRec, Personalized Exercise Recommendation with Semantically Grounded Knowledge Tracing, NeurIPS 2025.

Reckase, M.D. (2009). *Multidimensional Item Response Theory*, Springer.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A. and Klimov, O. (2017). Proximal Policy Optimization Algorithms, arXiv 1707.06347.

Tsutsumi, E., Kinoshita, R. and Ueno, M. (2021). Deep Item Response Theory as a Novel Test Theory, *Behaviormetrika*.

van der Linden, W.J. and Glas, C.A.W., eds. (2010). *Elements of Adaptive Testing*, Springer.

Vie, J.J. and Kashima, H. (2023). Deep Knowledge Tracing is an Implicit Dynamic Multidimensional IRT Model, arXiv 2309.12334, ICCE 2023.

Wallmark, J., Ramsay, J.O., Li, J. and Wiberg, M. (2024). Analyzing polytomous test data, a comparison between an information-based IRT model and GPCM, *Journal of Educational and Behavioral Statistics* 49(2).

Wilson, M., Karim, M.A. and Briggs, D.C. (2006). Using IRT to measure learning over time, in *Handbook of Statistics* 26, Elsevier.

Wright, B.D. and Stone, M.H. (1979). *Best Test Design*, MESA Press.

Yeung, C.K. (2019). Deep-IRT, Make Deep Learning Based Knowledge Tracing Explainable Using Item Response Theory, EDM 2019.
