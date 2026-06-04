# DRL-MAIRT, Research Brief

## Abstract

DRL-MAIRT extends MA-IRT, a deep ordinal item-response model with strong
recovery on synthetic GPCM data, into a real-time interactive job
recommender. The central design problem is to elicit a latent learner
state inside a budgeted interactive session (the user's ability and
preference profile under a 1D GPCM IRT model) and to convert that state
into a high-quality recommendation slate over an O\*NET occupation pool.
We pose this as a sequential decision problem with a partially observed
posterior on theta as the agent's belief, and we claim that a policy
trained against potential-shaped information gain plus a grounded
terminal slate utility yields stronger top-K recommendations than greedy
maximum-Fisher selection while preserving the calibration of the
underlying IRT belief. This brief covers the project state through M0
to M3 plus M1 step API plus M4-RL v2 simulator, summarizes the v1
preliminary results, and outlines the M5-RL to M8-RL roadmap.

## Background and motivation

Adaptive testing has a five-decade tradition of Bayesian item selection
under IRT (Lord 1980, Owen 1975), but classical CAT does not extend
cleanly to downstream-decision settings where the recommender's value
depends on multi-step trade-offs between elicitation and action. A
prior in-house attempt, CaRReL, paired a DQN with a 2D MLE theta and a
cosine-similarity reward against fixed job embeddings. The post-mortem
identified the failure mode. The reward was a closed-form function of
the policy's own state representation, so the optimal action was
deterministic, leaving the agent nothing to learn. We retain CaRReL only
as a reference-negative example.

MA-IRT provides a deep belief tracker with calibrated theta recovery
(r=0.96 on synthetic K=4 paper headlines), per-step posterior variance
via observed Fisher information, and a frozen GPCM head whose item
parameters are recoverable. The recommendation question becomes how to
build a policy on top of this belief that does work classical CAT
cannot do alone.

## System architecture

The pipeline has six components on top of frozen ma-irt. **BeliefTracker**
wraps the online step API to maintain a per-session posterior
`{theta_t, sigma_t, h_t}`. **JobTower** (formerly ItemTower) embeds 923
O\*NET occupations into a 64-dim L2-normalized space via a frozen
BGE-small-en-v1.5 text branch plus a small structured-feature head.
**RetrievalIndex** serves cosine top-K from the precomputed
embedding bank. **FisherItemSelector** is the classical CAT baseline
for next-item selection. **ReflectionLayer** updates the per-session
query vector from in-session likes and dislikes with a 0.2 cosine-shift
cap, never touching ma-irt. **DecisionController** is the heuristic v1
controller that arbitrates ask vs recommend vs terminate, scheduled
for replacement by a PPO policy in M8-RL.

## What has been accomplished

M0 landed the rl/ subdirectory scaffold, the eight locked decisions in
`rl/docs/spec.md`, and the O\*NET 2024 occupation pool (923 occupations
with title, description, tasks, work_zone, education, RIASEC). M1
implemented the ma-irt online step API on `feat/online-step-api` with
parity to atol=1e-5 across DKVMN, LSTM, and Transformer encoders and
CPU step latencies of 7 to 38 ms at t=200, all under budget. M2 built
JobTower, RetrievalIndex, and the pool-registration helper, with a
50-occupation pool-swap smoke test confirming pool-agnostic operation.
M3 produced the v1 synthetic generator with the mixed K questionnaire
bank and engagement mixture. M4-RL on `feat/v2-simulator-delta-j`
replaces v1 with a continuous-delta_j composite, K=5 ordinal responses,
removes the engagement mixture, and bumps N to 100k.

Preliminary results on the v1 synthetic cohort (sim_v1_dev, N=500).

| Metric | Value |
|---|---|
| Theta recovery (recovery preset N=5000, EAP on true items) | Pearson r = 0.978, RMSE = 0.207 |
| Theta recovery (dev preset N=500) | Pearson r = 0.975, RMSE = 0.224 |
| Overall like rate | 0.202 (target 0.20) |
| Engaged users mean like rate | 0.337 |
| O\*NET embedding mean pairwise cosine (d=64) | 0.64 |
| RIASEC primary silhouette | 0.18 |
| Work-zone silhouette | 0.32 |

Recommender baselines, Hit@10 over 57 held-out evaluable users from a
100-user test split, with 500-bootstrap CIs.

| Baseline | Hit@10 |
|---|---|
| Random | 0.070 [0.018, 0.140] |
| Popularity (train likes) | 0.263 [0.140, 0.378] |
| 1D theta-true match (Bayes oracle) | 0.158 [0.070, 0.263] |
| 1D theta-hat match (realistic) | 0.158 [0.070, 0.263] |

The headline figure at `rl/results/v1/plots/headline_v1.png` shows the
recovery scatter, the baseline bar chart, the like distribution by
engagement class, and the O\*NET UMAP colored by RIASEC primary code.

## V1 finding, the 4-valued delta_j artifact

The v1 simulator computed `delta_j` as z-scored `work_zone` alone.
O\*NET work_zone takes 4 distinct integer values across the 923-job
pool (zones 2 through 5). The preference function
`P(like | u, j) = sigmoid(lambda * (theta_u - delta_j) + bias)`
therefore collapses to 4 distinct score values for any user. Both the
oracle and the realistic theta-hat baselines pick the same equivalence
class (lowest delta_j, highest score) and tie-break uniformly at
random. Popularity exploits its deterministic tie-breaking. The
popularity-versus-oracle gap of 0.105 absolute is a tie-breaking
artifact, not a measurement of policy quality. The CIs overlap. This
diagnosis directly motivates M4-RL.

## V2 design and the M4-RL milestone

The v2 simulator replaces work_zone with a continuous composite,

```
delta_j = 0.45 * z(work_zone)
        + 0.35 * z(education_zscore)
        + 0.20 * z(complexity_composite)
        + epsilon, epsilon ~ N(0, 0.30) at fixed seed
```

where `complexity_composite` is the mean z-score across the O\*NET
work_activities importance fields. The target is at least 900 unique
delta_j values across the 923-job pool, standard deviation in
[0.9, 1.1], and a 1D Bayes-oracle Hit@10 above 0.40 (versus v1's 0.158).
The engagement mixture is removed and replaced by a per-user
heterogeneity term `lambda_u ~ LogNormal(log 1.5, 0.4)`. Responses
become K=5 GPCM ordinal observations
`y ~ GPCM(lambda_u * theta - delta_j, beta)` with fixed step thresholds
`beta = (-1.5, -0.5, 0.5, 1.5)`. Backward-compatible binary
`IsLiked = 1[y >= 3]` is preserved. ItemTower is renamed JobTower
across `rl/`.

The policy is PPO (Schulman et al. 2017) with behavioral cloning
warm-start from a 50/30/20 mixture of max-Fisher, ReflectionLayer-greedy,
and Thompson-sampling rollouts. The reward function decomposes into
four pieces.

```
r_t = (phi(s_t) - phi(s_{t-1}))                      potential shaping
    - c_ask * 1[a_t = ask]                           ask cost
    - c_exposure * max(0, rate(q_t) - r_max) * 1[ask] exposure penalty
    + 1[t = T] * (w_sl * r_SlateLift + w_pp * r_PPLL) terminal anchor

phi(s_t) = -0.5 * log(2 * pi * e * sigma_t^2)         capped at sigma_floor
r_SlateLift = U_sim(TopK_10(theta_hat_T)) - U_sim(TopK_10(theta_hat_0))
              under the simulator's hidden p_sim_like
r_PPLL = mean log P_GPCM(y_j_sim | theta_hat_T) over a session-fresh
              held-out probe set Probe_u of 20 jobs
```

with weights `w_sl = 1.0, w_pp = 0.5, c_ask = 0.02, c_exposure = 0.5,
r_max = 0.20, sigma_floor = 0.15`, all reward components normalized by
RunningMeanStd on the first 1000 rollouts then frozen. The state is a
96-dim vector concatenating `(theta_hat_t, log sigma_t, joint summary,
used-mask sketch, exposure-tally sketch, t/T, n_asked, n_likes,
n_dislikes)`. The action space is discrete of size 925 (923 ask actions
masked by no-repeat, probe-leakage, exposure-cap, plus recommend, plus
terminate). Horizon T_max = 30 with policy-initiated early termination,
gamma = 0.99, GAE lambda = 0.95.

## Preliminary v2 results

The M4-RL prelim experiment (workflow `waawjc9wt`) validated the
continuous-`delta_j` simulator on a 2000-user dev cohort (1600 train,
400 test, 356 evaluable held-out users with at least one positive). The
diagnostic gate passed.

| Quantity | v1 (sim_v1_dev) | v2 (sim_v2_dev) | Change |
|---|---|---|---|
| Unique `delta_j` values | 4 (of 923 jobs) | **923 (of 923 jobs)** | continuous |
| `delta_j` standard deviation | 1.0 | 1.0 | preserved |
| `delta_j` range | discrete bands | [-2.16, 2.58] | non-degenerate |
| Theta recovery Pearson r | 0.975 | 0.974 | preserved |
| Theta recovery RMSE | 0.224 | 0.222 | preserved |
| Overall IsLiked rate | 0.20 | 0.39 | denser positive signal |
| Hit@10, random | 0.070 | 0.157 | higher density of positives in larger candidate sets |
| Hit@10, popularity | 0.263 | 0.236 | popularity drops as likes spread across heterogeneous `lambda_u` |
| **Hit@10, 1D oracle (theta-true)** | **0.158** | **0.261** | **1.65x lift, expected ordering restored** |
| Hit@10, theta-hat (realistic) | 0.158 | 0.261 | ties oracle, EAP recovery is precise |

The headline reversal is that the v1 anomaly (popularity beats 1D
oracle at Hit@10) is **gone in v2**. Oracle (0.261) now sits above
popularity (0.236), random (0.157), and the difference is preserved
under bootstrap (the oracle and theta-hat 95 percent CIs are
[0.219, 0.305] versus popularity's [0.194, 0.278], with limited
overlap). The continuous `delta_j` composite gives each user a 923-way
ranked pool instead of 4 tied equivalence classes, so 1D matching has
actual granularity to discriminate and the oracle-versus-popularity
ordering becomes diagnostic again.

EAP theta recovery on the K=5 GPCM responses against the true
`delta_j` and `lambda_u` gives r = 0.974 RMSE = 0.222, essentially
identical to v1's r = 0.975 despite the swap from a binary sigmoid to
a per-user-discrimination GPCM with five ordinal categories. The K=5
response distribution is well-balanced (category counts roughly
15.5k/14.6k/15.0k/14.2k/14.8k across the 74k administered items).

Plots at `rl/results/v2/plots/`.

- `m4rl_delta_j_distribution.png`, the continuous distribution
  versus the v1 4-bucket staircase.
- `m4rl_theta_recovery.png`, EAP recovery scatter on v2 GPCM
  responses.
- `m4rl_baselines.png`, Hit@10 with bootstrap CIs on the four
  baselines.
- `m4rl_v1_vs_v2_baselines.png`, side-by-side comparison.
- `m4rl_response_distribution.png`, K=5 category histogram.

Implication for M5-RL. The trained UserTower must clear
**Hit@10 = 0.261** (v2 1D oracle) on held-out users to claim
contribution. Any headroom above that ceiling must come from
multi-dimensional matching against the JobTower embedding plus
sequence-level encoder history plus per-user `lambda_u` heterogeneity,
none of which a scalar `delta_j` 1D matcher can use.

## Mathematical formulation

### Item response model

Every questionnaire item j is parameterized by a discrimination
`alpha_j > 0` and `K - 1 = 4` ordered step thresholds
`beta_j = (beta_{j,1}, ..., beta_{j,4})`. Given a learner ability
`theta`, the probability that the response `y in {0, 1, ..., 4}` lands
in category k under the generalized partial credit model (Muraki 1992)
is

```
                exp( sum_{m=1..k} alpha_j (theta - beta_{j,m}) )
P(y = k | theta, alpha_j, beta_j) = --------------------------------------
                sum_{l=0..4} exp( sum_{m=1..l} alpha_j (theta - beta_{j,m}) )
```

with the empty inner sum equal to zero by convention so that category 0
has probability 1 over the denominator.

### Posterior on theta and observed Fisher information

Under a Gaussian prior `theta ~ N(0, 1)` and the GPCM likelihood, the
posterior on theta given a response history `H_t = ((j_1, y_1), ...,
(j_t, y_t))` admits a Laplace approximation at the maximum a posteriori
estimate `theta_hat_t`. The observed Fisher information at theta_hat_t
is

```
I(theta_hat_t) = sum_{s=1..t} alpha_{j_s}^2 * Var(Y_{j_s} | theta_hat_t)
```

where `Var(Y | theta)` is the category-response variance for the GPCM.
The posterior precision absorbing the unit Gaussian prior is
`I(theta_hat_t) + 1`, and the Laplace posterior standard deviation
emitted by the ma-irt step API is

```
sigma_t = 1 / sqrt( I(theta_hat_t) + 1 + jitter ),  jitter = 1e-6.
```

This is the quantity ma-irt's online step API returns alongside
`theta_hat_t` and the encoder hidden `h_t`.

### Preference field and slate utility

In the v2 simulator, the latent like rate for user u on job j is the
GPCM-implied tail probability of an ordinal response of 3 or 4,

```
p_sim_like(theta_u, j) = sum_{k=3..4} P(y = k | lambda_u * theta_u - delta_j, beta)
                       = P_GPCM(y >= 3 | lambda_u theta_u, delta_j, beta).
```

The slate utility of a recommended set S of K = 10 jobs is

```
U_sim(S, theta_u) = (1 / |S|) * sum_{j in S} p_sim_like(theta_u, j).
```

The terminal slate-lift reward compares the policy's final slate to the
slate that a system with no information about the user would construct,

```
r_SlateLift = U_sim( argtopK( q_phi(theta_hat_T, j) | j in pool ), theta_true_u )
            - U_sim( argtopK( q_phi(theta_hat_0, j) | j in pool ), theta_true_u )
```

where `q_phi` is the policy's job-scoring head and `theta_true_u` is
the simulator's hidden user parameter (never exposed to the policy).

### MDP

The decision process is `(S, A, P, R, gamma)` with

```
s_t = ( theta_hat_t, log sigma_t, joint_summary_t in R^64,
        used_mask_sketch in R^16, exposure_tally_sketch in R^8,
        t / T_max, n_asked, n_likes, n_dislikes,
        log( sigma_t / sigma_0 ),
        1[sigma_t < sigma_floor],
        1[sigma_t < sigma_recommend_ceiling] ) in R^96
```

action space `A = {ask_1, ..., ask_923, recommend, terminate}` of size
925, masked at runtime by no-repeat, probe-leakage, and exposure-cap
indicator vectors. Transitions are deterministic given the simulator
and the sampled ordinal response `y_t`. Horizon `T_max = 30` with
policy-initiated early termination. Discount `gamma = 0.99`. The reward
is the four-component sum given above.

### PPO objective with shaping

PPO optimizes a clipped surrogate over rollout minibatches,

```
L^{CLIP}(theta_pi) = E_t[ min( r_t(theta_pi) A_hat_t,
                               clip(r_t(theta_pi), 1 - eps, 1 + eps) A_hat_t ) ]

r_t(theta_pi) = pi_{theta_pi}(a_t | s_t) / pi_{theta_pi_old}(a_t | s_t)

A_hat_t = sum_{l=0..L} (gamma * lambda)^l * delta_{t+l},
          delta_t = R_t + gamma V(s_{t+1}) - V(s_t)
```

with clip eps = 0.2, GAE lambda = 0.95, gamma = 0.99. The total loss
adds a value head term `L^{VF} = 0.5 * (V(s_t) - V_target_t)^2` clipped
at the same eps, plus an entropy bonus `c_ent * H(pi)` annealed linearly
from 0.01 to 0.0 over the first 50 percent of training, plus a KL
early-stop trigger at 0.02.

### Behavioral cloning warm-start

Before any PPO update, the actor is trained for five epochs of
behavioral cloning against an ensemble teacher
`pi_teach = 0.5 * pi_MaxFisher + 0.3 * pi_Reflection + 0.2 * pi_Thompson`
with the cross-entropy loss

```
L_BC = - E_{s ~ rho_pi_teach}[ sum_a pi_teach(a | s) * log pi_theta_pi(a | s) ]
```

over 50,000 teacher rollouts collected on the v2 simulator. The five
epochs typically lift the BC actor to roughly 85 percent action-match
agreement against the teacher on a held-out validation slice.

## Theoretical position

This is not classical CAT with an RL wrapper. Greedy maximum-Fisher
selection is Bayes-optimal only when the per-step ask cost is zero and
the terminal objective decomposes additively over items. Both
conditions are violated. A strictly positive `c_ask = 0.02` makes
asking marginal items strictly suboptimal once their per-step
information return falls below the cost, which requires the policy to
solve a stopping problem with no closed-form CAT solution.

The potential-shaping term is policy-invariant by the Ng, Harada,
Russell (1999) theorem. For any potential function `Phi : S -> R` and
any policy pi, the optimal Q-function under the shaped reward
`R' = R + gamma * Phi(s') - Phi(s)` satisfies
`Q^*'(s, a) = Q^*(s, a) - Phi(s)`, so `argmax_a Q^*'(s, a) =
argmax_a Q^*(s, a)`. We instantiate `Phi(s_t) = phi(s_t) =
-0.5 * log(2 * pi * e * sigma_t^2)` (the negative differential entropy
of the Laplace posterior) so the shaping signal is exactly the
single-step information gain about theta. This densifies gradients
without biasing the optimum.

The slate-lift term depends on the simulator's hidden true preference
function parametrized by `theta_true_u` and the continuous `delta_j`
field, which the policy never observes. The only way to maximize
`r_SlateLift` is to drive `theta_hat_T` toward `theta_true_u` through
informative item choices, breaking the closed-form circularity that
killed the CaRReL design. The predictive log-likelihood probe `r_PPLL`
uses fresh per-session jobs masked from the candidate pool, so the
policy cannot inflate it through ma-irt self-consistency.

The claim regime is bounded to synthetic data. Sim-to-real transfer
is flagged for future work.

## Roadmap

| Label | Scope | Status |
|---|---|---|
| M4-RL | v2 simulator, continuous delta_j, K=5, no engagement, JobTower rename | merged on main, prelim eval in flight |
| M5-RL | StudentEnv gym wrapper with reward harness and probe sampling | next |
| M6-RL | BC warm-start, heuristic ensemble teachers | sequential |
| M7-RL | PPO trainer, 100k-user run, ablations on each reward component | sequential, ~6 GPU hours |
| M8-RL | DecisionController integration, end-to-end Pareto eval on (Hit@10, session length, exposure entropy) | final |

## Risks and open questions

- **Reward magnitude drift.** The potential-shaping signal collapses as
  the posterior tightens (`sigma_t` approaches `sigma_floor`). The
  terminal anchor must dominate at termination. Per-component
  RunningMeanStd standardization is the mitigation, with a hard
  constraint that no component exceeds 70 percent or falls under 5
  percent of absolute reward in expectation.
- **Simulator-policy collusion.** The probe set must be held out from
  both the candidate pool and the ma-irt update. A runtime assertion in
  StudentEnv enforces this; a unit test in M5-RL asserts that random
  rollouts never select a probe job.
- **Greedy max-Fisher tying PPO on Hit@10 alone.** If `c_ask` ends up
  being the only thing making PPO beat MFI, the DRL contribution
  narrows to "cost-aware stopping". The eval reports Pareto
  performance over (Hit@10, session length, exposure entropy) jointly.
- **External validity.** All headroom claims are conditional on the v2
  simulator's preference field, not on real user behavior. Cross-
  simulator robustness (train inside DKVMN-based ma-irt, evaluate
  inside Transformer-based ma-irt) is the v2 integrity check.

## References

Bassen, J. et al. (2020). Reinforcement learning for the adaptive
scheduling of educational activities. *CHI*.

Lindley, D. V. (1956). On a measure of the information provided by
an experiment. *Annals of Mathematical Statistics*.

Lord, F. M. (1980). *Applications of Item Response Theory to Practical
Testing Problems*. Erlbaum.

Muraki, E. (1992). A generalized partial credit model. *Applied
Psychological Measurement* 16(2), 159 to 176.

Ng, A. Y., Harada, D., and Russell, S. (1999). Policy invariance under
reward transformations. *ICML*.

Owen, R. J. (1975). A Bayesian sequential procedure for quantal
response. *JASA*.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O.
(2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.

Sympson, J. B., and Hetter, R. D. (1985). Controlling item-exposure
rates in computerized adaptive testing. *Proceedings of the Military
Testing Association*.
