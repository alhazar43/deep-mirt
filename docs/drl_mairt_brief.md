# DRL-MAIRT, Research Brief

## Abstract

DRL-MAIRT extends the MA-IRT deep ordinal item-response model into a real-time interactive job recommender. The central design problem is to elicit a latent learner state (ability theta and discrimination-difficulty-keyed preferences) within a budgeted interactive session and to convert that estimated state into a high-quality recommendation slate. We pose this as a sequential decision problem with a partially observed posterior over theta as the agent's belief, and we claim that a policy trained against potential-shaped information gain plus a grounded terminal slate utility yields stronger top-K recommendations than greedy maximum-Fisher selection while preserving the calibration of the underlying IRT belief.

## What has been done

The v1 simulator and infrastructure are committed on `main` through M0 through M3, with the M1 online step API parked on `feat/online-step-api`. The synthesis in `docs/drl_mairt_plan_v1.md` consolidates the retrieval scaffold (`rl/src/irtrec/retrieval/`), the v1 synthetic generator (`rl/src/irtrec/datagen/`), and the O*NET pool attachment (`rl/artifacts/onet_v1.parquet`). Unit tests cover retrieval and generation. The v1 recovery study revealed a critical failure mode. The job-difficulty delta_j was computed as a discrete bin over work-zone alone, producing only four unique values across 923 items. Under this degenerate preference field, popularity ranking matched the Bayes-oracle 1D theta at Hit@10, leaving no measurable headroom for IRT-style elicitation. The artifact, not the IRT formulation, was the cause.

## What is being planned and why

Milestone M4-RL replaces the v1 simulator with a v2 generative process whose preference field is genuinely informative. Each job receives a continuous delta_j composed of work-zone, education z-score, and an O*NET complexity composite (the mean z-score across importance-weighted work-activity fields), plus Gaussian noise at fixed seed. The engagement mixture is removed and replaced by a per-user log-normal scale lambda_u, the population is grown to 100k with a stratified 80k/10k/10k split, and responses become K=5 GPCM ordinal observations with fixed thresholds at (-1.5, -0.5, 0.5, 1.5). Backward-compatible binary IsLiked is preserved as 1[y >= 3]. ItemTower is renamed JobTower across `rl/` to match the domain. The policy is PPO (Schulman et al., 2017) with behavioral cloning warm-start from a 50/30/20 mix of max-Fisher, ReflectionLayer-greedy, and Thompson rollouts. The per-step reward is potential-based shaping phi(s_t) - phi(s_{t-1}) with phi defined as the negative differential entropy of the Gaussian posterior on theta, plus an ask cost c_ask and an exposure penalty. The terminal reward combines a slate-lift term against the simulator's hidden true preference and a posterior-predictive log-likelihood on a held-out probe set. Potential-based shaping is policy-invariant by the Ng, Harada, Russell (1999) theorem, so the dense per-step signal does not bias the optimum. The information-gain potential follows Bayesian sequential design (Lindley 1956, Owen 1975) and the classical Fisher-information formulation of CAT (Lord 1980). Terminal grounding in slate-lift forces the policy to drive theta_hat toward theta_true rather than toward self-consistency under MA-IRT.

## Theoretical position

This is not classical CAT with an RL wrapper. Greedy maximum-Fisher selection is Bayes-optimal only when the per-step ask cost is zero and the terminal objective decomposes additively over items. A strict c_ask=0.02 and a terminal slate-lift that depends on the full posterior at horizon T break both conditions. The optimal policy must trade immediate information gain against the marginal value of the asked item to the final slate, a non-myopic decision with no closed-form greedy solution. Multi-step credit assignment therefore has real work to do. The claim regime is bounded to synthetic data. Sim-to-real transfer to real labor-market interaction is flagged for future work.

## Risks and open questions

- Reward magnitude drift between the per-step potential and the terminal slate-lift may cause one term to dominate. We will monitor returns decomposition during PPO updates and rescale via r_max clipping.
- Simulator-policy collusion. The probe set must be held out from both the candidate pool and the MA-IRT update, otherwise the policy can self-confirm through the probe.
- Greedy maximum-Fisher and ReflectionLayer-greedy baselines may tie PPO at low ask budgets. We need budget-stratified evaluation to expose the multi-step advantage.
- External validity. All headroom claims are conditional on the v2 simulator's preference field, not on real user behavior.

## References

Lindley, D. V. (1956). On a measure of the information provided by an experiment. *Annals of Mathematical Statistics*.

Lord, F. M. (1980). *Applications of Item Response Theory to Practical Testing Problems*. Erlbaum.

Ng, A. Y., Harada, D., and Russell, S. (1999). Policy invariance under reward transformations. *ICML*.

Owen, R. J. (1975). A Bayesian sequential procedure for quantal response. *JASA*.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.

Bassen, J. et al. (2020). Reinforcement learning for the adaptive scheduling of educational activities. *CHI*.
