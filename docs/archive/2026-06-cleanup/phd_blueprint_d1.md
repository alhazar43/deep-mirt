# Direction 1: KT+IRT + DRL + Combined (depth-first round)

A PhD program in three threads. Thread A extends MA-GPCM as a measurement model. Thread B extends the candidate's deep reinforcement learning work to educational and vocational recommendation. Thread C is the new contribution, using MA-GPCM as the observation model inside a measurement-grounded planner. Thread D adds two optional extensions that strengthen the combined story. The signature of the program is the encoder-decoder modular single-pass design pattern from the IJAIED paper, generalized across response formats, encoder families, and decision-making problems.

## Literature scan (organized by thread)

### Thread A scan (KT, IRT, amortization, ordinal heads)

Deep knowledge tracing started with Piech et al. (2015), an LSTM that predicts the next correct/incorrect outcome from a one-hot interaction sequence. Zhang et al. (2017) introduced DKVMN, which separates a static key memory (concept slots) from a dynamic value memory (per-concept mastery), and is the encoder the candidate's MA-GPCM builds on. Yeung (2019) added an IRT-style readout on top of DKVMN, calling the result Deep-IRT, and several follow-ups have extended this in 2024 (for example a Deep-IRT with temporal convolutions and an independent student-item Deep-IRT). Pandey and Karypis (2019) proposed SAKT, the first self-attention KT model. SAINT (Choi et al., 2020) and SAINT+ (Shin et al., 2020) added Transformer encoder-decoder structure on EdNet. AKT (Ghosh, Heffernan, Lan, 2020) introduced monotonic attention with exponential decay to model forgetting. Most recent KT work follows two threads, namely deeper or wider Transformers (RouterKT 2025, k-sparse attention 2024) and hybrid causal or Bayesian structure (Transformer-Bayesian KT 2025). The pykt-toolkit (Liu et al.) gives the standard benchmark suite. ASSISTments 2009, ASSISTments 2017, and EdNet remain the canonical datasets, with ES-KT-24 (Lee et al., 2024) and a few smaller corpora providing new multimodal signals.

Cognitive diagnosis has its own modern neural strand. Wang et al. (NeuralCD, AAAI 2020) introduced a neural CDM with a monotonicity constraint, which the candidate's MA-GPCM design echoes through ordered cumulative logits. The 2024 survey by Liu et al. (arXiv:2407.05458) catalogs the field. DisenGCD (NeurIPS 2024) uses a meta multigraph to disentangle student-skill-exercise interactions. A Q-matrix constraint network (QNN, Behavior Research Methods 2024) shows that injecting the Q-matrix as a structural prior improves both fit and small-sample recovery, which connects directly to MA-GPCM's question-side discrimination structure.

On the IRT side, the multidimensional generalization (MIRT) has long had compensatory and partially compensatory forms, with the Reckase compensatory model dominant in practice. Tsutsumi et al. (2023, Psychometrika) developed a dynamical non-compensatory MIRT with variational approximation, the closest existing competitor to a neural MIRT KT. A 2024 paper by Sharpnack et al. (BanditCAT and AutoIRT, arXiv:2410.21033) uses contextual-bandit Thompson sampling with Fisher-information reward to administer CAT and uses AutoML in the calibration step, which is the cleanest example of ML+IRT integration in production. Li, Gibbons, Rockova (Deep Computerized Adaptive Testing, Psychometrika 2026, arXiv:2502.19275) combine Bayesian sparse multivariate IRT with double deep Q-learning for item selection, replacing the myopic Fisher-information rule with a learned policy.

For amortized inference, Zammit-Mangion, Sainsbury-Dale, Huser (2024, arXiv:2404.12484) provide the canonical review, framing neural amortization as a feedforward replacement for MCMC, ABC, and EM that pays a one-time training cost for instant inference at deployment. Hollmann et al. (TabPFN, ICLR 2023; TabPFN v2, Nature 2025) demonstrated prior-fitted networks as a tabular foundation model. The PFN paradigm (Muller et al., 2022) generates synthetic datasets from a prior, trains a Transformer to predict held-out points conditioned on a context set, and at test time performs in-context Bayesian inference in one forward pass. This is exactly the amortization regime under which MA-GPCM's "single forward pass" claim should be evaluated.

For ordinal heads, Cao et al. (CORAL 2019, 2020) and Shi et al. (CORN 2022) develop rank-consistent neural ordinal classifiers using K-1 binary tasks. Cumulative link networks (Vargas et al., 2019) wrap a sigmoid CDF around a 1-d projection. ONTRAMs (Sick, Hothorn 2020) unite deep features with classical ordinal regression. The MA-GPCM head, K-1 cumulative logits with an alpha-weighted theta minus beta term, sits inside this family but is psychometrically interpretable rather than purely predictive.

### Thread B scan (DRL for education and vocational recommendation)

Adaptive learning systems with RL go back at least to Beck et al. (2000) and were systematized by Doroudi, Aleven, Brunskill (IJAIED 2019, scoping review). Singla et al. (2014), Reddy et al. (2017), and Bassen et al. (2020) used contextual bandits and RL for problem selection. Li, Xu, Zhang, Chang (2023, JEBS) framed deep RL for adaptive learning systems, including a learned transition model used inside deep Q-learning. Sharpnack et al. (BanditCAT 2024) is the production version of the bandit branch.

For recommendation more broadly, Zhao et al. (NeurIPS 2018, "Recommendations with Negative Feedback via Pairwise Deep Reinforcement Learning"), Xin et al. (SIGIR 2020, "Self-Supervised Reinforcement Learning"), and the KuaiShou / Tencent retention papers (Cai et al. 2023, arXiv:2302.01724, and Bai et al. 2024, arXiv:2404.03637) define the contemporary template, namely supervised pretraining followed by an off-policy correction. Liu et al. (long-term engagement, 2019, arXiv:1902.05570) is the canonical retention RL paper. Wang et al. (Auction-based recommender, arXiv:2305.13747) treat long-term value as an on-policy objective.

For off-policy evaluation, Dudik, Erhan, Langford, Li (Doubly Robust Policy Evaluation and Learning, 2011 and 2014) is foundational. Su et al. (Doubly Robust with Shrinkage, ICML 2020) and Saito and Joachims (combinatorial OPE, RecSys 2024) extend the family. Jiang and Li (Doubly Robust off-policy for Markov decision processes, ICML 2016) and Thomas and Brunskill (MAGIC, ICML 2016) give the sequential variant. The MAGIC estimator is particularly relevant because educational tasks have long horizons and high variance.

For offline RL, Kumar et al. (CQL, NeurIPS 2020), Kostrikov et al. (IQL, ICLR 2022), Fujimoto and Gu (TD3+BC, NeurIPS 2021), and Yu et al. (COMBO, NeurIPS 2021) are the standard methods. Decision Transformer (Chen et al., NeurIPS 2021) reframes offline RL as conditional sequence modeling. Janner et al. (Trajectory Transformer, NeurIPS 2021) is the model-based analog.

For course and concept recommendation in MOOC and ITS settings, Zhang et al. (Hierarchical RL for course recommendation, AAAI 2019) and Reinforced MOOCs Concept Recommendation (ACM TWEB 2023) are direct precedents. Cai et al. (Neural Computing and Applications 2023) survey RL-based MOOC recommenders.

For vocational and career pathway recommendation, Lan et al. (Skill-based Career Path Modeling, IEEE BigData 2020) build a learned occupation embedding from resumes. Yamashita et al. (WSDM 2022, "Looking Further into the Future") frame career pathway prediction as a sequence problem. Skill-graph work with ESCO and ONET (Gugnani et al. ICDMW; ESCO Labour Market Intelligence reports) is the data-side primary source.

For reward modeling under uncertainty, Coste et al. (Diverse LoRA RM ensembles, arXiv:2401.00243), Zhai et al. (Uncertainty-aware RLHF, arXiv:2410.23726), and PURM (probabilistic uncertain reward model, arXiv:2503.22480) all show that overconfident reward models cause reward hacking, which translates directly to a known failure mode in educational RL where greedy difficulty matching can hack a "learning gain" proxy.

### Thread C scan (model-based RL, world models, OPE with learned models)

Ha and Schmidhuber (World Models, 2018) and Hafner et al. (PlaNet 2019, Dreamer 2020, DreamerV2 2021, DreamerV3 2023) define the world-model paradigm, with latent imagination as the planning substrate. MedDreamer (Wang et al. 2025, arXiv:2505.19785) is the most direct medical analog, planning in latent EHR space for clinical decision support. The simulator gap (sim-to-real) literature, including domain randomization (Tobin et al. 2017) and PolySim (2025, arXiv:2510.01708), gives the language for "the planner is acting in a learned dynamics model and the dynamics model is wrong somewhere".

For OPE with learned models, Voloshin et al. (Empirical Study of OPE, NeurIPS 2021) and Fu et al. (D4RL, arXiv:2004.07219) build the benchmark culture. Hanna et al. (Importance Sampling Policy Evaluation with an Estimated Behavior Policy, ICML 2019) is the closest result for our setting. Thomas, Theocharous, Ghavamzadeh (high-confidence off-policy evaluation, AAAI 2015) give the bound formalism we will need.

For causal latent state, Schoelkopf et al. (Toward Causal Representation Learning, 2021) is the field-defining survey. Khemakhem et al. (iVAE, AISTATS 2020) and Hyvarinen et al. (nonlinear ICA family) supply the identifiability scaffolding. von Kuegelgen et al. (NeurIPS 2024, "Disentangled Representation Learning in Non-Markovian Causal Systems") is the most current result for treatment effects in temporal latent models, which matches the educational recommender pattern.

### Thread D scan (open-world items, rubric-ordinal validation, open-ended scoring)

For open-world item arrival, the contextual-bandit cold-start literature (Li, Karatzoglou, Gentile 2010 for LinUCB cold-start; Bouneffouf et al. 2014; Sanz-Cruzado et al. 2023) is the direct template. Hu et al. (Cluster-based Bandits for new users, SIGIR 2021) shows that grouping items or users via learned representations recovers most of the cold-start gap.

For rubric-ordinal validation, the NAEP Math Scoring Challenge (NAEP/ETS 2023, Springer IJAIED 2024) and PISA constructed-response scoring are the natural targets. AES (automated essay scoring) literature, especially the rubric-aware models of Mathias and Bhattacharyya (2018) and the cross-prompt models of Ridley et al. (2021), already use 4-level or 6-level rubrics that fit MA-GPCM's ordinal head exactly. The 2024 LLM grand-prize work on NAEP shows that LLMs can hit human agreement on nine of ten items, but they do not produce structured item parameters.

## Thread A. MA-GPCM extensions (online KT+IRT)

The candidate's signature contribution is an encoder-decoder modular single-pass model that returns interpretable IRT parameters in real time. Thread A asks how far each axis of that architecture can be pushed without breaking the streaming amortized property. Five sub-papers, ordered by depth.

### Thrust A.1. PFN-style amortization for online IRT, with calibration guarantees

**Research question.** Can the MA-GPCM forward pass be reframed as a prior-fitted network, so that the model is pretrained once on a wide synthetic prior over IRT data-generating processes and then performs Bayesian inference for an unseen student or unseen test in a single forward pass at deployment time, with calibration guarantees comparable to MML or MCMC.

**Method.** Pretrain a Transformer-MA-GPCM hybrid on the PFN protocol of Muller et al. (2022). Each pretraining episode samples a synthetic environment from a structured prior over (Q, K, item difficulty distributions, slope distributions, drift kernels, missingness patterns). The encoder consumes the context set of past responses and the query item, and the IRT head emits theta with a predictive distribution rather than a point estimate. At test time, no retraining occurs. We measure (i) calibration of the theta posterior against the true posterior on synthetic data, (ii) point-estimate recovery against R-mirt EM on real ASSISTments and EdNet, (iii) latency per step.

**Baselines.** TabPFN v2 (Hollmann et al. 2025, Nature) applied naively to the tabular response matrix, R mirt EM, JAGS-MCMC, AutoIRT (Sharpnack et al. 2024) for the calibration estimator, DKT, AKT, SAINT, MA-GPCM itself in non-amortized form, and an isotonic regression calibration wrapper on top of MA-GPCM.

**Datasets.** Three synthetic suites already in the candidate's repo (static GPCM, staircase, random walk, block change), ASSISTments 2009, ASSISTments 2017, EdNet KT1, and a NAEP math constructed-response subset for rubric ordinal.

**Metrics.** RMSE on theta, alpha, beta against ground truth on synthetic. Coverage and width of credible intervals for the theta posterior. Expected calibration error of the predictive ordinal distribution. AUC and QWK for next-response prediction. Wall-clock latency per response on CPU and GPU.

**Hard problem.** Existing PFNs work on i.i.d. tabular data. MA-GPCM has explicit temporal structure and a structured psychometric likelihood. The open question is whether a PFN pretrained on a sufficiently rich prior over GPCM data-generating processes recovers the true posterior at test time without per-student fine-tuning, and what the prior needs to look like for the calibration to transfer to ASSISTments-scale data. A second open question is whether the amortization gap, the loss in posterior accuracy relative to MCMC, is bounded under reasonable smoothness conditions on the prior and the encoder.

**Formal claim.** Under standard regularity on the synthetic prior and a Lipschitz condition on the GPCM log-likelihood with respect to (theta, alpha, beta), the amortized posterior produced by the network converges in total variation to the true posterior as the prior coverage and the network capacity grow. The provable statement would be a finite-sample bound on TV between the amortized predictive and the true predictive in terms of (i) the KL between the synthetic prior and the deployment data DGP and (ii) the worst-case approximation error of the encoder family on the simulated tasks. This is the educational analog of the consistency result for PFNs in causal inference (frequentist consistency of PFNs, arXiv:2603.12037).

### Thrust A.2. Open-world item arrival with cold-start IRT

**Research question.** When a new item appears mid-stream with no calibration data, MA-GPCM cannot produce a meaningful alpha and beta. How can side information (item text, taxonomy tags, image features) be combined with the streaming encoder so that the IRT head emits calibrated parameters for the first occurrence of a new item, and the parameters tighten as responses accrue.

**Method.** Replace the LearnedEmbedding with a hybrid head that consumes item content features (BERT or a small encoder over the item text or HTML) and produces a prior over (alpha, beta). The DKVMN write step is unchanged. At each step the IRT head fuses the prior with the running posterior accumulated from observed responses, using either a Bayes-by-backprop layer or a hypernetwork. We benchmark against AutoIRT (Sharpnack et al. 2024) as the direct cold-start IRT method. The streaming property is preserved.

**Baselines.** AutoIRT, LinUCB cold-start, Cluster-based Bandits (Hu et al. 2021), a content-only baseline (BERT to alpha-beta directly), and MA-GPCM with random initialization for new items.

**Datasets.** Duolingo SLAM 2018, EdNet KT4 (contains item content), ASSISTments 2017, and a constructed split of NAEP where some items are held out from calibration.

**Metrics.** RMSE on alpha and beta for held-out new items as a function of the number of observed responses. Calibration of the predictive ordinal distribution on first-occurrence items. QWK on the cold-start subset.

**Hard problem.** Without responses, alpha and beta are unidentifiable from item content alone, so any cold-start estimate is a prior. The open question is how to design the prior so that it is shrunk toward content-similar items without inducing the rotation indeterminacy familiar from MIRT. The second open question is the cold-start to warm-start handoff. The model must continuously interpolate as the response count grows from zero to many without a discontinuity in the parameter estimates.

**Formal claim.** Under a smoothness assumption on the mapping from content embedding to true IRT parameters, the predictive ordinal probability of a cold-start item is close (in total variation) to that of its k nearest content neighbors with calibrated alpha and beta. The bound degrades linearly in the embedding Lipschitz constant and shrinks as observed responses accumulate, recovering the standard MA-GPCM rate after O(log Q) responses on the item.

### Thrust A.3. Multidimensional concept-aligned memory for per-skill feedback

**Research question.** MA-GPCM emits a scalar theta. Real instruction needs per-skill mastery to drive feedback. How can MA-GPCM be lifted to a D-dimensional theta vector whose dimensions are aligned to interpretable skills (Q-matrix rows), while keeping identifiability and the single-pass property.

**Method.** Generalize the IRT head to MIRT with a compensatory GPCM formulation, theta in R^D, alpha in R^D per item, beta in R^{K-1} per item, and Q-matrix mask on alpha. The DKVMN value memory rows are aligned to skill slots via a soft assignment learned jointly with a Q-matrix recovery loss. The orthogonality penalty on Corr(theta) - I from the candidate's existing kt-mirt prototype is the identifiability constraint. We add a partial compensation parameter mu per item (Bolt and Lall 2003 form) so the head smoothly interpolates compensatory and non-compensatory MIRT.

**Baselines.** R mirt for compensatory MIRT, smirt in R sirt for non-compensatory, NeuralCD and DisenGCD on the diagnostic side, AKT and SAINT on the prediction side. For per-skill feedback, MultiDim-DKT extensions in pykt-toolkit.

**Datasets.** Synthetic MIRT data (the candidate's kt-mirt prototype already supports D >= 1), ASSISTments 2009 with its skill tagging, EdNet KT4, and the FrAcT-Sub corpus where Q-matrix is known.

**Metrics.** RMSE on theta vector, alpha vector, beta. Per-skill QWK on held-out responses. Q-matrix recovery F1. The orthogonality of recovered theta in a held-out window.

**Hard problem.** D-dimensional theta has a rotation indeterminacy that scales like D^2 free parameters. Existing fixes (orthogonality penalty, anchor items, sparse Q-matrix) trade off interpretability against fit. The open question is whether a streaming encoder with a learned soft Q-matrix can recover the rotation up to a permutation of skills, and what the necessary identifiability conditions on the data and on the encoder look like in finite sequences.

**Formal claim.** Under a sparse-Q-matrix anchoring condition (every skill has at least three items loading exclusively on it), the MA-MIRT-GPCM model is identifiable up to permutation and sign of skills, and the orthogonality penalty plus Q-matrix mask is sufficient for the gradient flow to reach a globally identifiable point. This is the educational analog of the anchor result in identifiable factor analysis (sparse decoding for identifiable deep generative models, arXiv:2110.10804).

### Thrust A.4. Rubric-ordinal validation on NAEP and PISA constructed response

**Research question.** All evidence for MA-GPCM has been on proxy-ordinal ASSISTments (binary correctness collapsed into K=2 or K=3 by hint usage). Does the model recover meaningful item parameters when the responses are true rubric scores, and how does this compare to existing automated essay scoring and LLM-as-grader systems on the same constructed-response items.

**Method.** Take the NAEP Math Scoring Challenge corpus (10 items, public release after 2024) and PISA constructed-response items. Build a two-stage pipeline. Stage one is an LLM grader (or human rubric scores when available) producing K-level ordinal labels per response. Stage two is MA-GPCM consuming these labels in the candidate's existing GPCM form. The point is to test whether MA-GPCM gives parameter estimates consistent with the official NAEP IRT scoring (which uses MML on a calibrated form) while preserving the streaming property for new test takers.

**Baselines.** Official NAEP IRT scaling (R mirt EM), AES rubric models (Mathias and Bhattacharyya 2018; Ridley et al. 2021), LLM grader-only baseline (zero IRT structure), and Deep-IRT (Yeung 2019).

**Datasets.** NAEP Math Scoring Challenge data, PISA 2015 and 2018 released constructed-response items, and ASAP-AES as the AES baseline domain.

**Metrics.** QWK between predicted and rubric scores. Concurrent validity correlation between MA-GPCM theta and official NAEP scale scores. Item-level alpha and beta agreement with NAEP technical reports.

**Hard problem.** Constructed-response items are often noisy and have a small number of administrations per item, which violates the implicit large-sample assumption of streaming KT. The open question is whether the MA-GPCM prior structure (the K-1 ordered thresholds) acts as enough of a regularizer to recover stable parameters under thin item data, and whether the alpha estimates correlate with item-discrimination indices from the official NAEP MML pipeline.

**Formal claim.** Under a fixed grader noise model on the rubric labels, the bias in MA-GPCM alpha and beta estimates is bounded by a function of the grader error rate and the GPCM Fisher information at the true theta. The claim is a measurement-error correction result familiar from psychometrics, restated in the neural amortization setting.

### Thrust A.5. Foundation-pretrained encoder with MoE memory

**Research question.** DKVMN is a fixed-capacity attention over a fixed number of skill slots. Modern Transformers scale to long sequences with mixture-of-experts memory. Does swapping the DKVMN encoder for a Transformer with MoE memory rows preserve the IRT decoder's interpretability while improving prediction, and how does the gating signal correlate with skill assignment.

**Method.** Replace DKVMN with a small encoder-decoder Transformer where the value memory is realized as the keys of an MoE router (sparse top-k routing per token, e.g. RouterKT 2025 design). The IRT head remains a GPCM with separated ability pathway. We measure (i) prediction performance, (ii) router-skill alignment (do expert assignments cluster by ground-truth Q-matrix skills), (iii) whether the alpha-beta estimates remain identifiable under the larger encoder.

**Baselines.** RouterKT, SAINT, SAINT+, AKT, the candidate's existing MA-GPCM with DKVMN, and DKT2.

**Datasets.** EdNet KT1 (long sequences for scale), ASSISTments 2009, FrAcT-Sub for Q-matrix recovery.

**Metrics.** AUC, QWK, RMSE on parameters, expert assignment NMI against true skill labels.

**Hard problem.** MoE routing is notoriously unstable. The open question is whether the IRT head's gradient signal (from the GPCM likelihood) is informative enough to encourage skill-aligned routing, and whether the gating creates an identifiability problem of its own (expert label switching).

**Formal claim.** The encoder swap preserves the identifiability of the IRT head as long as the routing distribution is independent of the ordering of theta dimensions. This is a permutation-equivariance statement that can be checked empirically and likely proved under standard MoE assumptions.

## Thread B. DRL educational and vocational recommender

The candidate already has a working DRL recommender for education and vocational settings. Thread B deepens it along three axes, namely reward design, off-policy evaluation, and constraint structure. Four sub-papers.

### Thrust B.1. Calibrated reward models for learning gain under measurement error

**Research question.** Most RL-for-education systems use a hand-designed reward, usually a proxy for learning gain or engagement. When the reward is itself a learned model (estimated learning gain from observed responses), the policy can hack it. How can a reward model be built from streaming IRT-style theta estimates so that the optimizer cannot exploit measurement error.

**Method.** Build the reward as a function of MA-GPCM's theta posterior over time, with a deliberate pessimism on the theta variance, following the pessimistic reward model literature (PURM 2025; Zhai et al. 2024). The reward is r_t = E[theta_t] - lambda * sqrt(Var[theta_t]) - cost(action). Train the policy with offline RL (CQL or IQL) on logged tutoring data. Compare to engagement-only and click-only rewards.

**Baselines.** Engagement reward RL (KuaiShou retention paper template), CQL with raw outcome rewards, behavior cloning, AKT-based reward (point estimate).

**Datasets.** EdNet KT1 (treat the platform's item sequencing as the behavior policy), the candidate's existing vocational recommender logs, and a synthetic ITS environment built from the staircase and random-walk DGPs in the kt-gpcm repo.

**Metrics.** Estimated policy value under MAGIC OPE (Thomas, Brunskill 2016), regret against an oracle policy on synthetic, and on real data, on-policy A-B style estimates if available.

**Hard problem.** Reward hacking with learned reward models is well documented in RLHF (Coste et al. 2024; Zhai et al. 2024). The open question is whether the calibrated theta posterior from MA-GPCM gives a tight enough variance estimate to bound the hacking risk, and what the optimal lambda is in terms of measurement error variance and policy class capacity.

**Formal claim.** If MA-GPCM's theta posterior is calibrated (in the sense of Thrust A.1), the pessimistic reward r_t lower-bounds the true expected learning gain in finite samples, and the policy that maximizes this lower bound improves over the behavior policy with probability at least 1 - delta. This is a Bellman-consistent pessimism result (Xie et al. 2021) restated with a learned, calibrated observation model.

### Thrust B.2. Doubly robust OPE for educational tutoring

**Research question.** Educational platforms cannot easily A-B test alternative tutoring policies because the cost of a bad sequence is high. Doubly robust and MAGIC estimators give variance-reduced OPE, but they need a value-function model and a behavior policy estimate. Can MA-GPCM's predictive distribution provide both, at no extra computational cost, and what is the variance reduction relative to plain IPS.

**Method.** Use MA-GPCM as the value-model component of a doubly robust estimator. The behavior policy is estimated from the logged item sequence using a small autoregressive model. The DR estimator is the Dudik 2011 form, the MAGIC variant for the sequential case. We benchmark on synthetic environments where the true policy value is known by Monte Carlo rollout.

**Baselines.** Plain importance sampling, weighted importance sampling, DR with random forest value model, MAGIC with neural value model.

**Datasets.** Synthetic ITS environments built from kt-gpcm DGPs, where the ground truth value is known, then EdNet and ASSISTments for the real benchmark.

**Metrics.** RMSE of the OPE estimate against the Monte Carlo ground truth, variance of the estimator across seeds, coverage of high-confidence bounds (Thomas, Theocharous, Ghavamzadeh 2015).

**Hard problem.** The doubly robust property requires either the value model or the behavior model to be correctly specified. In education, both are misspecified. The open question is whether MA-GPCM's structured likelihood (GPCM with monotone IRT structure) is a sufficient inductive bias to make the value model close enough for DR to dominate IPS in finite samples.

**Formal claim.** Under mild regularity on the GPCM likelihood and a bounded misspecification of the value model, the MA-GPCM-based DR estimator has variance bounded by a factor 1 + eps below standard IPS, where eps depends on the GPCM Fisher information and the support overlap between logging and target policies. This is the standard DR variance bound (Dudik, Erhan, Langford 2014) restated for a GPCM value model.

### Thrust B.3. Constrained policies for ability-difficulty matching and exposure caps

**Research question.** CAT theory has decades of results on optimal item selection under Fisher information at the current ability estimate, with exposure caps to prevent item overuse. Modern deep RL recommenders ignore both. How can a deep policy be trained subject to (i) ability-difficulty Fisher matching and (ii) item exposure caps, in a way that does not sacrifice long-horizon value.

**Method.** Define a constraint set on the policy at each step. The Fisher-information constraint says argmax of E[Fisher(theta_t, item)] over the action set should be within some neighborhood of the chosen action, with a temperature that the policy learns. The exposure constraint says each item's empirical selection rate stays below a cap. We train the policy with constrained policy optimization (Achiam et al., CPO 2017) using MA-GPCM's running Fisher information as the constraint function.

**Baselines.** Pure Fisher-information CAT (van der Linden 1998), BanditCAT (Sharpnack et al. 2024), unconstrained DQN, CPO without educational constraints, fairness-of-exposure bandits (Wang et al. 2021).

**Datasets.** Live CAT simulators built from the candidate's synthetic suite, BanditCAT's published settings (Duolingo English Test calibration), and an MOOC course-recommendation log.

**Metrics.** Measurement precision (posterior variance of theta as a function of test length), exposure rate distribution (Gini, entropy), policy value, regret against a Fisher-only baseline.

**Hard problem.** Constrained policy optimization in deep RL is brittle. The open question is whether the structured constraints from psychometrics (which have known closed-form behavior in the linear-Gaussian case) admit a stable deep RL relaxation. A second question is whether the Fisher constraint and the exposure cap are jointly feasible at all sample sizes, and what the Pareto frontier looks like.

**Formal claim.** The constrained policy achieves measurement precision within a factor (1 + gamma) of pure Fisher-information CAT, where gamma scales with the exposure-cap tightness, while strictly dominating BanditCAT in long-horizon value under any logged reward model. The proof template is constrained policy improvement under a Lyapunov-style argument.

### Thrust B.4. Vocational and skill-pathway recommendation with IRT-graded skill mastery

**Research question.** Course recommendations in vocational settings should optimize long-horizon career outcomes (placement, wage growth) under a skill-graph constraint (prerequisite ordering). Can MA-GPCM-style ordinal mastery estimates over an ESCO or ONET skill graph be used as the state of a DRL recommender, with rewards tied to labor market signals.

**Method.** Build a state representation that is a D-dimensional ordinal mastery vector over the ESCO skill graph, populated by MA-GPCM applied to logged exercise responses or self-assessments. The action space is the set of next-courses on a course catalog (Coursera Open, MIT OCW). The reward is a hybrid of estimated learning gain and a downstream labor market signal (placement probability, wage gain, or skill demand growth from job-ad feeds). We train with offline RL on resume-trajectory data, evaluated under a doubly robust estimator.

**Baselines.** Embedding-based job-to-candidate matching (Lan et al. 2020), career-pathway prediction (Yamashita et al. 2022), HRL for MOOC course recommendation (Zhang et al. 2019), and an IPS-only baseline.

**Datasets.** Resume-trajectory datasets (e.g. the open-source ones used by Yamashita et al.), Coursera course catalog with associated ESCO skill tags, job-postings panel for labor market signal.

**Metrics.** OPE estimate of placement rate, OPE estimate of wage gain at 1, 3, 5 years, skill-graph coverage of recommended pathways, regret against domain-expert pathways.

**Hard problem.** The reward signal (career outcome) is delayed by years, sparse, and confounded by the labor market. The open question is whether a measurement-grounded state (MA-GPCM ordinal skill vector) gives the policy enough signal to learn under sparse delayed reward, and whether the policy generalizes across labor market shifts. A second open question is the multi-stakeholder fairness, namely whether the recommender systematically pushes underrepresented students toward lower-wage paths.

**Formal claim.** Under a Markov assumption on the labor market and a calibration assumption on the MA-GPCM skill state, the offline-RL recommender achieves a high-confidence lower bound on placement rate that strictly improves over the logging policy. This is the high-confidence policy improvement framework of Thomas et al. (AAAI 2015), specialized to a structured state space.

## Thread C. Combined. MA-GPCM as the observation model for measurement-grounded DRL

This is the new contribution. The architectural claim is that MA-GPCM, as built, is not just a measurement model but also a candidate world model for educational planning. The DKVMN value memory is a sufficient statistic for the student state, and the IRT decoder is the emission function. Together they form a partially observable Markov decision process where the state is interpretable (mastery on each skill), the emission is psychometrically grounded (GPCM), and the action is a tutoring choice. Five sub-papers in increasing order of ambition.

### Thrust C.1. MA-GPCM as the dynamics model of a Dreamer-style latent planner

**Research question.** Does planning in MA-GPCM's latent state recover the gains promised by Dreamer-style world models (Hafner et al. 2020), and does the IRT structure give a more sample-efficient model than an unstructured world model.

**Method.** Treat MA-GPCM as the observation model. Add a learned action-conditioned latent transition g(z_t, a_t) on top of the DKVMN value memory, where z_t is the value memory state. Train the actor and critic in latent imagination, following DreamerV3. The reward is the calibrated learning gain from Thrust B.1.

**Baselines.** DreamerV3 with raw observations (a one-hot interaction vector), DreamerV3 with a black-box VAE encoder, MedDreamer (Wang et al. 2025) adapted to educational data, behavior cloning, and Decision Transformer on tutoring logs.

**Datasets.** Synthetic ITS environment with known ground-truth dynamics (built from the staircase and random-walk DGPs), then EdNet and ASSISTments for the real evaluation.

**Metrics.** Sample efficiency (rollouts to reach a fixed return), interventional validity (do the planner's predicted gains match the realized gains in the simulator), interpretability of the latent transitions.

**Hard problem.** Dreamer's promise rests on rolling out the world model many steps without compounding error. The open question is whether the IRT-structured observation model reduces compounding error relative to a generic VAE, and what the formal sample-complexity advantage is when the observation model is correctly specified up to a Lipschitz constant.

**Formal claim.** When the GPCM observation model is correctly specified up to a bounded log-likelihood ratio, the latent rollout error grows at most polynomially with horizon, in contrast to the exponential growth observed for unstructured world models. This is a finite-horizon-error bound analogous to the simulation-lemma-style results in model-based RL theory (Janner et al. 2019).

### Thrust C.2. Interventional validity of planning in latent space

**Research question.** A planner that scores well on logged data may still fail under intervention. When MA-GPCM is used as a world model and the planner picks items to maximize predicted theta gain, do the actual theta trajectories in the true DGP match the planner's predictions, or is there a counterfactual gap.

**Method.** Build a controlled synthetic environment where the true DGP is known (the candidate's existing kt-gpcm staircase and random-walk DGPs are exactly this). Train MA-GPCM on logged data from a behavior policy. Train a planner against MA-GPCM. Deploy the planner in the true DGP and measure the gap between the predicted theta trajectory and the realized theta trajectory. Then introduce systematic errors in the observation model (mis-specified Q-matrix, biased item parameters) and measure the breakdown.

**Baselines.** A perfect-observation-model planner (uses the true DGP, oracle), a planner with a generic VAE world model, a model-free CQL policy.

**Datasets.** Synthetic only, because the truth is needed.

**Metrics.** Counterfactual gap between predicted and realized theta gain, as a function of observation-model accuracy. Calibration of the planner's predicted return distribution against realized returns.

**Hard problem.** Interventional validity is hard to test on real educational data because we cannot run the counterfactual. The open question is what the gap looks like under realistic levels of MA-GPCM misspecification, and whether there is a regularizer on the planner (a conservative term) that closes the gap to within a quantifiable bound. This connects to causal latent state work (Schoelkopf 2021; von Kuegelgen 2024) because the question is whether the latent state recovered from passive logs is the same latent state under intervention.

**Formal claim.** The interventional gap is bounded by the KL divergence between the conditional distribution of theta given action under the data-generating distribution and under the learned world model, plus a term scaling with the policy class capacity. Closing the gap requires either an interventional data assumption or a structural assumption (e.g. monotonicity of GPCM in theta, which MA-GPCM enforces).

### Thrust C.3. Causal latent state vs predictive latent state in MA-GPCM

**Research question.** Is MA-GPCM's theta a causal latent (do interventions on items actually move theta) or only a predictive latent (it correlates with future responses but is not a controllable variable). For a tutoring planner, only the causal interpretation is justified.

**Method.** Use the formal causal representation learning framework (Schoelkopf et al. 2021; Khemakhem et al. 2020). Identify what conditions on data, sequence length, and item diversity are required for MA-GPCM's recovered theta to coincide with the causal latent up to a smooth bijection. Use the candidate's synthetic environments to test the identification empirically. Compare against pi-VAE and iVAE under analogous conditions.

**Baselines.** iVAE, pi-VAE, NeuralCD, DisenGCD, and the candidate's MA-GPCM with and without the orthogonality penalty.

**Datasets.** Synthetic ITS with intervention data (need a DGP with confounders and instruments), plus the released causal-IRT toy datasets if available.

**Metrics.** Identifiability error (correlation between recovered and true theta up to a smooth bijection), interventional prediction error (does the model predict the right theta after a forced item change).

**Hard problem.** Identifiability for sequential latent variable models with time-varying latents is open. The open question is the minimal data condition (number of items, number of intervention types, sequence length) for theta to be identifiable as a causal variable, not just as a sufficient statistic for prediction.

**Formal claim.** Under a non-Markovian causal model (von Kuegelgen et al. 2024) with at least D + 1 distinct intervention regimes (one per skill plus a control), MA-GPCM's theta is identifiable up to a permutation and a smooth element-wise transformation. The orthogonality penalty resolves the rotation ambiguity, and the GPCM monotonicity resolves the sign.

### Thrust C.4. Bandits with IRT-style item information

**Research question.** Classical CAT chooses the item with maximum Fisher information at the current theta estimate. Modern bandit theory chooses the item with maximum expected reward minus exploration cost. Are these two principles compatible, and what is the right unified algorithm when the reward is a learning-gain estimate from MA-GPCM.

**Method.** Generalize Thompson sampling on alpha and beta to compute Fisher information at the sampled theta, and use Fisher information as the exploration bonus rather than as the reward (as BanditCAT does). The reward is the expected learning gain, computed by rolling MA-GPCM forward under a counterfactual response. This is essentially a Bayesian planning step inside the bandit.

**Baselines.** BanditCAT (Sharpnack 2024), classical Fisher-information CAT, UCB-style bandit with a learned reward, Thompson sampling on the reward only.

**Datasets.** Duolingo English Test (already used by BanditCAT), synthetic CAT, ASSISTments adaptive-mode subsets.

**Metrics.** Measurement precision (theta posterior variance) and learning gain (delta theta), jointly evaluated as a Pareto frontier.

**Hard problem.** The unified objective (maximize learning gain plus a Fisher-information bonus) has a non-standard exploration-exploitation tradeoff because the bonus rewards measurement, not the policy's return. The open question is whether there is a single regret bound that handles both, and whether the deep variant inherits the asymptotic optimality of top-two Thompson sampling (Thrust B.3 baselines).

**Formal claim.** Under standard regularity, the unified algorithm has a sublinear regret O(sqrt(T log T)) against the joint objective, with constants tied to MA-GPCM's GPCM Fisher information at the estimated theta. This is the bandit version of the joint planning-measurement result.

### Thrust C.5. Open-world bandit, planner, and IRT estimator running on a live MOOC

**Research question.** When MA-GPCM is deployed as the observation model, the planner as the recommender, and the bandit as the exploration policy, all running online and updating each other, does the system maintain calibration and policy improvement guarantees, or does the closed-loop coupling create new failure modes.

**Method.** Build the full system on top of an open MOOC platform (Khan Academy Khanmigo open API or OpenStax adaptive practice). Run a controlled deployment where some students see the unified system and some see a control policy (Fisher-information CAT for assessment, behavior-policy item ordering for recommendation). Use sequential A-B testing with bounded type-I error.

**Baselines.** Static MA-GPCM + behavior policy, dynamic MA-GPCM + BanditCAT, dynamic MA-GPCM + offline-RL recommender (Thread B), the full unified system.

**Datasets.** Live deployment.

**Metrics.** Calibration of MA-GPCM under closed-loop data, regret against a static-policy oracle on theta gain, exposure distribution, fairness across demographic groups.

**Hard problem.** Closed-loop data drifts away from the i.i.d. assumption used in offline training. The open question is whether the streaming amortized property of MA-GPCM, combined with periodic recalibration triggers, suffices to keep the system calibrated. A second open question is whether the planner-bandit coupling has any failure mode analogous to performative prediction (Perdomo et al. 2020).

**Formal claim.** Under a performative stability condition (the data-generating distribution induced by the policy is Lipschitz in the policy), the closed-loop system converges to a performative-optimal fixed point, with the bandit's exploration providing sufficient distributional support to maintain MA-GPCM calibration. This is the educational performative-prediction result.

## Thread D. Extensions (optional)

Two extensions to deepen the combined story.

### Thrust D.1. Continuous-time MA-GPCM via neural SDEs

Replace the discrete-step DKVMN dynamics with a neural stochastic differential equation on theta, with a GPCM emission at each observed interaction time. This is the continuous-time analog of MA-GPCM and resolves a known issue in KT, namely that real students answer items at irregular times and the model collapses the time dimension. Baseline against ODE-RNN (Rubanova et al. 2019) and neural SDE work (Li et al. 2020). The hard problem is identifiability of the drift and diffusion under sparse observation, with a formal claim along the lines of "the GPCM Fisher information at observation times suffices to identify the SDE drift up to a deterministic transformation of theta."

### Thrust D.2. MA-GPCM with text-conditional item embedding for open-ended responses

Replace the integer item-id with a text-conditional embedding from a pretrained LLM, allowing MA-GPCM to ingest items it has never seen before. This is a generalization of Thrust A.2 but to the textual setting. The hard problem is whether the LLM embedding preserves the monotonic-discrimination structure that GPCM requires, and whether a regularizer on the embedding (such as a sparse anchor item set with known parameters) is needed to keep the IRT head identifiable. Baseline against AutoIRT and Open-ended KT (GPT-OKT, Liu et al. 2022).

## Cross-cutting threads

Three themes run across all three threads and tie them together.

**Identifiability.** Every neural latent variable model has a rotation, scaling, or sign indeterminacy. MA-GPCM resolves the K=1 case with the monotonicity of GPCM and the ordering of beta thresholds. The MIRT lift in Thrust A.3 needs orthogonality plus Q-matrix sparsity. The causal-latent claim in Thrust C.3 needs interventions. A unified theme of the thesis is that identifiability in deep educational measurement models is not a single condition but a hierarchy of progressively stronger assumptions, and that the encoder-decoder modular design pattern is what makes these assumptions visible and testable. The formal scaffolding is the iVAE family (Khemakhem et al. 2020), the sparse-decoding family (Moran et al. 2022, arXiv:2110.10804), and the recent non-Markovian causal disentanglement work (von Kuegelgen et al. 2024).

**Validity.** The candidate's stance is educational AI, not psychometrics. But every claim about theta needs to be defended against the question, "is this the right latent variable for the educational decision at hand". Three notions of validity appear repeatedly, namely (i) predictive validity (does theta predict future responses, a standard KT metric), (ii) construct validity (does theta correlate with external measures, e.g. NAEP scale scores, transcript GPA), and (iii) interventional validity (does an action that changes theta actually produce the predicted learning gain, the Thread C question). The thesis will treat (i), (ii), (iii) as a single object, with the MA-GPCM design pattern as the encoder-decoder substrate and the GPCM head as the bridge to classical psychometric validation.

**Calibration.** The streaming single-pass property is only useful if the posterior is calibrated. Thrust A.1 makes this explicit through the PFN framing. Thrust B.1 uses calibration to bound reward-hacking risk. Thrust C.1 uses calibration to bound rollout error. The thesis can stake out a single quantitative target, namely an expected calibration error below a threshold on EdNet-scale data, and use it as the common acceptance criterion for every sub-paper.

## Open questions for the candidate

These are decisions the program rests on. The candidate should choose before the next round.

1. What is the right primary venue. IJAIED has accepted MA-GPCM and is the natural home for Thrust A. EDM and LAK are the natural homes for Thrusts B and D. AAAI, NeurIPS, ICML are the natural homes for Thrusts A.1 and C (formal claims). The candidate should commit to a venue mix that supports both an educational AI thesis and the formal claims.

2. How much real-data infrastructure is in reach. Thrusts A.4 (NAEP, PISA) and C.5 (live MOOC) need either institutional access or partnership. The program should specify which partnerships to pursue in years one and two, before Thrust C.5 starts.

3. What is the right scope for Thread B. The candidate already has a DRL recommender. Are we doing four sub-papers in Thread B or fewer, with the saved time going to Thread C or to Thread A's amortization work. The current blueprint assumes four, but three with a deeper Thread A is also viable.

4. The combined thread (C) is the most novel and the highest-risk. The candidate should commit to which two of C.1, C.2, C.3, C.4, C.5 are core thesis chapters and which are stretch goals. C.1 and C.3 are the most defensible. C.5 is the most ambitious.

5. The optional Thread D is a pure stretch. Continuous-time MA-GPCM (D.1) is a beautiful problem but is a separate research arc. Text-conditional MA-GPCM (D.2) is a near-term extension of Thrust A.2 and might fold into A.2. The candidate should decide whether either belongs in the dissertation or is a postdoc project.

Sources, in order of citation in the text. Piech et al., Deep Knowledge Tracing, NeurIPS 2015 (arXiv:1506.05908). Zhang et al., Dynamic Key-Value Memory Networks for Knowledge Tracing, WWW 2017 (arXiv:1611.08108). Yeung, Deep-IRT, EDM 2019. Pandey, Karypis, A Self-Attentive Model for Knowledge Tracing, EDM 2019. Choi et al., SAINT, L@S 2020 (arXiv:2002.07033). Shin et al., SAINT+, LAK 2021 (arXiv:2010.12042). Ghosh, Heffernan, Lan, Context-Aware Attentive Knowledge Tracing, KDD 2020 (arXiv:2007.12324). Wang et al., NeuralCD, AAAI 2020. Liu et al., A Survey of Models for Cognitive Diagnosis, arXiv:2407.05458 (2024). DisenGCD, NeurIPS 2024. Tsutsumi et al., Dynamical Non-compensatory MIRT, Psychometrika 2023. Sharpnack et al., BanditCAT and AutoIRT, arXiv:2410.21033 (2024). Li, Gibbons, Rockova, Deep Computerized Adaptive Testing, Psychometrika 2026 (arXiv:2502.19275). Zammit-Mangion, Sainsbury-Dale, Huser, Neural Methods for Amortized Inference, arXiv:2404.12484 (2024). Hollmann et al., TabPFN, ICLR 2023 and Nature 2025. Muller et al., Transformers Can Do Bayesian Inference, arXiv:2112.10510 (2021). Cao et al., CORAL (2019, 2020). Shi et al., CORN (2022). Vargas et al., Cumulative link networks (2019). Doroudi, Aleven, Brunskill, scoping review, IJAIED 2019. Li, Xu, Zhang, Chang, Deep RL for adaptive learning, JEBS 2023. Cai et al., Reinforcing User Retention, arXiv:2302.01724 (2023). Bai et al., Sequential Recommendation, arXiv:2404.03637 (2024). Liu et al., Long-term engagement, arXiv:1902.05570 (2019). Dudik, Erhan, Langford, Li, Doubly Robust, 2011 and JMLR 2014. Su et al., Doubly Robust with Shrinkage, ICML 2020 (arXiv:1907.09623). Saito and Joachims, combinatorial OPE, RecSys 2024. Jiang and Li, DR for MDPs, ICML 2016. Thomas and Brunskill, MAGIC, ICML 2016. Kumar et al., CQL, NeurIPS 2020 (arXiv:2006.04779). Kostrikov et al., IQL, ICLR 2022. Fujimoto and Gu, TD3+BC, NeurIPS 2021. Yu et al., COMBO, NeurIPS 2021. Chen et al., Decision Transformer, NeurIPS 2021. Zhang et al., HRL for MOOC, AAAI 2019. Reinforced MOOC concept recommendation, ACM TWEB 2023. Lan et al., Skill-based Career Path Modeling, IEEE BigData 2020. Yamashita et al., Career Pathway Prediction, WSDM 2022. Coste et al., Diverse LoRA RM ensembles, arXiv:2401.00243 (2024). Zhai et al., Uncertainty-aware RLHF, arXiv:2410.23726 (2024). PURM, arXiv:2503.22480 (2025). Ha and Schmidhuber, World Models, 2018. Hafner et al., PlaNet 2019, Dreamer 2020, DreamerV2 2021, DreamerV3 2023. Wang et al., MedDreamer, arXiv:2505.19785 (2025). Tobin et al., domain randomization, IROS 2017. PolySim, arXiv:2510.01708 (2025). Voloshin et al., OPE empirical study, NeurIPS 2021. Fu et al., D4RL, arXiv:2004.07219 (2020). Hanna, Stone, Niekum, IS with estimated behavior policy, ICML 2019. Thomas, Theocharous, Ghavamzadeh, high-confidence OPE, AAAI 2015. Schoelkopf et al., Toward Causal Representation Learning, 2021. Khemakhem et al., iVAE, AISTATS 2020. von Kuegelgen et al., Disentangled Representation Learning in Non-Markovian Causal Systems, NeurIPS 2024. Moran et al., Identifiable Deep Generative Models via Sparse Decoding, arXiv:2110.10804. Li et al., LinUCB cold-start, ICML 2010. Hu et al., Cluster-based Bandits, SIGIR 2021. Wang et al., Fairness of Exposure in Stochastic Bandits, ICML 2021 (arXiv:2103.02735). Achiam et al., Constrained Policy Optimization, ICML 2017. Settles and Meeder, Half-Life Regression, ACL 2016. Rubanova et al., ODE-RNN, NeurIPS 2019. Li et al., Neural SDEs, 2020. Perdomo et al., Performative Prediction, ICML 2020. Frequentist Consistency of PFNs for Causal Inference, arXiv:2603.12037.
