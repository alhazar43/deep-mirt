# Explainable Deep Learning for Educational and Vocational Decision Making

## 1. What I have built

**MA-GPCM** (under review). DKVMN encoder, separated ability pathway, GPCM decoder. One forward pass gives per-time $\theta_t$, per-item $\alpha_q$, step thresholds $\boldsymbol{\beta}_q$.

```
   (q_t, r_t) --> f_phi (DKVMN) --> h_t --> g_psi (GPCM) --> theta_t, alpha_q, beta_q
```

Recovery matches batch EM under stationary data, exceeds EM under discrete shift and continuous drift. Binary reduction at $K=2$ is competitive with DKT, DKVMN, SAKT, AKT, SimpleKT, SAINT+ on ASSISTments 2009/2017 (5-fold CV).

**DRL-IRT** (working manuscript). Cold-start job recommender. Classical IRT runs adaptive testing over RIASEC and Big Five for a 2-dim trait $\theta_t$. A DQN refines list $K_t$ via keep/add/remove on $s_t = (\theta_t, F_{K_t})$. Reward is log-difference of user-list alignment.

```
   adaptive testing: theta_t = argmax_theta sum_s log P_is(u_is | theta)
   DQN refinement:   a_t = argmax_z Q(s_t, z),  s_t = (theta_t, F_{K_t})
```

DQN_Q50 beats dot-product matching on Prec@5/10 and Recall@5/10.

Both maintain a structured latent user state via streaming inference and use it for decisions. MA-GPCM has a deep dynamic encoder, DRL-IRT has classical IRT. MA-GPCM is measurement-only, DRL-IRT couples to a policy.

---

## 2. Narrative

One method, one claim.

**Method.** Put a structured psychometric decoder in front of a deep sequential encoder. Train end to end. Encoder handles streaming interaction. Decoder produces interpretable IRT or CDM parameters. One forward pass, no batch step. MA-GPCM validated this for unidim KT. Multidim MA-IRT (Section 3) generalizes it. Every direction in Section 4 reuses it.

**Claim.** Educational and vocational AI systems make decisions about humans, and increasingly with or by AI. Current explanations are weak, either post-hoc interpretation over opaque states or aggregate accuracy. The claim is that decisions become explainable when the model produces interpretable IRT parameters as part of its forward pass, the decision is computed from those parameters, and the explanation cites them. The IRT parameters are the units of explanation. Without them, explanations are decorative.

Four directions in Section 4, four uses of MA-IRT.

- **A** uses MA-IRT parameters as the state for a DRL policy. Recommendations explained in trait and item-information units.
- **B** treats LLM components (rater, generator, content delivery) as structured contributions to MA-IRT estimates, not as oracles.
- **C** uses the MA-IRT decoder as the observation function of a world model. Planning in trait units across months and years.
- **D** estimates MA-IRT parameters when the test-taker is an LLM. Same parameter set, AI subject.

A secondary point. MA-IRT is application-neutral in the human-vs-AI sense. Built for humans, it estimates the same parameters for AI test-takers with no architectural change. Direction D exists because that observation is testable.

The thesis exists because nothing in education or vocation currently produces interpretable IRT parameters in single forward-pass form across these four cases.

---

## 3. The committed first step, multidim MA-IRT

MA-GPCM is scalar. Everything in Section 4 needs $\boldsymbol{\theta}_t \in \mathbb{R}^D$. RIASEC ($D=6$), Big Five ($D=5$), joint ($D=11$). Per-skill ability tracking in KT also needs multidim. So the first paper is multidim MA-IRT.

DKVMN memory stays. Ability pathway widens to a vector,

$$
\boldsymbol{\theta}_t = \mathrm{MLP}_\theta(h_t) \in \mathbb{R}^D.
$$

Decoder swaps. Multidim 2PL for binary,

$$
P(r = 1 \mid q, \boldsymbol{\theta}_t) = \sigma\!\big(\boldsymbol{\alpha}_q^\top \boldsymbol{\theta}_t - b_q\big), \quad \boldsymbol{\alpha}_q \in \mathbb{R}^D.
$$

Multidim GPCM for $K$-category ordinal,

$$
P(r = k \mid q, \boldsymbol{\theta}_t) = \frac{\exp\!\sum_{s=0}^{k}\big(\boldsymbol{\alpha}_q^\top \boldsymbol{\theta}_t - \beta_{q,s}\big)}{\sum_{k'=0}^{K-1}\exp\!\sum_{s=0}^{k'}\big(\boldsymbol{\alpha}_q^\top \boldsymbol{\theta}_t - \beta_{q,s}\big)}.
$$

Multidim DINA for CDM. Loading matrix $A = [\boldsymbol{\alpha}_q]_q$ is either a fixed Q from the questionnaire's published design or learned via sparse attention with $L_1$ penalty.

Two open problems.

**Rotational identifiability under streaming updates.** Batch MIRT is identified up to orthogonal rotation [16]. Anchor items pin the rotation. With a neural encoder updating online, rotation can drift without anchors. Test stability with and without anchors. Report rotational drift alongside parameter recovery.

**Sparse-attention Q-matrix recovery.** Sparse attention should recover the generating Q's sparsity pattern. Test on synthetic GPCM at $D = 2, 5, 6, 11$. Compare against R `mirt` and sparse Bayesian MIRT [20].

Validation. Synthetic data with known $(\boldsymbol{\theta}, A, B)$ at multiple $D$, plus RIASEC and Big Five where published Q-structure is ground truth. Per-dim Pearson and RMSE on $\boldsymbol{\theta}$, RMSE on $\boldsymbol{\alpha}_q$, rotational similarity.

---

## 4. Directions for further research

Four directions, options not a pipeline. Shared model is multidim MA-IRT, shared theoretical thread is identifiability of streaming neural IRT (final sub-section). Not all independent. B builds on A. C wraps A or B in a world model. D is independent. Realistic 2-year scope is multidim MA-IRT plus two to three of these, chosen as year-1 work surfaces problems.

### 4.A Joint multidim MA-IRT and DRL recommender

MA-IRT inside the decision loop, human-as-test-taker.

Replace classical IRT in DRL-IRT with multidim MA-IRT. Share the encoder. DQN sees $h_t$ instead of the 2D point estimate.

$$
h_t = f_\phi(\cdot), \quad (\boldsymbol{\theta}_t, A, B) = g_\psi(h_t), \quad Q(h_t, a) = Q_\omega(h_t, a),
$$

$$
\mathcal{L} = \mathcal{L}_{\text{meas}}(\boldsymbol{\theta}, A, B; r) + \lambda \, \mathcal{L}_{\text{DQN}}(Q; \text{TD targets}).
$$

**4.A.1 Item-agnostic encoding.** Same encoder for exercises, questionnaire items, jobs, courses. Content encoding $c_q$ is a learned id for stable-identity items, pretrained text for content-defined items. Test cross-domain transfer (KT to vocational and back). Validates the item-agnostic claim DRL-IRT makes but does not realize.

**4.A.2 Joint training stability.** As $\lambda$ grows the policy gradient pulls the encoder. Pure measurement ($\lambda = 0$) is the reference. Characterize how recovery degrades, find the range where the policy adds value without harming measurement. Real question, when the encoder also serves a policy gradient, is $\boldsymbol{\theta}_t$ still a measurement?

**4.A.3 OPE under learned observation.** Reward depends on $\boldsymbol{\theta}_t$, which is the learned output of $g_\psi$. Reward learning is partially identified under affine transformations [51]. Doubly robust OPE [49, 50] with the measurement head as the observation model returns policy value as a confidence set, not a point.

Deepest claim. Measurement and policy are not independent. Joint training closes a feedback loop that creates new identifiability problems, and partial-ID-aware OPE is the honest tool for that loop.

### 4.B LLM-mediated assessment with measurement accountability

MA-IRT when AI mediates the interaction. Builds on A.

LLMs grade constructed responses, generate items, deliver content. The MA-IRT decoder remains the measurement model. LLM components are structured contributions, not oracles.

Rater channel,

$$
P(m_q = k \mid r_t, c_q, c_{\text{resp}}) = \mathrm{Cat}\!\big(k; \mathrm{softmax}(W [c_q ; c_{\text{resp}}] + b + e_{r_t})\big).
$$

Generator prior for new item $q^\star$ with content $c_{q^\star}$,

$$
\boldsymbol{\alpha}_{q^\star} \sim \mathcal{N}\!\big(\boldsymbol{\mu}_\alpha(c_{q^\star}), \boldsymbol{\Sigma}_\alpha(c_{q^\star})\big), \quad \text{expose iff } \mathrm{tr}\, \boldsymbol{\Sigma}_{q^\star} < \tau.
$$

**4.B.1 LLM rater bias.** Position bias, length bias, rubric-conditioning effects [37, 38, 39] go beyond what many-facet Rasch [25] handles. State identifiability conditions for separating user trait from content-conditional rater bias. Test under controlled rater injection.

**4.B.2 Cold-start safety.** The exposure gate is the safety property. Test whether content-derived priors compress the cold-start window enough that uncalibrated LLM items can enter the live bank.

**4.B.3 Architecturally faithful explanations.** Project the policy gradient $\nabla_{h_t} Q$ onto $(\boldsymbol{\theta}_t, A, B)$. The reason is in measurement units, derived from the model that took the action, not from a separate LLM rationalization. Validate against counterfactual perturbations of $h_t$.

Deepest claim. An LLM placed as rater or generator is a measurement instrument with its own psychometric properties. Characterizing those is a measurement problem the field is not yet doing.

### 4.C Learner world model and long-horizon planning

MA-IRT at long horizons. Action space stays virtual. No physical interventions.

Wrap A (or B) in a world model. Latent dynamics,

$$
s_t = \mathrm{enc}(h_t), \quad s_{t+1} = f_{\text{world}}(s_t, a_t) + \varepsilon_t, \quad P(r_{t+1} \mid q_{t+1}, s_t) = g_\psi(\mathrm{dec}(s_t)).
$$

MA-IRT decoder is the observation. Planning over horizon $H$,

$$
a_t = \arg\max_{a} \mathbb{E}\!\left[\sum_{h=0}^{H} \gamma^h R(s_{t+h}, a_{t+h}) \;\middle|\; s_t, a_t = a, \pi\right].
$$

Two templates. DreamerV3 [47] for imagination-based planning. MedDreamer [48] for longitudinal latent-state planning with sparse irregular observations, the data shape closest to education. Alternative, LLM as world model via in-context simulation, RAP [63]. Compare both.

**4.C.1 Causal identifiability under intervention.** Adaptive recommendation correlates actions with the $\boldsymbol{\theta}_t$ trajectory. Disentangling intrinsic learning from policy-induced selection needs structural assumptions on $f_{\text{world}}$ or exploration noise. Identifiability tools from [45, 46]. State which counterfactuals are supported, with synthetic causal DGP tests.

**4.C.2 Credit assignment with measurement as dense signal.** Education rewards are sparse and delayed. Test whether the per-step $\boldsymbol{\theta}_t$ trajectory closes the credit-assignment gap, comparing against a world model that only sees final reward.

**4.C.3 Lifelong measurement under drift.** Items and populations drift over years. Online conformal [53] for running coverage on $\boldsymbol{\theta}_t$ intervals. Bayesian online changepoint [55] flags drift. Build a continual MA-IRT pipeline that updates online without forgetting past calibration.

Deepest claim. KT predicts but does not explain why. World models with structural assumptions can compute counterfactuals about learning under stated assumptions, with the MA-IRT trajectory as the dense signal observational RL alone lacks.

### 4.D MA-IRT for AI capability evaluation

MA-IRT with an LLM as the test-taker.

Same framework. Benchmark plays item bank. Streaming inference produces capability estimates. Adaptive selection [28] reduces benchmark size [67]. Multidim MA-IRT becomes multidim capability profiling.

$$
h_t = f_\phi(h_{t-1}, c_{q_t}, r_t), \quad r_t = \text{AI-response}(q_t), \quad \boldsymbol{\theta}_t^{\text{AI}} = g_\psi(h_t).
$$

**4.D.1 Multidim capability profiling.** Apply multidim MA-IRT to benchmark patterns across LLMs (MMLU, BBH, HumanEval). Recover capability factor structure. Compare against per-benchmark accuracy. Test whether multidim IRT reveals axes that aggregate accuracy hides.

**4.D.2 Capability across fine-tunes.** Track $\boldsymbol{\theta}_t^{\text{AI}}(\text{model})$ across model versions. Drift in capability becomes a measured quantity with calibrated uncertainty.

**4.D.3 Cross-model DIF.** Items LLM-A handles differently from LLM-B at matched overall capability suggest content-specific differences. Flag for audit. AI-side analog of classical differential item functioning.

Deepest claim. AI capability is itself a latent construct. Current evaluation by aggregate accuracy is 19th-century classical test theory. Multidim IRT recovers structure that aggregates hide.

### Cross-cutting identifiability thread

Every direction creates a new identifiability question. Section 3, rotational drift under streaming. A, policy bias on measurement. B, rater-channel separation. C, latent dynamics under intervention. D, cross-model invariance. A single theoretical treatment that addresses all five is one possible methodological contribution, independent of which directions are pursued in full.

---

## 5. Methodology

**Synthetic data.** MA-GPCM's four generators (static, block-shift, staircase, random walk) extend to multidim (Section 3), to LLM-rated free text (B), to causal intervention (C). DRL-IRT's centroid-mixture job-feature generator extends similarly.

**Real datasets.** ASSISTments 2009/2017 [57], EdNet [58], pyKT [59] for KT. RIASEC and Big Five from OpenPsychometrics for vocational. Deployment partner sought for A.3 OPE. Public LLM benchmark responses (MMLU, BBH, HumanEval) for D.

**Baselines.** DKT [3], DKVMN [4], SAKT [5], AKT [6], SimpleKT [7], SAINT+ [60]. R `mirt`, Stan, Deep-IRT [61], NeuralCD [18], sparse Bayesian MIRT [20]. Dot-product matching from DRL-IRT, BanditCAT [28], CQL [8]. DreamerV3 [47], RAP [63]. tinyBenchmarks [67].

**Evaluation.** Recovery (per-dim Pearson, RMSE) on synthetic. ACC, AUC, QWK on real KT. Prec@k, Recall@k on recommendation. Posterior coverage on held-out responses. OPE confidence sets under partial ID for A.3. Interventional validity on synthetic causal data for C.1. Capability recovery and DIF for D.

---

## 6. Contributions

**C1, multidim MA-IRT.** Committed first paper. Multidim generalization of MA-GPCM, sparse-attention Q-matrix learning, rotational stability under streaming.

**C2, Direction A.** Joint MA-IRT + DRL recommender. Item-agnostic encoding. Joint training stability. Partial-ID-aware OPE. Puts MA-IRT inside the decision loop.

**C3, Direction B.** LLM-mediated assessment. Structured rater channel, cold-start safety, faithful explanations. Preserves MA-IRT under AI mediation.

**C4, Direction C.** Learner world model. Causal identifiability, MA-IRT trajectory as dense credit signal, continual measurement. Extends MA-IRT to long horizons.

**C5, Direction D.** MA-IRT for AI evaluation. Multidim capability profiling, cross-fine-tune tracking, cross-model DIF. Applies MA-IRT to AI test-takers.

**C6, cross-cutting.** Single theoretical treatment of identifiability for streaming neural IRT covering rotational, policy-bias, rater-channel, dynamics, cross-model invariance.

2-year scope realistically delivers C1 plus two to three of C2-C6.

---

## References

[3] Piech, C., Spencer, J., Huang, J., Ganguli, S., Sahami, M., Guibas, L., and Sohl-Dickstein, J. Deep knowledge tracing. *Advances in Neural Information Processing Systems*, 2015. arXiv:1506.05908.

[4] Zhang, J., Shi, X., King, I., and Yeung, D.-Y. Dynamic key-value memory networks for knowledge tracing. *Proceedings of the 26th International Conference on World Wide Web*, 2017. arXiv:1611.08108.

[5] Pandey, S., and Karypis, G. A self-attentive model for knowledge tracing. *Proceedings of the 12th International Conference on Educational Data Mining*, 2019. arXiv:1907.06837.

[6] Ghosh, A., Heffernan, N., and Lan, A. S. Context-aware attentive knowledge tracing. *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2020. arXiv:2007.12324.

[7] Liu, Z., Liu, Q., Chen, J., Huang, S., Tang, J., and Luo, W. SimpleKT, a simple but tough-to-beat baseline for knowledge tracing. *International Conference on Learning Representations*, 2023. arXiv:2302.06881.

[8] Kumar, A., Zhou, A., Tucker, G., and Levine, S. Conservative Q-learning for offline reinforcement learning. *Advances in Neural Information Processing Systems*, 2020. arXiv:2006.04779.

[16] Reckase, M. D. *Multidimensional Item Response Theory.* Springer, 2009.

[18] Wang, F., Liu, Q., Chen, E., Huang, Z., Chen, Y., Yin, Y., Huang, Z., and Wang, S. Neural cognitive diagnosis for intelligent education systems. *AAAI*, 2020. arXiv:1908.08733.

[20] Chen, J., Chen, H., and Lin, Z. Identifiable cognitive diagnosis with sparse Bayesian multidimensional IRT. *arXiv*, 2023. arXiv:2310.17820.

[25] Linacre, J. M. *Many-Facet Rasch Measurement.* MESA Press, 1989.

[28] Sharpnack, J., Hao, K., Mulgrew, P., Garrard, C., Lash, M., Smith, B., and Tomkins, A. BanditCAT and AutoIRT. *arXiv*, 2024. arXiv:2410.21033.

[37] Gu, J., Jiang, X., Shi, Z., et al. A survey on LLM-as-a-judge. *arXiv*, 2024. arXiv:2411.15594.

[38] Zhao, W. X., et al. LLMs are biased teachers. *arXiv*, 2024. arXiv:2410.14012.

[39] Singhal, P., et al. A long way to go, length correlations in LLM-based reward models. *arXiv*, 2024. arXiv:2407.01085.

[45] Yao, W., et al. Disentangled representation learning in non-Markovian causal systems. *NeurIPS*, 2024.

[46] Moran, G. E., Sridhar, D., Wang, Y., and Blei, D. M. Identifiable deep generative models via sparse decoding. *arXiv*, 2021. arXiv:2110.10804.

[47] Hafner, D., Pasukonis, J., Ba, J., and Lillicrap, T. Mastering diverse domains through world models (DreamerV3). *arXiv*, 2023. arXiv:2301.04104.

[48] Liu, M., et al. MedDreamer, model-based RL with latent imagination for personalized clinical treatment. *arXiv*, 2025. arXiv:2505.19785.

[49] Su, Y., Dimakopoulou, M., Krishnamurthy, A., and Dudik, M. Doubly robust off-policy evaluation with shrinkage. *ICML*, 2020. arXiv:1907.09623.

[50] Bian, Z., and Shi, C. Doubly inhomogeneous reinforcement learning. *Journal of the American Statistical Association*, 2024.

[51] Skalse, J., Howe, N. H. R., Krasheninnikov, D., and Krueger, D. Invariance in policy optimisation and partial identifiability in reward learning. *ICML*, 2023. arXiv:2203.07475.

[53] Angelopoulos, A. N., Candes, E. J., and Tibshirani, R. J. Online conformal prediction with decaying step sizes. *ICML*, 2024. arXiv:2402.01139.

[55] Adams, R. P., and MacKay, D. J. C. Bayesian online changepoint detection. *arXiv*, 2007. arXiv:0710.3742.

[57] Feng, M., Heffernan, N., and Koedinger, K. Addressing the assessment challenge with an online system that tutors as it assesses. *UMUAI*, 19(3), 243 to 266, 2009.

[58] Choi, Y., et al. EdNet, a large-scale hierarchical dataset in education. *AIED*, 2020.

[59] Liu, Z., et al. pyKT. *NeurIPS*, 2022. arXiv:2206.11460.

[60] Shin, D., et al. SAINT+. *LAK*, 2021. arXiv:2010.12042.

[61] Yeung, C.-K. Deep-IRT. *EDM*, 2019. arXiv:1904.11738.

[63] Hao, S., Gu, Y., Ma, H., Hong, J. J., Wang, Z., Wang, D. Z., and Hu, Z. Reasoning with language model is planning with world model. *EMNLP*, 2023. arXiv:2305.14992.

[67] Polo, F. M., Weber, L., Choshen, L., Sun, Y., Xu, G., and Yurochkin, M. tinyBenchmarks. *ICML*, 2024. arXiv:2402.14992.
