# Explainable Deep Learning for Educational and Vocational Decision Making

**Doctoral Research Proposal**

Wenrui Yuan
Faculty of Behavioural, Management and Social Sciences, University of Twente
2026-05-22

---

## 1. Motivation

Two of my prior papers set up the program.

**MA-GPCM** (manuscript under review). A memory-augmented Generalized Partial Credit Model for ordinal knowledge tracing. A DKVMN encoder produces a streaming state $h_t$, a separated ability pathway produces per-time ability $\theta_t$, and a GPCM decoder produces per-item discrimination $\alpha_q$ and step thresholds $\boldsymbol{\beta}_q$. One forward pass per interaction. Recovery matches batch EM on stationary data and exceeds EM under discrete shift and continuous drift.

**DRL-IRT** (working manuscript). A cold-start job recommender for users with no resume or interaction history. Classical IRT runs adaptive testing over RIASEC and Big Five questionnaires, producing a 2-dim trait $\theta_t$. A Deep Q-Network refines the recommendation list $K_t$ via keep, add, remove actions. State $s_t = (\theta_t, F_{K_t})$. Reward is log-difference of user-list alignment plus a bounded terminal alignment. Outperforms a dot-product matching baseline at full calibration.

The two pieces share one structure. An interactive system maintains a latent user trait via streaming inference, uses it to make decisions, and updates it from observed responses. They differ in three coordinates.

| coordinate | MA-GPCM | DRL-IRT |
|---|---|---|
| inference | deep dynamic (DKVMN, GPCM) | classical static IRT (JML adaptive testing) |
| outputs | prediction + parameter recovery | recommendation list refinement |
| data | educational interaction logs | psychometric questionnaires |

This sets up the cell to fill. Dynamic deep IRT jointly trained with a DRL recommender, item-agnostic across education and vocation. Neither of my prior pieces fills it. MA-GPCM has the dynamic encoder but no action. DRL-IRT has the action but its trait is static after the questionnaire (Algorithm 1 of DRL-IRT terminates and freezes $\theta_T$). The thesis fills the cell and extends from it.

The unifying principle is this. An explainable AI system for educational and vocational decision making maintains a structured latent state of the user via a single forward pass, uses the state to take an action, and updates the state from the response. Measurement, action, and explanation share an encoder. The structured part of the state is what makes the explanation possible.

Three research directions follow. Direction 1 fills the cell. Direction 2 adds LLM components inside the agent, addressing the limitation DRL-IRT itself names ("relying solely on questionnaire responses can lead to overly self-referential recommendations"). Direction 3 lifts the per-session recommender to long-horizon planning under a world model. The remaining two years scope all three through what each requires, not through a calendar.

---

## 2. What I have

### 2.1 MA-GPCM

```
            (q_t, r_t)
                |
                v
       DKVMN encoder f_phi       (single forward pass)
                |
                v  h_t
       separated ability path
                |
                v
       GPCM decoder g_psi   --> theta_t, alpha_q, beta_q
                |
                v
       ordinal categorical:  P(r_{t+1} = k | q_{t+1}, h_t)
```

Validated on four synthetic generators (static, block-shift, staircase, random walk) plus ASSISTments 2009 and 2017 with five-fold CV. Recovery matches R `mirt` on stationary data. Binary reduction at $K=2$ is competitive with DKT, DKVMN, SAKT, AKT, SimpleKT, SAINT+.

### 2.2 DRL-IRT

State $s_t = [\theta_t \,\|\, c_t] \in \mathbb{R}^{2d}$ where $\theta_t \in \mathbb{R}^2$ is the IRT-estimated trait after $t$ questionnaire items and $c_t = \sum_{j \in K_t} \alpha_j(\theta_t, K_t) F_j$ is the attention-weighted feature aggregation over the current list.

```
adaptive testing (Algorithm 1 of DRL-IRT):
   theta_0 prior
   loop until termination:
       i_t   = argmax_i Fisher_info(theta_t, item_i)
       u_it  = observe(user, i_t)
       theta_t = argmax_theta sum_s log P_is(u_is | theta)
   return theta_T

DQN refinement at each step:
   s_t = (theta_t, F_{K_t})
   a_t = argmax_z Q(s_t, z)         z in {keep, add(j), remove(j)}
   K_{t+1} = apply(a_t, K_t)
   r_t = log sigmoid(theta_t . F_{K_t}) - log sigmoid(theta_{t-1} . F_{K_{t-1}})
```

DQN_Q50 (full calibration) outperforms a dot-product matching baseline on Prec@5/10 and Recall@5/10.

### 2.3 What is missing between MA-GPCM and DRL-IRT

Three gaps that the program closes.

**Gap 1, the trait is shallow in DRL-IRT.** Classical IRT under JML returns a 2D point estimate. The DQN sees only $(\theta_t, F_{K_t})$, not the response history or any uncertainty. MA-GPCM's $h_t$ is a richer state that carries dynamics and uncertainty.

**Gap 2, the trait is frozen after the questionnaire in DRL-IRT.** Once Algorithm 1 terminates, $\theta_T$ is fixed for the rest of the interaction. Real users keep generating signal (clicks on jobs, hours spent on training material, completion of micro-courses). MA-GPCM's streaming encoder consumes this signal natively.

**Gap 3, measurement and policy are trained separately in DRL-IRT.** IRT calibration uses 10000 historical responses via batch JML. The DQN is trained per-user via MDP rollouts. There is no gradient between them. Joint training shares an encoder and propagates information both ways.

The core technical contribution of the program is to close these three gaps in one architecture, with downstream extensions for LLMs (Direction 2) and long horizons (Direction 3).

---

## 3. Direction 1, Joint MA-IRT and Deep Reinforcement Learning Recommender

This is the cell to fill. Two coupled sub-areas, the MA-IRT architecture family that generalizes MA-GPCM, and the deep RL recommender that extends DRL-IRT by replacing classical IRT with MA-IRT and adding joint training.

### 3.1 The MA-IRT family

MA-GPCM is one instance. The general form,

$$
h_t = f_\phi(h_{t-1}, c_{q_t}, r_t), \qquad P(r_{t+1} = k \mid c_{q_{t+1}}, h_t) = g_\psi(h_t, c_{q_{t+1}}, k),
$$

with $c_{q_t}$ an item encoding (id embedding, learned content vector, or LLM text embedding) and $g_\psi$ an IRT, polytomous IRT, or cognitive diagnosis response model. MA-GPCM has $g_\psi = $ GPCM, $f_\phi = $ DKVMN, $c_{q_t} = $ learned id embedding.

Three extensions cover the operational space of educational and vocational measurement.

**Multi-dimensional MA-IRT.** Scalar $\theta_t$ becomes $\boldsymbol{\theta}_t \in \mathbb{R}^D$. RIASEC is $D=6$ (Realistic, Investigative, Artistic, Social, Enterprising, Conventional), Big Five is $D=5$, joint use is $D=11$. The decoder for binary RIASEC items is multidim 2PL,

$$
P(r = 1 \mid q, \boldsymbol{\theta}_t) = \sigma(\boldsymbol{a}_q^\top \boldsymbol{\theta}_t - b_q), \qquad \boldsymbol{a}_q \in \mathbb{R}^D.
$$

For Likert items (Big Five), multidim GPCM, exactly Equation (2) of DRL-IRT with $\boldsymbol{a}_i^\top \boldsymbol{\theta}$ replacing $a_i \theta$. The skill-item loading matrix is either fixed from the questionnaire's published Q-structure or learned via sparse attention with an $L_1$ penalty. Open question, identifiability of $\boldsymbol{\theta}_t$ and $\boldsymbol{a}_q$ under streaming updates with a neural encoder. Batch MIRT identifiability is known [16, 20]. Streaming, dynamic, neural variants are not.

**Item-agnostic encoding.** In MA-GPCM items are KT exercise ids. In DRL-IRT items are RIASEC or Big Five questionnaire items with $(a_i, b_i)$, and jobs are 2D features $F_j$ from a separate space. The MA-IRT extension uses a unified item encoding $c_q \in \mathbb{R}^m$ for any item, exercise, questionnaire item, job, or training program. The encoder is

$$
h_t = f_\phi(h_{t-1}, c_{q_t}, r_t)
$$

with $c_{q_t}$ coming from (i) a learned id table for items with stable identity, (ii) an LLM text encoder for items defined by content (Direction 2), or (iii) the existing $F_j$ for jobs as in DRL-IRT. This makes the architecture truly item-agnostic, fulfilling the claim DRL-IRT makes but does not fully realize.

**Long-horizon MA-IRT.** Multi-session and continuous-time. Encoder takes a time gap input,

$$
h_t = f_\phi(h_{t-1}, c_{q_t}, r_t, \Delta t_t).
$$

Continuous-time IRT [30] is the measurement baseline. Deliverable, ability tracking that respects irregular session spacing typical of career assessment (months between visits).

### 3.2 Deep RL recommender on MA-IRT state

Replace the classical-IRT trait $\theta_t$ in DRL-IRT with the MA-IRT memory state $h_t$. The policy now operates on $h_t$ (richer than the 2D point estimate of DRL-IRT) and gets gradient through the shared encoder during training. One forward pass produces measurement, action, and explanation.

```
        interactions (c_q_t, r_t) over time
                  |
                  v
       encoder f_phi  (DKVMN or attention-based)
                  |
                  v  h_t
       +----------+-----------+
       |                      |
       v                      v
   MA-IRT decoder g_psi   policy head Q_omega
   theta_t, a, b          Q(h_t, z)
                          z in {keep, add(j), remove(j)}
```

Joint loss,

$$
\mathcal{L} = \mathcal{L}_{\text{meas}}(\boldsymbol{\theta}, \boldsymbol{a}, \boldsymbol{b}; r) + \lambda \, \mathcal{L}_{\text{DQN}}(Q; \text{TD targets}),
$$

with $\mathcal{L}_{\text{meas}}$ from MA-GPCM (weighted ordinal cross-entropy) and $\mathcal{L}_{\text{DQN}}$ the temporal-difference loss using the reward from DRL-IRT Equations (6) and (7).

Three sub-questions.

**Q1, identifiability under joint training.** Increasing $\lambda$ biases $\boldsymbol{\theta}_t$ toward whatever serves the policy. Pure measurement ($\lambda = 0$) is the reference. How does parameter recovery degrade as $\lambda$ grows? On synthetic data with known generating IRT parameters and known reward, sweep $\lambda$ and report the curve. Deliverable, an operating range where the recommender adds value without harming measurement validity.

**Q2, off-policy validity under learned observation.** DRL-IRT uses simulation. Real deployment requires off-policy evaluation, and the reward in DRL-IRT depends on $\boldsymbol{\theta}_t$ which is the learned output of $g_\psi$. Skalse et al. [51] show reward learning is partially identifiable under affine transformations of utility, which transfers here. Use doubly robust OPE [49, 50] with the measurement head as the observation model and report policy value as a confidence set across the identifiable family.

**Q3, constrained policies.** Adaptive testing imposes exposure caps, content balancing, and Fisher information matching at the current trait [62]. These are added as a Lagrangian penalty,

$$
\mathcal{L}_{\text{DQN}}^{\text{constr}} = \mathcal{L}_{\text{DQN}} + \sum_{c \in \mathcal{C}} \mu_c \, \mathrm{ReLU}(\text{violation}_c).
$$

The recommender becomes a measurement-aware adaptive testing engine that respects the constraints the field requires.

### 3.3 Item-agnostic domain transfer

The item-agnostic encoding from 3.1 enables cross-domain validation as part of the deliverable. The same MA-IRT + DRL system handles

- KT-style learning content recommendation, items are exercises, responses are ordinal correctness, baseline is MA-GPCM
- DRL-IRT-style job recommendation, items are jobs with content embeddings, responses are interest ratings, baseline is DRL-IRT
- Vocational training recommendation, items are micro-courses, responses are engagement and completion

A model trained on one domain transfers to another by swapping the item content encoder and the response head, holding the encoder $f_\phi$ and the policy structure constant. The transfer experiment validates the item-agnostic claim that DRL-IRT states.

---

## 4. Direction 2, LLM-Influenced Agentic System

The limitation DRL-IRT explicitly names is self-referentiality. Likert questionnaires yield self-reported traits and the system has no other signal. Direction 2 addresses this with LLM components inside the agent, run on top of Direction 1. The architecture is the single forward pass below.

```
1. h_t            = f_phi(h_{t-1}, c_{q_t}, r_t)                 # perception (MA-IRT encoder)
2. (theta_t, a, b) = g_psi(h_t)                                  # measurement (MA-IRT decoder)
3. a_t            = pi_omega(h_t)                                # action (DRL policy, Direction 1.2)
4. content_t      = LLM_deliver(a_t, h_t)                        # action surface (sub-area 4.3)
5. r_{t+1}        = LLM_rate(q_{t+1}, response_{t+1})            # rater channel (sub-area 4.1)
                  or = direct_response(q_{t+1})                  # if MC or Likert
6. if a_t = introduce_new_item:
       q*         = LLM_generate(target_skill, target_difficulty)
       c_{q*}     = LLM_embed(q*)                                # cold-start prior (sub-area 4.2)
7. explanation_t  = trace(a_t, h_t, theta_t)                     # measurement-grounded reason
```

Each LLM role is one sub-area. Each addresses a specific gap in MA-GPCM or DRL-IRT.

### 4.1 LLM as response rater (line 5)

DRL-IRT is restricted to Likert items because classical IRT needs categorical responses. The agentic extension supports open-ended responses (career narratives, problem solutions, project descriptions) graded by an LLM. Treat the LLM score $m_q$ as a noisy ordinal observation of the true response category,

$$
P(m_q = k \mid r_t, c_q, c_{\text{resp}}) = \mathrm{Cat}\big(k; \mathrm{softmax}(W [c_q ; c_{\text{resp}}] + b + e_{r_t})\big),
$$

with $e_{r_t}$ a learned per-category offset. The MA-IRT decoder marginalizes over the rater distribution during likelihood. LLM raters exhibit content-correlated bias [37, 38, 39] beyond what many-facet Rasch [25] handles. The open question is identifiability of student or candidate trait when the rater channel is itself a learned function of response content.

### 4.2 LLM as item generator (line 6)

A new item $q^*$ enters the bank with content $c_{q^*}$ and no response data. Initialize its parameters from content-conditional prior heads trained on the existing calibrated bank,

$$
\boldsymbol{a}_{q^*} \sim \mathcal{N}(\mu_a(c_{q^*}), \sigma_a^2(c_{q^*})), \qquad b_{q^*} \sim \mathcal{N}(\mu_b(c_{q^*}), \sigma_b^2(c_{q^*})).
$$

Exposure of $q^*$ to users is gated by posterior width,

$$
\text{expose}(q^*) \iff \mathrm{tr}\,\boldsymbol{\Sigma}_{q^*} < \tau,
$$

so a deployment safety property follows. This applies uniformly to LLM-generated questionnaire items, KT exercises, and synthetic job descriptions, by Direction 1.3's item-agnostic encoding.

### 4.3 LLM as content delivery and explanation surface (line 4 and 7)

The policy from Direction 1.2 produces an action $a_t$ (target skill, item type, difficulty level). The LLM generates the actual content (hint, scaffolded prompt, worked example, item text) conditioned on $(a_t, h_t)$. The teacher-facing or counselor-facing explanation is computed by projecting the policy gradient $\nabla_{h_t} Q_\omega(h_t, a_t)$ onto the measurement components $(\boldsymbol{\theta}_t, \boldsymbol{a}, \boldsymbol{b})$, so the reason is expressed in trait and item-parameter units. Explanation faithfulness is enforced architecturally because the reason is derived from the model that produced the action, not from a separate LLM call.

### 4.4 Training the agent end to end

Joint loss,

$$
\mathcal{L}_{\text{agent}} = \mathcal{L}_{\text{meas}} + \lambda \mathcal{L}_{\text{DQN}} + \mu \mathcal{L}_{\text{rater}} + \nu \mathcal{L}_{\text{prior}}.
$$

Trained on synthetic data first (controlled rater bias, controlled item generation, known generating IRT parameters, known reward), then fine-tuned on logged interactions with the LLM rater and generator fixed at deployment.

---

## 5. Direction 3, World Model and Long-Horizon Trajectory

Both MA-GPCM and DRL-IRT are per-session. Educational and vocational decisions matter at the scale of months to years (degree completion, career transitions, lifelong learning). The per-session recommender of Direction 1.2 cannot plan over this horizon by construction. Direction 3 extends to a learner and career world model.

The setup,

$$
s_t = \mathrm{enc}(h_{1:t}), \qquad s_{t+1} = f_{\text{world}}(s_t, a_t) + \varepsilon_t, \qquad P(r_{t+1} \mid c_{q_{t+1}}, s_t) = g_\psi(\mathrm{dec}(s_t)),
$$

with MA-IRT as the observation function $g_\psi \circ \mathrm{dec}$. The policy plans by rolling out $f_{\text{world}}$,

$$
a_t = \arg\max_a \mathbb{E}\!\left[\sum_{h=0}^{H} \gamma^h R(s_{t+h}, a_{t+h}) \;\middle|\; s_t, a_t = a, \pi\right].
$$

DreamerV3 [47] is the methodological template. MedDreamer [48] adapts the same template to longitudinal clinical decision making, where the data shape (sparse irregular visits, drifting latent state, structured outcomes) matches the educational and vocational case closely.

Three sub-questions.

**Q1, identifiability of latent dynamics under intervention.** Adaptive recommendation correlates the policy's actions with the trajectory of $\boldsymbol{\theta}_t$. Disentangling intrinsic change (skill acquisition, interest shift, career maturation) from policy-induced selection requires structural assumptions on $f_{\text{world}}$ or exploration noise in the policy. Identifiability results for non-Markovian causal systems [45] and for sparse decoding [46] provide the tools. The investigation states which causal counterfactuals the world model supports.

**Q2, foundation representations for interaction.** Pretrain $f_\phi$ on large heterogeneous logs (KT and vocational interaction data jointly) and reuse it across downstream MA-IRT decoders. Time-series foundation models [31] supply the template. The investigation tests whether the pretrained representation transfers across platforms and item types with fewer responses needed for parameter recovery.

**Q3, lifelong measurement under drift.** Item parameters drift, populations shift, the calibration valid in January is not valid in July. Online conformal prediction [53] gives running coverage on the trait intervals. Bayesian online changepoint detection [55] flags when an item should be re-calibrated. The investigation builds a continual pipeline that updates MA-IRT online without forgetting past calibration.

Direction 3 is the most speculative. It is included because the operational target of an explainable educational and vocational AI system is long-horizon decision making, and the per-session recommender of Direction 1.2 is not sufficient for it.

---

## 6. Methodology

**Synthetic data.** Generators with known IRT parameters under stationary and non-stationary trait dynamics, the four already developed for MA-GPCM (static, block-shift, staircase, random walk) extended to multi-dim (Direction 1.1), to LLM-rated free text (Direction 2), and to intervention conditions with known causal structure (Direction 3). Job-feature generators from DRL-IRT (centroid mixture with Gaussian noise) extended to high-dim and to content-derived features.

**Real datasets.** ASSISTments 2009 and 2017 [57], EdNet [58], pyKT [59] for the KT side. RIASEC and Big Five public response data from OpenPsychometrics, as in DRL-IRT, for the vocational side. Logged interaction data from a deployment partner is sought for Direction 1.2 because off-policy evaluation requires a known behavior policy.

**Baselines.** KT side, DKT, DKVMN, SAKT, AKT, SimpleKT, SAINT+ [3, 4, 6, 7, 60]. IRT side, R `mirt` and Stan for batch calibration, Deep-IRT [61] and NeuralCD [18] for neural IRT. Recommender side, dot-product matching as in DRL-IRT, BanditCAT [28], conservative offline RL [8], the three DQN_Qn variants of DRL-IRT itself. World model side, DreamerV3 [47].

**Evaluation.** Recovery (Pearson, RMSE) on $(\boldsymbol{\theta}, \boldsymbol{a}, \boldsymbol{b})$ against generating values on synthetic data. Prediction (ACC, AUC, QWK) on real KT data. Prec@k and Recall@k on recommendation, as in DRL-IRT. Coverage of posterior intervals on held-out responses. For the recommender, off-policy estimated value with confidence sets respecting partial identification. For the world model, interventional validity on synthetic causal data.

---

## 7. Contributions

C1. **The MA-IRT family.** Multi-dim, item-agnostic, long-horizon generalization of MA-GPCM, with code, synthetic generators, and evaluation suite released.

C2. **Joint MA-IRT and deep RL recommender.** The core paper. Replace the classical IRT in DRL-IRT with MA-IRT, train the measurement head and policy head on a shared encoder, with partial-identification-aware off-policy evaluation and adaptive-testing-derived deployment constraints. Validated across KT and vocational domains under item-agnostic encoding.

C3. **LLM-influenced agentic system.** End-to-end trainable agent with MA-IRT perception, DRL action, LLM rater, LLM generator, LLM delivery, and architecturally faithful explanation. Identifiability conditions for separating user trait from content-correlated rater bias. Cold-start safety gate for LLM-generated items.

C4. **Learner and career world model.** MA-IRT as observation function in a world model supporting horizon-aware planning. Causal counterfactual claims under stated assumptions. Continual measurement pipeline for lifelong deployment.

The four contributions sit on top of the two prior manuscripts (MA-GPCM and DRL-IRT). Each is one paper if it lands cleanly. The thesis is the integrated artifact, the dynamic-deep-IRT plus DRL plus LLM plus world model architecture that neither prior manuscript delivers alone.

---

## References

[3] Piech, C., Spencer, J., Huang, J., Ganguli, S., Sahami, M., Guibas, L., and Sohl-Dickstein, J. Deep knowledge tracing. *Advances in Neural Information Processing Systems*, 2015. arXiv:1506.05908.

[4] Zhang, J., Shi, X., King, I., and Yeung, D.-Y. Dynamic key-value memory networks for knowledge tracing. *Proceedings of the 26th International Conference on World Wide Web*, 2017. arXiv:1611.08108.

[6] Ghosh, A., Heffernan, N., and Lan, A. S. Context-aware attentive knowledge tracing. *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2020. arXiv:2007.12324.

[7] Liu, Z., Liu, Q., Chen, J., Huang, S., Tang, J., and Luo, W. SimpleKT, a simple but tough-to-beat baseline for knowledge tracing. *International Conference on Learning Representations*, 2023. arXiv:2302.06881.

[8] Kumar, A., Zhou, A., Tucker, G., and Levine, S. Conservative Q-learning for offline reinforcement learning. *Advances in Neural Information Processing Systems*, 2020. arXiv:2006.04779.

[16] Reckase, M. D. *Multidimensional Item Response Theory.* Springer, 2009.

[18] Wang, F., Liu, Q., Chen, E., Huang, Z., Chen, Y., Yin, Y., Huang, Z., and Wang, S. Neural cognitive diagnosis for intelligent education systems. *Proceedings of the AAAI Conference on Artificial Intelligence*, 2020. arXiv:1908.08733.

[20] Chen, J., Chen, H., and Lin, Z. Identifiable cognitive diagnosis with sparse Bayesian multidimensional IRT. *arXiv*, 2023. arXiv:2310.17820.

[25] Linacre, J. M. *Many-Facet Rasch Measurement.* MESA Press, 1989.

[28] Sharpnack, J., Hao, K., Mulgrew, P., Garrard, C., Lash, M., Smith, B., and Tomkins, A. BanditCAT and AutoIRT, machine learning approaches to computerized adaptive testing and item calibration. *arXiv*, 2024. arXiv:2410.21033.

[30] Wang, X., Berger, J. O., and Burdick, D. S. Continuous-time longitudinal item response theory models. *arXiv*, 2021. arXiv:2109.13064.

[31] Das, A., Kong, W., Sen, R., and Zhou, Y. A decoder-only foundation model for time-series forecasting. *International Conference on Machine Learning*, 2024.

[37] Gu, J., Jiang, X., Shi, Z., et al. A survey on LLM-as-a-judge. *arXiv*, 2024. arXiv:2411.15594.

[38] Zhao, W. X., et al. LLMs are biased teachers, evaluating LLM bias in personalized education. *arXiv*, 2024. arXiv:2410.14012.

[39] Singhal, P., et al. A long way to go, investigating length correlations in LLM-based reward models. *arXiv*, 2024. arXiv:2407.01085.

[45] Yao, W., et al. Disentangled representation learning in non-Markovian causal systems. *Advances in Neural Information Processing Systems*, 2024.

[46] Moran, G. E., Sridhar, D., Wang, Y., and Blei, D. M. Identifiable deep generative models via sparse decoding. *arXiv*, 2021. arXiv:2110.10804.

[47] Hafner, D., Pasukonis, J., Ba, J., and Lillicrap, T. Mastering diverse domains through world models (DreamerV3). *arXiv*, 2023. arXiv:2301.04104.

[48] Liu, M., et al. MedDreamer, model-based reinforcement learning with latent imagination for personalized clinical treatment. *arXiv*, 2025. arXiv:2505.19785.

[49] Su, Y., Dimakopoulou, M., Krishnamurthy, A., and Dudik, M. Doubly robust off-policy evaluation with shrinkage. *International Conference on Machine Learning*, 2020. arXiv:1907.09623.

[50] Bian, Z., and Shi, C. Doubly inhomogeneous reinforcement learning. *Journal of the American Statistical Association*, 2024.

[51] Skalse, J., Howe, N. H. R., Krasheninnikov, D., and Krueger, D. Invariance in policy optimisation and partial identifiability in reward learning. *International Conference on Machine Learning*, 2023. arXiv:2203.07475.

[53] Angelopoulos, A. N., Candes, E. J., and Tibshirani, R. J. Online conformal prediction with decaying step sizes. *International Conference on Machine Learning*, 2024. arXiv:2402.01139.

[55] Adams, R. P., and MacKay, D. J. C. Bayesian online changepoint detection. *arXiv*, 2007. arXiv:0710.3742.

[57] Feng, M., Heffernan, N., and Koedinger, K. Addressing the assessment challenge with an online system that tutors as it assesses. *User Modeling and User-Adapted Interaction*, 19(3), 243 to 266, 2009.

[58] Choi, Y., et al. EdNet, a large-scale hierarchical dataset in education. *International Conference on Artificial Intelligence in Education*, 2020.

[59] Liu, Z., et al. pyKT, a python library to benchmark deep learning based knowledge tracing models. *Advances in Neural Information Processing Systems*, 2022. arXiv:2206.11460.

[60] Shin, D., et al. SAINT+, integrating temporal features for EdNet correctness prediction. *Proceedings of the 11th International Learning Analytics and Knowledge Conference*, 2021. arXiv:2010.12042.

[61] Yeung, C.-K. Deep-IRT, make deep learning based knowledge tracing explainable using item response theory. *Proceedings of the 12th International Conference on Educational Data Mining*, 2019. arXiv:1904.11738.

[62] Zhuang, Y., Liu, Q., Ning, Y., Huang, Z., Lin, R., Chen, E., Wu, J., and Wang, S. Bounded ability estimation for computerized adaptive testing. *Advances in Neural Information Processing Systems*, 2024.
