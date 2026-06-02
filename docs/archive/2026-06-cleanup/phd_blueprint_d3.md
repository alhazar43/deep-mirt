# Direction 3: Streaming Measurement (depth-first, free exploration)

## Why streaming measurement is the right organizing principle

The candidate's existing prototype, MA-GPCM, is not really a knowledge tracing paper. It is a measurement paper that happens to use a knowledge tracing backbone. Its signature is that every quantity an educational system normally computes offline (the student's ability, the items' discrimination and difficulty parameters, the ordinal response distribution) is produced as the output of a single forward pass over an incoming sequence. The encoder is the dynamics, the decoder is a calibrated measurement model, and the two are trained jointly so that interpretation is a property of the forward pass rather than a separate post-hoc estimation step. The candidate has stated, in the MA-GPCM paper, that the design pattern is to place an IRT decoder in front of a sequential encoder, so the same architecture absorbs new response formats by swapping the head. That sentence is the seed of an entire PhD program.

Direction 1 and Direction 2 each pick a vertical slice. D1 hard codes three threads with deep reinforcement learning in the middle and ends up with two largely independent research lines (measurement and policy) connected by a fairly thin interface. D2 drops DRL and reorganizes around encoder-decoder measurement, which is coherent but conservative: it generalizes MA-GPCM along the response-format axis and a few neighboring axes, and produces stronger psychometric papers, but it leaves the deployment story untouched. Neither D1 nor D2 commits to the claim that this candidate is actually making, which is the claim that classical educational AI has the wrong computational primitive. Classical educational AI computes offline batches and pushes them through a deployment pipeline. The candidate's stance is that the primitive should be a streaming forward pass.

Direction 3 takes that stance literally and asks what falls out of it across the entire educational measurement stack. The reframing is not cosmetic. If the primitive is streaming inference, then every classical pipeline step has to be re-derived in streaming form, often using a different statistical object. Item calibration becomes amortized item embedding plus online recalibration. Concept drift becomes a first-class output of the inference network, not a separate monitor. Person fit becomes a per-step likelihood ratio inside the same network. Differential item functioning becomes an online invariance audit running on the same stream. Intervention selection becomes a contextual bandit on top of a streaming, calibrated belief state, not a deep reinforcement learning loop that needs millions of trajectories. Free-response grading becomes a streaming LLM grader whose calibration is monitored online by the same invariance audit machinery. Open-world item arrival becomes a structured cold-start problem solved by a content-conditioned amortized item embedding network. None of these reframings are arbitrary. They follow from the choice to treat the forward pass as the unit of inference.

This is also a methodological commitment, not just an engineering one. Streaming inference forces the program to confront identifiability and recovery on the fly, calibration drift, anytime-valid testing, and online regret in a way that batch psychometrics never has to. It puts the candidate in conversation with neural posterior estimation (Cranmer et al., 2020; Lueckmann et al., 2021), prior-data-fitted networks (Mueller et al., 2022; Hollmann et al., 2023), online conformal prediction (Gibbs and Candes, 2021; Angelopoulos et al., 2024), drift-resilient in-context learning (Helli et al., 2024), and anytime-valid e-process testing (Ramdas et al., 2022; Pérez-Ortiz et al., 2024). None of these communities currently talk to the educational measurement community in a sustained way. A PhD program that bridges them, with MA-GPCM as the kernel, occupies a defensible and largely empty research area.

What does streaming framing buy that the other two cannot. Three things, concretely. First, it produces formal claims that have no equivalent in batch psychometrics, in particular online identifiability conditions, streaming calibration guarantees under bounded drift, and anytime-valid invariance tests. Second, it produces deployment-relevant artifacts (single-pass models that arrive at sound estimates within a session, that flag drift, that audit themselves) which are the artifacts that real platforms actually need. Third, it lets the candidate accumulate a portable design pattern across thrusts, so each paper makes the others stronger, rather than each paper being a self-contained contribution. The unit of evaluation across the program is the same: did the streaming forward pass produce a calibrated, identifiable, drift-aware measurement at time t, using only the data observed up to time t.

## Literature scan

The literature below is grouped by topic, with each entry verified during this round via WebSearch. Author-year citations in the rest of the document refer back to this list.

**Knowledge tracing core.** Piech et al., 2015 (Deep Knowledge Tracing) introduced LSTM-based KT. Yeung and Yeung, 2018 documented two key DKT failures (input reconstruction and waviness) and proposed prediction-consistent regularization. Zhang et al., 2017 introduced DKVMN with key-value memory. Yeung, 2019 introduced Deep-IRT, the closest predecessor to MA-GPCM. Liu et al., 2022 (pyKT) standardized benchmarking. Liu et al., 2023 (SimpleKT) showed that a properly tuned simple model is hard to beat. Cui et al., 2024 (GRKT) added cognitive-psychology-grounded retrieval and forgetting on top of GNN backbones. Liu et al., 2024 (Temporal Graph Memory Networks) introduced explicit decay over a graph memory.

**Knowledge tracing with LLMs and process data.** Scarlatos and Lan, 2024 (Dialogue KT) studied KT inside tutor-student dialogues with LLMs. Sonkar et al., 2024 (CLST) used a generative LM as a student tracer for cold start. Hicke et al., 2025 (Next Token Knowledge Tracing) reframed KT as next-token prediction with pretrained LLMs and showed strong cold-start generalization. Tang et al., 2022 (Process-BERT) used masked-token pretraining on educational process traces.

**IRT, neural psychometrics, identifiability.** Wu et al., 2020 (VIBO) introduced variational item response theory. Veldkamp et al., 2024 extended VAE-IRT to handle missing data. Liu et al., 2022 introduced importance-weighted VAEs for MIRT. Curi et al., 2019 introduced an early neural IRT estimator. Paassen et al., 2022 (Sparse Factor Autoencoders) addressed sparsity in IRT autoencoders. Wallmark et al., 2023 (Flexible Monotone IRT) used monotone neural nets for IRFs. Runje and Shankaranarayana, 2023 introduced constrained monotonic neural networks; the ICLR 2024 work on scalable monotone networks (Igel et al., 2024) extends this. Hosseinzadeh and Matlock Cole, 2024 studied MIRT recovery under cross-loading misspecification. Sun et al., 2023 introduced sparse Bayesian MIRT.

**Cognitive diagnosis and Q-matrix learning.** Xu et al., 2024 (QNN) used Q-matrix constraints to determine neural net connectivity. Liu et al., 2024 surveyed cognitive diagnosis models. The ICDM family proposed encoder-decoder cognitive diagnosis with student-centered graphs (cited inside the same survey).

**Amortized inference, simulation-based inference, neural processes.** Cranmer et al., 2020 (the foundational SBI survey). Lueckmann et al., 2021 benchmarked SNPE. Sharrock et al., 2024 (Sequential Neural Score Estimation) introduced score-based SBI. Gloeckler et al., 2024 (Active SNPE) added active learning to SBI. Garnelo et al., 2018 (Neural Processes). Foong et al., 2020 (Convolutional Conditional Neural Processes). Margossian and Blei, 2023 surveyed amortized inference. Müller et al., 2022 (Prior-Data Fitted Networks). Hollmann et al., 2023 (TabPFN). Helli et al., 2024 (Drift-Resilient TabPFN, NeurIPS 2024). Zaheer et al., 2017 (Deep Sets) underlies set-based amortized inference.

**Online and conformal calibration.** Gibbs and Candes, 2021 (Adaptive Conformal Inference). Angelopoulos et al., 2024 (Online Conformal Prediction with Decaying Step Sizes). Bhatnagar et al., 2024 (Online Conformal for Time Series). Deshpande et al., 2024 (Online Calibrated and Conformal Prediction). Bonkhoff et al., 2024 (Feature Fitted Online Conformal). Tibshirani et al., 2019 introduced conformal under covariate shift, which is a relevant batch predecessor.

**Sequential and anytime-valid testing.** Adams and MacKay, 2007 (Bayesian Online Change Point Detection). Ramdas et al., 2022 (e-processes survey). Pérez-Ortiz et al., 2024 (Anytime Validity is Free). Hammoud et al., 2024 (anytime-valid FDR with stopped e-BH). Ramdas et al., 2024 (post-hoc and anytime valid permutation tests).

**Drift detection.** Bifet and Gavalda, 2007 (ADWIN). Page, 1954 / Hinkley, 1971 (Page-Hinkley). Raab et al., 2020 (KSWIN). Cerqueira et al., 2022 (STUDD). Hinder et al., 2024 (concept drift monitoring survey). Lee et al., 2025 (Concept Drift Detection for Knowledge Tracing, EDM doctoral consortium). Lee et al., 2025 (arxiv 2511.00704) studied KT model robustness under student concept drift.

**Off-policy evaluation and contextual bandits.** Dudik et al., 2014 (Doubly Robust OPE). Su et al., 2020 (Switch-DR). Nie et al., 2024 (Doubly Inhomogeneous OPE, JASA). Chitnis et al., 2024 (CANDOR, counterfactual-annotated DR). Bhattacharya et al., 2024 (OPERA, automatic OPE re-weighting). Singla and Cong, 2024 (HOPE, human-centric OPE for e-learning). Liu and Wang, 2024 (off-policy evaluation under bandits with predicted context). Russo and Van Roy, 2018 / Agrawal and Goyal, 2013 (Thompson sampling baselines). Wang et al., 2021 (FairX-LinTS, fairness of exposure under TS).

**Differential item functioning and fairness.** Wang and Zhu, 2024 (DIF for continuous-response CAT). Kraus et al., 2024 (interpretable ML for DIF). Huang and Ishii, 2024 (multi-detector DIF). Suk and Lyu, 2024 (single-world intervention graphs for item fairness). Zhang and Lan, 2024 (reducing DIF via process data).

**Person fit and aberrant response detection.** The 2024 MLP-F line introduced neural-network-based machine learning person fit for cognitive diagnosis. Sinharay, 2020 surveyed log-normal fit for response times. Lee and von Davier, 2024 studied person fit corrections. The Cross Estimation Network (Wang et al., 2024) jointly estimates persons and items in a paired neural setup.

**LLM judges and automated grading.** Zheng et al., 2024 (LLM-as-Judge survey). Wang et al., 2024 (position bias in pairwise vs pointwise judges). Zhao et al., 2024 (Grade Like a Human). Saha et al., 2026 (Rubric-Conditioned LLM Grading). Liu et al., 2024 (rubric grounded RL). Kovac et al., 2024 (LLMs are biased teachers). Zhou et al., 2024 (length bias in LLM preference judges).

**Item generation and synthetic students.** Liu et al., 2024 (LLM respondents for item evaluation). Omopekunola and Kardanova, 2024 (LLM-generated physics items). Park and Lee, 2025 (multi-agent AIG framework). Laverghetta and Licato, 2024 (LLMs as psychometrically plausible respondents).

**Long-context and streaming architectures.** Gu and Dao, 2024 (Mamba). Munkhdalai et al., 2024 (Infini-Attention). Xiao et al., 2024 (StreamingLLM). Beck et al., 2024 (xLSTM).

**Computerized adaptive testing.** Wang and Liu, 2024 (NeurIPS, Collaborative CAT). Zhuang et al., 2022 (NCAT, RL-based CAT). Anderson et al., 2024 (BanditCAT and AutoIRT).

**Uncertainty in KT.** Mao et al., 2024 (Dynamic LENS, uncertainty-preserving DKT with state-space models). Liu and Liu, 2025 (Uncertainty-Aware KT survey).

This is the core reading list against which the thrusts below are constructed.

## Thrust 1. Streaming neural psychometrics

The first thrust generalizes MA-GPCM along the inference axis. The encoder-decoder pair already produces theta, alpha, and beta in a single forward pass on synthetic data and on proxy ordinal ASSISTments. The thrust pushes this in three directions, treating drift as a first-class output, allowing the encoder to be a foundation model, and putting formal streaming guarantees on the resulting estimates.

The signature problem of this thrust is identifiability online. In batch IRT we know exactly which rotations and scalings are unidentified, and we impose constraints to break them. In streaming IRT, the data arrives ordered by time and by student, and the trait scale is implicitly anchored by whichever items show up first. The candidate already knows that MA-GPCM tends to collapse the theta scale early and re-expand it later when more diverse items appear. That is an online identifiability artifact, and it has no name in the literature. This thrust proposes naming it, formalizing the streaming-identifiability conditions under which it cannot happen, and designing encoders that respect those conditions.

### Sub-paper 1.1. Streaming identifiability and recovery diagnostics for neural IRT

Research question. Given a streaming neural IRT model that consumes responses (q_t, r_t) and emits (theta_t, alpha_q, beta_q) at each step, what conditions on the input stream are sufficient to ensure that the streaming estimates converge to a fixed point that recovers the true IRT parameters up to a known equivalence class. State and prove a streaming identifiability theorem analogous to the batch identifiability conditions for the 2PL and GPCM, but expressed in terms of stream statistics (item coverage, ability coverage, monotonicity of the cumulative information matrix).

Method. Build on MA-GPCM. Add a recovery diagnostics module that, at every step, computes (i) the cumulative Fisher information matrix on the items observed so far, (ii) the current effective sample size per item, and (iii) an online Procrustes alignment of the running theta estimate to a reference scale anchored by a pre-specified set of anchor items. Train using the same combined ordinal loss but add an identifiability regularizer that penalizes degenerate rotations once cumulative information crosses a threshold.

Baselines. Static MIRT-EM via the R mirt package (already part of the candidate's pipeline). Variational IRT (Wu et al., 2020, VIBO). MA-GPCM without the identifiability regularizer. Deep-IRT (Yeung, 2019).

Datasets. Synthetic static and dynamic GPCM data (the candidate's existing data generators). ASSISTments 2009 and 2017. EdNet KT1 and KT2 (Choi et al., 2020). XES3G5M (Liu et al., 2023).

Evaluation. Item parameter recovery (Pearson correlation between estimated and true alpha, beta) as a function of stream length. Ability recovery (within-student correlation of theta_t with true theta_t). A novel streaming-identifiability gap, defined as the discrepancy between the estimate produced at time t and the estimate that would be produced if all data up to time t were processed in a batch under the same model. Calibration of K-1 cumulative logits via reliability diagrams.

Open theoretical claim. Under bounded item-coverage rates and a Lipschitz encoder, the streaming estimator converges in mean-square to the batch maximum-likelihood estimator at a rate O(1/sqrt(t)), and the streaming-identifiability gap closes at the same rate. The proof technique adapts standard stochastic approximation results (Robbins-Monro, Kushner-Yin) under the additional constraint that the encoder is a fixed-parameter function, not a parameter being updated. This is novel because the identifiability literature treats parameters as estimable; streaming neural IRT treats the encoder as fixed and the inference as a function of the stream, which changes the proof.

Hard problem. The encoder is trained on a meta-distribution of student trajectories, but at test time it is run on a single trajectory. Convergence to the batch MLE on that single trajectory is a property of the encoder, not of an estimator, and it requires the encoder to behave like a sufficient statistic for the IRT likelihood on the meta-distribution. Establishing this requires bridging the SBI literature on amortized posteriors (Lueckmann et al., 2021) with the streaming stochastic-approximation literature.

Connection to MA-GPCM. This is the most direct generalization. MA-GPCM is one architecture in the streaming-IRT family. The thrust treats it as a member of that family and asks what guarantees we can attach to the family.

### Sub-paper 1.2. Drift as a first-class output, not a monitor

Research question. Can a streaming neural IRT model emit, at each step, a calibrated posterior over both the current latent trait theta_t and a drift indicator d_t in {no change, gradual drift, abrupt change}, such that the drift indicator agrees with a Bayesian Online Change Point Detector (Adams and MacKay, 2007) applied to the true theta in the data-generating process.

Method. Augment the MA-GPCM decoder with a parallel drift head that emits a categorical distribution over drift types, trained on synthetic streams generated by the candidate's existing data generators (static, block, staircase, random walk). The drift head uses the same DKVMN read vector as the trait head but conditions on a longer effective context via a Mamba (Gu and Dao, 2024) or xLSTM (Beck et al., 2024) backbone. Training uses a weighted combination of the ordinal loss and a categorical cross-entropy on simulated drift labels.

Baselines. ADWIN (Bifet and Gavalda, 2007), KSWIN, Page-Hinkley applied to the running theta. Drift-Resilient TabPFN (Helli et al., 2024) repurposed for KT by treating each (question, response) pair as a tabular instance. The arxiv 2511.00704 KT-drift robustness study, which uses fixed KT models without an explicit drift head.

Datasets. Synthetic dynamic GPCM (the candidate's block, staircase, random walk generators). ASSISTments multi-year (the 2024 study used five academic years; the same data slice can be used to validate drift detection on a natural-history stream).

Evaluation. F1 of drift event detection at varying delay budgets (early vs late detection). Calibration of the drift posterior under simulated drift. Theta recovery RMSE conditional on drift type. False alarm rate on stationary streams.

Open theoretical claim. The drift head, when trained on a prior over drift dynamics that matches the test distribution, is equivalent to an amortized Bayesian online change point detector in the sense of Müller et al., 2022, with detection delay bounded by a function of the prior's expected change magnitude. This is a streaming analogue of the TabPFN result that in-context predictions approximate posterior predictives.

Hard problem. The drift label is unobserved at test time, and the synthetic drift generators may not match real student drift dynamics. The thrust requires either an extensive prior calibration study or a self-supervised drift surrogate constructed from response-time discontinuities and accuracy slopes.

Connection to MA-GPCM. The thrust makes drift detection a co-output of the same forward pass that produces theta. The encoder-decoder pattern is preserved; only the decoder branches.

### Sub-paper 1.3. Foundation encoders for streaming student modeling

Research question. Does pretraining a single encoder on a large meta-distribution of synthetic and real student trajectories, then attaching a MA-GPCM decoder, produce better streaming IRT recovery than per-platform training, particularly at the start of a new stream when context is short.

Method. Build a prior over student trajectories that combines (i) the candidate's GPCM data generator with sampled parameter distributions, (ii) ASSISTments-style ordinal sequences with sampled hyperparameters, and (iii) EdNet-style multi-skill long sequences. Pretrain a Mamba or transformer encoder via masked-token and next-token objectives on this meta-distribution. Freeze the encoder, train only the MA-GPCM decoder on each target task.

Baselines. Per-task MA-GPCM (the candidate's existing model). Next Token Knowledge Tracing (Hicke et al., 2025), which is the closest LLM-based predecessor. CLST (Sonkar et al., 2024). Drift-Resilient TabPFN with a synthetic KT prior.

Datasets. ASSISTments 2009, 2012, 2017. EdNet KT1, KT2. XES3G5M. The candidate's synthetic data as held-out probes for parameter recovery.

Evaluation. AUC and QWK on prediction tasks. Streaming theta recovery on synthetic probes (the synthetic data lets us measure recovery directly, which is impossible on real data). Few-shot transfer: AUC at session lengths 5, 10, 20, 50.

Open theoretical claim. Under the prior-data-fitted-network framework (Müller et al., 2022), a sufficiently expressive transformer pretrained on the KT meta-distribution approximates the Bayesian posterior over (theta_t, alpha, beta) given the observed stream, and therefore achieves Bayes-optimal recovery up to encoder capacity. This is novel as an educational measurement statement.

Hard problem. The KT meta-distribution is heterogeneous; ASSISTments items are not EdNet items, and a single encoder must handle both. The thrust requires designing an item-vocabulary-free encoding (probably text embeddings of item content combined with response-only tokens) so that the same encoder generalizes across item banks.

Connection to MA-GPCM. The decoder is preserved verbatim; only the encoder is replaced by a foundation model. The encoder-decoder design pattern is exactly the value here, because it lets the encoder change without re-deriving the measurement model.

### Sub-paper 1.4. Online calibration of cumulative logits under nonstationarity

Research question. The MA-GPCM K-1 cumulative logits are calibrated in-distribution, but on streams with drift the calibration degrades. Can online conformal prediction (Gibbs and Candes, 2021; Angelopoulos et al., 2024) be adapted to GPCM cumulative logits to maintain coverage of the predicted ordinal category, anytime, under bounded drift.

Method. Wrap the MA-GPCM decoder with an online conformal layer that maintains, at each step, a per-threshold conformity score and an adaptive quantile. The conformal prediction set is the smallest contiguous interval of ordinal categories that covers the true category with probability 1-alpha. The adaptive step size follows Angelopoulos et al., 2024 (decaying step sizes).

Baselines. Static conformal prediction (Tibshirani et al., 2019, with covariate shift correction). Vanilla MA-GPCM softmax probabilities. Dynamic LENS (Mao et al., 2024), which is a state-space DKT with uncertainty.

Datasets. Same as 1.1 plus a deliberately injected drift dataset (the candidate's block change generator with a large abrupt shift halfway through the stream).

Evaluation. Coverage and average set size at confidence levels 0.8, 0.9, 0.95, measured per time step and aggregated by drift type. Long-run miscoverage rate, which is the quantity that Gibbs and Candes guarantee. Adaptive coverage in the sense of conditional coverage given the drift indicator from sub-paper 1.2.

Open theoretical claim. Under bounded drift in the IRT data-generating process (formalized as a Wasserstein bound on the response distribution shift per step), the online conformal layer maintains 1-alpha long-run coverage with regret O(sqrt(T)). The proof is a direct adaptation of Angelopoulos et al., 2024 to ordinal targets with monotone cumulative logits.

Hard problem. Ordinal targets require coverage of a contiguous interval, not of a point estimate, and the conformity score must respect monotonicity in K. The thrust requires a new conformity score that is monotone in the cumulative logit gap.

Connection to MA-GPCM. The conformal layer is a strict add-on to the existing MA-GPCM decoder. It does not modify the forward pass; it consumes the forward pass and emits an additional, statistically-guaranteed output.

## Thrust 2. Open-world streaming measurement

Real platforms see a continuous arrival of new items (autogenerated quizzes, instructor-added problems, LLM-generated content) and new learners. Classical IRT requires that items have been calibrated. Classical KT models require that questions appear in the training vocabulary. Neither is true online. This thrust solves the open-world problem inside the streaming forward pass.

The signature problem is cold-start without recalibration. The candidate's existing learned item embeddings break when a new item arrives. The thrust replaces them with a content-conditioned amortized item embedding network that produces the IRT parameters of a new item from its content, with the same encoder-decoder pattern.

### Sub-paper 2.1. Content-conditioned amortized item parameters for cold-start IRT

Research question. Given an item content embedding c_q (text, LaTeX, image features), can a feed-forward network produce a prior over (alpha_q, beta_q) that, when combined with a small number of student responses, yields a posterior approximating the result of running full IRT calibration on a large student sample.

Method. Train an item-embedding network on the candidate's synthetic GPCM generator with item content sampled from a paired text prior (curriculum-aligned templates). The network outputs a mean and covariance over (alpha_q, beta_q). At test time, on a new item, the prior is updated by streaming Bayes using the responses observed so far. The update is itself amortized through a recurrent layer (Mamba block) so that inference remains single-pass.

Baselines. EM-based item calibration on small samples (R mirt). AutoIRT (Anderson et al., 2024). Sparse Factor Autoencoders (Paassen et al., 2022). The CLST cold-start approach (Sonkar et al., 2024). Content-based initialization for sequential recommendations (Pliakos et al., 2024).

Datasets. Synthetic data with curriculum-templated item content. ASSISTments 2017 with item text. The Next Token KT cold-start splits (Hicke et al., 2025), since they explicitly evaluate cold start.

Evaluation. Alpha and beta recovery RMSE as a function of the number of responses observed for the new item. Out-of-distribution detection (how well the network's predicted variance flags items whose content is dissimilar from the training distribution). Predictive AUC on responses to the new item.

Open theoretical claim. Under a content prior with Lipschitz mean and bounded covariance, the amortized posterior on (alpha_q, beta_q) achieves the Bayesian convergence rate O(1/sqrt(n)) for n responses, matching batch Bayes up to a constant that depends on the content prior's Lipschitz constant. This is a streaming analogue of standard amortized inference results (Cranmer et al., 2020).

Hard problem. Item content is heterogeneous (text vs LaTeX vs images vs diagrams) and the content embedding must be robust across these modalities. The thrust requires a multimodal content encoder trained jointly with the amortized item-parameter network.

Connection to MA-GPCM. The item-embedding lookup table in MA-GPCM is replaced by a content-conditioned function. The decoder is unchanged. This is the encoder-decoder pattern applied to the item axis instead of the student axis.

### Sub-paper 2.2. Learner cold-start via meta-learned prior over theta trajectories

Research question. New learners arrive without history. Can a meta-learned prior over theta_0 produce a useful initial estimate that converges to the per-student estimate faster than starting from a population mean.

Method. Train an amortized network that, given any auxiliary information about a new learner (course enrollment, demographic features when ethically permitted, item-content interactions in the first few responses), produces a prior over theta_0. The prior is then updated by streaming inference as responses arrive. The network is trained on the candidate's synthetic data with simulated auxiliary features.

Baselines. Population-mean initialization. The CLST generative initialization (Sonkar et al., 2024). Drift-Resilient TabPFN (Helli et al., 2024) used in the cold-start direction. A vanilla MA-GPCM without learner cold-start.

Datasets. ASSISTments multi-year, since student turnover provides natural cold starts. EdNet KT1.

Evaluation. RMSE of theta_t over the first 20 responses (early-stream convergence). Predictive AUC at session lengths 1, 5, 10. Calibration of the cold-start prior.

Open theoretical claim. The meta-learned prior achieves lower expected regret on the first k responses than any prior that does not condition on auxiliary information, for k below a problem-dependent threshold. After that threshold, both priors converge. This is a meta-learning regret bound on the cold-start window.

Hard problem. Auxiliary features carry fairness risk. The thrust requires an explicit fairness analysis showing that the cold-start prior does not introduce DIF (which sub-paper 4.2 will then audit online).

Connection to MA-GPCM. The student-side amortization mirror of 2.1. Together, they make MA-GPCM open-world along both the student and the item axes.

### Sub-paper 2.3. Streaming bank expansion and concept-drifting curricula

Research question. As new items are added to the bank over time, the item difficulty distribution drifts. Can a streaming forward pass maintain a calibrated, consistent ability scale across an evolving bank, without recalibrating the entire bank.

Method. Combine 2.1 with an anchoring procedure that uses a small set of pinned items as a stable reference scale. The anchoring is implemented as a regularizer on the streaming theta estimator. The anchor set is updated by an online procedure that detects items whose IRT parameters have drifted (using a per-item likelihood-ratio test against the amortized prior).

Baselines. Periodic full bank recalibration via mirt. Vertical scaling methods from classical psychometrics (Kolen and Brennan, 2014). MA-GPCM trained on the original bank only.

Datasets. ASSISTments 2009 + 2012 + 2015 + 2017 treated as a chronological stream with overlapping but expanding item banks.

Evaluation. Cross-year consistency of theta on common students (when available). Anchor-item parameter stability over the stream. AUC on next-year predictions.

Open theoretical claim. Under a bounded fraction of drifting items and a Lipschitz amortized item-prior, the streaming theta scale is identifiable across the entire timeline and matches the batch result up to an O(1/sqrt(t)) drift.

Hard problem. The anchor set is a partial solution; in real platforms, even anchors drift. The thrust requires a self-anchoring scheme that detects when anchor items themselves drift.

Connection to MA-GPCM. This thrust treats MA-GPCM as the persistent measurement device that survives bank changes. The encoder-decoder pattern is the artifact that persists.

## Thrust 3. Streaming intervention on calibrated state

The candidate also has DRL background. The streaming framing does not need to discard it, but it reframes intervention as a lightweight contextual decision on a calibrated state rather than as a deep RL loop. This is in line with the candidate's stated philosophy of single-pass deployment and is also the strictly correct statistical move, because deep RL in education is notoriously sample-inefficient and exploration-unsafe, while contextual bandits on a calibrated belief state are sample-efficient and have closed-form regret bounds.

The signature problem is that the intervention policy must operate on a measurement object that is itself being updated online. The policy state is not the raw response history; it is the streaming posterior over theta and the streaming reliability of that posterior.

### Sub-paper 3.1. Calibrated-state contextual bandits for next-item selection

Research question. Given a streaming posterior over theta_t produced by MA-GPCM, what is the regret of a Thompson-sampling contextual bandit that selects the next item to maximize expected information gain about theta, compared to a deep RL CAT policy (NCAT, Zhuang et al., 2022).

Method. The bandit state is (mean of theta_t, variance of theta_t, the running streaming-identifiability gap from sub-paper 1.1). Actions are items. Reward is the expected reduction in posterior variance, computed in closed form from the GPCM Fisher information. Thompson sampling samples theta from the posterior and selects the item with maximum Fisher information at the sampled theta.

Baselines. NCAT (Zhuang et al., 2022). CCAT (Wang and Liu, 2024). BanditCAT (Anderson et al., 2024). Maximum Fisher information CAT with the batch IRT calibration. Random item selection (as a sanity check).

Datasets. ASSISTments 2017 and synthetic GPCM data (the synthetic data allows true theta recovery to be the evaluation metric).

Evaluation. RMSE of final theta estimate at varying session lengths. Regret in the bandit sense, measured against the optimal item-selection policy on the synthetic data. Diversity of selected items.

Open theoretical claim. The Thompson bandit on the streaming posterior achieves regret O(sqrt(T) log K) for K items, matching standard contextual TS bounds (Russo and Van Roy, 2018), with the additional property that the calibrated-state version is well-defined even when MA-GPCM's posterior is not exactly Bayesian, provided the posterior is calibrated in the conformal sense.

Hard problem. The Fisher information depends on alpha and beta, which are also being estimated. Thompson sampling over the joint posterior is more expensive. The thrust requires a tractable approximation.

Connection to MA-GPCM. The bandit is a thin wrapper on top of MA-GPCM. It uses the existing posterior; it does not modify the forward pass.

### Sub-paper 3.2. Off-policy evaluation for streaming interventions

Research question. Given logged tutoring data from a deployed system, can we estimate, anytime, the value of a candidate streaming-intervention policy without deploying it.

Method. Build a doubly-robust OPE estimator on top of MA-GPCM that uses the streaming posterior as the propensity-adjusted state. Extend the OPERA framework (Bhattacharya et al., 2024) and HOPE (Singla and Cong, 2024) by replacing their fixed propensity model with the streaming MA-GPCM posterior. Use an anytime-valid e-process (Ramdas et al., 2022) to deliver a confidence sequence on the policy value that is valid under optional stopping.

Baselines. OPERA. HOPE. Standard doubly-robust OPE (Dudik et al., 2014). Direct method on a fitted MA-GPCM reward model.

Datasets. ASSISTments problem-skill builder logs. The CANDOR (Chitnis et al., 2024) benchmark dataset, since it has counterfactual annotations.

Evaluation. Estimator variance and bias on simulated streams (where ground truth is available). Confidence-sequence coverage. Conservative estimation under non-overlap (when the logged policy puts low probability on the candidate policy's actions).

Open theoretical claim. Under bounded drift in the streaming posterior and standard overlap, the doubly-robust streaming OPE estimator is consistent at the parametric rate, and the anytime-valid confidence sequence has coverage 1-alpha at every time t.

Hard problem. Real tutoring data has limited overlap (the deployed policy is greedy on a fitted MA-GPCM, leaving most counterfactual actions unexplored). The thrust requires either a deliberate exploration phase or a robust OPE estimator that degrades gracefully under near-overlap.

Connection to MA-GPCM. OPE is a deployment-grade application of MA-GPCM. The encoder-decoder pattern provides the propensity model; the OPE wrapper is again additive.

### Sub-paper 3.3. Exposure-fair streaming item selection

Research question. The streaming intervention policy of 3.1 may concentrate exposure on a small set of items. Can exposure-fairness constraints be enforced anytime, in the streaming forward pass, without sacrificing the regret bound by more than a known factor.

Method. Wrap the Thompson bandit of 3.1 with an exposure-control layer based on exADMM (Sato et al., 2024) and FairX-LinTS (Wang et al., 2021). The constraint is that each item's cumulative exposure remains within a per-item budget. The exposure constraint is enforced via a Lagrangian update that runs online.

Baselines. Unconstrained TS bandit. Random item selection. exADMM applied to a fixed IRT model.

Datasets. Same as 3.1 plus a fairness probe set with intentionally rare items.

Evaluation. Regret. Exposure inequality (Gini). Per-item recovery RMSE on rare items (the test of whether exposure fairness helps measurement at the bank tail).

Open theoretical claim. The Lagrangian-controlled TS bandit achieves regret O(sqrt(T) log K) plus a constraint-violation term that decays as O(1/sqrt(T)). This is a standard constrained-bandit result instantiated on a streaming measurement state.

Hard problem. Real exposure budgets are not known in advance; they have to be inferred from instructor preferences or platform constraints. The thrust requires an interactive specification protocol.

Connection to MA-GPCM. Exposure fairness is an audit on top of the streaming measurement. It does not modify the measurement.

## Thrust 4. Streaming AI-mediated assessment

LLM-graded responses, LLM-generated items, and LLM-simulated students are now common, and they all introduce streaming nonstationarity. A grader's bias drifts over model updates. A generated item's effective difficulty drifts as the LLM is fine-tuned. A simulated student's behavior drifts as the underlying model is updated. The thrust treats every LLM-mediated component as a stream and audits it inside the same MA-GPCM forward pass.

The signature problem is that LLM graders introduce a measurement-equivalent of differential item functioning, but on the grader rather than on the human group. The thrust formalizes this as grader-DIF and provides an online audit.

### Sub-paper 4.1. Streaming LLM grader with online calibration

Research question. Given an LLM grader producing partial-credit scores on short-answer or essay responses, can a streaming calibration layer maintain agreement with human graders anytime, under LLM updates and prompt drift.

Method. Wrap the LLM grader with a Platt-style streaming calibrator that maps raw LLM scores to calibrated ordinal categories using a small, continuously updated set of human-graded anchor responses. The calibrator is itself a streaming MA-GPCM-style model where the LLM serves as the encoder and the calibration head serves as the decoder. The calibrator emits both a calibrated score and a confidence sequence for that score.

Baselines. Raw LLM scores. Rubric-conditioned LLM grading (Saha et al., 2026). Few-shot calibrated graders (Zheng et al., 2024). Bias correction via regression (Wang et al., 2024).

Datasets. ASAP/ASAP++ essay scoring data. The PERSUADE corpus. The candidate's synthetic GPCM data with LLM-graded outputs as a controlled probe.

Evaluation. Agreement with human graders (QWK, weighted kappa). Calibration of the score posterior. Drift detection when the LLM version changes mid-stream. Length bias (do longer responses receive higher scores than human graders give them).

Open theoretical claim. Under bounded grader drift and a fixed human-anchor refresh rate, the streaming calibrator maintains kappa agreement with humans within a known band, anytime. The result is an anytime-valid analogue of the LLM-judge calibration analyses (Zhao et al., 2024).

Hard problem. Human anchor responses are expensive. The thrust requires an active-learning anchor selection that maximizes calibration value per human-graded response.

Connection to MA-GPCM. The LLM grader is the encoder; the calibration head is the decoder. The encoder-decoder design pattern is preserved verbatim across response formats. This is the strongest illustration of the candidate's own design-pattern claim.

### Sub-paper 4.2. Online DIF as anytime-valid invariance test

Research question. Can differential item functioning between groups be tested anytime, in the streaming forward pass, using an e-process over the per-item likelihood ratio of group-A and group-B responses, conditional on the running theta estimate.

Method. At each step, compute a per-item conditional likelihood ratio between group A and group B (matched on the running theta). Accumulate this into an e-process (Ramdas et al., 2022). Reject the null of no DIF when the e-process crosses 1/alpha. Group A and group B can be demographic groups (classical DIF), grader identities (grader-DIF from 4.1), or prompt versions (prompt-DIF).

Baselines. Mantel-Haenszel DIF (the standard batch test). Logistic regression DIF. SIBTEST. The multi-detector ensemble of Huang and Ishii, 2024. The interpretable-ML DIF of Kraus et al., 2024. The causal-DIF framework of Suk and Lyu, 2024.

Datasets. ASSISTments with demographic metadata (when available). The candidate's synthetic data with simulated group effects. Educational testing service publicly released data when accessible.

Evaluation. Detection delay vs detection power vs false alarm rate. Coverage of the anytime-valid p-value. Comparison with batch DIF results computed on the same data.

Open theoretical claim. The e-process for DIF is anytime-valid under the null, and achieves power 1-beta against a fixed alternative within O(log(1/beta) / KL-divergence) responses. This is a direct application of Ramdas et al., 2022 to educational measurement, which is novel.

Hard problem. Matching on the running theta introduces dependence between the test statistic and the running estimate. The e-process must be constructed to remain valid under this dependence.

Connection to MA-GPCM. The DIF test is a derived quantity from the streaming forward pass. It does not require a separate model.

### Sub-paper 4.3. Streaming generative items and effective-difficulty drift

Research question. When an LLM generates items at deployment time, the effective difficulty of each item depends on the prompt, the model version, and the student population. Can a streaming pipeline estimate effective difficulty within k student exposures, and detect when the prompt-induced difficulty drifts.

Method. Treat each generated item as a new cold-start item (use the network from 2.1). Track its difficulty estimate over student exposures. Combine with the drift head from 1.2 to detect prompt-induced difficulty drift. Test by deliberately drifting the prompt midway and measuring detection delay.

Baselines. Static IRT calibration after k exposures. The multi-agent AIG framework (Park and Lee, 2025). LLM-respondent calibration (Liu et al., 2024).

Datasets. The candidate's synthetic data with LLM-generated items keyed to a curriculum. A held-out probe set with controlled prompt drift.

Evaluation. Difficulty recovery as a function of k. Detection delay for prompt drift. Item-effective-difficulty calibration.

Open theoretical claim. The amortized item-parameter network plus drift head detects prompt-induced effective-difficulty drift with delay bounded by O(1/sqrt(k)) under bounded drift, matching the cold-start convergence rate of 2.1.

Hard problem. Prompts are themselves text; representing prompt drift as a measurable signal requires a prompt-embedding distance metric.

Connection to MA-GPCM. The decoder is unchanged. Only the item-side encoder is augmented with a prompt-conditional layer.

## Thrust 5. The streaming design pattern as a theory

The fifth thrust is partly a survey paper and partly a position paper. It collects the streaming results from the first four thrusts into a single statement of what streaming measurement requires, what it provides, and what its limits are. It is the candidate's manifesto paper, of the kind that AIED senior PhDs sometimes produce in their final year.

### Sub-paper 5.1. The streaming measurement framework: definition, requirements, guarantees

Content. Formalize the streaming measurement framework as a tuple (Encoder, Decoder, Online-Update-Rule, Audit-Layer) where the encoder is a fixed-parameter function from streams to summaries, the decoder is a measurement model (here, GPCM), the online-update rule is the forward pass, and the audit layer consumes the forward pass and emits drift, fairness, and uncertainty statements. State and prove the streaming-identifiability conditions, the calibration-under-bounded-drift conditions, and the open-world cold-start conditions in a single notation.

This paper is the one that ties the program together. It is a target for Psychometrika or for the JEDM Theory section, not for AIED.

Hard problem. Unifying notation across SBI, online conformal, e-processes, and IRT. None of these communities use compatible notation. The thrust requires inventing a common notation, which is itself a non-trivial contribution.

Connection to MA-GPCM. MA-GPCM is the running example.

### Sub-paper 5.2. Benchmarks and reproducibility for streaming measurement

Content. Release a benchmark suite for streaming IRT, drift detection, cold-start, online DIF, and online OPE in education, with the candidate's synthetic generators as part of the benchmark. The benchmark provides standardized streams, ground-truth IRT parameters, and ground-truth drift events.

Baselines. The full set of methods from 1.1 through 4.3.

Datasets. Synthetic streams plus public ASSISTments and EdNet slices arranged as streams.

Evaluation. Standardized metrics for streaming RMSE, streaming AUC, drift detection F1, conformal coverage, anytime-valid p-value coverage.

Open theoretical claim. None; this is a benchmark paper.

Hard problem. Real streams are not ground-truthed for drift, identifiability, or DIF. The benchmark must combine synthetic and real streams in a way that makes both meaningful.

Connection to MA-GPCM. MA-GPCM is the reference implementation.

## The streaming design pattern (theoretical core)

A streaming measurement model is a triple (F_psi, D_phi, A) where F_psi is an encoder with frozen parameters psi mapping a stream up to time t into a summary s_t, D_phi is a decoder with parameters phi mapping s_t into a measurement object m_t (theta_t, alpha_q, beta_q in the GPCM case), and A is an audit layer that consumes s_t and m_t and produces drift d_t, fairness f_t, and uncertainty u_t. The training procedure is meta-learning over a prior P over data-generating processes that includes the kinds of streams the system will see, and the encoder-decoder is trained so that on a held-out stream sampled from P, the decoder output approximates the Bayesian posterior over the measurement object given the stream.

The streaming measurement framework provides three formal properties when the meta-distribution is well-specified, the encoder is sufficiently expressive, and the audit layer is correctly instrumented. First, streaming identifiability: the decoder output at time t recovers the true measurement object up to a known equivalence class, at a rate that depends on stream-statistic quantities (item coverage, ability coverage, drift magnitude). Second, anytime-valid calibration: with the audit layer instantiated as online conformal plus an e-process, the calibration claim holds for every t, under optional stopping. Third, open-world extensibility: when the item or learner space expands, the encoder can absorb the expansion without retraining, provided the expansion lies in the support of the meta-distribution.

The framework requires three things in return. First, a meta-distribution that is rich enough to cover the deployment population. This is the bottleneck of the entire program, and it is where the candidate's synthetic data generators are load-bearing. Second, an encoder architecture that is amenable to streaming inference (recurrent or attention-with-cache; the Mamba family is a natural fit). Third, an audit layer that is statistically sound and computationally cheap, which is why we rely on online conformal and e-processes rather than full Bayesian posterior monitoring.

The framework does not provide everything. It does not provide causal claims about interventions; those require additional design (thrust 3). It does not provide robustness to adversarial inputs; that is a separate research line. It does not provide interpretability beyond the measurement model itself; the encoder remains a black box. These limits are honest, and the candidate should state them in the manifesto.

## Cross-cutting themes

Three themes thread through every thrust.

The first is calibration as the unifying metric. In sub-paper 1.4 calibration is the explicit object. In 1.2 calibration of the drift posterior is the open claim. In 2.1 calibration of the amortized item prior is what determines convergence. In 3.1 calibration of the streaming posterior is what makes the bandit's regret bound hold. In 4.1 calibration of the LLM grader is the central task. In 4.2 calibration of the DIF e-process is what makes anytime-valid testing possible. The program is, at its core, a calibration program.

The second is the encoder-decoder design pattern. Every thrust preserves the pattern. The encoder may be a Mamba, a transformer, a foundation model, or an LLM. The decoder is always a measurement model with explicit, interpretable parameters. The audit layer is always an additive online statistical layer. This pattern is the candidate's invariant.

The third is the synthetic-real bridge. The candidate's existing synthetic generators are the program's empirical core. They allow ground-truth IRT parameter recovery, ground-truth drift events, ground-truth DIF, and ground-truth intervention values, none of which are available on real data. The real data is used for prediction and deployment claims; the synthetic data is used for recovery and theoretical claims. Every thrust uses both.

## Open questions for the candidate

The following questions should shape the next round of refinement.

How much of the DRL stack does the candidate want to keep. Thrust 3 uses contextual bandits, which are much lighter than full DRL. If the candidate wants to keep more of the DRL machinery, it can be added as a thrust 3.4 (deep RL on the calibrated state, with the bandit as the baseline). If not, the lighter formulation is statistically stronger and easier to defend.

How aggressive a multimodal commitment is realistic. Thrust 4 already uses LLMs for grading and item generation. Adding response time, gaze, and hint usage as modalities would strengthen thrust 1 substantially, but it requires multimodal datasets that the candidate may not have. A reasonable middle path is to add response time only, since it is universally available.

How much theory the candidate wants to write. Sub-papers 1.1, 1.2, 1.4, 2.1, 3.1, 3.2, 3.3, 4.2 each carry a formal claim. The candidate's prior work is empirical. A clear decision is needed about whether to aim for AIED with empirical depth and lighter theory, or for Psychometrika and JMLR with full theorems. The program supports both, but the time allocation is different.

How to position MA-GPCM going forward. Once the streaming framework is articulated, MA-GPCM becomes one instance, not the centerpiece. The candidate should decide whether to keep it as a running example or to rename and rebrand it as the reference implementation of the framework. The latter is stronger for thesis coherence.

What the venue strategy looks like across years. AIED, EDM, LAK are the natural homes for sub-papers 1.3, 2.2, 2.3, 3.1, 3.3, 4.1, 4.3, 5.2. JMLR, Psychometrika, Journal of Educational Measurement are the natural homes for 1.1, 1.2, 1.4, 2.1, 3.2, 4.2, 5.1. NeurIPS and ICML are possible for 1.3, 2.1, and the manifesto if reframed. The candidate's stated preference is AIED-first, ML-secondary, so the major venues should be AIED for the empirical thrust 1, 2, 3 papers, and Psychometrika or JEDM for the theoretical companions.

Whether to open-source a streaming-measurement library. The benchmark paper 5.2 will need code. A standalone library, marketed as the streaming-IRT toolkit, would give the program a community footprint that pyKT enjoys. The candidate already has the MA-GPCM codebase as the seed; extending it to a full library is a year-long engineering investment that pays off in citations and reproducibility.

The strongest single-sentence pitch for this direction is the following. The candidate has built a single forward pass that does what classical educational AI does in five separate offline pipelines, and the PhD will prove that this is not a coincidence but a general principle of educational measurement at deployment scale.
