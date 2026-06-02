# Direction 2: DL-based Computational Psychometrics (depth-first round, no DRL)

This blueprint develops a single coherent doctoral program around one organizing principle, the encoder, decoder, single-pass paradigm that MA-GPCM instantiates. The encoder is any sequence model that maps a stream of learner interactions to a latent representation. The decoder is any structured psychometric head (binary IRT, GPCM, GRM, MIRT, DINA, NIDA, learned-Q, multistage adaptive testing) that turns that representation into calibrated parameters and predictions. The program asks when this pattern recovers a well-defined measurement model, when it can absorb new response formats by swapping the decoder, when it can absorb new content domains by swapping the encoder, and how it behaves under streaming arrival of new learners, new items, and LLM-mediated grading.

Reinforcement learning is excluded. Adaptive testing appears here as a decoder, not as a policy-learning problem.

## Literature scan (organized by thrust)

The encoder backbone of contemporary knowledge tracing began with DKT (Piech et al., 2015), which used an LSTM over response tuples. DKVMN (Zhang et al., 2017) introduced an attention based memory factored into a static key matrix indexed by concepts and a dynamic value matrix tracking mastery. SAKT (Pandey and Karypis, 2019) replaced recurrence with self attention. SAINT and SAINT+ (Choi et al., 2020a,b) used a Transformer encoder, decoder pair on exercise and response sequences with explicit temporal features and were validated on EdNet. AKT (Ghosh, Heffernan, and Lan, 2020) added monotonic exponentially decaying attention motivated by forgetting. SimpleKT (Liu, Liu, Chen, Huang, Luo, 2023) showed that a Rasch question embedding with dot product attention matches or beats more elaborate baselines. pyKT (Liu et al., NeurIPS 2022 Datasets and Benchmarks) standardised data splits and reported that many DLKT improvements vanish under leakage free evaluation. KTSTs (Toward Principled Transformers for Knowledge Tracing, 2025) treats KT as a set transformer task. RouterKT (2025) introduces mixture of experts attention. GKT (Nakagawa, Iwasawa, Matsuo, 2019) and DyGKT (2024) place a graph neural network over a concept graph.

On the decoder side, Deep IRT (Yeung, 2019) attached a Rasch style head to DKVMN to produce theta and difficulty. NeuralCD (Wang et al., 2020, journal version 2022) wraps DINA, DINO, MIRT in a neural multilayer architecture with monotonicity constraints. ICD (Qi et al., 2023) makes slip and guess explicit. Q matrix learning has moved from EM with Lasso (Fu et al., 2025) to restricted Boltzmann machines (Liu et al., 2020) and to sparse Bayesian MIRT (Chen, Filiz, Vanaja, 2023, arXiv 2310.17820). For ordinal response heads, CORN (Shi, Cao, Raschka, 2023, Pattern Analysis and Applications) provides rank consistent conditional ordinal heads. MA-GPCM extends this lineage by attaching a GPCM head with K minus one cumulative logits, recovering theta as a separated read pathway.

For inference, VIBO (Wu, Davis, Domingue, Piech, Goodman, EDM 2020) reformulates IRT as amortised variational inference, achieving up to two hundred times speedup on PISA and DuoLingo. Modeling IRT with stochastic variational inference (Natesan Batley et al., 2021, arXiv 2108.11579) and VAE based MIRT with correlated traits (Curi, Converse, Hajewski, Oliveira, 2021) demonstrate that amortised inference scales to dozens of latent dimensions. Variational Temporal IRT (Vie and Cousot, EDM 2023) extends amortisation to dynamic ability. Handling missing data in VAE based IRT (Veldkamp et al., 2025, British Journal of Mathematical and Statistical Psychology) addresses the masked observation problem that any streaming KT model must solve. The autoencoded sparse Bayesian IRT for the WD FAB (Graf et al., AISTATS 2023, arXiv 2210.10952) is the cleanest existing example of a Bayesian psychometric model embedded as a decoder under an encoder that amortises ability inference, almost exactly the design pattern MA-GPCM follows in the dynamic streaming setting.

Outside psychometrics, Prior Fitted Networks (Muller, Hollmann et al., ICLR 2022, Transformers Can Do Bayesian Inference) showed that a transformer trained on draws from a prior performs amortised Bayesian inference in one forward pass. TabPFN (Hollmann, Muller et al., 2023, 2025) scales this to tabular data with structural causal model priors. Distribution Transformers (Vasconcelos et al., 2025, arXiv 2502.02463) allow on the fly prior adaptation. CANVI (Patel et al., ICML 2024, arXiv 2305.14275) gives marginal coverage guarantees for amortised variational simulation based inference. Cranmer, Brehmer, Louppe (PNAS 2020) survey simulation based inference more broadly. TimesFM (Das et al., ICML 2024) and TimeGPT (Garza et al., 2024) are foundation models for time series and are a methodological cousin of any cross corpus pretrained KT encoder.

For LLM mediated assessment, AutoIRT (Ozyurt, Lalor et al., 2024, arXiv 2409.08823) calibrates IRT with AutoML from item content alone, enabling cold start. BERT IRT used in English language proficiency testing extends the same idea. Maeda et al. (Journal of Educational Measurement, 2025) use LLMs and explainable AI to predict DIF from item text. SINKT (Fu, Liu, et al., CIKM 2024) augments KT with LLM derived concept and question embeddings. Next Token Knowledge Tracing (2025, arXiv 2511.02599) shows pretrained LLM representations decode student behaviour without bespoke encoders. Language Bottleneck Models for Qualitative Knowledge State Modeling (2025, arXiv 2506.16982) puts natural language between the encoder and decoder. On the grading side, LLM as judge surveys (Gu et al., 2024, arXiv 2411.15594) and bias studies (Wang et al., 2024 on position bias, Park et al., 2024 EMNLP on judgement bias, arXiv 2506.22316 on scoring bias) document systematic noise channels for LLM raters that classical psychometric rater models do not capture.

For computerised adaptive testing as a decoder, BOBCAT (Ghosh and Lan, 2021, arXiv 2108.07386) frames item selection as bilevel optimisation. Zhuang et al. (NeurIPS 2024, Computerized Adaptive Testing via Collaborative Ranking) treat CAT as ranking. A 2024 machine learning survey of CAT (arXiv 2404.00712) maps the landscape.

This is the literature the program inherits and extends.

## Thrust 1, Decoder generality

The encoder produces a representation z_t. A psychometric decoder f maps z_t to (theta_t, item parameters) and emits a likelihood over the next response. The claim is that this pattern is decoder agnostic in the sense that, given identifiability constraints suited to the head, the same encoder recovers the true parameters of any psychometric model in a known family. MA-GPCM already validates this for GPCM. Each sub paper instantiates a different decoder and tests whether recovery is preserved.

### Sub-paper 1.1, A taxonomy of psychometric decoders under one streaming encoder

Research question. Given a fixed DKVMN encoder, do GPCM, GRM (Samejima graded response), PCM (Masters partial credit), Rasch, 2PL, and 3PL decoders all recover their respective ground truth item parameters from simulated streaming response data at competitive sample efficiency relative to marginal maximum likelihood with mirt.

Method. Implement six decoder heads sharing the same encoder. Each head exposes a small set of learnable item parameters whose shape depends on the model, alpha and beta for GPCM, a alpha and ordered c for GRM, c alone for PCM, b alone for Rasch, alpha and b for 2PL, alpha, b, c for 3PL. Train jointly on synthetic data generated under the corresponding data generating process and evaluate recovery (correlation, RMSE, calibration) on theta, alpha, and the model specific difficulty or threshold parameters.

Baselines. mirt (R), VIBO (Wu et al., 2020), Deep IRT (Yeung, 2019), Variational Temporal IRT (Vie and Cousot, 2023). For Rasch and 2PL the candidate inherits established baselines from those packages.

Datasets. Synthetic per head DGPs with K equals 2 or higher categories where applicable. Real data, ASSISTments 2009 and 2012 (Rasch, 2PL), DuoLingo HLR (2PL), PISA reading 2018 (GPCM), TIMSS science (GRM where applicable).

Metrics. Per parameter Spearman and Pearson recovery, RMSE, posterior coverage where available, prediction AUC and QWK, runtime to convergence.

Hard problem. Identifiability differs by head. GPCM thresholds slide under additive shifts to beta and theta unless beta is centered. 3PL is notoriously weakly identified for small c. The neural encoder concentrates capacity in z_t, so the decoder must carry the identifiability constraints, otherwise the encoder absorbs everything and the head parameters become arbitrary. The hard problem is to formulate per head constraints (centering, ordering, monotonicity, lower asymptote priors for 3PL) that are sufficient for asymptotic recovery without sacrificing predictive performance.

Formal claim. Under a fixed encoder of finite capacity and per head identification constraints C_h, the maximum likelihood estimator over the joint encoder, decoder is consistent for the item parameters of head h up to the equivalence class allowed by C_h, provided the encoder is sufficiently expressive to represent the true conditional ability process. State the claim, prove it under a simplifying linear encoder assumption, demonstrate it empirically for the deep encoder.

### Sub-paper 1.2, Cognitive diagnosis as a decoder, DINA and NIDA in a single forward pass

Research question. Can a streaming DKVMN, attention encoder feed a DINA, NIDA, or G DINA decoder and recover attribute mastery patterns competitive with NeuralCD and classical EM, while also tracking attribute mastery over time, which classical CDMs do not model.

Method. Encoder z_t is projected to a per attribute mastery vector alpha_t in [0,1]^K. The DINA decoder uses a known Q matrix, defining the latent class for item j as the conjunction of required attributes. The DINA likelihood involves slip s_j and guess g_j parameters which become learnable scalars per item. NIDA places attribute level noise. Compare a fixed Q matrix variant against a learnable Q matrix decoder regularised by an L1 penalty (Fu et al., 2025 style) on Q entries.

Baselines. NeuralCD (Wang et al., 2022 TKDE), classical EM DINA with the CDM R package, ICD (Qi et al., 2023).

Datasets. FrcSub, Math1, Math2 (canonical CDM datasets), ASSISTments with skill annotations, a synthetic dynamic DINA DGP where attribute mastery follows a staircase or random walk, an LLM annotated Q matrix on EdNet content.

Metrics. Attribute level mastery recovery, item parameter recovery, response prediction AUC, Q matrix recovery against the expert annotation in a held out validation set.

Hard problem. CDMs were designed for static testing. The honest extension to streaming sequences requires deciding whether alpha_t flips one attribute at a time (transition style) or moves continuously and is then thresholded for class labels. The candidate must commit to a generative story and confirm that the chosen story is identifiable.

Formal claim. The encoder, DINA decoder is identifiable up to label permutation on attribute order when the Q matrix is known and each attribute is required by at least three items not all of which require the same other attributes (the standard DINA identifiability condition extended pointwise in time).

### Sub-paper 1.3, Learned Q matrix as decoder structure, sparse priors versus content priors

Research question. When the Q matrix is unknown, can the decoder learn an interpretable, sparse Q matrix from response data alone, from content alone (LLM embeddings of item text), or from a combination, and how does each compare to an expert annotated Q matrix on downstream attribute mastery recovery and prediction.

Method. Two decoder heads. The data driven head learns a real valued Q in R^{J x K} with an L1 plus group sparsity penalty, then thresholds at evaluation time. The content driven head feeds an LLM embedding of each item through a small MLP to produce Q row predictions, akin to AutoIRT (Ozyurt et al., 2024) but for attribute structure rather than IRT parameters. A combined head uses content as a prior over Q with data driven refinement.

Baselines. Expert Q (gold standard), Lasso EM Q estimation (Fu et al., 2025), restricted Boltzmann machine Q learning (Liu et al., 2020), sparse Bayesian MIRT (Chen et al., 2023).

Datasets. Math1, Math2, FrcSub with known Q, EdNet and ASSISTments with partial Q, MMLU with section labels treated as soft Q rows.

Metrics. Q matrix recovery agreement with expert annotation, downstream alpha mastery recovery, response prediction. Important secondary metric, expert agreement on the learned Q, scored by a separate panel.

Hard problem. The Q matrix is identifiable only up to attribute permutation and certain structural rearrangements. With LLM content as a prior the permutation is fixed implicitly through attribute names. Without it, the program must explicitly enforce a canonical ordering, for instance by frequency or by ordering attributes along a low rank decomposition of the response matrix.

Formal claim. With sufficient response data and an LLM derived content prior that is informative in the sense of having positive mutual information with the true attribute assignment, the posterior over Q concentrates on the true equivalence class.

### Sub-paper 1.4, MIRT decoder and beyond unidimensionality

Research question. The MA-GPCM design is unidimensional. Replace the theta head with a D dimensional MIRT decoder. Can the same DKVMN encoder support compensatory MIRT, non compensatory MIRT, and bifactor structures, with identifiability enforced by orthogonality and reference loadings, and does the resulting model recover a known multidimensional structure on simulated data and yield interpretable trait estimates on PISA where reading sub domains are available.

Method. The IRT parameter extractor produces theta_t in R^D, alpha in R^{D x J}, beta as before. Identifiability constraints are imposed by an orthogonality penalty on the trait correlation matrix (the candidate has already implemented this for kt mirt) and by anchor items with prespecified loading structure (a bifactor anchor). Compare compensatory and partially compensatory likelihoods.

Baselines. mirt (R) MML estimation, ML2P VAE (Curi et al., 2021), the candidate's own MA-GPCM with D equals 1.

Datasets. Synthetic MIRT DGPs with D equals 1, 2, 3 and known loadings, PISA reading with three subscales, ASSISTments with explicit skill groups.

Metrics. Trait correlation recovery, factor loading recovery up to rotation, prediction performance, identifiability sensitivity (run multiple seeds and measure within seed and between seed correlation of recovered traits).

Hard problem. Rotational indeterminacy. In multidimensional factor models the loadings are identified only up to an orthogonal rotation. The candidate must either commit to a rotation (varimax, oblimin, target rotation against an LLM derived hypothesis matrix) or report all results in rotation invariant terms (subspace angles, factor congruence). The honest framing is the second.

Formal claim. Under compensatory MIRT with D known and an orthogonality constraint, the encoder, decoder is identified up to a signed permutation matrix. With a bifactor anchor item set the identification is exact.

### Sub-paper 1.5, Multistage adaptive testing as a decoder

Research question. Cast multistage adaptive testing as a decoder that, at each step, produces a categorical distribution over the next item conditional on theta_t and on past items. Training is supervised by simulated optimal item selection under maximum Fisher information, not by reinforcement learning. Does this learned policy match or exceed BOBCAT and maximum Fisher information at test length reduction while remaining differentiable through the encoder.

Method. The decoder is a softmax over the unused item bank, with logits given by predicted Fisher information evaluated at the current theta posterior, plus a small learned correction trained to mimic an oracle that has access to the full simulator. This is imitation learning, not policy gradient. The encoder is the standard MA-GPCM stack.

Baselines. Maximum Fisher information selection, BOBCAT (Ghosh and Lan, 2021), NCAT, collaborative ranking CAT (Zhuang et al., NeurIPS 2024).

Datasets. ASSISTments, EdNet, simulated GPCM and MIRT item banks of varying size.

Metrics. Test length to reach a target SEM, theta recovery RMSE at fixed test length, calibration of the predicted Fisher information.

Hard problem. Differentiating through item selection requires either a Gumbel softmax relaxation or a straight through estimator. Each has known bias. The candidate must choose carefully and document the trade off.

Formal claim. The imitation learned policy converges to the maximum Fisher information policy as the imitation oracle is queried more often, with sample complexity bounded by a constant depending on the encoder Lipschitz constant.

## Thrust 2, Encoder generality

The decoder fixes the measurement model. The encoder can vary widely. The thrust is to demonstrate that the same decoders work over very different encoders, and to push toward encoders that transfer across content domains.

### Sub-paper 2.1, Encoder zoo, controlled comparison under identical GPCM decoder

Research question. Fixing the GPCM decoder from MA-GPCM, which encoder family (LSTM, DKVMN, SAKT, SAINT, AKT, SimpleKT, set transformer, GNN over a concept graph, MoE attention) recovers item parameters and ability dynamics most accurately, and is recovery quality a function of in sample fit or of architectural priors over forgetting and concept structure.

Method. A controlled benchmark with identical decoder, identical data, identical training schedule, identical compute budget. Each encoder is implemented through pyKT (Liu et al., 2022) and exported with a shared interface to the GPCM head.

Baselines. The encoder zoo itself constitutes the comparison.

Datasets. All seven pyKT datasets, the synthetic suite from MA-GPCM (static, block change, staircase, random walk), EdNet, NeurIPS 2020 Education Challenge data.

Metrics. Recovery of theta, alpha, beta on synthetic data, AUC and QWK on real data, robustness to label leakage under the pyKT one window evaluation, parameter count and inference latency.

Hard problem. Decoupling encoder choice from encoder tuning. A weak result from a poorly tuned transformer is uninformative. The protocol must include a fair hyperparameter search budget per encoder and report performance under both equal compute and equal parameters.

Formal claim. None directly. The contribution is an honest leaderboard with an interpretable decoder, the first time KT models have been compared on parameter recovery rather than only on prediction.

### Sub-paper 2.2, A foundation encoder for knowledge tracing

Research question. Pretrain a single encoder on a large pooled corpus of student response sequences (EdNet, ASSISTments 09 to 17, Algebra 2005, Eedi NeurIPS 2020) with a self supervised next response objective, then attach task specific GPCM, MIRT, or DINA decoders for each held out dataset. Does the pretrained encoder improve recovery and downstream prediction compared to training from scratch, especially in low data settings.

Method. Pretraining loss combines a next response cross entropy and a masked item prediction loss. Item embeddings are tokenized either by a hash, by a learned codebook (VQ style), or by LLM derived item text embeddings (in the spirit of SINKT and Next Token Knowledge Tracing, 2025). Fine tuning attaches a decoder head and continues training on the target dataset.

Baselines. Train from scratch encoder, SINKT, simpleKT, AKT.

Datasets. Pretraining corpus as listed. Downstream datasets, ASSISTments 2009 (small), Riiid Answer Correctness, NeurIPS 2020 Education Challenge.

Metrics. Downstream AUC, QWK, recovery on a held out synthetic probe (a synthetic dataset matched in distribution to a target real dataset, used as a recovery probe).

Hard problem. Items differ across corpora. Without a content based tokenisation, item embeddings cannot transfer. With LLM content embeddings transfer is possible but introduces a confound, LLM content may leak information about correctness or curriculum order. The candidate must distinguish gains from architecture from gains from LLM content embeddings.

Formal claim. Open empirical claim, no theorem. The thesis is that foundation encoders for KT, analogous to TimesFM for time series, are viable and that the MA-GPCM decoder pattern is the right interface to attach to them.

### Sub-paper 2.3, GNN encoder over an item, skill graph and inductive evaluation on new items

Research question. Replace the DKVMN attention with a temporal GNN over a heterogeneous item, skill graph (in the spirit of GKT and DyGKT, 2024). With a GPCM decoder, does the GNN support inductive evaluation when a new item is added to the graph mid stream, scoring a learner on the new item using only its connectivity and content features.

Method. A heterogeneous graph with item, skill, and learner nodes. Messages flow item to skill, skill to skill, learner to item via response edges. The encoder reads at the learner node at time t. The decoder is GPCM. New items are inserted as nodes with content features and known skill edges (from expert tags or LLM tagging), no historical responses.

Baselines. DKVMN encoder with the same decoder, SINKT, DyGKT.

Datasets. ASSISTments, EdNet, a controlled split where ten percent of items are held out and revealed mid stream as new items.

Metrics. Inductive AUC on held out items, recovery of those items' alpha and beta under the GPCM decoder once enough responses accumulate, cold start latency (number of responses required to reach a given calibration accuracy).

Hard problem. Inductive item insertion under streaming inference. Most KT benchmarks assume a closed item bank. The honest evaluation requires a temporal split, not a random split, and must respect causality.

Formal claim. The encoder, GPCM decoder is jointly identifiable for new items in the streaming open world setting provided that each new item is linked to at least one previously calibrated item via skill edges with non zero learner overlap.

### Sub-paper 2.4, Cross domain transfer and linking diagnostics

Research question. Train the encoder on K to 12 math and evaluate it as a feature extractor for K to 12 science, college level algebra, and language learning data. Does an MA-GPCM decoder built on top of this cross domain encoder recover sensible item parameters in the target domain after light fine tuning, and can the same encoder serve as the equating bridge between two item banks that share no items but share a learner population.

Method. Pretrain encoder on math. Freeze encoder. Train MA-GPCM decoder on science. Compare to encoder fine tuned, encoder trained from scratch. For linking, compute item parameters on bank A and bank B using shared learners, then estimate the linear transformation between thetas in the two banks, with the encoder fixed across banks.

Baselines. Cross domain training from scratch, classical concurrent calibration via mirt with anchor items.

Datasets. ASSISTments math, ASSISTments physics if available, DuoLingo HLR, Algebra 2005 to 2006, EdNet.

Metrics. Downstream recovery and prediction, linking accuracy measured by recovery of known anchor item parameters when treated as unknown.

Hard problem. Concurrent calibration requires either anchor items or shared learners. The neural setting allows a third option, shared encoder. The candidate must show that the shared encoder option produces equivalent linking accuracy and is honest about when it fails (when the two banks measure very different constructs).

Formal claim. With a shared encoder of finite capacity and sufficiently expressive decoders on each bank, the implied learner ability across the two banks is identifiable up to a single affine transformation, the analog of mean variance linking in classical IRT.

## Thrust 3, Streaming amortized inference

This thrust treats the entire encoder, decoder model as a learned amortised posterior over learner and item parameters. The single forward pass replaces EM, MML, MCMC, and even non amortised variational EM. The methodological cousin is PFN and TabPFN (Muller, Hollmann, et al., 2022, 2023, 2025).

### Sub-paper 3.1, Amortized IRT as a benchmark against EM, MML, MCMC, VI

Research question. Treat MA-GPCM as a learned regression from response histories to (theta_t, alpha, beta). Across recovery accuracy, posterior coverage, calibration, and wall clock cost, how does it compare to mirt MML, Stan HMC, mean field variational Bayes with edstan, and VIBO at varying dataset sizes, sequence lengths, and number of items.

Method. Train MA-GPCM and a non amortised variational baseline (VIBO) on a matched synthetic suite. For posterior coverage, since the deterministic MA-GPCM produces only point estimates, augment it with a small Bayesian last layer (a Laplace approximation around the trained weights for the item parameter heads, or a heteroscedastic head that produces a Gaussian over alpha and beta). Compare nominal versus empirical coverage at multiple credible levels.

Baselines. mirt (MML), Stan HMC, edstan VI, VIBO, AutoIRT (Ozyurt et al., 2024).

Datasets. Synthetic GPCM, PISA reading, DuoLingo HLR, the FAB battery from Graf et al. (2023) as a clinical analogue.

Metrics. RMSE on theta, alpha, beta, log posterior coverage, calibration error on predicted response probabilities, wall clock cost amortised over training plus inference for varying numbers of new learners.

Hard problem. Amortisation gap. Amortised inference networks can underfit the true posterior even when the model is correctly specified. The candidate must quantify this gap and decide whether it is acceptable given the wall clock advantage.

Formal claim. With sufficient training simulations from the prior, the amortised posterior achieves nominal marginal coverage at a rate that depends polynomially on the encoder width, in the spirit of CANVI (Patel et al., 2024) but adapted to the IRT setting.

### Sub-paper 3.2, PFN style pretraining on a prior over GPCM tasks

Research question. Train a transformer in the PFN style, conditioning on a context of (item, response) pairs and a query item, to output the predicted response distribution under a posterior over (theta, alpha, beta) drawn from a prior. After training on synthetic tasks drawn from a hierarchical GPCM prior, does the model generalise to held out real datasets in a true in context fashion without any fine tuning, in the way TabPFN does for tabular tasks.

Method. Sample a prior over hyperparameters (number of categories, mean and variance of alpha, mean and variance of beta, possibly an autoregressive process for theta), sample a task, sample a sequence, train the PFN. At inference, feed a response history as context, the model outputs the predictive distribution for any query item, plus a learned proxy for theta if requested.

Baselines. MA-GPCM trained from scratch on the target dataset, VIBO, mirt.

Datasets. Synthetic suite for pretraining, ASSISTments, PISA, EdNet for in context evaluation.

Metrics. In context prediction AUC and QWK on real datasets, recovery on synthetic tasks held out from the prior, sample efficiency (number of context responses required to reach target accuracy).

Hard problem. The PFN must consume variable length context with variable item ids. Item identity must be either tokenised invariantly (a content embedding) or made permutation invariant via set attention. The cleanest answer is to feed item content via an LLM embedding plus a positional handle, but this couples results to the LLM. The candidate should run an ablation without content.

Formal claim. If the pretraining prior contains the data generating process of the target dataset, the PFN posterior converges to the Bayes posterior as the context length grows, in the sense formalised in Muller et al. (2022).

### Sub-paper 3.3, Coverage guarantees via conformal amortisation

Research question. Apply CANVI (Patel et al., 2024) to the GPCM amortised posterior to produce conformal predictive intervals for theta_t with guaranteed marginal coverage. Does this improve practical trust in the model for high stakes use (placement testing, formative feedback).

Method. Treat the model's predicted posterior as a candidate, conformalise it using a held out simulator calibration set, evaluate marginal coverage on real data via a heldout learner subset whose true theta is approximated by long run MAP from EM as an oracle proxy.

Baselines. Vanilla amortised posterior, mean field VI credible intervals, Bayesian neural network last layer.

Datasets. PISA, DuoLingo, synthetic.

Metrics. Empirical marginal coverage at multiple alpha, interval width, conditional coverage stratified by ability level, item count, and learner experience.

Hard problem. Calibrating against a simulator means trusting the simulator. If the simulator does not match the real data, coverage will fail to transfer. The candidate must include a misspecification stress test where simulator and target differ in known ways (Robust Variational NPE, arXiv 2509.05724, is the analogue in physics).

Formal claim. Coverage guarantees from CANVI carry over to the encoder, decoder amortised posterior, provided the calibration set is drawn from the same distribution as the test set.

### Sub-paper 3.4, Streaming posterior updates and online identifiability

Research question. In a streaming setting where responses arrive one at a time, can the model produce a sequence of well calibrated posteriors over theta_t that is online consistent, that is, the posterior contracts at the correct rate as more responses arrive.

Method. Quantify posterior contraction rate empirically against an EM oracle that re estimates from scratch after each response. Compare amortised, recurrent, and transformer encoders. Develop a diagnostic that flags when the streaming posterior is over or under confident relative to the oracle.

Baselines. EM oracle, online EM, particle filter on the GPCM state space.

Datasets. Synthetic with controlled signal to noise, EdNet for realism.

Metrics. Posterior contraction rate, calibration error as a function of t, time to detect a known shift in theta.

Hard problem. The encoder is a fixed function and so its posterior is biased by the training distribution. Under distribution shift the streaming posterior can be wrong in subtle ways. The candidate must build a monitor that detects this.

Formal claim. Under a known DGP that lies in the support of the training distribution, the streaming posterior contracts at the parametric rate n to the minus one half in total variation distance from the true posterior, up to an amortisation gap.

## Thrust 4, Generative items plus AI graded responses

This thrust takes seriously the modern reality that items are increasingly generated by LLMs and responses are increasingly graded by LLMs. Both interventions inject systematic non psychometric structure into the response stream that classical IRT silently absorbs into theta, alpha, or beta. The thrust separates that structure cleanly.

### Sub-paper 4.1, LLM as item generator with online IRT calibration

Research question. Use an LLM to generate items targeted at desired (alpha, beta), then calibrate generated items in an online stream of learner responses using MA-GPCM with an open ended item bank. Does the predicted alpha and beta of generated items match the realised alpha and beta after calibration, and how does this match degrade across difficulty levels, content domains, and LLM versions.

Method. Train a content prediction head on existing calibrated items, mapping LLM item embeddings to (alpha, beta). Use this head as the prior for calibration of new items, with response data updating the posterior. This is an explicit Bayesian update from a content prior to a response posterior, AutoIRT (Ozyurt et al., 2024) made dynamic and online.

Baselines. AutoIRT (offline content only calibration), classical calibration with fixed pilot data, no prior cold start calibration.

Datasets. K to 12 math item generation, MMLU style multiple choice generation, an LLM generated extension of an existing item bank with paired human ratings of difficulty and discrimination.

Metrics. Predicted versus realised parameter agreement, calibration speed (responses needed for predicted parameters to be within tolerance), drift across LLM versions, downstream recovery of theta with mixed banks.

Hard problem. Content prior misspecification. If the content head was trained on math, it will be miscalibrated for science. The candidate must quantify and visualise the prior, posterior gap as a function of content similarity, possibly using a content distance metric derived from the same LLM.

Formal claim. With a content prior that is informative in the sense of having lower entropy than the marginal prior, posterior contraction is faster than no prior baseline by a factor that depends on the mutual information between content and true item parameters.

### Sub-paper 4.2, LLM as rater with a learned structured noise channel

Research question. Model LLM grading as a noise channel between the true latent response category Y and the observed graded category Y tilde. Existing rater models in psychometrics use a single offset or a Cohen kappa per rater. LLM raters exhibit systematic, content dependent, position dependent, and option dependent biases (surveyed in Gu et al., 2024). Can a learned neural noise channel, attached to the MA-GPCM decoder, recover the true item parameters under noisy grading while also producing an interpretable model of LLM rater behaviour.

Method. Decoder factors as P(Y tilde given item, learner) equals sum over Y of P(Y tilde given Y, item, rater) P(Y given item, learner). The first factor is the noise channel, parameterised by an MLP that consumes item content, rater identity, and the position of the response, outputting a K by K confusion. The second factor is GPCM as before. Train end to end with an identifiability constraint that the noise channel reduces to identity for human raters used as anchors.

Baselines. Single offset rater model, hierarchical rater model (HRM), MA-GPCM ignoring rater identity.

Datasets. ASAP automated essay scoring with paired human and LLM scores, short answer datasets with LLM and human scoring (the BERT IRT setting from Ozyurt et al. but with LLM raters), a controlled study where the candidate runs three LLM raters over a fixed essay set.

Metrics. Recovery of item parameters under noisy grading, calibration of the noise channel (rater confusion matrix prediction), DIF in the noise channel by content topic and length.

Hard problem. Identifiability between the rater channel and item parameters. A constant biased rater can be absorbed into beta. The candidate must use human anchored raters or items with known difficulty to break this confound.

Formal claim. Given at least one anchor rater whose noise channel is constrained to identity, the model is identified for item parameters and for the LLM raters' channels.

### Sub-paper 4.3, DIF between generated and human authored items, between LLM and human graders

Research question. Are LLM generated items differentially functioning compared to human authored items, conditional on theta, and are LLM graders differentially functioning compared to human graders, conditional on the true response. Use the encoder, decoder architecture from 4.1 and 4.2 to produce a DIF statistic that operates on streaming data.

Method. DIF statistic is the gap between the model's predicted response probability under the generated item branch and the human item branch, conditional on a matched theta. Implement as a permutation test against a null where item source is randomised, computed online via a streaming permutation procedure.

Baselines. Mantel Haenszel DIF, logistic regression DIF, classical mirt anchored DIF.

Datasets. The mixed banks from 4.1 and 4.2, MMLU human plus LLM augmented, ASAP, the DIF prediction setting in Maeda et al. (2025).

Metrics. DIF detection power and Type I error against a synthetic ground truth, agreement with classical DIF methods on real data.

Hard problem. DIF for streaming open item banks is not well defined classically. The candidate must propose a clean operational definition (DIF at a given theta level integrated over the streaming observation density) and show that it reduces to classical DIF in the static special case.

Formal claim. The proposed streaming DIF statistic is consistent for the classical DIF effect size as the observation density approaches the static design density.

### Sub-paper 4.4, Joint LLM author and LLM rater calibration as a chained noise model

Research question. The end to end pipeline of LLM generated items graded by LLM raters has two stacked noise channels. Can the model identify each separately under realistic conditions, and what is the minimum anchor configuration (anchor items plus anchor raters) required for joint identifiability.

Method. Combine 4.1 and 4.2 into one model with two noise channels and a content prior over item parameters. Run identifiability experiments where anchors are removed one at a time.

Baselines. Stacked separate calibration (calibrate items using AutoIRT, then calibrate raters separately), naive end to end without anchors.

Datasets. Fully synthetic with known truth, plus a real LLM generated, LLM graded subset of ASAP.

Metrics. Recovery as a function of anchor count, sensitivity to anchor quality.

Hard problem. The composition of two unknown noise channels is a non trivial inverse problem. The candidate should connect this to identifiability results in independent component analysis and to chained measurement error models in econometrics.

Formal claim. With at least k anchor items per attribute and at least one anchor rater per content domain, the joint model is identified, where k depends on the rank of the item content embedding.

## Thrust 5, Per-skill feedback, person-fit, open-world streams

The decoders so far emit scalar or low dimensional theta. Modern educational use cases demand per skill mastery feedback, online detection of aberrant response patterns (rapid guessing, cheating, miscoded items), and seamless handling of new learners and items in production.

### Sub-paper 5.1, Multidimensional concept aligned memory for per skill ability

Research question. Replace the unidimensional theta head with a D dimensional theta_t aligned with named concepts (extracted from item text via an LLM or expert annotation). Does this produce interpretable per skill mastery feedback that agrees with human teacher judgement, and does it recover known skill substructure on synthetic data.

Method. The DKVMN value memory is partitioned along D concept slots. The decoder reads each slot independently to produce theta_t^(d), and the GPCM logit uses a alpha weighted dot product of theta_t with item loadings. The concept slots are aligned to named concepts by a contrastive loss between the slot read and an LLM concept embedding.

Baselines. MA-GPCM unidimensional, MIRT decoder from 1.4, NeuralCD.

Datasets. ASSISTments with skill annotation, PISA with subscales, EdNet with topic tags, a synthetic dataset where ground truth per skill ability is known.

Metrics. Per skill recovery, agreement with teacher judgement on a small panel study, prediction performance.

Hard problem. The slots can collapse onto each other (a known failure mode of multi head memory). The candidate must impose a diversity constraint, possibly an orthogonality penalty on slot reads averaged over a batch, possibly a contrastive separation from a concept embedding.

Formal claim. With concept aligned contrastive loss above a threshold, the slot assignment is identified up to a permutation that matches the named concepts.

### Sub-paper 5.2, Online person fit as a first class output

Research question. Person fit indices (Lz, l_z star, ECI, U3) flag aberrant response patterns. Classically they are computed post hoc on a static test. Can the encoder, decoder produce a streaming person fit index online, flagging aberrance within the session, and is this index calibrated against known aberrance categories (rapid guessing, item exposure, cheating via shared answer key).

Method. The model outputs a likelihood at each step. The cumulative log likelihood ratio of the predicted GPCM model against a saturated alternative is the basis for a streaming Lz statistic. Augment with a learned auxiliary head that predicts a discrete aberrance label trained on simulated aberrance.

Baselines. Static Lz computed at end of session, post hoc clustering of response patterns.

Datasets. Synthetic with injected aberrance, ASSISTments with response time based aberrance labels (rapid guessing).

Metrics. Aberrance detection AUC at varying session lengths, calibration of the streaming statistic, latency of detection (responses to detect at a given confidence level).

Hard problem. Distinguishing genuine ability change from aberrance. A learner who is starting to guess looks like a learner whose ability is dropping. The candidate must commit to a generative story (a switching process between an attentive and an aberrant state) and show that the model identifies the switch.

Formal claim. With a known switching DGP, the streaming person fit index is uniformly most powerful against the saturated alternative within the exponential family.

### Sub-paper 5.3, Open world item arrival and online linking

Research question. When new items arrive mid stream, with optional content features and optional skill tags, the encoder, decoder should score learners on these items immediately, calibrate them online, and integrate them into the bank without re training. What is the latency of calibration to a given precision, and how does it depend on content prior quality and on learner traffic.

Method. Build on 2.3 and 4.1. New items enter as nodes in the graph (or as new tokens via content embeddings). The MA-GPCM decoder uses a content prior for initial alpha, beta and then updates via streaming responses.

Baselines. Cold start with no prior, AutoIRT offline batch calibration, hierarchical empirical Bayes calibration.

Datasets. EdNet with simulated item arrival over time, a real platform stream if available via collaboration.

Metrics. Calibration latency, prediction quality during calibration, downstream theta accuracy.

Hard problem. New items arrive at a rate that may exceed the calibration rate, leading to a backlog of weakly calibrated items. The candidate must analyse this queueing problem and propose strategies (prioritised exposure of weakly calibrated items to high information learners).

Formal claim. Under stationary learner traffic with content prior of mutual information I_c with the truth, item calibration converges with rate proportional to exp(I_c) faster than the cold start baseline.

### Sub-paper 5.4, Open world learner arrival and zero shot scoring

Research question. When new learners arrive, the encoder must score them from their first response, ideally with prior information from demographic or prior course history. Can the encoder produce a useful initial theta_t from a meta learner state (a prior over thetas implied by the population) and update rapidly with responses.

Method. The encoder's initial state is a learned prior generated by a small meta network conditioned on optional learner features. With no features, the prior is the population mean. Update is the standard encoder forward pass.

Baselines. Cold start with zero initial state, classical empirical Bayes shrinkage to population mean.

Datasets. ASSISTments, EdNet, PISA where learner background features are available.

Metrics. Initial prediction accuracy (first three responses), time to converge to within tolerance of the EM oracle theta, fairness across demographic groups (no DIF in the implied prior).

Hard problem. Learner features can be discriminatory. The candidate must include a fairness audit that checks for DIF in the meta network output across protected groups.

Formal claim. With population mean prior alone, the encoder achieves the empirical Bayes lower bound on initial estimation error.

### Sub-paper 5.5, Capstone, an end to end open world MA system in production conditions

Research question. Combine 5.1 to 5.4 into a single open world system. Stream simulation that includes new items, new learners, LLM grading, and aberrance. Evaluate the system holistically over a six month simulated period.

Method. Integration of all components, with an emphasis on engineering, deployment, and a live evaluation harness.

Baselines. The component models in isolation, classical IRT with periodic batch recalibration.

Datasets. A long horizon synthetic stream, possibly with a partial real deployment via a collaborating tutoring platform.

Metrics. Holistic, prediction performance, calibration drift, time to detect aberrance, latency of new item calibration, fairness across groups, computational cost.

Hard problem. Composition of error sources. Each component is calibrated in isolation, but errors compound at the system level. The candidate must show that the system level error remains bounded.

Formal claim. Under bounded per component error, the system level theta recovery error is bounded by a polynomial in the per component errors.

## Cross cutting design pattern (the encoder, decoder paradigm as a theory)

The five thrusts share one statable theory. Define a measurement family M as a set of psychometric likelihoods (GPCM, GRM, MIRT, DINA, NIDA, learned Q variants). Define an encoder family E as a set of sequence models from response histories to latent vectors. The pattern is the composition f equals dec circ enc, where enc is in E and dec is in M's family of decoders. The thesis claims four properties for this composition.

Property one, modularity. Swap the decoder, change the response format. Swap the encoder, change the data domain. The training pipeline does not change.

Property two, identifiability transfer. If the classical measurement model is identifiable under constraints C_M, the neural composition is identifiable under C_M applied to the decoder, provided the encoder is sufficiently expressive. The encoder must not absorb identification mass that belongs to the decoder. This requires careful constraint placement.

Property three, streaming consistency. With a sufficiently rich pretraining distribution, the amortised posterior is consistent and contracts at the parametric rate on in distribution streams. Out of distribution streams require an explicit diagnostic.

Property four, modular calibration. New items and learners enter through their respective slots in the decoder and encoder, with content priors smoothing cold start. The system does not need full re calibration.

Stating and partially proving these four properties is the theoretical backbone of the thesis. The thesis is then a sequence of instantiations and falsifications of these properties across decoders, encoders, inference modes, AI mediated input, and open world deployment.

This theory absorbs MA-GPCM as the foundational case (GPCM decoder, DKVMN encoder, unidimensional, static item bank). Every sub paper above is a generalisation along one dimension. The candidate's existing capability is the load bearing instance from which the program extends.

## Open questions for the candidate

The blueprint commits to a depth first program. Several decisions remain open and shape the order of execution.

First, the choice between MIRT and CDM as the primary decoder generality vehicle. MIRT is closer to MA-GPCM and the candidate already has working code (kt mirt). CDM is closer to the per skill feedback goals of Thrust 5. A clean reading would be to do MIRT first as a natural extension and bring CDM in via the learned Q matrix sub paper, which is a transition object between the two views.

Second, the question of whether to invest in a true foundation encoder for KT (Sub paper 2.2). This is a compute heavy undertaking and would shape the second half of the doctoral program. Alternatively, the candidate could remain at the level of pretrained transformer backbones used off the shelf for content embedding and avoid pretraining their own KT foundation model.

Third, the depth of formal claims. The blueprint lists statable claims for nearly every sub paper. The candidate must decide which claims to prove rigorously (this is a mathematical thesis worth attempting on a small subset, perhaps Sub paper 1.1, 3.1, 4.4) and which to leave as conjectures with strong empirical evidence.

Fourth, the relationship with LLM mediated assessment (Thrust 4). This is the most rapidly moving area in the literature and will not be stable over a five year doctoral program. The candidate must decide whether to treat LLM raters and generators as moving targets (each paper a snapshot) or to commit to a fixed protocol that abstracts away the specific LLM (this is the cleaner research stance and the one the blueprint quietly favours).

Fifth, the engineering dimension. Thrust 5 culminates in a production grade open world system. This is unusual for an AIED dissertation and would require partnership with a tutoring platform. The candidate should decide whether to pursue this seriously or whether to scale down Sub paper 5.5 to a simulation only system.

Sixth, the venue strategy. The thrust structure naturally maps onto the candidate's venue preferences. Decoder generality and CDM work fit AIED and IJAIED. Encoder generality, foundation encoders, and amortised inference fit EDM, LAK, and the ML venues. LLM mediated assessment fits a mix of AIED, EDM, and the LLM evaluation track at NeurIPS or ICML. Person fit and open world streams fit LAK and Journal of Educational Measurement. The candidate's preferred primary venues (AIED, IJAIED, EDM, LAK) are well covered.

Seventh, and most strategically, what to do with the existing MA-GPCM submission to IJAIED. The blueprint assumes it is the foundational case of a larger program. If the IJAIED reviewers request major revisions, the candidate has the option of folding several revisions into early sub papers of the program rather than fighting the reviewers paper by paper. This is a question of programme management more than research.

The thesis statement that emerges from this blueprint is that streaming, single pass, encoder, decoder measurement is the natural successor to both classical psychometrics and to current deep knowledge tracing, that it can be instantiated across response formats, content domains, AI mediated input, and open world deployment, and that within this paradigm a small set of identifiability, calibration, and consistency properties hold and can be stated formally. MA-GPCM is the existence proof. The program is the generalisation.
