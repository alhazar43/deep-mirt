# PhD Blueprint, D2 (v2)

## Streaming Educational Measurement as a Unified Computational Theory

Wenrui Yuan. AIED-primary thesis. Measurement-only scope. DRL and intervention explicitly out.

---

## 1. What the theory actually is

Classical educational measurement is built on a single computational pattern. Collect a finite response matrix $Y \in \{0, 1, \dots, K-1\}^{N \times Q}$, posit a generative model $p(Y \mid \Theta, \Psi)$ with person parameters $\Theta$ and item parameters $\Psi$, and recover the posterior $p(\Theta, \Psi \mid Y)$ by an offline pass, typically marginal-likelihood EM or MCMC. Every standard operation, theta estimation, item calibration, cognitive diagnosis, DIF, linking, person-fit, item generation, CAT, and drift monitoring, is a different functional of the same posterior.

The thesis claim is that this entire stack admits a single computational reduction.

**The streaming reduction.** Replace the offline tuple $(Y_{\text{fixed}}, \text{EM}_{\text{offline}}, p_{\text{posterior}})$ with the streaming tuple $(y_{1\colon t}, f_\phi, q_\phi)$ where $y_t$ is the response event arriving at time $t$, $f_\phi$ is an encoder that maintains a finite sufficient state $h_t = f_\phi(h_{t-1}, y_t)$ in a single pass, and $q_\phi$ is an amortized decoder that maps $h_t$ to whatever measurement object is required, ability, item parameter, skill profile, fit statistic, drift flag. The encoder is trained once. At deployment the cost per response is constant in the size of the history.

This is more than a method. It is a *theoretical* commitment because it makes three falsifiable claims.

1. (Existence) For each batch operation in measurement, there is a streaming surrogate that recovers the batch object as a limit. Concretely, as the encoder capacity and training data grow, the streaming decoder converges in distribution to the offline posterior of the corresponding operation.
2. (Equivalence under stationarity) When the data-generating process is stationary and identifiable, the streaming and batch estimators agree on the same recovery functionals up to a finite-sample Monte Carlo gap.
3. (Strict dominance under non-stationarity) When the DGP is non-stationary, ability drifts, items drift, the population changes, the streaming decoder dominates the batch estimator on any time-localized functional, because the batch estimator does not have a within-time object to estimate.

The contribution is the body of equivalence and non-equivalence proofs and empirical demonstrations for a *connected* set of measurement operations under one architectural commitment. That is what makes it a theory and not a methodology. A methodology builds one system. A theory tells you which classical batch results survive the reduction, which fail, and why.

**Why "encoder-decoder" is load-bearing and not cosmetic.** A streaming primitive could be many things, a Kalman filter, an SMC particle system, an RNN, a transformer with KV cache. The thesis commits to encoder-decoder because every measurement operation decomposes the same way. The encoder produces a state sufficient for the *person side* (history of responses) and the *item side* (incoming item descriptor). The decoder reads off a measurement functional. Different operations differ only in the decoder head. The same encoder serves theta estimation, calibration, diagnosis, fit, and drift. This is the unification, one encoder, many heads, one training criterion that ties them.

---

## 2. The choice of operations

A thesis cannot contain all ten classical operations. The candidate is one person. I will argue inclusions and exclusions deliberately.

**Included (five thrusts).**

1. **Person measurement** (theta). The base case. MA-GPCM. Streaming reduction of marginal MLE for theta.
2. **Item calibration** (alpha, beta). Streaming reduction of online item calibration (OEM/MEM family, Wainer and Mislevy 2000, Ban et al. 2001, van der Linden and Ren 2020 for the Bayesian adaptive variant).
3. **Cognitive diagnosis** (skill mastery, Q-matrix). Streaming reduction of DINA/DINO and neural CD (Wang et al. AAAI 2020 NeuralNCD line).
4. **Drift and population change** (item parameter drift, ability drift, population shift). The frontier where the streaming primitive has nothing to compete against from the batch world. This thrust is where dominance is provable.
5. **Person-fit and aberrance** (lz, U3, response-time-aware fit). The streaming primitive makes within-session aberrance possible. Batch person-fit only fires post hoc.

**Excluded, with reasons.**

- **Linking and equating across forms or populations.** Excluded. Equating is fundamentally a *between-cohort* identification problem, not a streaming problem. Two cohorts that never co-occur in time cannot be linked by a streaming encoder without anchor items, and once you have anchor items linking reduces to concurrent calibration, which is already inside thrust 2. Putting equating in would either duplicate thrust 2 or open a between-cohort identification problem orthogonal to streaming. It is the cleanest exclusion.
- **DIF and measurement invariance.** Borderline. DIF asks whether item parameters differ across known groups. Reformulated as online change-point detection on per-item parameter traces, it becomes a special case of thrust 4 (drift). I fold it into thrust 4 rather than make it a separate thrust, because the streaming primitive for DIF is the same primitive as drift, namely sequential monitoring of the per-item decoder output against a null hypothesis of stationarity. Treating it as a separate thrust would be padding.
- **Item generation and bank growth.** Excluded. Item generation today means LLM-generated items. That is a generative-model thrust, not a measurement thrust, and it imports a whole different evaluation apparatus (item quality, content validity, fairness of generated content). It belongs in a different thesis. The thesis does say what *happens to* measurement when items arrive mid-stream (calibration of new items is thrust 2), but it does not generate them.
- **CAT and content sequencing.** Excluded. CAT is a policy problem. A measurement-only thesis stops at the posterior over theta. The moment you select the next item you are doing decision theory, not measurement. The candidate is explicitly excluding intervention/DRL from D2.
- **Intervention and recommendation.** Excluded for the same reason as CAT. Out of scope by user fiat.

Five thrusts. That is a defensible thesis size, three operations the candidate must complete and two more if time permits, with one (thrust 1) already done by MA-GPCM.

---

## 3. Where MA-GPCM sits

MA-GPCM is not just "the first chapter." It is the *existence proof* for the entire program.

The chain of reasoning. If even one batch operation, the most studied one, theta estimation, refused to admit a streaming reduction with batch-equivalent recovery, the thesis collapses on the first claim. MA-GPCM shows the reduction is possible for the base case. It recovers ground-truth theta, alpha, beta on synthetic data with single-pass inference. It does this with one encoder and one head, exactly the architectural shape the thesis commits to. That is why MA-GPCM is the floor.

What is *essential* about MA-GPCM for the thesis. The separated ability pathway (SIE), the encoder-decoder factorization, the single-pass evaluation criterion (within-session theta is read off without a second backward pass), and the empirical demonstration that streaming recovery matches batch EM on stationary DGPs and beats it on non-stationary DGPs (staircase, random walk, block).

What is *incidental*. The specific choice of DKVMN as the memory module. The specific ordinal head (GPCM vs GRM vs sequential). The specific embedding (LinDecay vs StaticItem vs Learned). The thesis must be robust to swapping these. If a transformer-based encoder beats DKVMN it is still the same thesis. The architectural commitment is encoder-decoder with a single-pass evaluation criterion, not DKVMN specifically.

This matters for the strict-reviewer attack in section 9. If a reviewer points at DKVMN and asks "why this memory module," the answer is "the choice is incidental, the thesis is about the encoder-decoder reduction, here are two follow-ups (thrust 2 and 3) where the encoder is different but the result holds."

---

## 4. The thrusts

### Thrust A. Streaming person measurement

**Batch baseline.** Marginal MLE or EAP for theta given calibrated item parameters, or full marginal MLE via EM (Bock and Aitkin 1981) for theta and items jointly.

**Streaming primitive.** Encoder $f_\phi$ maps response history to a state $h_t$. Decoder $q_\phi^{\theta}$ reads theta off $h_t$ at every step. Trained by likelihood on synthetic data with known generative parameters.

**Unification.** This is the canonical case of the encoder-decoder commitment. Every other thrust reuses $f_\phi$ and swaps the decoder.

**Open intellectual problem.** What is the *correct* identifiability discipline for streaming theta under non-stationarity. Batch theta is identified up to a global affine transform via anchor items. Streaming theta is identified up to *what*. The answer is non-trivial because the encoder can drift. The thrust must produce a formal statement.

**Success criteria.** (i) Recovery on stationary DGP matches batch EM within Monte Carlo error. (ii) Recovery on non-stationary DGP (staircase, RW, block) strictly dominates batch EM under any time-localized metric (within-session correlation, time-windowed RMSE). (iii) Single-pass inference cost is O(1) per response after training. (iv) Identifiability statement that pins down what "the right theta" means under streaming.

**Falsifier.** A stationary DGP on which streaming theta is biased relative to batch EM with no clear knob to remove the bias. Or, the gap on non-stationary DGPs vanishes when batch EM is run on sliding windows. Both would say streaming offers nothing over windowed batch.

**Sub-papers.**

*A1. MA-GPCM, the base case.* RQ: does a single-pass encoder-decoder recover GPCM parameters at parity with EM on stationary data and dominate on non-stationary data? Method, MA-GPCM with SIE on synthetic GPCM DGPs and ASSISTments. Dataset, the four synthetic DGPs and ASSISTments 2009. Intellectual contribution, this paper establishes that the streaming reduction exists for the foundational measurement operation. It is the first time the encoder-decoder commitment is shown to be both batch-equivalent on stationary problems and strictly better on non-stationary ones, with controlled synthetic DGPs that let the comparison be unambiguous. The paper introduces the within-session recovery metric that the rest of the thesis uses.

*A2. Identifiability of streaming theta.* RQ: what is theta identified up to in a streaming encoder. Method, formal analysis plus controlled experiments injecting known affine transforms into the DGP. Dataset, synthetic. Intellectual contribution, batch IRT has a clean identifiability theory (Bock and Aitkin, San Martin et al.). Streaming has none. The paper gives the first identifiability result for a streaming person-side estimator, showing theta is recovered up to a *time-varying* affine that is pinned down by item-side regularization. This result is reused throughout the thesis.

### Thrust B. Streaming item calibration

**Batch baseline.** Offline calibration on a fixed response matrix via marginal MLE. Online calibration in the CAT literature, OEM (Wainer and Mislevy 2000) and MEM (Ban et al. 2001), addresses item replenishment by anchoring new items to operational ones.

**Streaming primitive.** Items are first-class arriving objects. When a new item $j$ enters the bank, the encoder treats item-side parameters as an additional decoder head, $q_\phi^{\alpha, \beta}$, that conditions on the item descriptor and the response history of every student who has answered $j$. Calibration is single-pass over students for each new item, not a re-fit of the bank.

**Unification.** Same encoder $f_\phi$ as thrust A. The item-side decoder is a different head reading a different aspect of $h_t$, specifically the response prediction error at the item.

**Open intellectual problem.** Hot-start calibration. When item $j$ has been seen by ten students, by a hundred, by a thousand, the streaming decoder must produce a credible interval that is honest. The closed-form Cramer-Rao bounds for offline calibration do not apply because the streaming estimator is amortized. The thrust must produce calibrated uncertainty.

**Success criteria.** (i) For new items entering an established bank, streaming alpha and beta recovery matches MEM after the same number of responses. (ii) Streaming calibration produces credible intervals with empirical coverage near nominal on synthetic data. (iii) Throughput on Q in the thousands is order of magnitude better than MEM.

**Falsifier.** Streaming calibration is biased in the small-N regime (cold-start) and the bias does not shrink as N grows, or the uncertainty is miscalibrated by a factor that does not close with more data.

**Sub-papers.**

*B1. Streaming item calibration at parity with MEM.* RQ: can the streaming encoder calibrate new items as items arrive, with single-pass cost? Method, attach a calibration head to the MA-GPCM encoder. Synthetic streaming DGP that injects new items at known rates. Intellectual contribution, the paper closes the loop between MA-GPCM (which assumed calibrated items) and the item-side problem. It shows that the same encoder that produces theta also produces item parameters when given the right head. This is the first concrete instance of the "one encoder, many heads" claim that the thesis stands on.

*B2. Calibrated uncertainty for streaming items.* RQ: do streaming calibration intervals have the right coverage? Method, ensembling, MC dropout, and conformal post-hoc calibration applied to the calibration head, evaluated by empirical coverage on synthetic DGPs with known parameters. Intellectual contribution, batch psychometrics gives item parameter standard errors from the observed information matrix. The streaming literature has *no* honest counterpart. This paper provides one, and shows that under specified conditions the conformal procedure produces intervals indistinguishable from offline frequentist intervals. This is methodologically novel and is what makes streaming calibration deployable rather than just demonstrable.

### Thrust C. Streaming cognitive diagnosis

**Batch baseline.** DINA, DINO, G-DINA, neural cognitive diagnosis (Wang et al. AAAI 2020). All run on a fixed response matrix with a fixed Q-matrix. Skill profiles are estimated offline.

**Streaming primitive.** Replace the skill-profile-per-student decoder with a streaming head $q_\phi^{\text{skill}}$ that reads $h_t$ and emits a skill mastery vector $\boldsymbol{\alpha}_t \in [0, 1]^S$ updated every response. The Q-matrix is either known and fixed (the easier setting) or partially learned through soft attention over skills (the harder setting).

**Unification.** Same encoder. The skill decoder is a multi-output sigmoid head with a Q-matrix mask. Cross-thrust property, the skill mastery trace must be consistent with the theta trace from thrust A, because if a student masters more skills their ability on items requiring those skills should go up. This *cross-decoder consistency* is what makes thrust A and thrust C parts of the same thesis rather than two separate models.

**Open intellectual problem.** How do you reconcile a continuous theta with a binary skill profile when both come from the same encoder. The classical literature treats CDM and IRT as alternatives. The streaming primitive can produce both from one state. The thesis must address whether this is a feature (richer measurement) or a bug (overparameterization and identifiability collapse).

**Success criteria.** (i) Skill mastery recovery on synthetic DINA data is at parity with offline DINA EM. (ii) Within-session skill trajectories make pedagogical sense on real data (a student answers a sequence of items requiring skill $s$ correctly, mastery of $s$ rises monotonically). (iii) Cross-decoder consistency holds, predicted item probability from theta-decoder and from skill-decoder agree within bounds.

**Falsifier.** The two decoders systematically disagree on item probability, and forcing agreement degrades both. That would say the encoder cannot serve both heads simultaneously without identifiability loss, and the unification claim fails in this thrust.

**Sub-papers.**

*C1. Streaming DINA with shared encoder.* RQ: does a shared encoder produce both theta and skill profile traces consistently? Method, dual-head model trained jointly on synthetic data with both DINA and IRT-compatible generative parameters. Intellectual contribution, this is the first demonstration of *cross-paradigm consistency* in measurement. Batch psychometrics treats IRT and CDM as competing paradigms with different fit indices. The streaming primitive forces them into one architecture and asks whether they can coexist. The paper either shows they can (vindicating the unification) or shows where they collide (a genuine theoretical finding either way).

*C2. Streaming Q-matrix learning.* RQ: can the Q-matrix be inferred online as items arrive? Method, sparse attention over skills with $L_1$ regularization, validated on synthetic data with known Q. Intellectual contribution, offline Q-matrix learning (de la Torre, Liu et al.) is a major open problem with mixed success. Streaming Q-matrix learning has not been seriously attempted. The paper does not promise to solve it, but it formulates it correctly in the streaming language and provides the first reproducible benchmark with controlled DGPs.

### Thrust D. Streaming drift and DIF as one operation

**Batch baseline.** Item parameter drift detection by chained equating residuals (Donoghue and Isham 1998, Han 2012). DIF detection by Mantel-Haenszel, IRT-LRT, Wald tests, all on a fixed response matrix with a fixed group label. None of these are sequential procedures.

**Streaming primitive.** Run thrust B continuously. The calibration head produces a parameter trace $(\alpha_t, \beta_t)$ for every item $j$. A change-point detector $g_\phi$ monitors these traces against a stationarity null. DIF becomes a special case where the "groups" are time windows or response-conditional. Population shift becomes a change-point on aggregate statistics of $h_t$.

**Unification.** Same encoder, same item-side decoder. Add a sequential test on the decoder outputs. The test is generic, not psychometric, and reuses online change-point detection theory (Page CUSUM, Lorden, more recent literature including the arXiv 2006.03283 line).

**Open intellectual problem.** The batch world has no notion of *within-deployment* drift detection. There are no benchmarks, no agreed-upon metrics, no ground-truth DGPs. The thrust must construct the evaluation apparatus that did not previously exist. This is risk and opportunity in one package.

**Success criteria.** (i) On synthetic DGPs with injected drift, the change-point detector achieves near-optimal ARL (average run length) tradeoffs. (ii) On real data (ASSISTments, EdNet) the detector flags item parameter shifts that align with platform-level events (curriculum changes, item edits) verifiable from metadata. (iii) DIF detection on standard benchmarks (PISA released items) matches MH and IRT-LRT in recovery while running in a single pass.

**Falsifier.** The detector flags either nothing useful or too many things, and there is no operating point that competes with offline DIF/IPD methods. Or, recovered drift on real data has no platform-level corroboration, suggesting the detector is reading encoder noise.

**Sub-papers.**

*D1. Sequential item parameter drift detection.* RQ: can we detect IPD as it happens rather than at the end of the test window? Method, online CUSUM on $\beta_t$ traces from thrust B. Intellectual contribution, this paper introduces the first end-to-end pipeline for within-deployment drift detection. The thesis-level claim it supports is that drift is a *streaming-native* operation that has no honest batch counterpart, which is the strongest part of the dominance claim in section 1.

*D2. DIF as a special case of streaming drift.* RQ: does the same streaming primitive recover classical DIF results? Method, condition the change-point detector on a group indicator, evaluate against MH and IRT-LRT on standard benchmarks. Intellectual contribution, this paper unifies two previously separate literatures (drift monitoring and DIF analysis) under one operation. It also is the first DIF method that runs in a single pass, which has operational implications for live tests.

### Thrust E. Streaming person-fit and within-session aberrance

**Batch baseline.** lz (Drasgow), U3, and similar person-fit indices, all computed after the test on the whole response vector. They tell you that a person responded aberrantly, but not when.

**Streaming primitive.** Person-fit becomes a function $\text{fit}_t(h_t)$ that fires when the encoder's prediction over $y_t$ disagrees with the realized response by more than a calibrated threshold. The threshold is set by the same sequential testing machinery as thrust D.

**Unification.** Same encoder. The fit head is the simplest of all the decoders, a likelihood-ratio between the current $h_t$ and a null model.

**Open intellectual problem.** What is the *right* null for streaming person-fit. Batch person-fit uses the IRT model under the estimated theta as the null. Streaming has the same null on a moving target. The thrust must work out the analog of lz under streaming, which has not been done.

**Success criteria.** (i) Recovers known aberrant patterns (cheating, careless responding, guessing) at parity with lz on retrospective evaluation. (ii) Detects aberrance within the session, not just at the end. (iii) Calibrated false-positive rate.

**Falsifier.** The streaming statistic is unable to separate aberrance from encoder uncertainty in the early part of a session, and the only way to fix it is to wait until the session is long enough, which collapses the streaming advantage.

**Sub-papers.**

*E1. Streaming lz analog.* RQ: is there a sequential test statistic that recovers retrospective person-fit and adds within-session aberrance detection? Method, derive the streaming likelihood ratio under the encoder-induced null, evaluate on simulated cheating and careless-responding DGPs. Intellectual contribution, the paper closes the gap between retrospective person-fit (which knows everything happened but tells you only after the fact) and live monitoring (which has only been done with ad-hoc rules in industry). It establishes that the encoder-decoder framework gives person-fit for free, because the fit statistic is just a head on the same encoder.

---

## 5. The unification

Five thrusts. Why is this one thesis.

**Shared architecture.** One encoder $f_\phi$ trained once. Five decoder heads $\{q_\phi^{\theta}, q_\phi^{\alpha, \beta}, q_\phi^{\text{skill}}, q_\phi^{\text{drift}}, q_\phi^{\text{fit}}\}$. The same state $h_t$ feeds all of them. The encoder is the load-bearing object. The thesis stands or falls on whether one $h_t$ can serve all five.

**Shared evaluation.** Every thrust is evaluated on a common synthetic-DGP framework with three properties. (i) Known ground-truth parameters so recovery can be measured. (ii) Controllable non-stationarity so the dominance claim can be tested. (iii) A real-data complement so external validity is checked. The candidate has already built four DGPs for thrust A (v2, staircase, random walk, block). The thesis ships a DGP suite where each thrust has its synthetic ground truth.

**Shared theoretical commitments.** Three claims that bind the thrusts together.

1. (Existence) For each operation, the streaming surrogate exists and is consistent.
2. (Batch equivalence under stationarity) Under the right conditions, streaming and batch agree on recovery.
3. (Strict dominance under non-stationarity) When the DGP moves, streaming has objects to estimate that batch does not.

Every thrust restates these three claims in its own decoder. The thesis is a tour through five operations under one theoretical pattern. That is what "unified theory" means here. Not "the same model does five things." The same *reduction* applies to five things.

**Shared identifiability discipline.** Thrust A produces an identifiability statement for streaming theta. Thrusts B, C, D, E inherit it. If theta is identified up to a time-varying affine, item parameters in thrust B are identified up to the inverse transform, skill profiles in thrust C are identified up to a Q-matrix-mediated rotation, drift in thrust D must be invariant to the affine, and fit in thrust E must be too. The five thrusts are not independent on the identifiability question. They form one connected geometry.

This last point, identifiability propagation, is the strongest argument that this is theory and not architecture. If it were just architecture, the thrusts could be developed independently. Because the identifiability statement in A propagates constraints to B, C, D, E, the thesis is one object.

---

## 6. Venue strategy

Five thrusts, eight to ten sub-papers. The candidate cannot run them all to completion. Realistic plan, six papers across four years with two more if time permits.

| Sub-paper | Year | Venue | Reason |
|-----------|------|-------|--------|
| A1 MA-GPCM | submitted | IJAIED or BJET | AIED journal, foundational empirical paper, longer form |
| A2 Identifiability of streaming theta | year 2 | Psychometrika or BJMSP | Identifiability is a psychometrics-journal home, and a Psychometrika paper anchors the thesis's theoretical credibility |
| B1 Streaming calibration | year 2-3 | AIED conference or EDM | Methodological paper, conference home, broad community |
| B2 Calibrated uncertainty | year 3 | NeurIPS or ICML if framed as conformal/uncertainty, otherwise Psychometrika | ML venue if the conformal calibration result is sharp, psychometrics if the result is more applied |
| C1 Streaming DINA | year 3 | AAAI or LAK | NCDM's home is AAAI, LAK is the educational analytics venue |
| D1 Sequential drift | year 4 | EDM or Educational Measurement: Issues and Practice | EDM for the algorithmic paper, EM for the practitioner story |
| D2 DIF as streaming | year 4 (stretch) | Applied Psychological Measurement | DIF's traditional home |
| E1 Streaming person-fit | year 4 (stretch) | Psychometrika | The lz analog result is a theoretical contribution |

The venue logic. The thesis is AIED-primary, so AIED-conference / IJAIED / EDM / LAK / BJET take the empirical and applied work. Psychometrika takes the two papers with formal results (A2 and E1, plus possibly B2). ML venues are an option only when the result transcends the educational application, B2 is the most likely candidate because conformal calibration on amortized inference is of interest to the wider ML community.

The candidate gets one strong psychometrics paper (A2 or E1), three AIED/EDM papers (A1, B1, C1 or D1), and a stretch into ML if B2 lands. That is a defensible portfolio.

---

## 7. Risks and falsifiers

The thesis collapses or partially collapses under any of the following.

**Catastrophic risks.**

1. **Streaming theta is biased relative to batch EM on stationary DGPs and cannot be fixed.** Thrust A fails. The whole program loses its base case. Mitigation, MA-GPCM already shows this does not happen on the easiest case, so the catastrophic version of this risk is already retired.
2. **Cross-decoder consistency fails in thrust C.** The encoder cannot serve both IRT and CDM heads. The unification claim weakens from "one encoder for everything" to "one encoder for these three things." The thesis survives but the strongest version of the claim is gone. Mitigation, if this happens, restate the thesis with a smaller unified claim, four thrusts instead of five.
3. **No identifiability statement is provable in thrust A.** A2 cannot be written. The thesis still ships an empirical program but loses the theoretical anchor in psychometrics. Mitigation, identifiability under partial assumptions is still publishable, and several recent papers (San Martin et al., Casabianca and Lewis on weak identifiability) show how to write such papers without a full uniqueness theorem.

**Empirical risks that determine the result, not collapse the thesis.**

4. **The dominance claim under non-stationarity is real but small.** Streaming beats windowed batch EM by 5 percent of QWK, not 30 percent. The thesis is still correct but less exciting. The narrative becomes "streaming is the natural primitive, and on harder problems the gap widens," with thrust D supplying the harder problems.
5. **Real-data corroboration of drift in thrust D fails.** ASSISTments and EdNet do not give clean platform-level events that align with detected drift. The drift thrust becomes a synthetic-only contribution. The thesis ships but D1 is weaker.

**Open intellectual questions whose answers determine whether the unified theory holds.**

- Is streaming theta identified up to *a time-varying* affine, a *constant* affine, or something else entirely. The answer determines what A2 says and constrains what every other thrust can claim.
- Does cross-decoder consistency in thrust C require a hard constraint (probabilities must agree) or a soft one (probabilities are close in expectation). Hard constraints might be infeasible, soft constraints might let the heads drift apart.
- Is there a streaming analog of the observed information matrix that gives honest standard errors. If yes, thrust B's uncertainty story is on solid ground. If no, the thesis ships with conformal-only intervals, which is defensible but less elegant.

---

## 8. Defense against the strict-reviewer attack

A strict reviewer will say "you have not built a unified theory, you have built a unified architecture, which is a methodology not a theory." Here is the answer.

The objection treats "theory" and "methodology" as mutually exclusive. They are not. Plenty of accepted theories in computational fields are theories about what a class of methods can and cannot do. PAC learning is a theory about a class of algorithms. The minimax theorem is about a class of strategies. Stochastic approximation is a theory of online estimators. These are all "theories of methods."

The thesis is a theory in that sense, with three specific properties that elevate it above a methodology.

**Property one. It makes equivalence and non-equivalence claims that can be proved or refuted.** Existence, batch equivalence under stationarity, strict dominance under non-stationarity. These are mathematical statements with truth values. A pure methodology builds a system and says "it works." A theory makes claims that survive or fall on evidence and proof.

**Property two. It propagates constraints across instances.** The identifiability statement in thrust A constrains what thrusts B through E can possibly claim. A methodology does not have this structure. Five independent papers each calling their model "streaming X" would not have propagating constraints. The thesis does, because identifiability is a *theoretical* object that lives at the level of the encoder, not at the level of the head.

**Property three. It predicts new operations that have not yet been instantiated.** If the reduction is correct, every batch psychometric operation should have a streaming counterpart. The thesis instantiates five. The reviewer who says "show me a theory not a methodology" can be answered with "here are two operations the thesis did not have time to instantiate (linking across populations, item generation), here are the predictions the framework makes about what those streaming surrogates would look like, and here is what would make those predictions wrong." A methodology cannot do this. Methodologies are descriptive of the systems they build. Theories are predictive of systems not yet built.

If the reviewer pushes harder, "but you have only empirical evidence for five operations, not a closed-form theorem about all ten," the answer is that this is the normal state of a thesis in a computational measurement field. Psychometric theory itself is built this way, the Rasch family of theorems holds under specific assumptions, the rest is empirical. The thesis follows the same standard.

The final move, if the reviewer is still skeptical, is to point at the alternative. The alternative is to give up on unification and ship five independent papers each titled "Streaming X." That body of work would *be* a methodology, five tools without a shared theoretical commitment. The thesis as proposed *is* the theory, because it commits to the unification and makes claims that can be tested against it. The reviewer can disagree with the verdict on each claim, but cannot honestly say "this is not a theory."

---

## 9. What I think is hard, novel, and risky

My own read.

**The hardest part.** Thrust A2, identifiability of streaming theta under non-stationarity. Batch IRT identifiability is mature. The streaming version has no precedent I am aware of, and the answer is not obvious. The encoder can absorb arbitrary affine transformations into its weights, which means theta is only identified relative to the encoder, not absolutely. Writing a clean identifiability statement that is both true and useful is the single biggest piece of theoretical work in the thesis.

**The most novel part.** Thrust D, drift detection as a first-class measurement operation. The batch world has no honest competitor here. The thesis can show dominance unambiguously because the comparison is to "nothing exists." The risk is that without a competitor the evaluation apparatus has to be built from scratch, which is exactly what makes it both novel and risky.

**The most fragile part.** Thrust C cross-decoder consistency. I would not bet a thesis on this working out cleanly. If I were the candidate I would frame thrust C with two acceptable outcomes from the start, either the heads are consistent (vindication) or they are not (a theoretical finding about IRT-CDM incompatibility under shared encoding). Both outcomes are publishable. Both support the broader thesis. The framing must be hedged from day one.

**The thing the user should worry about most.** Time. Five thrusts in four years is tight. The realistic plan is to complete A and B, ship one paper in C or D, and have E as a stretch. The thesis remains a unified theory with three to four instances rather than five. The candidate should plan for that compression now, before year three when the choice has to be made.

**The thing the user should be most excited about.** The thesis offers something nobody in the AIED or psychometrics literature has put forward, a coherent computational reframing of the entire measurement stack under one architectural commitment with one identifiability discipline. Papers in the area have done streaming versions of single operations. None have argued for the unification. If the candidate gets even three of the five thrusts to land convincingly, the thesis stakes out a distinctive identity, "the researcher who reframed batch measurement into streaming primitives," and the remaining operations become a multi-decade research agenda the candidate owns.

---

Word count, approximately 4600.
