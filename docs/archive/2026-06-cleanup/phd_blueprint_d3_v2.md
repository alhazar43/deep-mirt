# PhD Blueprint D3, version 2

Streaming as the computational paradigm for measurement in the AI era.

---

## 1. The paradigm claim, argued

The first thing to admit is that "streaming" is a worn word. Machine learning has streaming losses, online optimization, continual learning. Statistics has sequential analysis and online Bayesian filtering. The claim here is not that streaming computation is new. The claim is that *measurement*, as a scientific practice with its own validity standards, has not yet been reorganized around streaming primitives, and that the AI era forces this reorganization.

Classical measurement, as it exists today, was designed when three things were true at once. Items were authored and pretested in long cycles. Respondents arrived in waves, took a fixed form, and produced a closed response matrix. The estimation step happened offline by EM or MCMC, was published, and froze the calibration until the next cycle. Almost every guarantee of measurement validity, classical reliability, factor structure, measurement invariance, differential item functioning, equating, person fit, was developed inside this batch frame.

The AI era violates all three premises simultaneously. Items can be generated on demand by LLMs and inserted live into a bank, with validity that is itself uncertain at the moment of insertion. Respondents do not assemble into closed cohorts, they pass through the system continuously and at different rates. The estimation step has no natural offline boundary because the data never closes. Even the construct itself can drift, because what a survey item means changes when the population changes, when the wording shifts under translation models, or when respondents interact with AI assistants while answering.

So the paradigm claim has a precise content. It says four things.

First, the *unit of computation* for measurement should be the arrival of a single response, not the closure of a response matrix. Every method, every estimator, every diagnostic, every validity check, should have a streaming form that updates on each new response and is consistent with the batch form in the limit of fixed data.

Second, the *unit of guarantee* should be a per-instant guarantee, not a per-cohort guarantee. Reliability at a moment, identifiability under current data, invariance up to time t. The batch frame produces guarantees that hold only after the cohort is closed, which is fine when the world also closes and reopens on the same calendar, and useless when it does not.

Third, the *unit of accountability* must include change detection as a first-class operation. In batch psychometrics, drift is a nuisance that violates the invariance assumption. In streaming, drift is a measured quantity, with its own estimator and its own uncertainty. The system reports not only "this is the score" but "this is the score and these are the items, populations, or constructs whose meaning has shifted since you last looked."

Fourth, the *unit of intervention* should be the same as the unit of computation. When measurement is streaming, the natural action object is also streaming, an adaptive next item, a recommendation, a tailored prompt, an alert that says invariance has broken. Measurement and action collapse into a closed loop on a single time axis.

What the claim does NOT say is also worth pinning down. It does not say that batch methods are wrong. EM on a closed dataset will remain a benchmark and a reference. It does not say that streaming is faster or cheaper, often it is not. It does not say that all measurement problems are streaming problems, some are inherently cross-sectional, like a one-time validation study. It says, narrowly, that the *load-bearing* measurement problems of the AI era are streaming, and that the field needs a coherent set of primitives, not a patchwork of online-IRT papers tacked onto a batch culture.

The reason this is a paradigm claim and not merely an engineering claim is that it changes what counts as a finished result. Under the batch frame, a finished result is a calibrated bank and a validated scale. Under the streaming frame, a finished result is a *running system* with stated per-instant guarantees and explicit drift telemetry. Those are different scientific objects.

## 2. Why this thesis must reach beyond education

If the thesis is just "streaming primitives for educational measurement," it is D2. D2 is defensible, conservative, and tight. It is also a missed opportunity, for the following reason.

The intellectual move that makes streaming measurement a paradigm and not a trick is that it identifies a *common computational shape* across domains that classical psychometrics, by sitting in separate journal communities, did not see was the same shape. Health PROs, vocational skill states, formative classroom signals, longitudinal personality, cognitive screening, professional certification, all of them are running the same operation, repeated ordinal or numeric responses under an assumed latent process, with item parameters and population structure both potentially drifting. The reason this is not visible is sociological, not technical. The psychometrics of PROMIS lives in health journals, the psychometrics of skill tests lives in I-O psychology, the psychometrics of KT lives in AIED and EDM. The candidate has the technical training to see across them. That is the leverage.

Going broader buys two things. It buys *generality of the primitive*, because if a streaming primitive works only on ordinal KT it is a specialized algorithm, while if the same primitive works on PROMIS short forms and on a vocational skill bank, it is a paradigm-level contribution. It also buys *evidence of paradigm necessity*, because if the same problem (drift, identifiability under continuous arrival, validity-at-time-t) recurs in unrelated fields, that is evidence that the batch frame is the limiting factor, not the domain.

The cost is real. Going broader risks the strict-reviewer attack discussed in section 9. It risks intellectual dilution, where the candidate ends up shallow in three fields. It also risks domain-credibility loss, because writing a health-measurement paper requires understanding clinical validity in a way that an AIED background does not automatically supply. The defense is in section 8.

The cleanest answer to "why broader" is this. The candidate is staking a claim on a *computational primitive*. Computational primitives are validated by the breadth of their applicability. A streaming-measurement primitive that runs only on KT is, in evidential terms, no different from an online-IRT paper, of which there are dozens. A streaming-measurement primitive that crosses fields and exposes a shared structure that the fields themselves had not seen is a paradigm contribution. The PhD has to do the second thing to justify the framing.

## 3. The thrust structure, derived

The structure of the thesis falls out of the paradigm claim above. There are five thrusts, not chosen by topical convenience but by what each of the four content points in section 1 demands, plus one out-of-domain thrust as required by D3.

### Thrust A. Streaming person measurement (the foundation)

The reason this thrust exists is that the paradigm cannot start anywhere else. If you cannot do streaming person measurement under known item parameters, you cannot do anything downstream. MA-GPCM lives here, scoped to the easiest case of ordinal KT.

What would need to be shown. That a streaming estimator of person ability under a defined response model is at least as accurate as offline EM on the same closed data, and produces a usable trajectory under non-stationary ability, and gives per-instant uncertainty that is calibrated against held-out responses. The third condition is the hardest and is what distinguishes a streaming measurement method from an online prediction method.

What falsifies it. If, under realistic data densities and ability drift, the streaming method's per-instant uncertainty intervals do not contain the held-out next response at the nominal rate, the thrust has failed. Predictive accuracy alone is not enough, the paradigm claim is about *measurement*, which means uncertainty about the latent quantity has to be honest, not just point predictions.

### Thrust B. Streaming item and population calibration

The reason this thrust exists is that in the AI era item banks are not stable. Items are added by LLMs, edited by humans, retired when they fail, and translated across populations. Item parameters are no longer something you calibrate once and freeze, they are a process. The same is true for population structure, demographic mix changes continuously on a platform.

What would need to be shown. A streaming joint estimator for items and persons that handles items entering and leaving the bank live, with explicit identifiability under continuous arrival, and with a defensible answer to "when has an item drifted enough that it should be recalibrated or retired." This last is the technical content. Drift detection on item parameters under a latent-variable model where the latent is itself drifting is a hard estimation problem and not one classical psychometrics solved, because it solved a static version of it.

What falsifies it. If, on realistic mixed-source banks (some LLM-generated, some human-authored), the streaming calibration cannot recover known item parameters under controlled drift injection, or cannot distinguish drift in items from drift in population, the thrust fails. The diagnostic of failure here is *confusion of sources*, the estimator says items drifted when populations drifted, or vice versa.

### Thrust C. Streaming validity, invariance, and fairness

The reason this thrust exists is that under the paradigm, every quality you used to certify offline now has to be certified continuously. Measurement invariance, differential item functioning, person fit, equating, all of them need streaming counterparts. This is the thrust where streaming measurement makes contact with the algorithmic fairness literature, because measurement invariance is the field's native language for fairness across groups, predating ML fairness by decades.

What would need to be shown. A streaming test of measurement invariance with stated false-alarm and miss rates under continuous arrival, a streaming DIF detector that does not require the analyst to declare focal and reference groups in advance, and a streaming person-fit signal that flags aberrant respondents in real time without inflating false flags during ordinary ability change. These are statistical tasks with rigorous offline forms and currently no satisfactory streaming forms.

What falsifies it. If the streaming invariance test has uncontrolled type-1 inflation when the population mixes, or if it misses real DIF that the offline LRT picks up, the thrust fails. This is the most statistically delicate thrust and the one where shortcuts will be most tempting.

### Thrust D. Streaming intervention and generation

The reason this thrust exists is that measurement under the new paradigm is not free-standing. Adaptive testing, item generation, recommendation, all of them sit on top of measurement and run on the same time axis. The intervention object becomes a per-instant policy that uses the streaming estimate and feeds back into the data stream. This is where the candidate's DRL background is load-bearing.

What would need to be shown. That a streaming intervention policy, built on top of the streaming measurement estimator, achieves the same or better adaptive efficiency than offline-calibrated CAT, that LLM-generated items can be inserted into the live bank with bounded validity loss, and that the closed loop between measurement and intervention does not destabilize the measurement estimator itself. The last point is the genuine intellectual problem here. When the intervention shapes the data that the measurement consumes, identifiability is no longer free.

What falsifies it. If closing the loop produces feedback pathologies, ability estimates that drift because of the policy rather than because of the learner, the thrust fails. This is a real risk, not a contrived one, the recommender-systems literature is full of such pathologies.

### Thrust E. The out-of-domain thrust

Treated separately in section 5. Its reason for existing is the paradigm claim itself, which is not earned without out-of-domain evidence.

I considered a sixth thrust on theoretical foundations, a kind of "what is the asymptotic theory of streaming psychometrics." I decided against carving it out as a separate thrust because the theory should be embedded in thrusts A through C, not pulled into its own silo. A separate theory thrust signals that the candidate is uncertain whether the application thrusts will hit, which they will. Treating theory as integrated rather than separate is the right structural choice.

## 4. Where MA-GPCM sits

MA-GPCM is the canonical first example under Thrust A. It is essential in three respects and incidental in two.

Essential, first, because it shows that the streaming primitive can match EM on recovery, which is the necessary parity condition. The paradigm dies on first contact if the streaming version is just worse than the batch version on the batch's own home turf. MA-GPCM clears this bar.

Essential, second, because it demonstrates that streaming measurement can do something EM cannot, namely produce a within-student trajectory of ability that is usable per-instant. The trajectory is what unlocks the rest of the paradigm. If you cannot answer "what is theta at time t" you cannot answer any of the downstream questions about drift, validity, or intervention.

Essential, third, because it operationalizes the *separation* between item structure and ability structure (the separated ability pathway). That separation is a small architectural commitment in MA-GPCM but is conceptually load-bearing, because under the paradigm, items and persons must be allowed to drift independently. A model that entangles them at the representation level cannot support the rest of the program.

Incidental, first, the ordinal response specifically. The GPCM is not the point. The point is a defined response model with a known offline calibration target. Binary, ordinal, continuous, count, all admit the same streaming construction.

Incidental, second, the educational setting. KT is the testbed because the candidate already has it, the synthetic data is rich, and EM is a clean baseline. There is nothing in MA-GPCM's design that ties it to education.

How does the candidate generalize from here. The plan is to recast MA-GPCM at the abstract level as "a streaming estimator for a latent quantity under a defined polytomous response model with a memory of item interactions." Once this abstraction is named, the same architecture, with appropriate response head and appropriate item representation, ports to PROMIS-style health responses, to skill-tag responses on a vocational platform, and to formative classroom signals. The port is not free, every domain forces a new technical problem (different category-collapsing behavior for sparse PRO responses, different item-bank dynamics for vocational tags), but those problems are the substance of the thesis, not obstacles to it.

## 5. The out-of-domain thrust, chosen and argued

I considered four serious candidates. Patient-reported outcomes via PROMIS-like CAT systems. Labor and vocational skill measurement. Simulation-based assessment in professional certification. Cognitive screening in aging populations. Each is intellectually rich.

I think the right choice is **patient-reported outcomes (PROMIS-style longitudinal symptom and function measurement)**. The argument is as follows.

PROs are the closest non-education domain where the measurement infrastructure already runs on IRT and CAT and where the streaming problem is *currently unsolved and clinically pressing*. PROMIS calibrated item banks across symptom domains (pain, fatigue, depression, physical function) and built CAT delivery on top. The calibration assumed cross-sectional invariance and stable population structure. The clinical use is increasingly longitudinal, the same patient is measured monthly across years, with disease progression, treatment effects, and aging all driving the latent quantity, and with item parameters subject to drift across calendar time, across translation versions, and across modes of administration (paper, web, phone, voice).

The clinical community knows this is a problem and the methodological community is producing papers on longitudinal measurement invariance and dynamic IRT, but the solutions are still batch in structure. They re-run invariance analyses on closed cohorts and they retrofit longitudinal models onto fixed datasets. There is no running system that tells a clinician at visit t whether the depression scale has drifted for this patient's cohort.

The technical problems that PROs create that ed-AI did not.

First, *sparsity per person is high and bursty*. A KT learner produces hundreds of responses per week. A PRO respondent produces five to twelve items per visit, with visits weeks or months apart. The streaming estimator must extract reliable trajectory from very low per-person data density. This forces a stronger commitment to population priors and hierarchical structure than KT requires.

Second, *the construct itself is contested*. KT models a relatively well-defined latent ability under a known curriculum. PROMIS constructs (depression, fatigue) are theoretical objects with real validity debate. Streaming measurement of a construct under epistemic uncertainty about the construct is harder than streaming measurement of a defined ability. The technical implication is that the thesis must distinguish drift-in-parameters from drift-in-construct-meaning, and produce diagnostics for both.

Third, *mode and translation invariance are first-class concerns*. A patient may answer the same scale in English on web, in Spanish on phone, and in voice via an AI agent. Each mode is a potential source of DIF. The streaming invariance test has to be sensitive to mode and translation, not just population.

Fourth, *the stakes are individual and clinical*. KT scores rarely have direct clinical action attached to them. PRO scores route patients into trials, into adjusted dosing, into mental-health referrals. The streaming per-instant uncertainty must therefore be honest enough to support a referral decision. This connects streaming measurement to clinical decision-support literature in a serious way.

Fifth, *the data ecosystem is institutional*. PROs live in EHRs, in trial registries, in patient portals. The streaming system has to interoperate with FHIR, with REDCap, with consortium-level data sharing protocols. This is unglamorous engineering work but it is what separates a paper from a research program.

The research program I would lay out, in the language of the thrust.

Paper E.1. A streaming PROMIS-style ability estimator across short-form longitudinal data, validated against retrospective trajectories on a public PROMIS dataset (HealthMeasures supplies several), with comparison against the standard PROMIS scoring service. Contribution, parity at the batch boundary and superior per-visit uncertainty.

Paper E.2. Streaming measurement invariance across modes, with a synthetic-translation mode-switching protocol and a real cross-mode dataset if obtainable. Contribution, a streaming DIF detector specialized to mode and translation, with stated operating characteristics.

Paper E.3. Closed-loop adaptive PRO administration. Combines Thrust D and Thrust E. Builds an adaptive CAT that uses the streaming estimator and selects the next short form under per-visit time budgets, with bounded validity loss certified at the patient level. This is the paper that becomes clinically interesting, because per-patient adaptation is what clinicians want and what offline CAT cannot deliver.

Paper E.4 (optional, stretch). Construct-drift detection. Distinguishes a patient whose depression scale items are functioning differently from a patient whose depression itself has changed. This is conceptually the deepest paper in the thesis. It requires distinguishing measurement drift from construct drift, which is the hardest distinction in psychometrics and one the field has not solved in streaming form.

The reason PROs beat vocational skill measurement, despite the candidate's DRL/vocational background, is that the vocational measurement field does not have a mature batch baseline to dethrone. PROMIS does. Replacing or extending PROMIS streaming is a clean, falsifiable claim with an existing community of evaluators. Vocational skill measurement does not yet have the same kind of consensus measurement instrument, which makes the contribution harder to certify. The candidate's vocational background can show up in Thrust D as the source of intervention-policy expertise, without needing to be the out-of-domain claim.

## 6. Concrete sub-paper sketches, by thrust

Three to four papers per thrust. Method and dataset one sentence. Contribution one paragraph.

### Thrust A. Streaming person measurement.

A.1. MA-GPCM, the foundation paper. Method, separated-ability streaming GPCM with memory. Dataset, synthetic GPCM and ASSISTments. Contribution. Establishes that a neural streaming estimator matches offline EM on recovery and exceeds it on per-instant trajectory accuracy under non-stationary ability. The intellectual claim is that streaming measurement is not a degraded form of batch measurement under non-stationarity, it is the appropriate form, and batch is the degenerate case when ability is static.

A.2. Calibrated uncertainty for streaming ability. Method, conformal or martingale-style sequential intervals layered on the streaming estimator. Dataset, ASSISTments and a held-out synthetic non-stationary benchmark. Contribution. Per-instant uncertainty for latent quantities is currently absent in KT and weakly developed in psychometrics. The paper supplies a coverage guarantee under distribution shift, which is what clinicians and educators actually need when they look at an ability score.

A.3. Continuous-time streaming. Method, an irregular-time variant of the streaming primitive that respects inter-response intervals rather than treating them as uniform. Dataset, real KT logs with timestamps. Contribution. Most KT models implicitly assume uniform spacing. Real interactions are bursty. The paper shows that respecting time gives better trajectory and connects the streaming primitive to the continuous-time IRT literature in health.

### Thrust B. Streaming item and population calibration.

B.1. Joint streaming calibration with item entry and exit. Method, factorized streaming estimator with item-state hidden Markov for entry/exit. Dataset, simulated bank with controlled entry rates plus a real platform log with known item retirement events. Contribution. The first calibration method that respects item lifecycle as a first-class part of the model rather than a data-cleaning step.

B.2. Distinguishing item drift from population drift. Method, identifiability analysis plus a streaming joint detector. Dataset, controlled-injection synthetic and a real demographic-shift cohort. Contribution. Identifies the *confusion regime* where item drift and population drift are mathematically indistinguishable from response data alone and shows what auxiliary signal (item authoring metadata, demographic covariates) suffices to disentangle them. This is the paper that turns "things drift" into a structured estimation problem with stated solvability conditions.

B.3. LLM-generated item calibration. Method, treat LLM-authored items as items with informative priors derived from the LLM itself, calibrate streamingly against incoming responses. Dataset, an LLM-generated bank plus crowdsourced responses. Contribution. Bridges automatic item generation (active in 2024 to 2025) and streaming calibration. The intellectual claim is that LLM-authored items should not be treated as black-box items to be re-calibrated from scratch, the LLM is itself a source of prior information whose quality is itself a measured quantity.

### Thrust C. Streaming validity, invariance, fairness.

C.1. Streaming measurement invariance test. Method, sequential likelihood-ratio test on item parameters across a moving cohort window with controlled type-1 error. Dataset, simulated population-mixing scenarios plus a real cohort with known demographic shift. Contribution. The first invariance test designed for continuous arrival rather than two closed cohorts. The technical core is establishing the sequential test's operating characteristics under latent-variable identifiability.

C.2. Streaming DIF discovery. Method, online detection of items whose parameters condition on group, without analyst-specified focal groups. Dataset, public datasets with known group structure plus a synthetic benchmark. Contribution. Moves DIF from a confirmatory analysis to a discovery operation, which is what the streaming frame requires. The conceptual move is from "test this item against this group" to "tell me which items are exhibiting group-dependent functioning right now."

C.3. Streaming person-fit. Method, sequential test for individual response patterns that deviate from the model. Dataset, synthetic aberrance injections plus real high-stakes test data. Contribution. Person-fit in classical psychometrics is a post-hoc index. In a streaming system, person-fit is a real-time anomaly signal that can flag aberrance during the response sequence, with applications to test security and to detecting AI-assisted cheating, which is itself an AI-era problem.

### Thrust D. Streaming intervention and generation.

D.1. Closed-loop CAT under streaming measurement. Method, adaptive item selection policy on top of the streaming estimator. Dataset, simulated CAT benchmarks and a fielded test if available. Contribution. Demonstrates that adaptive testing built on streaming measurement does not destabilize the estimator and achieves adaptive efficiency comparable to or better than offline-calibrated CAT. The intellectual claim is that streaming measurement and adaptive testing are not separate problems, they are the same problem viewed from two sides.

D.2. Streaming item generation with quality guarantees. Method, LLM-generated items inserted into a live bank under a quality gate that requires posterior validity above a threshold within a stated response budget. Dataset, an LLM-generated KT bank with known calibration targets. Contribution. Operationalizes "items can enter the bank live" as a measured engineering primitive with stated error rates, rather than as a slogan.

D.3. Recommender-style content sequencing as measurement-aware action. Method, sequencing policy that uses streaming ability and item information jointly. Dataset, an existing KT platform log. Contribution. Reframes content recommendation in education as a streaming measurement-action loop and shows that ignoring measurement uncertainty in sequencing creates feedback pathologies. This is where the candidate's DRL background does work.

### Thrust E. Out of domain (patient-reported outcomes).

Sketched in section 5. Four papers, E.1 through E.4.

## 7. Venue strategy

The venue strategy must be partitioned by which community will judge the contribution.

Foundation papers (A.1, A.2, A.3) go to AIED, EDM, LAK as primary, with strong ML venues (NeurIPS, ICLR, AAAI) as secondary targets for the methodological extensions. A.2 in particular has a clean conformal-prediction story and would do well at an ML venue. The reason to target AIED first is that the paradigm claim is most legible there, the methodological extensions get traction at ML venues.

Calibration papers (B.1, B.2) target *Psychometrika* and the *Journal of Educational Measurement* primarily. These are the journals that will recognize the identifiability content and that have the standing to certify a calibration method. B.3 (LLM items) is dual-target, EDM or AIED for the application side, *Psychometrika* for the calibration side.

Validity papers (C.1, C.2, C.3) target *Psychometrika*, *Applied Psychological Measurement*, and the *Journal of Educational and Behavioral Statistics*. These are the venues where invariance and DIF are taken seriously. C.3 (person-fit, also touching cheating detection) has a secondary path to security venues (NeurIPS workshops, FAccT for the fairness angle).

Intervention papers (D.1, D.2, D.3) go to AIED, EDM, and to ML venues that take adaptive testing seriously. D.3 specifically has a path to RecSys.

Out-of-domain papers. E.1 and E.2 target *Medical Care*, *Quality of Life Research*, *Statistics in Medicine*. These are the venues that publish PROMIS methodology. E.3 has a path to *Journal of the American Medical Informatics Association (JAMIA)* and *npj Digital Medicine* because of the clinical decision-support angle. E.4 is hardest, *Psychometrika* is the right home if the construct-drift methodology is rigorous, otherwise it lives in a measurement journal.

The paradigm claim itself, as a position piece, belongs in a journal that publishes opinion-form contributions with technical backing. *AERA Open* is a candidate. *Educational Researcher* if the candidate has institutional support to push there. The position piece is not the first paper, it is the *capstone*, written after the thesis has produced enough of the underlying primitives that the claim is supported by the work and not just by argument.

Which venues care about the paradigm claim itself versus the applications. The paradigm claim will be evaluated most seriously by *Psychometrika* and by the AIED community. The applications are evaluated domain by domain. The strategic insight is that the candidate should not try to make the paradigm claim its own venue, the claim is supported by the body of work, not by a single paper.

## 8. Risks and falsifiers

There are six risks worth naming.

The first is intellectual dilution across domains. I think this is the most serious risk. The defense is that the unifying methodological commitment, which I will state explicitly, is *streaming estimation of latent quantities under defined polytomous response models with per-instant uncertainty and drift accountability*. Every paper in the thesis must hit this signature. If a paper does not, it does not belong. PROs hit it, vocational skill measurement would hit it, formative classroom signals hit it. A paper that drifts into "let's apply attention to symptom data" without the latent-variable measurement structure does not hit it and would not be in the thesis. This is the test the candidate applies to every proposed paper.

The second is identifiability collapse in streaming joint estimation. When items, persons, and population structure all drift, identifiability is not automatic. A failure mode of Thrust B is that the streaming estimator confuses sources of drift and produces parameter estimates that fit the data but are non-identified. The defense is to commit upfront to a *taxonomy of streaming identifiability* that says what is identifiable from response data alone, what requires auxiliary signal, and what cannot be identified at all. The taxonomy is itself a contribution.

The third is uncertainty miscalibration. Per-instant uncertainty for latent variables is hard. The Bayesian community has a strong story for it (sequential Monte Carlo, variational filtering) but operational guarantees under distribution shift are limited. A failure mode is that the streaming intervals look right on synthetic data and fail on real data with non-stationarity. The defense is to commit to conformal-style coverage guarantees as a first-class part of Thrust A, not as an afterthought.

The fourth is closed-loop feedback pathology. If intervention shapes the data that measurement consumes, measurement can drift in pathological ways. The defense is to make this an explicit object of study in Thrust D rather than wishing it away.

The fifth is venue mismatch. ML venues will want bigger models and more impressive predictive performance, psychometric venues will want stronger identifiability theory, clinical venues will want longer real-data deployments. The defense is to write papers that are venue-aware rather than venue-blind, and to accept that the methodological core will appear in different framings in different venues. This is normal for paradigm-spanning work.

The sixth is the candidate's solo capacity. A five-thrust thesis is large. The defense is that the thrusts share infrastructure (the streaming estimator, the synthetic data generators, the evaluation harnesses) and that papers can co-evolve. Realistically, the candidate ships A in year 1, A and the start of B in year 2, B and C in year 3, D and E.1 in year 4, E.2 through E.4 and the capstone in year 5.

What would falsify the thesis as a whole. Two failures would. If the streaming primitive consistently underperforms offline EM on the offline benchmarks under realistic data, the paradigm claim collapses, because parity at the batch boundary is the entry ticket. If the streaming primitive ports to PROs and fails (uncalibrated uncertainty, no clinical signal, no methodological advance) the cross-domain claim collapses and the thesis reduces to D2. Both of these are real risks and both are testable in the first two years.

## 9. Defense against the strict-reviewer attack

The strict reviewer says, "this is three PhDs glued together. Streaming KT is one PhD. Streaming psychometrics is another. Streaming health measurement is a third. You cannot do all three. This is overreach masquerading as breadth."

The answer comes in three parts.

First, the unifying methodological commitment is genuinely one commitment, stated above. Streaming estimation of latent quantities under defined polytomous response models with per-instant uncertainty and drift accountability. Every paper in the thesis hits this signature. The thesis is not three PhDs, it is one methodological commitment exercised across three demonstration domains. The KT papers prove the primitive works in the testbed that has the cleanest baselines. The psychometric papers prove the primitive supports the validity guarantees the field demands. The PRO papers prove the primitive ports to a domain with different data structure, different stakes, and different baselines. Without all three, the paradigm claim is unearned.

Second, the depth-versus-breadth concern cuts the other way than the reviewer suggests. A purely educational thesis of equal scope (D1 or D2) would have to invent the entire intellectual structure of streaming measurement using only KT as the proving ground, which is a thin evidential base. By doing the proving ground in KT (where the candidate has technical advantage) and the porting in PROs (where the methodological community is mature and the baselines are recognized), the candidate gets *more* depth on the unifying claim, not less. Breadth, done correctly, is depth on the abstraction.

Third, the candidate is not claiming domain expertise in clinical psychometrics. The PRO thrust will be done in collaboration with the clinical-psychometric community, which has its own validation infrastructure. The methodological contribution belongs to the candidate. The domain validation belongs to the collaborators. This is how the methodological psychometrics literature has always worked, methodologists produce primitives, domain communities certify them. The thesis follows that pattern.

If the reviewer presses, "but you are also doing item generation and adaptive testing," the answer is that those thrusts are not domains, they are operations under the paradigm. Adaptive testing on top of streaming measurement is not a separate field, it is the natural action object once measurement runs continuously. The same is true of generation. The thrust structure mirrors the operational structure of measurement systems, not the topical structure of fields.

## 10. Why D3 is the most novel framing

D1 is closed-loop educational measurement. It is defensible. It is also incremental. The AIED community already has the components, measurement, recommendation, adaptive testing, and D1 ties them together. The contribution is integrative. It is publishable, it is fundable, it is unlikely to be paradigm-shifting.

D2 is unified streaming theory for educational measurement. It is more ambitious than D1. It builds the methodological primitives I described in Thrust A, B, C, D, but stays inside education. The contribution is methodological. It is the right thesis for a psychometrically-minded candidate who is risk-averse about leaving their field. It positions the candidate for a strong academic career inside educational measurement.

D3 is the paradigm claim. The unique contribution is that it asserts streaming measurement is a *transferable computational primitive*, not an educational method. The evidence for the assertion is the cross-domain port. D2 cannot make this assertion because D2 stays inside education. D1 cannot make this assertion because D1 is not even trying to. Only D3 can.

What does this buy the candidate intellectually. It positions the work as a contribution to *measurement science* rather than to a topical subfield. Measurement science as a category currently sits between statistics, psychometrics, and machine learning, and has no clean home. The candidate has the chance to claim a piece of that intellectual territory. Researchers who take that risk and succeed end up defining new venues, supervising students across fields, and serving as bridge figures between communities. Researchers who take that risk and fail end up with a thesis that is hard to place. The risk is real and the upside is also real.

What is most novel here. Three things.

First, the *naming and structuring* of streaming measurement as a paradigm. Plenty of online-IRT papers exist. There is, to my knowledge, no thesis that treats the full set of measurement operations (person, item, validity, intervention, generation, audit) under a unified streaming frame, and no thesis that demands cross-domain demonstration of the same primitive. The act of naming and structuring is itself the novelty.

Second, the *taxonomy of streaming identifiability*. This is technical content that the field needs. When items, persons, and population structure all drift continuously, identifiability becomes a genuine open problem, not a textbook lemma. Producing the taxonomy is a contribution that travels across all five thrusts.

Third, the *cross-domain primitive*. If the same streaming architecture, with appropriate response head, succeeds on KT and on PROMIS-style PROs, that is empirical evidence for the paradigm claim that no purely-educational thesis can produce. This is the load-bearing novelty and the riskiest part of the thesis.

Finally, what I would tell the candidate. The paradigm claim is the right framing. The risk is real and the candidate should know it. The defense against dilution is to enforce the methodological signature ruthlessly on every paper. The out-of-domain thrust must be done in serious collaboration with a clinical-psychometric group, not as a methodologist parachuting in. MA-GPCM is the right first paper. The capstone position piece is the right last paper. Everything in between is the body of work that earns the claim. The thesis is doable in five years and only doable in five years if the candidate is disciplined about the signature.
