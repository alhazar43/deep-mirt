# PhD Research Blueprint, Round 4 (compiled)

A single integrated document that pulls together my own analysis with the three v2 agents' analytical pieces (D1 closed-loop, D2 unified theory, D3 paradigm). Treats the three directions as alternatives, not as a menu, and identifies where they converge and where the candidate has to pick.

Companion files with the full depth-first arguments per direction are
- `phd_blueprint_d1_v2.md`, closed-loop educational AI on a streaming measurement primitive
- `phd_blueprint_d2_v2.md`, streaming educational measurement as a unified computational theory
- `phd_blueprint_d3_v2.md`, streaming as the right computational paradigm for measurement in the AI era

This compile distills the four pieces of thinking into one frame the candidate can act on.

---

## 1. The storyline (locked across all four pieces of thinking)

MA-GPCM exists because the computational architecture of educational measurement is from a different era. Classical psychometrics assumes a fixed item bank, a closed cohort, a frozen response matrix, and an offline EM pass. That world has gone. Modern learning platforms produce response streams, items enter the bank during deployment, LLMs grade on the fly, cohorts are open, the apparatus itself drifts. The batch primitive cannot serve this world.

The candidate's real intellectual question is "what does educational measurement look like when the computational primitive is no longer batch but streaming?" MA-GPCM is the first answer, scoped to the easiest setting (single cohort, fixed bank, defined ordinal response model, no AI in the apparatus). It proves the streaming primitive matches EM on recovery in that base case and does what EM cannot. That is the floor of the thesis, not the ceiling.

The candidate's identity is "a researcher who reframes batch educational measurement into streaming computational primitives." Paradigm claim, paradigm-scoped PhD. All four pieces of thinking (my analysis plus the three v2 agents) agree on this. Every difference between D1, D2, D3 is downstream of this shared frame.

---

## 2. What MA-GPCM is and is not (consensus across the v2 round)

Essential to the thesis claim
- Single-pass encoder that maintains per-student state from response history
- Decoder head that reads ability, item parameters, or other measurement quantities off that state
- Per-time, per-student theta produced in stream, matching EM on stationary recovery and beating EM on non-stationary recovery
- Separated ability pathway (SIE) that keeps item structure from leaking into the ability estimate

Incidental and swappable without breaking the thesis
- DKVMN as the specific memory backbone (transformer-based encoders would serve the same role)
- GPCM as the specific ordinal head (GRM, PCM, or other response models would not break the framing)
- The ordinal response format itself (binary, count, continuous responses admit the same construction)
- The educational setting (the same primitive ports to other measurement-heavy domains)

This essential vs incidental split is load-bearing. Reviewers who attack module choices have to be redirected to the architectural commitment, not allowed to pin the thesis to DKVMN or GPCM specifically. All three v2 agents made this split independently and converged on roughly the same answer.

---

## 3. Three thesis claims, side by side

### D1, closed-loop educational AI on a streaming measurement primitive

In batch psychometrics, calibration and policy are structurally decoupled *because EM forces it*, not because anyone thinks the decoupling is right. In streaming, the decoupling cannot hold because items enter mid-deployment, cohorts open, and the apparatus drifts. Closed-loop is the only framing in which the streaming primitive delivers value beyond efficiency. Every action measures, every measurement acts. The thesis demonstrates the coupling.

Five thrusts come out of the argument. Measurement primitive (MA-GPCM and its immediate generalizations), action primitive on streaming measurement (bandit, policy gradient, curriculum RL), closed-loop validity (the technical heart, off-policy validity with learned observation model, selection-aware likelihoods, calibration drift under policy, causal identifiability of learning under intervention), bank and apparatus (LLM-item cold-start, streaming DIF without pre-specified groups, self-reinforcing fairness), sim-to-real and deployment.

The candidate's DRL background is the action primitive, not a parallel thread. New technical work beyond standard DRL includes policies on non-Markovian streaming psychometric state, OPE with a learned observation model, reward design under partial identifiability of theta.

Closed-loop made precise. Three conditions hold simultaneously. Policy is a function of M's current state. M is updated using responses elicited under pi with selection non-ignorable. Pi is evaluated against M with cross-fitting that breaks the circularity. Four new technical problems fall out, selection bias in streaming theta, off-policy validity with a learned observation model, calibration drift induced by the policy, latent dynamics under intervention. These four problems are unsolved in standard DRL or standard psychometrics and they are the substance of Thrust 3.

### D2, a unified theory of streaming learner measurement

The encoder-decoder streaming primitive is the architectural commitment, and the thesis shows the streaming primitive subsumes a connected subset of the batch measurement stack under one identifiability discipline. The claim is theoretical, not architectural. It states three falsifiable propositions. Existence (for each batch operation there is a streaming surrogate that recovers the batch object as a limit), batch equivalence under stationarity (streaming and batch estimators agree up to a finite-sample Monte Carlo gap), strict dominance under non-stationarity (streaming has objects to estimate that batch does not).

Five thrusts. Streaming person measurement (MA-GPCM and the identifiability statement that propagates to every other thrust), streaming item calibration with calibrated uncertainty, streaming cognitive diagnosis with cross-decoder consistency against the theta head, streaming drift and DIF as one operation, streaming person-fit. Linking, item generation, CAT, and intervention are excluded with stated reasons. DRL is out by design.

The unification is not "same model does five things." It is "the same reduction applies to five things." Identifiability propagation across thrusts is the load-bearing argument that this is one thesis. If theta is identified up to a time-varying affine in Thrust A, item parameters in B are identified up to the inverse transform, skill profiles in C are identified up to a Q-matrix-mediated rotation, drift in D must be invariant to the affine, and fit in E must be too. The five thrusts share one identifiability geometry.

The strict-reviewer attack ("you have a unified architecture not a unified theory") is answered by three properties. The thesis makes equivalence and non-equivalence claims that can be proved or refuted, the identifiability statement in A propagates constraints to B through E, and the framework predicts streaming surrogates for operations the thesis did not instantiate.

### D3, streaming as the right computational paradigm for measurement in the AI era

The paradigm claim has four content points. Unit of computation is the arrival of a single response, not the closure of a response matrix. Unit of guarantee is per-instant, not per-cohort. Unit of accountability includes drift as a measured quantity with its own estimator and uncertainty. Unit of intervention runs on the same time axis as measurement.

The claim is validated by breadth of applicability. A streaming primitive that runs only on KT is, in evidential terms, no different from an online-IRT paper, of which there are dozens. A streaming primitive that crosses fields and exposes a shared structure that the fields themselves had not seen is a paradigm contribution. The PhD has to do the second thing to justify the framing.

Five thrusts. Streaming person measurement, streaming item and population calibration with item-vs-population drift disentanglement, streaming validity and invariance (including fairness as the field's native language for group-level measurement equivalence), streaming intervention and generation (where the candidate's DRL background does work but as one thrust not the spine), and one out-of-domain thrust.

The recommended out-of-domain thrust is patient-reported outcomes (PROMIS-style longitudinal symptom and function measurement). PROs win over vocational because PROMIS supplies a mature batch baseline that can be dethroned, while vocational measurement lacks a consensus instrument to dethrone. The PRO port creates five new technical problems that ed-AI did not, bursty sparse per-person data, contested constructs, mode-and-translation invariance, individual clinical stakes, institutional data ecosystem.

The paradigm claim is supported by the body of work, not by a single position piece. The position piece is the capstone, not the opener.

---

## 4. Where the three claims converge

Five points the candidate can trust regardless of which direction is picked.

First, **MA-GPCM is the existence proof, not a chapter-one example**. All three blueprints treat MA-GPCM as the foundation that licenses the rest. Without MA-GPCM the parity-with-EM claim has no instance. With MA-GPCM, the thesis can argue that the streaming primitive works on the easiest case, and the rest of the thesis is generalization.

Second, **MA-GPCM in its current form is the floor, not a thrust**. All three blueprints require at least one substantial extension of MA-GPCM before the measurement-primitive thrust can stand as a thrust rather than as a single paper. Candidate extensions are multidimensional theta, joint streaming calibration with selection awareness, open-cohort cold-start, and continuous-time variants.

Third, **drift detection is the strongest part of the dominance claim** because batch psychometrics has no honest competitor to streaming on within-deployment drift. D1 has it as Thrust 4, D2 has it as Thrust D (folded with DIF), D3 has it as Thrust C. Whichever framing is picked, drift detection earns most of the "streaming dominates batch" airtime.

Fourth, **identifiability under streaming and intervention is the technical heart**. D1 names it Thrust 3 (closed-loop validity). D2 names it Thrust A2 (the identifiability statement that propagates). D3 names it as the "taxonomy of streaming identifiability" that travels across all thrusts. Different framing, same problem. The strongest theoretical paper of the thesis lives here, with Psychometrika or AISTATS as the natural venue depending on whether the result is more measurement-theoretic or more estimation-theoretic.

Fifth, **the four risks that determine whether the thesis holds are the same across framings**. (i) Streaming coupling fails to beat periodic recalibration in realistic settings. (ii) Identifiability under adaptive selection or continuous arrival is not provable under realistic assumptions. (iii) Uncertainty under streaming is miscalibrated and conformal coverage cannot fix it. (iv) Real-data deployment does not materialize and the thesis stays synthetic-only. The candidate should be probing all four in year one.

---

## 5. Where the three claims diverge

The three directions disagree on three structural questions.

**Does DRL live inside the thesis or outside it.** D1 says inside, as the action primitive that closes the loop with measurement. D2 says outside, the measurement-only framing is purer without it. D3 says inside but as one thrust among five, with the heart being measurement, not action. This is the largest structural difference. The candidate's DRL background loads the cards toward D1 or D3.

**How broad is the empirical evidence.** D1 stays inside education and gets depth through closed-loop integration. D2 stays inside education and gets depth through unified identifiability discipline. D3 reaches out to at least one non-education domain (PROs) and gets depth through cross-domain transfer of the primitive. The trade is depth-inside-education versus breadth-across-domains.

**What kind of paper is the capstone.** D1 capstone is a deployment paper at IJAIED that runs the closed-loop primitive on real students. D2 capstone is a Psychometrika paper on streaming identifiability or person-fit that grounds the theory. D3 capstone is a position piece (AERA Open or Educational Researcher) that names the paradigm after the body of work has earned the claim. Different audiences read different artifacts as the climax of the thesis.

---

## 6. The thrust structures, compressed

D1 (closed-loop, with DRL as action primitive)
- T1 Measurement primitive, MA-GPCM plus joint streaming calibration, multidim theta, open cohort
- T2 Action primitive on streaming psychometric state, bandit, policy gradient, curriculum RL
- T3 Closed-loop validity, OPE with learned observation, selection-aware likelihoods, drift under policy, causal identifiability under intervention
- T4 Bank and apparatus, LLM-item cold-start, streaming DIF without pre-specified groups, self-reinforcing fairness
- T5 Sim-to-real and deployment

D2 (unified theory, no DRL)
- A Streaming person measurement (MA-GPCM plus identifiability statement)
- B Streaming item calibration with calibrated uncertainty
- C Streaming cognitive diagnosis with cross-decoder consistency
- D Streaming drift and DIF folded as one operation
- E Streaming person-fit

D3 (paradigm, with PRO out-of-domain)
- A Streaming person measurement with conformal coverage and continuous-time variant
- B Streaming item and population calibration with item-vs-population drift disentanglement
- C Streaming validity, invariance, fairness
- D Streaming intervention and generation (CAT, LLM-item generation, content sequencing)
- E Out-of-domain, PROMIS-style PROs across four sub-papers

All three structures land at five thrusts with three to four sub-papers each, fifteen to twenty papers total. That is a research agenda, not a publication list. The thesis itself is three to five flagship papers plus supporting workshop or short papers. The candidate should plan for that compression now.

---

## 7. The risk landscape

Six risks consolidated from the three v2 blueprints.

R1, **streaming dominance over batch is small in practice**. If streaming beats windowed batch EM by five percent of QWK rather than thirty, the framing is still correct but less exciting. Mitigation, run the comparison early on cases where the gap should be largest (open cohorts, fast bank turnover, within-session theta).

R2, **identifiability under streaming and intervention is partial only**. Confidence intervals on policy value may be wide, identifiability statements may hold under restrictive policy classes only. Mitigation, accept set-valued estimands and argue empirically that the loss from partial identification is small.

R3, **cross-decoder consistency fails in D2's Thrust C** (IRT-decoder and CDM-decoder cannot share one encoder cleanly). Mitigation, frame Thrust C with two acceptable outcomes from day one. Either consistency holds (vindication) or it does not (a theoretical finding about IRT-CDM incompatibility under shared encoding). Both are publishable.

R4, **PRO port fails in D3** (no clinical signal, no methodological advance, uncalibrated uncertainty). Mitigation, partner with a clinical-psychometric group in year one. Do not parachute in as a methodologist.

R5, **deployment access does not materialize**. The thesis stays simulation-only. Mitigation, start partnership conversations in year one. Even a single classroom of N=30 is enough for measurement-only validation.

R6, **the thesis is read as five loosely connected papers**. Mitigation, the introduction must earn the framing, not assert it. The reader should finish the introduction unable to imagine removing any thrust without breaking the argument.

---

## 8. My recommendation, and the reasoning

I think **D3 is the strongest framing for this candidate and this moment, D2 is the safest, D1 is in between**. Here is the reasoning.

D3's case rests on three points. First, the candidate's stated philosophy in the MA-GPCM paper is paradigm-flavored, not method-flavored. The paper's discussion section already gestures at the broader "what computational architecture for measurement" question. D3 commits to that gesture. Second, the cross-domain port (PROMIS-style PROs) is what turns the work from "online IRT, again" into "streaming as a transferable computational primitive across measurement domains." That is a paradigm-level contribution that neither D1 nor D2 can make. Third, the candidate's DRL background fits cleanly as Thrust D (intervention) without being the spine of the thesis, which avoids the D1 trap of looking like measurement-plus-recommendation glued together.

D3's risks are real. Intellectual dilution across domains is the largest. The defense is to enforce one methodological signature ruthlessly on every paper, "streaming estimation of latent quantities under defined polytomous response models with per-instant uncertainty and drift accountability." If a paper does not hit that signature, it does not belong. PRO port failure is the second largest risk. The defense is serious collaboration with a clinical-psychometric group, not a methodologist parachuting in.

D2 is the right framing if the candidate prefers depth-in-field over cross-domain ambition. D2's case is that the streaming-measurement unification is intellectually rich enough to fill a thesis without leaving education, the venue mix is mature and well-understood (AIED, EDM, LAK, Psychometrika), and the candidate builds an academic identity inside educational measurement without the cross-domain risk. The cross-decoder consistency in Thrust C is the fragile piece and should be framed with two acceptable outcomes from day one.

D1 is the right framing only if the candidate's DRL identity is central to how they want to position themselves, and if closed-loop deployment in education is the contribution they want to be known for. If DRL is a useful background but not the identity, D1 is incidentally choosing a frame that does not fit the candidate's stated philosophy.

The decision the candidate needs to make is not which thrusts to do but which paradigm-level claim to commit to. The thrusts follow from the claim, not the other way around.

---

## 9. Two questions to settle before the proposal

Q1, what cross-domain commitment is the thesis making. Cross-domain claim that requires the PRO port (D3), in-education unification claim (D2), or in-education closed-loop claim with DRL central (D1).

Q2, who is the candidate's primary academic community five years from now. Educational measurement (D2 native). Educational AI with measurement and intervention (D1 native). Computational measurement science as a cross-field identity (D3 native).

The answer to Q2 should drive the answer to Q1, not the other way around. The thesis is the artifact that positions the candidate for the community Q2 picks.

---

## 10. What stays open

Three things deliberately not committed to in this round.

The exact set of MA-GPCM extensions that turns Thrust A from a paper into a thrust. The three candidates (joint streaming calibration with selection awareness, multidim theta without a Q-matrix prior, open-cohort cold-start) are all defensible. The pick depends on which downstream thrusts the candidate prioritizes and what real data is reachable.

The specific sub-papers per thrust. Each v2 file proposes fifteen to twenty. The candidate will not write all of them. The compression to a publication-shaped plan happens after Q1 and Q2 are answered, not before.

The order of publication. The capstone is direction-dependent. The opener is always MA-GPCM. The middle is where the candidate has the most freedom and where year-by-year execution risk drives choices. That ordering is a year-one conversation with the supervisor, not a blueprint decision.

This is where the blueprint ends. The next document is the thesis proposal, and that requires the candidate's commitment on Q1 and Q2.
