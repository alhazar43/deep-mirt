# Paper boost plan (the plan for the plan)

Working object for the ultracode planning session that follows. Mission: elevate "Not All
Parameters Learn Alike" from solid-but-mild to a strong contribution, by widening the claim
structure and closing the gaps a skeptical senior reviewer would attack. This document is
MAINTAINED jointly by the lead (Fable5) and the adversarial supervisor during that session,
until it becomes a fully detailed paper plan (what has been worked / what must be worked).

Grounding: verdicts below come from a 4-scout landscape scan (KT, neural-IRT, broader ML,
stakes precedents) adjudicated by an Opus adversary on 2026-07-03 (workflow
paper-boost-horizon-scan). Evidence base: docs/exposure_rerun_results.md (all numbers),
docs/exposure_rerun_plan.md (design + novelty verdict + must-cites).

## 1. Cast and constitution (for the ultracode session)

- LEAD (Fable5, main loop): claims, synthesis, decisions, final calls. Proposes; never
  self-certifies.
- ADVERSARIAL SUPERVISOR (Opus, high effort): grounds, does not hinder. Every proposed claim
  or section gets a verdict GROUNDED / PLAUSIBLE(+named missing evidence) / FANTASY(kill or
  descope). Enforces the scope fence (section 7) and the honesty rules. The lead may appeal
  once with new evidence; the second verdict stands.
- RESEARCH SCIENTIST (Opus): think-to-code and code-to-think; designs experiments from claims,
  reads results back into claims.
- ML DEVELOPER (Sonnet): practical implementation; drivers, runs, aggregation. Follows the
  established scratch discipline (new _-prefixed files only, fold-level persistence,
  skip-done, determinism, single 8GB GPU sequential).
- INFO GATHERER (Haiku): mechanical lookups, file inventories, citation fetching.
- Protocol: lead proposes -> adversary verdicts -> only GROUNDED items enter the outline;
  PLAUSIBLE items enter the work queue with their named evidence; FANTASY items are logged
  as rejected with reasons (do not re-litigate). Every outline section must carry pointers to
  evidence on disk or to a queued run.

## 2. Operational definition of "not small"

The paper is big enough when ALL hold:
1. One spine law that organizes every result (not six ablations).
2. At least one result the literature PROVABLY lacks (the flagship).
3. A stake: a decision that visibly changes, not just a rho.
4. A reason-to-exist answer to "why not just use mirt".
5. An artifact the community can adopt (benchmark + protocol).
6. Every claim survives the adversary; the two refuted hypotheses stay refuted; the
   MML-dominance honesty stays front and center.

## 3. The claim structure (post-adjudication)

SPINE (V1, GROUNDED with a gate; BOUNDARY per P4 lens 2, BLOCKING, folded in): a parameter's
recovery is governed by its readout channel's statistical regime. Per-item channels (wide
keys, dkvmn slots, classical MML) scale with exposure and starve below it -- this half is
clean everywhere (mirt exactly flat-in-N; the decoupled channel 5.6x steeper in E). The
amortized half -- pooled channels scale with total sample size -- is DECODER-DEPENDENT and
must always be stated with its boundary: clean for nrm (R2 0.90), partial for gpcm (Q-level
offset), FAILS for 2pl where bank size dominates (its low per-response information on the
slope makes coverage the binding constraint). Never write "governed by total data"
unqualified. Decoupling = choosing the regime per parameter. One mechanism
explains: the shared-pool laggard, the decoupling fix, the E~30 crossover, the dkvmn reversal
(slots = built-in per-item channels), the transformer being worst (extreme pooling), the MML
ceiling, and the real-data coverage reversal.
- GATE (adversary, binding): lead every framing with the PER-ITEM MLE ORACLE CONTROL, so the
  claim is about the target's available information, not weight-sharing, else it reads as
  Cremer's amortization gap relabeled. Status: a two-stage per-item-MLE artifact exists from
  the OLD bed (docs/experiment_results.md negative-results; scratch _two_stage_alpha.py);
  VERIFY and, if needed, re-run cheaply on the new spiraled bed (P0 below).
- E* is a PREDICTION TO BE EARNED, not a result: a one-number post-hoc fit is not a
  derivation. Either derive it from the two information rates and test out-of-sample (other
  decoder, other Q), or present E~30 as measured, not derived.

FLAGSHIP (V5, GROUNDED): the real-data calibration tax. Refit classical MML (mirt) on the
IDENTICAL real matrices (EdNet, KDD) the neural readout used. The scout searched for this
exact comparison and found it ABSENT from the literature; structural gap. No ground truth on
real data, so it is a neural-vs-mirt CONCORDANCE + DOWNSTREAM-DIVERGENCE study (+ split-half
reliability per method), not recovery. This is also the only place the honest synthetic
verdict (MML dominates) can legitimately turn: misspecification (H != F) voids the classical
guarantee. Either outcome is a finding.

BACKBONE FIGURE (V3, GROUNDED, analysis-only): the two-law scaling figure from the existing
144-cell surface: amortized channels ~ f(total responses), per-item channels ~ g(exposure),
classical mirt flat-in-N at fixed E. Renders the spine falsifiable on a live, named design
question (NWEA ~2000/item guidance; ETS collateral information; SPICE).

STAKE (V4-lite, GROUNDED, nearly free): the INTERPRETATION-FLIP demo. Show a Deep-IRT-style
item ranking / "sharp item" labeling flips between two arms whose accuracy differs by 0.001.
Converts accuracy-blindness into a visible wrong decision. The full CAT simulator is OPTIONAL
(PLAUSIBLE): scout supplied the standard harm vocabulary (test length can quadruple at
calibration SD 0.1-0.2; rho<0.85 considered inadequate); none of it exists for neural
readouts, so results would be ours to establish. Decide after the free items land.

REASON-TO-EXIST (V6, PLAUSIBLE, cheap): held-out COLD-ITEM calibration on the existing bed:
amortized channels calibrate zero/near-zero-exposure items that classical MML cannot touch,
from cohort structure alone (no item content). Clean contrast with AutoIRT/SPICE (they add
NLP features). Claim ONLY the content-free floor; never claim it beats content-based.

ARTIFACT (V7 + V10, GROUNDED): release the spiraled exposure-controlled generator + the
surface + the certification protocol; plus the evaluation-pitfalls section (padding-bug
post-mortem, dense-bed artifacts, rank-vs-linear metrics, alpha-decile caps, category
starvation). Extends the pyKT reform frame from prediction AUC to parameter validity, which
the KT scout identifies as the move that reads MAJOR (EDM 2025 reform theme; explainable-KT
survey 2024 names missing evaluation metrics for explanations as the central unsolved
problem; trustworthy-education survey 2026).

BRIDGE, NARROW (V2, PLAUSIBLE): position the protocol as control-tasks-for-psychometric-
readouts; the general "prediction does not certify faithfulness" point is KNOWN (Hewitt-Liang
control tasks; Belinkov) and must be cited as such; only the audited setting + the two-regime
refinement are new. FRAMING ONLY (V8): one positioning sentence that the amortized regime is
the only one that exists online. KILLED (V9): the standalone architecture-probe build (no
checkpoints saved; drifts toward the forbidden representation-learning home); the
slots/flat/attention observation folds into the spine as evidence.

## 4. Pre-registered defenses (attack -> neutralization)

1. "Amortization gap / identifiability relabeled" -> oracle control front and center (P0);
   cite Cremer, Khemakhem/Roeder, and state what is new: the audited KT setting + the
   per-parameter regime split + the classical control.
2. "MML dominates, so why deploy the readout at all" -> V6 cold-start floor + V5 real-data
   misspecification turn + the online/sequential positioning sentence.
3. "A curiosity; no decision changes" -> V4-lite flip demo (and optionally the CAT simulator).

## 5. Work map (what has been worked / what must be worked)

BANKED (on disk, verified): 144-cell N x Q surface; toggle study incl. 10 NRM configs; dkvmn
reversal probe; overtraining trajectories; crossover E~30; mirt synthetic control (flat-in-N;
dominates absolute); accuracy-blindness table; novelty verdict + must-cites; two refuted
hypotheses; timing/n_params (efficiency panels deferred).

P0 VERIFY (haiku/sonnet, hours): (a) the per-item MLE oracle artifact -- what exists from the
old bed, is a new-bed re-run needed (likely: two-stage fit on 2-4 grid cells, sonnet, <1 GPU-h);
(b) confirm outputs/p2_mml last cell finished (bmhglcmvf); (c) inventory what the flip demo
needs (item rankings per arm; on disk in arrays.npz).
P1 FREE (analysis/writing, no GPU): V3 two-law figure; V4-lite flip demo; V10 pitfalls
section; V1 spine framing with oracle gate; scaling fits.
P2 RUNS (days, GPU+CPU): V5 real-data tax (EdNet + KDD: neural arms incl. decoupled + mirt
refit on identical matrices + split-half reliability per method; also close the old loose end:
the NRM one-key fix arm on EdNet); V6 cold-item experiment (existing bed, held-out items);
optional E* out-of-sample test; optional trajectory re-run with per-checkpoint prediction
metrics (the "selection lands past the measurement peak" figure currently lacks the
prediction curve).
P3 OUTLINE (lead + adversary): full section-level outline, every section with evidence
pointers; the adversary passes verdicts section by section.
P4 REVIEW: adversarial full-outline pass (the three attacks + scope fence as the rubric);
then handoff to writing (register: main_magpcm_ijaied.tex; writing rules per CLAUDE.md).

## 5b. EXECUTION STATUS (ultracode session, launched 2026-07-03)
- WF free-evidence-cpu (wf_0a7a69e7-d88): P0 verifications -> oracle control (both clamps,
  CPU per-item MLE) -> two-law scaling figure -> interpretation-flip demo. CPU ONLY.
  Outputs: outputs/p2_oracle/, outputs/p2_scaling/, outputs/p2_flip/.
- WF flagship-gpu-track (wf_f2675d5e-a86): OWNS THE GPU, serial. Real-data calibration tax
  (EdNet then KDD; full-matrix fits per arm + per-half params + matrix export -> mirt refit
  on identical matrices, incl. mirt's own split-half) -> cold-item skewed-exposure bed
  (tiers E {5,10,20,40} vs mirt) -> trajectory re-run with per-checkpoint prediction
  metrics -> Opus synthesis. Outputs: outputs/p2_realtax/, outputs/p2_coldstart/,
  outputs/p2_trajpred/. Includes the EdNet NRM one-key loose end.
- WF bibliography-lock (wf_737d118b-adc): DONE. 40 verified entries -> docs/boost_refs.bib
  (durable copy). Notables: BanditCAT (Sharpnack et al., PMLR 264, 2025) and AutoIRT
  (arXiv:2409.08823) are TWO papers, both included; explainable-KT survey = Bai et al.,
  Applied Intelligence 2024 (abstract states the missing-evaluation-metrics gap our protocol
  fills); trustworthy-education survey = Yu et al. arXiv:2601.21837 (preprint, cite as such);
  DKT second author is Bassen (arXiv metadata mangles it); Deep-IRT is single-author Yeung.
  UNCITABLE: the "NWEA ~2000 responses/item" figure (NWEA's own materials say ~1000
  field-test responses) -- for calibration-sample-size effects cite ban_comparative_2001; do
  NOT put the 2000/item number in the paper.
- NEXT (blocked on free-evidence + at least the flagship's EdNet leg): P3 outline drafting,
  lead + adversarial supervisor, maintained IN THIS DOCUMENT (section 9 to be appended as
  "The outline"), then P4 adversarial full pass.

## 6. Landscape hooks and must-cites (from the scan)

- KT reform current: pyKT (NeurIPS D&B 2022) as the methodological floor; EDM 2025 reform
  theme; explainable-KT survey (2024); trustworthy intelligent education survey (2026).
- Neural-IRT: Deep-IRT (Yeung 2019); Converse (AIED 2021); VIBO (EDM 2020, person-side
  amortization mirror); AutoIRT (Sharpnack et al. 2024/25, content-based cold start, MUST
  distinguish); SPICE; BanditCAT; Koenig/Spoden/Frey (Bayesian pooling folklore).
- Operational psychometrics: NWEA per-item calibration guidance (~2000/item); ETS collateral
  information; CAT harm literature (calibration-error -> test length / classification at cut
  scores; van der Linden-line simulation studies).
- Broader ML (cite-as-known, bridge only): Hewitt-Liang control tasks; Belinkov probing
  survey; Cremer amortization gap; Locatello; Khemakhem / Roeder identifiability.

## 7. Scope fence (binding on the ultracode session)

1. Home is knowledge tracing; IRT is the readable flavor. NEVER a learning-dynamics paper;
   not a representation-learning paper.
2. Do NOT claim the two-regime law or the faithfulness point as a new general ML result; the
   general claims are known; the audited setting and the classical control are what is new.
3. Do NOT spin, soften, or bury the classical-MML dominance on well-specified data. The
   paper's turn is real-data misspecification (V5) + cold-start (V6), not reweighting.
4. Do NOT build the standalone architecture probe (V9) or any streaming system (V8).
5. The refuted hypotheses (dynamic-head fix; decoupling-delays-degradation) stay refuted.
6. Do NOT claim content-free cold-start beats content-based calibration; claim the floor.

## 9. THE OUTLINE
### 9.0 SURGERY B DIRECTIVE (hostile-editor ruling, 2026-07-03; GOVERNS everything below)
The author's stacking critique was ADJUDICATED CORRECT for the outline as drafted (no
repeatable takeaway; weakest-link evaluation across ~12 co-equal results; flagship resting
on a near-null). The cure is B+E. The S-blocks below remain the accurate EVIDENCE MAP; their
ROLES change as follows, and prose is written to this structure, not to the draft-1 skeleton:

IDENTITY: a trust-and-verify AUDIT of interpretable readouts already in the wild (the
shipped Deep-IRT-line design IS the shared arm). Never deploy-instead-of-mirt.
ONE SENTENCE: "Interpretable KT parameters fail because the shared readout channel discards
recoverable information -- per-item estimation on the model's own ability estimates reaches
the classical ceiling -- and giving each parameter its own channel restores it; here is what
the failure costs an adaptive test."
ARC: SYMPTOM (real-data flip, 67% disagreement at tied accuracy; synthetic flip with truth)
-> CAUSE (the oracle ladder, both decoders; the audit facts are its setting, not co-equal
contributions) -> FIX + BOUNDARY (decoupling restores per-item-estimator behavior; E*~30;
selection-alignment property; NRM excluded, the screen catches it) -> CONSEQUENCE (CAT harm,
experiment E [VENUE GATE, running]).
CONTRIBUTIONS, EXACTLY 3: (1) the channel mechanism with the oracle decomposition; (2) the
decision-cost audit of current practice (flip + CAT); (3) the exposure-controlled benchmark
+ certification protocol, shipped as one artifact.
DEMOTIONS (validation/appendix, NEVER contributions): the 144-cell matrix (validation of
generality); the two-law scaling (state as: the per-item half is clean and universal, the
amortized half is a decoder-dependent SPECTRUM -- half-a-law honesty); real-data
head-to-head (concordance + no-calibration-catastrophe + parameter-specific edges; NEVER "a
win", NEVER "the protocol is the deliverable" -- editor: that phrasing is spin on a
near-null); cold items (difficulty-only, 2x mirt at E=5, coverage retracted); trajectory
(one figure, the selection-alignment property lives in FIX).
PROPOSITION (D): only for the UMUAI/TMLR variant and only if the C1-C4 derivation is
certified airtight by ml-math first; a loose proposition is worse than none.
VENUE GATE **RESOLVED (2026-07-03): MATERIAL.** Shared arm: 196.8% length inflation
[190,204] + worse theta at its own stop (+0.036) + 2.3pp excess misclassification at the cut;
decoupled halves it; mirt ~ oracle. Education-Q1 route OPEN: IEEE TLT first, UMUAI second
(with restricted D), CAEAI viable. TWO-CHANNEL refinement to the CONSEQUENCE step (from the
ablation): discrimination error inflates TEST LENGTH (85% of the excess); difficulty error
corrupts the DECISION (b-only = worst stop-point arm, +0.096; falsely confident posterior).
The one-sentence arc stands; CONSEQUENCE is written two-channel. CAVEATS to carry: CAT CIs
join the R1 seed-clustered re-analysis (margins survive n=5 comfortably); decoupling is a
PARTIAL fix in decision terms (157%, say so); ~250-item real banks cap harm generality.
SPLIT: rejected for now.

### 9.05 RIGOR ADJUSTMENT (three-specialist consult wf_f15135cc + lead reconciliation, 2026-07-03)
EXECUTION OUTCOMES (wf_ef0a77e7, 2026-07-03; full table docs/exposure_rerun_results.md Phase 8):
spine SURVIVES clustered inference end to end (ladder generalized to dkvmn+transformer+2nd
geometry, slots prediction CONFIRMED; R10 kills the key-capacity rebuttal; flagship holds
apples-to-apples). THREE CHANGES BIND THE WRITING: (1) COLD-ITEM LEG DIES (item-resampled
CIs span zero) -- drop it as a claim, retract "location pools" as a law, re-anchor Attack-2
on online/sequential + no-calibration-catastrophe; (2) "5.6x" RETRACTED (Fisher-z ratio
0.76-0.92), two-law figure qualitative; (3) CROSSOVER REWRITTEN: no confirmed synthetic
reversal (E=15 was a naive false positive); decoupled wins from E~30-60 up, below is
UNRESOLVED; the practice rule softens to "decouple unless exposure is very thin (roughly
under a few dozen); at very thin exposure the contrast is unresolved on synthetic beds and
reversed in the one real option-level case." R8: real K=3 thresholds mostly disordered and
the neural ordering was an export artifact -- polytomous real demotion is now MANDATORY.
EDITOR REVIEW (wf_343e90a1): **GO.** Fidelity confirmed on all six blocking findings.
STRIKES: R11 struck as a run (becomes a [W] label: selection-alignment is single-cell
illustrative, not a law); R10 tightened to width-16 only (gpcm+2pl, lstm). RESTORES: none;
no-ASSISTments UPHELD contingent on R3 landing. Coherence strikes applied to S1 (3
contributions) and S7.2 (phrasing ban). SEQUENCING: Wave 1 blocks writing = R1, R3, R2, R7
(+R8); Wave 2 during writing = R4-verify, R9, R10; Wave 3 = R5, R12, R6 (variant-gated).
Consensus findings accepted; disagreements resolved by the lead as noted. Owners: [A]=analysis-only
CPU, [G]=small GPU run, [W]=writing.

R1 [A, BLOCKING] CLUSTERED STATISTICS. 25 folds = 5 seeds x 5 correlated CV splits; effective
n=5. Re-run EVERY headline CI with seed-clustered bootstrap (resample seeds, folds nested):
oracle ladder RUNG CONTRASTS (channel > theta-noise as a paired inferential test, not point
estimates), toggle deltas, crossover, cold tiers (also resample ITEMS within tier), flip
Jaccards, NRM crater (resample persons). "25/25 folds" language becomes "5/5 seeds".
R2 [A, BLOCKING] LADDER GENERALITY, both directions. Second ENCODER (research-scientist,
highest value near-free): per-item MLE on banked dkvmn + transformer theta_hat at Q=200
N=2000 -- this TESTS the slots story; if dkvmn's ladder contradicts it, the spine narrows and
we say so. Second GEOMETRY (ml-math + psychometric): lstm ladder at one more (N,Q) grid cell
from banked npz. All CPU.
R3 [G, BLOCKING] APPLES-TO-APPLES REAL DATA (psychometric): the head-to-head table needs
mirt and neural on the SAME response definition. Run FIRST-ATTEMPT-ONLY neural arms
(EdNet+KDD, 2pl+gpcm, shared+decoupled) for the comparison table; keep repeats-native neural
as a separate labeled column (and as the paper point about sequential models).
R4 [A, BLOCKING] CAT LINKING CHECK (psychometric): confirm the harm sim (i) generates
responses from the TRUE DGP only (verified in its gate) and (ii) LINKS each arm's EAP to the
true scale before cut-score classification (else misclassification is a gauge artifact); fix
+ re-run affected metrics if not. Report BOTH fixed-length and SE-stop conditions; EAP prior
fixed across arms.
R5 [W+A] E* DE-PINNED: report E* in (15,60], "near ~30, not a measured constant"; clustered
CI at E=30 shown. Deployable rule survives as a rough threshold.
R6 [W] PROPOSITION D, RESTRICTED FORM ONLY (ml-math): the theorem is the CRLB dichotomy
(per-item estimators, incl. mirt, have error O(1/E_i) independent of N at fixed E -- it
PREDICTS mirt-flat-in-N); the capacity floor delta_W is an explicitly LABELED assumption
with the width-plateau (+0.056 at 2.3x params) as evidence, never claimed proven. The
drafted "variance floor independent of N" is REFUTED by our own data and dies.
R7 [A] SCALING REFIT: Fisher-z (or log-error) refit of both slopes; recompute the "5.6x"
ratio; scope to measured E range; call it a descriptive law. Amortized half stays a
decoder-dependent spectrum (asymmetric framing enforced).
R8 [A] COERCED-ORDINAL CHECK (psychometric): report step-threshold ORDERING for the real
GPCM fits (mirt + neural); ordered -> the K=3 coding is defensible; disordered -> demote
harder. Plus a construct-validity paragraph.
R9 [W] POLYTOMOUS DEMOTION (ml-math + psychometric): ALL K>2 real-data results become
labeled "coerced-ordinal robustness checks"; NRM reversal scoped to EdNet with an explicit
possible-coercion-artifact caveat; polytomous-real gap a named limitation.
R10 [G] NARROW-KEY CONTROL (research-scientist): decoupled with key width 16 (and 32 if
cheap) on lstm gpcm+2pl -- if narrow-key decoupled beats shared-w96, the win is channel
SEPARATION, not key capacity. Pre-empts the symmetric rebuttal.
R11 [G] SELECTION-ALIGNMENT REPLICATION: trajectory-with-prediction on 2pl N=2000 and gpcm
N=500 (else label the property single-cell illustrative; do not present as law).
R12 [A/W] Smaller accepted items: TOST also on AUC/QWK (or scope blindness to accuracy);
multiplicity declaration (surface descriptive, formal inference only on pre-registered
contrasts); TOST band justified by decision relevance; rank-certifies-ordering-not-
calibration sentence; dkvmn-probe labeling; K-coverage phrasing (2PL supplies K=2); the
artifact ships runnable (generator, configs, metrics, seeds, CI scripts, both artifact
post-mortems as reproductions).

REAL-DATA DECISION (lead, from all three): NO new dataset now. EdNet+KDD carry the core arc;
their DISAGREEMENT (mirt wins EdNet-a, neural wins KDD-a) is itself the finding. Ship with
R8+R9 honesty. ASSISTments named in limitations as the reflex objection, deferred to
revision; EEDI only if the NRM boundary is ever made headline (it is not). No fourth ever.
Note for CAT scope: ~250-item real banks cap harm generality -- state it.

### 9.1 Evidence map (draft-1 S-blocks; roles per 9.0)
Status tags: [BANKED evidence on disk] [PENDING wf_f2675d5e GPU track] [WRITE writing-only].
Every claim keyed to outputs/ or docs/exposure_rerun_results.md. Register: main_magpcm_ijaied.

S1 INTRODUCTION [WRITE]
- Hook: KT models increasingly ship IRT readouts read as measurement (Deep-IRT line); the
  field's explainability current names missing evaluation of explanations as the central gap
  (Bai 2024; Yu 2026). Nothing certifies the numbers.
- Opening stake (the flip sentence) [BANKED outputs/p2_flip; REWORDED per P4 lens 2]: two
  models indistinguishable in accuracy (|d acc| 0.006) disagree on 56% of the items an
  instructor would flag as most discriminating; the SHARED model's flags are wrong against
  truth 61% of the time while remaining internally stable (within-arm overlap 0.80) --
  stable and wrong applies to the shared arm ONLY; the decoupled arm is stable and
  substantially more correct (0.69 vs truth). The two arms disagree while accuracy-tied;
  that is the certification gap.
- MML dominance stated PLAINLY in abstract + intro (P4 lens 2: not deferred to S7): on
  well-specified data a classical MML refit (0.94-0.98) beats every neural arm including the
  best decoupled one (0.94-0.96 at high N; far above shared); the readout's case is
  sequential/online use, cold items, and misspecified data, and the paper tests exactly
  those. Decoupling is the best-available in-model lever, not the winner over classical.
- Contributions, EXACTLY 3 (editor coherence strike applied; matches 9.0): (1) the CHANNEL
  MECHANISM with the oracle decomposition (both decoders; ladder generality per R2);
  (2) the DECISION-COST AUDIT of current practice (synthetic + real flip; CAT harm);
  (3) the exposure-controlled BENCHMARK + CERTIFICATION PROTOCOL, shipped runnable as one
  artifact. The 144-cell matrix, the classical head-to-head, the scaling figure, cold items,
  and the trajectory are VALIDATION AND BOUNDARY MATERIAL inside these three, not
  contributions.

S2 RELATED WORK [WRITE, docs/boost_refs.bib]
- KT + IRT readouts (Deep-IRT, Converse, option tracing); pyKT reform precedent (prediction
  side; we extend to measurement). Neural-IRT recovery line (VIBO person-side amortization
  mirror; VTIRT; ML2P-VAE; beta4). Amortized inference (Cremer gap, cite-as-known).
  Cold-start calibration (AutoIRT/BanditCAT content-based; Koenig Bayesian pooling).
  Probing bridge, one paragraph, cite-as-known (Alain-Bengio, Hewitt-Liang control tasks,
  Belinkov): the general point is known; the audited setting + classical control are new.
  Classical calibration floors (Bock-Aitkin; Ban 2001 -- NOT the NWEA 2000/item figure).

S3 FRAMEWORK AND BENCHMARK [BANKED + WRITE]
- The tracer: swappable encoder {lstm, dkvmn, transformer} x decoder {2pl, gpcm K4, nrm K4},
  prediction loss only, params read post hoc. Fixed L=60 spiraled administration -> exact
  per-item exposure E = N*60/Q (design contribution). 2D budget grid N x Q (16 cells x 9
  models x 25 folds). Metrics: rank primary (gauge argument), alpha-decile guards, category
  floors, last-valid theta. [outputs/p2_exposure]
- Evaluation-pitfalls subsection or appendix [WRITE]: the padded-column theta artifact (r
  0.49 -> 0.97), dense-bed artifacts, rank-vs-linear -- how recovery evaluations break;
  motivates the protocol. Artifact release statement.

S4 THE AUDIT (empirical laws) [BANKED]
- 4.1 Prediction is blind: accuracy flat (max 4.2pp) across arms differing 0.16-0.50 in
  discrimination recovery; the flip demo as Figure 1. [p2_toggle, p2_flip]
- 4.2 The laggard is systematic and encoder-dependent: full surface; discrimination lags
  difficulty in every lstm/transformer cell; transformer worst (0.42-0.67); dkvmn REVERSES
  (its laggard is difficulty). [p2_exposure; exposure_rerun_results S1-2]
- 4.3 Budget geometry, STATED WITH ITS BOUNDARY (P4 blocking fix): classical mirt is exactly
  flat-in-N at fixed E everywhere; the shared readout rises with total data at fixed E
  (anti-diagonal +0.13..+0.33) BUT the total-responses collapse is decoder-dependent (clean
  nrm R2 0.90, partial gpcm, fails 2pl where Q dominates via low per-response slope
  information). The clean universal half is the per-item law; the amortized half is a
  spectrum. No linking penalty anywhere. [p2_exposure, p2_mml, p2_scaling]
- 4.4 Over-training and model selection [BANKED p2_trajectory + p2_trajpred]: theta peaks at
  ep~25-125 then collapses in BOTH arms while training loss improves. MEASURED: in the
  SHARED arm validation prediction plateaus ~ep89 while discrimination recovery peaks ~ep167,
  so the prediction-selected checkpoint loses 0.099 discrimination recovery; in the DECOUPLED
  arm all peaks collapse together (~72-82) and the selection loss is 0.001. NEW decoupling
  property, promote to S6: decoupling ALIGNS the model-selection schedule with measurement,
  making prediction-based early stopping measurement-safe.

S5 THE ACCOUNT (mechanism) [BANKED]
- One paragraph of structure: the gauge (alpha*theta) + where each parameter reads from.
- THE ORACLE LADDER (the paper's Table 2; N=2000/Q=200): shared readout 0.719 -> per-item
  MLE on the SAME theta_hat 0.934 ~= decoupled readout 0.941 -> true-theta clamp 0.979 ~=
  mirt 0.982. Decomposition (P4 audit-corrected): CHANNEL 0.22, THETA-NOISE 0.05,
  INFORMATION RESIDUAL 0.003 (effectively zero: with theta fixed to truth, per-item MLE
  saturates the classical ceiling -- STRONGER than the earlier 0.02 misquote). The deficit
  is the shared channel, not amortization; decoupling recovers per-item-estimator behavior
  inside end-to-end training. NAMING: keep "two-regime"; the third term is a residual, not a
  regime. GENERALITY (P4 blocking item CLOSED): the 2PL ladder is BANKED -- shared 0.553 ->
  two-stage 0.874 ~= decoupled 0.898 -> clamp 0.959 ~= mirt 0.962 at N=2000; channel
  +0.32/+0.36 dominant at both N. T2 carries BOTH decoders. Honest refinement for S6: the
  2PL theta-noise term nearly doubles at low N (+0.084 -> +0.140), so "data cannot cure"
  is a claim about the DOMINANT CHANNEL term; the smaller errors-in-variables term is
  N-sensitive and stated separately. [outputs/p2_oracle both decoders]
- The two-law figure (Figure 2): mirt flat-in-N, slope 0.072/decade in E; the decoupled
  channel 0.404/decade (5.6x steeper -- a per-item channel needing far more exposure than
  closed-form MLE); shared arm rides total data. [outputs/p2_scaling]
- The regime reading unifies: dkvmn slots = built-in per-item channels (its laggard is
  whatever stays pooled); transformer = extreme pooling; NRM couplings (slope-only
  decoupling is pathological, one-key wins) as identification within the item block.

S6 LEVERS AND BOUNDARIES [BANKED]
- Decoupling: +0.16..+0.50 everywhere; ARCHITECTURAL for 2pl (shared plateaus 0.42-0.61 at
  N=5000; data cannot cure it); capacity control: widening shared plateaus +0.056 below
  decoupled in ALL THREE decoders at 2.3x the params. [p2_toggle, p2_width]
- The crossover: decoupled loses below E* in (15,30] (2pl; gpcm <=15); discrimination-
  specific. Practice rule: decouple unless items see < ~30 responses. [p2_crossover]
- Refuted levers, reported: dynamic head (never wins, collapses NRM slope), decoupling-
  delays-degradation (peaks higher and earlier, decays as fast). [p2_toggle, p2_trajectory]

S7 THE CLASSICAL CEILING AND WHEN THE READOUT EARNS ITS KEEP
- 7.1 Synthetic tax [BANKED p2_mml; audit-corrected]: mirt 0.94-0.98 dominates; decoupling
  cuts the tax from ~0.26 to ~0.02-0.04; mirt ~= true-theta oracle (delta 0.003).
- 7.2 REAL-DATA HEAD-TO-HEAD [BANKED, COMPLETE; verify-gate passed]. FINAL READING: the
  pre-registered prediction is CONFIRMED in the form "the tax SHRINKS TO NEAR-PARITY with
  parameter- and dataset-specific edges": mirt keeps EdNet discrimination (0.88 vs
  0.75-0.82); neural decoupled edges KDD discrimination (0.786 vs 0.699-0.711) and EdNet
  difficulty (0.946-0.959 vs mirt-robust 0.848-0.901); KDD difficulty ~parity. Which
  estimator to trust is PARAMETER- and DATASET-specific and the reliability screen is how a
  practitioner finds out. (Editor phrasing ban enforced: report as concordance +
  no-calibration-catastrophe + parameter-specific edges; never as a win, never "the protocol
  is the deliverable".)
  VERIFY-GATE OUTCOME (promote to the pitfalls section as case study #2): the initially
  observed mirt-difficulty "crater" (0.13-0.24) was an ARTIFACT -- Pearson on mirt's
  b = -d/a ratio parameterization, which explodes when near-zero discriminations flip sign
  between halves; sign-aligned Spearman gives 0.81-0.90. Our own pipeline briefly produced
  the exact rank-vs-linear artifact the certification protocol warns about, and the gate
  caught it. Report this openly; it is the protocol's best advertisement.
  Methodological paragraph: ~17% repeated learner-item pairs; mirt requires
  first-attempt-only (local independence), the sequential model consumes repeats natively.
  NRM RESOLUTION [BANKED]: the synthetic winner (one-key decoupled) craters real EdNet slope
  reliability to 0.065, WORSE than the pathological control (0.685/0.224); shared (0.695/
  0.707) is the only healthy arm. TRIPLE-CONFIRMED coupling-independent reversal: synthetic
  guidance does not transfer for option-level NRM; decoupling recommendation is explicitly
  SCOPED to location-scale decoders on real data. Still optional in this slot: the
  real-data flip exhibit (cheap CPU from persisted full-fit params).
- 7.3 COLD ITEMS [LANDED p2_coldstart; FINAL-AUDIT CORRECTED]. The reason-to-exist is REAL
  but PARAMETER-SHAPED, and it rests on DIFFICULTY, not coverage: (i) the coverage claim is
  RETRACTED (final artifact: mirt fit 49-50/50 starved items; the earlier 9/50 figure did
  not survive audit -- never use it); (ii) DIFFICULTY: the neural advantage is LARGER than
  first quoted: E=5 0.756/0.798 vs mirt 0.378 (2x), E=10 0.809/0.815 vs 0.465, E=20 neural
  clearly ahead (0.866/0.879 vs 0.773); (iii) DISCRIMINATION: mirt wins every tier -- cold
  scale is unrecoverable for everyone. LAW REFINEMENT (quotable, spine-consistent): location
  is recoverable through pooling; scale is not. Small-n caveat: 10 items/tier x 5 seeds,
  bootstrap CIs mandatory. Attack-2's neutralization now cites the 2x difficulty advantage +
  the KDD discrimination edge + online/sequential, NOT coverage.
- 7.4 Decision guide (half page): have the matrix -> refit classically; online/sequential ->
  readout with the discipline below; one sentence on streaming (fence: no system claim).

S8 PRACTICE: THE CERTIFICATION PROTOCOL [WRITE, from banked numbers]
- Checklist: rank metrics + alpha-decile guards; exposure floors (E>=~300 K=4 thresholds);
  decouple above E*; early-stop near the recovery peak, not the prediction plateau;
  split-half reliability screen (catches the EdNet NRM crater); report vs the classical
  ceiling when computable. Efficiency note: lstm ~= dkvmn recovery at 1/10-1/30 the compute.

S9 LIMITATIONS [WRITE]
- Well-specified synthetic core; K=4 single; per-eigenmode inversion below alpha~1; NRM
  identification; general faithfulness point known (control tasks), our setting new; not a
  learning-dynamics account (fence).

S10 CONCLUSION [WRITE]

STATISTICAL PROTOCOL (P4 completeness BLOCKING, applies to every section): fold-level
bootstrap 95% CIs on all recovery metrics; toggle contrasts adjudicated by paired fold-mean
difference + bootstrap CI + rank-biserial (the locked blueprint rule); the "prediction is
blind" claim gets a TOST-style EQUIVALENCE test on accuracy (pre-registered equivalence band,
e.g. +/-1pp) paired with the CI-separated recovery difference -- blindness = accuracy
equivalent AND recovery non-equivalent, stated formally. Prediction-competence anchors in
S4.1: majority-class baseline + the mirt model's own held-out AUC on the same beds (from the
existing fits) so "the encoders predict competently" is shown, not assumed.
DATA/ETHICS STATEMENTS (P4): EdNet (CC BY-NC 4.0, Choi 2020) and KDD Cup 2010 (DataShop
terms) licenses + anonymization noted; reproducibility statement (all seeds deterministic,
generator + drivers released).

FIGURE/TABLE PLAN (P4-revised): F1 flip demo; F2 two-law scaling; F3 budget surface
heatmaps; T1 benchmark matrix (with CIs); T2 oracle ladder (gpcm + 2pl when landed); T3
toggle deltas + crossover; T4 real-data head-to-head [PENDING, GATING for the Q1 case]; T5
NRM 10-config summary PROMOTED to main text (P4: not appendix-only); T6 protocol checklist.
Appendix: pitfalls post-mortem, timing/efficiency, full surface tables.

P4 VERDICT LOG (2026-07-03): panel = smallness lens, honesty lens, completeness critic
(opus) + evidence audit (sonnet; 14 numbers confirmed, 1 blocking mismatch corrected here).
SMALLNESS VERDICT: "not small anymore, CONTINGENT on S7.2 landing" -- the oracle ladder is
what converts the ablation pile into one law with corollaries; the flagship real-data leg is
the single remaining smallness risk. All BLOCKING items actioned in this draft-2: scaling-law
boundary promoted (spine + 4.3 + 5), 2PL oracle ladder BANKED (channel dominant, claim
holds with the theta-noise refinement), statistical protocol added, information residual
corrected to 0.003. ALL EVIDENCE SLOTS BANKED incl. the real-data flip exhibit (67%
disagreement on real EdNet at near-tied accuracy; mirt agrees more with decoupled).

FINAL GATE VERDICT (2026-07-03, wf_9adb954e): **READY FOR THE WRITING PHASE.** All six
op-def points MET; all three pre-registered attacks NEUTRALIZED with banked numbers; no
evidence slot open. New-numbers audit: 6 confirmed exact; 3 mismatches all resolved above
(cold-start coverage RETRACTED + difficulty STRENGTHENED; NRM directory labels noted).

BINDING WRITING INSTRUCTIONS (from the final adversary; enforce in prose):
1. NEAR-PARITY FRAMING ORDER: reason-to-deploy leads with cold-start difficulty +
   online/sequential + the protocol; the real-data head-to-head then establishes "no
   calibration catastrophe, parameter-specific edges, the reliability screen is the
   deliverable." Written flagship-first, near-parity reads as "the readout adds nothing."
2. NRM: a well-scoped BOUNDARY. Do NOT claim the spine predicts the real NRM reversal (its
   mechanism is unpinned; exposure starvation was shown non-dominant); present it as the
   boundary the reliability screen catches (0.065 flagged; ranking shown to be noise). Keep
   NRM out of the decoupling recommendation.
3. CRATER ARTIFACT: one tight paragraph as pitfalls case #2 (honest strength); do not
   over-dwell.
4. CIs on the small-n slots (cold tiers, flip Jaccards) per the statistical protocol.

## 8b. Venue posture note: outline is written venue-agnostic at journal depth; trim path to
CAEAI/TLT (lead 4.1+7) vs TMLR (lead 5+4.3) decided at writing time.

## 8. Venue posture (RESET by the hostile-editor ruling, 2026-07-03)

GATED ON EXPERIMENT E (CAT harm, running): MATERIAL harm -> IEEE TLT first ("certification
benchmark + channel mechanism, a solid reusable learning-technology contribution"), UMUAI
second (add the proposition D if airtight), CAEAI conditional on the harm size. NEGLIGIBLE
harm -> Q1 education not honestly reachable; TMLR as the mechanism paper (ladder + regimes
+ D), where near-parity is not disqualifying. JEDM remains the community anchor/safety.
The editor's decision-letter one-liners per venue are in the ruling (wf_b632b5ac).
All routes require surgery B (section 9.0) and the honest demotion of near-parity.
