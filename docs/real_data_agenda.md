# Real-data validation agenda

The plan for moving the framework from synthetic proof of concept to
real-data validation. Planned by a Fable-5 high-level pass (2026-06-12),
executed by the main loop dispatching Opus and Sonnet agents, deep-research,
and workflows. Fable-5 is reserved for high-level planning, never execution.

Companion to `substrate/RESULTS.md` (the experiment ledger) and memory
`thesis-vision`. This file is the forward agenda, not the result log.

## Where we stand

Everything except the SLAM respondent-transfer number is synthetic. The
format-agnostic, item-agnostic, dynamic-tracking, and continual-extension
claims are proven only on data generated from known parameters. This agenda
turns them into real-data claims.

Three load-bearing facts the planning pass surfaced from the code.
- The flagship joint-format result (`substrate/jointfmt/`) uses a static
  per-respondent theta, not the LSTM encoder. Dynamic format-agnostic is an
  unbuilt integration, not a config flag.
- `substrate/core/` has no masking, no mini-batching, and a single fixed K,
  so it cannot ingest real variable-length, mixed-format data as is.
- The repo already has written adapters for Eedi, ASSISTments, EdNet, and
  SLAM in `rl/src/ordrec/data/`, but only SLAM raw data is on disk.

## The three agnosticisms, defined operationally

- **Format-agnostic.** One shared item representation supports multiple
  response formats. Test, an item observed in only format X is correctly
  placed on format Y's scale, beating an independent-fits baseline. Proven
  synthetically (0.989 / 0.977 vs noise). Real version needs a corpus where
  the same objects receive responses in two or more formats with a bridge
  subset seen in both.
- **Item-agnostic.** The scale does not depend on a closed item bank. Two
  halves, parameter invariance (anchored extension recovers the same item
  parameters as a full recalibration, benchmarked against the full fit's own
  seed reliability, not against 1.0) and ability invariance (theta from
  disjoint item subsets agrees for the same learners).
- **Respondent-agnostic.** The same machinery and item scale apply across
  respondent populations. Test, item difficulty calibrated on population A
  rank-predicts difficulty in population B. Current real number, human vs
  small-LLM Spearman 0.341, lower-bounded by pool homogeneity.

## Research question battery

- **RQ1 FORMAT-REAL.** The jointfmt cross-format placement result on real
  mixed-format data. Metric, held-out-format placement Spearman vs an
  independent-fits baseline (the gap is the claim). Blocked on data, true
  mixed-format corpora are rare. Best candidate, comparative-judgment writing
  corpora with both rubric scores and pairwise judgments over shared scripts.
  Caveat, Eedi binary vs distractor-ordinal is not format-agnosticism (one
  event, two labels), that is RQ7.
- **RQ2 ITEM-REAL.** Anchored extension on a real bank. Split a real bank,
  base-calibrate on A, anchored-extend B, compare to full recalibration, plus
  theta-perturbation of base learners. Runnable now on SLAM. Live threat,
  real anchor learners are self-selected, not designed, so anchor selection
  bias is itself a reportable finding.
- **RQ3 RESPONDENT-REAL-2.** Firm the 0.341 keystone with a heterogeneous
  model pool (Llama, Gemma, Phi, plus deliberately weak models to widen the
  ability spread). Pass at >= 0.40 with bootstrap CI. Secondary human-only
  variant, calibrate SLAM items on one L1 track, test ordering on another.
- **RQ4 CROSS-TEST.** Ability alignment across instruments. Two layers,
  static (theta_A predicts theta_B near the attenuation-corrected reliability
  ceiling) and dynamic (within-learner net-drift in theta_A tracks net-drift
  in theta_B). Needs an instrument-overlap-gradient control, alignment should
  decay sensibly with construct distance, otherwise a null is unfalsifiable.
- **RQ5 IDENT.** Person and item separability. Three probes, instrument
  leakage (classify instrument from ability-matched theta trajectories, pass
  at chance), variance decomposition (person variance dominates instrument
  and interaction), and the counterfactual respondent (a fixed LLM answering
  two real banks, true theta constant by construction, any gap is pure
  instrument effect, a design no human dataset can give). RQ5c runnable soon.
- **RQ6 DYNAMIC-REAL.** Does dynamic tracking survive real longitudinal data.
  Real data has no ground-truth theta and forward-prediction collapses into
  the KT accuracy contest the thesis refuses. Needs an external criterion
  (lichess player ratings over time, or ASSISTments releases linked to
  state-test scores). Otherwise demoted to consistency evidence, labeled.
- **RQ7 COERCION-ROBUSTNESS.** Does the scale survive how raw behavior is
  coerced to a format (Eedi binary vs K=4, EdNet correctness vs time-coerced
  K=4). Cheap, real, inoculates against "your formats are relabelings."
- **RQ8 INVARIANCE/DIF.** Do anchor-linked item parameters replicate across
  subpopulations (Eedi age bands, SLAM L1 tracks). Folds into RQ2/RQ3 runs.

Dependency graph, RQ2 and its infrastructure gate RQ4/RQ5a/RQ5b/RQ7/RQ8.
RQ1 gates on the dataset hunt. RQ3 and RQ5c gate on nothing. RQ6 gates on an
external-criterion dataset existing.

## Dataset to property map

- **SLAM en_es (have, wired).** K=3 ordinal from per-token binary, three
  exercise formats, three L1 tracks, 30-day longitudinal. Tests RQ2 now, RQ3,
  RQ8, weak RQ4. Cannot test true mixed format or RQ6.
- **Eedi NeurIPS 2020 (adapter written, data not on disk).** 4-option MC, ~17M
  answers, topic metadata, age, timestamps, Task 3 has expert pairwise
  judgments. Tests RQ2 at scale, RQ7, RQ4 topic-splits, RQ5a/b, RQ8. Verify
  whether Task-3 pairwise is quality or difficulty before claiming RQ1.
- **ASSISTments (adapter written).** Binary plus hints/attempts, school-year
  longitudinal, some releases linked to state-test scores. Tests RQ6 with an
  external criterion, RQ2, RQ7, RQ4.
- **EdNet KT1-KT4 (adapter written).** TOEIC prep, parts 1-7 a natural
  multi-instrument structure, long histories. Tests RQ4 best, RQ2 at scale,
  RQ7. Large, needs aggressive subsampling on 8GB.
- **CJ writing corpora (the hunt).** Rubric plus real pairwise over shared
  scripts. The only real keystone candidate for RQ1. Biggest acquisition risk.
- **ASAP essays plus local-LLM pairwise (RQ1 fallback).** Real graded scores
  plus model-generated comparisons, a weakened but honest RQ1, doubles as
  RQ5c material.
- **lichess puzzle DB.** External item ratings and external player rating
  trajectories. The strongest available external criterion for RQ6.

Minimal covering set, SLAM (have) plus Eedi (acquire) plus one of {CJ corpus,
ASAP+LLM judges} for RQ1, plus lichess only if RQ6 is pursued. EdNet is a
scale-up luxury, not a necessity.

## Model gaps to close (scoped by the P0.3 audit)

1. No real-data path, `fit()`/`_compute_nll` assume rectangular tensors with
   no mask, no ordrec-adapter to substrate bridge.
2. Full-batch only, will not scale to Eedi/EdNet on 8GB.
3. Single fixed K, mixed-format histories unrepresentable.
4. jointfmt uses static theta, the encoder has never driven the shared
   multi-head decoder. The one genuinely new architecture task.
5. Bradley-Terry extension unimplemented.
6. Anchored extension assumes designed anchor learners with pinned theta.

## Execution plan

### Phase 0, parallel information gathering (dispatched 2026-06-12)
- **P0.1** deep-research, theory front (invariance/separability/CJ-graded
  linking/continual calibration), novelty verdict per RQ.
- **P0.2** deep-research, dataset acquisition playbook, the CJ-corpus yes/no.
- **P0.3** Sonnet, code audit, capability matrix against the six gaps.
- **P0.4** Sonnet, SLAM item-split feasibility, anchor-overlap numbers.

### Phase 1, gated experiments
- **P1.1** real-data bridge, masking, mini-batching, ordrec-to-substrate
  module. Sonnet. Gate, P0.3 confirms feasibility. Blocks P1.2/P1.3.
- **P1.2** SLAM anchored extension, RQ2. The first real experiment. Sonnet
  build to an Opus design brief and Opus interpretation (anchor-bias and
  reliability-ceiling benchmarking are subtle). Gate, P1.1 plus P0.4.
- **P1.3** Eedi acquisition plus battery (RQ2 scale, RQ7, RQ4a/RQ5a/RQ5b with
  the overlap-gradient control). Gate, P0.2 access plus P1.2 pipeline.
- **P1.4** RQ1 real mixed-format, Path A real CJ corpus, Path B ASAP plus
  local-LLM judges (disjoint model families to avoid circularity). Gate, the
  P0.2 CJ verdict. Independent of P1.2/P1.3.
- **P1.5** RQ3 pool diversification plus RQ5c counterfactual respondent.
  Sonnet, long Ollama jobs via main-loop background bash only, single-flight.
  Gate, none.
- **P1.6** dynamic-encoder times joint-format integration, the new
  architecture. Sonnet build to an Opus brief. Gate, P0.1 novelty plus a real
  RQ1 target.

### Phase 2 (not dispatched)
RQ4b dynamic cross-instrument on EdNet, RQ6 on lichess or ASSISTments if P0.2
confirms, RQ8 DIF folded into P1.3/P1.5, writeup integration.

## Phase 0 verified findings (2026-06-12)

The dataset deep-research (adversarially verified, 23 of 25 claims confirmed)
revised several pre-research assumptions.

- **No public educational comparative-judgment plus graded-scores corpus
  exists.** The CJ / ACJ / No More Marking / Ofqual targets were not confirmed
  public with both signals on the same items. Nearest verified dual-signal
  corpus is JudgmentBench, but it is LLM-generated legal work judged by
  attorneys, no learner population. RQ1's clean keystone has no off-the-shelf
  dataset. Open paths, ASAP plus local-LLM judges (ASAP itself unverified
  here, a coverage gap), cross-dataset assembly, or an institutional CJ corpus.
- **Eedi is weaker than modeled.** Math-only 4-way multiple choice, binary
  plus unordered categorical (not ordinal). Task-3 pairwise is expert QUALITY
  judgment, not difficulty, AND the labels are not released. Question images
  are use-restricted. Eedi supports binary anchored extension, longitudinal
  tracking, and person/item separability only. It cannot serve RQ1 or any
  ordinal claim.
- **EdNet is the clean large-scale binary testbed.** 131M interactions,
  784k students, parts 1-7 a natural multi-instrument structure (the best
  available substrate for RQ4 cross-test alignment), direct download, CC
  BY-NC. Binary multiple-choice only, no external criterion (Scores table
  unreleased). This raises EdNet's priority above Eedi, reversing the earlier
  "cut EdNet" note.
- **SLAM all three tracks are public** on Harvard Dataverse (DOI
  10.7910/DVN/8SWHNO, CC BY-NC), es_en 35 MB and fr_en 16 MB beyond the en_es
  we have. A zero-friction respondent-transfer-across-populations test.
- **External criterion for RQ6 exists but is gated.** ASSISTments has an
  MCAS state-test linkage (1,393 students, ordinal four-category) and an NSC
  college-enrollment linkage (3,747 students), both via the WPI/institutional
  research route plus a data use agreement, not bundled downloads. The 2017
  competition links to STEM-vs-non-STEM first job. lichess and ASAP were not
  verified (coverage gap), so RQ6's external-criterion path stays uncertain.
- **Licenses.** EdNet and SLAM are CC BY-NC (non-commercial). Plan accordingly.

The theory sweep (P0.1, adversarially verified) returned a strong novelty
verdict, recorded in full in memory thesis-originality-sweep. Headlines.
- The cross-format shared NEURAL scale (one embedding read by a GPCM and a
  Bradley-Terry decoder) is novel. Every component is classical and separate,
  Kim and Lee 2006 mixed-format linking, Andrich 1978 (Bradley-Terry equals
  Rasch, the load-bearing cite for one logit scale), Brown and
  Maydeu-Olivares 2012 forced-choice to IRT. Nobody neuralized the fusion.
  This raises the value of RQ1 even on an imperfect testbed.
- Separability is not free (VIBO independent 0.1 vs conditional 0.9), and
  predictive AUC near 0.85 coexists with ability recovery near 0.4 to 0.5,
  which validates the rank-recovery metric and the refusal of the prediction
  trap. RQ5 is meaningful.
- Reframe the anchoring story. Xi and Bloem-Reddy 2023 show neural generative
  scales carry indeterminacies unresolvable even with infinite data unless an
  anchor or structured prior is imposed. Anchoring is the identifiability
  mechanism the theory requires, not borrowed cost-saving plumbing. This is a
  stronger positioning than the deck's current apologetic framing.
- Cross-instrument invariance of a learned scale is the least explored track,
  a true open gap. That is exactly RQ4 and RQ8.

Net effect on the plan. RQ2 (item, SLAM, running) and RQ3 (respondent, SLAM
plus tracks) keep clean paths. RQ4 should target EdNet parts, not Eedi. RQ1
needs a path decision (no clean corpus) but is confirmed novel and worth
pursuing. RQ6 stays the cloud-castle pending an external criterion. Phase 0
is complete.

## Highest-value first experiment

P1.2 SLAM anchored extension. Zero acquisition risk (data and adapter in
repo), converts a core synthetic-only claim to real, honestly posed without
ground truth (reference is full recalibration plus a seed-reliability ceiling,
no synthetic theta, no KT-accuracy trap), forces the masking/batching/bridge
infrastructure every later experiment reuses, and a failure is maximally
informative because it would cap the continual story everywhere.

## Critical triage

- **Answerable now.** RQ2 (SLAM in hand), RQ3 (cheap, local), RQ5c (the
  counterfactual-respondent design is the most original method in the battery,
  human data cannot produce it). RQ8 rides along nearly free.
- **Needs data we lack.** RQ1 properly (be ready for "no public CJ corpus
  exists", the ASAP+LLM fallback is respectable but must be stated as real
  scores with model-generated comparisons). RQ4 and RQ5a/b need a
  multi-instrument acquisition.
- **Ill-posed or premature.** RQ6 is the biggest cloud-castle, without an
  external trajectory it degenerates into the prediction trap, pursue only if
  Phase 0 confirms a criterion (lichess, at the cost of leaving education).
  RQ4 is unfalsifiable without the construct-overlap-gradient control. RQ5 as
  first phrased is not yet a measurement, the three probes are the proposed
  operationalization, sanity-check against the P0.1 literature first. Eedi
  binary-vs-ordinal as mixed format is partly circular, useful as RQ7 only.
- **Cut if forced.** EdNet (scale pain, marginal coverage over Eedi) and RQ6
  unless lichess survives Phase 0.
