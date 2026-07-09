# Paper v3 plan — Structured Response Heads for KT

Status: PLAN ONLY, paper untouched. Reconciles the current manuscript (v2,
jedm/main after the user's restructure) with
`kt_structured_response_heads_revision_plan.md` (the revision plan). v2 is
already mid-migration: Related Work is reshaped, `[TO-BE-REMOVED]` marks sit on
the decomposition and CAT blocks, TIMSS/EdNet case-study sections exist as
empty stubs, title is TBD. v3 = finish that migration, add the two new
evidence threads, and resolve the tensions below.

---

## 1. What v2 establishes (the asset inventory)

The claim chain as it stands, each item an asset v3 should not waste:

- **C1 Setting.** KT models with IRT-parameterized response heads: prediction
  is the only training objective; the head exposes named ability/item/option
  parameters. IRT is a decoder family, not a calibration procedure. (Framing
  already ML-native after the register rewrites.)
- **C2 Theory.** Gradient-path analysis of the shared head: the shared item
  embedding carries both an encoder-input (prediction) gradient and a
  parameter-score gradient; the separated key receives the score alone, whose
  stationary point is the per-item ML condition. Predicts *which* parameter
  groups suffer: weak-score groups (ordered discrimination) and
  symmetry-sensitive groups (nominal option-slope orientation, cured by the
  oriented NRM head). Predictions were recorded before the runs and held.
- **C3 Synthetic validation.** Across 3 encoders x 3 decoders x exposure
  conditions: separation lifts discrimination/option-slope recovery (~0.6 to
  ~0.9+) at prediction parity (within 1pp); ability recovery is preserved
  (fig_surface, now ability-only) and the shared NRM's ability collapse is the
  sharp exception; width sweep shows capacity does not substitute for
  separation (data exist for ALL THREE decoders, not just LSTM-GPCM — the
  revision plan's fear is moot); growth benchmarks show more data does not cure
  pooling.
- **C4 Stable-and-wrong.** Prediction accuracy, retraining stability, and
  split-half reliability all pass while parameters are wrong; only agreement
  with generating values catches it (tab:ritual, now fully synthetic). This is
  v2's sharpest rhetorical asset and v3's biggest landmine (see T1).
- **C5 Decision-level price.** CAT simulation: the recovery gap doubles test
  length and corrupts cut-score decisions; two-channel attribution
  (discrimination -> length, difficulty -> decisions). Currently marked
  TO-BE-REMOVED.
- **C6 Real data (prediction only).** The IRT head costs nothing vs direct
  prediction (DKT) on binary/ordinal and HELPS on nominal (~0.65 vs ~0.53
  option accuracy; classical MML at chance there, partly a coverage artifact);
  TIMSS QWK ~0.40 for every source. Recovery/concordance/δ on real data were
  run, came back encoder-dependent and unresolvable without truth, and were
  deliberately retired (docs/real_data_section_draft.md records the full story
  and the psychometric verdict).
- **Retired, stays retired.** The truth-free δ detector; real-data recovery
  claims; state-conditioned (dynamic) heads (failed the planted-heterogeneity
  gate: recovers ~0% of known α(state), hallucinates heterogeneity on static
  data — private, a rebuttal asset only).

## 2. What the revision plan brings

- **Repositioning.** From "audit of parameter recovery" to "structured
  response heads extend KT beyond binary correctness." Recovery becomes the
  *design justification*, not the paper.
- **A real-data story that needs no ground truth.** Prediction compatibility
  (have) + parameter *stability* (new) + substantive *response tracing* (new:
  TIMSS ordinal, EdNet nominal case studies). This is precisely the escape
  from the trap C6 hit empirically: real-data recovery validation is
  structurally impossible (no truth; the classical reference is a different
  estimand; the winner flips by encoder). Stability + tracing give real data a
  job it can actually do.
- **Demotions.** Refit ladder / gap decomposition and CAT to appendix; MML to
  "offline reference" in one paragraph; "Corrections to the Recovery
  Analysis" cut entirely.
- **The nominal win becomes load-bearing.** "Beyond binary correctness" is
  exactly where the heads *win on prediction* (NRM 0.65 vs direct 0.53, MML at
  chance). v2 buried this in a table; v3 welds the framing and the strongest
  real-data number together.
- **New artifacts.** Evaluation-design table; ΔRecovery-vs-ΔPrediction
  figure; stability table + stability-vs-exposure figure; TIMSS composite
  figure; EdNet composite figure; new title/abstract.
- **Venue widening.** Less JEDM-bound; CAEAI/IJAIED if case studies lead,
  DMKD/UMUAI if the structured-head/stability angle leads.

## 3. Tensions v3 must resolve (and the resolutions)

- **T1 — Stability vs tab:ritual.** v2's core sentence is "parameters that are
  wrong the same way every time look reliable"; v3's real-data evidence is a
  consistency measure. These collide unless the roles are explicit.
  *Resolution: a three-role evidence model, stated in one place and used
  everywhere.* (i) Synthetic recovery validates the DESIGN under known truth;
  (ii) real-data stability is a GATE — necessary, never sufficient: unstable
  parameters are uninterpretable, stable ones are merely *eligible* for
  inspection; (iii) case studies show UTILITY, what the parameters let a
  reader do. Keep tab:ritual in the discussion as the reason stability alone
  certifies nothing. Corollary rule: **stability is reported for the adopted
  (separated) design as an interpretation gate, never as an SH-vs-SK contest**
  — on our own data the shared head is sometimes the *more* stable one
  (smoothing), so a stability contest would argue against our fix and against
  our own theory of why consistency misleads.
- **T2 — One fix vs three contributions.** v2 was deliberately "one
  theory-backed fix." The plan's 3-contribution list re-expands scope. Fine if
  serial (concept -> fix -> use), but the verbs must stay calibrated: we
  *study* response heads (Deep-IRT lineage exists; not "introduce"), we
  *introduce* the separated path, we *evaluate* on real data. The fix stays
  the technical core.
- **T3 — Losing CAT weakens "why care."** v3's substitute motivation: wrong or
  unstable parameters mislead exactly the inspection uses the case studies
  showcase (item review, distractor analysis). One pointer sentence to the
  appendix CAT result keeps the assessment-side price available without
  pulling the paper into CAT.
- **T4 — Where the decomposition lives.** The full ladder goes to appendix,
  but the *amortization-gap* concept (refit-vs-head contrast on the model's
  own states) is the paper's most citable idea and the theory's measurement
  arm. Keep a compressed paragraph in the theory section: define Δ_amort in
  two sentences + pointer. Do not bury the name.
- **T5 — Terminology migration.** One sweep, one glossary: SH/SK (or
  shared/separated — pick once), "structured response heads" vs "IRT decoder,"
  "recovery" = synthetic-only vocabulary, real data speaks
  stability/agreement/tracing. Also decide "separated key (SK)" vs v2's
  "separate heads" — the plan's SK naming is cleaner; adopt at rewrite time.
- **T6 — MML-at-chance on NRM.** v3 leans harder on the nominal case study, so
  the MML collapse row will draw scrutiny; state the coverage artifact (only
  ~1/3 of items calibrable; uncovered items scored at chance) in the caption,
  not just prose.
- **T7 — Venue.** The case studies strengthen the education-facing fit;
  recommendation: stay CAEAI-first (the plan itself lists CAEAI "if
  educational case studies are emphasized"), keep DMKD/UMUAI as fallbacks.
  User decides.

## 4. v3 thesis and claim chain

One sentence: *IRT-parameterized response heads extend neural KT beyond binary
correctness; a theory-motivated separation of the item-parameter path makes
their named parameters recoverable (shown synthetically), stable (shown on
real data), and usable for partial-credit and option-level response tracing
(shown on TIMSS and EdNet).*

Chain, mapping v2 assets to v3 seats — nothing is wasted:

| v2 asset | v3 seat |
|---|---|
| C1 setting + heads family | §Model Framework (compressed; SH/SK figure kept, recaptioned "path separation") |
| C2 gradient theory | §Theory-Motivated Predictions (P1–P5 compact prose; derivations -> appendix) |
| Oracle decomposition | 1 paragraph (Δ_amort definition) + appendix ladder |
| C3 synthetic grid | §Synthetic Validation: summary table (new Δ columns), scatter, NEW Δ/Δ figure, width figure (all 3 decoders), fig_surface (ability, no-cost) |
| C4 tab:ritual | §Discussion "why prediction accuracy is not enough" |
| C5 CAT | Appendix + one pointer sentence |
| C6 prediction benchmark | §Real-Data Prediction Compatibility (extend metrics) |
| NRM oriented head | Part of the design (the symmetry-sensitive P5 case) |
| — new — | §Real-Data Parameter Stability (table + exposure figure) |
| — new — | §TIMSS Ordinal Tracing, §EdNet Option Tracing (currently empty stubs) |
| "Corrections to the Recovery Analysis" | CUT (plan's call; reads as lab notes) |

## 5. Evidence map: have vs need

**Have on disk, analysis-only (no training):**
- Item parameters for stability: 5 seeds x 5 folds x 3 encoders per real cell
  (2PL EdNet/KDD α,β 250; TIMSS GPCM α 31; EdNet NRM slopes+intercepts
  4220x4). Split(fold)-, seed-, and exposure-stratified Spearman all
  computable now. Encoder-stability optional from the same files.
- Width sweep: `outputs/p2_width`, SH w∈{8,16,32,64,96} + decoupled, all three
  decoders — Figure "width is not separation" is plot-only.
- Synthetic grid for the Δ/Δ figure and the reshaped summary table.
- Prediction benchmark incl. direct/DKT + MML rows (realdirect_table).

**Need — new compute, all small (E1–E2 are the only blocking items):**
- **E1 Export re-inference pass** (LSTM folds on TIMSS-GPCM + EdNet-NRM, and
  2PL for completeness): current fold JSONs store one summary difficulty per
  item (TIMSS beta is (31,), K=3 needs 2 step thresholds) and NO θ
  trajectories. Case studies and threshold-stability rows need: full GPCM step
  thresholds, NRM slopes+intercepts (verify), θ_t per learner, final θ per
  learner (for ability bands). Minutes per fold on LSTM; a `_p2_`-prefixed
  exporter reusing the realstudy loaders.
- **E2 Extended prediction metrics** for Table 3: AUC + NLL (binary), NLL
  (ordinal), option-accuracy + macro-F1 + NLL (nominal). Logits were not
  saved; light re-score pass, can share E1's runs.
- **E3 Stability computation + Fig stability-vs-exposure** (post-hoc on Have).
- **E4 Case-study analyses** (post-hoc once E1 lands): TIMSS category-
  probability curves, expected-score curves E[Y|θ], threshold distribution,
  2–3 learner trajectories; EdNet option-probability curves, correct-option
  slope-orientation distribution, distractor attractiveness by low/mid/high θ
  band, intercept-vs-option-frequency check, binary-vs-nominal comparison.
- **E5 Δ/Δ figure; E6 optional encoder stability; E7 optional MML
  high-exposure difficulty agreement ("agreement with offline reference,"
  never "validation").**

**Explicitly not doing:** real-data recovery/concordance, δ, dynamic heads,
new synthetic campaigns, retraining any shared/separate cells.

## 6. Figures and tables (target set)

Main paper: T1 evaluation-design table (replaces tab:beds role); F1 SH/SK
architecture (recaption); T2 synthetic summary with Acc_SH/Acc_SK/ΔAcc +
Recovery_SH/Recovery_SK/ΔRecovery blocks per parameter group + MML column; F2
recovered-vs-true scatter (simplified caption); F3 Δrecovery-vs-Δprediction;
F4 width-vs-separation; F5 fig_surface (ability); T3 real prediction
compatibility (adds Direct + metrics); T4 real stability (groups x
split/seed/exposure; "stability," never "invariance"); F6
stability-vs-exposure; F7 TIMSS composite (A category curves, B expected-score
curves, C learner trajectories); F8 EdNet composite (A option curves, B slope
orientation, C distractor-by-band). Appendix: full grid, ladder/decomposition,
CAT (incl. trade_off_shared if kept), capacity/robustness extras.
`pareto_escape.png` is committed but unused — user call: intro concept figure
or drop.

## 7. Writing plan (section by section)

1. **Title + abstract last.** Plan's best title: "Structured Response Heads
   for Knowledge Tracing Beyond Binary Correctness" (user owns the call; v2
   title is already TBD). Abstract follows the plan's draft direction but must
   keep one v2 sentence-idea: same objective, same accuracy, different
   parameters — that is why any of this matters.
2. **Intro:** plan's 6-step logic (KT -> binary collapse -> richer responses
   exist -> structured heads -> named ≠ interpretable -> what we do). Salvage
   v2's current opening (already close). State the non-goal explicitly: not a
   replacement for classical calibration.
3. **Related work:** already restructured (three subsections) — light touch.
4. **Model framework:** compress heads; gradient theory as P1–P5 prose +
   Δ_amort paragraph; derivations to appendix. Execute the two TO-BE-REMOVED
   blocks (move, don't delete — appendix).
5. **Experiments/design:** fold in T1; metrics subsection gains the
   stability definitions (split/seed/exposure Spearman) and the tracing
   quantities (P(Y=k|θ,item), E[Y|θ,item], option curves).
6. **Results:** synthetic (tightened, + F3/F4) -> real prediction (extended
   T3) -> real stability (T4+F6) -> TIMSS case study -> EdNet case study.
7. **Discussion:** what heads add to KT; why prediction is not enough
   (tab:ritual); how to read real-data parameters responsibly (no truth
   claims, exposure-gated, stability-gated). Cut "Corrections" subsection.
8. **Consistency sweep at the end:** terminology (T5), recovery-vocabulary
   audit, caption/prose agreement, banned-word list still enforced.

## 8. Execution order

- **Phase 0 (compute, ~half a day):** E1 exporter + runs; E2 metric re-score.
  Unblocks everything else.
- **Phase 1 (analysis):** E3 stability + F6; E5 Δ/Δ; F4 width plot; extend T3.
- **Phase 2 (case studies):** E4 analyses -> F7/F8 composites -> draft the two
  stub sections.
- **Phase 3 (restructure):** execute TO-BE-REMOVED moves to appendix; compress
  theory; new T1; discussion rewrite; cut Corrections.
- **Phase 4 (final):** abstract + title + intro final pass; T5 terminology
  sweep; full consistency + compile gate.
Each phase lands as its own commit(s); paper compiles at every step.

## 9. Open decisions (user-owned)

1. **Title** (plan's candidate vs alternatives) — needed by Phase 4 only.
2. **Venue** — CAEAI-first recommended; affects register nuance, not structure.
3. **SH/SK notation** adoption vs "shared/separated head" prose.
4. **Branch** — v3 on `main` in place, or on the existing `jedm/rewrite-ml`
   branch (currently a stale pre-restructure snapshot; would need reset).
5. **CAT in appendix vs cut entirely** (plan allows either; appendix
   recommended — it is finished work and answers "so what" cheaply).
6. **pareto_escape.png** — use or drop.
7. **Stability presentation** — confirm the T1 rule (gate for the adopted
   design, no SH-vs-SK stability contest).
