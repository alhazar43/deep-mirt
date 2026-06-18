# DuoLingo Mini Research Appendix

Appendix to `docs/duolingo_mini_plan.md`. Tables and citations that
support the plan's four research streams. Narrative that repeats plan
Sections 1, 4, 5, and 6 is trimmed here; read those sections for the
full framing and adversarial-verification prose.

---

## Stream 1. Duolingo collaboration landscape

### Gap named by Duolingo's own publications

| Paper | Venue | Binary-IRT gap stated |
|---|---|---|
| Interactive Reading Task (Attali et al. 2022) | Frontiers in AI | "Need for psychometric models combining discrete and continuous grades" |
| AutoIRT (Sharpnack et al. 2024) | arXiv 2409.08823 | Binary 2PL/3PL only; extension to richer item types acknowledged |
| BanditCAT (Sharpnack et al. 2025) | PMLR v264 workshop | Binary CAT; same open extension |
| BERT-IRT (Yancey et al. 2024) | BEA at ACL | Item cold-start via BERT features; binary only |
| Jump-Starting (McCarthy et al. 2021) | EMNLP | Parameter transfer for adaptive tests; binary |

### SLAM 2018 dataset summary

| Property | Value |
|---|---|
| License | CC0 |
| Source | Harvard Dataverse doi 10.7910/DVN/8SWHNO |
| Mirror | github.com/NYUCCL/duolingoSLAM |
| Tokens | ~7M |
| Learners | 6000+ |
| Tracks | en_es, es_en, fr_en |
| Exercise formats | reverse_translate, reverse_tap, listen |
| Label granularity | Per-token binary (correct/incorrect) |
| Ordinal coercion | Aggregate to per-exercise proportion-correct, threshold to K=3 or K=4 |
| Ground-truth theta | None. Prediction metrics only (AUC, log-loss). |

### Ordinal coercion scheme (K=3)

| Label | Condition |
|---|---|
| 0 (all-wrong) | proportion_correct == 0 |
| 1 (partial) | 0 < proportion_correct < 1 |
| 2 (all-correct) | proportion_correct == 1 |

Thresholds fit on train split only, persisted to coercion_artefacts.json.

Outreach sequencing and collaboration-path ranking: see plan Section 6.

---

## Stream 2. Mixed-format positioning: ma-irt as a new look at KT and IRT

### The three literatures and where ma-irt sits

| Literature | Method | Sequential | Ordinal | Calibrated scale | Amortized |
|---|---|---|---|---|---|
| Classical IRT | GPCM (Muraki 1992), MML (Bock and Aitkin 1981) | No (per-occasion re-fit) | Yes | Yes | No |
| Dynamic IRT | Bayesian state-space (Wang, Berger, Burdick 2013) | Yes | Possible | Yes | No |
| Deep KT | DKT (Piech et al. 2015), DKVMN (Zhang et al. 2017), AKT (Ghosh et al. 2020) | Yes | No (binary) | No | Yes |
| Deep-IRT (Yeung 2019 EDM) | 1PL + DKT | Yes | No | Unanchored | Yes |
| KTM (Vie and Kashima 2019 AAAI) | FM over side features | Partial | No | No | Yes |
| ma-irt | DKVMN/LSTM/Transformer + GPCM | Yes | Yes (K>=2) | Yes (alpha-norm) | Yes |

ma-irt occupies the empty cell: sequential, ordinal, calibrated, amortized.

### Single-theta claim: what it requires and what the evidence shows

Requirement: formats must load on one construct. Tested by cross-format
theta concordance on synthetic interleaved histories with known parameters.

Evidence on synthetic K=4: r_theta ~0.96, r_beta >0.95 (IJAIED main result).
Evidence on mixed-format: not yet measured (D3-D4 target).
Falsification control: remove anchor items, theta correlation should
collapse to chance.

### Defensible novelty statement

"We combine concurrent calibration and sequential ability estimation in
one amortized forward pass. Any grader that compresses a signal to an
integer in 0..K-1 lands on one maintained theta. No existing deep KT
model offers a polytomous, calibrated, format-agnostic sequential
estimator. Deep-IRT is the closest KT-side ancestor (1PL,
unanchored, binary). Classical concurrent calibration is the closest
IRT-side ancestor (polytomous, calibrated, per-occasion)."

### Mixed-format experiment design (D3-D4)

| Component | Description |
|---|---|
| Two format types | Format A (e.g. binary vocab), Format B (e.g. partial-credit dictation) |
| Shared anchor items | Items appearing in both format streams, fix scale across formats |
| Ground-truth | Known (theta, alpha, beta) from the synthetic generator |
| Target 1 | Cross-format theta concordance is high (r >= 0.9) with anchors |
| Target 2 | Alpha and beta recover within each format at synthetic ceiling |
| Target 3 | Held-out format predicted better than format-specific baselines |
| Falsification | Remove anchor items; theta correlation degrades to near-chance |

### Related-work citations for the positioning section

Muraki (1992). Applied Psychological Measurement.
Bock and Aitkin (1981). Psychometrika.
Wang, Berger and Burdick (2013). Annals of Applied Statistics.
Piech et al. (2015). NeurIPS.
Zhang, Shi, King, Yeung (2017). WWW.
Ghosh, Heffernan and Lan (2020). KDD.
Yeung (2019). EDM.
Vie and Kashima (2019). AAAI.
Rodriguez (2003). Journal of Educational Measurement.
Bradlow, Wainer and Wang (1999). Psychometrika.
Stocking and Lord (1983). Applied Psychological Measurement.
Kolen and Brennan (2014). Springer.

---

## Stream 3. Reality check: enable/disable classification

### Classification rules

- ENABLED-as-is: zero code anywhere, architecture already handles it.
- ENABLED-via-rl-adapter: one new adapter subclassing OrdinalDatasetBase,
  zero ma-irt source edits.
- NEEDS-minor-additive-ma-irt: small, flagged, reviewed-worktree edit
  (20-50 lines across at most two ma-irt files).
- BLOCKED-without-overhaul: requires new forward-signature, new model
  branches, or unresolved methodology. Out of scope.

### Full classification table

| Direction | Status | Key constraint |
|---|---|---|
| Rubric/LLM-graded open responses as GPCM items | ENABLED-as-is | Decoder forward is `(questions, responses)`, never inspects elicitation |
| SLAM per-token to per-exercise ordinal coercion | ENABLED-via-rl-adapter | Mirrors EdNetAdapter, one new class, ~100-150 lines |
| Response-time integration (new coercion table) | ENABLED-via-rl-adapter | K=6 fast/medium/slow coercion, zero ma-irt edits |
| Multi-session gap-token forgetting | ENABLED-via-rl-adapter | Reserve one item ID as session-boundary marker; honest caveat (learned perturbation, not decay) |
| Format-effect estimation (per-format alpha contrast) | ENABLED-via-rl-adapter | Post-hoc join of recovered alpha/beta table with format metadata |
| Encoder-invariance probe | ENABLED-as-is | Existing encoders + evaluate harness, zero new code |
| Cite-and-compare positioning | ENABLED-as-is | Prose only |
| Cross-test anchoring via merged ID space | NEEDS-minor-additive (FLAGGED) | No ma-irt source edit, but config n_questions raise + retrain = research-design commitment |
| Mixed-K item banks via per-category mask | NEEDS-minor-additive (FLAGGED) | ~20-50 lines: -inf mask in trainer._flatten_mask and GPCMLogits.forward; per-item K table in adapter schema |
| Item cold-start on unseen items | BLOCKED | q_embed sized to training bank at dkvmn.py:146; no feature-based pathway |
| D>1 multi-dimensional traits | BLOCKED | Rotation indeterminacy unresolved; user-confirmed out of scope |
| Response time as continuous third input | BLOCKED | Encoder forward signature fixed |
| Adaptive FORMAT selection in OrdRec action space | BLOCKED | Action space is flat item IDs; inflate-IDs workaround needs mandatory retrain |

### Code pointers for the two NEEDS-minor-additive items

Mixed-K per-category mask (D8):
- `ma-irt/training/trainer.py` lines 262-275: `_flatten_mask` applies a
  uniform K mask; needs an optional `per_item_K` argument.
- `ma-irt/models/components/irt.py` line 123: `GPCMLogits.forward`
  unconditionally emits K-1 thresholds; needs a -inf fill for positions
  above each item's true K_i.

Cross-test anchoring (D7):
- No ma-irt source edit required.
- rl/-side merged-ID adapter assigns unified IDs across two pseudo-banks.
- Config: raise n_questions to the merged bank size, retrain from scratch.
- Flagged because the retrain is a research-design commitment, not because
  it requires touching ma-irt.

### Ranked plan scope

The ranked opportunities in the plan (Section 6) draw only from
ENABLED-as-is, ENABLED-via-rl-adapter, and the two explicitly flagged
NEEDS-minor-additive items. All four BLOCKED directions are confined to
Appendix B of the plan and do not appear in the milestone sequence.

---

## Stream 4. AI-era struggles: key facts table

For the narrative framing and collaboration-ranking rationale, see plan
Section 5.

| Fact | Date | Detail |
|---|---|---|
| AI-first backlash | Apr 28 2025 | Von Ahn memo phasing out contractors, accepting "small hits on quality", social purge May 17 2025, two CEO walk-backs through Apr 2026. The quality concession became a lasting liability. |
| Business pressure | Feb 2026 | Stock down ~78% from May 2025 peak, DAU growth halved, full-year 2026 bookings guidance ~10.5%. |
| AI commoditization | 2025-2026 | Free frontier LLMs and real-time translation weaken the motive to learn languages, the core analyst concern. |
| AI-content gap | 2025 | 148 AI-generated courses with documented quality errors, 100-plus curriculum specialists laid off, no published benchmark of AI-generated vs human course learning outcomes. |
| DET cheating pressure | EMNLP 2024 | LLM-assisted cheating arms race, their contrastive detection paper reports 1.7x over classifiers at 0.1% false positive, institutional acceptance still growing. |
| Efficacy gap | Mar 2026 | Best external RCT (Kim et al., Studies in Second Language Acquisition, n=183) shows comparable-to-classroom not better, no published calibrated longitudinal ability estimate from their own data. |

### Adversarial verification verdict table, 2026-06-11

For the full verification narrative, see plan Section 5.

| Verdict | Finding |
|---|---|
| Survived 3-0 | AutoIRT and BanditCAT are dichotomous-only, exhaustive full-text search, zero polytomous terms |
| Survived 3-0 | Deployment was DET practice test only, two vocabulary item types |
| Survived 3-0 | S2A3 (June 2026, arXiv 2606.07364) names GRM or partial-credit extension as required future work, gap is author-acknowledged and unpublished |
| Survived 3-0 | C-test blanks and Interactive Reading parts are decomposed into separate binary pseudo-items, no partial-credit IRT |
| Refuted 0-3 | "ML-graded responses bypass IRT entirely", false for speaking and listening, true only for writing |
| Corrected | Operational layer already ingests continuous and explicit polytomous grades under an UNDOCUMENTED model class, so claim "published line is binary-only", never "Duolingo is binary-only" |
| Corrected | BanditCAT venue is PMLR v264 workshop, not ICML main. Read Aloud removed July 2025. The "discrete and continuous grades" goal sentence was never located verbatim, do not quote it |
