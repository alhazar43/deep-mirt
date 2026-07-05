# Paper plan v2.1 (the plan of record; merchandised architecture)

De novo paper; supersedes paper_plan.md and paper_boost_plan.md section 9. paper_boost_plan.md
remains the campaign log (rulings, rigor ledger). Evidence: docs/exposure_rerun_results.md
phases 1-9, all seed-clustered, twice audited. v2.1 change (user directive): same claims,
same honesty, new SELLING architecture -- the paper is packaged as a named disease, a home
test, bounded repairs, and an invoice; the reader is invited to DO something, not to
acknowledge an audit. Persuasion in the paper, candor in the claims, fabrication in neither.

## Title -- FROZEN (user pick 2026-07-03): "Detecting Item-Parameter Error in
Knowledge-Tracing Models Without Ground Truth" (slate candidate 2, instrument-forward: the
abstract's promise leads with the truth-free detector; sentence one still states the disease
scoped to the shared arm; invoice + mechanism carry the body). Superseded slate below.

## Title slate (record)
1. "Stable and Wrong: Auditing the Measurement Claims of Prediction-Trained Knowledge
   Tracing" (editor rank 1; "wrong" scoped to the shared arm in sentence one)
2. "Stable but Unfaithful: Auditing the IRT Readouts of Prediction-Trained Knowledge
   Tracing" (editor alternative; bridges the faithfulness literature)
"Not All Parameters Learn Alike" fully retired.

## The one sentence (editor fixes 1/3/6 applied)
The IRT parameters of prediction-trained knowledge tracing pass every TRUTH-FREE check while
being badly wrong, because the shared readout channel discards information the model already
has (per-item refitting on the model's own ability estimates nearly reaches the classical
ceiling); a truth-free slack test detects it, a per-item refit repairs the rankings people
interpret, an own-channel rebuild halves the decision cost, and the unrepaired channel
doubles adaptive-test length while corrupting cut scores in simulation.

## The story (what sells, who hurts)
- THE FEAR (editor fixes 1/2/4/6 applied): every TRUTH-FREE validation ritual PASSES on the
  shared readout -- accuracy (TOST-equivalent), rerun stability (0.80), real split-half
  reliability (0.75+) -- while 61% of its top-flagged items are wrong against truth and
  accuracy-tied models disagree on two thirds of their flags. The RELIABILITY rituals cannot
  detect the failure BECAUSE the wrong numbers are stable (the slack test can, truth-free --
  that is the product). Any published table that reads parameters off a SHARED readout is
  implicated. Ritual-table columns must each be CITED as a practice the lineage actually
  reports; cut "distribution plausibility" if no citation supports it. (All banked.)
- THE GIFTS: (i) the SLACK TEST, run it on your own model this afternoon, no ground truth
  (validated: r=0.986 vs wrongness, AUC 0.987, 1.2% false alarms); (ii) the REFIT, a
  no-retraining rank-repair for models already shipped (0.72 -> 0.93), honestly bounded to
  interpretation uses; (iii) the REBUILD (decoupling), the training-time fix that HALVES
  THE DECISION COST (157% vs oracle 100%, residual +0.6-0.8pp; never "repairs decisions").
- THE RECEIPTS: untreated, an adaptive test pays 196.8% length and +2.3pp misclassification;
  the two-channel attribution (a-error inflates the test, b-error makes it falsely
  confident); each repair carries its own price row.
- Reviewers get a REVIEWER CHECKLIST (demand the slack test); practitioners get a decision
  guide; theorists get the ladder. Everyone leaves with something -- that is the packaging.
- EDITOR FIX 5 (DONE, outputs/p2_cluster/cat_clustered.json): prose quotes SEED-CLUSTERED
  CAT intervals -- shared inflation [180.4,210.8], decoupled [144.1,168.0], b-only
  RMSE-at-stop [0.069,0.117], retrofit [111.8,199.8], all Phase-10 variants clear of the
  null. ONE NEW BAN: decoupled is NOT significantly better than oracle at its own stop
  (clustered CI straddles zero); never state it as a win over oracle. EDITOR FIX 6:
  one framing sentence precedes the retraction list (the validity gate working, not a
  program flailing); "in simulation" attaches to every cut-score claim.

## Title and venue (RULED, final editor gate wf_e0540be1)
VENUE: CAEAI FIRST (verified 6-day desk screen + 64-day decisions; five same-species
precedents, anchor Schmucker & Moore 2026; material education stake banked; APC likely
covered by the Twente NL-Elsevier deal -- USER CONFIRMS WITH LIBRARY). Accepted risk:
a desk editor reading it as AI-methods-primary; mitigated by leading the abstract and cover
letter with the deployed-decision harm and citing the in-scope anchor. TLT = immediate
fallback. TITLE (ranked six workshopped to CAEAI conventions; USER PICKS, editor's pick
first):
1. Wrong Item Parameters, Longer Tests: The Deployment Cost of Shared Readouts in
   Knowledge Tracing  [editor's pick: disease + invoice in seven words; subtitle scopes
   "wrong" to shared readouts per the binding rule]
2. Detecting Item-Parameter Error in Knowledge-Tracing Models Without Ground Truth
   [instrument-forward, most actionable; undersells mechanism + invoice]
3. The Shared-Readout Trap: Reproducible Item Parameters That Contradict the Truth in
   Knowledge Tracing
4. Can Prediction-Trained Knowledge Tracing Recover Item Parameters You Can Trust?
5. A Truth-Free Test for Item-Parameter Error in Knowledge Tracing, and How to Repair It
6. Why the Shared Readout in Knowledge Tracing Mis-Recovers Item Parameters, and What
   Fixes It

## Architecture (~13pp journal; arc = Disease, Cause, Invoice, Instruments, Choice)

S1 INTRODUCTION (~1.5pp). The promise; the disease in one paragraph (rituals pass, numbers
   wrong, scoped to the shared arm); who is hurt (published interpretations; deployed
   selection); the three gifts named; contributions EXACTLY 3: (C1) the located mechanism
   (estimator ladder + channel decomposition, three encoders, two geometries); (C2) the
   decision-cost receipts (flip + CAT with two-channel attribution); (C3) the instruments:
   the validated truth-free slack test + the exposure-controlled benchmark + protocol,
   shipped runnable. Honesty posture up front (classical near-parity; boundaries).

S2 BACKGROUND (~1.5pp). Deep-IRT lineage = the audited design; pyKT reform (prediction; we
   do measurement); neural-IRT recovery line; cite-as-known bridges (control tasks /
   faithfulness; amortization gap); classical calibration floors. Refs verified
   (docs/boost_refs.bib).

S3 INSTRUMENTS AND BEDS (~1.5pp). Tracer family; spiraled exact-exposure bed; 2D budget
   grid; rank metrics + gauge rationale; seed-clustered statistics (detail to appendix);
   classical reference (mirt, first-attempt matching; one sentence: mirt is exactly
   flat-in-N at fixed exposure, a per-item estimator); artifact statement. T1.

S4 THE DISEASE: stable and wrong (~1.5pp). THE RITUAL TABLE (new exhibit T2r, all numbers
   banked): checks a practitioner can run -- accuracy TOST, rerun stability, split-half
   reliability, distribution plausibility -- each PASS, vs truth FAIL (61% wrong flags).
   The flip, real (67% disagreement at tied accuracy) + synthetic; ENCODER-GENERAL and
   scaling with pooling: transformer 68% disagreement / 71% wrong flags at N=5000, 83%
   disagreement at N=2000 (Phase 10). "Wrong" scoped to the shared arm. F1.

S5 THE CAUSE: the readout channel (~2pp, the center). The estimator ladder T2 (sole
   keystone): shared 0.719 -> refit on OWN theta_hat 0.934 ~= decoupled 0.941 -> clamp
   0.979 ~= mirt 0.982; decomposition channel >> theta-noise >> ~0 (clustered CIs); both
   decoders (2pl channel +0.32/+0.36); generality with PREDICTED victims (dkvmn slots ->
   difficulty lags; transformer extreme pooling -> largest gap; second geometry); not
   capacity (key-16 beats shared-w96; width plateau, evidence Phase 8b).

S6 THE INVOICE: what the unrepaired channel costs (~1.5pp). CAT harm T4/F3: 196.8% length
   [190,204], worse theta at its own stop (+0.036), +2.3pp misclassification; two-channel
   attribution (a-error -> 85% of length inflation; b-error -> falsely confident stops,
   +0.096, the worst arm at its own stop). Simulation scope stated. ARCHITECTURAL
   CROSS-TEST (landed, Phase 10): the two-channel mechanism generalized across all three
   encoders; the transformer pre-registration HELD in full (343.9% inflation,
   length-dominated, a-only ablation 321.2%); dkvmn K=4 held; the dkvmn K=2
   pre-registration FAILED (length-dominated, not decision-dominated) and is REPORTED next
   to the held ones with the labeled post-hoc hypothesis (the falsely-confident signature
   requires accurate-a-with-biased-b; dkvmn has both degraded). Pre-registration with a
   reported miss is a credibility asset; never bury it.

S7 THE INSTRUMENTS AND REPAIRS, each priced (~2.5pp).
   7.1 THE SLACK TEST (the paper's product): slack = 1 - Spearman(readout, per-item refit on
       own theta_hat); truth-free; VALIDATED r=0.986 (n=275 fold-points, 11 cells), flag at
       tau~0.15 (sens 0.92, spec 0.99, AUC 0.987), 1.2% false alarms, catches bad decoupled
       folds too; caveat: inherits theta_hat quality. One figure (slack vs wrongness). T5.
   7.2 THE REFIT (no retraining), bounded: repairs RANK 0.72 -> 0.93 (what interpretation
       uses: flags, rankings) and ~52% of length inflation, but ~0% of the accuracy gap;
       WHY: the refit bank lives in the theta gauge (per-fold scale ratio r = -0.98 with
       harm) -- the scale gauge as the failure mode. Sell WITH the boundary: patch for
       interpretation, not for calibration. Retrofit CAT row.
   7.3 THE REBUILD (decoupling): 32/32 cells, 5/5 seeds; restores per-item-estimator
       behavior; halves the CAT invoice; NOW FULL-LADDER ON ALL THREE ENCODERS (Phase 10
       completion): dkvmn's pooled DIFFICULTY is N-flat under sharing (2pl b stuck
       0.57-0.67 at every N) and repaired by the own channel at every N -- the
       data-cannot-cure claim holds for BOTH victim types; and the repair holds on REAL
       EdNet for the shipped dkvmn architecture (a-reliability +0.044/+0.063, b unchanged).
       Careful real-data framing: the synthetic b-deficit is INVISIBLE to real split-half
       reliability by construction (consistency, not correctness) -- the ritual-table point
       recurring; the slack test is the truth-free detector. Selection-alignment
       (single-cell, illustrative); boundaries honest (thin exposure unresolved below
       E~30-60; NRM/option-level real data excluded, crater 0.065; dynamic heads refuted;
       stop early, both arms). T3.
   7.4 THE CLASSICAL REFIT when the matrix exists: near-parity, parameter- and
       dataset-specific edges (first-attempt-matched), no calibration catastrophe.

S8 THE CHOICE (~1pp, deploy-first per editor ruling). The readout is the only estimator
   alive DURING the sequential test, where the harm is paid and no refit exists; repeats
   point (~17%); then the ceiling and near-parity; the decision guide box: matrix -> refit
   classically; sequential -> rebuild + slack-test + reliability screen; shipped model ->
   slack test, then refit for interpretation only.

S9 WHAT THE PROTOCOL CAUGHT (~0.75pp). Four case studies (padded-theta, mirt-b Pearson
   crater, threshold-ordering export artifact, cold-item false positive), each one tight
   paragraph, shipped as runnable reproductions. Credibility, not score-keeping.

S10 LIMITATIONS (~1pp). Coerced-ordinal real data (thresholds mostly disordered; K>2 real
   results are robustness checks); K/L/bank-size scope; CAT is simulation over ~200-item
   banks; per-eigenmode inversion; well-specified synthetic core; ASSISTments named; the
   general faithfulness point is known, the located mechanism + classical control +
   validated truth-free instrument + decision price are what is new; not learning dynamics.
   RETRACTIONS LIST here, once, plainly (cold-start leg, 5.6x, E=15 reversal, dynamic head,
   decoupling-delays-degradation, linking penalty).

S11 CONCLUSION (~0.5pp). The decision guide restated in three sentences; the invitation
   (run the slack test on your model).

## Exhibits (7 main + appendix)
T1 beds/grid; T2r the ritual table (pass/pass/pass/pass/FAIL); F1 flip; T2 ladder; T4/F3 CAT
invoice (+cross-test); T5 slack validation figure; T3 repairs+boundary. APPENDIX: two-law
(qualitative), full surface, NRM 10-config, stats detail, timing, pitfalls reproductions.

## Binding writing rules (carried; editor-ruled)
Register main_magpcm_ijaied + writing-style memory; no em/en-dashes, no colons in prose.
MML near-parity plain in abstract+intro; never "protocol is the deliverable"; near-parity
never a win; "wrong" scoped to the shared arm; NRM outside the recommendation; "governed by
total data" never unqualified; E* never a constant; refit NEVER sold as decision-grade
(rank-repair only, gauge-bound); slack-test caveat (inherits theta quality) always attached;
abstract ~180 words, no four-clause chains; proposition (CRLB dichotomy) only for the
UMUAI/TMLR variant after ml-math certifies. Author = user; no agent attribution.

## Figure specifications (CAEAI/Elsevier, verified 2026-07-03)
Vector PDF accepted natively (or EPS), FONTS EMBEDDED (fonttype 42). Widths EXACT: single
column 90 mm (3.543 in), 1.5 column 140 mm (5.512 in), double 190 mm (7.480 in). Fonts
Arial/Helvetica or Times, >= 7 pt (6 pt absolute floor for sub/superscripts). Line weights
>= 0.25 pt (0.1 pt absolute). RGB fine (online-only, no color charges). Submission: separate
figure files, named Figure_1.pdf etc., one flat folder, no duplicate basenames. PGF not
mentioned anywhere; PDF is the format. Personal visual sign-off by the lead on every figure
BEFORE it enters the manuscript (user directive); agent vision review is a pre-filter only.

## Venue (verified facts, wf_0b40d3a8, 2026-07-03)
- CAEAI (Elsevier, gold OA): scope explicitly covers AI-algorithm research in education, no
  classroom study required; FIVE 2022-2026 precedents of exactly our species, incl.
  Schmucker & Moore 2026 (IRT item-flaw effects on difficulty/discrimination, existing
  items, no classroom) and psychometric audits of LLM scoring. VERIFIED FAST: ~6-day desk
  screen, ~64 days to post-review decision, ~25 weeks submission-to-publication. APC $2,940
  BUT the corresponding author is at Twente -- the Netherlands-Elsevier national OA
  agreement likely covers it (USER: verify with the Twente library before submission).
  Desk-reject risk profile (informal): AI-novelty-primary papers; ours is
  education-contribution-primary. One aggregator IF figure (23.4) unreliable, ignore.
- IEEE TLT (hybrid): fit moderate-to-strong; our exact measurement-audit framing is WHITE
  SPACE there; strong sibling precedent ("Stable Knowledge Tracing Using Causal Inference",
  TLT 2024, audit+repair, no classroom). BUT decision latency is publicly UNVERIFIABLE
  (treat as unknown risk on a PhD clock); 12-page IEEE format (we are ~13pp, trim or
  overlength charges); OA optional ($2,800) or free non-OA.
- UMUAI: prestige alternative, slow (details in the scout output file); only if time were free.
- LEAD RECOMMENDATION (updated on the verified facts): CAEAI PRIMARY (verified speed +
  precedent density + probable covered APC + met editor condition of material harm), TLT
  immediate fallback (brand + white space, unknown queue). TMLR fallback identity unchanged.
  Final editor pass adjudicates with the title workshop against the chosen venue.

## Status
Merchandised architecture drafted (v2.1) after the slack/retrofit validation (Phase 9).
PENDING: encoder-coverage closure (wf_b61f400d) results fold into S5/S6/S7.3; then the
FINAL editor pass with the changed question ("inviting without overclaim?"); then prose.
