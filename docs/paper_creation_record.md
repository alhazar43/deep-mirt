# Paper creation process record

## ROUND 6 (2026-07-05): NRM full-citizen rule + benchmark-all-encoders

User rules: (a) every figure/table carries NRM fully or is explicitly scoped, NO teasers
("either tell the full story or remove"); (b) benchmark all encoders wherever both classes
exist instead of LSTM-hiding. EXECUTED: NEW COMPUTE - NRM oracle decomposition extended to
dkvmn+transformer (8 cells x 25 folds, _p2_nrm_oracle_ext.py, 1305s) + missing
dkvmn/transformer 2PL refits run (_p2_oracle_ladder_matrix.py, 661s) -> full 3x3 ladder
matrix (ordered_decoder_matrix.json). FINDINGS INTEGRATED: amortization gap orders
encoders on both ordered decoders (transformer .48-.53 > lstm .22-.32 > dkvmn .075-.11,
all CIs exclude 0); "separation = trained refit" is ENCODER-DEPENDENT (transformer sep
trails its refit at reference, .806 vs .901 2PL; closes at N5000); NRM mitigation ordering
differs by encoder (dkvmn/transformer: separation dependable, dkvmn gap vanishes at N5000
+.02 n.s.; lstm: refit dependable); MML+RF(theta*) proven encoder-shared (bit-identical).
EXHIBITS: tab:ladder -> 9-row full matrix w/ Delta_amort CIs; fig_ladder -> 3 decoder
panels x 3 encoders overlaid (dodged, fold-backs visible); fig_nrm_oracle -> 3 encoder
panels; fig_cat -> 2 decoder panels, all cross-tested encoders (dkvmn-gpcm = the one
longer-test pair, trades length for 2/3 stop-error cut); fig_slack -> NRM exploratory
diamonds REMOVED (out-of-pool teaser); scope sentences: CAT ordered-only (no option-level
instrument + 4.6 bar), linking ordered-only (no single NRM scale), LSTM = walk-through
anchor (benchmark-first sentence). Encoder markers hoisted to _paperfig_style.py. 29pp.

## ROUND 5 (2026-07-04 night): title + floats + the gradient-flow theory ground

TITLE FROZEN (user): "On the Prediction-Recovery Trade-off in Interpretable Knowledge
Tracing". Floats: [t]->[htbp] + relaxed float fractions ended the post-references dump
(refs now p.24ish, all floats in body). Table 3 width fixed (footnotesize, p-column);
STAYS in Results on pushback (its center column is measured output, not literature).
GRADIENT-FLOW BLOCK integrated into 4.2 (sources: docs/slides/workshop.tex frames 82-96,
learning_dynamics_theory_support.md P1-P9b, theory_memo.md; ml-math-researcher authored
and honesty-checked): Eq flow (exact ODEs, separate key vs shared embedding with encoder
term u_q), Eq flowH (block Gauss-Newton curvature, flat alpha-direction vs steep
theta/beta, kappa throttling, 1-1/kappa contraction), three interpretation-labeled
readings (under-travel = Delta_amort; attenuation as slow-mode under-convergence,
size NOT read off kappa; opposite optimal stopping times). REFUSED per the honesty
checklist: kappa as quantitative predictor, population-limit laws, the convergence-tie,
"lowest-Fisher parameter" unconditional. No learning-dynamics contribution claim. 28pp
clean.

## ROUND 4 (2026-07-04 evening, user's eight-point critique)

1+1b. TRAINED the two never-run combos: {dkvmn, transformer} x NRM x separate heads, 4
budgets x 25 fits = 200 folds, 35 min GPU (new _p2_toggle_nrm_fill.py, smoke-gated,
skip-done). RESULT: recovery AND accuracy wins everywhere (acc deltas +0.002..+0.053, all
8 clustered CIs exclude zero; a_k 0.988/0.937 at N5000). Mass tables now 9x2 coverage,
ZERO dashes; dominance accounting = 16 certified + 20 audited (16/20 significantly more
accurate, 4 dkvmn-2pl indistinguishable), recovery + in all 36. 2. testbed purged
(Synthetic benchmarks). 3. ritual table -> metric language (Metric / Shared-head value /
Detects the error / Reference). 4. "oracle" KEPT (pushback: standard ML). 5. captions
tersened. 6. appendix figures promoted to main text (Appendix A deleted); TWO new figures:
fig_attenuation (log-log slope 0.51 vs 0.89, representative fold d1f4) and fig_nrm_oracle
(4-estimator NRM decomposition across budgets). 7. release vocabulary -> code release /
evaluation suite (measurement "artifact" kept, correct usage). 8. full figure scan:
frontier extended to 36 pairs (panel c all three encoders); fig_slack + 100 NRM
exploratory points (open diamonds); fig_cat stays GPCM (no option-level CAT instrument;
NRM barred from deployment); ladder stays ordered (NRM ordering differs, own chart).
HEREDOC LESSON: never pass LaTeX through bash heredocs (JSON layer eats backslashes ->
control chars); Write-tool script files only. 29pp compile clean. OPEN: title.

## ROUND 3 (2026-07-04, user's five-point critique)

1. TITLE: open; user directed "interpretable KT + prediction/item-parameter trade-off";
   candidates offered, discussion pending. 2. NRM: findability fixed (consistent acronym);
   ORACLE DECOMPOSITION RUN (new deep_irt/bench/_p2_nrm_oracle.py, per-item Bock-NRM MLE,
   100 folds, 102s CPU, outputs/p2_nrm_oracle/): MIXED regime (amortization gap +0.15..0.38
   AND theta-noise +0.11..0.48 both exclude zero at every N; refit-on-truth 0.94-0.99;
   separation certified only at N=500; refit = the dependable NRM mitigation); exploratory
   NRM delta r=0.97 kept OUTSIDE the validated pool. 3. GRADIENT VIEW: verified score
   equations + Fisher asymmetry (I_aa=(theta-b)^2 pq vanishing at theta=b) + the
   direction-vs-magnitude statement (asymmetry sets WHICH coordinate, amortization WHETHER)
   written into 4.2, ml-math-researcher signed. 4. Explicit decoder blocks (normalized GPCM
   + display NRM, hard coordinate named per architecture). 5. MASS TABLES: main (largest
   budget, 9 combos x 2 classes) + appendix (both budgets), from
   outputs/p2_mass_table/ (sanity cross-check exact); coverage 9/9 shared, 7/9 separate,
   dashes honest. FRONTIER v2: three panels by decoder, y = hard-parameter recovery, 28
   paired comparisons incl. DKVMN (beta) and NRM sweeps; pair audit: separate heads
   significantly more accurate in 8/12 new pairs, indistinguishable in dkvmn-2pl 4.
   28pp compile clean.

## REWORK 2 (2026-07-04, user directive: ML-native everything, up to 70% rewrite)

Thesis reframed on psychometric-consult APPROVE verdict: prediction and item-parameter
recovery do NOT compete; the shared-head default is Pareto-dominated (dominance, not
balance; regularization path explicitly contraindicated). Two consult-mandated additions
from banked data: the prediction-recovery frontier figure and a linking magnitude
diagnostic (shared-head log-alpha slope 0.28-0.62, all CIs excluding 1, vs separate-head
0.85-0.89; locations unbiased both classes) -- outputs/p2_magnitude/. Vocabulary contract
(docs/rework2_contract.md) applied at ~130 sites: model classes (shared-head vs
separate-head), oracle decomposition (+ three displayed gap equations Delta_amort/theta/
info), refit discrepancy delta (slack retired), mitigations, benchmarks, study; NRM got a
full results subsection (grid ordering 0.93/0.54/0.69; coupling structure; real reversal
with the corrected per-coordinate statement) + instrument-scope honesty. Figure system
rebuilt on _paperfig_style.py (usetex Computer Modern, unified palette, math labels):
frontier (new thesis figure), rank-scatter redesign, cost-plane redesign, discrepancy
self-contained redesign, restyles for the rest; lead visual gate on all. Gates: ML-native
cold read (blockers fixed: sixteen-scoping, abstract hedging to direction claims,
in-sample tags, channel/coinage purge) + delta integrity 6/6 PASS (equations verified
against implementation earlier; dominance wording, magnitude, NRM, frontier, register all
verified; 0.16->0.15 rounding aligned). Compile 25pp clean. OPEN: title (user pick; three
dominance-framed candidates offered), GenAI disclosure, funding, APC.


Manuscript: "Detecting Item-Parameter Error in Knowledge-Tracing Models Without Ground
Truth" (overleaf-sync/main_caeai.tex, 21 pp compiled). Author: Wenrui Yuan. Target: CAEAI
(Elsevier), IEEE TLT fallback. Pipeline run 2026-07-03/04, lead-authored prose throughout
(all section text written by the lead in the main loop; agents did scaffolding, figures,
verification, and evidence re-analysis only). Plan of record: docs/paper_plan_v2.md
(frozen). Evidence: docs/exposure_rerun_results.md phases 1-11 + Stage-4 addenda.

## Stage chronology and verdicts

- **Stage 2 WRITE.** Full draft against the frozen plan; abstract + 11 sections + back
  matter; five publication figures built under the dataviz discipline (vision-reviewed,
  then lead visual sign-off), venue-exact specs verified against Elsevier/CAEAI (90/140/
  190 mm, Arial embedded, vector PDF).
- **Stage 2.5 INTEGRITY (pre-review).** PASS; 4 precision nits fixed, including the
  shared-w96 mislabel corrected in the evidence record.
- **Style intervention (user directive).** Three published CAEAI papers read directly
  (OpenAlex repository mirrors after ScienceDirect bot-walls; Sci-Hub declined: papers
  are CC-licensed OA and Sci-Hub lacks post-2021 coverage). Binding contract
  docs/caeai_style_contract.md. Manuscript restructured to the venue's numbered IMRaD
  spine; all captions rebuilt to venue anatomy; campaign jargon swept.
- **Stage 3 REVIEW.** Five reviewers (psychometrics, ML/KT, venue fit, devil's advocate,
  cold reader). Verdicts: major / major / minor / no-reject / 20 friction items.
- **Stage 4 REVISE.** All 8 blocking items resolved; three answered with NEW evidence:
  (1) slack robustness (theta-quality bins, fail-safe degradation, real-EdNet silver
  validation, Spearman 1.000 with the classical verdict); (2) CAT at the largest budget
  (costs do not resolve with 2.5x data); (3) statistical hardening (phantom TOST purged
  and replaced by the stronger true claim; wild-cluster floor disclosed; LOCO threshold
  validation; mirt EM-cap disclosure). Point-by-point:
  docs/stage3_response_to_reviewers.md.
- **Stage 3' RE-REVIEW.** ACCEPT with 5 minor tweaks (2 prose colons, caption wording,
  repeats-native tag, orphan float refs); all applied.
- **Stage 4.5 FINAL INTEGRITY (from scratch).** FAIL round 1 (3 cosmetic traceability
  items: reliability range endpoints, 0.942 minimum, exhibit-fold labeling) -> fixed ->
  **PASS, zero issues**. 39/39 citations resolve; 100% of numbers trace; all 7 AI-failure
  modes CLEAR; register rules hold (no dashes, no prose colons).
- **Stage 5 FINALIZE.** overleaf-sync/submission/: manuscript.pdf (21 pp), main_caeai.tex,
  boost_refs.bib, Figure_1..Figure_5.pdf (venue naming; manifest maps to semantic names),
  figure_manifest.txt. Artifact hygiene: stale 1-minus-Jaccard summary strings in the
  flip JSONs regenerated with corrected wording (raw fields untouched).

## Material corrections made during the pipeline (all verified at source)

1. Jaccard/per-list conflation: originated in the flip script's own summary strings;
   all set-agreement headlines re-derived (61% wrong -> 44% wrong vs 18% repaired; 67%
   real disagreement -> 50%; stability 0.80 J -> 89% per-list). Figure regenerated.
2. Phantom equivalence test: no TOST artifact ever existed; replaced by the clustered
   +/-1pp criterion and the stronger fact that the decoupled arm never trails on accuracy.
3. Mechanism paragraph: global-gauge account (rank-preserving, wrong) replaced by the
   item-level amortization-gap account with linear-map mixing.
4. mirt stop-time excess is small but real (0.010): "indistinguishable from oracle"
   corrected before it entered review.

## Open items (author's, not pipeline's)

- GenAI-in-writing disclosure text (Elsevier requires a statement; placeholder in back
  matter). - Funding statement. - Twente library confirmation that the NL-Elsevier deal
  covers CAEAI's APC. - At submission: swap documentclass to elsarticle, set artifact URL,
  consider trimming the two longest captions if the venue asks.

## Where things live

Submission package: overleaf-sync/submission/. Reviews + responses:
docs/stage3_response_to_reviewers.md. Style contract: docs/caeai_style_contract.md.
Evidence: docs/exposure_rerun_results.md (+ outputs/p2_stat_hardening, p2_slack_robustness,
p2_cat_n5000, p2_cluster/cat_clustered.json). Figure sources: deep_irt/bench/_paperfig_*.py.
Nothing committed to git (repo rule: commit only on the author's word).
