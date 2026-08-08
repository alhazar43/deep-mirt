# Lens A2: theory-empirics-linkage (framing review, 2026-07-17)

# Theory-Empirics Linkage Audit: `overleaf-sync/main_caeai.tex`

## 0. Preliminary flag: vocabulary mismatch with the brief

The brief expects to find "the truth-free slack test (delta)," "bounded repairs," "CAT invoice," and a "stable-and-wrong disease" framing (these are `MEMORY.md`'s description of "the measurement-audit paper"). None of these terms, or a slack-test/delta construct of any kind, appear in `main_caeai.tex` (verified by full-text search, zero hits). The live draft's own header names it *"On the Prediction-Recovery Trade-off in Interpretable Knowledge Tracing"* (line 2), and a third, distinct framing exists in `docs/archive/kt_structured_response_heads_revision_plan.md` ("Structured Response Heads for Knowledge Tracing Beyond Binary Correctness"). Three different narrative skins are attested in the repo for what appears to be the same underlying campaign. That churn is itself relevant to the "glued" diagnosis (see §5), but everything below is grounded strictly in what `main_caeai.tex` actually contains.

## 1. Formal-element inventory (26 labeled equations + notation blocks)

Grouped by role, with consumption verdict:

| Group | Lines | Consumer | Verdict |
|---|---|---|---|
| KT task def (`eq:kt_objective`, `eq:state`, `eq:response_head`, `eq:theta_readout`) | 173-208 | Frames every later model; §3.2-4 instantiate it | live |
| 2PL / GPCM / NRM response models (`eq:2pl`, `eq:gpcm`, `eq:nrm`) | 213-250 | Table `tab:mass`, `tab:massfull`, `tab:real_prediction`, all real-data figures | live, heavily used |
| SH / SK head definitions (`eq:shared_head`, `eq:separate_head`) | 255-272 | `fig:arch`; every SH/SK column in every table | live, central spine |
| Ordered-head parameterization (`eq:ordered_head`) | 342-351 | Not directly tested; only indirectly via Table 1 discrimination numbers | thin (see §3.5) |
| NRM projection/orientation (`eq:nrm_projection`, `eq:nrm_head`, orthogonality identities) | 361-416 | `fig:ednet_nrm_shsk` ("keyed option has largest slope," `tab:app_ednet_2pl_shsk` row) | live |
| NRM stop-gradient routing (`eq:nrm_routed`, and its forward-pass equivalence) | 426-450 | `fig:reversal_bridge` panels (a)/(b) directly decompose the two routed terms; `tab:ednet_two_resolution` | live, **strongest theory-to-data chain in the paper** |
| Loss (`eq:pred_dist`, `eq:totalloss`) | 453-474 | Implicit in all training; not separately tested (doesn't need to be) | live (infrastructural) |
| Two-term/one-term gradient decomposition (`eq:item_score`, `eq:twoterm`, `eq:encoder_path`, `eq:oneterm`) | 494-564 | Motivates the SH/SK manipulation; **the specific quantities \(b_q\), \(J_{g_\psi}^\top s_q\) are never measured** — only the downstream recovery numbers are shown and declared "consistent with" | motivates design, not directly instrumented |
| SK stationarity = per-item refit equivalence | 566-578 | This is the theoretical definition of the refit estimator. Its empirical counterpart never appears (§3.1 below) | **theory present; matching evidence absent** |
| Ordered-head scores + Fisher information (`eq:scores`, `eq:scores2pl`, `eq:info2pl`) | 590-667 | One qualitative sentence at line 1004 ("consistent with the path analysis"); no binning by \(|\hat\theta-\beta|\), no operating-point-specific test | **weakly consumed, sharper prediction than what's tested** (§3.2) |
| NRM scores + incidental-parameter argument (`eq:scoresnrm`, `eq:nrm_theta_grad`, `eq:nrm_key_contrast`, Neyman-Scott/Lancaster cite) | 672-773 | No experiment varies item-bank size / \(n_q\) through this lens | **incidental-parameter claim fully orphaned** (§3.3) |
| Routed ability gradient (`eq:nrm_routed_theta`) | 764-773 | `fig:reversal_bridge` panel (c), `tab:ednet_two_resolution` ability row | live |
| "Three empirical implications" summary | 789-801 | Restated qualitatively in §4 prose | live (as narrative signpost) |

## 2. Reverse map: every table/figure and its motivator

| Figure/Table | Motivating formal element | Verdict |
|---|---|---|
| `tab:mass`, `tab:massfull`, `fig:dd`, `fig:scatter` | Eq 9/10 + full §3.2 gradient argument | tightly motivated |
| `tab:real_prediction` (specifically the **direct-predictor** rows/columns) | No equation anywhere defines an unconstrained "direct" head; §3 (`sec:instruments`) formalizes only SH and SK, never the baseline. The comparison is motivated by a stated question (l.813) but by no math | **orphan experiment** |
| `fig:agreement_both` | General "recovery-as-validity" stance (§2.3); not a specific equation | acceptably motivated (real-data recovery proxy), but the theory's specific claim (discrimination = vulnerable group) is not confirmed here — text itself says discrimination looks "broadly similar" under SH/SK on real data (l.1108), an unaddressed tension with the synthetic result |
| `fig:timss_shsk`, `tab:app_timss_item_thresholds` | GPCM def (Eq 7) | motivated |
| `fig:ednet_2pl_shsk` | 2PL def (Eq 5) | motivated |
| `tab:ednet_two_resolution`, `tab:app_ednet_2pl_shsk`, `fig:reversal_bridge`, `fig:ednet_nrm_shsk` | NRM routing math (Eq 13-14, 25-27) | **best-instrumented part of the paper** — theory and figure panels correspond term-for-term |
| Item-wise popular-option baseline (l.1275-1283) | Serves the accuracy claim's proper interpretation, not a formal element per se | fine, ordinary control |
| `tab:decision`, `tab:hyper`, `tab:beds` | Descriptive/hyperparameter scaffolding, not results | not applicable (see §4) |

## 3. Orphan theory (math with no, or negligible, empirical consumption)

**3.1 SK-stationarity-as-refit (l.566-578) and the whole "estimator ladder."** The theory explicitly *defines* the per-item refit as "the same score condition solved by a per-item refit with the learner states held fixed" (l.573-574). This is genuine, well-built theory. But no table or figure ever reports what the refit actually recovers. `tab:beds` row 2 promises "AM, RF\((\hat\theta^-)\), RF\((\theta^*)\), MML... source of recovery loss" (l.912-915) and cross-references `Section~\ref{sec:diagnostics}` twice (l.811, l.851) — **`sec:diagnostics` does not exist anywhere in the document; both `\ref`s will compile as "??".** The abstract states as a finding that the refit "recovers most of the offline maximum-likelihood reference" (l.49) with zero accompanying number anywhere in the manuscript. `tab:hyper` documents refit hyperparameters (Optimizer, Bounds, l.1725-1728) for an experiment whose output never appears. Also note "AM" (l.913) is used once, undefined, never again.

**3.2 Fisher-information / operating-point derivation (Eq `scores2pl`, `info2pl`, l.619-667).** Derives a specific, falsifiable claim: discrimination information is suppressed specifically when \(\hat\theta\approx\beta_q\) (near the item's own operating point), not just "on average." The paper never tests the conditional version. Its only empirical contact is one hedged sentence: *"This pattern is consistent with the path analysis in Section~\ref{sec:gradient}, where the shared encoder path most strongly affects the parameter groups whose response-head signal is weakest"* (l.1004). What's actually shown (Table 1) is the marginal fact (discrimination worse than difficulty), which is compatible with much cruder explanations too. The math promises a sharper claim than the data delivers.

**3.3 Incidental-parameter analogy (l.750-762, citing Neyman-Scott 1948 and Lancaster 2000).** Invoked to argue that \(\mathbf{a}_q^\perp\) noise contaminates ability as the item bank grows relative to per-item exposure \(n_q\). No experiment varies bank size/exposure through this specific lens (the natural test bed — the boundary/exposure runs — exists, see §4.1, but isn't read through this citation's frame anywhere).

**3.4 Two identifiability claims asserted without citation or derivation.** (i) l.858: "the response likelihood leaves location-scale choices and monotone transformations partially unidentified" (justifies using Spearman over Pearson). (ii) l.1013-1015 (`fig:scatter` caption): "the bilinear likelihood identifies the slope–ability product rather than either scale alone" (justifies an ad hoc common-rescale). Both are load-bearing methodological facts, both plausible, neither derived nor cited anywhere in §3.

**3.5 Disattenuation formula.** Named and cited (`\citet{spearman_proof_1904}`, l.1263) but never written out. Feeds exactly one number pair anywhere in the document (".59 SH; .63 SK," l.1252, a table footnote) and is never discussed in the flowing prose of §4.5. Thinnest-instantiated formal element in the paper — not a hard orphan (it has one consumer), but close.

**3.6 Quadratic weighted kappa (l.856).** Defined as a metric to be reported for ordered real data ("For ordered real data we also report quadratic weighted kappa as an ordinal-agreement measure") — never appears in `tab:real_prediction` or anywhere else. A promised, never-delivered measurement, cheap to fix (drop the sentence or add the column).

## 4. Orphan experiments / orphan promises (empirics asserted, never shown)

This is the sharpest material and doesn't cleanly fit either requested bucket — these are neither "unused math" nor "unmotivated data," they are **claimed findings with zero corresponding table, figure, or number anywhere in the manuscript**, despite methodological machinery (hyperparameters, design-table rows) proving the work was specified and, per the project's own artifact map, largely executed.

**4.1 Adaptive-testing / CAT simulation — fully absent.** Traced through the document:
- Abstract: *"In adaptive-testing simulations, the same gap increases test length and cut-score error"* (l.51).
- Intro, contribution 3: *"it shows that the same gap affects downstream adaptive testing by increasing test length and cut-score errors"* (l.93).
- `tab:beds` row: *"Adaptive testing / truth, MML, SH, SK, RF, ablations / simulated examinees / assessment-level limitations"* (l.922-925).
- The methods paragraph describing it is **commented out** (l.872-880).
- Discussion states results as fact: *"For the ordered response models, SH parameters increase test length and cut-score error relative to the generating-parameter reference. SK reduces these costs but does not remove them"* (l.1385-1387), citing a `Section~\ref{sec:downstream}` (l.1382) **that does not exist** — second broken cross-reference.
- Limitations (l.1466, l.1478) and Conclusion (l.1511-1515) restate it as established.
- `tab:hyper` carries a full "Adaptive testing simulation" hyperparameter block — stopping rule, fixed length, 61-point quadrature, 2000 examinees/fold (l.1719-1723).
No table, figure, or in-text digit for test length or cut-score error appears anywhere in §4. `kt-irt/docs/port/caeai_usage_map.json` independently confirms this: `outputs/p2_cat` is annotated *"the CAT figure/table is commented out of CAEAI"*, and `docs/paper_plan_v2.md` shows actual computed numbers ("196.8% length... +2.3pp misclassification") existing in project records outside the paper. The result was computed and then removed from the draft without scrubbing the five places that assert it.

**4.2 Capacity/width control — fully absent, and it backs a specific abstract clause.** Abstract line 50: *"The gap is therefore an item-wise amortization error rather than a data or capacity limit"* — ruling out "capacity limit" requires a shown capacity experiment. `tab:hyper` lists it: *"Narrow-key capacity control & item key 16..."* and *"Widened-embedding control & shared width 16 to 96 (width-sweep benchmark)"* (l.1708-1709). `tab:beds` row 3 promises *"Boundary and capacity runs / selected encoders and head widths"* (l.917-920). Limitations **admits** the gap directly: *"The shared-width capacity analysis currently covers the LSTM–GPCM setting and should be repeated for the other encoders and response heads"* (l.1475-1476) — confirming it exists in some partial form but is not shown anywhere in the body. `caeai_usage_map.json` confirms independently: `outputs/p2_narrowkey` — *"mentioned in tab:hyper but no CAEAI figure/table presents it"*; `outputs/p2_v3_width` — *"the width/pareto figures are NOT in CAEAI."*

**4.3 Item-bank-size (Q) grid — promised, not shown, and independently verified as computed-but-unused.** Design text: *"The main grid crosses learner count \(N\in\{500,1000,2000,5000\}\) with item-bank size \(Q\in\{200,500,1000,2000\}\)"* (l.821); `tab:beds` row 1 repeats *"Q=200 to 2000"* (l.909). Every displayed synthetic table (`tab:mass`, `tab:massfull`) varies only \(N\); no table has a \(Q\) column, and both cell sets are Q=200 only. `tab:massfull`'s own caption overclaims completeness: *"the two tables cover the benchmark with no cell repeated"* (l.1538-1539) — true only for the N-slice, not for the Q dimension the design paragraph defines as part of "the benchmark." I checked the results tree directly: `outputs/p2_exposure/grid_dkvmn_2pl_N1000_Q1000`, `...Q2000`, `...Q500`, `outputs/p2_exposure/ctrl_embdim_lstm_gpcm_N5000_Q2000`, etc. **exist on disk**, and the usage map records the reason for archiving that entire 641M tree as *"NO figure reads it."* So this isn't a promise that was never executed — it was run, is sitting in the results store, and is on a path to deletion.

**4.4 Boundary run at N=50.** Design text promises *"Boundary runs vary \(N\) from 50 to 5000 at \(Q=200\)"* (l.824); the lowest N shown in any table is 500 (`tab:massfull`). `sec:caught` (l.1452-1454) plausibly explains why: *"Some apparent differences at low item exposure did not persist under resampling... those regions are treated as unresolved rather than as evidence for either SH or SK."* Likely a defensible, deliberate suppression for validity reasons, but the design paragraph (l.824) was never walked back to match.

## 5. `tab:beds` as a scorecard for its own promises

`tab:beds` (l.890-933) is the paper's own table of contents for §4. Read against what the rest of the manuscript actually delivers:

1. Synthetic recovery grid — delivered, but only the N-slice at Q=200 (§4.3).
2. Gap decomposition (AM/RF/MML) — **not delivered** (§4.1-parallel; see §3.1).
3. Boundary and capacity runs — half delivered (N-boundary partial, capacity/width absent, §4.2/4.4).
4. Adaptive testing — **not delivered** (§4.1).
5. Real-data checks — delivered, and arguably undersold by its own role label *"predictive cost only"* (l.930), since the delivered content (external-agreement figures, TIMSS/EdNet tracing) goes beyond prediction cost.

Of five self-declared components, one is solid, one is solid-but-narrower-than-claimed, one is half-missing, and one is fully missing. That ratio, read straight off the paper's own roadmap table, is a fairly literal description of the "glued together" complaint.

## 6. Where the thread genuinely holds (constructive counterweight)

The SH/SK synthetic spine is not glued. Eq 9/10 (definitions) to `sec:gradient`'s two-term/one-term decomposition (mechanism, even if not directly instrumented) to `tab:mass`/`fig:dd`/`fig:scatter` (demonstration) to the real-data NRM routing chain (Eq 13-14/25-27 to `fig:reversal_bridge`, whose three panels correspond term-for-term to the theory's correctness/allocation split) is a complete, well-built theory-to-evidence chain and is the strongest material in the draft. If there is an "aha" already on the page, it lives here, not in the recovery-table headline number.

## 7. Cost note for the parent agent

Per standing constraints: the campaign is frozen and new experiments are expensive and out of scope to require casually. But §4.1-4.3's gaps look less like missing experiments and more like **cut-but-not-scrubbed** content — the underlying computations appear to already exist in `outputs/p2_cat`, `outputs/p2_narrowkey`, `outputs/p2_v3_width`, and `outputs/p2_exposure` per the usage map, and CAT summary numbers already exist in `docs/paper_plan_v2.md`. Surfacing (or explicitly, permanently cutting) these is a writing/table-building and provenance-verification cost, not a re-run cost, though the parent agent should treat "these outputs still exist" as unverified-by-me beyond directory listings and the usage map's own annotations, since they were parked rather than audited to the same standard as the frozen main grid. Separately, the two broken `\ref{sec:diagnostics}`/`\ref{sec:downstream}` cross-references (l.811, 851, 1382) are compile-time defects independent of any narrative decision and should be fixed regardless of which way the framing question is resolved.