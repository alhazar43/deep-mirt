# Lens A3: aha-audit (framing review, 2026-07-17)

I have everything I need. I've read the full draft, both governing plans (the frozen "merchandised" plan and the archived "response-tracing" revision consult), verified the section structure, confirmed the dangling references, and mapped every candidate against the actual results inventory. Here is the deliverable.

---

# AHA-AUDIT: main_caeai.tex

## Headline finding: the paper doesn't lack an aha — it deleted the one it was built around, and half-installed a different one without rewriting the frame

The "reads like glue" complaint is not a vibe. It is a mechanical, on-the-page collision of **three incompatible paper conceptions**, with the load-bearing sections of the first still referenced but physically absent:

1. **The frozen plan of record (`paper_plan_v2.md`)** merchandises a "Disease → Cause → Invoice → Instruments → Choice" paper whose designated **product is the truth-free slack test** (frozen title: *"Detecting Item-Parameter Error in Knowledge-Tracing Models Without Ground Truth"*), whose keystone is the **estimator ladder table** (`0.719 → 0.934 → mirt 0.982`), and whose payoff is the **CAT invoice** (196.8% test length).
2. **The archived revision consult (`kt_structured_response_heads_revision_plan.md`)** argues the opposite: *cut* the slack test, *cut* adaptive testing, *cut* the refit ladder to an appendix, and reframe around "structured response heads beyond binary correctness" with **TIMSS ordinal tracing + EdNet nominal tracing** as case studies.
3. **The header comment names a third framing neither plan endorsed**: `"On the Prediction-Recovery Trade-off in Interpretable Knowledge Tracing"` (line 2) — which plan v2 explicitly forbids ("never lead with the trade-off") and the revision plan also rejects ("Do not lead with 'prediction-recovery trade-off'").

**The draft executed the revision plan's grafts (TIMSS + EdNet case studies) and deletions (slack test, CAT tables) but never rewrote the abstract, intro, or discussion, which still promise the plan-of-record paper.** The result is a body that delivers paper B under a frame that promises paper A, titled as paper C. Concretely:

- **Abstract promises the CAT invoice** — "In adaptive-testing simulations, the same gap increases test length and cut-score error" (lines 51–52) — and the **refit decomposition** (line 49). **Contribution 3 promises CAT** (line 93). **The body contains no adaptive-testing results at all.** The design paragraph for it is commented out (lines 872–880); the discussion points to `Section~\ref{sec:downstream}` (line 1382) — **a label that does not exist**.
- **The "estimator ladder" is referenced twice as `Section~\ref{sec:diagnostics}` (lines 811, 851) and again in the discussion (line 1504) — that section also does not exist.** No ladder table is in the paper.
- **The slack test — the plan-of-record's "product" and the noun in its frozen title — appears zero times** (grep for slack/truth-free/ritual/stable-and-wrong returns nothing but one incidental "AUC" on line 107).
- The discussion and conclusion still carry literal `[FULL-REWORK]` tags in their headers (lines 1370, 1487). The draft is visibly mid-surgery.

So the ~310-line gradient-path theory (`sec:gradient`, lines 491–801) was written to establish "the located mechanism" behind a **disease and invoice the draft no longer shows**. The theory now hangs unattached to any payoff the reader sees — which is precisely the reader's complaint: *"gluing math (theory) and random code/datasets (empirical) to prove a point that was never there."* The point WAS there; it was cut. **This is the single most important thing for synthesis: the incoherence is a half-finished pivot, not an absence of material.**

A second structural reason for "glue," matching the author's own diagnosis:
- **TIMSS and EdNet point in opposite directions and share no thesis in the current draft.** TIMSS is a **null**: "SH and SK lead to essentially the same substantive ordinal interpretation" (line 1173), closing with "whether response structures show universal agreement is dataset-specific" (line 1176). EdNet is a **divergence**: items agree but persons and slopes don't. The draft never tells the reader why a stable case and a divergent case belong in the same paper.
- **GPCM and NRM feel glued because the synthetic machinery unifies them (same "where do item parameters come from" question, line 251) but the real-data half splits them** into two disconnected datasets. The decoders are bound by the SK argument; the datasets are bound by nothing the reader can feel.

**Cost reality (reassuring):** every deleted piece was *computed and still exists* — `outputs/p2_slack` + `p2_slack_robustness` (slack test), `outputs/p2_cat` + `p2_cat_retrofit` + `p2_cat_n5000` (invoice), `p2_nrm_repar/{arm1h,arm2,g0,oracle}` + `p2_scaling` (ladder), all archived per `caeai_usage_map.json`. **Nothing requires new GPU runs.** The fix is a spine decision plus a rewrite of the frame-bearing sections, not a new campaign.

---

## Per-candidate audit

### (a) Stable-and-wrong (reliable yet unrelated to truth)
- **Evidence in draft: mostly absent as a *demonstrated* phenomenon.** The plan-of-record's central exhibit — the ritual table (accuracy/rerun/split-half all PASS vs truth FAIL, "61% wrong flags") — is not in the paper. The raw material is half-present: split-half **reliability numbers sit in appendix `tab:app_ednet_2pl_shsk`** (α split/fold stability .755 SH / .847 SK, line 1666; c_k stability .45 SH / .83 SK, line 1673) but are **never connected to a wrongness claim**. The synthetic "stable yet wrong" is implied by SH discrimination compression (`fig:scatter`, lines 999–1004) but never stated as *reliable-and-wrong*.
- **Placement:** dispersed, unlabeled, no exhibit.
- **To make it the spine:** rebuild the ritual table from archived `p2_flip`/reliability artifacts (data exists). **Two blockers under the standing rulings:** (i) "stable and wrong" / "ritual table" are **invented labels** the constitution forbids; (ii) "wrong" is only demonstrable on synthetic (real data has no truth), so the disease is permanently scoped to the shared arm on synthetic data — a measurement-audit posture that pushes toward the AI-methods-primary desk-reject risk the plan itself flagged. **Highest conceptual punch, worst fit to the author's own constraints.**

### (b) Truth-free slack test (audit measurement WITHOUT ground truth)
- **Evidence in draft: entirely absent.** This is the plan-of-record's designated **product** and the noun in its **frozen title**. The validation exists in archived scratch (`p2_slack`: r=0.986 vs wrongness, AUC 0.987 per the plan). The draft deleted it wholesale.
- **Placement:** none.
- **To make it the spine:** new section + validation figure from archived data (medium cost, no new runs). **Highest ceiling of any candidate** — "you can detect the failure on your own shipped model with no answer key" is a genuine *wow*. **But it fights the constitution hardest:** invented label ("slack test"), most methods-primary framing (the plan flagged the CAEAI desk-reject risk), it inherits the θ̂-quality caveat, and **an instrument-led paper still needs a payoff the reader cares about — which was the CAT invoice, also deleted.** Restoring (b) means restoring the whole plan-A machine the draft dismantled.

### (c) Two-resolution EdNet (same learners at binary and nominal; items agree, persons partial)
- **Evidence in draft: fully present and the best-developed novel result.** `sec:ednet_two_resolution` (lines 1178–1367), `tab:ednet_two_resolution` (line 1221), backed by frozen artifacts (`inter_agreement.json`, `theta_cross_reading.json`). The numbers are clean and self-contained:
  - **Item locations agree across resolutions:** β vs keyed-c_k contrast .82/.83; model-implied p-value .95/.97 (lines 1238–1248).
  - **Slopes transfer weakly:** α vs keyed-a_k .21 SH / .46 SK (line 1242).
  - **Persons only partial:** final ability matched learners .18 SH / .33 SK, **disattenuated .59/.63** via fingerprint-matching 1,501/2,000 learners (line 1250, Spearman 1904 disattenuation).
  - **Binary control:** the 2PL item map is design-robust (β .998, α .978, line 1191) while ability is not (.36 SH → .54 SK, line 1206).
- **Placement:** last experimental subsection, buried under heavy hedging ("should not be interpreted as a SK–SH ranking," line 1363).
- **To make it the spine: LOW cost — repositioning and a sharper frame only.** Strongest fit to the standing rulings: **real education data, distractor/option substance (CAEAI-native), no invented vocabulary required.** Limitation: it is an *agreement/consistency* story, not a *truth* story; on its own it doesn't obviously need the synthetic recovery machinery, risking making the synthetic half look like an appendage.

### (d) One-repair (separated key improves recovery without prediction cost, across formats)
- **Evidence in draft: fully present and already the de facto spine.** `tab:mass` (line 941), `fig:dd` ("all eighteen recovery shows positive gap," line 985), `fig:scatter`. Strong: 2PL disc .553→.898 (LSTM), .373→.806 (transformer); GPCM .438→.900 (transformer); NRM .668→.916; accuracy within 1pp throughout (lines 978–979). Real-data repair echoes in appendix (α stability .755→.847).
- **Placement:** it *is* the current spine.
- **Assessment: strongest evidence, weakest aha.** This is a "we found a training trick" paper — a fix to a problem the reader hasn't been made to feel. **It is exactly the current draft, and the current draft is what's being criticized.** It cannot carry the paper alone; it is the *instrument*, not the *payoff*.

### (e) Stronger frame I find: the robustness hierarchy — "item locations port, discriminations are conditionally recoverable, persons are fragile"
This is the **only frame that makes the currently-disconnected pieces cohere, and every leg is already on the page:**
- **Item *location* (difficulty / keyed intercept / threshold ordering): robust to everything.** Synthetic β recovers .72–.85 under SH even while α collapses (`tab:mass`). Real: EdNet β design-robust .998 and cross-resolution .95/.97; **all 31 TIMSS items keep ordered thresholds under both designs** (`tab:app_timss_item_thresholds`).
- **Item *discrimination* / slopes: fragile under the shared readout, repairable by the separated key, transfers weakly across resolution (.21→.46).** This is exactly the tier the paper's entire machinery (theory + SK repair) acts on.
- **Person / ability: fragile everywhere, only partially recoverable (.18–.64), never fully** — the one quantity no design or resolution rescues.

This reframes TIMSS and EdNet from contradictory to complementary: **TIMSS shows the top tier (locations) is so robust that even the design choice doesn't matter; EdNet shows the middle and bottom tiers (slopes, persons) are where design and resolution bite.** Same hierarchy, two stress tests. It also earns the synthetic half (which establishes *which* tier SK repairs) and needs no new experiments. **Caveat to flag honestly:** the person-fragility leg is strongest on real EdNet; synthetic ability recovery isn't even in `tab:mass` (only accuracy/α/β), so that leg is under-reported in the current draft and would need a synthetic ability column surfaced (data exists, cheap).

---

## Ranked verdict

**Rank 1 — the aha that can carry the paper: (c) elevated by the (e) frame.** "Read the same learners and items at two resolutions and under two amortization paths; the difficulty map is one portable object across all four views, but discrimination and ability are not — and a one-line design change (the separated key) is the lever that moves discrimination from unreadable to readable at zero prediction cost." This is the highest-value option **that respects every standing ruling**: it is prediction/KT-home with IRT as flavor, uses only established names, is CAEAI-native (real data, distractor substance), needs **no new runs**, and it converts the paper's three orphaned pieces (synthetic recovery, TIMSS, EdNet) into one instrument with one payoff. It absorbs (d) as its mechanism and (a)'s "prediction accuracy doesn't certify the readout" as its motivation, without importing any banned vocabulary.

**Rank 2 — highest ceiling, worst constraint fit: (b) the truth-free slack test.** If the author is willing to relax the "no invented labels / not methods-primary" posture and rebuild the deleted plan-A machine (slack section + CAT invoice from archived data), this is the biggest genuine *wow*. Present it to the author as the "swing for the fences at TLT-not-CAEAI" option, explicitly noting it is the plan-of-record's own frozen spine that the draft abandoned — the author should consciously *confirm* the abandonment rather than let it stand by default.

**Rank 3 — strongest evidence, insufficient as a headline: (d) the one-repair.** Keep it as the instrument inside Rank 1; do not let it be the whole story, or the criticism recurs verbatim.

**Rank 4 — (a) stable-and-wrong:** powerful hook, but banned vocabulary + synthetic-only "wrong" + audit posture make it a poor headline under the constraints. Fold its motivating sentence into Rank 1's setup; do not build the ritual table.

### The 3-beat story Rank 1 implies (setup → instrument → payoff)

- **Setup.** Learning platforms increasingly log more than right/wrong — the selected option, partial-credit levels — and prediction-trained KT models expose named IRT readouts (difficulty, discrimination, ability) that people read as measurements. Which of those readouts can you actually trust off a model trained only to predict the next response?
- **Instrument.** Read the *same* EdNet learners and items at two resolutions (binary 2PL, nominal NRM) and under two amortization paths (shared vs separated key), with a synthetic known-parameter bed to calibrate what "agree" means and a matched-format TIMSS check for the ordinal case. Compare what survives.
- **Payoff.** A robustness hierarchy: **item difficulty is the same object across resolution and design; discrimination and option slopes are fragile under the shared readout and only conditionally recoverable; person ability is fragile everywhere.** The separated key is the single, prediction-free lever that promotes discrimination from the fragile tier to the portable tier — so what you may safely read off a KT model is a *stratified* answer, and one design choice widens the safe stratum.

---

### Concrete decisions the synthesis owes the author
1. **Pick one spine and rewrite the abstract, intro contributions, and discussion to it.** The current frame-body mismatch is the mechanical cause of "glue."
2. **Resolve the three dangling promissory notes** (`sec:diagnostics`, `sec:downstream`, the abstract's CAT sentence): either fulfill them (restore ladder/CAT from archived data) or delete them. Right now they are broken cross-references that a reviewer will catch immediately.
3. **Give TIMSS and EdNet a shared thesis** (the robustness hierarchy does this) or the "unrelated datasets" complaint stands.
4. **Note for the author, not to require:** restoring the slack test (b) or the CAT invoice is *possible and cheap in compute* (data exists) but *expensive in scope and venue risk* — present as an option with that price, per the frozen-campaign ruling.

Key file references: draft `C:/Users/steph/documents/deep-mirt/overleaf-sync/main_caeai.tex` (broken refs lines 811, 851, 1382; commented CAT design 872–880; two-resolution 1178–1367; `[FULL-REWORK]` tags 1370, 1487); frozen plan `C:/Users/steph/documents/deep-mirt/docs/paper_plan_v2.md` (slack-test-as-product, lines 39–43, 124–131); abandoned-alternative consult `C:/Users/steph/documents/deep-mirt/docs/archive/kt_structured_response_heads_revision_plan.md`; inventory confirming all cut results still exist `C:/Users/steph/documents/deep-mirt/kt-irt/docs/port/caeai_usage_map.json` (archive block, lines 359–401).