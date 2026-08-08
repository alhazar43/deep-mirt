# Lens A7: prior-consult (framing review, 2026-07-17)

# Consult on the archived revision plan vs. the current breadth-of-appeal criticism

## Finding 0 (load-bearing): the manuscript is mid-merge between two drafts that never got reconciled

`overleaf-sync/main_caeai.tex` (root, 1762 lines, `\title{TBD}`) is not a single coherently-drafted paper — it is a splice. A second file, `overleaf-sync/submission/main_caeai.tex` (1544 lines, title actually set: *"On the Prediction-Recovery Trade-off in Interpretable Knowledge Tracing"*), contains sections **absent from root**: `sec:diagnostics` ("Oracle Decomposition"), `sec:invoice` ("Assessment-Level Consequences of Parameter Error"), `sec:slack` ("Ground-Truth-Free Detection of Recovery Failure" — the slack test, r=0.986, τ=0.152, 92% sens/99% spec, matching `paper_plan_v2.md` §7.1 almost digit-for-digit), and `sec:refit`/`sec:rebuild` (the two repairs). Its real-data section is one compact subsection that explicitly disclaims tracing: *"with no recovered parameter in sight"* (submission, line 1260).

Root has replaced that compact section with ~330 new lines — `sec:ordinal_kt` (TIMSS) and `sec:ednet_two_resolution` (EdNet) — but in doing so **dropped** `sec:diagnostics`/`sec:invoice`/`sec:slack`/`sec:refit` outright. Evidence this is unfinished surgery, not a decision:
- `\ref{sec:diagnostics}` (root lines 811, 851) and `\ref{sec:downstream}` (root line 1382) are dangling — the targets don't exist in root.
- The CAT paragraph is commented out (root 872–881), yet the abstract (line 51), the contributions list (line 93), and the conclusion (1511–1515) still assert CAT results as delivered.
- `\section{Discussion [FULL-REWORK]}` and `\section{Conclusion [FULL-REWORK]}` (root 1370, 1487) are literal in-file admissions the reconciliation isn't done.

This matters for everything below: some of what looks like "archived-doc advice adopted" is really this graft, and some of what looks like "archived-doc advice adopted" (cutting CAT) directly **contradicts** the still-standing `paper_plan_v2.md` frozen architecture, which treats the CAT invoice as a full section and one of exactly three named contributions (C2). I can't tell from the files whether CAT/slack/refit are coming back or are being deliberately retired — that's a question for the author, not something to infer.

## Part 1 — Does the archived diagnosis anticipate the current criticism?

Archived doc's diagnosis, verbatim (lines 5–12): *"The current manuscript is technically coherent, but its center of gravity is too narrow: parameter recovery; amortized item-parameter gaps; synthetic known-parameter validation; adaptive-testing errors. That makes the paper feel strongly tied to EDM/JEDM or a narrow KT/IRT methodology audience."*

This is an **audience/scope-fit** complaint about a paper that (per the submission/ snapshot) had *no* TIMSS/EdNet tracing at all. The live criticism — "glued together," no storyline, GPCM/NRM and TIMSS/EdNet feel unrelated — is closer to the opposite complaint, aimed at a paper that now has *more* components. Nowhere does the archived doc anticipate "feels glued together." It doesn't diagnose the current disease.

Where the two connect is mechanical, not diagnostic. The archived remedy — add TIMSS ordinal tracing and EdNet nominal tracing (§10–§11) — was written as one bundled move: new case-study content *plus* a rewritten identity/contributions/discussion (§1, §3, §4, §17) explaining why those two datasets belong together. The identity rewrite was rejected (centerpiece-demotion). What's on the page now is the archived doc's real-data body **without** its connective frame. That half-adoption is a plausible, well-evidenced mechanical account of the current symptom, even though the archived document never predicted it.

On the GPCM/NRM half of the author's own diagnosis specifically: `sec:family`/`sec:gradient` unify 2PL/GPCM/NRM under one theory (same SH/SK gradient question, one derivation each) — that part is genuinely coherent. But the real-data section then segregates them one-per-dataset (GPCM only via TIMSS, NRM only via EdNet, no shared real ground) with no sentence explaining that the split is deliberate (the ordinal/nominal boundary the theory predicts) rather than a matter of dataset availability. The theory promises a family; the real-data section delivers two orphaned case studies.

**Fair-summarizer verdict:** the archived doc's *strategic* claim (reframe the whole paper's identity around "KT beyond binary correctness") was rightly rejected against the standing ruling. Its *tactical* claim — real data needs its own, non-recovery evaluation logic (stability/agreement, not ground-truth validation) — was sound and **is** what the current TIMSS/EdNet sections do; `sec:real_beyond_accuracy` (1036–1039) makes almost the identical move the archived doc prescribed. The document lost the fight it's remembered for and won a quieter one.

## Part 2 — Classification of every concrete move

**Already-absorbed:**
- 2PL/GPCM/NRM head family → `sec:family` eqs (213–250)
- SH/SK architecture figure → `fig:arch` (279–337)
- Synthetic recovery table → `tab:mass` (941–970)
- Recovered-vs-true scatter → `fig:scatter` (1006–1017)
- §8 Fig 3 "ΔRecovery vs ΔPrediction" → `fig:dd` (989–997), near-exact match
- §8 Table 3 real-data prediction table → `tab:real_prediction` (1052–1085); direct/SH/SK/MML present across EdNet/KDD/TIMSS, though the spec's NLL/QWK columns aren't shown (partial)
- §10 TIMSS case study (category curves, expected-score curves, threshold table, trajectories) → `sec:ordinal_kt` (1128–1176), `fig:timss_shsk` a–d, `tab:app_timss_item_thresholds` — near 1:1
- §11 EdNet option tracing → `sec:ednet_two_resolution` (1178–1366) — and the executed version (paired binary/nominal reading of one item bank) is a stronger idea than the archived spec, actually a real finding rather than a display
- §8/§9 "stability not invariance" wording discipline — followed throughout (e.g. 1665–1674 "Split/fold stability," never "invariance")
- §12 MML treated strictly as offline reference, never ground truth — followed (1057, 1372–1379)
- §1/§9 general logic that real data needs prediction+agreement+stability rather than recovery — adopted as `sec:real_beyond_accuracy`'s explicit rationale (1036–1045)

**Compatible with the audit centerpiece, simply not done:**
- §6 "compress the theory" — the opposite happened: `sec:gradient` runs ~300 main-text lines (491–802), including a new Neyman-Scott incidental-parameter argument beyond what archived doc anticipated. Doesn't conflict with the centerpiece, just an economy opportunity left on the table.
- §2 "cut 'Corrections to recovery analysis,' reads like internal lab notes" — not done; `sec:caught` survives in root (1432–1459) with the flagged tone intact. Notably, `submission/` had already independently fixed this by retitling the equivalent section "Internal Validity Checks and Failure-Mode Audits" (submission line 1334), matching `paper_plan_v2` §9's own packaging ("credibility, not score-keeping"). Root reverted to the plainer framing.
- §8/§9 dedicated stability table/figure — the numbers exist (scattered in appendix), the unifying exhibit doesn't.

**Centerpiece-conflicting (rejected territory, correctly absent):**
- §1 identity statement centered on "response tracing" as the thesis — abstract stays recovery/amortization-first in both drafts.
- §3 contribution list organized around "KT beyond binary correctness" — root's three contributions (89–93) are still validity-check / amortization-gap / CAT, structurally unchanged.
- §4 top-level Background reorganized around "KT Beyond Binary Correctness" — `sec:background` keeps its recovery/interpretability-checks organization (138).
- §15 "don't lead with 'prediction-recovery trade-off'" — `submission/`'s literal title *is* that exact phrase (line 31); root's title is unset but its abstract still leads with faithfulness-under-prediction.
- §16 title options — none adopted; live slate is `paper_plan_v2`'s disease/audit-forward six.
- §17 discussion reorganized around "what heads add to KT" — `sec:discussion`'s actual lens is "Offline Calibration and Sequential Use," an audit/deployment lens.
- §18 venue-broadening argument — moot; CAEAI-first is ruled independent of this argument.

**Ambiguous, flag don't infer:**
- §2 "cut adaptive testing from main" — *looks* followed (CAT gone from root body), but Finding 0 suggests it's more likely collateral damage from the TIMSS/EdNet graft than a deliberate endorsement of this specific archived-doc advice, and it sits against `paper_plan_v2`'s still-frozen S6/C2. Needs the author's call, not mine.

## Part 3 — 3 salvageable ideas that keep the audit as the spine

**1. One throughline sentence, in the audit's own vocabulary (cheapest, highest value).**
State once, in intro/discussion/conclusion: TIMSS and EdNet re-test the synthetic section's own question — do prediction-tied models agree on what you'd read off them — on two response geometries the theory (`sec:gradient`) predicts should behave oppositely. TIMSS (short, ordered, GPCM): SH/SK agree substantively (`fig:timss_shsk`, trajectory correlation ≈0.84). EdNet (nominal, NRM): SH/SK are prediction-tied but disagree on which external criteria they satisfy (SH wins log-loss, SK wins the person-side anchor and distractor point-biserial, `tab:ednet_two_resolution`). That's the shared-readout phenomenon from the synthetic audit, shown present on one real dataset and absent on another, tied to response geometry — an actual "aha," already fully computed, zero new experiments. Adapts archived §9/§17.1 but recast as an audit claim, not a KT-feature claim. Cost: prose only (abstract, 89–93, 1122–1124, discussion, conclusion).

**2. One consolidated real-data stability exhibit, positioned as the real-data analogue of `tab:mass`.**
Pull the already-computed split/fold-stability rows (`tab:app_ednet_2pl_shsk`, 1665–1674) and the TIMSS cross-fold-SD prose (1169–1170) into one compact main-text table spanning both datasets, captioned as the truth-free substitute for recovery where ground truth doesn't exist. This is archived §8/§9's unabsorbed proposal, which the archived doc itself flagged as recovery-adjacent rather than identity-reframing ("a strong real-data replacement for unavailable ground-truth recovery," line 533) — centerpiece-compatible even before the ruling. Cost: reorganizes existing numbers, no new runs.

**3. Reconcile the two drafts, then re-derive the contributions list from what survives.**
This is a decision, not a rewrite. Either (a) reintegrate `sec:diagnostics`/`sec:invoice`/`sec:slack`/`sec:refit` from `submission/` — updating their numbers, since submission's abstract scopes to "two encoders, two ordered decoders" while root's grid is wider, so the ladder/slack tables need re-deriving, not copy-pasting — or (b) confirm CAT/slack/refit are staying out and scrub every remaining reference (abstract 51, contributions 93, roadmap 95, the two dangling `\ref`s, conclusion 1511–1515). Either way, rewrite the contributions list and roadmap to match what the paper actually contains (including TIMSS/EdNet, currently unpromised), and resolve both `[FULL-REWORK]` tags in the same pass. This operationalizes the archived doc's most durable, venue-agnostic instruction — §2's keep/cut/move discipline, decide-then-follow-through — without touching which result is the centerpiece. Cost: (b) is cheap, a few short deletions/rewrites; (a) is a real cost (re-verifying ladder/slack numbers against the expanded grid) and should be flagged to the author as an extension with a price, not assumed.