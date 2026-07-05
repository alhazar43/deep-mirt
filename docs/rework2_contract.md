# Rework 2 contract (user directive 2026-07-04): ML-native vocabulary + reframe + figure system

Thesis (reframed, pending psychometric design consult): a study of whether a KT sequence
encoder with an IRT decoder balances next-response prediction and item-parameter recovery.
IRT/educational measurement is the flavor domain; every term must be standard ML or
standard psychometrics. NO invented/decorative vocabulary. Up to 70% prose rework licensed;
numbers frozen.

## Vocabulary replacement map (binding; applies to prose, tables, figures, captions)

| retire | replace with | rationale |
|---|---|---|
| arm(s) | model class / shared-head model / separate-head model; "parameter source" in CAT | user #1; probing-literature vocabulary |
| shared/decoupled readout (as arm names) | shared head / separate heads (readout stays for the generic act of reading out) | ML: prediction heads |
| estimator ladder / rung | oracle decomposition / oracle variants | standard: oracle experiments |
| slack (statistic) | refit discrepancy, symbol delta | "slack" collides with slack variables; exact name |
| audit (the paper's act) | study / analysis / evaluation | not a standard genre term |
| repair(s) | mitigation(s); or name the change (head separation, post-hoc re-estimation) | ML: mitigation |
| the rebuild | head separation (the architectural mitigation) | call it what it is |
| per-item refit | post-hoc per-item re-estimation (refit acceptable after definition) | precise |
| testbed(s) | synthetic benchmark / data-generating process (DGP) / simulation setup | user: "wtf is a testbed" |
| flags / top-20 flags | top-20 items by estimated discrimination; "flagged" only for the detector's decision | signal-detection usage only |
| symptom / disease | effect / failure / phenomenon | no medical drama |
| invoice / price | cost(s) | standard |
| stable and wrong (branding) | describe plainly: consistent across reruns yet far from truth (may appear once as a phrase, not as a coined term) | no self-invented brands |
| culprit / innocent / lament / drama declaratives | plain causal statements | cold-read list |
| honesty meta ("we state plainly", "offered as such", "honest about its limits") | state the content once, no meta | cold-read list |
| KEEP | readout, head, amortization gap, oracle, ablation, exposure, item bank, spiraled administration, discrimination/difficulty/step thresholds/ability, data budget, split-half reliability, false-alarm rate / sensitivity / specificity (signal detection), Pareto | standard in ML or psychometrics |

## Figure system (complete redesign, one style module)

_paperfig_style.py: text.usetex=True with Computer Modern (matches body), white background
(no canvas fills), despined, font sizes 8-9pt at final width, widths 90/140/190mm, one
palette across ALL figures: shared-head #E69F00 (orange), separate-head #0072B2 (blue),
oracle/truth black, classical MML gray, Okabe-Ito accents only; mathtext everywhere; CI
whiskers = seed-clustered 95%; compact layouts; every figure must carry its story
self-contained (axis labels define quantities in math; operating points labeled).

- F1 arch schematic (TikZ, in-doc): keep; relabel to shared head / separate heads.
- F2 REDESIGN recovered-vs-true rank scatters, shared vs separate side by side (one
  largest-budget fit, rho_s annotated, top-20 band shaded); replaces the dot-plot flip.
- F3 oracle decomposition chart: restyle (CM, white, no collisions), Delta_ch/Delta_theta/
  Delta_info annotations in math.
- F4 REDESIGN CAT cost plane: x = length inflation %, y = excess stop RMSE, one point per
  parameter source with clustered CI crosses; two-channel structure as geometry.
- F5 REDESIGN discrepancy validation: delta vs true rank error, axis labels in math,
  threshold tau labeled "detection threshold", operating-point annotation (sens/spec,
  in-sample vs held-out), classes in the standard palette.
- F6 nrm: restyle + class vocabulary.
- F7/F8 appendix (scaling, budget grid): restyle.
- F-NEW (pending consult): prediction-recovery plane (accuracy vs recovery per model class
  across cells) as the thesis figure.

## Rework obligations beyond vocabulary
- Reframe abstract/intro/discussion around prediction-recovery balance; the central
  empirical fact: separate heads never cost prediction (all 16 static pairs) while
  recovering parameters; the shared-head default forfeits recovery for nothing.
- Register: cut narrative scaffolding (cold-read list), flat declaratives, hedge only
  interpretation.
- Terminology consistency figures<->text (ablation, not attribution).
- Pending: psychometric-researcher verdict APPROVE (reframe prose only) vs REBUILD
  (minimal experiment additions, run before rewrite).
