# Response to the Stage-3 panel (2026-07-04)

Manuscript: overleaf-sync/main_caeai.tex. Every action verified in-text; evidence jobs in
outputs/p2_stat_hardening/, p2_slack_robustness/, p2_cat_n5000/. Decision was major
revision (R1, R2), minor (R3), no-reject (DA); all blocking items addressed below.

## Blocking items

1. **Jaccard conflation (R1-M1, R2-minor, Cold-12).** Confirmed at source: the flip
   script's own summary printed 1-Jaccard as "% disagreement". ALL set-agreement numbers
   re-derived as per-list fractions from the source JSONs and replaced (abstract, S1, 4.1,
   ritual table, both flip captions): 44% of shared flags wrong vs 18% decoupled; 39%
   between-arm synthetic; 50% between-arm EdNet; transformer 51%/71%, 55% wrong; stability
   89%. fig_flip regenerated to plot per-list overlap (load-time assertions on both raw J
   and conversion); lead visually signed off.
2. **Phantom equivalence test (R2-M4i, R1-M3).** Audit found NO TOST artifact ever
   existed. All equivalence language removed. Replaced by clustered-delta criterion (tied
   = clustered delta within +/-1pp, stated in 3.4) and the stronger true statement: in all
   16 static pairs the decoupled arm's accuracy >= shared (max gap 0.042 under transformer
   at thin budgets; exhibit budgets <= 0.006, tied). 4.1 rewritten accordingly.
3. **Mechanism paragraph wrong (R2-M1, R1-M6).** Rewritten (4.2): concedes global scale
   gauge is rank-preserving and innocent; locates the culprit at item level as amortized
   estimation through a narrow shared map (mixing per roeder_linear_2021); channel term
   named an amortization gap over items (cremer_inference_2018) here, in Background, and
   at the ladder definition. Capacity controls cited as support.
4. **Small-G inference (R1-M2, R2-M4ii).** Wild cluster bootstrap run: all headline
   contrasts at the G=5 floor (p_min=1/16) with 5/5 sign agreement. 3.4 now discloses the
   floor, leads with sign consistency, keeps clustered intervals as primary.
5. **CAT budget scoping (R1-M4, DA-4).** Answered with data, not scoping: new N=5000 CAT
   run (25 folds, clustered). Costs do NOT resolve: length 183.7 [155.6,214.8] (unresolved
   vs 196.8), stop error +0.043, misclassification +2.9pp, while a-recovery improves
   0.719->0.810. New paragraph in 4.3; intro claim now "a cost larger training budgets do
   not resolve". (No mirt fit exists at that cell; reported, not improvised.)
6. **Slack in-sample + circularity (R1-M5, DA-1).** LOCO cross-validation reported in 4.4
   (tau reproduces 10/11; sens 0.92; spec 0.91 held out; per-cell AUC 0.63-0.96, labeled
   in-sample vs held-out). Robustness paragraph added: survives worst observed theta
   tercile/decile (0.96/0.84); deliberate degradation fails safe (inflates slack, never
   masks; 8/8 bad readouts flagged in every condition); REAL EdNet silver validation
   (slack ranks 4 arms identically to mirt-disagreement, Spearman 1.000, same tau).
7. **Online niche asserted (DA-2).** 5.1 now states the boundary plainly: simulations
   consume offline banks; the claim is definitional; online pipeline named as the next
   experiment.
8. **Encoder-generality language (R2-M3).** "Exactly as predicted" removed for DKVMN
   (observed-then-confirmed stated honestly); transformer forecast kept with
   "version-controlled project log" provenance; "law" -> "regularity".

## Non-blocking items applied

R2: hyperparameter table (new Appendix B, values traced to source files; standard key 64,
narrow-key control 16 disambiguated); mirt EM-cap disclosure (3.5); NRM cite
thissen_steinberg_1986; r=0.986 harmonized. R1: attempt-filter tags (Table 5
repeats-native); reliability "low end of accepted range"; refit needs "model + training
responses". R3: embedding gloss; decision-rule table (tab:decision); tau portability
sentence; register pass ("substantially wrong" at bookends, slang neutralized); style
contract reconciled. Cold reader: all 20 friction items (glosses for victim/arm/spiraled/
toggle-cell/scale-mismatch; exposure->liability; budget vocabulary pinned; unit inline;
sentence splits; dkvmn discrimination-vs-difficulty explanation; 55%/27% rephrase;
eigenstructure dropped).

## Declined / deferred (with reasons)

- Misspecified-synthetic condition (DA-3): repo probe exists (run_misspecification_probe
  .py) but needs an array-dump hook for slack; deferred to revision-if-asked; real-data
  flip + silver validation already cover the misspecification direction.
- OSF registration (R2-M4iii): predictions were recorded in the version-controlled log,
  not OSF; text now says exactly that.
- Caption length band (R3-minor): invoice/flip captions slightly above 60 words; kept for
  completeness of the statistics note; trim at proof if the venue insists.
