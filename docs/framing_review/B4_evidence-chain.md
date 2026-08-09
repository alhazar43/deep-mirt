# Board B4: evidence-chain (results critique, 2026-08-09)

I have enough to deliver a grounded evidence-chain audit. All findings below are anchored to files/lines/artifacts I read.

---

# PART 1 — DEFECTS (ranked)

## FATAL

**F1. The TIMSS "ordered thresholds for every item" result is a `torch.sort` export artifact, not a finding.**
- Evidence: the GPCM head produces *unconstrained* thresholds — `core/decoders.py:419` `b = self.fc_b(emb) # unconstrained (sorted only at eval time)`. The recovery path the exporter uses calls `item_params_sorted`, which applies `params["beta"] = torch.sort(...).values` (`core/decoders.py:430`; also `core/model.py:658` `b_hat = np.sort(b_hat, axis=1)`; recovery entry `_p2_model.py:262`, `core/model.py:541`). So every exported threshold pair is sorted *before* the "ordered?" test.
- The appendix table (`tab:app_timss_item_thresholds`, fed verbatim by `p2_v3_arm1r/timss_thr_rows.tex`) shows "yes" for all 31×2 cells; the figure panel `fig:timss_shsk`(d) computes the "ordered fraction" as `(beta[:,0] < beta[:,1]).mean()` on the sorted betas (`_p2_v3_shsk.py:786-787`) — identically 100% by construction.
- The campaign's own R8 verdict says exactly this and instructs a demotion the paper did not perform: *"the neural fits' 100%-ordered is a torch.sort EXPORT ARTIFACT, not evidence. Polytomous real claims demote hard"* (`docs/exposure_rerun_results.md:456`).
- Collateral: sorting is order-statistic biased for near-tied items, so the reported threshold *locations* and cross-fold SDs (~0.4) are also biased apart for items whose raw steps are close/crossed — the "sufficiently stable for item-level inspection" claim (1169) inherits this.
- Reviewer would write: *"The claim that all items retain ordered thresholds under both designs (§4.4, lines 1162/1168; Table B.2) is tautological: thresholds are sorted at export (`decoders.py:430`). Report the fraction ordered on the raw, unsorted head output, or withdraw the claim. As written it is not evidence of preserved partial-credit structure."*

## MAJOR

**M2. The real-data SH/SK contrast is confounded with encoder item-embedding width; it is not the "same encoder" the methods define.**
- Evidence: `_p2_realstudy.py:167,169` — 2PL/GPCM **shared** uses `emb_dim=64, item_key_dim=None`; **separate** uses `emb_dim=8, item_key_dim=64`. The encoder's item embedding is 8× narrower in the SK arm. This contradicts the paper's construction (SK "leaves the encoder unchanged," `∂h/∂k=0`, lines 269-275, 850) and the synthetic construction (`ModelConfig.emb_dim=8` for both arms, `_p2_config.py:81`; toggle adds only the key, `_p2_toggle_sweep.py:178`).
- Consequence: the real-data person-side story ("SK gives the stronger ability anchor," r=.54 vs .36, line 1206) and the "design-robust item map / design-sensitive person map" split (1204-1207) have a mundane alternative: θ̂ comes from encoders of different item capacity (64 vs 8), while the item params come from *width-matched* readouts (both 64), so of course the item map is robust and the person map is not. The NRM real arms *are* width-matched (both `emb_dim=8`, `:172,174`), so this confound is specific to EdNet-2PL and TIMSS-GPCM — precisely the cells carrying the person-side and "design-robust" claims.
- Reviewer would write: *"In the real study the shared and separated arms do not share an encoder — the item embedding is 64 vs 8 (`_p2_realstudy.py:167-169`). The person-side differences you attribute to the amortization path are confounded with encoder capacity. Match the widths or drop the person-side interpretation."*

**M1. The NRM "every head beats direct" gain is a broken-baseline artifact; no head beats the trivial baseline.**
- Evidence: direct NRM accuracy is .526/.554/.565 (`realdirect_table.md:5-7`; paper 1065-1067), which is 9-13 pts *below* the item-wise popular-option baseline of ~.653 (appendix line 1680; body 1278). The IRT heads reach only .636-.648 — also below .653. The direct head routes all item information through one shared MLP over a 64-d embedding (`_p2_direct_model.py:89-93`), whereas the NRM head has per-item option intercepts `c_k` by construction — so the gap is a per-item output-parameter mismatch, not "item-option structure that direct predictors cannot recover from histories" (1093-1096).
- The paper concedes the baseline point at 1275-1283 but makes the "gains are large … structured heads are not merely post-hoc interpretability layers" claim at 1093-1096 without it — an internal contradiction between §4.3 and §4.5.
- Reviewer would write: *"A per-item majority-option rule (.653) beats every model in Table 3's EdNet-NRM column. The 'large gains over direct' (line 1095) reflect a direct baseline that underperforms a static prior, not predictive value of the NRM head. This cell licenses no positive nominal-prediction claim."*

**M3. The capacity controls that would defuse the synthetic confound are named but never reported.**
- The hyperparameter table lists a "narrow-key capacity control (item key 16)" and a "widened-embedding control (shared width 16 to 96)" (lines 1708-1709), but no numbers for either appear anywhere in 804-1367 or 1528-1762. On the page, the synthetic recovery gain (SK's width-64 key vs SH's width-8 embedding, `tab:mass`) is not separated from raw parameter-head capacity.
- The results exist off-page and would largely settle it (`exposure_rerun_results.md:384-389, 404-409`: shared-width-96 plateaus ≈0.056 below decoupled at 2.3× params; a width-16 key already beats the shared arm). They must be in the paper.
- Reviewer would write: *"SK differs from SH in both separation and readout width (8→64). Show the key-16 and width-96 controls; otherwise the recovery gain (Table 2) is not attributable to the amortization path."*

**M4. Every interpretive real-data exhibit rests on one encoder; the person-side and reversal exhibits on one fold.**
- Confirmed in code: `fig:agreement_both` (`_p2_v3_agreement_both_fig.py:57`), `fig:ednet_2pl_shsk` (`_p2_v3_ednet2pl_shsk.py:55-56`), `fig:timss_shsk` (suptitle "LSTM-SH vs LSTM-SK," `_p2_v3_shsk.py:816`), `fig:ednet_nrm_shsk`, and `fig:reversal_bridge` are all LSTM. Reversal/person-side quantities are fold-0 of 5 seeds (`_p2_v3_reversal_bridge.py:120,175,204` `traj_d{s}_f0`; captions 1303, 1648). The 3-encoder table (`tab:real_prediction`) replicates only *prediction* (near-tied accuracies); no interpretive claim — distractor point-biserial .60/.75, expected-score maps, threshold map, ability r — is shown for transformer or DKVMN.
- Reviewer would write: *"The entire real-data interpretation is a single encoder; its person-side numbers are a single fold. Provide the distractor-agreement, threshold, and ability contrasts across all three encoders with seed-clustered intervals, as you did for synthetic recovery."*

**M5. Zero misspecification-robustness evidence exists.**
- `run_misspecification_probe.py` and `tests/test_misspecification_probe.py` are present; no output files exist anywhere under `results/` (verified). The 7-violation probe (local dependence, noisy/disordered thresholds, response style, DIF, exposure imbalance, drifting θ) was never run. Every real-data recovery/agreement claim assumes the fitted IRT model is a reasonable description; nothing tests that.
- Reviewer would write: *"The recovery thesis is demonstrated only under a perfectly specified generator. With no misspecification stress test, there is no basis for transferring 'accuracy hides parameter error, and SK repairs it' to the real datasets you foreground."*

**M6. DKVMN real-data cells are single-/few-fold and the note misstates their coverage.**
- `realdirect_table.md:22,28-29`: dkvmn-direct = 3 folds; dkvmn IRT-shared = 6/1/1/1; separate = 1/1/1/1. `realstudy_table_dkvmn.md` shows 4-6 of 25 folds, mostly seed-0; the routed DKVMN-NRM (`p2_v3_arm1r/dkvmn_nrm_ednet_*`) is 5 folds all from seed 0 (`d0_f0..d0_f4`). The paper note (1082-1083) says "DKVMN-SH uses 5-6 folds" — true for one column (EdNet-2PL) and false for the other three (single fold).
- Reviewer would write: *"DKVMN rows are not comparable in weight to the 25-fold LSTM/transformer rows; three of four DKVMN-SH cells are a single fold. State per-cell n or drop DKVMN from the headline table."*

## MINOR

**m1. "Prediction changes remain small" (fig:dd, 985/995) is false for GPCM/NRM.** `tab:mass`: transformer GPCM +3.3pp (.459→.492), transformer NRM +4.0pp (.449→.489), LSTM GPCM/NRM +1.5pp — all above the paper's own 1pp tie threshold (870). The clean "separable quantities" story holds only for 2PL (accuracy tied, recovery moves); for GPCM/NRM, SK improves *both*, which is "SK dominates," a different and weaker claim.

**m2. Disattenuated person-side correlations lean on reliability ≈0.25.** `tab:ednet_two_resolution` reports "disattenuated .59 SH/.63 SK" by dividing raw .18/.33 by √(cross-seed reproducibilities .245/.506) (`theta_cross_reading.json`). Cross-seed reproducibility is a non-standard reliability proxy; at .245 it nearly triples the correlation and *compresses* the SH/SK gap (.18/.33 → .59/.63), undercutting the raw-contrast "SK stronger person-side" prose.

**m3. KDD is decoration, and used only where it ties.** KDD appears solely in `tab:real_prediction` (accuracy tie ~.82-.85; no case study/figure/interpretive claim — grep). Where KDD is informative it is unflattering and omitted: `realstudy_table_dkvmn.md:11-12,23` shows KDD MML-concordance .41/.51 and truth-free slack .57/.53 (the health screen flags it). It adds no evidence to any claim.

**m4. MML EdNet-NRM is apples-to-oranges.** Paper MML = .609† on 34% coverage (`realdirect_table.md:46` coverage 0.34; all-position value 0.279); daggered, but placed beside full-coverage .636-.645 routed heads.

**m5. `tab:beds` and the hyperparameter table advertise an adaptive-testing experiment the results never report.** The design table lists "Adaptive testing" (922-925) and Table B.4 lists its constants (1719-1723), but the experiment paragraph is commented out (872-880) and no CAT exhibit appears in 804-1367. (The CAT results exist and are strong in the campaign — `exposure_rerun` Phase 7/11 — so this is a self-inflicted omission of the paper's most direct downstream-impact evidence, not a null.)

**Tensions a careful reader will catch (evidence-chain, board item e):**
- SH beats SK on NRM prediction (1099) while SK beats SH on NRM recovery (`tab:mass`) and distractor agreement (.75 vs .60, 1332) — disclosed as "distinct goals," but it means the synthetic headline "separation is ~free" does *not* replicate on the one real nominal dataset (SK there costs top-1 and NLL).
- "Design-robust item map" (β r=.998) coexisting with "design-sensitive person map" is explained by M2 (item readouts width-matched; encoders not).
- The synthetic `tab:mass` NRM "SK" column comes from the routed/reparameterized head (`p2_v3_arm1r/tabmass_nrm.md`; the plain mass-table has NRM "missing_on_disk," `mass_table.md:11-17`) — a different lever than the plain decoupled key used for the 2PL/GPCM "SK." Consistent with the methods' NRM definition, but not the "same manipulation" a reader assumes across the table.

---

# PART 2 — WHAT SURVIVES (claims I would accept, with honest effect sizes)

1. **Synthetic, well-specified 2PL: next-response accuracy does not determine discrimination rank-recovery.** Under a shared width-8 item embedding, discrimination ρ is .37-.55; giving the parameter head its own wider key raises it to .81-.91 with accuracy unchanged within 1pp (`tab:mass`/`mass_table.md`, 25 fits/cell, seed-clustered CIs, sign-consistent across 5 datasets). This is clean and well-powered. Effect: Δacc < .01, Δρ_discrimination ≈ +.3-.4. This is the paper's real result.

2. **The direction generalizes across encoders and to GPCM in synthetic** (`tab:massfull`), with the caveat that for GPCM/NRM the separated arm also gains 1.5-4pp accuracy — so state it as "SK dominates on both axes," not "separable."

3. **Real-data prediction parity on binary/ordinal:** IRT-parameterized heads impose no systematic next-response accuracy cost vs a same-backbone direct head on EdNet-2PL, KDD-2PL, TIMSS-GPCM (within ~1-2pp, mixed sign; `tab:real_prediction`). Defensible as stated.

4. **Item-location robustness on EdNet-2PL:** learned difficulty rank is near-invariant to the readout path (β ρ=.998) and tracks the empirical p-value (−.97) and MML β (.73) (`tab:app_ednet_2pl_shsk`). Location recovers; consistent with the campaign's "location recoverable, scale not."

Everything **nominal (NRM)** and everything **person-side (ability)** is weak (r≈.18-.54), confounded (M2), single-encoder/single-fold (M4), or baseline-dominated (M1). At my venue these are "suggestive, not established." The **TIMSS ordinal-ordering** claim (F1) does not survive at all as written.

---

# PART 3 — THE ONE DEMAND

**Run the misspecification probe that is already built (`run_misspecification_probe.py`) and report whether the accuracy-hides-error phenomenon and the SK repair survive a mis-specified generator.**

Rationale: the paper's entire thesis is demonstrated only under a generator that matches the fitted model exactly (θ∼N(0,1), lognormal α, ordered β). Real assessment data are misspecified by construction, and the paper has **zero** evidence either way (M5). This single analysis is what separates "a clean synthetic curiosity" from "a claim about real measurement," and the code exists.

Concrete design:
- Keep the reference cohort (N=2000, Q=200, E=600), 5 datasets × 5 folds, seed-clustered CIs — identical to `tab:mass` so results drop into the same figure.
- Apply the seven violations one at a time and jointly, at low/medium/high dose: local dependence (item pair residual correlation), noisy thresholds, response-style shift, threshold disorder, DIF, exposure imbalance, drifting θ.
- Fit the same SH and SK 2PL/GPCM heads. Report, per condition: Δaccuracy(SK−SH), Δρ_discrimination(SK−SH), and the **truth-free slack** value (already validated, `exposure_rerun` Phase 9) so the diagnostic is shown to fire (or not) under each violation.
- Pre-register the reading: SK's recovery advantage should shrink monotonically with dose; the claim to defend is that it does not *reverse* and that slack still separates healthy from unhealthy readouts under misspecification.

Verdict impact: if the SK advantage and the slack diagnostic survive misspecification, the synthetic result earns generalization to the real datasets the paper foregrounds, and points 1-2 of "what survives" become a genuine contribution. If they collapse or reverse, the synthetic headline is an artifact of perfect specification and must not be presented as guidance for real data.

Close second (real-data cleanup, if only one real-side change is possible): re-run EdNet-2PL and TIMSS-GPCM real study with the encoder item-embedding width **matched** across SH and SK (both `emb_dim=8`, SK adds `item_key_dim=64` — the synthetic construction), across all three encoders and 25 folds, reporting the person-side and agreement contrasts with intervals. This removes M2 and extends M4 in one run; if SK's person-side edge survives width-matching it is real, and if it collapses it was capacity.