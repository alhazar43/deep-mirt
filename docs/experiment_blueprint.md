# Experiment blueprint (revised): "Not All Parameters Learn Alike"

Status: revised via a 5-agent ultracode workflow (diagnose-wheels + experiment-design + psychometric, then
synthesize, then adversarial verify), 2026-07-01. Adversarial verdict was REVISE (targeted, not structural); the
fixes are FOLDED IN below and flagged [ADV]. This is the experiment + code plan of record. Nothing runs until the
user resolves the open decisions at the end.

## Storyline (6 steps, fixed by the user)
1. Synthetic BENCHMARK proving encoders and decoders are swappable. Prediction loss only, no IRT NLL. Per decoder:
   2PL binary CE, GPCM (K=4, ordinal) cumulative-link ordinal CE (not softmax CE, not WOL), NRM (K=4, nominal) softmax CE.
2. Benchmark is a FIXED setup: shared item embedding, static discrimination, no decoupling, no dynamic. Toggles later.
   Terminology: "item key" and "value embedding" only.
3. Report BOTH predictive accuracy AND parameter recovery (rho or r per parameter).
4. Result: prediction good, ability ok, GPCM/2PL discrimination (alpha) recovers badly; the NRM slope (a_k) does not
   lag (the control). Math motivates why -> the study.
5. Hypothesis: slow parameters need their own embedding channel. Toggles: 4 for 2PL, 4 for GPCM, up to 10 for NRM.
   Find the minimal-sufficient set.
6. Prove step 5 on realistic synthetic + real data (shuffled order, variable length, variable exposure, larger banks;
   reference ma-irt configs, do not copy its generator).

## Ground truth verified against the repo (these change the build)
1. The `_p2_*` scaffold and `configs_p2/` EXIST already. Reuse, do not create.
2. NRM selective-decoupling routing already lives in scratch: `_nrm_asym.py` (A_ONLY_DEC, C_ONLY_DEC x static/dynamic)
   and `_nrm_repr.py` (SHARED / DECOUPLED-one-key / ALL_DECOUPLED-separate-keys x static/dynamic/dynamic-a_k). Core
   `NRMDecoder` only offers shared vs both-on-one-key. Promote this routing into `_p2_nrm_channels.py`; do not rebuild.
3. GPCM ordinal-CE is genuinely ABSENT. GPCM currently trains on `WeightedOrdinalLoss` (WOL), which the user bans.
   `_p2_ordinal_ce.py` is a real build.
4. LOAD-BEARING (`decoders.py:374-420`): the static-decouple toggle routes ONLY difficulty (beta) to the item key;
   the discrimination (alpha) reads the thin value and rides the item key ONLY when `state_alpha=True`
   (decoupled+dynamic). So the naive 2x2 does NOT cleanly cross "discrimination has its own channel" x
   "discrimination is dynamic." See Option A below.

## A. Fixed benchmark (steps 1-3)
Grid: {lstm, transformer, dkvmn} x {2PL, GPCM K=4, NRM K=4} = 9 cells. Fixed setup = `DeepIRTEngine` default
(`decouple=False`, `state_alpha=False`, `item_key_dim=None`). Runs on the primary bed (section C): Q=200, K=4
(K=2 for 2PL), incomplete administration. 150 epochs max, Adam, grad clip, EARLY STOPPING (patience ~10, restore
best). Same ground-truth data + same partition across the 9 cells; report `n_params` per cell (forecloses the
capacity-artifact rebuttal).

### GPCM ordinal-CE (exact, the only build)
From GPCM per-category logits psi (psi_0=0, psi_k = cumsum alpha(theta-beta_c)): F_k = P(Y>=k) = sum_{j>=k} softmax(psi)_j,
k=1..K-1; cut targets t_k = 1[y>=k]; loss L(y) = -sum_k [t_k log F_k + (1-t_k) log(1-F_k)]. Compute log F_k,
log(1-F_k) by partitioned log-sum-exp over the tail/head of psi (stable, no clamps). Strictly proper (unbiased probs,
which the recovery claim requires); NOT order-blind softmax CE; reduces EXACTLY to 2PL binary CE at K=2 (unifies the
family). Methods caveat to STATE: this is the graded-response cumulative likelihood, not GPCM's adjacent-category
likelihood; applied to GPCM cumulative probs it is still proper and Fisher-consistent for (alpha,beta) under a GPCM DGP.

### [ADV] The "no IRT NLL" framing, restated correctly
2PL binary CE IS the 2PL likelihood and NRM softmax CE IS the NRM likelihood, so "no IRT NLL" cannot mean "never the
model NLL." State the two actual principles: (i) the loss is the order-appropriate STRICTLY-PROPER scoring rule for the
response type (binary->BCE, nominal->softmax CE, ordinal->cumulative-link CE; softmax is order-blind, WOL is improper);
(ii) "no IRT NLL" means theta/alpha/beta are AMORTIZED encoder/decoder outputs, not free per-person/item MLE parameters.

### Recovery metrics (per parameter)
- ability theta: Pearson r (after one global sign) primary; Spearman rho secondary; dynamic scored as net drift.
- discrimination alpha (GPCM/2PL) and slope a_k (NRM): Spearman rho HEADLINE (the multiplicative scale gauge leaves
  magnitude unidentified, rank is gauge-free); plus a linked-scale regression coefficient (<1 = attenuation) and RMSE
  for the magnitude diagnostic.
- difficulty beta / b (GPCM/2PL) and intercept c_k (NRM): Pearson r (after mean-shift link); sort GPCM thresholds; report pooled and per-step.
Predictive accuracy: 2PL AUC + acc; GPCM quadratic-weighted kappa + acc; NRM top-1 + macro-AUC.

### [ADV] Replication unit -- NOT "5-fold, paired Wilcoxon across folds" (HARD BLOCKER)
Paired Wilcoxon on 5 folds has minimum two-sided p = 2/2^5 = 0.0625 > 0.05, so it can NEVER reach significance and
every "significantly above baseline / indistinguishable from best" decision would be inert. USE the scaffold's
Monte-Carlo replication instead: >=15 replicates per cell (e.g. 5 data seeds x 3 init seeds, a FRESH ground-truth bank
per data seed), paired across replicates (Wilcoxon signed-rank + rank-biserial). This rescues significance, drops the
unneeded `_p2_cv.py` build, and gives independent-bank replication, which is STRONGER for a recovery claim than
one-bank K-fold (recovery is scored against known ground truth, so K-fold cross-validates only prediction/theta, not
recovery). This overrides the user's "5 CV" -- see open decision 1.

### [ADV] Early-stopping honesty + no leakage
Discrimination recovery peaks early (~ep50) then decays while prediction keeps improving, so prediction-early-stopping
stops PAST the alpha-recovery peak, a pessimistic but correct baseline (using ground-truth recovery to pick the stop
would be oracle circularity). STATE this; assert the stopping rule (metric, patience, restore-best) is byte-identical
across all cells. Early-stop on an INNER training slice; report prediction on the untouched held-out portion.

### [ADV] Swappability is an equivalence claim
"Accuracy flat across encoders" needs a pre-registered negligible-margin band with CIs inside it (TOST-style), not a
non-significant difference test.

### Step-4 transition (math motivation)
Fisher-information figure via `_p2_cat.gpcm_fisher_information` alongside recovery. The multiplicative scale gauge, one
paragraph: alpha multiplies theta, so alpha->alpha/c, theta->c theta leaves every probability unchanged (magnitude
unidentified, rank survives); the discrimination score d logP/d alpha ~ (theta-beta) p(1-p) vanishes at theta=beta and
carries no alpha prefactor, whereas difficulty scores ~ alpha p(1-p) ride the alpha factor, so discrimination sits lowest
on the Fisher-leverage ordering. Present the algebra at K=2 (cumulative-link = 2PL = exact) and state it generalizes.

## B. Toggleable study (step 5)
### The routing catch (ground-truth 4): Option A recommended
Build `_p2_gpcm_alpha_key.py` (a GPCMDecoder subclass whose STATIC fc_a reads the item key when `item_key_dim` is set),
so decoupled-static genuinely puts the discrimination on its own channel and the 2x2 cleanly crosses
{discrimination-channel} x {discrimination-dynamic}. Option B (accept the entanglement, frame the static arm as
difficulty-decoupled) is cheaper but confounds the headline factor with dynamics. See open decision 2.
### Configs
- 2PL: 4 = {shared, decoupled} x {static, dynamic}. GPCM: 4, same 2x2 (the headline decoder; carry its full 2x2 into 6a).
- NRM [all 10 setups per decision 3]: 5 decoupling configs x {static, dynamic}. Configs 1 shared, 2 a_k-only,
  3 c_k-only, 4 both-one-key, 5 both-separate-keys. The full run decides which NRM parameter needs its own item key;
  do NOT presume the slope a_k is the hard one (the deck's preliminary evidence points the other way, to the intercept
  c_k). Report the static/dynamic contrast per config (the dynamic head is the robust real-data lever, and NRM tests
  whether it helps a parameter whose Fisher is not low).
### Encoder budget: full toggle grid on LSTM; replicate {baseline, minimal-sufficient winner} on transformer + dkvmn.
### Minimal-sufficient rule: lowest-capacity toggle whose discrimination Spearman is significantly above baseline and
indistinguishable from the best corner, paired across the >=15 replicates; guardrails theta/location/accuracy must not regress.
### [ADV] Scientific caveat to preflag: in GPCM, decoupling gives the discrimination its own item key. In NRM the
deck's run finds the OPPOSITE parameter needs it -- the intercept c_k, not the slope a_k (decoupling c_k alone leaves
a_k ~0.96 on the narrow value; decoupling a_k alone breaks c_k to ~0.39; separate keys for both add nothing). So do
NOT presume the slope a_k is the hard NRM parameter; the full 10-setup run decides. This is NRM's role as the control:
the representation trade-off is paid by whatever readout shares ability's embedding, while Fisher information is a
separate mechanism governing the recovery rate and where the dynamic head helps. Re-confirm on the primary bed (the
prior result predates it).

## C. Primary synthetic bed (ma-irt-aligned): build `_p2_datagen_realistic.py`
This ma-irt-style incomplete-administration bank is the PRIMARY bed for the benchmark (A), the toggles (B), and the
proof, not a separate "realistic" leg. Reuse `datagen._gpcm_probs`/`_sample` + the NRM softmax draw; extend the dataset
containers; do not edit `datagen.py`. Reference ma-irt `configs/{block_q200_k4, randomwalk}.yaml` for shape only; the
generator stays ours.
- Bank: Q=200 items primary (Q=500 a later stress), K=4 for GPCM/NRM, K=2 for 2PL.
- Administration IS the only noise. Each learner is administered a RANDOM selection of items; the number of items per
  learner ~ Uniform(40,80) integer (exactly ma-irt's range). Selection is theta-INDEPENDENT (non-adaptive).
- Responses are CLEAN IRT draws from the GPCM/NRM/2PL model. NO guessing, NO lapse, NO response noise; realism comes
  purely from the random, incomplete administration.
- Learners N ~ 2000, scaled so each item gets ~500-600 takers (per-item minimum ~100-200 takers).
- Ability: static theta ~ N(0,1) primary; a random-walk drift arm as robustness.
- Items: discrimination ~ LogNormal(0,0.5); GPCM difficulty = ordered step thresholds; NRM slope a_k and intercept c_k
  centered.
- Emit the per-learner administered (seen) set; score recovery on SEEN items only, exposure-stratified.
- Keep one small DENSE control bed (everyone sees all items) alongside the primary.
Order is shuffled; variable length follows the Uniform(40,80) administration count (+ boolean mask; padding is
tail-only, the encoder is causal, `fit(mask=...)` masks the loss). Data-sufficiency floors: >=100-200 obs/item; check
per-category counts; keep exposure theta-independent.
### [ADV] Seen-mask fix (biases the contrast otherwise): the static `recover()` path returns NO seen key (only the
state_alpha path does), so unseen items keep at-init embeddings and the shared-static baseline would be scored on
garbage, artificially depressing the very baseline the fix is compared to. `_p2_datagen_realistic` must emit the
per-learner administered set; `_p2_engine` computes the global seen-union and threads it to `item_recovery(seen=...)`
for ALL configs. Also: per-learner holdout = min(h, floor(frac*T_n)) so short learners keep history;
exposure-stratified + length-stratified scoring.

## D. Real data (step 6b): split-half reliability (no ground truth)
Fill `_p2_reliability.py` (split_half_reliability, reliability_by_coverage, spearman_brown, item_bootstrap_ci). Split
learners into halves, fit each, correlate per-item recovered params on the common bank (discrimination / slope a_k by
Spearman, difficulty / intercept c_k by Pearson), Spearman-Brown correct; theta split by items. {shared, fix} contrast,
reliability guarded by accuracy,
stratified by coverage. Datasets: EdNet/KDD/ASSISTments 2PL; EdNet options NRM; coerced-ordinal EdNet GPCM (flag the
coercion honestly). This leg adjudicates decoupling-vs-dynamic on real data (prior evidence favors the dynamic head).

## E. Reuse vs build (no reinvention)
REUSE (Codex-owned, read-only): 3 encoders (`core/encoder.py`, `dkvmn_encoder.py`, `transformer_encoder.py`);
3 decoders (`core/decoders.py`); 2PL BCE + NRM softmax CE (`model.py`, `CombinedLoss`); fixed baseline (DeepIRTEngine
default); toggles (`item_key_dim`, `state_alpha`); recovery (`metrics_bench.item_recovery`, `theta_recovery_*`,
`nrm_metrics.item_recovery` with `seen=`); prediction metrics; Fisher (`_p2_cat.gpcm_fisher_information`); config+resume
(`_p2_config`, `_p2_sweep`); variable-length training (`DeepIRTModel.fit(mask=...)`).
BUILD (new `_`-prefixed): `_p2_ordinal_ce.py` (the cumulative-link CE; unit-test K=2 == 2PL BCE); `_p2_model.py`
(`_build_loss_fn` override for GPCM + an early-stop fit loop, since `fit`'s callback only notifies); `_p2_engine.py`
(construct `_P2Model`; masked `predict_heldout` for variable length); `_p2_datagen_realistic.py`; `_p2_nrm_channels.py`
(promote the `_nrm_asym`/`_nrm_repr` routing); `_p2_gpcm_alpha_key.py` (Option A).
FILL-INS (existing stubs): `_p2_aggregate.py` (bootstrap_ci, aggregate_rows, wilcoxon_comparison, build_swap_matrix,
build_kendalls_w); `_p2_run_cell.py` (NRM recovery path via `model.recover_item_params` + `nrm_metrics.item_recovery`;
wire `agg`; early stop); `_p2_config.py` (add `train.gpcm_loss` = ordinal_ce|wol default ordinal_ce, + a data.regime/
exposure block); `_p2_reliability.py`; `_p2_apply_gauge` only for the alpha magnitude diagnostic.
[ADV] CUT `_p2_cv.py` -- a true K-fold learner harness is unnecessary and inferior here; use `run_cell`'s Monte-Carlo
seed loop as the replication/pairing unit.
Build order: (1) ordinal_ce + `_p2_model` + `_p2_engine` + gpcm_loss field, unit-test K=2; (2) `_p2_aggregate`;
(3) early stop + wire agg (MC replicates); (4) NRM data + recovery in `run_one`; (5) `_p2_datagen_realistic` (the
primary bed, section C) + masked engine + the dense control bed; (6) 9 benchmark configs on the primary bed, run
steps 1-4; (7) toggle configs (+ `_p2_gpcm_alpha_key` Option A, `_p2_nrm_channels`); (8) real-data via `_p2_reliability`.

## F. Run count and compute
At 5 fits/cell: 45 benchmark + 55 + 30 toggle + 100 realistic + 50 real = ~280 new fits. [ADV] With the MC-replication
fix (>=15/cell) the benchmark and toggle counts roughly triple, so the true envelope is ~450-800 fits, ~2-3 GPU-days on
the 8 GB RTX 4060. Primary-bed fits (Q=200, incomplete administration) are a few minutes each; the heavy tail is the
Q>=500 stress -- gate it behind the Q=200 primary. Recount precisely once the replication unit (open decision 1) is fixed.

## Decisions (LOCKED 2026-07-01)
1. Replication = **5-FOLD LEARNER CV** (user's choice over the MC recommendation). Since 5 paired folds cannot reach
   p<0.05 via Wilcoxon, toggle comparisons are adjudicated by the **paired fold-mean difference + bootstrap 95% CI +
   rank-biserial effect size**, NOT a 5-sample significance test. Report recovery as fold-mean +/- bootstrap CI.
2. GPCM/2PL routing = **Option A**: fix the decoupling so **discrimination** (2PL/GPCM's single multiplicative
   parameter) routes to its OWN item key in the STATIC arm (current code routes only difficulty there). Clean
   {discrimination-decoupled} x {static, dynamic} 2x2.
3. NRM = **run ALL 10 setups** (5 decoupling configs x static/dynamic).
4. The ability-item coupling / structural-identifiability arm (E2) = **DROPPED** as invented drift not in the deck;
   the mechanism is the multiplicative scale-gauge coupling only. Stay on the 6-step storyline.

Full workflow output (three diagnoses + synthesis + adversarial verdict) is in the run transcript for this session.
