# Project Handoff (START HERE)

Last updated 2026-07-01. Orientation for a fresh conversation. This handoff is
scoped to the CURRENT active work: the **Q-MIRT transfer / active-learning paper**
(the "show learning via transfer" line) and the **NRM parameter-representation
study**. Older parked tracks (OrdRec/Duolingo, Chapter-0 ma-irt) are one-liners at
the end; do not start there.

Repo root: `C:/Users/steph/documents/deep-mirt`. Canonical branch
`feat/prediction-loss`.

## 1. What the active work is

Two threads, both prediction-trained neural IRT, both live on `feat/prediction-loss`.

**Thread A -- Q-MIRT: show LEARNING via cross-concept TRANSFER.** A dynamic,
multi-concept (MA-GPCM) model that tracks a NAMED per-concept ability over time and
answers: when a learner practices concept A, does performance on a related concept B
move. Framed as "fixed measurement, moving STATE" (NOT "evolving theta" -- IRT rejects
an evolving trait): item parameters stay fixed/identified, only the per-concept state
moves; learning is shown OBSERVABLY (anchor-item score growth, held-out prediction),
not by waving the latent. Recovery of the static MA-GPCM is already settled and is NOT
the contribution; the contribution is the active, structured cross-concept change.

**Thread B -- NRM parameter representation.** Extends the workshop deck ("Not All
Parameters Learn Alike", docs/slides/workshop.tex). See section 4; a redo is IN FLIGHT.

## 2. deep_irt structure (the framework both threads sit on)

`DeepIRTModel` (`deep_irt/core/model.py`): swappable sequence encoder + swappable IRT
decoder, trained end-to-end on a PREDICTION loss (IRT is a readout flavor; no IRT-NLL).

- Encoders (`core/encoder.py`): `lstm` (default), `transformer`, `dkvmn`. All expose
  `theta_for_prediction`, `state_for_prediction`, `aligned_theta_and_state`,
  `item_val_emb` (thin, feeds encoder/theta), `item_key_emb` (wide, feeds readouts).
- Decoders (`core/decoders.py`): `gpcm` (ordered K cats, WeightedOrdinalLoss),
  `binary` (2PL, BCE), `nrm` (unordered K options, CE), `bt` (pairwise).
- **Decoupled architecture** (`decouple=True`, default for gpcm/binary): `state_alpha`
  reads discrimination from a state-conditioned head (DYNAMIC); `item_key_dim=64` is a
  separate wide KEY table for the static readouts (DECOUPLED) while the thin value table
  feeds theta only. **`decouple` is a NO-OP for `nrm` and `bt`** -- NRM has no
  decoupled/dynamic heads yet (this is the gap thread B must build).
- Single-shift causal alignment: theta at step t is a function of history strictly before t.
- IRT params recovered AFTER training from frozen decoder weights (`recover_item_params`).

Codex owns `deep_irt/core/*`, `deep_irt/bench/run_*.py`, `datagen.py`, `engines.py` --
do NOT edit; extend additively from scratch (`deep_irt/bench/_*.py`, gitignored) or new
modules. Scratch files are `_`-prefixed and gitignored.

## 3. Thread A (Q-MIRT transfer) -- overnight campaign results

Full per-venue log with every number and root-cause: `docs/overnight_transfer_active_campaign.md`.
Models/generators in `deep_irt/bench/_qmirt_*.py` (all gitignored scratch).

### 3.1 The model (technical)
`deep_irt/bench/_qmirt_state_model.py` (`ExplicitStateModel`) and
`_qmirt_state_model_fb.py` (`ExplicitStateModelFB`, the DEFAULT). Explicit per-concept
state `z` (D concepts), causal transition:
```
z_{t+1,c} = decay_c * z_{t,c} + own_gain_c * Q[item_t, c]
          + resp_feedback_c            # FB only: Q-gated PSI-KT innovation, own-concept
          + (prac_t @ G.T)[c]          # G = G_raw * (1 - eye(D)) = SOLE cross-concept route
```
- Item params (a_j discriminations, delta_j thresholds) fit static in Stage 1 then FROZEN;
  Stage 2 releases G with an L1 penalty (no-transfer is the default). Compensatory GPCM
  readout (logit = sum_c Q[j,c]*a_{j,c}*z_{t,c} - delta). Soft GPCM likelihood.
- ACTIVE-CHANGE isolation is structural: z_A never enters z_B's update; cross-concept
  only via G[B,A]; G-zero control gives pure decay on non-practiced concepts to ~1e-8.
- FB fix (PSI-KT): a Q-gated `resp_proj(one_hot(r_t))` innovation updates only the
  answered concept's own state (isolation preserved), which restored individual
  learning recovery.

### 3.2 The metric (technical)
`deep_irt/bench/_qmirt_forecast.py`. The gauge-free primary metric is the masked-forecast
"active gap": condition on [0,T_cond) with real responses, forecast [T_cond,T) with
responses MASKED, target concept measured-not-practiced while a source is practiced (so
the target's only forecast route is G[target,source]). active_gap = (No-G minus With-G)
forecast NLL on the target, WITHIN-CONDITION (item params + split-state cancel). Read
matched-null paired (transfer minus same-seed null), NEVER a fitted G against zero (the
fitted G carries a per-seed additive offset). Generator: `_qmirt_datagen2.py` (directed
exponential-approach learning curves, pure/anchor items, transfer G_matrix, decoupling
episodes, drift modes).

### 3.3 Findings (venues 0-4)
- **OBJ2 active change: achieved.** Structural isolation + the within-condition forecast
  control. Passive LSTM cannot forecast the target's rise without its responses.
- **OBJ1 transfer real: achieved (direction/existence; magnitude gauge-bound).** Forecast
  active gap +0.22 to +0.36 on the target, ~0 on controls and the null twin.
- **Survives confounds** (venue 2): correlated-no-transfer ~0, curriculum co-scheduling ~0
  in aggregate, shuffle-order COLLAPSES to 0, reverse-direction 0. Residual: pure
  co-scheduling is non-identified (collinear practice) -> needs decoupling episodes
  (stated requirement, same shape as the measurement-invariance gauge).
- **Survives noisy/non-monotone theta** (venue 3/3b): the clean-curve model FABRICATES on
  non-monotone data (null gap 96% of active gap); the PSI-KT mean-reverting (OU)
  transition CLEARS it, and with the regularizer loosened to l1=0.001 the signal survives
  at power (+0.066, 9/9 seeds).
- **Individual learning recovery 0.80** with FB (from 0.37 without response feedback).
- **Robust across active mechanisms** (venue 4): linear own-gain, mastery-ceiling gain,
  and rate+forgetting all carry real active transfer (~+0.36).
- **Measurement invariance** (person-learning vs item-drift): PROVEN-WITH-A-STATED-
  ASSUMPTION. Uniform global item drift = location gauge (unprovable from responses);
  differential drift is detectable via the early-vs-late anchor-stability check scored
  against the Q-induced baseline; the fixed-item model is fooled otherwise. Carry the
  differential-invariance check as a companion + state the no-uniform-anchor-drift
  assumption (standard IRT equating posture).

### 3.4 Honest sizing + corrections
- Magnitude is gauge-bound throughout (direction/existence only).
- Everything is SYNTHETIC under correct specification (estimator = generator family),
  D=3, small seed counts.
- CORRECTION made in venue 4: venue-1's "beats a passive LSTM by +0.63" was confounded
  (free vs frozen decoder) and model-seed-specific. The robust "active" operationalization
  is the WITHIN-CONDITION no-G control, not the passive-LSTM comparison.

### 3.5 Open (Thread A)
- **D-scaling to D=5,8**: needs the masked-forecast harness GENERALIZED (the D=3 version
  bakes concept roles, the single edge, and the practice/measure schedule into a fixed
  `ITEM_SEQ_TOTAL` in `_qmirt_forecast.py:151+`). Direct-G recovery is resp_proj-
  confounded, so the gauge-free forecast metric is the one to generalize. Build a SILENT
  runner (json-only) -- workflow agents twice crashed on the 32k output-token limit here.
- **KDD Cup 2010 real data**: needs that harness + a KC->concept mapping. No ground truth,
  so the claim steps down to predictive-improvement (transfer model beats no-transfer on
  held-out) + differential-invariance + seed stability. Judgment-heavy; do with the user.

## 4. Thread B (NRM parameter representation) -- corrected study IN FLIGHT

The workshop deck (docs/slides/workshop.tex) studies, for GPCM: prediction-trained neural
IRT recovers each item parameter by its FISHER LEVERAGE (difficulty fast, discrimination
slow); a SHARED embedding forces a capacity trade-off (wide helps discrimination but
overfits ability; difficulty indifferent); DECOUPLING (narrow value + wide key) escapes
it; DYNAMIC (state-conditioned) heads reach the escape faster and rescue reliability on
real data (EdNet, KDD). Synthetic setup: GPCM, known theta/alpha/beta, LSTM encoder,
N=800, Q=60, K=4, prediction loss, reported at 150 epochs, >=8 seeds, 95% CIs. Two lenses:
recovery (vs truth) and reliability (split-half agreement, works without ground truth).

**The NRM question (correctly framed):** NRM gives each category a slope `a_k` and
intercept `c_k`. `a_k` is slope-on-ability like GPCM alpha but its Fisher is
near-symmetric with `c_k` (I_a/I_c ~ 0.90, not 5-10x). So run the SAME architectural
sweep -- {shared, decoupled} x {static, dynamic} for the a_k and c_k readouts, plus the
shared-width sweep -- and measure recovery + reliability for theta, a_k, c_k, and the
TRADE-OFFS theta<->a_k, theta<->c_k, a_k<->c_k. Because a_k is a slope but NOT low-Fisher,
this dissociates whether the shared-embedding trade-off comes from being a SLOPE
(representation) or from LOW FISHER. Build the decoupled/dynamic NRM heads additively
(decouple is a no-op for nrm in core). Prior data point (memory nrm-decoder-default):
a naive state-conditioned a_k did not earn the default (unstable, no gain) -- re-examine
properly. Real data only if synthetic is meaningful: NRM fits OPTION TRACING (modeling
which multiple-choice option a learner selects; search for datasets then).

The earlier "objective A" pass (Fisher symmetry + recovery-trajectory + gauge audit) is
CORRECT as far as it goes and produced 3 gauge-clean slides now in the deck, but it did
NOT run the architectural sweep -- that is the redo. Files: `deep_irt/bench/_nrm_leverage.py`,
`_nrm_gates.py`; results `_nrm_leverage.json`, `_nrm_gates.json`.

## 5. Operating conventions (carry into the new conversation)

- **Env.** `source ~/anaconda3/etc/profile.d/conda.sh && conda activate research`, then
  `export PYTHONPATH=".;rl/src;ma-irt"` (Windows `;` separator) and
  `export KMP_DUPLICATE_LIB_OK=TRUE`. Tests: `python -m pytest deep_irt/tests/`. CUDA is
  an RTX 4060 Laptop 8 GB (single GPU -> runs are sequential).
- **Do NOT edit Codex-owned files**: `deep_irt/core/*`, `deep_irt/bench/run_*.py`,
  `datagen.py`, `engines.py`. Extend additively in `_`-prefixed gitignored scratch.
- **Execution discipline (learned the hard way).** Run training SYNCHRONOUSLY in the
  foreground and WAIT -- do NOT launch detached/background jobs (they silently die).
  Scripts write full results to a JSON; agents return SHORT summaries (<600 words) --
  do NOT paste per-cell logs (agents crash on the 32k output-token limit).
- **Model economy.** Subagents on sonnet, trivial on haiku; reserve the top model for the
  main loop, planning, verification. Decompose independent work.
- **Writing style (strict).** No em-dashes or en-dashes, no colons in flowing prose,
  American English. Use ESTABLISHED names, never invent labels (memory use-established-names).
  Slides: noun-phrase titles reused as bold summary leads, terse bullets, grant-then-qualify.
- **Staging.** Never `git add -A`; explicit paths only. Never stage `__pycache__`,
  `outputs/`, `*/data/`, `archive/`. No Co-Authored-By / Claude attribution; author = user.
- **PSI-KT is AGPL** -- reference its design (per-concept state, OU mean-reverting transition
  with a learned concept graph, generative ELBO, per-learner traits, single-concept readout)
  but never vendor its code into deep_irt.

## 6. Immediate next steps
1. Thread B: build the decoupled/dynamic NRM heads + run the shared/decoupled x
   static/dynamic sweep with the workshop synthetic setup; report the theta<->a_k,
   theta<->c_k, a_k<->c_k trade-offs (recovery + reliability). (Workflow in flight.)
2. Thread A: hand-build the D>3 masked-forecast harness (silent runner) to close
   D-scaling, then the KDD option (with the user).
3. If Thread B synthetic is meaningful, find an option-tracing real dataset.

## Parked / separate tracks (one-liners, do not start here)
- **ma-irt** (`ma-irt/`): frozen Chapter-0 deep ordinal IRT, IJAIED submission. Submodule;
  additive configs only.
- **OrdRec** (`rl/`): parked ExRec-style ordinal item-recommendation (E1-E4.7 + D1 SLAM).
  Prior handoff details in git history / `docs/duolingo_mini_plan.md` if that track resumes.

## Pointers
- `docs/overnight_transfer_active_campaign.md` -- Thread A per-venue log (numbers + root-causes).
- `docs/slides/workshop.tex` (XeLaTeX, SimplePlusAIC theme) + `docs/LEARNING_DYNAMICS_STUDY.md`
  -- Thread B framework and the GPCM result.
- `deep_irt/README.md` -- framework API and the decoupled/dynamic architecture.
- `deep_irt/bench/_qmirt_*.py`, `_nrm_*.py` -- scratch models/generators/probes (gitignored).
