# Resubmission experiment plan (2026-08-13)

Audit-grounded implementation plan for the five resubmission tasks
(gradient-isolated SH control, key-width ablation completion, matched
EdNet-250 2PL, cross-format analyses, GPCM recovery audit). Produced
from a four-way read-only code/store audit (agent reports in the
session record; every claim below carries a file:line citation there).
No runs launched yet; this document is the GO gate.

## A. Audit verdicts (the eight questions)

### A1. What exists, what is missing

- SH/SK synthetic cells exist for all 9 encoder x decoder combinations
  at N=2000/Q200, 25 fits each (kt-irt/results/p2_toggle + the ported
  NRM cells). BUT: held-out NLL was never recorded for 2pl/gpcm cells
  (metrics_bench returns acc/qwk/auc only), no weights were saved, and
  no held-out probabilities persist -- so paired NLL against a new
  third path CANNOT be obtained from the existing stores.
- Key-width evidence: outputs/p2_width = SHARED-embedding width sweep
  (lstm only, all three decoders, emb 8..96, SK pinned at 64);
  outputs/p2_narrowkey = the ONLY SK-key-width variation in the repo:
  lstm x {2pl, gpcm} at key=16, 25 folds each. Transformer and DKVMN
  have ZERO width evidence anywhere.
- Matched EdNet-250: no frozen histories exist. bank_manifest.npz
  stores only the answer key + learner count; every consumer rebuilds
  the bank via load_full+restrict_top. restrict_top uses an unstable
  argsort -- deterministic on this machine/numpy, NOT guaranteed
  portable if item counts tie at rank 250. The Eedi bank.npz precedent
  (items, resp, correct_opt) is the freeze template.
- GPCM recovery: the headline synthetic b_spearman is SORTED learned
  steps vs generating steps that are sorted at generation --
  order-blind by construction. Only two call sites ever pass
  sort_beta=False (the misspec battery and the TIMSS rawbeta rerun).
  The stored arrays npz contain the SORTED export only (verified: 100%
  ascending rows); raw-order statistics for the historical synthetic
  campaigns are NOT rescorable (sort is not invertible, no permutation
  stored).
- Integrity catch: the P1 NLL/ECE exhibit has NO committed generator,
  store, or weights trail -- its numbers are not reproducible from the
  repository. The new grid re-derives paired NLL properly and archives
  it per fit.

### A2. Gradient isolation: clean? YES, exactly.

The item table is gathered TWICE per forward: heads gather it in
model/engine code; dynamics gathers it inside each encoder's
_direct_hidden. Detaching the encoder-internal gather gives "heads
live, dynamics detached" with zero new parameters and BIT-IDENTICAL
forward values (torch.equal, no tolerance -- .detach() is a value view
and consumes no RNG). Sites:

- lstm: core/encoder.py:267 (one gather)
- transformer: core/transformer_encoder.py:106 in _tokens (one gather)
- dkvmn: core/dkvmn_encoder.py:151 AND :153 -- two gathers feeding
  three dynamics paths (addressing query via key_proj, value write via
  value_proj, summary fuse reusing the line-151 keys). DKVMN's Mk is a
  slot memory, not a per-item table, so the addressing query IS
  derived from the item embedding; both gathers must be detached.

Design: flag `detach_dynamics_item_val: bool = False` on
BaseSeqEncoder + a `_val_for_dynamics` helper; three call-site edits;
threaded ModelConfig -> _p2_run_cell._build_engine -> engines ->
_P2Engine (which REBUILDS the model copying resolved attrs -- the flag
must be copied there or it silently drops) -> _make_encoder's common
dict (NOT encoder_kwargs; the lstm branch ignores kw). Causal
single-shift is applied downstream of the detach and is untouched.

GUARD: under SK the value table's only gradient IS the dynamics path,
so detach+SK freezes the table at init. The flag raises when combined
with item_key_dim.

NRM: the routed arm1r head reads the same shared embedding via a
head-side gather (unaffected); its own theta.detach() routing composes
independently. NRM routing stays untouched, as required.

### A3. Matched EdNet-250 2PL: obstacles

One hard blocker, three mandatory steps:

- BLOCKER: run_fold_hard attaches the routed NRM head UNCONDITIONALLY
  (_p2_realstudy_hardnrm.py:147) and raises for non-NRM decoders. The
  2PL driver neutralizes HEAD_ATTACH via the same monkeypatch pattern
  the drivers already use, and overwrites the three NRM provenance row
  fields.
- STEP 0 (mandatory): freeze results/p2_nrm250/bank.npz = (items,
  resp option codes, correct_opt) built ONCE on this machine, assert
  key == bank_manifest correct_opt, then point the NRM/MML/2PL
  consumers at it (additive load-frozen path; rebuild stays the
  fallback). This closes the argsort-tie portability risk and makes
  "same histories" a checkable object instead of a determinism
  argument.
- Folds match automatically: fold assignment is a pure function of
  (N, data_seed) (default_rng(42+seed*100).permutation(N), split 5);
  init_seed likewise decoder-independent. The correspondence test
  asserts binary == (resp == correct_opt[items]) at every valid
  position, train/val index equality against the NRM units' persisted
  traj npz, and key identity.
- Rows via the hard harness carry NLL + full alpha/beta vectors
  (nrm_metrics acc dict), unlike the plain realstudy path. Caveat
  disclosed: its auc is macro one-vs-rest, not the old realstudy
  binary auc key.
- MML 2PL reference on the exact bank: defensible and cheap -- reuse
  the first-attempt dedup convention + _p2_mml_real.R itemtype 2PL +
  the split-half (seed 21) self-agreement ceiling. 3 R fits, minutes.

### A4. GPCM recovery semantics (Task 5 resolution)

- Historical headline values: SORTED (verdict certain, code path
  cited). They are preserved untouched.
- Raw-step recovery for the NEW grid: the new driver records BOTH raw
  and sorted per fit (plus within-item step statistics separated from
  the flattened correlation, mirroring the misspec driver), so the
  sorted-vs-raw comparison report covers all 27 synthetic cells
  without touching any historical store.
- Historical raw evidence that already exists and enters the report:
  the misspec battery (beta_raw in rows) and the TIMSS rawbeta rerun
  (ordered fraction .43 raw vs 1.00 sorted).
- Flags for the report: the flattened (Q,K-1) correlation mixes
  between-item location with within-item spread; the global (not
  per-item) sign alignment is a nonstandard identification handle;
  the state-conditioned recovery path always sorts.

## B. Experiment matrix

| store | grid | fits | compute |
|---|---|---|---|
| results/p2_gradiso | 3 enc x 3 dec x 3 paths (SH, SH-isolated, SK) x 5 seeds x 5 folds, N2000/Q200 static, one driver capturing acc+auc/qwk+NLL, raw+sorted beta, theta (Spearman+Pearson), full raw vectors npz | 675 | cluster autopilot + local 4060; DKVMN-NRM cells heaviest; est. 25-40 GPU-h total, ~1 day wall split |
| outputs-style narrowkey fill -> results/p2_narrowkey_fill | {transformer, dkvmn} x {2pl, gpcm} x SK key=16, N2000/Q200, 25 fits | 100 | local overnight or 2 cluster jobs |
| results/p2_ednet250_2pl | 3 enc x {SH, SK} x 25 on the frozen bank (binary responses) | 150 | split local+cluster; est. 8-15 GPU-h |
| MML 2PL reference (full + halves) | 3 R fits | 0 GPU | CPU, minutes |
| cross-format analyses (T4) | none | 0 | CPU |
| sorted-vs-raw report (T5) | none (new grid + existing raw stores) | 0 | CPU |

Total new training: 925 fits. SH/SK synthetic cells are RERUN inside
p2_gradiso rather than reused because paired per-fit NLL and raw-beta
capture do not exist in the historical stores and cannot be rescored
(no weights, no probs); historical stores remain the record for the
published numbers and are never overwritten.

## C. Files

Create:
- bench/_p2_gradiso.py (grid driver; extends the toggle-cell pattern
  with the third path, NLL for all decoders, raw+sorted beta capture)
- bench/_p2_ednet250_bank.py (one-time freeze + loader used by all
  matched consumers; additive)
- bench/_p2_ednet250_2pl.py (matched 2PL driver + HEAD_ATTACH
  neutralization + provenance-corrected rows)
- bench/_p2_mml_ednet250_2pl.py (2PL reference + split-half ceiling)
- bench/_p2_crossformat.py (T4: kappa correspondence per seed for
  SH/SK; H_q model-implied correctness on a common reference grid,
  with the caveat that H_q approximates marginal correctness both
  models fit, so near-ceiling agreement is expected and stated)
- bench/_p2_beta_sortraw_report.py (T5 comparison report)
- tests/test_detach_dynamics.py (5 assertions: exact forward
  invariance incl. loss; parameter identity; all-other-grads
  bit-identical + item-table grad differs + equals manual reference;
  NRM-decoupled zero-grad leak check catching a missed dkvmn gather;
  causal alignment invariant)
- tests/test_matched_bank_correspondence.py

Modify (additive, defaults preserve behavior):
- core/encoder.py, core/transformer_encoder.py, core/dkvmn_encoder.py
  (flag + helper + gather call sites)
- core/model.py (kwarg -> _make_encoder common dict)
- bench/_p2_config.py, bench/_p2_run_cell.py, bench/engines.py,
  bench/_p2_engine.py (thread the flag; the _P2Engine rebuild copy is
  the silent-drop trap), cell dict + fold row record the path
- CLAUDE.md: the stated test path kt-irt/src/deep_irt/tests/ is empty;
  the real suite is kt-irt/tests/ (stale doc fix)

## D. Execution order

1. Core flag + tests; run test_detach_dynamics on all 3 encoders x 3
   decoders locally (minutes).
2. Freeze the ednet250 bank + correspondence test.
3. Launch p2_gradiso (cluster autopilot + local split) and the
   narrowkey fill.
4. Launch p2_ednet250_2pl + the MML 2PL reference.
5. T4 + T5 analyses when fits land; paired seed-level t (df=4)
   throughout; no dataset-clustered bootstrap.

Model economy for the build: core detach + engine threading + both
test files on the strong model; driver clones and runners delegated;
report scripts delegated.

## E. Corrections to the drafted change plan (from the audit)

1. Reusing existing SH/SK synthetic cells for the 3x3x3 grid is not
   possible for the NLL endpoint (never recorded for 2pl/gpcm; no
   weights; no probs) -- the grid reruns all three paths under one
   driver. Historical stores untouched.
2. Task 5's "rescore without refitting" is impossible for the
   historical synthetic campaigns (sorted-only arrays); the new grid
   supplies the raw-step statistics, and the report covers historical
   raw evidence where it exists (misspec battery, TIMSS rawbeta).
3. detach + SK is a degenerate combination (freezes the item table);
   the flag guards against it rather than allowing a fourth condition
   by accident.
4. The matched 2PL run must go through the HARD harness (for NLL and
   traj persistence), with the NRM head-attach neutralized -- not
   through the plain realstudy fold runner.
5. Disclosed provenance hole: the P1 NLL exhibit is not reproducible
   from the tree; p2_gradiso's per-fit NLL supersedes it with an
   archived trail.
