# Public-Release Session Status, 2026-06-03

Snapshot of where the ma-irt public-release prep stands at the end of
this session. Detailed phase definitions live in
`docs/cleanup/PIPELINE_OPT_PLAN.md`; this doc tracks which phases are
done and which are still queued.

## Done

### P1, dedup and methodological clean-up
Completed pre-session (see `docs/cleanup/PIPELINE_OPT_PLAN.md` §P1).

### P2, modular interface refactor
Completed pre-session. ABCs at `ma-irt/models/registry.py`; encoders
under `ma-irt/models/encoders/`; decoders under `ma-irt/models/decoders/`.

### P3, backbone integration
Completed in this session, ten committed milestones:

| Milestone | Commit | Scope |
|---|---|---|
| P3.5a | `36d3f4a` | width-match assertion in `check_compatible` |
| P3.5b | `782847b` | config schema for transformer encoder + AdamW path |
| P3.5c | `d6476d3` | implement TransformerEncoder (SAKT-style) + wrapper + 13 tests |
| P3.5c-fix | `4e3b926` | shift value stream right by one (SAKT response-leak fix) |
| P3.5d | `8bffc5d` | transformer recovery parity test + K=4 fold0 bulk config |
| P3.6a | `a4892a9` | LSTMEncoder + LSTMGPCM wrapper + 14 tests |
| P3.6b | `c4dc1de` | LSTM recovery parity test + K=4 fold0 bulk config |
| P3.7a | `938572c` | drop harmful MLP_theta subtraction from non-DKVMN encoders |
| P3.7b | `9603057` | split-residual encoder restores full IRT recovery parity |
| P3.7c | `cc35965` | q-free ability attention as MA-GPCM default |
| P3.7d | `1430df0` | rename separate_attention → ability_query (principled siblings) |
| P3.7e | `4b97b9a` | rename ability_query → item_conditioned (bool flag) |

Headline results:

- Transformer and LSTM encoders join DKVMN under the unified Encoder
  ABC; the same GPCM decoder runs over all three.
- Split-residual encoder (P3.7b) makes the structural separation
  principle universal: transformer and LSTM now match or beat DKVMN on
  every IRT recovery metric (K=4 fold0).
- The `item_conditioned` flag exposes the paper MA-GPCM (True) and the
  P3.7c q-free variant (False) as principled siblings. The two paper
  R2 headlines pin `item_conditioned: true`.
- pytest passes 145/1-skipped throughout.
- R2 invariant on the 6 paper headlines reloads bit-for-bit at every
  milestone.

## Remaining

### P4, decoder family (stretch goal)

Add four IRT-family decoders alongside the existing GPCM:

| Head | Plan reference | Status |
|---|---|---|
| `GRMHead` (graded response) | PIPELINE_OPT_PLAN.md §P4.1 | not started |
| `PCMHead` (partial credit) | §P4.2 | not started |
| `MIRTHead` (multidim 2PL) | §P4.3 | not started |
| `DINAHead` (cognitive diagnosis, separate ABC) | §P4.4 | not started |

Plus a synthetic GRM DGP generator (`scripts/data_gen.py --decoder
grm`) and two test files (`test_decoder_swap.py`,
`test_grm_recovery.py`). Cost: 8–10 agent rounds + 30–60 min GPU per
GRM recovery fold.

Paper baseline uses only GPCM, so P4 is optional for the release.
Reasons to do it: positions the framework as a true decoder-family
host, not just a GPCM tool. Reasons to skip: more code surface than
the paper currently exercises.

### P5, computational optimisations

| Item | Plan reference | Status |
|---|---|---|
| B3 vectorise recovery accumulation (`utils/recovery.py`, `index_add_`) | §P5.1 | not started |
| B5 `num_workers >= 2` + `persistent_workers=True` on ASSISTments | §P5.2 | not started |
| B7 `torch.cuda.amp.autocast` behind `cfg.training.amp` | §P5.3 | not started |
| B1 `torch.compile(model)` behind `cfg.training.compile_model` | §P5.4 | not started |
| B6 lift `.cpu()` out of per-batch eval loop | §P5.5 | not started |

Cost: 4–6 agent rounds, ~1–2 h wall-clock for per-flag training pass.

### P6, packaging polish for public repo

Most of P6 was absorbed by the S1–S11 + restore series and the
P3-rework R14 cleanup. Remaining items:

| Item | Plan reference | Status |
|---|---|---|
| `LICENSE` at repo root (MIT) | §P6.1 | not started |
| `CONTRIBUTING.md` | §P6.2 | not started |
| `.github/workflows/ci.yml` (pytest on push + PR, Ubuntu + Windows) | §P6.3 | not started |
| `docs/encoders.md`, `docs/decoders.md` (after P3 and P4 land) | §P6.4 | partial (P3 done, P4 pending) |
| Move legacy top-level dirs into `legacy/` (T2 in INVESTIGATION_GIT_STATUS) | §P6.5 | not started |

Cost: 3–4 agent rounds + 1 CI iteration cycle for Windows-specific
issues.

## Multi-seed CV for P3.7 claims (not yet a phase)

The P3.7b (split-residual) and P3.7c (q-free attention) results were
measured on one fold and one seed each. The scientist's earlier
analysis recommended 5-fold × 3-seed CV before any external claim of
dominance. This is research validation rather than release blocker.

## Recommended next steps

1. **Push current state to origin/main**, 12 commits queued (P3.5a
   through P3.7e). The repo is at a clean stopping point.
2. **Tag `v0.1-paper`** at the current HEAD as a reproducibility
   anchor (the scientist's earlier suggestion). Cheap insurance for
   "the state of the codebase at paper submission".
3. Decide on P4. If skipping, move directly to P5 (computational
   optimisations) since they have no architectural dependency on P4.
4. P6 packaging at the end, includes CI workflow that exercises the
   final architectural surface.

## File reference

- Master plan: `docs/cleanup/PIPELINE_OPT_PLAN.md`
- Architecture: `docs/cleanup/DESIGN_MODULAR.md`,
  `docs/cleanup/DESIGN_BACKBONES.md`
- Codebase audit: `docs/cleanup/INVESTIGATION_CODEBASE.md`
- SOTA references: `docs/cleanup/INVESTIGATION_SOTA.md`
- Baseline numbers (paper R2): `docs/cleanup/BASELINE_2026-06-02.md`
