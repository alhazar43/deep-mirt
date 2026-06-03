# Pipeline optimisation plan, 2026-06-03

Synthesised from `INVESTIGATION_CODEBASE.md`, `INVESTIGATION_SOTA.md`,
`INVESTIGATION_GIT_STATUS.md`, and `BASELINE_2026-06-02.md`. Six phases,
P1 through P6, lowest risk first. Each phase declares scope, files
touched (inventory only), verification command and acceptance threshold,
rollback strategy, and estimated cost in agent rounds plus verification
wall-clock.

The hill to die on is MA-GPCM K=4 Synthetic-Static recovery (Table 3) and
ASSIST2009 binary K=2 (Table 2). Both pass at the 0.5% gate today from
cached sidecars. Tolerance is `effective_tol = max(0.005 * |published
mean|, published_sigma)` per the baseline document. A >0.5% drift on any
of the 34 currently-passing cells triggers a full lookback.

R2 invariant. Any phase that touches model code (P2, P3, P4, P5) starts
with a test that asserts the cached baseline numbers are reproducible
from a single fold, before any refactor begins.

---

## P1, dedup and methodological clean-up

Lowest risk. No model surgery, no retraining. Targets findings A1, A2,
A5, A6, B8 in `INVESTIGATION_CODEBASE.md`.

### Scope
1. Extract `ma-irt/utils/linking.py` from `compute_linking.py`,
   `evaluate.py`, `plot_recovery_split.py`, `plot_trajectory_comparison.py`,
   `plot_block_and_rw.py`. Five surface names collapse to four
   functions (`mean_sigma_link`, `mean_sigma_coefs`, `apply_mean_sigma`,
   `link_alpha_logspace`, `zscore`). Pin with a fixed-input regression
   hash so R and Python cannot drift.
2. Move `build_model(cfg, device, n_students)` to
   `ma-irt/models/__init__.py`. Replaces the three parallel dispatch
   tables in `scripts/train.py:65`, `scripts/evaluate.py:154-214`,
   `scripts/compute_linking.py:35`, and the model-specific path in
   `scripts/plot_recovery_split.py:67`. Move the
   `_patch_monotonic_beta_state_dict` legacy loader next to it.
3. Unify metric source. Make `utils/metrics.py` the single source for
   QWK, Kendall tau, MAE, AUC, accuracy. `scripts/evaluate.py` calls
   it after concat. Removes the sklearn vs torch divergence (A6).
4. Audit `focal_weight: 0.0` across `configs/bulk/*.yaml`. If every
   active config is 0.0, drop the field from the schema and strip
   from configs in a follow-up commit. `FocalLoss` already does not
   exist in `training/losses.py`.
5. Lift the `scipy.stats.spearmanr` import out of the per-batch
   function in `scripts/evaluate.py:145`. Trivial.
6. README sync. The recent `435e2df` rewrite is current. Re-check
   `CLAUDE.md` against the new factory location.

### Files touched (inventory)
- New, `ma-irt/utils/linking.py`, `ma-irt/tests/test_linking.py`.
- Edited, `ma-irt/models/__init__.py`, `ma-irt/scripts/train.py`,
  `ma-irt/scripts/evaluate.py`, `ma-irt/scripts/compute_linking.py`,
  `ma-irt/scripts/plot_recovery_split.py`,
  `ma-irt/scripts/plot_trajectory_comparison.py`,
  `ma-irt/scripts/plot_block_and_rw.py`, `ma-irt/config/types.py`
  (drop `focal_weight` if audit confirms), `ma-irt/configs/bulk/*.yaml`
  (strip field, follow-up only).

### Verification
```
cd ma-irt && PYTHONPATH=. KMP_DUPLICATE_LIB_OK=TRUE pytest tests/ -v
cd ma-irt && PYTHONPATH=. python -c "from models import build_model; \
  from config.loader import load_config; \
  cfg = load_config('configs/smoke/magpcm_smoke.yaml'); \
  m = build_model(cfg, 'cpu', n_students=10); \
  import torch; q = torch.randint(1, 11, (2, 5)); r = torch.zeros(2, 5, dtype=torch.long); \
  sid = torch.arange(2); out = m(sid, q, r); \
  print({k: tuple(v.shape) for k, v in out.items() if hasattr(v, 'shape')})"
```
Acceptance, 68 of 68 pytest pass, sanity build under one second per
model type (`magpcm`, `dkvmn_softmax`, `static_gpcm`, `dynamic_gpcm`,
`dkt`, `dkvmn`, `deep_irt`).

### Rollback
Single revert commit per surface (linking, factory, metrics, focal,
imports). All edits are additive or one-to-one substitutions.

### Cost
3 agent rounds. Verification under 30 s wall-clock.

---

## P2, modular interface refactor

Defines `EncoderBackbone` and `ResponseDecoder` ABCs following the py-irt
decorator-on-ABC pattern recommended in `INVESTIGATION_SOTA.md`
section (d). Resolves A3, A4, A7, A8 plus the modularity gap in
section (b).

### Scope
1. Add `ma-irt/models/registry.py` with `EncoderBackbone(nn.Module,
   abc.ABC)` and `ResponseDecoder(nn.Module, abc.ABC)`. Each owns a
   class-level `_registry` dict and a `from_name(name, **cfg)`
   classmethod. The encoder ABC declares
   `forward(batch) -> EncoderOutput`, where `EncoderOutput` is a
   typed dataclass with `student_summary`, `joint_summary`,
   `item_embed`, `responses`, `mask`, `attention`. The decoder ABC
   declares `forward(enc_out) -> DecoderOutput` with `theta`, `alpha`,
   `beta`, `logits`, `probs`. Both ABCs declare class attributes
   `needs_student_id`, `produces_irt_params`, `compatible_with` to
   replace the script-level `NEEDS_STUDENT_ID` and `HAS_IRT_PARAMS`
   guards.
2. Extract `DKVMNEncoder` from `models/magpcm.py`, `dkvmn_softmax.py`,
   `dkvmn.py`, `deep_irt.py`. Owns the attention pre-pass, the
   read-before-write loop, and the value-memory init. Three current
   embedding modes (`onehot`, `learned`, `static_item`) move to
   `models/encoders/value_embedding.py`.
3. Extract `GPCMDecoder` from `models/components/irt.py` plus
   `GPCMLogits`. Owns the `theta`, `alpha`, `beta` extraction plus
   the cumulative logit head. The `separate_theta` flag becomes a
   decoder-side choice of which hidden stream to read.
4. Backward-compat shim. Keep `MAGPCM`, `DynamicGPCM`, `StaticGPCM`,
   `DKVMNSoftmax`, `DKT`, `DKVMN`, `DeepIRT` as thin composition
   wrappers that select the right encoder and decoder by name. Old
   YAML configs continue to load without change. New configs may use
   `model.encoder` and `model.decoder` directly.
5. Unify forward arity, all models accept `(student_ids, q, r)` and
   ignore unused arguments. Removes the `_model_type` monkey patch
   from `scripts/train.py:72-75` and the trainer dispatch at
   `training/trainer.py:244-254`.
6. Lift `alpha_from_raw` from the uncommitted edit in
   `models/components/irt.py` into the new decoder, and call it from
   `StaticGPCM`, `DynamicGPCM`, `MAGPCM`. Resolves T7 readiness item 1.

### Files touched (inventory)
- New, `ma-irt/models/registry.py`, `ma-irt/models/encoders/__init__.py`,
  `ma-irt/models/encoders/dkvmn.py`,
  `ma-irt/models/encoders/value_embedding.py`,
  `ma-irt/models/encoders/static.py`,
  `ma-irt/models/encoders/recurrent_theta.py`,
  `ma-irt/models/encoders/lstm.py`, `ma-irt/models/decoders/__init__.py`,
  `ma-irt/models/decoders/gpcm.py`,
  `ma-irt/models/decoders/softmax.py`,
  `ma-irt/models/decoders/rasch.py`,
  `ma-irt/models/decoders/binary.py`,
  `ma-irt/tests/test_registry.py`,
  `ma-irt/tests/test_interface_compliance.py`,
  `ma-irt/tests/test_baseline_reproduction.py`.
- Edited, `ma-irt/models/magpcm.py`, `dynamic_gpcm.py`, `static_gpcm.py`,
  `dkvmn_softmax.py`, `dkt.py`, `dkvmn.py`, `deep_irt.py`,
  `models/__init__.py`, `training/trainer.py`, `scripts/train.py`,
  `scripts/evaluate.py`, `scripts/compute_linking.py`,
  `config/types.py`.

### Verification (R2 gate)
1. Before any refactor, land `test_baseline_reproduction.py` that
   loads one MA-GPCM K=4 fold0 `best.pt`, runs `scripts/evaluate.py
   single` against `data/v2_q200_k4`, and asserts
   `|observed - cached| <= effective_tol` for QWK, MAE, r_alpha,
   r_beta, r_theta, accuracy.
2. After refactor, the same test must pass without retraining. The
   ASSIST2009 fold0 binary `best.pt` must reproduce AUC and ACC
   inside the published sigma.
3. Interface tests assert every registered encoder and decoder
   satisfies its ABC, that `from_name` lookup works, and that the
   forward signature returns the typed dataclass.
4. `pytest tests/ -v` shows 68+ pass.

Acceptance, 100% of currently-passing cells (34 of 36 from
BASELINE_2026-06-02.md) reproduce within `effective_tol`. Kendall tau
gap is documented as a sidecar definition issue, not a refactor
regression.

### Rollback
Each model file's shim is a single import-and-compose block. Reverting
the shim restores the inlined class. Decoder and encoder modules are
additive. Forward-arity unification is the riskiest sub-step, isolate
it in its own commit so a revert does not undo the registry.

### Cost
8 to 10 agent rounds. Verification 5 to 15 min wall-clock per fold
on CPU, under 1 min on GPU per model type.

---

## P3, backbone integration

Add `SAKTEncoder`, `SAINTPlusEncoder`, `AKTEncoder`,
`SimpleKTEncoder` under the P2 ABC. Sources from
`INVESTIGATION_SOTA.md` section (a), canonical pyKT implementations.

### Scope
1. Port `pykt.models.sakt.SAKT`, `saint_plus_plus.SAINTPlus`,
   `akt.AKT`, `simplekt.SimpleKT` into `ma-irt/models/encoders/`,
   each registered with `@EncoderBackbone.register("sakt")` etc.
   Adapt the forward signature to consume the `EncoderOutput`
   dataclass.
2. AKT's Rasch-style item reparameterisation moves into a
   companion decoder utility, since `INVESTIGATION_SOTA.md` flags it
   as decoder-side. The encoder hands back `d_output` plus the
   item-pid embedding decomposition.
3. SAINT+ stays as dual-stream encoder, both streams folded into
   `EncoderOutput.joint_summary` and `student_summary`.
4. One synthetic K=4 verification config per new backbone, paired
   with the existing `GPCMDecoder`. Small training budget,
   single fold, lower epoch count, just enough to confirm the
   composed model trains and converges.
5. License audit. SAKT, AKT, SimpleKT, SAINT+ pyKT ports are MIT
   licensed. Header attribution in each new file.

### Files touched (inventory)
- New, `ma-irt/models/encoders/sakt.py`,
  `ma-irt/models/encoders/saint_plus.py`,
  `ma-irt/models/encoders/akt.py`,
  `ma-irt/models/encoders/simplekt.py`,
  `ma-irt/configs/smoke/{sakt,saintpp,akt,simplekt}_gpcm_smoke.yaml`,
  `ma-irt/configs/bulk/bench_{sakt,akt,saintpp}_static_q200_k4_pykt_fold0.yaml`,
  `ma-irt/tests/test_encoder_swap.py`.
- Edited, `ma-irt/models/encoders/__init__.py`,
  `ma-irt/models/registry.py` (only if a new ABC method surfaces).

### Verification
```
cd ma-irt && PYTHONPATH=. KMP_DUPLICATE_LIB_OK=TRUE python scripts/train.py \
  --config configs/bulk/bench_sakt_static_q200_k4_pykt_fold0.yaml
cd ma-irt && KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python scripts/evaluate.py single \
  --config configs/bulk/bench_sakt_static_q200_k4_pykt_fold0.yaml \
  --checkpoint outputs/bench_sakt_static_q200_k4_pykt_fold0/best.pt \
  --data-dir data/v2_q200_k4
```
Acceptance, MA-GPCM-on-NewEncoder K=4 fold0 recovery is within
`max(0.05, 5 * effective_tol)` of MA-GPCM-on-DKVMN fold0 on QWK,
r_alpha, r_beta, r_theta. The tolerance is intentionally loose,
SAKT and SAINT+ are not expected to match DKVMN exactly on the
recovery row, but they must be in the same neighbourhood, not random.

### Rollback
Each backbone is an isolated file with one registration line. Revert
by deleting the file and its config. No effect on MA-GPCM-on-DKVMN.

### Cost
6 to 8 agent rounds for port plus smoke, plus one training pass per
backbone (4 backbones x 5 to 30 min GPU per fold0 = 20 to 120 min).

---

## P4, decoder family

Add `GRMHead`, `PCMHead`, `DINAHead`, `MIRTHead` under the P2 ABC.
Decoder family observation from `INVESTIGATION_SOTA.md` section (c),
GPCM/GRM/PCM/NRM share one base, DINA needs a sibling
`CognitiveDiagnosisDecoder` ABC.

### Scope
1. `GRMHead`. Graded response model. Differs from GPCM in the
   cumulative link, sigmoid of `alpha * (theta - beta_k)` directly,
   then differencing to get category probabilities. `n_traits`
   continues to switch single-trait vs MIRT.
2. `PCMHead`. Partial credit model. Rasch-style GPCM where alpha is
   constrained to 1. Implements as `GPCMHead(alpha_mode="fixed_one")`.
3. `MIRTHead`. Multidimensional 2PL, the K=2 special case of GRM with
   vector theta and vector alpha. Verifies the n_traits axis.
4. `DINAHead`. Separate ABC, `CognitiveDiagnosisDecoder`. Consumes
   skill mastery vector and Q-matrix, emits per-category probability.
   The encoder ABC's `EncoderOutput.student_summary` must expose
   skill logits, so add a `skill_summary` optional field to
   `EncoderOutput` rather than overloading `student_summary`. Gated
   by Q-matrix availability in the dataset.
5. Extend `scripts/data_gen.py` with a `--decoder grm` flag to
   produce a synthetic GRM DGP for the head's recovery test. Reuse
   the existing item-parameter sampler.

### Files touched (inventory)
- New, `ma-irt/models/decoders/grm.py`, `pcm.py`, `mirt.py`,
  `ma-irt/models/decoders/dina.py`,
  `ma-irt/models/cdm_registry.py` (separate ABC),
  `ma-irt/scripts/data_gen_grm.py`,
  `ma-irt/configs/smoke/{grm,pcm,mirt,dina}_smoke.yaml`,
  `ma-irt/tests/test_decoder_swap.py`,
  `ma-irt/tests/test_grm_recovery.py`.
- Edited, `ma-irt/models/decoders/__init__.py`,
  `ma-irt/scripts/data_gen.py` (factor out the sampler).

### Verification
1. GPCM head with the new ABC reproduces the cached MA-GPCM K=4
   fold0 baseline within `effective_tol`. This is the regression
   gate from P2 rerun under the new decoder routing.
2. GRM head on a synthetic GRM DGP (K=4, Q=200, 5000 students)
   recovers `r_alpha >= 0.85`, `r_beta >= 0.95`, `r_theta >= 0.94`
   on one fold. These thresholds match the GPCM-on-GPCM baseline
   at K=4 minus 0.02 slack.
3. MIRT head on a synthetic 2-trait GPCM DGP recovers both trait
   correlations above 0.85.
4. DINA head on a synthetic DINA DGP (small Q-matrix, K=2) recovers
   guess and slip parameters within 0.05 MAE.

### Rollback
Each head is one file plus one registration. Revert by deleting.
The GPCM baseline is unaffected because GPCM head sits on the same
ABC.

### Cost
8 to 10 agent rounds, includes the small DGP generator extensions.
GRM DGP training pass 30 to 60 min GPU per fold.

---

## P5, computational optimisations

Targets B1, B3, B5, B6, B7 in `INVESTIGATION_CODEBASE.md`. Strictly
gated behind the P2 baseline-reproduction test.

### Scope
1. B3, vectorise recovery accumulation with `index_add_` in
   `ma-irt/utils/recovery.py`. The new home for `accumulate_item_params`
   and `accumulate_theta`. Replaces three copy-pasted Python
   `(B, S)` loops in `scripts/evaluate.py:284-336`,
   `compute_linking.py:59-89`, `plot_recovery_split.py:78-95`.
2. B5, set `num_workers >= 2` and `persistent_workers=True` in the
   ASSISTments-scale configs. Synthetic configs stay at 0 workers
   for Windows safety.
3. B7, gate `torch.cuda.amp.autocast` behind
   `cfg.training.amp: bool = False`. Keep the GPCM head in float32
   via explicit cast, since the cumulative sum near alpha=0 can
   underflow in fp16.
4. B1, gate `torch.compile(model)` behind
   `cfg.training.compile_model: bool = False`. Pure speedup if it
   reproduces; revert per model if it does not.
5. B6, lift `.cpu()` transfers out of the per-batch loop in
   `scripts/evaluate.py`. One transfer per evaluation pass instead
   of one per batch.
6. B1.1, hoist attention pre-compute and q_embed in the as-yet
   un-optimised `models/dkvmn.py` and `deep_irt.py` loops to match
   the MA-GPCM pattern. This piggy-backs on P2's encoder extraction,
   so by P5 the loop has already moved into `DKVMNEncoder`.

### Files touched (inventory)
- New, `ma-irt/utils/recovery.py`,
  `ma-irt/tests/test_recovery_accumulator.py`,
  `ma-irt/scripts/_profile_pipeline.py` (microbench, not committed).
- Edited, `ma-irt/scripts/evaluate.py`,
  `ma-irt/scripts/compute_linking.py`,
  `ma-irt/scripts/plot_recovery_split.py`,
  `ma-irt/training/trainer.py`, `ma-irt/config/types.py`,
  `ma-irt/configs/bulk/bench_*_assist2009_*.yaml`.

### Verification
1. P2 baseline reproduction test still passes. Same 34 cells, same
   tolerance.
2. Microbench shows the recovery accumulator runs 10x or better
   versus the Python loop on a 1000-student test set.
3. AMP-on and compile-on training of MA-GPCM K=4 fold0 produces a
   metric set within `effective_tol` of the AMP-off, compile-off
   baseline. If not, leave the flag default False with documentation.

### Rollback
Each optimisation is a flag. Default off on the public configs. A
regression revert is a one-line YAML edit.

### Cost
4 to 6 agent rounds. Verification 1 to 2 hours wall-clock if a
training pass per flag is needed.

---

## P6, packaging for public repo

Targets the top-level repo polish requested in the candidate's
directive 5. Independent of model code, can land in parallel with
P3 or P4.

### Scope
1. `LICENSE` at repo root. MIT is the right choice given the pyKT
   ports in P3 carry MIT.
2. `CONTRIBUTING.md` at repo root. Describes the environment setup
   from `CLAUDE.md`, the test command, the encoder and decoder
   registration pattern from P2, and the baseline tolerance gate.
3. `pyproject.toml` at repo root. PEP 621 metadata, declares
   `ma-irt` as the import root, lists runtime deps, declares
   `pytest` as the test runner and `ruff` as the linter. No
   `setup.py`, modern packaging only.
4. `.github/workflows/ci.yml`. Runs `pytest tests/ -v` on push to
   `main` and on PRs. Cache the conda environment, run on Ubuntu
   plus Windows. Add a separate job for the baseline reproduction
   test against a checked-in K=4 fold0 `best.pt` if the checkpoint
   is under the GitHub LFS limit, otherwise gate behind a manual
   trigger.
5. `docs/` directory polish. The existing `docs/pipeline.md` and
   `docs/architecture.md` from `f4eb5b7` and `24a3cfc` move under
   `docs/`. Add `docs/encoders.md` and `docs/decoders.md` after P3
   and P4 land.
6. Repo root layout. Move the top-level legacy directories
   (`akt/`, `deep-1pl/`, `deep-gpcm/`, `dkt-ori/`, `dkvmn-ori/`,
   `mirt-dkvmn/`, `pykt/`, `archive_sigma03_20260422_0534`,
   `_overleaf_old/`, top-level `figures/`) into `legacy/` per
   legacy Tx T2. This is the pending T2 in
   `INVESTIGATION_GIT_STATUS.md` section (b).

### Files touched (inventory)
- New, `LICENSE`, `CONTRIBUTING.md`, `pyproject.toml`,
  `.github/workflows/ci.yml`, `docs/encoders.md`, `docs/decoders.md`.
- Moved, the legacy root directories into `legacy/`. Submodule
  pointers preserved.
- Edited, `README.md` to point at the new `pyproject.toml` install
  instructions.

### Verification
1. `pip install -e .` from the repo root succeeds in a clean conda
   env.
2. `pytest tests/ -v` runs from the repo root via the new
   `pyproject.toml` test configuration, 68+ pass.
3. GitHub Actions CI job runs green on a draft PR.
4. README install instructions reproduce the smoke train command.

### Rollback
Each artifact is a single new file or a single move. Revert by
deletion or `git mv` back. Legacy directory move is the riskiest
sub-step, do it in its own commit so a revert is one operation.

### Cost
3 to 4 agent rounds for the files, plus one CI iteration cycle to
chase Windows-specific issues. Verification 5 to 10 min per CI run.

---

## What I will NOT do this round

1. Retrain the paper's reported numbers. The cached `best.pt` and
   sidecars from `BASELINE_2026-06-02.md` are the source of truth.
   P2, P3, P4, P5 verify against them, they do not regenerate them.
2. Break YAML config backward compatibility. Every existing
   `configs/bulk/*.yaml` and `configs/assistments/*.yaml` continues
   to load and produce the same model. New configs may use the
   `encoder` and `decoder` keys directly, old ones go through the
   shim.
3. Touch the published metrics in the paper or in `overleaf-sync/`.
   The Kendall tau column gap noted in `BASELINE_2026-06-02.md`
   section 5.2 is a sidecar definition issue. P1 fixes the
   computation, the paper number stays.
4. Change `model.separate_theta` semantics. The MA-GPCM ablation
   surface is part of the paper. It becomes a decoder flag in P2
   but it stays addressable from YAML by the same name.
5. Move the R baseline (`scripts/mirt_baseline_all_k.R`) into
   Python. R mirt remains the GPCM EM reference. P1 pins the Python
   linking against a fixed-input hash, the R copy continues to live
   in parallel.
6. Touch the ASSISTments retraining under StaticItem (commit
   `e6b6b9d`). Those configs and outputs are frozen.
7. Delete `ma-irt/outputs/`. Sidecar metrics and checkpoints are
   protected per legacy T8. P6 may add a `.gitignore` rule for
   future runs but does not touch existing artifacts.
8. Add new dependencies beyond what pyKT requires for the SAKT,
   SAINT+, AKT, SimpleKT ports in P3. No `transformers`, no
   `lightning`, no `hydra`. The HuggingFace and Lightning patterns
   noted in `INVESTIGATION_SOTA.md` section (b) are inspiration
   only.
9. Run `git rebase -i`, `git push --force`, or any history rewrite.
   All phases land as forward commits per workflow R1.
10. Modify code or move files in this round. This document is
    inventory and design only.
