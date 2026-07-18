# kt-mirt vendoring report

Date: 2026-07-17. Source: `kt-irt` @ `df3aee1fcfaff20f8ba7784c59853ec4ab528696`
(`git -C kt-irt rev-parse HEAD`). Vendored tree: `kt-mirt/src/kt_mirt/`.
`kt-irt/` itself was only read, never modified.

## 1. Source study

Read `kt-irt/pyproject.toml` and all 10 files in `kt-irt/src/deep_irt/core/`
(`__init__.py`, `model.py`, `encoder.py`, `transformer_encoder.py`,
`dkvmn_encoder.py`, `decoders.py`, `decoders_ext.py`, `losses.py`,
`anchor.py`, `realdata.py`) plus all 13 files in `kt-irt/tests/` (+
`conftest.py`). Grepped every file in `core/` for `^(import|from)\s+deep_irt`
and for `pandas|scipy|sklearn|matplotlib|yaml` imports.

**Finding: `core/` is already fully self-contained.** Every cross-module
import inside `core/` targets another `deep_irt.core.*` file; there are zero
imports of `deep_irt.bench.*` or `deep_irt.pipeline.*` anywhere in `core/`.
No surgery (feature cuts) was needed on any core file -- only mechanical
import-path rewrites.

**Finding: the real runtime dependency footprint is smaller than
`kt-irt/pyproject.toml` declares.** kt-irt's `[project.dependencies]` lists
numpy, scipy, torch, pandas, scikit-learn, matplotlib, pyyaml -- but those
cover the whole `deep_irt` tree (`bench/`, `pipeline/`, etc.), not `core/`
alone. Grepping `core/` turned up only `numpy` (used in `model.py` for
`_recover_item_params_state_alpha`'s occurrence-averaging, and imported but
apparently unused in `anchor.py`) and `torch`. No file in `core/` imports
pandas, scipy, sklearn, matplotlib, or yaml.

## 2. File manifest

### Package skeleton (new)
- `kt-mirt/pyproject.toml` -- name `kt-mirt`, version `0.0.1`, src layout,
  package `kt_mirt`, setuptools build backend (mirrors kt-irt's pattern).
- `kt-mirt/src/kt_mirt/__init__.py` -- `__version__ = "0.0.1"`.
- `kt-mirt/README.md`, `kt-mirt/.gitignore`.
- `kt-mirt/tests/__init__.py`, `kt-mirt/tests/conftest.py` (src/ path
  bootstrap, adapted from kt-irt's).

### Vendored core (10 files, `kt-irt/src/deep_irt/core/*.py` ->
`kt-mirt/src/kt_mirt/core/*.py`)

| file | cross-module imports rewritten | other imports (unchanged) |
|---|---|---|
| `__init__.py` | 5 lines: `encoder`, `decoders`, `anchor`, `model`, `realdata` | -- |
| `model.py` | 4 top-level lines (`encoder`, `decoders`, `anchor`, `losses`) + 2 lazy imports inside `_make_encoder` (`transformer_encoder.TransformerEncoder`, `dkvmn_encoder.DKVMNEncoder`) | `numpy`, `torch`, stdlib |
| `encoder.py` | none (no cross-module imports) | `torch` |
| `transformer_encoder.py` | 1 line (`encoder.BaseSeqEncoder`) | `torch` |
| `dkvmn_encoder.py` | 1 line (`encoder.BaseSeqEncoder`) | `torch` |
| `decoders.py` | none | `torch` |
| `decoders_ext.py` | none | `torch`, `math` |
| `losses.py` | none | `torch`, `math` |
| `anchor.py` | 2 lines (`encoder.LSTMEncoder`, `decoders.{GPCMDecoder,Binary2PLDecoder}`) | `numpy` (imported, unused), `torch` |
| `realdata.py` | none | `torch` |

Every rewrite is the mechanical pattern `deep_irt.core.X` -> `kt_mirt.core.X`.
No file needed a vendored copy of anything outside `core/`, and no feature
was cut from any core file.

**Docstring/comment prose deliberately left unchanged.** Several files carry
historical design-rationale prose that says "`deep_irt`" (e.g. `losses.py`'s
"Copied verbatim from `ma-irt/utils/losses.py` ... so `deep_irt` stays
self-contained", `model.py`'s opening line `"""deep_irt.py -- Top-level
API..."""`, `realdata.py`'s references to "deep_irt tensors"). These are
left byte-identical to source on purpose: vendoring rewrites imports, not
prose, and touching it risked introducing meaning drift for zero functional
gain. The two exceptions are `tests/test_decoders_ext.py`'s docstring "Run:"
lines, which are actionable usage instructions (not design rationale) and
were updated to the new path; see below.

### Tests (`kt-mirt/tests/`)

Ported from `kt-irt/tests/` (imports rewritten `deep_irt.core` ->
`kt_mirt.core`):
- `test_swap_encoders.py`, `test_masking_and_bridge.py`,
  `test_causal_alignment.py`, `test_decoupled_alpha.py`, `test_alpha_link.py`
  -- copied verbatim apart from import lines.
- `test_decoders_ext.py` -- import rewrite + the two docstring "Run:" lines
  updated to the new path (`kt-mirt/tests/...` instead of
  `deep_irt/tests/...`).
- `test_nrm_decoder.py` -- import rewrite, **plus one surgery**: cut
  `test_cross_format_transfer` and its `nrmfmt`-on-`sys.path` setup. That
  test depended on a `data_gen`/`train`/`metrics` comparison harness living
  in a sibling `nrmfmt/` directory that is not part of `deep_irt.core` and
  was never vendored (the source test itself already guarded it with
  `pytest.importorskip`, i.e. it was already known-optional infra). Rather
  than carry a permanently-skipping test and an unused `sys.path` hack, it
  was dropped; a comment in the file records why.

Skipped (not ported -- all depend on `deep_irt.bench.*`, which is out of
scope for a core-only vendor):
- `test_alpha_map_geometry.py` (`deep_irt.bench.analyze_alpha_map_geometry`)
- `test_direct_alpha_geometry.py` (`deep_irt.bench.datagen`,
  `deep_irt.bench.run_direct_alpha_geometry`)
- `test_alpha_map_bench.py` (`deep_irt.bench.run_alpha_map_bench`) -- its
  core-relevant substance (the named alpha-map formulas on `GPCMDecoder`)
  is independently covered by the ported
  `test_alpha_link.py::test_named_alpha_maps_are_exact`, so no coverage was
  actually lost by skipping this file.
- `test_neural_map_isolation.py` (`deep_irt.bench.run_neural_map_isolation`)
- `test_alpha_residual_null.py` (`deep_irt.bench.run_alpha_residual_null`,
  `deep_irt.bench.datagen`)
- `test_misspecification_probe.py` (`deep_irt.bench.datagen`,
  `deep_irt.bench.run_misspecification_probe`)

New (written for this vendor -- the source suite exercises `losses.py` and
`anchor.py` only *indirectly* through `DeepIRTModel.fit`/`.extend`, and
never unit-tests `BradleyTerryDecoder` or GPCM/Binary log-probability
normalization at the decoder level at all; all three gaps are things the
task explicitly asked the ported suite to cover):
- `test_losses.py` (10 tests) -- `compute_class_weights` (uniform/balanced/
  sqrt_balanced/zero-count floor), `WeightedOrdinalLoss` (matches weighted
  CE at zero ordinal penalty, penalizes far misses more, sum-vs-mean
  reduction, gradient flow), `CombinedLoss` (matches `WeightedOrdinalLoss`
  at weight=1, collapses to plain CE at weight=0 and never builds the
  ordinal submodule).
- `test_anchor.py` (8 tests) -- `anchored_extend` (output shapes, freezes
  encoder+decoder, touches only the new embedding rows), `build_extended_
  encoder` (shape/weight-copy correctness, and a behavioral check that the
  extended encoder reproduces the base encoder's theta exactly on
  base-item-only sequences), plus end-to-end `DeepIRTModel.extend()`
  (recovers `a_ext`/`b_ext`, `track(use_extended=True)` works) and its two
  guard rails (`NotImplementedError` for non-LSTM backbones and for the
  `bt` decoder).
- `test_decoders_core.py` (12 tests) -- `GPCMDecoder` (log-probs sum to 1,
  `psi_0==0` anchor, sorted-beta contract, alpha strictly positive, NLL
  gradient flow, K=2 structure) and `Binary2PLDecoder` (log-probs sum to 1,
  `binary_logit` matches the log-prob difference and its sigmoid matches
  P(y=1), NLL matches `F.binary_cross_entropy_with_logits`), and
  `BradleyTerryDecoder` (strength shape, `nll_pairs` gradient flow, and a
  sanity check that a confident-correct pairwise prediction costs less than
  a confident-wrong one).

One test-authoring bug was found and fixed during the "iterate until green"
step: `test_weighted_ordinal_loss_zero_penalty_matches_weighted_ce`'s first
draft compared against `F.cross_entropy(..., weight=w, reduction="mean")`,
but `WeightedOrdinalLoss`'s `"mean"` reduction is a **plain** per-sample
mean of the already class-weighted per-sample CE terms, not PyTorch's own
weighted-mean convention (which normalizes by the sum of the sample weights
actually used, not by N). The two conventions diverge whenever class
weights are non-uniform. This is a fact about `WeightedOrdinalLoss`'s
existing (unchanged) semantics, not a defect introduced by vendoring; the
test was corrected to reduce `"none"` and average by hand, with a comment
recording the convention for future readers.

## 3. Dependencies

`kt-mirt/pyproject.toml`:
- Runtime (`[project.dependencies]`): `numpy>=1.24`, `torch>=2.1` -- the
  only two third-party imports anywhere in `core/`.
- Dev/test (`[project.optional-dependencies].dev`): `pytest>=7.4`,
  `scipy>=1.11` -- scipy is needed only by the ported
  `test_decoders_ext.py` and `test_nrm_decoder.py`, which use
  `scipy.stats.spearmanr`/`pearsonr` for synthetic-recovery Spearman
  checks. No `core/` module uses scipy; it is test-only.

Deliberately **not** carried over from kt-irt's runtime deps: `scipy`
(test-only here, see above), `pandas` (not imported anywhere in `core/`,
including `realdata.py`), `scikit-learn`, `matplotlib`, `pyyaml` (none used
in `core/`; all are `bench`/`pipeline` concerns).

## 4. Install and test results

```
source ~/anaconda3/etc/profile.d/conda.sh && conda activate research
export KMP_DUPLICATE_LIB_OK=TRUE
pip install -e kt-mirt          # numpy/torch already satisfied in `research`
python -m pytest kt-mirt/tests -q
```

Result: **139 passed, 0 skipped, 0 failed** (also re-ran clean under
`-W error::DeprecationWarning`, no warnings surfaced). Per-file counts:

| file | tests |
|---|---|
| `test_alpha_link.py` | 21 |
| `test_anchor.py` (new) | 8 |
| `test_causal_alignment.py` | 12 |
| `test_decoders_core.py` (new) | 12 |
| `test_decoders_ext.py` | 14 |
| `test_decoupled_alpha.py` | 22 |
| `test_losses.py` (new) | 10 |
| `test_masking_and_bridge.py` | 15 |
| `test_nrm_decoder.py` | 5 |
| `test_swap_encoders.py` | 20 |
| **total** | **139** |

CPU-only throughout (`device=torch.device("cpu")` in every test); nothing
required CUDA, so nothing needed a skip-with-reason marker.

## 5. Interface summary (for the design phase)

Shape conventions used everywhere: `B`/`N` = batch of learners, `T` =
sequence length (interactions per learner), `K` = `n_cats` (response
categories -- **one global K for the whole model instance**, not per item),
`Q`/`num_items` = item-bank size, `emb_dim` = thin per-item/response
embedding width (drives theta), `hidden_dim` = backbone hidden/state width,
`item_key_dim` = optional wide per-item embedding width feeding *only* the
decoder's static alpha/beta heads (never the backbone input).

### `DeepIRTModel` (`kt_mirt.core.model`)
High-level API: one encoder + one decoder + one prediction loss.
```
DeepIRTModel(num_items, emb_dim=8, hidden_dim=32, n_cats=4,
             decoder="gpcm"|"binary"|"bt"|"nrm", correct_option=0,
             state_alpha=None, item_key_dim=None, alpha_log_scale=None,
             alpha_pos_map=None, alpha_pos_kwargs=None, state_beta=False,
             decouple=True, encoder="lstm"|"transformer"|"dkvmn",
             encoder_kwargs=None, device=cpu, seed=0,
             ordinal_penalty=0.5, class_weight_strategy="sqrt_balanced")
```
- `fit(item_ids (N,T), responses (N,T), n_epochs=300, lr=1e-2, mask=None
  (N,T) bool, batch_size=None, callback=None, grad_clip_norm=None) ->
  {"final_loss", "final_nll", "train_time"}`.
- `fit_pairs(item_emb_i, item_emb_j, outcome, ...)` -- `bt` only.
- `track(item_ids, responses, use_extended=False) -> theta (N,T)`.
- `recover_item_params(item_ids=None, responses=None, use_extended=False)
  -> dict`, shape depends on decoder (`{"alpha":(Q,), "beta":(Q,K-1)}` for
  gpcm/binary, `+"seen":(Q,) bool` in state_alpha mode;
  `{"alpha":(Q,K),"intercept":(Q,K)}` for nrm; `{"strength":(Q,)}` for bt).
- `extend(n_ext, anchor_theta, ext_item_ids, ext_responses, ...) -> dict`
  (LSTM-only; raises `NotImplementedError` for other backbones or `bt`).
- **Item bank is a single flat namespace of size `num_items`; there is no
  concept/KC axis anywhere in this class today.**

### Encoders (`kt_mirt.core.encoder`, `.transformer_encoder`,
`.dkvmn_encoder`)
`BaseSeqEncoder` owns the shared IRT head; each backbone implements one
method:
```
_direct_hidden(item_ids (B,T), responses (B,T)) -> h (B,T,hidden_dim)
```
Public accessors (identical across backbones -- this is the swappability
contract):
- `encode(item_ids, responses) -> theta (B,T)` -- raw, responsive.
- `aligned_theta_and_state(item_ids, responses) -> (theta (B,T), state
  (B,T,hidden_dim))` -- single-shift causal alignment (theta/state at step
  t is a function of history strictly before t).
- `theta_for_prediction` / `state_for_prediction` -- the two halves of the
  above; `state_for_prediction` is the **natural attachment point for any
  new per-step readout head** (it is already causal, item-blind for the
  current step, and is exactly what `state_alpha`'s `fc_a_state` head
  reads today).
- `get_final_theta(item_ids, responses) -> theta (B,)`.

Embedding tables (all live on the encoder, shared between encoding and
decoding): `item_val_emb: Embedding(num_items, emb_dim)` (thin, feeds the
backbone/theta), `resp_emb: Embedding(n_cats, emb_dim)`, optional
`item_key_emb: Embedding(num_items, item_key_dim)` (wide, decoder-only,
never touches the backbone -- built only when `item_key_dim` is set).

- `LSTMEncoder(num_items, emb_dim=8, hidden_dim=32, n_cats=4,
  item_key_dim=None)` -- `nn.LSTM(2*emb_dim, hidden_dim)`. Default backbone;
  the only one `anchor.py` supports.
- `TransformerEncoder(..., n_heads=4, n_layers=2, max_seq_len=512,
  dropout=0.0)` -- causal self-attention; token width == `hidden_dim`
  (must be divisible by `n_heads`).
- `DKVMNEncoder(..., memory_size=20, key_dim=None(=emb_dim))` -- key-value
  memory network; per-step read-then-write addressing over `memory_size`
  slots.

**Per-KC attachment point #1 (ability read-out):** `theta_proj:
Linear(hidden_dim, 1)` is the single scalar-ability head. A multi-KC
variant needs either `Linear(hidden_dim, C)` (C concepts) read at every
step and gathered/selected by the current item's KC id, or a separate
per-KC head reading `state_for_prediction` the same way `fc_a_state`
already does -- no backbone change required for the latter.

**Per-KC attachment point #2 (item->KC mapping):** does not exist yet.
Nothing in `core/` has a notion of "concept". A KC id per item would need
either a fixed lookup buffer threaded through encoder+decoder calls, or a
learned `kc_key_emb` table built exactly parallel to `item_key_emb` (wide,
decoder-only, decoupled from the backbone input) if KC identity itself
should be embedded rather than hard-assigned.

### Decoders (`kt_mirt.core.decoders`, `.decoders_ext`)
Shared contract: `item_params(emb, ...) -> dict`, `log_probs(...)` or
`log_density(...)`, `nll(...) -> scalar`.
- `GPCMDecoder(emb_dim, n_cats, state_dim=None, item_key_dim=None,
  alpha_log_scale=None, alpha_pos_map=None, alpha_pos_kwargs=None,
  state_alpha=True, state_beta=False)`:
  - `item_params(emb (...,emb_dim), state=None (...,state_dim),
    item_key=None (...,item_key_dim)) -> {"alpha":(...,1), "beta":
    (...,K-1)}`.
  - `logits(theta (B,)or(B,1), alpha (B,1), beta (B,K-1)) -> (B,K)` via
    `psi_0=0, psi_k=cumsum(alpha*(theta-beta))`.
  - `log_probs(...) -> (B,K)`, `nll(theta (N,), item_val (N,emb_dim),
    responses (N,), state=None, item_key=None) -> scalar`.
  - The alpha positivity map is swappable (`softplus` default, `exp`, or
    13 named maps in `alpha_pos_map`) -- this is the existing precedent
    for "pluggable per-item transform," a useful pattern to reuse for any
    per-KC parameter transform.
- `Binary2PLDecoder(emb_dim, ...)` -- thin `GPCMDecoder(n_cats=2)` wrapper;
  adds `binary_logit(theta, item_val, ...) -> (N,)` for BCE training.
- `BradleyTerryDecoder(emb_dim)` -- `item_strength(item_val) -> (...,1)`;
  `nll_pairs(emb_i, emb_j, outcome, reg_strength=0.01) -> scalar`. Item-only
  (never reads theta at all) -- structurally the odd one out among the four
  decoders.
- `NRMDecoder(emb_dim, n_options, correct_option=0, item_key_dim=None)` --
  `item_params -> {"alpha":(...,K) sum-to-zero per-option slope,
  "intercept":(...,K) sum-to-zero}`; `logits = alpha*theta + intercept`
  (no cumsum -- options are unordered).
- `decoders_ext.py` (vendored but **not** re-exported by `core/__init__.py`,
  same as in kt-irt -- import directly, e.g.
  `from kt_mirt.core.decoders_ext import LogNormalRTDecoder`):
  `LogNormalRTDecoder` (response time; `item_params -> {"beta":(...,1)
  time-intensity, "log_sigma":(...,1)}`), `BetaResponseDecoder` (bounded
  continuous score in (0,1); `item_params -> {"a","b","phi"}`),
  `PoissonCountDecoder` (counts; `item_params -> {"b", "a"?}`). All three
  follow the same `item_params`/`log_density-or-log_probs`/`nll` contract
  as the ordinal decoders, so a per-KC pattern validated on `GPCMDecoder`
  should transfer directly.

**Per-KC attachment point #3 (item parameters):** every decoder's
`item_params` reads only `emb`/`item_key` (both purely item-indexed) plus
an optional `state`. A per-KC difficulty/discrimination structure (e.g. a
shared KC-level prior with an item-level deviation) would concatenate a
gathered KC embedding into the existing `fc_a`/`fc_a_state`/`fc_b` inputs
-- exactly the mechanism `item_key_dim` already uses to decouple alpha/beta
capacity from theta capacity, so it is a proven pattern to extend.

### Losses (`kt_mirt.core.losses`)
- `compute_class_weights(targets, n_classes, strategy="sqrt_balanced"|
  "balanced"|other=uniform, device=None) -> (n_classes,)`.
- `WeightedOrdinalLoss(n_categories, class_weights=None,
  ordinal_penalty=0.5, reduction="mean"|"sum")`: `forward(logits (N,K),
  targets (N,)) -> scalar`; CE scaled by `1 + ordinal_penalty *
  |argmax(logits) - target|`. `"mean"` is a **plain** per-sample average
  (divides by N), not PyTorch's own weighted-CE convention (divides by the
  sum of sample weights) -- the two diverge whenever class weights are
  non-uniform; see the test note above.
- `CombinedLoss(n_categories, class_weights=None,
  weighted_ordinal_weight=1.0, ordinal_penalty=0.5)`: `weight>0` ->
  `WeightedOrdinalLoss`; `weight==0` -> plain `nn.CrossEntropyLoss` (this
  is how the `nrm` format gets a no-ordinal-penalty loss).
- **Per-KC attachment point #4 (loss):** `CombinedLoss` is response-format
  keyed only (ordinal vs nominal), with no notion of per-KC weighting.
  `compute_class_weights` already generalizes to "weights per label
  distribution" -- a per-KC variant would call it once per KC's response
  histogram instead of once globally.

### Anchoring (`kt_mirt.core.anchor`)
- `anchored_extend(encoder, decoder, anchor_theta (n_anchor,),
  ext_item_ids (n_anchor,E) local {0..E-1}, ext_responses (n_anchor,E),
  n_epochs=300, lr=1e-2, device=cpu) -> {"ext_emb_weight":(E,emb_dim),
  "n_params", "train_time", "final_nll"}`. Freezes encoder+decoder, trains
  only E new `item_val_emb` rows against `decoder.nll(...)` with anchor
  theta held fixed. Decoder-agnostic (anything exposing `.nll(theta_flat,
  emb_flat, resp_flat)`); `bt` is excluded (uses `nll_pairs` instead, not
  wired into this function).
- `build_extended_encoder(base_encoder, ext_emb_weight (E,emb_dim),
  device) -> LSTMEncoder(num_items=B+E)`, weights copied and frozen.
  **Hardcoded to `LSTMEncoder`** -- it does not dispatch on backbone type,
  which is exactly why `DeepIRTModel.extend()` raises `NotImplementedError`
  for `transformer`/`dkvmn`. A backbone-agnostic version (or
  `build_extended_transformer_encoder`/`..._dkvmn_encoder` siblings) is a
  prerequisite if per-KC work needs item-bank extension on a non-LSTM
  backbone.
- **Per-KC attachment point #5 (anchoring):** only the ITEM axis is
  extensible today; there is no "extend the KC/concept set" primitive. If
  per-KC structure introduces its own learnable table (KC embeddings, or a
  per-KC alpha/beta prior), anchoring that table needs a new, parallel
  mechanism to this one -- nothing here generalizes for free.

### `realdata` bridge (`kt_mirt.core.realdata`)
`collate_adapter_items(items: list[dict]|adapter, pad_id=0) ->
CollatedBatch(item_ids (N,T_max) long 0-based, responses (N,T_max) long,
mask (N,T_max) bool, student_ids (N,), seq_lens (N,))`. Adapter dict
contract: `{"student_id": int, "questions": 1-D LongTensor (1-based),
"responses": 1-D LongTensor}`. **No KC field anywhere in this contract.**
Multi-KC data would need a parallel `"kc_ids"` field (same shape as
`"questions"`, one KC id per interaction, or a list of ids if items map to
multiple KCs) threaded through padding/masking exactly like `item_ids` is.

No filesystem paths are hardcoded anywhere in the vendored core; every
loader that accepts data takes it as tensors/dicts already in memory.
