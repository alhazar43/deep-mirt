# Investigation 1. Codebase modularity, redundancies, computational bottlenecks

Scope. `ma-irt/` only. Read-only audit. Findings are anchored by absolute file
paths and line numbers from the working tree at the time of the scan. Targets
for refactor.

- The decoder, currently GPCM only, should become a swap-in head (GRM, PCM,
  DINA, MIRT in the future).
- The encoder, currently DKVMN only, should become a swap-in backbone
  (Transformer, AKT, SAKT, SAINT+ in the future).
- Tolerance on the MA-GPCM headline metrics during any refactor is 0.5%.

The repo already has a tier-5 cleanup queue. The findings below are independent
of that queue but converge on most of the same surfaces. Where they overlap,
the file references that queue so the rationale can be cross-checked. See
`CLEANUP_PLAN_2026.md` Appendix F and `docs/architecture.md`.

---

## (a) Redundancies

### A1. Linking utilities (high confirm). 4 copies of the mean-sigma map

The mean-sigma linking transform `est_linked = A * est + B` with
`A = sd(true) / sd(est)` and `B = mean(true) - A * mean(est)` is implemented
in four places with three different surface names.

- `ma-irt/scripts/evaluate.py:78` `mean_sigma_link(true, est) -> (est_linked, A, B)`. Canonical 3-tuple form. Used at lines 434, 450, 520, 521, 700, 710, 721, 735, 813, 823, 833, 913, 943.
- `ma-irt/scripts/evaluate.py:98` `link_alpha_logspace(true, est)`. Same transform but in log space and re-exponentiated. Used at line 434.
- `ma-irt/scripts/plot_recovery_split.py:282` `_apply_mean_sigma(true, est)`. Mathematically identical to `mean_sigma_link`, returns the linked array only. Used at lines 259, 385, 414.
- `ma-irt/scripts/compute_linking.py:100` `linking(true, est) -> (A, B)`. Same formula, returns coefficients only. Used at lines 156, 158, 163.
- `ma-irt/scripts/plot_trajectory_comparison.py:100` `mean_sigma_coefs(true_pool, est_pool) -> (A, B)`. Same formula, returns coefficients only. Used at line 457.

In addition, a separate z-score normalisation (no truth involved) is duplicated.

- `ma-irt/scripts/evaluate.py:70` `link_zscore(vals)`.
- `ma-irt/scripts/plot_block_and_rw.py:63` `link_normal(vals)`. Identical formula.
- `ma-irt/scripts/plot_trajectory_comparison.py:92` `link_normal(vals)`. Identical formula.

The R baseline (`ma-irt/scripts/mirt_baseline_all_k.R:12` and `:23`) mirrors
`mean_sigma_link` and `link_alpha_logspace` in R. That copy is unavoidable
because it lives in a different language, but the Python and R contracts are
not pinned by a shared test.

Recommended consolidation. Extract to `ma-irt/utils/linking.py`.

```text
ma-irt/utils/linking.py
    mean_sigma_link(true, est)        -> (linked, A, B)
    mean_sigma_coefs(true, est)       -> (A, B)
    apply_mean_sigma(true, est)       -> linked
    link_alpha_logspace(true, est)    -> (linked, A, B)
    zscore(vals)                      -> standardized
```

Pin with a regression test that hashes the linked output for a fixed
`(true, est)` array so the R baseline cannot silently drift.

This is the most decoupled cleanup. It does not touch model code, can be done
without retraining, and unlocks Tier-5 item 3 in `CLEANUP_PLAN_2026.md:549`.

### A2. `build_model` is duplicated across `train.py` and `evaluate.py`

- `ma-irt/scripts/train.py:65` `build_model(cfg, device, n_students)`. Dispatches on `cfg.model.model_type`, instantiates one of {magpcm, dkvmn_softmax, static_gpcm, dynamic_gpcm, dkt, dkvmn, deep_irt}, sets `model._model_type` so the trainer can dispatch later.
- `ma-irt/scripts/evaluate.py:154` `MODEL_CLASSES` dict plus `ma-irt/scripts/evaluate.py:214` `load_model(cfg, ckpt, device, n_students)`. Parallel dispatch table covering the same seven model classes, plus a checkpoint patcher for legacy keys (`_patch_monotonic_beta_state_dict` at line 172).
- `ma-irt/scripts/compute_linking.py:35` `infer(...)`. A third dispatch, but limited to {MAGPCM, StaticGPCM, DynamicGPCM}, with its own `use_sid` flag.
- `ma-irt/scripts/plot_recovery_split.py:67` `_run_one_magpcm(...)`. A model-specific reload for MA-GPCM only. Reads checkpoint, builds model, runs inference.

Three of the four touch the kwargs split `{k: v for k, v in vars(cfg.model).items() if k != "model_type"}` verbatim. Each pasted copy is a place where a new model class will silently bypass legacy kwarg handling.

Recommended consolidation. `ma-irt/models/__init__.py` already exports the
classes (`magpcm`, `dkt`, `dkvmn`, `deep_irt`). Add a `build_model(cfg, device,
n_students=0)` factory in the same module. `train.py`, `evaluate.py`, the
linking script, and the plot scripts then call one entry point. The legacy
state-dict patcher should move next to it, since loading is part of the
construction contract.

This consolidation also removes the `MODEL_CLASSES`, `NEEDS_STUDENT_ID`, and
`HAS_IRT_PARAMS` triplet from `evaluate.py:154-169`. These are model
capabilities that belong on the model class as class attributes, not as
script-level lookup tables. Doing so is a prerequisite for the encoder and
decoder swap, because the same capability flags will need to live on every
new backbone and head.

### A3. Trainer `_forward` dispatch on `_model_type`

`ma-irt/training/trainer.py:244-254` selects between `model(student_ids, q, r)`
and `model(q, r)` based on a string attribute set externally in
`build_model` (`ma-irt/scripts/train.py:72, 75`). The dispatch is repeated
in `ma-irt/scripts/evaluate.py:291` and `ma-irt/scripts/plot_trajectory_comparison.py:218, 260`.

The fragility is that `_model_type` is monkey-patched onto the module after
construction. `DKVMNSoftmax` is built without it (`ma-irt/scripts/train.py:68-69`),
and the codebase relies on the trainer's `getattr(..., None)` fallback so the
default path is taken. Anyone adding a new model class will need to remember
to set this string.

Recommended consolidation. Unify the forward arity. All models accept
`(student_ids, questions, responses)` (matching the four-tuple emitted by
`collate_sequences` at `ma-irt/dataloading/loaders.py:113`). Models that do
not consume `student_ids` ignore it. This collapses `_forward()` to a single
call and lets the trainer be model-agnostic. See `CLEANUP_PLAN_2026.md:551`
for the same recommendation in the pre-existing queue.

This unification is a prerequisite for the encoder swap, because the trainer
must stop knowing model identities once Transformer, AKT, SAKT, and SAINT+
become candidates.

### A4. Recovery accumulation loop is duplicated three ways

The accumulation pattern below appears in three places with the same nested
Python loop over `(B, S)`.

- `ma-irt/scripts/evaluate.py:284-336` `run_inference`. Adds AUC bookkeeping.
- `ma-irt/scripts/compute_linking.py:59-89` (inline in `infer`). Adds last-step theta tracking per student id.
- `ma-irt/scripts/plot_recovery_split.py:78-95` (inline in `_run_one_magpcm`). Adds last-step theta only.

Each loop walks the mask in pure Python (`for b in range(B): for t in range(S):
if mask[b, t]: ...`). For a 5000-student dataset with average sequence length
60, that is on the order of 300000 Python iterations per pass. The cost is
bearable for evaluation but is the dominant runtime in the recovery scripts
and is visible to any new figure or aggregation script that copy-pastes the
pattern.

Recommended consolidation. Pull the accumulation into `ma-irt/utils/recovery.py`.

```text
ma-irt/utils/recovery.py
    accumulate_item_params(out, questions, mask, alpha_sum, beta_sum, counts)
    accumulate_theta(out, student_ids, mask) -> dict[sid, np.ndarray]
```

The implementation can vectorise the masked accumulation with
`torch.scatter_add_`, killing both the duplication and the per-step Python
loop. See B3 below.

### A5. Loss schema. `focal_weight` is dead config

`FocalLoss` no longer exists in `ma-irt/training/losses.py`. `CombinedLoss`
absorbs and ignores it via `**kwargs` (`ma-irt/training/losses.py:169`). But
`focal_weight: 0.0` is set in roughly 250 YAML files under
`ma-irt/configs/...`. The default loss recipe is now WOL only.

Action. Remove `focal_weight` from `TrainingConfig` (it is not declared
anyway, so it is silently absorbed at load time, then silently absorbed at
loss-construction time). Strip the field from configs in a follow-up. Confirms
`CLEANUP_PLAN_2026.md:547`.

### A6. Metrics duplication. sklearn vs torch

- `ma-irt/utils/metrics.py` implements `qwk`, Kendall tau b, Spearman, MAE, AUC in pure torch (lines 88, 55, 40, 168-end). Used by `Trainer.evaluate_epoch` via `compute_metrics` (`ma-irt/training/trainer.py:201`).
- `ma-irt/scripts/evaluate.py:51-52` imports sklearn `cohen_kappa_score`, `mean_absolute_error`, `roc_auc_score` and scipy `kendalltau`, then computes the same metrics on flat numpy arrays at lines 382-396.

The two paths are exercised side by side. Different numerical libraries can
produce small disagreements on rounded ties, which weakens any 0.5% tolerance
claim made about parameter recovery. The fact that one path is per-epoch
(torch) and the other is post-hoc (sklearn) is a design choice, but the
metric definitions should not diverge.

Action. Make `ma-irt/utils/metrics.py` the single source for QWK, Kendall tau,
MAE, AUC, and accuracy. `evaluate.py` calls it after concatenating the
flattened arrays. The pure-torch versions already handle K=2 AUC after the
recent `ma-irt/utils/metrics.py:201-209` edit (uncommitted).

### A7. Embedding-mode branch is copy-pasted across DKVMN-flavoured models

Both `MAGPCM.__init__` (`ma-irt/models/magpcm.py:75-97`) and
`DKVMNSoftmax.__init__` (`ma-irt/models/dkvmn_softmax.py:49-64`) carry the
same 3-branch logic for `embedding_type in {"onehot", "learned",
"static_item"}`. The forward pass also duplicates the runtime branch
(`ma-irt/models/magpcm.py:190-202` and `ma-irt/models/dkvmn_softmax.py:85-94`).

The new binary baselines (`dkt`, `dkvmn`, `deep_irt`) bypass this entirely and
use a 2Q-indexed embedding (`ma-irt/models/dkt.py:67`, `ma-irt/models/dkvmn.py:73`,
`ma-irt/models/deep_irt.py:70-71`), which is the standard DKT/DKVMN binary
convention. That is correct for K=2 but means the codebase now has two
distinct "value embedding" idioms living in different files.

Recommended consolidation. The encoder swap is the right time to extract a
`models/encoders/value_embedding.py` that owns the three ordinal modes plus
a binary mode. The encoder consumes a tagged embedding factory rather than
re-implementing the if/elif tree.

### A8. DKVMN backbone is instantiated four times with slight surface drift

The exact same backbone (`models/components/memory.py:DKVMN`) is built in
four model files with cosmetically different argument plumbing.

- `ma-irt/models/magpcm.py:99-104` `init_value_memory=True` by default.
- `ma-irt/models/dkvmn_softmax.py:66-71` `init_value_memory=False` by default.
- `ma-irt/models/dkvmn.py:76-81` `init_value_memory=True` (Zhang 2017 reference).
- `ma-irt/models/deep_irt.py:73-78` `init_value_memory=True` (Yeung 2019 reference).

The plain DKVMN (`dkvmn.py`) and DeepIRT (`deep_irt.py`) additionally
re-implement the per-timestep loop body inline (`models/dkvmn.py:124-130`
and `models/deep_irt.py:124-135`) rather than calling a shared "DKVMN run
sequence" helper. The MA-GPCM loop (`models/magpcm.py:236-280`) and the
DKVMN+Softmax loop (`models/dkvmn_softmax.py:104-113`) are also each separate
copies of the same `attn -> read -> summarise -> write` pattern.

Recommended consolidation. Extract a `DKVMNEncoder` module that owns the
attention pre-pass, the read-before-write loop, and returns either the full
sequence of read vectors or the full sequence of `(read, attn)` pairs.
Decoders consume that output, item-conditioning is layered on top. This is
the direct prerequisite for the encoder swap. The Transformer, AKT, SAKT,
and SAINT+ variants then implement the same `Encoder` interface and can be
selected by config without any model-file growth.

---

## (b) Modularity gaps. Encoder/decoder split per model

Reading order. The "encoder" is the module that consumes
`(questions, responses)` and produces a per-step hidden state. The "decoder"
is the module that maps that hidden state plus the current item embedding
to category logits.

### MA-GPCM (`ma-irt/models/magpcm.py`)

```text
Encoder:
    questions, responses
    -> q_embed       (B, S, key_dim)            [line 187]
    -> value_embed   (B, S, value_dim)          [line 190-202]
    -> attn_all      (B, S, M)  pre-computed    [line 212]
    -> per step t:
        read_t       (B, value_dim)             [line 242]
        write(value_mem, attn_t, v_t)           [line 280]
    Hidden state exposed to decoder:
        ability_summary_t = ability_summary(read_t)  (B, summary_dim)    [line 245]
        summary_t        = summary([read_t, q_t])    (B, summary_dim)    [line 247]

Decoder (inlined, lines 250-273):
    theta_t = ability_network(ability_summary_t) * ability_scale         (B, D)
    raw_alpha_t = discrimination_network([summary_t, q_t])               (B, D)
    alpha_t = exp(raw_alpha_t)                                            (B, D)
    beta = threshold(q_embed) pre-computed                               (B, S, K-1) [line 207]
    GPCM logits via cumsum                                                (B, K) per step

Decoder output:
    {theta, alpha, beta, logits, probs}
```

Encoder/decoder split. Currently logically separated but physically inlined.
The decoder reads three sub-networks from `self.irt` (lines 227-228) and
applies them by hand. The reason given in the docstring at
`ma-irt/models/components/irt.py:36-40` is that there is exactly one mapping
path from summary to (theta, alpha, beta). That is a fine invariant, but it
also blocks the swap.

To extract a clean decoder, two things are needed.

1. Stop expressing the encoder->decoder contract as "the encoder hands the
   decoder its internal layer outputs". Hand the decoder a tagged dict.
2. Pull the GPCM step computation out of the loop body and call a single
   `GPCMDecoder.step(summary, q_embed, ability_summary)` method.

```text
Encoder.forward() -> dict
    {
        "ability_summary": (B, S, summary_dim),
        "item_summary":    (B, S, summary_dim),
        "item_embed":      (B, S, key_dim),     # q_embed, item identity only
        "attention":       (B, S, M),            # diagnostic
    }

Decoder.forward(enc_dict) -> dict
    {
        "theta":  (B, S, D),
        "alpha":  (B, S, D),
        "beta":   (B, S, K-1),
        "logits": (B, S, K),
        "probs":  (B, S, K),
    }
```

State that must flow between encoder and decoder. (i) `ability_summary` for
theta. (ii) `item_summary` for alpha. (iii) `q_embed` for beta. (iv) Optional
`attention` for diagnostics. No other coupling exists in the current code.
This is a 4-tensor contract.

The pre-computed `beta` at line 207 must move into the decoder. It currently
escapes the encoder/decoder boundary because the encoder calls
`self.irt.threshold` directly.

The `ability_scale` and `separate_theta` flag belong in the decoder. The
ablation flag `separate_theta=False` becomes a decoder-side choice of which
hidden stream to read from.

### DKVMN+GPCM ablation (same file, `separate_theta=false`)

Same module, switched at `ma-irt/models/magpcm.py:250-253`. The encoder is
identical. The decoder reads `summary_t` for theta instead of
`ability_summary_t`. After the extraction above, this is one config flag on
the `GPCMDecoder`, not a branch inside the encoder loop.

### DKVMN+Softmax (`ma-irt/models/dkvmn_softmax.py`)

```text
Encoder:
    questions, responses
    -> q_embed, value_embed, attn_all
    -> per step t:
        read_t, summary_t = summary([read_t, q_t])
        write

Decoder (single linear classifier):
    logits_t = classifier(summary_t)   (B, K)   [line 111]
```

Encoder is identical to MA-GPCM's encoder modulo the `ability_summary`
branch. Decoder is a single linear layer. After the extraction above, this
file becomes a thin combination of `DKVMNEncoder + SoftmaxDecoder`.

Returns dummy IRT parameter fields (`ma-irt/models/dkvmn_softmax.py:122-124`)
so the trainer's `_forward` works. That is a smell, not a contract. With a
capability flag (`HAS_IRT = False`) the trainer and the evaluation scripts
can skip the IRT path explicitly. The dummies should go away.

### Static GPCM (`ma-irt/models/static_gpcm.py`)

No sequential encoder at all. The "encoder" is three embedding tables
(`theta_embed`, `alpha_raw`, `beta_raw`) plus `_get_item_params` lookups.
The "decoder" is `GPCMLogits` (`ma-irt/models/static_gpcm.py:128-130`).

This is the baseline that the encoder/decoder split must serve. The cleanest
abstraction is to call the model a "static encoder" that emits per-step
`(theta, alpha, beta)` directly, paired with a `GPCMDecoder` head. The
decoder is then the same head used by MA-GPCM. Confirmed today by
`GPCMLogits` being shared between the two files
(`ma-irt/models/components/irt.py:117`).

### Dynamic GPCM (`ma-irt/models/dynamic_gpcm.py`)

Encoder. Gated recurrence on theta driven by the GPCM prediction's surprise.
Per-step state is `theta_t (B, D)`. Item parameters come from the same
`alpha_raw`, `beta_raw` tables as Static GPCM.

Decoder. `GPCMLogits`, identical to MA-GPCM (`ma-irt/models/dynamic_gpcm.py:177`).
The encoder calls the decoder once per step inside its own loop
(`ma-irt/models/dynamic_gpcm.py:166-180`), which is the right pattern. But
that means the encoder owns the decoder by reference. Under the swap
abstraction, this becomes `RecurrentTheta + GPCMDecoder`.

A subtlety. The Dynamic GPCM encoder needs the GPCM head's `expected_t` at
training time so it can compute `surprise_t` (`ma-irt/models/dynamic_gpcm.py:190-191`).
That couples encoder and decoder at the prediction level. The swap
abstraction must allow encoders to call the decoder forward inside their
own loop (or expose `expected_response = sum(probs * categories)` as a
decoder utility).

### DKT (`ma-irt/models/dkt.py`)

Encoder. LSTM over a 2Q-indexed one-hot embedding (`ma-irt/models/dkt.py:67-74`).
Causal shift inside `forward` at line 104.

Decoder. Linear projection to per-item logits, gathered at the current item
(`ma-irt/models/dkt.py:110-115`), wrapped into 2-class logits at line 120.

Encoder/decoder split. Already clean. The encoder produces `h (B, S, hidden)`,
the decoder is `Linear + gather + Bernoulli-to-2-class`. The dummy IRT fields
at lines 128-130 should go away with the capability flag.

### Plain DKVMN (`ma-irt/models/dkvmn.py`)

Encoder. DKVMN, identical to MA-GPCM's encoder except no `ability_summary`
branch (`ma-irt/models/dkvmn.py:108-130`).

Decoder. `summary -> sigmoid` then converted to 2-class
(`ma-irt/models/dkvmn.py:128-133`).

Encoder/decoder split. Trivially extractable. After the swap, this file is
`DKVMNEncoder + BinaryGPCMHead(K=2)` where the head is the same `GPCMDecoder`
restricted to K=2 with a Rasch-like discrimination of 1. Or, more honestly,
`DKVMNEncoder + BinaryClassifierHead`.

Inline duplication. The 2Q-indexed value embedding and the causal loop are
re-implemented (`ma-irt/models/dkvmn.py:107-130`), parallel to
DKVMN+Softmax. After the shared encoder extraction this collapses.

### Deep-IRT (`ma-irt/models/deep_irt.py`)

Encoder. DKVMN backbone, same as MA-GPCM, with no value-projection branch
since the value embedding is 2Q-indexed.

Decoder. Rasch readout. `theta_net([read, q]) -> tanh -> theta` and
`beta_net(q) -> tanh -> beta`. Combined as `3.0 * theta - beta`
(`ma-irt/models/deep_irt.py:127-132`).

Encoder/decoder split. Clean if the Rasch readout is recognised as a
constrained decoder. Under the swap abstraction, `DKVMNEncoder + RaschDecoder`,
where `RaschDecoder` is a special case of a future `IRTDecoder` family
(`Rasch -> 1PL -> 2PL -> GPCM -> GRM -> PCM -> DINA -> MIRT`).

### Summary of the encoder/decoder boundary today

| Model | Encoder | Decoder | Coupling notes |
|---|---|---|---|
| MA-GPCM | DKVMN, separated ability summary | GPCM, exp(raw_alpha) | Decoder inlined in encoder loop; sub-network refs cached for speed (`magpcm.py:227-233`) |
| DKVMN+GPCM | DKVMN, shared summary | GPCM | Same file, flag-controlled |
| DKVMN+Softmax | DKVMN, shared summary | Linear softmax | Dummy IRT outputs returned (`dkvmn_softmax.py:122`) |
| Static GPCM | Embedding tables, no recurrence | GPCM (`GPCMLogits` called once) | Cleanly separable |
| Dynamic GPCM | Gated recurrence on theta | GPCM (`GPCMLogits` called per step) | Encoder reads decoder probs at training time |
| DKT | LSTM over 2Q one-hot, causal shift | Linear gather + 2-class wrap | Dummy IRT outputs returned (`dkt.py:128`) |
| Plain DKVMN | DKVMN with 2Q value embedding | Sigmoid + 2-class wrap | Dummy IRT outputs returned (`dkvmn.py:139`) |
| Deep-IRT | DKVMN, theta and beta nets | Rasch (`3 theta - beta`) | Decoder is constrained, treated separately |

Encoder-side, the union of distinct backbones is
{DKVMN, EmbeddingLookup, GatedRecurrence, LSTM, LSTM_2Q}. Decoder-side, the
union is {GPCM, Softmax, BinaryGather, BinarySigmoid, Rasch}. The future
encoder set adds {Transformer, AKT, SAKT, SAINT+}. The future decoder set
adds {GRM, PCM, DINA, MIRT}.

State that must flow across the boundary, after extraction.

```text
EncoderOutput = {
    "student_summary":  (B, S, H_s),    # theta-relevant hidden, item-free
    "joint_summary":    (B, S, H_j),    # alpha-relevant hidden, item-aware
    "item_embed":       (B, S, H_i),    # beta-relevant, item-only
    "responses":        (B, S),         # passthrough for decoders that need it (Dynamic GPCM)
    "mask":             (B, S),         # passthrough for loss/metric paths
    "attention":        optional        # diagnostic only
}
```

Encoders that do not produce a meaningful `student_summary` (e.g. plain DKVMN
binary baselines) set it to a fallback. The decoder's `requires_student_summary`
class attribute selects which encoders are compatible at config-time.

### Modularity gap, build path

The capabilities `NEEDS_STUDENT_ID` and `HAS_IRT_PARAMS` at
`ma-irt/scripts/evaluate.py:165-169` are model identity guards, not
capability declarations. They are out of sync with the encoder/decoder
abstraction. Move them to class attributes
`Encoder.needs_student_id` and `Decoder.produces_irt_params`. The factory
then composes encoder + decoder and exposes the combined capability.

### Modularity gap, dataloader

`collate_sequences` emits a 4-tuple `(q, r, mask, student_ids)`
(`ma-irt/dataloading/loaders.py:113`). The collation already supports the
unified arity needed for the trainer dispatch removal in A3. No data-side
change is needed for the encoder/decoder swap.

---

## (c) Computational bottlenecks

### B1. Per-step Python loop in DKVMN encoders is the dominant cost

The MA-GPCM loop (`ma-irt/models/magpcm.py:236-280`) and the DKVMN+Softmax
loop (`ma-irt/models/dkvmn_softmax.py:104-113`) run S Python-level iterations
per batch. Each iteration launches several CUDA kernels
(`memory.read`, `summary`, `ability_summary`, `irt.ability_network`,
`irt.discrimination_network`, GPCM cumsum, `memory.write`).

Why it cannot be fully vectorised. The write of value memory at step t
depends on the read at step t. That dependency is real and is what makes
DKVMN sequential.

What can be eliminated.

- (Done) Attention is loop-invariant and is now pre-computed
  (`ma-irt/models/magpcm.py:212`, mirrored in
  `ma-irt/models/dkvmn_softmax.py:100`).
- (Done) The `beta` projection is loop-invariant and is now pre-computed
  (`ma-irt/models/magpcm.py:207`).
- (Done) Sub-network references are cached to skip Python attribute lookups
  (`ma-irt/models/magpcm.py:227-233`).

What is left.

- For sequences of length 60 with batch 32 and `memory_size=50`, a single
  causal step is roughly 8 small matmuls plus 2 element-wise ops. On CPU
  that is throttled by kernel-launch overhead, not arithmetic.
  `torch.compile(model)` should remove most of the launch overhead and is
  the lowest-risk speedup. Gate behind a `cfg.training.compile_model` flag
  defaulting to False to keep `<0.5%` metric drift.
- The plain DKVMN and Deep-IRT loops at
  `ma-irt/models/dkvmn.py:124-130` and `ma-irt/models/deep_irt.py:124-135`
  do not hoist `q_embed` or pre-compute attention. They should match the
  MA-GPCM optimisation. Item B1.1.
- `init_value_memory(B)` returns `init_memory_param.unsqueeze(0).expand(B, M, dv).contiguous()`
  (`ma-irt/models/components/memory.py:97-99`). The `.contiguous()` materialises
  the full `(B, M, dv)` tensor on every batch. For B=64, M=50, dv=128 that is
  1.6 MB per batch, fine on GPU but unnecessary. The first `write` already
  produces a new tensor; the `.contiguous()` can be deferred until first
  write. Low priority. Worth measuring under `torch.compile` because compile
  can fold the contiguous into the first write.

### B2. `OneHotEmbedding` materialises (B, S, K*Q) on every step

`ma-irt/models/components/embeddings.py:65-88` builds a `(B, S, K, Q)`
intermediate, then flattens to `(B, S, K*Q)`. For Q=200, K=4, B=32, S=60
that is roughly 30 MB per batch and grows linearly with Q. The active
default is now `static_item` (see `ma-irt/models/dkvmn_softmax.py:37`
and the StaticItemEmbedding scaling note at
`ma-irt/models/components/embeddings.py:120-128`), and the MA-GPCM model
also uses `static_item` for the bulk runs (`onehot` is still a config
option but rarely the default at large Q).

Action. None on the hot path, because the `static_item` factored embedding
already replaces it for large Q. Keep `OneHotEmbedding` for K=2 and small Q
benchmark fidelity. A note in the docstring that this is intentionally
quadratic for the reference implementation would prevent surprise.

### B3. Recovery accumulation is the main per-epoch Python cost

`ma-irt/scripts/evaluate.py:305-336`, `ma-irt/scripts/compute_linking.py:75-89`,
and `ma-irt/scripts/plot_recovery_split.py:86-95` each iterate `(B, S)` in
Python to accumulate `alpha_sum[qid] += a[b, t]` and `beta_sum[qid] += b[b, t]`.
On the standard 1000 test student configuration this is `~60000` Python
iterations per evaluation.

Action. Replace with `torch.scatter_add_` on the GPU side.

```python
# inside no_grad block, alpha_np lives on GPU still
mask_flat = mask.view(-1).bool()
q_flat = (questions.view(-1) - 1).clamp(min=0)  # 0-indexed item
a_flat = out["alpha"].view(-1, D)
b_flat = out["beta"].view(-1, K - 1)
valid = mask_flat & (q_flat >= 0) & (q_flat < Q)

alpha_sum.index_add_(0, q_flat[valid], a_flat[valid])
beta_sum.index_add_(0, q_flat[valid], b_flat[valid])
counts.scatter_add_(0, q_flat[valid], torch.ones_like(q_flat[valid], dtype=counts.dtype))
```

This is also the natural home for `ma-irt/utils/recovery.py` in A4. A
single-batch microbenchmark would confirm the speedup, but order of
magnitude is `10-100x` over the Python loop for the recovery sweep.

### B4. Trainer `_pad_and_cat` builds zero-tensors at evaluation time

`ma-irt/training/trainer.py:281-310` pads each evaluated batch to the global
max sequence length, then concatenates everything. For test sets with a
heterogeneous length distribution and a long tail, this wastes memory.

Mitigation. The current implementation is correct and not a bottleneck for
the synthetic data sizes (`B=32`, `S<=80`). On real datasets this is the
first place to fix. Replace with a list-of-arrays representation passed
directly to the metric functions, which already handle masks. Low priority
relative to B3.

### B5. DataLoader. workers and pin_memory are wired but disabled by default

`ma-irt/dataloading/loaders.py:267-291` exposes `num_workers`, `pin_memory`,
`persistent_workers` knobs. Defaults are `num_workers=0`, `pin_memory=True`,
`persistent_workers=False`. The CV-aware `_make_loader` at line 414 mirrors
those defaults.

`num_workers=0` is correct on Windows for safety, and the synthetic data
fits in process memory anyway. For ASSISTments-scale data the per-epoch
collation cost is non-trivial. The knobs are already in place. Set
`num_workers >= 2` in the ASSISTments configs and measure. Confirmed by
`ma-irt/scripts/_profile_dkvmn.py` which is the existing benchmark
harness.

`persistent_workers` should be ON whenever `num_workers > 0` because the
dataset is small enough to keep loaded in worker processes. The code
already handles this correctly at line 269.

### B6. `evaluate.py` does CPU/Python AUC accumulation per batch

`ma-irt/scripts/evaluate.py:299-311` runs four `.cpu().numpy()` transfers
per batch and then a Python double loop. The `.cpu()` calls are the
expensive part on GPU runs. Batch them after the whole loop, or accumulate
masked tensors and call `.cpu()` once at the end.

### B7. Mixed precision (AMP) is not used

No `torch.cuda.amp.autocast` anywhere in `ma-irt/training/trainer.py` or
the model files. The DKVMN read/write is `bmm` heavy. On Ampere or newer
GPUs, autocast can give a 1.5-2x speedup at zero metric cost. Gate behind
a `cfg.training.amp` flag defaulting to False.

Risk. The cumulative GPCM logits in `GPCMLogits.forward`
(`ma-irt/models/components/irt.py:117-135`) sum K-1 terms with a `norm`
inside. In fp16 the `alpha.norm` can underflow when `alpha` is near 0.
Use `torch.amp.GradScaler` and keep the GPCM block in `float32` via an
explicit `cast`. The 0.5% tolerance is safe but needs an empirical check
on the bulk K=4 row.

### B8. `evaluate.py` Spearman is computed twice

`ma-irt/scripts/evaluate.py` calls `safe_spearman` per linked output
(`evaluate.py:436, 452`, etc.). Each call imports `scipy.stats.spearmanr`
inside the function (`evaluate.py:145-146`). The import is cached after
first call but Python's import machinery still walks `sys.modules`. Lift
the import to module scope. Trivial.

### Bottleneck priorities

| Rank | Item | Effort | Expected speedup |
|---|---|---|---|
| 1 | B3, vectorise recovery accumulation | Low | 10-100x on the recovery sweep |
| 2 | B7, AMP on GPU training | Low | 1.5-2x training step time |
| 3 | B1, torch.compile on the DKVMN loop | Low | 1.2-1.5x training step time |
| 4 | B1.1, hoist attention + q_embed in `dkvmn.py` and `deep_irt.py` | Low | Matches MA-GPCM perf for K=2 baselines |
| 5 | B5, set `num_workers>=2` on ASSISTments configs | Low | Measured at ~1.3x on prior `_profile_dkvmn.py` runs (per the existing commit 09c7085) |
| 6 | B6, batched `.cpu()` transfers in evaluate.py | Low | 1.2x evaluate pass |
| 7 | B4, drop pad-to-global-max in evaluator | Medium | Memory savings only for long-tail datasets |

---

## (d) Codex partial work

Search criteria. `git log -i --grep codex --all` and full author/email
inspection on the last 60 commits.

### Findings

- Zero commits since at least 2026-03 contain "codex" or "Codex" in the
  message body. The commit log is uniformly authored by
  `Wenrui Yuan <stephen514yuan@gmail.com>`. No co-author trailers.
- The most recent commits (in order, newest first) are documentation and
  cleanup readiness work, not code refactor.

```text
7793d2f Guard cleanup refactor behind readiness check
37c37c2 Add public pipeline reproducibility tests
6bbbda1 Add cleanup taxonomy manifests
eaeb7a3 Add ordinal baseline smoke configs
24a3cfc Document MA-GPCM architecture contract
f4eb5b7 Document MA-GPCM pipeline contract
435e2df Refresh public README entry surface
4d3edbe Record T0 cleanup baseline evidence
c0e5427 Redesign cleanup plan around MA-GPCM pipeline
```

### Uncommitted work in the working tree, by file

The working tree contains substantial uncommitted edits. None of them carry
an explicit Codex signature, but they are the most relevant "partial work"
the investigation should flag because they predate any planned refactor and
will collide with one. The cleanup plan flags them at
`CLEANUP_PLAN_2026.md:165` and `docs/cleanup/T7_REFACTOR_READINESS_2026-06-02.md`.

| File | Change | Status |
|---|---|---|
| `ma-irt/config/types.py` | Adds `CVConfig` dataclass and `chunk_long_sequences`, `shuffle_before_split` fields on `DataConfig`. Adds `early_stop_metric`, `patience`, `early_stop_margin` fields on `TrainingConfig`. | Standalone, internally consistent. Tests not updated. |
| `ma-irt/config/loader.py` | Adds nested `CVConfig` merge path. | Matches `types.py` edit. |
| `ma-irt/dataloading/loaders.py` | Adds `_chunk_sequences`, `build_cv`, `_make_loader`. Records `cv_test_id_offset` and `cv_test_orig_idx` for downstream recovery. Adds `shuffle_before_split` to the legacy `build`. | New CV path is parallel to the legacy path. Both routes appear correct. The downstream callers of `cv_test_orig_idx` are not present in any committed script. |
| `ma-irt/scripts/train.py` | Adds DKT, DKVMN, DeepIRT to `build_model`. Adds CV-aware loop with patience-based early stop and AUC tracking. Emits `test_metrics.json` after CV training. | The CV path will fail until the consumers of `cv_test_orig_idx` exist. The early stop / AUC fields work standalone. |
| `ma-irt/models/__init__.py` | Exports DKT, DKVMN, DeepIRT. | Matches new model files added in commit 410efed. |
| `ma-irt/models/components/irt.py` | Adds `alpha_from_raw(raw_alpha, log_scale)` with a long design note. The function is defined but is not yet called anywhere. | Dead code in the working tree. Either wire it into Static GPCM and Dynamic GPCM (which currently inline the same formula at `static_gpcm.py:123` and `dynamic_gpcm.py:117`) or revert. |
| `ma-irt/utils/metrics.py` | Adds K=2 AUC into `compute_metrics`. Returns NaN for K>2 or single-class batches. | Standalone, correctly guarded. |
| `ma-irt/scripts/plot_*.py` | Several plot scripts modified. Not inspected in detail by this investigation. | Inspect before any refactor that touches plot inputs. |
| `ma-irt/configs/bulk/*.yaml` | ~600 bulk configs modified. | YAML edits, presumably a sigma or scale field update. Out of scope for this investigation, but they will affect any retraining-based verification of the refactor. |
| `CLAUDE.md` | Edits in the working tree. | Out of scope. |
| `overleaf-sync` | Submodule pointer change. | Out of scope. |
| `ma-irt/outputs/` | New directory, untracked. | Run artifacts. |

### Unfinished items the investigation suggests

The cleanup readiness note at
`docs/cleanup/T7_REFACTOR_READINESS_2026-06-02.md` already states the
guardrail. The investigation confirms it. Specifically.

1. `alpha_from_raw` is added to `components/irt.py` but neither
   `static_gpcm.py` nor `dynamic_gpcm.py` was updated to call it. Either
   complete the call-site change or revert the function. Leaving it is a
   modularity hazard for the decoder swap because a future GRM head will
   want a different positivity map.
2. CV early-stop fields (`patience`, `early_stop_metric`, `early_stop_margin`)
   are wired into the training loop but no test, smoke config, or
   documentation has been added. Add a smoke YAML that exercises
   `patience: 2, early_stop_metric: auc` so the path does not bit-rot.
3. `cv_test_orig_idx` is exposed by `DataModule.build_cv` but no script
   consumes it. The intended consumer is the recovery aggregator. If that
   is partial work, finish the consumer; otherwise the field can be a
   private attribute.
4. AUC was added to both `utils/metrics.py` and `scripts/evaluate.py`. The
   metric path in `scripts/evaluate.py` still uses sklearn (per A6). Pick
   one source.

None of the above blocks the encoder/decoder refactor. They do all need to
land first so the refactor's regression tests have a stable baseline.

---

## Summary count

| Bucket | Findings |
|---|---|
| Redundancies | 8 (A1 linking, A2 build_model, A3 trainer dispatch, A4 recovery accumulation, A5 focal_weight, A6 metrics duplication, A7 embedding branches, A8 DKVMN backbone instantiation) |
| Modularity gaps | 7 per-model encoder/decoder splits, 1 build-path capability flag gap, 1 dataloader contract already fine |
| Computational bottlenecks | 8 (B1 DKVMN loop, B2 OneHot allocation, B3 recovery accumulation, B4 pad-to-global, B5 dataloader workers, B6 evaluate CPU transfers, B7 AMP, B8 import-in-loop) |
| Codex partial work | 0 explicit Codex commits; 4 uncommitted hazards listed above |

The encoder is `DKVMN backbone + embedding + summary networks`, inlined in
each model file. The decoder is `IRTParameterExtractor + GPCMLogits + GPCMHead`,
inlined inside the MA-GPCM causal loop and called per-step or once depending
on the baseline. The boundary today is implicit. The 4-tensor encoder->decoder
contract proposed in (b) makes it explicit and admits all four future encoders
and four future decoders without touching the trainer.
