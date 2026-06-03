# P2 migration design, EncoderBackbone and ResponseDecoder

Status, design only. No model files are edited in this round. Execution lands
in a follow-up phase where each migration is one commit, gated by the
R2 baseline test in `ma-irt/tests/test_baseline_reproduction.py` and a new
per-model regression-snapshot test.

The spine is `ma-irt/models/registry.py` (commit `8c8ceb7`), which introduces

- `EncoderBackbone(nn.Module, abc.ABC)` with class attribute `needs_student_id`
  and a class-level `_registry` keyed by name.
- `ResponseDecoder(nn.Module, abc.ABC)` with class attribute
  `produces_irt_params` and its own `_registry`.
- `EncoderOutput` and `DecoderOutput` dataclasses carrying
  `student_summary`, `joint_summary`, `item_embed`, `responses`, `mask`,
  `attention`, `extra` for the encoder, and `logits`, `probs`, `theta`,
  `alpha`, `beta`, `skill_profile`, `extra` for the decoder.

The `alpha_from_raw` helper from `ma-irt/models/components/irt.py`
(commit `613b812`) is unused today and must land inside the migrated
`gpcm` decoder.

## Scope, the shared-backbone family only

The unified Encoder + Decoder framework applies to the **shared-backbone
family**. Two production models are explicitly **exempt** from migration
and keep their current self-contained structure.

- `static_gpcm.py` (`StaticGPCM`) is **exempt**. It owns a per-student
  theta embedding table and a per-item parameter table. No DKVMN
  encoder, no sequence encoder. The class is not wrapped under
  `EncoderBackbone`. The class gains a single advertisement attribute,
  `produces_irt_params = True`, so downstream code can route on the same
  flag the decoder family carries. Nothing else changes.
- `dynamic_gpcm.py` (`DynamicGPCM`) is **exempt**. Gated recurrent theta
  with per-item lookup, no shared memory backbone. Same treatment as
  `StaticGPCM`, single `produces_irt_params = True` advertisement.
- `GPCM (EM)` lives in R (`scripts/mirt_baseline_all_k.R`) and is already
  outside the Python framework. No change.

Rationale for the exemption. `StaticGPCM` and `DynamicGPCM` are pure IRT
estimators with per-student and per-item lookup tables. There is no
shared encoder body to migrate, and forcing them under the
`EncoderBackbone` ABC adds a stateless wrapper around the embedding
lookups that does not improve reuse. Their forward signature already
diverges from the shared family (they take `student_ids`), and the
trainer dispatch already special-cases them. Migration of these two
files is **not** a goal of P2 and should not be attempted by future
contributors. If a future model genuinely shares the per-item table
machinery with these two, the right move is to extract a `PerItemIRT`
mixin, not to coerce these two into `EncoderBackbone`.

The migration target is the **six shared-backbone models** below.

| Paper model    | File                | encoder_type | decoder_type   | Notes                            |
|----------------|---------------------|--------------|----------------|----------------------------------|
| MA-GPCM        | `magpcm.py`         | `dkvmn`      | `gpcm`         | `separate_theta=True`            |
| DKVMN+GPCM     | `magpcm.py`         | `dkvmn`      | `gpcm`         | `separate_theta=False` (same cls)|
| DKVMN+Softmax  | `dkvmn_softmax.py`  | `dkvmn`      | `softmax`      | K-way classifier                 |
| DKVMN (binary) | `dkvmn.py`          | `dkvmn`      | `binary`       | K=2 only                         |
| Deep-IRT       | `deep_irt.py`       | `dkvmn`      | `rasch`        | K=2 only, alpha fixed at 1       |
| DKT            | `dkt.py`            | `dkt_gru`    | `binary`       | K=2 only, GRU-style LSTM encoder |

The MA-GPCM and DKVMN+GPCM rows share `magpcm.py`. The ablation switch
is the existing `separate_theta` flag carried on the `dkvmn` encoder.
Five model files in total, six registry entries.

The framework adds one `DKVMNEncoder` reused by five of the six entries,
one `DKTEncoder` (LSTM-based, registered under `dkt_gru`), and four
decoders, `GPCMDecoder` (polytomous), `SoftmaxDecoder` (K-way), 
`BinaryDecoder` (K=2, sigmoid lift), `RaschDecoder` (K=2, alpha frozen
at 1).

---

## (a) Per-model migration notes

For each migrated model, the current forward signature, the cleanest
split between the encoder body and the decoder body, the adapter logic
the `EncoderOutput` dataclass needs, helper wiring, and a risk grade.
The two exempt models appear last for completeness, with their treatment
under the framework, not a migration.

### 1. `dkvmn_softmax.py`, `DKVMNSoftmax`, risk LOW

- Forward, `forward(questions, responses) -> dict`. Returns
  `logits (B,S,K)`, `probs (B,S,K)`, plus dummy zero `theta`, ones
  `alpha`, zeros `beta` so downstream eval code does not crash.
- Encoder body, lines 81 to 113. Embedding, `q_embed`, value embedding,
  DKVMN attention pre-compute, causal read-write loop, `summary` network
  on `[read_t, q_t]`. Encoder ends at `summary_t` per timestep.
- Decoder body, line 111 plus 115 to 116. One linear classifier on
  `summary_t` followed by softmax.
- Split point, the encoder returns `joint_summary = summary_all (B,S,d_s)`
  and copies the same tensor to `student_summary` so the ABC contract is
  satisfied. The decoder reads `joint_summary` only.
- `EncoderOutput` shape, `joint_summary` filled, `student_summary` set
  equal to `joint_summary`, `item_embed = q_embed`, `responses`
  passthrough, `mask = None`, `attention = attn_all`.
- Helpers, no `alpha_from_raw` use (no IRT structure). The new encoder
  is registered as `dkvmn`, the new decoder as `softmax`.
- Why low, encoder and decoder are already disjoint in code, the
  classifier is a single linear plus softmax, the dummy IRT fields
  trivially port into `DecoderOutput` (theta, alpha, beta stay `None`,
  the wrapper synthesises the zero / one tensors on the way out to keep
  the legacy dict shape).

### 2. `dkvmn.py`, `DKVMN` (binary, Zhang 2017), risk LOW

- Forward, `forward(questions, responses) -> dict`, returns
  `logits (B,S,2)`, `probs (B,S,2)`, dummy IRT fields.
- Encoder body, lines 104 to 130. Same skeleton as `DKVMNSoftmax`, with
  `qa_embed` for the 2Q value side. Encoder ends at `summary_t`.
- Decoder body, lines 129, 132 to 134. Scalar sigmoid logit `z` then the
  `[-z/2, z/2]` two-category lift. This is the `BinaryDecoder` registered
  under `binary`.
- Split point, identical to `DKVMNSoftmax`. The split lands at the
  `summary_t` boundary.
- `EncoderOutput`, identical shape to `DKVMNSoftmax`. The value-side
  difference (`qa_embed` vs `OneHot` or `StaticItemEmbedding`) is
  absorbed into the `DKVMNEncoder` constructor's `embedding_type` knob.
  The single new encoder accepts both the 2Q `qa_embed` flavour (binary)
  and the ordinal flavours used by `MAGPCM` / `DKVMNSoftmax`.
- Helpers, no `alpha_from_raw`.
- Why low, almost identical to model 1. Confirms the `dkvmn` encoder
  supports the binary value flavour from the same class.

### 3. `deep_irt.py`, `DeepIRT`, risk LOW to MED

- Forward, `forward(questions, responses) -> dict`, returns
  `logits (B,S,2)`, `probs (B,S,2)`, real `theta (B,S,1)` and
  `beta (B,S,1)`, and a constant `alpha (B,S,1) = ones` (Rasch).
- Encoder body, lines 103 to 117 up to the per-step `read_t` and
  `q_emb` passthrough. The `beta_net` over `q_emb` is item-only and
  runs as a batched pre-compute on line 118. The encoder must expose
  both the per-step `read_t` and the per-step `q_emb`.
- Decoder body, lines 127 to 137. `theta_net` on `[read, q]`,
  `rasch_scale * theta - beta`, two-category lift. This is the
  `RaschDecoder` registered under `rasch`. It produces real `theta`
  and `beta` and a constant `alpha`. The decoder owns the `rasch_scale`
  buffer (3.0 per Yeung 2019).
- Split point, the encoder returns `student_summary = read_all (B,S,d_v)`
  and `item_embed = q_emb (B,S,d_k)`. The decoder concatenates them
  inside its own MLP. `beta_net` belongs to the decoder, not the
  encoder, since it is decoder-family specific (Rasch).
- `EncoderOutput`, `student_summary = read_all`, `joint_summary = None`,
  `item_embed = q_emb`, `attention = attn_all`, `responses` passthrough.
- Helpers, no `alpha_from_raw` (Rasch has no learned alpha).
- Why medium-ish, the decoder owns the `beta_net` and the `rasch_scale`
  buffer. Checkpoint keys move from `model.beta_net.*` to
  `decoder.beta_net.*` under the shim, see (c).

### 4. `dkt.py`, `DKT`, risk LOW

- Forward, `forward(questions, responses) -> dict`, returns
  `logits (B,S,2)` and dummy IRT fields.
- Encoder body, lines 88 to 109. 2Q one-hot index, causal shift, LSTM,
  dropout. Encoder ends at `h (B,S,hidden)`.
- Decoder body, lines 110 to 121. Per-item linear, gather, two-category
  lift. The new decoder is `BinaryDecoder` (same class used by `DKVMN`)
  with an internal per-item linear `nn.Linear(hidden_dim, n_questions)`
  enabled only for the DKT path. The cleanest split is to register a
  thin `DKTBinaryDecoder` subclass that owns the per-item linear plus
  gather, then reuse the `[-z/2, z/2]` lift from the parent. Treat as
  the same decoder family registered under a distinct name, `dkt_binary`,
  if reuse via composition turns out to be cleaner.
- Split point, encoder returns `student_summary = h`,
  `item_embed = None` since DKT has no item-side embedding, and the
  decoder reads `questions` from `enc_out.extra["questions"]` to drive
  the gather.
- `EncoderOutput`, `student_summary = h`, `joint_summary = None`,
  `item_embed = None`, `responses` passthrough, `mask = None`,
  `extra = {"questions": questions}`. The new encoder is registered as
  `dkt_gru`.
- Helpers, no `alpha_from_raw`.
- Why low, the encoder is a stock LSTM and the decoder is one linear
  plus a gather.

### 5. `magpcm.py`, `MAGPCM` (separate_theta=True, paper headline), risk HIGH

- Forward, `forward(questions, responses) -> dict`, returns real
  `theta (B,S,D)`, `alpha (B,S,D)`, `beta (B,S,K-1)`, `logits`, `probs`.
- Encoder body, lines 182 to 247 plus 280. Embeddings, DKVMN attention
  precompute, value memory init, causal loop, the separate
  `ability_summary` and `summary` pathways, write-back to memory. The
  encoder ends at `(ability_summary_t, summary_t, q_t)` per step, or
  equivalently at the per-sequence
  `(ability_summary_all, summary_all, q_embed)`.
- Decoder body, lines 207, 249 to 273, 283. The decoder is the
  `IRTParameterExtractor` triple (ability_network,
  discrimination_network, threshold) plus `GPCMLogits + GPCMHead`.
  Note line 207 already batches beta as a single matmul on `q_embed`.
- Split strategy, encoder returns
  `student_summary = ability_summary_all (B,S,d_s)`,
  `joint_summary = summary_all (B,S,d_s)`,
  `item_embed = q_embed (B,S,d_k)`,
  `responses` passthrough,
  `attention = attn_all`,
  `extra = {"separate_theta": self.separate_theta}`.
- Decoder, `GPCMDecoder`, takes `student_summary` for theta if
  `separate_theta=True`, otherwise `joint_summary`. Concatenates
  `joint_summary` with `item_embed` to feed the
  `discrimination_network`. Applies `alpha_from_raw(raw_alpha, 1.0)`
  for MA-GPCM (per the asymmetry note at `irt.py:151`). Applies
  `threshold` to `item_embed`. Runs `GPCMLogits + GPCMHead`.
- The current `MAGPCM.forward` inlines the IRT sub-networks for speed
  (cached direct refs at lines 227 to 232). The refactor must preserve
  that performance contract. Choice, the encoder runs the causal loop
  and pre-collects `(ability_summary_all, summary_all, q_embed,
  attn_all)`, then the decoder applies its three sub-networks as three
  single matmul calls over the full sequence. This is exactly the
  line-207 trick extended to alpha and theta and preserves the
  kernel-fusion intent. The per-step `value_mem` write stays in the
  encoder.
- `EncoderOutput`, all fields populated.
- Helpers, `alpha_from_raw` lands inside `GPCMDecoder` as the canonical
  mapping, log_scale = 1.0 for `MAGPCM`.
- Why high, this is the paper headline model and the R2 baseline gate
  runs on its checkpoints. Any drift larger than the tolerance band in
  `BASELINE_2026-06-02.md` (RMSE_theta sigma 0.01, r_theta sigma 0.001,
  ACC sigma 0.001) blocks the migration.

### 6. `magpcm.py`, `MAGPCM` (separate_theta=False, DKVMN+GPCM ablation), risk HIGH

- Same class as model 5 with `separate_theta=False`. No new code, the
  registry decision routes through the same wrapper.
- The decoder reads `joint_summary` instead of `student_summary` for
  theta. One branch, decided once inside `GPCMDecoder` from
  `EncoderOutput.extra["separate_theta"]`.
- Migration is a single commit shared with model 5. Both models pass
  the same regression-snapshot test (one fixture per `separate_theta`
  setting).

### Exempt, `static_gpcm.py`, `StaticGPCM`

- **No migration commit.** The class stays as is.
- Add one class attribute, `produces_irt_params = True`, advertised on
  the model rather than on a decoder. Downstream code reads
  `getattr(model, "produces_irt_params", False)` and treats it the same
  as a decoder-side flag.
- Trainer dispatch keeps the existing `student_ids`-aware path. After
  the migration of the other models, the trainer reads
  `getattr(model, "needs_student_id", False)` from the model itself,
  which `StaticGPCM` advertises with `needs_student_id = True`. The
  shared-family wrappers forward the same attribute by delegating to
  `self.encoder.needs_student_id`.

### Exempt, `dynamic_gpcm.py`, `DynamicGPCM`

- **No migration commit.** Same treatment as `StaticGPCM`.
- Add `produces_irt_params = True` and `needs_student_id = True` class
  attributes. The internal gated theta recurrence remains intact.
- The `alpha_log_scale` and per-item table machinery stays in the
  class. If a future refactor wants to share these tables across both
  exempt models, extract a `PerItemGPCMTables` mixin then. Out of scope
  for P2.

---

## (b) Migration sequence with justification

Order, lowest risk first, paper-headline last. Each migration is its own
commit with a verification step. Standard rollback is `git revert <sha>`.
Six commits in total, one per shared-backbone model. `StaticGPCM` and
`DynamicGPCM` are not in this list.

1. **`dkvmn_softmax`**, LOW. First migration. Clean encoder body and a
   single-linear decoder. Establishes the `DKVMNEncoder` shape under the
   registry, with the value side set to `ordinal_static_item`.
2. **`dkvmn`**, LOW. Same `DKVMNEncoder`, value-side flag flipped to
   `qa_2Q`. Decoder is the new `binary`. Confirms the encoder supports
   both value flavours from a single class.
3. **`deep_irt`**, LOW to MED. `DKVMNEncoder` again, with the new
   `rasch` decoder owning `theta_net`, `beta_net`, `rasch_scale`.
   Confirms the decoder can own item-only nets like `beta_net`.
4. **`dkt`**, LOW. New `DKTEncoder` (LSTM-based), reuses the binary
   decoder family with a `dkt_binary` variant that owns the per-item
   linear plus gather. First migration that does not use `DKVMNEncoder`.
5. **`magpcm`, both `separate_theta` settings**, HIGH. One commit migrates
   both. Paper headline. Lands last so every other migration has already
   proven the registry, the `EncoderOutput` shape, the `alpha_from_raw`
   helper inside `gpcm`, and the decoder reuse pattern. The R2 baseline
   gate is exercised on every prior step but bites hardest here.

Justification, the order is monotone in risk and monotone in the number
of new spine components touched. Each step adds at most one new encoder
or decoder to the registry. The `DKVMNEncoder` is written once at step
1 and reused at steps 2, 3, and 5. The `binary` decoder family is
written once at step 2 and specialised at step 4. The `gpcm` decoder is
written once at step 5, the only step that has to interact with the
R2 baseline gate at high tolerance.

The exempt models do not appear in this sequence. The five-commit count
plus the one shared `magpcm` commit gives six commits, matching the
six registry entries.

---

## (c) Backward-compatibility shim

Goal, every existing YAML config and every existing checkpoint continues
to work unchanged after the migration. The `model_type` string in YAML
remains the public knob. The existing concrete class (`MAGPCM`,
`DKVMNSoftmax`, `DKVMN`, `DeepIRT`, `DKT`) remains importable, becomes a
one-line composition wrapper, and preserves its `state_dict` keys.

`StaticGPCM` and `DynamicGPCM` are exempt and need no shim. Their YAML
configs and checkpoints are already compatible because their class is
unchanged.

### Shim layer mapping

Defined in `ma-irt/models/__init__.py`. One dict mapping the YAML
`model_type` to a `(encoder_name, decoder_name, extra_kwargs)` triple.

```python
# ma-irt/models/__init__.py
_SHIM_MAP: dict[str, tuple[str, str, dict]] = {
    "magpcm":         ("dkvmn",       "gpcm",         {}),
    "dkvmn_softmax":  ("dkvmn",       "softmax",      {}),
    "dkvmn":          ("dkvmn",       "binary",       {}),
    "deep_irt":       ("dkvmn",       "rasch",        {}),
    "dkt":            ("dkt_gru",     "dkt_binary",   {}),
    # Exempt, no entry. build_model dispatches these to the legacy class
    # directly.
    # "static_gpcm" -> StaticGPCM
    # "dynamic_gpcm" -> DynamicGPCM
}
```

The MA-GPCM versus DKVMN+GPCM ablation switch is the existing
`separate_theta` flag on the `dkvmn` encoder constructor, no separate
entry needed.

### Concrete class skeleton (shared-family models only)

The existing `MAGPCM`, `DKVMNSoftmax`, `DKVMN`, `DeepIRT`, `DKT` classes
become thin composition wrappers. The constructor signature is unchanged
so every YAML and every test that imports the class keeps working. The
`state_dict` keys are preserved by naming the submodules `encoder` and
`decoder` and by mirroring the original sub-attribute names inside them.
Where exact key parity is required (loading a pre-migration checkpoint),
a `_patch_legacy_keys` hook renames on load. Pattern, shown for `MAGPCM`.

```python
# ma-irt/models/magpcm.py (post-migration)
from .registry import EncoderBackbone, ResponseDecoder

class MAGPCM(nn.Module):
    """Composition wrapper, kept for YAML / checkpoint compatibility."""

    needs_student_id: bool = False  # delegates to self.encoder

    def __init__(self, n_questions: int, n_categories: int = 5, **kwargs):
        super().__init__()
        sep = kwargs.pop("separate_theta", True)
        self.encoder = EncoderBackbone.from_name(
            "dkvmn",
            n_questions=n_questions,
            n_categories=n_categories,
            separate_theta=sep,
            **{k: v for k, v in kwargs.items() if k in _ENC_KWARGS},
        )
        self.decoder = ResponseDecoder.from_name(
            "gpcm",
            input_dim=self.encoder.summary_dim,
            n_questions=n_questions,
            n_categories=n_categories,
            question_dim=self.encoder.key_dim,
            **{k: v for k, v in kwargs.items() if k in _DEC_KWARGS},
        )

    @property
    def needs_student_id(self) -> bool:
        return self.encoder.needs_student_id

    @property
    def produces_irt_params(self) -> bool:
        return self.decoder.produces_irt_params

    def forward(self, questions, responses) -> dict:
        enc_out = self.encoder(questions, responses)
        dec_out = self.decoder(enc_out)
        return {
            "theta":  dec_out.theta,
            "alpha":  dec_out.alpha,
            "beta":   dec_out.beta,
            "logits": dec_out.logits,
            "probs":  dec_out.probs,
        }

    def load_state_dict(self, state_dict, strict: bool = True):
        sd = _patch_legacy_keys(state_dict)
        return super().load_state_dict(sd, strict=strict)
```

For the binary baselines (`DKT`, `DKVMN`, `DKVMNSoftmax`) whose pre-
migration forward returned dummy zero `theta`, ones `alpha`, zeros
`beta`, the wrapper synthesises those tensors after the decoder call so
the dict shape is bit-exact compatible. Cost, three `torch.zeros` /
`torch.ones` allocations per forward, negligible.

### State-dict key preservation

Pre-migration keys for `MAGPCM` are flat, e.g. `q_embed.weight`,
`memory.key_memory`, `summary.0.weight`, `ability_summary.0.weight`,
`irt.ability_network.weight`, `irt.discrimination_network.weight`,
`irt.threshold.weight`, `embedding.W_resp.weight`, and so on.

Post-migration keys are nested, e.g. `encoder.q_embed.weight`,
`encoder.memory.key_memory`, `encoder.summary.0.weight`,
`encoder.ability_summary.0.weight`, `decoder.irt.ability_network.weight`,
`decoder.irt.discrimination_network.weight`,
`decoder.irt.threshold.weight`, `encoder.embedding.W_resp.weight`.

`_patch_legacy_keys` prepends `encoder.` to any non-IRT key and
`decoder.` to any IRT key, then composes with the existing
`patch_monotonic_beta_state_dict` for the older `threshold_base /
threshold_gaps` checkpoints. The patch is keyed off the absence of any
`encoder.` or `decoder.` prefix in the input state_dict, so it is a
no-op for newly trained models. The new function lives alongside the
existing patch in `models/__init__.py` and is composed before it.

```python
def _patch_legacy_keys(state_dict):
    if any(k.startswith(("encoder.", "decoder.")) for k in state_dict):
        return state_dict  # already new layout
    IRT_PREFIXES = ("irt.", "gpcm_logits.", "gpcm_head.")
    out = {}
    for k, v in state_dict.items():
        if k.startswith(IRT_PREFIXES):
            out[f"decoder.{k}"] = v
        else:
            out[f"encoder.{k}"] = v
    return out
```

Verification, the R2 baseline gate after the `magpcm` migration loads
the bench checkpoints documented in `BASELINE_2026-06-02.md` and the
ASSIST2009 fold0 checkpoint, exercising the patch on real pre-migration
state_dicts. For `DeepIRT`, `DKVMN`, `DKVMNSoftmax`, `DKT`, the patch
follows the same rule, all top-level non-IRT keys become `encoder.*`,
the only IRT-shaped keys (`beta_net.*` on `DeepIRT`) become `decoder.*`.
The patch is shared across all five wrappers.

---

## (d) Test plan

Tests live under `ma-irt/tests/`. Three layers, regression-snapshot,
registry, baseline gate (existing, just referenced).

### Per-model regression snapshot

For each migrated wrapper, a `tests/test_migration_<model>.py` with one
test that

1. Sets `torch.manual_seed(0)`.
2. Loads a tiny pre-migration weight snapshot
   `tests/snapshots/<model>_premigration.pt`, saved as a one-off commit
   immediately before the migration of that model starts (single
   construction with `torch.manual_seed(0)`).
3. Builds the post-migration class with the same constructor kwargs.
4. Loads the snapshot weights into the post-migration class via the
   legacy key patch.
5. Runs both forwards on a fixed small input
   `(B=2, S=8, K=cfg.n_categories, Q=8)` and asserts each output dict
   field is bitwise equal under `torch.allclose(atol=0, rtol=0)`.

For `magpcm`, two snapshots are saved, one per `separate_theta` setting,
and both are checked. The snapshots are tiny (<1 MB each) and live
alongside the test, not in `outputs/`. The fixed seed plus the legacy
key patch guarantee the pre-migration weights load deterministically
into the new layout.

The exempt models (`StaticGPCM`, `DynamicGPCM`) do not get a migration
snapshot test. Their existing forward-shape tests cover them.

### Registry tests

One `tests/test_registry_coverage.py` parametrised over the six
shared-family `model_type`s. Asserts

```python
EncoderBackbone.from_name(enc_name, **enc_cfg)  # constructs
ResponseDecoder.from_name(dec_name, **dec_cfg)  # constructs
issubclass(type(model.encoder), EncoderBackbone)
issubclass(type(model.decoder), ResponseDecoder)
model.encoder.needs_student_id in (True, False)
model.decoder.produces_irt_params in (True, False)
```

A second parametrisation covers the two exempt models and asserts the
class-level advertisements are present, `StaticGPCM.produces_irt_params
== True`, `DynamicGPCM.produces_irt_params == True`,
`StaticGPCM.needs_student_id == True`,
`DynamicGPCM.needs_student_id == True`, without touching the registry.

### R2 baseline gate

`tests/test_baseline_reproduction.py`, the load-bearing test. Not
duplicated, just referenced. Must pass after each migration commit. The
gate skips with a clear message when sidecar files are missing (fresh
clone), and aborts with a tolerance-band diff message otherwise.

### Optional, shape and gradient sanity

One generic `tests/test_forward_shapes.py` parametrised over all seven
`model_type`s (six shared-family plus two exempt, the latter passed
their `student_ids` path). Asserts the output dict carries the right
shapes `(B, S, K)`, `(B, S, D)`, `(B, S, K-1)` and that
`loss.backward()` runs without `nan` gradients. Cheap, runs in seconds,
catches any silent shape regression introduced by the shim.

---

## (e) Trainer and evaluator refactor plan

### `training/trainer.py`, line 244 to 254

Current dispatch keys off `getattr(self.model, "_model_type", None)` and
the literal set `("static_gpcm", "dynamic_gpcm")`. Replace with a
single attribute read. The attribute is advertised by the model in both
the shared-family wrappers (delegated via `@property` to
`self.encoder.needs_student_id`) and the exempt models (set directly on
the class).

```python
def _forward(self, questions, responses, student_ids):
    needs_sid = getattr(self.model, "needs_student_id", False)
    if needs_sid and student_ids is not None:
        return self.model(student_ids, questions, responses)
    return self.model(questions, responses)
```

This removes the literal `("static_gpcm", "dynamic_gpcm")` set without
moving the exempt models into the registry. The `_model_type` tag on
the model object becomes vestigial. Keep it set for one release for any
downstream tooling, then remove.

### `scripts/evaluate.py`, lines 153, 201, 202, 1034, 1105, 1131

Current dispatch keys off the package-level sets `NEEDS_STUDENT_ID` and
`HAS_IRT_PARAMS`. Both are removed and replaced with attribute reads on
the model. The exempt models advertise the same flags so this works
without special-casing them.

```python
needs_sid = getattr(model, "needs_student_id", False)
has_irt   = getattr(model, "produces_irt_params", False)
```

The literal branches at lines 1105 and 1131 (`if model_type ==
"static_gpcm"`) are about whether to use MLE or forward-pass linking
for theta, which is genuinely method-level dispatch and not
encoder-level metadata. Keep those literal checks but read `model_type`
from a single location (the wrapper's `_model_type` tag or the model
class name) instead of threading it through every function. Mark as a
follow-up cleanup, not blocking on P2.

### `alpha_from_raw` wiring

Lands inside the `GPCMDecoder` constructor. For `MAGPCM`, the encoder
hands the discrimination summary input, the decoder runs the
`discrimination_network` and then `alpha_from_raw(raw, log_scale=1.0)`.
One helper, one call site inside `GPCMDecoder`, one source of truth for
the shared-backbone family.

The exempt `StaticGPCM` and `DynamicGPCM` continue to call
`alpha_from_raw` from inside their own forward as a follow-up cleanup,
not part of P2. Their inline `torch.exp(self.alpha_log_scale *
self.alpha_raw[questions])` (line 123 in `static_gpcm.py`, line 117 in
`dynamic_gpcm.py`) is already mathematically the helper, replacing it
in-place is a one-line edit. Schedule that as a P2.1 cleanup, after the
shared-family migration is green.

---

## (f) Per-step verification commands

Run from the repo root with `PYTHONPATH=ma-irt` set. The standard env
preamble is `source ~/anaconda3/etc/profile.d/conda.sh && conda activate
research && export PYTHONPATH=ma-irt KMP_DUPLICATE_LIB_OK=TRUE`.

```bash
# Common pre-flight for every step
cd ma-irt && PYTHONPATH=. pytest tests/test_baseline_reproduction.py -v
cd ma-irt && PYTHONPATH=. pytest tests/test_registry_coverage.py -v
cd ma-irt && PYTHONPATH=. pytest tests/test_forward_shapes.py -v
```

Per migration commit, run the model-specific snapshot test plus the
three above. Order matches section (b).

1. `cd ma-irt && PYTHONPATH=. pytest tests/test_migration_dkvmn_softmax.py tests/test_baseline_reproduction.py tests/test_registry_coverage.py tests/test_forward_shapes.py -v`
2. `cd ma-irt && PYTHONPATH=. pytest tests/test_migration_dkvmn.py tests/test_baseline_reproduction.py tests/test_registry_coverage.py tests/test_forward_shapes.py -v`
3. `cd ma-irt && PYTHONPATH=. pytest tests/test_migration_deep_irt.py tests/test_baseline_reproduction.py tests/test_registry_coverage.py tests/test_forward_shapes.py -v`
4. `cd ma-irt && PYTHONPATH=. pytest tests/test_migration_dkt.py tests/test_baseline_reproduction.py tests/test_registry_coverage.py tests/test_forward_shapes.py -v`
5. `cd ma-irt && PYTHONPATH=. pytest tests/test_migration_magpcm.py tests/test_baseline_reproduction.py tests/test_registry_coverage.py tests/test_forward_shapes.py -v`

A commit may not be pushed until its line above passes green.

After the `magpcm` commit, run the full test suite as a final gate.

```bash
cd ma-irt && PYTHONPATH=. pytest tests/ -v
```

---

## (g) Rollback

Each migrated model is one commit. A bad migration is one
`git revert <sha>` away. The revert restores the previous concrete class
plus its `state_dict` key layout. The legacy key patch is additive, so
it keeps working after a revert of any later commit. Order of revert
matches reverse of (b), with `magpcm` revertible independently of
earlier steps because the shim layer keeps every prior model on the
same registry contract.

If two consecutive migrations are bad, revert both in one
`git revert <sha1> <sha2>` call (newest first). The R2 baseline gate is
the authoritative signal, if it goes red, revert and investigate before
re-attempting.

The exempt models cannot regress under P2 because they are not touched.
The only edits they receive are two class-level attribute additions
(`produces_irt_params = True`, `needs_student_id = True`). A bad attribute
addition is reverted in isolation without touching any encoder /
decoder code.
