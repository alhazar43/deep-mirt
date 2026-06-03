# Design 1, modular Encoder and Decoder ABCs plus registry

Status, design only. No code edits in this round. Reviewed against
`PIPELINE_OPT_PLAN.md` section P2, `INVESTIGATION_SOTA.md` section (d),
and `INVESTIGATION_CODEBASE.md` section (b).

The design adopts the py-irt pattern (decorator on ABC, scoped per family)
applied twice. One `EncoderBackbone` ABC for the backbone family. One
`ResponseDecoder` ABC for the polytomous and binary IRT decoder family. A
sibling `CognitiveDiagnosisDecoder` ABC is sketched for the CDM family
(DINA, DINO) but is not required by the current 4 active model types and
will land later (P4).

The design preserves the 4-tensor encoder to decoder contract identified
in `INVESTIGATION_CODEBASE.md` section (b), namely `student_summary`,
`joint_summary`, `item_embed`, `attention`, plus passthrough `responses`
and `mask`. This contract was verified against every current model
(MA-GPCM, DKVMN+GPCM ablation, DKVMN+Softmax, StaticGPCM, DynamicGPCM, DKT,
plain DKVMN, Deep-IRT) and against the four planned encoders (SAKT,
SAINT+, AKT, SimpleKT).

---

## 1. Encoder ABC

File, `ma-irt/models/registry.py` for the ABC and registry primitives.
Subclasses live under `ma-irt/models/encoders/`.

```python
# ma-irt/models/registry.py
from __future__ import annotations
import abc
from dataclasses import dataclass
from typing import ClassVar, Optional
import torch
from torch import Tensor, nn


@dataclass
class EncoderOutput:
    """Tagged output of any EncoderBackbone.

    student_summary, theta-relevant hidden, item-free, shape (B, S, H_s).
    joint_summary,   alpha-relevant hidden, item-aware, shape (B, S, H_j).
    item_embed,      beta-relevant item-only, shape (B, S, H_i).
    responses,       passthrough for decoders that re-enter the loop
                     (DynamicGPCM, AKT-Rasch reparam), shape (B, S).
    mask,            passthrough for loss and metric paths, shape (B, S).
    attention,       optional, diagnostic only, shape (B, S, M) for DKVMN
                     or (B, n_heads, S, S) for Transformer-family.
    """
    student_summary: Tensor
    joint_summary: Tensor
    item_embed: Tensor
    responses: Tensor
    mask: Tensor
    attention: Optional[Tensor] = None


class EncoderBackbone(nn.Module, abc.ABC):
    """Encoder over (questions, responses) producing a per-step hidden state.

    Subclasses register via @EncoderBackbone.register("name") and implement
    forward returning an EncoderOutput. The shape contract is enforced by
    the dataclass plus the tests in test_encoder_interface.py.

    Class attributes declare capabilities the trainer and scripts need
    without inspecting the instance type.
    """

    _registry: ClassVar[dict[str, type["EncoderBackbone"]]] = {}

    # Capability flags, replace the script-level NEEDS_STUDENT_ID and
    # HAS_IRT_PARAMS triplet in scripts/evaluate.py:165-169.
    needs_student_id: ClassVar[bool] = False
    produces_student_summary: ClassVar[bool] = True
    produces_attention: ClassVar[bool] = False
    compatible_decoders: ClassVar[tuple[str, ...]] = ()  # empty means any

    @classmethod
    def register(cls, name: str):
        def deco(sub: type["EncoderBackbone"]) -> type["EncoderBackbone"]:
            if name in cls._registry:
                raise ValueError(f"encoder {name!r} already registered")
            cls._registry[name] = sub
            return sub
        return deco

    @classmethod
    def from_name(cls, name: str, **cfg) -> "EncoderBackbone":
        if name not in cls._registry:
            raise KeyError(
                f"encoder {name!r} not registered, available "
                f"{sorted(cls._registry.keys())}"
            )
        return cls._registry[name](**cfg)

    @classmethod
    def list_encoders(cls) -> list[str]:
        return sorted(cls._registry.keys())

    @abc.abstractmethod
    def forward(
        self,
        q: Tensor,           # (B, S) item ids, 1-indexed, 0 is pad
        r: Tensor,           # (B, S) response ids
        mask: Tensor,        # (B, S) bool
        student_ids: Optional[Tensor] = None,  # (B,) for StaticGPCM only
    ) -> EncoderOutput:
        ...

    # Optional hooks. Default implementations are no-ops.
    def reset_state(self) -> None:
        """Clear any persistent recurrent state. Stateless by default."""
        return None

    def get_attention_weights(self) -> Optional[Tensor]:
        """Return the most recent attention tensor or None."""
        return None
```

Encoder fit matrix against the four current backbones and the four planned
ones.

| Encoder | student_summary | joint_summary | item_embed | attention | needs_student_id |
|---|---|---|---|---|---|
| `dkvmn` (MA-GPCM, DKVMN+GPCM, DKVMN+Softmax) | `ability_summary` | `summary([read, q])` | `q_embed` | DKVMN attn | False |
| `dkvmn_binary` (DKT-style 2Q embed) | read | read | `q_embed` | DKVMN attn | False |
| `static` (StaticGPCM) | `theta_embed[sid]` broadcast | `theta_embed[sid]` broadcast | `q_embed` | None | True |
| `recurrent_theta` (DynamicGPCM) | gated theta state | gated theta state | `q_embed` | None | False |
| `lstm` (DKT) | LSTM hidden | LSTM hidden | item one-hot | None | False |
| `sakt` (planned) | post-attn xemb | post-attn xemb | exercise emb | self-attn | False |
| `saintpp` (planned) | encoder stream | decoder stream | exercise emb | cross-attn | False |
| `akt` (planned) | d_output | d_output | `q_embed + pid * var` | monotonic | False |
| `simplekt` (planned) | d_output | d_output | `q_embed` | self-attn | False |

The contract holds. Encoders that do not maintain a separated student
pathway (DKT, DKVMN+Softmax, plain DKVMN, SAKT, SimpleKT) set
`student_summary = joint_summary` and let the decoder's `separate_theta`
flag remain a no-op. Encoders that have no meaningful attention diagnostic
(static, recurrent_theta, lstm) set `attention = None` and
`produces_attention = False`.

---

## 2. Decoder ABC

File, `ma-irt/models/registry.py` for the ABC. Subclasses live under
`ma-irt/models/decoders/`.

```python
# continued in ma-irt/models/registry.py
@dataclass
class DecoderOutput:
    """Tagged output of any ResponseDecoder.

    For IRT decoders (GPCM, GRM, PCM, MIRT), theta, alpha, beta are
    populated. For pure classification decoders (Softmax, BinaryClassifier),
    those fields are None.

    logits, shape (B, S, K), pre-softmax category scores.
    probs,  shape (B, S, K), normalized category probabilities.
    """
    logits: Tensor
    probs: Tensor
    theta: Optional[Tensor] = None   # (B, S, D)
    alpha: Optional[Tensor] = None   # (B, S, D)
    beta: Optional[Tensor] = None    # (B, S, K-1)


class ResponseDecoder(nn.Module, abc.ABC):
    """Polytomous IRT and binary classification response decoder.

    GPCM, GRM, PCM, NRM, and MIRT (as K=2 vector-trait case) live under
    this ABC. Pure-softmax and binary-classifier heads also live here,
    they simply set the IRT fields on DecoderOutput to None.

    DINA and DINO need a different input contract (skill mastery vector
    plus Q-matrix) and will register under a sibling
    CognitiveDiagnosisDecoder ABC when P4 lands.
    """

    _registry: ClassVar[dict[str, type["ResponseDecoder"]]] = {}

    # Capability flags.
    produces_irt_params: ClassVar[bool] = True
    needs_separate_student_summary: ClassVar[bool] = False
    n_categories: ClassVar[Optional[int]] = None  # None means any K

    @classmethod
    def register(cls, name: str):
        def deco(sub: type["ResponseDecoder"]) -> type["ResponseDecoder"]:
            if name in cls._registry:
                raise ValueError(f"decoder {name!r} already registered")
            cls._registry[name] = sub
            return sub
        return deco

    @classmethod
    def from_name(cls, name: str, **cfg) -> "ResponseDecoder":
        if name not in cls._registry:
            raise KeyError(
                f"decoder {name!r} not registered, available "
                f"{sorted(cls._registry.keys())}"
            )
        return cls._registry[name](**cfg)

    @classmethod
    def list_decoders(cls) -> list[str]:
        return sorted(cls._registry.keys())

    @abc.abstractmethod
    def forward(self, enc: EncoderOutput) -> DecoderOutput:
        ...

    # Optional hook for encoders that need expected-value feedback inside
    # their own loop (DynamicGPCM surprise term).
    def expected_response(self, dec: DecoderOutput) -> Tensor:
        """Expected category index under probs, shape (B, S)."""
        K = dec.probs.shape[-1]
        cats = torch.arange(K, device=dec.probs.device, dtype=dec.probs.dtype)
        return (dec.probs * cats).sum(dim=-1)
```

Decoder fit matrix against the five planned heads.

| Decoder | theta | alpha | beta | logits | probs | n_categories |
|---|---|---|---|---|---|---|
| `gpcm` | from `student_summary` or `joint_summary` (flag) | from `joint_summary` | from `item_embed` | cumulative sum | softmax | any K |
| `grm` | same | same | sigmoid of `alpha(theta - beta_k)`, differenced | per-category diff | softmax-equivalent | any K |
| `pcm` | same | fixed 1 | from `item_embed` | cumulative sum | softmax | any K |
| `mirt` | vector D | vector D | from `item_embed` | dot product | softmax / sigmoid | K=2 |
| `dina` (sibling ABC, P4) | skill mastery | (guess, slip) | Q-matrix | per-category | sigmoid | K=2 |
| `softmax` | None | None | None | linear of `joint_summary` | softmax | any K |
| `binary` (DKT, plain DKVMN) | None | None | None | gather or linear | sigmoid wrapped to 2-class | K=2 |
| `rasch` (Deep-IRT) | from `student_summary` | fixed 1 | `tanh(beta_net(item_embed))` | `c * theta - beta` | sigmoid | K=2 |

The contract holds across the polytomous family and across the binary
heads. DINA is excluded by design and goes on the sibling ABC.

---

## 3. Registry

The decorator and factory primitives are on the ABCs themselves
(section 1 and section 2). The discoverable namespaces are package level.

```python
# ma-irt/models/encoders/__init__.py
"""Encoder backbones, importable side effects register each subclass."""
from ma_irt.models.registry import EncoderBackbone, EncoderOutput
from ma_irt.models.encoders import dkvmn        # noqa, registers "dkvmn"
from ma_irt.models.encoders import dkvmn_binary # noqa, registers "dkvmn_binary"
from ma_irt.models.encoders import static       # noqa, registers "static"
from ma_irt.models.encoders import recurrent_theta  # noqa
from ma_irt.models.encoders import lstm         # noqa, registers "lstm"
# P3 ports register here once landed
# from ma_irt.models.encoders import sakt, saint_plus, akt, simplekt

__all__ = ["EncoderBackbone", "EncoderOutput"]
```

```python
# ma-irt/models/decoders/__init__.py
"""Response decoders, importable side effects register each subclass."""
from ma_irt.models.registry import ResponseDecoder, DecoderOutput
from ma_irt.models.decoders import gpcm         # noqa, registers "gpcm"
from ma_irt.models.decoders import softmax      # noqa, registers "softmax"
from ma_irt.models.decoders import binary       # noqa, registers "binary"
from ma_irt.models.decoders import rasch        # noqa, registers "rasch"
# P4 heads register here once landed
# from ma_irt.models.decoders import grm, pcm, mirt

__all__ = ["ResponseDecoder", "DecoderOutput"]
```

Lookup is `EncoderBackbone.from_name("dkvmn", **cfg)` and
`ResponseDecoder.from_name("gpcm", **cfg)`. Listing is
`EncoderBackbone.list_encoders()` and `ResponseDecoder.list_decoders()`,
shaped after timm's `list_models`.

Side-effect-on-import is a conscious choice. Per
`INVESTIGATION_SOTA.md` section (d), this is how py-irt, timm, and
torchvision all surface their registries. The alternative (explicit
register on each subclass file imported by a central manifest) requires
the central manifest, which is the if/elif anti-pattern from pyKT we are
trying to avoid.

---

## 4. Unified `build_model`

File, `ma-irt/models/__init__.py`. Single entry point that composes one
registered encoder with one registered decoder.

```python
# ma-irt/models/__init__.py
from __future__ import annotations
import torch
from torch import nn
from ma_irt.config.types import ModelConfig, BaseConfig
from ma_irt.models.registry import (
    EncoderBackbone, ResponseDecoder, EncoderOutput, DecoderOutput,
)
import ma_irt.models.encoders  # noqa, populates EncoderBackbone._registry
import ma_irt.models.decoders  # noqa, populates ResponseDecoder._registry


class MaIrtModel(nn.Module):
    """Composed encoder plus decoder.

    The composition is the only place that knows both halves. Trainer,
    evaluator, and scripts hold a MaIrtModel and call its forward, never
    inspect the encoder or decoder type.
    """

    def __init__(self, encoder: EncoderBackbone, decoder: ResponseDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        # Surface capability flags for the trainer and scripts.
        self.needs_student_id = encoder.needs_student_id
        self.produces_irt_params = decoder.produces_irt_params

    def forward(
        self,
        student_ids: torch.Tensor,
        q: torch.Tensor,
        r: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict:
        enc = self.encoder(q, r, mask, student_ids=student_ids)
        dec = self.decoder(enc)
        # Surface the legacy dict shape that trainer.py and evaluate.py
        # already expect. Keeps R2 baseline tests green.
        return {
            "logits": dec.logits,
            "probs": dec.probs,
            "theta": dec.theta,
            "alpha": dec.alpha,
            "beta": dec.beta,
            "mask": enc.mask,
            "attention": enc.attention,
        }


def build_model(
    cfg: BaseConfig,
    device: str = "cpu",
    n_students: int = 0,
) -> MaIrtModel:
    """Single source of truth for model construction.

    Replaces the three parallel dispatch tables in scripts/train.py:65,
    scripts/evaluate.py:154 and :214, scripts/compute_linking.py:35, and
    the model-specific path in scripts/plot_recovery_split.py:67.

    The cfg.model.model_type field is honored via the back-compat shim
    in _shim_legacy_model_type. New configs may set
    cfg.model.encoder_type and cfg.model.decoder_type directly.
    """
    mcfg: ModelConfig = cfg.model
    encoder_name, decoder_name, extras = _resolve_encoder_decoder(mcfg)

    encoder_kwargs = _encoder_kwargs(encoder_name, mcfg, n_students)
    decoder_kwargs = _decoder_kwargs(decoder_name, mcfg, extras)

    encoder = EncoderBackbone.from_name(encoder_name, **encoder_kwargs)
    decoder = ResponseDecoder.from_name(decoder_name, **decoder_kwargs)
    model = MaIrtModel(encoder, decoder).to(device)

    # Legacy state-dict patcher (monotonic beta key rewrite) moves here,
    # see scripts/evaluate.py:172 _patch_monotonic_beta_state_dict.
    model._legacy_patch = _patch_monotonic_beta_state_dict  # type: ignore[attr-defined]
    return model
```

`ModelConfig` gains two optional fields (`encoder_type`, `decoder_type`)
in `config/types.py`. They default to None. The legacy `model_type` field
stays and is consulted by the shim when the new fields are absent.

---

## 5. Backward compatibility shim

```python
# ma-irt/models/__init__.py, continued
_LEGACY_MODEL_TYPE_MAP: dict[str, tuple[str, str, dict]] = {
    # Active 4 model types under the new ABCs.
    "magpcm":         ("dkvmn",            "gpcm",    {"separate_theta": True}),
    "dkvmn_softmax":  ("dkvmn",            "softmax", {}),
    "static_gpcm":    ("static",           "gpcm",    {"separate_theta": False}),
    "dynamic_gpcm":   ("recurrent_theta",  "gpcm",    {"separate_theta": False}),
    # K=2 binary baselines (added in the uncommitted train.py edit).
    "dkt":            ("lstm",             "binary",  {}),
    "dkvmn":          ("dkvmn_binary",     "binary",  {}),
    "deep_irt":       ("dkvmn_binary",     "rasch",   {}),
}


def _resolve_encoder_decoder(mcfg: ModelConfig) -> tuple[str, str, dict]:
    # New-config path. encoder_type and decoder_type take precedence.
    enc = getattr(mcfg, "encoder_type", None)
    dec = getattr(mcfg, "decoder_type", None)
    if enc is not None and dec is not None:
        return enc, dec, {}
    if enc is not None or dec is not None:
        raise ValueError(
            "encoder_type and decoder_type must be set together, "
            "or omit both and use model_type"
        )
    # Legacy path.
    mt = mcfg.model_type
    if mt not in _LEGACY_MODEL_TYPE_MAP:
        raise KeyError(
            f"model_type {mt!r} not recognised, known "
            f"{sorted(_LEGACY_MODEL_TYPE_MAP.keys())}"
        )
    return _LEGACY_MODEL_TYPE_MAP[mt]


def _encoder_kwargs(name: str, mcfg: ModelConfig, n_students: int) -> dict:
    # Per-encoder kwarg slicing. The DKVMN family consumes embedding_type,
    # key_dim, value_dim, memory_size. The static family consumes
    # n_students. The lstm family consumes hidden_dim. Each encoder
    # advertises its constructor kwargs via a class-level _expected_kwargs
    # tuple to keep the slicing self-describing.
    ...


def _decoder_kwargs(name: str, mcfg: ModelConfig, extras: dict) -> dict:
    # extras carries the shim-derived flags such as separate_theta.
    # Each decoder advertises its constructor kwargs via _expected_kwargs.
    ...


def _patch_monotonic_beta_state_dict(sd: dict) -> dict:
    """Legacy key rewrite, originally at scripts/evaluate.py:172."""
    ...
```

What the shim guarantees.

1. Every YAML in `configs/bulk/`, `configs/assistments/`, `configs/smoke/`
   that sets `model.model_type = magpcm | dkvmn_softmax | static_gpcm |
   dynamic_gpcm | dkt | dkvmn | deep_irt` continues to load and produce
   the same model.
2. Every cached `best.pt` checkpoint listed in `BASELINE_2026-06-02.md`
   continues to load. The monotonic-beta state-dict patcher moves into
   `build_model` so the load path is unchanged from the caller's view.
3. `cfg.model.separate_theta` keeps the same YAML name and semantics. It
   is read by the shim, packed into `extras`, and handed to the GPCM
   decoder.

What the shim does not do.

1. It does not promote new `encoder_type` and `decoder_type` keys into
   the existing YAMLs. Bulk configs stay on `model_type`. New
   smoke configs and new bench configs (P3, P4) opt into the new keys.
2. It does not change the semantics of any current model. The DKVMN
   encoder produces the same tensors at the same step indices. The GPCM
   decoder applies the same cumulative-sum head. R2 baseline
   reproduction (`test_baseline_reproduction.py` in P2) is the
   load-bearing test.

---

## 6. Test sketches

Four new files. One or two paradigmatic test functions per file. Full
test bodies come in the implementation round.

### `tests/test_encoder_interface.py`

```python
import pytest
import torch
from ma_irt.models.registry import EncoderBackbone, EncoderOutput


@pytest.mark.parametrize("name", EncoderBackbone.list_encoders())
def test_encoder_forward_returns_typed_output(name):
    """Every registered encoder returns an EncoderOutput with the
    documented shapes. Catches contract drift at registration time."""
    enc = EncoderBackbone.from_name(name, **_min_kwargs_for(name))
    B, S = 2, 5
    q = torch.randint(1, 11, (B, S))
    r = torch.zeros(B, S, dtype=torch.long)
    mask = torch.ones(B, S, dtype=torch.bool)
    out = enc(q, r, mask, student_ids=torch.arange(B))
    assert isinstance(out, EncoderOutput)
    assert out.student_summary.shape[:2] == (B, S)
    assert out.joint_summary.shape[:2] == (B, S)
    assert out.item_embed.shape[:2] == (B, S)
    assert out.mask.shape == (B, S)


def test_dkvmn_encoder_preserves_attention_diagnostic():
    """DKVMN sets produces_attention=True and surfaces (B, S, M)."""
    ...
```

### `tests/test_decoder_interface.py`

```python
import pytest
import torch
from ma_irt.models.registry import (
    ResponseDecoder, DecoderOutput, EncoderOutput,
)


@pytest.mark.parametrize("name", ResponseDecoder.list_decoders())
def test_decoder_forward_returns_typed_output(name):
    dec = ResponseDecoder.from_name(name, **_min_kwargs_for(name))
    enc = _fake_encoder_output(B=2, S=5, H=16, K=4)
    out = dec(enc)
    assert isinstance(out, DecoderOutput)
    B, S, K = enc.mask.shape[0], enc.mask.shape[1], 4
    assert out.logits.shape == (B, S, K)
    assert out.probs.shape == (B, S, K)
    if dec.produces_irt_params:
        assert out.theta is not None and out.alpha is not None
        assert out.beta is not None and out.beta.shape[-1] == K - 1


def test_gpcm_decoder_separate_theta_flag_routes_correctly():
    """separate_theta=True reads student_summary, False reads
    joint_summary. The ablation flag must remain addressable from YAML."""
    ...
```

### `tests/test_registry.py`

```python
import pytest
import torch
from torch import nn
from ma_irt.models.registry import (
    EncoderBackbone, ResponseDecoder, EncoderOutput, DecoderOutput,
)


def test_duplicate_registration_raises():
    @EncoderBackbone.register("dup_probe")
    class _A(EncoderBackbone):
        def forward(self, q, r, mask, student_ids=None):
            ...
    with pytest.raises(ValueError, match="already registered"):
        @EncoderBackbone.register("dup_probe")
        class _B(EncoderBackbone):
            def forward(self, q, r, mask, student_ids=None):
                ...


def test_from_name_unknown_lists_available():
    with pytest.raises(KeyError, match="not registered"):
        EncoderBackbone.from_name("nope_not_a_real_encoder")
```

### `tests/test_backcompat_shim.py`

```python
import pytest
import torch
from ma_irt.config.loader import load_config
from ma_irt.models import build_model, MaIrtModel


@pytest.mark.parametrize("model_type,encoder,decoder", [
    ("magpcm",        "dkvmn",           "gpcm"),
    ("dkvmn_softmax", "dkvmn",           "softmax"),
    ("static_gpcm",   "static",          "gpcm"),
    ("dynamic_gpcm",  "recurrent_theta", "gpcm"),
])
def test_legacy_model_type_resolves_to_expected_pair(
    model_type, encoder, decoder, tmp_path,
):
    """The shim maps every legacy model_type to a known (encoder, decoder)
    pair without YAML edits. This is the R2 contract."""
    cfg = _minimal_cfg(tmp_path, model_type=model_type)
    m = build_model(cfg, device="cpu", n_students=10)
    assert isinstance(m, MaIrtModel)
    assert type(m.encoder).__name__.lower().startswith(encoder.replace("_", ""))
    assert type(m.decoder).__name__.lower().startswith(decoder)


def test_existing_bulk_config_loads_unchanged():
    """An untouched configs/bulk/*.yaml continues to build a model."""
    cfg = load_config("configs/bulk/continuous_static_gpcm_q200_k4_s0.yaml")
    m = build_model(cfg, device="cpu", n_students=5000)
    assert m.needs_student_id is True   # static encoder needs sid
```

The four test files plus the P2 `test_baseline_reproduction.py` are the
R2 gate. The gate must pass on a single fold before and after the
refactor.

---

## Design decisions made

1. **Two ABCs, not one.** Encoder and Decoder are distinct families with
   different forward signatures. Following py-irt, each family gets its
   own `_registry` dict and its own `from_name`. CDM goes on a third
   sibling ABC when DINA lands (P4).
2. **Side-effect-on-import registration.** Mirrors py-irt, timm, and
   torchvision. The alternative central manifest is the pyKT
   anti-pattern.
3. **`EncoderOutput` and `DecoderOutput` as dataclasses, not dicts.**
   Static type checkability, plus the contract is documented in one
   place. The legacy dict shape is reconstructed inside `MaIrtModel.forward`
   so the trainer and evaluator do not need to be updated in the same
   round.
4. **Capability flags on the class, not on the script.** `needs_student_id`,
   `produces_irt_params`, `produces_attention` live on the ABC classes.
   Removes the `MODEL_CLASSES`, `NEEDS_STUDENT_ID`, `HAS_IRT_PARAMS`
   tables from `scripts/evaluate.py:154-169`.
5. **Forward arity unified.** Every encoder accepts
   `(q, r, mask, student_ids=None)`. Encoders that do not need
   `student_ids` ignore the kwarg. Removes the `_model_type` monkey
   patch and the trainer's two-branch dispatch (A3).
6. **`separate_theta` moves to the decoder.** The MA-GPCM ablation
   flag stays addressable from YAML by the same name but is read by the
   `GPCMDecoder`, not by the encoder. This matches the
   `INVESTIGATION_CODEBASE.md` finding that the flag selects which hidden
   stream the decoder reads.
7. **Pre-computed `beta` projection moves into the decoder.** The
   current encoder calls `self.irt.threshold(q_embed)` at
   `magpcm.py:207`. After P2 the decoder owns the threshold network and
   computes `beta` from `enc.item_embed`.
8. **DynamicGPCM coupling resolved via decoder hook.** The encoder needs
   the GPCM expected response inside its own loop. The decoder ABC
   exposes `expected_response(dec_out)` so the encoder can call the
   decoder forward once per step without re-entering the registry.
9. **Legacy state-dict patcher moves next to `build_model`.** Loading is
   part of construction.
10. **`encoder_type` and `decoder_type` are additive ModelConfig fields.**
    Both default to None. The shim resolves them from `model_type` when
    absent. Old YAMLs need zero edits.

## Open questions for the user

1. **AKT Rasch reparameterization placement.** The pyKT and authors'
   AKT compute `q_embed_data = q_embed + pid * variation` inside the
   encoder before the FC head, per `INVESTIGATION_SOTA.md` section
   (a). The investigation flags this as a decoder concern. Confirm,
   should the AKT encoder return the decomposed `(q_embed, pid_embed,
   variation)` triple via `EncoderOutput.item_embed` as a tuple, or
   should AKT carry its own `AKTRaschGPCMDecoder` subclass that knows
   how to apply the reparameterization? The first is more general, the
   second is closer to the published AKT design.
2. **SAINT+ dual-stream handling.** SAINT+ has two streams (exercise
   encoder, response decoder) and the published code couples them in
   `forward`. Confirm, do we fold both into `joint_summary` as the
   final dec-side hidden, treat the exercise stream as `item_embed`,
   and leave `student_summary` equal to `joint_summary` (matching the
   AKT and SAKT pattern), or do we widen `EncoderOutput` with an
   optional `exercise_summary` field?
3. **Embedding factory placement.** The three ordinal value embeddings
   (`onehot`, `learned`, `static_item`) currently live in
   `models/components/embeddings.py`. The P2 plan moves them under
   `models/encoders/value_embedding.py`. Confirm, should the value
   embedding be (a) a class attribute on each DKVMN-family encoder,
   (b) a separately registered factory the encoder constructor selects
   by name, or (c) left in `components/` and imported by the encoder?
   Recommendation, option (b), parallel to the encoder and decoder
   registries.
4. **CDM ABC, now or P4.** The sibling `CognitiveDiagnosisDecoder` ABC
   for DINA is sketched here but not implemented. Confirm, should the
   ABC stub land in this round so future P4 work has a placeholder, or
   wait until P4 actually needs it?
5. **`MaIrtModel.forward` dict shape.** The composed model returns the
   legacy 7-key dict (`logits`, `probs`, `theta`, `alpha`, `beta`,
   `mask`, `attention`) for trainer and evaluator compatibility.
   Confirm, is it acceptable to leak `mask` and `attention` into the
   public forward output, or should the composed model return only
   `DecoderOutput` and have the trainer pull `mask` from the batch
   directly? The first is lower-friction for P2, the second is
   cleaner.
6. **Reset-state semantics for stateful encoders.** `reset_state()` is
   declared as an optional no-op hook. The recurrent_theta encoder
   (DynamicGPCM) has no persistent cross-batch state today, so the hook
   is unused. Confirm, do we need this hook now for any planned
   encoder (in-context Transformer with KV cache, for instance), or
   defer until a stateful backbone actually arrives?
