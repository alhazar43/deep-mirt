# Investigation 2. SOTA encoder backbones and modular encoder-decoder patterns

Scope. Map out how recent KT papers (SAKT, SAINT+, AKT, SimpleKT) implement their encoders, how adjacent ML libraries swap backbones, and how IRT/CDM codebases organize response models. Goal is a recommendation for a single PyTorch pattern that lets `ma-irt/` support swappable encoders (DKVMN, Transformer, AKT, SAKT, SAINT+) under one interface and swappable decoders (GPCM, GRM, PCM, DINA, MIRT) under another.

All cited URLs were fetched via WebFetch and returned 200 OK except where noted in section (e).

## (a) SOTA KT backbones with code links

### SAKT, Pandey and Karypis 2019, arXiv 1907.06837

Canonical pyKT implementation at `pykt/models/sakt.py` (https://github.com/pykt-team/pykt-toolkit/blob/main/pykt/models/sakt.py).

```
__init__(self, num_c, seq_len, emb_size, num_attn_heads, dropout,
         num_en=2, emb_type="qid", emb_path="", pretrain_dim=768)
forward(self, q, r, qry, qtest=False) -> p  or  (p, xemb)
```

Hidden state. `xemb` of shape `(B, T, emb_size)`, the post-attention contextual embedding for each step.

Encoder/head separation. Clean. Stacked attention `Blocks` produce `xemb`, then a single `nn.Linear` plus sigmoid forms the head. Setting `qtest=True` returns `xemb` alongside predictions, which is exactly the seam we would target.

A second widely cited port is `arshadshk/SAKT-pytorch` (https://github.com/arshadshk/SAKT-pytorch), but it lacks the masking and concept handling that pyKT adds.

### SAINT+, Shin et al. LAK 2021, arXiv 2010.12042

Canonical pyKT implementation at `pykt/models/saint_plus_plus.py` (https://github.com/pykt-team/pykt-toolkit/blob/main/pykt/models/saint_plus_plus.py).

```
__init__(self, num_q, num_c, seq_len, emb_size, num_attn_heads,
         dropout, n_blocks=1, emb_type="qid", emb_path="", pretrain_dim=768)
forward(self, in_ex, in_cat, in_res, qtest=False) -> p  or  (p, hidden)
```

Hidden state. Decoder output `(B, T, emb_size)` after cross-attention with the encoder stack.

Encoder/head separation. Partial. The transformer encoder (exercise stream) and decoder (response stream) are two separate `nn.ModuleList`s and the final `nn.Linear` head is isolated, but the encoder-decoder coupling is hard-wired in `forward`, so swapping just the backbone requires keeping both streams together. Third-party ports (`shivanandmn/SAINT_plus-Knowledge-Tracing-`, `Chang-Chia-Chi/SaintPlus-Knowledge-Tracing-Pytorch`, `arshadshk/SAINT-pytorch`) follow the same dual-stream layout.

### AKT, Ghosh et al. KDD 2020, arXiv 2007.12324

Two reference points. Authors' release at https://github.com/arghosh/AKT (`akt.py`) and the pyKT mirror at `pykt/models/akt.py` (https://github.com/pykt-team/pykt-toolkit/blob/main/pykt/models/akt.py).

Authors' signature.
```
__init__(self, n_question, n_pid, d_model, n_blocks, kq_same, dropout, model_type,
         final_fc_dim=512, n_heads=8, d_ff=2048, l2=1e-5, separate_qa=False)
forward(self, q_data, qa_data, target, pid_data=None)
  -> (loss, sigmoid_predictions, valid_count)
```

pyKT signature.
```
forward(self, q_data, target, pid_data=None, qtest=False)
  -> predictions  or  (predictions, intermediate_emb)
```

Hidden state. `d_output` of shape `(B, T, d_model)`, produced by the `Architecture` module's two-block design where `blocks_1` self-attends over QA history and `blocks_2` attends over questions while monotonically masking future responses.

Encoder/head separation. Tightly coupled but recoverable. The Rasch-style item embedding (`q_embed_data = q_embed + pid * variation`) is computed inside `forward` and concatenated with `d_output` before the FC head. The encoder seam exists, but the Rasch regularizer and the `q_embed`/`pid_embed` decomposition belong with the IRT decoder rather than the backbone. Important point for the abstraction. AKT's monotonic attention is part of the backbone, the Rasch reparameterization is part of the decoder.

### SimpleKT, Liu et al. ICLR 2023, arXiv 2302.06881

Canonical pyKT implementation at `pykt/models/simplekt.py` (https://github.com/pykt-team/pykt-toolkit/blob/main/pykt/models/simplekt.py).

```
__init__(..., d_model, n_blocks, n_question, n_pid, emb_type,
         final_fc_dim, num_attn_heads, ...)
forward(self, dcur, qtest=False, train=True) -> predictions [, intermediates]
```

Forward takes a `dcur` dict containing question, concept, and response sequences. This is pyKT's standardized batch container.

Hidden state. `d_output` of shape `(B, T, d_model)`, dot-product attention without distance decay (the simplification from AKT).

Encoder/head separation. Clean. The `Architecture` module produces `d_output`, then a sequential `Linear -> ReLU -> Dropout -> Linear` head consumes `[d_output, q_embed]`. SimpleKT is the clearest worked example in pyKT of "transformer encoder feeds a tiny head". For our purposes, SimpleKT shows the seam we want.

### Common shape contract across the four backbones

Every backbone produces a per-timestep hidden state `h_t in R^{d_model}`, shape `(B, T, d_model)`, indexed so `h_t` summarizes information up to (and excluding, for AKT/SimpleKT) the current response. This is the same contract DKVMN exposes via its read vector `r_t`. The seam is real and consistent across the literature.

## (b) Encoder-decoder patterns in adjacent ML

**HuggingFace transformers.** `PretrainedConfig` + `PreTrainedModel` + `AutoModel`. Models register with `AutoConfig.register("model_type_string", ConfigClass)` and `AutoModel.register(ConfigClass, ModelClass)`. The `from_pretrained` machinery dispatches on the config's `model_type` attribute. Powerful but heavyweight, the contract assumes the HF hub, weight serialization, and a sharded download story. Overkill for a research codebase.

**torchvision.models.** Registry of bare functions with `register_model`, plus thin discovery via `list_models()` and `get_model(name, **config)`. Simple by design. The pattern is described at https://pytorch.org/blog/easily-list-and-initialize-models-with-new-apis-in-torchvision/ and the API docs at https://docs.pytorch.org/vision/main/generated/torchvision.models.get_model.html.

**timm `_registry.py`.** Same idea, slightly fancier. `@register_model` decorator populates a module-level `_model_entrypoints` dict, plus `list_models` with fnmatch filters and `model_entrypoint(name)` for lookup. Source at https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/_registry.py. The essential pattern (decorator + dict + factory) is ~50 lines.

**PyTorch Lightning LightningModule.** Not a registry at all, just a convention. The `LightningModule` wraps an `encoder` and `decoder` as attributes, exposes `forward`, and lets the trainer drive optimization. Useful as the wrapper around a registered backbone + head, not as the registration mechanism itself.

## (c) IRT/CDM decoder patterns

### R mirt (Chalmers)

Driven by an `itemtype` string fed to the `mirt()` function. Supported values include `Rasch`, `2PL`, `3PL`, `3PLu`, `4PL`, `graded`, `grsm`, `gpcm`, `nominal`, `ideal`, `PC2PL`, `PC3PL`, `2PLNRM`, `3PLNRM`, `3PLuNRM`, `4PLNRM`. Internally these dispatch to S4 classes (`dich`, `graded`, `gpcm`, `nominal`, ...) under a common abstract `AllItemsClass`. The dispatch surface is "string -> class", which is functionally a registry. Source PDF at https://cran.r-project.org/web/packages/mirt/mirt.pdf.

### py-irt (Lalor and Rodriguez 2022, arXiv 2203.01282)

Cleanest Python reference for our purposes. `py_irt/models/abstract_model.py` defines

```python
class IrtModel(abc.ABC):
    @classmethod
    def register(cls, name): ...        # decorator factory
    @classmethod
    def from_name(cls, name): ...       # registry lookup
    @abc.abstractmethod
    def get_model(self): pass
    @abc.abstractmethod
    def get_guide(self): pass
    @abc.abstractmethod
    def export(self) -> Dict[str, Any]: pass
```

Subclasses opt in via `@IrtModel.register("2pl")` on the class itself (see `py_irt/models/two_param_logistic.py`). Sibling files implement 1PL, 2PL, 4PL, multidim 2PL, amortized 1PL. This is the canonical "decorator on the ABC" pattern, scoped to one library. Source at https://github.com/nd-ball/py-irt/tree/master/py_irt/models.

### EduCDM (NeuralCD authors' lab, bigdata-ustc)

Houses IRT, MIRT, DINA, NCDM, FuzzyCDF, KaNCD, MCD, IRR, ICD as sibling subpackages. The base class `EduCDM/meta.py` is minimal,

```python
class CDM(object):
    def __init__(self, *args, **kwargs): pass
    def train(self, *args, **kwargs): raise NotImplementedError
    def eval(self, *args, **kwargs):  raise NotImplementedError
    def save(self, *args, **kwargs):  raise NotImplementedError
    def load(self, *args, **kwargs):  raise NotImplementedError
```

No registry decorator. Each model is imported explicitly from its subpackage. Source at https://github.com/bigdata-ustc/EduCDM. The contract is too thin to be useful as a decoder ABC (it does not constrain inputs/outputs of the response function at all), but it is a useful negative example.

### Shared decoder family observation

The four sources agree on the structural family.

- **Polytomous ordinal under one base.** GPCM, GRM, PCM, NRM share the contract "given `theta in R^D`, `alpha in R^D`, and step parameters `beta in R^{K-1}`, return per-category probabilities `(B, T, K)`". `mirt` makes this explicit via the `itemtype` switch. `ma-irt`'s `MIRTGPCMLogits` is already one instance of this contract.
- **CDM under a separate base.** DINA and DINO consume binary skill vectors and a Q-matrix, with guess/slip parameters. The input contract is `(skill_mastery_vec, q_vec) -> P(correct)`, distinct from the IRT contract. EduCDM keeps them as sibling subpackages, not under a common decoder ABC.
- **MIRT as 2PL with vector ability.** Both `mirt` and EduCDM treat MIRT as a "2PL where `theta` is `R^D` and `alpha` is `R^D`", not as a separate model family. This validates `ma-irt`'s existing decision to use `n_traits` as a config toggle on the same code path.

Practical conclusion. One `ResponseDecoder` ABC over polytomous models (GPCM, GRM, PCM, NRM, 2PL/MIRT as the K=2 case) is natural. CDM (DINA, DINO) needs a separate ABC because the input contract is different (skill vector + Q-matrix, not continuous trait). Forcing them under one base will leak abstractions.

## (d) PyTorch registry/factory patterns, ranked for fit

| Pattern | Lines of glue | Discoverability | Type safety | Fit for `ma-irt/` |
|---|---|---|---|---|
| HF `AutoModel` + `PretrainedConfig` | ~200, plus hub assumptions | High | High via config classes | Overkill |
| timm `register_model` decorator | ~50 | High via `list_models` | Loose (functions) | Good |
| torchvision `get_model` | ~30 | High via `list_models` | Loose | Good |
| py-irt `@Base.register(name)` on ABC | ~20, scoped to one ABC | Medium | Strong, ABC enforces methods | Best |
| pyKT `init_model` if/elif chain | ~80, grows linearly | Low, must read source | None | Avoid |
| Hand-rolled dict + explicit import | ~10 | Medium | None | Reasonable fallback |

**Recommendation.** py-irt's pattern, applied twice. One `EncoderBackbone(nn.Module, abc.ABC)` with `@EncoderBackbone.register("dkvmn"|"akt"|"sakt"|"saintpp"|"simplekt")` and one `ResponseDecoder(nn.Module, abc.ABC)` with `@ResponseDecoder.register("gpcm"|"grm"|"pcm"|"2pl"|"mirt")`. CDM gets its own `CognitiveDiagnosisDecoder` ABC if/when DINA lands.

Sketch.

```python
# ma-irt/models/registry.py
class EncoderBackbone(nn.Module, abc.ABC):
    _registry: dict[str, type["EncoderBackbone"]] = {}

    @classmethod
    def register(cls, name):
        def deco(sub):
            cls._registry[name] = sub
            return sub
        return deco

    @classmethod
    def from_name(cls, name, **cfg) -> "EncoderBackbone":
        return cls._registry[name](**cfg)

    @abc.abstractmethod
    def forward(self, q: Tensor, r: Tensor, mask: Tensor) -> Tensor:
        """Return h of shape (B, T, d_model). h[:, t] summarizes info up to step t."""
```

Then `magpcm.py` becomes a thin wrapper that composes a registered encoder with a registered decoder, instead of containing both in one class. Existing `MAGPCM`, `DynamicGPCM`, `StaticGPCM` register as encoders; `MIRTGPCMLogits` registers as a decoder. The current config switch `model_type: "magpcm" | "static_gpcm" | "dynamic_gpcm" | "dkvmn_softmax"` becomes `encoder: str` + `decoder: str` in `ModelConfig`, with the four current strings as preset shortcuts for backward compatibility.

### Why this pattern over the others

1. **Enforces the shape contract via the ABC.** SAKT, AKT, SimpleKT, and DKVMN all emit `(B, T, d_model)`. Encoding that as an abstract method signature catches mistakes at registration time, not at runtime three forward passes later. timm and torchvision do not enforce this because their backbones produce heterogeneous outputs.
2. **Decoder swap is the actual research question.** The PhD work is at the IRT/KT intersection. The decoder ABC is the more important seam (GPCM vs GRM vs MIRT vs DINA), and py-irt's pattern is the only one that demonstrably scales to a sibling family of response models in a research codebase.
3. **Compatible with existing config-driven workflow.** `config/loader.py` already reads `model_type` from YAML. Replacing the if/elif in the model factory with `EncoderBackbone.from_name(cfg.encoder, **kwargs)` is a one-line change at the call site.
4. **Avoids pyKT's anti-pattern.** pyKT's `init_model` is a 40-branch if/elif chain. Each new backbone touches one central file. The decorator-on-ABC pattern keeps each backbone self-contained.

Trade-off acknowledged. The ABC pattern requires every encoder to fit one forward signature `(q, r, mask) -> h`. AKT's monotonic attention needs the question stream separated from the response stream, and SAINT+ needs both as separate inputs. The signature should be `(q, r, mask, **extras)` with `**extras` documented per-encoder, or use a typed batch dataclass like pyKT's `dcur` dict. Recommend the dataclass route, it gives static typing without forcing every encoder to ignore irrelevant kwargs.

## (e) Verified URLs

Every URL below was fetched via WebFetch during this investigation and returned content (200 OK). No 404s encountered.

Knowledge tracing papers.
- https://arxiv.org/abs/1907.06837 -- SAKT (Pandey, Karypis 2019)
- https://arxiv.org/abs/2010.12042 -- SAINT+ (Shin et al. 2021, LAK)
- https://arxiv.org/abs/2007.12324 -- AKT (Ghosh, Heffernan, Lan 2020, KDD)
- https://arxiv.org/abs/2302.06881 -- SimpleKT (Liu et al. 2023, ICLR)
- https://arxiv.org/abs/1908.08733 -- NeuralCD (Wang et al. 2020, AAAI)
- https://arxiv.org/abs/2203.01282 -- py-irt (Lalor, Rodriguez 2022)

Knowledge tracing repositories.
- https://github.com/pykt-team/pykt-toolkit -- pyKT toolkit, canonical implementations
- https://github.com/pykt-team/pykt-toolkit/blob/main/pykt/models/sakt.py -- SAKT
- https://github.com/pykt-team/pykt-toolkit/blob/main/pykt/models/saint.py -- SAINT
- https://github.com/pykt-team/pykt-toolkit/blob/main/pykt/models/saint_plus_plus.py -- SAINT+
- https://github.com/pykt-team/pykt-toolkit/blob/main/pykt/models/akt.py -- AKT (pyKT mirror)
- https://github.com/pykt-team/pykt-toolkit/blob/main/pykt/models/simplekt.py -- SimpleKT
- https://github.com/arghosh/AKT -- AKT authors' release
- https://github.com/arshadshk/SAKT-pytorch -- third-party SAKT port
- https://github.com/shivanandmn/SAINT_plus-Knowledge-Tracing- -- third-party SAINT+ port
- https://github.com/Chang-Chia-Chi/SaintPlus-Knowledge-Tracing-Pytorch -- third-party SAINT+ port
- https://github.com/arshadshk/SAINT-pytorch -- third-party SAINT port

IRT and CDM repositories.
- https://github.com/nd-ball/py-irt -- py-irt
- https://github.com/nd-ball/py-irt/tree/master/py_irt/models -- model directory with `abstract_model.py`
- https://github.com/bigdata-ustc/EduCDM -- EduCDM (IRT, MIRT, DINA, NCDM, ...)
- https://cran.r-project.org/web/packages/mirt/mirt.pdf -- R mirt CRAN PDF
- https://github.com/eribean/girth -- girth (alternate Python IRT)
- https://github.com/cjurban/deepirtools -- deepirtools
- https://github.com/mhw32/variational-item-response-theory-public -- variational IRT
- https://github.com/17zuoye/pyirt -- pyirt (older EM-based)

Registry pattern references.
- https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/_registry.py -- timm registry
- https://pytorch.org/blog/easily-list-and-initialize-models-with-new-apis-in-torchvision/ -- torchvision list_models/get_model
- https://docs.pytorch.org/vision/main/generated/torchvision.models.get_model.html -- torchvision get_model API
- https://huggingface.co/docs/transformers/model_doc/auto -- HF AutoModel docs
- https://huggingface.co/docs/transformers/main_classes/backbones -- HF backbone API
- https://lightning.ai/docs/pytorch/stable/common/lightning_module.html -- LightningModule docs

Not 404, but worth noting.
- `arshadshk/SAKT-pytorch` README references `sakt.py`, but the raw fetch of `https://raw.githubusercontent.com/arshadshk/SAKT-pytorch/master/sakt.py` returned 404. The file lives elsewhere in that repo. Use the pyKT implementation as the canonical SAKT reference.
