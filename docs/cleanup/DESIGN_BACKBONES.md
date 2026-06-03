# Design 2, backbone-swap targets, 2026-06-03

Companion to `DESIGN_MODULAR.md` (Encoder ABC, written in parallel).
This document covers `SAKTEncoder`, `SAINTPlusEncoder`, `AKTEncoder`
under the new `Encoder` ABC contract `forward(q, r, ...) -> dict`
with `h_t` as the per-step contextual hidden state.

References used for every design choice come from
`INVESTIGATION_SOTA.md` section (a) and the canonical pyKT files
(MIT licensed) at `https://github.com/pykt-team/pykt-toolkit/blob/main/pykt/models/`.

Source of the equivalence target, `BASELINE_2026-06-02.md` section 4.2.
Config inventory for cost, `ma-irt/configs/bulk/bench_magpcm_static_q200_k4_pykt_fold0.yaml`.

Inventory and design only. No code is modified by this round.

---

## 1. SAKTEncoder, arXiv 1907.06837

### Layer composition

Pure self-attention encoder over the response stream. Two-stream input
embedding, the exercise embedding queries against a key-value
projection of the interaction history. Stacked attention blocks at the
end of the encoder, output is the contextual `h_t` per timestep.

Composition follows `pykt/models/sakt.py` (`SAKT`, `Blocks`).

1. Interaction embedding `E_x in R^{2Q x d_model}`, indexed by `2*q + r`
   for binary KT, generalised below for K-ary GPCM. `nn.Embedding(2*n_q+1, d_model, padding_idx=0)`.
2. Exercise embedding `E_e in R^{Q x d_model}` for the query stream.
   `nn.Embedding(n_q+1, d_model, padding_idx=0)`.
3. Learned positional embedding `P in R^{T_max x d_model}`. SAKT uses
   absolute learned positions, not sinusoidal. Match pyKT.
4. `num_en` self-attention `Blocks` of `n_heads=8`, FFN dim `d_model`,
   `dropout=0.2`. Causal mask, key padding mask from `q==0`.
5. Final post-attention representation `xemb in R^{B x T x d_model}`.
   In pyKT this feeds a single Linear plus sigmoid, but for MA-IRT the
   GPCM decoder consumes `xemb` directly.

### Polytomous adaptation

SAKT was published for binary KT. For K-ary GPCM the interaction
index becomes `K*(q-1) + r + 1`, embedding table size `K*Q + 1` with
the same `padding_idx=0`. This matches MA-GPCM's `OneHotEmbedding`
indexing scheme for `K*Q` and is the minimal change to keep the
exercise stream intact.

### Class header

```python
class SAKTEncoder(Encoder):
    """Self-attention KT backbone, polytomous-ready.

    Args:
        n_questions: item bank Q.
        n_categories: ordinal K, used to size the interaction table.
        d_model: hidden size d, output is (B, T, d_model).
        n_heads: attention heads in each block, 8 in the paper.
        n_layers: stacked attention blocks (num_en), 2 in the paper.
        seq_len: maximum sequence length T_max for learned positions.
        dropout: dropout rate inside each block, 0.2 in the paper.
    """
    h_dim: int  # = d_model, advertised to the decoder

    def __init__(
        self,
        n_questions: int,
        n_categories: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        seq_len: int = 200,
        dropout: float = 0.2,
    ) -> None: ...

    def forward(
        self,
        q: Tensor,   # (B, T) long, item ids in [1, Q], 0 = pad
        r: Tensor,   # (B, T) long, ordinal responses in [0, K-1]
        mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        # h_t shape (B, T, d_model). h_t[:, t] summarises responses 1..t
        # under causal self-attention. Decoder reads h_t at step t.
        return {"h_t": xemb, "q_embed": q_e, "mask": pad_mask}
```

### Expected `h_t`

Shape `(B, T, d_model)`. Causal contextual embedding, position `t`
attends to positions `<= t`. Semantically equivalent to DKVMN's
read vector `read_t`, indexed at the same granularity. No item key
is concatenated, the GPCM decoder still owns the `q_embed` channel
for beta and alpha.

---

## 2. SAINTPlusEncoder, arXiv 2010.12042

### Layer composition

Encoder-decoder Transformer. Encoder consumes the exercise stream,
decoder consumes the response stream and cross-attends to the
encoder output. SAINT+ adds elapsed-time and lag-time features to
the response stream over the original SAINT.

Composition follows `pykt/models/saint_plus_plus.py` (`SAINTPlus`).

1. Encoder input, exercise stream
   - `E_q in R^{Q x d_model}`, item embedding.
   - `E_c in R^{C x d_model}`, KC embedding, optional and zeroed when
     KCs are not provided (synthetic GPCM has none, so this is zero).
   - Positional encoding, sinusoidal as in the original Transformer.
   - `n_blocks` standard Transformer encoder layers, `n_heads=8`,
     FFN dim `4 * d_model`, `dropout=0.2`.
2. Decoder input, response stream
   - `E_r in R^{K x d_model}`, response embedding. K is the ordinal
     bucket count, generalises the binary embedding.
   - SAINT+ extras, elapsed time and lag time projected to `d_model`
     and added. Not present in synthetic data, the projection is
     gated by `cfg.use_time_features`.
   - Positional encoding, sinusoidal.
3. `n_blocks` decoder layers, each with self-attention over the
   response stream, cross-attention to encoder output, FFN.
4. Decoder output `d_out in R^{B x T x d_model}`. In pyKT this feeds a
   sigmoid head. For MA-IRT the GPCM decoder consumes `d_out`.

### Encoder-decoder split versus the MA-IRT Encoder ABC

SAINT+ has its own internal encoder-decoder split. Important
clarification, the MA-IRT `Encoder` ABC is one abstraction layer
above SAINT+. The whole SAINT+ encoder-decoder stack lives inside
`SAINTPlusEncoder`, and the MA-IRT `Decoder` (the GPCM head) takes
over from SAINT+'s decoder output. In effect

```
SAINT+'s encoder -+
                  +-- SAINTPlusEncoder.forward returns d_out as h_t
SAINT+'s decoder -+
                                      |
                                      v
                          MA-IRT GPCMDecoder consumes h_t
                            (replaces SAINT+'s sigmoid head)
```

This keeps the SAINT+ cross-attention intact, the MA-IRT decoder
just substitutes a polytomous GPCM head for the binary sigmoid.

### Class header

```python
class SAINTPlusEncoder(Encoder):
    """Encoder-decoder Transformer KT backbone, SAINT+ (LAK 2021).

    The MA-IRT GPCMDecoder replaces SAINT+'s final sigmoid head.
    """
    h_dim: int  # = d_model

    def __init__(
        self,
        n_questions: int,
        n_categories: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_blocks: int = 4,
        seq_len: int = 200,
        dropout: float = 0.2,
        use_time_features: bool = False,
    ) -> None: ...

    def forward(
        self,
        q: Tensor,            # (B, T) long
        r: Tensor,            # (B, T) long, polytomous
        mask: Tensor | None = None,
        elapsed: Tensor | None = None,  # (B, T) float, SAINT+ only
        lag: Tensor | None = None,      # (B, T) float, SAINT+ only
    ) -> dict[str, Tensor]:
        # h_t shape (B, T, d_model), decoder output of SAINT+
        return {"h_t": d_out, "q_embed": e_q, "mask": pad_mask}
```

### Expected `h_t`

Shape `(B, T, d_model)`. Same contract as SAKT, the decoder reads
position `t` as the prediction for response `t`. Note the SAINT+
decoder is already causal (shifted self-attention plus cross-attention
to a causal encoder output), so no extra masking is required at the
MA-IRT decoder boundary.

### Open question 1, time features

`scripts/data_gen.py` does not emit `elapsed` or `lag`. Two options
for the equivalence run

- A. Force `use_time_features=False` on synthetic data. SAINT+
  collapses to SAINT plus the K-ary response embedding. This is the
  cleanest equivalence target.
- B. Synthesise `lag = 1.0` and `elapsed = mean_value` constants.
  Embedding still trains but adds zero information.

Recommendation, A. Document in the smoke config header.

---

## 3. AKTEncoder, arXiv 2007.12324

### Layer composition

Two-block architecture, monotonic distance-decay attention. The
encoder produces `d_output` of shape `(B, T, d_model)`, and a
Rasch-style item reparameterisation feeds both the encoder and the
downstream head.

Composition follows `pykt/models/akt.py` (`AKT`, `Architecture`,
`MonotonicMultiheadAttention`).

1. Question concept embedding `c_embed in R^{C x d_model}`. For
   synthetic GPCM where there is one KC per item, `C = Q`.
2. Item pid embedding `pid_embed in R^{Q x d_model}` plus a
   per-question variation vector `q_embed_diff in R^{Q x d_model}`.
   The Rasch reparameterisation is
   `q_embed_data = c_embed + pid_embed * q_embed_diff`,
   computed inside the encoder.
3. Interaction embedding, similar reparameterisation
   `qa_embed_data = qa_embed + pid_embed * qa_embed_diff`.
4. `blocks_1`, n_blocks of self-attention over the interaction
   stream `qa_embed_data` with monotonic attention.
5. `blocks_2`, n_blocks of monotonic cross-attention from
   `q_embed_data` queries to `blocks_1` keys and values, masked so
   that step `t` cannot peek at the current response.
6. `d_output in R^{B x T x d_model}`, the final block_2 output.

### Rasch embedding handoff to MA-IRT decoder

This is the architecturally interesting decision and the part that
`INVESTIGATION_SOTA.md` flagged as decoder-side. The Rasch
reparameterisation produces a per-step item embedding that is
already a learned analogue of the IRT 1PL `theta - b` form.
`d_output` carries the latent state that, in pyKT, is concatenated
with `q_embed_data` and fed to a 2-layer MLP for the binary head.

Two design choices for MA-IRT.

- A. **Keep Rasch inside the encoder.** Treat the Rasch part as an
  AKT internal, expose only `d_output` as `h_t`. The MA-IRT GPCM
  decoder builds alpha and beta from its own `q_embed` channel,
  unaware of the AKT Rasch reparameterisation. Pro, drop-in encoder,
  decoder code unchanged. Con, the AKT Rasch terms duplicate the
  MA-IRT GPCM alpha and beta, double-parameterising the item.
- B. **Surface the Rasch embedding to the decoder.** The encoder
  emits both `h_t = d_output` and `q_embed_rasch = q_embed_data`
  (the Rasch-reparameterised item embedding). The MA-IRT GPCM
  decoder consumes `q_embed_rasch` instead of a fresh `nn.Embedding`,
  reusing the AKT learned item representation. Pro, no
  double-parameterisation, beta and alpha read from the AKT-tuned
  embedding. Con, the GPCM decoder needs to accept an external
  `q_embed` channel and skip its own embedding when present.

Recommendation, **B**. The MA-IRT GPCM decoder already separates
`q_embed` from the read summary (see `models/components/irt.py`
`IRTParameterExtractor`, `threshold` and `discrimination_network`
both consume `question_dim`). The decoder ABC already needs to
accept an item embedding, AKT just supplies a richer one. The
encoder dict returns `{"h_t", "q_embed", "mask"}` and the decoder
prefers `q_embed` when present over its internal table.

This decision propagates into the modular design. Flag it in
`DESIGN_MODULAR.md`, the `Encoder` ABC's return dict needs `q_embed`
as a first-class optional field, not a private hook.

### Class header

```python
class AKTEncoder(Encoder):
    """Context-aware monotonic attention KT (AKT, KDD 2020).

    Rasch reparameterised item embedding is surfaced so the GPCM
    decoder can reuse it (recommendation B above).
    """
    h_dim: int  # = d_model

    def __init__(
        self,
        n_questions: int,
        n_categories: int,
        n_pid: int | None = None,  # set to n_questions when one KC per item
        d_model: int = 256,
        n_blocks: int = 1,
        n_heads: int = 8,
        d_ff: int = 1024,
        kq_same: bool = True,
        dropout: float = 0.05,
        separate_qa: bool = False,
    ) -> None: ...

    def forward(
        self,
        q: Tensor,
        r: Tensor,
        mask: Tensor | None = None,
        pid: Tensor | None = None,  # (B, T), defaults to q
    ) -> dict[str, Tensor]:
        # h_t = d_output, shape (B, T, d_model)
        # q_embed = q_embed_data, Rasch-reparameterised item embed
        return {"h_t": d_output, "q_embed": q_embed_data, "mask": pad_mask}
```

### Expected `h_t`

Shape `(B, T, d_model)`. Position `t` summarises the response
history up to (and excluding) step `t` under monotonic attention.
The accompanying `q_embed_data` at position `t` is the
Rasch-reparameterised item embedding for the item at `t`, shape
`(B, T, d_model)`.

### Open question 2, AKT L2 on the Rasch variation

`akt.py` adds a small L2 penalty (`l2=1e-5`) on the Rasch variation
parameters during training. In pyKT this is exposed via
`forward(..., target=...) -> (loss, preds, valid_count)`. The MA-IRT
trainer does not expect the encoder to contribute to the loss.

Resolution, **lift the L2 into the MA-IRT training loop**. The
encoder exposes the relevant parameters via a property
`akt_l2_params: list[nn.Parameter]`, and `training/trainer.py`
adds `cfg.training.akt_rasch_l2 * sum(p.norm() for p in params)`
to the total loss when the encoder is AKT. Default 1e-5 to match
pyKT.

---

## 4. Mapping `h_t` to the MA-IRT decoder

Summary of `h_t` shape, semantics, and whether a projection layer is
needed before the GPCM decoder.

The current MA-IRT GPCM decoder (`models/components/irt.py`,
`IRTParameterExtractor`) takes `input_dim = summary_dim` (default
50) for the ability network and `input_dim + question_dim` for the
discrimination network. Each new encoder produces `h_t` with
`d_model` channels, which is typically larger than `summary_dim`.

Two acceptable strategies.

- **Strategy P (projection).** Add a thin `nn.Linear(d_model, summary_dim)`
  inside the encoder wrapper, advertise `summary_dim` as the public
  `h_dim`. Pro, decoder unchanged, registry contract unaltered. Con,
  loses information when `d_model > summary_dim`, an extra linear
  layer per encoder.
- **Strategy D (rebuild decoder for d_model).** Construct the GPCM
  decoder with `input_dim = d_model` so the IRT sub-networks fit the
  encoder output directly. Pro, no information loss, no extra layer.
  Con, the decoder's input_dim becomes encoder-dependent, breaks
  the "decoder is built once for the dataset" pattern.

Recommendation, **D for all three new encoders**. The MA-IRT
decoder already accepts `input_dim` as a constructor argument, so
the wrapper just builds the decoder with `input_dim = encoder.h_dim`.
This keeps the decoder lossless and avoids burying a Linear inside
each encoder.

Per-backbone summary.

| Encoder | h_t shape | h_dim | q_embed surfaced | Projection layer | Decoder input_dim |
|---|---|---|---|---|---|
| `DKVMNEncoder` (existing) | `(B, T, value_dim)` | `value_dim`, default 64 | yes, from `nn.Embedding(Q+1, key_dim)` | no | `summary_dim`, summary network sits inside the wrapper, contract preserved |
| `SAKTEncoder` | `(B, T, d_model)` | `d_model`, default 256 | yes, from `E_e` | none, decoder built with `input_dim=d_model` | `d_model` |
| `SAINTPlusEncoder` | `(B, T, d_model)` | `d_model`, default 256 | yes, from encoder side exercise embedding | none, decoder built with `input_dim=d_model` | `d_model` |
| `AKTEncoder` | `(B, T, d_model)` | `d_model`, default 256 | yes, Rasch reparameterised, see section 3 | none, decoder built with `input_dim=d_model` | `d_model`, decoder reads AKT's `q_embed_data` for alpha and beta |

DKVMN keeps its current summary network as part of its own
encoder wrapper, so it still exposes `h_dim = summary_dim = 50`.
The three Transformer backbones expose `h_dim = d_model` directly,
no internal summary FC. The decoder is built per-encoder by passing
`encoder.h_dim` to `IRTParameterExtractor`.

### Projection layer signatures

If reviewers prefer Strategy P after all, the signature is identical
per encoder.

```python
class _OutputProjection(nn.Module):
    """Optional projection from encoder d_model to decoder summary_dim."""
    def __init__(self, d_model: int, summary_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, summary_dim)
        self.act = nn.Tanh()
    def forward(self, h: Tensor) -> Tensor:
        # h: (B, T, d_model), returns (B, T, summary_dim)
        return self.act(self.proj(h))
```

Adding this is a one-line change per encoder, so the design is
reversible.

---

## 5. Verification plan, MA-GPCM equivalence

The recovery hill, MA-GPCM K=4 Synthetic-Static (Table 1 ACC, QWK,
MAE plus Table 3 r_alpha, r_beta, r_theta), is the gate.

### Baseline config

`ma-irt/configs/bulk/bench_magpcm_static_q200_k4_pykt_fold0.yaml`
trains `MAGPCM` with `embedding_type=learned`, `value_dim=64`,
`summary_dim=50`, on `data/v2_q200_k4`. The baseline metric row
from `BASELINE_2026-06-02.md` section 4.2, K=4, 5-fold mean.

| Metric | Baseline mean | sigma | Acceptance tolerance, 0.5% rule |
|---|---|---|---|
| ACC | 52.7843 | 0.0882 | 52.78 +/- 0.264 |
| QWK | 0.6806 | 0.0005 | 0.6806 +/- 0.00341 |
| MAE | 0.6074 | 0.0011 | 0.6074 +/- 0.00304 |
| r_alpha | 0.8938 | 0.0087 | 0.8938 +/- 0.009 |
| r_beta | 0.9675 | 0.0019 | 0.9675 +/- 0.00484 |
| r_theta | 0.9574 | 0.0005 | 0.9574 +/- 0.00479 |

Tolerance bracket per metric is `effective_tol = max(0.005 * |mean|, sigma_published)`,
matching `BASELINE_2026-06-02.md`.

### Per-backbone verification

For each of `SAKTEncoder`, `SAINTPlusEncoder`, `AKTEncoder`,
construct a paired smoke + equivalence config under
`ma-irt/configs/bulk/bench_<encoder>_static_q200_k4_pykt_fold0.yaml`.
Each config swaps the encoder while keeping

- The dataset (`data/v2_q200_k4`).
- The fold split (fold 0 from `cv.split_seed=42`).
- The decoder (`GPCMDecoder`).
- The loss (WOL with `weighted_ordinal_weight=1.0`,
  `ordinal_penalty=0.5`).
- The optimiser (Adam, lr 1e-3, ReduceLROnPlateau on QWK).
- Total epochs (200) and grad clip (1.0).

The equivalence test runs `scripts/train.py` then `scripts/evaluate.py single`
and asserts each of the 6 metrics above is within its tolerance band.

Equivalence here is a research target, not a strict gate. Transformer
encoders are not expected to reach exact DKVMN numbers, the goal is
to confirm the MA-IRT decoder still recovers IRT parameters under a
different upstream representation. Document the result for each
backbone, flag any metric outside the tolerance, but do not block the
P3 phase on a SAKT vs DKVMN dead-tie.

Note. `PIPELINE_OPT_PLAN.md` section P3 already loosens the tolerance
for the recovery row to `max(0.05, 5 * effective_tol)`. The 0.5%
gate in this section is the strict version, used only to document
how close each backbone gets.

### Synthetic-only constraint

All three equivalence runs use `data/v2_q200_k4` so ground-truth
alpha, beta, theta are available. ASSIST2009 has no ground truth
and is excluded from the equivalence gate. ASSIST2009 retraining
under the new encoders is a separate exercise, gated on ACC and AUC
only.

---

## 6. Cost estimate

Baseline K=4 fold0 config trains for 200 epochs with batch size 64
on `data/v2_q200_k4` (Q=200, K=4, 5000 students, max_seq_len 200,
chunk_long_sequences true). MA-GPCM with `value_dim=64`,
`summary_dim=50`, learned embeddings sits at roughly the lower end
of the Transformer cost curve.

Cost reference points

- MA-GPCM K=4 fold0, 200 epochs. Observed sub-30 min GPU per
  PIPELINE_OPT_PLAN.md section P3, "5 to 30 min GPU per fold0".
- Parameter count roughly, key_dim 64 + value_dim 64 + summary 50,
  plus IRT extractor, around 0.5 M parameters.

Per-backbone estimate for the K=4 fold0 equivalence run with the
config presets in this design.

| Encoder | d_model | n_blocks/layers | Param count, rough | Time per epoch vs MA-GPCM | Total per fold0, GPU |
|---|---|---|---|---|---|
| DKVMN (baseline) | n/a | n/a | ~0.5 M | 1.0x | 5 to 30 min |
| `SAKTEncoder` | 256 | 2 | ~2.0 M | 1.5x to 2.0x | 10 to 60 min |
| `SAINTPlusEncoder` | 256 | 4 enc, 4 dec | ~5.0 M | 3.0x to 4.0x | 20 to 120 min |
| `AKTEncoder` | 256 | 1 + 1 monotonic blocks | ~2.5 M | 2.0x to 2.5x | 15 to 80 min |

The Transformer backbones have heavier per-step compute but the
sequence is short (max 200), so the quadratic attention cost is
manageable. SAINT+ is the slowest because of the dual-stream block
count.

Total cost for the K=4 fold0 sweep across all three new backbones,
roughly 45 to 260 GPU minutes. Plan budget, one half-day GPU.

If the equivalence test is extended to the full 5 folds, multiply by
5. Recommendation, run fold0 only at first, gate the 5-fold sweep on
fold0 success.

---

## 7. Open questions

1. **Polytomous SAKT.** No canonical implementation. Recommend
   `K*(q-1) + r + 1` interaction indexing, matches MA-GPCM's
   `OneHotEmbedding`. Needs a one-line note in the smoke config.
2. **SAINT+ time features on synthetic data.** Recommend
   `use_time_features=False` for the equivalence run, see section 2
   open question 1.
3. **AKT Rasch handoff strategy A versus B.** Recommend B (surface
   `q_embed_data` to the decoder). Strategy A is the safer
   drop-in, but it double-parameterises the item. Decision needs
   sign-off before P3 lands.
4. **AKT L2 on Rasch variation.** Recommend lifting the 1e-5 L2 into
   the trainer via `encoder.akt_l2_params`, see section 3 open
   question 2.
5. **Decoder input_dim policy.** Recommend Strategy D, build the
   decoder with `input_dim = encoder.h_dim`. Strategy P (projection)
   is the reversible fallback.
6. **Cost extrapolation to ASSIST2009.** ASSIST2009 has longer
   sequences and the attention cost scales quadratically. None of
   the three Transformer backbones has been profiled at that scale
   inside this repo, the section 6 numbers are synthetic-only.

---

## Files this document points to (inventory)

Reads only.

- `INVESTIGATION_SOTA.md` section (a), four canonical pyKT references.
- `PIPELINE_OPT_PLAN.md` section P3.
- `BASELINE_2026-06-02.md` sections 3 and 4.2.
- `ma-irt/configs/bulk/bench_magpcm_static_q200_k4_pykt_fold0.yaml`.
- `ma-irt/models/magpcm.py`.
- `ma-irt/models/components/irt.py` (`IRTParameterExtractor`).

To be added in P3 (new, designed here, not written).

- `ma-irt/models/encoders/sakt.py`.
- `ma-irt/models/encoders/saint_plus.py`.
- `ma-irt/models/encoders/akt.py`.
- `ma-irt/configs/bulk/bench_sakt_static_q200_k4_pykt_fold0.yaml`.
- `ma-irt/configs/bulk/bench_saintpp_static_q200_k4_pykt_fold0.yaml`.
- `ma-irt/configs/bulk/bench_akt_static_q200_k4_pykt_fold0.yaml`.
- `ma-irt/configs/smoke/{sakt,saintpp,akt}_gpcm_smoke.yaml`.
- `ma-irt/tests/test_encoder_swap.py`.

No file is written or moved by this document.
