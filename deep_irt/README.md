# deep_irt

Dynamic assessment framework: a swappable sequence encoder plus swappable IRT
decoders, trained end-to-end on a prediction loss. IRT is a readout flavor; the
loss scores decoder logits, not an IRT-likelihood NLL.

## Core classes

`DeepIRTModel` (in `deep_irt/core/model.py`) is the top-level API.

```python
from deep_irt.core import DeepIRTModel

model = DeepIRTModel(num_items=500, decoder="gpcm", encoder="lstm")
model.fit(item_ids, responses, n_epochs=300)          # (N, T) tensors
theta   = model.track(item_ids, responses)            # (N, T) trajectory
params  = model.recover_item_params(item_ids, responses)  # dict of arrays
```

`DeepIRTEngine` wraps `DeepIRTModel` for benchmark runners (see `deep_irt/bench/`).

## Swappable encoders

Pass `encoder=` to select the sequence backbone. All subclass `BaseSeqEncoder`
and expose the same interface (`theta_for_prediction`, `state_for_prediction`,
`aligned_theta_and_state`, `item_val_emb`, `item_key_emb`).

| `encoder=`      | Class                | Notes                                         |
|-----------------|----------------------|-----------------------------------------------|
| `"lstm"`        | `LSTMEncoder`        | Default. Plain `nn.LSTM`.                     |
| `"transformer"` | `TransformerEncoder` | Causal self-attention. Knobs: `n_heads`, `n_layers`, `max_seq_len`, `dropout` (via `encoder_kwargs`). |
| `"dkvmn"`       | `DKVMNEncoder`       | DKVMN-style key-value memory. Knobs: `memory_size`, `key_dim` (via `encoder_kwargs`). |

## Swappable decoders

Pass `decoder=` to select the IRT response model.

| `decoder=`   | Class               | Response format                          | Training loss   |
|--------------|---------------------|------------------------------------------|-----------------|
| `"gpcm"`     | `GPCMDecoder`       | Ordered K categories (default K=4).      | WeightedOrdinalLoss (WOL) |
| `"binary"`   | `Binary2PLDecoder`  | Binary {0, 1}. Thin GPCM wrapper at K=2. | BCE             |
| `"nrm"`      | `NRMDecoder`        | Unordered K options (multiple-choice).   | Cross-entropy (CE) |
| `"bt"`       | `BradleyTerryDecoder` | Pairwise comparisons (no theta).       | BCE on pairs; fit via `fit_pairs()` |

## Prediction-loss training

The framework is prediction-home: the encoder and decoder are trained jointly
to minimize a format-keyed prediction loss on the decoder's logits. No
IRT-likelihood NLL is computed during training.

- GPCM: `WeightedOrdinalLoss` with sqrt-balanced class weights (ma-irt's recipe).
  The `ordinal_penalty` kwarg (default 0.5) controls the ordinal distance term.
- Binary: `binary_cross_entropy_with_logits` on the single 2PL logit.
- NRM: plain cross-entropy (options carry no order).

IRT parameters (alpha, beta, a_k, c_k, strength) are recovered AFTER training
from the frozen decoder weights via `recover_item_params`.

## Decoupled architecture (default for gpcm/binary)

`decouple=True` (the default) activates the validated deep-irt configuration
for the GPCM and binary decoders:

- `state_alpha=True`: discrimination is read from a state-conditioned head
  `fc_a_state([state, item_key])`, where `state` is the prediction-aligned
  LSTM hidden (the same causal state that produces theta). Occurrence-averaged
  per item at recovery (ma-irt's IRTParameterExtractor recipe).
- `item_key_dim=64`: a separate wide item KEY table (`item_key_emb`, 64-wide)
  feeds the static alpha and beta readouts. The thin VALUE table (`item_val_emb`,
  `emb_dim`-wide) feeds the encoder/theta only. Width follows inverse Fisher:
  alpha is low-information and capacity-hungry, theta is well-determined and is
  hurt by a fat encoder input.
- `alpha_log_scale=1.0`: the exp positivity transform for discrimination (ma-irt's
  convention; the MLP head absorbs the scale).

Pass `decouple=False` for the plain (legacy) item-only static decoder. `decouple`
is a no-op for `"nrm"` and `"bt"`.

## Single-shift causal alignment

The encoder is run ONCE per forward pass. The ability used to predict the
response at step t is always a function of the interaction history strictly
before t (the single-shift alignment contract). State-conditioned discrimination
reads the SAME causal hidden, so there is no second encoder stream.

## Anchored extension

`model.extend(n_ext, anchor_theta, ext_item_ids, ext_responses)` adds new item
embeddings while freezing the encoder and base decoder. Supported for `"gpcm"`
and `"binary"` with `encoder="lstm"`. See `deep_irt/core/anchor.py`.

## Key files

| Path | Purpose |
|------|---------|
| `deep_irt/core/model.py` | `DeepIRTModel` (encoder + decoder + training loop) |
| `deep_irt/core/encoder.py` | `BaseSeqEncoder`, `LSTMEncoder` (item_val_emb / item_key_emb, causal shift) |
| `deep_irt/core/decoders.py` | `GPCMDecoder`, `Binary2PLDecoder`, `NRMDecoder`, `BradleyTerryDecoder` |
| `deep_irt/core/anchor.py` | Anchored extension (freeze + fit new embeddings) |
| `deep_irt/core/losses.py` | `CombinedLoss` (WeightedOrdinalLoss + CE) |
| `deep_irt/bench/` | Benchmark runners (`DeepIRTEngine`, synthetic + real-data probes) |
| `deep_irt/tests/` | 139+ tests (causal alignment, decoupled alpha, decoder contracts, encoder swap) |

## Further reading

- `docs/LEARNING_DYNAMICS_STUDY.md` -- the Fisher-conditioning study: why
  decoupling alpha from theta buys a convergence-rate advantage on alpha rank
  recovery, and how the advantage scales with K.
- `deep_irt/RESULTS.md` -- the experiment ledger (anchoring, dynamic theta,
  multi-format decoders, SLAM real-data, EdNet separability).
