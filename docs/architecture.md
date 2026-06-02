# MA-GPCM Architecture

This document explains the model structure that cleanup work must preserve.
It is grounded in `ma-irt/models/magpcm.py` and the component modules under
`ma-irt/models/components/`.

## Core Claim

MA-GPCM is not just "DKVMN plus a classifier." It is a memory-augmented
ordinal knowledge tracing model that exposes both:

- prediction outputs: category logits and probabilities
- measurement outputs: `theta`, `alpha`, and `beta`

The paper-critical intervention is the separated ability pathway:

```text
separate_theta = true   -> MA-GPCM
separate_theta = false  -> DKVMN+GPCM ablation
```

When `separate_theta` is true, ability is estimated from the DKVMN read
state only. Item identity is excluded from the ability summary. Item
parameters remain item-conditioned. This is the inductive bias that protects
IRT parameter recovery.

## Input Tensors

`MAGPCM.forward(questions, responses)` receives:

```text
questions: LongTensor, shape (B, S)
responses: LongTensor, shape (B, S)
```

Where:

- `B` is batch size
- `S` is sequence length after padding
- question ID 0 is padding/unknown
- real question IDs are 1-based
- response categories are integers in `0..K-1`

The model returns:

```text
theta:  (B, S, D)
alpha:  (B, S, D)
beta:   (B, S, K-1)
logits: (B, S, K)
probs:  (B, S, K)
```

## Encoder

The encoder combines item embeddings, response/value embeddings, and a
Dynamic Key-Value Memory Network.

### Item Query Embedding

The item ID `q_t` is embedded by:

```text
self.q_embed: Embedding(n_questions + 1, key_dim, padding_idx=0)
```

The resulting `q_embed` is used for:

- DKVMN attention over key memory
- item-conditioned IRT parameter extraction
- threshold (`beta`) projection

### Response/Value Embedding

The observed pair `(q_t, r_t)` is encoded into a value vector `v_t`.
Supported embedding modes are:

- `onehot`: one-hot ordinal item-response encoding, projected to value dim
- `learned`: learned item embedding plus ordinal response feature
- `static_item`: static item-response embedding that directly emits value dim

The value embedding is what gets written into dynamic value memory after the
model reads the previous state.

### DKVMN Memory

Implemented in `models/components/memory.py`.

The memory has:

- static key memory: learned, shared across sequences
- dynamic value memory: initialized per batch and updated through time

At each timestep:

```text
attention_t = attention(q_embed_t)
read_t      = read(value_memory_t, attention_t)
value_memory_{t+1} = write(value_memory_t, attention_t, value_embed_t)
```

The implementation precomputes attention for the full sequence because
attention depends on question embeddings and static key memory, not on the
dynamic value memory. The value-memory read/write loop remains causal.

## Causal Convention

For each timestep `t`, the model:

1. reads the memory state before writing the current response
2. estimates `theta_t`, `alpha_t`, and `beta_t`
3. computes GPCM logits/probabilities
4. writes the current response embedding into memory

This read-before-write convention means timestep outputs are based on the
student state before incorporating the current observed response.

## Decoder

The decoder converts the memory read and item embedding into IRT parameters
and then into GPCM logits.

### Ability Summary

```text
ability_summary_t = ability_summary(read_t)
```

This path receives only `read_t`. It does not receive item identity.

When `separate_theta` is true:

```text
theta_t = ability_network(ability_summary_t) * ability_scale
```

This is the MA-GPCM path.

### Item Summary

```text
summary_t = summary(concat(read_t, q_embed_t))
```

This path receives the student state and item identity. It is used for
item-conditioned discrimination.

### Shared-Path Ablation

When `separate_theta` is false:

```text
theta_t = ability_network(summary_t) * ability_scale
```

This is the DKVMN+GPCM ablation. It can still predict ordinal responses, but
the ability estimator now sees item identity through `summary_t`. The paper
uses this contrast to show that prediction quality alone is not enough:
parameter recovery depends on the separated pathway.

### Discrimination

Discrimination is positive:

```text
raw_alpha_t = discrimination_network(concat(summary_t, q_embed_t))
alpha_t = exp(raw_alpha_t)
```

The current MA-GPCM implementation applies the exponential directly inside
`MAGPCM.forward`.

### Thresholds

Thresholds are projected from item embeddings:

```text
beta = threshold(q_embed)
```

They are computed for the whole sequence before the causal loop because they
depend on item identity, not on dynamic memory state.

## GPCM Head

For category 0, the cumulative logit baseline is 0.

For categories `1..K-1`, the model accumulates step values:

```text
interaction_t = dot(alpha_t, theta_t)
alpha_norm_t  = norm(alpha_t)
step_h        = interaction_t - alpha_norm_t * beta_h
logit_k       = cumulative_sum(step_1 ... step_k)
```

For one trait (`D=1`), this reduces to the standard scalar GPCM form:

```text
sum_h alpha * (theta - beta_h)
```

`models/heads/gpcm.py` then applies softmax over the category dimension to
produce probabilities.

## Model Family Semantics

| Model | Encoder | Decoder/head | Interpretable recovery? |
|---|---|---|---|
| MA-GPCM | DKVMN | separated theta + GPCM | yes |
| DKVMN+GPCM | DKVMN | shared theta + GPCM | yes, ablation |
| DKVMN+Softmax | DKVMN | K-way softmax | no |
| Static GPCM | none/retrieval | static IRT parameters | yes |
| Dynamic GPCM | recurrent theta | GPCM | yes |
| DKT/DKVMN/Deep-IRT | binary KT | binary prediction | no |

Prediction-only baselines may return placeholder parameter fields internally
for trainer compatibility. Those placeholders are not measurement outputs.

## Cleanup Constraints

Any cleanup touching these surfaces must run modular tests and, if behavior
changes are possible, a prediction/recovery check:

- `models/magpcm.py`
- `models/components/memory.py`
- `models/components/irt.py`
- `models/heads/gpcm.py`
- model construction in `scripts/train.py` and `scripts/evaluate.py`
- data-loader shape, mask, and metadata synchronization behavior

Performance-sensitive invariant:

```text
MA-GPCM (`magpcm`, `separate_theta: true`) must not lose paper-level
prediction or recovery performance.
```

Docs-only cleanup does not require benchmark reruns. Core-code cleanup does.
