# MIRT Engineering Notes for Deep-GPCM

## Files and Responsibilities
- `deep-gpcm/models/components/irt_layers.py`: extend parameter extractor and GPCM logits for multi-dim theta and alpha.
- `deep-gpcm/models/implementations/deep_gpcm.py`: wire trait dimension and item/KC mapping into forward pass.
- `deep-gpcm/models/components/embeddings.py`: optional KC-aware embedding or feature augmentation.
- `deep-gpcm/data/loaders.py`: load KC metadata and batch it alongside questions/responses.
- `deep-gpcm/utils/data_utils.py`: validate KC shapes and consistency.
- `deep-gpcm/config/*.py`: add `n_traits`, `kc_dim`, `kc_mapping` parameters.

## Proposed Module Changes

### 1) IRT Parameter Extractor
**Current**: scalar theta, scalar alpha, vector beta.
**Target**: vector theta (D), vector alpha (D), vector beta (K-1).

- Ability network: `Linear(input_dim, D)`
- Discrimination network: `Linear(input_dim + question_dim, D)`
- GPCM logits: use dot product `a_j^T theta_t`.

### 2) KC-to-Trait Mapping
Add a module `KCMapping`:
- Input: `kc_vector` (C dims)
- Output: `a_j` (D dims)
- Options: `linear`, `linear+relu`, `linear+softplus` (for non-negativity)

### 3) Model Forward Flow
- Build or load `kc_vector` per item.
- Obtain `a_j` from KC mapping (or from item embedding if no KC data).
- Compute theta_t from summary network.
- Compute logits via MIRT-GPCM rule.

### 4) Data Pipeline
- Extend dataset metadata format to include KC data.
- Allow either:
  - `item_kc_matrix.json` (shape: n_items x n_kc)
  - `item_kc_ids.json` (list of KC indices per item)
- Batch KC features based on item IDs.

### 5) Backward Compatibility
- If `n_traits == 1`, fall back to scalar behavior.
- If KC metadata missing, use learned item embeddings for a_j.

## Minimal API Sketch
```python
# in IRTParameterExtractor
self.ability_network = nn.Linear(input_dim, n_traits)
self.discrimination_network = nn.Linear(input_dim + question_dim, n_traits)

# in GPCMProbabilityLayer
logit_k = sum_{h<k} (a_dot_theta - beta_h)
```

## Validation Strategy
- Add a tiny synthetic dataset with known KC->trait mapping.
- Run `python deep-gpcm/train.py --model deep_gpcm --dataset synthetic_...`
- Verify logits shape, loss stability, and ordinal metrics.
