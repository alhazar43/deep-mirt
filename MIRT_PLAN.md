# MIRT Extension Plan (Deep-GPCM)

## Goal
Extend Deep-GPCM into a multi-dimensional DKVMN + polytomous MIRT system that models psychometric states (latent traits) instead of scalar knowledge states, with a principled KC-to-trait mapping.

## Scope
- Primary: `deep-gpcm/`
- Reference only: `akt/`, `deep-1pl/`, `dkvmn-ori/`, `dkvmn-torch/`

## Core Deliverables
1. Multi-dimensional IRT parameterization (theta/alpha vectors) in the IRT layers.
2. KC-to-trait mapping mechanism (linear or learned), linked to item parameters.
3. Data pipeline support for KC features or item-to-KC mappings.
4. Configurable trait dimensionality and mapping strategy.
5. Training/evaluation compatibility with existing ordinal losses.

## High-Level Steps
1. **IRT math update**: Replace scalar ability/discrimination with D-dimensional traits and dot-product ability. Update GPCM logits accordingly.
2. **Model wiring**: Inject trait dimensionality and mapping into `DeepGPCM` and attention variants.
3. **Data schema**: Add KC metadata loading and batching.
4. **Config & CLI**: Expose `n_traits`, `kc_dim`, `kc_mapping` parameters.
5. **Validation**: Add small synthetic or toy dataset with KC features to ensure end-to-end flow.

## Risks & Mitigations
- **Instability from extra degrees of freedom**: add optional norm constraints or weight decay for KC->trait mappings.
- **Data missing KC info**: allow fallback to identity mapping or learnable item embeddings.
- **Ordinal loss compatibility**: keep logits shape consistent (`batch x seq x K`).

## Definition of Done
- End-to-end training on a synthetic dataset with KC mappings.
- Clear module boundaries for trait extraction vs. KC mapping.
- Minimal changes required for existing models to run with `n_traits=1`.
