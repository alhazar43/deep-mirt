# Subproject Comparison Notes (Deep-MIRT Context)

## Purpose
Summarize theoretical and engineering differences across subprojects so the new MIRT-focused codebase can reuse the right components and avoid incompatible assumptions.

## Deep-GPCM (Primary Reference)
**Theory**: DKVMN + IRT (theta/alpha/beta) + GPCM for polytomous responses. Ordinal loss mix (QWK/ordinal/focal) aligns with ordered categories.
**Engineering**: Modular components for embeddings, memory, IRT layers, and attention. Data loader supports ordinal category sequences. See:
- `deep-gpcm/models/components/memory_networks.py`
- `deep-gpcm/models/components/irt_layers.py`
- `deep-gpcm/models/implementations/deep_gpcm.py`
- `deep-gpcm/data/loaders.py`

## Deep-IRT (deep-1pl)
**Theory**: DKVMN with scalar ability and difficulty for binary outcomes; no explicit ordinal modeling.
**Engineering**: TensorFlow 1.x, tightly coupled model graph, DKVMN memory with linear head. See `deep-1pl/model.py`.

## DKVMN Original (dkvmn-ori)
**Theory**: DKVMN for binary knowledge tracing; logistic output with masking.
**Engineering**: MXNet symbolic graph, scalar prediction head. See `dkvmn-ori/code/python3/model.py`.

## DKVMN PyTorch (dkvmn-torch)
**Theory**: Same DKVMN binary KT formulation; no IRT or ordinal handling.
**Engineering**: PyTorch DKVMN with BCE loss and masking. See `dkvmn-torch/model/model.py`.

## AKT (akt)
**Theory**: Transformer-based KT with attention blocks; no memory network or IRT parameterization.
**Engineering**: PyTorch multi-head attention stack with BCE loss. See `akt/akt.py`.

## Implications for New MIRT Codebase
- Use Deep-GPCM for polytomous IRT/GPCM structure and losses.
- Use DKVMN components for temporal state handling, but replace scalar theta/alpha with D-dimensional traits.
- Treat AKT as attention reference only; it does not align with psychometric parameterization.
- Treat deep-1pl and dkvmn-* as binary KT references only.
