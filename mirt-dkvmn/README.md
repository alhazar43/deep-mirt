# MIRT-DKVMN (Skeleton)

This is a fresh, production-oriented skeleton for a multidimensional DKVMN model with polytomous MIRT heads. It is designed to extend ideas from Deep-GPCM while keeping modular, testable components and a clean separation between data, model, and training layers.

## Structure
- `src/mirt_dkvmn/models/`: core model code (memory, embeddings, IRT heads).
- `src/mirt_dkvmn/data/`: dataset loading and schema validation.
- `src/mirt_dkvmn/training/`: loss composition, trainer, and optimization utilities.
- `src/mirt_dkvmn/config/`: configuration objects for model/data/training.
- `scripts/`: entry points for training/evaluation (placeholders).
- `configs/`: configuration files (placeholders).
- `tests/`: minimal shape and integration checks.

## Initial Scope
- Multidimensional theta/alpha with MIRT-GPCM logits.
- DKVMN memory with concept-level value states (traits = value_dim).

## Concept-Aligned Memory
When `concept_aligned_memory=true`, the model ties `value_dim == n_traits` and uses the DKVMN read vector directly as `theta` (latent traits). This keeps the design close to original DKVMN while making the trait dimension explicit.

## Notes
- For the theory and broader plan, see `MIRT_PLAN.md` and `MIRT_THEORY.md` in the parent folder.
- This skeleton intentionally avoids framework lock-in; you can add dependencies and implementations as needed.
