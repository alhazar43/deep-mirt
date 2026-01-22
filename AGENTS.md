# Repository Guidelines

## Project Structure & Module Organization
- `deep-gpcm/`: Primary project. Training/eval/analysis entry points are `main.py`, `train.py`, and `evaluate.py`.
- `deep-gpcm/model/` and `deep-gpcm/optimization/`: Core model components and hyperparameter optimization logic.
- `deep-gpcm/analysis/` and `deep-gpcm/utils/`: IRT analysis, plotting, and dataset utilities.
- `deep-gpcm/results/` and `deep-gpcm/save_models/`: Generated artifacts, metrics, and checkpoints.
- Other folders (`akt/`, `deep-1pl/`, `dkvmn-ori/`, `dkvmn-torch/`) are legacy or reference implementations; avoid changing them unless a change is explicitly scoped there.

## Build, Test, and Development Commands
Deep-GPCM requires the `vrec-env` conda environment.

- Activate env: `source ~/anaconda3/etc/profile.d/conda.sh && conda activate vrec-env`
- Full pipeline (recommended): `python deep-gpcm/main.py --dataset synthetic_500_200_4 --epochs 30 --hyperopt`
- Single model training: `python deep-gpcm/train.py --model deep_gpcm --dataset synthetic_500_200_4 --epochs 30 --n_folds 0`
- Evaluation: `python deep-gpcm/evaluate.py --all --dataset synthetic_OC`
- IRT analysis: `python deep-gpcm/analysis/irt_analysis.py --dataset synthetic_OC`

There is no top-level build script; `deep-gpcm/` owns its runtime and dependencies.

## Coding Style & Naming Conventions
- Python is the primary language; follow the style already used in each subproject.
- Indentation is 4 spaces; keep existing module naming and CLI flag patterns (e.g., `--dataset`, `--model`).
- No repository-wide formatter or linter is configured; avoid reformatting unrelated code.

## Testing Guidelines
- No automated test suite is wired at the repo root.
- For Deep-GPCM changes, validate with a small run (e.g., short epochs or a small synthetic dataset).
- Keep generated artifacts under `deep-gpcm/results/` and `deep-gpcm/save_models/` only.

## Commit & Pull Request Guidelines
- Git history is minimal (`"Add back old projects"`, `"Setting up parent repo"`), so there is no enforced commit convention.
- Keep commits focused on `deep-gpcm/` and describe datasets, model types, and key flags used.
- PRs should include the exact command used to validate (plus a short result note).

## Configuration & Data Notes
- Deep-GPCM requires `vrec-env`; do not assume system Python.
- Large datasets are checked in; avoid duplicating or renaming dataset files unless necessary.

## Deep-GPCM Extension Focus (MIRT + KC Mapping)
- Changes should target DKVMN-based components in `deep-gpcm/model/` and their configs.
- When adding multi-dimensional IRT (MIRT), be explicit about the latent dimension size and how KC dimensions map to it.
- Prefer configuration-driven mapping (e.g., config flags or JSON) over hardcoded assumptions.

## Agent Instructions
- Always activate `vrec-env` for any runnable Deep-GPCM command.
- Be punctual and precise during analysis: identify file paths, relevant modules, and assumptions before proposing changes.
