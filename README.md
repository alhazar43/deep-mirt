# deep-mirt

Modular encoder by decoder framework for ordinal knowledge tracing with
recoverable IRT parameters.

The flagship model **MA-GPCM** combines a DKVMN encoder with a GPCM decoder
and predicts ordinal student responses while recovering item discrimination
`alpha`, item step thresholds `beta`, and per-step ability `theta`. The same
framework hosts Transformer (SAKT-style) and LSTM encoders, plus Rasch,
binary, and softmax decoders. The Models table below lists all
nine pre-built model wrappers.

The paper "MA-GPCM: Modular Ability-tracking with Generalized Partial Credit
Model" (under review at IJAIED 2026) reproduces from the configs under
`ma-irt/configs/bulk/` and the cached checkpoints under
`ma-irt/outputs/<experiment>/best.pt`.

Headline recovery on `static_q200_k4` (fold 0): MA-GPCM reaches
`r_theta=0.96`, `r_alpha=0.89`, `r_beta=0.97`; the Transformer and LSTM
encoders reach `r_theta=0.98`, `r_alpha=0.92`, `r_beta=0.97`.

## Install

```bash
git clone https://github.com/alhazar43/deep-mirt
cd deep-mirt/ma-irt
pip install -r requirements.txt
```

GPU PyTorch builds are picked from
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

## Quick start

Smoke train, run from `ma-irt/`.

```bash
# 1. Generate a small synthetic dataset
python data_gen.py static --name smoke_test \
    --n_students 120 --n_questions 20 --n_cats 4 \
    --min_seq 10 --max_seq 25 --output_dir data --seed 42

# 2. Train MA-GPCM for one epoch
python train.py --config configs/smoke_dkvmn_gpcm.yaml \
    --dataset smoke_test --epochs 1

# 3. Evaluate prediction metrics and IRT recovery
python evaluate.py single --config configs/smoke_dkvmn_gpcm.yaml \
    --checkpoint outputs/smoke_test/best.pt \
    --data-dir data/smoke_test
```

`configs/smoke_transformer_gpcm.yaml` swaps in the Transformer encoder;
the other `configs/smoke_*.yaml` files cover the remaining model types.

## Models

Select via `model.model_type` in any YAML config. The encoder by decoder
composition (`model.encoder` and `model.decoder` blocks) builds the same
pieces from named components, see `docs/architecture.md`.

| Paper name | `model_type` | IRT params |
|---|---|---|
| MA-GPCM | `magpcm` (`separate_theta: true`) | theta, alpha, beta |
| DKVMN+GPCM | `magpcm` (`separate_theta: false`) | theta, alpha, beta |
| Transformer+GPCM | `transformer_gpcm` | theta, alpha, beta |
| LSTM+GPCM | `lstm_gpcm` | theta, alpha, beta |
| Static GPCM | `static_gpcm` | theta, alpha, beta |
| Dynamic GPCM | `dynamic_gpcm` | theta, alpha, beta |
| DKVMN+Softmax | `dkvmn_softmax` | none |
| DKT, DKVMN, Deep-IRT | `dkt`, `dkvmn`, `deep_irt` | none (K=2) |

## Reference docs

- `docs/architecture.md` — model structure, ability pathway, GPCM head
- `docs/pipeline.md` — train, evaluate, datagen data flow
- `docs/config_taxonomy.md` — YAML schema and config conventions
- `docs/script_taxonomy.md` — script entry points and their roles

## License

[MIT](LICENSE).
