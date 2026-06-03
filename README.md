# deep-mirt

Modular encoder by decoder framework for ordinal knowledge tracing with
recoverable IRT parameters.

The flagship model **MA-GPCM** pairs a DKVMN encoder with a GPCM decoder to
predict ordinal student responses while recovering item discrimination
`alpha`, step thresholds `beta`, and per-step ability `theta` in a single
forward pass. The same framework hosts Transformer (SAKT-style) and LSTM
encoders, plus Rasch, binary, and softmax decoders.

The paper *"MA-GPCM: Modular Ability-tracking with Generalized Partial
Credit Model"* (under review at IJAIED 2026) reproduces from the configs
in `ma-irt/configs/bulk/` and the cached checkpoints under
`ma-irt/outputs/<experiment>/best.pt`.

## Install

```bash
git clone https://github.com/alhazar43/deep-mirt
cd deep-mirt/ma-irt
pip install -r requirements.txt
```

GPU PyTorch builds are picked from
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

## Usage

The active project lives in [`ma-irt/`](ma-irt/). See its
[README](ma-irt/README.md) for the script usage manual, model list, and
configuration layout.

## Reference docs

- [`docs/architecture.md`](docs/architecture.md) — model structure, ability
  pathway, GPCM head
- [`docs/pipeline.md`](docs/pipeline.md) — train, evaluate, datagen data flow
- [`docs/config_taxonomy.md`](docs/config_taxonomy.md) — YAML schema and
  config conventions
- [`docs/script_taxonomy.md`](docs/script_taxonomy.md) — script entry
  points and their roles

## License

[MIT](LICENSE).
