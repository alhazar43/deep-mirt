# irtrec, the DRL-MAIRT recommender

A realtime interactive job recommender built on top of the `ma-irt`
deep IRT belief tracker. This package lives at `deep-mirt/rl/` and
depends on the sibling `ma-irt/` package for the encoder, the
decoder, and the online step API.

## Canonical reference

The full v1 design, milestone plan, and contracts are documented in
[`docs/drl_mairt_plan_v1.md`](../docs/drl_mairt_plan_v1.md). That
document is the source of truth. This README is intentionally short.

## Install

From the repository root,

```bash
pip install -e rl/
```

The frozen text encoder, FAISS index, and PyTorch are pulled in as
declared dependencies.

## Run

The FastAPI service entrypoint is provided once M5 lands,

```bash
python -m irtrec.service.app
```

## Test

```bash
cd rl && pytest tests/
```

## Layout

See Section 12 of the plan document for the canonical directory
tree.
