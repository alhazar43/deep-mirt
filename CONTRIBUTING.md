# Contributing to deep-mirt

This repository implements MA-GPCM and the surrounding model family for
polytomous knowledge tracing with explicit IRT parameter recovery. Active
code lives in [`ma-irt/`](ma-irt/). The paper is in
[`overleaf-sync/`](overleaf-sync/).

## Environment

```bash
conda create -n research python=3.12
conda activate research
pip install -r ma-irt/requirements.txt
# PyTorch is pinned separately to match your CUDA version, see the comment at
# the top of requirements.txt.
```

On Windows, set `KMP_DUPLICATE_LIB_OK=TRUE` before running anything that
imports torch.

## Running the tests

```bash
cd ma-irt
PYTHONPATH=. KMP_DUPLICATE_LIB_OK=TRUE pytest tests/ -v
```

The suite has three independent gates.

| Test file | What it pins | Why |
|---|---|---|
| `tests/test_linking.py` | IRT linking math, plus regression hashes | The Python and R baselines must agree on the linking constants |
| `tests/test_registry.py` | Encoder and decoder ABC contract | Required for any new backbone or decoder |
| `tests/test_baseline_reproduction.py` | MA-GPCM headline metrics on the K=4 Synthetic-Static and K=2 ASSIST2009 fold 0 sidecars | The R2 invariant from [`PIPELINE_OPT_PLAN.md`](PIPELINE_OPT_PLAN.md). Any architectural change that violates this gate is rolled back |

`test_baseline_reproduction.py` reads cached `recovery_metrics.json` and
`test_metrics.json` from `ma-irt/outputs/bench_magpcm_*_pykt_fold0/`. If the
checkpoint directory is absent (fresh clone), the test skips rather than
fails.

## Adding a new sequential encoder

1. Implement the encoder as a subclass of
   [`EncoderBackbone`](ma-irt/models/registry.py), returning an
   `EncoderOutput` dataclass from `forward()`.
2. Register the implementation with `@register_encoder("name")`.
3. Add a smoke YAML under `ma-irt/configs/smoke/` so the encoder is reachable
   from the bulk runner.
4. Add a test under `ma-irt/tests/` asserting the forward returns the
   expected shapes.
5. For a paper-graded comparison against MA-GPCM-on-DKVMN, train on the
   K=4 Synthetic-Static config and confirm the metric envelope per
   `PIPELINE_OPT_PLAN.md` P3.

See [`INVESTIGATION_SOTA.md`](INVESTIGATION_SOTA.md) for the recommended
implementation patterns drawn from pyKT, py-irt, and torchvision.

## Adding a new response decoder

1. Implement the decoder as a subclass of
   [`ResponseDecoder`](ma-irt/models/registry.py), returning a `DecoderOutput`
   dataclass.
2. Register with `@register_decoder("name")`.
3. For polytomous IRT decoders, follow the GPCM/GRM/PCM family shape, theta
   and alpha in $\mathbb{R}^D$, beta in $\mathbb{R}^{K-1}$. For CDM
   decoders, populate `skill_profile` and leave the IRT-side fields empty.
4. Extend `scripts/data_gen.py` if the new decoder needs a matching DGP.

## Architecture and pipeline docs

- [`docs/architecture.md`](docs/architecture.md), the MA-GPCM architecture
  contract.
- [`docs/pipeline.md`](docs/pipeline.md), end-to-end data flow.
- [`docs/cleanup/`](docs/cleanup/), tier-by-tier cleanup records.
- [`PIPELINE_OPT_PLAN.md`](PIPELINE_OPT_PLAN.md), the in-flight optimization
  plan, P1 through P6.

## Workflow conventions

The project uses a forward-only commit history. Cleanup, refactor, and
feature work each land as a sequence of small commits with an explicit
verification record in [`cleanup_log.md`](cleanup_log.md).

When in doubt, prefer
1. one logical change per commit,
2. running `pytest tests/ -v` before each commit, and
3. a commit message that names the verification result.

Do not run `git rebase -i`, `git push --force`, or any other history rewrite
without an explicit request.

## Style notes

- American English.
- No em-dashes or en-dashes in prose. Commas, periods, and parentheses are
  preferred.
- No colons inside prose (math and code are exempt).
- No semicolons in prose.
- Match the candidate's voice from the MA-GPCM paper and the existing
  `docs/architecture.md` for any new technical writing.

## License

MIT. See [`LICENSE`](LICENSE).
