# Cleanup Round 2: Naming, Structure, and Dead Code

**Date**: 2026-03-29
**Status**: Phases A, B, C DONE. Phase D deferred to bulk retrain.

---

## Key Finding: Research repos use FLAT layout, not src/ layout

pyKT, AKT, simpleKT, DKT, and IRT repos all use flat structure:
```
project/
  models/
  data/
  configs/
  scripts/
  train.py
```

NOT: `project/src/project_name/models/`

Our `kt-gpcm/src/kt_gpcm/` adds unnecessary nesting and requires `PYTHONPATH=src` everywhere.

---

## Proposed New Structure

Rename `kt-gpcm/` to `ma-irt/` with flat layout:

```
ma-irt/
  models/                    # was src/kt_gpcm/models/
    magpcm.py               # was kt_gpcm.py (DeepGPCM -> MAGPCM)
    dkvmn_softmax.py        # unchanged
    dynamic_gpcm.py         # unchanged
    static_gpcm.py          # unchanged
    components/
      memory.py
      irt.py
      embeddings.py
    heads/
      gpcm.py
  training/                  # was src/kt_gpcm/training/
    trainer.py
    losses.py
  data/                      # was src/kt_gpcm/data/ (code, not datasets)
    loaders.py
  config/                    # was src/kt_gpcm/config/
    types.py
    loader.py
  datasets/                  # was data/ (generated datasets, gitignored)
  outputs/                   # training outputs (gitignored except figures/)
  configs/                   # YAML experiment configs
  scripts/                   # pipeline scripts
  tests/
  README.md
  requirements.txt
```

**Key changes:**
- No more `src/kt_gpcm/` wrapper
- Package name: `ma_irt` (imports: `from models.magpcm import MAGPCM`)
- Or simpler: direct imports with `PYTHONPATH=ma-irt` -> `from models.magpcm import MAGPCM`

---

## Naming Renames (paper alignment)

### P0 Critical (reader confusion if mismatched)

| What | Current | New (matches paper) |
|------|---------|---------------------|
| Main model class | `DeepGPCM` | `MAGPCM` |
| Config model_type | `"deepgpcm"` | `"magpcm"` |
| Main model file | `kt_gpcm.py` | `magpcm.py` |
| Embedding: One-hot | `"linear_decay"` / `LinearDecayEmbedding` | `"onehot"` / `OneHotEmbedding` |
| Embedding: Learned | `"separable"` | `"learned"` |
| Package name | `kt_gpcm` | `ma_irt` (or flat imports) |

### P1 Important

| What | Current | New |
|------|---------|-----|
| Dataset dirs | `v2_q200_k4` | `static_q200_k4` |
| Experiment prefix | `rq1_*`, `rq4_*`, `rq5_*` | `static_*`, `scaling_*`, `imbalance_*` |
| Forward loop var | `q_embed` | `k_t` or `key_embed` (paper uses k_t) |
| Summary attribute | `self.summary` | `self.item_summary` (paper: "item summary") |

### P2 Docstrings (mechanical find-replace)

~15 files with "Deep-GPCM", "memirt", "DKVMN-GPCM" fossils -> "MA-GPCM" / "MA-IRT"

---

## Scripts to Delete (6 scripts, ~2325 lines)

| Script | Lines | Reason |
|--------|-------|--------|
| `compute_all_recovery_v3.py` | 614 | Superseded by `evaluate.py` |
| `eval_block_and_rw.py` | 655 | Superseded by `evaluate.py` |
| `eval_dynamic_seeds.py` | 208 | Superseded by `evaluate.py` |
| `gen_dynamic_seed_configs.py` | 95 | One-shot, configs exist |
| `estimate_theta_eap.py` | 111 | Never imported |
| `run_all_dynamic_k.py` | 642 | One-shot pipeline |

## Dead Source Code to Remove

| Location | Item | Why |
|----------|------|-----|
| `losses.py` | `FocalLoss` class | Never instantiated (focal_weight always 0) |
| `losses.py` | `qwk_weight` in CombinedLoss | Ghost param, never read |
| `losses.py` | Focal branch in CombinedLoss | Dead conditional |
| `types.py` | `memory_add_activation` field | Inert, DKVMN hardcodes tanh |
| `embeddings.py` | `embed_dim` in LinearDecayEmbedding | Unused |
| `memory.py` | `self.n_questions` in DKVMN | Stored, never read |
| `trainer.py` | 4 regularization penalty methods (~60 lines) | Always off in every config |
| `base.yaml` | Stale focal_weight=0.5 | Contradicts code defaults |

---

## Execution Plan

### Phase A: Delete superseded scripts + dead code (safe, non-breaking)
1. Delete 6 scripts
2. Remove FocalLoss, simplify CombinedLoss
3. Remove memory_add_activation, embed_dim, DKVMN.n_questions
4. Remove trainer regularization penalties
5. Fix base.yaml
6. Run tests
7. Commit

### Phase B: Rename classes and config values (BREAKING for checkpoints)
1. `DeepGPCM` -> `MAGPCM`, file `kt_gpcm.py` -> `magpcm.py`
2. `"deepgpcm"` -> `"magpcm"` in build_model() and all configs
3. `"linear_decay"` -> `"onehot"`, `LinearDecayEmbedding` -> `OneHotEmbedding`
4. `"separable"` -> `"learned"`
5. Update all imports, tests, scripts
6. Update evaluate.py checkpoint patching for backward compat
7. Run tests
8. Commit

### Phase C: Directory restructure (MAJOR, do last)
1. Flatten `src/kt_gpcm/` to top-level `models/`, `training/`, `data/`, `config/`
2. Rename `kt-gpcm/` to `ma-irt/`
3. Update all PYTHONPATH references, CLAUDE.md, README.md
4. Update .gitignore paths
5. Run tests
6. Commit

### Phase D: Rename datasets and experiment prefixes (deferred to bulk retrain)
- `v2_q200_k4` -> `static_q200_k4` (requires regeneration)
- `rq1_*` -> `static_*` etc. (requires retraining)
- Do this during the bulk retrain, not as a separate step

---

## Risk Assessment

- **Phase A**: Safe. No external dependencies.
- **Phase B**: Breaks all existing checkpoints (class names in state_dict keys change). Need checkpoint migration or retrain.
- **Phase C**: Breaks all import paths. Requires updating every script and test. Large diff but mechanically simple.
- **Phase D**: Requires bulk retrain. Deferred.

**Recommendation**: Do Phase A now. Phase B+C together during bulk retrain (when all checkpoints are regenerated anyway). Phase D as part of bulk retrain.
