# Config Taxonomy

This manifest classifies the current `ma-irt/configs/` tree before any
config archival or consolidation. It is a cleanup planning document, not a
move/delete request.

Inventory date: 2026-06-02.

Current YAML count: 2,294.

## Directory Counts

| Directory | YAML count | Status |
|---|---:|---|
| `configs/` | 47 | Mixed public smoke, base, and root dynamic-DGP configs |
| `configs/bulk/` | 1,652 | Main generated/frozen benchmark matrix |
| `configs/dynamic_seeds/` | 160 | Legacy dynamic seed matrix |
| `configs/experiments/rq1/` | 100 | Legacy RQ1 experiment matrix |
| `configs/experiments/rq4/` | 60 | Legacy RQ4 scaling/embedding matrix |
| `configs/experiments/rq5/` | 20 | Legacy RQ5 imbalance matrix |
| `configs/experiments/ablation/` | 5 | Legacy ablation matrix |
| `configs/_archive_s0p5/` | 125 | Existing archive |
| `configs/tmp_alpha1/` | 125 | Temporary alpha=1 matrix |

## Classification Rules

Status values:

- **KEEP**: public smoke, base, or paper-critical reproduction config.
- **KEEP-GENERATED**: generated configs that reproduce paper tables/figures.
- **REVIEW**: may support paper sections, figures, or appendices; inspect
  before moving.
- **ARCHIVE-CANDIDATE**: legacy, superseded, temporary, or older protocol.
- **EXISTING-ARCHIVE**: already archived by path; leave in place until a
  later archive policy is approved.

No config should be moved until its pattern is checked against
`CLEANUP_VERIFICATION_2026.md`, `benchmarks.md`, and active paper figures.

## Public Smoke and Base Configs

| Pattern/file | Count | Status | Rationale |
|---|---:|---|---|
| `base.yaml` | 1 | KEEP | Dataclass-default mirror/reference config. |
| `smoke.yaml` | 1 | KEEP | Public MA-GPCM smoke config. |
| `smoke_dkvmn_gpcm.yaml` | 1 | KEEP | Public DKVMN+GPCM ablation smoke config. |
| `smoke_dkvmn_softmax.yaml` | 1 | KEEP | Public DKVMN+Softmax smoke config. |
| `smoke_static_gpcm.yaml` | 1 | KEEP | Public Static GPCM smoke config. |
| `smoke_dynamic_gpcm.yaml` | 1 | KEEP | Public Dynamic GPCM smoke config. |
| `smoke_dkt.yaml` | 1 | KEEP | Binary DKT smoke config. |
| `smoke_dkvmn.yaml` | 1 | KEEP | Binary DKVMN smoke config. |
| `smoke_deep_irt.yaml` | 1 | KEEP | Binary Deep-IRT smoke config. |

## Root Dynamic-DGP Configs

| Pattern | Count | Status | Rationale |
|---|---:|---|---|
| `block_q200_k*.yaml` | 17 | REVIEW | Root-level block-change DGP configs. Used by dynamic-DGP experiments, but likely superseded by generated/bulk variants. |
| `rw_q200_k*.yaml` | 17 | REVIEW | Root-level random-walk DGP configs. Used by dynamic-DGP experiments, but likely superseded by generated/bulk variants. |
| `staircase_q200_k4*.yaml` | 4 | REVIEW | Root-level staircase DGP configs. Verify figure/table role before moving. |

## Bulk Configs

`configs/bulk/` currently contains 1,652 YAML files.

| Pattern | Count | Status | Rationale |
|---|---:|---|---|
| `bench_*.yaml` | 770 | KEEP-GENERATED | Five-fold pyKT-style benchmark configs. Includes paper-critical static, binary, ASSISTments, continuous, and discrete bench configs. |
| `continuous_*.yaml` | 160 | KEEP-GENERATED | Continuous/random-walk dynamic-DGP matrix; paper/figure support likely. |
| `discrete_*.yaml` | 160 | KEEP-GENERATED | Discrete/block dynamic-DGP matrix; paper/figure support likely. |
| `assist2009_*.yaml` | 55 | REVIEW | Older/proxy ASSISTments configs; verify relation to current `bench_*assist2009*` configs and figures. |
| `assistments_*.yaml` | 35 | REVIEW | Older/proxy ASSISTments configs; verify relation to current `bench_*assist2017*` configs and figures. |
| `static_*.yaml` | 217 | REVIEW | Older seeded static synthetic configs and variants; some may be prior protocol, ablation, learned, or alpha=1 variants. |
| `imbalance*.yaml` | 100 | REVIEW | Imbalance extension matrix; verify paper section/appendix role. |
| `scaling*.yaml`, `scalability*.yaml` | 155 | REVIEW | Scaling/scalability extension matrix; verify paper section/appendix role. |

The `bench_*` family is the safest current reproduction surface because
`CLEANUP_VERIFICATION_2026.md` maps paper tables directly to that naming
scheme.

## Dynamic Seed Configs

| Directory | Count | Status | Rationale |
|---|---:|---|---|
| `dynamic_seeds/` | 160 | ARCHIVE-CANDIDATE | Legacy seed matrix for block/random-walk DGPs. Current paper verification primarily uses `configs/bulk/` and pyKT-fold configs. Verify no plotting script still expects this path before moving. |

## Experiment RQ Configs

| Directory | Count | Status | Rationale |
|---|---:|---|---|
| `experiments/rq1/` | 100 | ARCHIVE-CANDIDATE | Legacy RQ1 matrix predating current bulk configs. |
| `experiments/rq4/` | 60 | REVIEW | Scaling/embedding experiments. May overlap with paper scaling section. |
| `experiments/rq5/` | 20 | REVIEW | Imbalance experiments. May overlap with paper imbalance section. |
| `experiments/ablation/` | 5 | REVIEW | Ablation configs. Keep until ablation narrative is finalized. |

## Existing Archive and Temporary Configs

| Directory | Count | Status | Rationale |
|---|---:|---|---|
| `_archive_s0p5/` | 125 | EXISTING-ARCHIVE | Already archived by name. Leave until a later archive policy is approved. |
| `tmp_alpha1/` | 125 | ARCHIVE-CANDIDATE | Temporary alpha=1 matrix. Keep only if alpha=1 ablation remains in the paper or appendix. |

## Paper-Critical Config Families

These patterns must not be moved without running the verification suite in
`CLEANUP_VERIFICATION_2026.md`.

```text
configs/bulk/bench_static_gpcm_static_q200_k{3,4,5,6}_pykt_fold{0,1,2,3,4}.yaml
configs/bulk/bench_dynamic_gpcm_static_q200_k{3,4,5,6}_pykt_fold{0,1,2,3,4}.yaml
configs/bulk/bench_dkvmn_softmax_static_q200_k{3,4,5,6}_pykt_fold{0,1,2,3,4}.yaml
configs/bulk/bench_dkvmn_gpcm_static_q200_k{2,3,4,5,6}_pykt_fold{0,1,2,3,4}.yaml
configs/bulk/bench_magpcm_static_q200_k{2,3,4,5,6}_pykt_fold{0,1,2,3,4}.yaml
configs/bulk/bench_{dkt,dkvmn,deep_irt}_static_q200_k2_pykt_fold{0,1,2,3,4}.yaml
configs/bulk/bench_{dkt,dkvmn,deep_irt,dkvmn_gpcm,magpcm}_synthetic5_v{0,1,2,3,4}_pykt_fold{0,1,2,3,4}.yaml
configs/bulk/bench_{dkt,dkvmn,deep_irt,dkvmn_gpcm,magpcm}_assist2009_bin_pykt_fold{0,1,2,3,4}.yaml
configs/bulk/bench_{dkt,dkvmn,deep_irt,dkvmn_gpcm,magpcm}_assist2017_bin_pykt_fold{0,1,2,3,4}.yaml
configs/bulk/bench_{static_gpcm,dynamic_gpcm,dkvmn_gpcm,magpcm}_assist{2009,2017}_ord_k4_pykt_fold{0,1,2,3,4}.yaml
configs/bulk/bench_{static_gpcm,dynamic_gpcm,dkvmn_gpcm,magpcm}_continuous_q200_k{3,4,5,6}_pykt_fold{0,1,2,3,4}.yaml
configs/bulk/bench_{static_gpcm,dynamic_gpcm,dkvmn_gpcm,magpcm}_discrete_q200_k{3,4,5,6}_pykt_fold{0,1,2,3,4}.yaml
```

## Next Action

Before config moves:

1. Generate a machine-readable manifest from the pattern groups above.
2. Verify that every `KEEP` and `KEEP-GENERATED` pattern exists.
3. Move only obvious archive families first, such as `tmp_alpha1/`, after
   confirming they are not cited by `overleaf-sync/main.tex`, current README
   docs, or plotting scripts.
4. Run smoke tests and the relevant paper-critical subset after each move.
