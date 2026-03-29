# TODO — MA-IRT Project

Integrated cleanup and bulk retrain plan. See RETRAIN_PLAN.md for full dataset/model specs.

---

## Phase A: Dead Code Removal (safe, non-breaking)

- [ ] Delete 6 superseded scripts (compute_all_recovery_v3, eval_block_and_rw, eval_dynamic_seeds, gen_dynamic_seed_configs, estimate_theta_eap, run_all_dynamic_k)
- [ ] Remove FocalLoss class from losses.py
- [ ] Simplify CombinedLoss (remove focal branch, qwk_weight ghost param)
- [ ] Remove memory_add_activation from ModelConfig + model constructors
- [ ] Remove embed_dim from LinearDecayEmbedding
- [ ] Remove DKVMN.n_questions (stored, never read)
- [ ] Remove trainer regularization penalty methods (always off)
- [ ] Fix base.yaml stale defaults
- [ ] Fix plot_theta_temporal.py hardcoded paths
- [ ] Run tests, verify pipeline
- [ ] Commit

## Phase B: Class + Config Renames (BREAKING for checkpoints)

- [ ] `DeepGPCM` -> `MAGPCM`, file `kt_gpcm.py` -> `magpcm.py`
- [ ] `"deepgpcm"` -> `"magpcm"` in build_model() and all configs
- [ ] `"linear_decay"` -> `"onehot"`, `LinearDecayEmbedding` -> `OneHotEmbedding`
- [ ] `"separable"` -> `"learned"`
- [ ] Update evaluate.py checkpoint patching for backward compat
- [ ] Update all imports, tests, scripts
- [ ] Fix ~15 files with "Deep-GPCM"/"memirt" docstring fossils
- [ ] Run tests
- [ ] Commit

## Phase C: Directory Restructure

- [ ] Flatten `kt-gpcm/src/kt_gpcm/` to `ma-irt/models/`, `ma-irt/training/`, etc.
- [ ] Rename `kt-gpcm/` to `ma-irt/`
- [ ] Update all PYTHONPATH references (no more `PYTHONPATH=src`)
- [ ] Update .gitignore paths
- [ ] Update CLAUDE.md, README.md
- [ ] Run tests
- [ ] Commit

## Phase 1: Dataset Generation

- [ ] 5 static: static_q200_k{2,3,4,5,6} (N=5000)
- [ ] 3 scalability: static_q{500,1000,2000}_k4 (N=5000/5000/10000)
- [ ] 5 discrete: discrete_q200_k{2,3,4,5,6} (3-level staircase)
- [ ] 5 continuous: continuous_q200_k{2,3,4,5,6} (random walk)
- [ ] 3 imbalanced: imbalanced_q200_k4_{mild,severe,extreme}
- [ ] 1 bimodal: imbalanced_q200_k4_bimodal (0.5*N(-1.5,0.25) + 0.5*N(1.5,0.25))
  - Add --bimodal flag to data_gen_imbalanced.py

## Phase 2: Config Generation

- [ ] Write gen_all_configs.py to programmatically generate all ~350 configs
- [ ] Generate configs for all model x dataset x seed combinations

## Phase 3: Static Training (~6h GPU)

- [ ] MA-GPCM: 5K x 5 seeds = 25 runs
- [ ] DKVMN+Softmax: 5K x 5 seeds = 25 runs
- [ ] Dynamic GPCM: 5K x 5 seeds = 25 runs
- [ ] GPCM (SGD): 5K x 5 seeds = 25 runs
- [ ] GPCM (EM): 5K x 1 run = 5 R mirt calibrations

## Phase 4: Dynamic Training (~8h GPU)

- [ ] Discrete: 4 models x 4K (3..6) x 5 seeds = 80 runs
- [ ] Continuous: 4 models x 4K (3..6) x 5 seeds = 80 runs
- [ ] DKVMN+GPCM: K=4 x 1 seed, discrete + continuous = 2 runs (trajectory figures)

## Phase 5: Scalability + Imbalanced (~6h GPU)

- [ ] Scalability: 3 embeddings x 4Q x 5 seeds = 60 runs
- [ ] Imbalanced: 4 conditions x 5 seeds = 20 runs

## Phase 6: Evaluation

- [ ] evaluate.py static on all static checkpoints -> eval_static.csv
- [ ] evaluate.py dynamic on all dynamic checkpoints -> eval_dynamic.csv
- [ ] evaluate.py for scalability -> eval_scaling.csv
- [ ] evaluate.py for imbalanced -> eval_imbalanced.csv

## Phase 7: Figures

### Main body (K=4)
- [ ] Temporal theta convergence
- [ ] Item parameter recovery scatter
- [ ] Discrete trajectory comparison (MA-GPCM, DKVMN+GPCM, Dynamic GPCM)
- [ ] Continuous trajectory comparison (same 3 models)

### Appendix (K=2,3,5,6)
- [ ] Temporal theta (4 figures)
- [ ] Item scatter (4 figures)
- [ ] Learner trajectories (4 figures)

## Phase 8: Tables

- [ ] Table 1: Ordinal prediction (static, K=3..6)
- [ ] Table 2: Parameter recovery (static, K=3..6)
- [ ] Table 3: Binary compatibility (K=2)
- [ ] Table 4: Discrete prediction (K=3..6)
- [ ] Table 5: Discrete recovery (K=3..6) — with traj RMSE + median r
- [ ] Table 6: Continuous prediction (K=3..6)
- [ ] Table 7: Continuous recovery (K=3..6) — with traj RMSE + median r
- [ ] Table 8: Item representation (Q=200..2000)
- [ ] Table 9: Imbalance robustness (K=4, 4 conditions)

## Phase 9: Paper Updates

- [ ] Update all table numbers from new results
- [ ] Update dataset descriptions (Ordinal-Block -> discrete 3-staircase, rename to Ordinal-Staircase)
- [ ] Update prose referencing specific numbers
- [ ] Final abstract pass
