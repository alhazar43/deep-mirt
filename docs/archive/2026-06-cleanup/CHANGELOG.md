# Changelog

## 2026-03-29 — Paper Reorganization + Codebase Cleanup

### Paper (overleaf-sync/)

- [x] Reorganize results: merge binary compat into static section, split dynamic into block/RW
- [x] Merge block+RW into single "Dynamic Ability Tracking" section with 2 arguments (separated pathway, distributed memory)
- [x] Methodology: unconstrained beta as default, monotonic as optional regularization
- [x] Rename "shared architecture" to "DKVMN+GPCM" throughout (~10 locations)
- [x] Add DKVMN+GPCM as named baseline in models list
- [x] Remove monotonic ablation entirely from results, intro, roadmap
- [x] Reframe "ordinal inductive bias" to "cumulative logit structure" (4 locations)
- [x] Rewrite intro contribution list (elevate SIE + separated pathway)
- [x] Rewrite conclusion three-result summary
- [x] Add disordered-thresholds-as-diagnostic point in Discussion
- [x] New trajectory figures: 2-row layout (examples + population error), 3 models
- [x] Fix figure titles/captions to match paper style (K=4, Q=200)
- [x] Add flexMIRT and ConQuest citations to ref.bib

### Codebase (kt-gpcm/)

- [x] Phase 1+2: Remove dead config fields (response_dim, use_separable_embed), fix loss defaults, remove QWKLoss
- [x] Phase 3: Remove monotonic_betas entirely (BREAKING)
- [x] Phase 4a: Archive 25 dead scripts, 10 shell scripts, 3 R debug scripts, 80 legacy configs
- [x] Phase 4b-c: Clean monotonic_betas from 643 configs, archive 54 old root configs
- [x] Phase 4d: Create unified evaluate.py (all model types, all DGPs, v3 linking)
- [x] Phase 6: Update CLAUDE.md and README.md
- [x] Gitignore data/, outputs/, archive/, legacy dirs (kt-mirt, mirt-dkvmn, deep-gpcm)
- [x] Untrack old data archives, output archives, generated configs, stale docs

## Cleanup Round 2 (2026-03-29, evening)

### Phase A: Dead code removal
- [x] Delete 6 superseded scripts (compute_all_recovery_v3, eval_block_and_rw, eval_dynamic_seeds, gen_dynamic_seed_configs, estimate_theta_eap, run_all_dynamic_k)
- [x] Remove FocalLoss class, simplify CombinedLoss to WOL-only
- [x] Remove memory_add_activation (inert), embed_dim (unused), DKVMN.n_questions (stored never read)
- [x] Remove 4 trainer regularization penalty methods (always off)
- [x] Remove focal_weight, qwk_weight, 4 penalty weight config fields
- [x] Fix base.yaml stale defaults
- [x] Tests: 56 pass, smoke train verified

### Phase B: Class renames (BREAKING)
- [x] DeepGPCM -> MAGPCM, file kt_gpcm.py -> magpcm.py
- [x] model_type "deepgpcm" -> "magpcm" in 161 YAML configs + dispatch
- [x] "linear_decay" -> "onehot", LinearDecayEmbedding -> OneHotEmbedding
- [x] "separable" -> "learned"
- [x] Update all docstring fossils (Deep-GPCM, memirt, DKVMN-GPCM -> MA-GPCM/MA-IRT)
- [x] Tests: 56 pass, smoke train verified

### Phase C: Directory restructure
- [x] Flatten src/kt_gpcm/ to top-level models/, training/, config/, dataloading/
- [x] Rename kt-gpcm/ to ma-irt/
- [x] Update all imports (from kt_gpcm.X -> from X)
- [x] PYTHONPATH now points to ma-irt/ (not ma-irt/src/)
- [x] Update .gitignore, CLAUDE.md, README.md
- [x] Tests: 56 pass, smoke train verified

### Validation
- [x] Static K=4 train+eval: all 4 models match Phase A numbers exactly (zero regression)

### Data + Training

- [x] Change model defaults: monotonic_betas=False, embedding_type=static_item
- [x] Create staircase data generator (3-level discrete ability change)
- [x] Generate staircase datasets K=3,4,5,6
- [x] Create staircase configs (MA-GPCM, DKVMN+GPCM, Dynamic GPCM)
- [x] Train 3 staircase models (K=4, seed=42)
- [x] Generate staircase + RW trajectory figures
