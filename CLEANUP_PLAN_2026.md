# Cleanup Plan 2026

Procedural, tiered cleanup of the `deep-mirt` repository. Each tier is independently authorizable. Tiers are ordered safe-first. No file is modified until a tier is explicitly approved.

The active project is `ma-irt/`. Reference paths in this plan are absolute Windows paths so they can be cross-checked without changing directories. The plan assumes that the paper experiments (the six models in `benchmarks.md`) must remain reproducible to within 1% on ACC, AUC, QWK, $r_\alpha$, $r_\beta$, $r_\theta$.

A parallel research-scientist agent owns (a) the canonical experiment list to preserve and (b) the minimal verification suite. This plan references those outputs at the points where they are needed.

Hard facts established before writing this plan:
- `ma-irt/scripts/run_all_experiments.sh` does NOT exist on disk, despite being cited in `benchmarks.md` and `ma-irt/README.md`. The actual orchestrator is `ma-irt/scripts/run_bulk_retrain.sh`, which iterates `configs/bulk/`.
- Live config directories under `ma-irt/configs/` are root-level YAMLs, `bulk/`, `experiments/`, and `dynamic_seeds/`. The `generated/` and `baselines/` directories referenced in `ma-irt/CLEANUP.md` and `ma-irt/CLEANUP_PLAN.md` are already gone.
- `ma-irt/configs/bulk/` has 1652 files across 844 distinct stems (model x dataset x seed).
- `ma-irt/outputs/` is 1.7 GB.
- Top-level `figures/` (188 KB) is not referenced by `overleaf-sync/main.tex`. The paper reads `overleaf-sync/figures/`.
- The seven candidate legacy repos at the root total 2.3 GB. Among them, `dkvmn-ori/data/synthetic/` and `deep-1pl/data/synthetic/` are read by `ma-irt/scripts/_build_pykt_synthetic5.py` and `_convert_yeung_synthetic.py`. `deep-gpcm/data/assist2009_dkvmn/` is referenced by a docstring in `convert_dkvmn_format.py`. The rest are not imported by any `ma-irt` code.

---

## Tier 0. Sanity hygiene

Files that are unambiguously build artifacts or scratch logs. Zero risk to paper experiments.

### Inventory

**Top-level (`C:\Users\steph\documents\deep-mirt\`)**
- `texput.log`
- `IJAIED-sub.zip` (submission archive; user should confirm before delete)

**`ma-irt/` root**
- `paper.aux`, `paper.log`, `texput.log` (LaTeX droppings, no `paper.tex` exists)
- `baseline_run.log`, `bulk_run.log`, `outputs_q500_v9a.log`, `regenerate_plots.log`
- `_check_rmse.py`, `_render_tables.py` (one-off scratch scripts at the root, not in `scripts/`)
- `figures_tmp/` (working directory for figure regeneration, never referenced)
- `logs/` (`retrain_v2.log`, `rq1_training.log`; superseded by per-run logs under `outputs/`)
- `__pycache__/` at root and in every package subdir

**`overleaf-sync/`**
- `title_page.aux`, `title_page.log` (LaTeX droppings; `title_page.pdf` is kept since it is built into `main.tex`)

**`ma-irt/outputs/` log spew**
- 39 top-level `*.log` and `*.out` files (`_assist2017_bin.log`, `_chunked_bench_*.log`, `alpha1_*.log`, `ablation_training_*.log`, `bench_*.log`, etc.). These are sweep-orchestrator stdout dumps, not per-run training logs. The per-run logs live inside `outputs/<run>/train.log` and are preserved.

**Pyc / pycache (recursive)**
- All `__pycache__/` directories under `ma-irt/`, `kt-mirt/`, `mirt-dkvmn/` (12 directories total)
- Stray `*.pyc` files

### Evidence

- LaTeX droppings have no source `.tex` in the same directory (e.g. `ma-irt/paper.tex` does not exist), so the `.aux`/`.log` cannot be regenerated and serve no purpose.
- Sweep logs in `ma-irt/outputs/*.log` are all dated before 2026-04-29; they are not referenced by `run_bulk_retrain.sh` or by any reporting script (`generate_tables.py`, `_aggregate_bench.py`).
- `__pycache__/` is regenerated on first import.
- `_check_rmse.py` and `_render_tables.py` at `ma-irt/` root are not in `scripts/` and not imported anywhere.

### Procedure

1. Pre-state scan. Save the inventory list as a file (`cleanup_t0_inventory.txt`) containing the absolute path of every candidate. Compute `du -sb` of the union.
2. Confirm git status is clean of any uncommitted changes touching the listed files (`git status --short -- <path>` per directory). Anything modified gets excluded from this tier.
3. Delete in groups, one git commit per group. Suggested groups.
   - a. Recursive `__pycache__` and `*.pyc`.
   - b. LaTeX droppings (root, `ma-irt/`, `overleaf-sync/`).
   - c. Top-level scratch logs in `ma-irt/` (`baseline_run.log`, `bulk_run.log`, etc.).
   - d. `ma-irt/outputs/*.log` and `*.out` top-level dumps.
   - e. `ma-irt/_check_rmse.py`, `ma-irt/_render_tables.py`, `ma-irt/figures_tmp/`, `ma-irt/logs/`.
   - f. Optional, ask first. `IJAIED-sub.zip` at root.
4. Verification step. Run `cd ma-irt && PYTHONPATH=. pytest tests/ -v`. No verification suite needed beyond tests since no source or config file is touched.
5. Record-keeping. Save the deletion list as `cleanup_t0_deleted.txt` and the `git log` of the deletion commit(s).
6. Post-state scan. Re-run the inventory query and confirm zero residue.

### Risk and rollback

- Worst case. A user-modified scratch file gets deleted. Rollback via `git restore` or `git checkout HEAD~1 -- <path>` since everything is in one commit per group.
- The `__pycache__` group cannot break anything since Python regenerates it.
- `IJAIED-sub.zip` is the manuscript submission artifact. Defer until the user confirms whether it is needed for the IJAIED record.

---

## Tier 1. Stale planning markdown

Markdown that was once a working plan and is now archeology. The risk is loss of context, not loss of functionality.

### Inventory

| Path | Date in header | Status |
|---|---|---|
| `BENCH_OPT_PLAN.md` | mid-Apr 2026 | Bench sweep is complete (Table 2 of `benchmarks.md`) |
| `BINARY_BENCH_TODO.md` | mid-Apr 2026 | Same. Binary CV results are in the paper |
| `CHANGELOG.md` | 2026-03-29 onward | Manual log, superseded by `git log` |
| `CODE_CHANGES_2026-03-29.md` | 2026-03-29 | Status header reads "PLANNING"; work has shipped |
| `PAPER_CHANGES_2026-03-29.md` | 2026-03-29 | Status "IN PROGRESS"; the paper has moved on |
| `PAPER_NOTES.md` | early 2026 | Cross-ref ledger; obsoleted by current paper |
| `PYKT_REFACTOR_PLAN.md` | mid-Apr 2026 | Refactor done. Bench configs exist |
| `RETRAIN_PLAN.md` (root) | 2026-03-29 | Superseded by `ma-irt/PLAN_sigma05_bulk_retrain.md` and current `run_bulk_retrain.sh` |
| `TODO.md` (root) | 2026-03-29 | Same. Cleanup and retrain are now in active scripts |
| `phd_blueprint_d1.md`, `phd_blueprint_d1_v2.md` | recent | PhD planning, superseded by `phd_research_proposal.md` |
| `phd_blueprint_d2.md`, `phd_blueprint_d2_v2.md` | recent | Same |
| `phd_blueprint_d3.md`, `phd_blueprint_d3_v2.md` | recent | Same |
| `phd_research_blueprint.md`, `phd_research_blueprint_v4.md` | recent | Working draft, the final is `phd_research_proposal.md` |
| `proxy-ord-mapping.md` | recent | Methodology note; check whether referenced by the paper |
| `ma-irt/CLEANUP.md` | 2026-03-29 | This tier's predecessor, Phase A-C done |
| `ma-irt/CLEANUP_PLAN.md` | 2026-03-29 | Same |
| `ma-irt/NOTES_linking_appendix.md` | 2026-04 | Source material for an appendix; **probably keep** until paper is final |
| `ma-irt/PLAN_sigma05_bulk_retrain.md` | 2026-04 | Sigma sweep done, bulk retrain ran |
| `ma-irt/RETRAIN_PLAN.md` | early 2026 | Predecessor to the sigma plan |
| `ma-irt/REVIEW_converged.md`, `REVIEW_psychometric.md`, `REVIEW_research_scientist.md` | 2026-04 | Reviewer simulations; **keep** until paper resubmission is closed |
| `ma-irt/TODO_alternating_optim.md` | recent | Static GPCM alternating-optim study; check if still open |

### Evidence

- Status banners inside several files explicitly say "PLANNING" or "IN PROGRESS" but the referenced code has shipped (e.g. naming renames, monotonic-betas removal, the sigma sweep).
- The phd_blueprint series has six v1/v2 drafts plus a v4 compilation plus `phd_research_proposal.md`. `git log` shows the most recent commit message is "Rewrite PhD proposal: narrative, multidim MA-IRT prereq, four directions", which is the proposal file. The earlier drafts are working notes.
- `CHANGELOG.md` overlaps with `git log` and offers no incremental information.

### Procedure

1. Pre-state scan. List the absolute paths and the SHA-1 of each file; save as `cleanup_t1_inventory.txt`.
2. Classify each item as KEEP, MOVE, or DELETE. The user reviews this classification before any file moves.
   - Default proposal. MOVE all DELETE-candidates to `docs/archive/2026-06-cleanup/` (newly created) rather than git-delete. Preserves the writing without polluting the root.
   - DEFAULT KEEP. `ma-irt/NOTES_linking_appendix.md`, `ma-irt/REVIEW_*.md`, `phd_research_proposal.md`.
   - DEFAULT MOVE. Everything else listed above.
3. Minimal-blast-radius change. `git mv` each file to the archive directory. No content edits.
4. Verification step. Run `grep -rE "(BENCH_OPT_PLAN|RETRAIN_PLAN|CLEANUP_PLAN|TODO\.md|PAPER_NOTES)" --include="*.py" --include="*.sh" --include="*.tex" --include="*.md" .` and confirm only allowed callers (the moved files themselves, or this plan) remain. The paper LaTeX must not reference any of them.
5. Record-keeping. Commit message lists every moved file. `cleanup_t1_moves.txt` captures source-to-destination pairs.
6. Post-state scan. Re-run the inventory and confirm the root is down to README, CLAUDE.md, benchmarks.md, the active phd_research_proposal.md, and live project directories.

### Risk and rollback

- Worst case. A planning doc was actually a live design document for ongoing work. Rollback is a single `git mv` reversal.
- Mitigation. The DEFAULT KEEP list is biased toward retention. If in doubt, KEEP. Anything actually obsolete will be detected by re-inventorying in a future round.

---

## Tier 2. Legacy reference repos at root

Seven vendored repos and one large data dump live alongside `ma-irt/`. Two of them are referenced by `ma-irt` scripts as data sources. The rest are reference reading.

### Inventory

| Path | Size | Referenced by `ma-irt/` | Disposition candidate |
|---|---|---|---|
| `mirt-dkvmn/` | 127 KB | none | MOVE to `legacy/` |
| `deep-gpcm/` | 1.9 GB | `scripts/convert_dkvmn_format.py` (docstring only) | MOVE to `legacy/`, prune `data/` first |
| `deep-1pl/` | 48 MB | `scripts/_convert_yeung_synthetic.py` reads `deep-1pl/data/synthetic/` | KEEP `data/synthetic/`, MOVE rest to `legacy/` |
| `dkt-ori/` | 22 MB | none | MOVE to `legacy/` |
| `dkvmn-ori/` | 62 MB | `scripts/_build_pykt_synthetic5.py` reads `dkvmn-ori/data/synthetic/` | KEEP `data/synthetic/`, MOVE rest to `legacy/` |
| `akt/` | 91 MB | none | MOVE to `legacy/` |
| `pykt/` | 102 MB | none (the project has its own bench configs; pyKT itself is not imported) | MOVE to `legacy/` |
| `assisstment-raw/` | 701 MB | indirectly via `convert_assistments*.py` reading converted CSVs from elsewhere | Verify before touching |
| `archive_sigma03_20260422_0534/` | 2.6 GB | none | MOVE to `legacy/archives/` or external storage |
| `submission_2026-05-09/` | 3.5 MB | none | MOVE to `legacy/submissions/` |
| `_overleaf_old/` | 4 KB | none | DELETE candidate (just one `artifacts/` subdir) |
| `figures/` (top-level) | 188 KB | none, paper reads `overleaf-sync/figures/` | DELETE candidate |
| `elsarticle/` | small | LaTeX class files | KEEP if `overleaf-sync` does not bundle the class; verify |
| `kt-mirt/` | 11 MB | none | Verify if it is an active sibling project before touching |
| `recsys25_v1_3.pdf` | small | reference reading | MOVE to `docs/references/` |

### Evidence

- `grep -rE "(mirt-dkvmn|deep-1pl|dkt-ori|dkvmn-ori|deep-gpcm|akt|pykt)/" --include="*.py" --include="*.sh" --include="*.yaml" ma-irt/` returns five hits, all in scripts. Four of those hits are in scripts whose names begin with underscore (one-off):
  - `ma-irt/scripts/_orbit_align_static_experiment.py`
  - `ma-irt/scripts/_k4_digest.sh`
  - `ma-irt/scripts/_build_pykt_synthetic5.py` (reads `dkvmn-ori/data/synthetic/`)
  - `ma-irt/scripts/_convert_yeung_synthetic.py` (reads `deep-1pl/data/synthetic/`)
- The fifth is `ma-irt/scripts/convert_dkvmn_format.py`, where the reference is only in the docstring. No live file reads `deep-gpcm/`.
- Python imports. `grep -rE "^(import|from) (pykt|akt)" ma-irt/` returns zero hits.
- Paper LaTeX. `grep -rE "(figures/|submission_2026|elsarticle)" overleaf-sync/*.tex` confirms the paper points at `overleaf-sync/figures/`, not the root `figures/`. The class file `elsarticle.cls` should be checked separately (paper says `\documentclass[review,12pt,authoryear]{elsarticle}`).
- `kt-mirt/` is an independent sibling project (own `src/`, `tests/`, `logs/`). Verify before any move.

### Procedure

1. Pre-state scan. For every candidate directory, run `du -sh` and `find <dir> -name "*.csv" -o -name "*.txt" -newer <date>` to see if anything is recent.
2. Verify data dependencies. For `dkvmn-ori/`, `deep-1pl/`, `deep-gpcm/`, run the relevant ma-irt script with `--help` (or a dry-run) to confirm the data file is still expected on disk and not embedded in `ma-irt/data/`. Document each verification.
3. Create `legacy/` and `legacy/archives/` directories at the repo root. Add to `.gitignore` if they should not be tracked.
4. Two-stage move. Per directory in the table.
   - Stage A. Move the parts not referenced by ma-irt. Example. For `deep-1pl/`, move everything except `data/synthetic/`. Or, more conservative, leave `deep-1pl/` in place but document the dependency in a `LEGACY_DATA.md`.
   - Stage B. After two weeks of no regressions, move the data-source directories too once the `_build_pykt_synthetic5.py` and `_convert_yeung_synthetic.py` are confirmed never to run again (they are underscore-prefixed one-offs).
5. Minimal-blast-radius change. The first move should be `mirt-dkvmn/`, `dkt-ori/`, `akt/`, `pykt/`, `_overleaf_old/`, and the top-level `figures/`. These have zero references.
6. Verification step. Run the **Tier 2 verification suite** from the research-scientist agent. Minimum requirement, the paper-critical experiments listed in their canonical-experiment manifest reproduce their headline numbers to within 1%. Suggested floor.
   - Smoke train per model type on `configs/smoke.yaml`.
   - One bulk seed for `static_q200_k4` MA-GPCM and DKVMN+Softmax. Compare ACC, QWK, $r_\alpha$, $r_\beta$, $r_\theta$ to `benchmarks.md` Table 1 row K=4.
7. Record-keeping. `cleanup_t2_moves.txt` lists source-to-destination. Output of the verification commands archived to `cleanup_t2_verify.txt`.
8. Post-state scan. Re-run the inventory and update the table with new sizes.

### Risk and rollback

- Worst case. A move breaks a data-loading path that is exercised only in an underscore-prefixed script the user runs manually.
- Rollback. `git mv` reversal, or for large data dumps that were never committed to git, restore from local backup or external storage.
- Mitigation. Stage A is reversible in one command. Defer Stage B until the next round.

---

## Tier 3. Dead code inside `ma-irt/`

Source modules, scripts, and configs that are unreachable from the live entry points. Some are reachable but obsolete (already covered by ma-irt/CLEANUP.md's Phase A; verify whether any survived).

### Live entry points (anything not transitively reachable from this set is a candidate)

- `ma-irt/scripts/train.py`
- `ma-irt/scripts/evaluate.py`
- `ma-irt/scripts/data_gen.py`, `data_gen_block.py`, `data_gen_randomwalk.py`, `data_gen_staircase.py`, `data_gen_imbalanced.py`
- `ma-irt/scripts/plot_metrics.py`, `plot_recovery_split.py`, `plot_trajectory_comparison.py`, `plot_theta_temporal.py`, `plot_learner_trajectories.py`, `plot_block_and_rw.py`, `plot_assistments_*.py`
- `ma-irt/scripts/generate_tables.py`, `gen_all_configs.py`, `gen_bench_configs.py`
- `ma-irt/scripts/run_bulk_retrain.sh`, `run_bench_sweep.sh`, `run_bench_phases_chain.sh`, `run_learned_sweep.sh`, `run_assist2009_ord.sh`, `run_bench_extra_2015_2017.sh`
- `ma-irt/scripts/mirt_baseline_all_k.R`, `mirt_predict.R`
- `ma-irt/tests/`

### Inventory (modules)

| Path | Status |
|---|---|
| `ma-irt/models/magpcm.py` | LIVE |
| `ma-irt/models/static_gpcm.py` | LIVE |
| `ma-irt/models/dynamic_gpcm.py` | LIVE |
| `ma-irt/models/dkvmn_softmax.py` | LIVE |
| `ma-irt/models/dkt.py` | LIVE (binary K=2 baseline, Table 2) |
| `ma-irt/models/dkvmn.py` | LIVE (binary K=2 baseline, Table 2) |
| `ma-irt/models/deep_irt.py` | LIVE (binary K=2 baseline, Table 2) |
| `ma-irt/models/components/{memory,irt,embeddings}.py` | LIVE |
| `ma-irt/models/heads/gpcm.py` | LIVE |
| `ma-irt/training/{trainer,losses}.py` | LIVE |
| `ma-irt/dataloading/loaders.py` | LIVE |
| `ma-irt/utils/metrics.py` | LIVE |
| `ma-irt/config/{loader,types}.py` | LIVE |

No dead modules found in the library. Either the previous round (`ma-irt/CLEANUP.md` Phase A) removed them or the verbose list in `CLEANUP_PLAN.md` Section 9 overstated dead code. The remaining dead-code work is in scripts and configs.

### Inventory (scripts, underscore-prefixed and one-off)

Underscore-prefixed scripts (33 total in `ma-irt/scripts/`):
- `_aggregate_bench.py`, `_aggregate_pykt_results.py`
- `_bench_table_draft.tex`, `_bench_writeup_draft.md`
- `_build_pykt_synthetic5.py`, `_convert_yeung_synthetic.py`
- `_emit_k4_tables.py`, `_extract_row.py`
- `_gen_chunked_bench_configs.py`, `_gen_imb_scale_pykt_configs.py`, `_gen_pykt_configs.py`, `_gen_table_rows.py`
- `_k4_digest.sh`, `_linking_learned.py`
- `_orbit_align_static_experiment.py`, `_profile_dkvmn.py`, `_profile_dkvmn_report.md`
- `_reeval_discrete.sh`, `_run_*.sh` (12 files), `_verify_*.py` (3 files)

All underscore-prefixed files are by convention one-offs. Inspection of the run-orchestrator `run_bulk_retrain.sh` confirms only `_extract_row.py` is called by a live shell script; `_aggregate_bench.py` is called by `_run_chunked_bench_seeds.sh` (itself underscore-prefixed, transitively dead).

Other obsolete scripts (per `ma-irt/CLEANUP_PLAN.md` Section 3.3 and verified against grep):
- `aggregate_recovery_v4.py` (v5 supersedes it)
- `analyze_threshold_ordering.py`, `compare_alpha1.py`, `diag_alpha_collapse.py`, `investigate_wol_threshold.py` (diagnostics)
- `eval_retrained.py` (one-off post-retrain eval)
- `gen_alpha1_configs.py`, `gen_raw_alpha_configs.py`, `gen_ablation_configs.py`, `gen_table_updates.py` (one-off config generators, output is in `configs/tmp_alpha1/` which is itself stale)
- `monitor.py`, `monitor_retrain.sh` (live monitoring for now-finished runs)
- `convert_dkvmn_format.py` (assists conversion, may still be referenced; verify)

Shell scripts to triage:
- `eval_all_collect.sh`, `eval_remaining.sh`, `eval_and_compare_learned.sh`, `rerun_*.sh`, `resweep_eval_all.sh`, `retrain_baselines.sh`, `run_after_chain.sh`, `run_after_imbalance.sh`, `run_remaining.sh`, `run_imbalance_extension.sh`, `train_ablations.sh`, `train_alpha1.sh`, `train_learned_repr.sh`, `gen_ablation_data.sh`
- Each of these reads a config dir or a fixed seed list and runs `train.py` / `evaluate.py`. Mostly orchestration that has been superseded by `run_bulk_retrain.sh`. Audit one-by-one and either fold into the bulk runner or archive.

### Inventory (configs)

- Root-level `ma-irt/configs/`. Mix of legacy (`block_*`, `rw_*`, `staircase_*` per-model files predating `bulk/`), and live smoke configs (`smoke.yaml`, `smoke_*.yaml`, `base.yaml`). The block/rw/staircase root files are superseded by `bulk/<dgp>_<model>_q200_k<K>_s<seed>.yaml`.
- `ma-irt/configs/_archive_s0p5/`. Already archived by name. Verify no live script reads it.
- `ma-irt/configs/tmp_alpha1/`. 50+ files for the alpha-prior-1 ablation. Inspect whether the ablation made it into the paper.
- `ma-irt/configs/dynamic_seeds/`. 160 files. Only referenced in `evaluate.py`'s usage docstring. Not iterated by `run_bulk_retrain.sh`. Likely superseded by `bulk/{continuous,discrete}_*` configs.
- `ma-irt/configs/experiments/`. 205 files across `rq1/`, `rq4/`, `rq5/`, `ablation/`. The bulk runner does not iterate these. Either fold into `bulk/` (rename and merge) or archive.

### Evidence

- Underscore prefix is a project-wide convention for one-offs (33 scripts follow it).
- `grep -E "scripts/" run_bulk_retrain.sh` shows only `train.py`, `evaluate.py`, and `_extract_row.py` are invoked. The bulk runner is the canonical pipeline.
- `_extract_row.py` is the only underscore-prefixed file that is live and must be kept.
- `ma-irt/configs/dynamic_seeds/` is only mentioned in `evaluate.py`'s usage block; no script iterates it.
- `ma-irt/configs/experiments/` is mentioned in `README.md` and `evaluate.py` docstrings; no script iterates it.
- The bulk runner's `configs/bulk/` set already covers static (K=2 to 6), discrete (block / staircase, K=3 to 6), continuous (random walk, K=3 to 6), assistments, and assist2009_ord across five seeds. Everything else duplicates the matrix in a previous naming scheme.

### Procedure

1. Pre-state scan. For each script and config directory, produce a reachability report.
   - For scripts. Search for the filename in `*.py`, `*.sh`, `*.md`. Tag as `LIVE_CALLED`, `ONLY_DOCS`, `UNREFERENCED`.
   - For configs. Match against `run_bulk_retrain.sh`'s expected paths and the regex it builds.
   - Output. `cleanup_t3_reachability.csv` with columns `path,kind,status,callers`.
2. User reviews `cleanup_t3_reachability.csv` and overrides any classification before any file moves.
3. Move (do not delete) `UNREFERENCED` items to `ma-irt/archive/scripts/` and `ma-irt/archive/configs/`. The archive directory already exists in the repo per `ma-irt/CLEANUP.md`. If it has been pruned, recreate it.
4. Verification step. **This is where the research-scientist agent's minimal verification suite is required.** Minimum.
   - Tests. `cd ma-irt && PYTHONPATH=. pytest tests/ -v`.
   - Smoke training. One epoch per model type (`magpcm`, `dkvmn_softmax`, `static_gpcm`, `dynamic_gpcm`, `dkt`, `dkvmn`, `deep_irt`) on `configs/smoke.yaml`.
   - Headline reproducibility. Pick a single (dgp, K, seed) cell from each paper table.
     - Table 1 K=4. `configs/bulk/static_magpcm_q200_k4_s42.yaml` to `outputs/static_magpcm_q200_k4_s42/best.pt`, evaluate with `scripts/evaluate.py single`, compare ACC, QWK, MAE to the K=4 row of `benchmarks.md` to within 1%.
     - Table 2. `configs/bulk/bench_deep_irt_assist2009_bin_pykt_fold0.yaml` if a checkpoint exists, else skip with a note.
     - Table 3 K=4. Same as Table 1 but check $r_\alpha$, $r_\beta$, $r_\theta$.
   - The research-scientist agent's manifest extends this list; use their version as the source of truth.
5. Record-keeping. Commit one tier-3 group at a time. `cleanup_t3_actions.txt` records every move. Verification CSVs go to `ma-irt/outputs/cleanup_verify/`.
6. Post-state scan. Re-grep for any of the moved filenames in `*.py`, `*.sh`, `*.md`. Hits should be limited to the archive and this plan.

### Risk and rollback

- Worst case. A config or script removed in this tier is silently used by a sweep script that runs only on the user's machine and not in CI. Symptom: a `MISSING_CONFIG` line appears in the next bulk sweep summary.
- Mitigation. Move, do not delete. Restore is one `git mv` away. The bulk-runner already prints `MISSING_CONFIG` and continues, so a missing config does not abort a sweep.
- Medium risk. The `experiments/` and `dynamic_seeds/` directories may still be referenced by an unsynced research notebook. If the user has such a notebook, surface the dependency before archiving.

---

## Tier 4. Config consolidation

Once Tier 3 has classified configs, this tier rationalizes the surviving set.

### Inventory

After Tier 3 the live config set should be.
- `ma-irt/configs/base.yaml`
- `ma-irt/configs/smoke.yaml` and `ma-irt/configs/smoke_*.yaml` (3 files for DKT, DKVMN, Deep-IRT smoke)
- `ma-irt/configs/bulk/` (1652 files, the canonical sweep matrix)

Open questions to settle.
- Are the per-DGP files at root (`block_q200_k3.yaml`, `rw_q200_k4.yaml`, `staircase_q200_k4.yaml` etc.) still needed for single-shot runs, or are all real users on `bulk/`?
- Are `bench_*_chunked30*` configs current, or did they get superseded by the pyKT-style 5-fold configs (`*_pykt_fold[0-4].yaml`)?
- Inside `bulk/`, both `assist2009_*` and `assist2009_ord_*` exist with overlapping model lists. One pair is binary, one is ordinal. Confirm the bulk runner targets exactly the live set.

### Procedure

1. Pre-state scan. Build a matrix.
   - Rows. `(dgp, K, model_type, seed)` for synthetic; `(dataset, fold, model)` for benchmarks.
   - Columns. config exists? checkpoint exists? eval CSV exists?
   - Source. iterate `configs/bulk/*.yaml`, `outputs/*/best.pt`, `outputs/*/recovery_metrics.json`.
   - Output. `cleanup_t4_matrix.csv`.
2. Cross-reference with the canonical-experiment manifest from the research-scientist agent. Any cell present in `bulk/` but not in the manifest is a candidate for archiving.
3. Decisions per group.
   - Synthetic static / discrete / continuous q200, K in {2,3,4,5,6}, models in {magpcm, dkvmn_gpcm, dkvmn_softmax, dynamic_gpcm, static_gpcm}, seeds in {0,1,7,42,123}. KEEP (these underlie Tables 1 and 3).
   - Benchmark configs (assist2009 binary, assist2017 binary, assist2015, synthetic5 x 5 versions x 5 folds, static_q200_k2 x 5 folds), models {dkt, dkvmn, deep_irt, dkvmn_gpcm, magpcm}. KEEP (these underlie Table 2).
   - `assist2009_ord_*` (ordinal experiment on ASSISTments). KEEP if the paper still reports the section. Confirm against `overleaf-sync/main.tex`.
   - `assistments_*` (ASSISTments 2017 separate pass). KEEP if Section in the paper still exists.
   - `bench_*_chunked` vs `bench_*_pykt_fold*`. KEEP the pykt_fold variants; archive `_chunked` and `_chunked30` if they are pre-pykt iterations.
   - Anything in `_archive_s0p5/` and `tmp_alpha1/`. ARCHIVE (the names already signal this).
4. Apply moves into `ma-irt/configs/_archive_2026/` (date-stamped). One commit per group.
5. Verification step. Re-run `bash scripts/run_bulk_retrain.sh --skip-existing --no-assist --no-eval --models magpcm --seeds 42 --Ks 4 --dgps static`. Should pick up one config, find an existing checkpoint, and print `SKIPPED`. Failure of this sanity check means a referenced config was archived in error.
6. Record-keeping. Save `cleanup_t4_matrix.csv` and `cleanup_t4_moves.txt`.
7. Post-state scan. Recount configs in `bulk/`. Target reduction. About 30% (rough estimate, depends on benchmark scope).

### Risk and rollback

- Worst case. A retraining run launches and skips a config that should exist. The bulk runner logs `MISSING_CONFIG` and continues.
- Rollback. Move from `_archive_2026/` back to `bulk/`. Single command.
- Medium risk. The research-scientist agent's manifest may be incomplete on the first pass. Defer ambiguous cells (`assist2009_ord_*` for example) to a second sub-tier rather than archive them on first guess.

---

## Tier 5. Deeper code refactoring (optional, document only)

Authorized only after Tiers 0 to 4 land and the codebase is stable for at least one full bulk retrain.

### Candidates

1. **Loss schema cleanup.** Remove `focal_weight` from `TrainingConfig` and from `CombinedLoss` if every active config has it at 0 (already true). Keep `FocalLoss` as a class for future experiments.
2. **`use_separable_embed` legacy flag** in `models/magpcm.py` and `models/dkvmn_softmax.py`. The previous CLEANUP_PLAN flagged it; verify whether the rename round actually completed.
3. **Linking utility extraction.** `link_alpha`, `link_normal`, `link_theta_irt` are duplicated in `compute_linking.py`, `evaluate.py`, `plot_recovery_split.py`. Extract to `ma-irt/utils/linking.py`. Pure refactor with regression tests on the recovery numbers.
4. **`build_model` move.** Currently lives only in `scripts/train.py`. `evaluate.py` re-implements model construction. Move to `ma-irt/models/__init__.py` and have both scripts import it.
5. **Trainer dispatch** in `training/trainer.py._forward()`. The `_model_type` attribute is set on the model in `build_model`. Refactor to a polymorphic `model.forward()` that accepts the same `(student_ids, questions, responses)` arity, even if some args are unused. Removes the dispatch fragility.
6. **README sync.** `ma-irt/README.md` references `scripts/run_all_experiments.sh`, `scripts/plot_recovery.py`, and `scripts/plot_recovery_figure.py`, none of which exist. Update to the actual entry points (`run_bulk_retrain.sh`, `plot_recovery_split.py`).
7. **`base.yaml` reconciliation.** Audit `ma-irt/configs/base.yaml` against the live config defaults; remove stale fields after the deprecation window from Tier 3.

### Procedure (template, per item)

1. Open an issue describing the change and the verification.
2. Write a regression test that pins the current behavior (recovery numbers, loss values, output keys).
3. Implement the change.
4. Run the **Tier 3 verification suite** plus the new regression test.
5. Land in one commit per item.

### Risk and rollback

- These changes touch live code. Each item must pass the full verification suite. Rollback by `git revert`.
- Document only for now. Do not act on Tier 5 until the user authorizes after Tier 4 closes.

---

## Coordination with the research-scientist agent

The research-scientist agent will produce, in parallel.

1. **Canonical-experiment manifest.** A CSV listing every `(dgp, K, model, seed)` and every `(dataset, fold, model)` cell required to reproduce `benchmarks.md` Tables 1 to 3 and the figures in `overleaf-sync/main.tex`. Input to Tier 3 (which scripts produce these results) and Tier 4 (which configs are needed).
2. **Verification suite.** A Python or shell script that, given a checkpoint directory, prints a pass/fail per headline metric with a 1% tolerance against `benchmarks.md`. Input to Tier 3 and Tier 4 verification steps. Suggested location, `ma-irt/scripts/verify_paper_metrics.py`.

Tier 0 and Tier 1 do not need either input. Tier 2 needs the verification suite only for the staged Stage B move. Tiers 3, 4, and 5 require both.

## Authorization workflow

For each tier in order.

1. User reads the tier's section in this plan.
2. User approves or amends the inventory.
3. Assistant runs the procedure and produces the artifacts listed under "Record-keeping".
4. Assistant runs the verification step.
5. User confirms verification passed.
6. Commit lands. Tier closes. Move to next.

Authorization is per-tier. No tier executes until the prior tier's verification has passed and is recorded.
