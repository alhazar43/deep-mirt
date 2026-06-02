# Plan: σ=0.3 → σ=0.5 regeneration + Full Retrain

## Preamble

**Scope.** Switch true α distribution from LogNormal(0, 0.3) to LogNormal(0, 0.5), matching the mirt (Chalmers 2012) default. Archive current artifacts, regenerate synthetic data, retrain all paper-relevant models, regenerate all tables and plots referenced in `main.tex`.

**σ=0.5 confirmation.** User wrote "lognormal(0.5, 0)" which is read as LogNormal(0, 0.5) since σ=0 is degenerate. Confirm before Phase 1.

**Models after latest setup.** MA-GPCM uses α raw from network (no exp, no prior). DKVMN+GPCM shares the same `MAGPCM` class with `separate_theta: false` so its fresh retrains also get raw α. Static GPCM and Dynamic GPCM use `α = exp(α_log_scale · α_raw)` with `α_log_scale=1.0`, no prior. DKVMN+Softmax has no α pathway.

**K coverage.** Paper uses K ∈ {2, 3, 4, 5, 6} for Static (K=2 is a binary compatibility check), and K ∈ {3, 4, 5, 6} for Discrete and Continuous. K=2 is regenerated and retrained for Static only; Discrete and Continuous remain K=3-6.

**Ablations postponed.** The Imbalanced variants and the Scalability study (Q=200..2000) are postponed to Phase 7. Paper sections and tables that reference them stay intact with current σ=0.3 numbers during Phases 1-6. They are regenerated in Phase 7 after the main bulk lands.

**ASSISTments retrain.** Real dataset files stay in place. Trained models get rebuilt using the newest configs (MA-GPCM and DKVMN+GPCM raw α; Static/Dynamic GPCM with α_log_scale=1.0). Paper claims about ASSISTments are threshold-ordering-rate based and do not quote α magnitudes, so no numeric claims invalidate.

**Track-and-commit per phase.** At the end of each phase, stage the specific files touched, commit with a scoped message, push if appropriate. Canonical workflow (adapt per phase):

```bash
cd C:/Users/steph/documents/deep-mirt
git status                                  # sanity check what changed
git add <scoped paths>                      # stage only what this phase owns
git commit -m "Phase N: <short scope>"      # scoped commit message
# push deferred until bulk retrain succeeds, unless phase is safe-to-push
```

Each phase below lists its exact `add` set under a "Track and commit" subsection so rollback is cheap.

---

## Phase 0. Archive

**Paper reference.** Affected artifacts are every table, figure, and CSV downstream of the synthetic DGPs. Preserve them so we can roll back or cross-reference.

### What to archive

| Category | Paths |
|---|---|
| Synthetic datasets | `ma-irt/data/static_q200_k{2,3,4,5,6}` `ma-irt/data/discrete_q200_k{3,4,5,6}` `ma-irt/data/continuous_q200_k{3,4,5,6}` `ma-irt/data/v2_q200_k4_{mild,severe,extreme}_imb` `ma-irt/data/v2_q200_k4` (alias of Static K=4 used in ablation paths) `ma-irt/data/block_q200_k{2,3,4,5,6}` `ma-irt/data/rw_q200_k{2,3,4,5,6}` `ma-irt/data/staircase_q200_k{3,4,5,6}` |
| Synthetic-5 and ASSIST2015 | not archived (not regenerated; binary-benchmark only) |
| Assistments proxy-ordinal data | not archived (real data, no regeneration) |
| All training outputs | `ma-irt/outputs/*` archived wholesale; per user, assistments output folders can go into the archive too |
| Paper figures | `overleaf-sync/figures/` snapshot |

### Commands

From repo root (`C:/Users/steph/documents/deep-mirt`):

```bash
STAMP=$(date +%Y%m%d_%H%M)
ARCHIVE=archive_sigma03_${STAMP}
mkdir -p "$ARCHIVE/data" "$ARCHIVE/outputs" "$ARCHIVE/figures"

# Synthetic datasets (do NOT archive assistments_ord_k4 / assist2009_ord_k4 / assist2015 / synthetic5)
for d in static_q200_k{2,3,4,5,6} discrete_q200_k{3,4,5,6} continuous_q200_k{3,4,5,6} \
         v2_q200_k4 v2_q200_k4_{mild,severe,extreme}_imb \
         block_q200_k{2,3,4,5,6} rw_q200_k{2,3,4,5,6} staircase_q200_k{3,4,5,6}; do
  [ -d "ma-irt/data/$d" ] && mv "ma-irt/data/$d" "$ARCHIVE/data/"
done

# All outputs (user confirmed assistments outputs can also be archived)
mv ma-irt/outputs/* "$ARCHIVE/outputs/" 2>/dev/null
mkdir -p ma-irt/outputs

# Paper figures snapshot
cp -r overleaf-sync/figures "$ARCHIVE/figures/figures_sigma03" 2>/dev/null || true

echo "Archive at $ARCHIVE"
```

**Paper verification.** After archive, the paper currently references every table listed below that will be regenerated. No `.tex` edits in Phase 0 — we preserve current numbers until Phase 4.

**Track and commit.**
```bash
# Archive directory is outside the ma-irt source tree, but record its name
git add ma-irt/.gitignore  # if archive_sigma03_*/ needs to be ignored
git commit -m "Phase 0: archive σ=0.3 datasets and outputs"
```

**Rename `assistments_ord_k4` → `assist2017_ord_k4`.** Current folder slug conflicts with `assist2009_ord_k4` naming and with configs that say `assist2009_*` but point at 2017 data. Phase 0 rename step:

```bash
mv ma-irt/data/assistments_ord_k4 ma-irt/data/assist2017_ord_k4
# Then grep all configs/scripts that reference assistments_ord_k4 and update
grep -rl "assistments_ord_k4" ma-irt/configs ma-irt/scripts | \
  xargs sed -i 's|assistments_ord_k4|assist2017_ord_k4|g'
```

Also rename `assist2009_*` configs that actually point at 2017 data to `assist2017_*` for consistency. Inspect before mass-renaming.

---

## Phase 1. Data Regeneration (σ=0.5)

**Paper reference.** `overleaf-sync/main.tex` line 364 (σ=0.3 → σ=0.5 in DGP description). Also cite Chalmers (2012) for the default.

### Patch generators

σ is hard-coded in four scripts. Preferred fix is a `--alpha_sigma` CLI flag on each, default 0.5. Minimal fix is an in-place sed.

| File | Line | Change |
|---|---|---|
| `ma-irt/scripts/data_gen.py` | 81 + docstring 24 | `sigma=0.3` → `sigma=0.5` |
| `ma-irt/scripts/data_gen_block.py` | 77 | `sigma=0.3` → `sigma=0.5` |
| `ma-irt/scripts/data_gen_randomwalk.py` | 71 | `sigma=0.3` → `sigma=0.5` |
| `ma-irt/scripts/data_gen_staircase.py` | 103 | `sigma=0.3` → `sigma=0.5` |
| `ma-irt/scripts/data_gen_imbalanced.py` | 82 | `sigma=0.3` → `sigma=0.5` |

### Regeneration commands

From `ma-irt/`:

```bash
# Static: K in {2,3,4,5,6}
for K in 2 3 4 5 6; do
  python scripts/data_gen.py --name static_q200_k${K} --n_students 5000 \
    --n_questions 200 --n_cats ${K} --min_seq 20 --max_seq 80 \
    --seed 42 --output_dir data
done

# Discrete staircase: K in {3,4,5,6}  (paper: T=60, three blocks of 20)
for K in 3 4 5 6; do
  python scripts/data_gen_staircase.py --name discrete_q200_k${K} \
    --n_students 5000 --n_questions 200 --n_cats ${K} \
    --seq_len 60 --seed 42 --output_dir data
done

# Continuous random walk: K in {3,4,5,6}  (paper: seq_len [40, 80])
for K in 3 4 5 6; do
  python scripts/data_gen_randomwalk.py --name continuous_q200_k${K} \
    --n_students 5000 --n_questions 200 --n_cats ${K} \
    --min_seq 40 --max_seq 80 --seed 42 --output_dir data
done

```
# Imbalanced variants postponed to Phase 7.

**Paper verification.**
- Static description (line 364) matches `data_gen.py` defaults (N=5000, Q=200, seq ∈ [20,80], β base from N(base, 0.5)).
- Discrete description (line 377) matches `data_gen_staircase.py` (T=60, three 20-item blocks, δ_1 ~ N(0.5, 0.3), δ_2 ~ N(0.4, 0.3), clipping [−0.5, 2.0]).
- Continuous description (line 379) matches `data_gen_randomwalk.py` (μ_drift ~ N(0.02, 0.01), σ_innov=0.1).
- Imbalanced generator left untouched but not invoked.

Post-regen sanity check: mean(α) ≈ 1.13, std(α) ≈ 0.60 for each dataset (vs. 1.05 / 0.32 under σ=0.3).

**Deliverables.** New data directories under `ma-irt/data/` with sigma=0.5 in each metadata.json.

**Track and commit.**
```bash
git add ma-irt/scripts/data_gen.py ma-irt/scripts/data_gen_block.py \
        ma-irt/scripts/data_gen_randomwalk.py ma-irt/scripts/data_gen_staircase.py \
        ma-irt/scripts/data_gen_imbalanced.py
git commit -m "Phase 1: data generators default to sigma=0.5 (mirt default)"
# Data directories themselves are gitignored. metadata.json change is captured in the code patch above.
```

---

## Phase 2. Model Config Check

**Paper reference.** Models and Baselines section (line 394-404). Training Setup (line 411).

### Checklist per config

For every bulk config in `ma-irt/configs/bulk/` matching the paper's experimental grid.

| Model | `model_type` | Special flags | Epochs |
|---|---|---|---|
| Static GPCM (`static_gpcm`) | static_gpcm | `alpha_log_scale: 1.0` required | 150 |
| Dynamic GPCM (`dynamic_gpcm`) | dynamic_gpcm | `alpha_log_scale: 1.0` required | 50 |
| MA-GPCM (`magpcm`) | magpcm | `separate_theta: true` (default); no prior flags | 30 |
| DKVMN+GPCM (`dkvmn_gpcm`) | magpcm | `separate_theta: false` | 30 |
| DKVMN+Softmax (`dkvmn_softmax`) | dkvmn_softmax | — | 30 |
| GPCM (EM) | R mirt | runs via `scripts/mirt_baseline_all_k.R` | — |

### Required config patches

Every `{static,discrete,continuous}_{static,dynamic}_gpcm_q200_k{3,4,5,6}_s{0,1,7,42,123}.yaml` (and K=2 variants where they exist) must carry `alpha_log_scale: 1.0` in the `model` section. Today, most default to 0.3 via the dataclass, so most need patching.

```bash
cd ma-irt/configs/bulk
for f in \
  {static,discrete,continuous}_static_gpcm_q200_k{2,3,4,5,6}_s{0,1,7,42,123}.yaml \
  {static,discrete,continuous}_dynamic_gpcm_q200_k{2,3,4,5,6}_s{0,1,7,42,123}.yaml; do
  [ -f "$f" ] || continue
  grep -q 'alpha_log_scale' "$f" && continue
  sed -i 's/^  ability_scale: 1\.0$/  ability_scale: 1.0\n  alpha_log_scale: 1.0/' "$f"
done

# Verify: should print nothing
for f in {static,discrete,continuous}_{static,dynamic}_gpcm_q200_k{3,4,5,6}_s*.yaml; do
  grep -L 'alpha_log_scale: 1.0' "$f"
done
```

### Other config audits

1. No config sets `alpha_prior_sigma` or `alpha_weight_decay` for bulk paper runs. Confirm. Ablation variants (`_wd`, `_wdprior`, `_noprior`) are separate and not touched.
2. Epochs match the table above across all bulk configs (spot-check several per DGP).
3. Seeds `{0, 1, 7, 42, 123}` present in all (dgp, K, model) cells.
4. Dataset names match Phase 1 outputs (`static_q200_k{K}`, `discrete_q200_k{K}`, `continuous_q200_k{K}`). If Phase 1 changed names, update `data.dataset_name` in all bulk configs.

**Paper verification.**
- Line 394-404 (Models and Baselines): six models listed. Bulk must cover all five PyTorch models; R mirt EM runs separately.
- Line 411 (Training Setup): batch=64, Adam lr=1e-3, ReduceLROnPlateau (factor 0.8, patience 5), no dropout, no early stopping, full epochs. Spot-check configs.
- Line 364 (Static DGP description): will be updated in Phase 4 when σ=0.3 is replaced with σ=0.5.

**Deliverables.** All bulk configs confirmed paper-consistent. No training yet.

**Track and commit.**
```bash
git add ma-irt/configs/bulk/
git commit -m "Phase 2: alpha_log_scale=1.0 on Static/Dynamic GPCM bulk configs"
```

---

## Phase 3. Bulk Retrain

**Paper reference.** Table 1-style recovery (`tab:irt_recovery_k` line 522), Discrete (`tab:block_recovery` line 729), Continuous (`tab:rw_recovery` line 827), and full appendix tables.

### Scope (350 PyTorch runs + 13 R EM runs)

| DGP | K | Models | Seeds | Cells |
|---|---|---|---|---|
| Static | {2, 3, 4, 5, 6} | all 5 | {0, 1, 7, 42, 123} | 125 |
| Discrete | {3, 4, 5, 6} | all 5 | same | 100 |
| Continuous | {3, 4, 5, 6} | all 5 | same | 100 |
| ASSISTments 2017 | K=4 | all 5 | same | 25 |
| EM baseline (R mirt) | Static {2,3,4,5,6}, Discrete {3,4,5,6}, Continuous {3,4,5,6} | 1 EM | N/A | 13 R runs |

**Total: 350 PyTorch + 13 R**. Imbalanced and Scalability runs are postponed to Phase 7.

### Orchestration

`ma-irt/scripts/run_bulk_retrain.sh` + `ma-irt/scripts/aggregate_recovery.py`.

```bash
#!/usr/bin/env bash
# ma-irt/scripts/run_bulk_retrain.sh
set -uo pipefail

DGPS=(static discrete continuous)
MODELS=(static_gpcm dynamic_gpcm magpcm dkvmn_gpcm dkvmn_softmax)
SEEDS=(0 1 7 42 123)
# K sets are per-DGP: K=2 is Static-only. See inner loop for branching.
SKIP_EXISTING=0
RUN_EVAL=1
CONFIG_DIR="configs/bulk"
SUMMARY="outputs/bulk_summary_$(date +%Y%m%d_%H%M%S).csv"

# argparse: --dgps --Ks --models --seeds --skip-existing --no-eval --config-dir

export PYTHONPATH=.
export KMP_DUPLICATE_LIB_OK=TRUE

mkdir -p outputs
echo "config,status,train_sec,eval_sec,best_epoch,qwk,r_theta,r_alpha,r_beta" > "$SUMMARY"

total=0; done_n=0; failed=0; skipped=0
for s in "${SEEDS[@]}"; do
  for dgp in "${DGPS[@]}"; do
    # Per-DGP K set: K=2 only for Static.
    if [[ "$dgp" == "static" ]]; then ks=(2 3 4 5 6); else ks=(3 4 5 6); fi
    for k in "${ks[@]}"; do
      for m in "${MODELS[@]}"; do
        ((total++))
        name="${dgp}_${m}_q200_k${k}_s${s}"
        cfg="$CONFIG_DIR/${name}.yaml"
        out="outputs/${name}"
        ckpt="${out}/best.pt"

        [[ ! -f "$cfg" ]] && { echo "$cfg,MISSING_CONFIG,,,,,,," >> "$SUMMARY"; ((failed++)); continue; }
        [[ -f "$ckpt" && $SKIP_EXISTING -eq 1 ]] && { ((skipped++)); echo "$cfg,SKIPPED,,,,,,," >> "$SUMMARY"; continue; }

        t0=$SECONDS
        if python scripts/train.py --config "$cfg" 2>&1 | tee "${out}.train.log"; then
          train_s=$((SECONDS-t0))
          t1=$SECONDS; eval_s=0
          if [[ $RUN_EVAL -eq 1 ]]; then
            data_dir=$(python -c "import yaml; c=yaml.safe_load(open('$cfg')); print(c['data']['data_dir'])")
            python scripts/evaluate.py single \
              --config "$cfg" --checkpoint "$ckpt" --data-dir "$data_dir" \
              >> "${out}.eval.log" 2>&1 || true
            eval_s=$((SECONDS-t1))
          fi
          python scripts/_extract_row.py "$out" "$cfg" "$train_s" "$eval_s" >> "$SUMMARY"
          ((done_n++))
        else
          echo "$cfg,FAILED,,,,,,," >> "$SUMMARY"; ((failed++))
        fi
      done
    done
  done
done

echo "done: $done_n failed: $failed skipped: $skipped total: $total"
```

Seed is outermost so partial runs cover all `(dgp, K, model)` cells early.

### ASSISTments 2017 orchestration (separate loop)

The ASSISTments 2017 proxy-ordinal dataset is K=4 real data. Configs are in `configs/bulk/assist2009_*_learned_s{0,1,7,42,123}.yaml`. (Filename slug says "2009" but the dataset directory resolved is `assistments_ord_k4`, which is the 2017 proxy-ordinal file. Confirm before launch if ambiguous.)

Append after the synthetic loop in the same script:

```bash
ASSIST_MODELS=(static_gpcm dynamic_gpcm magpcm dkvmn_gpcm dkvmn_softmax)
for s in "${SEEDS[@]}"; do
  for m in "${ASSIST_MODELS[@]}"; do
    name="assist2009_${m}_learned_s${s}"
    cfg="$CONFIG_DIR/${name}.yaml"
    # train + inline eval (same block as synthetic loop above)
  done
done
```

No ground-truth recovery here; eval emits prediction metrics (QWK, ACC, MAE) and item parameters for qualitative inspection.

### EM baseline (R mirt)

Separate pass after PyTorch bulk.

```bash
cd ma-irt
Rscript scripts/mirt_baseline_all_k.R --dgp static --Ks 2,3,4,5,6
Rscript scripts/mirt_baseline_all_k.R --dgp discrete --Ks 3,4,5,6
Rscript scripts/mirt_baseline_all_k.R --dgp continuous --Ks 3,4,5,6
```

### Wall-clock estimates (RTX 4060 Laptop, single GPU)

65 synthetic runs per model (25 Static {K=2-6} + 20 Discrete {K=3-6} + 20 Continuous {K=3-6}) + 5 ASSISTments = 70 runs per model.

| Model | Per-run epochs | Per-run sec | Total runs | Total hours |
|---|---|---|---|---|
| Static GPCM | 150 | ~90 | 70 | 1.75 |
| Dynamic GPCM | 50 | ~420 | 70 | 8.2 |
| MA-GPCM | 30 | ~72 | 70 | 1.4 |
| DKVMN+GPCM | 30 | ~72 | 70 | 1.4 |
| DKVMN+Softmax | 30 | ~60 | 70 | 1.2 |
| Inline eval | — | ~15 | 350 | 1.5 |
| R mirt EM | — | ~60-120 per K | 13 | ~0.3 |
| **Total** | | | **350 + 13** | **~15.7** |

Plan for overnight plus morning slack.

**Paper verification.**
- Output `recovery_metrics.json` keys must include `r_alpha`, `r_beta_mean`, `r_theta`, `rmse_alpha_raw`, `rmse_beta_mean_raw`, `rmse_theta_raw`, `bias_alpha_raw`, `bias_beta_mean_raw`, `bias_theta_raw`, `val_qwk`, `val_acc`, `val_mae`, `rho_alpha`, `rho_beta_mean`, `rho_theta`. These back the main-text tables.
- Raw metrics feed the paper (per latest decision). Linked metrics can still be present in the JSON but are not used.

**Deliverables.**
- `ma-irt/scripts/run_bulk_retrain.sh` (orchestrator per skeleton above).
- `ma-irt/scripts/_extract_row.py` (helper that reads `outputs/<name>/recovery_metrics.json` and emits one CSV row). **Does not exist today, must be created before launch.**
- 350 PyTorch checkpoints + per-run `recovery_metrics.json` + `bulk_summary_*.csv`.
- 13 R mirt JSON outputs.

### Dry-run cell (mandatory pre-launch)

Run one end-to-end cell before committing the overnight batch. ~5 min.

```bash
cd ma-irt
# (a) regen one tiny dataset to exercise the generator patch
python scripts/data_gen.py --name _smoke_static_q40_k4 --n_students 200 \
  --n_questions 40 --n_cats 4 --min_seq 20 --max_seq 40 --seed 42 --output_dir data

# (b) one fast model end-to-end (override epochs via --epochs flag if supported)
PYTHONPATH=. KMP_DUPLICATE_LIB_OK=TRUE python scripts/train.py \
  --config configs/bulk/static_magpcm_q200_k4_s42.yaml --epochs 2

# (c) inline eval emits recovery_metrics.json
PYTHONPATH=. python scripts/evaluate.py single \
  --config configs/bulk/static_magpcm_q200_k4_s42.yaml \
  --checkpoint outputs/static_magpcm_q200_k4_s42/best.pt \
  --data-dir data/static_q200_k4

# (d) confirm _extract_row.py parses cleanly
python scripts/_extract_row.py outputs/static_magpcm_q200_k4_s42 \
  configs/bulk/static_magpcm_q200_k4_s42.yaml 60 10

# (e) plot_theta_temporal.py patched and renders
python scripts/plot_theta_temporal.py --K 4 \
  --deepgpcm-config configs/bulk/static_magpcm_q200_k4_s42.yaml \
  ...  # full arg set
```

If any step fails, fix before launching overnight.

**Track and commit.**
```bash
# Orchestration + helper scripts after dry-run passes
git add ma-irt/scripts/run_bulk_retrain.sh ma-irt/scripts/_extract_row.py
git commit -m "Phase 3 prep: bulk retrain orchestrator + row extractor"
# Checkpoints (best.pt, last.pt, metrics.csv) are gitignored; commit after bulk run.
# After bulk completes:
git add ma-irt/outputs/bulk_summary_*.csv
git commit -m "Phase 3: bulk retrain summary CSV (350 PyTorch + 13 R runs)"
```

---

## Phase 4. Results Aggregation

**Paper reference.** Every result table in main text and appendix.

### Aggregator

`ma-irt/scripts/aggregate_recovery.py` reads all `outputs/*/recovery_metrics.json`, joins with R mirt outputs, pivots per-(DGP, K) to 5-seed mean ± std.

Emits:

| CSV | Rows | Columns | Paper target |
|---|---|---|---|
| `outputs/recovery_summary.csv` | all runs | raw per-seed metrics | appendix cross-check |
| `outputs/recovery_table_static_K{3,4,5,6}.csv` | 6 models | r/RMSE/bias per (α,β,θ) | `tab:irt_recovery_k` 522, `tab:irt_recovery_k_full` 1096 |
| `outputs/recovery_table_discrete_K{3,4,5,6}.csv` | same | same | `tab:block_recovery` 729 |
| `outputs/recovery_table_continuous_K{3,4,5,6}.csv` | same | same | `tab:rw_recovery` 827 |
| `outputs/prediction_table_{static,discrete,continuous}_K{2,3,4,5,6}.csv` | same | QWK/ACC/MAE/τ | `tab:comp_results` 451, `tab:combined_perf` 500, `tab:block_prediction` 683, `tab:rw_prediction` 781 |
| `outputs/prediction_table_assist2017.csv` | 5 models | QWK/ACC/MAE/τ (no ground-truth recovery) | ASSISTments row of `tab:assistments_pred` 888 |

### Paper verification

Walk each table in the paper and confirm the regenerated CSV has the same row/column set.

- **`tab:comp_results`** (line 451): primary prediction table, Static K=2-6, all 6 models. → `prediction_table_static_K{2,3,4,5,6}.csv`.
- **`tab:combined_perf`** (line 500): combined Static/Binary main table. → `prediction_table_static_K*.csv` plus `prediction_table_binary.csv` (binary pass, separate).
- **`tab:irt_recovery_k`** (line 522): main recovery table, Static. → `recovery_table_static_K*.csv` with 5-col layout `r_α, r̄_β, r_θ, RMSE_θ, bias_θ`.
- **`tab:recovery`** (line 605): kept as σ=0.3 during Phases 1-6, regenerated in Phase 7.
- **`tab:imbalance`** (line 648): kept as σ=0.3 during Phases 1-6, regenerated in Phase 7.
- **`tab:block_prediction`** (line 683) + **`tab:block_recovery`** (line 729): Discrete.
- **`tab:rw_prediction`** (line 781) + **`tab:rw_recovery`** (line 827): Continuous.
- **`tab:recovery_appendix`** (line 1041): kept as σ=0.3 during Phases 1-6, regenerated in Phase 7.
- **`tab:imbalance_appendix`** (line 1072): kept as σ=0.3 during Phases 1-6, regenerated in Phase 7.
- **`tab:irt_recovery_k_full`** (line 1096): full recovery appendix.
- **`tab:assistments_pred`** (line 888): ASSISTments prediction. → `prediction_table_assist2017.csv`.

**Deliverables.** Per-DGP per-K CSVs ready for direct paste into `main.tex` tables.

**Track and commit.**
```bash
git add ma-irt/scripts/aggregate_recovery.py
git add ma-irt/outputs/recovery_summary.csv ma-irt/outputs/recovery_table_*.csv \
        ma-irt/outputs/prediction_table_*.csv
git commit -m "Phase 4: recovery and prediction CSVs aggregated across 5 seeds"
```

---

## Phase 5. Plots

**Paper reference.** All figures referenced in `main.tex`.

### Figures to regenerate

| Label | Line | Script | Inputs | Output |
|---|---|---|---|---|
| `fig:recovery_k4_item` | 585 | `plot_recovery_split.py` | Static K=4 checkpoints (MA-GPCM, Static GPCM, Dynamic GPCM) | `figures/recovery_k4_item.pdf` |
| `fig:theta_temporal_k4` | 578 | `plot_theta_temporal.py` | Static K=4 (MA-GPCM, Dynamic GPCM) | `figures/theta_temporal_k4.pdf` |
| `fig:block_traj` | 677 | `plot_trajectory_comparison.py` | Discrete K=4 (MA-GPCM, DKVMN+GPCM, Dynamic GPCM) | `figures/discrete_trajectories_k4.pdf` |
| `fig:rw_traj` | 775 | `plot_trajectory_comparison.py` | Continuous K=4 (same three) | `figures/continuous_trajectories_k4.pdf` |
| `fig:recovery_k{2,3,5,6}_item` | 1008, 1015, 1022, 1029 | `plot_recovery_split.py` | Static per K | `figures/recovery_k{K}_item.pdf` |
| `fig:theta_temporal_k{2,3,5,6}` | 978, 985, 992, 999 | `plot_theta_temporal.py` | Static per K | `figures/theta_temporal_k{K}.pdf` |
| `fig:assistments_theta` | 910 | `plot_assistments_theta.py` | Assistments 2017 | `figures/assistments_theta_population.pdf` |
| `fig:assistments_item_params` | 920 | `plot_assistments_item_params.py` | Assistments 2017 | `figures/assistments_item_params_3model.pdf` |

### Regeneration commands

```bash
cd ma-irt

# Recovery scatters (Static K=2-6)
for K in 2 3 4 5 6; do
  python scripts/plot_recovery_split.py \
    --deepgpcm-config  configs/bulk/static_magpcm_q200_k${K}_s42.yaml \
    --deepgpcm-checkpoint outputs/static_magpcm_q200_k${K}_s42/best.pt \
    --static-config    configs/bulk/static_static_gpcm_q200_k${K}_s42.yaml \
    --static-checkpoint outputs/static_static_gpcm_q200_k${K}_s42/best.pt \
    --dynamic-config   configs/bulk/static_dynamic_gpcm_q200_k${K}_s42.yaml \
    --dynamic-checkpoint outputs/static_dynamic_gpcm_q200_k${K}_s42/best.pt \
    --output ../overleaf-sync/figures/recovery_k${K}
done

# Theta temporal (Static per K)
for K in 2 3 4 5 6; do
  python scripts/plot_theta_temporal.py \
    --deepgpcm-config  configs/bulk/static_magpcm_q200_k${K}_s42.yaml \
    --deepgpcm-checkpoint outputs/static_magpcm_q200_k${K}_s42/best.pt \
    --dynamic-config   configs/bulk/static_dynamic_gpcm_q200_k${K}_s42.yaml \
    --dynamic-checkpoint outputs/static_dynamic_gpcm_q200_k${K}_s42/best.pt \
    --output ../overleaf-sync/figures/theta_temporal_k${K}
done

# Trajectory (Discrete and Continuous K=4)
for dgp in discrete continuous; do
  python scripts/plot_trajectory_comparison.py \
    --dataset-type $dgp --data-dir data/${dgp}_q200_k4 \
    --magpcm-config     configs/bulk/${dgp}_magpcm_q200_k4_s42.yaml \
    --magpcm-checkpoint outputs/${dgp}_magpcm_q200_k4_s42/best.pt \
    --dkvmn-gpcm-config     configs/bulk/${dgp}_dkvmn_gpcm_q200_k4_s42.yaml \
    --dkvmn-gpcm-checkpoint outputs/${dgp}_dkvmn_gpcm_q200_k4_s42/best.pt \
    --dynamic-config     configs/bulk/${dgp}_dynamic_gpcm_q200_k4_s42.yaml \
    --dynamic-checkpoint outputs/${dgp}_dynamic_gpcm_q200_k4_s42/best.pt \
    --output ../overleaf-sync/figures/${dgp}_trajectories_k4
done

# Assistments (real data, separate retrain path — unaffected by σ change)
python scripts/plot_assistments_theta.py ... # existing invocation
python scripts/plot_assistments_item_params.py ... # existing invocation
```

### Raw-scale audit for plot scripts

Already verified:
- `plot_recovery_split.py`: raw α vs raw truth, raw β vs raw truth, raw θ KDE.
- `plot_trajectory_comparison.py`: raw θ_t trajectories.
- `plot_assistments_*`: no synthetic truth to link against; shows raw estimates.

### Phase 5 prerequisite: patch `plot_theta_temporal.py` to raw

Currently `normalize_theta` (lines 79-85) z-scores both estimated and true θ. Per user decision, drop the z-score. Minimal three-line replacement:

```python
def normalize_theta(thetas_list, theta_true):
    """Return raw estimated trajectories and raw true theta."""
    return list(thetas_list), np.asarray(theta_true)
```

This preserves the call-site contract (`deep_normed, true_normed = ...` at lines 175-176) so nothing downstream breaks. Apply before running the Phase 5 regen loop.

**Paper verification.** After regenerating every figure, diff the old and new PDFs via `pdfdiff` or visual spot-check. Expect magnitudes to shift (wider α range means wider scatters and steeper theta trajectories) but relative model ordering to hold.

**Deliverables.** Every figure under `overleaf-sync/figures/` replaced with σ=0.5 versions. Paper compiles with `pdflatex main.tex` and renders new figures.

**Track and commit.**
```bash
# plot_theta_temporal.py patched before regen
git add ma-irt/scripts/plot_theta_temporal.py
git commit -m "Phase 5 prep: drop z-score normalization in plot_theta_temporal"
# Figures live in the overleaf-sync submodule
cd overleaf-sync
git add figures/recovery_k*_item.pdf figures/theta_temporal_k*.pdf \
        figures/discrete_trajectories_k4.pdf figures/continuous_trajectories_k4.pdf \
        figures/assistments_*.pdf
git commit -m "Phase 5: regenerate figures under σ=0.5 raw-scale convention"
cd ..
```

---

## Phase 6. Paper Update

**File.** `overleaf-sync/main.tex` and `ref.bib`.

### Required edits (Phases 1-6 only; ablation sections untouched)

| Section | Line | Change |
|---|---|---|
| Static DGP description | 364 | `\mathrm{LogNormal}(0, 0.3)` → `\mathrm{LogNormal}(0, 0.5)` + Chalmers (2012) citation |
| Static recovery tables | 451, 500, 522, 1096 | Replace numeric content from Phase 4 CSVs |
| Discrete recovery tables | 683, 729 | Replace numeric content from Phase 4 CSVs |
| Continuous recovery tables | 781, 827 | Replace numeric content from Phase 4 CSVs |
| ASSISTments prediction | 888 | Replace numeric content from Phase 4 CSVs |
| Main-text prose | 563 onward | Spot-check magnitudes; update `bias_α` range in line 566 and similar if the numbers shift |
| **Ablation-dependent sections (LEFT UNCHANGED during Phases 1-6)** | — | `tab:recovery` (605), `tab:recovery_appendix` (1041), `tab:imbalance` (648), `tab:imbalance_appendix` (1072), `\subsubsection{Scalability and Robustness}` (589), intro line 66. These carry σ=0.3 numbers until Phase 7 refreshes them. |
| `ref.bib` | — | No new entries needed; `chalmers_mirt_2012` already present |

**Paper verification.** After all edits, recompile `pdflatex main.tex`. Should produce ~40 pages, no undefined references, no bib warnings that weren't already there.

**Track and commit.**
```bash
cd overleaf-sync
git add main.tex ref.bib
git commit -m "Phase 6: σ=0.5 paper numbers, DGP description, Chalmers citation"
cd ..
```

---

---

## Phase 7. Ablation Refresh (deferred)

**Paper reference.** `\subsubsection{Scalability and Robustness}` (main.tex lines 589-662) plus the appendix tables `tab:recovery_appendix` (1041) and `tab:imbalance_appendix` (1072).

### Scope

- **Imbalanced**: 3 θ-prior variants at K=4 (mild, severe, extreme), 5 models, 5 seeds = 75 runs.
- **Scalability**: Q ∈ {200, 500, 1000, 2000} on Static K=4, 5 models, 5 seeds = 100 runs (subset depending on how many Q values we keep).

Total Phase 7 runs ≈ 75-175 depending on scalability grid choice. Budget ~3-6 hours on RTX 4060.

### Actions

1. Regenerate Imbalanced datasets at σ=0.5 with the Phase 1 generators:
   ```bash
   for prior in mild:0.5 severe:1.0 extreme:1.5; do
     name=v2_q200_k4_${prior%:*}_imb
     tmean=${prior#*:}
     python scripts/data_gen_imbalanced.py --name $name --n_students 5000 \
       --n_questions 200 --n_cats 4 --theta_mean $tmean --theta_std 1.0 \
       --min_seq 20 --max_seq 80 --seed 42 --output_dir data
   done
   ```
2. Regenerate Scalability datasets at σ=0.5 for each Q value.
3. Launch bulk retrain on these configs (same `run_bulk_retrain.sh` with `--dgps imbalanced scalability`, or a separate driver).
4. Aggregate into `recovery_table_imbalance.csv` and `recovery_table_scalability.csv`.
5. Regenerate any figures tied to these tables.
6. Update main.tex lines 605 (`tab:recovery`), 648 (`tab:imbalance`), 1041 (`tab:recovery_appendix`), 1072 (`tab:imbalance_appendix`), and 66 (intro claims) with the σ=0.5 numbers.

**Track and commit.**
```bash
# Code patches first (if run_bulk_retrain.sh needs ablation grid)
git add ma-irt/scripts/run_bulk_retrain.sh
git commit -m "Phase 7 prep: ablation grid support in bulk orchestrator"
# After bulk completes
git add ma-irt/outputs/recovery_table_imbalance.csv ma-irt/outputs/recovery_table_scalability.csv
git commit -m "Phase 7: imbalance and scalability refresh under σ=0.5"
cd overleaf-sync
git add main.tex figures/
git commit -m "Phase 7: update ablation sections to σ=0.5"
cd ..
```

**Deliverables.** Main.tex ablation sections carry σ=0.5 numbers. Paper is internally consistent.

---

## Execution order (actionable)

1. **Phase 0**: archive + rename `assistments_ord_k4` to `assist2017_ord_k4`. ~10 min. Commit.
2. σ=0.5 already confirmed.
3. **Phase 1**: patch generators and regenerate data (Static K=2-6, Discrete K=3-6, Continuous K=3-6). ~20 min. Commit.
4. **Phase 2**: patch `alpha_log_scale: 1.0` into Static/Dynamic GPCM configs. Verify epochs and seeds. ~5 min. Commit.
5. **Phase 3 prep**: write `scripts/_extract_row.py`, write `scripts/run_bulk_retrain.sh`, patch `plot_theta_temporal.py` to raw. ~20 min. Commit prep.
6. **Dry-run cell** (regen tiny dataset, train 2 epochs, eval, extract_row, render one figure). ~5 min.
7. **Phase 3**: bulk retrain (350 PyTorch + 13 R mirt). ~15.7 hours. Overnight. Commit summary after.
8. **Phase 4**: aggregate CSVs. ~10 min. Commit.
9. **Phase 5**: regenerate all figures (main Phase 1-6 set, ablation figures stay σ=0.3 until Phase 7). ~15 min. Commit.
10. **Phase 6**: paper edits for non-ablation sections (DGP desc, Static/Discrete/Continuous tables, ASSISTments prediction). Ablation sections kept as σ=0.3. Recompile. ~30 min. Commit.
11. **Phase 7** (deferred): Imbalanced and Scalability retrain under σ=0.5, paper sections updated. ~3-6 hours. Commit.

**Total active time to Phase 6**: ~1.75 hours. **Total wall time to Phase 6**: ~17 hours. Phase 7 can run later as an independent batch.

## Risks and open questions

- **EM baseline time**. R mirt on K=6 can be slow; budget 5-10 min per run.
- **Synthetic-5 and ASSIST2015 binary bench**. Not σ-dependent, not archived (per Phase 0). Old checkpoints were archived; restore or retrain as needed for `tab:combined_perf` binary rows.
- **ASSISTments filename vs data directory**. Configs are named `assist2009_*_learned_s*.yaml` but the data dir resolves to `assistments_ord_k4` (2017 proxy-ordinal). Confirm before launch that the 2009 slug is the intended 2017 experiment and not the excluded 2009 dataset.
- **K=2 Static** safe under σ=0.5 (per research-scientist review), 2PL ICCs just sharpen. No degeneracy risk.
- **Raw θ KDE comparability across models**. Both MA-GPCM and Dynamic GPCM are trained against raw N(0,1) truth, so raw trajectories are on comparable scale. No visual fairness issue.
- **ASSISTments α magnitudes**. Paper claims are threshold-ordering-rate based (line 881), not α magnitudes. Retraining with raw α does not invalidate any numeric claim.
