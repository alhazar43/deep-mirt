#!/bin/bash
# Comprehensive bulk training script
# Q ∈ {200, 500, 1000, 2000}
# K ∈ {2, 3, 4, 5, 6}
# Embeddings: linear_decay, separable, static_item
# Plus ablations at Q=200, K=4

set -e
export PYTHONPATH=src
export KMP_DUPLICATE_LIB_OK=TRUE

echo "=========================================="
echo "COMPREHENSIVE BULK TRAINING"
echo "=========================================="
echo ""

# ============================================================================
# STEP 1: Generate all datasets
# ============================================================================
echo "=== STEP 1: Generating datasets ==="
echo ""

Q_SIZES=(200 500 1000 2000)
K_VALUES=(2 3 4 5 6)

for Q in "${Q_SIZES[@]}"; do
    for K in "${K_VALUES[@]}"; do
        DATASET_NAME="large_q${Q}_k${K}"
        echo "Generating ${DATASET_NAME}..."

        if [ $Q -eq 200 ]; then
            N_STUDENTS=5000
            MIN_SEQ=20
            MAX_SEQ=80
        elif [ $Q -eq 500 ]; then
            N_STUDENTS=5000
            MIN_SEQ=20
            MAX_SEQ=80
        elif [ $Q -eq 1000 ]; then
            N_STUDENTS=5000
            MIN_SEQ=20
            MAX_SEQ=80
        else  # Q=2000
            N_STUDENTS=3000
            MIN_SEQ=50
            MAX_SEQ=120
        fi

        python scripts/data_gen.py \
            --name ${DATASET_NAME} \
            --n_students ${N_STUDENTS} \
            --n_questions ${Q} \
            --n_cats ${K} \
            --min_seq ${MIN_SEQ} \
            --max_seq ${MAX_SEQ} \
            --output_dir data
    done
done

echo ""
echo "=== Dataset generation complete ==="
echo ""

# ============================================================================
# STEP 2: Generate all config files
# ============================================================================
echo "=== STEP 2: Generating config files ==="
echo ""

mkdir -p configs/generated

# Function to create config
create_config() {
    local EXP_NAME=$1
    local Q=$2
    local K=$3
    local EMBEDDING=$4
    local DATASET=$5
    local MONOTONIC=${6:-true}
    local FOCAL_WEIGHT=${7:-0.5}
    local WOL_WEIGHT=${8:-0.5}

    cat > configs/generated/${EXP_NAME}.yaml <<EOF
base:
  experiment_name: "${EXP_NAME}"
  device: "cuda"
  seed: 42

model:
  n_questions: ${Q}
  n_categories: ${K}
  n_traits: 1
  memory_size: 50
  key_dim: 64
  value_dim: 64
  summary_dim: 50
  embedding_type: "${EMBEDDING}"
  ability_scale: 1.0
  dropout_rate: 0.0
  memory_add_activation: "tanh"
  init_value_memory: false
  monotonic_betas: ${MONOTONIC}

training:
  epochs: 15
  batch_size: 64
  lr: 0.001
  grad_clip: 1.0
  focal_weight: ${FOCAL_WEIGHT}
  weighted_ordinal_weight: ${WOL_WEIGHT}
  ordinal_penalty: 0.5
  lr_patience: 10
  lr_factor: 0.9
  attention_entropy_weight: 0.0
  theta_norm_weight: 0.0
  alpha_prior_weight: 0.0
  beta_prior_weight: 0.0

data:
  data_dir: "data"
  dataset_name: "${DATASET}"
  train_split: 0.8
  min_seq_len: 10
EOF
}

# Main experiments: Q × K × Embedding
EMBEDDINGS=("linear_decay" "separable" "static_item")

for Q in "${Q_SIZES[@]}"; do
    for K in "${K_VALUES[@]}"; do
        DATASET="large_q${Q}_k${K}"
        for EMB in "${EMBEDDINGS[@]}"; do
            EXP_NAME="q${Q}_k${K}_${EMB}"
            echo "Creating config: ${EXP_NAME}"
            create_config ${EXP_NAME} ${Q} ${K} ${EMB} ${DATASET}
        done
    done
done

# Ablations at Q=200, K=4
echo "Creating ablation configs..."

# Loss ablations
create_config "q200_k4_focal_only" 200 4 "static_item" "large_q200_k4" true 1.0 0.0
create_config "q200_k4_wol_only" 200 4 "static_item" "large_q200_k4" true 0.0 1.0

# Monotonic constraint ablation
create_config "q200_k4_unconstrained" 200 4 "static_item" "large_q200_k4" false 0.5 0.5

echo ""
echo "=== Config generation complete ==="
echo ""

# ============================================================================
# STEP 3: Training
# ============================================================================
echo "=== STEP 3: Training all models ==="
echo ""

TOTAL_EXPERIMENTS=$((${#Q_SIZES[@]} * ${#K_VALUES[@]} * ${#EMBEDDINGS[@]} + 3))
CURRENT=0

for CONFIG in configs/generated/*.yaml; do
    CURRENT=$((CURRENT + 1))
    EXP_NAME=$(basename ${CONFIG} .yaml)

    echo "[$CURRENT/$TOTAL_EXPERIMENTS] Training ${EXP_NAME}..."
    python scripts/train.py --config ${CONFIG}
    echo ""
done

echo "=== Training complete ==="
echo ""

# ============================================================================
# STEP 4: Parameter recovery analysis
# ============================================================================
echo "=== STEP 4: Computing parameter recovery ==="
echo ""

for CONFIG in configs/generated/*.yaml; do
    EXP_NAME=$(basename ${CONFIG} .yaml)
    CHECKPOINT="outputs/${EXP_NAME}/best.pt"

    if [ -f ${CHECKPOINT} ]; then
        echo "Computing recovery for ${EXP_NAME}..."

        # Run eval_metrics
        python scripts/eval_metrics.py \
            --config ${CONFIG} \
            --checkpoint ${CHECKPOINT} || true

        # Generate recovery plots
        python scripts/plot_recovery.py \
            --config ${CONFIG} \
            --checkpoint ${CHECKPOINT} \
            --output outputs/${EXP_NAME} || true
    fi
done

echo ""
echo "=== Parameter recovery complete ==="
echo ""

# ============================================================================
# STEP 5: Learner trajectories (skipped - requires multi-model comparison)
# ============================================================================
echo "=== STEP 5: Learner trajectories (skipped) ==="
echo "Note: Trajectory plotting requires comparing multiple models."
echo "Run plot_learner_trajectories.py manually for specific comparisons."
echo ""

# ============================================================================
# STEP 6: Summary statistics
# ============================================================================
echo "=== STEP 6: Generating summary statistics ==="
echo ""

python << 'PYEOF'
import pandas as pd
import os
from pathlib import Path

results = []

for metrics_file in Path('outputs').rglob('metrics.csv'):
    exp_name = metrics_file.parent.name
    df = pd.read_csv(metrics_file)

    if len(df) > 0:
        best_idx = df['val_qwk'].idxmax()
        best = df.iloc[best_idx]

        results.append({
            'experiment': exp_name,
            'best_epoch': int(best['epoch']),
            'qwk': float(best['val_qwk']),
            'acc': float(best['val_categorical_accuracy']),
            'tau': float(best['val_kendall_tau'])
        })

if results:
    summary_df = pd.DataFrame(results)
    summary_df = summary_df.sort_values('experiment')
    summary_df.to_csv('outputs/summary_all_experiments.csv', index=False)

    print("\n=== Summary Statistics ===")
    print(f"Total experiments: {len(results)}")
    print(f"\nBest QWK: {summary_df['qwk'].max():.3f} ({summary_df.loc[summary_df['qwk'].idxmax(), 'experiment']})")
    print(f"Mean QWK: {summary_df['qwk'].mean():.3f}")
    print(f"\nSummary saved to: outputs/summary_all_experiments.csv")
else:
    print("No results found!")
PYEOF

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE!"
echo "=========================================="
echo ""
echo "Results saved to: outputs/"
echo "Summary: outputs/summary_all_experiments.csv"
echo ""
