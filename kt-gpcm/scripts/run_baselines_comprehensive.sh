#!/bin/bash
# Run 4 baselines for all existing datasets
# Baselines: Static GPCM, Dynamic GPCM, DKVMN+Softmax, DKVMN+Ordinal
# Then generate recovery plots and trajectory plots

set -e
export PYTHONPATH=src
export KMP_DUPLICATE_LIB_OK=TRUE

echo "=========================================="
echo "BASELINE MODELS - COMPREHENSIVE RUN"
echo "=========================================="
echo ""

# Get all datasets
DATASETS=($(ls data/ | grep "large_q"))
echo "Found ${#DATASETS[@]} datasets"
echo ""

# ============================================================================
# STEP 1: Generate baseline configs
# ============================================================================
echo "=== STEP 1: Generating baseline configs ==="
echo ""

mkdir -p configs/baselines

# Function to create baseline config
create_baseline_config() {
    local EXP_NAME=$1
    local Q=$2
    local K=$3
    local DATASET=$4
    local MODEL_TYPE=$5

    cat > configs/baselines/${EXP_NAME}.yaml <<EOF
base:
  experiment_name: "${EXP_NAME}"
  device: "cuda"
  seed: 42

model:
  model_type: "${MODEL_TYPE}"
  n_questions: ${Q}
  n_categories: ${K}
  n_traits: 1
  memory_size: 50
  key_dim: 64
  value_dim: 64
  summary_dim: 50
  embedding_type: "static_item"
  ability_scale: 1.0
  dropout_rate: 0.0
  memory_add_activation: "tanh"
  init_value_memory: false
  monotonic_betas: true

training:
  epochs: 15
  batch_size: 64
  lr: 0.001
  grad_clip: 1.0
  focal_weight: 0.5
  weighted_ordinal_weight: 0.5
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

# Generate configs for all datasets and baselines
BASELINES=("static_gpcm" "dynamic_gpcm" "dkvmn_softmax" "dkvmn_ordinal")

for DATASET in "${DATASETS[@]}"; do
    # Extract Q and K from dataset name
    Q=$(echo $DATASET | sed 's/large_q\([0-9]*\)_k.*/\1/')
    K=$(echo $DATASET | sed 's/large_q[0-9]*_k\([0-9]*\)/\1/')

    for BASELINE in "${BASELINES[@]}"; do
        EXP_NAME="${DATASET}_${BASELINE}"
        echo "Creating config: ${EXP_NAME}"
        create_baseline_config ${EXP_NAME} ${Q} ${K} ${DATASET} ${BASELINE}
    done
done

echo ""
echo "=== Config generation complete ==="
echo ""

# ============================================================================
# STEP 2: Training all baselines
# ============================================================================
echo "=== STEP 2: Training all baseline models ==="
echo ""

TOTAL_EXPERIMENTS=$((${#DATASETS[@]} * ${#BASELINES[@]}))
CURRENT=0

for CONFIG in configs/baselines/*.yaml; do
    CURRENT=$((CURRENT + 1))
    EXP_NAME=$(basename ${CONFIG} .yaml)

    echo "[$CURRENT/$TOTAL_EXPERIMENTS] Training ${EXP_NAME}..."
    python scripts/train.py --config ${CONFIG}
    echo ""
done

echo "=== Training complete ==="
echo ""

# ============================================================================
# STEP 3: Parameter recovery for Static and Dynamic GPCM
# ============================================================================
echo "=== STEP 3: Computing parameter recovery for GPCM models ==="
echo ""

for DATASET in "${DATASETS[@]}"; do
    Q=$(echo $DATASET | sed 's/large_q\([0-9]*\)_k.*/\1/')
    K=$(echo $DATASET | sed 's/large_q[0-9]*_k\([0-9]*\)/\1/')

    STATIC_EXP="${DATASET}_static_gpcm"
    DYNAMIC_EXP="${DATASET}_dynamic_gpcm"
    DEEPGPCM_EXP="q${Q}_k${K}_static_item"

    STATIC_CKPT="outputs/${STATIC_EXP}/best.pt"
    DYNAMIC_CKPT="outputs/${DYNAMIC_EXP}/best.pt"
    DEEPGPCM_CKPT="outputs/${DEEPGPCM_EXP}/best.pt"

    if [ -f ${STATIC_CKPT} ] && [ -f ${DYNAMIC_CKPT} ] && [ -f ${DEEPGPCM_CKPT} ]; then
        echo "Generating split recovery plots for ${DATASET}..."

        # Generate split recovery plots
        python scripts/plot_recovery_split.py \
            --deepgpcm-config configs/generated/${DEEPGPCM_EXP}.yaml \
            --deepgpcm-checkpoint ${DEEPGPCM_CKPT} \
            --static-config configs/baselines/${STATIC_EXP}.yaml \
            --static-checkpoint ${STATIC_CKPT} \
            --dynamic-config configs/baselines/${DYNAMIC_EXP}.yaml \
            --dynamic-checkpoint ${DYNAMIC_CKPT} \
            --output-dir outputs/recovery_plots \
            --output-prefix ${DATASET} || true
    fi
done

echo ""
echo "=== Parameter recovery complete ==="
echo ""

# ============================================================================
# STEP 4: Trajectory plots
# ============================================================================
echo "=== STEP 4: Generating trajectory plots ==="
echo ""

for DATASET in "${DATASETS[@]}"; do
    Q=$(echo $DATASET | sed 's/large_q\([0-9]*\)_k.*/\1/')
    K=$(echo $DATASET | sed 's/large_q[0-9]*_k\([0-9]*\)/\1/')

    DEEPGPCM_EXP="q${Q}_k${K}_static_item"
    SOFTMAX_EXP="${DATASET}_dkvmn_softmax"
    DYNAMIC_EXP="${DATASET}_dynamic_gpcm"

    DEEPGPCM_CKPT="outputs/${DEEPGPCM_EXP}/best.pt"
    SOFTMAX_CKPT="outputs/${SOFTMAX_EXP}/best.pt"
    DYNAMIC_CKPT="outputs/${DYNAMIC_EXP}/best.pt"

    if [ -f ${DEEPGPCM_CKPT} ] && [ -f ${SOFTMAX_CKPT} ] && [ -f ${DYNAMIC_CKPT} ]; then
        echo "Generating trajectory plots for ${DATASET}..."

        python scripts/plot_learner_trajectories.py \
            --deepgpcm-config configs/generated/${DEEPGPCM_EXP}.yaml \
            --deepgpcm-checkpoint ${DEEPGPCM_CKPT} \
            --softmax-config configs/baselines/${SOFTMAX_EXP}.yaml \
            --softmax-checkpoint ${SOFTMAX_CKPT} \
            --dynamic-config configs/baselines/${DYNAMIC_EXP}.yaml \
            --dynamic-checkpoint ${DYNAMIC_CKPT} \
            --output-dir outputs/trajectory_plots/${DATASET} || true
    fi
done

echo ""
echo "=== Trajectory plots complete ==="
echo ""

# ============================================================================
# STEP 5: Summary statistics
# ============================================================================
echo "=== STEP 5: Generating summary statistics ==="
echo ""

python << 'PYEOF'
import pandas as pd
from pathlib import Path

results = []

for metrics_file in Path('outputs').rglob('metrics.csv'):
    exp_name = metrics_file.parent.name

    # Skip if not a baseline experiment
    if not any(x in exp_name for x in ['static_gpcm', 'dynamic_gpcm', 'dkvmn_softmax', 'dkvmn_ordinal']):
        continue

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
    summary_df.to_csv('outputs/summary_baselines.csv', index=False)

    print("\n=== Baseline Summary Statistics ===")
    print(f"Total baseline experiments: {len(results)}")
    print(f"\nBest QWK: {summary_df['qwk'].max():.3f} ({summary_df.loc[summary_df['qwk'].idxmax(), 'experiment']})")
    print(f"Mean QWK: {summary_df['qwk'].mean():.3f}")
    print(f"\nSummary saved to: outputs/summary_baselines.csv")

    # Compare by model type
    print("\n=== Average QWK by Model Type ===")
    for model_type in ['static_gpcm', 'dynamic_gpcm', 'dkvmn_softmax', 'dkvmn_ordinal']:
        subset = summary_df[summary_df['experiment'].str.contains(model_type)]
        if len(subset) > 0:
            print(f"{model_type:20s}: {subset['qwk'].mean():.3f} (n={len(subset)})")
else:
    print("No baseline results found!")
PYEOF

echo ""
echo "=========================================="
echo "ALL BASELINE EXPERIMENTS COMPLETE!"
echo "=========================================="
echo ""
echo "Results saved to: outputs/"
echo "Baseline summary: outputs/summary_baselines.csv"
echo "Recovery plots: outputs/recovery_plots/"
echo "Trajectory plots: outputs/trajectory_plots/"
echo ""
