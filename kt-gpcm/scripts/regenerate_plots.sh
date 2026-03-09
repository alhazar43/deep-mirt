#!/bin/bash
# Fix and regenerate recovery and trajectory plots
# After baseline training completion

set -e
export PYTHONPATH=src
export KMP_DUPLICATE_LIB_OK=TRUE

echo "=========================================="
echo "REGENERATING RECOVERY & TRAJECTORY PLOTS"
echo "=========================================="
echo ""

# Get all datasets
DATASETS=($(ls data/ | grep "large_q"))
echo "Found ${#DATASETS[@]} datasets"
echo ""

# ============================================================================
# STEP 1: Generate split recovery plots
# ============================================================================
echo "=== STEP 1: Generating split recovery plots ==="
echo ""

mkdir -p outputs/recovery_plots

SUCCESS_COUNT=0
FAIL_COUNT=0

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
        echo "Generating split recovery for ${DATASET}..."

        OUTPUT_FILE="outputs/recovery_plots/${DATASET}_recovery"

        python scripts/plot_recovery_split.py \
            --deepgpcm-config configs/generated/${DEEPGPCM_EXP}.yaml \
            --deepgpcm-checkpoint ${DEEPGPCM_CKPT} \
            --static-config configs/baselines/${STATIC_EXP}.yaml \
            --static-checkpoint ${STATIC_CKPT} \
            --dynamic-config configs/baselines/${DYNAMIC_EXP}.yaml \
            --dynamic-checkpoint ${DYNAMIC_CKPT} \
            --output ${OUTPUT_FILE} 2>&1 | grep -E "(Saved|Error)" || true

        if [ -f "${OUTPUT_FILE}_student.pgf" ]; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            echo "  ✓ Success"
        else
            FAIL_COUNT=$((FAIL_COUNT + 1))
            echo "  ✗ Failed"
        fi
    else
        echo "Skipping ${DATASET} - missing checkpoints"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

echo ""
echo "Recovery plots: ${SUCCESS_COUNT} success, ${FAIL_COUNT} failed"
echo ""

# ============================================================================
# STEP 2: Generate trajectory plots
# ============================================================================
echo "=== STEP 2: Generating trajectory plots ==="
echo ""

mkdir -p outputs/trajectory_plots

SUCCESS_COUNT=0
FAIL_COUNT=0

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
        echo "Generating trajectories for ${DATASET}..."

        mkdir -p outputs/trajectory_plots/${DATASET}

        python scripts/plot_learner_trajectories.py \
            --deepgpcm-config configs/generated/${DEEPGPCM_EXP}.yaml \
            --deepgpcm-checkpoint ${DEEPGPCM_CKPT} \
            --softmax-config configs/baselines/${SOFTMAX_EXP}.yaml \
            --softmax-checkpoint ${SOFTMAX_CKPT} \
            --dynamic-config configs/baselines/${DYNAMIC_EXP}.yaml \
            --dynamic-checkpoint ${DYNAMIC_CKPT} \
            --output-dir outputs/trajectory_plots/${DATASET} 2>&1 | grep -E "(Saved|Error)" || true

        if [ -f "outputs/trajectory_plots/${DATASET}/learner_trajectories.pgf" ]; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            echo "  ✓ Success"
        else
            FAIL_COUNT=$((FAIL_COUNT + 1))
            echo "  ✗ Failed"
        fi
    else
        echo "Skipping ${DATASET} - missing checkpoints"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

echo ""
echo "Trajectory plots: ${SUCCESS_COUNT} success, ${FAIL_COUNT} failed"
echo ""

# ============================================================================
# STEP 3: Summary
# ============================================================================
echo "=========================================="
echo "POST-PROCESSING COMPLETE!"
echo "=========================================="
echo ""

# Count generated files
RECOVERY_FILES=$(find outputs/recovery_plots -name "*.pgf" 2>/dev/null | wc -l)
TRAJECTORY_FILES=$(find outputs/trajectory_plots -name "*.pgf" 2>/dev/null | wc -l)

echo "Recovery plots: ${RECOVERY_FILES} files in outputs/recovery_plots/"
echo "Trajectory plots: ${TRAJECTORY_FILES} files in outputs/trajectory_plots/"
echo ""
