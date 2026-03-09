#!/bin/bash
# Regenerate recovery split plots for all K and Q combinations

set -e

export PYTHONPATH=src

# Define all Q and K combinations from current experiments
Q_VALUES=(200 500 1000 2000)
K_VALUES=(2 3 4 5 6)

echo "Regenerating recovery plots for all Q and K combinations..."

for Q in "${Q_VALUES[@]}"; do
    for K in "${K_VALUES[@]}"; do
        DATASET="large_q${Q}_k${K}"

        # Check if all three baseline checkpoints exist
        # Use dkvmn_ordinal (which is actually DeepGPCM) as the main model
        DEEP_CONFIG="configs/baselines/${DATASET}_dkvmn_ordinal.yaml"
        DEEP_CKPT="outputs/${DATASET}_dkvmn_ordinal/best.pt"

        STATIC_CONFIG="configs/baselines/${DATASET}_static_gpcm.yaml"
        STATIC_CKPT="outputs/${DATASET}_static_gpcm/best.pt"

        DYNAMIC_CONFIG="configs/baselines/${DATASET}_dynamic_gpcm.yaml"
        DYNAMIC_CKPT="outputs/${DATASET}_dynamic_gpcm/best.pt"

        if [[ -f "$DEEP_CKPT" && -f "$STATIC_CKPT" && -f "$DYNAMIC_CKPT" ]]; then
            echo "Processing Q=${Q}, K=${K}..."
            OUTPUT_DIR="outputs/${DATASET}_dkvmn_ordinal"

            python scripts/plot_recovery_split.py \
                --deepgpcm-config "$DEEP_CONFIG" \
                --deepgpcm-checkpoint "$DEEP_CKPT" \
                --static-config "$STATIC_CONFIG" \
                --static-checkpoint "$STATIC_CKPT" \
                --dynamic-config "$DYNAMIC_CONFIG" \
                --dynamic-checkpoint "$DYNAMIC_CKPT" \
                --output "${OUTPUT_DIR}/recovery_split"

            echo "  ✓ Generated ${OUTPUT_DIR}/recovery_split_student.pgf"
            echo "  ✓ Generated ${OUTPUT_DIR}/recovery_split_item.pgf"
        else
            echo "Skipping Q=${Q}, K=${K} (missing checkpoints)"
        fi
    done
done

echo ""
echo "Done! Recovery plots regenerated."
