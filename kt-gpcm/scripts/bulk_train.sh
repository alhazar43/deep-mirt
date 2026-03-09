#!/bin/bash
# Bulk training script for all K=4 experiments
# Run from kt-gpcm/ directory with: bash scripts/bulk_train.sh

set -e
export PYTHONPATH=src

echo "=========================================="
echo "Starting bulk training - all K=4 experiments"
echo "=========================================="

# RQ1: Ordinal Prediction across K∈{2,3,4,5}
echo ""
echo "=== RQ1: Ordinal Prediction (K∈{2,3,4,5}) ==="
for k in 2 3 4 5; do
    echo "Training deepgpcm_k${k}_s42..."
    python scripts/train.py --config configs/deepgpcm_k${k}_s42.yaml

    echo "Training softmax_k${k}_s42..."
    python scripts/train.py --config configs/softmax_k${k}_s42.yaml

    echo "Training ordinal_k${k}_s42..."
    python scripts/train.py --config configs/ordinal_k${k}_s42.yaml

    echo "Training static_gpcm_k${k}_s42..."
    python scripts/train.py --config configs/static_gpcm_k${k}_s42.yaml

    echo "Training dynamic_gpcm_k${k}_s42..."
    python scripts/train.py --config configs/dynamic_gpcm_k${k}_s42.yaml
done

# RQ2: Learner Trajectories (K=4)
echo ""
echo "=== RQ2: Learner Trajectories (K=4) ==="
echo "Already covered by deepgpcm_k4_s42, softmax_k4_s42, dynamic_gpcm_k4_s42"

# RQ3: IRT Parameter Recovery (K=4)
echo ""
echo "=== RQ3: IRT Parameter Recovery (K=4) ==="
echo "Already covered by deepgpcm_k4_s42, static_gpcm_k4_s42, dynamic_gpcm_k4_s42"

# RQ4: Scalable Item Encoding (Q∈{500,1000,5000}, K=4)
echo ""
echo "=== RQ4: Scalable Item Encoding (K=4, varying Q) ==="
echo "Training large_q500_v9a (LinearDecay, Q=500, K=4)..."
python scripts/train.py --config configs/large_q500_v9a.yaml

echo "Training large_q500_v9b (StaticItem, Q=500, K=4)..."
python scripts/train.py --config configs/large_q500_v9b.yaml

echo "Training large_q1000_v9a (LinearDecay, Q=1000, K=4)..."
python scripts/train.py --config configs/large_q1000_v9a.yaml

echo "Training large_q1000_v9b (StaticItem, Q=1000, K=4)..."
python scripts/train.py --config configs/large_q1000_v9b.yaml

echo "Training large_q5000_linear (LinearDecay, Q=5000, K=4)..."
python scripts/train.py --config configs/large_q5000_linear.yaml

echo "Training large_q5000_separable (Separable, Q=5000, K=4)..."
python scripts/train.py --config configs/large_q5000_separable.yaml

echo "Training large_q5000_static (StaticItem, Q=5000, K=4)..."
python scripts/train.py --config configs/large_q5000_static.yaml

# RQ5: Ablation Studies (K=4)
echo ""
echo "=== RQ5: Ablation Studies (K=4) ==="
echo "Training ablation_focal_k4_s42 (Focal loss only)..."
python scripts/train.py --config configs/ablation_focal_k4_s42.yaml

echo "Training ablation_wol_k4_s42 (WOL only)..."
python scripts/train.py --config configs/ablation_wol_k4_s42.yaml

echo "Training ablation_nomonot_k4_s42 (No monotonic constraint)..."
python scripts/train.py --config configs/ablation_nomonot_k4_s42.yaml

echo "Training seqlen_s20_40_k4_s42 (Short sequences)..."
python scripts/train.py --config configs/seqlen_s20_40_k4_s42.yaml

echo "Training seqlen_s20_100_k4_s42 (Long sequences)..."
python scripts/train.py --config configs/seqlen_s20_100_k4_s42.yaml

echo ""
echo "=========================================="
echo "Bulk training completed!"
echo "=========================================="
