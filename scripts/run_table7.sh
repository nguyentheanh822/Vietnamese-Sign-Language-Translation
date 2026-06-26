#!/bin/bash
# Phase 7 - Table 7: Fair comparison of 5 models
# Uses GPUs 1, 4, 5 only (0,2,3 occupied by other users/services)
# Uses taskset to limit CPU core usage for server safety

set -e
cd /workspace/yhnn/VSL_GH
source venv/bin/activate

echo "=========================================="
echo "Phase 7 - Table 7: Fair Model Comparison"
echo "=========================================="
echo "All models: input_dim=822, frame_ce_weight=0.3, ctc_weight=0.0"
echo ""

mkdir -p results/run_tb7_stgcn results/run_tb7_transformer results/run_tb7_tcn results/run_tb7_mamba results/run_tb7_conformer

# Conformer already trained (run_p7_grid_w03), skip it
echo "[INFO] Conformer already trained as run_p7_grid_w03 (BLEU=82.27). Copying checkpoint..."
cp results/run_p7_grid_w03/checkpoint_best.pt results/run_tb7_conformer/checkpoint_best.pt 2>/dev/null || true
cp results/run_p7_grid_w03/train_log.txt results/run_tb7_conformer/train_log.txt 2>/dev/null || true

# Round 1: ST-GCN (GPU 1), Transformer (GPU 4), TCN (GPU 5)
echo ""
echo "[Round 1] Training ST-GCN (GPU 1), Transformer (GPU 4), TCN (GPU 5)..."
echo "Started at: $(date)"

CUDA_VISIBLE_DEVICES=1 taskset -c 0-15 python source/train.py --config source/config/phase7/config_tb7_stgcn.yaml > results/run_tb7_stgcn/train.log 2>&1 &
PID1=$!
echo "  ST-GCN PID=$PID1 on GPU 1, CPU cores 0-15"

CUDA_VISIBLE_DEVICES=4 taskset -c 16-31 python source/train.py --config source/config/phase7/config_tb7_transformer.yaml > results/run_tb7_transformer/train.log 2>&1 &
PID2=$!
echo "  Transformer PID=$PID2 on GPU 4, CPU cores 16-31"

CUDA_VISIBLE_DEVICES=5 taskset -c 32-47 python source/train.py --config source/config/phase7/config_tb7_tcn.yaml > results/run_tb7_tcn/train.log 2>&1 &
PID3=$!
echo "  TCN PID=$PID3 on GPU 5, CPU cores 32-47"

echo ""
echo "Waiting for Round 1 to finish..."
wait $PID1
echo "  ST-GCN done at $(date)"
wait $PID2
echo "  Transformer done at $(date)"
wait $PID3
echo "  TCN done at $(date)"

# Round 2: Mamba (GPU 1)
echo ""
echo "[Round 2] Training Mamba (GPU 1)..."
echo "Started at: $(date)"

CUDA_VISIBLE_DEVICES=1 taskset -c 0-15 python source/train.py --config source/config/phase7/config_tb7_mamba.yaml > results/run_tb7_mamba/train.log 2>&1 &
PID4=$!
echo "  Mamba PID=$PID4 on GPU 1, CPU cores 0-15"

wait $PID4
echo "  Mamba done at $(date)"

echo ""
echo "=========================================="
echo "All Table 7 training COMPLETE!"
echo "=========================================="
echo "Finished at: $(date)"
