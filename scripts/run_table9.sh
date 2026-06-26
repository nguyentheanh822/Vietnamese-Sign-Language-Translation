#!/bin/bash
# Table 9: Feature Ablation (Parallel with safe num_workers=2)

set -e
cd /workspace/yhnn/VSL_GH
source venv/bin/activate

echo "=========================================="
echo "Table 9: Feature Ablation (TCN Supervised)"
echo "Parallel mode: GPU 1 and GPU 3 (num_workers=2)"
echo "=========================================="

mkdir -p results/run_tb9_no_velocity results/run_tb9_no_face results/run_tb9_multi_view

echo ""
echo "[Round 1] Training no_velocity (GPU 1) + no_face (GPU 3)..."
echo "Started: $(date)"

CUDA_VISIBLE_DEVICES=1 taskset -c 0-15 python source/train.py \
    --config source/config/phase7/table9/config_tb9_no_velocity.yaml \
    > results/run_tb9_no_velocity/train.log 2>&1 &
PID1=$!
echo "  no_velocity PID=$PID1 on GPU 1, cores 0-15"

CUDA_VISIBLE_DEVICES=3 taskset -c 16-31 python source/train.py \
    --config source/config/phase7/table9/config_tb9_no_face.yaml \
    > results/run_tb9_no_face/train.log 2>&1 &
PID2=$!
echo "  no_face     PID=$PID2 on GPU 3, cores 16-31"

wait $PID1 && echo "  no_velocity done: $(date)"
wait $PID2 && echo "  no_face     done: $(date)"

echo ""
echo "[Round 2] Training multi_view (GPU 1)..."
echo "Started: $(date)"

CUDA_VISIBLE_DEVICES=1 taskset -c 0-15 python source/train.py \
    --config source/config/phase7/table9/config_tb9_multi_view.yaml \
    > results/run_tb9_multi_view/train.log 2>&1 &
PID3=$!
echo "  multi_view PID=$PID3 on GPU 1, cores 0-15"

wait $PID3 && echo "  multi_view done: $(date)"

echo ""
echo "=========================================="
echo "All Table 9 training COMPLETE!"
echo "Finished: $(date)"
echo "=========================================="
