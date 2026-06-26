#!/usr/bin/env bash
# run_table9_ablation.sh — Chay Ablation Study (Table 9)
# Ghi chú: Chạy trên GPUs: 3, 4, 5

set -e
cd /workspace/yhnn/VSL_GH

mkdir -p results/run_v2
mkdir -p results/run_no_face
mkdir -p results/run_no_vel

echo "============================================================"
echo "  Table 9: Ablation Study"
echo "============================================================"

# Full model
CUDA_VISIBLE_DEVICES=0 nohup bash -c "source venv/bin/activate && python source/train.py --config source/config/config.yaml" > results/run_v2/train.log 2>&1 &
PID1=$!
echo "Started Full Model on GPU 0 (PID: $PID1)"

# w/o face
CUDA_VISIBLE_DEVICES=1 nohup bash -c "source venv/bin/activate && python source/train.py --config source/config/config_no_face.yaml" > results/run_no_face/train.log 2>&1 &
PID2=$!
echo "Started w/o Face on GPU 1 (PID: $PID2)"

# w/o velocity
CUDA_VISIBLE_DEVICES=2 nohup bash -c "source venv/bin/activate && python source/train.py --config source/config/config_no_vel.yaml" > results/run_no_vel/train.log 2>&1 &
PID3=$!
echo "Started w/o Velocity on GPU 2 (PID: $PID3)"

echo "Waiting for Ablations to finish..."
wait $PID1 $PID2 $PID3

echo "Table 9 Ablations completed!"
