#!/usr/bin/env bash
# run_table8_gloss_supervision.sh — Chay Gloss Supervision (Table 8)
# Ghi chú: Kịch bản này chuẩn bị cho 4 phương pháp giám sát nhãn.

set -e
cd /workspace/yhnn/VSL_GH

mkdir -p results/run_tb8_gloss_free
mkdir -p results/run_tb8_ctc
mkdir -p results/run_tb8_supervised
mkdir -p results/run_tb8_ctc_supervised

echo "============================================================"
echo "  Table 8: Gloss Supervision"
echo "============================================================"

# Gloss-Free
CUDA_VISIBLE_DEVICES=3 nohup bash -c "source venv/bin/activate && python source/train.py --config source/config/config_tb8_gloss_free.yaml" > results/run_tb8_gloss_free/train.log 2>&1 &
PID1=$!
echo "Started Gloss-Free on GPU 3 (PID: $PID1)"

# CTC
CUDA_VISIBLE_DEVICES=4 nohup bash -c "source venv/bin/activate && python source/train.py --config source/config/config_tb8_ctc.yaml" > results/run_tb8_ctc/train.log 2>&1 &
PID2=$!
echo "Started CTC on GPU 4 (PID: $PID2)"

# Supervised
CUDA_VISIBLE_DEVICES=5 nohup bash -c "source venv/bin/activate && python source/train.py --config source/config/config_tb8_supervised.yaml" > results/run_tb8_supervised/train.log 2>&1 &
PID3=$!
echo "Started Supervised (Frame-level CE) on GPU 5 (PID: $PID3)"

echo "Waiting for Batch 1 to finish..."
wait $PID1 $PID2 $PID3

echo "============================================================"
echo "  Table 8: Batch 2"
echo "============================================================"

# CTC + Supervised
CUDA_VISIBLE_DEVICES=3 nohup bash -c "source venv/bin/activate && python source/train.py --config source/config/config_tb8_ctc_supervised.yaml" > results/run_tb8_ctc_supervised/train.log 2>&1 &
PID4=$!
echo "Started CTC+Supervised on GPU 3 (PID: $PID4)"

echo "Waiting for Batch 2 to finish..."
wait $PID4

echo "Table 8 Gloss Supervision runs completed!"
