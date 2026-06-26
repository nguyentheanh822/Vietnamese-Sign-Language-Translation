#!/bin/bash
# Table 10: LOSO (Leave-One-Subject-Out) Signer Generalization
# Parallel execution on GPU 1 and GPU 3 safely (num_workers=2)

set -e
cd /workspace/yhnn/VSL_GH
source venv/bin/activate

mkdir -p results/run_tb10_loso_s01
mkdir -p results/run_tb10_loso_s02
mkdir -p results/run_tb10_loso_s03
mkdir -p results/run_tb10_loso_s04
mkdir -p results/run_tb10_loso_s05
mkdir -p results/run_tb10_loso_s06

echo "============================================================"
echo "  Table 10: LOSO Signer Generalization (Batch 1: s01, s02)"
echo "============================================================"

CUDA_VISIBLE_DEVICES=1 taskset -c 0-15 python source/train.py --config source/config/phase7/table10/config_tb10_loso_s01.yaml > results/run_tb10_loso_s01/train.log 2>&1 &
PID1=$!
echo "Started S01 on GPU 1 (PID: $PID1)"

CUDA_VISIBLE_DEVICES=3 taskset -c 16-31 python source/train.py --config source/config/phase7/table10/config_tb10_loso_s02.yaml > results/run_tb10_loso_s02/train.log 2>&1 &
PID2=$!
echo "Started S02 on GPU 3 (PID: $PID2)"

wait $PID1 $PID2
echo "Batch 1 finished!"

echo "============================================================"
echo "  Table 10: LOSO Signer Generalization (Batch 2: s03, s04)"
echo "============================================================"

CUDA_VISIBLE_DEVICES=1 taskset -c 0-15 python source/train.py --config source/config/phase7/table10/config_tb10_loso_s03.yaml > results/run_tb10_loso_s03/train.log 2>&1 &
PID3=$!
echo "Started S03 on GPU 1 (PID: $PID3)"

CUDA_VISIBLE_DEVICES=3 taskset -c 16-31 python source/train.py --config source/config/phase7/table10/config_tb10_loso_s04.yaml > results/run_tb10_loso_s04/train.log 2>&1 &
PID4=$!
echo "Started S04 on GPU 3 (PID: $PID4)"

wait $PID3 $PID4
echo "Batch 2 finished!"

echo "============================================================"
echo "  Table 10: LOSO Signer Generalization (Batch 3: s05, s06)"
echo "============================================================"

CUDA_VISIBLE_DEVICES=1 taskset -c 0-15 python source/train.py --config source/config/phase7/table10/config_tb10_loso_s05.yaml > results/run_tb10_loso_s05/train.log 2>&1 &
PID5=$!
echo "Started S05 on GPU 1 (PID: $PID5)"

CUDA_VISIBLE_DEVICES=3 taskset -c 16-31 python source/train.py --config source/config/phase7/table10/config_tb10_loso_s06.yaml > results/run_tb10_loso_s06/train.log 2>&1 &
PID6=$!
echo "Started S06 on GPU 3 (PID: $PID6)"

wait $PID5 $PID6
echo "Batch 3 finished!"

echo "============================================================"
echo "All 6 LOSO folds completed!"
echo "============================================================"
