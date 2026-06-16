#!/bin/bash
# start_sota_train.sh
echo "Bat dau huan luyen ST-GCN + Transformer tren GPU 0..."
CUDA_VISIBLE_DEVICES=0 python3 source/train.py --config source/config/config_sota.yaml > results/run_SOTA_stgcn/train_output.txt 2>&1 &
PID=$!
echo "Tien trinh train dang chay ngam (PID: $PID)"
