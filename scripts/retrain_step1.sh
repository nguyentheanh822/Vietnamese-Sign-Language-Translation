#!/bin/bash
source venv/bin/activate

echo "Starting GRU Training on GPU 1..."
mkdir -p results/run_tb7_gru
CUDA_VISIBLE_DEVICES=1 nohup python source/train.py --config source/config/config_tb7_gru.yaml > results/run_tb7_gru/train.log 2>&1 &

echo "Starting BiLSTM Training on GPU 2..."
mkdir -p results/run_tb7_bilstm
CUDA_VISIBLE_DEVICES=2 nohup python source/train.py --config source/config/config_tb7_bilstm.yaml > results/run_tb7_bilstm/train.log 2>&1 &

echo "Both trainings launched in background!"
