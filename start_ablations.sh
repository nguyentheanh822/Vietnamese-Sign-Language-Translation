#!/bin/bash
echo "Starting No-Gloss Ablation on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python3 source/train.py --config source/config/config_no_gloss.yaml > results/run_no_gloss/train_output.txt 2>&1 &
PID1=$!

echo "Starting No-Face Ablation on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python3 source/train.py --config source/config/config_no_face.yaml > results/run_no_face/train_output.txt 2>&1 &
PID2=$!

echo "Both processes started! PID1=$PID1, PID2=$PID2"
