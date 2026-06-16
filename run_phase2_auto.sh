#!/bin/bash
echo "Waiting for extract_keypoints.py to finish..."
while pgrep -f "extract_keypoints.py" > /dev/null; do
    sleep 30
done
echo "Extraction finished!"

echo "Starting Training Phase 2 (300 sentences, No Gloss)..."
mkdir -p results/run_phase2
CUDA_VISIBLE_DEVICES=0 python3 source/train.py --config source/config/config_phase2.yaml > results/run_phase2/train_output.txt 2>&1

echo "Training finished. Starting Evaluation..."
CUDA_VISIBLE_DEVICES=0 python3 source/evaluate.py --config source/config/config_phase2.yaml --checkpoint results/run_phase2/checkpoint_best.pt --split test > results/run_phase2/eval_test.txt 2>&1

echo "Pipeline completed successfully!"
