#!/bin/bash
# Table 8: Supervision Ablation (Parallel with safe num_workers=2)

set -e
cd /workspace/yhnn/VSL_GH
source venv/bin/activate

echo "=========================================="
echo "Table 8: Supervision Ablation (TCN)"
echo "Parallel mode: GPU 1 and GPU 3 (num_workers=2 -> SAFE)"
echo "=========================================="

mkdir -p results/run_tb8_gloss_free results/run_tb8_ctc_only \
          results/run_tb8_supervised results/run_tb8_ctc_supervised

# 'supervised' = TCN w03 from Table 7 (BLEU=82.64) - reuse, no retraining
echo "[INFO] 'supervised' reuses run_tb7_tcn (Test BLEU=82.64). Skipping."
cp results/run_tb7_tcn/checkpoint_best.pt results/run_tb8_supervised/checkpoint_best.pt 2>/dev/null || true

echo ""
echo "[Round 1] Training gloss_free (GPU 1) + ctc_only (GPU 3)..."
echo "Started: $(date)"

CUDA_VISIBLE_DEVICES=1 taskset -c 0-15 python source/train.py \
    --config source/config/phase7/table8/config_tb8_gloss_free.yaml \
    > results/run_tb8_gloss_free/train.log 2>&1 &
PID1=$!
echo "  gloss_free  PID=$PID1 on GPU 1, cores 0-15"

CUDA_VISIBLE_DEVICES=3 taskset -c 16-31 python source/train.py \
    --config source/config/phase7/table8/config_tb8_ctc_only.yaml \
    > results/run_tb8_ctc_only/train.log 2>&1 &
PID2=$!
echo "  ctc_only    PID=$PID2 on GPU 3, cores 16-31"

wait $PID1 && echo "  gloss_free done: $(date)"
wait $PID2 && echo "  ctc_only   done: $(date)"

echo ""
echo "[Round 2] Training ctc_supervised (GPU 1)..."
echo "Started: $(date)"

CUDA_VISIBLE_DEVICES=1 taskset -c 0-15 python source/train.py \
    --config source/config/phase7/table8/config_tb8_ctc_supervised.yaml \
    > results/run_tb8_ctc_supervised/train.log 2>&1 &
PID3=$!
echo "  ctc_supervised PID=$PID3 on GPU 1, cores 0-15"

wait $PID3 && echo "  ctc_supervised done: $(date)"

echo ""
echo "=========================================="
echo "All Table 8 training COMPLETE!"
echo "Finished: $(date)"
echo "=========================================="
