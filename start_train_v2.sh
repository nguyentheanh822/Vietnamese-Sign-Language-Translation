#!/usr/bin/env bash
# start_train_v2.sh — Kill old processes va start train moi voi GPU 0,1

set -e

cd /workspace/yhnn/VSL_GH

# Kill cac python processes cu cua user nay trong VSL_GH (tru script nay)
OLD_PIDS=$(ps aux | grep python3 | grep -v grep | grep -v "start_train" | awk '{print $2}')
if [ -n "$OLD_PIDS" ]; then
    echo "[INFO] Killing old python3 processes: $OLD_PIDS"
    echo "$OLD_PIDS" | xargs kill -9 2>/dev/null || true
    sleep 3
fi

# Kiem tra GPU
echo "[INFO] GPU status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader 2>/dev/null | head -4

# Tao thu muc ket qua
mkdir -p results/run_v2

# Test nhanh dataset (chay o foreground)
echo "[INFO] Testing dataset with new normalization..."
CUDA_VISIBLE_DEVICES=0,1 python3 -c "
import sys; sys.path.insert(0,'source')
from data.dataset import VSLDataset
from utils.vocab import Vocab, build_translation_vocab
gloss_vocab = Vocab.load('data/gloss_vocab.txt')
trans_vocab = build_translation_vocab('data/dataset.json')
ds = VSLDataset('/workspace/yhnn/VSL_GH', 'val', gloss_vocab, trans_vocab, 300, 20, 30, False)
item = ds[0]
print(f'  Keypoint shape: {tuple(item[\"keypoints\"].shape)}  <- should be (T, 822)')
print(f'  Video: {item[\"video_id\"]}')
print('  Dataset OK!')
" 2>&1 | grep -v "^I0" | grep -v "^W0"

# Start training voi nohup (background), GPU 0 va 1
echo "[INFO] Starting training in background (GPU 0,1)..."
CUDA_VISIBLE_DEVICES=0,1 nohup python3 source/train.py \
    --config source/config/config.yaml \
    > results/run_v2/train_output.txt 2>&1 &

TRAIN_PID=$!
echo "[INFO] Training PID: $TRAIN_PID"
echo "$TRAIN_PID" > results/run_v2/train.pid

# Cho 30 giay va in output
sleep 30
echo ""
echo "[INFO] First 50 lines of training output:"
head -50 results/run_v2/train_output.txt 2>/dev/null || echo "(no output yet)"
echo ""
echo "[INFO] Training is running! Monitor with:"
echo "  tail -f /workspace/yhnn/VSL_GH/results/run_v2/train_output.txt"
