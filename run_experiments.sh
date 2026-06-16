#!/usr/bin/env bash
# run_experiments.sh — Chay 4 cau hinh Skeleton Transformer song song
#
# Timeline:
#   t=0     : Run A (beam eval trên checkpoint_best run_v2, GPU 0, ~5 phut)
#   t=0     : Run C (GPU 1, train ~2-3 gio)
#   t=5min  : Run B (GPU 0, train ~2-3 gio, sau khi A xong)
#   t~B_done: Run D (GPU 0, train ~2-3 gio, sau khi B xong)
#
# Ghi log rieng cho moi run:
#   results/run_A_beam/eval_output.txt
#   results/run_B_small/train_output.txt
#   results/run_C_dropout/train_output.txt
#   results/run_D_ctc05/train_output.txt

set -e
cd /workspace/yhnn/VSL_GH

echo "============================================================"
echo "  VSL-GH Experiments — $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ── GPU status ──────────────────────────────────────────────────
echo ""
echo "[INFO] GPU status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader 2>/dev/null

# ── Tao thu muc ket qua ─────────────────────────────────────────
mkdir -p results/run_A_beam
mkdir -p results/run_B_small
mkdir -p results/run_C_dropout
mkdir -p results/run_D_ctc05

# ════════════════════════════════════════════════════════════════
# RUN A — Beam Search Eval tren checkpoint_best cua run_v2
# (khong train, chi eval, ~5 phut)
# ════════════════════════════════════════════════════════════════
echo ""
echo "[RUN A] Beam search eval on run_v2/checkpoint_best.pt ..."

CUDA_VISIBLE_DEVICES=0 python3 source/evaluate.py \
    --config source/config/config.yaml \
    --checkpoint results/run_v2/checkpoint_best.pt \
    --split val \
    --beam_size 4 \
    --output results/run_A_beam/val_beam4.json \
    2>&1 | tee results/run_A_beam/eval_val_output.txt

CUDA_VISIBLE_DEVICES=0 python3 source/evaluate.py \
    --config source/config/config.yaml \
    --checkpoint results/run_v2/checkpoint_best.pt \
    --split test \
    --beam_size 4 \
    --output results/run_A_beam/test_beam4.json \
    2>&1 | tee results/run_A_beam/eval_test_output.txt

echo "[RUN A] DONE! Ket qua luu tai results/run_A_beam/"

# ════════════════════════════════════════════════════════════════
# RUN C — High Dropout (GPU 1) — Bat dau ngay, chay background
# ════════════════════════════════════════════════════════════════
echo ""
echo "[RUN C] Starting training on GPU 1 (high dropout/regularization)..."

CUDA_VISIBLE_DEVICES=1 nohup python3 source/train.py \
    --config source/config/config_C_dropout.yaml \
    > results/run_C_dropout/train_output.txt 2>&1 &

PID_C=$!
echo "$PID_C" > results/run_C_dropout/train.pid
echo "[RUN C] PID=$PID_C (GPU 1)"

# ════════════════════════════════════════════════════════════════
# RUN B — Small Model (GPU 0) — Bat dau sau khi A xong
# ════════════════════════════════════════════════════════════════
echo ""
echo "[RUN B] Starting training on GPU 0 (small model)..."

CUDA_VISIBLE_DEVICES=0 nohup python3 source/train.py \
    --config source/config/config_B_small.yaml \
    > results/run_B_small/train_output.txt 2>&1 &

PID_B=$!
echo "$PID_B" > results/run_B_small/train.pid
echo "[RUN B] PID=$PID_B (GPU 0)"

# ════════════════════════════════════════════════════════════════
# Cho B xong, sau do chay D tren GPU 0
# ════════════════════════════════════════════════════════════════
echo ""
echo "[INFO] B va C dang chay song song. Doi B hoan tat thi chay D..."
echo "[INFO] Monitor:"
echo "  tail -f results/run_B_small/train_output.txt"
echo "  tail -f results/run_C_dropout/train_output.txt"

wait $PID_B
echo ""
echo "[RUN B] Training DONE! (PID $PID_B)"

# ════════════════════════════════════════════════════════════════
# RUN D — Balanced CTC (GPU 0) — Sau khi B xong
# ════════════════════════════════════════════════════════════════
echo ""
echo "[RUN D] Starting training on GPU 0 (balanced CTC 0.5/0.5)..."

CUDA_VISIBLE_DEVICES=0 nohup python3 source/train.py \
    --config source/config/config_D_ctc05.yaml \
    > results/run_D_ctc05/train_output.txt 2>&1 &

PID_D=$!
echo "$PID_D" > results/run_D_ctc05/train.pid
echo "[RUN D] PID=$PID_D (GPU 0)"

# ════════════════════════════════════════════════════════════════
# Tong ket
# ════════════════════════════════════════════════════════════════
echo ""
echo "============================================================"
echo "  Tat ca experiments da duoc khoi dong!"
echo "  Run A  : DONE (results/run_A_beam/)"
echo "  Run B  : DONE (results/run_B_small/)"
echo "  Run C  : Dang chay - PID $PID_C (GPU 1)"
echo "  Run D  : Dang chay - PID $PID_D (GPU 0)"
echo ""
echo "  Monitor C: tail -f results/run_C_dropout/train_output.txt"
echo "  Monitor D: tail -f results/run_D_ctc05/train_output.txt"
echo ""
echo "  Sau khi C va D xong, chay eval:"
echo "  bash eval_all_runs.sh"
echo "============================================================"
