#!/usr/bin/env bash
# run_phase6.sh — Chạy 13 mô hình Ablation (Phase 6)
# Sử dụng GPU: 4 và 5

set -e
cd /workspace/yhnn/VSL_GH

# Định nghĩa các mảng config
TABLE_8_CONFIGS=(
    "tb8_conformer_glossfree"
    "tb8_conformer_ctc"
    "tb8_conformer_supervised"
    "tb8_conformer_ctc_supervised"
)

TABLE_9_CONFIGS=(
    "tb9_conformer_no_vel"
    "tb9_conformer_no_face"
    "tb9_conformer_multiview"
)

TABLE_10_CONFIGS=(
    "tb10_conformer_loso_s01"
    "tb10_conformer_loso_s02"
    "tb10_conformer_loso_s03"
    "tb10_conformer_loso_s04"
    "tb10_conformer_loso_s05"
    "tb10_conformer_loso_s06"
)

# Kết hợp tất cả config lại
ALL_CONFIGS=("${TABLE_8_CONFIGS[@]}" "${TABLE_9_CONFIGS[@]}" "${TABLE_10_CONFIGS[@]}")

echo "============================================================"
echo "  PHASE 6: Chạy ${#ALL_CONFIGS[@]} Thực nghiệm (Bảng 8, 9, 10)"
echo "============================================================"

# Hàm chạy 1 model (nhận GPU, Core, và config_name)
run_model() {
    local GPU=$1
    local CORES=$2
    local CONFIG=$3
    local LOG_DIR="results/run_phase6_${CONFIG}"
    local LOG_FILE="${LOG_DIR}/train.log"

    mkdir -p "$LOG_DIR"
    
    # Kiem tra xem da train xong chua
    if [ -f "$LOG_DIR/checkpoint_best.pt" ] && grep -q "Training done!" "$LOG_FILE" 2>/dev/null; then
        echo "[BỎ QUA] $CONFIG đã train xong trước đó."
        return
    fi
    
    local RESUME_ARGS=""
    if [ -f "$LOG_DIR/checkpoint_best.pt" ]; then
        RESUME_ARGS="--resume $LOG_DIR/checkpoint_best.pt"
        echo "[RESUME] Tìm thấy checkpoint cũ, tiếp tục train $CONFIG..."
    fi
    
    echo "[BẮT ĐẦU] Chạy $CONFIG trên GPU $GPU (Core $CORES)"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 taskset -c $CORES bash -c "source venv/bin/activate && CUDA_VISIBLE_DEVICES=$GPU python source/train.py --config source/config/phase6/config_${CONFIG}.yaml $RESUME_ARGS" >> "$LOG_FILE" 2>&1
    echo "[HOÀN THÀNH] $CONFIG trên GPU $GPU"
}

# Chạy tuần tự tất cả các mô hình trên GPU 4
for CONFIG in "${ALL_CONFIGS[@]}"; do
    run_model 4 "40-45" "$CONFIG"
done

echo "============================================================"
echo "TẤT CẢ THỰC NGHIỆM ĐÃ HOÀN TẤT!"
echo "Chạy lệnh python scripts/extract_phase6_results.py để xem kết quả."
