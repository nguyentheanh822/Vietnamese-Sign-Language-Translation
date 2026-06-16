#!/bin/bash
# ============================================================
# VSL-GH Smart Backup Script
# Chỉ backup những gì cần thiết (~20GB thay vì 96GB)
# ============================================================

DEST="$1"  # Truyền đường dẫn đích vào tham số

if [ -z "$DEST" ]; then
    echo "❌ Thiếu đường dẫn đích!"
    echo "   Cách dùng: bash backup_smart.sh /path/to/destination"
    echo "   Ví dụ:     bash backup_smart.sh /media/usb/VSL_GH_backup"
    exit 1
fi

SRC="/workspace/yhnn/VSL_GH"
echo "============================================"
echo " VSL-GH Smart Backup"
echo " Nguồn : $SRC"
echo " Đích   : $DEST"
echo "============================================"

# Tạo thư mục đích
mkdir -p "$DEST"

# ------------------------------------------------------------
# 1. SOURCE CODE (268 KB) - Toàn bộ
# ------------------------------------------------------------
echo ""
echo "[1/5] 📂 Backup source code..."
rsync -av --progress "$SRC/source/" "$DEST/source/"

# ------------------------------------------------------------
# 2. ANNOTATIONS (8.4 MB) - Toàn bộ
# ------------------------------------------------------------
echo ""
echo "[2/5] 📝 Backup annotations..."
rsync -av --progress "$SRC/data/annotations/" "$DEST/data/annotations/"

# ------------------------------------------------------------
# 3. SPLITS (44 KB) - Toàn bộ
# ------------------------------------------------------------
echo ""
echo "[3/5] 📋 Backup data splits..."
rsync -av --progress "$SRC/data/splits/" "$DEST/data/splits/"

# ------------------------------------------------------------
# 4. KEYPOINTS (584 MB) - Toàn bộ
# ------------------------------------------------------------
echo ""
echo "[4/5] 🦴 Backup keypoints..."
rsync -av --progress "$SRC/data/keypoints/" "$DEST/data/keypoints/"

# ------------------------------------------------------------
# 5. VIDEOS (18 GB) - Toàn bộ
# ------------------------------------------------------------
echo ""
echo "[5/5] 🎥 Backup videos (18GB - có thể mất nhiều thời gian)..."
rsync -av --progress "$SRC/data/videos/" "$DEST/data/videos/"

# ------------------------------------------------------------
# BONUS: Chỉ lấy checkpoint_best.pt từ mỗi run (bỏ qua ep*.pt)
# ------------------------------------------------------------
echo ""
echo "[BONUS] 🏆 Backup best checkpoints từ mỗi run..."
for run_dir in "$SRC/results"/*/; do
    run_name=$(basename "$run_dir")
    best_ckpt="$run_dir/checkpoint_best.pt"
    if [ -f "$best_ckpt" ]; then
        mkdir -p "$DEST/results/$run_name"
        rsync -av --progress "$best_ckpt" "$DEST/results/$run_name/"
        echo "   ✅ $run_name/checkpoint_best.pt"
    else
        echo "   ⚠️  $run_name: không có checkpoint_best.pt"
    fi
done

# Copy thêm log/metrics nếu có (nhẹ)
for run_dir in "$SRC/results"/*/; do
    run_name=$(basename "$run_dir")
    for ext in log txt json csv yaml; do
        find "$run_dir" -maxdepth 2 -name "*.$ext" | while read f; do
            rel=$(realpath --relative-to="$SRC/results/$run_name" "$f")
            mkdir -p "$DEST/results/$run_name/$(dirname $rel)"
            cp "$f" "$DEST/results/$run_name/$rel" 2>/dev/null
        done
    done
done

# ------------------------------------------------------------
# Tổng kết
# ------------------------------------------------------------
echo ""
echo "============================================"
echo " ✅ BACKUP HOÀN TẤT!"
echo " 📁 Thư mục đích: $DEST"
echo " 📊 Dung lượng đã backup:"
du -sh "$DEST"
echo "============================================"
