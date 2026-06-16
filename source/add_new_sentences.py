import os
import json
import pandas as pd
from pathlib import Path

def main():
    base_dir = Path("/workspace/yhnn/VSL_GH/data")
    excel_path = base_dir / "new_sentences.xlsx"
    dataset_json_path = base_dir / "dataset.json"
    videos_dir = base_dir / "videos"

    if not excel_path.exists():
        print(f"Lỗi: Không tìm thấy file Excel tại {excel_path}")
        return

    # 1. Đọc file Excel
    # Dựa vào ảnh cung cấp, file có các cột có thể là "STT" và "Câu Tiếng Việt"
    try:
        df = pd.read_excel(excel_path, engine='openpyxl')
    except Exception as e:
        print(f"Lỗi khi đọc file Excel: {e}")
        return

    # Tim ten cot chinh xac
    stt_col = None
    text_col = None
    for col in df.columns:
        col_lower = str(col).lower()
        if "stt" in col_lower:
            stt_col = col
        elif "câu" in col_lower or "việt" in col_lower or "translation" in col_lower:
            text_col = col

    if stt_col is None or text_col is None:
        print(f"Không thể tự động nhận dạng cột STT và Câu Tiếng Việt. Các cột hiện có: {df.columns.tolist()}")
        print("Vui lòng đổi tên cột về chuẩn 'STT' và 'Câu Tiếng Việt'.")
        return

    # Tao dictionary map tu STT sang Text
    # Vi du: {151: "Tôi đau đầu", 152: ...}
    sentence_map = {}
    for index, row in df.iterrows():
        try:
            stt = int(row[stt_col])
            text = str(row[text_col]).strip()
            # chuan hoa cau: thuong them dau cham ket thuc neu chua co
            if not text.endswith('.'):
                text += " ."
            sentence_map[stt] = text
        except ValueError:
            continue

    print(f"Đã tải {len(sentence_map)} câu dịch từ Excel.")

    # 2. Đọc dataset.json hiện tại
    if dataset_json_path.exists():
        with open(dataset_json_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    else:
        dataset = []
        print("Không tìm thấy dataset.json cũ, sẽ tạo mới.")

    # Đếm số dòng hiện tại
    existing_ids = {item["id"] for item in dataset}
    print(f"Dataset hiện tại có {len(existing_ids)} videos.")

    # 3. Quét thư mục videos để tìm các video mới
    # Giả định video name format: SENT151_S01_R01_F.mp4
    video_files = list(videos_dir.glob("*.mp4"))
    
    new_entries = 0
    for vf in video_files:
        vid_id = vf.stem  # e.g. SENT151_S01_R01_F
        
        if vid_id in existing_ids:
            continue  # Đã có trong json rồi
            
        # Parse STT từ vid_id. Format: SENT151 -> lay so 151
        parts = vid_id.split("_")
        if len(parts) < 3:
            continue
            
        sent_part = parts[0]
        try:
            sent_num = int(sent_part.replace("SENT", ""))
        except ValueError:
            print(f"Skip file: {vid_id} (Không parse được số SENT)")
            continue

        if sent_num not in sentence_map:
            print(f"Cảnh báo: Video {vid_id} không có bản dịch tương ứng trong Excel (STT {sent_num}).")
            continue

        # Phan bo Split: Train, Val, Test giông hệt cũ
        # S01-S04: train, S05: val, S06: test
        signer_part = parts[1] # e.g. S01
        
        split = "train"
        if signer_part == "S05":
            split = "val"
        elif signer_part == "S06":
            split = "test"

        entry = {
            "id": vid_id,
            "signer": signer_part,
            "translation": sentence_map[sent_num],
            "gloss_sequence": [""],  # Bỏ trống gloss
            "split": split
        }
        dataset.append(entry)
        new_entries += 1

    # 4. Ghi đè file dataset.json
    if new_entries > 0:
        with open(dataset_json_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"✅ Đã thêm {new_entries} video mới vào dataset.json thành công!")
    else:
        print("Không có video mới nào được thêm. Vui lòng kiểm tra lại ID video và STT trong Excel.")

if __name__ == "__main__":
    main()
