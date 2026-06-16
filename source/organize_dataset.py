"""
organize_dataset.py
-------------------
Script tu dong sap xep files sau khi upload tu Windows len server.
Chuyen mp4 -> data/videos/ va txt -> data/annotations/
Sau do sinh ra splits/train.txt, splits/val.txt, splits/test.txt

Quy uoc:
  S01 - S04  =>  train (4 signers x 3 repetitions = 12 clips/sentence)
  S05        =>  val   (1 signer  x 1 repetition  =  1 clip/sentence)
  S06        =>  test  (1 signer  x 1 repetition  =  1 clip/sentence)

Usage:
  python source/organize_dataset.py
"""

import os
import shutil
from pathlib import Path

# ---- Paths ----------------------------------------------------------------
BASE_DIR    = Path("/workspace/yhnn/VSL_GH")
RAW_DIR     = BASE_DIR / "data" / "raw"
VIDEO_DIR   = BASE_DIR / "data" / "videos"
ANNOT_DIR   = BASE_DIR / "data" / "annotations"
SPLITS_DIR  = BASE_DIR / "data" / "splits"

TRAIN_SIGNERS = {"S01", "S02", "S03", "S04"}
VAL_SIGNERS   = {"S05"}
TEST_SIGNERS  = {"S06"}

# ---- Step 1: Move files ---------------------------------------------------
print("=" * 60)
print("Step 1: Phan loai va di chuyen files...")
print("=" * 60)

mp4_count = 0
txt_count = 0
skip_count = 0

for f in sorted(RAW_DIR.rglob("*")):
    if not f.is_file():
        continue
    
    ext = f.suffix.lower()
    stem = f.stem  # ten file khong co duoi, vi du: SENT001_S01_R01_F
    
    if ext == ".mp4":
        dst = VIDEO_DIR / f.name
        if not dst.exists():
            shutil.move(str(f), str(dst))
            mp4_count += 1
        else:
            skip_count += 1
    
    elif ext == ".txt":
        dst = ANNOT_DIR / f.name
        if not dst.exists():
            shutil.move(str(f), str(dst))
            txt_count += 1
        else:
            skip_count += 1

print(f"  Videos  moved : {mp4_count}")
print(f"  TXT     moved : {txt_count}")
print(f"  Skipped (dup) : {skip_count}")

# ---- Step 2: Verify pairing -----------------------------------------------
print("\nStep 2: Kiem tra cap doi video <-> annotation...")
video_stems = {f.stem for f in VIDEO_DIR.glob("*.mp4")}
annot_stems = {f.stem for f in ANNOT_DIR.glob("*.txt")}

missing_annot = video_stems - annot_stems
missing_video = annot_stems - video_stems

if missing_annot:
    print(f"  [WARN] {len(missing_annot)} video THIEU annotation:")
    for x in sorted(missing_annot)[:5]:
        print(f"    - {x}")
    if len(missing_annot) > 5:
        print(f"    ... va {len(missing_annot)-5} file khac")
else:
    print("  [OK] Tat ca videos deu co annotation!")

if missing_video:
    print(f"  [WARN] {len(missing_video)} annotation THIEU video")
else:
    print("  [OK] Tat ca annotations deu co video!")

# ---- Step 3: Generate splits ----------------------------------------------
print("\nStep 3: Sinh train/val/test splits...")

train_list, val_list, test_list, unknown_list = [], [], [], []

for stem in sorted(video_stems):
    # Ten file: SENT001_S01_R01_F
    parts = stem.split("_")
    if len(parts) < 4:
        unknown_list.append(stem)
        continue
    
    signer = parts[1]  # VD: S01
    
    if signer in TRAIN_SIGNERS:
        train_list.append(stem)
    elif signer in VAL_SIGNERS:
        val_list.append(stem)
    elif signer in TEST_SIGNERS:
        test_list.append(stem)
    else:
        unknown_list.append(stem)

splits = {
    "train": train_list,
    "val":   val_list,
    "test":  test_list,
}

for split_name, items in splits.items():
    split_file = SPLITS_DIR / f"{split_name}.txt"
    with open(split_file, "w", encoding="utf-8") as fp:
        fp.write("\n".join(items))
    print(f"  {split_name:5s}: {len(items):4d} clips  -> {split_file}")

if unknown_list:
    print(f"  [WARN] {len(unknown_list)} file khong xac dinh duoc split: {unknown_list[:3]}")

# ---- Step 4: Summary ------------------------------------------------------
print("\n" + "=" * 60)
print("TONG KET DATASET")
print("=" * 60)
print(f"  Total videos      : {len(video_stems)}")
print(f"  Total annotations : {len(annot_stems)}")
print(f"  Train clips       : {len(train_list)}")
print(f"  Val   clips       : {len(val_list)}")
print(f"  Test  clips       : {len(test_list)}")

# Dem so sentence
sentences = {stem.split("_")[0] for stem in video_stems}
print(f"  So cau (sentences): {len(sentences)}")

# Dem unique glosses tu annotation files
print("\nDang dem unique glosses tu annotation files...")
glosses = set()
for txt_file in ANNOT_DIR.glob("*.txt"):
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            # Format: Gloss  \t\t  start  \t  end  \t  GLOSS_LABEL
            if parts[0].strip() == "Gloss" and len(parts) >= 4:
                gloss_label = parts[-1].strip()
                if gloss_label:
                    glosses.add(gloss_label)

print(f"  Unique glosses    : {len(glosses)}")

# Luu gloss vocabulary
vocab_file = BASE_DIR / "data" / "gloss_vocab.txt"
with open(vocab_file, "w", encoding="utf-8") as f:
    f.write("<blank>\n<unk>\n")
    for g in sorted(glosses):
        f.write(g + "\n")
print(f"  Gloss vocab saved : {vocab_file}")
print("\n[DONE] Dataset da duoc to chuc xong!")
