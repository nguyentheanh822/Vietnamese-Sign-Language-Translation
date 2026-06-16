"""
generate_metadata.py
---------------------
Sinh file dataset.json chua metadata day du cho tung video.
Dung cho: paper, thesis, public release, keypoint extraction pipeline.

Output:
  - data/dataset.json         : metadata tung video
  - data/dataset_stats.json   : thong ke tong the cho paper
  - data/README.md            : mo ta dataset

Usage:
  python source/generate_metadata.py
"""

import json
import re
from pathlib import Path
from collections import defaultdict, Counter

# ---- Paths ----------------------------------------------------------------
BASE_DIR    = Path("/workspace/yhnn/VSL_GH")
VIDEO_DIR   = BASE_DIR / "data" / "videos"
ANNOT_DIR   = BASE_DIR / "data" / "annotations"
SPLITS_DIR  = BASE_DIR / "data" / "splits"
DATA_DIR    = BASE_DIR / "data"

# ---- Load splits ----------------------------------------------------------
def load_split(split_name):
    f = SPLITS_DIR / f"{split_name}.txt"
    return set(f.read_text(encoding="utf-8").splitlines())

train_set = load_split("train")
val_set   = load_split("val")
test_set  = load_split("test")

def get_split(stem):
    if stem in train_set: return "train"
    if stem in val_set:   return "val"
    if stem in test_set:  return "test"
    return "unknown"

# ---- Parse annotation file ------------------------------------------------
def parse_annotation(txt_path):
    """
    Parse file annotation dang:
      Gloss    00:00:00.010    00:00:00.860    TÔI
      Translation    00:00:00.000    00:00:03.000    Câu dịch tiếng Việt
    """
    glosses = []
    translation = ""

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"\t+", line)
            if len(parts) < 4:
                continue

            record_type = parts[0].strip()
            start_time  = parts[1].strip()
            end_time    = parts[2].strip()
            content     = parts[3].strip()

            if record_type == "Gloss":
                glosses.append({
                    "gloss":      content,
                    "start_time": start_time,
                    "end_time":   end_time,
                })
            elif record_type == "Translation":
                translation = content

    return glosses, translation

# ---- Time string -> seconds -----------------------------------------------
def time_to_seconds(t):
    """'HH:MM:SS.mmm' -> float seconds"""
    try:
        h, m, s = t.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except:
        return 0.0

# ---- Build metadata -------------------------------------------------------
print("Dang xay dung metadata cho 2100 videos...")

dataset = []
all_glosses_flat = []
sentence_translations = {}   # sent_id -> translation text
sent_gloss_lengths = defaultdict(list)

video_files = sorted(VIDEO_DIR.glob("*.mp4"))

for video_path in video_files:
    stem = video_path.stem  # SENT001_S01_R01_F
    parts = stem.split("_")

    if len(parts) < 4:
        print(f"  [WARN] Ten file khong dung format: {stem}")
        continue

    sent_id  = parts[0]       # SENT001
    signer   = parts[1]       # S01
    rep      = parts[2]       # R01
    view     = parts[3]       # F

    sent_num    = int(sent_id[4:])   # 1
    signer_num  = int(signer[1:])    # 1
    rep_num     = int(rep[1:])       # 1

    # Parse annotation
    annot_path = ANNOT_DIR / f"{stem}.txt"
    if not annot_path.exists():
        print(f"  [WARN] Khong co annotation: {stem}")
        glosses, translation = [], ""
    else:
        glosses, translation = parse_annotation(annot_path)

    # Tinh tong thoi luong tu gloss cuoi
    duration_sec = 0.0
    if glosses:
        duration_sec = time_to_seconds(glosses[-1]["end_time"])

    # Tong thoi luong cua sentence (lay tu Translation neu co)
    split = get_split(stem)

    # Gloss sequence (chi cac tu, khong timestamp)
    gloss_sequence = [g["gloss"] for g in glosses]
    all_glosses_flat.extend(gloss_sequence)

    # Luu translation theo sentence (dung cho thong ke)
    if sent_id not in sentence_translations:
        sentence_translations[sent_id] = translation

    entry = {
        "id":               stem,
        "sentence_id":      sent_id,
        "sentence_num":     sent_num,
        "signer_id":        signer,
        "signer_num":       signer_num,
        "repetition":       rep,
        "repetition_num":   rep_num,
        "view":             view,
        "split":            split,
        "video_file":       f"videos/{stem}.mp4",
        "annotation_file":  f"annotations/{stem}.txt",
        "fps":              30,
        "resolution":       "1080x1080",
        "duration_sec":     round(duration_sec, 3),
        "num_glosses":      len(gloss_sequence),
        "gloss_sequence":   gloss_sequence,
        "glosses_detail":   glosses,
        "translation":      translation,
    }
    dataset.append(entry)
    sent_gloss_lengths[sent_id].append(len(gloss_sequence))

# Sort by sentence, signer, repetition
dataset.sort(key=lambda x: (x["sentence_num"], x["signer_num"], x["repetition_num"]))

# ---- Save dataset.json ---------------------------------------------------
out_json = DATA_DIR / "dataset.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)
print(f"Saved: {out_json}  ({len(dataset)} entries)")

# ---- Compute stats -------------------------------------------------------
print("\nDang tinh thong ke...")

gloss_counter = Counter(all_glosses_flat)
num_unique_glosses = len(gloss_counter)

durations = [e["duration_sec"] for e in dataset if e["duration_sec"] > 0]
gloss_counts = [e["num_glosses"] for e in dataset if e["num_glosses"] > 0]

# Word count in translations
trans_word_counts = []
for e in dataset:
    if e["translation"]:
        trans_word_counts.append(len(e["translation"].split()))

split_counts = Counter(e["split"] for e in dataset)

stats = {
    "dataset_name":         "VSL-GH",
    "description":          "Vietnamese Sign Language - Gloss and Hand (Front View)",
    "version":              "1.0.0",
    "total_videos":         len(dataset),
    "total_sentences":      len(sentence_translations),
    "total_signers":        6,
    "train_signers":        4,
    "val_signers":          1,
    "test_signers":         1,
    "split_counts": {
        "train": split_counts["train"],
        "val":   split_counts["val"],
        "test":  split_counts["test"],
    },
    "unique_glosses":       num_unique_glosses,
    "total_gloss_tokens":   len(all_glosses_flat),
    "video_specs": {
        "resolution":       "1080x1080",
        "fps":              30,
        "aspect_ratio":     "1:1",
        "view":             "Front",
        "background":       "Green screen",
    },
    "duration_stats_sec": {
        "min":    round(min(durations), 3) if durations else 0,
        "max":    round(max(durations), 3) if durations else 0,
        "mean":   round(sum(durations)/len(durations), 3) if durations else 0,
    },
    "gloss_per_sentence_stats": {
        "min":    min(gloss_counts) if gloss_counts else 0,
        "max":    max(gloss_counts) if gloss_counts else 0,
        "mean":   round(sum(gloss_counts)/len(gloss_counts), 2) if gloss_counts else 0,
    },
    "translation_word_stats": {
        "min":    min(trans_word_counts) if trans_word_counts else 0,
        "max":    max(trans_word_counts) if trans_word_counts else 0,
        "mean":   round(sum(trans_word_counts)/len(trans_word_counts), 2) if trans_word_counts else 0,
    },
    "top_30_glosses": [
        {"gloss": g, "count": c}
        for g, c in gloss_counter.most_common(30)
    ],
}

out_stats = DATA_DIR / "dataset_stats.json"
with open(out_stats, "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print(f"Saved: {out_stats}")

# ---- Print summary -------------------------------------------------------
print("\n" + "=" * 60)
print("THONG KE DATASET (de dung trong Paper/Thesis)")
print("=" * 60)
print(f"  Ten dataset       : VSL-GH (Vietnamese Sign Language - Gloss/Hand)")
print(f"  Tong so videos    : {stats['total_videos']}")
print(f"  Tong so cau       : {stats['total_sentences']}")
print(f"  So nguoi ky hieu  : {stats['total_signers']} (Train:{stats['train_signers']}, Val:{stats['val_signers']}, Test:{stats['test_signers']})")
print(f"  Splits            : Train={stats['split_counts']['train']}, Val={stats['split_counts']['val']}, Test={stats['split_counts']['test']}")
print(f"  Unique glosses    : {stats['unique_glosses']}")
print(f"  Tong gloss tokens : {stats['total_gloss_tokens']}")
print(f"  Thoi luong video  : {stats['duration_stats_sec']['min']}s - {stats['duration_stats_sec']['max']}s (TB: {stats['duration_stats_sec']['mean']}s)")
print(f"  Gloss/cau         : {stats['gloss_per_sentence_stats']['min']} - {stats['gloss_per_sentence_stats']['max']} (TB: {stats['gloss_per_sentence_stats']['mean']})")
print(f"  Tu/cau dich       : {stats['translation_word_stats']['min']} - {stats['translation_word_stats']['max']} (TB: {stats['translation_word_stats']['mean']})")
print(f"\n  Top 10 glosses pho bien nhat:")
for item in stats['top_30_glosses'][:10]:
    print(f"    {item['gloss']:20s}  {item['count']:4d} lan")
print("\n[DONE] Metadata da duoc tao xong!")
