# VSL-GH: Vietnamese Sign Language Dataset

VSL-GH is a large-scale, high-quality dataset for Vietnamese Sign Language (VSL) translation. It features multiple signers, dual-camera views (Frontal and Side), and detailed temporal boundary annotations.

## Dataset Highlights
- **Size:** 4,200 video clips (Frontal) + 4,200 video clips (Side-view)
- **Signers:** 6 independent signers
- **Annotations:** Gloss-level and translation-level sequences with accurate temporal boundaries.
- **Keypoints:** Extracted using MediaPipe Holistic (137 landmarks per frame).

## Project Structure
- `data/`: Contains dataset JSON metadata, vocabularies, and splits.
- `source/`: Contains the PyTorch source code for the models (ST-GCN, Transformer) and dataloaders.
- `scripts/`: Shell scripts to reproduce baseline experiments (Table 7, 8, 9, 10).
- `results/`: Contains training logs and checkpoints for the experiments.

| Attribute | Value |
|---|---|
| Total videos | 8,400 (4,200 Frontal + 4,200 Lateral) |
| Total sentences | 300 |
| Total signers | 6 |
| Unique glosses | 466 |
| Total gloss tokens (in 300 sentences) | 1,174 |
| Total gloss tokens (across 8,400 videos) | ~32,800 |
| Avg. glosses/sentence | 3.91 |
| Avg. words/translation | 6.2 |
| Avg. video duration | 3.6 s |
| Video resolution | 1080 × 1080 (1:1) |
| Frame rate | 30 fps |
| Background | Green screen (controlled) |
| View | Frontal + Lateral (Side) |

### Splits

| Split | Signers | Videos (per view) | Total Videos |
|---|---|---|---|
| Train | S01, S02, S03, S04 (×3 reps each) | 3,600 | 7,200 |
| Val | S05 (×1 rep) | 300 | 600 |
| Test | S06 (×1 rep) | 300 | 600 |

The val and test signers are **unseen during training**, enabling evaluation of **signer-independent generalization**.

## File Naming Convention

```
SENT{sentence_id}_S{signer_id}_R{repetition}_F.mp4
```

Example: `SENT001_S01_R02_F.mp4`
- `SENT001` — Sentence #1
- `S01` — Signer #1
- `R02` — Repetition 2
- `F` — Front view

## Annotation Format

Each video has a corresponding `.txt` file with the same base name:

```
Gloss       HH:MM:SS.mmm    HH:MM:SS.mmm    GLOSS_LABEL
...
Translation HH:MM:SS.mmm    HH:MM:SS.mmm    Câu dịch tiếng Việt
```

Example (`SENT001_S01_R01_F.txt`):
```
Gloss       00:00:00.010    00:00:00.860    TÔI
Gloss       00:00:00.860    00:00:01.430    HẸN
Gloss       00:00:01.430    00:00:01.900    TRƯỚC
Gloss       00:00:01.900    00:00:03.000    CHƯA
Translation 00:00:00.000    00:00:03.000    Tôi có phải chờ lâu không?
```

## Directory Structure

```
VSL-GH/
├── data/
│   ├── videos/            # 8,400 × .mp4 (4,200 Frontal + 4,200 Lateral)
│   ├── annotations/       # 4,200 × .txt (gloss + translation)
│   ├── keypoints/         # 8,400 × .npy (MediaPipe Holistic)
│   ├── splits/
│   │   ├── train.txt      # 3,600 video IDs
│   │   ├── val.txt        # 300 video IDs
│   │   └── test.txt       # 300 video IDs
│   ├── dataset.json       # Full metadata per video
│   ├── dataset_stats.json # Dataset-level statistics
│   └── gloss_vocab.txt    # 466 unique glosses + special tokens
├── results/               # Experiment results
└── source/                # Source code
```

## Usage

```python
import json

# Load full metadata
with open("data/dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Filter by split
train_data = [d for d in dataset if d["split"] == "train"]
val_data   = [d for d in dataset if d["split"] == "val"]
test_data  = [d for d in dataset if d["split"] == "test"]

# Access a sample
sample = train_data[0]
print(sample["translation"])      # Vietnamese sentence
print(sample["gloss_sequence"])   # ['TÔI', 'HẸN', 'TRƯỚC', 'CHƯA']
print(sample["video_file"])       # 'videos/SENT001_S01_R01_F.mp4'
```

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Extract keypoints (if not already downloaded): 
   `python source/extract_keypoints.py --num_workers 8`
3. Train the baseline model:
   `bash scripts/run_table7_baselines.sh`
4. Evaluate a trained model:
   `python source/evaluate.py --config source/config/config.yaml --checkpoint results/run_v2/checkpoint_best.pt`

## Citation
If you use this dataset, please cite the corresponding paper.

## License
This project and dataset are released under the MIT License. See `LICENSE` for more details.
