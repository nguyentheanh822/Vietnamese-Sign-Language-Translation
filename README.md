# VSL-GH: Vietnamese Sign Language Dataset (Gloss & Hand)

## Overview

**VSL-GH** is a Vietnamese continuous sign language dataset collected for research on Sign Language Recognition (CSLR) and Sign Language Translation (SLT). It contains front-view videos of signers performing Vietnamese Sign Language (VSL) sentences, with time-aligned gloss annotations and Vietnamese text translations.

## Dataset Statistics

| Attribute | Value |
|---|---|
| Total videos | 8, 400 |
| Total sentences | 150 |
| Total signers | 6 |
| Unique glosses | 231 |
| Total gloss tokens | 8,113 |
| Avg. glosses/sentence | 3.86 |
| Avg. words/translation | 5.77 |
| Avg. video duration | 3.10 s |
| Video resolution | 1080 × 1080 (1:1) |
| Frame rate | 30 fps |
| Background | Green screen (controlled) |
| View | Front |

### Splits

| Split | Signers | Videos |
|---|---|---|
| Train | S01, S02, S03, S04 (×3 reps each) | 1,800 |
| Val | S05 (×1 rep) | 150 |
| Test | S06 (×1 rep) | 150 |

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
│   ├── videos/            # 2,100 × .mp4
│   ├── annotations/       # 2,100 × .txt (gloss + translation)
│   ├── keypoints/         # MediaPipe Holistic keypoints (.npy)
│   ├── splits/
│   │   ├── train.txt      # 1,800 video IDs
│   │   ├── val.txt        # 150 video IDs
│   │   └── test.txt       # 150 video IDs
│   ├── dataset.json       # Full metadata per video
│   ├── dataset_stats.json # Dataset-level statistics
│   └── gloss_vocab.txt    # 231 unique glosses + special tokens
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

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{vsl_gh_2026,
  title     = {VSL-GH: A Vietnamese Continuous Sign Language Dataset with Gloss Annotations},
  author    = {[Author Names]},
  year      = {2026},
  publisher = {[University/Institution]},
  note      = {[Thesis/Paper title]}
}
```

## License

This dataset is released under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

You are free to share and adapt the material for any purpose, as long as appropriate credit is given.

## Contact

For questions about the dataset, please contact: [your email]
