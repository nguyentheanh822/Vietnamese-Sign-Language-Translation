import json
import os
from pathlib import Path

def time_to_frames(time_str: str, fps: float = 30.0) -> int:
    """Convert HH:MM:SS.mmm to frame index"""
    h, m, s = time_str.split(":")
    seconds = int(h) * 3600 + int(m) * 60 + float(s)
    return int(round(seconds * fps))

def parse_annotation(filepath: str, fps: float = 30.0):
    boundaries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = [p for p in line.strip().replace('\t', ' ').split(' ') if p]
            if len(parts) >= 4 and parts[0] == "Gloss":
                start_str = parts[1]
                end_str = parts[2]
                gloss = parts[3]
                try:
                    start_frame = time_to_frames(start_str, fps)
                    end_frame = time_to_frames(end_str, fps)
                    boundaries.append({
                        "start": start_frame,
                        "end": end_frame,
                        "gloss": gloss
                    })
                except Exception as e:
                    print(f"Error parsing line: {line.strip()} -> parts: {parts}")
                    raise e
    return boundaries

def main():
    base_dir = Path("/workspace/yhnn/VSL_GH")
    dataset_path = base_dir / "data" / "dataset.json"
    ann_dir = base_dir / "data" / "annotations"
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    updated = 0
    for entry in dataset:
        video_id = entry["id"]
        ann_file = ann_dir / f"{video_id}.txt"
        
        if ann_file.exists():
            boundaries = parse_annotation(str(ann_file), fps=entry.get("fps", 30.0))
            entry["gloss_frame_boundaries"] = boundaries
            updated += 1
            
    # Save back
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        
    print(f"Added frame boundaries to {updated}/{len(dataset)} entries in dataset.json")

if __name__ == "__main__":
    main()
