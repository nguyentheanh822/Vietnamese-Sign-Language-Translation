import json
import argparse
from pathlib import Path

def generate_loso_splits(dataset_path: str, output_dir: str):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Find all unique signers
    signers = sorted(list(set(d.get("signer_id", "S01") for d in data if "signer_id" in d)))
    print(f"Found signers: {signers}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for i, test_signer in enumerate(signers):
        # We'll pick the next signer as validation to be consistent
        val_signer = signers[(i + 1) % len(signers)]
        
        # The rest are training
        loso_data = []
        for d in data:
            entry = dict(d)
            signer = entry.get("signer_id")
            if signer == test_signer:
                entry["split"] = "test"
            elif signer == val_signer:
                entry["split"] = "val"
            else:
                entry["split"] = "train"
            loso_data.append(entry)

        out_name = Path(output_dir) / f"dataset_loso_{test_signer.lower()}.json"
        with open(out_name, "w", encoding="utf-8") as f:
            json.dump(loso_data, f, ensure_ascii=False, indent=2)
        print(f"Generated {out_name} (Test: {test_signer}, Val: {val_signer})")

if __name__ == "__main__":
    generate_loso_splits("data/dataset.json", "data")
