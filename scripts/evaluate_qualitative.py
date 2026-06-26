#!/usr/bin/env python3
"""
evaluate_qualitative.py
Doc file ket qua du doan JSON tu evaluate.py, tinh BLEU score cho tung cau, 
va xuat ra 2 cau tot nhat (Good cases) va 2 cau te nhat (Bad cases).
"""

import json
import argparse
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to predictions JSON file")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    predictions = data.get("predictions", [])
    if not predictions:
        print("No predictions found in JSON.")
        return

    chencherry = SmoothingFunction()
    
    results = []
    for p in predictions:
        vid = p["video_id"]
        ref_tokens = p["reference"].split()
        hyp_tokens = p["hypothesis"].split()
        
        # Calculate sentence level BLEU
        # Reference expects list of lists of tokens
        score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=chencherry.method1)
        
        results.append({
            "video_id": vid,
            "reference": p["reference"],
            "hypothesis": p["hypothesis"],
            "bleu": score
        })

    # Sort by BLEU score
    results.sort(key=lambda x: x["bleu"], reverse=True)

    print("=" * 60)
    print("QUALITATIVE RESULTS: 2 BEST CASES")
    print("=" * 60)
    for i in range(2):
        r = results[i]
        print(f"[{r['video_id']}] (BLEU: {r['bleu']*100:.2f})")
        print(f"  Ref: {r['reference']}")
        print(f"  Hyp: {r['hypothesis']}\n")

    print("=" * 60)
    print("QUALITATIVE RESULTS: 2 WORST CASES")
    print("=" * 60)
    # Take from the end, ensuring we don't pick empty references if any, but let's just pick the last 2
    for i in range(1, 3):
        r = results[-i]
        print(f"[{r['video_id']}] (BLEU: {r['bleu']*100:.2f})")
        print(f"  Ref: {r['reference']}")
        print(f"  Hyp: {r['hypothesis']}\n")

if __name__ == "__main__":
    main()
