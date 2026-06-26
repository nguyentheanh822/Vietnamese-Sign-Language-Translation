#!/usr/bin/env python3
import json
import glob
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../source'))
from utils.metrics import bleu4, batch_wer, chrf_score, rouge_l_score

def main():
    print(f"{'Model Directory':<40s} | {'BLEU-4':<8s} | {'WER (%)':<8s} | {'chrF':<8s} | {'ROUGE-L':<8s}")
    print("-" * 80)
    for f in sorted(glob.glob("results/*/test_predictions.json")):
        dir_name = os.path.dirname(f)
        
        with open(f, "r", encoding="utf-8") as file:
            data = json.load(file)
            
        predictions = data.get("predictions", [])
        if not predictions:
            continue
            
        all_hyps = [p["hypothesis"].split() for p in predictions]
        all_refs = [p["reference"].split() for p in predictions]
        
        b4 = bleu4(all_hyps, all_refs)
        wer = batch_wer(all_hyps, all_refs) * 100
        
        # for chrF and ROUGE-L, we might need sacrebleu and rouge_score.
        # if they fail, we return -1
        try:
            chrf = chrf_score(all_hyps, all_refs)
        except Exception as e:
            chrf = -1.0
            
        try:
            rouge = rouge_l_score(all_hyps, all_refs)
        except Exception as e:
            rouge = -1.0
            
        print(f"{dir_name:<40s} | {b4:>6.2f}   | {wer:>6.2f}   | {chrf:>6.2f}   | {rouge:>6.2f}")

if __name__ == "__main__":
    main()
