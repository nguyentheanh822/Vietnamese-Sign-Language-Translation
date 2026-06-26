import json, glob, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../source'))
from utils.metrics import bleu4, batch_wer, chrf_score, rouge_l_score

dirs = [
    "results/eval_tb7_transformer",
    "results/eval_tb7_stgcn",
    "results/eval_tb7_mamba",
    "results/eval_tb7_conformer",
    "results/eval_tb7_tcn",
    
    "results/eval_tb8_ctc",
    "results/eval_tb8_ctc_sup",
    "results/eval_tb8_gloss_free",
    
    "results/eval_tb9_no_face",
    "results/eval_tb9_no_vel",
    "results/eval_tb9_multi_view",
    
    "results/eval_tb10_s01",
    "results/eval_tb10_s02",
    "results/eval_tb10_s03",
    "results/eval_tb10_s04",
    "results/eval_tb10_s05",
    "results/eval_tb10_s06",
]

print(f"{'Model Experiment':<30s} | {'BLEU-4':<8s} | {'WER (%)':<8s} | {'chrF':<8s} | {'ROUGE-L':<8s}")
print("-" * 75)

for d in dirs:
    f = os.path.join(d, "eval_predictions.json")
    if not os.path.exists(f):
        continue
    with open(f, "r", encoding="utf-8") as file:
        data = json.load(file)
    predictions = data.get("predictions", [])
    if not predictions: continue
    
    all_hyps = [p["hypothesis"].split() for p in predictions]
    all_refs = [p["reference"].split() for p in predictions]
    
    b4 = bleu4(all_hyps, all_refs)
    wer = batch_wer(all_hyps, all_refs) * 100
    try:
        chrf = chrf_score(all_hyps, all_refs)
    except:
        chrf = -1.0
    try:
        rouge = rouge_l_score(all_hyps, all_refs)
    except:
        rouge = -1.0
        
    name = d.replace("results/eval_", "")
    print(f"{name:<30s} | {b4:>6.2f}   | {wer:>6.2f}   | {chrf:>6.2f}   | {rouge:>6.2f}")

