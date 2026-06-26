#!/bin/bash
source venv/bin/activate
cd /workspace/yhnn/VSL_GH

# Helper function
evaluate_model() {
    config=$1
    checkpoint=$2
    out_dir=$3
    split=$4
    
    mkdir -p $out_dir
    if [ ! -f $checkpoint ]; then
        echo "Missing $checkpoint"
        return
    fi
    
    echo "Evaluating $config on split=$split ..."
    CUDA_VISIBLE_DEVICES=1 python source/evaluate.py \
        --config $config \
        --checkpoint $checkpoint \
        --split $split \
        --output $out_dir/eval_predictions.json || echo "Failed to evaluate $config"
}

# Table 7
evaluate_model source/config/config_tb7_transformer_ctc_ce.yaml results/run_tb7_transformer_ctc_ce/checkpoint_best.pt results/eval_tb7_transformer test
evaluate_model source/config/config_tb7_stgcn_transformer.yaml results/run_tb7_stgcn_transformer/checkpoint_best.pt results/eval_tb7_stgcn test
evaluate_model source/config/phase7/table7/config_tb7_mamba.yaml results/run_tb7_mamba/checkpoint_best.pt results/eval_tb7_mamba test
evaluate_model source/config/phase7/table7/config_tb7_conformer.yaml results/run_tb7_conformer/checkpoint_best.pt results/eval_tb7_conformer test
evaluate_model source/config/phase7/table8/config_tb8_supervised.yaml results/run_tb8_supervised/checkpoint_best.pt results/eval_tb7_tcn test

# Table 8
evaluate_model source/config/phase7/table8/config_tb8_ctc.yaml results/run_tb8_ctc/checkpoint_best.pt results/eval_tb8_ctc test
evaluate_model source/config/phase7/table8/config_tb8_ctc_supervised.yaml results/run_tb8_ctc_supervised/checkpoint_best.pt results/eval_tb8_ctc_sup test
evaluate_model source/config/phase7/table8/config_tb8_gloss_free.yaml results/run_tb8_gloss_free/checkpoint_best.pt results/eval_tb8_gloss_free test

# Table 9
evaluate_model source/config/phase7/table9/config_tb9_no_face.yaml results/run_tb9_no_face/checkpoint_best.pt results/eval_tb9_no_face test
evaluate_model source/config/phase7/table9/config_tb9_no_vel.yaml results/run_tb9_no_vel/checkpoint_best.pt results/eval_tb9_no_vel test
evaluate_model source/config/phase7/table9/config_tb9_multi_view.yaml results/run_tb9_multi_view/checkpoint_best.pt results/eval_tb9_multi_view test

# Table 10 (LOSO)
for i in {1..6}; do
    evaluate_model source/config/phase7/table10/config_tb10_loso_s0$i.yaml results/run_tb10_loso_s0$i/checkpoint_best.pt results/eval_tb10_s0$i val
done

echo "All evaluations finished! Computing metrics..."

# Run a python script to compute everything and print beautifully
cat << 'EOF' > scripts/compute_all_phase7.py
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

EOF

python scripts/compute_all_phase7.py > results/phase7_full_metrics.txt
cat results/phase7_full_metrics.txt
