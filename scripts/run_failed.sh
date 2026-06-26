#!/bin/bash
source venv/bin/activate
cd /workspace/yhnn/VSL_GH

evaluate_model() {
    config=$1
    checkpoint=$2
    out_dir=$3
    split=$4
    
    mkdir -p $out_dir
    echo "Evaluating $config on split=$split ..."
    CUDA_VISIBLE_DEVICES=1 python source/evaluate.py \
        --config $config \
        --checkpoint $checkpoint \
        --split $split \
        --output $out_dir/eval_predictions.json || echo "Failed to evaluate $config"
}

evaluate_model source/config/phase7/config_tb7_mamba.yaml results/run_tb7_mamba/checkpoint_best.pt results/eval_tb7_mamba test
evaluate_model source/config/phase7/config_tb7_conformer.yaml results/run_tb7_conformer/checkpoint_best.pt results/eval_tb7_conformer test
evaluate_model source/config/phase7/table8/config_tb8_ctc_only.yaml results/run_tb8_ctc/checkpoint_best.pt results/eval_tb8_ctc test

python scripts/compute_all_phase7.py > results/phase7_full_metrics.txt
cat results/phase7_full_metrics.txt
