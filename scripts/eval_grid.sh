#!/bin/bash
source venv/bin/activate

echo "Evaluating grid search models on Test set..."

configs=("config_p7_grid_w01.yaml" "config_p7_grid_w03.yaml" "config_p7_grid_w05.yaml" "config_p7_grid_w08.yaml" "config_p7_grid_w10.yaml")
dirs=("run_p7_grid_w01" "run_p7_grid_w03" "run_p7_grid_w05" "run_p7_grid_w08" "run_p7_grid_w10")

for i in "${!configs[@]}"; do
    conf="source/config/phase7/${configs[$i]}"
    d="results/${dirs[$i]}"
    ckpt="$d/checkpoint_best.pt"
    
    if [ -f "$ckpt" ]; then
        echo "----------------------------------------"
        echo "Evaluating ${dirs[$i]} (weight: ${configs[$i]})"
        python source/evaluate.py --config "$conf" --checkpoint "$ckpt"
    else
        echo "----------------------------------------"
        echo "Model ${dirs[$i]} did not finish or checkpoint_best.pt missing."
    fi
done
