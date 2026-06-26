#!/bin/bash
source venv/bin/activate

seeds=(123 456 789)

echo "Starting Step 4: Multi-seed Training on GPU 5..."

for seed in "${seeds[@]}"; do
    # Gloss-Free configs
    cfg_gf="source/config/config_tb8_gloss_free_s${seed}.yaml"
    cp source/config/config_tb8_gloss_free.yaml "$cfg_gf"
    sed -i "s|results/run_tb8_gloss_free|results/run_tb8_gloss_free_s${seed}|g" "$cfg_gf"
    echo "  seed: $seed" >> "$cfg_gf"
    
    # Supervised configs
    cfg_sup="source/config/config_tb8_supervised_s${seed}.yaml"
    cp source/config/config_tb8_supervised.yaml "$cfg_sup"
    sed -i "s|results/run_tb8_supervised|results/run_tb8_supervised_s${seed}|g" "$cfg_sup"
    echo "  seed: $seed" >> "$cfg_sup"
    
    echo "-----------------------------------"
    echo "Training Gloss-Free Seed $seed"
    mkdir -p results/run_tb8_gloss_free_s${seed}
    CUDA_VISIBLE_DEVICES=5 python source/train.py --config "$cfg_gf" > results/run_tb8_gloss_free_s${seed}/train.log 2>&1
    
    echo "Training Supervised Seed $seed"
    mkdir -p results/run_tb8_supervised_s${seed}
    CUDA_VISIBLE_DEVICES=5 python source/train.py --config "$cfg_sup" > results/run_tb8_supervised_s${seed}/train.log 2>&1
done

echo "All Multi-seed trainings completed!"
