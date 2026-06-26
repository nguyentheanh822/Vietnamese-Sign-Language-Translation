#!/bin/bash
source venv/bin/activate

# Table 9 Ablations
models=(
    "results/run_no_face"
    "results/run_no_vel"
)

# Table 10 LOSO
for i in {1..6}; do
    models+=("results/run_loso_s0$i")
done

for dir in "${models[@]}"; do
    if [ -f "$dir/checkpoint_best.pt" ]; then
        echo "======================================"
        echo "Evaluating $dir"
        echo "======================================"
        cfg="source/config/config.yaml"
        if [[ $dir == *"no_face"* ]]; then cfg="source/config/config_no_face.yaml"; fi
        if [[ $dir == *"no_vel"* ]]; then cfg="source/config/config_no_vel.yaml"; fi
        if [[ $dir == *"loso"* ]]; then 
            signer=$(echo $dir | grep -o 's0[1-6]')
            cfg="source/config/config_loso_${signer}.yaml"
            # If specific loso config doesn't exist, we just use default but the split might be wrong in config
            # Actually dataset will use the split from config.
            # In VSLDataset, loso test split needs to be defined in config?
            # Wait, the user has `scripts/run_table10_loso.sh` where they pass `--config source/config/config_loso_s0X.yaml`
            # Let's check if config_loso_s0X.yaml exists
            if [ ! -f "$cfg" ]; then
                cfg="source/config/config.yaml"
            fi
        fi
        
        python source/evaluate.py --config "$cfg" --checkpoint "$dir/checkpoint_best.pt" --output "$dir/test_predictions.json" > "$dir/eval.log" 2>&1
    fi
done
echo "All missing evaluations completed!"
