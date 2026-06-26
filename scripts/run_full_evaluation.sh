#!/bin/bash
source venv/bin/activate

for dir in results/run_*; do
    if [ -d "$dir" ] && [ -f "$dir/checkpoint_best.pt" ]; then
        run_name=$(basename "$dir")
        
        # Try to find the exact config file
        cfg="source/config/config_${run_name#run_}.yaml"
        
        # Exceptions for default runs
        if [[ "$run_name" == "run" || "$run_name" == "run_v2" || "$run_name" == "run_phase2" ]]; then 
            cfg="source/config/config.yaml"
        fi
        
        if [ ! -f "$cfg" ]; then
            echo "Warning: Config $cfg not found for $dir. Defaulting to config.yaml"
            cfg="source/config/config.yaml"
        fi
        
        echo "======================================"
        echo "Evaluating $dir using $cfg"
        echo "======================================"
        
        python source/evaluate.py --config "$cfg" --checkpoint "$dir/checkpoint_best.pt" --output "$dir/test_predictions.json" > "$dir/eval.log" 2>&1
    fi
done
echo "All evaluations completed!"
