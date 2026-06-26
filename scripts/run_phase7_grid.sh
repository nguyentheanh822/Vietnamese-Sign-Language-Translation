#!/bin/bash
source venv/bin/activate

echo "Starting Grid Search for frame_ce_weight..."

# Round 1
echo "Running weight 0.1 on GPU 0 and weight 0.3 on GPU 3..."
CUDA_VISIBLE_DEVICES=0 taskset -c 0-15 python source/train.py --config source/config/phase7/config_grid_0.1.yaml > results/grid_0.1.log 2>&1 &
PID1=$!
CUDA_VISIBLE_DEVICES=3 taskset -c 16-31 python source/train.py --config source/config/phase7/config_grid_0.3.yaml > results/grid_0.3.log 2>&1 &
PID2=$!
wait $PID1
wait $PID2

# Round 2
echo "Running weight 0.5 on GPU 0 and weight 0.8 on GPU 3..."
CUDA_VISIBLE_DEVICES=0 taskset -c 0-15 python source/train.py --config source/config/phase7/config_grid_0.5.yaml > results/grid_0.5.log 2>&1 &
PID1=$!
CUDA_VISIBLE_DEVICES=3 taskset -c 16-31 python source/train.py --config source/config/phase7/config_grid_0.8.yaml > results/grid_0.8.log 2>&1 &
PID2=$!
wait $PID1
wait $PID2

# Round 3
echo "Running weight 1.0 on GPU 0..."
CUDA_VISIBLE_DEVICES=0 taskset -c 0-15 python source/train.py --config source/config/phase7/config_grid_1.0.yaml > results/grid_1.0.log 2>&1 &
wait $!

echo "Grid search completed!"
