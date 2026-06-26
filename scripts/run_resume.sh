#!/usr/bin/env bash
# run_resume.sh — Chay tiep cac model bi loi hoac chua hoan thanh sau khi restart

set -e
cd /workspace/yhnn/VSL_GH

echo "============================================================"
echo "  Resume missing/failed runs on GPUs 0, 1, 2, 3, 4"
echo "============================================================"

# Table 7: GRU
CUDA_VISIBLE_DEVICES=0 nohup bash -c "source venv/bin/activate && python source/train.py --config source/config/config_tb7_gru.yaml" > results/run_tb7_gru/train.log 2>&1 &
PID1=$!
echo "Started GRU on GPU 0 (PID: $PID1)"

# Table 7: BiLSTM
CUDA_VISIBLE_DEVICES=1 nohup bash -c "source venv/bin/activate && python source/train.py --config source/config/config_tb7_bilstm.yaml" > results/run_tb7_bilstm/train.log 2>&1 &
PID2=$!
echo "Started BiLSTM on GPU 1 (PID: $PID2)"

# Table 7: ST-GCN
CUDA_VISIBLE_DEVICES=2 nohup bash -c "source venv/bin/activate && python source/train.py --config source/config/config_tb7_stgcn_transformer.yaml" > results/run_tb7_stgcn_transformer/train.log 2>&1 &
PID3=$!
echo "Started ST-GCN on GPU 2 (PID: $PID3)"

# Table 7: Transformer (CTC+CE)
CUDA_VISIBLE_DEVICES=3 nohup bash -c "source venv/bin/activate && python source/train.py --config source/config/config_tb7_transformer_ctc_ce.yaml" > results/run_tb7_transformer_ctc_ce/train.log 2>&1 &
PID4=$!
echo "Started Transformer (CTC+CE) on GPU 3 (PID: $PID4)"

# Table 8: CTC + Supervised
CUDA_VISIBLE_DEVICES=4 nohup bash -c "source venv/bin/activate && python source/train.py --config source/config/config_tb8_ctc_supervised.yaml" > results/run_tb8_ctc_supervised/train.log 2>&1 &
PID5=$!
echo "Started CTC+Supervised on GPU 4 (PID: $PID5)"

echo "Waiting for all resumed runs to finish..."
wait $PID1 $PID2 $PID3 $PID4 $PID5

echo "All resumed runs completed!"
