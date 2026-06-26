#!/usr/bin/env bash
# run_table7_benchmark.sh — Chay Benchmark (Table 7)
# Ghi chú: Kịch bản này so sánh 5 phương pháp trên GPUs: 3, 4, 5

set -e
cd /workspace/yhnn/VSL_GH

mkdir -p results/run_tb7_gru
mkdir -p results/run_tb7_bilstm
mkdir -p results/run_tb7_transformer_gloss_free
mkdir -p results/run_tb7_stgcn_transformer
mkdir -p results/run_tb7_transformer_ctc_ce

echo "============================================================"
echo "  Table 7: Benchmark 5 methods (Batch 1: GRU, BiLSTM, Trans-GF)"
echo "============================================================"

# GRU Seq2Seq
CUDA_VISIBLE_DEVICES=0 nohup bash -c "source venv/bin/activate && python source/train.py --config source/config/config_tb7_gru.yaml" > results/run_tb7_gru/train.log 2>&1 &
PID1=$!
echo "Started GRU on GPU 0 (PID: $PID1)"

# BiLSTM Seq2Seq
CUDA_VISIBLE_DEVICES=1 nohup bash -c "source venv/bin/activate && python source/train.py --config source/config/config_tb7_bilstm.yaml" > results/run_tb7_bilstm/train.log 2>&1 &
PID2=$!
echo "Started BiLSTM on GPU 1 (PID: $PID2)"

# Transformer (Gloss-Free)
CUDA_VISIBLE_DEVICES=2 nohup bash -c "source venv/bin/activate && python source/train.py --config source/config/config_tb7_transformer_gloss_free.yaml" > results/run_tb7_transformer_gloss_free/train.log 2>&1 &
PID3=$!
echo "Started Transformer (Gloss-Free) on GPU 2 (PID: $PID3)"

echo "Waiting for Batch 1 to finish..."
wait $PID1 $PID2 $PID3

echo "============================================================"
echo "  Table 7: Batch 2 (STGCN, Trans-CTC-CE)"
echo "============================================================"

# ST-GCN + Transformer
CUDA_VISIBLE_DEVICES=0 nohup bash -c "source venv/bin/activate && python source/train.py --config source/config/config_tb7_stgcn_transformer.yaml" > results/run_tb7_stgcn_transformer/train.log 2>&1 &
PID4=$!
echo "Started ST-GCN on GPU 0 (PID: $PID4)"

# Transformer (CTC+CE)
CUDA_VISIBLE_DEVICES=1 nohup bash -c "source venv/bin/activate && python source/train.py --config source/config/config_tb7_transformer_ctc_ce.yaml" > results/run_tb7_transformer_ctc_ce/train.log 2>&1 &
PID5=$!
echo "Started Transformer (CTC+CE) on GPU 1 (PID: $PID5)"

echo "Waiting for Batch 2 to finish..."
wait $PID4 $PID5

echo "Table 7 Benchmark runs completed!"
