#!/usr/bin/env bash
# eval_all_runs.sh — Eval tat ca runs (B, C, D) sau khi train xong
# Chay: bash eval_all_runs.sh
# Output: results/run_*/val_beam4.json va test_beam4.json

set -e
cd /workspace/yhnn/VSL_GH

echo "============================================================"
echo "  Eval All Runs — $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

eval_run() {
    local NAME=$1
    local CONFIG=$2
    local CKPT=$3
    local OUTDIR=$4
    local GPU=${5:-0}

    if [ ! -f "$CKPT" ]; then
        echo "[SKIP] $NAME: checkpoint not found at $CKPT"
        return
    fi

    echo ""
    echo "──── $NAME ($(basename $CKPT)) ────"

    echo "  [val] beam=4 ..."
    CUDA_VISIBLE_DEVICES=$GPU python3 source/evaluate.py \
        --config "$CONFIG" \
        --checkpoint "$CKPT" \
        --split val \
        --beam_size 4 \
        --output "$OUTDIR/val_beam4.json" \
        2>&1 | grep -E "BLEU|WER|Loaded|EVAL"

    echo "  [test] beam=4 ..."
    CUDA_VISIBLE_DEVICES=$GPU python3 source/evaluate.py \
        --config "$CONFIG" \
        --checkpoint "$CKPT" \
        --split test \
        --beam_size 4 \
        --output "$OUTDIR/test_beam4.json" \
        2>&1 | grep -E "BLEU|WER|Loaded|EVAL"

    echo "  → Saved to $OUTDIR/"
}

# Run v2 (baseline — da co ket qua greedy, gio lay beam)
eval_run "Run_v2 (baseline)"  \
    source/config/config.yaml \
    results/run_v2/checkpoint_best.pt \
    results/run_A_beam \
    0

# Run B — Small model
eval_run "Run_B_small"  \
    source/config/config_B_small.yaml \
    results/run_B_small/checkpoint_best.pt \
    results/run_B_small \
    0

# Run C — High Dropout
eval_run "Run_C_dropout"  \
    source/config/config_C_dropout.yaml \
    results/run_C_dropout/checkpoint_best.pt \
    results/run_C_dropout \
    1

# Run D — Balanced CTC
eval_run "Run_D_ctc05"  \
    source/config/config_D_ctc05.yaml \
    results/run_D_ctc05/checkpoint_best.pt \
    results/run_D_ctc05 \
    0

# ── Tong hop ket qua ────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  TONG HOP KET QUA (beam=4)"
echo "============================================================"
echo "  Run           | BLEU-4 Val | BLEU-4 Test"
echo "  --------------|------------|------------"

for run in run_A_beam run_B_small run_C_dropout run_D_ctc05; do
    VAL_FILE="results/$run/val_beam4.json"
    TEST_FILE="results/$run/test_beam4.json"
    if [ -f "$VAL_FILE" ] && [ -f "$TEST_FILE" ]; then
        VAL=$(python3 -c "import json; d=json.load(open('$VAL_FILE')); print(f\"{d['bleu4']:.2f}%\")" 2>/dev/null || echo "N/A")
        TST=$(python3 -c "import json; d=json.load(open('$TEST_FILE')); print(f\"{d['bleu4']:.2f}%\")" 2>/dev/null || echo "N/A")
        printf "  %-14s | %-10s | %s\n" "$run" "$VAL" "$TST"
    else
        printf "  %-14s | %-10s | %s\n" "$run" "N/A" "N/A"
    fi
done
echo "============================================================"
