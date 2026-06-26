import os
import re
from pathlib import Path

def extract_best_metrics(log_file):
    best_bleu = 0.0
    best_wer = 0.0
    # Search for "Best model saved (BLEU=...)"
    # Or "Epoch ... | BLEU4=..."
    # Actually train.log has: "Epoch ... | Train=... | Val=... | BLEU4=5.55 | Time=18s"
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Find all BLEU scores
    bleu_matches = re.findall(r"BLEU4=([\d\.]+)", content)
    if bleu_matches:
        bleu_scores = [float(x) for x in bleu_matches]
        best_bleu = max(bleu_scores)
        
    # WER isn't printed directly in the log by default, it usually needs evaluate.py
    # If the log contains WER, we'd extract it here. 
    # For now we just return BLEU-4.
    return best_bleu, best_wer

def parse_runs():
    base_dir = Path("/workspace/yhnn/VSL_GH/results")
    results = {}
    
    if not base_dir.exists():
        return results

    for run_dir in base_dir.iterdir():
        if run_dir.is_dir() and run_dir.name.startswith("run_"):
            log_path = run_dir / "train.log"
            if log_path.exists():
                bleu, wer = extract_best_metrics(log_path)
                results[run_dir.name] = {"bleu": bleu, "wer": wer}
    return results

def main():
    results = parse_runs()
    
    print("=====================================================")
    print("   TỔNG HỢP KẾT QUẢ CÁC BẢNG (BLEU-4)                ")
    print("=====================================================")
    
    # Table 7
    print("\n[Table 7: Benchmark 5 methods]")
    tb7_keys = {
        "GRU Seq2Seq": "run_tb7_gru",
        "BiLSTM Seq2Seq": "run_tb7_bilstm",
        "Transformer (Gloss-Free)": "run_tb7_transformer_gloss_free",
        "ST-GCN + Transformer": "run_tb7_stgcn_transformer",
        "Transformer (CTC+CE)": "run_tb7_transformer_ctc_ce"
    }
    for name, folder in tb7_keys.items():
        score = results.get(folder, {}).get("bleu", "Running/N/A")
        print(f"{name:30s} : {score}")

    # Table 8
    print("\n[Table 8: Gloss Supervision]")
    tb8_keys = {
        "Gloss-Free": "run_tb8_gloss_free",
        "CTC (Sequence-level)": "run_tb8_ctc",
        "Supervised (Frame-level CE)": "run_tb8_supervised",
        "CTC + Supervised": "run_tb8_ctc_supervised"
    }
    for name, folder in tb8_keys.items():
        score = results.get(folder, {}).get("bleu", "Running/N/A")
        print(f"{name:30s} : {score}")

    # Table 9
    print("\n[Table 9: Ablation Study]")
    tb9_keys = {
        "Full model": "run_v2",
        "w/o face": "run_no_face",
        "w/o velocity": "run_no_vel"
    }
    for name, folder in tb9_keys.items():
        score = results.get(folder, {}).get("bleu", "Running/N/A")
        print(f"{name:30s} : {score}")

    # Table 10
    print("\n[Table 10: LOSO Signer Generalization]")
    tb10_keys = {
        "Test S01": "run_loso_s01",
        "Test S02": "run_loso_s02",
        "Test S03": "run_loso_s03",
        "Test S04": "run_loso_s04",
        "Test S05": "run_loso_s05",
        "Test S06": "run_loso_s06"
    }
    scores = []
    for name, folder in tb10_keys.items():
        res = results.get(folder, {}).get("bleu", "Running/N/A")
        print(f"{name:30s} : {res}")
        if isinstance(res, float) and res > 0:
            scores.append(res)
    
    if len(scores) == 6:
        mean_bleu = sum(scores) / 6
        print(f"{'Mean':30s} : {mean_bleu:.2f}")

    print("\n=====================================================")

if __name__ == "__main__":
    main()
