import os
import re

RESULTS_DIR = "results"
PHASE6_PREFIX = "run_phase6_"

def get_best_bleu(log_path):
    if not os.path.exists(log_path):
        return "Not started"
    
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Check if finished
    match = re.search(r"Training done! Best BLEU-4 = ([\d\.]+)", content)
    if match:
        return match.group(1)
        
    # If not finished, find the max BLEU so far
    bleus = re.findall(r"BLEU4=([\d\.]+)", content)
    if bleus:
        max_bleu = max([float(b) for b in bleus])
        return f"Running (Max so far: {max_bleu})"
        
    return "Running (No epoch finished yet)"

def main():
    print("==================================================")
    print("      KẾT QUẢ THỰC NGHIỆM PHASE 6 (BẢNG 8, 9, 10)  ")
    print("==================================================\n")
    
    # Table 8
    print("## Bảng 8: Impact of Gloss Supervision (Base: Conformer)")
    t8_configs = [
        ("Gloss-Free", "tb8_conformer_glossfree"),
        ("CTC", "tb8_conformer_ctc"),
        ("CTC + Supervised", "tb8_conformer_ctc_supervised"),
        ("Supervised (Frame CE)", "tb8_conformer_supervised")
    ]
    for name, folder in t8_configs:
        log = os.path.join(RESULTS_DIR, f"{PHASE6_PREFIX}{folder}", "train.log")
        bleu = get_best_bleu(log)
        print(f"{name:25s} : {bleu}")
        
    print("\n")
    
    # Table 9
    print("## Bảng 9: Ablation Study on Input Features (Base: Supervised Conformer)")
    t9_configs = [
        ("w/o velocity features", "tb9_conformer_no_vel"),
        ("w/o face keypoints", "tb9_conformer_no_face"),
        ("Full model (front-view)", "tb8_conformer_supervised"),
        ("+ Side-view (multi-view)", "tb9_conformer_multiview")
    ]
    for name, folder in t9_configs:
        log = os.path.join(RESULTS_DIR, f"{PHASE6_PREFIX}{folder}", "train.log")
        bleu = get_best_bleu(log)
        print(f"{name:25s} : {bleu}")
        
    print("\n")
    
    # Table 10
    print("## Bảng 10: Signer Generalization LOSO (Base: Supervised Conformer)")
    t10_configs = [
        "tb10_conformer_loso_s01",
        "tb10_conformer_loso_s02",
        "tb10_conformer_loso_s03",
        "tb10_conformer_loso_s04",
        "tb10_conformer_loso_s05",
        "tb10_conformer_loso_s06"
    ]
    sum_bleu = 0.0
    count = 0
    for folder in t10_configs:
        log = os.path.join(RESULTS_DIR, f"{PHASE6_PREFIX}{folder}", "train.log")
        bleu = get_best_bleu(log)
        signer = folder.split('_')[-1].upper()
        print(f"Test Signer {signer:13s} : {bleu}")
        try:
            val = float(bleu)
            sum_bleu += val
            count += 1
        except:
            pass
            
    if count == 6:
        print("-" * 40)
        print(f"Mean BLEU-4               : {sum_bleu/6:.2f}")

if __name__ == "__main__":
    main()
