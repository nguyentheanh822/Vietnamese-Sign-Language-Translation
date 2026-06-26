import yaml
import os
from pathlib import Path

BASE_CONFIG = "source/config/config_sota_conformer.yaml"
OUTPUT_DIR = "source/config/phase6"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(BASE_CONFIG, "r", encoding="utf-8") as f:
    base = yaml.safe_load(f)

def write_config(name, modifications):
    cfg = yaml.safe_load(yaml.dump(base)) # deep copy
    
    # Apply modifications
    for k, v in modifications.items():
        keys = k.split('.')
        current = cfg
        for key in keys[:-1]:
            current = current[key]
        current[keys[-1]] = v
        
    # Update save_dir
    cfg["training"]["save_dir"] = f"results/run_phase6_{name}"
    
    out_path = os.path.join(OUTPUT_DIR, f"config_{name}.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"Generated: {out_path}")

# ==========================================
# Table 8: Impact of Gloss Supervision
# Base Architecture: Conformer
# ==========================================

# (a) Gloss-Free (No gloss loss, only Translation CE)
write_config("tb8_conformer_glossfree", {
    "training.ctc_weight": 0.0,
    "training.frame_ce_weight": 0.0,
    "training.ce_weight": 1.0
})

# (b) CTC (CTC only, no frame CE)
write_config("tb8_conformer_ctc", {
    "training.ctc_weight": 0.5,
    "training.frame_ce_weight": 0.0,
    "training.ce_weight": 0.5
})

# (c) Supervised (Frame CE only, no CTC) -> THIS IS THE ULTIMATE MODEL
write_config("tb8_conformer_supervised", {
    "training.ctc_weight": 0.0,
    "training.frame_ce_weight": 1.5,
    "training.ce_weight": 1.0
})

# (d) CTC + Supervised (Both)
write_config("tb8_conformer_ctc_supervised", {
    "training.ctc_weight": 0.3,
    "training.frame_ce_weight": 1.5,
    "training.ce_weight": 0.7
})

# ==========================================
# Table 9: Ablation Study on Input Features
# Base: tb8_conformer_supervised (The Ultimate Model)
# ==========================================

# w/o velocity
write_config("tb9_conformer_no_vel", {
    "training.ctc_weight": 0.0,
    "training.frame_ce_weight": 1.5,
    "training.ce_weight": 1.0,
    "data.input_dim": 411,
    "data.use_velocity": False
})

# w/o face
write_config("tb9_conformer_no_face", {
    "training.ctc_weight": 0.0,
    "training.frame_ce_weight": 1.5,
    "training.ce_weight": 1.0,
    "data.input_dim": 612, # (25 + 21*2)*3*2 = 402 if no face, wait. 
    # Actually, the dataset class handles the slicing. If use_face=False, input_dim is smaller.
    # Pose: 25*3=75. Hands: 42*3=126. Total = 201. With vel = 402.
    "data.input_dim": 402,
    "data.use_face": False
})

# multi-view (S+F)
write_config("tb9_conformer_multiview", {
    "training.ctc_weight": 0.0,
    "training.frame_ce_weight": 1.5,
    "training.ce_weight": 1.0,
    "data.input_dim": 1644, # 822 * 2 = 1644
    "data.view_mode": "FS"
})

# Note: The "Full model (front-view)" in Table 9 is just "tb8_conformer_supervised"

# ==========================================
# Table 10: Signer Generalization (LOSO)
# Base: tb8_conformer_supervised (The Ultimate Model)
# ==========================================

signers = ["S01", "S02", "S03", "S04", "S05", "S06"]
for signer in signers:
    write_config(f"tb10_conformer_loso_{signer.lower()}", {
        "training.ctc_weight": 0.0,
        "training.frame_ce_weight": 1.5,
        "training.ce_weight": 1.0,
        "data.dataset_json": f"data/dataset_loso_{signer.lower()}.json"
    })

print("Done generating 13 configs.")
