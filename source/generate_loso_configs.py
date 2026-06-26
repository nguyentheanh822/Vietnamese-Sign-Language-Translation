import yaml
import copy
from pathlib import Path

def generate_loso_configs(base_config_path: str):
    with open(base_config_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
        
    signers = ["s01", "s02", "s03", "s04", "s05", "s06"]
    
    for s in signers:
        cfg = copy.deepcopy(base_cfg)
        # Update dataset JSON
        cfg["data"]["dataset_json"] = f"data/dataset_loso_{s}.json"
        
        # Update save dir
        cfg["training"]["save_dir"] = f"results/run_loso_{s}"
        
        # We also need to configure GPUs if we want to run them in parallel
        # Let's just leave gpus: [0] or similar so we can override by CUDA_VISIBLE_DEVICES
        cfg["training"]["gpus"] = [0]
        
        # Write out
        out_path = Path("source/config") / f"config_loso_{s}.yaml"
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            
        print(f"Generated {out_path}")

if __name__ == "__main__":
    generate_loso_configs("source/config/config.yaml")
