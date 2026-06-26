import yaml
import copy
from pathlib import Path

def generate_table8_configs(base_config_path: str):
    with open(base_config_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
        
    experiments = {
        "gloss_free": {"ctc_weight": 0.0, "frame_ce_weight": 0.0, "ce_weight": 1.0},
        "ctc": {"ctc_weight": 0.5, "frame_ce_weight": 0.0, "ce_weight": 0.5},
        "supervised": {"ctc_weight": 0.0, "frame_ce_weight": 0.5, "ce_weight": 0.5},
        "ctc_supervised": {"ctc_weight": 0.25, "frame_ce_weight": 0.25, "ce_weight": 0.5},
    }
    
    Path("source/config").mkdir(parents=True, exist_ok=True)

    for name, weights in experiments.items():
        cfg = copy.deepcopy(base_cfg)
        cfg["training"]["ctc_weight"] = weights["ctc_weight"]
        cfg["training"]["frame_ce_weight"] = weights["frame_ce_weight"]
        cfg["training"]["ce_weight"] = weights["ce_weight"]
        cfg["training"]["save_dir"] = f"results/run_tb8_{name}"
        # Force single GPU for these test runs
        cfg["training"]["gpus"] = [0]
        
        out_path = Path("source/config") / f"config_tb8_{name}.yaml"
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            
        print(f"Generated {out_path} with weights: {weights}")

if __name__ == "__main__":
    generate_table8_configs("source/config/config.yaml")
