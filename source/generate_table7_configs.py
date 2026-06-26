import yaml
import copy
from pathlib import Path

def generate_table7_configs(base_config_path: str):
    with open(base_config_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
        
    experiments = {
        "gru": {"type": "gru", "ctc_weight": 0.5, "ce_weight": 0.5},
        "bilstm": {"type": "bilstm", "ctc_weight": 0.5, "ce_weight": 0.5},
        "transformer_gloss_free": {"type": "transformer", "ctc_weight": 0.0, "ce_weight": 1.0},
        "stgcn_transformer": {"type": "stgcn", "ctc_weight": 0.5, "ce_weight": 0.5},
        "transformer_ctc_ce": {"type": "transformer", "ctc_weight": 0.5, "ce_weight": 0.5},
    }
    
    Path("source/config").mkdir(parents=True, exist_ok=True)

    for name, params in experiments.items():
        cfg = copy.deepcopy(base_cfg)
        cfg["model"]["type"] = params["type"]
        cfg["training"]["ctc_weight"] = params["ctc_weight"]
        cfg["training"]["ce_weight"] = params["ce_weight"]
        # Default frame CE is 0 for benchmark
        cfg["training"]["frame_ce_weight"] = 0.0
        cfg["training"]["save_dir"] = f"results/run_tb7_{name}"
        # Force single GPU for these test runs
        cfg["training"]["gpus"] = [0]
        
        out_path = Path("source/config") / f"config_tb7_{name}.yaml"
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            
        print(f"Generated {out_path} with params: {params}")

if __name__ == "__main__":
    generate_table7_configs("source/config/config.yaml")
