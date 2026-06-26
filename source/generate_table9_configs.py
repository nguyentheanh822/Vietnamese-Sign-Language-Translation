import yaml
from pathlib import Path

def create_config(base_cfg, changes, out_path):
    import copy
    cfg = copy.deepcopy(base_cfg)
    for k1, v1 in changes.items():
        for k2, v2 in v1.items():
            if k1 not in cfg: cfg[k1] = {}
            cfg[k1][k2] = v2
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

def main():
    base_file = "source/config/config_tb8_supervised.yaml"
    with open(base_file, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    configs = {
        "config_tb9_fs.yaml": {
            "data": {"view_mode": "FS", "input_dim": 1644},
            "training": {"save_dir": "results/run_tb9_fs"}
        }
    }

    out_dir = Path("source/config")
    for name, changes in configs.items():
        create_config(base_cfg, changes, out_dir / name)
        print(f"Generated {name}")

if __name__ == "__main__":
    main()
