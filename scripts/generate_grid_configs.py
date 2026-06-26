import yaml
import os

base_config_path = 'source/config/phase6/config_tb8_conformer_supervised.yaml'
with open(base_config_path, 'r') as f:
    base_cfg = yaml.safe_load(f)

weights = [0.1, 0.3, 0.5, 0.8, 1.0]
os.makedirs('source/config/phase7', exist_ok=True)

for w in weights:
    cfg = base_cfg.copy()
    cfg['training']['frame_ce_weight'] = w
    # Just to be safe, set save_dir to a specific dir
    cfg['training']['save_dir'] = f'results/run_phase7_grid_{w}'
    
    # Restrict GPUs to 0 and 3 for this particular phase as they are 100% free
    # I'll create a script that assigns specific GPUs later, but we can set 
    # it to [0] for now since we will override it via CUDA_VISIBLE_DEVICES
    cfg['training']['gpus'] = [0]
    
    out_path = f'source/config/phase7/config_grid_{w}.yaml'
    with open(out_path, 'w') as f:
        yaml.dump(cfg, f, sort_keys=False)
    print(f'Created {out_path}')
