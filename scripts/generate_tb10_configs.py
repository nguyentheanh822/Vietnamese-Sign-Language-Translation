import yaml
import copy
import os

base = {
    'augmentation': {
        'flip_prob': 0.5,
        'noise_std': 0.02,
        'rotation_deg': 20,
        'scale_factor': [0.85, 1.15],
        'speed_factor': [0.6, 1.4],
    },
    'data': {
        'annotations_dir': 'data/annotations',
        'base_dir': '/workspace/yhnn/VSL_GH',
        'gloss_vocab': 'data/gloss_vocab.txt',
        'input_dim': 822,
        'keypoints_dir': 'data/keypoints',
        'max_gloss_len': 20,
        'max_seq_len': 300,
        'max_trans_len': 30,
        'splits_dir': 'data/splits',
        'use_velocity': True,
        'use_face': True,
        'view_mode': 'F',
    },
    'evaluation': {'beam_size': 4, 'max_decode_len': 30},
    'model': {
        'activation': 'relu',
        'd_model': 512,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'nhead': 8,
        'num_decoder_layers': 6,
        'num_encoder_layers': 6,
        'type': 'tcn',
    },
    'training': {
        'batch_size': 16,
        'ce_weight': 1.0,
        'clip_grad_norm': 1.0,
        'ctc_weight': 0.0,
        'epochs': 150,
        'frame_ce_weight': 0.3, # Best supervised strategy
        'learning_rate': 5.0e-05,
        'lr_warmup_steps': 1000,
        'num_workers': 2, # Safe for parallel
        'patience': 30,
        'save_every': 5,
        'weight_decay': 0.001,
    }
}

os.makedirs('source/config/phase7/table10', exist_ok=True)

for i in range(1, 7):
    s_id = f"s0{i}"
    cfg = copy.deepcopy(base)
    cfg['data']['dataset_json'] = f"data/dataset_loso_{s_id}.json"
    cfg['training']['save_dir'] = f"results/run_tb10_loso_{s_id}"
    
    out_path = f"source/config/phase7/table10/config_tb10_loso_{s_id}.yaml"
    with open(out_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"Created: {out_path}")
