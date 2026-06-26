import yaml, copy, os

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
        'dataset_json': 'data/dataset.json',
        'gloss_vocab': 'data/gloss_vocab.txt',
        'input_dim': 822,
        'use_velocity': True,
        'use_face': True,
        'view_mode': 'F',
        'keypoints_dir': 'data/keypoints',
        'max_gloss_len': 20,
        'max_seq_len': 300,
        'max_trans_len': 30,
        'splits_dir': 'data/splits',
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
        'clip_grad_norm': 1.0,
        'epochs': 150,
        'learning_rate': 5.0e-05,
        'lr_warmup_steps': 1000,
        'num_workers': 2,
        'patience': 30,
        'save_every': 5,
        'weight_decay': 0.001,
        'ce_weight': 1.0,
        'ctc_weight': 0.0,
        'frame_ce_weight': 0.3, # Best from Table 8
        'save_dir': '',
    },
}

experiments = [
    ('no_velocity', False, True, 'F', 411, 'Ablation: No Velocity'),
    ('no_face',     True, False, 'F', 402, 'Ablation: No Face Keypoints'),
    ('multi_view',  True, True, 'FS', 1644, 'Ablation: Multi-view (Front + Side)'),
]

os.makedirs('source/config/phase7/table9', exist_ok=True)
for name, use_vel, use_face, view_mode, dim, desc in experiments:
    cfg = copy.deepcopy(base)
    cfg['data']['use_velocity'] = use_vel
    cfg['data']['use_face'] = use_face
    cfg['data']['view_mode'] = view_mode
    cfg['data']['input_dim'] = dim
    cfg['training']['save_dir'] = f'results/run_tb9_{name}'
    
    path = f'source/config/phase7/table9/config_tb9_{name}.yaml'
    with open(path, 'w') as f:
        f.write(f'# Table 9: {desc}\n')
        yaml.dump(cfg, f, sort_keys=False, allow_unicode=True)
    print(f'Created: {path} [dim={dim}, vel={use_vel}, face={use_face}, view={view_mode}]')
