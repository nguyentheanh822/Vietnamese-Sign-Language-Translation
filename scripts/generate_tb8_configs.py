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
        'num_workers': 2,   # Reduced from 4 to 2 for server safety
        'patience': 30,
        'save_every': 5,
        'weight_decay': 0.001,
        'ce_weight': 1.0,
        'ctc_weight': 0.0,
        'frame_ce_weight': 0.0,
        'save_dir': '',
    },
}

strategies = [
    ('gloss_free',       0.0, 0.0,  'No gloss supervision'),
    ('ctc_only',         0.5, 0.0,  'CTC loss only'),
    ('supervised',       0.0, 0.3,  'Supervised frame-level CE (manual labels)'),
    ('ctc_supervised',   0.3, 0.3,  'CTC + Supervised combined'),
]

os.makedirs('source/config/phase7/table8', exist_ok=True)
for name, ctc_w, frame_w, desc in strategies:
    cfg = copy.deepcopy(base)
    cfg['training']['ctc_weight'] = ctc_w
    cfg['training']['frame_ce_weight'] = frame_w
    cfg['training']['save_dir'] = f'results/run_tb8_{name}'
    path = f'source/config/phase7/table8/config_tb8_{name}.yaml'
    with open(path, 'w') as f:
        f.write(f'# Table 8: {desc}\n')
        yaml.dump(cfg, f, sort_keys=False, allow_unicode=True)
    print(f'Created: {path}  [ctc={ctc_w}, frame_ce={frame_w}, num_workers=2]')

print('\nAll Table 8 configs recreated with num_workers=2!')
