import yaml
import copy

# Base config from the best grid search result (w=0.3)
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
    'evaluation': {
        'beam_size': 4,
        'max_decode_len': 30,
    },
    'model': {
        'activation': 'relu',
        'd_model': 512,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'nhead': 8,
        'num_decoder_layers': 6,
        'num_encoder_layers': 6,
        'type': 'conformer',  # Will be overridden
    },
    'training': {
        'batch_size': 16,
        'ce_weight': 1.0,
        'clip_grad_norm': 1.0,
        'ctc_weight': 0.0,
        'epochs': 150,
        'frame_ce_weight': 0.3,  # Best from grid search
        'gpus': [0],
        'learning_rate': 5.0e-05,
        'lr_warmup_steps': 1000,
        'num_workers': 4,
        'patience': 30,
        'save_dir': '',  # Will be overridden
        'save_every': 5,
        'weight_decay': 0.001,
    },
}

# 5 models for Table 7
models = [
    {
        'name': 'stgcn',
        'type': 'stgcn',
        'extra_model': {'num_nodes': 137, 'coords': 6},
    },
    {
        'name': 'transformer',
        'type': 'transformer',
        'extra_model': {},
    },
    {
        'name': 'tcn',
        'type': 'tcn',
        'extra_model': {},
    },
    {
        'name': 'mamba',
        'type': 'mamba',
        'extra_model': {},
    },
    {
        'name': 'conformer',
        'type': 'conformer',
        'extra_model': {},
    },
]

for m in models:
    cfg = copy.deepcopy(base)
    cfg['model']['type'] = m['type']
    cfg['model'].update(m['extra_model'])
    cfg['training']['save_dir'] = f"results/run_tb7_{m['name']}"
    
    path = f"source/config/phase7/config_tb7_{m['name']}.yaml"
    with open(path, 'w') as f:
        yaml.dump(cfg, f, sort_keys=False, allow_unicode=True)
    print(f"Created: {path}")

print("\nAll Table 7 configs created!")
