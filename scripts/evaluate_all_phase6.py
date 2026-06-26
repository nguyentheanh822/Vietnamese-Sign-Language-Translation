#!/usr/bin/env python3
"""
evaluate_all_phase6.py — Evaluate ALL Phase 6 models with full metrics.
Loads each checkpoint_best.pt, runs greedy decode on val set,
and computes BLEU-4, WER, chrF, ROUGE-L.
"""

import os, sys, yaml, json, torch
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../source'))

from data.dataset import VSLDataset, collate_fn
from torch.utils.data import DataLoader
from models.conformer_model import ConformerSLTModel
from utils.vocab import Vocab, build_translation_vocab
from utils.metrics import bleu4, batch_wer, chrf_score, rouge_l_score

def evaluate_model(config_path, checkpoint_path, device):
    """Load model from config+checkpoint, evaluate on val, return all metrics."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_dir = cfg["data"]["base_dir"]
    gloss_vocab = Vocab.load(f"{base_dir}/{cfg['data']['gloss_vocab']}")
    trans_vocab = build_translation_vocab(f"{base_dir}/{cfg['data']['dataset_json']}")

    val_ds = VSLDataset(
        base_dir, split="val",
        gloss_vocab=gloss_vocab, trans_vocab=trans_vocab,
        max_seq_len=cfg["data"]["max_seq_len"],
        max_gloss_len=cfg["data"]["max_gloss_len"],
        max_trans_len=cfg["data"]["max_trans_len"],
        augment=False,
        use_face=cfg["data"].get("use_face", True),
        use_velocity=cfg["data"].get("use_velocity", True),
        view_mode=cfg["data"].get("view_mode", "F"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=False, collate_fn=collate_fn,
        num_workers=2, pin_memory=True,
    )

    model = ConformerSLTModel(
        vocab_size=len(trans_vocab),
        d_model=cfg["model"]["d_model"],
        nhead=cfg["model"]["nhead"],
        num_encoder_layers=cfg["model"]["num_encoder_layers"],
        num_decoder_layers=cfg["model"]["num_decoder_layers"],
        dim_feedforward=cfg["model"]["dim_feedforward"],
        dropout=cfg["model"]["dropout"],
        input_dim=cfg["data"]["input_dim"],
        max_seq_len=cfg["data"]["max_seq_len"],
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    all_hyps, all_refs = [], []
    with torch.no_grad():
        for batch in val_loader:
            src = batch["keypoints"].to(device)
            src_mask = batch["kp_mask"].to(device)
            preds = model.greedy_decode(
                src, src_mask,
                bos_idx=trans_vocab.bos_idx,
                eos_idx=trans_vocab.eos_idx,
                max_len=cfg["evaluation"]["max_decode_len"],
            )
            pred_tokens = [trans_vocab.decode(row.tolist()) for row in preds]
            ref_tokens = [trans_vocab.decode(row.tolist()) for row in batch["trans_ids"]]
            all_hyps.extend(pred_tokens)
            all_refs.extend(ref_tokens)

    b4 = bleu4(all_hyps, all_refs)
    w = batch_wer(all_hyps, all_refs) * 100
    try:
        cf = chrf_score(all_hyps, all_refs)
    except Exception:
        cf = -1.0
    try:
        rl = rouge_l_score(all_hyps, all_refs)
    except Exception:
        rl = -1.0

    return b4, w, cf, rl


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    configs = {
        # Table 8
        "tb8_conformer_glossfree":      ("Gloss-Free",           "T8"),
        "tb8_conformer_ctc":            ("CTC-based",            "T8"),
        "tb8_conformer_supervised":     ("Supervised (Frame CE)","T8"),
        "tb8_conformer_ctc_supervised": ("CTC + Supervised",     "T8"),
        # Table 9
        "tb9_conformer_no_vel":    ("w/o velocity",  "T9"),
        "tb9_conformer_no_face":   ("w/o face",      "T9"),
        "tb9_conformer_multiview": ("Multi-view",    "T9"),
        # Table 10
        "tb10_conformer_loso_s01": ("LOSO S01", "T10"),
        "tb10_conformer_loso_s02": ("LOSO S02", "T10"),
        "tb10_conformer_loso_s03": ("LOSO S03", "T10"),
        "tb10_conformer_loso_s04": ("LOSO S04", "T10"),
        "tb10_conformer_loso_s05": ("LOSO S05", "T10"),
        "tb10_conformer_loso_s06": ("LOSO S06", "T10"),
    }

    results = {}
    print(f"{'Name':<30s} | {'BLEU-4':>7s} | {'WER(%)':>7s} | {'chrF':>7s} | {'ROUGE-L':>7s}")
    print("-" * 75)

    for folder, (name, table) in configs.items():
        config_path = f"source/config/phase6/config_{folder}.yaml"
        ckpt_path = f"results/run_phase6_{folder}/checkpoint_best.pt"
        if not os.path.exists(ckpt_path):
            print(f"{name:<30s} | {'N/A':>7s} | {'N/A':>7s} | {'N/A':>7s} | {'N/A':>7s}")
            continue
        b4, w, cf, rl = evaluate_model(config_path, ckpt_path, device)
        print(f"{name:<30s} | {b4:7.2f} | {w:7.2f} | {cf:7.2f} | {rl:7.2f}")
        results[folder] = {"name": name, "table": table, "bleu4": b4, "wer": w, "chrf": cf, "rougel": rl}

    # Save to JSON for later use
    with open("results/phase6_full_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/phase6_full_metrics.json")


if __name__ == "__main__":
    main()
