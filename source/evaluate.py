"""
evaluate.py -- Danh gia mo hinh tren tap val hoac test.

Usage:
  python source/evaluate.py --split val
  python source/evaluate.py --split test
  python source/evaluate.py --split val --checkpoint results/run/checkpoint_best.pt
  python source/evaluate.py --split val --beam_size 4
  python source/evaluate.py --split test --beam_size 4 --output results/run/test_beam4.json
"""

import os, sys, argparse, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from torch.utils.data import DataLoader

from data.dataset import VSLDataset, collate_fn
from models.slt_model import SLTModel
from models.sota_model import SOTAModel
from utils.vocab import Vocab, build_translation_vocab
from utils.metrics import batch_wer, bleu4
import yaml


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        default="source/config/config.yaml")
    parser.add_argument("--checkpoint",    default=None, help="Path to .pt checkpoint")
    parser.add_argument("--split",         default="test", choices=["train", "val", "test"])
    parser.add_argument("--output",        default=None,  help="JSON output file for predictions")
    parser.add_argument("--beam_size",     type=int, default=1,
                        help="Beam size (1 = greedy, >=2 = beam search)")
    parser.add_argument("--length_penalty", type=float, default=0.6,
                        help="Length penalty alpha for beam search (0=none, 0.6=default)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    base_dir = cfg["data"]["base_dir"]

    # -----------------------------------------------------------------------
    # Vocabulary -- PHAI load tu file da save khi train (khong rebuild)
    # Neu rebuild, Counter.most_common() co the cho thu tu khac -> token IDs sai
    # -----------------------------------------------------------------------
    gloss_vocab = Vocab.load(f"{base_dir}/{cfg['data']['gloss_vocab']}")

    trans_vocab_path = f"{base_dir}/data/trans_vocab.txt"
    if Path(trans_vocab_path).exists():
        trans_vocab = Vocab.load(trans_vocab_path)
        print(f"Trans vocab loaded from file: {len(trans_vocab)} tokens")
    else:
        # Fallback: rebuild (co the sai neu thu tu khac voi luc train)
        trans_vocab = build_translation_vocab(f"{base_dir}/{cfg['data']['dataset_json']}")
        print(f"[WARN] trans_vocab.txt not found, rebuilt: {len(trans_vocab)} tokens")

    print(f"Gloss vocab size : {len(gloss_vocab)}")
    print(f"Trans vocab size : {len(trans_vocab)}")

    # Dataset
    ds = VSLDataset(
        base_dir, split=args.split,
        gloss_vocab=gloss_vocab, trans_vocab=trans_vocab,
        max_seq_len=cfg["data"]["max_seq_len"],
        max_gloss_len=cfg["data"]["max_gloss_len"],
        max_trans_len=cfg["data"]["max_trans_len"],
        augment=False,
        use_face=cfg["data"].get("use_face", True),
    )
    # Giam batch size khi beam search de tranh OOM
    eval_batch = cfg["training"]["batch_size"]
    if args.beam_size > 1:
        eval_batch = max(1, eval_batch // args.beam_size)

    loader = DataLoader(
        ds, batch_size=eval_batch,
        shuffle=False, collate_fn=collate_fn,
        num_workers=cfg["training"]["num_workers"],
    )

    # Model
    m_cfg = cfg["model"]
    use_sota = m_cfg.get("use_sota", False)
    if use_sota:
        model = SOTAModel(
            input_dim=cfg["data"]["input_dim"],
            gloss_vocab_size=len(gloss_vocab),
            trans_vocab_size=len(trans_vocab),
            d_model=m_cfg["d_model"],
            nhead=m_cfg["nhead"],
            num_encoder_layers=m_cfg["num_encoder_layers"],
            num_decoder_layers=m_cfg["num_decoder_layers"],
            dim_feedforward=m_cfg["dim_feedforward"],
            dropout=m_cfg["dropout"],
            pad_idx=trans_vocab.pad_idx,
            num_nodes=m_cfg.get("num_nodes", 137),
            coords=m_cfg.get("coords", 6),
        ).to(device)
    else:
        model = SLTModel(
            input_dim=cfg["data"]["input_dim"],
            gloss_vocab_size=len(gloss_vocab),
            trans_vocab_size=len(trans_vocab),
            d_model=m_cfg["d_model"],
            nhead=m_cfg["nhead"],
            num_encoder_layers=m_cfg["num_encoder_layers"],
            num_decoder_layers=m_cfg["num_decoder_layers"],
            dim_feedforward=m_cfg["dim_feedforward"],
            dropout=m_cfg["dropout"],
            pad_idx=trans_vocab.pad_idx,
        ).to(device)

    # Load checkpoint -- uu tien: arg > checkpoint_best
    ckpt_path = args.checkpoint or f"{base_dir}/results/run/checkpoint_best.pt"
    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model_state = ckpt.get("model_state", ckpt)
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        epoch    = ckpt.get("epoch", "?")
        best_b   = ckpt.get("best_bleu", "?")
        print(f"Loaded checkpoint: {ckpt_path}")
        print(f"  epoch={epoch}, best_bleu={best_b}")
        if missing:
            print(f"  [WARN] Missing keys: {missing}")
        if unexpected:
            print(f"  [WARN] Unexpected keys: {unexpected}")
    else:
        print(f"[WARN] No checkpoint at {ckpt_path}, using random weights!")

    model.eval()

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------
    decode_mode = "beam_search" if args.beam_size > 1 else "greedy"
    print(f"\nDecoding mode: {decode_mode}"
          + (f" (beam={args.beam_size}, lp={args.length_penalty})" if args.beam_size > 1 else ""))

    all_hyps, all_refs, all_ids = [], [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            src      = batch["keypoints"].to(device)
            src_mask = batch["kp_mask"].to(device)

            if args.beam_size > 1:
                preds = model.beam_search_decode(
                    src, src_mask,
                    bos_idx=trans_vocab.bos_idx,
                    eos_idx=trans_vocab.eos_idx,
                    beam_size=args.beam_size,
                    max_len=cfg["evaluation"]["max_decode_len"],
                    length_penalty=args.length_penalty,
                )
            else:
                preds = model.greedy_decode(
                    src, src_mask,
                    bos_idx=trans_vocab.bos_idx,
                    eos_idx=trans_vocab.eos_idx,
                    max_len=cfg["evaluation"]["max_decode_len"],
                )

            for pred_ids, ref_ids, vid in zip(preds, batch["trans_ids"], batch["video_ids"]):
                hyp = trans_vocab.decode(pred_ids.tolist())
                ref = trans_vocab.decode(ref_ids.tolist())
                all_hyps.append(hyp)
                all_refs.append(ref)
                all_ids.append(vid)

            if (batch_idx + 1) % 5 == 0:
                print(f"  Processed {batch_idx+1}/{len(loader)} batches...")

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------
    bleu = bleu4(all_hyps, all_refs)
    wer_score = batch_wer(all_hyps, all_refs)

    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS -- split={args.split}")
    print("=" * 60)
    print(f"  Samples   : {len(all_hyps)}")
    print(f"  BLEU-4    : {bleu:.2f}")
    print(f"  WER       : {wer_score * 100:.2f}%")
    print(f"  Decode    : {decode_mode}" +
          (f" beam={args.beam_size}" if args.beam_size > 1 else ""))

    print("\n--- Sample predictions (first 10) ---")
    for vid, hyp, ref in zip(all_ids[:10], all_hyps[:10], all_refs[:10]):
        print(f"  [{vid}]")
        print(f"    Ref: {' '.join(ref)}")
        print(f"    Hyp: {' '.join(hyp)}")

    # Save predictions
    out_path = args.output
    if not out_path:
        suffix = f"beam{args.beam_size}" if args.beam_size > 1 else "greedy"
        out_path = f"{base_dir}/results/run/{args.split}_{suffix}_predictions.json"

    results = {
        "split":    args.split,
        "bleu4":    bleu,
        "wer":      wer_score * 100,
        "decode":   decode_mode,
        "beam_size": args.beam_size,
        "checkpoint": str(ckpt_path),
        "predictions": [
            {"video_id": vid, "reference": " ".join(ref), "hypothesis": " ".join(hyp)}
            for vid, ref, hyp in zip(all_ids, all_refs, all_hyps)
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nPredictions saved to: {out_path}")


if __name__ == "__main__":
    main()
