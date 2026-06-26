"""
train.py — Training script cho VSL-GH Sign Language Translation.

Usage:
  CUDA_VISIBLE_DEVICES=0,1 python source/train.py
  CUDA_VISIBLE_DEVICES=0,1 python source/train.py --resume results/run/checkpoint_best.pt
"""

import os, sys, argparse, json, time, math
from pathlib import Path

# Them source dir vao path
sys.path.insert(0, str(Path(__file__).parent))

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel

from data.dataset import VSLDataset, collate_fn
from models.slt_model import SLTModel
from models.sota_model import SOTAModel
from models.gru_model import GRUSLTModel
from models.bilstm_model import BiLSTMSLTModel
from models.conformer_model import ConformerSLTModel
from models.st_transformer import STTransformerSLTModel
from models.tcn_model import TCNSLTModel
from models.mamba_model import MambaSLTModel
import numpy as np
import random
from utils.vocab import Vocab, build_translation_vocab
from utils.metrics import batch_wer, bleu4, ids_to_tokens


def load_config(path="source/config/config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def warmup_cosine_schedule(step, warmup_steps, total_steps, min_lr=1e-6, max_lr=1e-4):
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def compute_loss(model, batch, gloss_vocab, trans_vocab, cfg, device):
    """Tinh CTC loss + CE loss cho 1 batch."""
    src       = batch["keypoints"].to(device)    # (B, T, 411)
    src_mask  = batch["kp_mask"].to(device)      # (B, T)
    gloss_ids = batch["gloss_ids"].to(device)    # (B, G)
    gloss_lens = batch["gloss_lens"].to(device)  # (B,)
    trans_ids = batch["trans_ids"].to(device)    # (B, L)
    trans_lens = batch["trans_lens"].to(device)  # (B,)
    seq_lens  = batch["seq_lens"].to(device)     # (B,)

    # Decoder input: bo EOS, giu BOS
    tgt_in  = trans_ids[:, :-1]    # (B, L-1) input  [BOS, w1, w2, ...]
    tgt_out = trans_ids[:, 1:]     # (B, L-1) target [w1, w2, ..., EOS]
    tgt_mask = batch["trans_mask"].to(device)[:, :-1]

    # Forward pass
    trans_logits, ctc_log_probs = model(src, src_mask, tgt_in, tgt_mask)

    # ── CE Loss (translation) ────────────────────────────────────────────
    B, L, V = trans_logits.shape
    ce_loss = nn.CrossEntropyLoss(
        ignore_index=trans_vocab.pad_idx,
        label_smoothing=0.1,
    )(
        trans_logits.reshape(B * L, V),
        tgt_out.reshape(B * L),
    )

    # ── CTC Loss (gloss) ─────────────────────────────────────────────────
    # ctc_log_probs: (T, B, gloss_vocab)
    # Tinh input_lengths tu seq_lens (so frame thuc te)
    input_lengths = seq_lens.clamp(max=ctc_log_probs.size(0))
    gloss_lens_clamp = gloss_lens.clamp(min=1)

    ctc_loss = nn.CTCLoss(
        blank=gloss_vocab.blank_idx,
        zero_infinity=True,
    )(
        ctc_log_probs,
        gloss_ids,
        input_lengths,
        gloss_lens_clamp,
    )

    # ── Frame-level CE Loss (Supervised Gloss) ───────────────────────────
    frame_ce_loss = 0.0
    w_frame = cfg["training"].get("frame_ce_weight", 0.0)
    if w_frame > 0.0:
        log_probs = ctc_log_probs.permute(1, 2, 0) # (B, V, T)
        frame_gloss_ids = batch["frame_gloss_ids"].to(device) # (B, T)
        frame_gloss_ids_masked = frame_gloss_ids.clone()
        frame_gloss_ids_masked[batch["kp_mask"]] = -100 # Ignore padding
        frame_ce_loss = nn.NLLLoss(ignore_index=-100)(log_probs, frame_gloss_ids_masked)

    w_ctc = cfg["training"].get("ctc_weight", 0.0)
    w_ce  = cfg["training"].get("ce_weight", 1.0)
    
    total = w_ctc * ctc_loss + w_ce * ce_loss
    if w_frame > 0.0:
        total += w_frame * frame_ce_loss

    return total, ce_loss.item(), ctc_loss.item() if w_ctc > 0.0 else (frame_ce_loss.item() if w_frame > 0.0 else 0.0)


def evaluate(model, loader, gloss_vocab, trans_vocab, cfg, device):
    """Chay evaluate tren val/test set, tra ve BLEU-4 va Loss trung binh."""
    model.eval()
    total_loss, total_ce, total_ctc, n = 0, 0, 0, 0
    all_hyps, all_refs = [], []

    with torch.no_grad():
        for batch in loader:
            loss, ce, ctc = compute_loss(
                model, batch, gloss_vocab, trans_vocab, cfg, device
            )
            total_loss += loss.item()
            total_ce   += ce
            total_ctc  += ctc
            n += 1

            # Greedy decode
            src      = batch["keypoints"].to(device)
            src_mask = batch["kp_mask"].to(device)
            preds = model.module.greedy_decode(
                src, src_mask,
                bos_idx=trans_vocab.bos_idx,
                eos_idx=trans_vocab.eos_idx,
                max_len=cfg["evaluation"]["max_decode_len"],
            ) if hasattr(model, "module") else model.greedy_decode(
                src, src_mask,
                bos_idx=trans_vocab.bos_idx,
                eos_idx=trans_vocab.eos_idx,
                max_len=cfg["evaluation"]["max_decode_len"],
            )

            # Convert IDs to tokens
            pred_tokens = [
                trans_vocab.decode(row.tolist()) for row in preds
            ]
            # Reference: bo BOS/EOS
            ref_tokens = [
                trans_vocab.decode(row.tolist())
                for row in batch["trans_ids"]
            ]
            all_hyps.extend(pred_tokens)
            all_refs.extend(ref_tokens)

    bleu = bleu4(all_hyps, all_refs)
    avg_loss = total_loss / max(n, 1)
    return avg_loss, bleu, all_hyps[:5], all_refs[:5]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="source/config/config.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume")
    parser.add_argument("--gpu", default="0", help="GPU id to use")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ── GPU setup ───────────────────────────────────────────────────────────
    # NOTE: CUDA_VISIBLE_DEVICES phai duoc set tu shell command truoc khi chay script
    # Vi du: CUDA_VISIBLE_DEVICES=0,1 python3 source/train.py
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Set random seeds for reproducibility
    seed = cfg["training"].get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    n_gpu = torch.cuda.device_count()
    print(f"Using device: {device} | Visible GPUs: {n_gpu}")
    for i in range(n_gpu):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # ── Vocabulary ──────────────────────────────────────────────────────────
    base_dir = cfg["data"]["base_dir"]
    gloss_vocab = Vocab.load(f"{base_dir}/{cfg['data']['gloss_vocab']}")
    trans_vocab = build_translation_vocab(f"{base_dir}/{cfg['data']['dataset_json']}")
    print(f"Gloss vocab size : {len(gloss_vocab)}")
    print(f"Trans vocab size : {len(trans_vocab)}")

    # Save trans vocab for evaluation
    trans_vocab.save(f"{base_dir}/data/trans_vocab.txt")

    # ── Datasets & DataLoaders ──────────────────────────────────────────────
    aug = cfg.get("augmentation", {})
    train_ds = VSLDataset(
        base_dir, split="train",
        gloss_vocab=gloss_vocab, trans_vocab=trans_vocab,
        max_seq_len=cfg["data"]["max_seq_len"],
        max_gloss_len=cfg["data"]["max_gloss_len"],
        max_trans_len=cfg["data"]["max_trans_len"],
        augment=True,
        aug_flip_prob=aug.get("flip_prob", 0.5),
        aug_speed=tuple(aug.get("speed_factor", [0.8, 1.2])),
        aug_noise_std=aug.get("noise_std", 0.01),
        aug_rotation_deg=aug.get("rotation_deg", 15.0),
        use_face=cfg["data"].get("use_face", True),
        use_velocity=cfg["data"].get("use_velocity", True),
        view_mode=cfg["data"].get("view_mode", "F"),
    )
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

    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=True, collate_fn=collate_fn,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=False, collate_fn=collate_fn,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Model ───────────────────────────────────────────────────────────────
    m_cfg = cfg["model"]
    model_type = m_cfg.get("type", "transformer")
    
    if model_type == "stgcn":
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
        print("=> Initialized SOTAModel (ST-GCN + Transformer)")
    elif model_type == "gru":
        model = GRUSLTModel(
            input_dim=cfg["data"]["input_dim"],
            gloss_vocab_size=len(gloss_vocab),
            trans_vocab_size=len(trans_vocab),
            hidden_dim=m_cfg.get("d_model", 512),
            emb_dim=m_cfg.get("d_model", 512) // 2,
            num_layers=m_cfg.get("num_encoder_layers", 2),
            dropout=m_cfg.get("dropout", 0.1),
            pad_idx=trans_vocab.pad_idx,
        ).to(device)
        print("=> Initialized GRUSLTModel")
    elif model_type == "bilstm":
        model = BiLSTMSLTModel(
            input_dim=cfg["data"]["input_dim"],
            gloss_vocab_size=len(gloss_vocab),
            trans_vocab_size=len(trans_vocab),
            hidden_dim=m_cfg.get("d_model", 512),
            emb_dim=m_cfg.get("d_model", 512) // 2,
            num_layers=m_cfg.get("num_encoder_layers", 2),
            dropout=m_cfg.get("dropout", 0.1),
            pad_idx=trans_vocab.pad_idx,
        ).to(device)
        print("=> Initialized BiLSTMSLTModel")
    elif model_type == 'conformer':
        model = ConformerSLTModel(
            vocab_size=len(trans_vocab),
            d_model=cfg["model"]["d_model"],
            nhead=cfg["model"]["nhead"],
            num_encoder_layers=cfg["model"]["num_encoder_layers"],
            num_decoder_layers=cfg["model"]["num_decoder_layers"],
            dim_feedforward=cfg["model"]["dim_feedforward"],
            dropout=cfg["model"]["dropout"],
            input_dim=cfg["data"]["input_dim"],
            max_seq_len=cfg["data"]["max_seq_len"]
        ).to(device)
        print("=> Initialized ConformerSLTModel")
    elif model_type == 'st_transformer':
        model = STTransformerSLTModel(
            vocab_size=len(trans_vocab),
            d_model=cfg["model"]["d_model"],
            nhead=cfg["model"]["nhead"],
            num_encoder_layers=cfg["model"]["num_encoder_layers"],
            num_decoder_layers=cfg["model"]["num_decoder_layers"],
            dim_feedforward=cfg["model"]["dim_feedforward"],
            dropout=cfg["model"]["dropout"],
            num_nodes=137,
            coords=6,
            max_seq_len=cfg["data"]["max_seq_len"]
        ).to(device)
        print("=> Initialized STTransformerSLTModel")
    elif model_type == 'tcn':
        model = TCNSLTModel(
            vocab_size=len(trans_vocab),
            gloss_vocab_size=len(gloss_vocab),
            d_model=cfg["model"]["d_model"],
            nhead=cfg["model"]["nhead"],
            num_encoder_layers=cfg["model"]["num_encoder_layers"],
            num_decoder_layers=cfg["model"]["num_decoder_layers"],
            dim_feedforward=cfg["model"]["dim_feedforward"],
            dropout=cfg["model"]["dropout"],
            input_dim=cfg["data"]["input_dim"],
            max_seq_len=cfg["data"]["max_seq_len"]
        ).to(device)
        print("=> Initialized TCNSLTModel")
    elif model_type == 'mamba':
        model = MambaSLTModel(
            vocab_size=len(trans_vocab),
            gloss_vocab_size=len(gloss_vocab),
            d_model=cfg["model"]["d_model"],
            nhead=cfg["model"]["nhead"],
            num_encoder_layers=cfg["model"]["num_encoder_layers"],
            num_decoder_layers=cfg["model"]["num_decoder_layers"],
            dim_feedforward=cfg["model"]["dim_feedforward"],
            dropout=cfg["model"]["dropout"],
            input_dim=cfg["data"]["input_dim"],
            max_seq_len=cfg["data"]["max_seq_len"]
        ).to(device)
        print("=> Initialized MambaSLTModel")
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
        print("=> Initialized baseline SLTModel (Transformer)")

    # Single GPU (DataParallel khong tuong thich voi PyTorch 1.10.1 + CUDA 12.2 driver)
    model = model.to(device)
    print(f"Model tren GPU: {next(model.parameters()).device}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # ── Optimizer & Scheduler ───────────────────────────────────────────────
    t_cfg = cfg["training"]
    optimizer = optim.AdamW(
        model.parameters(),
        lr=t_cfg["learning_rate"],
        weight_decay=t_cfg["weight_decay"],
    )

    total_steps = t_cfg["epochs"] * len(train_loader)
    warmup_steps = t_cfg["lr_warmup_steps"]

    # ── Load checkpoint (resume) ────────────────────────────────────────────
    save_dir = Path(base_dir) / t_cfg["save_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_bleu = 0.0
    patience_count = 0

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model_state = ckpt.get("model_state", ckpt)
        model.load_state_dict(model_state, strict=False)
        optimizer.load_state_dict(ckpt.get("optimizer_state", {}))
        start_epoch = ckpt.get("epoch", 0)
        best_bleu   = ckpt.get("best_bleu", 0.0)
        print(f"Resumed from epoch {start_epoch}, best BLEU={best_bleu:.2f}")

    # ── Training Loop ───────────────────────────────────────────────────────
    global_step = start_epoch * len(train_loader)
    log_file = open(save_dir / "train_log.txt", "a")
    
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, t_cfg["epochs"]):
        model.train()
        epoch_loss, epoch_ce, epoch_ctc = 0, 0, 0
        t0 = time.time()

        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            accumulation_steps = t_cfg.get("accumulation_steps", 1)
            
            # LR warmup + cosine decay (update lr only on optimization step)
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                global_step += 1
                lr_now = warmup_cosine_schedule(
                    global_step, warmup_steps, total_steps // accumulation_steps,
                    min_lr=1e-6, max_lr=t_cfg["learning_rate"]
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_now

            with torch.cuda.amp.autocast():
                loss, ce, ctc = compute_loss(
                    model, batch, gloss_vocab, trans_vocab, cfg, device
                )
                loss_scaled = loss / accumulation_steps
                
            scaler.scale(loss_scaled).backward()
            
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), t_cfg["clip_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item()
            epoch_ce   += ce
            epoch_ctc  += ctc

            if (step + 1) % 20 == 0:
                print(
                    f"  Ep{epoch+1} [{step+1}/{len(train_loader)}] "
                    f"loss={loss.item():.4f} ce={ce:.4f} ctc={ctc:.4f} "
                    f"lr={lr_now:.2e}"
                )

        # ── End of epoch ────────────────────────────────────────────────────
        avg_train = epoch_loss / len(train_loader)
        val_loss, val_bleu, hyp_examples, ref_examples = evaluate(
            model, val_loader, gloss_vocab, trans_vocab, cfg, device
        )        
        elapsed = time.time() - t0

        log_line = (
            f"Epoch {epoch+1:03d} | "
            f"Train={avg_train:.4f} | "
            f"Val={val_loss:.4f} | "
            f"BLEU4={val_bleu:.2f} | "
            f"Time={elapsed:.0f}s"
        )
        print(log_line)
        log_file.write(log_line + "\n")
        log_file.flush()

        # In 3 vi du dich
        print("  [Examples]")
        for h, r in zip(hyp_examples[:3], ref_examples[:3]):
            print(f"    Ref: {' '.join(r)}")
            print(f"    Hyp: {' '.join(h)}")

        # Save best
        if val_bleu > best_bleu:
            best_bleu = val_bleu
            patience_count = 0
            ckpt_path = save_dir / "checkpoint_best.pt"
            torch.save({
                "epoch":          epoch + 1,
                "model_state":    model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_bleu":      best_bleu,
                "val_loss":       val_loss,
            }, ckpt_path)
            print(f"  ✓ Best model saved (BLEU={best_bleu:.2f})")
        else:
            patience_count += 1
            if patience_count >= t_cfg["patience"]:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Save periodic checkpoint
        if (epoch + 1) % t_cfg["save_every"] == 0:
            torch.save({
                "epoch":          epoch + 1,
                "model_state":    model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_bleu":      best_bleu,
            }, save_dir / f"checkpoint_ep{epoch+1:03d}.pt")

    log_file.close()
    print(f"\nTraining done! Best BLEU-4 = {best_bleu:.2f}")


if __name__ == "__main__":
    main()
