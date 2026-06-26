import torch
import torch.nn as nn
from torchaudio.models import Conformer
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class ConformerSLTModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, input_dim=822, max_seq_len=300):
        super().__init__()
        
        # 1. Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # 2. Conformer Encoder
        # Note: torchaudio Conformer expects (B, T, d_model) and lengths
        self.encoder = Conformer(
            input_dim=d_model,
            num_heads=nhead,
            ffn_dim=dim_feedforward,
            num_layers=num_encoder_layers,
            depthwise_conv_kernel_size=31,
            dropout=dropout
        )
        
        # 3. Standard Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # 4. Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # 5. Gloss projection (for CTC / frame-level CE)
        self.gloss_proj = nn.Linear(d_model, vocab_size)
        
        # 6. Target Embedding & Positional Encoding
        self.tgt_emb = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_len)

    def generate_square_subsequent_mask(self, sz: int):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_mask, tgt, tgt_mask=None):
        # src: (B, T, 822)
        # src_mask: (B, T) boolean mask where True means padding
        # tgt: (B, L)
        # tgt_mask: (B, L) padding mask
        
        # Encoder phase
        src_emb = self.input_proj(src) # (B, T, d_model)
        
        B, T, _ = src_emb.shape
        if src_mask is not None:
            lengths = (~src_mask).sum(dim=1).to(src_emb.device).int()
        else:
            lengths = torch.full((B,), T, dtype=torch.int32, device=src_emb.device)
            
        memory, _ = self.encoder(src_emb, lengths) # memory: (B, T, d_model)
        
        gloss_logits = self.gloss_proj(memory) # (B, T, V)
        
        # Decoder phase
        tgt_emb = self.pos_encoder(self.tgt_emb(tgt)) # (B, L, d_model)
        
        tgt_causal_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        outs = self.decoder(
            tgt_emb, memory,
            tgt_mask=tgt_causal_mask,
            memory_mask=None,
            tgt_key_padding_mask=tgt_mask,
            memory_key_padding_mask=src_mask
        ) # (B, L, d_model)
        
        translation_logits = self.output_proj(outs) # (B, L, V)
        
        ctc_log_probs = torch.nn.functional.log_softmax(gloss_logits, dim=-1)
        ctc_log_probs = ctc_log_probs.permute(1, 0, 2).float() # (T, B, V)
        
        return translation_logits, ctc_log_probs

    @torch.no_grad()
    def greedy_decode(self, src, src_mask, bos_idx, eos_idx, max_len=50):
        src_emb = self.input_proj(src)
        B, T, _ = src_emb.shape
        if src_mask is not None:
            lengths = (~src_mask).sum(dim=1).to(src_emb.device).int()
        else:
            lengths = torch.full((B,), T, dtype=torch.int32, device=src_emb.device)
        memory, _ = self.encoder(src_emb, lengths)

        ys = torch.full((B, 1), bos_idx, dtype=torch.long, device=src.device)
        finished = torch.zeros(B, dtype=torch.bool, device=src.device)

        for _ in range(max_len):
            tgt_emb = self.pos_encoder(self.tgt_emb(ys))
            tgt_causal_mask = self.generate_square_subsequent_mask(ys.size(1)).to(ys.device)
            outs = self.decoder(
                tgt_emb, memory,
                tgt_mask=tgt_causal_mask,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=src_mask
            )
            next_logits = self.output_proj(outs[:, -1])
            next_token = next_logits.argmax(-1)
            next_token = torch.where(finished, torch.full_like(next_token, eos_idx), next_token)
            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
            finished = finished | (next_token == eos_idx)
            if finished.all():
                break
        return ys[:, 1:]
