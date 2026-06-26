import torch
import torch.nn as nn
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

class STTransformerSLTModel(nn.Module):
    """
    Spatio-Temporal Transformer:
    1. Spatial Transformer: Attention across 137 joints for each frame.
    2. Temporal Transformer: Attention across T frames.
    """
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=4, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, num_nodes=137, coords=6, max_seq_len=300):
        super().__init__()
        self.num_nodes = num_nodes
        self.coords = coords
        
        # Spatial Projection
        self.spatial_proj = nn.Linear(coords, d_model)
        self.spatial_pos_emb = nn.Parameter(torch.randn(1, num_nodes, d_model))
        
        spatial_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.spatial_encoder = nn.TransformerEncoder(spatial_layer, num_layers=2)
        
        # Temporal Encoder
        self.temporal_pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_len)
        temporal_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=num_encoder_layers)
        
        # Standard Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Projections
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.gloss_proj = nn.Linear(d_model, vocab_size)
        self.tgt_emb = nn.Embedding(vocab_size, d_model)

    def generate_square_subsequent_mask(self, sz: int):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_mask, tgt, tgt_mask=None):
        # src: (B, T, 822)
        # src_mask: (B, T)
        # tgt: (B, L)
        
        B, T, _ = src.shape
        # Reshape to (B*T, V, C) for Spatial Transformer
        x = src.view(B*T, self.num_nodes, self.coords)
        x = self.spatial_proj(x) # (B*T, V, d_model)
        x = x + self.spatial_pos_emb
        
        x = self.spatial_encoder(x) # (B*T, V, d_model)
        
        # Average pooling across joints
        x = x.mean(dim=1) # (B*T, d_model)
        
        # Reshape back to (B, T, d_model)
        x = x.view(B, T, -1)
        
        # Temporal Transformer
        x = self.temporal_pos_encoder(x)
        memory = self.temporal_encoder(x, src_key_padding_mask=src_mask)
        
        gloss_logits = self.gloss_proj(memory) # (B, T, V)
        
        # Decoder phase
        tgt_emb = self.temporal_pos_encoder(self.tgt_emb(tgt)) # (B, L, d_model)
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
        B, T, _ = src.shape
        x = src.view(B*T, self.num_nodes, self.coords)
        x = self.spatial_proj(x)
        x = x + self.spatial_pos_emb
        x = self.spatial_encoder(x)
        x = x.mean(dim=1)
        x = x.view(B, T, -1)
        x = self.temporal_pos_encoder(x)
        memory = self.temporal_encoder(x, src_key_padding_mask=src_mask)

        ys = torch.full((B, 1), bos_idx, dtype=torch.long, device=src.device)
        finished = torch.zeros(B, dtype=torch.bool, device=src.device)

        for _ in range(max_len):
            tgt_emb = self.temporal_pos_encoder(self.tgt_emb(ys))
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
