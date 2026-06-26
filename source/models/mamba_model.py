import torch
import torch.nn as nn
import torch.nn.functional as F
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

class SimplifiedMambaBlock(nn.Module):
    """
    A simplified, pure-PyTorch implementation of a Mamba/SSM block.
    Uses Pre-LayerNorm and clamped state values for training stability.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        
        self.norm = nn.LayerNorm(d_model)
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        
        # Initialize A_log with smaller values for stability
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stability."""
        nn.init.xavier_uniform_(self.in_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.x_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
        # Initialize dt_proj bias to make initial dt small
        with torch.no_grad():
            self.dt_proj.bias.fill_(-4.0)  # softplus(-4) ≈ 0.018, keeps dt small

    def forward(self, hidden_states):
        B, L, _ = hidden_states.shape
        
        # Pre-LayerNorm
        x_normed = self.norm(hidden_states)
        
        xz = self.in_proj(x_normed)
        x, z = xz.chunk(2, dim=-1)
        
        # Conv step
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)
        x = F.silu(x)
        
        # SSM step
        x_dbl = self.x_proj(x)
        delta, B_mat, C_mat = torch.split(x_dbl, [1, self.d_state, self.d_state], dim=-1)
        
        delta = F.softplus(self.dt_proj(delta))
        # Clamp delta to prevent explosion
        delta = delta.clamp(max=1.0)
        
        A = -torch.exp(self.A_log.float())
        
        # Emulate the SSM scan with clamped state
        y = torch.zeros_like(x)
        state = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        
        for t in range(L):
            dt = delta[:, t].unsqueeze(-1)
            dA = torch.exp(dt * A)
            dB = dt * B_mat[:, t].unsqueeze(1)
            
            state = dA * state + dB * x[:, t].unsqueeze(-1)
            # Clamp state to prevent numerical overflow
            state = state.clamp(-10.0, 10.0)
            y[:, t] = (state * C_mat[:, t].unsqueeze(1)).sum(dim=-1)
            
        y = y + x * self.D
        y = y * F.silu(z)
        out = self.out_proj(y)
        out = self.dropout(out)
        
        return out

class MambaSLTModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, input_dim=822, max_seq_len=300, gloss_vocab_size=None):
        super().__init__()
        # gloss_vocab_size for CTC head (235), vocab_size for translation (442)
        _gloss_size = gloss_vocab_size if gloss_vocab_size is not None else vocab_size
        
        # 1. Input projection (with LayerNorm for stability)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        
        # 2. Mamba Encoder Blocks (Pre-LN residual)
        self.encoder_blocks = nn.ModuleList([
            SimplifiedMambaBlock(d_model=d_model, dropout=dropout) for _ in range(num_encoder_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model)
        
        # 3. Standard Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # 4. Output projections
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.gloss_proj = nn.Linear(d_model, _gloss_size)  # CTC/Frame-CE must use gloss_vocab_size
        
        # 5. Target Embedding
        self.tgt_emb = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_len)

    def generate_square_subsequent_mask(self, sz: int):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_mask, tgt, tgt_mask=None):
        memory = self.input_proj(src)
        
        for block in self.encoder_blocks:
            memory = memory + block(memory)  # Residual connection (block has pre-norm inside)
        memory = self.norm_f(memory)
        
        gloss_logits = self.gloss_proj(memory)
        
        tgt_emb = self.pos_encoder(self.tgt_emb(tgt))
        tgt_causal_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        outs = self.decoder(
            tgt_emb, memory,
            tgt_mask=tgt_causal_mask,
            memory_mask=None,
            tgt_key_padding_mask=tgt_mask,
            memory_key_padding_mask=src_mask
        )
        
        translation_logits = self.output_proj(outs)
        
        ctc_log_probs = torch.nn.functional.log_softmax(gloss_logits, dim=-1)
        ctc_log_probs = ctc_log_probs.permute(1, 0, 2).float()
        
        return translation_logits, ctc_log_probs

    @torch.no_grad()
    def greedy_decode(self, src, src_mask, bos_idx, eos_idx, max_len=50):
        memory = self.input_proj(src)
        for block in self.encoder_blocks:
            memory = memory + block(memory)
        memory = self.norm_f(memory)

        B = src.size(0)
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
