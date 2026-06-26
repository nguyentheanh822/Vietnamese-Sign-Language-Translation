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

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNSLTModel(nn.Module):
    def __init__(self, vocab_size, num_encoder_layers=6, num_decoder_layers=6, d_model=512, nhead=8,
                 dim_feedforward=2048, dropout=0.1, input_dim=822, max_seq_len=300, gloss_vocab_size=None):
        super().__init__()
        # gloss_vocab_size for CTC head (235), vocab_size for translation (442)
        _gloss_size = gloss_vocab_size if gloss_vocab_size is not None else vocab_size
        
        # 1. Input projection
        self.is_multi_view = (input_dim == 1644)
        if self.is_multi_view:
            self.input_proj = nn.Sequential(
                nn.Linear(822, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model)
            )
            self.fusion_proj = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model)
            )
        
        # 2. TCN Encoder (6 levels, matching d_model)
        num_channels = [d_model] * num_encoder_layers
        self.encoder = TemporalConvNet(d_model, num_channels, kernel_size=3, dropout=dropout)
        
        # 3. Standard Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # 4. Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # 5. Gloss projection (for CTC/Frame-CE — must use gloss_vocab_size)
        self.gloss_proj = nn.Linear(d_model, _gloss_size)
        
        # 6. Target Embedding & Positional Encoding
        self.tgt_emb = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_len)

    def generate_square_subsequent_mask(self, sz: int):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_mask, tgt, tgt_mask=None):
        if self.is_multi_view:
            # src is (B, T, 1644). Extract Front (0-411, 822-1233) and Side (411-822, 1233-1644)
            src_f = torch.cat([src[:, :, :411], src[:, :, 822:1233]], dim=-1)
            src_s = torch.cat([src[:, :, 411:822], src[:, :, 1233:1644]], dim=-1)
            
            emb_f = self.input_proj(src_f)
            emb_s = self.input_proj(src_s)
            
            src_emb = self.fusion_proj(torch.cat([emb_f, emb_s], dim=-1)) # (B, T, d_model)
        else:
            src_emb = self.input_proj(src) # (B, T, d_model)
        
        # TCN expects (B, C, T)
        src_emb = src_emb.permute(0, 2, 1)
        memory = self.encoder(src_emb)
        memory = memory.permute(0, 2, 1) # Back to (B, T, d_model)
        
        gloss_logits = self.gloss_proj(memory) # (B, T, V)
        
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
        if self.is_multi_view:
            src_f = torch.cat([src[:, :, :411], src[:, :, 822:1233]], dim=-1)
            src_s = torch.cat([src[:, :, 411:822], src[:, :, 1233:1644]], dim=-1)
            emb_f = self.input_proj(src_f)
            emb_s = self.input_proj(src_s)
            src_emb = self.fusion_proj(torch.cat([emb_f, emb_s], dim=-1))
        else:
            src_emb = self.input_proj(src)
        src_emb = src_emb.permute(0, 2, 1)
        memory = self.encoder(src_emb)
        memory = memory.permute(0, 2, 1)

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
