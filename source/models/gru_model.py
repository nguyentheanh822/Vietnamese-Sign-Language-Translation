"""
models/gru_model.py
-------------------
Baseline GRU-based Seq2Seq model for Sign Language Translation.
Includes Attention mechanism and CTC auxiliary loss for glosses.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(
            hidden_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
    def forward(self, src, src_mask=None):
        # src: (B, T, input_dim)
        x = self.input_proj(src) # (B, T, H)
        outputs, hidden = self.gru(x)
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs, mask=None):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # repeat hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1) # (B, T, H)
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2))) # (B, T, H)
        attention = self.v(energy).squeeze(2) # (B, T)
        
        if mask is not None:
            attention = attention.masked_fill(mask == True, -1e10)
            
        return F.softmax(attention, dim=1)

class GRUDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, num_layers=2, dropout=0.1, pad_idx=0):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)
        self.attention = Attention(hidden_dim)
        
        self.gru = nn.GRU(
            hidden_dim + emb_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc_out = nn.Linear(hidden_dim * 2 + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs, mask=None):
        input = input.unsqueeze(1) # (B, 1)
        embedded = self.dropout(self.embedding(input)) # (B, 1, E)
        
        # Use top layer hidden state for attention
        a = self.attention(hidden[-1], encoder_outputs, mask) # (B, T)
        a = a.unsqueeze(1) # (B, 1, T)
        
        weighted = torch.bmm(a, encoder_outputs) # (B, 1, H)
        
        rnn_input = torch.cat((embedded, weighted), dim=2) # (B, 1, E + H)
        
        output, hidden = self.gru(rnn_input, hidden)
        
        embedded = embedded.squeeze(1) # (B, E)
        output = output.squeeze(1) # (B, H)
        weighted = weighted.squeeze(1) # (B, H)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        
        return prediction, hidden

class GRUSLTModel(nn.Module):
    """
    GRU-based Seq2Seq model for Sign Language Translation
    with CTC auxiliary loss for gloss prediction.
    """
    def __init__(
        self,
        input_dim: int,
        gloss_vocab_size: int,
        trans_vocab_size: int,
        hidden_dim: int = 512,
        emb_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        
        self.encoder = GRUEncoder(input_dim, hidden_dim, num_layers, dropout)
        self.decoder = GRUDecoder(trans_vocab_size, emb_dim, hidden_dim, num_layers, dropout, pad_idx)
        
        # CTC Head for auxiliary loss
        self.ctc_head = nn.Linear(hidden_dim, gloss_vocab_size)
        self.pad_idx = pad_idx
        
    def forward(self, src, src_mask, tgt, tgt_mask=None, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        trans_vocab_size = self.decoder.output_dim
        
        trans_logits = torch.zeros(batch_size, tgt_len, trans_vocab_size).to(src.device)
        
        encoder_outputs, hidden = self.encoder(src, src_mask)
        
        ctc_logits = self.ctc_head(encoder_outputs)
        ctc_log_probs = F.log_softmax(ctc_logits, dim=-1)
        ctc_log_probs = ctc_log_probs.permute(1, 0, 2)
        
        input = tgt[:, 0]
        
        for t in range(1, tgt_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs, src_mask)
            trans_logits[:, t, :] = output
            
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = tgt[:, t] if teacher_force else top1
            
        return trans_logits, ctc_log_probs
