"""
models/bilstm_model.py
----------------------
Baseline BiLSTM-based Seq2Seq model for Sign Language Translation.
Includes Attention mechanism and CTC auxiliary loss for glosses.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim // 2, # bidirectional, so // 2
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
    def forward(self, src, src_mask=None):
        x = self.input_proj(src) # (B, T, H)
        outputs, (hidden, cell) = self.lstm(x)
        # outputs: (B, T, H)
        # hidden: (num_layers*2, B, H//2)
        
        # Combine bidirectional states for decoder
        # hidden is [forward_1, backward_1, forward_2, backward_2]
        # we need to concatenate forward and backward for each layer
        num_layers = self.lstm.num_layers
        batch_size = src.shape[0]
        hidden_dim = self.lstm.hidden_size * 2
        
        h_comb = hidden.view(num_layers, 2, batch_size, -1)
        h_comb = torch.cat((h_comb[:, 0, :, :], h_comb[:, 1, :, :]), dim=2)
        
        c_comb = cell.view(num_layers, 2, batch_size, -1)
        c_comb = torch.cat((c_comb[:, 0, :, :], c_comb[:, 1, :, :]), dim=2)
        
        return outputs, (h_comb, c_comb)

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs, mask=None):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        
        if mask is not None:
            attention = attention.masked_fill(mask == True, -1e4)
            
        return F.softmax(attention, dim=1)

class LSTMDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, num_layers=2, dropout=0.1, pad_idx=0):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)
        self.attention = Attention(hidden_dim)
        
        self.lstm = nn.LSTM(
            hidden_dim + emb_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc_out = nn.Linear(hidden_dim * 2 + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell, encoder_outputs, mask=None):
        input = input.unsqueeze(1) # (B, 1)
        embedded = self.dropout(self.embedding(input)) # (B, 1, E)
        
        a = self.attention(hidden[-1], encoder_outputs, mask) # (B, T)
        a = a.unsqueeze(1) # (B, 1, T)
        
        weighted = torch.bmm(a, encoder_outputs) # (B, 1, H)
        
        rnn_input = torch.cat((embedded, weighted), dim=2) # (B, 1, E + H)
        
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        
        embedded = embedded.squeeze(1) # (B, E)
        output = output.squeeze(1) # (B, H)
        weighted = weighted.squeeze(1) # (B, H)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        
        return prediction, hidden, cell

class BiLSTMSLTModel(nn.Module):
    """
    BiLSTM-based Seq2Seq model for Sign Language Translation
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
        
        self.encoder = BiLSTMEncoder(input_dim, hidden_dim, num_layers, dropout)
        self.decoder = LSTMDecoder(trans_vocab_size, emb_dim, hidden_dim, num_layers, dropout, pad_idx)
        
        self.ctc_head = nn.Linear(hidden_dim, gloss_vocab_size)
        self.pad_idx = pad_idx
        
    def forward(self, src, src_mask, tgt, tgt_mask=None, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        trans_vocab_size = self.decoder.output_dim
        
        trans_logits = torch.zeros(batch_size, tgt_len, trans_vocab_size).to(src.device)
        
        encoder_outputs, (hidden, cell) = self.encoder(src, src_mask)
        
        ctc_logits = self.ctc_head(encoder_outputs)
        ctc_log_probs = F.log_softmax(ctc_logits, dim=-1)
        ctc_log_probs = ctc_log_probs.permute(1, 0, 2)
        
        input = tgt[:, 0]
        
        for t in range(1, tgt_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs, src_mask)
            trans_logits[:, t, :] = output
            
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = tgt[:, t] if teacher_force else top1
            
        return trans_logits, ctc_log_probs

    @torch.no_grad()
    def greedy_decode(self, src, src_mask, bos_idx, eos_idx, max_len=50):
        batch_size = src.shape[0]
        device = src.device
        
        encoder_outputs, (hidden, cell) = self.encoder(src, src_mask)
        
        ys = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        input = ys[:, 0]
        
        for _ in range(max_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs, src_mask)
            next_token = output.argmax(1)
            
            next_token = torch.where(finished, torch.full_like(next_token, eos_idx), next_token)
            
            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
            finished = finished | (next_token == eos_idx)
            if finished.all():
                break
                
            input = next_token
            
        return ys[:, 1:]
