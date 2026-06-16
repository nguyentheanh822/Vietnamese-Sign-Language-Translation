"""
models/sota_model.py
--------------------
Spatio-Temporal Graph Convolutional Network + Transformer Architecture (ST-GCN-Transformer).
Replaces the standard Linear projection in SLTModel with an adaptive ST-GCN architecture.
"""

from .slt_model import SLTModel
from .stgcn import STGCN_Frontend

class SOTAModel(SLTModel):
    def __init__(
        self,
        input_dim: int,
        gloss_vocab_size: int,
        trans_vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pad_idx: int = 0,
        num_nodes: int = 137,
        coords: int = 6,
    ):
        # Initialize standard Transformer
        super().__init__(
            input_dim=input_dim,
            gloss_vocab_size=gloss_vocab_size,
            trans_vocab_size=trans_vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pad_idx=pad_idx,
        )
        
        # Override the linear input projection with ST-GCN Front-end
        self.input_proj = STGCN_Frontend(
            in_features=input_dim, 
            num_nodes=num_nodes, 
            coords=coords, 
            d_model=d_model, 
            dropout=dropout
        )
