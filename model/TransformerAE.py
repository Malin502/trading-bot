from __future__ import annotations

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (L, D)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (L,1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1,L,D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        t = x.size(1)
        return x + self.pe[:, :t, :]


class TransformerAE(nn.Module):
    """
    Input:  x (B, T, F)
    Output: recon (B, T, F), latent (B, Z)
    """
    def __init__(
        self,
        feature_dim: int,
        seq_len: int = 56,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        latent_dim: int = 32,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.d_model = d_model

        self.in_proj = nn.Linear(feature_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max(256, seq_len + 1))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # (B,T,D) -> mean over T -> (B,D) -> (B,Z)
        self.to_latent = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, latent_dim),
        )

        # Decoder: use proper TransformerDecoder with cross-attention
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        # Learnable queries for each time step (tgt tokens)
        self.tgt_queries = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.tgt_queries, mean=0.0, std=0.02)

        # Optionally inject z into decoding as an extra memory token
        self.z_to_mem = nn.Linear(latent_dim, d_model)

        self.out_proj = nn.Linear(d_model, feature_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,F)
        h = self.in_proj(x)          # (B,T,D)
        h = self.pos_enc(h)
        memory = self.encoder(h)     # (B,T,D)

        # Pool over time to produce z
        pooled = memory.mean(dim=1)  # (B,D)
        z = self.to_latent(pooled)   # (B,Z)
        return z

    def forward(self, x: torch.Tensor):
        # Encode
        h = self.in_proj(x)
        h = self.pos_enc(h)
        memory = self.encoder(h)         # (B,T,D)

        pooled = memory.mean(dim=1)      # (B,D)  ※時間方向に平均
        z = self.to_latent(pooled)       # (B,Z)

        # Inject z as an extra memory token (helps decoder conditioning)
        z_mem = self.z_to_mem(z).unsqueeze(1)     # (B,1,D)
        memory2 = torch.cat([z_mem, memory], dim=1)  # (B,1+T,D)

        # Decode: tgt is learnable queries + positional encoding
        tgt = self.tgt_queries.repeat(x.size(0), 1, 1)  # (B,T,D)
        tgt = self.pos_enc(tgt)

        dec = self.decoder(tgt=tgt, memory=memory2)  # (B,T,D) cross-attn included
        recon = self.out_proj(dec)                   # (B,T,F)

        return recon, z