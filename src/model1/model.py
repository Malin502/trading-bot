from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)

class Model1ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, width: int = 256, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.GELU(),
            nn.Dropout(dropout),
            *[ResidualMLPBlock(width, dropout) for _ in range(depth)],
            nn.LayerNorm(width),
        )
        self.head_ret = nn.Linear(width, 1)
        self.head_risk = nn.Linear(width, 1)

        # 初期化（軽く効く）
        nn.init.zeros_(self.head_ret.bias)
        nn.init.zeros_(self.head_risk.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        pred_ret = self.head_ret(h).squeeze(-1)    # (B,)
        pred_risk = F.softplus(self.head_risk(h).squeeze(-1))  # (B,), downside magnitude >= 0
        return pred_ret, pred_risk
