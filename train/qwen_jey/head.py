"""T8.1 决策头：M6 图 N——W_q/W_k 1024→128（带 bias），dot/√128 → softmax。

参数量 2×(1024×128＋128)＝262,400（表 M1 🔥 唯一新增件）。训推共用同一模块。
"""
from __future__ import annotations

import math

import torch
from torch import nn


class DecisionHead(nn.Module):
    def __init__(self, hidden: int = 1024, dim: int = 128):
        super().__init__()
        self.query = nn.Linear(hidden, dim, bias=True)
        self.key = nn.Linear(hidden, dim, bias=True)
        self.dim = dim

    def forward(self, q_hidden: torch.Tensor, k_hidden: torch.Tensor) -> torch.Tensor:
        """q_hidden [B,H]；k_hidden [B,N,H] → logits [B,N]（未 softmax）。"""
        u = self.query(q_hidden)                       # [B,d]
        v = self.key(k_hidden)                         # [B,N,d]
        return torch.einsum("bd,bnd->bn", u, v) / math.sqrt(self.dim)

    def probs(self, q_hidden, k_hidden) -> torch.Tensor:
        return torch.softmax(self.forward(q_hidden, k_hidden), dim=-1)
