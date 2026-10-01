"""SwiGLU Feed-Forward Network. Plan §3.5, §9.2.8.

SwiGLU(x) = (W_1 · x ⊙ SiLU(W_3 · x)) · W_2

where:
    W_1: (d_ff, d_model) — gate linear
    W_3: (d_ff, d_model) — up-projection
    W_2: (d_model, d_ff) — down-projection
    d_ff = ⌈(8/3) · d_model⌉ rounded to nearest 8

The 8/3 ratio compensates for the gating halving effective width vs
standard 4d expansion.

Weight initialization (plan §10):
    W_1, W_3: Xavier uniform, gain = 1/√2 (two paths multiply)
    W_2:      Xavier uniform
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dpnca.config import DPNCAConfig


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network. Plan §3.5.

    FFN(x) = (W_1·x ⊙ SiLU(W_3·x)) · W_2
    Dropout applied after down-projection (before residual add).
    """

    def __init__(self, config: DPNCAConfig) -> None:
        super().__init__()
        d = config.d_model
        d_ff = config.d_ff

        # Gate projection
        self.w1 = nn.Linear(d, d_ff, bias=False)
        # Up-projection (through SiLU activation)
        self.w3 = nn.Linear(d, d_ff, bias=False)
        # Down-projection
        self.w2 = nn.Linear(d_ff, d, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        # Initialization — plan §10
        gain_gate = 1.0 / math.sqrt(2.0)
        nn.init.xavier_uniform_(self.w1.weight, gain=gain_gate)
        nn.init.xavier_uniform_(self.w3.weight, gain=gain_gate)
        nn.init.xavier_uniform_(self.w2.weight)

    def forward(self, x: Tensor) -> Tensor:
        """Apply SwiGLU FFN.

        Args:
            x: (B, L, d_model)

        Returns:
            (B, L, d_model)
        """
        gate = self.w1(x)                    # (B, L, d_ff)
        up = F.silu(self.w3(x))              # (B, L, d_ff)
        hidden = gate * up                   # (B, L, d_ff) — element-wise product
        out = self.w2(hidden)                # (B, L, d_model)
        return self.dropout(out)
