"""Causal Aggregation Channels (CAC). Plan V5 §2.

Multi-timescale exponential moving averages providing global context
summaries to both the model stream and hardware channels. Each scale k
maintains a running state s_l^{(k)} that integrates information at a
characteristic half-life.

Per-position update (K scales):
    s_l^{(k)} = (1 - β_k) · s_{l-1}^{(k)} + β_k · GELU(W_k · x_l + b_k)

Output:
    agg_output = W_out · [s_l^{(1)} ‖ ... ‖ s_l^{(K)}]      → model stream
    hw_injection = W_hw · [s_l^{(1)} ‖ ... ‖ s_l^{(K)}]      → hardware channels

Default half-lives: [4, 16, 64, 256, 1024] positions.

The gradient at distance D decays as (1-β_k)^D. Slow scales (large half-life,
small β) maintain gradient flow over very long distances — a natural
"gradient highway" without explicit skip connections.

Complexity: O(K · d_a · L) — negligible compared to perception/pathway costs.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dpnca.config import DPNCAConfig

# Default target half-lives for the K aggregation scales
_DEFAULT_HALF_LIVES = (4, 16, 64, 256, 1024)


class CausalAggregation(nn.Module):
    """Multi-timescale causal EMA with dual output to model and hardware.

    Maintains K exponential moving averages at geometrically spaced
    half-lives. Each scale projects input to d_a dimensions, applies GELU,
    and updates via EMA. The concatenated state is projected to both
    d_model (for residual fusion) and C_H (for hardware injection).

    Decay rates β_k = σ(γ_k) are learnable, initialised from target
    half-lives via β = ln(2) / t_{1/2}.
    """

    def __init__(self, config: DPNCAConfig) -> None:
        super().__init__()
        K = config.n_aggregation_scales
        d_a = config.d_aggregation
        d = config.d_model
        C_H = config.n_hw_channels

        self.n_scales = K
        self.d_aggregation = d_a

        # Per-scale input projection: x ∈ R^d → R^{d_a}
        self.projections = nn.ModuleList([
            nn.Linear(d, d_a) for _ in range(K)
        ])

        # Learnable decay rates (stored pre-sigmoid)
        # β_k = σ(γ_k), initialised so that half-life = t_{1/2,k}
        half_lives = _DEFAULT_HALF_LIVES[:K]
        target_betas = [math.log(2) / hl for hl in half_lives]
        init_logits = [math.log(b / (1.0 - b)) for b in target_betas]
        self.decay_logits = nn.Parameter(torch.tensor(init_logits))

        # Output projection to model dimension
        self.W_out = nn.Linear(K * d_a, d, bias=True)

        # Output projection to hardware channels
        self.W_hw = nn.Linear(K * d_a, C_H, bias=False)

        self._init_weights(config)

    def _init_weights(self, config: DPNCAConfig) -> None:
        """Xavier uniform init with residual output scaling."""
        for proj in self.projections:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)
        nn.init.xavier_uniform_(self.W_out.weight, gain=1.0 / math.sqrt(config.n_layers))
        nn.init.zeros_(self.W_out.bias)
        nn.init.xavier_uniform_(self.W_hw.weight)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Multi-scale EMA forward pass.

        Args:
            x: (B, L, d) — pre-normed input

        Returns:
            agg_output: (B, L, d) — aggregated context for residual fusion
            hw_injection: (B, L, C_H) — context summary for hardware channels
        """
        B, L, d = x.shape
        K = self.n_scales
        d_a = self.d_aggregation
        betas = torch.sigmoid(self.decay_logits)  # (K,)

        # Compute projected + activated inputs for all scales
        # Each: (B, L, d_a)
        projected = [F.gelu(proj(x)) for proj in self.projections]

        # Sequential EMA along L for each scale
        # Vectorized: s_l = β · Σ_{i=0}^{l} (1-β)^{l-i} · p_i
        # This is a causal weighted sum with decay matrix W[l,m] = β·(1-β)^{l-m}
        pos = torch.arange(L, device=x.device)
        rel = pos[:, None] - pos[None, :]  # (L, L)
        causal_mask = (rel >= 0)  # lower-triangular including diagonal

        all_states = []
        for k in range(K):
            beta_k = betas[k]
            alpha_k = 1.0 - beta_k  # decay factor
            p_k = projected[k]  # (B, L, d_a)

            # Build causal EMA weight matrix: W[l,m] = β · α^{l-m} for l ≥ m
            W_k = torch.where(
                causal_mask,
                beta_k * alpha_k.float() ** rel.clamp(min=0).float(),
                torch.zeros(1, device=x.device),
            )  # (L, L)

            # Apply: s = W_k @ p_k → (B, L, d_a)
            states_k = torch.matmul(W_k.unsqueeze(0), p_k)  # (1, L, L) @ (B, L, d_a)
            all_states.append(states_k)

        # Concatenate all scales: (B, L, K·d_a)
        agg_concat = torch.cat(all_states, dim=-1)

        # Dual output
        agg_output = self.W_out(agg_concat)     # (B, L, d)
        hw_injection = self.W_hw(agg_concat)    # (B, L, C_H)

        return agg_output, hw_injection
