"""Hebbian Associative Memory (HAM). Plan V5 §1, DVTM Spec §4.4.

Content-addressable cross-position memory using outer-product accumulation
with learnable decay — mathematically equivalent to causal linear attention
with ELU+1 feature map, but derived from Hebbian plasticity principles.

Memory update (per head h, position l):
    M_l^{(h)} = λ^{(h)} · M_{l-1}^{(h)} + φ(k_l^{(h)}) · φ(v_l^{(h)})^T
    z_l^{(h)} = λ^{(h)} · z_{l-1}^{(h)} + φ(k_l^{(h)})

Retrieval:
    r_l^{(h)} = M_l^{(h)} · φ(q_l^{(h)}) / (z_l^{(h)}^T · φ(q_l^{(h)}) + ε)

Output:
    HAM(x_l) = W_O · concat(r_l^{(1)}, ..., r_l^{(H)})

Feature map: φ(z) = ELU(z) + 1 (strictly positive, Lipschitz continuous)

Complexity: O(d² · L) total — linear in sequence length.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dpnca.config import DPNCAConfig

# Numerical stability constants (Spec §14.5)
_EPS = 1e-6
_MEMORY_NORM_THRESHOLD = 1000.0


def _elu_plus_one(x: Tensor) -> Tensor:
    """Feature map φ(z) = ELU(z) + 1, floored at ε for strict positivity."""
    return F.elu(x, alpha=1.0) + 1.0 + _EPS


class HebbianAssociativeMemory(nn.Module):
    """Multi-head Hebbian associative memory with learnable decay.

    Implements content-addressable retrieval via outer-product accumulation
    of key-value associations, decayed by a per-head learnable factor.
    This is the NCA-native equivalent of causal linear attention, derived
    from the Hebbian concentration update in DVTM Spec §4.4.

    The decay λ^{(h)} = σ(α^{(h)}) is parameterised via sigmoid to
    ensure λ ∈ (0, 1). Initialised to 0.99 (half-life ≈ 69 positions).
    """

    def __init__(self, config: DPNCAConfig) -> None:
        super().__init__()
        self.n_heads = config.n_memory_heads
        self.d_head = config.d_memory_head_effective
        self.d_model = config.d_model
        d_inner = self.n_heads * self.d_head

        # Q, K, V projections — shared gain = 1/√3 (attention convention)
        self.W_Q = nn.Linear(self.d_model, d_inner, bias=False)
        self.W_K = nn.Linear(self.d_model, d_inner, bias=False)
        self.W_V = nn.Linear(self.d_model, d_inner, bias=False)

        # Output projection — gain = 1/√n_layers (residual scaling)
        self.W_O = nn.Linear(d_inner, self.d_model, bias=False)

        # Per-head learnable decay: α stored pre-sigmoid
        # sigmoid^{-1}(0.99) ≈ 4.595
        decay_init = math.log(config.memory_decay_init / (1.0 - config.memory_decay_init))
        self.decay_logit = nn.Parameter(
            torch.full((self.n_heads,), decay_init)
        )

        self._init_weights(config)

    def _init_weights(self, config: DPNCAConfig) -> None:
        """Xavier uniform init with prescribed gains."""
        gain_qkv = 1.0 / math.sqrt(3)
        nn.init.xavier_uniform_(self.W_Q.weight, gain=gain_qkv)
        nn.init.xavier_uniform_(self.W_K.weight, gain=gain_qkv)
        nn.init.xavier_uniform_(self.W_V.weight, gain=gain_qkv)

        gain_o = 1.0 / math.sqrt(config.n_layers)
        nn.init.xavier_uniform_(self.W_O.weight, gain=gain_o)

    def forward(self, x: Tensor) -> Tensor:
        """Hebbian memory forward pass: project → accumulate → retrieve → output.

        Vectorized implementation: the sequential recurrence
            M_l = λ M_{l-1} + φ(k_l)⊗v_l,  r_l = M_l φ(q_l) / (z_l^T φ(q_l) + ε)
        is equivalent to causal linear attention with exponential decay weights:
            S[l,m] = λ^{l-m} · φ(q_l)^T φ(k_m)  for m ≤ l
            r_l = Σ_m S[l,m] v_m  /  Σ_m S[l,m]
        This avoids O(L) sequential Python iterations; uses O(L²) memory which
        is acceptable for L ≤ 2048.

        Args:
            x: (B, L, d) — pre-normed input

        Returns:
            output: (B, L, d) — memory retrieval projected back to model dim
        """
        B, L, _ = x.shape
        H = self.n_heads
        d_h = self.d_head

        # Project to multi-head Q, K, V
        q = self.W_Q(x).view(B, L, H, d_h)  # (B, L, H, d_h)
        k = self.W_K(x).view(B, L, H, d_h)
        v = self.W_V(x).view(B, L, H, d_h)

        # Apply feature map to Q and K (not V — values are raw)
        q_phi = _elu_plus_one(q)  # (B, L, H, d_h)
        k_phi = _elu_plus_one(k)

        # Per-head decay factor λ ∈ (0, 1)
        decay = torch.sigmoid(self.decay_logit)  # (H,)

        # Build causal decay matrix D[l,m] = λ^{l-m} for l ≥ m, else 0
        # Computation done in float32 for numerical stability of decay powers
        pos = torch.arange(L, device=x.device)
        rel = pos[:, None] - pos[None, :]  # (L, L), rel[l,m] = l - m
        causal_mask = (rel >= 0)  # lower-triangular including diagonal

        # D: (H, L, L) — per-head decay weights
        # decay[:, None, None] ** rel.clamp(min=0) computes λ^|l-m| for each head
        D = torch.where(
            causal_mask[None, :, :],
            decay[:, None, None].float() ** rel[None, :, :].clamp(min=0).float(),
            torch.zeros(1, device=x.device),
        )  # (H, L, L)

        # Compute attention scores:
        # S[b,h,l,m] = φ(q_l)^T φ(k_m) · D[h, l, m]
        # q_phi: (B, L, H, d_h) → permute to (B, H, L, d_h)
        q_t = q_phi.permute(0, 2, 1, 3)  # (B, H, L, d_h)
        k_t = k_phi.permute(0, 2, 1, 3)  # (B, H, L, d_h)
        v_t = v.permute(0, 2, 1, 3)      # (B, H, L, d_h)

        # (B, H, L, d_h) @ (B, H, d_h, L) → (B, H, L, L)
        S = torch.matmul(q_t, k_t.transpose(-2, -1)) * D[None, :, :, :]

        # Normalise per position: denominator = sum over source positions
        denom = S.sum(dim=-1, keepdim=True) + _EPS  # (B, H, L, 1)

        # Retrieve: output = (S / denom) @ V
        output = torch.matmul(S / denom, v_t)  # (B, H, L, d_h)

        # Reshape to (B, L, H*d_h) and project
        output = output.permute(0, 2, 1, 3).reshape(B, L, H * d_h)
        return self.W_O(output)
