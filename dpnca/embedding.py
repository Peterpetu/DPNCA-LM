"""Token embedding with Rotary Position Encoding (RoPE). Plan §9.2.1–9.2.3.

TokenEmbedding: vocab lookup with weight-tied output projection.
RotaryPositionalEncoding: RoPE applied per-cell after grid reshape.

RoPE rotates channel pairs at each column position l using θ_i = 10000^{-2i/C}.
All rows r at the same column l receive identical rotations (row is compute
dimension, not positional).

Implementation note — channel pairing convention:
    The plan §9.2.3 specifies INTERLEAVED pairing: channels (2i, 2i+1).
    This implementation uses the SPLIT-HALF convention: channels (i, i+C/2),
    following the LLaMA / Hugging Face standard. Both are mathematically
    equivalent orthogonal rotations preserving the key RoPE properties:
    (1) norm preservation, (2) relative position encoding via dot-product,
    (3) decaying influence with distance. The split-half form is preferred
    for computational efficiency (contiguous memory access).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from dpnca.config import DPNCAConfig


class RotaryPositionalEncoding(nn.Module):
    """Rotary Position Embedding applied to NCA grid channels. Plan §9.2.3.

    Pre-computes cos/sin tables for positions [0, max_seq_len) and caches them.
    Applied to channel pairs (2i, 2i+1) along the C dimension of the grid.
    """

    def __init__(self, n_channels: int, max_seq_len: int = 2048) -> None:
        super().__init__()
        assert n_channels % 2 == 0, f"n_channels must be even for RoPE, got {n_channels}"
        self.n_channels = n_channels

        # θ_i = 10000^{-2i/C} for i = 0, ..., C/2 - 1
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, n_channels, 2).float() / n_channels))
        self.register_buffer("inv_freq", inv_freq)  # (C/2,)

        # Pre-compute cos/sin tables
        self._build_cache(max_seq_len)

    def _build_cache(self, max_seq_len: int) -> None:
        """Build and cache cos/sin tables for positions [0, max_seq_len)."""
        positions = torch.arange(max_seq_len, dtype=self.inv_freq.dtype)  # (L,)
        # Outer product: (L,) × (C/2,) → (L, C/2)
        freqs = torch.outer(positions, self.inv_freq)
        # Duplicate for both sin and cos: (L, C) where C = 2 × C/2
        emb = torch.cat([freqs, freqs], dim=-1)  # (L, C)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)  # (L, C)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)  # (L, C)
        self._cached_len = max_seq_len

    def forward(self, grid: Tensor, seq_len: int) -> Tensor:
        """Apply RoPE to NCA grid.

        Args:
            grid: (B, C, R, L) — NCA grid in NCHW format
            seq_len: actual sequence length (L)

        Returns:
            grid with RoPE applied: (B, C, R, L)
        """
        if seq_len > self._cached_len:
            self._build_cache(seq_len)

        # cos/sin tables: (L, C) → reshape for broadcasting with (B, C, R, L)
        cos = self.cos_cached[:seq_len]  # (L, C)
        sin = self.sin_cached[:seq_len]  # (L, C)

        # Reshape to (1, C, 1, L) for broadcasting
        cos = cos.T.unsqueeze(0).unsqueeze(2)  # (1, C, 1, L)
        sin = sin.T.unsqueeze(0).unsqueeze(2)  # (1, C, 1, L)

        # Split into even/odd channel pairs and apply rotation
        # grid shape: (B, C, R, L)
        C = grid.shape[1]
        half_C = C // 2
        x_even = grid[:, :half_C, :, :]   # (B, C/2, R, L) — channels 0,1,...,C/2-1
        x_odd = grid[:, half_C:, :, :]     # (B, C/2, R, L) — channels C/2,...,C-1

        cos_half = cos[:, :half_C, :, :]   # (1, C/2, 1, L)
        sin_half = sin[:, :half_C, :, :]   # (1, C/2, 1, L)

        # Rotation: [cos -sin; sin cos] applied to (even, odd) pairs
        rotated_even = x_even * cos_half - x_odd * sin_half
        rotated_odd = x_even * sin_half + x_odd * cos_half

        return torch.cat([rotated_even, rotated_odd], dim=1)


class TokenEmbedding(nn.Module):
    """Token embedding with weight-tied output projection. Plan §9.2.1.

    Input:  token_ids (B, L) → embeddings (B, L, d_model)
    Output (tied): hidden (B, L, d_model) → logits (B, L, V) via E^T

    Weight initialization: N(0, 1/√d) per plan §10.
    """

    def __init__(self, config: DPNCAConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Initialize: N(0, 1/√d_model) — plan §10
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0 / math.sqrt(config.d_model))

    @property
    def weight(self) -> Tensor:
        """Embedding weight matrix E ∈ R^{V × d}, used for weight tying."""
        return self.embedding.weight

    def forward(self, token_ids: Tensor) -> Tensor:
        """Look up token embeddings.

        Args:
            token_ids: (B, L) integer token IDs ∈ {0, ..., V-1}

        Returns:
            embeddings: (B, L, d_model)
        """
        return self.embedding(token_ids)

    def project_to_logits(self, hidden: Tensor) -> Tensor:
        """Weight-tied output projection: logits = hidden @ E^T.

        Args:
            hidden: (B, L, d_model) — final hidden states

        Returns:
            logits: (B, L, V)
        """
        return hidden @ self.embedding.weight.T
