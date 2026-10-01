"""Mixture-of-pathways NCA update rule. Plan §3.3.3, §9.2.5.2–9.2.5.4.

Adapted from V3's FusedAttentionUpdate. Computes M specialized pathway
updates with hardware-conditioned routing:

    α_m = softmax(W_gate · H / T_soft)_m           — routing weights
    Δ_m = W_m · P + b_m                             — per-mode update
    ΔG  = Σ_m α_m ⊙ Δ_m                             — weighted sum
    G^{s+1} = G^s + η_s · LayerNorm_C(ΔG)           — residual update

Pathway weights W_m and gate W_gate are SHARED across all T NCA steps
within a layer. Only step sizes η_s and LayerNorm parameters are per-step.

Weight initialization (plan §10):
    W_m:    Xavier uniform, gain = 1/√M
    b_m:    Zero
    W_gate: Zero (uniform routing α_m = 1/M at init)
    T_soft: 1.0
    η_s:    step_size_init (default 0.1)
    LN γ:   Ones
    LN β:   Zero
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dpnca.config import DPNCAConfig


class PathwayUpdate(nn.Module):
    """Mixture-of-pathways NCA update with per-step LayerNorm and step sizes.

    Plan §3.3.3: M pathway weights are shared across NCA steps. Each step s
    has its own learnable step size η_s and LayerNorm parameters (γ_s, β_s).
    """

    def __init__(self, config: DPNCAConfig) -> None:
        super().__init__()
        C = config.n_channels
        M = config.n_modes
        T = config.n_steps
        C_H = config.n_hw_channels
        P_dim = config.n_perception_channels  # (n_k + 1) * C

        self.n_modes = M
        self.n_steps = T
        self.n_channels = C

        # ── Pathway weights (SHARED across NCA steps) ──────────────────
        # W: (M, C, P_dim) — each mode projects perceived features to update
        self.W = nn.Parameter(torch.empty(M, C, P_dim))
        # b: (M, C) — per-mode bias
        self.b = nn.Parameter(torch.zeros(M, C))

        # Xavier uniform, gain = 1/√M — plan §10
        gain = 1.0 / math.sqrt(M)
        for m in range(M):
            nn.init.xavier_uniform_(self.W.data[m], gain=gain)

        # ── Gate projection (SHARED across NCA steps) ──────────────────
        # W_gate: (M, C_H) — projects hardware to routing logits
        self.W_gate = nn.Parameter(torch.zeros(M, C_H))  # zero init → uniform routing

        # Learnable softmax temperature — plan §3.3.3
        self.T_soft = nn.Parameter(torch.tensor(1.0))

        # ── Content-modulated routing (V5) ─────────────────────────────
        # When enabled, routing logits incorporate grid content alongside hardware
        if config.content_modulated_routing:
            # W_gate_content: (M, C) — projects row-averaged grid to routing logits
            self.W_gate_content = nn.Parameter(torch.zeros(M, C))
        else:
            self.W_gate_content = None

        # ── Per-step parameters (NOT shared) ───────────────────────────
        # Step sizes η_s — T scalars, one per NCA step
        self.step_sizes = nn.ParameterList([
            nn.Parameter(torch.tensor(config.step_size_init))
            for _ in range(T)
        ])

        # Per-step LayerNorm across C channels — T × (γ, β) ∈ R^C each
        self.step_norms = nn.ModuleList([
            nn.LayerNorm(C) for _ in range(T)
        ])

    def compute_routing(self, H: Tensor, G: Tensor | None = None) -> Tensor:
        """Compute pathway routing weights from hardware (and optionally grid content).

        V4: α = softmax(W_gate · H / T_soft)
        V5: α = softmax((W_gate · H + W_gate_content · mean_R(G)) / T_soft)

        Content modulation is zero-initialised so V5 starts with V4 behaviour
        and content influence is discovered during training.

        Args:
            H: (B, C_H, R, L) — hardware channels
            G: (B, C, R, L) — current grid state (optional, for content routing)

        Returns:
            alpha: (B, M, R, L) — routing weights summing to 1 over M
        """
        # W_gate: (M, C_H), H: (B, C_H, R, L) → logits: (B, M, R, L)
        logits = torch.einsum("mc, bcrL -> bmrL", self.W_gate, H)

        # Content-modulated routing (V5)
        if self.W_gate_content is not None and G is not None:
            # Row-average grid: (B, C, R, L) → (B, C, L) via mean over R
            G_bar = G.mean(dim=2)  # (B, C, L)
            # W_gate_content: (M, C), G_bar: (B, C, L) → (B, M, L)
            content_logits = torch.einsum("mc, bcL -> bmL", self.W_gate_content, G_bar)
            # Broadcast over R dimension
            logits = logits + content_logits.unsqueeze(2)  # (B, M, R, L)

        # Temperature-scaled softmax over modes
        alpha = F.softmax(logits / (self.T_soft + 1e-8), dim=1)
        return alpha

    def compute_delta(self, P: Tensor, alpha: Tensor) -> Tensor:
        """Compute weighted pathway update ΔG.

        Δ_m = W_m · P + b_m   for each mode m
        ΔG  = Σ_m α_m ⊙ Δ_m

        Args:
            P: (B, P_dim, R, L) — perceived features
            alpha: (B, M, R, L) — routing weights

        Returns:
            delta_G: (B, C, R, L) — weighted update
        """
        B, P_dim, R, L = P.shape

        # All modes at once: W (M, C, P_dim) × P (B, P_dim, R, L) → (B, M, C, R, L)
        P_flat = P.reshape(B, P_dim, R * L)  # (B, P_dim, R*L)
        # einsum: (M, C, P_dim) × (B, P_dim, N) → (B, M, C, N) where N = R*L
        delta_all = torch.einsum("mcp, bpn -> bmcn", self.W, P_flat)  # (B, M, C, R*L)
        delta_all = delta_all.reshape(B, self.n_modes, self.n_channels, R, L)
        # Add bias: (M, C) → (1, M, C, 1, 1)
        delta_all = delta_all + self.b[None, :, :, None, None]

        # Weighted sum over modes: α (B, M, R, L) → (B, M, 1, R, L)
        alpha_expanded = alpha.unsqueeze(2)  # (B, M, 1, R, L)
        delta_G = (alpha_expanded * delta_all).sum(dim=1)  # (B, C, R, L)

        return delta_G

    def apply_step(self, G: Tensor, delta_G: Tensor, step_idx: int) -> Tensor:
        """Apply one NCA step: G^{s+1} = G^s + η_s · LayerNorm_C(ΔG).

        LayerNorm is applied across the C channel dimension at each spatial
        position (b, r, l) independently.

        Args:
            G: (B, C, R, L) — current grid state
            delta_G: (B, C, R, L) — weighted update
            step_idx: NCA step index s ∈ {0, ..., T-1}

        Returns:
            G_next: (B, C, R, L) — updated grid state
        """
        B, C, R, L = delta_G.shape
        eta = self.step_sizes[step_idx]
        norm = self.step_norms[step_idx]

        # LayerNorm_C: normalize across C at each spatial position
        # Reshape to (B*R*L, C) for nn.LayerNorm, then reshape back
        delta_flat = delta_G.permute(0, 2, 3, 1).reshape(B * R * L, C)  # (N, C)
        delta_normed = norm(delta_flat)                                   # (N, C)
        delta_normed = delta_normed.reshape(B, R, L, C).permute(0, 3, 1, 2)  # (B, C, R, L)

        return G + eta * delta_normed
