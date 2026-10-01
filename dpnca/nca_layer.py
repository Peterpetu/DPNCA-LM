"""NCA Layer: single NCA block + FFN with pre-norm residuals. Plan §3.4, §4 Phase 4.

NCABlock: Reshape → RoPE → T causal NCA steps → Hardware modulation → Flatten
NCATransformerLayer: PreNorm → NCABlock + HAM + CAC (gated fusion) → Residual → PreNorm → FFN → Residual

The NCA block is where the core NCA dynamics happen:
    1. Reshape x ∈ R^{B×L×d} → G ∈ R^{B×C×R×L}
    2. Apply RoPE to encode positional information
    3. Initialize hardware H from learned parameters (+ optional hw_injection)
    4. For s = 0..T-1:
       a. P = Perceive(G)                  — multi-scale causal perception
       b. α = routing(H, G)                — hardware + content routing (V5)
       c. ΔG = Σ_m α_m · (W_m · P + b_m)  — weighted pathway computation
       d. G = G + η_s · LN_C(ΔG)           — residual update with per-step LN
       e. If (s+1) % K_mod == 0: modulate hardware
    5. Flatten G back to R^{B×L×d}

V5 additions:
    - HebbianAssociativeMemory (HAM): parallel branch for content-addressable retrieval
    - CausalAggregation (CAC): parallel branch for multi-timescale context
    - Gated residual fusion: x + g_nca·NCA + g_mem·HAM + g_agg·CAC
    - Hardware injection from previous layer's CAC output
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from dpnca.config import DPNCAConfig
from dpnca.embedding import RotaryPositionalEncoding
from dpnca.ffn import SwiGLU
from dpnca.hardware import HardwareChannels
from dpnca.pathway_update import PathwayUpdate
from dpnca.perception import CausalPerception


class NCABlock(nn.Module):
    """Single NCA block: perception → pathway update → hardware modulation.

    Operates in NCHW format internally. Reshapes between (B,L,d) and (B,C,R,L).
    Each block has its own perception, pathway update, and hardware channels.

    When dilation is enabled (config.enable_dilation=True), each NCA step s
    uses a different dilation rate d(ℓ, s) computed from the layer index ℓ
    and step index s. This implements WaveNet-style exponential dilation
    within the NCA iteration loop, expanding the receptive field without
    adding any parameters.
    """

    def __init__(self, config: DPNCAConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_channels = config.n_channels       # C
        self.n_compute_rows = config.n_compute_rows  # R
        self.n_steps = config.n_steps              # T
        self.hw_mod_interval = config.hw_mod_interval

        self.perception = CausalPerception(config)
        self.pathway_update = PathwayUpdate(config)
        self.hardware = HardwareChannels(config)
        self.rope = RotaryPositionalEncoding(config.n_channels, config.max_seq_len)

        # State dropout (inverted) — plan §9.7
        self.state_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

    def _reshape_to_grid(self, x: Tensor) -> Tensor:
        """Reshape (B, L, d) → (B, C, R, L). Plan §9.2.2.

        x_{b,l,r·C+c} = G_{b,c,r,l}
        """
        B, L, d = x.shape
        # (B, L, d) → (B, L, R, C) → (B, R, L, C) → (B, C, R, L)
        G = x.view(B, L, self.n_compute_rows, self.n_channels)
        G = G.permute(0, 3, 1, 2)    # (B, C, L, R)
        G = G.permute(0, 1, 3, 2)    # (B, C, R, L)
        return G

    def _flatten_from_grid(self, G: Tensor) -> Tensor:
        """Flatten (B, C, R, L) → (B, L, d). Plan §9.2.6.

        Inverse of _reshape_to_grid: G_{b,c,r,l} → x_{b,l,r·C+c}
        """
        B, C, R, L = G.shape
        # (B, C, R, L) → (B, C, L, R) → (B, L, C, R) → (B, L, R, C) → (B, L, d)
        x = G.permute(0, 3, 2, 1)    # (B, L, R, C)
        x = x.reshape(B, L, R * C)
        return x

    def forward(self, x: Tensor, hw_injection: Tensor | None = None) -> Tensor:
        """NCA block forward pass.

        Args:
            x: (B, L, d) — pre-normed input from residual stream
            hw_injection: (B, L, C_H) — optional injection from previous layer's CAC

        Returns:
            y: (B, L, d) — NCA block output (to be added as residual)
        """
        B, L, d = x.shape

        # 1. Reshape to NCA grid
        G = self._reshape_to_grid(x)  # (B, C, R, L)

        # 2. Apply RoPE — plan §9.2.3
        G = self.rope(G, L)

        # 3. Initialize hardware
        H = self.hardware.init_for_sequence(B, L)  # (B, C_H, R, L)

        # 3b. Inject aggregation context from previous layer (V5)
        if hw_injection is not None:
            # hw_injection: (B, L, C_H) → (B, C_H, 1, L), broadcast over R
            hw_inj_grid = hw_injection.permute(0, 2, 1).unsqueeze(2)  # (B, C_H, 1, L)
            H = H + hw_inj_grid

        # 4. NCA iteration — T steps
        # For hardware modulation: track running statistics using Welford online algorithm
        running_mean = torch.zeros_like(G)
        running_m2 = torch.zeros_like(G)  # running sum of squared deviations
        welford_count = 0

        # Accumulate routing weights across all T steps for pathway balance loss
        # Plan §9.3: ᾱ_m = (1/BTRL) Σ_{b,s,r,l} α_{b,m,r,l}
        alpha_accum = None

        for s in range(self.n_steps):
            # Compute dilation rate for this layer and step
            dilation = self.config.compute_dilation(self.layer_idx, s)

            # 4a. Perceive: multi-scale causal features (with dilation)
            P = self.perception(G, dilation=dilation)  # (B, (n_k+1)·C, R, L)

            # 4b. Compute routing from hardware (+ optional content modulation)
            alpha = self.pathway_update.compute_routing(H, G=G)  # (B, M, R, L)

            # Accumulate alpha for pathway balance loss (detach to avoid
            # double backward through the routing computation)
            if alpha_accum is None:
                alpha_accum = alpha.detach().clone()
            else:
                alpha_accum = alpha_accum + alpha.detach()

            # 4c. Compute weighted pathway update
            delta_G = self.pathway_update.compute_delta(P, alpha)  # (B, C, R, L)

            # 4d. Residual NCA step with per-step LayerNorm and step size
            G = self.pathway_update.apply_step(G, delta_G, step_idx=s)

            # State dropout during training
            if self.training and self.state_dropout is not None:
                G = self.state_dropout(G)

            # 4e. Update running statistics for hardware modulation (Welford)
            welford_count += 1
            delta_welford = G - running_mean
            running_mean = running_mean + delta_welford / welford_count
            running_m2 = running_m2 + delta_welford * (G - running_mean)

            # 4f. Hardware modulation every K_mod steps
            if (s + 1) % self.hw_mod_interval == 0 and (s + 1) < self.n_steps:
                running_var = running_m2 / max(welford_count, 1)
                H = self.hardware.modulate(H, running_mean, running_var)
                # Reset running stats for next modulation window
                running_mean = torch.zeros_like(G)
                running_m2 = torch.zeros_like(G)
                welford_count = 0

        # Store final H for smoothness loss and mean alpha for pathway balance
        self._H_final = H
        self._alpha_mean = alpha_accum / self.n_steps  # (B, M, R, L)

        # 5. Flatten back to (B, L, d)
        y = self._flatten_from_grid(G)  # (B, L, d)
        return y

    def get_routing_weights(self) -> Tensor | None:
        """Return step-averaged routing weights from last forward pass."""
        return getattr(self, "_alpha_mean", None)

    def get_hardware_loss(self) -> Tensor:
        """Get hardware smoothness loss for this block."""
        if hasattr(self, "_H_final"):
            return self.hardware.get_smoothness_loss(self._H_final)
        return torch.tensor(0.0)


class NCATransformerLayer(nn.Module):
    """Pre-norm residual NCA + HAM + CAC (gated) + Pre-norm residual FFN. Plan §3.4 + V5.

    V4 Forward:
        x = x + NCABlock(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    V5 Forward:
        x_norm = LayerNorm(x)
        x = x + gate_nca·NCABlock(x_norm) + gate_mem·HAM(x_norm) + gate_agg·CAC(x_norm)
        x = x + FFN(LayerNorm(x))

    When V5 features are disabled, falls back to V4 behaviour.
    """

    def __init__(self, config: DPNCAConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.config = config
        self.norm1 = nn.LayerNorm(config.d_model)
        self.nca_block = NCABlock(config, layer_idx=layer_idx)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn = SwiGLU(config)

        # V5: Hebbian Associative Memory
        if config.enable_memory:
            from dpnca.memory import HebbianAssociativeMemory
            self.memory = HebbianAssociativeMemory(config)
        else:
            self.memory = None

        # V5: Causal Aggregation Channels
        if config.enable_aggregation:
            from dpnca.aggregation import CausalAggregation
            self.aggregation = CausalAggregation(config)
        else:
            self.aggregation = None

        # V5: Gated residual fusion
        if config.enable_gated_fusion:
            d = config.d_model
            self.gate_nca = nn.Linear(d, d)
            nn.init.zeros_(self.gate_nca.weight)
            nn.init.constant_(self.gate_nca.bias, 2.0)  # sigmoid(2) ≈ 0.88

            if self.memory is not None:
                self.gate_mem = nn.Linear(d, d)
                nn.init.zeros_(self.gate_mem.weight)
                nn.init.constant_(self.gate_mem.bias, 0.0)  # sigmoid(0) = 0.5
            else:
                self.gate_mem = None

            if self.aggregation is not None:
                self.gate_agg = nn.Linear(d, d)
                nn.init.zeros_(self.gate_agg.weight)
                nn.init.constant_(self.gate_agg.bias, -0.85)  # sigmoid(-0.85) ≈ 0.3
            else:
                self.gate_agg = None
        else:
            self.gate_nca = None
            self.gate_mem = None
            self.gate_agg = None

    def forward(
        self, x: Tensor, hw_injection: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """Pre-norm residual NCA + HAM + CAC (gated) + FFN.

        Args:
            x: (B, L, d) — residual stream
            hw_injection: (B, L, C_H) — optional aggregation from previous layer

        Returns:
            x: (B, L, d) — updated residual stream
            hw_inj_next: (B, L, C_H) | None — aggregation output for next layer
        """
        x_norm = self.norm1(x)

        # Parallel branches on same pre-normed input
        nca_out = self.nca_block(x_norm, hw_injection=hw_injection)

        mem_out = self.memory(x_norm) if self.memory is not None else None

        agg_out = None
        hw_inj_next = None
        if self.aggregation is not None:
            agg_out, hw_inj_next = self.aggregation(x_norm)

        # Gated residual fusion (V5) or simple residual (V4)
        if self.gate_nca is not None:
            combined = torch.sigmoid(self.gate_nca(x)) * nca_out
            if mem_out is not None and self.gate_mem is not None:
                combined = combined + torch.sigmoid(self.gate_mem(x)) * mem_out
            if agg_out is not None and self.gate_agg is not None:
                combined = combined + torch.sigmoid(self.gate_agg(x)) * agg_out
            x = x + combined
        else:
            x = x + nca_out
            if mem_out is not None:
                x = x + mem_out
            if agg_out is not None:
                x = x + agg_out

        # FFN sub-layer with pre-norm residual
        x = x + self.ffn(self.norm2(x))

        return x, hw_inj_next

    def get_hardware_loss(self) -> Tensor:
        """Proxy to NCA block's hardware smoothness loss."""
        return self.nca_block.get_hardware_loss()
