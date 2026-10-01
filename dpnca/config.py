"""DPNCA-LM configuration. Plan Part 2 §3.8, Part 3 §4 Phase 1.

Defines DPNCAConfig dataclass and predefined model configurations
(Tiny through Large). All hyperparameters have documented defaults,
valid ranges, and justifications.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class DPNCAConfig:
    """Configuration for the Deep Parallel NCA Language Model.

    Notation (matching plan §9.1):
        d = d_model         — model hidden dimension
        R = n_compute_rows  — vertical compute dimension of NCA grid
        C = n_channels      — channels per cell (= d / R)
        T = n_steps         — NCA iterations per layer
        M = n_modes         — mixture-of-pathways modes
        C_H = n_hw_channels — hardware channels per cell
    """

    # ── Model dimensions ──────────────────────────────────────────────
    d_model: int = 512
    n_layers: int = 6
    n_compute_rows: int = 8     # R
    n_channels: int = 64        # C = d_model / R

    # ── NCA dynamics ──────────────────────────────────────────────────
    n_steps: int = 6            # T: NCA iterations per layer
    n_modes: int = 8            # M: mixture-of-pathways modes
    perception_kernels: tuple[int, ...] = (3, 5, 7)
    step_size_init: float = 0.1 # initial learnable step size η_s

    # ── Hardware channels ─────────────────────────────────────────────
    n_hw_channels: int = 16     # C_H
    hw_mod_interval: int = 4    # K_mod: NCA steps between hardware updates
    hw_ema_decay: float = 0.95  # λ_hw: EMA decay for hardware modulation

    # ── FFN ────────────────────────────────────────────────────────────
    ffn_multiplier: float = 8.0 / 3.0  # SwiGLU expansion ratio
    ffn_type: str = "swiglu"           # "swiglu" (Option A)

    # ── Sequence ──────────────────────────────────────────────────────
    max_seq_len: int = 2048

    # ── Vocabulary ────────────────────────────────────────────────────
    vocab_size: int = 50257     # GPT-2 BPE

    # ── Dilation schedule ────────────────────────────────────────────
    #
    # WaveNet-style exponential dilation for long-range receptive field.
    # Within each NCA layer, step s uses dilation d_s = base^s (intra-layer).
    # Across layers, layer ℓ multiplies by an inter-layer growth factor.
    # Effective dilation for layer ℓ, step s:
    #     d(ℓ, s) = inter_layer_dilation_base^ℓ × intra_step_dilation_base^s
    # clamped to max_dilation.
    #
    # With defaults (intra=2, inter=2, max=512, T=6, N=12):
    #     RF ≈ 1 + Σ_ℓ Σ_s (k_max - 1) × d(ℓ,s) ≈ 1.5M positions
    #
    # Biological justification: exponential dilation mirrors the exponential
    # growth of axonal projection distances across cortical layers (V1→IT).
    # Setting enable_dilation=False recovers the original dense perception.
    enable_dilation: bool = False  # opt-in; False preserves backward compat
    intra_step_dilation_base: int = 2   # dilation growth base within T steps
    inter_layer_dilation_base: int = 2  # dilation growth base across layers
    max_dilation: int = 512             # clamp to prevent excessive sparsity

    # ── Hebbian Associative Memory (V5) ─────────────────────────────
    n_memory_heads: int = 8
    d_memory_head: int = 0          # 0 = auto (d_model // n_memory_heads)
    memory_decay_init: float = 0.99  # initial λ (half-life ≈ 69 positions)
    enable_memory: bool = False      # opt-in; False preserves V4 compat

    # ── Causal Aggregation Channels (V5) ──────────────────────────────
    n_aggregation_scales: int = 5    # K: number of EMA timescales
    d_aggregation: int = 64          # d_a: per-scale hidden dimension
    enable_aggregation: bool = False  # opt-in

    # ── Content-Modulated Routing (V5) ────────────────────────────────
    content_modulated_routing: bool = False  # opt-in

    # ── Gated Fusion (V5) ─────────────────────────────────────────────
    enable_gated_fusion: bool = False  # opt-in

    # ── Regularization ────────────────────────────────────────────────
    dropout: float = 0.1
    lambda_helm: float = 0.001       # Helmholtz entropy regularisation
    lambda_pathway_balance: float = 0.01
    lambda_hw_smooth: float = 0.1  # scaled for mean-normalized hardware loss

    # ── Memory efficiency ─────────────────────────────────────────────
    gradient_checkpointing: bool = False  # Recompute activations to save VRAM

    # ── Derived properties ────────────────────────────────────────────

    @property
    def d_ff(self) -> int:
        """SwiGLU FFN intermediate dimension, rounded up to multiple of 8."""
        raw = int(math.ceil(self.d_model * self.ffn_multiplier))
        return ((raw + 7) // 8) * 8  # round to nearest 8 for hardware efficiency

    @property
    def n_perception_channels(self) -> int:
        """Total perception output channels: (n_kernels + 1) * C (identity + convs)."""
        return (len(self.perception_kernels) + 1) * self.n_channels

    @property
    def d_memory_head_effective(self) -> int:
        """Effective per-head memory dimension. Auto-derives from d_model if d_memory_head == 0."""
        if self.d_memory_head > 0:
            return self.d_memory_head
        return self.d_model // self.n_memory_heads

    def compute_dilation(self, layer_idx: int, step_idx: int) -> int:
        """Compute the effective dilation rate for a given layer and NCA step.

        V5 schedule (when T >= 12): first T//2 steps use dilation 1 (dense local),
        remaining steps use exponential dilation 2^{i+1} (sparse long-range).

        V4 schedule (legacy): d(ℓ, s) = clamp(inter^ℓ × intra^s, max=max_dilation).

        When enable_dilation is False, always returns 1 (standard dense convolution).

        Args:
            layer_idx: layer index ℓ ∈ [0, n_layers)
            step_idx: NCA step index s ∈ [0, n_steps)

        Returns:
            Integer dilation rate ≥ 1.
        """
        if not self.enable_dilation:
            return 1
        T = self.n_steps
        local_steps = T // 2
        if step_idx < local_steps:
            return 1
        exp_idx = step_idx - local_steps
        d = 2 ** (exp_idx + 1)
        return min(d, self.max_dilation)

    def __post_init__(self) -> None:
        assert self.d_model > 0, f"d_model must be positive, got {self.d_model}"
        assert self.d_model % self.n_compute_rows == 0, (
            f"d_model ({self.d_model}) must be divisible by n_compute_rows ({self.n_compute_rows})"
        )
        assert self.n_channels == self.d_model // self.n_compute_rows, (
            f"n_channels ({self.n_channels}) must equal d_model // n_compute_rows "
            f"({self.d_model // self.n_compute_rows})"
        )
        assert self.n_steps >= 1, f"n_steps must be >= 1, got {self.n_steps}"
        assert self.n_modes >= 1, f"n_modes must be >= 1, got {self.n_modes}"
        assert self.n_layers >= 1, f"n_layers must be >= 1, got {self.n_layers}"
        assert all(k >= 3 and k % 2 == 1 for k in self.perception_kernels), (
            f"All perception kernels must be odd and >= 3, got {self.perception_kernels}"
        )
        assert 0.0 <= self.dropout < 1.0, f"dropout must be in [0, 1), got {self.dropout}"
        assert self.n_hw_channels > 0, f"n_hw_channels must be positive, got {self.n_hw_channels}"
        assert self.hw_mod_interval >= 1, f"hw_mod_interval must be >= 1, got {self.hw_mod_interval}"
        if self.hw_mod_interval >= self.n_steps:
            import warnings
            warnings.warn(
                f"hw_mod_interval ({self.hw_mod_interval}) >= n_steps ({self.n_steps}): "
                f"hardware modulation will never fire. Set hw_mod_interval < n_steps "
                f"for online hardware adaptation.",
                stacklevel=2,
            )
        assert 0.0 < self.hw_ema_decay < 1.0, (
            f"hw_ema_decay must be in (0, 1), got {self.hw_ema_decay}"
        )
        # Dilation parameter validation
        assert self.intra_step_dilation_base >= 1, (
            f"intra_step_dilation_base must be >= 1, got {self.intra_step_dilation_base}"
        )
        assert self.inter_layer_dilation_base >= 1, (
            f"inter_layer_dilation_base must be >= 1, got {self.inter_layer_dilation_base}"
        )
        assert self.max_dilation >= 1, (
            f"max_dilation must be >= 1, got {self.max_dilation}"
        )
        # V5 feature validation
        if self.enable_memory:
            assert self.n_memory_heads >= 1, (
                f"n_memory_heads must be >= 1, got {self.n_memory_heads}"
            )
            d_mem_eff = self.d_memory_head if self.d_memory_head > 0 else self.d_model // self.n_memory_heads
            assert self.d_model % self.n_memory_heads == 0 or self.d_memory_head > 0, (
                f"d_model ({self.d_model}) must be divisible by n_memory_heads ({self.n_memory_heads}) "
                f"when d_memory_head is auto (0)"
            )
            assert 0.0 < self.memory_decay_init < 1.0, (
                f"memory_decay_init must be in (0, 1), got {self.memory_decay_init}"
            )
        if self.enable_aggregation:
            assert self.n_aggregation_scales >= 1, (
                f"n_aggregation_scales must be >= 1, got {self.n_aggregation_scales}"
            )
            assert self.d_aggregation >= 1, (
                f"d_aggregation must be >= 1, got {self.d_aggregation}"
            )


# ── Predefined configurations ────────────────────────────────────────

DPNCA_TINY = DPNCAConfig(
    d_model=256,
    n_layers=4,
    n_compute_rows=8,
    n_channels=32,
    n_steps=4,
    n_modes=4,
    n_hw_channels=8,
    hw_mod_interval=2,  # Must be < T so modulation fires at least once
    perception_kernels=(3, 5),
    max_seq_len=512,
    dropout=0.1,
)

DPNCA_SMALL = DPNCAConfig(
    d_model=512,
    n_layers=6,
    n_compute_rows=8,
    n_channels=64,
    n_steps=6,
    n_modes=8,
    n_hw_channels=16,
    perception_kernels=(3, 5, 7),
    max_seq_len=1024,
    dropout=0.1,
)

DPNCA_BASE = DPNCAConfig(
    d_model=768,
    n_layers=12,
    n_compute_rows=12,
    n_channels=64,
    n_steps=8,
    n_modes=8,
    n_hw_channels=16,
    perception_kernels=(3, 5, 7),
    max_seq_len=2048,
    dropout=0.1,
)

DPNCA_160M = DPNCAConfig(
    d_model=1024,
    n_layers=12,
    n_compute_rows=8,
    n_channels=128,
    n_steps=6,
    n_modes=8,
    n_hw_channels=16,
    hw_mod_interval=3,
    perception_kernels=(3, 5, 7),
    max_seq_len=1024,
    dropout=0.1,
    gradient_checkpointing=True,
)

DPNCA_LARGE = DPNCAConfig(
    d_model=1024,
    n_layers=24,
    n_compute_rows=16,
    n_channels=64,
    n_steps=8,
    n_modes=16,
    n_hw_channels=32,
    perception_kernels=(3, 5, 7),
    max_seq_len=2048,
    dropout=0.1,
)


# ── V5 configurations (HAM + CAC + content routing + gated fusion) ──

DPNCA_V5_TINY = DPNCAConfig(
    d_model=256,
    n_layers=4,
    n_compute_rows=8,
    n_channels=32,
    n_steps=8,
    n_modes=4,
    n_hw_channels=8,
    hw_mod_interval=2,
    perception_kernels=(3, 5),
    max_seq_len=512,
    dropout=0.1,
    enable_dilation=True,
    enable_memory=True,
    n_memory_heads=4,
    memory_decay_init=0.99,
    enable_aggregation=True,
    n_aggregation_scales=3,
    d_aggregation=32,
    content_modulated_routing=True,
    enable_gated_fusion=True,
)

DPNCA_V5_SMALL = DPNCAConfig(
    d_model=512,
    n_layers=8,
    n_compute_rows=8,
    n_channels=64,
    n_steps=12,
    n_modes=8,
    n_hw_channels=16,
    hw_mod_interval=3,
    perception_kernels=(3, 5, 7),
    max_seq_len=1024,
    dropout=0.1,
    enable_dilation=True,
    enable_memory=True,
    n_memory_heads=8,
    memory_decay_init=0.99,
    enable_aggregation=True,
    n_aggregation_scales=5,
    d_aggregation=64,
    content_modulated_routing=True,
    enable_gated_fusion=True,
)

DPNCA_V5_160M = DPNCAConfig(
    d_model=768,
    n_layers=12,
    n_compute_rows=8,
    n_channels=96,
    n_steps=6,
    n_modes=8,
    n_hw_channels=16,
    hw_mod_interval=3,
    perception_kernels=(3, 5, 7),
    max_seq_len=1024,
    dropout=0.1,
    gradient_checkpointing=True,
    enable_dilation=True,
    enable_memory=True,
    n_memory_heads=8,
    memory_decay_init=0.99,
    enable_aggregation=True,
    n_aggregation_scales=5,
    d_aggregation=64,
    content_modulated_routing=True,
    enable_gated_fusion=True,
)
