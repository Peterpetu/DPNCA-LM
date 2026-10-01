"""Hardware channels — online neuromodulation. Plan §3.3.4, §9.2.5.5.

Hardware channels provide cell-specific, position-specific processing context
that modulates pathway routing. They bridge the biological concept of
neuromodulation (dopamine, serotonin, etc.) into the NCA framework.

Initialization: H_0 ∈ R^{1 × C_H × R × 1} — learned params broadcast across
    batch and sequence length. Replaces the genome-grown hardware of V3.

Online modulation (every K_mod NCA steps):
    stats = [sg(μ_G) ∥ sg(σ²_G)] ∈ R^{B × 2C × R × L}  (stop-gradient)
    H^{s+K} = (1 - λ_hw) · H^s + λ_hw · σ(W_H · stats)

Gradient graph (plan §3.3.4):
    - W_H IS a learned parameter, updated via gradient descent
    - State statistics have stop_gradient: W_H gets gradients from ROUTING
      (via α_m) but NOT from NCA state dynamics (via G)
    - Hardware VALUES do not receive gradients from cross-entropy loss
    - This creates desired timescale separation: hardware adapts slowly
      based on aggregate statistics, not per-step gradient signal

Weight initialization (plan §10):
    H_0:  U(-0.1, 0.1)
    W_H:  Xavier uniform
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from dpnca.config import DPNCAConfig


class HardwareChannels(nn.Module):
    """Hardware channels with EMA online modulation. Plan §3.3.4."""

    def __init__(self, config: DPNCAConfig) -> None:
        super().__init__()
        C_H = config.n_hw_channels
        C = config.n_channels
        R = config.n_compute_rows

        self.n_hw_channels = C_H
        self.n_channels = C
        self.n_compute_rows = R
        self.hw_ema_decay = config.hw_ema_decay
        self.hw_mod_interval = config.hw_mod_interval

        # Learned hardware initialization: (1, C_H, R, 1) — broadcast across B, L
        # U(-0.1, 0.1) — plan §10
        self.H_init = nn.Parameter(torch.empty(1, C_H, R, 1))
        nn.init.uniform_(self.H_init, -0.1, 0.1)

        # Modulation projection: stats (2C channels) → hardware (C_H channels)
        # W_H: (C_H, 2C) — Xavier uniform — plan §10
        self.W_H = nn.Linear(2 * C, C_H, bias=True)
        nn.init.xavier_uniform_(self.W_H.weight)
        nn.init.zeros_(self.W_H.bias)

        # Track total hardware drift for smoothness loss
        self._H_initial: Tensor | None = None

    def init_for_sequence(self, batch_size: int, seq_len: int) -> Tensor:
        """Expand H_0 to full (B, C_H, R, L) and store initial for loss.

        Args:
            batch_size: B
            seq_len: L

        Returns:
            H: (B, C_H, R, L) — initial hardware for this sequence
        """
        H = self.H_init.expand(batch_size, -1, -1, seq_len).clone()
        self._H_initial = H.detach().clone()
        return H

    def modulate(self, H: Tensor, G_history_mean: Tensor, G_history_var: Tensor) -> Tensor:
        """EMA hardware update from stopped-gradient state statistics.

        H' = (1 - λ) · H + λ · σ(W_H · [sg(μ_G) ∥ sg(σ²_G)])

        Args:
            H: (B, C_H, R, L) — current hardware
            G_history_mean: (B, C, R, L) — running mean of state channels
            G_history_var: (B, C, R, L) — running variance of state channels

        Returns:
            H_new: (B, C_H, R, L) — modulated hardware
        """
        # Stop gradient on state statistics — plan §3.3.4
        mu = G_history_mean.detach()
        var = G_history_var.detach()

        # Concatenate: (B, 2C, R, L)
        stats = torch.cat([mu, var], dim=1)

        # Project: need (B, R, L, 2C) for nn.Linear → (B, R, L, C_H)
        B, _, R, L = stats.shape
        stats_nhwc = stats.permute(0, 2, 3, 1)  # (B, R, L, 2C)
        h_new_raw = self.W_H(stats_nhwc)         # (B, R, L, C_H)
        h_new_raw = torch.sigmoid(h_new_raw)      # bound to (0, 1)
        h_new_raw = h_new_raw.permute(0, 3, 1, 2)  # (B, C_H, R, L)

        # EMA update
        lam = self.hw_ema_decay
        H_new = (1.0 - lam) * H + lam * h_new_raw

        return H_new

    def get_smoothness_loss(self, H_final: Tensor) -> Tensor:
        """Hardware smoothness penalty: mean(‖H_final - H_init‖²). Plan §9.3.

        Uses .mean() over (B, C_H, R, L) to make the penalty independent
        of batch size and sequence length.

        Returns:
            Scalar mean-squared hardware drift.
        """
        if self._H_initial is None:
            return torch.tensor(0.0, device=H_final.device, dtype=H_final.dtype)
        return (H_final - self._H_initial).pow(2).mean()
