"""Causal multi-scale perception for NCA grid. Plan §9.2.5.1.

Adapted from V3's CausalMultiScalePerceive. Extracts local neighborhood
features using depthwise Conv2d with causal horizontal padding and
circular vertical padding (toroidal rows).

For each kernel size k ∈ perception_kernels:
    F_k = DepthwiseConv2D(G, kernel=(k,k), groups=C, dilation=d)
    Horizontal padding: pad_left = d*(k-1), pad_right = 0  (causal)
    Vertical padding: circular ⌊d*(k-1)/2⌋ on each side  (toroidal)

Output: P = [G ∥ F_3 ∥ F_5 ∥ F_7] ∈ R^{B × (n_k+1)C × R × L}

The identity branch (G itself) provides skip-connection at the perception
level, ensuring gradient flow even if convolution weights are small.

Dilation support (WaveNet-style):
    When dilation > 1, the convolution samples every d-th position rather
    than every adjacent position. This expands the receptive field without
    adding parameters — dilation is a runtime parameter of Conv2d, not a
    learned weight. The causal padding formula generalises:
        effective_kernel = d * (k - 1) + 1
        causal_pad_left = d * (k - 1)
    This is the standard dilated convolution identity.

    Biological analog: dilated convolutions mirror axonal projections
    in cortical layers, where higher layers (V4, IT) have exponentially
    larger receptive fields via sparse long-range connections.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dpnca.config import DPNCAConfig


class CausalPerception(nn.Module):
    """Causal multi-scale 2D perception for the NCA grid. Plan §3.3.2.

    Input:  G ∈ R^{B × C × R × L}  (NCHW format)
    Output: P ∈ R^{B × (n_k+1)·C × R × L}  (identity + n_k conv branches)

    Each depthwise Conv2d operates independently per channel (groups=C).
    Perception Conv2d weights: Kaiming uniform, mode='fan_in' — plan §10.
    Perception Conv2d biases: Zero — plan §10.
    """

    def __init__(self, config: DPNCAConfig) -> None:
        super().__init__()
        C = config.n_channels
        self.perception_kernels = config.perception_kernels

        self.convs = nn.ModuleList()
        for k in self.perception_kernels:
            conv = nn.Conv2d(
                in_channels=C,
                out_channels=C,
                kernel_size=k,
                padding=0,       # we handle padding manually for causal + circular
                groups=C,        # depthwise: each channel has its own k×k filter
                bias=True,
            )
            # Kaiming uniform, fan_in (plan §10)
            nn.init.kaiming_uniform_(conv.weight, mode="fan_in", nonlinearity="linear")
            nn.init.zeros_(conv.bias)
            self.convs.append(conv)

        # Total output channels: identity (C) + n_kernels × C
        self.n_output_channels = (len(self.perception_kernels) + 1) * C

    def _causal_pad(self, G: Tensor, k: int, dilation: int = 1) -> tuple[Tensor, int]:
        """Apply asymmetric padding: circular vertical + causal horizontal.

        For a kernel of size k with dilation d, the effective kernel size is
        k_eff = d * (k - 1) + 1. Causal padding places k_eff - 1 zeros on the
        left (horizontal) and k_eff // 2 circular padding on each side (vertical).

        Vertical dilation is clamped so circular padding never exceeds R
        (PyTorch constraint: circular pad must not wrap more than once).

        Args:
            G: (B, C, R, L) — NCHW grid state
            k: kernel size (square k×k)
            dilation: dilation rate (≥ 1)

        Returns:
            Padded tensor and the (possibly clamped) vertical dilation rate.
        """
        R = G.shape[2]

        # Vertical dilation clamped: pad_v must be < R for circular padding
        # pad_v = (d_v*(k-1))//2, so d_v_max such that d_v*(k-1)//2 < R
        max_dilation_v = max(1, (2 * R - 1) // (k - 1)) if k > 1 else dilation
        dilation_v = min(dilation, max_dilation_v)

        k_eff_v = dilation_v * (k - 1) + 1
        k_eff_h = dilation * (k - 1) + 1  # horizontal uses full dilation

        pad_v = (k_eff_v - 1) // 2
        # Step 1: circular vertical (toroidal rows)
        G = F.pad(G, (0, 0, pad_v, pad_v), mode="circular")
        # Step 2: causal horizontal (zeros on left only, nothing on right)
        pad_h = k_eff_h - 1
        G = F.pad(G, (pad_h, 0, 0, 0), mode="constant", value=0.0)
        return G, dilation_v

    def forward(self, G: Tensor, dilation: int = 1) -> Tensor:
        """Multi-scale causal perception with optional dilation.

        When dilation=1 (default), this is standard dense perception.
        When dilation>1, each convolution samples every d-th position,
        expanding the receptive field by factor d without adding parameters.

        The same Conv2d weights are used regardless of dilation — dilation
        is a runtime parameter, not a weight modification. This is critical:
        the NCA learns local interaction rules that are then applied at
        different spatial scales via dilation.

        Args:
            G: (B, C, R, L) — NCA grid state in NCHW format
            dilation: dilation rate for convolutions (≥ 1)

        Returns:
            P: (B, (n_k+1)·C, R, L) — perceived features with identity branch
        """
        branches = [G]  # identity branch
        for conv, k in zip(self.convs, self.perception_kernels):
            G_padded, dilation_v = self._causal_pad(G, k, dilation=dilation)
            # Conv2d with runtime dilation: same weights, different spacing
            # Vertical dilation may be clamped if R is small
            branches.append(
                F.conv2d(
                    G_padded,
                    conv.weight,
                    conv.bias,
                    stride=1,
                    padding=0,
                    dilation=(dilation_v, dilation),
                    groups=conv.groups,
                )
            )
        return torch.cat(branches, dim=1)
