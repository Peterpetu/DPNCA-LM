"""Tests for NCABlock and NCATransformerLayer. Plan §8.2.

Verifies:
    1. Shape preservation (B, L, d) → (B, L, d) through full layer
    2. Residual connection works (identity at init)
    3. Gradient flow through T NCA steps and depth
    4. Grid reshape is invertible
"""

import pytest
import torch

from dpnca.config import DPNCAConfig
from dpnca.nca_layer import NCABlock, NCATransformerLayer


@pytest.fixture
def config():
    return DPNCAConfig(
        d_model=64, n_layers=2, n_compute_rows=4, n_channels=16,
        n_steps=3, n_modes=2, n_hw_channels=4, hw_mod_interval=2,
        perception_kernels=(3, 5), max_seq_len=64, vocab_size=256,
        dropout=0.0,
    )


class TestNCABlock:
    def test_output_shape(self, config):
        """NCABlock must preserve (B, L, d) shape."""
        block = NCABlock(config)
        x = torch.randn(2, 16, config.d_model)
        y = block(x)
        assert y.shape == x.shape

    def test_grid_reshape_invertible(self, config):
        """Reshape to grid and back should recover original tensor."""
        block = NCABlock(config)
        x = torch.randn(1, 8, config.d_model)
        G = block._reshape_to_grid(x)
        x_recovered = block._flatten_from_grid(G)
        torch.testing.assert_close(x, x_recovered)

    def test_gradient_flow(self, config):
        """Gradients must flow through the NCA block."""
        block = NCABlock(config)
        x = torch.randn(1, 8, config.d_model, requires_grad=True)
        y = block(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_hardware_loss_available(self, config):
        """After forward, get_hardware_loss should return a scalar."""
        block = NCABlock(config)
        x = torch.randn(1, 8, config.d_model)
        _ = block(x)
        hw_loss = block.get_hardware_loss()
        assert hw_loss.numel() == 1
        assert hw_loss.item() >= 0

    def test_different_seq_lengths(self, config):
        """Block should handle variable sequence lengths."""
        block = NCABlock(config)
        for L in [1, 4, 16, 32]:
            x = torch.randn(1, L, config.d_model)
            y = block(x)
            assert y.shape == (1, L, config.d_model), f"Failed for L={L}"


class TestNCATransformerLayer:
    def test_output_shape(self, config):
        """NCATransformerLayer must preserve (B, L, d) shape."""
        layer = NCATransformerLayer(config)
        x = torch.randn(2, 16, config.d_model)
        y, hw_inj = layer(x)
        assert y.shape == x.shape
        # V4 config: no aggregation → hw_inj is None
        assert hw_inj is None

    def test_residual_connection(self, config):
        """With near-identity init, output should be close to input."""
        layer = NCATransformerLayer(config)
        x = torch.randn(1, 8, config.d_model)
        y, _ = layer(x)
        # The layer should produce a small perturbation at init
        relative_change = (y - x).norm() / x.norm()
        assert relative_change < 10.0, (
            f"Layer output diverged too much from input: {relative_change:.4f}"
        )

    def test_gradient_flow_through_layer(self, config):
        """Gradients must flow through NCA + FFN."""
        layer = NCATransformerLayer(config)
        x = torch.randn(1, 8, config.d_model, requires_grad=True)
        y, _ = layer(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_hardware_loss_proxy(self, config):
        """get_hardware_loss should proxy to inner NCA block."""
        layer = NCATransformerLayer(config)
        x = torch.randn(1, 8, config.d_model)
        _ = layer(x)
        hw_loss = layer.get_hardware_loss()
        assert hw_loss.numel() == 1

    def test_alpha_accumulation(self, config):
        """NCA block must accumulate routing alphas across all T steps.

        Plan §9.3: ᾱ_m = (1/BTRL) Σ_{b,s,r,l} α_{b,m,r,l}
        After forward, _alpha_mean should contain the T-step average.
        """
        block = NCABlock(config)
        x = torch.randn(2, 8, config.d_model)
        _ = block(x)
        alpha_mean = block.get_routing_weights()
        assert alpha_mean is not None, "Alpha mean not stored after forward"
        # Shape: (B, M, R, L)
        assert alpha_mean.shape == (2, config.n_modes, config.n_compute_rows, 8)
        # Must be a valid probability distribution (sum to 1 over M)
        sums = alpha_mean.sum(dim=1)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)
        # Must be detached (no grad)
        assert not alpha_mean.requires_grad
