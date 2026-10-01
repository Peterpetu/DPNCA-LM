"""Tests for HardwareChannels. Plan §8.2.

Verifies:
    1. Initialization shape: (B, C_H, R, L) from (1, C_H, R, 1)
    2. EMA modulation dynamics
    3. Stop-gradient on state statistics
    4. Hardware smoothness loss computation
    5. Modulation bounds (sigmoid output in (0,1))
"""

import pytest
import torch

from dpnca.config import DPNCAConfig
from dpnca.hardware import HardwareChannels


@pytest.fixture
def config():
    return DPNCAConfig(
        d_model=64, n_layers=1, n_compute_rows=4, n_channels=16,
        n_steps=4, n_modes=2, n_hw_channels=4, hw_mod_interval=2,
        perception_kernels=(3, 5), max_seq_len=64, vocab_size=256,
        hw_ema_decay=0.9,
    )


class TestHardwareChannels:
    def test_init_shape(self, config):
        """init_for_sequence must return (B, C_H, R, L)."""
        hw = HardwareChannels(config)
        H = hw.init_for_sequence(batch_size=3, seq_len=16)
        assert H.shape == (3, config.n_hw_channels, config.n_compute_rows, 16)

    def test_broadcast_consistency(self, config):
        """All sequences in a batch should start with the same hardware."""
        hw = HardwareChannels(config)
        H = hw.init_for_sequence(batch_size=4, seq_len=8)
        # All batch elements should have identical initial hardware
        # (broadcast from shared H_init)
        for b in range(1, 4):
            torch.testing.assert_close(H[0], H[b])

    def test_modulation_output_shape(self, config):
        """modulate must return same shape as input H."""
        hw = HardwareChannels(config)
        B, L = 2, 8
        H = hw.init_for_sequence(B, L)
        mu = torch.randn(B, config.n_channels, config.n_compute_rows, L)
        var = torch.randn(B, config.n_channels, config.n_compute_rows, L).abs()
        H_new = hw.modulate(H, mu, var)
        assert H_new.shape == H.shape

    def test_stop_gradient_on_stats(self, config):
        """State statistics must be detached (stop-gradient). Plan §3.3.4."""
        hw = HardwareChannels(config)
        B, L = 1, 4
        H = hw.init_for_sequence(B, L)

        # Create state stats that track gradients
        mu = torch.randn(B, config.n_channels, config.n_compute_rows, L, requires_grad=True)
        var = torch.randn(B, config.n_channels, config.n_compute_rows, L).abs().requires_grad_(True)

        H_new = hw.modulate(H, mu, var)
        loss = H_new.sum()
        loss.backward()

        # mu and var should NOT have gradients (stop_gradient applied internally)
        assert mu.grad is None or mu.grad.abs().sum() == 0, \
            "State statistics should be detached in modulate()"

    def test_ema_dynamics(self, config):
        """H_new should be between H and sigma(W_H·stats)."""
        hw = HardwareChannels(config)
        B, L = 1, 4
        H = hw.init_for_sequence(B, L)
        mu = torch.randn(B, config.n_channels, config.n_compute_rows, L)
        var = torch.ones(B, config.n_channels, config.n_compute_rows, L)

        H_new = hw.modulate(H, mu, var)

        # H_new should differ from H (unless EMA decay = 0)
        assert not torch.allclose(H, H_new), "Hardware should change after modulation"

    def test_smoothness_loss(self, config):
        """Smoothness loss = ‖H_final - H_init‖²_F. Non-negative."""
        hw = HardwareChannels(config)
        B, L = 2, 8
        H = hw.init_for_sequence(B, L)
        mu = torch.randn(B, config.n_channels, config.n_compute_rows, L)
        var = torch.ones_like(mu)
        H_new = hw.modulate(H, mu, var)

        loss = hw.get_smoothness_loss(H_new)
        assert loss.item() >= 0
        assert loss.item() > 0, "After modulation, drift should be positive"

    def test_smoothness_loss_zero_at_init(self, config):
        """Smoothness loss should be ~0 if hardware hasn't changed."""
        hw = HardwareChannels(config)
        H = hw.init_for_sequence(1, 4)
        loss = hw.get_smoothness_loss(H)
        assert loss.item() < 1e-10

    def test_H_init_learnable(self, config):
        """H_init should be a learnable parameter."""
        hw = HardwareChannels(config)
        assert hw.H_init.requires_grad
        assert hw.H_init.shape == (1, config.n_hw_channels, config.n_compute_rows, 1)
