"""Tests for PathwayUpdate. Plan §8.2.

Verifies:
    1. Near-identity initialization (zero bias, zero gate → uniform routing)
    2. Output shape correctness
    3. Gradient flow through all T steps
    4. Routing weights sum to 1 over modes
    5. Per-step LayerNorm and step sizes function correctly
"""

import pytest
import torch

from dpnca.config import DPNCAConfig
from dpnca.pathway_update import PathwayUpdate


@pytest.fixture
def config():
    return DPNCAConfig(
        d_model=64, n_layers=1, n_compute_rows=4, n_channels=16,
        n_steps=3, n_modes=4, n_hw_channels=4, hw_mod_interval=2,
        perception_kernels=(3, 5), max_seq_len=64, vocab_size=256,
    )


class TestPathwayUpdate:
    def test_routing_sums_to_one(self, config):
        """Routing weights must sum to 1.0 over the mode dimension."""
        pu = PathwayUpdate(config)
        B, C_H, R, L = 2, config.n_hw_channels, config.n_compute_rows, 8
        H = torch.randn(B, C_H, R, L)
        alpha = pu.compute_routing(H)  # (B, M, R, L)
        sums = alpha.sum(dim=1)  # (B, R, L) — should all be 1.0
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)

    def test_uniform_routing_at_init(self, config):
        """With zero-initialized gate, routing should be uniform: α_m = 1/M."""
        pu = PathwayUpdate(config)
        M = config.n_modes
        H = torch.randn(1, config.n_hw_channels, config.n_compute_rows, 4)
        alpha = pu.compute_routing(H)
        expected = torch.full_like(alpha, 1.0 / M)
        torch.testing.assert_close(alpha, expected, atol=1e-5, rtol=1e-5)

    def test_delta_shape(self, config):
        """compute_delta must return (B, C, R, L)."""
        pu = PathwayUpdate(config)
        B, R, L = 2, config.n_compute_rows, 8
        P_dim = config.n_perception_channels
        P = torch.randn(B, P_dim, R, L)
        alpha = torch.softmax(torch.randn(B, config.n_modes, R, L), dim=1)
        delta = pu.compute_delta(P, alpha)
        assert delta.shape == (B, config.n_channels, R, L)

    def test_near_identity_at_init(self, config):
        """At initialization, NCA update should produce small deltas."""
        pu = PathwayUpdate(config)
        B, C, R, L = 1, config.n_channels, config.n_compute_rows, 8
        G = torch.randn(B, C, R, L)
        P_dim = config.n_perception_channels
        P = torch.randn(B, P_dim, R, L)
        H = torch.randn(B, config.n_hw_channels, R, L)

        alpha = pu.compute_routing(H)
        delta = pu.compute_delta(P, alpha)
        G_new = pu.apply_step(G, delta, step_idx=0)

        # Update should be small relative to G (step_size=0.1 + small weights)
        relative_change = (G_new - G).norm() / G.norm()
        assert relative_change < 1.0, f"Relative change too large: {relative_change}"

    def test_gradient_flow_through_steps(self, config):
        """Gradients must flow through all T NCA steps."""
        pu = PathwayUpdate(config)
        B, C, R, L = 1, config.n_channels, config.n_compute_rows, 8
        G = torch.randn(B, C, R, L, requires_grad=True)
        P_dim = config.n_perception_channels
        H = torch.randn(B, config.n_hw_channels, R, L)

        # Simulate T steps
        state = G
        for s in range(config.n_steps):
            P = torch.randn(B, P_dim, R, L)  # simplified — real would come from perception
            alpha = pu.compute_routing(H)
            delta = pu.compute_delta(P, alpha)
            state = pu.apply_step(state, delta, step_idx=s)

        loss = state.sum()
        loss.backward()
        assert G.grad is not None
        assert G.grad.abs().sum() > 0

    def test_per_step_parameters(self, config):
        """Each NCA step should have its own step size and LayerNorm."""
        pu = PathwayUpdate(config)
        assert len(pu.step_sizes) == config.n_steps
        assert len(pu.step_norms) == config.n_steps

        # Step sizes should all be initialized to step_size_init
        for eta in pu.step_sizes:
            assert abs(eta.item() - config.step_size_init) < 1e-6

    def test_pathway_weight_shared_across_steps(self, config):
        """W and b should be shared (same object) across steps."""
        pu = PathwayUpdate(config)
        # W and b are single parameters, not duplicated per step
        assert pu.W.shape == (config.n_modes, config.n_channels, config.n_perception_channels)
        assert pu.b.shape == (config.n_modes, config.n_channels)
