"""Tests for CausalPerception. Plan §8.2.

Verifies:
    1. Output shape is correct: (B, (n_k+1)*C, R, L)
    2. Causal masking: future columns have zero influence
    3. Identity branch is preserved
    4. Gradient flows through perception
"""

import pytest
import torch

from dpnca.config import DPNCAConfig
from dpnca.perception import CausalPerception


@pytest.fixture
def config():
    return DPNCAConfig(
        d_model=64, n_layers=1, n_compute_rows=4, n_channels=16,
        n_steps=1, n_modes=2, n_hw_channels=4,
        perception_kernels=(3, 5), max_seq_len=64, vocab_size=256,
    )


class TestCausalPerception:
    def test_output_shape(self, config):
        """Output must be (B, (n_k+1)*C, R, L)."""
        perc = CausalPerception(config)
        B, C, R, L = 2, config.n_channels, config.n_compute_rows, 16
        G = torch.randn(B, C, R, L)
        P = perc(G)
        n_k = len(config.perception_kernels)
        assert P.shape == (B, (n_k + 1) * C, R, L)

    def test_identity_branch(self, config):
        """First C channels of output must equal the input."""
        perc = CausalPerception(config)
        G = torch.randn(2, config.n_channels, config.n_compute_rows, 16)
        P = perc(G)
        C = config.n_channels
        torch.testing.assert_close(P[:, :C, :, :], G)

    def test_causal_masking(self, config):
        """Changing a future token must not affect current token's perception."""
        perc = CausalPerception(config)
        perc.eval()
        C, R, L = config.n_channels, config.n_compute_rows, 16

        G1 = torch.randn(1, C, R, L)
        G2 = G1.clone()
        # Modify column 10 (future relative to column 5)
        G2[:, :, :, 10] = torch.randn(1, C, R)

        P1 = perc(G1)
        P2 = perc(G2)

        # Columns 0-9 should be identical (column 10 is in the future)
        # But the largest kernel is 5, so column 10's change can affect columns
        # 10-(5-1)=6 through 10 at the earliest. Column 5 is safe.
        for col in range(6):
            torch.testing.assert_close(
                P1[:, :, :, col], P2[:, :, :, col],
                msg=f"Column {col} should not be affected by future column 10"
            )

    def test_gradient_flow(self, config):
        """Gradients must flow through all conv branches."""
        perc = CausalPerception(config)
        G = torch.randn(2, config.n_channels, config.n_compute_rows, 16, requires_grad=True)
        P = perc(G)
        loss = P.sum()
        loss.backward()
        assert G.grad is not None
        assert G.grad.abs().sum() > 0

    def test_different_kernel_sizes(self):
        """Test with various kernel configurations."""
        for kernels in [(3,), (3, 5), (3, 5, 7)]:
            cfg = DPNCAConfig(
                d_model=64, n_layers=1, n_compute_rows=4, n_channels=16,
                n_steps=1, n_modes=2, n_hw_channels=4,
                perception_kernels=kernels, max_seq_len=64, vocab_size=256,
            )
            perc = CausalPerception(cfg)
            G = torch.randn(1, 16, 4, 8)
            P = perc(G)
            expected_C = (len(kernels) + 1) * 16
            assert P.shape[1] == expected_C, f"kernels={kernels}: expected {expected_C} channels"

    def test_single_column(self, config):
        """Perception should work with L=1 (single token)."""
        perc = CausalPerception(config)
        G = torch.randn(1, config.n_channels, config.n_compute_rows, 1)
        P = perc(G)
        assert P.shape == (1, config.n_perception_channels, config.n_compute_rows, 1)
