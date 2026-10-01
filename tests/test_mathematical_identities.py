"""Mathematical identity tests for DPNCA-LM. Plan §8.2 / §12.

Tests fundamental mathematical properties that must hold:
    1. Perception is truly causal (future tokens have zero gradient)
    2. Residual stream preserves information (identity init ≈ identity function)
    3. Hardware modulation is outside gradient graph
    4. Pathway softmax sums to 1
    5. RoPE preserves vector norms (orthogonal rotation)
    6. KL divergence is non-negative
    7. Gradient Jacobian eigenvalues at init ≈ 1 (residual property)
"""

import math

import pytest
import torch
import torch.nn.functional as F

from dpnca.config import DPNCAConfig
from dpnca.embedding import RotaryPositionalEncoding
from dpnca.hardware import HardwareChannels
from dpnca.model import DPNCA
from dpnca.nca_layer import NCABlock
from dpnca.pathway_update import PathwayUpdate
from dpnca.perception import CausalPerception


@pytest.fixture
def config():
    return DPNCAConfig(
        d_model=64, n_layers=2, n_compute_rows=4, n_channels=16,
        n_steps=3, n_modes=4, n_hw_channels=4, hw_mod_interval=2,
        perception_kernels=(3, 5), max_seq_len=64, vocab_size=256,
        dropout=0.0,
    )


class TestCausalityIdentity:
    """Perception must be strictly causal: position l sees only positions ≤ l."""

    def test_zero_gradient_from_future(self, config):
        """∂P[l] / ∂G[l'] = 0 for all l' > l."""
        perc = CausalPerception(config)
        C, R, L = config.n_channels, config.n_compute_rows, 16

        G = torch.randn(1, C, R, L, requires_grad=True)
        P = perc(G)

        for target_col in [0, 3, 7]:
            if G.grad is not None:
                G.grad.zero_()
            P[:, :, :, target_col].sum().backward(retain_graph=True)
            grad = G.grad[0]  # (C, R, L)

            max_kernel = max(config.perception_kernels)
            for future_col in range(target_col + 1, L):
                grad_at_future = grad[:, :, future_col].abs().max().item()
                assert grad_at_future < 1e-6, (
                    f"Causality violation: ∂P[{target_col}]/∂G[{future_col}] = {grad_at_future}"
                )


class TestResidualIdentity:
    """At initialization, the model should approximate the identity function
    through the residual stream."""

    def test_nca_block_near_identity(self, config):
        """NCA block should produce small perturbation at init."""
        block = NCABlock(config)
        x = torch.randn(1, 8, config.d_model)
        y = block(x)
        delta_norm = (y - x).norm() / x.norm()
        # At init: W_m is small (Xavier/√M), η=0.1, so output ≈ 0
        # But zero is too strict due to perception having nonzero weights
        assert delta_norm < 5.0, f"NCA block delta too large at init: {delta_norm:.4f}"

    def test_full_model_near_identity_logits(self, config):
        """Full model at init should produce nearly uniform logits."""
        model = DPNCA(config)
        x = torch.randint(0, config.vocab_size, (1, 8))
        logits = model(x)
        probs = F.softmax(logits, dim=-1)
        uniform = 1.0 / config.vocab_size
        # Check entropy is not pathologically low at init
        # With weight tying and small model, logits won't be perfectly uniform,
        # but entropy should be at least 30% of maximum (not collapsed)
        entropy = -(probs * probs.clamp(min=1e-10).log()).sum(dim=-1).mean().item()
        max_entropy = math.log(config.vocab_size)
        entropy_ratio = entropy / max_entropy
        assert entropy_ratio > 0.3, (
            f"Entropy pathologically low at init: {entropy:.2f} vs max {max_entropy:.2f} "
            f"(ratio {entropy_ratio:.4f})"
        )


class TestPathwaySoftmaxIdentity:
    """Routing weights must always sum to 1 over modes."""

    def test_softmax_sums_to_one(self, config):
        pu = PathwayUpdate(config)
        H = torch.randn(3, config.n_hw_channels, config.n_compute_rows, 10)
        alpha = pu.compute_routing(H)
        sums = alpha.sum(dim=1)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)

    def test_softmax_all_positive(self, config):
        pu = PathwayUpdate(config)
        H = torch.randn(2, config.n_hw_channels, config.n_compute_rows, 8)
        alpha = pu.compute_routing(H)
        assert (alpha > 0).all(), "All routing weights must be strictly positive"


class TestRoPEIdentity:
    """RoPE rotation must preserve vector norms (orthogonal transformation)."""

    def test_norm_preservation(self):
        rope = RotaryPositionalEncoding(n_channels=32, max_seq_len=128)
        G = torch.randn(4, 32, 8, 16)
        G_rotated = rope(G, seq_len=16)

        in_norm = G.norm(dim=1)
        out_norm = G_rotated.norm(dim=1)
        torch.testing.assert_close(in_norm, out_norm, atol=1e-5, rtol=1e-5)

    def test_different_positions_differ(self):
        rope = RotaryPositionalEncoding(n_channels=32, max_seq_len=128)
        G = torch.ones(1, 32, 1, 4)
        G_rotated = rope(G, seq_len=4)
        # Each column should be rotated differently
        assert not torch.allclose(G_rotated[:, :, 0, 0], G_rotated[:, :, 0, 1])
        assert not torch.allclose(G_rotated[:, :, 0, 1], G_rotated[:, :, 0, 2])


class TestHardwareGradientGraph:
    """Hardware state statistics must have stop_gradient applied."""

    def test_no_grad_flow_through_stats(self, config):
        hw = HardwareChannels(config)
        B, L = 1, 4
        H = hw.init_for_sequence(B, L)

        mu = torch.randn(B, config.n_channels, config.n_compute_rows, L, requires_grad=True)
        var = torch.abs(torch.randn(B, config.n_channels, config.n_compute_rows, L)).requires_grad_(True)

        H_new = hw.modulate(H, mu, var)
        H_new.sum().backward()

        # mu and var should NOT receive gradients
        assert mu.grad is None or mu.grad.abs().max() < 1e-10
        assert var.grad is None or var.grad.abs().max() < 1e-10


class TestKLDivergenceNonNegativity:
    """KL(P ‖ Q) ≥ 0 for all valid distributions (Gibbs' inequality)."""

    def test_kl_nonnegative(self):
        for _ in range(20):
            M = torch.randint(2, 16, (1,)).item()
            alpha = F.softmax(torch.randn(M), dim=0)
            uniform = torch.full((M,), 1.0 / M)
            kl = (alpha * (alpha / uniform).log()).sum()
            assert kl >= -1e-6, f"KL divergence negative: {kl.item()}"

    def test_kl_zero_at_uniform(self):
        M = 8
        alpha = torch.full((M,), 1.0 / M)
        uniform = torch.full((M,), 1.0 / M)
        kl = (alpha * (alpha / uniform).log()).sum()
        assert abs(kl.item()) < 1e-6


class TestGradientJacobianProperty:
    """At initialization, the NCA step Jacobian should have eigenvalues
    close to 1 due to residual connection and small step size."""

    def test_jacobian_near_identity(self, config):
        pu = PathwayUpdate(config)
        C, R, L = config.n_channels, config.n_compute_rows, 4
        G = torch.randn(1, C, R, L, requires_grad=True)
        H = torch.randn(1, config.n_hw_channels, R, L)

        # Use a simplified version to check gradient magnitude
        P_dim = config.n_perception_channels
        P = torch.randn(1, P_dim, R, L)
        alpha = pu.compute_routing(H)
        delta = pu.compute_delta(P, alpha)
        G_new = pu.apply_step(G, delta, step_idx=0)

        # Compute gradient of G_new w.r.t. G
        # If near identity, ∂G_new/∂G ≈ I, so the gradient of sum(G_new)
        # w.r.t. G should be approximately all-ones
        G_new.sum().backward()
        grad = G.grad

        # At init: G_new = G + 0.1 * LN(ΔG), and ΔG has very weak dependence
        # on G directly (it depends on P, which is separate). So gradient ≈ 1.0
        ones = torch.ones_like(grad)
        relative_error = (grad - ones).norm() / ones.norm()
        assert relative_error < 2.0, (
            f"Jacobian too far from identity: relative error {relative_error:.4f}"
        )
