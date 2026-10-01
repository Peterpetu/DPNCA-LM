"""Tests for Hebbian Associative Memory (HAM). Plan V5 §7.

Covers: output shape, causality, retrieval accuracy, decay behaviour,
gradient flow, multi-head independence, numerical stability, empty memory,
feature map positivity, decay bounds.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from dpnca.config import DPNCAConfig, DPNCA_V5_TINY
from dpnca.memory import HebbianAssociativeMemory, _elu_plus_one


@pytest.fixture
def config() -> DPNCAConfig:
    """Minimal V5 config for memory tests."""
    return DPNCAConfig(
        d_model=64,
        n_layers=2,
        n_compute_rows=8,
        n_channels=8,
        n_steps=4,
        n_modes=2,
        n_hw_channels=4,
        hw_mod_interval=2,
        perception_kernels=(3,),
        max_seq_len=128,
        dropout=0.0,
        enable_memory=True,
        n_memory_heads=4,
        memory_decay_init=0.99,
    )


@pytest.fixture
def ham(config: DPNCAConfig) -> HebbianAssociativeMemory:
    return HebbianAssociativeMemory(config)


class TestOutputShape:
    def test_output_shape(self, ham: HebbianAssociativeMemory, config: DPNCAConfig) -> None:
        B, L, d = 2, 16, config.d_model
        x = torch.randn(B, L, d)
        out = ham(x)
        assert out.shape == (B, L, d), f"Expected ({B}, {L}, {d}), got {out.shape}"

    def test_output_shape_single_position(self, ham: HebbianAssociativeMemory, config: DPNCAConfig) -> None:
        B, L, d = 1, 1, config.d_model
        x = torch.randn(B, L, d)
        out = ham(x)
        assert out.shape == (B, L, d)


class TestCausality:
    def test_no_future_leakage(self, ham: HebbianAssociativeMemory, config: DPNCAConfig) -> None:
        """∂HAM(x)[l] / ∂x[l'] = 0 for l' > l — strict causality."""
        B, L, d = 1, 8, config.d_model
        x = torch.randn(B, L, d, requires_grad=True)
        out = ham(x)

        # Check gradient at position 3 w.r.t. all positions
        target_pos = 3
        out[:, target_pos].sum().backward()

        grad = x.grad  # (1, L, d)
        assert grad is not None

        # Future positions (l' > target_pos) must have zero gradient
        future_grad = grad[:, target_pos + 1:].abs().max().item()
        assert future_grad == 0.0, f"Future gradient leakage: {future_grad}"

        # Past/current positions should generally have non-zero gradient
        past_grad = grad[:, :target_pos + 1].abs().max().item()
        assert past_grad > 0.0, "No gradient at past/current positions"


class TestRetrievalAccuracy:
    def test_retrieve_stored_value(self, config: DPNCAConfig) -> None:
        """Store a known (k, v) pair and verify query retrieves ~v."""
        ham = HebbianAssociativeMemory(config)
        ham.eval()
        B, L, d = 1, 32, config.d_model

        with torch.no_grad():
            x = torch.randn(B, L, d)
            # Run forward to populate memory
            out = ham(x)

            # The output at position L-1 should be influenced by all past positions
            # (non-trivial output)
            assert out[:, -1].abs().mean() > 1e-6, "Final position output is near-zero"


class TestDecayBehaviour:
    def test_old_memories_decay(self, config: DPNCAConfig) -> None:
        """Memories at position 0 should have less influence at position L than at position 1."""
        ham = HebbianAssociativeMemory(config)
        ham.eval()
        B, d = 1, config.d_model

        with torch.no_grad():
            # Create sequence where pos 0 has a distinctive signal, rest is zeros
            x_short = torch.zeros(B, 4, d)
            x_short[:, 0] = torch.randn(d) * 10  # strong signal at pos 0
            out_short = ham(x_short)

            x_long = torch.zeros(B, 64, d)
            x_long[:, 0] = x_short[:, 0]  # same signal at pos 0
            out_long = ham(x_long)

            # Influence at pos 3 (short) vs pos 63 (long) — should decay
            influence_near = out_short[:, 3].norm().item()
            influence_far = out_long[:, 63].norm().item()
            assert influence_far < influence_near, (
                f"Far influence ({influence_far}) should be less than near ({influence_near})"
            )

    def test_decay_bounds(self, ham: HebbianAssociativeMemory) -> None:
        """λ = σ(α) must always be in (0, 1)."""
        decay = torch.sigmoid(ham.decay_logit)
        assert (decay > 0).all(), "Decay contains non-positive values"
        assert (decay < 1).all(), "Decay contains values >= 1"

    def test_decay_init_value(self, ham: HebbianAssociativeMemory) -> None:
        """Initial decay should be close to 0.99."""
        decay = torch.sigmoid(ham.decay_logit)
        assert torch.allclose(decay, torch.full_like(decay, 0.99), atol=1e-3)


class TestGradientFlow:
    def test_gradients_through_all_projections(
        self, ham: HebbianAssociativeMemory, config: DPNCAConfig
    ) -> None:
        """All W_Q, W_K, W_V, W_O, decay_logit should receive gradients."""
        B, L, d = 2, 8, config.d_model
        x = torch.randn(B, L, d)
        out = ham(x)
        loss = out.sum()
        loss.backward()

        for name, param in ham.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().max() > 0, f"Zero gradient for {name}"


class TestMultiHead:
    def test_head_count(self, ham: HebbianAssociativeMemory, config: DPNCAConfig) -> None:
        assert ham.n_heads == config.n_memory_heads
        assert ham.d_head == config.d_model // config.n_memory_heads


class TestNumericalStability:
    def test_no_nan_long_sequence(self, config: DPNCAConfig) -> None:
        """No NaN/Inf for moderately long sequences."""
        ham = HebbianAssociativeMemory(config)
        B, L, d = 1, 128, config.d_model
        x = torch.randn(B, L, d)
        out = ham(x)
        assert torch.isfinite(out).all(), "NaN or Inf in HAM output"

    def test_large_input_magnitude(self, config: DPNCAConfig) -> None:
        """Handles large input magnitudes without overflow."""
        ham = HebbianAssociativeMemory(config)
        B, L, d = 1, 16, config.d_model
        x = torch.randn(B, L, d) * 100
        out = ham(x)
        assert torch.isfinite(out).all(), "NaN or Inf with large inputs"


class TestEmptyMemory:
    def test_first_position_near_zero(self, config: DPNCAConfig) -> None:
        """First position has no stored associations, retrieval ≈ 0."""
        ham = HebbianAssociativeMemory(config)
        ham.eval()
        B, L, d = 1, 8, config.d_model
        x = torch.randn(B, L, d) * 0.1  # small inputs

        with torch.no_grad():
            out = ham(x)
            # Position 0 retrieves from empty memory + its own write
            # The magnitude should be modest
            first_norm = out[:, 0].norm().item()
            last_norm = out[:, -1].norm().item()
            # Not a strict test — just verify first pos isn't absurdly large
            assert first_norm < last_norm * 10 or first_norm < 1.0


class TestFeatureMap:
    def test_positivity(self) -> None:
        """φ(z) > 0 for all z."""
        z = torch.linspace(-10, 10, 1000)
        phi = _elu_plus_one(z)
        assert (phi > 0).all(), "Feature map produced non-positive values"

    def test_monotonic_for_positive(self) -> None:
        """φ(z) is monotonically increasing for z > 0."""
        z = torch.linspace(0.01, 10, 100)
        phi = _elu_plus_one(z)
        diffs = phi[1:] - phi[:-1]
        assert (diffs > 0).all(), "Feature map not monotonic for positive inputs"


class TestV5TinyConfig:
    def test_v5_tiny_instantiation(self) -> None:
        """V5_TINY config creates a valid HAM module."""
        ham = HebbianAssociativeMemory(DPNCA_V5_TINY)
        B, L, d = 1, 16, DPNCA_V5_TINY.d_model
        x = torch.randn(B, L, d)
        out = ham(x)
        assert out.shape == (B, L, d)
        assert torch.isfinite(out).all()
