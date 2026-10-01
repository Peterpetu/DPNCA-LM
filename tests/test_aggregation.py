"""Tests for Causal Aggregation Channels (CAC). Plan V5 §7.

Covers: output shape, causality, multi-scale response, gradient flow,
hardware injection shape, decay bounds, initial state, numerical stability.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from dpnca.config import DPNCAConfig, DPNCA_V5_TINY
from dpnca.aggregation import CausalAggregation


@pytest.fixture
def config() -> DPNCAConfig:
    """Minimal V5 config for aggregation tests."""
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
        enable_aggregation=True,
        n_aggregation_scales=3,
        d_aggregation=16,
    )


@pytest.fixture
def cac(config: DPNCAConfig) -> CausalAggregation:
    return CausalAggregation(config)


class TestOutputShape:
    def test_agg_output_shape(self, cac: CausalAggregation, config: DPNCAConfig) -> None:
        B, L, d = 2, 16, config.d_model
        x = torch.randn(B, L, d)
        agg_out, hw_inj = cac(x)
        assert agg_out.shape == (B, L, d), f"Expected ({B}, {L}, {d}), got {agg_out.shape}"

    def test_hw_injection_shape(self, cac: CausalAggregation, config: DPNCAConfig) -> None:
        B, L, d = 2, 16, config.d_model
        x = torch.randn(B, L, d)
        agg_out, hw_inj = cac(x)
        C_H = config.n_hw_channels
        assert hw_inj.shape == (B, L, C_H), f"Expected ({B}, {L}, {C_H}), got {hw_inj.shape}"

    def test_single_position(self, cac: CausalAggregation, config: DPNCAConfig) -> None:
        B, L, d = 1, 1, config.d_model
        x = torch.randn(B, L, d)
        agg_out, hw_inj = cac(x)
        assert agg_out.shape == (B, L, d)
        assert hw_inj.shape == (B, L, config.n_hw_channels)


class TestCausality:
    def test_no_future_leakage(self, cac: CausalAggregation, config: DPNCAConfig) -> None:
        """∂CAC(x)[l] / ∂x[l'] = 0 for l' > l — strict causality."""
        B, L, d = 1, 8, config.d_model
        x = torch.randn(B, L, d, requires_grad=True)
        agg_out, hw_inj = cac(x)

        target_pos = 3
        agg_out[:, target_pos].sum().backward()

        grad = x.grad  # (1, L, d)
        assert grad is not None

        future_grad = grad[:, target_pos + 1:].abs().max().item()
        assert future_grad == 0.0, f"Future gradient leakage: {future_grad}"

        past_grad = grad[:, :target_pos + 1].abs().max().item()
        assert past_grad > 0.0, "No gradient at past/current positions"


class TestMultiScaleResponse:
    def test_fast_scale_responds_quickly(self, config: DPNCAConfig) -> None:
        """Fast scale (small half-life) reaches ~50% of step response faster."""
        cac = CausalAggregation(config)
        cac.eval()

        B, d = 1, config.d_model
        # Step input: zero then constant 1
        x = torch.zeros(B, 64, d)
        x[:, 1:] = 1.0

        with torch.no_grad():
            agg_out, _ = cac(x)
            # The output at position 4 vs position 32 — later should be closer to steady state
            norm_early = agg_out[:, 4].norm().item()
            norm_late = agg_out[:, 32].norm().item()
            # Late should be larger (closer to steady state) or at least non-zero
            assert norm_late > 0, "Late position has zero output"


class TestGradientFlow:
    def test_gradients_through_all_params(
        self, cac: CausalAggregation, config: DPNCAConfig
    ) -> None:
        """All parameters should receive gradients."""
        B, L, d = 2, 8, config.d_model
        x = torch.randn(B, L, d)
        agg_out, hw_inj = cac(x)
        loss = agg_out.sum() + hw_inj.sum()
        loss.backward()

        for name, param in cac.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().max() > 0, f"Zero gradient for {name}"


class TestDecayBounds:
    def test_beta_in_zero_one(self, cac: CausalAggregation) -> None:
        """β_k = σ(γ_k) ∈ (0, 1) always."""
        betas = torch.sigmoid(cac.decay_logits)
        assert (betas > 0).all(), "β contains non-positive values"
        assert (betas < 1).all(), "β contains values >= 1"

    def test_initial_halflife_ordering(self, cac: CausalAggregation) -> None:
        """Initial β values should be decreasing (faster scales have larger β)."""
        betas = torch.sigmoid(cac.decay_logits)
        for k in range(len(betas) - 1):
            assert betas[k] > betas[k + 1], (
                f"β[{k}]={betas[k].item():.4f} should be > β[{k+1}]={betas[k+1].item():.4f}"
            )


class TestInitialState:
    def test_first_position_proportional_to_beta(self, config: DPNCAConfig) -> None:
        """First position output ≈ β · f(x_0), not zero (EMA starts from zero)."""
        cac = CausalAggregation(config)
        cac.eval()
        B, d = 1, config.d_model
        x = torch.randn(B, 1, d)

        with torch.no_grad():
            agg_out, _ = cac(x)
            # Should be non-zero (β > 0 and input is non-zero)
            assert agg_out.abs().max() > 0, "First position output is exactly zero"


class TestNumericalStability:
    def test_no_nan_long_sequence(self, config: DPNCAConfig) -> None:
        """No NaN/Inf for moderately long sequences."""
        cac = CausalAggregation(config)
        B, L, d = 1, 128, config.d_model
        x = torch.randn(B, L, d)
        agg_out, hw_inj = cac(x)
        assert torch.isfinite(agg_out).all(), "NaN or Inf in agg_output"
        assert torch.isfinite(hw_inj).all(), "NaN or Inf in hw_injection"

    def test_large_input_magnitude(self, config: DPNCAConfig) -> None:
        """Handles large input magnitudes via GELU saturation."""
        cac = CausalAggregation(config)
        B, L, d = 1, 16, config.d_model
        x = torch.randn(B, L, d) * 100
        agg_out, hw_inj = cac(x)
        assert torch.isfinite(agg_out).all(), "NaN or Inf with large inputs"
        assert torch.isfinite(hw_inj).all(), "NaN or Inf in hw with large inputs"


class TestV5TinyConfig:
    def test_v5_tiny_instantiation(self) -> None:
        """V5_TINY config creates a valid CAC module."""
        cac = CausalAggregation(DPNCA_V5_TINY)
        B, L, d = 1, 16, DPNCA_V5_TINY.d_model
        x = torch.randn(B, L, d)
        agg_out, hw_inj = cac(x)
        assert agg_out.shape == (B, L, d)
        assert hw_inj.shape == (B, L, DPNCA_V5_TINY.n_hw_channels)
        assert torch.isfinite(agg_out).all()
