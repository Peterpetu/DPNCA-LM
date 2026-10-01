"""Tests for SwiGLU FFN. Plan §8.2.

Verifies:
    1. Output shape preservation
    2. Correct intermediate dimension
    3. Gradient flow
"""

import pytest
import torch

from dpnca.config import DPNCAConfig
from dpnca.ffn import SwiGLU


@pytest.fixture
def config():
    return DPNCAConfig(
        d_model=64, n_layers=1, n_compute_rows=4, n_channels=16,
        n_steps=1, n_modes=2, n_hw_channels=4,
        perception_kernels=(3,), max_seq_len=64, vocab_size=256,
        dropout=0.0,
    )


class TestSwiGLU:
    def test_output_shape(self, config):
        ffn = SwiGLU(config)
        x = torch.randn(2, 10, config.d_model)
        y = ffn(x)
        assert y.shape == x.shape

    def test_intermediate_dim(self, config):
        """d_ff should be ⌈8/3 × d_model⌉ rounded to 8."""
        ffn = SwiGLU(config)
        d_ff = config.d_ff
        assert ffn.w1.weight.shape == (d_ff, config.d_model)
        assert ffn.w3.weight.shape == (d_ff, config.d_model)
        assert ffn.w2.weight.shape == (config.d_model, d_ff)

    def test_gradient_flow(self, config):
        ffn = SwiGLU(config)
        x = torch.randn(1, 5, config.d_model, requires_grad=True)
        y = ffn(x)
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_no_bias(self, config):
        """SwiGLU should have no bias terms (bias=False)."""
        ffn = SwiGLU(config)
        assert ffn.w1.bias is None
        assert ffn.w2.bias is None
        assert ffn.w3.bias is None
