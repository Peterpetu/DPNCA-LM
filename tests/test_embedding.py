"""Tests for embedding and RoPE. Plan §8.2.

Verifies:
    1. Embedding shape and initialization
    2. RoPE is correctly applied to channel pairs
    3. RoPE rotation is position-dependent
    4. All rows at the same column get identical rotations
    5. Weight-tied logit projection
"""

import math

import pytest
import torch

from dpnca.config import DPNCAConfig
from dpnca.embedding import RotaryPositionalEncoding, TokenEmbedding


@pytest.fixture
def config():
    return DPNCAConfig(
        d_model=64, n_layers=1, n_compute_rows=4, n_channels=16,
        n_steps=1, n_modes=2, n_hw_channels=4,
        perception_kernels=(3,), max_seq_len=64, vocab_size=256,
    )


class TestTokenEmbedding:
    def test_forward_shape(self, config):
        emb = TokenEmbedding(config)
        ids = torch.randint(0, config.vocab_size, (2, 10))
        out = emb(ids)
        assert out.shape == (2, 10, config.d_model)

    def test_weight_property(self, config):
        emb = TokenEmbedding(config)
        assert emb.weight.shape == (config.vocab_size, config.d_model)

    def test_logit_projection(self, config):
        emb = TokenEmbedding(config)
        h = torch.randn(1, 5, config.d_model)
        logits = emb.project_to_logits(h)
        assert logits.shape == (1, 5, config.vocab_size)

    def test_init_std(self, config):
        """Embedding should be initialized with std ≈ 1/√d."""
        emb = TokenEmbedding(config)
        expected_std = 1.0 / math.sqrt(config.d_model)
        actual_std = emb.weight.std().item()
        assert abs(actual_std - expected_std) < 0.1 * expected_std


class TestRotaryPositionalEncoding:
    def test_output_shape(self):
        rope = RotaryPositionalEncoding(n_channels=16, max_seq_len=64)
        G = torch.randn(2, 16, 4, 8)
        out = rope(G, seq_len=8)
        assert out.shape == G.shape

    def test_position_dependent(self):
        """Different positions should get different rotations."""
        rope = RotaryPositionalEncoding(n_channels=16, max_seq_len=64)
        G = torch.ones(1, 16, 4, 8)
        out = rope(G, seq_len=8)
        # Different columns should have different values
        assert not torch.allclose(out[:, :, 0, 0], out[:, :, 0, 1])

    def test_same_rotation_across_rows(self):
        """All rows at the same column must receive identical rotations. Plan §9.2.3."""
        rope = RotaryPositionalEncoding(n_channels=16, max_seq_len=64)
        # Use input where all rows at same column are identical
        G = torch.randn(1, 16, 1, 8).expand(1, 16, 4, 8).clone()
        out = rope(G, seq_len=8)
        # All rows should still be identical (same rotation applied)
        for r in range(1, 4):
            torch.testing.assert_close(out[:, :, 0, :], out[:, :, r, :])

    def test_cache_extension(self):
        """Cache should extend for sequences longer than initial max."""
        rope = RotaryPositionalEncoding(n_channels=16, max_seq_len=32)
        G = torch.randn(1, 16, 4, 64)
        out = rope(G, seq_len=64)  # should auto-extend cache
        assert out.shape == G.shape

    def test_orthogonal_rotation(self):
        """RoPE should preserve vector norm (rotation is orthogonal)."""
        rope = RotaryPositionalEncoding(n_channels=16, max_seq_len=64)
        G = torch.randn(1, 16, 4, 8)
        out = rope(G, seq_len=8)
        # Norm should be approximately preserved at each spatial position
        in_norm = G.norm(dim=1)   # (1, 4, 8)
        out_norm = out.norm(dim=1)
        torch.testing.assert_close(in_norm, out_norm, atol=1e-5, rtol=1e-5)
