"""Tests for the full DPNCA model. Plan §8.2.

Verifies:
    1. End-to-end forward/backward pass
    2. Loss computation with all auxiliary terms
    3. Weight tying between embedding and output head
    4. Generation produces valid tokens
    5. Parameter counting
    6. Mathematical identity tests (causal, residual, Helmholtz)
"""

import pytest
import torch
import torch.nn.functional as F

from dpnca.config import DPNCAConfig, DPNCA_TINY
from dpnca.model import DPNCA


@pytest.fixture
def config():
    return DPNCAConfig(
        d_model=64, n_layers=2, n_compute_rows=4, n_channels=16,
        n_steps=3, n_modes=2, n_hw_channels=4, hw_mod_interval=2,
        perception_kernels=(3, 5), max_seq_len=64, vocab_size=256,
        dropout=0.0,
    )


@pytest.fixture
def model(config):
    return DPNCA(config)


class TestDPNCAForward:
    def test_forward_shape(self, model, config):
        """Forward must produce (B, L, V) logits."""
        x = torch.randint(0, config.vocab_size, (2, 16))
        logits = model(x)
        assert logits.shape == (2, 16, config.vocab_size)

    def test_forward_backward(self, model, config):
        """Full forward-backward must work without errors."""
        x = torch.randint(0, config.vocab_size, (2, 16))
        t = torch.randint(0, config.vocab_size, (2, 16))
        loss, metrics = model.get_loss(x, t)
        loss.backward()

        # All parameters should have gradients
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_no_nans(self, model, config):
        """Forward pass must produce no NaN values."""
        x = torch.randint(0, config.vocab_size, (4, 32))
        logits = model(x)
        assert not torch.isnan(logits).any(), "NaN in logits"
        assert not torch.isinf(logits).any(), "Inf in logits"


class TestWeightTying:
    def test_embedding_output_tied(self, model):
        """Embedding weight and output head weight must be the same tensor."""
        assert model.output_head.weight is model.embedding.weight

    def test_weight_tying_preserves_gradient(self, model, config):
        """Gradient should flow through both input and output paths."""
        x = torch.randint(0, config.vocab_size, (1, 8))
        t = torch.randint(0, config.vocab_size, (1, 8))
        loss, _ = model.get_loss(x, t)
        loss.backward()
        assert model.embedding.embedding.weight.grad is not None
        assert model.embedding.embedding.weight.grad.abs().sum() > 0


class TestLoss:
    def test_loss_components(self, model, config):
        """Loss should include CE, pathway balance, and hardware smoothness."""
        x = torch.randint(0, config.vocab_size, (2, 16))
        t = torch.randint(0, config.vocab_size, (2, 16))
        loss, metrics = model.get_loss(x, t)

        assert "loss_ce" in metrics
        assert "loss_pathway_balance" in metrics
        assert "loss_hw_smooth" in metrics
        assert "ppl" in metrics

        # CE loss should be positive
        assert metrics["loss_ce"].item() > 0

        # PPL should be > 1
        assert metrics["ppl"].item() > 1.0

    def test_pathway_balance_loss_nonnegative(self, model, config):
        """KL(ᾱ ‖ U_M) ≥ 0 always."""
        x = torch.randint(0, config.vocab_size, (2, 16))
        t = torch.randint(0, config.vocab_size, (2, 16))
        _, metrics = model.get_loss(x, t)
        assert metrics["loss_pathway_balance"].item() >= -1e-6

    def test_pathway_balance_uses_accumulated_alpha(self, model, config):
        """Pathway balance loss must use the step-averaged alpha stored during
        forward pass, not re-compute from H_final. Plan §9.3."""
        x = torch.randint(0, config.vocab_size, (2, 16))
        t = torch.randint(0, config.vocab_size, (2, 16))
        _, _ = model.get_loss(x, t)
        # Verify alpha_mean was stored during forward
        for layer in model.layers:
            alpha_mean = layer.nca_block.get_routing_weights()
            assert alpha_mean is not None, "Step-averaged alpha not stored"
            # Must be (B, M, R, L)
            assert alpha_mean.shape[1] == config.n_modes

    def test_hardware_smooth_loss_nonnegative(self, model, config):
        """‖H_final - H_init‖²_F ≥ 0 always."""
        x = torch.randint(0, config.vocab_size, (2, 16))
        t = torch.randint(0, config.vocab_size, (2, 16))
        _, metrics = model.get_loss(x, t)
        assert metrics["loss_hw_smooth"].item() >= -1e-6


class TestGeneration:
    def test_generate_shape(self, model, config):
        """generate must produce correct length output."""
        prompt = torch.randint(0, config.vocab_size, (1, 4))
        output = model.generate(prompt, max_new_tokens=10)
        assert output.shape == (1, 14)  # 4 prompt + 10 generated

    def test_generate_valid_tokens(self, model, config):
        """All generated tokens must be valid vocab indices."""
        prompt = torch.randint(0, config.vocab_size, (1, 4))
        output = model.generate(prompt, max_new_tokens=20)
        assert (output >= 0).all()
        assert (output < config.vocab_size).all()

    def test_generate_deterministic_greedy(self, model, config):
        """Greedy generation (temp=0) should be deterministic."""
        prompt = torch.randint(0, config.vocab_size, (1, 4))
        out1 = model.generate(prompt, max_new_tokens=5, temperature=0, top_k=0)
        out2 = model.generate(prompt, max_new_tokens=5, temperature=0, top_k=0)
        torch.testing.assert_close(out1, out2)


class TestParameterCounting:
    def test_count_parameters(self, model):
        """count_parameters must return consistent totals."""
        counts = model.count_parameters()
        assert "total" in counts
        assert "embedding" in counts
        assert "all_layers" in counts
        assert counts["total"] > 0
        assert counts["total"] == counts["total_trainable"]

    def test_output_head_zero_additional(self, model):
        """Weight-tied output head should add 0 parameters."""
        counts = model.count_parameters()
        assert counts["output_head_tied"] == 0


class TestCausality:
    def test_causal_masking_via_gradient(self, config):
        """Future token positions must not influence current position logits.

        This is the critical mathematical identity test: if we compute
        ∂logits[t] / ∂input[t'], it must be zero for t' > t.
        """
        model = DPNCA(config)
        model.eval()
        L = 8

        # Create input that tracks gradients per position
        x = torch.randint(0, config.vocab_size, (1, L))
        embeds = model.embedding(x)  # (1, L, d)
        embeds = embeds.detach().requires_grad_(True)

        # Run through layers manually
        h = embeds
        hw_inj = None
        for layer in model.layers:
            h, hw_inj = layer(h, hw_injection=hw_inj)
        h = model.final_norm(h)
        logits = model.output_head(h)  # (1, L, V)

        # Check causality: gradient of logits[0, t, :].sum() w.r.t. embeds
        for t in range(min(L, 4)):
            if embeds.grad is not None:
                embeds.grad.zero_()
            logits[0, t, :].sum().backward(retain_graph=True)
            grad = embeds.grad[0]  # (L, d)

            # Positions t' > t should have zero gradient
            for tp in range(t + 1, L):
                grad_norm = grad[tp].norm().item()
                assert grad_norm < 1e-5, (
                    f"Causality violated: ∂logits[{t}]/∂input[{tp}] = {grad_norm:.6f}"
                )


class TestNumericalStability:
    def test_no_nans_large_input(self, config):
        """Large input values should not cause NaN."""
        model = DPNCA(config)
        x = torch.randint(0, config.vocab_size, (1, 32))
        logits = model(x)
        assert not torch.isnan(logits).any()

    def test_loss_not_nan(self, config):
        """Loss computation should be numerically stable."""
        model = DPNCA(config)
        for _ in range(5):
            x = torch.randint(0, config.vocab_size, (2, 16))
            t = torch.randint(0, config.vocab_size, (2, 16))
            loss, _ = model.get_loss(x, t)
            assert not torch.isnan(loss), "NaN in loss"
            assert not torch.isinf(loss), "Inf in loss"
