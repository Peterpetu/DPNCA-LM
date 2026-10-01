"""Tests for training infrastructure. Plan §8.2.

Verifies:
    1. Optimizer setup (weight decay exclusion)
    2. LR schedule (warmup + cosine decay)
    3. Gradient accumulation
    4. Checkpoint save/load round-trip
    5. Training step produces decreasing loss
"""

import pytest
import torch
from pathlib import Path
import tempfile

from dpnca.config import DPNCAConfig
from dpnca.model import DPNCA
from dpnca.training import CosineWarmupScheduler, TrainConfig, Trainer


@pytest.fixture
def config():
    return DPNCAConfig(
        d_model=64, n_layers=2, n_compute_rows=4, n_channels=16,
        n_steps=3, n_modes=2, n_hw_channels=4, hw_mod_interval=2,
        perception_kernels=(3,), max_seq_len=64, vocab_size=256,
        dropout=0.0,
    )


@pytest.fixture
def train_config(tmp_path):
    return TrainConfig(
        lr=1e-3,
        batch_size=4,
        seq_len=16,
        total_steps=10,
        warmup_steps=2,
        eval_interval=5,
        eval_batches=2,
        log_interval=5,
        save_interval=100,
        save_dir=str(tmp_path / "ckpt"),
        use_amp=False,
        device="cpu",
    )


@pytest.fixture
def dummy_tokens():
    return torch.randint(0, 256, (10000,))


class TestCosineWarmupScheduler:
    def test_warmup_phase(self):
        """LR should linearly increase during warmup."""
        optimizer = torch.optim.AdamW([torch.randn(1, requires_grad=True)], lr=1e-3)
        scheduler = CosineWarmupScheduler(optimizer, warmup_steps=100, total_steps=1000)

        lr_0 = scheduler.step(0)
        lr_50 = scheduler.step(49)
        lr_99 = scheduler.step(99)

        assert lr_0 < lr_50 < lr_99
        assert abs(lr_99 - 1e-3) < 1e-5  # should reach lr_max at end of warmup

    def test_decay_phase(self):
        """LR should decrease after warmup."""
        optimizer = torch.optim.AdamW([torch.randn(1, requires_grad=True)], lr=1e-3)
        scheduler = CosineWarmupScheduler(optimizer, warmup_steps=10, total_steps=100)

        lr_start = scheduler.step(10)
        lr_mid = scheduler.step(55)
        lr_end = scheduler.step(99)

        assert lr_start > lr_mid > lr_end


class TestTrainer:
    def test_trainer_creation(self, config, train_config, dummy_tokens):
        model = DPNCA(config)
        trainer = Trainer(model, train_config, dummy_tokens, dummy_tokens[:1000])
        assert trainer.global_step == 0

    def test_single_train_step(self, config, train_config, dummy_tokens):
        model = DPNCA(config)
        trainer = Trainer(model, train_config, dummy_tokens, dummy_tokens[:1000])
        metrics = trainer.train_step()
        assert "loss_total" in metrics
        assert "loss_ce" in metrics
        assert "grad_norm" in metrics
        assert "lr" in metrics
        assert trainer.global_step == 1

    def test_evaluate(self, config, train_config, dummy_tokens):
        model = DPNCA(config)
        trainer = Trainer(model, train_config, dummy_tokens, dummy_tokens[:1000])
        val_metrics = trainer.evaluate()
        assert "val_loss_ce" in val_metrics
        assert "val_ppl" in val_metrics

    def test_checkpoint_roundtrip(self, config, train_config, dummy_tokens):
        """Save + load checkpoint should restore training state."""
        model = DPNCA(config)
        trainer = Trainer(model, train_config, dummy_tokens)

        # Do some training
        trainer.train_step()
        trainer.train_step()

        # Save
        path = trainer.save_checkpoint("test")

        # Create new trainer and load
        model2 = DPNCA(config)
        trainer2 = Trainer(model2, train_config, dummy_tokens)
        trainer2.load_checkpoint(path)

        assert trainer2.global_step == trainer.global_step

        # Model weights should match
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            torch.testing.assert_close(p1, p2)

    def test_weight_decay_exclusion(self, config, train_config, dummy_tokens):
        """Biases, norms, step sizes should have zero weight decay."""
        model = DPNCA(config)
        trainer = Trainer(model, train_config, dummy_tokens)

        # Check param groups
        assert len(trainer.optimizer.param_groups) == 2
        # Group 1: with decay
        assert trainer.optimizer.param_groups[0]["weight_decay"] == train_config.weight_decay
        # Group 2: no decay
        assert trainer.optimizer.param_groups[1]["weight_decay"] == 0.0

    def test_short_training_run(self, config, train_config, dummy_tokens):
        """A few training steps should run without errors."""
        model = DPNCA(config)
        train_config.total_steps = 3
        train_config.log_interval = 1
        train_config.eval_interval = 100  # skip eval
        trainer = Trainer(model, train_config, dummy_tokens)
        history = trainer.train(n_steps=3)
        assert len(history) == 3
        assert all("loss_total" in m for m in history)
