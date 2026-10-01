"""End-to-end training loop for DPNCA-LM. Plan §3.7, §4 Phase 7.

Standard PyTorch training loop with:
    - AdamW optimizer with cosine annealing + linear warmup
    - Gradient clipping (max_norm=1.0)
    - Mixed precision (AMP) with GradScaler
    - Gradient accumulation for large effective batch sizes
    - Periodic evaluation on validation set
    - Checkpoint saving (best + periodic)
    - Structured metric logging

Key hyperparameters (plan §4 Phase 7):
    lr = 3e-4, weight_decay = 0.1, betas = (0.9, 0.95)
    warmup_steps = 2000, grad_clip = 1.0
    batch_size = 64, seq_len = 512 (default)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import autocast, GradScaler

from dpnca.config import DPNCAConfig
from dpnca.model import DPNCA

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training hyperparameters. Plan §4 Phase 7."""

    # Optimization
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip: float = 1.0

    # Schedule
    warmup_steps: int = 2000
    total_steps: int = 100_000

    # Batching
    batch_size: int = 64
    seq_len: int = 512
    grad_accum_steps: int = 1

    # Evaluation
    eval_interval: int = 500
    eval_batches: int = 20

    # Checkpointing
    save_interval: int = 5000
    save_dir: str = "checkpoints/v4"

    # Mixed precision
    use_amp: bool = True

    # Logging
    log_interval: int = 50

    # Device
    device: str = "cuda"


class CosineWarmupScheduler:
    """Cosine annealing with linear warmup. Plan §3.7.

    lr(t) = lr_max × t / warmup_steps                  for t < warmup_steps
    lr(t) = lr_min + 0.5(lr_max - lr_min)(1 + cos(π·progress))  otherwise

    where progress = (t - warmup) / (total - warmup).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        lr_min_ratio: float = 0.1,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr_max = optimizer.param_groups[0]["lr"]
        self.lr_min = self.lr_max * lr_min_ratio

    def step(self, current_step: int) -> float:
        """Update learning rate and return current lr."""
        if current_step < self.warmup_steps:
            # Linear warmup
            lr = self.lr_max * (current_step + 1) / self.warmup_steps
        else:
            # Cosine decay
            progress = (current_step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            progress = min(progress, 1.0)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                1.0 + math.cos(math.pi * progress)
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


class Trainer:
    """DPNCA-LM Trainer. Plan §4 Phase 7.

    Handles the complete training pipeline: optimizer setup, LR scheduling,
    gradient accumulation, mixed precision, evaluation, and checkpointing.
    """

    def __init__(
        self,
        model: DPNCA,
        train_config: TrainConfig,
        train_tokens: Tensor,
        val_tokens: Tensor | None = None,
    ) -> None:
        self.model = model
        self.tc = train_config
        self.train_tokens = train_tokens
        self.val_tokens = val_tokens
        self.device = torch.device(train_config.device)

        # Move model to device
        self.model.to(self.device)

        # Optimizer
        self.optimizer = self._build_optimizer()

        # LR scheduler
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=train_config.warmup_steps,
            total_steps=train_config.total_steps,
        )

        # Mixed precision
        self._device_type = "cuda" if self.device.type == "cuda" else "cpu"
        self.scaler = GradScaler(self._device_type, enabled=train_config.use_amp)

        # Checkpoint directory
        self.save_dir = Path(train_config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.metrics_history: list[dict[str, float]] = []

    def _build_optimizer(self) -> torch.optim.AdamW:
        """Build AdamW optimizer with weight decay exclusion for norms/biases."""
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # No weight decay on bias, LayerNorm, step sizes, temperature
            if (
                "bias" in name
                or "norm" in name
                or "step_size" in name
                or "T_soft" in name
                or "H_init" in name
            ):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.tc.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return torch.optim.AdamW(
            param_groups,
            lr=self.tc.lr,
            betas=self.tc.betas,
            eps=self.tc.eps,
        )

    def _get_batch(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Sample random windows from token tensor. Plan §4 Phase 7.3.

        Args:
            tokens: (N_total,) flat token tensor

        Returns:
            input_ids: (B, seq_len)
            target_ids: (B, seq_len)
        """
        B = self.tc.batch_size
        L = self.tc.seq_len
        max_start = len(tokens) - L - 1

        # Random start positions
        starts = torch.randint(0, max_start, (B,))

        input_ids = torch.stack([tokens[s : s + L] for s in starts])
        target_ids = torch.stack([tokens[s + 1 : s + L + 1] for s in starts])

        return input_ids.long().to(self.device), target_ids.long().to(self.device)

    def train_step(self) -> dict[str, float]:
        """Single training step with gradient accumulation.

        Returns:
            metrics dict with loss components
        """
        self.model.train()
        accumulated_metrics: dict[str, float] = {}
        n_accum = self.tc.grad_accum_steps

        for micro_step in range(n_accum):
            input_ids, target_ids = self._get_batch(self.train_tokens)

            with autocast(device_type=self._device_type, enabled=self.tc.use_amp):
                loss, metrics = self.model.get_loss(input_ids, target_ids)
                # Scale loss for gradient accumulation
                scaled_loss = loss / n_accum

            self.scaler.scale(scaled_loss).backward()

            # Accumulate metrics
            for k, v in metrics.items():
                val = v.item() if isinstance(v, Tensor) else v
                accumulated_metrics[k] = accumulated_metrics.get(k, 0.0) + val / n_accum

        # Gradient clipping (unscale first for correct norm computation)
        self.scaler.unscale_(self.optimizer)
        grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.tc.grad_clip
        )
        accumulated_metrics["grad_norm"] = grad_norm.item()

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        # LR step
        lr = self.scheduler.step(self.global_step)
        accumulated_metrics["lr"] = lr

        self.global_step += 1
        return accumulated_metrics

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Evaluate on validation set. Returns average metrics."""
        if self.val_tokens is None or len(self.val_tokens) < self.tc.seq_len + 1:
            return {}

        self.model.eval()
        total_metrics: dict[str, float] = {}
        n_batches = self.tc.eval_batches

        for _ in range(n_batches):
            input_ids, target_ids = self._get_batch(self.val_tokens)

            with autocast(device_type=self._device_type, enabled=self.tc.use_amp):
                _, metrics = self.model.get_loss(input_ids, target_ids)

            for k, v in metrics.items():
                val = v.item() if isinstance(v, Tensor) else v
                total_metrics[k] = total_metrics.get(k, 0.0) + val / n_batches

        return {f"val_{k}": v for k, v in total_metrics.items()}

    def save_checkpoint(self, tag: str = "latest") -> Path:
        """Save model + optimizer + scheduler state."""
        path = self.save_dir / f"dpnca_{tag}.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "global_step": self.global_step,
                "best_val_loss": self.best_val_loss,
                "config": self.model.config,
                "train_config": self.tc,
            },
            path,
        )
        logger.info(f"Checkpoint saved: {path}")
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model + optimizer + scheduler state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        logger.info(f"Loaded checkpoint from step {self.global_step}")

    def train(self, n_steps: int | None = None) -> list[dict[str, float]]:
        """Full training loop.

        Args:
            n_steps: number of steps (defaults to train_config.total_steps)

        Returns:
            List of all metrics dicts
        """
        steps = n_steps or self.tc.total_steps
        start_step = self.global_step
        end_step = start_step + steps

        logger.info(
            f"Starting training: steps {start_step}→{end_step}, "
            f"batch_size={self.tc.batch_size}, seq_len={self.tc.seq_len}, "
            f"lr={self.tc.lr}, device={self.device}"
        )

        # Log parameter count
        param_counts = self.model.count_parameters()
        logger.info(f"Model parameters: {param_counts['total']:,} total")

        t_start = time.time()
        tokens_processed = 0

        while self.global_step < end_step:
            step_metrics = self.train_step()
            tokens_processed += self.tc.batch_size * self.tc.seq_len * self.tc.grad_accum_steps

            # Logging
            if self.global_step % self.tc.log_interval == 0:
                elapsed = time.time() - t_start
                tok_per_sec = tokens_processed / max(elapsed, 1e-6)
                step_metrics["tok_per_sec"] = tok_per_sec

                ce_str = f"ce={step_metrics.get('loss_ce', 0):.4f}"
                ppl_str = f"ppl={step_metrics.get('ppl', 0):.1f}"
                lr_str = f"lr={step_metrics.get('lr', 0):.2e}"
                grad_str = f"grad={step_metrics.get('grad_norm', 0):.2f}"
                speed_str = f"tok/s={tok_per_sec:.0f}"
                aux_path = step_metrics.get('loss_pathway_balance', 0)
                aux_hw = step_metrics.get('loss_hw_smooth', 0)
                aux_str = f"aux(path={aux_path:.3f},hw={aux_hw:.3f})"

                logger.info(
                    f"[step {self.global_step}] {ce_str} {ppl_str} "
                    f"{lr_str} {grad_str} {aux_str} {speed_str}"
                )

            # Evaluation
            if (
                self.global_step % self.tc.eval_interval == 0
                and self.global_step > start_step
            ):
                val_metrics = self.evaluate()
                step_metrics.update(val_metrics)

                val_loss = val_metrics.get("val_loss_ce", float("inf"))
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best")
                    logger.info(f"New best val_ce: {val_loss:.4f}")

                val_ppl = val_metrics.get("val_ppl", 0)
                logger.info(f"[eval step {self.global_step}] val_ce={val_loss:.4f} val_ppl={val_ppl:.1f}")

            # Periodic checkpoint
            if self.global_step % self.tc.save_interval == 0 and self.global_step > start_step:
                self.save_checkpoint(f"step_{self.global_step}")

            self.metrics_history.append(step_metrics)

        # Final checkpoint
        self.save_checkpoint("final")
        total_time = time.time() - t_start
        logger.info(
            f"Training complete: {steps} steps in {total_time:.1f}s, "
            f"best val_ce={self.best_val_loss:.4f}"
        )

        return self.metrics_history
