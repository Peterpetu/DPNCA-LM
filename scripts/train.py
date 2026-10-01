"""Train DPNCA-LM.

Usage:
    python scripts/train.py --config tiny --device cuda
    python scripts/train.py --config small --steps 50000
    python scripts/train.py --config tiny --profile smoke  # Quick smoke test
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dpnca.config import (
    DPNCA_BASE,
    DPNCA_LARGE,
    DPNCA_SMALL,
    DPNCA_TINY,
    DPNCA_160M,
    DPNCA_V5_TINY,
    DPNCA_V5_SMALL,
    DPNCA_V5_160M,
    DPNCAConfig,
)
from dpnca.diagnostics import model_summary
from dpnca.model import DPNCA
from dpnca.training import TrainConfig, Trainer
from dpnca.data.tokenizer import GPT2Tokenizer
from dpnca.data.dataset import CorpusData

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


CONFIG_MAP = {
    "tiny": DPNCA_TINY,
    "small": DPNCA_SMALL,
    "base": DPNCA_BASE,
    "160m": DPNCA_160M,
    "large": DPNCA_LARGE,
    "v5_tiny": DPNCA_V5_TINY,
    "v5_small": DPNCA_V5_SMALL,
    "v5_160m": DPNCA_V5_160M,
}

# Predefined training profiles
PROFILES = {
    "smoke": TrainConfig(
        lr=3e-4,
        batch_size=8,
        seq_len=128,
        total_steps=200,
        warmup_steps=20,
        eval_interval=50,
        log_interval=10,
        save_interval=100,
        grad_accum_steps=1,
        use_amp=True,
    ),
    "short": TrainConfig(
        lr=3e-4,
        batch_size=32,
        seq_len=256,
        total_steps=5000,
        warmup_steps=500,
        eval_interval=500,
        log_interval=50,
        save_interval=2500,
        grad_accum_steps=1,
        use_amp=True,
    ),
    "full": TrainConfig(
        lr=3e-4,
        batch_size=64,
        seq_len=512,
        total_steps=100_000,
        warmup_steps=2000,
        eval_interval=500,
        log_interval=50,
        save_interval=5000,
        grad_accum_steps=1,
        use_amp=True,
    ),
    "pile_250m": TrainConfig(
        lr=6e-4,
        batch_size=8,
        seq_len=1024,
        total_steps=7_630,
        warmup_steps=200,
        eval_interval=250,
        eval_batches=20,
        log_interval=10,
        save_interval=1000,
        grad_accum_steps=4,
        use_amp=True,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DPNCA-LM (V4)")
    parser.add_argument(
        "--config",
        choices=list(CONFIG_MAP.keys()),
        default="tiny",
        help="Model size configuration",
    )
    parser.add_argument(
        "--profile",
        choices=list(PROFILES.keys()),
        default="short",
        help="Training profile (smoke/short/full)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=None, help="Override total steps")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--seq-len", type=int, default=None, help="Override sequence length")
    parser.add_argument("--dataset", default="tinystories", help="Dataset name")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens to load from dataset")
    parser.add_argument("--grad-accum", type=int, default=None, help="Override gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")
    args = parser.parse_args()

    # Model config
    model_config = CONFIG_MAP[args.config]
    logger.info(f"Model config: DPNCA-{args.config.upper()}")

    # Training config
    train_config = PROFILES[args.profile]
    train_config.device = args.device
    train_config.save_dir = f"checkpoints/v4_{args.config}_{args.profile}"

    # Apply overrides
    if args.steps is not None:
        train_config.total_steps = args.steps
    if args.lr is not None:
        train_config.lr = args.lr
    if args.batch_size is not None:
        train_config.batch_size = args.batch_size
    if args.seq_len is not None:
        train_config.seq_len = args.seq_len
    if args.grad_accum is not None:
        train_config.grad_accum_steps = args.grad_accum

    # Seeding
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    logger.info(f"Random seed: {args.seed}")

    # Effective tokens per step
    eff_batch = train_config.batch_size * train_config.seq_len * train_config.grad_accum_steps
    total_tokens = eff_batch * train_config.total_steps
    logger.info(
        f"Effective batch: {eff_batch:,} tokens/step "
        f"({train_config.batch_size}×{train_config.seq_len}×{train_config.grad_accum_steps}), "
        f"total ~{total_tokens/1e6:.0f}M tokens over {train_config.total_steps} steps"
    )

    # Load data
    logger.info(f"Loading dataset: {args.dataset}")
    tokenizer = GPT2Tokenizer()
    corpus = CorpusData(args.dataset, tokenizer, max_tokens=args.max_tokens)
    logger.info(f"Train tokens: {len(corpus.train_tokens):,}, Val tokens: {len(corpus.val_tokens):,}")

    # Create model
    model = DPNCA(model_config)
    logger.info(model_summary(model))

    # Create trainer
    trainer = Trainer(
        model=model,
        train_config=train_config,
        train_tokens=corpus.train_tokens,
        val_tokens=corpus.val_tokens,
    )

    # Resume if requested
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    metrics_history = trainer.train()

    # Final summary
    logger.info(f"Training complete. Best val CE: {trainer.best_val_loss:.4f}")
    if metrics_history:
        final = metrics_history[-1]
        logger.info(
            f"Final metrics: ce={final.get('loss_ce', 0):.4f}, "
            f"ppl={final.get('ppl', 0):.1f}, "
            f"total={final.get('loss_total', 0):.4f}"
        )


if __name__ == "__main__":
    main()
