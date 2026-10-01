"""Dataset loading and corpus iterator factory for DVTM training.

Supports TinyStories (HuggingFace) and Pile (monology/pile-uncopyrighted).
Tokenizes once with GPT-2 BPE, caches to disk as flat int32 tensor.
Training draws random windows — no sequential iteration, no epochs.

Modeled after §2 of TRAINING_INFRASTRUCTURE_BLUEPRINT.md.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterator

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class CorpusData:
    """Tokenized corpus cached as a flat int32 tensor.

    First call downloads + tokenizes (slow). Subsequent calls load
    from cache (fast).
    """

    MAX_TOKENS = 2_500_000_000  # Hard cap at 2.5B tokens

    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        max_tokens: int | None = None,
        cache_dir: str | None = None,
        cache_name: str | None = None,
    ) -> None:
        """
        Args:
            dataset_name: "tinystories" or "pile"
            tokenizer:    must have .encode(text) -> list[int]
            max_tokens:   cap total tokens (for Pile subsets)
            cache_dir:    directory for cached .pt file
            cache_name:   filename for cache, e.g. "tinystories_gpt2.pt"
        """
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens or self.MAX_TOKENS

        if cache_dir is None:
            cache_dir = str(Path(__file__).parent.parent.parent / "data_cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if cache_name is None:
            suffix = f"_{self.max_tokens // 1_000_000}M" if max_tokens else ""
            cache_name = f"{dataset_name}{suffix}_gpt2.pt"
        self._cache_name = cache_name

        self.tokens = self._load_data()
        self.n_tokens = len(self.tokens)

        # 90/10 train/val split by token position
        split_point = int(self.n_tokens * 0.9)
        self.train_tokens = self.tokens[:split_point].contiguous()
        self.val_tokens = self.tokens[split_point:].contiguous()

        logger.info(
            f"Dataset: {self.n_tokens:,} tokens "
            f"({self.tokens.element_size() * self.n_tokens / 1024**2:.0f}MB), "
            f"train={len(self.train_tokens):,}, val={len(self.val_tokens):,}"
        )

    def _load_data(self) -> Tensor:
        cache_file = self.cache_dir / self._cache_name

        if cache_file.exists():
            logger.info(f"Loading cached tokens from {cache_file.name}...")
            tokens = torch.load(cache_file, map_location="cpu", weights_only=True)
            if len(tokens) > self.max_tokens:
                tokens = tokens[: self.max_tokens].clone()
            if tokens.dtype == torch.int64:
                tokens = tokens.to(torch.int32)
            return tokens.contiguous()

        if self.dataset_name == "tinystories":
            return self._tokenize_tinystories(cache_file)
        elif self.dataset_name == "pile":
            return self._tokenize_pile(cache_file)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name!r}")

    def _tokenize_tinystories(self, cache_file: Path) -> Tensor:
        logger.info("Loading TinyStories from HuggingFace...")
        from datasets import load_dataset

        dataset = load_dataset(
            "roneneldan/TinyStories", split="train"
        )

        logger.info(f"Tokenizing {len(dataset):,} stories...")
        all_tokens: list[int] = []
        for i, story in enumerate(dataset):
            toks = self.tokenizer.encode(story["text"])
            all_tokens.extend(toks)
            if (i + 1) % 10000 == 0:
                logger.info(f"  {i + 1:,} stories ({len(all_tokens):,} tokens)")
            if len(all_tokens) >= self.max_tokens:
                all_tokens = all_tokens[: self.max_tokens]
                break

        tokens_tensor = torch.tensor(all_tokens, dtype=torch.int64)
        logger.info(f"Caching {len(tokens_tensor):,} tokens to {cache_file}...")
        torch.save(tokens_tensor, cache_file)

        return tokens_tensor.to(torch.int32).contiguous()

    def _tokenize_pile(self, cache_file: Path) -> Tensor:
        logger.info("Loading Pile (uncopyrighted) from HuggingFace (streaming)...")
        from datasets import load_dataset

        dataset = load_dataset(
            "monology/pile-uncopyrighted",
            split="train",
            streaming=True,
        )

        logger.info(f"Tokenizing up to {self.max_tokens:,} tokens...")
        all_tokens: list[int] = []
        count = 0
        for doc in dataset:
            toks = self.tokenizer.encode(doc["text"])
            all_tokens.extend(toks)
            count += 1
            if count % 10000 == 0:
                logger.info(f"  {count:,} docs ({len(all_tokens):,} tokens)")
            if len(all_tokens) >= self.max_tokens:
                all_tokens = all_tokens[: self.max_tokens]
                break

        tokens_tensor = torch.tensor(all_tokens, dtype=torch.int64)
        logger.info(f"Caching {len(tokens_tensor):,} tokens to {cache_file}...")
        torch.save(tokens_tensor, cache_file)

        return tokens_tensor.to(torch.int32).contiguous()


def create_corpus_iterator_factory(
    data: CorpusData,
    T_cycle: int,
    batch_size: int,
    embedding,
    device: torch.device,
    split: str = "train",
) -> Callable[[], Iterator[tuple[Tensor, Tensor]]]:
    """Create a factory that returns fresh corpus iterators.

    Each factory() call returns a new infinite iterator yielding
    (o_sequence, token_ids) tuples:
        o_sequence: (T_cycle, batch_size, N_v) dense embedded tensors
        token_ids:  (T_cycle, batch_size) long tensor of original token IDs

    The factory pattern is required by evolution.py — each individual
    in each generation gets a fresh iterator.

    Args:
        data:       CorpusData with tokenized corpus
        T_cycle:    tokens per sequence
        batch_size: sequences per batch
        embedding:  TokenEmbedding with .encode(token_ids) → (*, N_v)
        device:     target device
        split:      "train" or "val"

    Returns:
        Callable returning Iterator[tuple[Tensor, Tensor]]
    """
    split_tokens = data.train_tokens if split == "train" else data.val_tokens
    n = len(split_tokens)

    def factory() -> Iterator[tuple[Tensor, Tensor]]:
        def _iter():
            while True:
                # Random start positions (inclusive of final valid position)
                starts = torch.randint(0, n - T_cycle + 1, (batch_size,))
                # Gather (batch, T_cycle) windows of token IDs
                offsets = torch.arange(T_cycle, dtype=torch.long)
                indices = starts.unsqueeze(1) + offsets.unsqueeze(0)  # (B, T)
                token_ids = split_tokens[indices].to(dtype=torch.long)  # (B, T)

                # Embed: (B, T) → (B, T, N_v) → transpose to (T, B, N_v)
                with torch.no_grad():
                    embedded = embedding.encode(token_ids.to(device))  # (B, T, N_v)
                o_sequence = embedded.permute(1, 0, 2).contiguous()  # (T, B, N_v)

                # token_ids: (B, T) → (T, B) to match o_sequence layout
                token_ids_tb = token_ids.permute(1, 0).contiguous().to(device)

                yield o_sequence, token_ids_tb

        return _iter()

    return factory


def create_corpus_iterator_factory_v2(
    data: CorpusData,
    T_cycle: int,
    batch_size: int,
    device: torch.device,
    split: str = "train",
) -> Callable[[], Iterator[Tensor]]:
    """Create a factory for v2 corpus iterators (token IDs only).

    Returns iterators yielding (T_cycle, batch_size) long tensors.
    No embedding needed — InferenceNCA handles token I/O internally.

    Args:
        data:       CorpusData with tokenized corpus
        T_cycle:    tokens per sequence
        batch_size: sequences per batch
        device:     target device
        split:      "train" or "val"

    Returns:
        Callable returning Iterator[Tensor]
    """
    split_tokens = data.train_tokens if split == "train" else data.val_tokens
    n = len(split_tokens)

    def factory() -> Iterator[Tensor]:
        def _iter():
            while True:
                starts = torch.randint(0, n - T_cycle + 1, (batch_size,))
                offsets = torch.arange(T_cycle, dtype=torch.long)
                indices = starts.unsqueeze(1) + offsets.unsqueeze(0)
                token_ids = split_tokens[indices].to(dtype=torch.long)
                # (batch, T_cycle) → (T_cycle, batch)
                yield token_ids.permute(1, 0).contiguous().to(device)
        return _iter()

    return factory
