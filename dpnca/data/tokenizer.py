"""GPT-2 tokenizer wrapper using tiktoken.

Provides the minimal TokenizerInterface required by training and
chat scripts. Uses the GPT-2 BPE encoding (50257 tokens) via
tiktoken for fast, dependency-light tokenization.
"""

from __future__ import annotations

import tiktoken


class GPT2Tokenizer:
    """GPT-2 BPE tokenizer wrapping tiktoken.

    Properties:
        vocab_size:   50257
        eos_token_id: 50256 (<|endoftext|>)
    """

    def __init__(self) -> None:
        self._enc = tiktoken.get_encoding("gpt2")

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab  # 50257

    @property
    def eos_token_id(self) -> int:
        return self._enc.eot_token  # 50256

    def encode(self, text: str) -> list[int]:
        """Encode text to GPT-2 BPE token IDs."""
        return self._enc.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text."""
        return self._enc.decode(token_ids)
