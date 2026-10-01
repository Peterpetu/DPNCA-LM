"""DPNCA-LM: an attention-free language model built on an iterative neural
cellular automaton.

A pre-norm transformer skeleton in which the attention sublayer is replaced by
T steps of a causal 2D neural cellular automaton over an (R x L) grid of
C-channel cells. The only cross-position operator in the model is a set of
depthwise causal convolutions inside the NCA perception step; every other
component is strictly position-local.

Architecture overview:
    TokenEmbedding + RoPE -> [NCATransformerLayer x N_layers] -> LayerNorm -> tied OutputHead

Each NCATransformerLayer:
    PreNorm -> NCABlock (T causal NCA steps with a hardware-gated mixture of pathways) -> Residual
    PreNorm -> SwiGLU FFN -> Residual

The optional V5 modules (Hebbian associative memory, causal aggregation
channels, gated fusion) are present in this package but were disabled in the
reported 158.9M-parameter run; see README.md for the exact configuration and
the measured results.
"""

from dpnca.config import (
    DPNCAConfig,
    DPNCA_TINY, DPNCA_SMALL, DPNCA_BASE, DPNCA_LARGE,
    DPNCA_V5_TINY, DPNCA_V5_SMALL, DPNCA_V5_160M,
)
from dpnca.model import DPNCA

__all__ = [
    "DPNCAConfig",
    "DPNCA_TINY",
    "DPNCA_SMALL",
    "DPNCA_BASE",
    "DPNCA_LARGE",
    "DPNCA_V5_TINY",
    "DPNCA_V5_SMALL",
    "DPNCA_V5_160M",
    "DPNCA",
]
