"""Shared fixtures for V4 tests."""

import pytest
import torch

from dpnca.config import DPNCAConfig, DPNCA_TINY


@pytest.fixture
def tiny_config():
    """Minimal config for fast tests."""
    return DPNCAConfig(
        d_model=64,
        n_layers=2,
        n_compute_rows=4,
        n_channels=16,
        n_steps=3,
        n_modes=2,
        n_hw_channels=4,
        hw_mod_interval=2,
        perception_kernels=(3, 5),
        max_seq_len=64,
        vocab_size=256,
        dropout=0.0,
    )


@pytest.fixture
def device():
    return torch.device("cpu")
