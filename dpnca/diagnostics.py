"""Diagnostics for DPNCA-LM. Plan §4 Phase 8.

Provides diagnostic metrics for monitoring model health during training:
    - pathway_usage_entropy: measures pathway mode balance
    - hardware_drift: measures hardware adaptation magnitude
    - effective_receptive_field: estimates information propagation distance
    - nca_convergence_rate: measures NCA state convergence speed
    - helmholtz_energy: computes full Helmholtz decomposition
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from dpnca.model import DPNCA


@torch.no_grad()
def pathway_usage_entropy(model: DPNCA, input_ids: Tensor) -> dict[str, float]:
    """Measure pathway routing entropy across layers. Plan §8.2.

    Higher entropy = more balanced usage (good).
    Max entropy = ln(M) when routing is uniform.
    Zero entropy = complete collapse to one mode (bad).

    Args:
        model: DPNCA model (after forward pass or will run one)
        input_ids: (B, L) — sample batch

    Returns:
        Dict with per-layer entropy and average
    """
    model.eval()
    _ = model(input_ids)

    results: dict[str, float] = {}
    M = model.config.n_modes
    max_entropy = math.log(M)

    for i, layer in enumerate(model.layers):
        # Use step-averaged alpha accumulated during forward pass
        alpha_mean = layer.nca_block.get_routing_weights()
        if alpha_mean is not None:
            alpha_bar = alpha_mean.mean(dim=(0, 2, 3))  # (M,)
            alpha_bar = alpha_bar.clamp(min=1e-8)
            entropy = -(alpha_bar * alpha_bar.log()).sum().item()
            results[f"layer_{i}_entropy"] = entropy
            results[f"layer_{i}_entropy_ratio"] = entropy / max_entropy

    if results:
        entropy_values = [v for k, v in results.items() if k.endswith("_entropy") and "ratio" not in k]
        results["mean_entropy"] = sum(entropy_values) / len(entropy_values)
        results["mean_entropy_ratio"] = results["mean_entropy"] / max_entropy

    return results


@torch.no_grad()
def hardware_drift(model: DPNCA, input_ids: Tensor) -> dict[str, float]:
    """Measure total hardware drift (H_final - H_init) per layer.

    Args:
        model: DPNCA model (will run forward)
        input_ids: (B, L) — sample batch

    Returns:
        Dict with per-layer Frobenius norm of drift
    """
    model.eval()
    _ = model(input_ids)

    results: dict[str, float] = {}
    for i, layer in enumerate(model.layers):
        hw = layer.nca_block.hardware
        if hasattr(layer.nca_block, "_H_final") and hw._H_initial is not None:
            drift = (layer.nca_block._H_final - hw._H_initial).norm().item()
            results[f"layer_{i}_hw_drift"] = drift

    if results:
        drifts = list(results.values())
        results["mean_hw_drift"] = sum(drifts) / len(drifts)

    return results


@torch.no_grad()
def effective_receptive_field(
    model: DPNCA,
    input_ids: Tensor,
    layer_idx: int = 0,
    target_pos: int = -1,
) -> Tensor:
    """Estimate effective receptive field via gradient magnitude.

    Computes ∂output[target_pos] / ∂input[all_pos] to see how much
    each input position contributes to the target position's output.

    Args:
        model: DPNCA model
        input_ids: (1, L) — single example
        layer_idx: which layer to measure
        target_pos: output position to trace back from (default: last)

    Returns:
        importance: (L,) — per-position gradient magnitude
    """
    model.eval()
    # Need gradients for this
    input_embeds = model.embedding(input_ids)  # (1, L, d)
    input_embeds = input_embeds.detach().requires_grad_(True)

    # Run only through the target layer
    x = input_embeds
    for i, layer in enumerate(model.layers):
        x = layer(x)
        if i == layer_idx:
            break

    # Compute gradient of sum at target position
    if target_pos < 0:
        target_pos = x.shape[1] + target_pos

    target = x[0, target_pos, :].sum()
    target.backward()

    # Gradient magnitude per position
    importance = input_embeds.grad[0].norm(dim=-1)  # (L,)
    return importance


@torch.no_grad()
def nca_convergence_rate(
    model: DPNCA,
    input_ids: Tensor,
    layer_idx: int = 0,
) -> list[float]:
    """Measure NCA state change magnitude at each step.

    If the NCA converges, the delta magnitude should decrease over steps.

    Args:
        model: DPNCA model
        input_ids: (B, L)
        layer_idx: which layer's NCA block to instrument

    Returns:
        List of delta magnitudes per NCA step
    """
    model.eval()

    target_block = model.layers[layer_idx].nca_block
    x = model.embedding(input_ids)

    # Get pre-normed input for the target layer
    for i in range(layer_idx):
        x = model.layers[i](x)
    x_normed = model.layers[layer_idx].norm1(x)

    # Manually run NCA steps to capture deltas
    G = target_block._reshape_to_grid(x_normed)
    G = target_block.rope(G, x.shape[1])
    H = target_block.hardware.init_for_sequence(x.shape[0], x.shape[1])

    delta_magnitudes = []
    for s in range(target_block.n_steps):
        G_prev = G.clone()
        P = target_block.perception(G)
        alpha = target_block.pathway_update.compute_routing(H)
        delta_G = target_block.pathway_update.compute_delta(P, alpha)
        G = target_block.pathway_update.apply_step(G, delta_G, step_idx=s)

        delta_mag = (G - G_prev).norm().item()
        delta_magnitudes.append(delta_mag)

    return delta_magnitudes


@torch.no_grad()
def model_summary(model: DPNCA) -> str:
    """Generate a summary of model architecture and parameters."""
    counts = model.count_parameters()
    lines = [
        f"DPNCA-LM Summary",
        f"{'='*50}",
        f"d_model:        {model.config.d_model}",
        f"n_layers:       {model.config.n_layers}",
        f"n_compute_rows: {model.config.n_compute_rows}",
        f"n_channels:     {model.config.n_channels}",
        f"n_steps:        {model.config.n_steps}",
        f"n_modes:        {model.config.n_modes}",
        f"n_hw_channels:  {model.config.n_hw_channels}",
        f"d_ff:           {model.config.d_ff}",
        f"vocab_size:     {model.config.vocab_size}",
        f"max_seq_len:    {model.config.max_seq_len}",
        f"dropout:        {model.config.dropout}",
        f"{'-'*50}",
        f"Total params:   {counts['total']:,}",
        f"Trainable:      {counts['total_trainable']:,}",
        f"Embedding:      {counts['embedding']:,}",
        f"All layers:     {counts['all_layers']:,}",
    ]
    return "\n".join(lines)
