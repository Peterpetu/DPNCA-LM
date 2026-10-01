"""DPNCA-LM: Deep Parallel NCA Language Model. Plan §3.2, §4 Phase 6.

Top-level model class assembling all components:
    TokenEmbedding → [NCATransformerLayer × N_layers] → LayerNorm → OutputHead

The output head is weight-tied with the embedding matrix (E^T projection).

Provides:
    forward(input_ids) → logits
    get_loss(input_ids, target_ids) → (loss, metrics_dict)
    generate(prompt_ids, max_new_tokens, ...) → token_ids
    count_parameters() → dict
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from dpnca.config import DPNCAConfig
from dpnca.embedding import TokenEmbedding
from dpnca.nca_layer import NCATransformerLayer


class DPNCA(nn.Module):
    """Deep Parallel NCA Language Model. Plan §3.2, §4 Phase 6.

    Architecture:
        embed(tokens) → [NCATransformerLayer × n_layers] → final_norm → logits
    """

    def __init__(self, config: DPNCAConfig) -> None:
        super().__init__()
        self.config = config
        self._use_grad_ckpt = config.gradient_checkpointing

        # Token embedding (weight-tied with output)
        self.embedding = TokenEmbedding(config)

        # Stack of NCA transformer layers
        self.layers = nn.ModuleList([
            NCATransformerLayer(config, layer_idx=i) for i in range(config.n_layers)
        ])

        # Final layer norm before output projection
        self.final_norm = nn.LayerNorm(config.d_model)

        # Output head — weight-tied with embedding
        self.output_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output_head.weight = self.embedding.weight  # weight tying

    def forward(self, input_ids: Tensor) -> Tensor:
        """Forward pass: tokens → logits.

        Args:
            input_ids: (B, L) integer token IDs

        Returns:
            logits: (B, L, V) — unnormalized log-probabilities
        """
        # 1. Token embedding
        x = self.embedding(input_ids)  # (B, L, d)

        # 2. Pass through NCA transformer layers, threading aggregation state
        hw_injection = None
        for layer in self.layers:
            if self._use_grad_ckpt and self.training:
                x, hw_injection = grad_checkpoint(
                    layer, x, hw_injection, use_reentrant=False
                )
            else:
                x, hw_injection = layer(x, hw_injection=hw_injection)

        # 3. Final norm + output projection
        x = self.final_norm(x)        # (B, L, d)
        logits = self.output_head(x)   # (B, L, V)

        return logits

    def get_loss(
        self,
        input_ids: Tensor,
        target_ids: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute total loss with auxiliary regularization terms.

        Loss = L_CE + λ_path · L_path + λ_hw · L_hw

        Plan §9.3: Cross-entropy + pathway balance KL + hardware smoothness.

        Args:
            input_ids: (B, L) — input token IDs
            target_ids: (B, L) — target token IDs (shifted by 1)

        Returns:
            total_loss: scalar
            metrics: dict with per-component losses
        """
        logits = self.forward(input_ids)  # (B, L, V)

        # Cross-entropy loss
        B, L, V = logits.shape
        loss_ce = F.cross_entropy(
            logits.reshape(B * L, V),
            target_ids.reshape(B * L),
            reduction="mean",
        )

        # Pathway balance loss: KL(ᾱ ‖ U_M) — plan §9.3
        loss_path = self._pathway_balance_loss()

        # Hardware smoothness loss: ‖H_final - H_init‖²_F — plan §9.3
        loss_hw = self._hardware_smoothness_loss()

        # Helmholtz entropy loss: encourage routing diversity — V5 §6
        loss_helm = self._helmholtz_entropy_loss()

        total_loss = (
            loss_ce
            + self.config.lambda_pathway_balance * loss_path
            + self.config.lambda_hw_smooth * loss_hw
            + self.config.lambda_helm * loss_helm
        )

        metrics = {
            "loss_total": total_loss.detach(),
            "loss_ce": loss_ce.detach(),
            "loss_pathway_balance": loss_path.detach(),
            "loss_hw_smooth": loss_hw.detach(),
            "loss_helmholtz": loss_helm.detach(),
            "ppl": torch.exp(loss_ce.detach()).clamp(max=1e4),
        }

        return total_loss, metrics

    def _pathway_balance_loss(self) -> Tensor:
        """KL(ᾱ ‖ U_M) summed across layers. Plan §9.3.

        Uses step-averaged routing weights accumulated during the forward pass.
        ᾱ_m = (1/BTRL) Σ_{b,s,r,l} α_{b,m,r,l}^{ℓ,s}
        KL(ᾱ ‖ U_M) = Σ_m ᾱ_m ln(M · ᾱ_m) = ln M - H(ᾱ)
        """
        total_kl = torch.tensor(0.0, device=next(self.parameters()).device)
        M = self.config.n_modes

        for layer in self.layers:
            # Use step-averaged alpha accumulated during forward pass
            alpha_mean = layer.nca_block.get_routing_weights()
            if alpha_mean is not None:
                # Average across B, R, L to get per-mode usage ᾱ_m
                alpha_bar = alpha_mean.mean(dim=(0, 2, 3))  # (M,)
                # KL(ᾱ ‖ uniform) = Σ_m ᾱ_m ln(M · ᾱ_m)
                alpha_bar = alpha_bar.clamp(min=1e-8)
                kl = (alpha_bar * (alpha_bar * M).log()).sum()
                total_kl = total_kl + kl

        return total_kl

    def _hardware_smoothness_loss(self) -> Tensor:
        """Sum hardware smoothness penalties across layers. Plan §9.3."""
        total_hw = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.layers:
            total_hw = total_hw + layer.get_hardware_loss()
        return total_hw

    def _helmholtz_entropy_loss(self) -> Tensor:
        """Mean negative entropy of routing distributions. V5 §6.

        L_helm = -(1/NL) Σ_ℓ Σ_l H(α_{ℓ,l})

        where H(α) = -Σ_m α_m ln(α_m). Minimising this loss maximises
        routing entropy, preventing premature pathway collapse.
        Grounded in DVTM Spec §8.2 variational free energy.
        """
        total_neg_entropy = torch.tensor(0.0, device=next(self.parameters()).device)
        count = 0

        for layer in self.layers:
            alpha_mean = layer.nca_block.get_routing_weights()
            if alpha_mean is not None:
                # alpha_mean: (B, M, R, L) — step-averaged routing weights
                # Entropy per (b, r, l): H = -Σ_m α_m ln(α_m)
                alpha_clamped = alpha_mean.clamp(min=1e-8)
                neg_entropy = (alpha_clamped * alpha_clamped.log()).sum(dim=1)  # (B, R, L)
                total_neg_entropy = total_neg_entropy + neg_entropy.mean()
                count += 1

        if count > 0:
            total_neg_entropy = total_neg_entropy / count
        return total_neg_entropy

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> Tensor:
        """Autoregressive generation. Plan §11.

        Uses simple full-forward approach (no sliding window cache for V4.0).
        Cache optimization deferred to V4.1.

        Args:
            prompt_ids: (B, L_prompt) — prompt token IDs
            max_new_tokens: maximum number of tokens to generate
            temperature: sampling temperature
            top_k: top-k filtering (0 = greedy)

        Returns:
            generated_ids: (B, L_prompt + max_new_tokens) — full sequence
        """
        self.eval()
        ids = prompt_ids.clone()

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len if needed
            input_ids = ids if ids.shape[1] <= self.config.max_seq_len else ids[:, -self.config.max_seq_len:]

            logits = self.forward(input_ids)        # (B, L, V)
            next_logits = logits[:, -1, :]           # (B, V) — last position

            # Temperature scaling
            if temperature != 1.0:
                next_logits = next_logits / temperature

            # Top-k filtering
            if top_k > 0:
                top_k_vals, _ = next_logits.topk(min(top_k, next_logits.size(-1)))
                threshold = top_k_vals[:, -1].unsqueeze(-1)
                next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))

            probs = F.softmax(next_logits, dim=-1)

            if top_k == 0 and temperature == 0:
                next_token = probs.argmax(dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

            ids = torch.cat([ids, next_token], dim=1)

        return ids

    def count_parameters(self) -> dict[str, int]:
        """Parameter count breakdown by component. Plan §3.8."""
        counts: dict[str, int] = {}

        # Embedding (counted once due to weight tying)
        counts["embedding"] = sum(p.numel() for p in self.embedding.parameters())

        # Per-layer breakdown
        total_layers = 0
        for i, layer in enumerate(self.layers):
            layer_params = sum(p.numel() for p in layer.parameters())
            counts[f"layer_{i}"] = layer_params
            total_layers += layer_params

        counts["all_layers"] = total_layers

        # Final norm
        counts["final_norm"] = sum(p.numel() for p in self.final_norm.parameters())

        # Output head (tied, so 0 additional)
        counts["output_head_tied"] = 0

        # Grand total
        counts["total"] = sum(p.numel() for p in self.parameters())
        counts["total_trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return counts
