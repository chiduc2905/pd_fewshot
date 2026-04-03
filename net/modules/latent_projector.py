"""Latent evidence projector for SC-LFI."""

from __future__ import annotations

import torch
import torch.nn as nn


class LatentEvidenceProjector(nn.Module):
    """Project backbone tokens into the latent evidence space.

    This latent space is not used for image generation. It is the compact
    support/query evidence space where class-conditional distributions live.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim or max(input_dim, latent_dim))
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.output_norm = nn.LayerNorm(latent_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() < 2:
            raise ValueError(f"tokens must have at least 2 dimensions, got {tuple(tokens.shape)}")
        return self.output_norm(self.network(tokens))
