"""Token projection utilities for hierarchical SW-based few-shot heads."""

from __future__ import annotations

import torch
import torch.nn as nn


class TokenSetProjector(nn.Module):
    """Project token sets to a shared metric space."""

    def __init__(self, input_dim: int, output_dim: int | None = None) -> None:
        super().__init__()
        output_dim = output_dim or input_dim
        self.proj = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(tokens))
