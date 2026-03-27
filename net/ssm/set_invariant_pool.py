"""Permutation-invariant pooling for class-level support aggregation."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class SetInvariantMemoryPool(nn.Module):
    """Pool a support token set into learned memory tokens and a summary."""

    def __init__(
        self,
        dim: int,
        num_memory_tokens: int = 4,
        num_heads: int = 4,
        ffn_multiplier: int = 2,
    ) -> None:
        super().__init__()
        if num_memory_tokens <= 0:
            raise ValueError("num_memory_tokens must be positive")

        hidden_dim = max(dim, dim * ffn_multiplier)
        attn_heads = num_heads if dim % num_heads == 0 else 1

        self.memory_queries = nn.Parameter(torch.randn(1, num_memory_tokens, dim) * 0.02)
        self.input_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, attn_heads, dropout=0.0, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, attn_heads, dropout=0.0, batch_first=True)
        self.self_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.summary_proj = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.summary_norm = nn.LayerNorm(dim)

    def forward(self, elements: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        squeeze_batch = False
        if elements.dim() == 2:
            elements = elements.unsqueeze(0)
            squeeze_batch = True
        elif elements.dim() != 3:
            raise ValueError(
                "elements must have shape (Elements, Dim) or (Batch, Elements, Dim), "
                f"got {tuple(elements.shape)}"
            )

        element_repr = self.input_norm(elements)
        queries = self.memory_queries.expand(element_repr.shape[0], -1, -1)

        pooled, _ = self.cross_attn(queries, element_repr, element_repr)
        memory_tokens = self.cross_norm(queries + pooled)

        self_attended, _ = self.self_attn(memory_tokens, memory_tokens, memory_tokens)
        memory_tokens = self.self_norm(memory_tokens + self_attended)
        memory_tokens = memory_tokens + self.ffn(self.ffn_norm(memory_tokens))

        summary_input = torch.cat([memory_tokens.mean(dim=1), element_repr.mean(dim=1)], dim=-1)
        summary = self.summary_norm(self.summary_proj(summary_input) + memory_tokens.mean(dim=1))

        if squeeze_batch:
            return memory_tokens.squeeze(0), summary.squeeze(0)
        return memory_tokens, summary

