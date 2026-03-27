"""Clean Transformer modules for WARN few-shot models."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


def _build_2d_sincos_position_embedding(
    height: int,
    width: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid spatial size: {(height, width)}")
    if dim <= 0:
        raise ValueError(f"Invalid embedding dim: {dim}")

    base_dim = max(1, dim // 4)
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    y = y.reshape(-1, 1)
    x = x.reshape(-1, 1)

    omega = torch.arange(base_dim, device=device, dtype=dtype)
    omega = 1.0 / (10000 ** (omega / max(float(base_dim), 1.0)))

    x_proj = x * omega.unsqueeze(0)
    y_proj = y * omega.unsqueeze(0)
    pos = torch.cat(
        [torch.sin(x_proj), torch.cos(x_proj), torch.sin(y_proj), torch.cos(y_proj)],
        dim=-1,
    )
    if pos.shape[-1] < dim:
        pos = torch.cat(
            [pos, torch.zeros(pos.shape[0], dim - pos.shape[-1], device=device, dtype=dtype)],
            dim=-1,
        )
    elif pos.shape[-1] > dim:
        pos = pos[:, :dim]
    return pos.unsqueeze(0)


class TransformerTokenBlock(nn.Module):
    """Pre-norm Transformer encoder block."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        attn_heads = num_heads if dim % num_heads == 0 else 1
        hidden_dim = max(dim, dim * ffn_multiplier)
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, attn_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attn_input = self.attn_norm(tokens)
        attended, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        tokens = tokens + self.attn_dropout(attended)
        tokens = tokens + self.ffn_dropout(self.ffn(self.ffn_norm(tokens)))
        return tokens


class IntraImageTransformerEncoder(nn.Module):
    """Transformer encoder over image tokens with 2D sine-cosine positions."""

    def __init__(
        self,
        dim: int,
        depth: int = 1,
        num_heads: int = 4,
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if depth < 0:
            raise ValueError("depth must be non-negative")
        self.blocks = nn.ModuleList(
            [
                TransformerTokenBlock(
                    dim=dim,
                    num_heads=num_heads,
                    ffn_multiplier=ffn_multiplier,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor, spatial_hw: Tuple[int, int]) -> torch.Tensor:
        pos = _build_2d_sincos_position_embedding(
            spatial_hw[0],
            spatial_hw[1],
            tokens.shape[-1],
            device=tokens.device,
            dtype=tokens.dtype,
        )
        tokens = tokens + pos
        for block in self.blocks:
            tokens = block(tokens)
        return self.output_norm(tokens)


class QueryConditionedBasisDistiller(nn.Module):
    """Distill a routed support evidence set into a compact query-conditioned basis."""

    def __init__(
        self,
        dim: int,
        num_basis_tokens: int = 8,
        num_heads: int = 4,
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_basis_tokens <= 0:
            raise ValueError("num_basis_tokens must be positive")
        attn_heads = num_heads if dim % num_heads == 0 else 1
        hidden_dim = max(dim, dim * ffn_multiplier)

        self.base_queries = nn.Parameter(torch.randn(1, num_basis_tokens, dim) * 0.02)
        self.query_bias = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.query_norm = nn.LayerNorm(dim)
        self.support_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, attn_heads, dropout=dropout, batch_first=True)
        self.cross_dropout = nn.Dropout(dropout)
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, attn_heads, dropout=dropout, batch_first=True)
        self.self_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.ffn_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, query_summary: torch.Tensor, support_tokens: torch.Tensor) -> torch.Tensor:
        if query_summary.dim() != 2:
            raise ValueError(
                "query_summary must have shape (Batch, Dim), "
                f"got {tuple(query_summary.shape)}"
            )
        if support_tokens.dim() != 3:
            raise ValueError(
                "support_tokens must have shape (Batch, Tokens, Dim), "
                f"got {tuple(support_tokens.shape)}"
            )

        queries = self.base_queries.expand(query_summary.shape[0], -1, -1)
        queries = queries + self.query_bias(query_summary).unsqueeze(1)

        attended, _ = self.cross_attn(
            self.query_norm(queries),
            self.support_norm(support_tokens),
            self.support_norm(support_tokens),
            need_weights=False,
        )
        basis = queries + self.cross_dropout(attended)

        refined, _ = self.self_attn(
            self.self_norm(basis),
            self.self_norm(basis),
            self.self_norm(basis),
            need_weights=False,
        )
        basis = basis + self.self_dropout(refined)
        basis = basis + self.ffn_dropout(self.ffn(self.ffn_norm(basis)))
        return self.output_norm(basis)


class SupportSetBasisDistiller(nn.Module):
    """Distill an unordered support token measure into a compact basis."""

    def __init__(
        self,
        dim: int,
        num_basis_tokens: int = 8,
        num_heads: int = 4,
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_basis_tokens <= 0:
            raise ValueError("num_basis_tokens must be positive")
        attn_heads = num_heads if dim % num_heads == 0 else 1
        hidden_dim = max(dim, dim * ffn_multiplier)

        self.base_queries = nn.Parameter(torch.randn(1, num_basis_tokens, dim) * 0.02)
        self.query_norm = nn.LayerNorm(dim)
        self.support_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, attn_heads, dropout=dropout, batch_first=True)
        self.cross_dropout = nn.Dropout(dropout)
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, attn_heads, dropout=dropout, batch_first=True)
        self.self_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.ffn_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, support_tokens: torch.Tensor) -> torch.Tensor:
        if support_tokens.dim() != 3:
            raise ValueError(
                "support_tokens must have shape (Batch, Tokens, Dim), "
                f"got {tuple(support_tokens.shape)}"
            )

        basis = self.base_queries.expand(support_tokens.shape[0], -1, -1)
        attended, _ = self.cross_attn(
            self.query_norm(basis),
            self.support_norm(support_tokens),
            self.support_norm(support_tokens),
            need_weights=False,
        )
        basis = basis + self.cross_dropout(attended)

        refined, _ = self.self_attn(
            self.self_norm(basis),
            self.self_norm(basis),
            self.self_norm(basis),
            need_weights=False,
        )
        basis = basis + self.self_dropout(refined)
        basis = basis + self.ffn_dropout(self.ffn(self.ffn_norm(basis)))
        return self.output_norm(basis)
