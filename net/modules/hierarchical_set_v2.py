"""Set-aware modules reused by the pulse_fewshot consensus-slot model."""

from __future__ import annotations

import torch
import torch.nn as nn

from net.ssm.hierarchical_ssm import HierarchicalClassAggregator


class SetConditionedShotRefiner(nn.Module):
    """Refine shot embeddings using shared class-memory tokens."""

    def __init__(self, dim: int, num_heads: int = 4, ffn_multiplier: int = 2) -> None:
        super().__init__()
        hidden_dim = max(dim, dim * ffn_multiplier)
        attn_heads = num_heads if dim % num_heads == 0 else 1

        self.shot_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, attn_heads, dropout=0.0, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim)
        self.film = nn.Linear(dim, dim * 2)
        self.film_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(
        self,
        shot_embeddings: torch.Tensor,
        memory_tokens: torch.Tensor,
        class_summary: torch.Tensor,
    ) -> torch.Tensor:
        squeeze_batch = False
        if shot_embeddings.dim() == 2:
            shot_embeddings = shot_embeddings.unsqueeze(0)
            squeeze_batch = True
        if memory_tokens.dim() == 2:
            memory_tokens = memory_tokens.unsqueeze(0)
        if class_summary.dim() == 1:
            class_summary = class_summary.unsqueeze(0)

        if memory_tokens.shape[0] == 1 and shot_embeddings.shape[0] > 1:
            memory_tokens = memory_tokens.expand(shot_embeddings.shape[0], -1, -1)
        if class_summary.shape[0] == 1 and shot_embeddings.shape[0] > 1:
            class_summary = class_summary.expand(shot_embeddings.shape[0], -1)

        attended, _ = self.cross_attn(
            self.shot_norm(shot_embeddings),
            self.memory_norm(memory_tokens),
            self.memory_norm(memory_tokens),
        )
        refined = self.cross_norm(shot_embeddings + attended)

        film_scale, film_bias = self.film(class_summary).chunk(2, dim=-1)
        refined = self.film_norm(
            refined * (1.0 + 0.5 * torch.tanh(film_scale).unsqueeze(1))
            + film_bias.unsqueeze(1)
        )
        refined = refined + self.ffn(self.ffn_norm(refined))

        if squeeze_batch:
            return refined.squeeze(0)
        return refined


class ConsensusClassAggregator(nn.Module):
    """Base class seed plus consensus-weighted support summary."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.base_aggregator = HierarchicalClassAggregator(dim)
        self.score = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        shot_embeddings: torch.Tensor,
        class_memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if shot_embeddings.dim() != 2:
            raise ValueError(
                "shot_embeddings must have shape (Shot, Dim), "
                f"got {tuple(shot_embeddings.shape)}"
            )
        if class_memory.dim() != 1:
            raise ValueError(
                "class_memory must have shape (Dim,), "
                f"got {tuple(class_memory.shape)}"
            )

        base_repr = self.base_aggregator(shot_embeddings, class_memory)
        base_expanded = base_repr.unsqueeze(0).expand(shot_embeddings.shape[0], -1)
        logits = self.score(
            torch.cat(
                [
                    shot_embeddings,
                    base_expanded,
                    torch.abs(shot_embeddings - base_expanded),
                ],
                dim=-1,
            )
        ).squeeze(-1)
        shot_weights = torch.softmax(logits, dim=0)
        weighted_summary = torch.sum(shot_weights.unsqueeze(-1) * shot_embeddings, dim=0)
        fused = self.fusion(torch.cat([base_repr, weighted_summary], dim=-1))
        class_repr = self.norm(base_repr + fused)
        return class_repr, shot_weights

